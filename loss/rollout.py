import torch

from helper import EOS
from opts.dis_opts import dis_opts
from opts.cuda_opts import USE_CUDA
from util.sequence_mask import sequence_mask


class Rollout(object):
    def __init__(self, generator, discriminator):
    
        self.generator = generator
        self.discriminator = discriminator

    def get_reward(self, src_seqs, response_seqs, src_lens, response_lens, num_rollout=16):
        """
            - src_seqs:         (seq_len, batch_size)
            - src_lens:         [1] * batch_size

            - response_seqs:    (batch_size, max_out_len)
            - response_lens:    (batch_size)
        """
        
        max_response_len = response_seqs.size(1)
        batch_size = src_seqs.size(1)

        rewards = torch.zeros(batch_size, max_response_len)     # (batch_size, max_response_len)
        
        query_seqs = self.pad_seqs(src_seqs.t())
        if USE_CUDA:
            query_seqs = query_seqs.cuda()


        # - response_seqs: (max_out_len, batch_size)
        response_seqs = response_seqs.t()
        
        for i in range(num_rollout):
            for l in range(1, max_response_len):
                state = response_seqs[:l]   # (state_len, batch_size)

                """
                    - out_seqs:           (batch_size, max_out_len)
                    - out_lens:           (batch_size)
                    - decoder_outputs:    (batch_size, max_out_len, vocab_size)
                """
                out_seqs, out_lens, decoder_outputs = self.generator(src_seqs, src_lens, enable_dropout=True, state=state)
            

                """
                    - query_seqs:     (batch_size, query_max_seq_len>=20)
                    - response_seqs:  (batch_size, response_max_seq_len>=20)
                    - reward:         (batch_size, 1)
                """
                
                out_seqs = self.pad_seqs(out_seqs)
                if USE_CUDA:
                    out_seqs = out_seqs.cuda()

                reward = - self.discriminator(query_seqs, out_seqs)[:, 1]

                rewards[:, l-1] += reward.cpu()
                
                del out_seqs, out_lens, decoder_outputs, reward
                torch.cuda.empty_cache()


            # for the last token, there's no need to do the response again
            out_seqs = self.pad_seqs(response_seqs.t())
            if USE_CUDA:
                out_seqs = out_seqs.cuda()

            reward = - self.discriminator(query_seqs, out_seqs)[:, 1]
            
            rewards[:, max_response_len-1] += reward.cpu()
            
            del out_seqs, reward
            torch.cuda.empty_cache()


        mask = sequence_mask(response_lens).float()   # (batch_size, max_response_len)
        rewards = rewards * mask                      # (batch_size, max_response_len)
        rewards = rewards / num_rollout               # (batch_size, max_response_len)

        del query_seqs, mask
        torch.cuda.empty_cache()

        return rewards


    def pad_seqs(self, seqs):
        # - seqs: (batch_size, max_out_len)

        seqs_len = seqs.size(1)
        if seqs_len  < dis_opts.conv_padding_len:    
            padded_seqs = torch.zeros(seqs.size(0), dis_opts.conv_padding_len).long()
            for i, seq in enumerate(seqs):
                padded_seqs[i, :seqs_len] = seq
            return padded_seqs

        else:
            return seqs


## -----------------------------------------------------------------------------------

from models.generator import generator
from models.discriminator import discriminator

rollout = Rollout(generator, discriminator)