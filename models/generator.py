import torch
import torch.nn as nn
import torch.nn.functional as F

from opts.gen_opts import gen_opts
from helper import PAD, SOS, EOS, UNK
from util.sequence_mask import sequence_mask


class Generator(nn.Module):
    def __init__(self, encoder, decoder, USE_CUDA=True):
        super(Generator, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.USE_CUDA = USE_CUDA

    def forward(self, src_seqs, src_lens, enable_dropout=True, state=None):
        """
            - src_seqs: Tensor, shape: (seq_len, batch_size)
            - src_lens:  list,  shape: [1] * batch_size
        """

        batch_size = src_seqs.size(1)
        src_lens = torch.LongTensor(src_lens)   # (batch_size)

        input_seq = torch.LongTensor([SOS] * batch_size)
        
        out_seqs = torch.zeros(gen_opts.max_seq_len, batch_size).long()
        out_lens = torch.zeros(batch_size).long()
        decoder_outputs = torch.zeros(gen_opts.max_seq_len, batch_size, self.decoder.vocab_size)

        if self.USE_CUDA:
            src_seqs = src_seqs.cuda()
            src_lens = src_lens.cuda()
            input_seq = input_seq.cuda()

        if enable_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        # - encoder_outputs: (max_seq_len, batch_size, hidden_size)
        # - encoder_hidden:  (num_layers, batch_size, num_directions * hidden_size)
        encoder_outputs, encoder_hidden = self.encoder(src_seqs)

        decoder_hidden = encoder_hidden


        state_len = 0

        if state is not None:
            """
                state is used to do the monte carlo search

                - state: (state_len, batch_size)
            """
            state_len = state.size(0)
            
            for t in range(state_len):
                # - decoder_output   : (batch_size, vocab_size)
                # - decoder_hidden   : (num_layers, batch_size, hidden_size)
                # - attention_weights: (batch_size, max_src_len)
                decoder_output, decoder_hidden, attention_weights \
                = self.decoder(input_seq, decoder_hidden, encoder_outputs, src_lens)
                
                decoder_outputs[t] = decoder_output

                out_seqs[t] = state[t]  # (batch_size)

                if (state[t] == EOS).sum().item() != 0:     # someone has finished generating seqs
                    out_lens += ((out_lens == 0) * (state[t] == EOS) * (t+1)).long()

                input_seq = state[t]    # (batch_size)
                if self.USE_CUDA:
                    input_seq = input_seq.cuda()


        for t in range(state_len, gen_opts.max_seq_len):

            # - decoder_output   : (batch_size, vocab_size)
            # - decoder_hidden   : (num_layers, batch_size, hidden_size)
            # - attention_weights: (batch_size, max_src_len)
            decoder_output, decoder_hidden, attention_weights \
            = self.decoder(input_seq, decoder_hidden, encoder_outputs, src_lens)

            decoder_outputs[t] = decoder_output


            if state is not None:
                # Sample a word from the given probability distribution
                prob = F.softmax(decoder_output, dim=1)     # (batch_size, vocab_size)
                token_id = prob.multinomial(1)              # (batch_size, 1)
            else:
                # Choose top word from decoder's output
                # - prob:       (batch_size, 1)
                # - token_id:   (batch_size, 1)
                prob, token_id = decoder_output.topk(1)

            # (batch_size, 1) -> (batch_size)
            token_id = token_id.squeeze(1)
            
            out_seqs[t] = token_id

            if (token_id == EOS).sum().item() != 0:     # someone has finished generating seqs
                out_lens += ((out_lens == 0) * (token_id == EOS).cpu() * (t+1)).long()

            if (out_lens == 0).sum().item() == 0:       # everyone has finished generating seqs
                break

            input_seq = token_id

        # if someone doesn't have lens, that means, it ended the for-loop without meeting <EOS>
        out_lens += ((out_lens == 0) * gen_opts.max_seq_len).long()     

        max_out_len = out_lens.max().item()
        
        # (max_out_len, batch_size, vocab_size) -> (batch_size, max_out_len, vocab_size)
        decoder_outputs = decoder_outputs[:max_out_len].transpose(0,1)

        # (max_out_len, batch_size) -> (batch_size, max_out_len)
        out_seqs = out_seqs[:max_out_len].transpose(0,1)
        mask = sequence_mask(out_lens)                      # (batch_size, max_out_len)
        out_seqs.masked_fill_(1 - mask, PAD)                # (batch_size, max_out_len)

        del src_seqs, src_lens, input_seq, mask, batch_size, encoder_outputs, encoder_hidden, decoder_output,  \
            decoder_hidden, attention_weights, state_len, prob, token_id, max_out_len
        
        torch.cuda.empty_cache()
        
        """
            - out_seqs:           (batch_size, max_out_len)
            - out_lens:           (batch_size)
            - decoder_outputs:    (batch_size, max_out_len, vocab_size)
        """
        return out_seqs, out_lens, decoder_outputs


## -----------------------------------------------------------------------------------

from opts.cuda_opts import USE_CUDA
from models.encoder import encoder
from models.decoder import decoder

generator = Generator(encoder=encoder,
                      decoder=decoder,
                      USE_CUDA=USE_CUDA)