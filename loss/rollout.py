import torch
import torch.nn as nn

from helper import EOS
from dataset.dis_collate_fn import _pad_sequences
from opts.cuda_opts import USE_CUDA


class Rollout(object):
    """Roll-out policy"""
    def __init__(self, responsor, discriminator):
        self.responsor = responsor
        self.discriminator = discriminator

    def get_reward(self, query_sent, query_seq, ori_response_seq, num_rollout, max_out_len):
        """

        e.g. 
            query_sent:     ["what", "do", "you", "have"]
            response_sent:  ["I", "have", "an", "apple"]
            state:          ["I"],
                            ["I", "have"],
                            ["I", "have", "an"]

            Rollouts:
            time            state               sample                  Reward(accumulate)
            -----           ----------------    ----------------    D   ------------------
            1               I                   I x x x x .... x    ->  rewards[0] += pred
            2               I have              I have x x ... x    ->  rewards[1] += pred
            3               I have an           I have an x .. x    ->  rewards[2] += pred
            4(last)         I have an apple     I have an apple     ->  rewards[3] += pred

        """
        
        query_seq = _pad_sequences([query_seq])[0]  # (batch=1, seq_len>=20)
        rewards = torch.zeros(1, max_out_len)   # (batch_size=1, max_out_len)
        if USE_CUDA:
            query_seq = query_seq.cuda()
            rewards = rewards.cuda()


        seq_len = len(ori_response_seq)


        for i in range(num_rollout):
            for l in range(1, seq_len):
                state = ori_response_seq[0:l]
                response_seq = responsor.forward(query_sent, state=state)

                # e.g.
                #   response_seq = [4, 5, 6, 7, 8, 2]
                #   -> tensor([4, 5, 6, 7, 8, 2, 0, 0, 0, 0, 0, ..., 0])    # pad seqence to make seq_len >= dis_ops.conv_padding_len
                response_seq = _pad_sequences([response_seq])[0]   # (batch=1, seq_len>=20)    
                if USE_CUDA:
                    response_seq = response_seq.cuda()

                # (batch_size=1, num_classes=2) -> (batch_size=1, 1)
                reward = - self.discriminator(query_seq, response_seq)[:, 1]
                
                rewards[:, l-1] += reward


            # for the last token, there's no need to do the response again
            response_seq = _pad_sequences([ori_response_seq])[0]  # (batch=1, seq_len>=20)
            if USE_CUDA:
                response_seq = response_seq.cuda()

            reward = - self.discriminator(query_seq, response_seq)[:, 1]
            rewards[:, seq_len-1] += reward


        return rewards / num_rollout



## -----------------------------------------------------------------------------------

from evaluator.responsor import responsor
from models.discriminator import discriminator

rollout = Rollout(responsor, discriminator)