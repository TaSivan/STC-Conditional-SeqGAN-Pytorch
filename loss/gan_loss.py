import torch
import torch.nn as nn
import torch.nn.functional as F

from opts.cuda_opts import USE_CUDA
from loss.rollout import rollout

"""Reward-Refined NLLLoss Function for adversarial training of Generator"""
def gan_loss(src_sents, src_seqs, fake_response_seqs, decoder_outputs, batch_size, num_rollout, max_out_len):
    
    # Rollout
    rewards = torch.zeros(batch_size, max_out_len)
    if USE_CUDA:
        rewards = rewards.cuda()

    # src_seqs: (seq_len, batch_size) -> (batch_size, seq_len)
    src_seqs = src_seqs.transpose(0, 1)

    for i in range(batch_size):
        query_sent = src_sents[i]
        query_seq = src_seqs[i].tolist()    # [1] * seq_len
        ori_response_seq = fake_response_seqs[i]
        # num_rollout = 16

        # (batch_size=1, max_out_len)
        rewards[i] = rollout.get_reward(query_sent, query_seq, ori_response_seq, num_rollout, max_out_len)

    decoder_outputs = F.log_softmax(decoder_outputs, dim=2)

    # (batch_size, max_out_len) * (batch_size, max_out_len)
    loss = decoder_outputs.topk(1)[0].squeeze(2) * rewards
    loss = - loss.sum()

    return loss