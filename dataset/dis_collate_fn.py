from dataset.dis_dataset import dis_dataset
from opts.dis_opts import dis_opts

import torch

def dis_collate_fn(data):

    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        if max(lens) > dis_opts.conv_padding_len:
            padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        else:
            padded_seqs = torch.zeros(len(seqs), dis_opts.conv_padding_len).long()
        
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lens

    query_sents, response_sents, query_seqs, response_seqs, labels = zip(*data)

    
    # Merge sequences (from tuple of 1D tensor to 2D tensor)
    query_seqs, query_lens = _pad_sequences(query_seqs)            ## (batch, max_seq_len), (batch)
    response_seqs, response_lens = _pad_sequences(response_seqs)   ## (batch, max_seq_len), (batch)

    return query_sents, response_sents, query_seqs, response_seqs, query_lens, response_lens, labels