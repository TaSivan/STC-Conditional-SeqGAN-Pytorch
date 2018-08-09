import torch

def collate_fn(data):
    """
    Creates mini-batch tensors from (src_sent, tgt_sent, src_seq, tgt_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    
    Args:
        data: list of tuple (src_sents, tgt_sents, src_seqs, tgt_seqs)
        - src_sents, tgt_sents: batch of original tokenized sentences
        - src_seqs, tgt_seqs: batch of original tokenized sentence ids
    Returns:
        - src_sents, tgt_sents (tuple): batch of original tokenized sentences
        - src_seqs, tgt_seqs (tensor): (max_src_len, batch_size)
        - src_lens, tgt_lens (tensor): (batch_size)
       
    """
    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lens

    # Sort a list by *source* sequence length (descending order) to use `pack_padded_sequence`.
    # The *target* sequence is not sorted <-- It's ok, cause `pack_padded_sequence` only takes
    # *source* sequence, which is in the EncoderRNN
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # Seperate source and target sequences.
    src_sents, tgt_sents, src_seqs, tgt_seqs = zip(*data)
    
    # Merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lens = _pad_sequences(src_seqs)   ## (batch, seq_len), (batch)
    tgt_seqs, tgt_lens = _pad_sequences(tgt_seqs)   ## (batch, seq_len), (batch)
    
    # (batch, seq_len) => (seq_len, batch)
    src_seqs = src_seqs.transpose(0,1)
    tgt_seqs = tgt_seqs.transpose(0,1)

    return src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens



"""

In[0]:

    src_seqs = [np.random.randint(0,100,(len)).tolist() for len in [20, 12, 16]]
    src_seqs

Out[0]:

    [[79, 22, 10, 38, 67, 73, 33, 60, 78, 94, 35, 49, 30, 33, 85, 71, 72, 75, 19, 46],
    [22, 57, 97, 19, 95, 30, 67, 3, 47, 21, 39, 25],
    [97, 28, 47, 49, 55, 73, 94, 69, 35, 51, 10, 27, 12, 85, 42, 69]]

In[1]:

    src_seqs, src_lens = _pad_sequences(src_seqs)

In[2]:

    src_seqs

Out[2]:

    tensor([[79, 22, 10, 38, 67, 73, 33, 60, 78, 94, 35, 49, 30, 33, 85, 71, 72, 75,19, 46],
        [22, 57, 97, 19, 95, 30, 67,  3, 47, 21, 39, 25,  0,  0,  0,  0,  0,  0,  0,  0],
        [97, 28, 47, 49, 55, 73, 94, 69, 35, 51, 10, 27, 12, 85, 42, 69,  0,  0,  0,  0]])

In[3]:

    src_lens

Out[3]:

    [20, 12, 16]

"""