import torch

from opts.dis_opts import dis_opts

def dis_collate_fn(data):

    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]

        # Kernel size of discriminator can't be greater than actual input size
        if max(lens) >= dis_opts.conv_padding_len:
            padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        else:
            padded_seqs = torch.zeros(len(seqs), dis_opts.conv_padding_len).long()

        for i, seq in enumerate(seqs):
            padded_seqs[i, :lens[i]] = torch.LongTensor(seq)
        return padded_seqs

    """
        Input:
            - query_seqs:       [ (seq_len) ] * batch_size
            - response_seqs:    [ (seq_len) ] * batch_size
            - labels:           [ 1 ] * batch_size

        Output:
            - query_seqs:       (batch_size, max_seq_len>=dis_opts.conv_padding_len)
            - response_seqs:    (batch_size, max_seq_len>=dis_opts.conv_padding_len)
            - labels:           [ 1 ] * batch_size
    """

    query_seqs, response_seqs, labels = zip(*data)

    query_seqs = _pad_sequences(query_seqs)         # (batch_size, max_seq_len>=20)
    response_seqs = _pad_sequences(response_seqs)   # (batch_size, max_seq_len>=20)

    return query_seqs, response_seqs, labels