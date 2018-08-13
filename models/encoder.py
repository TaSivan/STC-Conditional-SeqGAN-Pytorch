import torch
import torch.nn as nn

from opts import device

class Encoder(nn.Module):
    def __init__(self, embedding=None, rnn_type='LSTM', hidden_size=256, num_layers=1, dropout=0.3, bidirectional=True, fixed_embeddings=False):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size // self.num_directions


        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        if embedding is not None:
            self.embedding.weight.data.copy_(embedding.weight)

        if fixed_embeddings:
            self.embedding.weight.requires_grad = False

        self.word_vec_size = self.embedding.embedding_dim

        self.rnn_type = rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
                           input_size=self.word_vec_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           dropout=self.dropout,
                           bidirectional=self.bidirectional)

        self.init_parameters()



    def forward(self, src_seqs):
        emb = self.embedding(src_seqs)              # (max_seq_len, batch_size) -> (max_seq_len, batch_size, emb_dim)
        batch_size = src_seqs.size(1)
        hidden = self.initHidden(batch_size)        # (num_layers * num_directions, batch_size, hidden_size)
        outputs, hidden = self.rnn(emb, hidden)     # (max_seq_len, batch, hidden_size), (num_layers * num_directions, batch_size, hidden_size)

        if self.bidirectional:
            hidden = self._cat_directions(hidden)
        return outputs, hidden                      # (max_seq_len, batch, hidden_size), (num_layers, batch_size, num_directions * hidden_size)

    def _cat_directions(self, hidden):
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)

        return hidden

    def initHidden(self, batch_size):
        if self.rnn_type == "LSTM":
            return (torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=device),
                    torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=device))
        else:
            return torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=device)

    def init_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("embedding"): continue
            if param.requires_grad == True:
                if param.ndimension() >= 2:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.constant_(param, 0)


## -----------------------------------------------------------------------------------


from helper import PAD ,SOS, EOS, UNK
from opts import LOAD_CHECKPOINT, opts, USE_CUDA, USE_PARALLEL

if LOAD_CHECKPOINT:
    from opts import checkpoint
    from dataset.gen_dataset import gen_dataset
    encoder = Encoder(embedding=nn.Embedding(len(gen_dataset.vocab.token2id), opts.word_vec_size, padding_idx=PAD),
                        hidden_size=opts.hidden_size,
                        num_layers=opts.num_layers,
                        dropout=opts.dropout,
                        bidirectional=opts.bidirectional,
                        fixed_embeddings=opts.fixed_embeddings)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
else:
    from embedding.load_emb import embedding
    encoder = Encoder(embedding=embedding,
                        hidden_size=opts.hidden_size,
                        num_layers=opts.num_layers,
                        dropout=opts.dropout,
                        bidirectional=opts.bidirectional,
                        fixed_embeddings=opts.fixed_embeddings)

if USE_CUDA:
    if USE_PARALLEL:
        encoder = nn.DataParallel(encoder)  ## by default, device_ids is list of all device ids, and output_device uses device_ids[0]
    encoder.cuda()

print(encoder)
