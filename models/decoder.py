import torch
import torch.nn as nn
from torch.nn import functional as F

from util.sequence_mask import sequence_mask


class Decoder(nn.Module):
    def __init__(self, encoder, embedding, attention=True, bias=True, dropout=0.3, fixed_embeddings=False):
        super(Decoder, self).__init__()
        
        self.hidden_size = encoder.hidden_size * encoder.num_directions
        self.num_layers = encoder.num_layers
        self.dropout = dropout
        self.attention = attention
        
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding.weight.data.copy_(embedding.weight)

        if fixed_embeddings:
            self.embedding.weight.requires_grad = False
        
        self.vocab_size = self.embedding.num_embeddings
        self.word_vec_size = self.embedding.embedding_dim
        
        self.rnn_type = encoder.rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
                           input_size=self.word_vec_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           dropout=self.dropout)
        
        if self.attention:
            self.W_a = nn.Linear(encoder.hidden_size * encoder.num_directions,
                                 self.hidden_size, bias=bias)
            self.W_c = nn.Linear(encoder.hidden_size * encoder.num_directions + self.hidden_size,
                                 self.hidden_size, bias=bias)
            
        self.W_s = nn.Linear(self.hidden_size, self.vocab_size, bias=bias)

        self.init_parameters()

    def forward(self, input_seq, decoder_hidden, encoder_outputs, src_lens):
        
        # (batch_size) -> (seq_len=1, batch_size)
        input_seq = input_seq.unsqueeze(0)
        
        # (seq_len=1, batch_size) -> (seq_len=1, batch_size, emb_dim) 
        emb = self.embedding(input_seq)
        
        # rnn returns:
        # - decoder_output: (seq_len=1, batch_size, hidden_size)
        # - decoder_hidden: (num_layers, batch_size, hidden_size)
        decoder_output, decoder_hidden = self.rnn(emb, decoder_hidden)

        # (seq_len=1, batch_size, hidden_size) -> (batch_size, seq_len=1, hidden_size)
        decoder_output = decoder_output.transpose(0, 1)
        
        if self.attention:
            # attention_scores: (batch_size, seq_len=1, max_src_len)
            attention_scores = torch.bmm(decoder_output, self.W_a(encoder_outputs).permute(1, 2, 0))

            # attention_mask: (batch_size, seq_len=1, max_src_len)
            attention_mask = sequence_mask(src_lens).unsqueeze(1)

            # Fills elements of tensor with `-float('inf')` where `mask` is 1.
            attention_scores.data.masked_fill_(1 - attention_mask.data, -float('inf'))

            # attention_weights: (batch_size, seq_len=1, max_src_len) -> (batch_size, max_src_len) for `F.softmax` 
            # -> (batch_size, seq_len=1, max_src_len)
            attention_weights = F.softmax(attention_scores.squeeze(1), dim=1).unsqueeze(1)

            # context_vector:
            # (batch_size, seq_len=1, max_src_len) * (batch_size, max_src_len, encoder_hidden_size * num_directions)
            # -> (batch_size, seq_len=1, encoder_hidden_size * num_directions)
            context_vector = torch.bmm(attention_weights, encoder_outputs.transpose(0, 1))

            # concat_input: (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size)
            concat_input = torch.cat([context_vector, decoder_output], -1)

            # (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size) -> (batch_size, seq_len=1, decoder_hidden_size)
            concat_output = torch.tanh(self.W_c(concat_input))
            
            # Prepare returns:
            # (batch_size, seq_len=1, max_src_len) => (batch_size, max_src_len)
            attention_weights = attention_weights.squeeze(1)
        else:
            attention_weights = None
            concat_output = decoder_output
        
        # (batch_size, seq_len=1, decoder_hidden_size) => (batch_size, seq_len=1, vocab_size)
        output = self.W_s(concat_output)    
        
        # Prepare returns:
        # (batch_size, seq_len=1, vocab_size) => (batch_size, vocab_size)
        output = output.squeeze(1)
        
        del src_lens
        
        return output, decoder_hidden, attention_weights

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
from models.encoder import encoder
from opts.gen_opts import LOAD_GEN_CHECKPOINT, gen_opts
from opts.cuda_opts import USE_CUDA

if LOAD_GEN_CHECKPOINT:
    from opts.gen_opts import gen_checkpoint
    from dataset.gen_dataset import gen_dataset

    decoder = Decoder(encoder=encoder, 
                      embedding=nn.Embedding(len(gen_dataset.vocab.token2id), gen_opts.word_vec_size, padding_idx=PAD), 
                      attention = gen_opts.attention, 
                      dropout=gen_opts.dropout, 
                      fixed_embeddings=gen_opts.fixed_embeddings)

    decoder.load_state_dict(gen_checkpoint['decoder_state_dict'])
    
else:
    from embedding.load_emb import embedding
    
    decoder = Decoder(encoder=encoder, 
                      embedding=embedding, 
                      attention = gen_opts.attention, 
                      dropout=gen_opts.dropout, 
                      fixed_embeddings=gen_opts.fixed_embeddings)

if USE_CUDA:
    decoder.cuda()

print(decoder)