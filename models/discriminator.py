import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """A CNN for text classification

    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """
    def __init__(self, num_classes, vocab_size, emb_dim, dropout):
        super(Discriminator, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)                    ## (5000, 300)

        self.filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_dim)) for (n, f) in zip(self.num_filters, self.filter_sizes)
        ])
        """
        input: (16, 1, seq_len, 300)

            filter_sizes resembles the "N" of N-grams

            (1, 100, ( 1, 300))  ->  (16, 100, seq_len, 1)       ->  (16, 100, seq_len)        
            (1, 200, ( 2, 300))  ->  (16, 200, seq_len - 1, 1)   ->  (16, 200, seq_len - 1)
            (1, 200, ( 3, 300))  ->  (16, 200, seq_len - 2, 1)   ->  (16, 200, seq_len - 2)
            (1, 200, ( 4, 300))  ->  (16, 200, seq_len - 3, 1)   ->  (16, 200, seq_len - 3)
            (1, 200, ( 5, 300))  ->  (16, 200, seq_len - 4, 1)   ->  (16, 200, seq_len - 4)
            (1, 100, ( 6, 300))  ->  (16, 100, seq_len - 5, 1)   ->  (16, 100, seq_len - 5)
            (1, 100, ( 7, 300))  ->  (16, 100, seq_len - 6, 1)   ->  (16, 100, seq_len - 6)
            (1, 100, ( 8, 300))  ->  (16, 100, seq_len - 7, 1)   ->  (16, 100, seq_len - 7)
            (1, 100, ( 9, 300))  ->  (16, 100, seq_len - 8, 1)   ->  (16, 100, seq_len - 8)
            (1, 100, (10, 300))  ->  (16, 100, seq_len - 9, 1)   ->  (16, 100, seq_len - 9)
            (1, 160, (15, 300))  ->  (16, 160, seq_len - 14, 1)  ->  (16, 160, seq_len - 14)
            (1, 160, (20, 300))  ->  (16, 160, seq_len - 19, 1)  ->  (16, 160, seq_len - 19)
        """        
        self.highway = nn.Linear(sum(self.num_filters), sum(self.num_filters))    ## (1720, 1720)
        self.dropout = nn.Dropout(p=dropout)                            ##  0.75
        self.lin = nn.Linear(sum(self.num_filters), num_classes)             ## (1720, 2)
        self.softmax = nn.LogSoftmax()
        self.init_parameters()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)

        (16, 100) -> (16, 100, 300) -> (16, 1, 100, 300)
            
        (16, 1, 100, 300) -> convs -> pools -> pred
        ├── (16, 100, 100, 1) -> (16, 100, 100) -> (16, 100, 1) -> (16, 100) ──|
        ├── (16, 200,  99, 1) -> (16, 200,  99) -> (16, 200, 1) -> (16, 200) ──|
        ├── (16, 200,  98, 1) -> (16, 200,  98) -> (16, 200, 1) -> (16, 200) ──|
        ├── (16, 200,  97, 1) -> (16, 200,  97) -> (16, 200, 1) -> (16, 200) ──|
        ├── (16, 200,  96, 1) -> (16, 200,  96) -> (16, 200, 1) -> (16, 200) ──|
        ├── (16, 100,  95, 1) -> (16, 100,  95) -> (16, 100, 1) -> (16, 100) ──|   (cat)
        ├── (16, 100,  94, 1) -> (16, 100,  94) -> (16, 100, 1) -> (16, 100) ──|───────────> (16, 1720)
        ├── (16, 100,  93, 1) -> (16, 100,  93) -> (16, 100, 1) -> (16, 100) ──|
        ├── (16, 100,  92, 1) -> (16, 100,  92) -> (16, 100, 1) -> (16, 100) ──|
        ├── (16, 100,  91, 1) -> (16, 100,  91) -> (16, 100, 1) -> (16, 100) ──|
        ├── (16, 160,  86, 1) -> (16, 160,  86) -> (16, 160, 1) -> (16, 160) ──|
        ├── (16, 160,  81, 1) -> (16, 160,  81) -> (16, 160, 1) -> (16, 160) ──|

        pred -> highway
        (16, 1720) -> (16, 1720)

        Highway Networks
        Ref: https://arxiv.org/pdf/1505.00387.pdf
             https://blog.csdn.net/guoyuhaoaaa/article/details/54093913

        """ 
        emb = self.emb(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim    ## (16, 1, 100, 300)
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)                          ## (16, 1720)
        pred = F.sigmoid(highway) *  F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        pred = self.softmax(self.lin(self.dropout(pred)))     ## (16, 2)
        return pred

    def init_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.constant_(self.bias, 0)

        # for param in self.parameters():
        #     param.data.uniform_(-0.05, 0.05)
