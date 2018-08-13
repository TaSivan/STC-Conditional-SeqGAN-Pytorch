import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """A CNN for text classification

    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """
    def __init__(self, embedding, dropout=0.75, fixed_embeddings=False):
        super(Discriminator, self).__init__()

        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding.weight.data.copy_(embedding.weight)
        
        if fixed_embeddings:
            self.embedding.weight.requires_grad = False

        self.word_vec_size = self.embedding.embedding_dim

        self.filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

        self.num_classes = 2

        self.convs_q = nn.ModuleList([
            nn.Conv2d(1, n, (f, embedding.embedding_dim)) for (n, f) in zip(self.num_filters, self.filter_sizes)
        ])

        self.convs_r = nn.ModuleList([
            nn.Conv2d(1, n, (f, embedding.embedding_dim)) for (n, f) in zip(self.num_filters, self.filter_sizes)
        ])

        self.highway = nn.Linear(2 * sum(self.num_filters), 2 * sum(self.num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(2 * sum(self.num_filters), self.num_classes)
        
        self.init_parameters()
    
    def forward(self, query_seqs, response_seqs):
        """
        Ref:
            http://www.aclweb.org/anthology/D14-1181
            https://blog.csdn.net/chuchus/article/details/77847476
            http://aclweb.org/anthology/D17-1065

        Args:
            query_seqs:     (batch_size, query_max_seq_len)
            response_seqs:  (batch_size, response_max_seq_len)

            Assumed:
                batch_size=16
                max_seq_len=100
                emb_dim=300

            (16, 100) -> (16, 100, 300) -> (16, 1, 100, 300)
                
            (16, 1, 100, 64) 
            ├──> convs -----> unsqueeze(3) ------> pools --------> pred ───────────|
            ├── (16, 100, 100, 1) -> (16, 100, 100) -> (16, 100, 1) -> (16, 100) ──|
            ├── (16, 200,  99, 1) -> (16, 200,  99) -> (16, 200, 1) -> (16, 200) ──|
            ├── (16, 200,  98, 1) -> (16, 200,  98) -> (16, 200, 1) -> (16, 200) ──|
            ├── (16, 200,  97, 1) -> (16, 200,  97) -> (16, 200, 1) -> (16, 200) ──|
            ├── (16, 200,  96, 1) -> (16, 200,  96) -> (16, 200, 1) -> (16, 200) ──|
            ├── (16, 100,  95, 1) -> (16, 100,  95) -> (16, 100, 1) -> (16, 100) ──|   (cat)
            ├── (16, 100,  94, 1) -> (16, 100,  94) -> (16, 100, 1) -> (16, 100) ──|───────────> (16, 1720)
            ├── (16, 100,  93, 1) -> (16, 100,  93) -> (16, 100, 1) -> (16, 100) ──|           = (16, num_filters_sum)
            ├── (16, 100,  92, 1) -> (16, 100,  92) -> (16, 100, 1) -> (16, 100) ──|
            ├── (16, 100,  91, 1) -> (16, 100,  91) -> (16, 100, 1) -> (16, 100) ──|
            ├── (16, 160,  86, 1) -> (16, 160,  86) -> (16, 160, 1) -> (16, 160) ──|
            ├── (16, 160,  81, 1) -> (16, 160,  81) -> (16, 160, 1) -> (16, 160) ──|

        """


        # Response

        ## (batch_size, 1, max_seq_len, emb_dim)
        emb_q = self.embedding(query_seqs).unsqueeze(1)  

        ## [(batch_size, num_filter, conv_length)]
        convs_q = [F.relu(conv(emb_q)).squeeze(3) for conv in self.convs_q]

        ## [(batch_size, num_filter)]
        pools_q = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs_q]

        ## (batch_size, num_filters_sum)
        pred_q = torch.cat(pools_q, 1)
    

        # Condition

        ## (batch_size, 1, max_seq_len, emb_dim)
        emb_r = self.embedding(response_seqs).unsqueeze(1)
        
        ## [(batch_size, num_filter, conv_length)]
        convs_r = [F.relu(conv(emb_r)).squeeze(3) for conv in self.convs_r]

        ## [(batch_size, num_filter)]
        pools_r = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs_r]
        
        ## (batch_size, num_filters_sum)
        pred_r = torch.cat(pools_r, 1)



        ## (batch_size, 2 * num_filters_sum)
        pred = torch.cat([pred_q, pred_r], 1)


        """

        Highway Networks
        Ref: https://arxiv.org/pdf/1505.00387.pdf
             https://blog.csdn.net/guoyuhaoaaa/article/details/54093913

        """

        ## (batch_size, 2 * num_filters_sum)
        highway = self.highway(pred)
        # pred = F.sigmoid(highway) *  F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        pred = torch.sigmoid(highway) *  F.relu(highway) + (1. - torch.sigmoid(highway)) * pred

        ## (batch_size, num_classes=2)
        pred = F.log_softmax(self.lin(self.dropout(pred)), dim=1)
        return pred

    def init_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("embedding"): continue
            if param.requires_grad == True:
                if param.ndimension() >= 2:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.constant_(param, 0)


## -----------------------------------------------------------------------------------

from embedding.load_emb import embedding
from opts.dis_opts import dis_opts
from opts.cuda_opts import USE_CUDA

discriminator = Discriminator(embedding=embedding,
                              dropout=dis_opts.dropout, 
                              fixed_embeddings=dis_opts.fixed_embeddings)

if USE_CUDA:
    discriminator.cuda()