import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalDiscriminator(nn.Module):
    """A CNN for text classification

    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """
    def __init__(self, dropout=0.75, embedding=None, fixed_embeddings=False):
        super(ConditionalDiscriminator, self).__init__()

        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        if embedding is not None:
            self.embedding.weight.data.copy_(embedding.weight)
        self.fixed_embeddings = fixed_embeddings
        self.embedding.weight.requires_grad = self.fixed_embeddings
        self.word_vec_size = self.embedding.embedding_dim


        self.filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

        self.num_classes = 2

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embedding.embedding_dim)) for (n, f) in zip(self.num_filters, self.filter_sizes)
        ])

        self.highway = nn.Linear(2 * sum(self.num_filters), 2 * sum(self.num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(sum(self.num_filters), self.num_classes)
        self.softmax = nn.LogSoftmax()
        
        self.init_parameters()
    
    def forward(self, x, condition):
        """
        Ref:
            http://www.aclweb.org/anthology/D14-1181
            https://blog.csdn.net/chuchus/article/details/77847476
            http://aclweb.org/anthology/D17-1065

        Args:
            x: (batch_size * max_seq_len)

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
        emb_x = self.emb(x).unsqueeze(1)  

        ## [(batch_size, num_filter, conv_length)]
        convs_x = [F.relu(conv(emb_x)).squeeze(3) for conv in self.convs]

        ## [(batch_size, num_filter)]
        pools_x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs_x]

        ## (batch_size, num_filters_sum)
        pred_x = torch.cat(pools_x, 1)
    

        # Condition

        ## (batch_size, 1, max_seq_len, emb_dim)
        emb_c = self.emb(condition).unsqueeze(1)
        
        ## [(batch_size, num_filter, conv_length)]
        convs_c = [F.relu(conv(emb_c)).squeeze(3) for conv in self.convs]

        ## [(batch_size, num_filter)]
        pools_c = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs_c]
        
        ## (batch_size, num_filters_sum)
        pred_c = torch.cat(pools_c, 1)



        ## (batch_size, 2 * num_filters_sum)
        pred = torch.cat([pred_x, pred_c], 1)


        """

        Highway Networks
        Ref: https://arxiv.org/pdf/1505.00387.pdf
             https://blog.csdn.net/guoyuhaoaaa/article/details/54093913

        """

        ## (batch_size, 2 * num_filters_sum)
        highway = self.highway(pred)
        pred = F.sigmoid(highway) *  F.relu(highway) + (1. - F.sigmoid(highway)) * pred

        ## (batch_size, num_classes=2)
        pred = self.softmax(self.lin(self.dropout(pred)))
        return pred

    def init_parameters(self):
        for param in self.parameters():
            if param.requires_grad == True:
                if param.ndimension() >= 2:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.constant_(param, 0)