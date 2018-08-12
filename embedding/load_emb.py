import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from tqdm import tqdm

from helper import REPO_DIR
from dataset.gen_dataset import gen_dataset
from helper import PAD ,SOS, EOS, UNK

def load_emb(pretrained_emb, pretrained_w2id, vocab):

    word_vec_size = pretrained_emb.shape[1]
    vocab_size = len(vocab.token2id)
    embedding = nn.Embedding(vocab_size, word_vec_size, padding_idx=PAD)
    unk_count = 0
    
    for token, index in tqdm(vocab.token2id.items()):
        if index == PAD:
            continue
        elif index in [SOS, EOS, UNK]:
            embedding.weight[index] = torch.randn(word_vec_size)
        elif token in pretrained_w2id:
            idx = pretrained_w2id[token]
            embedding.weight[index] = torch.from_numpy(pretrained_emb[idx])
        else:
            embedding.weight[index] = embedding.weight[UNK]
            unk_count += 1
    
    print('- Unknown word count: {}'.format(unk_count))
    print('='*100 + '\n')
        
    return embedding

## -----------------------------------------------------------------------------------

with open(os.path.join(REPO_DIR, "wiki_weibo_embedding_normalize.npy"), "rb") as f:
    pretrained_embedding_model = np.load(f)

with open(os.path.join(REPO_DIR, "wiki_weibo_id2w_normalize.pkl"), "rb") as f:
    pretrained_id2w = pickle.load(f)

pretrained_w2id = {}
for i, w in enumerate(pretrained_id2w):
    pretrained_w2id[w] = i

embedding = load_emb(pretrained_embedding_model, pretrained_w2id, gen_dataset.vocab)
print(embedding)