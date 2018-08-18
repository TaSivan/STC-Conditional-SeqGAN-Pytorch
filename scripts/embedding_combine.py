import os
import numpy as np
import pickle
from gensim.models import KeyedVectors, Word2Vec
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.join(BASE_DIR, "repository")

"""

Pre-trained embedding source
    
    Wiki:
        https://github.com/Kyubyong/wordvectors

    Weibo:
        https://github.com/Embedding/Chinese-Word-Vectors

"""

# Note that you must also have `zh.bin.syn0.npy`, `zh.bin.syn1neg.npy` in repository.
model_wiki = Word2Vec.load(os.path.join(REPO_DIR, "zh.bin"))
model_weibo = KeyedVectors.load_word2vec_format(os.path.join(REPO_DIR, "sgns.weibo.char"))


def load_emb(model):
    id2w = []
    w2id = {}
    n_words = 0
    embedding = np.zeros((len(model.vocab), model.vector_size)).astype('float32')

    for word in model.vocab.keys():
        id2w.append(word)
        w2id[word] = n_words
        embedding[n_words] = model[word]
        n_words += 1

    return embedding, id2w, w2id


embedding_wiki, id2w_wiki, w2id_wiki = load_emb(model_wiki.wv) 
embedding_weibo, id2w_weibo, w2id_weibo = load_emb(model_weibo) 


# Normalize

embedding_wiki_normalize = normalize(embedding_wiki)
embedding_weibo_normalize = normalize(embedding_weibo)

# PCA

common_keys = model_wiki.wv.vocab.keys() & model_weibo.vocab.keys()

concats = []
for key in common_keys:
    concats.append(np.concatenate([embedding_wiki_normalize[w2id_wiki[key]], embedding_weibo_normalize[w2id_weibo[key]]]))
concats = np.vstack(concats)

pca = PCA(n_components=300)
common_tokens_embedding = pca.fit_transform(concats)


# Combine

w2id = {}
id2w = []
n_words = 0

for word in model_weibo.vocab.keys():
    id2w.append(word)
    w2id[word] = n_words
    n_words += 1
    
for word in model_wiki.wv.vocab.keys():
    if word not in common_keys:
        id2w.append(word)
        w2id[word] = n_words
        n_words += 1

combine_embedding_model = np.zeros((n_words, 300), dtype=np.float32)

for word in model_weibo.vocab.keys():
    idx = w2id[word]
    if word not in common_keys:
        combine_embedding_model[idx] = embedding_weibo_normalize[w2id_weibo[word]]

for word in model_wiki.wv.vocab.keys():
    idx = w2id[word]
    if word not in common_keys:
        combine_embedding_model[idx] = embedding_wiki_normalize[w2id_wiki[word]]

for i, word in enumerate(common_keys):
    idx = w2id[word]
    combine_embedding_model[idx] = common_tokens_embedding[i]


# Save

if not os.path.exists(REPO_DIR):
    os.makedirs(REPO_DIR)

SAVE_EMB_FILE = os.path.join(REPO_DIR, "wiki_weibo_embedding_normalize.npy")
SAVE_IDX_FILE = os.path.join(REPO_DIR, "wiki_weibo_id2w_normalize.pkl")

with open(SAVE_EMB_FILE, 'wb') as f:
    np.save(f, combine_embedding_model)

with open(SAVE_IDX_FILE, 'wb') as f:
    pickle.dump(id2w, f)