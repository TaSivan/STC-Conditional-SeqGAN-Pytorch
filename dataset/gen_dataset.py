import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import Counter
import os, pickle, random

from helper import REPO_DIR
from util.AttrDict import AttrDict
from helper import PAD, SOS, EOS, UNK

class GenDataset(Dataset):
    def __init__(self, training_pairs, vocab=None, counter=None, filter_vocab=True, max_vocab_size=50000):
        """ Note: If src_vocab, tgt_vocab is not given, it will build both vocabs.
            Args: 
            - src_path, tgt_path: text file with tokenized sentences.
            - src_vocab, tgt_vocab: data structure is same as self.build_vocab().
        """
        print('='*100)
        print('- Loading and tokenizing training sentences...')
        self.src_sents, self.tgt_sents = self.load_sents(training_pairs)
    
        if vocab is None:
            if counter is None:
                print('- Building source counter...')
                self.counter = self.build_counter(training_pairs)
            else:
                self.counter = counter
            print('- Building source vocabulary...')
            self.vocab = self.build_vocab(self.counter, filter_vocab, max_vocab_size)
        else:
            self.vocab = vocab
                        
        print('='*100)
        print('Dataset Info:')
        print('- Number of training pairs: {}'.format(self.__len__()))
        print('- Vocabulary size: {}'.format(len(self.vocab.token2id)))
        print('='*100 + '\n')
    
    def __len__(self):
        return len(self.src_sents)
    
    def __getitem__(self, index):
        src_sent = self.src_sents[index]
        tgt_sent = self.tgt_sents[index]
        src_seq = self.tokens2ids(src_sent, self.vocab.token2id, append_SOS=False, append_EOS=True)
        tgt_seq = self.tokens2ids(tgt_sent, self.vocab.token2id, append_SOS=False, append_EOS=True)

        return src_sent, tgt_sent, src_seq, tgt_seq
    
    def load_sents(self, sent_pairs):
        src_sents, tgt_sents = zip(*sent_pairs)
        return src_sents, tgt_sents

    def build_counter(self, sents):
        counter = Counter()
        for p_sent, r_sent in tqdm(sents):
            counter.update(p_sent)
            counter.update(r_sent)
        return counter
    
    def build_vocab(self, counter, filter_vocab, max_vocab_size):
        vocab = AttrDict()
        vocab.token2id = {'<PAD>': PAD, '<SOS>': SOS, '<EOS>': EOS, '<UNK>': UNK}
        
        if filter_vocab:
            vocab.token2id.update({token: _id+4 for _id, (token, count) in tqdm(enumerate(counter.most_common(max_vocab_size)))})
        else:
            vocab.token2id.update({token: _id+4 for _id, (token, count) in tqdm(enumerate(counter.most_common()))})
        
        vocab.id2token = {v:k for k,v in tqdm(vocab.token2id.items())}    
        return vocab
    
    def tokens2ids(self, tokens, token2id, append_SOS=True, append_EOS=True):
        seq = []
        if append_SOS: seq.append(SOS)
        seq.extend([token2id.get(token, UNK) for token in tokens])
        if append_EOS: seq.append(EOS)
        return seq

## -----------------------------------------------------------------------------------

from opts.gen_opts import gen_opts

with open(os.path.join(REPO_DIR, "training_pairs_seg.pkl"), 'rb') as f:
    training_pairs = pickle.load(f)

def get_gen_dataset(num_data=None):
    if num_data:
        assert(num_data <= len(training_pairs))
        gen_dataset = GenDataset(training_pairs=random.sample(training_pairs, num_data),
                                 filter_vocab=gen_opts.filter_vocab,
                                 max_vocab_size=gen_opts.max_vocab_size)        
    else:
        gen_dataset = GenDataset(training_pairs=training_pairs,
                                 filter_vocab=gen_opts.filter_vocab,
                                 max_vocab_size=gen_opts.max_vocab_size)
    
    return gen_dataset

gen_dataset = get_gen_dataset()