import torch
from torch.utils.data import Dataset
from tqdm import tqdm
# import numpy as np
import random

class DisDataset(Dataset):
    def __init__(self, training_pairs=None, gen_iter=None, generator=None):
        
        if training_pairs:
            self.dialogue_pairs, self.labels = self.load_data_by_training_pairs(training_pairs)
        else:
            self.dialogue_pairs, self.labels = self.load_data_by_generator(gen_iter, generator)

        print('='*100)
        print('Dataset Info:')
        print('- Number of training pairs: {}'.format(self.__len__()))
        print('='*100 + '\n')


    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, index):
        query_seq, response_seq = self.dialogue_pairs[index]
        label = self.labels[index]

        # - query_seq:      (seq_len)
        # - response_seq:   (seq_len)
        # - label:          int, 1 or 0
        return query_seq, response_seq, label


    def load_data_by_training_pairs(self, training_pairs):
        dialogue_pairs, labels = [], []
        for (query_seqs, response_seqs), label in tqdm(training_pairs):
            
            query_seqs = torch.LongTensor(query_seqs)
            response_seqs = torch.LongTensor(response_seqs)
            
            dialogue_pairs.append((query_seqs, response_seqs))
            labels.append(label)

        return dialogue_pairs, labels


    def load_data_by_generator(self, gen_iter, generator):
        dialogue_pairs, labels = [], []
        for batch_id, batch_data in tqdm(enumerate(gen_iter)):
            # - sents:  [ [token] * seq_len ] * batch_size
            # - seqs:   (seq_len, batch_size)
            # - lens:   [1] * batch_size
            query_sents, response_sents, query_seqs, response_seqs, query_lens, response_lens = batch_data

            batch_size = query_seqs.size(1)

            # - out_seqs:           (batch_size, max_out_len)
            # - out_lens:           (batch_size)
            # - decoder_outputs:    (batch_size, max_out_len, vocab_size)
            out_seqs, out_lens, decoder_outputs = generator(query_seqs, query_lens, enable_dropout=False)

            # (seq_len, batch_size) -> (batch_size, seq_len)
            query_seqs = query_seqs.t()         
            response_seqs = response_seqs.t()

            # - seq: (max_seq_len)
            real_pairs = list(zip(query_seqs, response_seqs))  # [(query_seq, response_seq)] * batch_size
            fake_pairs = list(zip(query_seqs, out_seqs))       # [(query_seq, out_seq)] * batch_size

            dialogue_pairs.extend(real_pairs)
            labels.extend([1] * batch_size)
            
            dialogue_pairs.extend(fake_pairs)
            labels.extend([0] * batch_size)

        return dialogue_pairs, labels