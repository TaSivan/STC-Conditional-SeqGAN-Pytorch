import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

class DisDataset(Dataset):
    def __init__(self, gen_dataset, responsor, nums_data=None):
        self.gen_dataset = gen_dataset
        self.responsor = responsor
        self.nums_data = nums_data
        self.dialogue_pairs, self.labels = self.load_data()

        print('='*100)
        print('Dataset Info:')
        print('- Number of training pairs: {}'.format(self.__len__()))
        print('='*100 + '\n')


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        query_sent, response_sent = self.dialogue_pairs[index]
        label = self.labels[index]
        query_seq = self.gen_dataset.tokens2ids(query_sent, self.gen_dataset.vocab.token2id, append_SOS=False, append_EOS=True)
        response_seq = self.gen_dataset.tokens2ids(response_sent, self.gen_dataset.vocab.token2id, append_SOS=False, append_EOS=True)

        return query_sent, response_sent, query_seq, response_seq, label


    def load_data(self):
        dialogue_pairs, labels = [], []

        data_range = np.arange(self.gen_dataset.__len__())
        np.random.shuffle(data_range)
        if self.nums_data:
            data_range = data_range[:self.nums_data]
            
        for index in tqdm(data_range):
            query_sent, real_response_sent, query_seq, real_response_seq = self.gen_dataset.__getitem__(index)
            
            fake_response_seq = self.responsor.forward(query_sent)
            fake_response_sent = self.responsor.ids2tokens(fake_response_seq)

            dialogue_pairs.append((query_sent, real_response_sent))
            labels.append(1)
            
            dialogue_pairs.append((query_sent, fake_response_sent))
            labels.append(0)

        return dialogue_pairs, labels