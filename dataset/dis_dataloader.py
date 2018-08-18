import torch
from torch.utils.data import DataLoader
import random

from dataset.gen_dataloader import get_gen_iter
from dataset.dis_dataset import DisDataset
from dataset.dis_collate_fn import dis_collate_fn
from opts.dis_opts import dis_opts

from dataset.gen_dataset import get_gen_dataset
from models.generator import generator

def get_dis_iter(training_pairs=None, num_data=None, num_workers=4):
    
    if training_pairs:
        if num_data:
            dis_dataset = DisDataset(random.sample(training_pairs, num_data))
        else:
            dis_dataset = DisDataset(training_pairs)

    else:
        if num_data:
            gen_dataset = get_gen_dataset(num_data=num_data/2)
        else:
            from dataset.gen_dataset import gen_dataset

        # 64 is the batch_size which could work best
        gen_iter = get_gen_iter(gen_dataset=gen_dataset,
                                batch_size=64)
        
        dis_dataset = DisDataset(gen_iter, generator)
    
    
    dis_iter = DataLoader(dataset=dis_dataset,
                          batch_size=dis_opts.batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          collate_fn=dis_collate_fn)

    return dis_iter