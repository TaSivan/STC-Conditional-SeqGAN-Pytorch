import torch
from torch.utils.data import DataLoader

from dataset.dis_dataset import DisDataset
from dataset.dis_collate_fn import dis_collate_fn
from opts.dis_opts import dis_opts

from dataset.gen_dataset import gen_dataset
from evaluator.responsor import responsor

def get_dis_iter(nums_data=None):
    dis_dataset = DisDataset(gen_dataset, responsor, nums_data=nums_data)
    
    dis_iter = DataLoader(dataset=dis_dataset,
                          batch_size=dis_opts.batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=dis_collate_fn)

    return dis_iter