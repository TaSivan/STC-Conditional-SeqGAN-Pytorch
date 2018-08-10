import torch
from torch.utils.data import DataLoader

from dataset.dis_dataset import dis_dataset
from dataset.dis_collate_fn import dis_collate_fn
from opts.dis_opts import dis_opts

dis_iter = DataLoader(dataset=dis_dataset,
                      batch_size=dis_opts.batch_size,
                      shuffle=True,
                      num_workers=4,
                      collate_fn=dis_collate_fn)