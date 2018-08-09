import torch
from torch.utils.data import DataLoader

from dataset.gen_dataset import gen_dataset
from dataset.gen_collate_fn import collate_fn
from opts import opts

gen_iter = DataLoader(dataset=gen_dataset,
                        batch_size=opts.batch_size,
                        shuffle=True,
                        num_workers=4,
                        collate_fn=collate_fn)