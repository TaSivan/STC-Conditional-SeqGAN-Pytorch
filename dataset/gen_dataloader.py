import torch
from torch.utils.data import DataLoader

from dataset.gen_dataset import gen_dataset
from dataset.gen_collate_fn import gen_collate_fn
from opts.gen_opts import gen_opts

gen_iter = DataLoader(dataset=gen_dataset,
                      batch_size=gen_opts.batch_size,
                      shuffle=True,
                      num_workers=4,
                      collate_fn=gen_collate_fn)