import torch
from torch.utils.data import DataLoader

# from dataset.gen_dataset import gen_dataset
from dataset.gen_collate_fn import gen_collate_fn
from opts.gen_opts import gen_opts

def get_gen_iter(gen_dataset, batch_size, num_workers=4):
    gen_iter = DataLoader(dataset=gen_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          collate_fn=gen_collate_fn)
    return gen_iter