import os
import torch

from helper import DIS_CHECKPOINT_DIR

from util.AttrDict import AttrDict
from util.checkpoint import load_checkpoint

# If enabled, load checkpoint.
LOAD_DIS_CHECKPOINT = False

if LOAD_DIS_CHECKPOINT:
    # Modify this path.
    dis_checkpoint_path = os.path.join(DIS_CHECKPOINT_DIR, "<Your filename>")
    dis_checkpoint = load_checkpoint(dis_checkpoint_path)
    dis_opts = dis_checkpoint['opts']

else:
    dis_opts = AttrDict()

    # Configure models
    dis_opts.dropout = 0.75
    dis_opts.fixed_embeddings = False

    # Configure optimization
    dis_opts.learning_rate = 0.001
    dis_opts.momentum = 0.9
    
    # Configure training
    dis_opts.conv_padding_len = 20
    dis_opts.batch_size = 64
    dis_opts.num_epochs = 2
    dis_opts.print_every_step = 500
    dis_opts.save_every_step = 5000
    # dis_opts.num_epochs = 2
    # dis_opts.print_every_step = 50
    # dis_opts.save_every_step = 100
    
    # Configure adversarial training
    dis_opts.adversarial_num_epoch = 1