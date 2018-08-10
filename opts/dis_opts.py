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
    # dis_opts.word_vec_size = 300
    dis_opts.dropout = 0.75
    dis_opts.fixed_embeddings = False

    # Configure optimization
    dis_opts.learning_rate = 0.001
    
    # Configure training
    # dis_opts.max_seq_len = 100 # max sequence length to prevent OOM.
    dis_opts.conv_padding_len = 20
    dis_opts.num_epochs = 5
    dis_opts.batch_size = 16
    dis_opts.print_every_step = 500
    dis_opts.save_every_step = 5000