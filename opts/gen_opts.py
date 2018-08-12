import os
import torch

from helper import GEN_CHECKPOINT_DIR

from util.AttrDict import AttrDict
from util.checkpoint import load_checkpoint

# If enabled, load checkpoint.
LOAD_GEN_CHECKPOINT = False

if LOAD_GEN_CHECKPOINT:
    # Modify this path.
    gen_checkpoint_path = os.path.join(GEN_CHECKPOINT_DIR, "<Your filename>")
    gen_checkpoint = load_checkpoint(gen_checkpoint_path)
    gen_opts = gen_checkpoint['opts']

else:
    gen_opts = AttrDict()

    # Configure models
    gen_opts.word_vec_size = 300
    gen_opts.rnn_type = 'LSTM'
    gen_opts.hidden_size = 512
    gen_opts.num_layers = 2
    gen_opts.dropout = 0.3
    gen_opts.bidirectional = True
    gen_opts.attention = True
    gen_opts.fixed_embeddings = False

    # Configure optimization
    gen_opts.learning_rate = 0.001
    
    # Configure training
    gen_opts.max_seq_len = 100 # max sequence length to prevent OOM.
    gen_opts.batch_size = 16
    gen_opts.num_epochs = 1
    gen_opts.print_every_step = 500
    gen_opts.save_every_step = 5000
    
    # Configure vocabulary size
    gen_opts.filter_vocab = False
    gen_opts.max_vocab_size = 50000 # work only if filter_vocab is True
