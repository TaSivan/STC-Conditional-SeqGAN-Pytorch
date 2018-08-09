import os
import torch

from helper import CHECKPOINT_DIR
from util.AttrDict import AttrDict
from util.checkpoint import load_checkpoint

# If enabled, load checkpoint.
LOAD_CHECKPOINT = False

if LOAD_CHECKPOINT:
    # Modify this path.
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "seq2seq_2018-08-09 12:17:24_epoch_2_iter_625_loss_5.65_step_1250.pt")
    checkpoint = load_checkpoint(checkpoint_path)
    opts = checkpoint['opts']

else:
    opts = AttrDict()

    # Configure models
    opts.word_vec_size = 300
    opts.rnn_type = 'LSTM'
    opts.hidden_size = 512
    opts.num_layers = 2
    opts.dropout = 0.3
    opts.bidirectional = True
    opts.attention = True
#     opts.share_embeddings = True
#     opts.pretrained_embeddings = True
    opts.fixed_embeddings = False
#     opts.tie_embeddings = True # Tie decoder's input and output embeddings

    # Configure optimization
#     opts.max_grad_norm = 2
    opts.learning_rate = 0.001
#     opts.weight_decay = 1e-5 # L2 weight regularization

    # Configure training
    opts.max_seq_len = 100 # max sequence length to prevent OOM.
    opts.num_epochs = 5
    opts.batch_size = 16
    opts.print_every_step = 500
    opts.save_every_step = 5000


""" Enable GPU training """
USE_CUDA = torch.cuda.is_available()
print('Use_CUDA={}'.format(USE_CUDA))
if USE_CUDA:
    device = torch.device("cuda" if USE_CUDA else "cpu")
    # print(device)
    # You can change device by `torch.cuda.set_device(device_id)`
    print('current_device={}'.format(torch.cuda.current_device()))


USE_PARALLEL = False
# USE_PARALLEL = True if torch.cuda.device_count() > 1 else False
print('Use_Parallel={}'.format(USE_PARALLEL))
if USE_PARALLEL:
    print("use", torch.cuda.device_count(), "GPUs")
