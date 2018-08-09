import torch.optim as optim

from models.encoder import encoder
from models.decoder import decoder
from opts import LOAD_CHECKPOINT, opts

encoder_optim = optim.Adam([p for p in encoder.parameters() if p.requires_grad], lr=opts.learning_rate)
decoder_optim = optim.Adam([p for p in decoder.parameters() if p.requires_grad], lr=opts.learning_rate)

if LOAD_CHECKPOINT:
    from opts import checkpoint
    encoder_optim.load_state_dict(checkpoint["encoder_optim_state_dict"])
    decoder_optim.load_state_dict(checkpoint["decoder_optim_state_dict"])