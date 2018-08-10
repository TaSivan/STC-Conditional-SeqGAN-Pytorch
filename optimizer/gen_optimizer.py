import torch.optim as optim

from models.encoder import encoder
from models.decoder import decoder
from opts.gen_opts import LOAD_GEN_CHECKPOINT, gen_opts

encoder_optim = optim.Adam([p for p in encoder.parameters() if p.requires_grad], lr=gen_opts.learning_rate)
decoder_optim = optim.Adam([p for p in decoder.parameters() if p.requires_grad], lr=gen_opts.learning_rate)

if LOAD_GEN_CHECKPOINT:
    from opts.gen_opts import gen_checkpoint
    encoder_optim.load_state_dict(gen_checkpoint["encoder_optim_state_dict"])
    decoder_optim.load_state_dict(gen_checkpoint["decoder_optim_state_dict"])