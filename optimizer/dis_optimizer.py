import torch.optim as optim

from models.discriminator import discriminator
from opts.dis_opts import LOAD_DIS_CHECKPOINT, dis_opts

dis_optim = optim.SGD([p for p in discriminator.parameters() if p.requires_grad], lr=dis_opts.learning_rate, momentum=dis_opts.momentum)

if LOAD_DIS_CHECKPOINT:
    from opts.dis_opts import dis_checkpoint
    dis_optim.load_state_dict(dis_checkpoint["dis_optim_state_dict"])