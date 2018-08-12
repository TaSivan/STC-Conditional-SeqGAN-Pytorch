import torch
import torch.nn as nn

from models.discriminator import discriminator
from optimizer.dis_optimizer import dis_optim

criterion = nn.NLLLoss(reduction='sum')


def dis_trainer(query_seqs, response_seqs, labels, discriminator, dis_optim, USE_CUDA=True):

    batch_size = query_seqs.size(0)
    assert(batch_size == response_seqs.size(0))
    assert(batch_size == labels.size(0))

    # Training mode (enable dropout)
    discriminator.train()

    if USE_CUDA:
        query_seqs = query_seqs.cuda()
        response_seqs = response_seqs.cuda()
        labels = labels.cuda()

    # (batch_size, 2)
    # query_seqs:    (batch, max_seq_len)
    # response_seqs: (batch, max_seq_len)
    pred = discriminator(query_seqs, response_seqs)

    loss = criterion(pred, labels)
    num_corrects = (pred.max(1)[1] == labels).sum().item()

    dis_optim.zero_grad()
    loss.backward()
    dis_optim.step()

    return loss.item(), num_corrects