import torch
import torch.nn.functional as F

from util.sequence_mask import sequence_mask

def masked_cross_entropy(logits, target, length):

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
        
    The code is same as:
    
    weight = torch.ones(tgt_vocab_size)
    weight[padding_idx] = 0
    criterion = nn.CrossEntropyLoss(weight.cuda(), size_average)
    loss = criterion(logits_flat, losses_flat)
    
    """
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))

    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=1)

    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    
    # losses_flat: (batch * max_len, 1)
    losses_flat = - torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())

    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))

    # Note: mask need to bed casted to float!
    losses = losses * mask.float()

    # loss = losses.sum() / mask.float().sum()
    loss = losses.sum()
    
    return loss