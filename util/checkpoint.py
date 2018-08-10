import os
import torch

from helper import GEN_CHECKPOINT_DIR, DIS_CHECKPOINT_DIR

def load_checkpoint(checkpoint_path):
    # It's weird that if `map_location` is not given, it will be extremely slow.
    return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)


def save_gen_checkpoint(opts, experiment_name, encoder, decoder, encoder_optim, decoder_optim,
                    epoch, num_iters, loss, global_step):
    checkpoint = {
        'opts': opts,
        'global_step': global_step,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'encoder_optim_state_dict': encoder_optim.state_dict(),
        'decoder_optim_state_dict': decoder_optim.state_dict()
    }
    
    filename = '%s_epoch_%d_iter_%d_loss_%.2f_step_%d.pt' % (experiment_name, epoch, num_iters, loss, global_step)
    
    checkpoint_path = os.path.join(GEN_CHECKPOINT_DIR, filename)
    
    if not os.path.exists(GEN_CHECKPOINT_DIR):
        os.makedirs(GEN_CHECKPOINT_DIR)

    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def save_dis_checkpoint(opts, experiment_name, discriminator, dis_optim, epoch, num_iters, loss, accuracy, global_step):

    checkpoint = {
        'opts': opts,
        'global_step': global_step,
        'discriminator_state_dict': discriminator.state_dict(),
        'dis_optim_state_dict': dis_optim.state_dict(),
    }

    filename = '%s_epoch_%d_iter_%d_loss_%.2f_accu_%.2f_step_%d.pt' % (experiment_name, epoch, num_iters, loss, accuracy, global_step)

    checkpoint_path = os.path.join(DIS_CHECKPOINT_DIR, filename)
    
    if not os.path.exists(DIS_CHECKPOINT_DIR):
        os.makedirs(DIS_CHECKPOINT_DIR)

    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path