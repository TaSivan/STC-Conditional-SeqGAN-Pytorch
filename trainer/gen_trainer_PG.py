import torch

from loss.gan_loss import gan_loss

def gen_trainer_PG(src_seqs, src_lens, generator, encoder_optim, decoder_optim, num_rollout=16, USE_CUDA=True):
    """
        - src_seqs:     Tensor,     shape: (seq_len, batch_size)
        - src_lens:     list,       shape: [1] * batch_size
    """    

    # - out_seqs:           (batch_size, max_out_len)
    # - out_lens:           (batch_size)
    # - decoder_outputs:    (batch_size, max_out_len, vocab_size)
    out_seqs, out_lens, decoder_outputs = generator(src_seqs, src_lens, enable_dropout=True)

    loss = gan_loss(src_seqs, out_seqs, src_lens, out_lens, decoder_outputs, num_rollout=num_rollout)
    
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    loss.backward()

    encoder_optim.step()
    decoder_optim.step()

    num_words = out_lens.sum().item()

    del out_seqs, out_lens, decoder_outputs
    torch.cuda.empty_cache()

    return loss.item(), num_words