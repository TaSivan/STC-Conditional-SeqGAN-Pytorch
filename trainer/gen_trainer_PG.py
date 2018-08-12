import torch
import torch.nn.functional as F

from helper import PAD ,SOS, EOS, UNK
from util.sequence_mask import sequence_mask
from opts.gen_opts import gen_opts
from loss.gan_loss import gan_loss

def gen_trainer_PG(src_sents, src_seqs, src_lens, encoder, decoder, encoder_optim, decoder_optim, USE_CUDA=True):

    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Last batch might not have the same size as we set to the `batch_size`
    batch_size = src_seqs.size(1)
    
    # Pack tensors to variables for neural network inputs (in order to autograd)
    src_lens = torch.LongTensor(src_lens)

    # Decoder's input
    input_seq = torch.LongTensor([SOS] * batch_size)
    
    
    # Store all decoder's outputs.
    # **CRUTIAL** 
    # Don't set:
    # >> decoder_outputs = Variable(torch.zeros(max_tgt_len, batch_size, decoder.vocab_size))
    # Varying tensor size could cause GPU allocate a new memory causing OOM, 
    # so we intialize tensor with fixed size instead:
    # `gen_opts.max_seq_len` is a fixed number, unlike `max_tgt_len` always varys.
    decoder_outputs = torch.zeros(gen_opts.max_seq_len, batch_size, decoder.vocab_size)
    out_lens = torch.zeros(batch_size).long()

    # Move variables from CPU to GPU.
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        src_lens = src_lens.cuda()
        input_seq = input_seq.cuda()
        decoder_outputs = decoder_outputs.cuda()
        
    # -------------------------------------
    # Training mode (enable dropout)
    # -------------------------------------
    encoder.train()
    decoder.train()
    
    # -------------------------------------
    # Zero gradients, since optimizers will accumulate gradients for every backward.
    # -------------------------------------
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
        
    # -------------------------------------
    # Forward encoder
    # -------------------------------------
    # encoder_outputs: (max_seq_len, batch_size=1, hidden_size)
    # encoder_hidden:  (num_layers, batch_size=1, num_directions * hidden_size)
    # -------------------------------------
    encoder_outputs, encoder_hidden = encoder(src_seqs)

    # -------------------------------------
    # Forward decoder
    # -------------------------------------
    # Initialize decoder's hidden state as encoder's last hidden state.
    decoder_hidden = encoder_hidden
    
    # Run through decoder one time step at a time.
    
    for t in range(gen_opts.max_seq_len):

        # decoder returns:
        # - decoder_output   : (batch_size, vocab_size)
        # - decoder_hidden   : (num_layers, batch_size, hidden_size)
        # - attention_weights: (batch_size, max_src_len)
        decoder_output, decoder_hidden, attention_weights = decoder(input_seq, decoder_hidden,
                                                                    encoder_outputs, src_lens)

        # Store decoder outputs.
        decoder_outputs[t] = decoder_output
        
        # Next input is current target
        # input_seq = tgt_seqs[t]
        prob, token_id = decoder_output.topk(1)     # (batch_size, 1), (batch_size, 1)

        if (token_id == EOS).sum().item() == 0:     # everybody finish generates
            out_lens += 1   # append EOS
            break
        else:
            out_lens += (token_id == EOS).view(-1).long()   # (batch)

        input_seq = token_id.view(-1)                       # (batch_size)


    max_out_len = out_lens.max().item()

    # (max_out_len, batch_size, vocab_size) -> (batch_size, max_out_len, vocab_size)
    decoder_outputs = decoder_outputs[:max_out_len].transpose(0,1)

    out_seqs = decoder_outputs.topk(1)[1].squeeze(2)    # (batch_size, max_out_len)
    mask = sequence_mask(out_lens)                      # (batch_size, max_out_len)
    if USE_CUDA:
        mask = mask.cuda()
    out_seqs.masked_fill_(1 - mask, PAD)                # (batch_size, max_out_len)

    fake_response_seqs = []
    for i, out_seq in enumerate(out_seqs):
        fake_response_seqs.append(out_seq[:out_lens[i]].tolist())   

    loss = gan_loss(src_sents=src_sents,
                    src_seqs=src_seqs, 
                    fake_response_seqs=fake_response_seqs, 
                    decoder_outputs=decoder_outputs, 
                    batch_size=batch_size, 
                    num_rollout=16, 
                    max_out_len=max_out_len)

    # -------------------------------------
    # Backward and optimize
    # -------------------------------------
    # Backward to get gradients w.r.t parameters in model.
    loss.backward()

    # Update parameters with optimizers
    encoder_optim.step()
    decoder_optim.step()

    num_words = out_lens.sum().item()

    return loss.item(), num_words