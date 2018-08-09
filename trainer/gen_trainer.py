import torch

from loss.masked_cross_entropy import masked_cross_entropy
from opts import USE_PARALLEL
from helper import PAD ,SOS, EOS, UNK


def gen_trainer(src_seqs, tgt_seqs, src_lens, tgt_lens, encoder, 
                decoder, encoder_optim, decoder_optim, opts, USE_CUDA=True):    
    
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Last batch might not have the same size as we set to the `batch_size`
    batch_size = src_seqs.size(1)
    assert(batch_size == tgt_seqs.size(1))
    
    # Pack tensors to variables for neural network inputs (in order to autograd)

    src_lens = torch.LongTensor(src_lens)
    tgt_lens = torch.LongTensor(tgt_lens)

    # Decoder's input
    input_seq = torch.LongTensor([SOS] * batch_size)
    
    # Decoder's output sequence length = max target sequence length of current batch.
    max_tgt_len = tgt_lens.max()
    
    # Store all decoder's outputs.
    # **CRUTIAL** 
    # Don't set:
    # >> decoder_outputs = Variable(torch.zeros(max_tgt_len, batch_size, decoder.vocab_size))
    # Varying tensor size could cause GPU allocate a new memory causing OOM, 
    # so we intialize tensor with fixed size instead:
    # `opts.max_seq_len` is a fixed number, unlike `max_tgt_len` always varys.
    if USE_PARALLEL:
        decoder_outputs = torch.zeros(opts.max_seq_len, batch_size, decoder.module.vocab_size)
    else:
        decoder_outputs = torch.zeros(opts.max_seq_len, batch_size, decoder.vocab_size)

    # Move variables from CPU to GPU.
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        tgt_seqs = tgt_seqs.cuda()
        src_lens = src_lens.cuda()
        tgt_lens = tgt_lens.cuda()
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
    for t in range(max_tgt_len):
        
        # decoder returns:
        # - decoder_output   : (batch_size, vocab_size)
        # - decoder_hidden   : (num_layers, batch_size, hidden_size)
        # - attention_weights: (batch_size, max_src_len)
        decoder_output, decoder_hidden, attention_weights = decoder(input_seq, decoder_hidden,
                                                                    encoder_outputs, src_lens)

        # Store decoder outputs.
        decoder_outputs[t] = decoder_output
        
        # Next input is current target
        input_seq = tgt_seqs[t]
        
        
    # -------------------------------------
    # Compute loss
    # -------------------------------------
    loss = masked_cross_entropy(
        decoder_outputs[:max_tgt_len].transpose(0,1).contiguous(), 
        tgt_seqs.transpose(0,1).contiguous(),
        tgt_lens
    )
    

    # -------------------------------------
    # Backward and optimize
    # -------------------------------------
    # Backward to get gradients w.r.t parameters in model.
    loss.backward()
    
    # Update parameters with optimizers
    encoder_optim.step()
    decoder_optim.step()
    
    # return loss.item(), attention_weights
    # return loss.item() / tgt_lens.float().sum()

    return loss.item(), tgt_lens.sum().item()