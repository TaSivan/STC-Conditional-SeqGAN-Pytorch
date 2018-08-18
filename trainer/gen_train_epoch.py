import torch

from datetime import datetime
from tqdm import tqdm

from opts.gen_opts import LOAD_GEN_CHECKPOINT, gen_opts
from opts.cuda_opts import USE_CUDA
from trainer.gen_trainer import gen_trainer
from evaluator.responsor import responsor
from util.checkpoint import save_gen_checkpoint


model_name = 'seq2seq'
observe_query = "没有高考，你拼得过官二代吗？"


def print_statistics(epoch, num_epochs, num_iters, gen_iter, global_step, loss):
    
    print('='*100)
    print('Training log:')
    print('- Epoch: {}/{}'.format(epoch, num_epochs))
    print('- Iter: {}/{}'.format(num_iters, gen_iter.__len__()))
    print('- Global step: {}'.format(global_step))
    print('- Loss: {}'.format(loss))

    print()
    
    print("post:")
    print(observe_query)
    print("response:")
    out_text = responsor.response(observe_query)
    print(out_text)
    
    print('='*100 + '\n')

def save_checkpoint_training(encoder, decoder, encoder_optim, decoder_optim, epoch, num_iters, loss, global_step):
    savetime = ('%s' % datetime.now()).split('.')[0]
    experiment_name = '{}_{}'.format(model_name, savetime)

    checkpoint_path = save_gen_checkpoint(gen_opts, experiment_name, encoder, decoder, encoder_optim, 
                                          decoder_optim, epoch, num_iters, loss, global_step)
    
    print('='*100)
    print('Save checkpoint to "{}".'.format(checkpoint_path))
    print('='*100 + '\n')


def train_gen(encoder, decoder, encoder_optim, decoder_optim,
              num_epochs, gen_iter, save_every_step=5000, print_every_step=500):

    # For saving checkpoint
    if LOAD_GEN_CHECKPOINT:
        from opts.gen_opts import gen_checkpoint
        global_step = gen_checkpoint['global_step']
    else:
        global_step = 0


    # --------------------------
    # Start training
    # --------------------------
    save_total_words = 0
    print_total_words = 0
    save_total_loss = 0
    print_total_loss = 0


    print("BEFORE TRAINING")
    print('-'*50)
    print("post:")
    print(observe_query)
    print('-'*50)
    out_text = responsor.response(observe_query)
    print("response:")
    print(out_text)
    print('='*100 + '\n')


    ### Ignore the last batch ?????
    for epoch in range(1, num_epochs+1):
        for batch_id, batch_data in tqdm(enumerate(gen_iter)):

            # Unpack batch data
            src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens = batch_data
            
            # Ignore the last batch if the batch size is less than that we set
            # if len(src_lens) < batch_size: continue

            # Ignore batch if there is a long sequence.
            max_seq_len = max(src_lens + tgt_lens)
            if max_seq_len > gen_opts.max_seq_len:
                print('[!] Ignore batch: sequence length={} > max sequence length={}'.format(max_seq_len, gen_opts.max_seq_len))
                continue
                
            # Train.
            loss, num_words = gen_trainer(src_seqs, tgt_seqs, src_lens, tgt_lens,
                                          encoder, decoder, encoder_optim, decoder_optim, gen_opts, USE_CUDA)
            
            # Statistics.
            global_step += 1
            save_total_loss += loss
            print_total_loss += loss
            save_total_words += num_words
            print_total_words += num_words

            # Print statistics.
            if (batch_id + 1) % print_every_step == 0:
                print_loss = print_total_loss / print_total_words
                
                print_statistics(epoch, num_epochs, batch_id+1, gen_iter, global_step, print_loss)     
                
                print_total_loss = 0
                print_total_words = 0

                del print_loss

            # Save checkpoint.
            if (batch_id + 1) % save_every_step == 0:
                save_loss = save_total_loss / save_total_words
                
                save_checkpoint_training(encoder, decoder, encoder_optim, decoder_optim, epoch, batch_id + 1, save_loss, global_step)
                
                save_total_loss = 0
                save_total_words = 0

                del save_loss
                
            # Free memory
            del src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, loss, num_words
            torch.cuda.empty_cache()       

        num_iters = gen_iter.__len__()

        if num_iters % print_every_step != 0:
            print_loss = print_total_loss / print_total_words

            print_statistics(epoch, num_epochs, num_iters, gen_iter, global_step, print_loss)
            
            print_total_loss = 0
            print_total_words = 0

            del print_loss
        
        if num_iters % save_every_step != 0:
            save_loss = save_total_loss / save_total_words

            save_checkpoint_training(encoder, decoder, encoder_optim, decoder_optim, epoch, num_iters, save_loss, global_step)
            
            save_total_loss = 0
            save_total_words = 0
            
            del save_loss

        del num_iters