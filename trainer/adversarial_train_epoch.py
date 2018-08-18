import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from dataset.dis_dataloader import get_dis_iter
from dataset.dis_collate_fn import dis_collate_fn

from opts.gen_opts import gen_opts
from opts.GAN_opts import GAN_opts
from opts.cuda_opts import USE_CUDA

from util.checkpoint import save_gen_checkpoint

from trainer.gen_trainer_PG import gen_trainer_PG
from trainer.dis_train_epoch import train_dis

from evaluator.responsor import responsor


G_model_name = 'SeqGAN-Generator'
D_model_name = 'SeqGAN-Discriminator'
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
    experiment_name = '{}_{}'.format(G_model_name, savetime)

    checkpoint_path = save_gen_checkpoint(gen_opts, experiment_name, encoder, decoder, encoder_optim, 
                                          decoder_optim, epoch, num_iters, loss, global_step)
    
    print('='*100)
    print('Save checkpoint to "{}".'.format(checkpoint_path))
    print('='*100 + '\n')


def train_adversarial(generator, discriminator, encoder_optim, decoder_optim, dis_optim,
                      gen_iter, gen_dataset, num_epochs, print_every_step, save_every_step, num_rollout):

    save_total_words = 0
    print_total_words = 0
    save_total_loss = 0
    print_total_loss = 0

    global_step = 0

    print("BEFORE TRAINING")
    print('-'*50)
    print("post:")
    print(observe_query)
    print('-'*50)
    out_text = responsor.response(observe_query)
    print("response:")
    print(out_text)
    print('='*100 + '\n')


    for epoch in range(1, num_epochs+1):
        for batch_id, batch_data in tqdm(enumerate(gen_iter)):
            
            # G steps
            for _ in range(1):
            
                # Unpack batch data
                src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens = batch_data
                
                max_seq_len = max(src_lens)
                if max_seq_len > gen_opts.max_seq_len:
                    print('[!] Ignore batch: sequence length={} > max sequence length={}'.format(max_seq_len, gen_opts.max_seq_len))
                    continue

                loss, num_words = gen_trainer_PG(src_seqs, src_lens, generator, encoder_optim, 
                                                 decoder_optim, num_rollout=num_rollout, USE_CUDA=True)

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
                    
                    save_checkpoint_training(generator.encoder, generator.decoder, encoder_optim, 
                                             decoder_optim, epoch, batch_id + 1, save_loss, global_step)
                    
                    save_total_loss = 0
                    save_total_words = 0

                    del save_loss

                del src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, loss, num_words
                torch.cuda.empty_cache()
                

            # D steps
            
            dis_iter = get_dis_iter(num_data=GAN_opts.batch_size * GAN_opts.d_step_repeat_times,
                                    num_workers=1)

            train_dis(discriminator=discriminator,
                      dis_optim=dis_optim,
                      num_epochs=GAN_opts.dis_num_epoch,
                      dis_iter=dis_iter,
                      save_every_step=GAN_opts.D_save_every_step,
                      print_every_step=GAN_opts.D_print_every_step,
                      model_name=D_model_name)