import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

from opts.dis_opts import dis_opts
from opts.cuda_opts import USE_CUDA
from util.checkpoint import save_dis_checkpoint
from trainer.dis_trainer import dis_trainer

model_name = 'cnn_classifier'

def print_statistics(epoch, num_epochs, num_iters, dis_iter, global_step, loss, accuracy):
    print('='*100)
    print('Training log:')
    print('- Epoch: {}/{}'.format(epoch, num_epochs))
    print('- Iter: {}/{}'.format(num_iters, dis_iter.__len__()))
    print('- Global step: {}'.format(global_step))
    print('- Loss: {}'.format(loss))
    print('- Accuracy: {}'.format(accuracy))
    print('='*100 + '\n')


def save_checkpoint_training(discriminator, dis_optim, epoch, num_iters, loss, accuracy, global_step):
    savetime = ('%s' % datetime.now()).split('.')[0]
    experiment_name = '{}_{}'.format(model_name, savetime)

    checkpoint_path = save_dis_checkpoint(dis_opts, experiment_name, discriminator, dis_optim,
                                          epoch, num_iters, loss, accuracy, global_step)

    print('='*100)
    print('Save checkpoint to "{}".'.format(checkpoint_path))
    print('='*100 + '\n')


def train_dis(discriminator ,dis_optim, num_epochs, dis_iter, save_every_step=5000, print_every_step=500):

    global_step = 0

    print_total_loss = 0
    print_total_corrects = 0
    print_total_num_data = 0

    save_total_loss = 0
    save_total_corrects = 0
    save_total_num_data = 0

    for epoch in range(1, num_epochs+1):
        for batch_id, batch_data in tqdm(enumerate(dis_iter)):
            query_sents, response_sents, query_seqs, response_seqs, query_lens, response_lens, labels = batch_data

            ## does it matter???
            # # Ignore batch if there is a long sequence.
            # max_seq_len = max(query_lens + response_lens)
            # if max_seq_len > dis_opts.max_seq_len:
            #     print('[!] Ignore batch: sequence length={} > max sequence length={}'.format(max_seq_len, dis_opts.max_seq_len))
            #     continue

            batch_size = query_seqs.size(0)

            # (batch,)
            labels = torch.tensor(labels)

            loss, num_corrects = dis_trainer(query_seqs, response_seqs, labels,
                                             discriminator, dis_optim, USE_CUDA=USE_CUDA)

            global_step += 1

            save_total_loss += loss
            save_total_corrects += num_corrects
            save_total_num_data += batch_size

            print_total_loss += loss
            print_total_corrects += num_corrects
            print_total_num_data += batch_size


            if (batch_id + 1) % print_every_step == 0:
                print_accuracy = print_total_corrects / print_total_num_data
                print_loss = print_total_loss / print_total_num_data

                print_statistics(epoch, num_epochs, batch_id + 1, dis_iter, global_step, print_loss, print_accuracy)

                print_total_loss = 0
                print_total_corrects = 0
                print_total_num_data = 0

                del print_accuracy, print_loss

            if (batch_id + 1) % save_every_step == 0:
                save_accuracy = save_total_corrects / save_total_num_data
                save_loss = save_total_loss / save_total_num_data

                save_checkpoint_training(discriminator, dis_optim, epoch, batch_id + 1, save_loss, save_accuracy, global_step)

                save_total_loss = 0
                save_total_corrects = 0
                save_total_num_data = 0

                del save_accuracy, save_loss


            del query_sents, response_sents, query_seqs, response_seqs, query_lens, response_lens, labels


        num_iters = dis_iter.__len__()

        if num_iters % print_every_step != 0:
            print_accuracy = print_total_corrects / print_total_num_data
            print_loss = print_total_loss / print_total_num_data

            print_statistics(epoch, num_epochs, batch_id + 1, dis_iter, global_step, print_loss, print_accuracy)

            print_total_loss = 0
            print_total_corrects = 0
            print_total_num_data = 0

            del print_accuracy, print_loss

        if num_iters % save_every_step != 0:
            save_accuracy = save_total_corrects / save_total_num_data
            save_loss = save_total_loss / save_total_num_data

            save_checkpoint_training(discriminator, dis_optim, epoch, num_iters, save_loss, save_accuracy, global_step)

            save_total_loss = 0
            save_total_corrects = 0
            save_total_num_data = 0

            del save_accuracy, save_loss

        del num_iters

## -----------------------------------------------------------------------------------



from dataset.dis_dataloader import dis_iter
from models.discriminator import discriminator
from optimizer.dis_optimizer import dis_optim

num_epochs = dis_opts.num_epochs
print_every_step = dis_opts.print_every_step
save_every_step = dis.opts.save_every_step

train_dis(discriminator=discriminator,
          dis_optim=dis_optim,
          num_epochs=num_epochs,
          dis_iter=dis_iter,
          save_every_step=save_every_step,
          print_every_step=print_every_step)
