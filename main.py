from opts.gen_opts import gen_opts
from opts.dis_opts import dis_opts

from dataset.gen_dataloader import gen_iter
from dataset.gen_dataset import gen_dataset
from dataset.dis_dataloader import get_dis_iter

from models.encoder import encoder
from models.decoder import decoder
from models.discriminator import discriminator

from optimizer.gen_optimizer import encoder_optim, decoder_optim
from optimizer.dis_optimizer import dis_optim

from trainer.gen_train_epoch import train_gen
from trainer.dis_train_epoch import train_dis
from trainer.adversarial_train_epoch import train_adversarial


if __name__ == "__main__":

    # Pretrain Generator using MLE

    train_gen(encoder=encoder, 
              decoder=decoder, 
              encoder_optim=encoder_optim, 
              decoder_optim=decoder_optim, 
              num_epochs=gen_opts.num_epochs, 
              gen_iter=gen_iter, 
              save_every_step=gen_opts.save_every_step, 
              print_every_step=gen_opts.print_every_step)


    # Pretrain Discriminator

    dis_iter = get_dis_iter()

    train_dis(discriminator=discriminator, 
              dis_optim=dis_optim, 
              num_epochs=dis_opts.num_epochs, 
              dis_iter=dis_iter, 
              save_every_step=dis_opts.save_every_step, 
              print_every_step=dis_opts.print_every_step)


    # Adversarial Training

    train_adversarial(encoder=encoder,
                      decoder=decoder,
                      discriminator=discriminator,
                      encoder_optim=encoder_optim,
                      decoder_optim=decoder_optim,
                      dis_optim=dis_optim, 
                      gen_iter=gen_iter,
                      gen_dataset=gen_dataset,
                      num_epochs=1, 
                      print_every_step=1, 
                      save_every_step=1000)