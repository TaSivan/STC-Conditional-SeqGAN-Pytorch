import opts.seed
from opts.gen_opts import gen_opts
from opts.dis_opts import dis_opts
from opts.GAN_opts import GAN_opts

from dataset.gen_dataset import gen_dataset
from dataset.gen_dataloader import get_gen_iter
from dataset.dis_dataloader import get_dis_iter
from scripts.write_dis_dataset import get_training_pairs

from models.encoder import encoder
from models.decoder import decoder
from models.generator import generator
from models.discriminator import discriminator

from optimizer.gen_optimizer import encoder_optim, decoder_optim
from optimizer.dis_optimizer import dis_optim

from trainer.gen_train_epoch import train_gen
from trainer.dis_train_epoch import train_dis
from trainer.adversarial_train_epoch import train_adversarial


if __name__ == "__main__":


    # Pretrain Generator using MLE

    gen_iter = get_gen_iter(gen_dataset=gen_dataset,
                            batch_size=gen_opts.batch_size)

    train_gen(encoder=encoder, 
              decoder=decoder, 
              encoder_optim=encoder_optim, 
              decoder_optim=decoder_optim, 
              num_epochs=gen_opts.num_epochs, 
              gen_iter=gen_iter, 
              save_every_step=gen_opts.save_every_step, 
              print_every_step=gen_opts.print_every_step)


    # Pretrain Discriminator

    training_pairs = get_training_pairs()

    dis_iter = get_dis_iter(training_pairs=training_pairs, num_workers=0)

    train_dis(discriminator=discriminator, 
              dis_optim=dis_optim, 
              num_epochs=dis_opts.num_epochs, 
              dis_iter=dis_iter, 
              save_every_step=dis_opts.save_every_step, 
              print_every_step=dis_opts.print_every_step)


    # Adversarial Training

    """
        Sometimes, use multiple workers for DataLoader would cause weird problem
        see this: https://github.com/pytorch/pytorch/issues/8976
    """
    gen_iter = get_gen_iter(gen_dataset=gen_dataset,
                            batch_size=GAN_opts.batch_size,
                            num_workers=2)

    
    train_adversarial(generator=generator,
                      discriminator=discriminator,
                      encoder_optim=encoder_optim,
                      decoder_optim=decoder_optim,
                      dis_optim=dis_optim, 
                      gen_iter=gen_iter,
                      gen_dataset=gen_dataset,
                      num_epochs=1, 
                      print_every_step=GAN_opts.G_print_every_step, 
                      save_every_step=GAN_opts.G_save_every_step,
                      num_rollout=GAN_opts.num_rollout)