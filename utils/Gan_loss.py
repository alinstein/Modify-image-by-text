#code modifed from https://github.com/hanzhanggit/StackGAN
import torch
from torch import nn
import torch.utils.data
import torchvision
import torchvision.utils as vutils

def compute_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels,
                               conditions):
    # Binary Cross Entropy loss function.                           
    criterion = torch.nn.BCELoss()
    batch_size = real_imgs.size(0)
    cond = conditions.detach()
    fake = fake_imgs.detach()
    #print("fake",fake.shape)
    real_features = netD(real_imgs)
    fake_features = netD(fake)
    # real pairs
    inputs = (real_features, cond)
    #print(real_features.shape, cond.shape)
    real_logits = netD.module.get_cond_logits(real_features, cond)
    errD_real = criterion(real_logits, real_labels)
    # wrong pairs
    inputs = (real_features[:(batch_size-1)], cond[1:])
    wrong_logits = netD.module.get_cond_logits(real_features[:(batch_size-1)], cond[1:])
    errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # fake pairs
    inputs = (fake_features, cond)
    #print("real featue",real_features.shape)
    #print("fake features",fake_features.shape)
    fake_logits = netD.module.get_cond_logits(fake_features, cond)
    errD_fake = criterion(fake_logits, fake_labels)
    if netD.module.get_uncond_logits is not None:

        real_logits =netD.module.get_uncond_logits(real_features)
        fake_logits = netD.module.get_uncond_logits(fake_features)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    #print(errD, errD_real.data, errD_wrong.data, errD_fake.data)    
    return errD, errD_real.item(), errD_wrong.item(), errD_fake.item()



def compute_generator_loss(netD, fake_imgs, real_labels, conditions):
    # Binary Cross Entropy loss function.
    criterion = torch.nn.BCELoss()
    cond = conditions.detach()

    fake_features =netD(fake_imgs)
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = netD.module.get_cond_logits(fake_features, cond)
    errD_fake = criterion(fake_logits, real_labels)
    if netD.module.get_uncond_logits is not None:
        fake_logits = netD.module.get_uncond_logits(fake_features)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    return errD_fake

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def save_img_results(data_img, fake, epoch, image_dir):
    num = 64
    fake = fake[0:num]
    # data_img is changed to [0,1]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples.png' % image_dir,
            normalize=True)
        # fake.data is still [-1, 1]
        vutils.save_image(
            fake.data, '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
