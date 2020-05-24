import argparse
import sys
import time
import os, errno
import datasets
from Models import img_text_composition_models
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import nn
import torch.utils.data
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm as tqdm
import pickle
from Models.StackGAN_Model import weights_init, STAGE1_G, STAGE1_D,STAGE2_D,STAGE2_G
import torchvision.utils as vutils
import sys
from config import parse_opt
from utils.Gan_loss import compute_discriminator_loss,compute_generator_loss
from utils.Gan_loss import save_img_results,KL_loss
torch.set_num_threads(3)



def train(opt, trainset):

    # TIRG model for extracting features
    texts = [t for t in trainset.get_all_texts()]
    model = img_text_composition_models.TIRG(texts, embed_dim=opt.embed_dim)
    if torch.cuda.is_available(): 
       model = model.cuda()
    #loading CSS dataset
    trainloader = trainset.get_loader(
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=opt.loader_num_workers)
    
    
    #making the dictionary for saving 
    if not os.path.exists(opt.save_dir):
        try:
                        os.makedirs(opt.save_dir)    
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    #Loading best model
    bestmodel_file = os.path.join(opt.save_dir, "best_model.pth") 

    if os.path.exists(bestmodel_file):
            print("Checkpoint found! Resuming")
            print("Using this model to extract features")
            if torch.cuda.is_available(): 
                checkpoint = torch.load(bestmodel_file)
            else:
                checkpoint = torch.load(bestmodel_file,map_location='cpu' )  
            #loading the checkpoint model and other paramerters using load_state_dict function
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
            #load the weight for TIRG model
            print("Please give the TIRG model is ./save dictionary")
            sys.exit()


    #Changing the model to eval mode to extract features
    model.eval()


    #checking if the feature set is already extracted
    #if the features are extracted this step will be skipped
    #Also checks the features for Stage 2 or Stage 1 is available
    if opt.stage==1:
        feature = os.path.join(opt.save_dir, "GAN1.pkl") 
    else:    
        feature = os.path.join(opt.save_dir, "GAN2a.pkl") 
  
    #following block extract features using TIRG model
    #to reduce the computational time instead of computing each time
    if  os.path.exists(feature):
        print("Features found! Resuming")
    else:
        print("Extracting Features")        
        with open(feature, 'wb') as f:
            Data_list=[]
            for data in tqdm(trainloader, desc='Feature Extraction' ):
                #loading the files(target image and source image and text)
                #before passing into TIRG model
                assert type(data) is list
                img1 = np.stack([d['source_img_data'] for d in data])
                img1 = torch.from_numpy(img1).float()
                if torch.cuda.is_available():
                    img1 = img1.clone().detach().cuda()
                else:
                    img1 = img1.clone().detach()
                        
                img2 = np.stack([d['target_img_data'] for d in data])
                img2 = torch.from_numpy(img2).float()
                if torch.cuda.is_available():
                    img2 = img2.clone().detach().cuda()
                else:
                    img2 = img2.clone().detach()
                
                mods = [str(d['mod']['str']) for d in data]
                mods = [t for t in mods]

                #extracting the features from the model as per the paper
                #composed Features
                img1_f = model.compose_img_text(img1, mods)
                img1_f = model.normalization_layer(img1_f)
                #Target Features
                img2_f = model.extract_img_feature(img2)
                img2_f = model.normalization_layer(img2_f)
                #Source Features
                img3_f = model.extract_img_feature(img1)
                img3_f = model.normalization_layer(img3_f)
                
                feature_dict = {"img1_f":img1_f,"img1":img1,"mods":mods,"img3_f":img3_f,"img2_f":img2_f,"img2":img2}
                #extracted features are saved for later use
                pickle.dump(feature_dict, f)  
        f.close()

    #Looking for GPU or CPU for processing
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    
    #making the dictionary for saving ( STAGE1 )
    if not os.path.exists(opt.save_dir_GAN):
                        os.makedirs(opt.save_dir_GAN)
    #location of stage 1 model weigts
    bestmodel_gan=opt.bestmodel_gan
    bestmodel_file_GAN = os.path.join(opt.save_dir_GAN, bestmodel_gan) 
    
    #Model creation (GAN)
    if opt.stage == 1:

        # Create the generator.
        netG = STAGE1_G()
        # Apply the weights_init() function to randomly initialize all
        # weights to mean=0.0, stddev=0.2
        netG.apply(weights_init)
        # Print the model.
        print(netG)
        # Create the discriminator.
        netD = STAGE1_D()
        # Apply the weights_init() function to randomly initialize all
        # weights to mean=0.0, stddev=0.2
        netD.apply(weights_init)
        # Print the model.
        print(netD)
        #folder of logs
        log_dir = os.path.join(opt.log_dir1) 
        #folder for generated images
        save_image= opt.image_GAN
        model_save=opt.save_dir_GAN
        if not os.path.exists(model_save):
                        os.makedirs(model_save) 
        
        #Choice for more than 2 GPU
        #train Second stage of GPU with atleast 2 GPUS
        # if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG = nn.DataParallel(netG)
        netG.to(device)
        netD = nn.DataParallel(netD)
        netD.to(device)
        # else:   
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     netG.to(device)
        #     netD.to(device)
        #If the weights exist loading the weights
        if os.path.exists(bestmodel_file_GAN):
                print("Resuming from the previous point ! ")
                if torch.cuda.is_available(): 
                    checkpoint = torch.load(bestmodel_file_GAN)
                else:
                    checkpoint = torch.load(bestmodel_file_GAN,map_location='cpu' )  
                #loading the checkpoint model and other paramerters using load_state_dict function
                netG.load_state_dict(checkpoint['generator'])
                netD.load_state_dict(checkpoint['discriminator'])

    #Training of stage II 
    #for training stage II, stage I should be trained before
    else :
        print("Stage II training started")
        #loading the stage1 generator model
        Stage1_G = STAGE1_G()
        Stage1_G=torch.nn.DataParallel(Stage1_G)
        
        #passing the stage 1 model to stage 2 
        netG = STAGE2_G(Stage1_G)
        #intializing the weights
        netG.apply(weights_init)
        print(netG)

        log_dir = os.path.join(opt.log_dir2) 
        #folder for generated images
        save_image= opt.image_GAN2
        #save model
        model_save=opt.save_dir_GAN2
        if not os.path.exists(model_save):
                        os.makedirs(model_save) 

        #Loading stage II discriminator
        netD = STAGE2_D()
        print(netD)
        netD.apply(weights_init)   
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            netG = torch.nn.DataParallel(netG)
            netG.to(device)
            netD = torch.nn.DataParallel(netD)
            netD.to(device)
        else:   
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            netG.to(device)
            netD.to(device)   

        if os.path.exists(bestmodel_file_GAN):
            state_dict =  torch.load(bestmodel_file_GAN)
            netG.module.STAGE1_G.load_state_dict(state_dict['generator'])
            netG.module.STAGE1_G=netG.module.STAGE1_G.module
            print('Load Stage I weights')

        else:
            #Stage I model weights are required to train stage II
            print("Please give the Stage1_G weights path")
            sys.exit()      



    nz = opt.nz  #dimension of random vector
    batch_size = opt.batch_size
    #generating the random latent vector 
    noise = Variable(torch.FloatTensor(batch_size, nz))
    fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1),
                    volatile=True)
    #Labels 1s with batch size for real labels
    #Labels 0s with batch size for fake labels
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
    
    #passing the variables to GPU
    noise, fixed_noise = noise.to(device), fixed_noise.to(device)
    real_labels, fake_labels = real_labels.to(device), fake_labels.to(device)

    #Loading the Generator, discriminator learning rate  for the parmeters
    generator_lr = opt.GENERATOR_LR
    discriminator_lr = opt.DISCRIMINATOR_LR
    lr_decay_step = opt.LR_DECAY_EPOCH
    
    # Optimizer for the discriminator.
    optimizerD = \
        torch.optim.Adam(netD.parameters(),
                    lr=opt.DISCRIMINATOR_LR, betas=(0.5, 0.999))
    netG_para = []
    #Applicable if 2nd stage training
    #for stage II training Stage I generator weights are to be 
    #conatant through the training period
    for p in netG.parameters():
        if p.requires_grad:
            netG_para.append(p)
    # Optimizer for the generator.
    optimizerG = torch.optim.Adam(netG_para,
                            lr=opt.GENERATOR_LR,
                            betas=(0.5, 0.999))
    count = 0  
    #Folder for storing logs
    #making the dictionary for saving logs
    if not os.path.exists(log_dir):
                        os.makedirs(log_dir)  
    #making the dictionary for saving images
    if not os.path.exists(save_image):
                        os.makedirs(save_image) 
    summary_writer = SummaryWriter(log_dir)
    
    #bestmodel_file_GAN = os.path.join(opt.save_dir_GAN, "model_epoch_120.pth.pth") 
    #Total number of batches
    length = (19000//opt.batch_size)
    #Reading the stored features, that we extracted using TIRG model 
    file = open(feature, 'rb')
    iters = 0

    print("Starting Training Loop...")

    for epoch in range(opt.nepochs):

        if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr
        count = 0
        file = open(feature, 'rb')
        for i in tqdm(range(length)):
            
            data = pickle.load(file)
            ######################################################
            # (1) Prepare training data
            ######################################################

            # Transfer data tensor to GPU/CPU (device)
            real_images = Variable(data['img2']).to(device)
            # Correct Conditions (Embedding features)
            embedding = Variable(data['img2_f']).to(device)
            
            #print(i)
            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            _, fake_imgs, mu, logvar =  netG(embedding, noise)

            ############################
            # (3) Update D network
            ###########################
            netD.zero_grad()
            errD, errD_real, errD_wrong, errD_fake = \
                compute_discriminator_loss(netD, real_images, fake_imgs,
                                            real_labels, fake_labels,
                                            mu)
            
            errD.backward()
            optimizerD.step()
            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            errG = compute_generator_loss(netD, fake_imgs,
                                            real_labels, mu)
            kl_loss = KL_loss(mu, logvar)
            errG_total = errG + kl_loss * 2.0
            #print(errG_total)
            errG_total.backward()
            optimizerG.step()

            count = count + 1
            if i % 100 == 0:

                    summary_writer.add_scalar('D_loss', errD.item(),epoch*length+i)
                    summary_writer.add_scalar('D_loss_real', errD_real,epoch*length+i)
                    summary_writer.add_scalar('D_loss_wrong', errD_wrong,epoch*length+i)
                    summary_writer.add_scalar('D_loss_fake', errD_fake,epoch*length+i)
                    summary_writer.add_scalar('G_loss', errG.item(),epoch*length+i)
                    summary_writer.add_scalar('KL_loss', kl_loss.item(),epoch*length+i)

                    # save the image result for each epoch
                    #inputs = (embedding, fixed_noise)
                    lr_fake, fake, _, _ = netG(embedding, fixed_noise)
                    save_img_results(real_images, fake, epoch,save_image)
                    if lr_fake is not None:
                        save_img_results(None, lr_fake, epoch, save_image)
            end_t = time.time()
            print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                    Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                    
                '''
                % (epoch, opt.nepochs, i,length,
                    errD.item(), errG.item(), kl_loss.item(),
                    errD_real, errD_wrong, errD_fake))

        if epoch % opt.save_epoch == 0:
            torch.save({
                'generator' : netG.state_dict(),

                'discriminator' : netD.state_dict(),
                'optimizerG' : optimizerG.state_dict(),
                'optimizerD' : optimizerD.state_dict(),
                'params' : opt
                }, os.path.join(model_save, 'model_epoch_{}.pth'.format(epoch)))
        file.close()

    torch.save({
                'generator' : netG.state_dict(),
                'discriminator' : netD.state_dict(),
                'optimizerG' : optimizerG.state_dict(),
                'optimizerD' : optimizerD.state_dict(),
                'params' : opt
                },  os.path.join(model_save,'model_final.pth'))



def load_dataset(opt):
  """Loads the input datasets."""
  print ('Reading dataset ', opt.dataset)
  if opt.dataset == 'css3d':
    trainset = datasets.CSSDataset(
        path=opt.dataset_path,
        stage=opt.stage,
        split='train',
        model='GAN',
        domain_trans='False',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  print ('trainset size:', len(trainset))
  return trainset


  
def main():
  #Loading the config details
  opt = parse_opt()
  print ('Arguments:')
  for k in opt.__dict__.keys():
    print ('    ', k, ':', str(opt.__dict__[k]))

  trainset = load_dataset(opt)

  train(opt, trainset)


if __name__ == '__main__':
  main()

  #code modifed from https://github.com/hanzhanggit/StackGAN