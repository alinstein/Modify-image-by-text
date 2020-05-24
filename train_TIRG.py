#Code is referenced and modified from following site :https://github.com/google/tirg

import argparse
import sys
import time
import os
import datasets
from Models import img_text_composition_models
from utils.load_data import load_dataset
import numpy as np
from tensorboardX import SummaryWriter
import test_retrieval
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm as tqdm
from config import parse_opt


torch.set_num_threads(3)


def train_loop(opt, texts,logger, trainset, testset):

  """Builds the model and related optimizer."""
  print ('Creating model and optimizer for', opt.model)
  if opt.model == 'imgonly':
    model = img_text_composition_models.SimpleModelImageOnly(
        texts, embed_dim=opt.embed_dim)
  elif opt.model == 'textonly':
    model = img_text_composition_models.SimpleModelTextOnly(
        texts, embed_dim=opt.embed_dim)
  elif opt.model == 'concat':
    model = img_text_composition_models.Concat(texts, embed_dim=opt.embed_dim)
  elif opt.model == 'tirg':
    model = img_text_composition_models.TIRG(texts, embed_dim=opt.embed_dim)
  elif opt.model == 'tirg_lastconv':
    model = img_text_composition_models.TIRGLastConv(
        texts, embed_dim=opt.embed_dim)
  else:
    print ('Invalid model', opt.model)
    print ('available: imgonly, textonly, concat, tirg or tirg_lastconv')
    sys.exit()
  
  if torch.cuda.is_available(): 
       model = model.cuda()



  # create optimizer
  params = []
  # low learning rate for pretrained layers on real image datasets
  if opt.dataset != 'css3d':
    params.append({
        'params': [p for p in model.img_model.fc.parameters()],
        'lr': opt.learning_rate
    })
    params.append({
        'params': [p for p in model.img_model.parameters()],
        'lr': 0.1 * opt.learning_rate
    })
  #Since learning rate for img_model model is defined above.
  #Following step assigns learning rate for other parameters in model.  
  params.append({'params': [p for p in model.parameters()]})
  for _, p1 in enumerate(params):  # remove duplicated params
    for _, p2 in enumerate(params):
      if p1 is not p2:
        for p11 in p1['params']:
          for j, p22 in enumerate(p2['params']):
            if p11 is p22:
              p2['params'][j] = torch.tensor(0.0, requires_grad=True)
  optimizer = torch.optim.SGD(
      params, lr=opt.learning_rate, momentum=0.9, weight_decay=opt.weight_decay)
  
  tr_writer = SummaryWriter(log_dir=os.path.join(opt.log_dir, "train"))
  va_writer = SummaryWriter(log_dir=os.path.join(opt.log_dir, "valid"))

  # Create log directory and save directory if it does not exist
  if not os.path.exists(opt.log_dir):
                       os.makedirs(opt.log_dir)
  if not os.path.exists(opt.save_dir):
                       os.makedirs(opt.save_dir)

  best_va_acc = 0  # to check if best validation accuracy   

  # Prepare checkpoint file and model file to save and load from  
  checkpoint_file = os.path.join(opt.save_dir, "checkpoint.pth")
  bestmodel_file = os.path.join(opt.save_dir, "best_model.pth")      

  # Check for existing training results. If it existst, and the configuration
    # is set to resume `config.resume_TIRG==True`, resume from previous training. If
    # not, delete existing checkpoint.
  if os.path.exists(checkpoint_file):
        if opt.resume_TIRG:

            print("Checkpoint found! Resuming")
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            best_va_acc = checkpoint['best_va_acc']
        else:
            os.remove(checkpoint_file)            



  best_va_acc=0

  print( 'Begin training')
  losses_tracking = {}
  it = 0
  epoch = -1
  tic = time.time()
  #Starting Training Process
  while it < opt.num_iters:
        epoch += 1

        # show/log stats
        print ('It', it, 'epoch', epoch, 'Elapsed time', round(time.time() - tic,
                                                            4), opt.comment)
        tic = time.time()
        for loss_name in losses_tracking:
            avg_loss = np.mean(losses_tracking[loss_name][-len(trainloader):])
            print ('    Loss', loss_name, round(avg_loss, 4))
            logger.add_scalar(loss_name, avg_loss, it)
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], it)

        # test in the model every 10 epoches
        if epoch % 10 == 0:
            print("REACHED epoch % 10 == 0")
            tests = []

            for name, dataset in [('train', trainset), ('test', testset)]:
                t = test_retrieval.test(opt, model, dataset)
                tests += [(name + ' ' + metric_name, metric_value)
                        for metric_name, metric_value in t]
            count=0   
            #tesing results are stored in logs         
            for metric_name, metric_value in tests:
                logger.add_scalar(metric_name, metric_value, it)
                print ('    ', metric_name, round(metric_value, 4))

                va_writer.add_scalar(metric_name,metric_value,epoch)

                count=count+1
                print("<CURRENT>",count,best_va_acc,metric_value)
                #Saving the model with model of higher Recall for K=1
                if best_va_acc < metric_value and count==6:
                    print("saving the best checkpoint")
                    print("Rewritting",best_va_acc, "by", metric_value)
                    best_va_acc=metric_value

                    print(epoch)
                    state = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'opt': opt,
                            }
                    torch.save(state, bestmodel_file)

        model.train()
        trainloader = trainset.get_loader(
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=opt.loader_num_workers)

        #loop for a epoch
        for data in tqdm(trainloader, desc='Training for epoch ' + str(epoch)):
            it += 1
            assert type(data) is list
            #loading the images 
            #Converting to cuda tensor if availalable
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
            # compute loss
            losses = []
            if opt.loss == 'soft_triplet':
                loss_value = model.compute_loss(
                    img1, mods, img2, soft_triplet_loss=True)
            elif opt.loss == 'batch_based_classification':
                loss_value = model.compute_loss(
                    img1, mods, img2, soft_triplet_loss=False)
            else:
                print('Invalid loss function', opt.loss)
                sys.exit()
            loss_name = opt.loss
            loss_weight = 1.0
            losses += [(loss_name, loss_weight, loss_value)]
            total_loss = sum([
                        loss_weight * loss_value
                        for loss_name, loss_weight, loss_value in losses
                        ])
            assert not torch.isnan(total_loss)
            losses += [('total training loss', None, total_loss)]

            # track losses
            for loss_name, loss_weight, loss_value in losses:
                if loss_name not in losses_tracking:
                    losses_tracking[loss_name] = []
                losses_tracking[loss_name].append(float(loss_value))

            # gradient descend
            #print(total_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Saving the loss and accuracy for tensorboardX
        tr_writer.add_scalar("Loss",total_loss,epoch)
        #Saving state model and other parameters
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_va_acc':best_va_acc,
                    }, checkpoint_file)    
      
        # decay learing rate
        if it >= opt.learning_rate_decay_frequency and it % opt.learning_rate_decay_frequency == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.1

  print('Finished training')

 
def main():
  opt = parse_opt()
  print ('Arguments:')
  for k in opt.__dict__.keys():
    print ('    ', k, ':', str(opt.__dict__[k]))
  #creating log file for saving results
  logger = SummaryWriter(comment=opt.comment)
  print ('Log files saved to', logger.file_writer.get_logdir())
  for k in opt.__dict__.keys():
    logger.add_text(k, str(opt.__dict__[k]))

  trainset, testset = load_dataset(opt,opt.domain_trans)
  train_loop(opt, [t for t in trainset.get_all_texts()],
                 logger, trainset, testset)
  logger.close()


if __name__ == '__main__':
  main()

