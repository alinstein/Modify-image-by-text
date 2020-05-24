#Code is referenced and modified from following site :https://github.com/google/tirg

import datasets
import torch
import torch.utils.data
import torchvision


def load_dataset(opt,domain_trans):
  """Loads the input datasets."""
  print ('Reading dataset ', opt.dataset)
  if opt.dataset == 'css3d':
    trainset = datasets.CSSDataset(
        path=opt.dataset_path,
        split='train',        
        stage=opt.stage,
        model='TIRG',
        domain_trans=domain_trans,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.CSSDataset(
        path=opt.dataset_path,
        split='test',
        stage=opt.stage,
        model='TIRG',
        domain_trans=domain_trans,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  print ('trainset size:', len(trainset))
  print ('testset size:', len(testset))
  return trainset, testset
