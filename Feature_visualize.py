import sys
import time
import os
import datasets
from Models import img_text_composition_models
import numpy as np
from tensorboardX import SummaryWriter
import test_retrieval
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm as tqdm


from config import parse_opt, str2bool
from utils.load_data import load_dataset


def main():
        opt = parse_opt()
        print ('Arguments:')
        for k in opt.__dict__.keys():
            print ('    ', k, ':', str(opt.__dict__[k]))



        trainset, testset = load_dataset(opt,False)
        texts=[t for t in trainset.get_all_texts()]
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=2)

        dataiter = iter(trainloader)
        data = dataiter.next()
        #print(data)
        img1 = data["source_img_data"]
        img2 = data['target_img_data']
        #print("shape of images",img1.shape,img2.shape)

        img_grid1 = torchvision.utils.make_grid(img1)
        img_grid2 = torchvision.utils.make_grid(img2)

      
        writer = SummaryWriter(log_dir=os.path.join(opt.log_dir, "source"))
        writer2 = SummaryWriter(log_dir=os.path.join(opt.log_dir, "target"))
        if not os.path.exists(opt.log_dir):
                       os.makedirs(opt.log_dir)
        
        # write to tensorboard
        writer.add_image('source',img_grid1)
        writer2.add_image('target',img_grid2)

        model = img_text_composition_models.TIRG(texts, embed_dim=opt.embed_dim)
        if torch.cuda.is_available(): 
            model = model.cuda()

        checkpoint_file = os.path.join(opt.save_dir, "best_model_final.pth")
        checkpoint = torch.load(checkpoint_file,map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        if torch.cuda.is_available():
                img1 = img1.clone().detach().cuda()
        else:
                img1 = img1.clone().detach()


        if torch.cuda.is_available():
              img2 = img2.clone().detach().cuda()
        else:
              img2 = img2.clone().detach()


        mods = [data["mod"]["str"]]


        mod_img1 = model.compose_img_text(img1, mods[0])
        mod_img1 = model.normalization_layer(mod_img1)

        img1 = model.compose_img_text(img1, mods[0])

        img2 = model.extract_img_feature(img2)
        img2 = model.normalization_layer(img2)
        print(img1.shape,mod_img1.shape)
        metadata=['Source1','Source2','Comp_img1','Comp_img2','Target_1','Target_2']
        features=torch.cat((img1,mod_img1,img2),0)

        writer.add_embedding(features,
                    metadata=metadata
                )

if __name__ == '__main__':
  main()
