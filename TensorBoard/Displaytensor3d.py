import argparse
import sys
import time
import os
import dataset2
import img_text_composition_models
import numpy as np
from tensorboardX import SummaryWriter
import test_retrieval
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm as tqdm


from new_main import str2bool,load_dataset1


def parse_opt():
  """Parses the input arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', type=str, default='')
  parser.add_argument('--comment', type=str, default='test_notebook')
  parser.add_argument('--dataset', type=str, default='css3d')
  parser.add_argument('--dataset_path', type=str, default='../CSS')
  parser.add_argument('--model', type=str, default='TIRGR')
  parser.add_argument('--embed_dim', type=int, default=512)
  parser.add_argument('--learning_rate', type=float, default=1e-2)
  parser.add_argument('--learning_rate_decay_frequency', type=int, default=9999999)
  parser.add_argument("--log_dir", type=str,
                       default="./logs1",
                       help="Directory to save logs and current model")
  parser.add_argument("--save_dir", type=str,
                       default="./save",
                       help="Directory to save the best model")
  parser.add_argument("--resume", type=str2bool,
                       default=True,
                       help="Whether to resume training from existing checkpoint")                     
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--weight_decay', type=float, default=1e-6)
  parser.add_argument('--num_iters', type=int, default=210000)
  parser.add_argument('--loss', type=str, default='soft_triplet')
  parser.add_argument('--loader_num_workers', type=int, default=4)
  parser.add_argument("--rep_intv", type=int,
                       default=2,
                       help="Report interval")
  
  args = parser.parse_args()
  return args

def main():
        opt = parse_opt()
        print ('Arguments:')
        for k in opt.__dict__.keys():
            print ('    ', k, ':', str(opt.__dict__[k]))



        trainset, testset = load_dataset1(opt)
        texts=[t for t in trainset.get_all_texts()]
        batch=2
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch)

        dataiter = iter(trainloader)
        data = dataiter.next()


        #print(data)
        img1 = data["3D"]["source_img_data"]
        img2 = data["3D"]['target_img_data']
        
        img3 = data["2D"]["source_img_data"]
        img4 = data["2D"]['target_img_data']






        img_grid1 = torchvision.utils.make_grid(img1)
        img_grid2 = torchvision.utils.make_grid(img2)

        img_grid3 = torchvision.utils.make_grid(img3)
        img_grid4 = torchvision.utils.make_grid(img4)

      
        writer = SummaryWriter(log_dir=os.path.join(opt.log_dir, "source"))
        writer2 = SummaryWriter(log_dir=os.path.join(opt.log_dir, "target"))

        writer3 = SummaryWriter(log_dir=os.path.join(opt.log_dir, "source1"))
        writer4 = SummaryWriter(log_dir=os.path.join(opt.log_dir, "target1"))

        if not os.path.exists(opt.log_dir):
                       os.makedirs(opt.log_dir)
        
        # write to tensorboard
        writer.add_image('source',img_grid1)
        writer2.add_image('target',img_grid2)

        writer3.add_image('source',img_grid3)
        writer4.add_image('target',img_grid4)
        #TIRG
        #TIRGRevGrad
        model = img_text_composition_models.TIRGRevGrad(texts, embed_dim=opt.embed_dim)
        if torch.cuda.is_available(): 
                  model = model.cuda()
        #checkpointGR
        checkpoint_file = os.path.join(opt.save_dir, "checkpointGR.pth")
        checkpoint = torch.load(checkpoint_file,map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        # writer.add_graph(model, img1)
        # writer.close()

        # img1 = torch.from_numpy(img1).float()
        if torch.cuda.is_available():
                img1 = img1.clone().detach().cuda()
        else:
                img1 = img1.clone().detach()

        if torch.cuda.is_available():
                img3 = img3.clone().detach().cuda()
        else:
                img3 = img3.clone().detach()
        
        if torch.cuda.is_available():
                img4 = img4.clone().detach().cuda()
        else:
                img4 = img4.clone().detach()

        print(img1.shape,img3.shape)
        if torch.cuda.is_available():
              img2 = img2.clone().detach().cuda()
        else:
              img2 = img2.clone().detach()

        # print(data)
        print(data["3D"]["mod"]["str"])      
            
        # mods = [(d["mod"]['str']) for d in data]
        mods = [data["3D"]["mod"]["str"]]

        print(mods)

        model.eval()
        mod_img1 = model.compose_img_text(img1, mods[0])
        mod_img1 = model.normalization_layer(mod_img1)

        img1 = model.compose_img_text(img1, mods[0])

        img2 = model.extract_img_feature(img2)
        img2 = model.normalization_layer(img2)
        
        mod_img3 = model.compose_img_text(img3.float() , mods[0])
        mod_img3 = model.normalization_layer(mod_img3)
        img3 = model.compose_img_text(img3.float() , mods[0])
        img4 = model.extract_img_feature(img4.float())
        img4 = model.normalization_layer(img4)



        print(img1.shape,mod_img1.shape)
        m=np.concatenate((np.zeros(3*batch),np.ones(3*batch)))  
        #x=np.array((0,1,2,3,4,5,6,7,8)).astype(int)
        #m=np.concatenate((x,np.ones(3*batch))  
        
        metadata=m.astype(int).tolist()
        metadata=[0,1,2,3,4,5,6,6,6,7,7,7]
        print(metadata)

        

        features=torch.cat((img1,mod_img1,img2,img3,mod_img3,img4),0)
        print(features.shape)

        writer.add_embedding(features,
                    metadata=metadata,
                )


    





if __name__ == '__main__':
  main()
