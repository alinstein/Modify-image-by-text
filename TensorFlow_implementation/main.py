import argparse
import sys
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import datasets
from datasets import CSSDataset
from test import test1
from TIRGmodel import TIRG
from textmodel import TextLSTMModel
from tqdm import tqdm as tqdm
import PIL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import re
import time
import json
from glob import glob
import pickle
from loss import compute_soft_triplet_loss_
#import tensorflow_hub as hub

def str2bool(v):
    return v.lower() in ("true", "1")

def parse_opt():
  """Parses the input arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', type=str, default='')
  parser.add_argument('--comment', type=str, default='test_notebook')
  parser.add_argument('--dataset', type=str, default='css3d')
  parser.add_argument('--dataset_path', type=str, default='./')
  parser.add_argument('--model', type=str, default='tirg')
  parser.add_argument('--embed_dim', type=int, default=1024)
  parser.add_argument('--learning_rate', type=float, default=1e-2)
  parser.add_argument('--learning_rate_decay_frequency', type=int, default=9999999)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--weight_decay', type=float, default=1e-6)
  parser.add_argument('--resume', type=str2bool, default=False)
  
  parser.add_argument('--num_iters', type=int, default=210000)
  parser.add_argument('--loss', type=str, default='soft_triplet')
  parser.add_argument("--save_dir",type=str,default="./save")
  parser.add_argument('--loader_num_workers', type=int, default=4)
  args = parser.parse_args()
  return args


def load_image(metadata):  
    opt = parse_opt()
    split='train'
    target_img=metadata['target_img_id']
    source_img=metadata['source_img_id']
    mod_id=metadata['mod_id']
    mod_str=metadata['mod_str']
 
    img=tf.io.read_file(source_img, name=None)
    s_img = tf.image.decode_png(img, channels=3)
    #_img=tf.image.per_image_standardization(s_img)
    s_img = tf.image.convert_image_dtype(s_img, tf.float32)
    s_img=tf.image.resize(s_img, [64, 64]) 
    img=tf.io.read_file(target_img, name=None)
    #mg=tf.image.per_image_standardization(img)
    t_img = tf.image.decode_png(img, channels=3)
    t_img = tf.image.convert_image_dtype(t_img, tf.float32)
    t_img=tf.image.resize(t_img, [64, 64]) 
    return {'target':t_img,'source':s_img,'mod_id':mod_id,'mod_str':mod_str}


def main():
    opt = parse_opt()
    split='train'
    path=opt.dataset_path
    img_path = path + '/images/'
    img_path1= path +'./images/css_%s_%06d.png'
    img_path = img_path + ('/css_%s_%06d.png' % (split, int(3)))
    
    opt = parse_opt()
    testset = CSSDataset(
            path=path,
            split='test')
    xx = testset.get_test_queries()
    metadataT={}
    metadataT['mod']=[]
    metadataT['source_img_id']=[]
    metadataT['target_caption']=[]

    all_img_name_vector = []
    for x in (xx):
        data=x
        metadataT['target_caption'].append(data['target_caption'])
        source_img_id=data['source_img_id']
        metadataT['source_img_id'].append(( img_path1 % (split, int(source_img_id))))
        mod_str=data['mod']['str']
        metadataT['mod'].append(mod_str)

    split='train'
    xx=CSSDataset(opt.dataset_path)
    metadata={}
    metadata['target_img_id']=[]
    metadata['source_img_id']=[]
    metadata['mod_id']=[]
    metadata['mod_str']=[]

    all_img_name_vector = []
    for x in range(len(xx)):
        image_id = ('../CSS/images/css_%s_%06d.png' % (split, int(x)))
        all_img_name_vector.append(image_id)
        
        data=xx.generate_random_query_target()
        target_img_id=data['target_img_id']
        metadata['target_img_id'].append((img_path1 % (split, int(target_img_id))))
        source_img_id=data['source_img_id']
        metadata['source_img_id'].append((img_path1 % (split, int(source_img_id))))
        mod_id=data['mod']['id']
        metadata['mod_id'].append(mod_id)
        mod_str=data['mod']['str']
        metadata['mod_str'].append(mod_str)

    image_dataset=tf.data.Dataset.from_tensor_slices(metadata)
    image_dataset = image_dataset.map(load_image).batch(opt.batch_size)

    it = iter(image_dataset)
    x=next(it)

    texts=[t for t in xx.get_all_texts()]
    embed_dim=opt.embed_dim
    text_model = TextLSTMModel(
                        texts_to_build_vocab=texts,
                        word_embed_dim=embed_dim,
                        lstm_hidden_dim=embed_dim)
    
    IMG_SHAPE = (64, 64, 3)
    base_model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', \
                        input_shape=IMG_SHAPE, pooling=None, classes=1000)
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))
    base_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 0
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False
    img_extractor = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dense(embed_dim, input_shape=( 2*embed_dim,))
    ])

    #building Model
    TIRGmodel=TIRG(texts,opt.embed_dim)
    temp=tf.random.uniform([opt.batch_size,opt.embed_dim])
    temp_img=tf.random.uniform([opt.batch_size,64, 64, 3])
    img_extractor(temp_img)
    text_model(temp)
    TIRGmodel(temp,temp)

    #img_extractor.summary()
    optimizer1 = tf.keras.optimizers.RMSprop(0.001, decay=opt.weight_decay, momentum=0.9, epsilon=1.0)
    optimizer2 = tf.keras.optimizers.RMSprop(0.01, decay=opt.weight_decay, momentum=0.9, epsilon=1.0)
    optimizer3 = tf.keras.optimizers.RMSprop(0.01, decay=opt.weight_decay, momentum=0.9, epsilon=1.0)
    
    if not os.path.exists(opt.save_dir):
                       os.makedirs(opt.save_dir)
    checkpoint_txt = os.path.join(opt.save_dir, "checkpoint_txt.h5")
    checkpoint_img = os.path.join(opt.save_dir, "checkpoint_img.h5")
    checkpoint_tirg = os.path.join(opt.save_dir, "checkpoint_tirg.h5")
    
    if os.path.exists(checkpoint_txt):
        if opt.resume:
            print("Checkpoint found! Resuming")
            img_extractor.load_weights(checkpoint_img)
            text_model.load_weights(checkpoint_txt)
            TIRGmodel.load_weights(checkpoint_tirg)
  

    def train_step(x):
        loss = 0
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3:
                text_features=text_model(x['mod_str'])
                imgS=tf.squeeze(img_extractor(x['source']))
                imgT=tf.squeeze(img_extractor(x['target']))
        
                C_feature = TIRGmodel( imgS,text_features)
                imgT = TIRGmodel.normalize(imgT)

                loss = compute_soft_triplet_loss_(C_feature,imgT)



        gradients_of_generator = tape1.gradient(loss,TIRGmodel.trainable_variables) 
        optimizer2.apply_gradients(zip(gradients_of_generator, TIRGmodel.trainable_variables))
   
        gradients_of_d1 = tape2.gradient(loss,text_model.trainable_variables) 
        optimizer3.apply_gradients(zip(gradients_of_d1, text_model.trainable_variables))
        
        gradients_of_d = tape3.gradient(loss,img_extractor.trainable_variables) 
        optimizer1.apply_gradients(zip(gradients_of_d, img_extractor.trainable_variables))

        return loss

    EPOCHS = 500
    num_steps=19012
    loss_plot=[]


    for epoch in (range(EPOCHS)):

        if epoch % 20 == 3:
            tests = []
            img_extractor.training = False
            text_model.training = False
            TIRGmodel.training = False
            for name, dataset in [ ('test', testset)]:
                    t = test1(opt, TIRGmodel,img_extractor,text_model, dataset)
                    tests += [(name + ' ' + metric_name, metric_value)
                            for metric_name, metric_value in t]
            for metric_name, metric_value in tests:
                    print ('    ', metric_name, round(metric_value, 4))

        
        start = time.time()
        total_loss = 0
        batchloss = 0
        batch=0
        img_extractor.training = True
        text_model.training = True
        TIRGmodel.training = True
        print("Epoch",epoch,)
        for  x in tqdm((image_dataset)):
            batch+=1
            t_loss = train_step(x)
            # batchloss1+=t_loss1
            total_loss += t_loss

        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)
                
        print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                            total_loss/num_steps))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        # Save the weights
        text_model.save_weights(checkpoint_txt)
        img_extractor.save_weights(checkpoint_img)
        TIRGmodel.save_weights(checkpoint_tirg)


if __name__ == '__main__':
  main()
