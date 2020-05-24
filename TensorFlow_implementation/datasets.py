#Code is referenced and modified from following site :https://github.com/google/tirg

import argparse
import sys
import os
import numpy as np
import tensorflow as tf
import PIL
from PIL import Image
import re
import numpy as np
from glob import glob
import pickle


class BaseDataset():
  """Base class for a dataset."""

  def __init__(self):
    super(BaseDataset, self).__init__()
    # image set
    self.imgs = []
    # tesr set
    self.test_queries = []

  def get_test_queries(self):
    return self.test_queries

  def get_all_texts(self):
    raise NotImplementedError

  def __getitem__(self, idx):
  
    return self.generate_random_query_target()
  def load_data(self):
        return self.generate_random_query_target()
        

  def generate_random_query_target(self):
    raise NotImplementedError

  def get_img(self, idx, raw_img=False):
    raise NotImplementedError


class CSSDataset(BaseDataset):
  """CSS dataset."""

  def __init__(self, path, split='train', transform=None):
    super(CSSDataset, self).__init__()
    # location of images
    path=str(path)
    self.img_path = (path) + '/images/'
    # transformation variable
    self.transform = transform
    self.split = split
    # loading the numpy file
    #changed in numpy format
    self.data = np.load(path + '/css_toy_dataset_novel2_small.dup.npy',encoding='latin1',allow_pickle=True).item()
    # selecting mods as per split (train of test)
    # mod- modification, number of mods and object_img are different here
    self.mods = self.data[self.split]['mods']
    self.imgs = []
    # retriving image and labels from the dataset
    #change has_key to 'in' for python3
    #Stores the objects and labels and caption in list - image
    for objects in self.data[self.split]['objects_img']:
      label = len(self.imgs)
      if ('labels') in self.data[self.split]:
                label = self.data[self.split]['labels'][label]

      self.imgs += [{
          'objects': objects,
          'label': label,
          'captions': [str(label)]
          }]

    self.imgid2modtarget = {}
    for i in range(len(self.imgs)):
      self.imgid2modtarget[i] = []
    #Make combination of (from - to)
    # it seems that for each object has more than one from-to elements
    for i, mod in enumerate(self.mods):
      for k in range(len(mod['from'])):
        f = mod['from'][k]
        t = mod['to'][k]
        self.imgid2modtarget[f] += [(i, t)]

    self.generate_test_queries_()

  #This function combines 'from id' , 'to_id' and 'caption'
  # {'source_img_id': 798, 'target_caption': '1000', 'mod': {'str': 'make middle-left large circle green'}}
  def generate_test_queries_(self):
    print("generate_test_queries_")
    test_queries = []
    for mod in self.mods:
      for i, j in zip(mod['from'], mod['to']):
        test_queries += [{
            'source_img_id': i,
            'target_caption': self.imgs[j]['captions'][0],
            'mod': {'str': mod['to_str']}
        }]
    self.test_queries = test_queries

  def get_1st_training_query(self):
   
    #randomly selecting a mod, where i mod index 
    #j is index for selection of from and to image 
    i = np.random.randint(0, len(self.mods))
    mod = self.mods[i]
    j = np.random.randint(0, len(mod['from']))
    self.last_from = mod['from'][j]
    self.last_mod = [i]
    return mod['from'][j], i, mod['to'][j]

  def get_2nd_training_query(self):
    #last_from is the varible stores the index of 'from' image from previous training query
    modid, new_to = random.choice(self.imgid2modtarget[self.last_from])
    #if modid is already used choose a new one
    while modid in self.last_mod:
      modid, new_to = random.choice(self.imgid2modtarget[self.last_from])
    #Once this is used it is added into last_mod so that it won't used again
    self.last_mod += [modid]
    # mod = self.mods[modid]
    return self.last_from, modid, new_to

  def generate_random_query_target(self):
    #when last_mod is not intialized get_1st_training_query is called
    # once last_mod == 2, get_1st_training_query is called and last_mod is reset to [] empty
    
    try:
      
      if len(self.last_mod) < 2:
        
        img1id, modid, img2id = self.get_2nd_training_query()

      else:
        img1id, modid, img2id = self.get_1st_training_query()
       
        
    except:
      img1id, modid, img2id = self.get_1st_training_query()
     

    out = {}
    out['source_img_id'] = img1id
    #create the source image
    #out['source_img_data'] = self.get_img(img1id)
    out['target_img_id'] = img2id
    #create the target image 
    #out['target_img_data'] = self.get_img(img2id)
    #output image contain source image,target image and modification string and their corresponding ids
    out['mod'] = {'id': modid, 'str': self.mods[modid]['to_str']}
    return out

  def __len__(self):
    return len(self.imgs)

  def get_all_texts(self):
    return [mod['to_str'] for mod in self.mods]

  def get_img(self, idx, raw_img=False, get_2d=False):
    """Gets CSS images."""
    def generate_2d_image(objects):
      img = np.ones((64, 64, 3))
      colortext2values = {
          'gray': [87, 87, 87],
          'red': [244, 35, 35],
          'blue': [42, 75, 215],
          'green': [29, 205, 20],
          'brown': [129, 74, 25],
          'purple': [129, 38, 192],
          'cyan': [41, 208, 208],
          'yellow': [255, 238, 51]
      }
      for obj in objects:
        s = 4.0
        if obj['size'] == 'large':
          s *= 2
        c = [0, 0, 0]
        for j in range(3):
          c[j] = 1.0 * colortext2values[obj['color']][j] / 255.0
        y = obj['pos'][0] * img.shape[0]
        x = obj['pos'][1] * img.shape[1]
        if obj['shape'] == 'rectangle':
          img[int(y - s):int(y + s), int(x - s):int(x + s), :] = c
        if obj['shape'] == 'circle':
          for y0 in range(int(y - s), int(y + s) + 1):
            x0 = x + (abs(y0 - y) - s)
            x1 = 2 * x - x0
            img[y0, int(x0):int(x1), :] = c
        if obj['shape'] == 'triangle':
          for y0 in range(int(y - s), int(y + s)):
            x0 = x + (y0 - y + s) / 2
            x1 = 2 * x - x0
            x0, x1 = min(x0, x1), max(x0, x1)
            img[y0, int(x0):int(x1), :] = c
      return img

    if self.img_path is None or get_2d:
      img = generate_2d_image(self.imgs[idx]['objects'])
    else:
      img_path = self.img_path + ('/css_%s_%06d.png' % (self.split, int(idx)))
      with open(img_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((64,64))
        img = tf.keras.preprocessing.image.img_to_array(
                img)
        img = tf.image.per_image_standardization(img)

    if raw_img:
      return img
    if self.transform:
      img = self.transform(img)
    return img