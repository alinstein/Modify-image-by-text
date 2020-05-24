import argparse
import sys
import os
import numpy as np
import tensorflow as tf
import datasets
import test
import matplotlib.pyplot as plt
import re
import time
#import tensorflow_hub as hub

class ConCatModule(tf.keras.Model):
    
  def __init__(self):
    super(ConCatModule, self).__init__()

  def call(self, x):
    x = tf.concat(x, axis=1)
    return x

class TIRG(tf.keras.Model):
  def __init__(self, texts, embed_dim):
        super(TIRG, self).__init__().__init__(name='')
        self.norm_s = tf.Variable(([1.0]),trainable=True)
        self.a = tf.Variable(([1,10, 0.9998, 1.0]),trainable=True)
        
        self.layer1=tf.keras.layers.LayerNormalization()
        self.gated_feature_composer = tf.keras.models.Sequential([
                        ConCatModule(), 
                        tf.keras.layers.BatchNormalization(axis=-1), 
                        tf.keras.layers.Activation('relu'),
                        tf.keras.layers.Dense(embed_dim, input_shape=( 2*embed_dim,)),
                        ])

        self.res_info_composer = tf.keras.models.Sequential([
                        ConCatModule(), 
                        tf.keras.layers.BatchNormalization(axis=-1), 
                        tf.keras.layers.Activation('relu'),
                        tf.keras.layers.Dense(2 * embed_dim, input_shape=(2* embed_dim,)),
                        tf.keras.layers.Activation('relu'),
                        tf.keras.layers.Dense( embed_dim, input_shape=(2* embed_dim,))
                        ])         

  def normalize(self,imgT):
    return  self.norm_s*tf.math.l2_normalize(imgT,axis=1)

  def call(self, img_features, text_features):   

    f1 = self.gated_feature_composer((img_features, text_features))
    f2 = self.res_info_composer((img_features, text_features))
    f = tf.keras.activations.sigmoid(f1) * img_features * self.a[0] + f2 * self.a[1] 

    f = self.norm_s* tf.math.l2_normalize(f,axis=1)
 
    return  f