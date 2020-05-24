import argparse
import sys
import os
import numpy as np
import tensorflow as tf
from TIRGmodel import TIRG
from textmodel import TextLSTMModel
import re


def compute_batch_based_classification_loss_(mod_img1, img2):
    x=tf.matmul(mod_img1, tf.transpose(img2, perm=[ 1, 0]))
    labels = (range(x.shape[0]))
    return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, x))


def compute_soft_triplet_loss_( mod_img1, img2):
    triplets = []
    labels = list(range(mod_img1.shape[0])) + list(range(img2.shape[0]))
    for i in range(len(labels)):
      triplets_i = []
      for j in range(len(labels)):
        if labels[i] == labels[j] and i != j:
          for k in range(len(labels)):
            if labels[i] != labels[k]:
              triplets_i.append([i, j, k])
      tf.random.shuffle(triplets_i)
      triplets += triplets_i[:3]

    assert (triplets and len(triplets) < 2000)


    soft_triplet_loss=TripletLoss()
    return soft_triplet_loss(tf.concat([mod_img1, img2], axis= 0, name='concat'), triplets)


def pairwise_distances(x, y=None):

  x_norm = tf.reshape(tf.reduce_sum(x**2,1), [-1, 1])

  if y is not None:
        y_t = tf.transpose(y, perm=[ 1, 0])
        y_norm = tf.reshape(tf.reduce_sum(y**2,1), [-1, 1])
  else:
    y_t = tf.transpose(x,  perm=[1, 0]) 
    y_norm = tf.reshape(x_norm,[1,-1])
  dist = x_norm + y_norm 
  dist = dist - 2.0 * tf.matmul(x, y_t)
  
  return tf.clip_by_value(dist, clip_value_min=0,clip_value_max=np.inf)


class TripletLoss(tf.keras.Model):
  """Class for the triplet loss."""
  def __init__(self, pre_layer=None):
    super(TripletLoss, self).__init__()
    self.pre_layer = pre_layer

  def call(self, x, triplets):
    
    if self.pre_layer is not None:
      x = self.pre_layer(x)
    
    self.triplets = triplets
    self.triplet_count = len(triplets)

    self.distances = pairwise_distances(x)
    loss = 0.0
    triplet_count = 0.0
    correct_count = 0.0
    wrong=0
    
    for i, j, k in self.triplets:
      w = 1.0
      triplet_count += w
      loss += w * tf.math.log(1 + tf.math.exp(self.distances[i, j] - self.distances[i, k]))
      if self.distances[i, j] < self.distances[i, k]:
        correct_count += 1
      else:
        wrong+=1

    loss /= triplet_count
    return (loss)