import numpy as np
from tqdm import tqdm as tqdm
import tensorflow as tf

def test1(opt, TIRG_m,IMG_m,Text_m, testset):
  """Tests a model over the given testset."""
  
  test_queries = testset.get_test_queries()

  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  if test_queries:
 
    # compute test query features
    imgs = []
    mods = []
    #this loop appends the test_queries till it reaches the batch size 
    #and extract the features from the image and test using test model 
    for t in tqdm(test_queries):
      #retriving the source image from id( or creating the image)
      imgs += [testset.get_img(t['source_img_id'])]
      mods += [t['mod']['str']]

      if len(imgs) >= opt.batch_size or t is test_queries[-1]:
        # print("INside loop 1")
        if 'torch' not in str(type(imgs[0])):
          imgs = [(d) for d in imgs]
        imgs = tf.stack(imgs)
        mods = [t for t in mods]
        img_f = tf.squeeze(IMG_m(imgs))
        Text_f = Text_m(mods) 
        f= TIRG_m(img_f,Text_f)
        all_queries += [f]
        imgs = []
        mods = []
        
    all_queries = np.concatenate(all_queries)
    all_target_captions = [t['target_caption'] for t in test_queries]
   
    
    # compute all image features for target
    imgs = []
    for i in tqdm(range(len(testset.imgs))):
      imgs += [testset.get_img(i)]
      if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
        if 'torch' not in str(type(imgs[0])):
          imgs = [(d) for d in imgs]
        
        imgs = tf.stack(imgs)
        imgs = tf.squeeze(IMG_m(imgs))
        imgs=TIRG_m.normalize(imgs)
        all_imgs += [imgs]
        imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [img['captions'][0] for img in testset.imgs]

  else:
    print("Testing on Training data")
    # use training queries to approximate training retrieval performance
    imgs0 = []
    imgs = []
    mods = []
    for i in range(10000):
      item = testset[i]
      imgs += [item['source_img_data']]
      mods += [item['mod']['str']]
      if len(imgs) > opt.batch_size or i == 9999:
        imgs = tf.stack(imgs)
        mods = [t for t in mods]



        img_f = IMG_m(imgs)
        Text_f = Text_m(mods) 
        f= TIRG_m(img_f,Text_f)
        
        all_queries += [f]
        imgs = []
        mods = []
      imgs0 += [item['target_img_data']]
      if len(imgs0) > opt.batch_size or i == 9999:
        imgs0 = tf.stack(imgs0)
        imgs0 = IMG_m(imgs0)
        imgs0 = TIRG_m.normalize(imgs0)
        all_imgs += [imgs0]
        imgs0 = []
      all_captions += [item['target_caption']]
      all_target_captions += [item['target_caption']]
    all_imgs = np.concatenate(all_imgs)
    all_queries = np.concatenate(all_queries)

  # feature normalization
  # for i in range(all_queries.shape[0]):
  #   all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  # for i in range(all_imgs.shape[0]):
  #   all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  sims = all_queries.dot(all_imgs.T)
  if test_queries:
    for i, t in enumerate(test_queries):
      sims[i, t['source_img_id']] = -10e10  # remove query image
   
  nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]


  out = []
  nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]

  for k in [1, 5, 10, 50, 100]:
    r = 0.0
    for i, nns in enumerate(nn_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(nn_result)
    out += [('recall_top' + str(k) + '_correct_composition', r)]
  return out