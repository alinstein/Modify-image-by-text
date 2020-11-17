# Retrieve target image from composed image and text

### Retrieval Model
![Results](https://github.com/alinstein/Modify-image-by-text/blob/master/diagram2.png)
![Results](https://github.com/alinstein/Modify-image-by-text/blob/master/diagram3.png)

### Generative Model
![Results](https://github.com/alinstein/Modify-image-by-text/blob/master/diagram.png)


This project implements a deep learning neural network model to retrieve the target image from a given image and a text containing desired modifications in the input image.
The text and the image are composed using the TIRG model, use this composed feature to retrieve the target image. An experimental study to generate the target image is also implemented
using StackGAN. This project is implemented as part of course project for Deep Learning Course at University of Victoria.

More details could be found at: 'Report.pdf'.

## Getting Started

##### Model TIRG is trained using the PYTHON file "train_TIRG.py":

* Download the dataset and give the location of the dataset in config.py.
* Change the following according to the needs: batch_size, epochs, dataset location.
* Load the pre-trained model in 'save' directory if needed.

```
bash download.sh
unzip CSS.zip
python train_TIRG.py --batch_size=64  --num_iters=210000
```

##### StackGAN is trained using the PYTHON file "train_GAN.py":
* Training consist of two stages, stage I and Stage II.
* Train the TIRG model or use the pretrained TIRG weights before training.

* To train Stage I:
```
 python train_GAN.py --stage=1 --batch_size=64
```
* To train Stage II: 
```
python train_GAN.py --stage=2 --batch_size=8
```
* Change the following according to the needs: batch_size, epochs, lr (learning rate).
* Load the pre-trained model in 'save' directory if needed.

##### TensorFlow for TIRG model:
To train in TensorFlow:
```
bash download.sh
unzip CSS.zip
python main.py
```

## Dataset 
* [CSS 16K](https://drive.google.com/open?id=1wPqMw-HKmXUG2qTgYBiTNUnjz83hA2tY) (1 GB): 
*  Extract and load the dataset outside the folder.

## Download the pretrained model 
* [TIRG](https://drive.google.com/file/d/1P7jcEbp-bHW18Wib1WWpsgsr1VXMIjkp/view?usp=sharing). Pretrained model is trained on 1 NVIDIA GeForce GTX 1080Ti  for 20 hours(450 epoches). 
* [StackGAN Stage I](https://drive.google.com/file/d/1SXltrXZGxJZa1PrnAkjre8T9izO0SDWi/view?usp=sharing). Pretrained model is trained on 2 NVIDIA Tesla T4 (120 epoches). 
* [StackGAN Stage II](https://drive.google.com/file/d/1niMnrb504ELmmg8k92XXs6cBTtegOfBn/view?usp=sharing). Pretrained model is trained on 2 NVIDIA Tesla T4 (90 epoches). 

## Reference

This project was implemented taking reference from the following papers: 

[Composing Text and Image for Image Retrieval - An Empirical Odyssey (arXiv 2018)](https://arxiv.org/abs/1812.07119)
**[Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays]

[StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks (arXiv 2016)](https://arxiv.org/abs/1612.03242)
**[Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas]
