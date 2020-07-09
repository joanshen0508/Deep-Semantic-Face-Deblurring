# Deep semantic face deblurring
=======================================================================================

Ziyi Shen, Wei-sheng Lai, Tingfa Xu, Jan Kautz and Ming-Hsuan Yang 

Deep Semantic Face Deblurring

IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018

=======================================================================================

In this paper, we propose to deblur face images using a multiscale network. A face parsing neural network is embedded into the deblurring framework, a multiple loss functions is applied to constraint the model. It is capable of deblurring the face image with more accurate semantic details.

The whole testing datasets have been released on our project website
https://sites.google.com/site/ziyishenmi/cvpr18_face_deblur

If you want test our method on your own face data, please align your face data fisrtly.

Requirements 
MATLAB (We test with MATLAB R2016a on Windows 10)


Test Pre-trained Models

Compile matconvnet:
-> cd matconvnet-1.0-beta22/matlab
-> vl_compilenn('enableGpu', 1, 'enableCudnn', 1)
-> cd ../../

We provide the matconvnet in,
./DL_deblur_net./matconvnet-1.0-beta22./matlab


Run  'main_deblur18.m' to test the example.png

or 

You also can run 'main_deblur_all.m' to test the blur images in ./blur_image and ./real_blur_image.




We provide two models here.

1. net_G_P_S_F_GAN and net_P_P_S_F_GAN
Our model with L1 loss + parsing Loss + structure loss +feathure loss

2. net_G_P_S_F and net_G_P_S_F

Our model with L1 loss + parsing Loss + structure loss + feathure loss + adversarial loss
