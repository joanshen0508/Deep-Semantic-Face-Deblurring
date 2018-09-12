The code is implementation code for the following paper: 

Ziyi Shen, Wei-sheng Lai, Tingfa Xu, Jan Kautz and Ming-Hsuan Yang 
Deep Semantic Face Deblurring
IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018

Requirements 
MATLAB (We test with MATLAB R2016a on Windows 10)


Test Pre-trained Models

Compile matconvnet:
>> cd matconvnet-1.0-beta22/matlab
>> vl_compilenn('enableGpu', 1, 'enableCudnn', 1)
>> cd ../../

We provide the matconvnet in,
./DL_deblur_net./matconvnet-1.0-beta22./matlab


Run  'main_deblur18.m' to test the example.png

or 

You also can run 'main_deblur_all.m' to test the blur images in ./blur_image and ./real_blur_image.

In addition, the whole testing datasets have been released on our project website
https://sites.google.com/site/ziyishenmi/cvpr18_face_deblur


We provide two models here.

1. net_G_P_S_F_GAN and net_P_P_S_F_GAN
Our model with L1 loss + parsing Loss + structure loss +feathure loss

2. net_G_P_S_F and net_G_P_S_F

Our model with L1 loss + parsing Loss + structure loss + feathure loss + adversarial loss

