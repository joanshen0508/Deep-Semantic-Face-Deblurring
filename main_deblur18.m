load('net_G_P_S_F.mat');
load('net_P_P_S_F.mat');
run ./matconvnet-1.0-beta22/matlab/vl_setupnn.m;
grayBlur=single(imread('example.png'));
blurImg=grayBlur;
if max(blurImg(:)>1)
    blurImg = blurImg/256;
end

deblur=DL_deblur_net18(blurImg,net_G,net_P);
imwrite(deblur,['example_deblur.png']);

figure;
subplot(1, 2, 1); imshow(blurImg);
subplot(1, 2, 2); imshow(deblur);


