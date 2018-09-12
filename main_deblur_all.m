clear all;
clc;
run ./matconvnet-1.0-beta22/matconvnet-1.0-beta22/matlab/vl_setupnn.m;
load('net_G_P_S_F.mat');
load('net_P_P_S_F.mat');
M=dir('./real/*.png');
for n=1:(length(M))
    a=M(n).name;
    blur=sprintf('./real/%s',a);
    grayBlur=single(imread(blur));
    blurImg=grayBlur;
    if max(blurImg(:)>1)
        blurImg = blurImg/256;
    end
                
    blurImg(blurImg>1) = 1; blurImg(blurImg<(0))=0;
    
    deblur=DL_deblur_net18(blurImg,net_G,net_P);
    d = a(1:end-4);
    imwrite(deblur,['./deblur_image/',num2str(d),'_deblur.png']);
    
end