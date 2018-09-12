function [outIm] = DL_deblur_net18(blurImg, net_G, net_P)


blurImg=imresize(blurImg,[128 128]);
blurImg0=imresize(blurImg,[64 64]);
% gpu or cpu
blurImg = gpuArray(blurImg);
blurImg0=gpuArray(blurImg0);
net_G=move(net_G,'gpu') ;
net_P=move(net_P,'gpu') ; %please use the gpu

%%%%%%%%%%%%%%%%%%%%%%%%% net_P
% downsample

convImg_1 = vl_nnconv(blurImg,net_P(1,1).w, net_P(1,1).b,'pad',[1 1 1 1],'stride',[1,1],'cuDNN');
batchImg_1 =vl_nnbnorm(convImg_1,net_P(1,1).bw, net_P(1,1).bb,'epsilon',1.0000e-04,'cuDNN');
reluImg_1 =vl_nnrelu(batchImg_1,[], 'leak', 0.0);
poolImng_1 =vl_nnpool(reluImg_1,[2,2],'pad', 0, 'stride', [2 ,2],'method','max');

convImg_2 = vl_nnconv(poolImng_1,net_P(1,2).w, net_P(1,2).b,'pad',[1 1 1 1],'stride',[1,1],'cuDNN');
batchImg_2 =vl_nnbnorm(convImg_2,net_P(1,2).bw, net_P(1,2).bb, 'epsilon',1.0000e-04,'cuDNN');
reluImg_2 =vl_nnrelu(batchImg_2,[], 'leak', 0.0);
poolImng_2 =vl_nnpool(reluImg_2,[2,2],'pad', 0, 'stride', [2 ,2],'method','max');

convImg_3 = vl_nnconv(poolImng_2,net_P(1,3).w, net_P(1,3).b,'pad',[2 2 2 2],'stride',[1,1],'cuDNN');
batchImg_3 =vl_nnbnorm(convImg_3,net_P(1,3).bw, net_P(1,3).bb,'epsilon',1.0000e-04,'cuDNN');
reluImg_3 =vl_nnrelu(batchImg_3,[], 'leak', 0.0);
poolImng_3 =vl_nnpool(reluImg_3,[2,2],'pad', 0, 'stride', [2 ,2],'method','max');

convImg_4 = vl_nnconv(poolImng_3,net_P(1,4).w, net_P(1,4).b,'pad',[1 1 1 1],'stride',[1,1],'cuDNN');
batchImg_4 =vl_nnbnorm(convImg_4,net_P(1,4).bw, net_P(1,4).bb,'epsilon',1.0000e-04,'cuDNN');
reluImg_4 =vl_nnrelu(batchImg_4,[], 'leak', 0.0);
poolImng_4 =vl_nnpool(reluImg_4,[2,2],'pad', 0, 'stride', [2 ,2],'method','max');

convImg_5 = vl_nnconv(poolImng_4,net_P(1,5).w, net_P(1,5).b,'pad',[1 1 1 1],'stride',[1,1],'cuDNN');
batchImg_5 =vl_nnbnorm(convImg_5,net_P(1,5).bw, net_P(1,5).bb,'epsilon',1.0000e-04,'cuDNN');
reluImg_5 =vl_nnrelu(batchImg_5,[], 'leak', 0.0);
poolImng_5 =vl_nnpool(reluImg_5,[2,2],'pad', 0 , 'stride', [2 ,2],'method','max');

convImg_6 = vl_nnconv(poolImng_5,net_P(1,6).w, net_P(1,6).b,'pad',[1 1 1 1],'stride',[1,1],'cuDNN');
batchImg_6 =vl_nnbnorm(convImg_6,net_P(1,6).bw, net_P(1,6).bb,'epsilon',1.0000e-04,'cuDNN');
reluImg_6 =vl_nnrelu(batchImg_6,[], 'leak', 0.0);

% upsample
de_deconv_1=vl_nnconvt(reluImg_6,net_P(1,7).upw,net_P(1,7).upb,'upsample',[2 2],'crop',[1 1 1 1],'numGroups',1,'cuDNN');
de_convImg_1 =vl_nnconv(de_deconv_1,net_P(1,7).w,net_P(1,7).b,'pad',[1 1 1 1],'stride',[1 1],'cuDNN');
de_batchImg_1=vl_nnbnorm(de_convImg_1,net_P(1,7).bw,net_P(1,7).bb,'epsilon',1.0000e-04,'cuDNN');
de_reluImng_1=vl_nnrelu(de_batchImg_1,[],'leak',0.0);
de_sumImg_1=(de_reluImng_1+reluImg_5);

de_deconv_2=vl_nnconvt(de_sumImg_1,net_P(1,8).upw,net_P(1,8).upb,'upsample',[2 2],'crop',[1 1 1 1],'numGroups',1,'cuDNN');
de_convImg_2 =vl_nnconv(de_deconv_2,net_P(1,8).w,net_P(1,8).b,'pad',[1 1 1 1],'stride',[1 1],'cuDNN');
de_batchImg_2=vl_nnbnorm(de_convImg_2,net_P(1,8).bw,net_P(1,8).bb,'epsilon',1.0000e-04,'cuDNN');
de_reluImng_2=vl_nnrelu(de_batchImg_2,[],'leak',0.0);
de_sumImg_2=(de_reluImng_2+reluImg_4);

de_deconv_3=vl_nnconvt(de_sumImg_2,net_P(1,9).upw,net_P(1,9).upb,'upsample',[2 2],'crop',[1 1 1 1],'numGroups',1,'cuDNN');
de_convImg_3 =vl_nnconv(de_deconv_3,net_P(1,9).w,net_P(1,9).b,'pad',[1 1 1 1],'stride',[1 1],'cuDNN');
de_batchImg_3=vl_nnbnorm(de_convImg_3,net_P(1,9).bw,net_P(1,9).bb,'epsilon',1.0000e-04,'cuDNN');
de_reluImng_3=vl_nnrelu(de_batchImg_3);
de_sumImg_3=(de_reluImng_3+reluImg_3);

de_deconv_4=vl_nnconvt(de_sumImg_3,net_P(1,10).upw,net_P(1,10).upb,'upsample',[2 2],'crop',[1 1 1 1],'numGroups',1,'cuDNN');
de_convImg_4 =vl_nnconv(de_deconv_4,net_P(1,10).w,net_P(1,10).b,'pad',[1 1 1 1],'stride',[1 1],'cuDNN');
de_batchImg_4=vl_nnbnorm(de_convImg_4,net_P(1,10).bw,net_P(1,10).bb,'epsilon',1.0000e-04,'cuDNN');
de_reluImng_4=vl_nnrelu(de_batchImg_4,[],'leak',0.0);
de_sumImg_4=(de_reluImng_4+reluImg_2);

de_deconv_5=vl_nnconvt(de_sumImg_4,net_P(1,11).upw,net_P(1,11).upb,'upsample',[2 2],'crop',[1 1 1 1],'numGroups',1,'cuDNN');
de_convImg_5 =vl_nnconv(de_deconv_5,net_P(1,11).w,net_P(1,11).b,'pad',[1 1 1 1],'stride',[1 1],'cuDNN');
de_batchImg_5=vl_nnbnorm(de_convImg_5,net_P(1,11).bw,net_P(1,11).bb,'epsilon',1.0000e-04,'cuDNN');
de_reluImg_5=vl_nnrelu(de_batchImg_5,[],'leak',0.0);

de_convImg_parsing=vl_nnconv(de_reluImg_5,net_P(1,12).w,net_P(1,12).b,'pad',[1 1 1 1],'stride',[1 1]);

%% %%%%%%%%%%%%%%%%%%%%%%% net_G
G_softImg_1=vl_nnsoftmax(de_convImg_parsing);
G_poolImng_1 =vl_nnpool(G_softImg_1,[2,2],'pad', [0 0 0 0], 'stride', [2 ,2],'method','max');
G_concatImg_1=vl_nnconcat({blurImg0,G_poolImng_1},3);

G_convImg_1=vl_nnconv(G_concatImg_1,net_G(1,1).w,net_G(1,1).b,'pad',[5 5 5 5],'stride',[1 1],'cuDNN');
G_reluImg_1=vl_nnrelu(G_convImg_1,[],'leak',0.0);

G_convImg_2=vl_nnconv(G_reluImg_1,net_G(1,2).w,net_G(1,2).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_reluImg_2=vl_nnrelu(G_convImg_2,[],'leak',0.0);

G_convImg_3=vl_nnconv(G_reluImg_2,net_G(1,3).w,net_G(1,3).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_reluImg_3=vl_nnrelu(G_convImg_3,[],'leak',0.0);

G_res1_convImg_1_1=vl_nnconv(G_reluImg_3,net_G(1,4).w,net_G(1,4).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res1_reluImg_1_1=vl_nnrelu(G_res1_convImg_1_1,[],'leak',0.0);
G_res1_convImg_1_2=vl_nnconv(G_res1_reluImg_1_1,net_G(1,5).w,net_G(1,5).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res1_sunImg_1 =(G_res1_convImg_1_2+G_reluImg_3);
G_res1_reluImg_1_2=vl_nnrelu(G_res1_sunImg_1,[],'leak',0.0);

G_res1_convImg_2_1=vl_nnconv(G_res1_reluImg_1_2,net_G(1,6).w,net_G(1,6).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res1_reluImg_2_1=vl_nnrelu(G_res1_convImg_2_1,[],'leak',0.0);
G_res1_convImg_2_2=vl_nnconv(G_res1_reluImg_2_1,net_G(1,7).w,net_G(1,7).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res1_sunImg_2 =(G_res1_convImg_2_2+G_res1_sunImg_1);
G_res1_reluImg_2_2=vl_nnrelu(G_res1_sunImg_2,[],'leak',0.0);

G_res1_convImg_3_1=vl_nnconv(G_res1_reluImg_2_2,net_G(1,8).w,net_G(1,8).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res1_reluImg_3_1=vl_nnrelu(G_res1_convImg_3_1,[],'leak',0.0);
G_res1_convImg_3_2=vl_nnconv(G_res1_reluImg_3_1,net_G(1,9).w,net_G(1,9).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res1_sunImg_3 =(G_res1_convImg_3_2+G_res1_sunImg_2);
G_res1_reluImg_3_2=vl_nnrelu(G_res1_sunImg_3,[],'leak',0.0);

G_res1_convImg_4_1=vl_nnconv(G_res1_reluImg_3_2,net_G(1,10).w,net_G(1,10).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res1_reluImg_4_1=vl_nnrelu(G_res1_convImg_4_1,[],'leak',0.0);
G_res1_convImg_4_2=vl_nnconv(G_res1_reluImg_4_1,net_G(1,11).w,net_G(1,11).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res1_sunImg_4 =(G_res1_convImg_4_2+G_res1_sunImg_3);
G_res1_reluImg_4_2=vl_nnrelu(G_res1_sunImg_4,[],'leak',0.0);

G_res1_convImg_5_1=vl_nnconv(G_res1_reluImg_4_2,net_G(1,12).w,net_G(1,12).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res1_reluImg_5_1=vl_nnrelu(G_res1_convImg_5_1,[],'leak',0.0);
G_res1_convImg_5_2=vl_nnconv(G_res1_reluImg_5_1,net_G(1,13).w,net_G(1,13).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res1_sunImg_5 =(G_res1_convImg_5_2+G_res1_sunImg_4);
G_res1_reluImg_5_2=vl_nnrelu(G_res1_sunImg_5,[],'leak',0.0);

G_convImg_4=vl_nnconv(G_res1_reluImg_5_2,net_G(1,14).w,net_G(1,14).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_reluImg_4=vl_nnrelu(G_convImg_4,[],'leak',0.0);

G_convImg_5=vl_nnconv(G_reluImg_4,net_G(1,15).w,net_G(1,15).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_reluImg_5=vl_nnrelu(G_convImg_5,[],'leak',0.0);

G_convImg_6=vl_nnconv(G_reluImg_5,net_G(1,16).w,net_G(1,16).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');

% SCALE 2
de_deconv=vl_nnconvt(G_convImg_6,net_G(1,17).w,net_G(1,17).b,'upsample',[2 2],'crop',[1 1 1 1],'numGroups',1,'cuDNN');
G_concatImg_12=vl_nnconcat({de_deconv,blurImg},3);
G_concatImg_2=vl_nnconcat({G_concatImg_12,G_softImg_1},3);

G_convImg2_1=vl_nnconv(G_concatImg_2,net_G(1,18).w,net_G(1,18).b,'pad',[5 5 5 5],'stride',[1 1],'cuDNN');
G_reluImg2_1=vl_nnrelu(G_convImg2_1,[],'leak',0.0);

G_convImg2_2=vl_nnconv(G_reluImg2_1,net_G(1,19).w,net_G(1,19).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_reluImg2_2=vl_nnrelu(G_convImg2_2,[],'leak',0.0);

G_convImg2_3=vl_nnconv(G_reluImg2_2,net_G(1,20).w,net_G(1,20).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_reluImg2_3=vl_nnrelu(G_convImg2_3,[],'leak',0.0);

G_res2_convImg_1_1=vl_nnconv(G_reluImg2_3,net_G(1,21).w,net_G(1,21).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res2_reluImg_1_1=vl_nnrelu(G_res2_convImg_1_1,[],'leak',0.0);
G_res2_convImg_1_2=vl_nnconv(G_res2_reluImg_1_1,net_G(1,22).w,net_G(1,22).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res2_sunImg_1 =(G_res2_convImg_1_2+G_reluImg2_3);
G_res2_reluImg_1_2=vl_nnrelu(G_res2_sunImg_1,[],'leak',0.0);

G_res2_convImg_2_1=vl_nnconv(G_res2_reluImg_1_2,net_G(1,23).w,net_G(1,23).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res2_reluImg_2_1=vl_nnrelu(G_res2_convImg_2_1,[],'leak',0.0);
G_res2_convImg_2_2=vl_nnconv(G_res2_reluImg_2_1,net_G(1,24).w,net_G(1,24).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res2_sunImg_2 =(G_res2_convImg_2_2+G_res2_sunImg_1);
G_res2_reluImg_2_2=vl_nnrelu(G_res2_sunImg_2,[],'leak',0.0);

G_res2_convImg_3_1=vl_nnconv(G_res2_reluImg_2_2,net_G(1,25).w,net_G(1,25).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res2_reluImg_3_1=vl_nnrelu(G_res2_convImg_3_1,[],'leak',0.0);
G_res2_convImg_3_2=vl_nnconv(G_res2_reluImg_3_1,net_G(1,26).w,net_G(1,26).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res2_sunImg_3 =(G_res2_convImg_3_2+G_res2_sunImg_2);
G_res2_reluImg_3_2=vl_nnrelu(G_res2_sunImg_3,[],'leak',0.0);

G_res2_convImg_4_1=vl_nnconv(G_res2_reluImg_3_2,net_G(1,27).w,net_G(1,27).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res2_reluImg_4_1=vl_nnrelu(G_res2_convImg_4_1,[],'leak',0.0);
G_res2_convImg_4_2=vl_nnconv(G_res2_reluImg_4_1,net_G(1,28).w,net_G(1,28).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res2_sunImg_4 =(G_res2_convImg_4_2+G_res2_sunImg_3);
G_res2_reluImg_4_2=vl_nnrelu(G_res2_sunImg_4,[],'leak',0.0);

G_res2_convImg_5_1=vl_nnconv(G_res2_reluImg_4_2,net_G(1,29).w,net_G(1,29).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res2_reluImg_5_1=vl_nnrelu(G_res2_convImg_5_1,[],'leak',0.0);
G_res2_convImg_5_2=vl_nnconv(G_res2_reluImg_5_1,net_G(1,30).w,net_G(1,30).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_res2_sunImg_5 =(G_res2_convImg_5_2+G_res2_sunImg_4);
G_res2_reluImg_5_2=vl_nnrelu(G_res2_sunImg_5,[],'leak',0.0);

G_2convImg_4=vl_nnconv(G_res2_reluImg_5_2,net_G(1,31).w,net_G(1,31).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_2reluImg_4=vl_nnrelu(G_2convImg_4,[],'leak',0.0);

G_2convImg_5=vl_nnconv(G_2reluImg_4,net_G(1,32).w,net_G(1,32).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');
G_2reluImg_5=vl_nnrelu(G_2convImg_5,[],'leak',0.0);

G_2convImg_6=vl_nnconv(G_2reluImg_5,net_G(1,33).w,net_G(1,33).b,'pad',[2 2 2 2],'stride',[1 1],'cuDNN');


outIm=gather(G_2convImg_6);
end