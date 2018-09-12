% clear all; clc;
% load('G:\Dropbox\Semantic Face Deblurring\code\run_shenzy_multi_parsing_deblur_combine_random_new_mr3_s_2level_test\model\G1252p_s_f.mat');
% % net_G =zeros(1,numel(net.params)/2);
% j=1;
% for i =1:2:numel(net.params)
%     net_G(1,j).w=net.params(i).value;
%     net_G(1,j).b=net.params(i+1).value;
%     j=j+1;
% 
% end
% save('G-P-S-F-GAN.mat','net_G'); 

clear all; clc;
load('G:\Dropbox\Semantic Face Deblurring\code\run_shenzy_multi_parsing_deblur_combine_random_new_mr3_s_2level_test\model\P1252p_s_f.mat');
% net_G =zeros(1,numel(net.params)/2);
j=1;
for i =1:5:30
    net_P(1,j).w=net.params(i).value;
    net_P(1,j).b=net.params(i+1).value;
    net_P(1,j).bw=net.params(i+2).value;
    net_P(1,j).bb=net.params(i+3).value;
    net_P(1,j).bm=net.params(i+4).value;
    j=j+1;
       
end
for i=31:7:65
    net_P(1,j).upw = net.params(i).value;
    net_P(1,j).upb = net.params(i+1).value;
    net_P(1,j).w = net.params(i+2).value;
    net_P(1,j).b = net.params(i+3).value;
    net_P(1,j).bw = net.params(i+4).value;
    net_P(1,j).bb = net.params(i+5).value;
    net_P(1,j).bm =net.params(i+6).value;
    j=j+1;
end

    net_P(1,j).w=net.params(66).value;
    net_P(1,j).b=net.params(67).value;
    
save('P-P-S-F-GAN.mat','net_P'); 

