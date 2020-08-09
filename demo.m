clear all
close all

load KADID_Data2.mat

path = '/home/domonkos/Desktop/QualityAssessment/Databases/kadid10k/images';

net    = alexnet;
layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'};

numberOfImages = size(dmos, 1);
Scores = zeros(numberOfImages, 1);

parfor i=1:numberOfImages
    if(mod(i,1000)==0)
        disp(i);
    end
    imgDist  = imread( char(strcat(path, filesep, string(dist_img(i)))) );
    imgRef   = imread( char(strcat(path, filesep, string(ref_img(i)))) );
    Scores(i)= ssimcnn(imgRef, imgDist, net, layers);
end

disp(['Pearson linear correlation coefficient: ', num2str(round(corr(dmos,Scores),3))])
disp(['Spearman rank order correlation coefficient: ', num2str(round(corr(dmos,Scores,'Type','Spearman'),3))])
disp(['Kendall rank order correlation coefficient: ', num2str(round(corr(dmos,Scores,'Type','Kendall'),3))])