%% 1. original SRC
addpath('C:\Users\wangy\Downloads\spgl1-2.0'); % Using spgl1 library to implement src
clear all; close all;
load('./YaleB_32x32.mat');
for p=1:5
    outlabel=[];r=[];count=0;x=[];
    [X_train,y_train,X_test,y_test]=split_train_test(fea,gnd,p*10);
    C=length(unique(y_train));
    % Basis pursuit denoising
    sigma = 0;       % Desired ||Ax - b||_2
    opts = spgSetParms('verbosity',1);
    for i=1:size(X_test,1)
        x(:,i) = spg_bpdn(X_train', X_test(i,:)', sigma, opts);
        for j=1:C
            idx=find(y_train==j);
            startidx=idx(1,1);endidx=idx(end,1);
            r(j)=norm(X_test(i,:)'-(X_train(startidx:endidx,:)'*x(startidx:endidx,i)));
        end
        [~,index]=min(r);
        outlabel=[outlabel;index];
    end
    % Accuracy
    idx=find(outlabel==y_test);
    err_src(p)=(1-length(idx)/length(y_test))*100;
end
err_src

figure(1)
plot(10:10:50,err_src)
xlabel('Number of Trainings Samples ')
ylabel('Error Rate (%)')
title("SRC error for original image")

%% 2. different features based on SRC 
clear all; close all;
load('./YaleB_32x32.mat');
N=size(fea,1);
%% PCA-SRC
avg=mean(fea,1);
fea=fea-avg;
evector=pca(fea);
evector=evector(:,1:50);
pca=fea*evector;

[X_train,y_train,X_test,y_test]=split_train_test(pca, gnd, 40);
err_src_pca=test_src(X_train,y_train,X_test,y_test)
% LDA-SRC
[X_train, y_train, X_test, y_test] = split_train_test(fea, gnd, 40);
[prj_test,prj_train] = LDA(X_train, X_test,40);
err_src_lda=test_src(prj_train,y_train,prj_test,y_test)
%% HOG-SRC
hog=[];
for i=1:N
    a=reshape(fea(i,:),[32,32]);
    hog=[hog;double(extractHOGFeatures(a,'CellSize',[8 8]))];
end
%
[X_train, y_train, X_test, y_test] = split_train_test(hog, gnd, 40);
err_src_hog=test_src(X_train,y_train,X_test,y_test)
% LBP-SRC
lbp=[];
for i=1:N
    a=reshape(fea(i,:),[32,32]);
    lbp=[lbp;double(extractLBPFeatures(a,'CellSize',[8 8]))];
end

[X_train, y_train, X_test, y_test] = split_train_test(lbp, gnd, 40);
err_src_lbp=test_src(X_train,y_train,X_test,y_test)