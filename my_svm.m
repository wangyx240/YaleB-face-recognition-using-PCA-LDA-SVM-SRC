%% SVM with original images
load('./YaleB_32x32.mat');
for i=1:5
    [X_train,y_train,X_test,y_test]=split_train_test(fea, gnd, i*10);
    t = templateSVM('Standardize',true);
    Model = fitcecoc(X_train,y_train,'Learners',t);
    [label,score] = predict(Model,X_test);
    idx=find(label==y_test);
    err_svm(i)=(1-length(idx)/length(y_test))*100
end

figure(1)
plot(10:10:50,err_svm)
xlabel('Number of Trainings Samples ')
ylabel('Error Rate (%)')
title("SVM error for original image")

%% 2. different features based on SVM 
clear all; close all;
load('./YaleB_32x32.mat');
N=size(fea,1);
%% PCA-SVM
avg=mean(fea,1);
fea=fea-avg;
[evector]=pca(fea);
evector=evector(:,4:100);
pca=fea*evector;

[X_train,y_train,X_test,y_test]=split_train_test(pca, gnd, 40);
t = templateSVM('Standardize',true);
Model = fitcecoc(X_train,y_train,'Learners',t);
[label,score] = predict(Model,X_test);
idx=find(label==y_test);
err_svm_PCA=(1-length(idx)/length(y_test))*100
%% LDA-SVM
[X_train, y_train, X_test, y_test] = split_train_test(fea, gnd, 40);
[prj_test,prj_train] = LDA(X_train, X_test,40);
t = templateSVM('Standardize',true);
Model = fitcecoc(prj_train,y_train,'Learners',t);
[label,score] = predict(Model,prj_test);
idx=find(label==y_test);
err_svm_LDA=(1-length(idx)/length(y_test))*100
%% HOG-SVM
hog=[]
for i=1:N
    a=reshape(fea(i,:),[32,32]);
    hog=[hog;double(extractHOGFeatures(a,'CellSize',[4 4]))];
end
[X_train, y_train, X_test, y_test] = split_train_test(hog, gnd, 40);
t = templateSVM('Standardize',true);
Model = fitcecoc(X_train,y_train,'Learners',t);
[label,score] = predict(Model,X_test);
idx=find(label==y_test);
err_svm_HOG=(1-length(idx)/length(y_test))*100
%% LBP-SVM
lbp=[]
for i=1:N
    a=reshape(fea(i,:),[32,32]);
    lbp=[lbp;double(extractLBPFeatures(a,'CellSize',[3 3]))];
end
[X_train, y_train, X_test, y_test] = split_train_test(lbp, gnd, 40);
t = templateSVM('Standardize',true);
Model = fitcecoc(X_train,y_train,'Learners',t);
[label,score] = predict(Model,X_test);
idx=find(label==y_test);
err_svm_LBP=(1-length(idx)/length(y_test))*100
