%% PCA
clear all
load('./YaleB_32x32.mat');
for p=1:5
    [X_train, y_train, X_test, y_test] = split_train_test(fea, gnd, 10*p);
    avg=mean(X_train,1);
    X_train=X_train-avg;
    evector=pca(X_train);
    evector=evector(:,4:300);% abandon first 3 eigen vectors as the paper says, it does help with result
    train_project=X_train*evector;
    
    count=0;
    for j=1:length(y_test)
        test_project(j,:)=(X_test(j,:)-avg)*evector;
        similarity = arrayfun(@(n) 1 / (1 + norm(train_project(n,:) - test_project(j,:))), 1:length(y_train));
        [~, idx] = max(similarity);
        label=y_train(idx,1);
        if label == y_test(j)
            count = count + 1;
        end
    end
    acc=count/length(y_test);
    err_pca(p)=(1-acc)*100;
end
err_pca
%%
figure(1)
plot(10:10:50,err_pca)
xlabel('Number of Trainings Samples ')
ylabel('Error Rate (%)')
title("PCA error for original image")