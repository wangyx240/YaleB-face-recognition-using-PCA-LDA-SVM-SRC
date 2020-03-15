%% LDA
clear all
load('./YaleB_32x32.mat');
for p=1:5
    ratio=p*10;
    [X_train, y_train, X_test, y_test] = split_train_test(fea, gnd, ratio);
    [prj_test,prj_train] = LDA(X_train, X_test,ratio);
    
    % test for error
    count=0;
    for j=1:length(y_test)
        similarity = arrayfun(@(n) 1 / (1 + norm(prj_train(n,:) - prj_test(j,:))), 1:length(y_train));
        [~, idx] = max(similarity);
        label=y_train(idx,1);
        if label == y_test(j)
            count = count + 1;
        end
    end
    acc=count/length(y_test);
    err_lda(p)=(1-acc)*100;
end
err_lda
%%
figure(1)
plot(10:10:50,err_lda)
xlabel('Number of Trainings Samples ')
ylabel('Error Rate (%)')
title("LDA error for original image")