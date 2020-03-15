function [err_src] = test_src(X_train,y_train,X_test,y_test)
addpath('C:\Users\wangy\Downloads\spgl1-2.0');
outlabel=[];r=[];count=0;x=[];
C=length(unique(y_train));
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
err_src=(1-length(idx)/length(y_test))*100;
end

