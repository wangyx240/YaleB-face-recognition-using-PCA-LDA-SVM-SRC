function [X_train, y_train,  X_test, y_test] = split_train_test(X, y, ratio)

m = size(X, 1);
y_labels = unique(y); 
d = [1:m]';

X_train = [];
y_train= [];

for i = 1:38
    comm_i = find(y == y_labels(i));
    if isempty(comm_i) 
        continue;
    end
    size_comm_i = length(comm_i);
    rp = randperm(size_comm_i);
    rp_ratio = rp(1:ratio);
    ind = comm_i(rp_ratio);
    X_train = [X_train; X(ind, :)];
    y_train = [y_train; y(ind, :)];
    d = setdiff(d, ind);
end

X_test = X(d, :);
y_test = y(d, :);

end