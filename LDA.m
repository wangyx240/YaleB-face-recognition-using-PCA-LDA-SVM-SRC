function [prj_test,prj_train] = LDA(X_train, X_test,ratio)
    % my own LDA function
    avg = mean(X_train,1);
    X_train = X_train - avg;
    
    % first calculate pca
    evector=pca(X_train);
    evector=evector(:,1:300);
    train_project=X_train*evector;
    X_test = X_test - avg;
    test_project=X_test*evector;
    
    % calculate means for each class
    m = mean(train_project,1);
    m_k = [];
    for i = 1:ratio:size(train_project,1)
        m_k_tmp = mean(train_project(i:i+8,:),1);
        m_k = [m_k; m_k_tmp];
    end
    
    % compute between-class scatter
    S_b = 0;
    for k = 1:38
        S_b = S_b + ratio*(m_k(k,:) - m)'*(m_k(k,:) - m);
    end
    
    % compute within-class scatter
    S_w = 0;
    for k = 1:38
        S_k = 0;
        for i = 1:ratio
            S_k_tmp = (train_project((k-1)*ratio+i,:)-m_k(k,:))'*(train_project((k-1)*ratio+i,:)-m_k(k,:));
            S_k = S_k + S_k_tmp;
        end
        S_w = S_w + S_k;
    end
    
    % LDA
    L = 38;
    [V, D] = eigs(S_b, S_w, L-1);
    prj_test = test_project*V;    % dimension of V can be changed
    prj_train = train_project*V;
end

