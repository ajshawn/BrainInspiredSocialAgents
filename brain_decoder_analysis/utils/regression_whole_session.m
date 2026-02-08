function [proj_train,proj_test,r2_train,r2_test,pcc_test,mse,npc] = regression_whole_session(x,y,nfold)
% run linear regression (run each col of y)
for i = 1:size(y,2)
    % take half for train and test
    for fold = 1:nfold
        nfr_per_fold = floor(size(x,1)/nfold);
        test = nfr_per_fold*(fold-1)+1:nfr_per_fold*fold;
        train = setdiff(1:size(x,1),test);
        % zscore training and test set 
        train_set = zscore(x(train,:)); 
        test_set = zscore(x(test,:));
        % run pca
        [coeff,score,~,~,explained,mu] = pca(train_set);
        npc = find(cumsum(explained)>85,1);
        md = fitlm(score(:,1:npc),y(train,i));
        test_score = (test_set-mu)*coeff(:,1:npc);
        proj_train(i,train) = predict(md,score(:,1:npc));
        proj_test(i,test) = predict(md,test_score);
    end
    r2_train(i) = r_squared(y(1:size(proj_train,2),i), proj_train(i,:));
    r2_test(i) = r_squared(y(1:size(proj_test,2),i), proj_test(i,:));
    pcc_test(i) = corr2(y(1:size(proj_test,2),i), proj_test(i,:)');
    mse(i) = mean((proj_test(i,:)'- y(1:size(proj_test,2),i)).^2);
end
end