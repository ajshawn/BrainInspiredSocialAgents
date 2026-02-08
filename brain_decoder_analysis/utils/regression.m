function [proj_train,proj_test,r2_train,r2_test,pcc_test,mse,npc] = regression(x,y,nfold)
n_trial = size(x,1)/100;
trial_rand = randperm(n_trial);
for i = 1:size(y,2)
    % take half for train and test
    for fold = 1:nfold
        ntr_per_fold = floor(n_trial/nfold);
        train_tr = trial_rand(ntr_per_fold*(fold-1)+1:ntr_per_fold*fold);
        test_tr = setdiff(1:n_trial,train_tr);
        test = trial_time_indices(test_tr, 100);
        train = trial_time_indices(train_tr, 100);
        % run pca
        [coeff,score,~,~,explained,mu] = pca(x(train,:));
        npc = find(cumsum(explained)>80,1);
        md = fitlm(score(:,1:npc),y(train,i));
        test_score = (x(test,:)-mu)*coeff(:,1:npc);
        proj_train(i,train) = predict(md,score(:,1:npc));
        proj_test(i,test) = predict(md,test_score);
    end
    r2_train(i) = r_squared(y(1:size(proj_train,2),i), proj_train(i,:));
    r2_test(i) = r_squared(y(1:size(proj_test,2),i), proj_test(i,:));
    pcc_test(i) = corr2(y(1:size(proj_test,2),i), proj_test(i,:)');
    mse(i) = mean((proj_test(i,:)'- y(1:size(proj_test,2),i)).^2);
end
end