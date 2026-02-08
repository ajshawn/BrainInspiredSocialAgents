function [proj_train,proj_test,r2_train,r2_test,r2_testsh,pcc_test,pcc_testsh,mse,npc] = regression_shuffle(x,y,npc,nfold)
% run linear regression (run each col of y)
n_trial = size(x,1)/100;
[proj_train,proj_test,r2_train,r2_test,pcc_test,mse,npc] = regression(x,y,nfold);
parfor i = 1:10
    trial_sh = randperm(n_trial);
    new_idx = trial_time_indices(trial_sh, 100);
    y_sh = y(new_idx,:);
    [~,~,~,r2_testi(i,:),pcc_testi(i,:),~,~] = regression(x,y_sh,nfold);
end
r2_testsh = mean(r2_testi,1);
pcc_testsh = mean(pcc_testi,1);
end