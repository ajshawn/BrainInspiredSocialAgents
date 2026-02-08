%%
clear
load("D:\Linfan\attention_project\nguyen_data\dlx_E_Full.mat")
%%
test_r2all = [];
train_r2all = [];
test_pccall = [];
test_r2sh = [];
train_r2sh = [];
test_pccsh = [];
nfold = 5;
timeline = [-5,5];
run_shuffle = false;
run_timeline = true;
fr = 15;
for p = 1:length(E_Dlx.exp)
    for a = 1:2
        feature_keys = {'Self_x','Self_y','Ptn_x','Ptn_y','Ego_x','Ego_y','distance'};
        track_self = permute(E_Dlx.exp{p}{a}.SLEAP.PositionMat,[1,3,2]);
        track_other = permute(E_Dlx.exp{p}{setdiff(1:2,a)}.SLEAP.PositionMat,[1,3,2]);
        neural = zscore(E_Dlx.exp{p}{a}.dFFCalciumTraces);
        % body x/y
        coop_features = extract_features(track_self,track_other);
        % average every 3 frames (fr = 10)
        %coop_features = compressMatrix(coop_features,3,1);
        %neural = compressMatrix(neural,3,1);
        %% regression space
        if run_timeline
            timex = timeline(1)*fr:5:timeline(2)*fr;
            parfor x = 1:length(timex)
                neural_ = circshift(neural,timex(x),1);
                [proj_train,proj_test,r2_train(x,:),r2_test(x,:),pcc_test(x,:),mse,~] = regression(neural_(-timeline(1)*fr:end-timeline(2)*fr,:),coop_features(-timeline(1)*fr:end-timeline(2)*fr,:),nfold);
            end
            test_r2all((p-1)*2+a,:,:) = r2_test;
            train_r2all((p-1)*2+a,:,:) = r2_train;
            test_pccall((p-1)*2+a,:,:) = pcc_test;
        else
        proj_train = []; proj_test = [];
        [proj_train,proj_test,r2_train,r2_test,pcc_test,mse,npc(p)] = regression(neural,coop_features,nfold);
        test_r2all((p-1)*2+a,:) = r2_test;
        train_r2all((p-1)*2+a,:) = r2_train;
        test_pccall((p-1)*2+a,:) = pcc_test;
        end
        %% shuffle
        if run_shuffle
            parfor s = 1:10
            neural_sh = circshift(neural,randperm(size(neural,1),1),1);
            [~,~,r2_trains(s,:),r2_tests(s,:),pcc_tests(s,:),~,~] = regression(neural_sh,coop_features,nfold);
            end
            test_r2sh((p-1)*2+a,:) = mean(r2_tests,1);
            train_r2sh((p-1)*2+a,:) = mean(r2_trains,1);
            test_pccsh((p-1)*2+a,:) = mean(pcc_tests,1);
        end
    end
end
%% plot timeline 
figure
x = timex/fr;
for i = 1:7
    subplot(4,2,i)
    line_ste_shade(squeeze(test_pccall_(:,:,i)),1,"x",x); 
    xlabel('predicted time'); ylabel('PCC'); ylim([-0.2 0.6])
    title(feature_keys{i});
end
