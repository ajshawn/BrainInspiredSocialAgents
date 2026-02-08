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
        %% plot on axis - self x and y
%         f = figure;
%         subplot(2,3,1)
%         scatter(proj_test(1,:),proj_test(2,:),5,'filled','MarkerFaceColor','k','MarkerFaceAlpha',0.2); hold on
%         xlabel('predicted self X'); ylabel('predicted self Y')
%         subplot(2,3,2)
%         mdl = fitlm(proj_test(1,:),coop_features(1:size(proj_test,2),1),'Intercept',false);
%         plot(mdl);legend off
%         xlabel('predicted x'); ylabel('true x');
%         %scatter(proj_test(5,:),features(test,5),5,"filled",'MarkerFaceAlpha',0.1);
%         subplot(2,3,3)
%         plot(fitlm(proj_test(2,:),coop_features(1:size(proj_test,2),2),'Intercept',false))
%         xlabel('predicted y'); ylabel('true y');legend off
%         %scatter(proj_test(6,:),features(test,6),5,"filled",'MarkerFaceAlpha',0.1);
%         % plot on axis - partner x and y
%         subplot(2,3,4)
%         scatter(proj_test(5,:),proj_test(6,:),5,'filled','MarkerFaceColor','k','MarkerFaceAlpha',0.2); hold on
%         xlabel('predicted partn. X'); ylabel('predicted partn. Y')
%         subplot(2,3,5)
%         mdl = fitlm(proj_test(5,:),coop_features(1:size(proj_test,2),5),'Intercept',false);
%         plot(mdl); legend off
%         xlabel('predicted x'); ylabel('true x');
%         %scatter(proj_test(5,:),features(test,5),5,"filled",'MarkerFaceAlpha',0.1);
%         subplot(2,3,6)
%         plot(fitlm(proj_test(6,:),coop_features(1:size(proj_test,2),6),'Intercept',false))
%         legend off
%         xlabel('predicted y'); ylabel('true y');
%         %scatter(proj_test(6,:),features(test,6),5,"filled",'MarkerFaceAlpha',0.1);
%         print(f,'-dpng','-r200',['model-fit-',num2str(p),num2str(a),'.png'])

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
%%
function [proj_train,proj_test,r2_train,r2_test,pcc_test,mse,npc] = regression(x,y,nfold)
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
function R2 = r_squared(actual, predicted)
%R_SQUARED Computes the coefficient of determination (R²)
% Ensure inputs are column vectors
actual    = actual(:);
predicted = predicted(:);
% Check for equal length
if length(actual) ~= length(predicted)
    error('Input vectors "actual" and "predicted" must be the same length.');
end
% Compute residual sum of squares (SS_res)
SS_res = sum((actual - predicted).^2);
% Compute total sum of squares (SS_tot)
SS_tot = sum((actual - mean(actual)).^2);
% Compute R-squared
R2 = 1 - (SS_res / SS_tot);
end
function features = extract_features(track_self,track_other)
% body x/y
rel_x = []; rel_y = [];
for f = 1:size(track_self,1)
    nose = squeeze(track_self(f,1,:));
    othernose = squeeze(track_other(f,1,:));
    head = squeeze(track_self(f,2,:));
    otherhead = squeeze(track_other(f,2,:));
    rel_x(f) = nose(1) - othernose(1);
    rel_y(f) = nose(2) - othernose(2);
    distance(f) = sqrt(rel_x(f)^2 + rel_y(f)^2);
    %[rel_x_ang(f),rel_y_ang(f)] = point_to_line_xy(squeeze(track_other(f,6,:))', nose, head);
end
%features = [squeeze(track_self(:,1,:)),squeeze(track_other(:,1,:)),rel_x',rel_y',rel_x_ang',rel_y_ang'];
features = [squeeze(track_self(:,1,:)),squeeze(track_other(:,1,:)),rel_x',rel_y',distance'];
end
function [rel_x,rel_y] = point_to_line_xy(pt, v1, v2)
% pt: n x 2
% v1 and v2 1x2
v1=v1(:)';
v2=v2(:)';
v1_ = repmat(v1,size(pt,1),1);
v2_ = repmat(v2,size(pt,1),1);
%
a = v1_ - v2_;
b = pt - v2_;
% Relative x
rel_y = dot(a,b,2) ./ vecnorm(a,2,2);
% Relative y
a_perp = [a(:,2), -a(:,1)];
rel_x = dot(b, a_perp, 2) ./ vecnorm(a,2,2);
end