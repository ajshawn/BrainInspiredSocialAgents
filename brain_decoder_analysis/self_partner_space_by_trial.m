clear
load('D:\Linfan\cooperation\miniscope_new\E_compiled_17_pairs.mat')
save_path = 'D:\Linfan\attention_project\cooperation';
mkdir(save_path)
%% extract sleap features - objective and egocentric
train_r2all = [];
test_r2all = [];
test_r2shuf = [];
test_pccall = [];
test_pccshuf = [];
angle_mat = [];
angle_plane1 = [];
angle_plane2 = [];
shuffle = true; 
make_plot = true;
angles = false;
for p = 1:17
    for a = 1:2
        feature_keys = {'Self_x','Self_y','Ptn_x','Ptn_y','Ego_x','Ego_y'};
        track_self = E{p}{1}.tracks(:,:,:,a);
        track_other = E{p}{1}.tracks(:,:,:,setdiff(1:2,a));
        % body x/y
        features = extract_features(track_self,track_other);
        % average every 3 frames (fr = 10)
        timeidx = E{p}{1}.session(:,1);
        coop_features = compressMatrix(features(timeidx,:),3,1);
        neural = compressMatrix(E{p}{a}.FiltTraces(timeidx,:),3,1);
        if a == 1
            correct_poke = compressMatrix(any(E{p}{a}.bvMat(timeidx,[19,20]),2),3,1)>0;
            miss_poke = compressMatrix(E{p}{a}.bvMat(timeidx,7),3,1)>0;
        else
            correct_poke = compressMatrix(any(E{p}{a}.bvMat(timeidx,[21,22]),2),3,1)>0;
            miss_poke = compressMatrix(E{p}{a}.bvMat(timeidx,24),3,1)>0;
        end
        approach = compressMatrix(E{p}{a}.bvMat(timeidx,5),3,1)>0;
        % take time period only between drink and poke (move toward poke)
        % use the next frame (poking frame) as baseline subtracted
        % threshold: < 450
        timeidx = false(size(correct_poke,1),1);
        % the last time at drink area
        cc = bwconncomp(correct_poke);
        neural_ = []; coop_features_ = []; approach_ = []; t = 0;
        trialid = [];
        % 100 frames per trial
        for i = 1:cc.NumObjects
            start = find(coop_features(1:cc.PixelIdxList{i}(1),1)>450,1,'last');
            if isempty(start) || any(correct_poke(start:cc.PixelIdxList{i}(1)-1))
                continue
            end
            t = t+1;
            timeidx(start:cc.PixelIdxList{i}(1)-1) = 1;
            len(t) = cc.PixelIdxList{i}(1) - start;
            % add neural
            neural_ori = neural(start:cc.PixelIdxList{i}(1)-1,:)-neural(cc.PixelIdxList{i}(1),:);
            neural_ = [neural_;interp1(1:len(t),neural_ori,linspace(1,len(t),100))];
            coop_features_ = [coop_features_;interp1(1:len(t),coop_features(start:cc.PixelIdxList{i}(1)-1,:),linspace(1,len(t),100))];
            approach_ = [approach_;interp1(1:len(t),double(approach(start:cc.PixelIdxList{i}(1)-1,:)),linspace(1,len(t),100))];
            trialid = [trialid,repelem(t,100)];
        end
        % same for miss poke
        % cc = bwconncomp(miss_poke);
        % for i = 1:cc.NumObjects
        %     start = find(coop_features(1:cc.PixelIdxList{i}(1),1)>450,1,'last');
        %     timeidx(start:cc.PixelIdxList{i}(1)) = 1;
        % end
        %neural = neural(timeidx,:);
        neural = neural_;
        coop_features = coop_features_;
        approach = approach_>0;
        % correct_poke = correct_poke(timeidx,:);
        % miss_poke = miss_poke(timeidx,:);
        % approach = approach(timeidx,:);

        % regression space
        if shuffle
        proj_train = []; proj_test = [];
        nfold = 2;
        [proj_train,proj_test,r2_train,r2_test,r2_testsh,pcc_test,pcc_testsh,mse,npc(p,a)] = regression_shuffle(neural,coop_features,npc,nfold);
        test_r2all(2*(p-1)+a,:) = r2_test;
        test_r2shuf(2*(p-1)+a,:) = r2_testsh;
        train_r2all(2*(p-1)+a,:) = r2_train;
        test_pccall(2*(p-1)+a,:) = pcc_test;
        test_pccshuf(2*(p-1)+a,:) = pcc_testsh;
        end
        if angles
        % calculate the angle between axes
        [angle_axis,angle_plane_1,angle_plane_2,pseudo_deg,npc(p,a)] = angle_calculate(neural,coop_features);
        angle_mat(:,:,2*(p-1)+a) = angle_axis;
        angle_plane1(:,:,2*(p-1)+a) = angle_plane_1;
        angle_plane2(:,:,2*(p-1)+a) = angle_plane_2;
        pseudo_angle(:,:,2*(p-1)+a) = pseudo_deg;
        end
        % % during approach vs not during approach
        % [~,proj_test_a,~,r2_appr,~] = regression(neural(approach,:),coop_features(approach,5),npc,nfold);
        % [~,proj_test_na,~,r2_nappr,~] = regression(neural(~approach,:),coop_features(~approach,5),npc,nfold);
        % f = figure;
        % subplot(1,2,1)
        % y = coop_features(approach,5);
        % if length(proj_test_a) ~= length(y)
        %     y = y(1:end-1);
        % end
        % scatter(proj_test_a,y); box off
        % xlabel('Predicted partn.'); ylabel('True partn.'); title(['During Approach, R2=',num2str(r2_appr)])
        % subplot(1,2,2)
        % y = coop_features(~approach,5);
        % if length(proj_test_na) ~= length(y)
        %     y = y(1:end-1);
        % end
        % scatter(proj_test_na,y)
        % xlabel('Predicted partn.'); ylabel('True partn.'); title(['Not during Approach, R2=',num2str(r2_nappr)])
        % print(f,'-dpng','-r200',['appr-',num2str(p),num2str(a),'.png'])
        if make_plot
        %% plot on axis - self x and y
        f = figure;
        subplot(2,3,1)
        scatter(proj_test(1,:),proj_test(2,:),5,'filled','MarkerFaceColor','k','MarkerFaceAlpha',0.2); hold on
        xlabel('predicted self X'); ylabel('predicted self Y')
        subplot(2,3,2)
        mdl = fitlm(proj_test(1,:),coop_features(1:size(proj_test,2),1),'Intercept',false);
        plot(mdl);legend off
        xlabel('predicted x'); ylabel('true x');
        %scatter(proj_test(5,:),features(test,5),5,"filled",'MarkerFaceAlpha',0.1);
        subplot(2,3,3)
        plot(fitlm(proj_test(2,:),coop_features(1:size(proj_test,2),2),'Intercept',false))
        xlabel('predicted y'); ylabel('true y');legend off
        %scatter(proj_test(6,:),features(test,6),5,"filled",'MarkerFaceAlpha',0.1);
        % plot on axis - partner x and y
        subplot(2,3,4)
        scatter(proj_test(5,:),proj_test(6,:),5,'filled','MarkerFaceColor','k','MarkerFaceAlpha',0.2); hold on
        xlabel('predicted partn. X'); ylabel('predicted partn. Y')
        subplot(2,3,5)
        mdl = fitlm(proj_test(5,:),coop_features(1:size(proj_test,2),5),'Intercept',false);
        plot(mdl); legend off
        xlabel('predicted x'); ylabel('true x');
        %scatter(proj_test(5,:),features(test,5),5,"filled",'MarkerFaceAlpha',0.1);
        subplot(2,3,6)
        plot(fitlm(proj_test(6,:),coop_features(1:size(proj_test,2),6),'Intercept',false))
        legend off
        xlabel('predicted y'); ylabel('true y');
        %scatter(proj_test(6,:),features(test,6),5,"filled",'MarkerFaceAlpha',0.1);
        print(f,'-dpng','-r200',['model-fit-',num2str(p),num2str(a),'.png'])
        %% plot true and neural space - self x and partner x , on the first 20 trials
        f = figure;
        for i = 1:20%trialid(end)
            subplot(1,2,1)
            x = coop_features(trialid==i,1)';
            y = coop_features(trialid==i,5)';
            z = zeros(size(x));
            col = 1:sum(trialid==i);
            surface([x;x],[y;y],[z;z],[col;col],...
                'facecol','no',...
                'edgecol','interp',...
                'linew',2);
            xlim([0 500]); ylim([-500 500]); xlabel('True self x'); ylabel('True partner x')
            subplot(1,2,2)
            x = proj_test(1,trialid==i);
            y = proj_test(5,trialid==i);
            z = zeros(size(x));
            col = 1:sum(trialid==i);
            surface([x;x],[y;y],[z;z],[col;col],...
                'facecol','no',...
                'edgecol','interp',...
                'linew',2);
            xlim([0 500]); ylim([-500 500]); xlabel('Predicted self x'); ylabel('Predicted partner x')
        end
        print(f,'-dpng','-r200',['selfx-partnerx-trajec-',num2str(p),num2str(a),'.png'])
        %% label the trials for waiting, waited and synchronous
        f = figure;
        trial = [];
        warped_x = [];
        warped_y = [];
        warped_x_p = [];
        warped_y_p = [];
        for i = 1:trialid(end)
            x = coop_features(trialid==i,1)';
            y = coop_features(trialid==i,5)';
            z = zeros(size(x));
            col = 1:sum(trialid==i);
            % classify
            if y(15)< 0 && any(y<=-300)
                trial(i) = 2;
                subplot(2,4,2)
            elseif y(1)>= 250 && ~any(y<=-200)
                trial(i) = 1;
                subplot(2,4,1)
            elseif all(abs(y)<100)
                trial(i) = 3;
                subplot(2,4,3)
            elseif y(1)>= 250 && any(y<=-250)
                trial(i) = 4;
                subplot(2,4,4)
            else
                continue
            end
            % compute warp
            warped_x = [warped_x;interp1(0:1/(length(x)-1):1,x,0:1/199:1)];
            warped_y = [warped_y;interp1(0:1/(length(y)-1):1,y,0:1/199:1)];
            %surface([x;x],[y;y],[z;z],[col;col],'facecol','no','edgecol','interp','edgealpha',0.2,'linew',2);
            %xlim([0 500]); ylim([-500 400])
            %subplot(2,4,4+trial(i))
            x = proj_test(1,trialid(1:length(proj_test))==i);
            y = proj_test(5,trialid(1:length(proj_test))==i);
            warped_x_p = [warped_x_p;interp1(0:1/(length(x)-1):1,x,0:1/199:1)];
            warped_y_p = [warped_y_p;interp1(0:1/(length(y)-1):1,y,0:1/199:1)];
            %z = zeros(size(x));
            %col = 1:sum(trialid(1:length(proj_test))==i);
            %surface([x;x],[y;y],[z;z],[col;col],'facecol','no','edgecol','interp','edgealpha',0.2,'linew',2);
            %xlim([0 500]); ylim([-500 400])
        end
        % plot warped trace and average (200 frames)
        trialmatch = trial(trial~=0);
        for i = 1:3
            subplot(2,4,i)
            take = find(trialmatch == i);
            hl = plot(warped_x(take,:)',warped_y(take,:)','-','Color',[0, 0, 0, 0.2]);
            hold on
            x = mean(warped_x(take,:));
            y = mean(warped_y(take,:));
            z = zeros(size(x));
            col = 1:200;
            surface([x;x],[y;y],[z;z],[col;col],...
                'facecol','no',...
                'edgecol','interp',...
                'edgealpha',1,...
                'linew',2);
            xlim([0 450]); ylim([-450 450]); box off
            subplot(2,4,4+i)
            take = find(trialmatch == i);
            hl = plot(warped_x_p(take,:)',warped_y_p(take,:)','-','Color',[0, 0, 0, 0.2]);
            hold on
            x = mean(warped_x_p(take,:));
            y = mean(warped_y_p(take,:));
            z = zeros(size(x));
            surface([x;x],[y;y],[z;z],[col;col],...
                'facecol','no',...
                'edgecol','interp',...
                'edgealpha',1,...
                'linew',2);
            xlim([0 400]); ylim([-400 400]); box off
        end
        print(f,'-dpng','-r200',['selfx-partnerx-bytrials-',num2str(p),num2str(a),'.png'])
        % %% plot warped
        % figure
        % for i = 1:size(warped_x,1)
        %     subplot(1,4,trialmatch(i))
        %     x = warped_x(i,:);
        %     y = warped_y(i,:);
        %     z = zeros(size(x));
        %     col = 1:200;
        %     surface([x;x],[y;y],[z;z],[col;col],...
        %         'facecol','no',...
        %         'edgecol','interp',...
        %         'edgealpha',1,...
        %         'linew',2);
        % end
        % %% plot true and neural space - self x and relative partner x
        % figure
        % for i = 1:trialid(end)
        %     subplot(2,4,1)
        %      x = coop_features(trialid==i,1)';
        %     y = coop_features(trialid==i,5)';
        %     z = zeros(size(x));
        %     col = 1:sum(trialid==i);
        %     surface([x;x],[y;y],[z;z],[col;col],...
        %         'facecol','no',...
        %         'edgecol','interp',...
        %         'linew',2);
        %     xlim([0 500]); ylim([-500 400])
        %     subplot(1,2,2)
        %     x = proj_test(1,trialid==i);
        %     y = proj_test(5,trialid==i);
        %     z = zeros(size(x));
        %     col = 1:sum(trialid==i);
        %     surface([x;x],[y;y],[z;z],[col;col],...
        %         'facecol','no',...
        %         'edgecol','interp',...
        %         'linew',2);
        %     xlim([0 500]); ylim([-500 400])
        % end
        end
        %% tDR ?
        % question: how to implement changing variables within a trial?
    end
end
%%
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
function out = align_trials(event_times, input,window)
cc = bwconncomp(event_times);
onsets = cellfun(@(i) i(1),cc.PixelIdxList);
out = [];
for i = 1:length(onsets)
    if onsets(i)+window(1)<=0 || onsets(i)+window(2)>length(input)
        continue
    end
    out = [out,input(onsets(i)+window(1):onsets(i)+window(2))];
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

function [proj_train,proj_test,r2_train,r2_test,r2_testsh,pcc_test,pcc_testsh,mse,npc] = regression_shuffle(x,y,npc,nfold)
% run linear regression (run each col of y)
n_trial = size(x,1)/100;
[proj_train,proj_test,r2_train,r2_test,pcc_test,mse,npc] = regression(x,y,nfold);
parfor i = 1:100
    trial_sh = randperm(n_trial);
    new_idx = trial_time_indices(trial_sh, 100);
    y_sh = y(new_idx,:);
    [~,~,~,r2_testi(i,:),pcc_testi(i,:),~,~] = regression(x,y_sh,nfold);
end
r2_testsh = mean(r2_testi,1);
pcc_testsh = mean(pcc_testi,1);
end
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
function features= extract_features(track_self,track_other)
% body x/y
rel_x = []; rel_y = [];
for f = 1:size(track_self,1)
    nose = squeeze(track_self(f,1,:));
    othernose = squeeze(track_other(f,1,:));
    head = squeeze(track_self(f,2,:));
    otherhead = squeeze(track_other(f,2,:));
    %[rel_x(f),rel_y(f)] = point_to_line_xy(squeeze(track_other(f,5,:))', nose, head);
    rel_x(f) = nose(1) - othernose(1);
    rel_y(f) = nose(2) - othernose(2);
end
features = [squeeze(track_self(:,1,:)),squeeze(track_other(:,1,:)),rel_x',rel_y'];
end
function idx = trial_time_indices(trials, trialLength)
starts = (trials-1) * trialLength + 1;
ends   = trials * trialLength;

idx = arrayfun(@(s,e) s:e, starts, ends, 'UniformOutput', false);
idx = [idx{:}];   % concatenate
end

function theta_deg = angle_between_axes(u, v)
dot_product = dot(u, v);
norm_u = norm(u);
norm_v = norm(v);
cos_theta = dot_product / (norm_u * norm_v);
theta_rad = acos(cos_theta);
theta_deg = real(rad2deg(theta_rad));
end
function [theta_deg,pseudo_deg] = plane_angles(v1, u1, v2, u2)
    % plane_angles: principal angles between two planes
    % Plane 1 = span(v1, u1)
    % Plane 2 = span(v2, u2)
    % Form basis matrices
    B1 = [v1(:), u1(:)];  % ensure column vectors
    B2 = [v2(:), u2(:)];
    % Orthonormal bases via reduced QR
    [Q1, ~] = qr(B1, 0);
    [Q2, ~] = qr(B2, 0);
    % Compute matrix of inner products
    C = Q1' * Q2;   % 2x2
    % Singular values of C are cosines of principal angles
    s = svd(C);
    % Numerical safety: clamp to [-1, 1]
    s = max(min(s, 1), -1);
    % Principal angles in radians
    theta = acos(s);
    % Convert to degrees
    theta_deg = rad2deg(theta);
    % pseudo-angle 
    theta = acos(s(1)*s(2));
    pseudo_deg = rad2deg(theta);
end
function [angles,angle_plane_1,angle_plane_2,pseudo_deg,npc] = angle_calculate(x,y)
    axis = []; angles = [];
    for i = 1:size(y,2)
        [coeff,score,~,~,explained,mu] = pca(x);
        npc = find(cumsum(explained)>80,1);
        md = fitlm(score(:,1:npc),y(:,i));
        axis(:,i) = coeff(:,1:npc)*table2array(md.Coefficients(2:end,1));
    end
    for i = 1:size(y,2)
        for j = i:size(y,2)
            angles(i,j) = angle_between_axes(axis(:,i), axis(:,j));
        end
    end
    for i = 1:3 % angle between self and allo/ego partner 
        for j = i:3
            [theta_deg,pseudo_deg(i,j)] = plane_angles(axis(:,(i-1)*2+1), axis(:,(i-1)*2+2), axis(:,(j-1)*2+1), axis(:,(j-1)*2+2));
            angle_plane_1(i,j) = theta_deg(1);
            angle_plane_2(i,j) = theta_deg(2);
        end
    end
end