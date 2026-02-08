for p = 1:17
    a = 1;
feature_keys = {'Self_x','Self_y','Ptn_x','Ptn_y','Ego_x','Ego_y'};
track_self = E{p}{a}.tracks(:,:,:,1);
track_other = E{p}{a}.tracks(:,:,:,2);
% body x/y
features = extract_features(track_self,track_other);
% average every 3 frames (fr = 10)
timeidx = E{p}{a}.session(:,1);
coop_features = compressMatrix(features(timeidx,:),3,1);
neural = compressMatrix(E{p}{a}.FiltTraces(timeidx,:),3,1);
correct_poke = compressMatrix(any(E{p}{a}.bvMat(timeidx,[19,20]),2),3,1)>0;
miss_poke = compressMatrix(E{p}{a}.bvMat(timeidx,4),3,1)>0;
approach = compressMatrix(E{p}{a}.bvMat(timeidx,5),3,1)>0;
% take time period only between drink and poke (move toward poke)
% use the next frame (poking frame) as baseline subtracted
% threshold: < 450 
timeidx = false(size(correct_poke,1),1); 
% the last time at drink area 
cc = bwconncomp(correct_poke);
neural_ = []; trialid = []; t = 0;
for i = 1:cc.NumObjects
    start = find(coop_features(1:cc.PixelIdxList{i}(1),1)>450,1,'last');
    if isempty(start) || any(correct_poke(start:cc.PixelIdxList{i}(1)-1))
        continue
    end
    t = t+1;
    timeidx(start:cc.PixelIdxList{i}(1)-1) = 1;
    len(t) = cc.PixelIdxList{i}(1) - start;
    % add neural 
    neural_ = [neural_;neural(start:cc.PixelIdxList{i}(1)-1,:)-neural(cc.PixelIdxList{i}(1),:)];
    % trial index 
    trialid = [trialid,repelem(t,len(t))];
end
% same for miss poke
% cc = bwconncomp(miss_poke);
% for i = 1:cc.NumObjects
%     start = find(coop_features(1:cc.PixelIdxList{i}(1),1)>450,1,'last');
%     timeidx(start:cc.PixelIdxList{i}(1)) = 1;
% end
%neural = neural(timeidx,:);
neural = neural_;
coop_features = coop_features(timeidx,:);
correct_poke = correct_poke(timeidx,:);
miss_poke = miss_poke(timeidx,:);
approach = approach(timeidx,:);
% pls
proj_train = []; proj_test = [];
%[proj_train,proj_test,r2_train,r2_test,mse] = regression(neural,coop_features,npc,nfold);
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(neural,coop_features(:,5:6),2);
varexp_e(p,1) = sum(PCTVAR(1,:));
varexp_e(p,2) = sum(PCTVAR(2,:));
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(neural,coop_features(:,1:2),2);
varexp_s(p,1) = sum(PCTVAR(1,:));
varexp_s(p,2) = sum(PCTVAR(2,:));
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(neural,coop_features(:,3:4),2);
varexp_p(p,1) = sum(PCTVAR(1,:));
varexp_p(p,2) = sum(PCTVAR(2,:));
end
%%
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