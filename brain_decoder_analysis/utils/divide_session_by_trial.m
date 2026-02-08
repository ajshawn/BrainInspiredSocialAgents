function [timeidx,neural,coop_features,trialid] = divide_session_by_trial(correct_poke,features,neural_raw)
timeidx = false(size(correct_poke,1),1);
% divide the session into trials - from leaving the water to entering poke 
cc = bwconncomp(correct_poke);
neural = []; coop_features = []; t = 0;
trialid = [];
for i = 1:cc.NumObjects
    start = find(features(1:cc.PixelIdxList{i}(1),1)>450,1,'last');
    if isempty(start) || any(correct_poke(start:cc.PixelIdxList{i}(1)-1))
        continue
    end
    t = t+1;
    timeidx(start:cc.PixelIdxList{i}(1)-1) = 1;
    len(t) = cc.PixelIdxList{i}(1) - start;
    % add neural
    neural_ori = neural_raw(start:cc.PixelIdxList{i}(1)-1,:)-neural_raw(cc.PixelIdxList{i}(1),:);
    neural = [neural;interp1(1:len(t),neural_ori,linspace(1,len(t),100))];
    coop_features = [coop_features;interp1(1:len(t),features(start:cc.PixelIdxList{i}(1)-1,:),linspace(1,len(t),100))];
    trialid = [trialid,repelem(t,100)];
end
end