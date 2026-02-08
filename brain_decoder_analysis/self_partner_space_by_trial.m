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
        feature_keys = {'Self_x','Self_y','Ptn_x','Ptn_y','Ego_x','Ego_y','distance'};
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
        %% TODO: tDR ?
        % question: how to implement changing variables within a trial?
    end
end
