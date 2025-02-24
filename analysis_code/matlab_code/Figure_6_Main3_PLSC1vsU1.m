% Example updated script
% -------------------------------------------------------------------------
% If you put everything in one .m file, make sure you have "getBVCols.m"
% at the bottom (after the main code). Or place getBVCols in a separate file.
% -------------------------------------------------------------------------

% Define the video paths
video_paths = {
  '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles/',
  '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles/',
  '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles/',
};
suffices = {
  'cp7357', 'cp9651', 'AH'
};
num_episode = 100;

% ------------------------------------------------------------------------
% Define the different bv_types you want to analyze:
% (Feel free to adjust to your use case.)
% ------------------------------------------------------------------------
bv_types = {
    'act+ori+sta',         % discrete actions + orientation + stamina
    'act',                 % discrete actions only
    'ori+sta',            % orientation + stamina
    'act_prob+ori+sta',    % action probabilities + orientation + stamina
    'act_prob'             % action probabilities only
};

% Settings
num_perm = 50;
thr = 0.01;
thrSig = 1.5;

tic
for ii = 1:numel(video_paths)
    video_path = video_paths{ii};
    suffix = suffices{ii};

    % Define predator and prey IDs
    if contains(video_path, 'open')
        predator_ids = 0:2;
        prey_ids = 3:12;
    else
        predator_ids = 0:4;
        prey_ids = 5:12;
    end

    % Load your Python pickle data (PLSC_results_dict.pkl)
    pkl_file_name = fullfile(video_path, 'PLSC_results_dict.pkl');
    fid = py.open(pkl_file_name, 'rb');
    data = py.pickle.load(fid);

    % --------------------------------------------------------------------
    % Now loop over each bv_type
    % --------------------------------------------------------------------
    for bvt_idx = 1:numel(bv_types)
        current_bv_type = bv_types{bvt_idx};

        % Build predator/prey columns for this bv_type
        predator_bv_cols = getBVCols(0, current_bv_type);  % For agent_id=0
        prey_bv_cols     = getBVCols(1, current_bv_type);  % For agent_id=1

        % Initialize accumulators for each bv_type
        mAll = [];
        posIdxAll = [];
        negIdxAll = [];
        sigIdxAll = [];

        % ----------------------------------------------------------------
        % Loop through each prey_id and predator_id
        % ----------------------------------------------------------------
        for prey_id = prey_ids

            for predator_id = predator_ids

                bv_path  = fullfile(video_path, sprintf('%d_%d_info.csv', predator_id, prey_id));
                net_path = fullfile(video_path, sprintf('%d_%d_network_states.csv', predator_id, prey_id));

                bv_data  = readtable(bv_path);
                net_data = readtable(net_path);

                % Fill missing orientation columns, if any
                bv_data = filling_missing_orientation_one_hot_vec(bv_data);

                % Normalize network state data
                net_data = array2table(zscore(table2array(net_data)), ...
                    'VariableNames', net_data.Properties.VariableNames);

                % Extract hidden_0 (predator) and hidden_1 (prey)
                prey_net_data     = net_data(:, contains(net_data.Properties.VariableNames, 'hidden_1'));
                predator_net_data = net_data(:, contains(net_data.Properties.VariableNames, 'hidden_0'));

                % Restrict bv_data to the columns for this bv_type
                bv_data = bv_data(:, [predator_bv_cols; prey_bv_cols]);
                bv_data = array2table(zscore(table2array(bv_data)), ...
                    'VariableNames', bv_data.Properties.VariableNames);

                prey_net_data     = table2array(prey_net_data);
                predator_net_data = table2array(predator_net_data);

                % Optionally, if you used computeNonRedundantVar or something else,
                % you could do so here. But your code just references "xx", so maybe
                % it's for debugging or placeholders. It's up to you:

                xx = {};
                xx{1} = table2array(bv_data(:, prey_bv_cols));
                xx{2} = table2array(bv_data(:, predator_bv_cols));
                % (If needed, do something with xx now)

                % Now the logic with your Python dictionary:
                % We get the rank for this pair (predator_id, prey_id)
                plsc_dims = cell( data{sprintf('%d_%d', predator_id, prey_id)}{'rank'} );

                % Now loop over each episode, 1..num_episode
                for episode = 1:num_episode

                    % Extract 1000 frames per episode
                    R1 = zscore(predator_net_data(1+(episode-1)*1000 : episode*1000, :));
                    R2 = zscore(prey_net_data(1+(episode-1)*1000 : episode*1000, :));

                    [U,V,~,dU,dV] = getSharedSpace(R1,R2);

                    dim = double(plsc_dims{episode}.item());
                    if dim < 128
                        l1 = pca(dU(:, dim+1:end));
                        l2 = pca(dV(:, dim+1:end));

                        R{1} = [U(:,1), U(:,dim+1:end)*l1(:,1)];
                        R{2} = [V(:,1), V(:,dim+1:end)*l2(:,1)];

                        for agent = 1:2
                            % The difference measure
                            m = abs(R{agent}(:,1)) - abs(R{agent}(:,2));
                            z1 = abs(R{agent}(:,1));
                            z2 = abs(R{agent}(:,2));

                            % Identify significant cells
                            x1Idx = z1 > (mean(z1) + thrSig*std(z1));
                            x2Idx = z2 > (mean(z2) + thrSig*std(z2));
                            x3Idx = (x1Idx + x2Idx) == 2;  % both

                            sig1Idx = ((x1Idx - x2Idx) == 1);   % only x1
                            sig2Idx = ((x1Idx - x2Idx) == -1);  % only x2

                            % Accumulate
                            mAll = [mAll; m];
                            posIdxAll = [posIdxAll; sig1Idx];
                            negIdxAll = [negIdxAll; sig2Idx];
                            sigIdxAll = [sigIdxAll; x3Idx];
                        end
                    end
                end % end episode loop

                toc
            end % end predator_id loop
        end % end prey_id loop

        %% Plotting the histogram
        figure();
        hold on;
        histogram(mAll(logical(posIdxAll)), 'FaceColor',[0.8,0,0], ...
            'BinWidth',0.01, 'FaceAlpha',0.4);
        histogram(mAll(logical(negIdxAll)), 'FaceColor',[0,0,0.8], ...
            'BinWidth',0.01, 'FaceAlpha',0.4);
        histogram(mAll(logical(sigIdxAll)), 'FaceColor',[0.5,0,0.5], ...
            'BinWidth',0.01, 'FaceAlpha',0.8);

        ylabel('Cells Count');
        title(['Total Cells:',num2str(numel(mAll)),' - Iter ',num2str(num_episode)]);
        xlabel({'|W_{PLSC1}| - |W_{U1}|'});
        legend({'PLSC1','U1','Both'}, 'Location','eastoutside');
        hold off;

        % Save figure: incorporate current_bv_type into name
        safe_bv_type = strrep(current_bv_type, '+', '_');  % replace '+' with '_'
        figure_name = ['fig_plsc1_vs_u1_' suffix '_' safe_bv_type '.png'];
        saveas(gcf, figure_name);

        %% Save results
        results = struct();
        results.mAll       = mAll;
        results.posIdxAll  = posIdxAll;
        results.negIdxAll  = negIdxAll;
        results.sigIdxAll  = sigIdxAll;

        % Construct a unique file name for each bv_type
        output_mat_name = ['./results/plsc1_vs_u1_' suffix '_' safe_bv_type '.mat'];
        save(output_mat_name, 'results');

    end % end of loop over bv_types

end % end of loop over video_paths

toc
disp('All done!');

% -------------------------------------------------------------------------
% Below is the helper function to build BV columns for each agent/bv_type.
% You can put it in a separate file named getBVCols.m if you prefer.
% -------------------------------------------------------------------------
function bv_cols = getBVCols(agent_id, bv_type)
    % This function returns a list of column names based on bv_type.
    % Examples of bv_type strings:
    %   'act' -> "actions_{agent_id}_0..7"
    %   'act_prob' -> "actions_prob_{agent_id}_0..8"
    %   'ori' -> "orientations_{agent_id}_0..3"
    %   'sta' -> "STAMINA_{agent_id}"
    %   combos like 'act+ori+sta', 'act_prob+ori+sta', etc.

    bv_cols = {};

    if contains(bv_type, 'act_prob')
        % 9 action probability columns
        for i = 0:7
            bv_cols{end+1} = sprintf('actions_prob_%d_%d', agent_id, i);
        end
    end

    if contains(bv_type, 'act') && ~contains(bv_type, 'act_prob')
        % 8 discrete action one-hot columns
        for i = 0:7
            bv_cols{end+1} = sprintf('actions_%d_%d', agent_id, i);
        end
    end

    if contains(bv_type, 'ori')
        % 4 orientation columns
        for i = 0:3
            bv_cols{end+1} = sprintf('orientations_%d_%d', agent_id, i);
        end
    end

    if contains(bv_type, 'sta')
        % stamina column
        bv_cols{end+1} = sprintf('STAMINA_%d', agent_id);
    end

    bv_cols = bv_cols(:);
end
