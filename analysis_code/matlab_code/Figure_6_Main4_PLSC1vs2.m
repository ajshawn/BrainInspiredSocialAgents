%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN SCRIPT WITH BV_TYPES LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

% -------------------------------------------------------------------------
% 1) Define the bv_types you want to analyze
%    (Add or remove combinations as you see fit.)
% -------------------------------------------------------------------------
bv_types = {
    'act',               % Just discrete action one-hots (0..7)
    'act+ori+sta',       % Discrete actions + orientation + stamina
    'ori+sta',           % Orientation + stamina
    'act_prob+ori+sta',  % Action probabilities + orientation + stamina
    'act_prob'           % Action probabilities only
};

% If you previously had:
% predator_bv_cols = strcat('actions_0_', string(0:7))'; ...
% ...comment those out, since we'll build them dynamically now.

% Other parameters
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

    % Load PLSC_results_dict.pkl via Python
    pkl_file_name = [video_path '/PLSC_results_dict.pkl'];
    fid = py.open(pkl_file_name, 'rb');
    data = py.pickle.load(fid);

    % ---------------------------------------------------------------------
    % 2) Outer loop over each bv_type
    % ---------------------------------------------------------------------
    for bvt_idx = 1:numel(bv_types)
        current_bv_type = bv_types{bvt_idx};

        % Build columns for predator (agent_id=0) and prey (agent_id=1)
        predator_bv_cols = getBVCols(0, current_bv_type);
        prey_bv_cols     = getBVCols(1, current_bv_type);

        % Re-initialize accumulators for *this* bv_type
        mAll = [];
        posIdxAll = [];
        negIdxAll = [];
        sigIdxAll = [];

        % -----------------------------------------------------------------
        % Loop over each prey_id (and inside that, predator_id)
        % -----------------------------------------------------------------
        for prey_id = prey_ids
            for predator_id = predator_ids

                bv_path  = fullfile(video_path, sprintf('%d_%d_info.csv', predator_id, prey_id));
                net_path = fullfile(video_path, sprintf('%d_%d_network_states.csv', predator_id, prey_id));

                bv_data  = readtable(bv_path);
                net_data = readtable(net_path);

                % Check and handle missing "orientations_0_*" / "orientations_1_*"
                bv_data = filling_missing_orientation_one_hot_vec(bv_data);

                % Normalize network state data
                net_data = array2table(zscore(table2array(net_data)), ...
                    'VariableNames', net_data.Properties.VariableNames);

                % Split net_data into hidden_0 (predator) and hidden_1 (prey)
                prey_net_data     = net_data(:, contains(net_data.Properties.VariableNames, 'hidden_1'));
                predator_net_data = net_data(:, contains(net_data.Properties.VariableNames, 'hidden_0'));

                % Restrict bv_data to the columns for this bv_type
                bv_data = bv_data(:, [predator_bv_cols; prey_bv_cols]);
                bv_data = array2table(zscore(table2array(bv_data)), ...
                    'VariableNames', bv_data.Properties.VariableNames);

                prey_net_data     = table2array(prey_net_data);
                predator_net_data = table2array(predator_net_data);

                % (Optional) If you need the "xx" structure for something else:
                xx = {};
                xx{1} = table2array(bv_data(:, prey_bv_cols));
                xx{2} = table2array(bv_data(:, predator_bv_cols));
                % Now do your PLSC dimension extraction:
                plsc_dims = cell(data{sprintf('%d_%d', predator_id, prey_id)}{'rank'});

                % For each episode, do your calculations
                for episode = 1:num_episode
                    R1 = zscore(predator_net_data(1+(episode-1)*1000 : episode*1000, :));
                    R2 = zscore(prey_net_data(1+(episode-1)*1000 : episode*1000, :));

                    [U,V,~,dU,dV] = getSharedSpace(R1,R2);
                    % In this version, you take PLSC1 and PLSC2:
                    R{1} = U(:, 1:2);  % columns 1..2
                    R{2} = V(:, 1:2);

                    % For each agent (1 or 2)
                    for agent = 1:2
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
                        mAll      = [mAll; m];
                        posIdxAll = [posIdxAll; sig1Idx];
                        negIdxAll = [negIdxAll; sig2Idx];
                        sigIdxAll = [sigIdxAll; x3Idx];
                    end
                end % episode loop

                toc
            end % predator_id loop
        end % prey_id loop

        % -----------------------------------------------------------------
        % 3) Plotting
        % -----------------------------------------------------------------
        figure();
        hold on;
        histogram(mAll(logical(posIdxAll)), 'FaceColor',[0.8,0,0], ...
            'BinWidth',0.01, 'FaceAlpha',0.4);
        histogram(mAll(logical(negIdxAll)), 'FaceColor',[0,0,0.8], ...
            'BinWidth',0.01, 'FaceAlpha',0.4);
        histogram(mAll(logical(sigIdxAll)), 'FaceColor',[0.5,0,0.5], ...
            'BinWidth',0.01, 'FaceAlpha',0.8);

        ylabel('Cells Count');
        title(['Total Cells: ' num2str(numel(mAll)) ...
               ' - Iter ' num2str(num_episode)]);
        xlabel({'|W_{PLSC1}| - |W_{PLSC2}|'});

        legend({'PLSC1','PLSC2','Both'}, 'Location','eastoutside');
        hold off;

        % We incorporate current_bv_type into the filename
        safe_bv_type = strrep(current_bv_type, '+', '_');  % replace '+' with '_'
        fig_name = ['fig_plsc1_vs_2_' suffix '_' safe_bv_type '.png'];
        saveas(gcf, fig_name);

        % -----------------------------------------------------------------
        % 4) Save the results
        % -----------------------------------------------------------------
        results = struct();
        results.mAll       = mAll;
        results.posIdxAll  = posIdxAll;
        results.negIdxAll  = negIdxAll;
        results.sigIdxAll  = sigIdxAll;

        output_mat_name = ['./results/plsc1_vs_2_' suffix '_' safe_bv_type '.mat'];
        save(output_mat_name, 'results');

    end % end of loop over bv_types

end % end of loop over video_paths

toc
disp('All done!');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HELPER FUNCTION: getBVCols
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function bv_cols = getBVCols(agent_id, bv_type)
    % This function dynamically constructs columns based on bv_type.
    % Examples:
    %   'act' -> actions_{agent_id}_0..7
    %   'act_prob' -> actions_prob_{agent_id}_0..8
    %   'ori' -> orientations_{agent_id}_0..3
    %   'sta' -> STAMINA_{agent_id}
    %   combos: 'act+ori+sta', 'act_prob+ori+sta', etc.

    bv_cols = {};

    % If includes 'act_prob', then add probability columns 0..8
    if contains(bv_type, 'act_prob')
        for i = 0:7
            bv_cols{end+1} = sprintf('actions_prob_%d_%d', agent_id, i);
        end
    end

    % If includes 'act' and not 'act_prob', then add one-hot columns 0..7
    if contains(bv_type, 'act') && ~contains(bv_type, 'act_prob')
        for i = 0:7
            bv_cols{end+1} = sprintf('actions_%d_%d', agent_id, i);
        end
    end

    % If includes 'ori', add orientations columns 0..3
    if contains(bv_type, 'ori')
        for i = 0:3
            bv_cols{end+1} = sprintf('orientations_%d_%d', agent_id, i);
        end
    end

    % If includes 'sta', add stamina
    if contains(bv_type, 'sta')
        bv_cols{end+1} = sprintf('STAMINA_%d', agent_id);
    end

    bv_cols = bv_cols(:);  % convert cell row -> cell column (optional)
end
