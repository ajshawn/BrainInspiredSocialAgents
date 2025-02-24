% Define the video path
video_paths = {
  '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles/',
  '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles/',
  '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles/',
};
suffices = {
  'cp7357', 'cp9651', 'AH'
  };
bv_types = {
  'act+ori+sta',
  'act',
  'ori+sta',
  'act_prob+ori+sta',
  'act_prob',
  }
num_episode = 100;


% Define simple action_cols, orientation_cols, and stamina_cols
predator_bv_cols = strcat('actions_0_', string(0:7))';
predator_bv_cols = [predator_bv_cols; strcat('orientations_0_', string(0:3))'; 'STAMINA_0'];
prey_bv_cols = strcat('actions_1_', string(0:7))';
prey_bv_cols = [prey_bv_cols; strcat('orientations_1_', string(0:3))'; 'STAMINA_1'];
% Initialize KFold (no direct equivalent in MATLAB for sklearn's KFold)
% You would manually need to implement cross-validation if necessary
num_perm = 50;
thr=0.01; thrSig = 1.5;
tic
for ii = 1:numel(video_paths)
  mAll=[];posIdxAll=[];negIdxAll=[];neuIdxAll=[];sigIdxAll=[];
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

  % Initialize KFold (no direct equivalent in MATLAB for sklearn's KFold)
  % You would manually need to implement cross-validation if necessary
  num_perm = 50;
  results = [];

  tic
  % Loop through each prey_id
  for prey_id = prey_ids

    % Concatenate data from multiple files
    for predator_id = predator_ids
      bv_path = fullfile(video_path, sprintf('%d_%d_info.csv', predator_id, prey_id));
      net_path = fullfile(video_path, sprintf('%d_%d_network_states.csv', predator_id, prey_id));
      bv_data = readtable(bv_path);
      net_data = readtable(net_path);
      % Check and handle missing "orientations_0_*" and "orientations_1_*" fields
      bv_data = filling_missing_orientation_one_hot_vec(bv_data);
      % Normalize network state data
      net_data = array2table(zscore(table2array(net_data)), 'VariableNames', net_data.Properties.VariableNames);
      predator_net_data = net_data(:, contains(net_data.Properties.VariableNames, 'hidden_0'));
      prey_net_data = net_data(:, contains(net_data.Properties.VariableNames, 'hidden_1'));

      bv_data = bv_data(:, [predator_bv_cols; prey_bv_cols]);
      bv_data = array2table(zscore(table2array(bv_data)), 'VariableNames', bv_data.Properties.VariableNames);

      % net_data = table2array(net_data);
      prey_net_data = table2array(prey_net_data);
      predator_net_data = table2array(predator_net_data);
      % bv_prey_data = table2array(bv_prey_data);
      xx = {};
      xx{1} = table2array(bv_data(:, [prey_bv_cols]));
      xx{2} = table2array(bv_data(:, [predator_bv_cols]));
      prey_score = computeNonRedundantVar(xx, prey_net_data, 10, num_perm);

      xx = {};
      xx{1} = table2array(bv_data(:, [predator_bv_cols]));
      xx{2} = table2array(bv_data(:, [prey_bv_cols]));
      predator_score = computeNonRedundantVar(xx, predator_net_data, 10, num_perm);
      results.(['pair_' num2str(predator_id) '_' num2str(prey_id) '_prey']) = prey_score;
      results.(['pair_' num2str(predator_id) '_' num2str(prey_id) '_predator']) = predator_score;
      toc
    end

  end
  %%
  % Loop through each prey_id
  for prey_id_mask = 1:numel(prey_ids)
    prey_id = prey_ids(prey_id_mask);
    bv_data = [];
    net_data = [];

    % Concatenate data from multiple files
    for predator_id = predator_ids
      bv_path = fullfile(video_path, sprintf('%d_%d_info.csv', predator_id, prey_id));
      net_path = fullfile(video_path, sprintf('%d_%d_network_states.csv', predator_id, prey_id));
      bv_temp = readtable(bv_path);
      net_temp = readtable(net_path);
      bv_data = [bv_data; bv_temp];
      net_data = [net_data; net_temp];
    end

    % Normalize network state data
    net_data = array2table(zscore(table2array(net_data)), 'VariableNames', net_data.Properties.VariableNames);
    net_data = net_data(:, contains(net_data.Properties.VariableNames, 'hidden_1'));
    bv_data = filling_missing_orientation_one_hot_vec(bv_data);
    bv_data = bv_data(:, [predator_bv_cols; prey_bv_cols]);
    bv_data = array2table(zscore(table2array(bv_data)), 'VariableNames', bv_data.Properties.VariableNames);

    net_data = table2array(net_data);
    % bv_prey_data = table2array(bv_prey_data);

    xx = {};
    xx{1} = table2array(bv_data(:, [prey_bv_cols]));
    xx{2} = table2array(bv_data(:, [predator_bv_cols]));
    test = computeNonRedundantVar(xx, net_data, 10, num_perm)
    results.(['prey_' num2str(prey_id)]) = test;
    toc
  end

  for predator_id = predator_ids
    bv_data = [];
    net_data = [];

    % Concatenate data from multiple files
    for prey_id = prey_ids
      bv_path = fullfile(video_path, sprintf('%d_%d_info.csv', predator_id, prey_id));
      net_path = fullfile(video_path, sprintf('%d_%d_network_states.csv', predator_id, prey_id));
      bv_temp = readtable(bv_path);
      net_temp = readtable(net_path);
      bv_data = [bv_data; bv_temp];
      net_data = [net_data; net_temp];

    end

    % Normalize network state data
    net_data = array2table(zscore(table2array(net_data)), 'VariableNames', net_data.Properties.VariableNames);
    net_data = net_data(:, contains(net_data.Properties.VariableNames, 'hidden_0'));
    bv_data = filling_missing_orientation_one_hot_vec(bv_data);

    bv_data = bv_data(:, [predator_bv_cols; prey_bv_cols]);
    bv_data = array2table(zscore(table2array(bv_data)), 'VariableNames', bv_data.Properties.VariableNames);

    net_data = table2array(net_data);
    % bv_prey_data = table2array(bv_prey_data);
    xx = {};

    xx{1} = table2array(bv_data(:, [predator_bv_cols]));
    xx{2} = table2array(bv_data(:, [prey_bv_cols]));
    test = computeNonRedundantVar(xx, net_data, 10, num_perm)
    results.(['predator_' num2str(predator_id)]) = test;
    toc
  end

  % save('bv_rep.mat', 'results')

  %%
  zero_mat = zeros(5,5);
  prey_self_rep_matrix = zero_mat;
  predator_self_rep_matrix = zero_mat;
  prey_partner_rep_matrix = zero_mat;
  predator_partner_rep_matrix = zero_mat;
  for aId = predator_ids
    for bId = prey_ids
      preyField = sprintf('pair_%d_%d_prey', aId, bId);
      predatorField = sprintf('pair_%d_%d_predator', aId, bId);

      if isfield(results, preyField) && isfield(results, predatorField)
        prey_self_rep_matrix(aId+1, bId+1) = results.(preyField){1};
        predator_self_rep_matrix(aId+1, bId+1) = results.(predatorField){1};
        prey_partner_rep_matrix(aId+1, bId+1) = results.(preyField){2};
        predator_partner_rep_matrix(aId+1, bId+1) = results.(predatorField){2};
      end
    end
  end


  % Creating heatmaps with annotations
  figure;
  subplot(2, 2, 1);
  h1 = heatmap(prey_self_rep_matrix);
  h1.Title = 'Prey Self Rep Matrix';
  h1.XLabel = 'BId';
  h1.YLabel = 'AId';
  h1.ColorbarVisible = 'on';
  h1.CellLabelFormat = '%0.2f';

  subplot(2, 2, 2);
  h2 = heatmap(predator_self_rep_matrix);
  h2.Title = 'Predator Self Rep Matrix';
  h2.XLabel = 'BId';
  h2.YLabel = 'AId';
  h2.ColorbarVisible = 'on';
  h2.CellLabelFormat = '%0.2f';

  subplot(2, 2, 3);
  h3 = heatmap(prey_partner_rep_matrix);
  h3.Title = 'Prey Partner Rep Matrix';
  h3.XLabel = 'BId';
  h3.YLabel = 'AId';
  h3.ColorbarVisible = 'on';
  h3.CellLabelFormat = '%0.2f';

  subplot(2, 2, 4);
  h4 = heatmap(predator_partner_rep_matrix);
  h4.Title = 'Predator Partner Rep Matrix';
  h4.XLabel = 'BId';
  h4.YLabel = 'AId';
  h4.ColorbarVisible = 'on';
  h4.CellLabelFormat = '%0.2f';

  % Save the figure as a PNG file
  saveas(gcf, 'interaction_heatmaps_annotated.png');

  %%
  % Load the .mat file
  % data = load('bv_rep_finished.mat');
  % results = data.results;

  %%
  % Define the range of indices for prey and predator
  prey_indices = 0:4;
  predator_indices = 0:4;

  % Initialize arrays to store self and partner representations
  prey_self_rep = zeros(length(prey_indices), 1);
  prey_partner_rep = zeros(length(prey_indices), 1);
  predator_self_rep = zeros(length(predator_indices), 1);
  predator_partner_rep = zeros(length(predator_indices), 1);

  % Extract data for prey
  for i = 1:length(prey_indices)
    preyField = sprintf('prey_%d', prey_indices(i));
    if isfield(results, preyField)
      prey_self_rep(i) = results.(preyField){1}; % Self representation
      prey_partner_rep(i) = results.(preyField){2}; % Partner representation
    end
  end

  % Extract data for predator
  for j = 1:length(predator_indices)
    predatorField = sprintf('predator_%d', predator_indices(j));
    if isfield(results, predatorField)
      predator_self_rep(j) = results.(predatorField){1}; % Self representation
      predator_partner_rep(j) = results.(predatorField){2}; % Partner representation
    end
  end

  % Plotting
  figure;
  subplot(2, 1, 1);
  bar([prey_self_rep, prey_partner_rep]);
  title('Prey Self vs. Partner Representation');
  legend('Self Rep', 'Partner Rep');
  set(gca, 'XTickLabel', arrayfun(@num2str, prey_indices, 'UniformOutput', false));
  xlabel('Prey Index');
  ylabel('Representation Value');

  subplot(2, 1, 2);
  bar([predator_self_rep, predator_partner_rep]);
  title('Predator Self vs. Partner Representation');
  legend('Self Rep', 'Partner Rep');
  set(gca, 'XTickLabel', arrayfun(@num2str, predator_indices, 'UniformOutput', false));
  xlabel('Predator Index');
  ylabel('Representation Value');

  % Adjust layout
  % tight_layout();


  %% Now save those matrix
  result_dict = struct;
  result_dict.prey_self_rep_matrix = prey_self_rep_matrix;
  result_dict.predator_self_rep_matrix = predator_self_rep_matrix;
  result_dict.prey_partner_rep_matrix = prey_partner_rep_matrix;
  result_dict.predator_partner_rep_matrix = predator_partner_rep_matrix;
  result_dict.prey_self_rep_overall = prey_self_rep;
  result_dict.predator_self_rep_overall = predator_self_rep;
  result_dict.prey_partner_rep_overall = prey_partner_rep;
  result_dict.predator_partner_rep_overall = predator_partner_rep;

  save(['./results/'...
    'partner_rep_result' suffix '.mat'], 'result_dict');
end