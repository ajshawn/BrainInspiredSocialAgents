% 
% def process_episode(data_directory, trial_directory, checkpoint, case_name, file_suffix, eId, pls, timesteps=500, num_perm=100, permutations=None):
%   """Function to process a single episode."""
%   data_path = f"{trial_directory}out_files/behavior_output_{checkpoint}iters_test_{eId + 1}{file_suffix}"
%   data = sio.loadmat(data_path)
%   R1 = data["R1"][:timesteps]
%   R2 = data["R2"][:timesteps]
trial_names = ["v3d_21", "v3d_22", "v3d_2", "v3d_1", "v1d_21", "v1d_22", ];
for trial_name = trial_names
for run = 1:26
try
    trial_directory = fullfile(data_directory, [trial_name, '/'], ...
                               ['RNN_output_Arena', trial_name(2:end), '_rec3_inp0_clip0p3_run', num2str(run), '/']);
    if ~exist(trial_directory, 'dir')
        trial_directory = fullfile(data_directory2, [trial_name, '/'], ...
                                   ['RNN_output_Arena', trial_name(2:end), '_rec3_inp0_clip0p3_run', num2str(run), '/']);
    end
    if ~exist(trial_directory, 'dir')
        fprintf('%s not found\n', trial_directory);
        continue; % This should be inside a loop to work; otherwise, you need to handle it differently.
    end
    fprintf('processing trial %s\n', trial_directory);
catch ME
    disp('Error processing directory path:');
    disp(getReport(ME));
end