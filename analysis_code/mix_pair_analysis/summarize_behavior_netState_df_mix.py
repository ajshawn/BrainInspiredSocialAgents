import os
import pickle
import numpy as np
import pandas as pd
import scipy as sp
from joblib import Parallel, delayed

# Define your helper function for computing relative position based on orientation.
def ori_position(A_pos, A_orient, B_pos):
  orientation_transform = {
    0: lambda x, y: (x, y),  # UP
    1: lambda x, y: (y, -x),  # RIGHT
    2: lambda x, y: (-x, -y),  # DOWN
    3: lambda x, y: (-y, x)  # LEFT
  }
  transform = orientation_transform[A_orient]
  delta_x = B_pos[0] - A_pos[0]
  delta_y = B_pos[1] - A_pos[1]
  relative_x, relative_y = transform(delta_x, delta_y)
  return (relative_x, relative_y)


def process_pair_folder(pair_folder, analysis_out_dir, skip_existing=False):
  """
  Processes one pair folder. For each pickle file (e.g., "0_5_100.pkl")
  in the folder, it loads the results and aggregates info and network states.
  Writes two CSV files in analysis_out_dir using the base pair folder name.
  """


  print(f"Processing folder: {pair_folder}")
  info_dict = {}
  network_states_dict = {}
  serial_data_dict = {}  # if needed; otherwise you can remove it

  # Optionally, use the folder name (without the _episode_pickles suffix) as the title.
  base_pair = os.path.basename(pair_folder).replace("_episode_pickles", "")

  # Define output file names based on the base folder name (without the trailing "_episode_pickles")
  out_base = os.path.join(analysis_out_dir, base_pair)
  info_file_name = f"{out_base}_info"
  network_states_file_name = f"{out_base}_network_states"

  if skip_existing:
    if os.path.exists(info_file_name + ".csv") and os.path.exists(network_states_file_name + ".pkl"):
      print(f"Skipping existing files: {info_file_name}.csv and {network_states_file_name}.pkl")
      return
  # Loop over all .pkl files in the folder.
  pkl_files = sorted([f for f in os.listdir(pair_folder) if f.endswith('.pkl')])
  if not pkl_files:
    print(f"No pickle files found in {pair_folder}")
    return

  for pkl_file in pkl_files:
    file_path = os.path.join(pair_folder, pkl_file)
    try:
      with open(file_path, 'rb') as f:
        results = pickle.load(f)
    except Exception as e:
      print(f"Error loading {file_path}: {e}")
      continue

    # Initialize serial_data_dict entry once per folder if needed.
    if base_pair not in serial_data_dict:
      serial_data_dict[base_pair] = {k: [] for k in
                                     ['STAMINA', 'POSITION', 'ORIENTATION', 'rewards', 'actions', 'distances']}

    # Process the pickle content (assuming results is a list of dicts)
    # Append data for each episode (each file corresponds to one episode)
    # Note: adjust field names if needed.
    serial_data_dict[base_pair]['STAMINA'].append([info['STAMINA'] for info in results])
    serial_data_dict[base_pair]['POSITION'].append([info['POSITION'] for info in results])
    serial_data_dict[base_pair]['ORIENTATION'].append([info['ORIENTATION'] for info in results])
    serial_data_dict[base_pair]['rewards'].append([info['rewards'] for info in results])
    serial_data_dict[base_pair]['actions'].append([info['actions'] for info in results])

    duration = len(results)
    # Fill info_dict with basic per-timestep info.
    for key in ['STAMINA', 'POSITION', 'ORIENTATION', 'rewards', 'actions']:
      for agent_id in range(2):
        key_id = f"{key}_{agent_id}"
        if key_id not in info_dict:
          info_dict[key_id] = []
        info_dict[key_id].extend([info[key][agent_id] for info in results])

    # Compute relative positions between agents.
    positions = np.array([info['POSITION'] for info in results])
    position_0_1_x = positions[:, 0, 0] - positions[:, 1, 0]
    position_0_1_y = positions[:, 0, 1] - positions[:, 1, 1]
    position_1_0_x = -position_0_1_x
    position_1_0_y = -position_0_1_y
    for key, values in zip(['rel_position_0_1_x', 'rel_position_0_1_y', 'rel_position_1_0_x', 'rel_position_1_0_y'],
                           [position_0_1_x, position_0_1_y, position_1_0_x, position_1_0_y]):
      if key not in info_dict:
        info_dict[key] = []
      info_dict[key].extend(values)

    # Compute orientation relative positions.
    orientations = np.array([info['ORIENTATION'] for info in results]).astype(int)
    for agent_id in range(2):
      ori_positions = []
      for i in range(len(orientations)):
        opponent_id = 1 - agent_id
        ori_positions.append(ori_position(positions[i, agent_id], orientations[i, agent_id], positions[i, opponent_id]))
      ori_positions = np.array(ori_positions).astype(int)
      for ii, xyname in enumerate(['x', 'y']):
        key_name = f"ori_position_{agent_id}_{opponent_id}_{xyname}"
        if key_name not in info_dict:
          info_dict[key_name] = []
        info_dict[key_name].extend(ori_positions[:, ii])

    # Process network states if available.
    # Assume each info dict in results has keys like 'hidden' and 'cell' that are lists for each agent.
    for state_key in ['hidden', 'cell']:
      for agent_id in range(2):
        # Extract the state for the given agent from each time step.
        states = [info[state_key][agent_id] for info in results if state_key in info]
        if states:
          # If the state is a vector, convert to numpy array.
          states = np.array(states)
          # For each dimension in the state vector, append to the dict.
          for dim in range(states.shape[1]):
            key_name = f"{state_key}_{agent_id}_{dim}"
            if key_name not in network_states_dict:
              network_states_dict[key_name] = []
            network_states_dict[key_name].extend(states[:, dim])

    # (Optional) Process additional fields like actions probabilities, distances, etc.
    # Your original code contains more computations; include those as needed.

  # After processing all pickle files in the folder, convert dictionaries to DataFrames.
  info_df = pd.DataFrame(info_dict)
  network_states_df = pd.DataFrame(network_states_dict)
  # You may also process serial_data_dict if desired.

  # Save DataFrames to CSV.
  info_df.to_csv(info_file_name + ".csv", index=False)
  # network_states_df.to_csv(network_states_file_name + ".csv", index=False)
  # pickle those dataframes if needed
  # info_df.to_pickle(info_file_name + ".pkl")
  network_states_df.to_pickle(network_states_file_name + ".pkl")
  print(f"Saved info and network states to {info_file_name}.csv and {network_states_file_name}.pkl")

if __name__ == '__main__':
  # Define the base folder where the mixed pair folders are located.
  base_mix_dir = "/home/mikan/e/Documents/GitHub/social-agents-JAX/results/mix"
  # Define output folder for analysis results.
  analysis_out_dir = os.path.join(base_mix_dir, "analysis_results")
  if not os.path.exists(analysis_out_dir):
    os.makedirs(analysis_out_dir)

  # List all folders in base_mix_dir that end with "_episode_pickles"
  pair_folders = [os.path.join(base_mix_dir, d) for d in os.listdir(base_mix_dir)
                  if os.path.isdir(os.path.join(base_mix_dir, d)) and d.endswith("_episode_pickles")]
  if not pair_folders:
    print("No pair folders found in", base_mix_dir)

  # Optionally, process in parallel using joblib, or simply loop over them.
  # Here we use a simple loop.
  # for folder in pair_folders:
  #   process_pair_folder(folder, analysis_out_dir)
  Parallel(n_jobs=25)(delayed(process_pair_folder)(folder, analysis_out_dir, skip_existing=True)
                      for folder in pair_folders)

  print("All analysis files have been generated in:", analysis_out_dir)
