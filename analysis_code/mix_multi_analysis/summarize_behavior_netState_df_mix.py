import os
import pickle
import numpy as np
import pandas as pd
import scipy as sp
from joblib import Parallel, delayed
from group_analysis_utils.helper import parse_agent_roles
from group_analysis_utils.segmentation import mark_death_periods
import logging
from collections import defaultdict
import argparse

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


def process_folder(folder, analysis_out_dir, skip_existing=False):
  """
  Processes one pair folder. For each pickle file (e.g., "0_5_100.pkl")
  in the folder, it loads the results and aggregates info and network states.
  Writes two CSV files in analysis_out_dir using the base pair folder name.
  """
  try:
    pkl_dir = os.path.join(folder, 'episode_pickles')
    assert os.path.exists(pkl_dir), f"Pickle directory not found: {pkl_dir}"
    files = sorted(f for f in os.listdir(pkl_dir) if f.endswith('.pkl'))
    assert files, f"No pickle files found in {pkl_dir}"
  except AssertionError as e:
    logging.error(e)
    return None

  role, src = parse_agent_roles(os.path.basename(folder))
  predators = [i for i, r in role.items() if r == 'predator']
  preys = [i for i, r in role.items() if r == 'prey']


  print(f"Processing folder: {folder}")
  info_dict = defaultdict(list)
  network_states_dict = {}
  serial_data_dict = {}  # if needed; otherwise you can remove it

  # Optionally, use the folder name (without the _episode_pickles suffix) as the title.
  base_pair = os.path.basename(folder).replace("_episode_pickles", "")

  # Define output file names based on the base folder name (without the trailing "_episode_pickles")
  out_base = os.path.join(analysis_out_dir, base_pair)
  info_file_name = f"{out_base}_info"
  network_states_file_name = f"{out_base}_network_states"

  if skip_existing:
    if os.path.exists(info_file_name + ".csv") and os.path.exists(network_states_file_name + ".pkl"):
      print(f"Skipping existing files: {info_file_name}.csv and {network_states_file_name}.pkl")
      return
  # Loop over all .pkl files in the folder.
  pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith('.pkl')])
  if not pkl_files:
    print(f"No pickle files found in {folder}")
    return

  for pkl_file in pkl_files:
    results = pickle.load(open(os.path.join(folder, 'episode_pickles', pkl_file), 'rb'))

    # Initialize serial_data_dict entry once per folder if needed.
    # if base_pair not in serial_data_dict:
    #   serial_data_dict[base_pair] = {k: [] for k in
    #                                  ['STAMINA', 'POSITION', 'ORIENTATION', 'rewards', 'actions', 'distances', 'death']}

    # Process the pickle content (assuming results is a list of dicts)
    # Append data for each episode (each file corresponds to one episode)
    # Note: adjust field names if needed.
    # serial_data_dict[base_pair]['STAMINA'].append([info['STAMINA'] for info in results])
    # serial_data_dict[base_pair]['POSITION'].append([info['POSITION'] for info in results])
    # serial_data_dict[base_pair]['ORIENTATION'].append([info['ORIENTATION'] for info in results])
    # serial_data_dict[base_pair]['rewards'].append([info['rewards'] for info in results])
    # serial_data_dict[base_pair]['actions'].append([info['actions'] for info in results])
    # serial_data_dict[base_pair]['death'].append([mark_death_periods(info['STAMINA']) for info in results])

    # duration = len(results)
    # Fill info_dict with basic per-timestep info.
    num_agents = len(role)  # Or len(results[0]['STAMINA']) if role isn't available
    general_keys = ['STAMINA', 'ORIENTATION', 'rewards', 'actions']
    all_keys = ['STAMINA', 'ORIENTATION', 'rewards', 'actions', 'POSITION_x', 'POSITION_y']

    # Iterate through each agent first
    for agent_id in range(num_agents):
      # Prepare lists for each key and agent
      for key in general_keys:
        key_id = f"{key}_{agent_id}"
        # No need for `if key_id not in info_dict:` due to defaultdict
        # info_dict[key_id] will be an empty list if not accessed before

      # Now, iterate through each timestep (info dictionary) for the current agent
      for info in results:
        # Populate general keys
        for key in general_keys:
          info_dict[f"{key}_{agent_id}"].append(info[key][agent_id])

        # Populate POSITION data
        pos_data = info['POSITION'][agent_id]
        info_dict[f"POSITION_x_{agent_id}"].append(pos_data[0])
        info_dict[f"POSITION_y_{agent_id}"].append(pos_data[1])

      # After collecting all data for the current agent, compute death labels
      # This ensures mark_death_periods receives the full history for this agent
      stamina_history = info_dict[f"STAMINA_{agent_id}"][:len(results)]
      death_labels = mark_death_periods([val['STAMINA'][agent_id] for val in results])
      info_dict[f"death_{agent_id}"].extend(death_labels)

    # Compute relative positions between agents.
    # positions = np.array([info['POSITION'] for info in results])
    # position_0_1_x = positions[:, 0, 0] - positions[:, 1, 0]
    # position_0_1_y = positions[:, 0, 1] - positions[:, 1, 1]
    # position_1_0_x = -position_0_1_x
    # position_1_0_y = -position_0_1_y
    # for key, values in zip(['rel_position_0_1_x', 'rel_position_0_1_y', 'rel_position_1_0_x', 'rel_position_1_0_y'],
    #                        [position_0_1_x, position_0_1_y, position_1_0_x, position_1_0_y]):
    #   if key not in info_dict:
    #     info_dict[key] = []
    #   info_dict[key].extend(values)

    # # Compute orientation relative positions.
    # orientations = np.array([info['ORIENTATION'] for info in results]).astype(int)
    # for agent_id in range(2):
    #   ori_positions = []
    #   for i in range(len(orientations)):
    #     opponent_id = 1 - agent_id
    #     ori_positions.append(ori_position(positions[i, agent_id], orientations[i, agent_id], positions[i, opponent_id]))
    #   ori_positions = np.array(ori_positions).astype(int)
    #   for ii, xyname in enumerate(['x', 'y']):
    #     key_name = f"ori_position_{agent_id}_{opponent_id}_{xyname}"
    #     if key_name not in info_dict:
    #       info_dict[key_name] = []
    #     info_dict[key_name].extend(ori_positions[:, ii])

    # Process network states if available.
    # Assume each info dict in results has keys like 'hidden' and 'cell' that are lists for each agent.
    for state_key in ['hidden', 'cell']:
      for agent_id in range(len(role)):
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

  p = argparse.ArgumentParser()
  p.add_argument('--base_dir', default='/home/mikan/e/GitHub/social-agents-JAX/results/RandomForest20250424_2v4_rollouts/',)
  p.add_argument('--out_dir', default='/home/mikan/e/GitHub/social-agents-JAX/results/RandomForest20250424_2v4_rollouts/analysis_results/',
                  help='Output directory for the analysis results')
  p.add_argument('--jobs', type=int, default=50)
  args = p.parse_args()

  base_dir = args.base_dir
  out_dir = args.out_dir
  os.makedirs(out_dir, exist_ok=True)

  # Find all valid pair folders
  folders = [os.path.join(base_dir, d)
             for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d))
             and 'episode_pickles' in os.listdir(os.path.join(base_dir, d))
             and 'simplified10x10' not in d]
  folders = [os.path.join(args.base_dir, d) for d in os.listdir(args.base_dir)
             if os.path.isdir(os.path.join(args.base_dir, d)) and 'episode_pickles' in os.listdir(
             os.path.join(args.base_dir, d))]
  if not folders:
    folders = [
      os.path.join(args.base_dir, d1, d2)  # <-- full path we keep
      for d1 in os.listdir(args.base_dir)  # level‑1 dirs
      for d2 in os.listdir(os.path.join(args.base_dir, d1))  # level‑2 dirs
      if os.path.isdir(os.path.join(args.base_dir, d1, d2)) and
         'episode_pickles' in os.listdir(os.path.join(args.base_dir, d1, d2))
    ]
  # Optionally, process in parallel using joblib, or simply loop over them.
  # Here we use a simple loop.
  # for folder in folders:
  #   process_folder(folder, out_dir)
  Parallel(n_jobs=20)(delayed(process_folder)(folder, out_dir, skip_existing=False)
                      for folder in folders)

  print("All analysis files have been generated in:", out_dir)
