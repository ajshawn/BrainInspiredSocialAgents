import os
import pickle
import numpy as np
import pandas as pd
# import scipy as sp # scipy was imported but not used in the provided snippet
from joblib import Parallel, delayed
from group_analysis_utils.helper import parse_agent_roles
from group_analysis_utils.segmentation import mark_death_periods
import logging
from collections import defaultdict
import argparse  # Added
from pathlib import Path  # Added
import re  # Added


# Define your helper function for computing relative position based on orientation.
# This function is defined in code 3 but not directly used in the main processing loop shown.
# It's kept here as it was part of the original code.
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


def process_folder(run_folder_str: str, analysis_out_dir_str: str, skip_existing: bool = False):
  """
  Processes one run_folder. For each pickle file in the 'episode_pickles'
  subdirectory, it loads the results and aggregates info and network states.
  Writes two files (one CSV for info, one PKL for network_states)
  in analysis_out_dir, named using checkpoint number and run folder name.
  """
  run_folder = Path(run_folder_str)
  analysis_out_dir = Path(analysis_out_dir_str)

  try:
    pkl_dir = run_folder / 'episode_pickles'
    assert pkl_dir.exists(), f"Pickle directory not found: {pkl_dir}"
    files = sorted(pkl_dir.glob("*.pkl"))
    assert files, f"No pickle files found in {pkl_dir}"
  except AssertionError as e:
    logging.error(e)
    return None

  # Infer ckpt number N from the parent of run_folder
  # e.g. run_folder = .../mix_RF_ckpt42/<run-subdir>
  ckpt_dir = run_folder.parent
  # This regex assumes a pattern like "mix_RF_ckpt<number>". Adjust if the new dataset's pattern is different.
  m = re.search(r"mix_RF_ckpt(\d+)", ckpt_dir.name)
  if not m:
    # Fallback if primary regex doesn't match, try a more generic "ckpt<number>" or "ckpt_<number>"
    m = re.search(r"ckpt_?(\d+)", ckpt_dir.name)
    if not m:
      logging.warning(
        f"Could not parse checkpoint number from {ckpt_dir.name} for run {run_folder.name}. Using 'unknown_ckpt'.")
      ckpt_num_str = "unknown_ckpt"
    else:
      ckpt_num_str = m.group(1)
  else:
    ckpt_num_str = m.group(1)

  run_name = run_folder.name

  # Define output file names
  info_file = analysis_out_dir / f"ckpt{ckpt_num_str}_{run_name}_info.csv"
  network_states_file = analysis_out_dir / f"ckpt{ckpt_num_str}_{run_name}_network_states.pkl"

  if skip_existing and info_file.exists() and network_states_file.exists():
    logging.info(f"Skipping existing files for {run_folder.name} (ckpt {ckpt_num_str})")
    return

  role, src = parse_agent_roles(run_folder.name)
  # predators = [i for i, r in role.items() if r == 'predator'] # Defined but not used in snippet
  # preys = [i for i, r in role.items() if r == 'prey'] # Defined but not used in snippet

  logging.info(f"Processing folder: {run_folder.name} (ckpt {ckpt_num_str})")
  info_dict = defaultdict(list)
  network_states_dict = defaultdict(list)  # Changed to defaultdict for convenience

  for pkl_path in files:
    try:
      with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    except Exception as e:
      logging.error(f"Error loading pickle file {pkl_path}: {e}")
      continue  # Skip this problematic file

    if not results:  # Check if results is empty
      logging.warning(f"No data in pickle file: {pkl_path}")
      continue

    # Determine num_agents. Prefer len(role) if available and consistent.
    # Fallback to data if role is not perfectly aligned or for robustness.
    try:
      num_agents = len(role)
      # Quick check: ensure first record has data for this many agents if possible
      if results and 'STAMINA' in results[0] and len(results[0]['STAMINA']) != num_agents:
        logging.warning(
          f"Mismatch between len(role)={num_agents} and data agents={len(results[0]['STAMINA'])} in {pkl_path}. Using data length.")
        num_agents = len(results[0]['STAMINA'])
    except TypeError:  # if role is None or not a sized object
      logging.warning(
        f"Role information not available or invalid for {run_folder.name}. Inferring num_agents from data in {pkl_path}.")
      if results and 'STAMINA' in results[0] and hasattr(results[0]['STAMINA'], '__len__'):
        num_agents = len(results[0]['STAMINA'])
      else:
        logging.error(f"Cannot determine number of agents for {pkl_path}. Skipping this file.")
        continue

    general_keys = ['STAMINA', 'ORIENTATION', 'rewards', 'actions']

    # Data for current episode (single pkl file)
    current_episode_stamina_per_agent = [[] for _ in range(num_agents)]

    for timestep_data in results:
      for agent_id in range(num_agents):
        # Populate general keys
        for key in general_keys:
          try:
            info_dict[f"{key}_{agent_id}"].append(timestep_data[key][agent_id])
            if key == 'STAMINA':
              current_episode_stamina_per_agent[agent_id].append(timestep_data[key][agent_id])
          except (IndexError, KeyError) as e:
            logging.warning(
              f"Missing data for {key}_{agent_id} at a timestep in {pkl_path}. Error: {e}. Appending NaN.")
            info_dict[f"{key}_{agent_id}"].append(np.nan)
            if key == 'STAMINA':
              current_episode_stamina_per_agent[agent_id].append(
                {'STAMINA': np.nan})  # Adjust if STAMINA structure is different

        # Populate POSITION data
        try:
          pos_data = timestep_data['POSITION'][agent_id]
          info_dict[f"POSITION_x_{agent_id}"].append(pos_data[0])
          info_dict[f"POSITION_y_{agent_id}"].append(pos_data[1])
        except (IndexError, KeyError) as e:
          logging.warning(
            f"Missing position data for agent {agent_id} at a timestep in {pkl_path}. Error: {e}. Appending NaN.")
          info_dict[f"POSITION_x_{agent_id}"].append(np.nan)
          info_dict[f"POSITION_y_{agent_id}"].append(np.nan)

    # Compute death labels per agent for the current episode
    for agent_id in range(num_agents):
      # Ensure that the structure passed to mark_death_periods is as expected by the function
      # Original code: mark_death_periods([val['STAMINA'][agent_id] for val in results])
      # Assuming mark_death_periods expects a list of stamina values for one agent over time
      stamina_for_agent_episode = [s_val for s_val in current_episode_stamina_per_agent[agent_id]]
      if stamina_for_agent_episode:  # only if there's stamina data
        # If stamina values are dicts like {'STAMINA': value}, extract the value. Adjust if structure differs.
        if isinstance(stamina_for_agent_episode[0], dict) and 'STAMINA' in stamina_for_agent_episode[0]:
          stamina_values = [s['STAMINA'] for s in stamina_for_agent_episode]
        else:  # assume it's already a list of numerical stamina values
          stamina_values = stamina_for_agent_episode
        death_labels = mark_death_periods(stamina_values)
        info_dict[f"death_{agent_id}"].extend(death_labels)
      elif results:  # if results were processed but this agent had no stamina
        info_dict[f"death_{agent_id}"].extend([np.nan] * len(results))

    # Process network states if available.
    for state_key in ['hidden', 'cell']:
      if results and state_key in results[0]:  # Check if key exists in the data
        for agent_id in range(num_agents):
          try:
            states = [timestep_data[state_key][agent_id] for timestep_data in results]
            if states:
              states = np.array(states)
              if states.ndim == 1:  # If states are scalar, reshape to (N, 1)
                states = states.reshape(-1, 1)
              for dim in range(states.shape[1]):
                network_states_dict[f"{state_key}_{agent_id}_{dim}"].extend(states[:, dim])
          except (IndexError, KeyError) as e:
            logging.warning(f"Missing network state {state_key} for agent {agent_id} in {pkl_path}. Error: {e}")
            # Extend with NaNs to maintain row alignment if other agents/dims have data for this episode
            if results:
              network_states_dict[f"{state_key}_{agent_id}_{0 if states.ndim == 1 else dim}"].extend(
                [np.nan] * len(results))

  # After processing all pickle files in the folder, convert dictionaries to DataFrames.
  if not info_dict and not network_states_dict:
    logging.warning(f"No data extracted from any pickle files in {run_folder}. Skipping output generation.")
    return

  try:
    info_df = pd.DataFrame(info_dict)
    info_df.to_csv(info_file, index=False)
    logging.info(f"Saved info data to {info_file}")
  except Exception as e:
    logging.error(f"Failed to save info_df for {run_folder.name}: {e}")

  try:
    network_states_df = pd.DataFrame(network_states_dict)
    network_states_df.to_pickle(network_states_file)
    logging.info(f"Saved network states to {network_states_file}")
  except Exception as e:
    logging.error(f"Failed to save network_states_df for {run_folder.name}: {e}")


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

  ap = argparse.ArgumentParser(
    description="Process episode pickle files to extract aggregated info and network states.")
  ap.add_argument('--base_dir', default='../../results/mix_RF3',  # Example: similar to code 2's mix_RF3
                  help="Root directory containing checkpoint folders (e.g., mix_RF_ckpt*) which in turn contain run subdirectories with 'episode_pickles'.")
  ap.add_argument('--jobs', '-j', type=int, default=60,
                  help="Number of parallel jobs to run.")
  ap.add_argument('--skip_existing', action='store_true',
                  help="If set, skip processing for folders where output files already exist.")
  ap.add_argument('--output_dir_name', default="analysis_aggregated_states",
                  help="Name of the subdirectory within base_dir where results will be saved.")
  ap.add_argument('--ckpt_pattern', default="mix_RF_ckpt*",
                  help="Pattern to find checkpoint directories within base_dir (e.g., 'ckpt_*', 'model_run_*').")

  args = ap.parse_args()

  base = Path(args.base_dir)
  analysis_main_out_dir = base / args.output_dir_name  # Output directory defined by arg
  analysis_main_out_dir.mkdir(exist_ok=True, parents=True)

  run_folders_to_process = []
  # Use glob to find checkpoint directories based on the pattern
  for ckpt_dir_path in base.glob(args.ckpt_pattern):
    if not ckpt_dir_path.is_dir():
      continue
    for run_sub_dir_path in ckpt_dir_path.iterdir():
      # Check if 'episode_pickles' exists and is a directory within the run_sub_dir_path
      if run_sub_dir_path.is_dir() and (run_sub_dir_path / "episode_pickles").is_dir():
        # Apply filters if any - e.g. 'simplified10x10' from original code 3
        # if 'simplified10x10' not in run_sub_dir_path.name:
        run_folders_to_process.append(str(run_sub_dir_path))

  if not run_folders_to_process:
    logging.warning(f"No run folders with 'episode_pickles' found under {base} using pattern {args.ckpt_pattern}.")
  else:
    logging.info(f"Found {len(run_folders_to_process)} run folders to process.")

    Parallel(n_jobs=args.jobs)(
      delayed(process_folder)(folder_path, str(analysis_main_out_dir), args.skip_existing)
      for folder_path in sorted(run_folders_to_process)
    )
    logging.info(f"All analysis files have been generated in: {analysis_main_out_dir}")