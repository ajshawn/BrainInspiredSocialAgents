#!/usr/bin/env python3
"""
analyze_mixed_pair_results_parallel.py

This script processes the reorganized mixed cross‐rollout result folders in parallel.
Each folder in
  /home/mikan/e/Documents/GitHub/social-agents-JAX/results/mix/analysis_results
that ends with "_episode_pickles" corresponds to a unique predator–prey pair.
For each such folder, the script loads all episode pickle files, aggregates detailed
(serial) data and cumulative summary metrics (including a new stuck rate metric),
and saves two CSV files into a "combined" subdirectory.
For example, if a folder is named:
  OR20250210_agent3_dim256_vs_AH20250210_agent11_dim256_episode_pickles
the script will output:
  OR20250210_agent3_dim256_vs_AH20250210_agent11_dim256_serial_results.csv
  OR20250210_agent3_dim256_vs_AH20250210_agent11_dim256_cumulative_results.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


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


def compute_stuck_rate(pos, min_duration=50):
  """
  Given a 2D position array pos of shape (T,2), compute the fraction of timesteps that are "stuck".
  A contiguous block of timesteps is marked as stuck if the agent's position does not change
  for at least min_duration timesteps. The entire duration of that block is then marked as stuck.
  Returns a float between 0 and 1 (the stuck rate).
  """
  T = pos.shape[0]
  stuck_indicator = np.zeros(T, dtype=int)
  i = 0
  while i < T - 1:
    # If current and next positions are identical (allowing for floating-point equality)
    if np.allclose(pos[i], pos[i + 1]):
      j = i
      while j < T - 1 and np.allclose(pos[j], pos[j + 1]):
        j += 1
      # Now, positions from index i to j (inclusive) did not change.
      if (j - i + 1) >= min_duration:
        stuck_indicator[i:j + 1] = 1
      i = j + 1
    else:
      i += 1
  return np.mean(stuck_indicator), stuck_indicator


def process_pair_folder(folder_path):
  """
  Process one pair folder.
  folder_path: Directory containing episode pickle files with names like "0_5_100.pkl".
  Returns two dictionaries: one with detailed (serial) data and one with cumulative summary metrics.
  """
  print(f"Processing folder: {folder_path}")
  base_pair = os.path.basename(folder_path).replace("_episode_pickles", "")
  # For simplicity, we use base_pair as key.
  serial_data_dict = {
    base_pair: {key: [] for key in ['STAMINA', 'POSITION', 'ORIENTATION', 'rewards',
                                    'actions', 'distances', 'stuck_indicator']}} # 'INVENTORY', 'READY_TO_SHOOT' are empty
  cumulative_dict = {base_pair: {key: [] for key in ['round', 'time_per_round', 'prey_move_distances_per_round',
                                                     'predator_move_distances_per_round',
                                                     'num_apple_collected_per_round',
                                                     'num_acorn_collected_per_round', 'prey_rotate_per_round',
                                                     'predator_rotate_per_round',
                                                     'time_on_grass_per_round', 'time_off_grass_per_round',
                                                     'frac_off_grass_per_round',
                                                     'frac_moving_away_per_round', 'frac_time_in_3_steps',
                                                     'frac_time_in_5_steps',
                                                     'pred_stuck_rate', 'prey_stuck_rate',
                                                     ]}}

  info_dict = {}  # Optional detailed per-timestep info.

  pkl_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pkl')])
  if not pkl_files:
    print(f"No pickle files found in {folder_path}")
    return None, None

  for pkl_file in pkl_files:
    file_path = os.path.join(folder_path, pkl_file)
    try:
      with open(file_path, 'rb') as f:
        results = pickle.load(f)
    except Exception as e:
      print(f"Error loading {file_path}: {e}")
      continue

    # Append data for this episode.
    # serial_data_dict[base_pair]['INVENTORY'].append([info['INVENTORY'] for info in results])
    # serial_data_dict[base_pair]['READY_TO_SHOOT'].append([info['READY_TO_SHOOT'] for info in results])
    serial_data_dict[base_pair]['STAMINA'].append([info['STAMINA'] for info in results])
    serial_data_dict[base_pair]['POSITION'].append([info['POSITION'] for info in results])
    serial_data_dict[base_pair]['ORIENTATION'].append([info['ORIENTATION'] for info in results])
    serial_data_dict[base_pair]['rewards'].append([info['rewards'] for info in results])
    serial_data_dict[base_pair]['actions'].append([info['actions'] for info in results])

    # Compute distances.
    positions = np.array([info['POSITION'] for info in results])  # shape: (T, num_agents, 2)
    num_agents = 2
    distances = []
    for i in range(num_agents):
      pos = positions[:, i, :]
      dist = np.linalg.norm(np.diff(pos, axis=0), axis=1)
      dist = np.concatenate(([0], dist))
      distances.append(dist)
    serial_data_dict[base_pair]['distances'].append(distances)

    # Compute stuck rates for each agent over the entire episode.
    pred_pos = positions[:, 0, :]
    prey_pos = positions[:, 1, :]
    pred_stuck, pred_stuck_ind = compute_stuck_rate(pred_pos, min_duration=20)
    prey_stuck, prey_stuck_ind = compute_stuck_rate(prey_pos, min_duration=20)
    serial_data_dict[base_pair]['stuck_indicator'].append([pred_stuck_ind.astype(bool), prey_stuck_ind.astype(bool)])

    # (Optional) Populate info_dict with per-timestep data.
    for key in ['STAMINA', 'POSITION', 'ORIENTATION', 'rewards', 'actions']:
      info_dict.setdefault(key, []).extend([info[key] for info in results])

    # Relative positions.
    rel_pos_0_1_x = positions[:, 0, 0] - positions[:, 1, 0]
    rel_pos_0_1_y = positions[:, 0, 1] - positions[:, 1, 1]
    rel_pos_1_0_x = -rel_pos_0_1_x
    rel_pos_1_0_y = -rel_pos_0_1_y
    for k, v in zip(['rel_position_0_1_x', 'rel_position_0_1_y', 'rel_position_1_0_x', 'rel_position_1_0_y'],
                    [rel_pos_0_1_x, rel_pos_0_1_y, rel_pos_1_0_x, rel_pos_1_0_y]):
      info_dict.setdefault(k, []).extend(v)

    # Orientation-based relative positions.
    orientations = np.array([info['ORIENTATION'] for info in results]).astype(int)
    for agent_id in range(num_agents):
      ori_pos = []
      for i in range(len(orientations)):
        opp_id = 1 - agent_id
        ori_pos.append(ori_position(positions[i, agent_id], orientations[i, agent_id], positions[i, opp_id]))
      ori_pos = np.array(ori_pos).astype(int)
      key_x = f"ori_position_{agent_id}_{1 - agent_id}_x"
      key_y = f"ori_position_{agent_id}_{1 - agent_id}_y"
      info_dict.setdefault(key_x, []).extend(ori_pos[:, 0])
      info_dict.setdefault(key_y, []).extend(ori_pos[:, 1])

    # Compute cumulative summary metrics per episode.
    rewards_arr = np.array([info['rewards'] for info in results])
    safe_grass = [[i, j] for i in [8, 9, 10] for j in [4, 5, 6]]
    t_catch = np.where(rewards_arr[:, 0] == 1)[0]
    t_apple = np.where(rewards_arr[:, 1] == 1)[0]
    t_acorn = np.where(rewards_arr[:, 1] > 1)[0]
    t_respawn = t_catch + 21
    t_respawn = np.insert(t_respawn, 0, 0)

    for round_idx, t_start in enumerate(t_respawn[:-1]):
      t_end = min(t_respawn[round_idx + 1], len(rewards_arr))
      t_leave_safe = None
      for ti in range(t_start, t_end):
        if list(positions[ti, 1]) not in safe_grass:
          t_leave_safe = ti
          break
      if t_leave_safe is None:
        continue
      t_catch_i_arr = t_catch[t_catch >= t_leave_safe]
      if len(t_catch_i_arr) == 0:
        continue
      t_catch_i = t_catch_i_arr[0]
      time_per_round = t_catch_i - t_leave_safe
      num_apple = len(t_apple[(t_apple >= t_leave_safe) & (t_apple < t_end)])
      num_acorn = len(t_acorn[(t_acorn >= t_leave_safe) & (t_acorn < t_end)])
      prey_move = np.sum(distances[1][t_start:t_end])
      predator_move = np.sum(distances[0][t_start:t_end])
      orientations_arr = np.array([info['ORIENTATION'] for info in results])
      prey_rotate = np.sum(np.abs(orientations_arr[1][t_start:t_end] - orientations_arr[1][t_start - 1:t_end - 1]))
      predator_rotate = np.sum(np.abs(orientations_arr[0][t_start:t_end] - orientations_arr[0][t_start - 1:t_end - 1]))
      time_on = 0
      time_off = 0
      for ti in range(t_start, t_catch_i):
        if list(positions[ti, 1]) in safe_grass:
          time_on += 1
        else:
          time_off += 1
      frac_off = time_off / (time_on + time_off) if (time_on + time_off) > 0 else np.nan

      # Now check for each time step in this round if the prey's position change cause a longer distance with the predator
      if t_leave_safe > 0 and t_catch_i < len(rewards_arr):
        distance_to_predator = np.linalg.norm(
          positions[t_leave_safe - 1:t_catch_i - 1, 0] - positions[t_leave_safe:t_catch_i, 1], axis=1)
        t_moved = [i - t_leave_safe for i in range(t_leave_safe, t_catch_i) if
                   (positions[i, 1] != positions[i - 1, 1]).any()]
        distance_to_predator = distance_to_predator[t_moved]
        t_moving_away = np.sum(
          [distance_to_predator[i] > distance_to_predator[i - 1] for i in range(1, len(distance_to_predator))])
        frac_moving_away = t_moving_away / len(t_moved)
        cumulative_dict[base_pair]['frac_moving_away_per_round'].append(frac_moving_away)
        manhattan_distance = np.sum(np.abs(positions[t_leave_safe:t_catch_i, 0] - positions[t_leave_safe:t_catch_i, 1]),
                                    axis=1)
        frac_time_in_3_steps = np.sum(manhattan_distance < 3) / len(manhattan_distance)
        cumulative_dict[base_pair]['frac_time_in_3_steps'].append(frac_time_in_3_steps)
        frac_time_in_5_steps = np.sum(manhattan_distance < 5) / len(manhattan_distance)
        cumulative_dict[base_pair]['frac_time_in_5_steps'].append(frac_time_in_5_steps)
      else:
        cumulative_dict[base_pair]['frac_moving_away_per_round'].append(np.nan)
        cumulative_dict[base_pair]['frac_time_in_3_steps'].append(np.nan)
        cumulative_dict[base_pair]['frac_time_in_5_steps'].append(np.nan)

      # Compute stuck rates for each agent over the current round.
      pred_stuck, pred_stuck_ind = compute_stuck_rate(pred_pos[t_start:t_catch_i], min_duration=20)
      prey_stuck, prey_stuck_ind = compute_stuck_rate(prey_pos[t_start:t_catch_i], min_duration=20)
      cumulative_dict[base_pair]['pred_stuck_rate'].append(pred_stuck)
      cumulative_dict[base_pair]['prey_stuck_rate'].append(prey_stuck)

      cumulative_dict[base_pair]['round'].append(f"{pkl_file.replace('.pkl', '')}_round{round_idx}")
      cumulative_dict[base_pair]['time_per_round'].append(time_per_round)
      cumulative_dict[base_pair]['num_apple_collected_per_round'].append(num_apple)
      cumulative_dict[base_pair]['num_acorn_collected_per_round'].append(num_acorn)
      cumulative_dict[base_pair]['prey_move_distances_per_round'].append(prey_move)
      cumulative_dict[base_pair]['predator_move_distances_per_round'].append(predator_move)
      cumulative_dict[base_pair]['prey_rotate_per_round'].append(prey_rotate)
      cumulative_dict[base_pair]['predator_rotate_per_round'].append(predator_rotate)
      cumulative_dict[base_pair]['time_on_grass_per_round'].append(time_on)
      cumulative_dict[base_pair]['time_off_grass_per_round'].append(time_off)
      cumulative_dict[base_pair]['frac_off_grass_per_round'].append(frac_off)
      # (Other metrics can be added similarly.)

  # Compute overall means for each summary metric.
  for key in list(cumulative_dict[base_pair].keys()):
    if cumulative_dict[base_pair][key]:
      try:
        cumulative_dict[base_pair][f"mean_{key}"] = [np.nanmean(cumulative_dict[base_pair][key])]
      except:
        pass


  return serial_data_dict, cumulative_dict


def process_and_save(folder_path, out_dir, ignore_existing=False):
  base_name = os.path.basename(folder_path).replace("_episode_pickles", "")
  cumulative_out = os.path.join(out_dir, f"{base_name}_cumulative_results.csv")
  serial_out = os.path.join(out_dir, f"{base_name}_serial_results.pkl")
  if ignore_existing and (os.path.exists(serial_out)) and os.path.exists(cumulative_out):
    print(f"{base_name} analysis results already exist. Skipping.")
    return
  result = process_pair_folder(folder_path)
  if result is None:
    return
  serial_dict, cumulative_dict = result
  # Use the folder's base name (without the "_episode_pickles" suffix) as output base.
  pair_key = list(serial_dict.keys())[0]
  # TODO: uncomment this when cumulative_df is needed.
  # cumulative_df = pd.DataFrame().from_dict(cumulative_dict[pair_key], orient='index').T
  # cumulative_df.to_csv(cumulative_out, index=False)

  # serial_df.to_csv(serial_out, index=False)
  with open(serial_out, 'wb') as f:
    pickle.dump(serial_dict, f)

  print(f"Saved analysis for {base_name}")


def main():
  # Base directory for pair folders.
  base_dir = "/home/mikan/e/Documents/GitHub/social-agents-JAX/results/mix/"
  # Output directory for combined analysis.
  out_dir = os.path.join(base_dir, "analysis_results")
  os.makedirs(out_dir, exist_ok=True)

  pair_folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
                  if os.path.isdir(os.path.join(base_dir, d)) and d.endswith("_episode_pickles")]
  if not pair_folders:
    print("No pair folders found in", base_dir)
    return

  # Process each pair folder in parallel.
  Parallel(n_jobs=90)(delayed(process_and_save)(folder, out_dir) for folder in pair_folders)
  print("All analysis files have been generated.")
  # for folder in pair_folders:
  #   process_and_save(folder, out_dir)

if __name__ == '__main__':
  main()
