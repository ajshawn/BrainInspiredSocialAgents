#!/usr/bin/env python3
"""
analyze_mixed_pair_results_parallel_episode.py

This script processes the reorganized mixed cross‐rollout result folders in parallel.
Each folder in
  /home/mikan/e/Documents/GitHub/social-agents-JAX/results/mix/analysis_results
that ends with "_episode_pickles" corresponds to a unique predator–prey pair.
For each such folder, the script loads all episode pickle files, aggregates episode-wise
summary metrics (including a new stuck rate metric), and saves a CSV file into a "combined"
subdirectory.
For example, if a folder is named:
  OR20250210_agent3_dim256_vs_AH20250210_agent11_dim256_episode_pickles
the script will output:
  OR20250210_agent3_dim256_vs_AH20250210_agent11_dim256_episode_results.csv
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
  folder_path: Directory containing episode pickle files.
  Returns a dictionary with episode-wise summary metrics.
  """
  print(f"Processing folder: {folder_path}")
  base_pair = os.path.basename(folder_path).replace("_episode_pickles", "")
  episode_data_dict = {base_pair: {key: [] for key in [
    'num_rounds', 'mean_time_per_round',
    'prey_move_distances_per_episode', 'predator_move_distances_per_episode',
    'num_apple_collected_per_episode', 'num_acorn_collected_per_episode',
    'prey_reward_per_episode', 'predator_reward_per_episode',
    'prey_rotate_per_episode', 'predator_rotate_per_episode',
    'time_on_grass_per_episode', 'time_off_grass_per_episode',
    'time_on_grass_nonstuck_per_episode', 'time_off_grass_nonstuck_per_episode',
    'frac_off_grass_per_episode', 'frac_off_grass_nonstuck_per_episode',
    'frac_moving_away_per_episode', 'frac_time_within_3_steps', 'frac_time_within_5_steps',
    'pred_stuck_rate', 'prey_stuck_rate',
    'mean_predator_stamina', 'mean_prey_stamina',
    'mean_predator_stamina_nonstuck', 'mean_prey_stamina_nonstuck'
  ]}}

  pkl_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pkl')])
  if not pkl_files:
    print(f"No pickle files found in {folder_path}")
    return None

  for pkl_file in pkl_files:
    file_path = os.path.join(folder_path, pkl_file)
    try:
      with open(file_path, 'rb') as f:
        results = pickle.load(f)
    except Exception as e:
      print(f"Error loading {file_path}: {e}")
      continue

    positions = np.array([info['POSITION'] for info in results])
    rewards_arr = np.array([info['rewards'] for info in results])
    orientations_arr = np.array([info['ORIENTATION'] for info in results])
    stamina_arr = np.array([info['STAMINA'] for info in results])

    num_agents = 2
    distances = [np.linalg.norm(np.diff(positions[:, i, :], axis=0), axis=1) for i in range(num_agents)]
    distances = [np.concatenate(([0], dist)) for dist in distances]

    safe_grass = [[i, j] for i in [8, 9, 10] for j in [4, 5, 6]]
    t_catch = np.where(rewards_arr[:, 0] == 1)[0]
    t_respawn = np.insert(t_catch + 21, 0, 0)

    round_time_list = []
    prey_move_list = []
    predator_move_list = []
    apple_count_list = []
    acorn_count_list = []
    prey_rotate_list = []
    predator_rotate_list = []
    time_on_list = []
    time_off_list = []
    time_on_nonstuck_list = []
    time_off_nonstuck_list = []
    time_moving_away_list = []
    time_moved_list = []
    time_within_3_steps_list = []
    time_within_5_steps_list = []
    pred_stuck_list = []
    prey_stuck_list = []
    prey_rewards_list = []
    pred_rewards_list = []
    pred_stamina_list = []
    prey_stamina_list = []
    pred_stamina_nonstuck_list = []
    prey_stamina_nonstuck_list = []

    # We intentionally ignore the last episode since it is not complete
    for round_idx, t_start in enumerate(t_respawn[:-1]):
      t_end = min(t_respawn[round_idx + 1], len(rewards_arr))
      t_leave_safe = next((ti for ti in range(t_start, t_end) if list(positions[ti, 1]) not in safe_grass), None)
      if t_leave_safe is None:
        continue
      t_catch_i = next((ti for ti in t_catch if ti >= t_leave_safe), None)
      if t_catch_i is None:
        continue
      time_per_round = t_catch_i - t_leave_safe
      round_time_list.append(time_per_round)

      apple_count_list.append(np.sum((rewards_arr[t_leave_safe:t_end, 1] == 1)))
      acorn_count_list.append(np.sum((rewards_arr[t_leave_safe:t_end, 1] > 1)))
      prey_move_list.append(np.sum(distances[1][t_start:t_end] == 1)) # anything beyond 1 is due to death
      predator_move_list.append(np.sum(distances[0][t_start:t_end] == 1))
      if t_start > 0:
        prey_rotate_list.append(
          np.sum(np.abs(np.diff(orientations_arr[t_start:t_end, 1]))))
        predator_rotate_list.append(
          np.sum(np.abs(np.diff(orientations_arr[t_start:t_end, 0]))))

      pred_stuck, pred_stuck_time_points = compute_stuck_rate(positions[t_start:t_catch_i, 0], min_duration=20)
      prey_stuck, prey_stuck_time_points = compute_stuck_rate(positions[t_start:t_catch_i, 1], min_duration=20)
      pred_stuck_list.append(pred_stuck)
      prey_stuck_list.append(prey_stuck)
      prey_rewards_list.append(np.sum(rewards_arr[t_start:t_catch_i, 1]))
      pred_rewards_list.append(np.sum(rewards_arr[t_start:t_catch_i, 0]))
      pred_stamina_list.extend(stamina_arr[t_start:t_catch_i, 0])
      prey_stamina_list.extend(stamina_arr[t_start:t_catch_i, 1])
      pred_stamina_nonstuck_list.extend(stamina_arr[t_start:t_catch_i, 0][pred_stuck_time_points == 0])
      prey_stamina_nonstuck_list.extend(stamina_arr[t_start:t_catch_i, 1][prey_stuck_time_points == 0])

      time_on = np.sum([1 for ti in range(t_start, t_catch_i) if list(positions[ti, 1]) in safe_grass])
      time_off = time_per_round - time_on
      time_on_nonstuck = np.sum([1 for ti in range(t_start, t_catch_i) if
                                 (list(positions[ti, 1]) in safe_grass) and (0 == prey_stuck_time_points[ti - t_start])])
      time_off_nonstuck =  np.sum([1 for ti in range(t_start, t_catch_i) if
                                    (list(positions[ti, 1]) not in safe_grass) and (0 == pred_stuck_time_points[ti - t_start])])

      time_on_list.append(time_on)
      time_off_list.append(time_off)
      time_on_nonstuck_list.append(time_on_nonstuck)
      time_off_nonstuck_list.append(time_off_nonstuck)


      if t_leave_safe >= 0 and t_catch_i <= t_end:
        if t_leave_safe == 0:
          t_leave_safe = 1
        distance_to_predator = np.linalg.norm(
          positions[t_leave_safe - 1:t_catch_i - 1, 0] - positions[t_leave_safe:t_catch_i, 1], axis=1)
        moved_indices = [i - t_leave_safe for i in range(t_leave_safe, t_catch_i) if
                         (positions[i, 1] != positions[i - 1, 1]).any()]
        distance_to_predator_moved = distance_to_predator[
          [i - t_leave_safe for i in range(t_leave_safe, t_catch_i) if (positions[i, 1] != positions[i - 1, 1]).any()]]
        time_moving_away_list.append(np.sum([1 for i in range(1, len(distance_to_predator_moved)) if
                                             distance_to_predator_moved[i] > distance_to_predator_moved[i - 1]]))
        time_moved_list.append(len(moved_indices))
        manhattan_distance = np.sum(np.abs(positions[t_leave_safe:t_catch_i, 0] - positions[t_leave_safe:t_catch_i, 1]),
                                    axis=1)
        time_within_3_steps_list.append(np.sum(manhattan_distance < 3))
        time_within_5_steps_list.append(np.sum(manhattan_distance < 5))
      else:
        time_moving_away_list.append(np.nan)
        time_moved_list.append(np.nan)
        time_within_3_steps_list.append(np.nan)
        time_within_5_steps_list.append(np.nan)

    episode_data_dict[base_pair]['num_rounds'].append(len(round_time_list))
    episode_data_dict[base_pair]['mean_time_per_round'].append(np.nanmean(round_time_list) if round_time_list else np.nan)
    episode_data_dict[base_pair]['prey_move_distances_per_episode'].append(np.nansum(prey_move_list))
    episode_data_dict[base_pair]['predator_move_distances_per_episode'].append(np.nansum(predator_move_list))
    episode_data_dict[base_pair]['num_apple_collected_per_episode'].append(np.nansum(apple_count_list))
    episode_data_dict[base_pair]['num_acorn_collected_per_episode'].append(np.nansum(acorn_count_list))
    episode_data_dict[base_pair]['prey_rotate_per_episode'].append(np.nansum(prey_rotate_list))
    episode_data_dict[base_pair]['predator_rotate_per_episode'].append(np.nansum(predator_rotate_list))
    episode_data_dict[base_pair]['time_on_grass_per_episode'].append(np.nansum(time_on_list))
    episode_data_dict[base_pair]['time_off_grass_per_episode'].append(np.sum(time_off_list))
    episode_data_dict[base_pair]['time_on_grass_nonstuck_per_episode'].append(np.nansum(time_on_nonstuck_list))
    episode_data_dict[base_pair]['time_off_grass_nonstuck_per_episode'].append(np.nansum(time_off_nonstuck_list))
    episode_data_dict[base_pair]['frac_off_grass_per_episode'].append(
      np.sum(time_off_list) / (np.nansum(time_on_list) + np.nansum(time_off_list)) if (np.nansum(time_on_list) + np.nansum(
        time_off_list)) > 0 else np.nan)
    episode_data_dict[base_pair]['frac_off_grass_nonstuck_per_episode'].append(
      np.nansum(time_off_nonstuck_list) / (np.nansum(time_on_nonstuck_list) + np.nansum(time_off_nonstuck_list)) if (np.sum(
        time_on_nonstuck_list) + np.sum(time_off_nonstuck_list)) > 0 else np.nan)
    episode_data_dict[base_pair]['frac_moving_away_per_episode'].append(
      np.nansum(time_moving_away_list) / np.nansum(time_moved_list) if time_moved_list else np.nan)
    episode_data_dict[base_pair]['frac_time_within_3_steps'].append(
      np.nansum(time_within_3_steps_list) / np.nansum(round_time_list) if round_time_list else np.nan)
    episode_data_dict[base_pair]['frac_time_within_5_steps'].append(
      np.nansum(time_within_5_steps_list) / np.nansum(round_time_list) if round_time_list else np.nan)
    episode_data_dict[base_pair]['pred_stuck_rate'].append(np.nanmean(pred_stuck_list) if pred_stuck_list else np.nan)
    episode_data_dict[base_pair]['prey_stuck_rate'].append(np.nanmean(prey_stuck_list) if prey_stuck_list else np.nan)
    episode_data_dict[base_pair]['prey_reward_per_episode'].append(np.nansum(prey_rewards_list))
    episode_data_dict[base_pair]['predator_reward_per_episode'].append(np.nansum(pred_rewards_list))
    episode_data_dict[base_pair]['mean_predator_stamina'].append(np.nanmean(pred_stamina_list))
    episode_data_dict[base_pair]['mean_prey_stamina'].append(np.nanmean(prey_stamina_list))
    episode_data_dict[base_pair]['mean_predator_stamina_nonstuck'].append(np.nanmean(pred_stamina_nonstuck_list))
    episode_data_dict[base_pair]['mean_prey_stamina_nonstuck'].append(np.nanmean(prey_stamina_nonstuck_list))

  return episode_data_dict


def process_and_save(folder_path, out_dir):
  result = process_pair_folder(folder_path)
  if result is None:
    return
  base_name = os.path.basename(folder_path).replace("_episode_pickles", "")
  episode_out = os.path.join(out_dir, f"{base_name}_episode_results.csv")
  pair_key = list(result.keys())[0]
  episode_df = pd.DataFrame().from_dict(result[pair_key], orient='index').T
  episode_df.to_csv(episode_out, index=False)

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
  Parallel(n_jobs=50)(delayed(process_and_save)(folder, out_dir) for folder in pair_folders)
  print("All analysis files have been generated.")
  # for folder in pair_folders:
  #   process_and_save(folder, out_dir)


if __name__ == '__main__':
  main()
