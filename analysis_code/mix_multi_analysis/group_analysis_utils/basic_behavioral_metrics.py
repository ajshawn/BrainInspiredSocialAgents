#!/usr/bin/env python3
"""

analyze_mixed_results_flexible.py

General-purpose analysis for *N* predators vs *M* preys.

Defines helper functions for per-agent and per-pair metrics, and
`process_pair_folder` to load episodes, compute all metrics, and return
a DataFrame. `main()` scans a base directory of pair folders and
outputs one CSV per pair.
"""
import os
import re
import pickle
import numpy as np
import pandas as pd
import importlib
from joblib import Parallel, delayed
from analysis_code.mix_multi_analysis.group_analysis_utils.helper import parse_agent_roles
from analysis_code.mix_multi_analysis.group_analysis_utils.segmentation import mark_death_periods
# from analysis_code.mix_multi_analysis.group_analysis_utils.gather_and_fence import compute_good_gathering_segmented, compute_successful_fencing_and_helpers_segmented, compute_invalid_interactions_segmented
from analysis_code.mix_multi_analysis.group_analysis_utils.position import ori_position
from analysis_code.mix_multi_analysis.group_analysis_utils.map_utils import get_safe_tiles
def compute_agent_move_distance(positions):
  _, A, _ = positions.shape
  return {i: float(np.sum(np.any(np.diff(positions[:, i], axis=0), axis=1)))
          for i in range(A)}


def compute_num_rotation(orientations):
  _, A = orientations.shape
  return {i: int(np.sum(np.abs(np.diff(orientations[:, i])))) for i in range(A)}


def compute_collect_counts(rewards, predators, preys):
  num_apple = {i: int(np.sum(rewards[:, i] == 1)) for i in preys}
  num_acorn = {i: int(np.sum(rewards[:, i] > 1)) for i in preys}
  num_catch = {i: int(np.sum(rewards[:, i] == 1)) for i in predators}
  return num_apple, num_acorn, num_catch

def compute_stuck_rate_all(positions, min_duration=50, death_labels=None):
  stuck_dict = {}
  for i in range(positions.shape[1]):
    stuck_rate, stuck_t = compute_stuck_rate(positions[:, i], min_duration)
    valid_stuck_t = stuck_t[death_labels[i] == 1]
    stuck_dict[i] = np.sum(valid_stuck_t) / len(valid_stuck_t) if len(valid_stuck_t) > 0 else np.nan
  return stuck_dict

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


def compute_death_counts(stamina, duration=100):
  deaths = {}
  for i in range(stamina.shape[1]):
    labels = mark_death_periods(stamina[:, i], duration)
    deaths[i] = int(np.sum((labels[:-1] == 1) & (labels[1:] == 0)))
  return deaths


def compute_grass_time_per_life(positions, death_labels, preys, safe_grass):
  on_time, off_time, frac_off = {}, {}, {}
  T = positions.shape[0]
  for i in preys:
    segments, alive = [], False
    for t in range(T):
      if death_labels[i][t] == 1 and not alive:
        start, alive = t, True
      if death_labels[i][t] == 0 and alive:
        segments.append((start, t));
        alive = False
    if alive:
      segments.append((start, T))
    on_list, off_list, frac_list = [], [], []
    for s, e in segments:
      on = sum(tuple(positions[t, i]) in safe_grass for t in range(s, e))
      total = e - s
      off = total - on
      on_list.append(on)
      off_list.append(off)
      frac_list.append(off / total if total > 0 else np.nan)
    on_time[i], off_time[i], frac_off[i] = on_list, off_list, frac_list
  return on_time, off_time, frac_off

## TODO: below shall be archived
def get_safe_grass_locations(folder_path: str):
  """
  Dynamically load the Meltingpot substrate config matching folder_path
  and extract all coordinates of safe grass tiles from its ASCII map.
  """
  # Infer substrate module name, e.g. 'predator_prey__open_debug'

  m = re.search(r'(predator_prey__[^_]+)', folder_path)
  substrate = m.group(1) if m else 'predator_prey__open_debug'
  module_name = f"meltingpot.python.configs.substrates.{substrate}"
  try:
    mod = importlib.import_module(module_name)
  except ImportError:
    raise ImportError(f"Could not import substrate config {module_name}")
  # Decide map size: if 'open' substrate, use default; else smaller_13x13
  kwargs = {} if 'open' in substrate else {'smaller_13x13': True}
  cfg = mod.get_config(**kwargs)
  ascii_map = cfg.layout.ascii_map.strip('\n').splitlines()
  char_map = cfg.layout.char_prefab_map

  safe_positions = []
  for y, row in enumerate(ascii_map):
    for x, ch in enumerate(row):
      val = char_map.get(ch)
      # string mapping
      if isinstance(val, str) and val.startswith('safe_grass'):
        safe_positions.append((x, y))
      # dict mapping with list of prefabs
      elif isinstance(val, dict) and 'list' in val and 'safe_grass' in val['list']:
        safe_positions.append((x, y))
  return safe_positions

def compute_pairwise_distance_mean(positions, death_labels):
  _, A, _ = positions.shape
  dist = {}
  for i in range(A):
    for j in range(i + 1, A):
      dists = np.sum(abs(positions[:, i] - positions[:, j]), axis=1)
      dists = dists[(death_labels[i] == 1) & (death_labels[j] == 1)]
      dist[f"dist_{i}to{j}"] = float(np.mean(dists))
  return dist


# --- Core processing ---
def process_pair_folder(folder_path):
  base = os.path.basename(folder_path).split('predator_prey__')[0]
  role, src = parse_agent_roles(base)
  predators = [i for i, r in role.items() if r == 'predator']
  preys = [i for i, r in role.items() if r == 'prey']

  pkl_dir = os.path.join(folder_path, 'episode_pickles')
  files = sorted(f for f in os.listdir(pkl_dir) if f.endswith('.pkl'))
  if not files:
    return None

  all_episode_dfs = []
  for fname in files:
    # load one episode
    with open(os.path.join(pkl_dir, fname), 'rb') as fp:
      episode_data = pickle.load(fp)
    # build arrays just for this episode
    pos = np.array([d['POSITION'] for d in episode_data])
    ori = np.array([d['ORIENTATION'] for d in episode_data])
    act = np.array([d['actions'] for d in episode_data])
    rew = np.array([d['rewards'] for d in episode_data])
    sta = np.array([d['STAMINA'] for d in episode_data])

    # compute death_labels for this episode
    death_lbl = {i: mark_death_periods(sta[:, i]) for i in range(sta.shape[1])}

    # compute your metrics exactly as before, but now per-episode
    move_dist = compute_agent_move_distance(pos)
    rotation = compute_num_rotation(ori)
    apple, acorn, catch = compute_collect_counts(rew, predators, preys)
    death_cnt = compute_death_counts(sta)  # counts per agent
    stuck_rate = compute_stuck_rate_all(pos, death_labels=death_lbl)

    # load safe grass from substrate
    safe = get_safe_tiles(folder_path, map='smaller_13x13')
    on_t, off_t, fo = compute_grass_time_per_life(pos, death_lbl, preys, safe)
    pair_dist = compute_pairwise_distance_mean(pos, death_lbl)
    # good = compute_good_gathering_segmented(pos, predators, preys, death_labels=death_lbl)
    # fence = compute_successful_fencing_and_helpers_segmented(pos, ori, act, rew,
    #                                                predators, preys, death_lbl)
    # invalid = compute_invalid_interactions_segmented(act, rew, pos, ori, sta,
    #                                        predators, preys)

    # assemble one DataFrame row for this episode
    row = {
      'trial_name': base,
      'role': role,
      'source': src,
      'episode': fname.replace('.pkl', ''),
      # 'fencing_events': fence,
      # 'invalid_events': invalid,
      # flatten the per-agent / per-pair scalars:
      **{f"move_{i}": move_dist[i] for i in move_dist},
      **{f"rot_{i}": rotation[i] for i in rotation},
      **{f"apple_{i}": apple[i] for i in apple},
      **{f"acorn_{i}": acorn[i] for i in acorn},
      **{f"catch_{i}": catch[i] for i in catch},
      **{f"death_{i}": death_cnt[i] for i in death_cnt},
      **{f"on_grass_{i}": on_t[i] for i in on_t},
      **{f"off_grass_{i}": off_t[i] for i in off_t},
      **{f"frac_off_{i}": fo[i] for i in fo},
      **pair_dist,
      # **good
    }
    all_episode_dfs.append(pd.DataFrame([row]))

  # now concatenate all episodes for this predator–prey folder
  return pd.concat(all_episode_dfs, ignore_index=True)

def process_and_save(folder_path, out_dir):
  """
  Wrapper to process one folder and save its metrics CSV.
  """
  df = process_pair_folder(folder_path)
  if df is None:
    return
  # Derive base filename from folder
  base = os.path.basename(folder_path).split('predator_prey__')[0]
  out_path = os.path.join(out_dir, f"{base}_metrics.csv")
  df.to_csv(out_path, index=False)
  pkl_path = os.path.join(out_dir, f"{base}_metrics.pkl")
  df.to_pickle(pkl_path)
  print(f"Saved: {os.path.basename(out_path)}")


if __name__ == '__main__':
  base_dir = "../../../results/mix_2_4/"
  out_dir = os.path.join(base_dir, "analysis_results")
  os.makedirs(out_dir, exist_ok=True)

  # Find all valid pair folders
  folders = [os.path.join(base_dir, d)
             for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d))
             and 'episode_pickles' in os.listdir(os.path.join(base_dir, d))
             and 'simplified10x10' not in d]

  # Process in parallel using all available CPUs
  Parallel(n_jobs=50)(
    delayed(process_and_save)(folder, out_dir)
    for folder in folders
  )
  # for folder in folders:
  #   process_and_save(folder, out_dir)
  print("All folders processed.")
