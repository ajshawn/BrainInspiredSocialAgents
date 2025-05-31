#!/usr/bin/env python3
import os
import re
import logging
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

# ————————————————————————————————————————————————————————————— #
# Assume these functions are imported or defined above:
# parse_agent_roles, mark_death_periods, get_safe_grass_locations,
# compute_agent_move_distance, compute_num_rotation,
# compute_collect_counts, compute_death_counts, compute_stuck_rate_all,
# compute_grass_time_per_life, compute_pairwise_distance_mean
# ————————————————————————————————————————————————————————————— #
from group_analysis_utils.basic_behavioral_metrics import *

def process_folder(folder, out_dir, skip_existing=False):
  base = os.path.basename(folder).split('predator_prey__')[0]
  env = os.path.basename(folder).split('predator_prey__')[1]
  out_pkl = os.path.join(out_dir, base + '_metrics.pkl')

  if skip_existing:
    if os.path.exists(out_pkl):
      logging.info(f"Skipping existing file: {out_pkl}")
      return None
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

  print(f"Processing folder: {folder} with roles: {role}")


  # load safe grass once
  safe = get_safe_tiles(os.path.basename(folder), map="smaller_13x13")

  rows = []
  for pkl_file in files:
    with open(os.path.join(pkl_dir, pkl_file), "rb") as fp:
      episode = pickle.load(fp)

    pos = np.array([d["POSITION"]    for d in episode])
    ori = np.array([d["ORIENTATION"] for d in episode])
    act = np.array([d["actions"]     for d in episode])
    rew = np.array([d["rewards"]     for d in episode])
    sta = np.array([d["STAMINA"]     for d in episode])

    death_lbl = {i: mark_death_periods(sta[:, i]) for i in range(sta.shape[1])}

    # compute per-episode metrics
    move_dist = compute_agent_move_distance(pos)
    rotation = compute_num_rotation(ori)
    apple, acorn, catch = compute_collect_counts(rew, predators, preys)
    death_cnt = compute_death_counts(sta)
    stuck_rate = compute_stuck_rate_all(pos, death_labels=death_lbl)

    on_t, off_t, frac_off = compute_grass_time_per_life(
      pos, death_lbl, preys, safe
    )
    pair_dist = compute_pairwise_distance_mean(pos, death_lbl)

    # assemble one row
    row = {
      "trial_name": base,
      "role": role,
      "source": src,
      "episode": pkl_file,
      **{f"move_{i}": move_dist[i] for i in move_dist},
      **{f"rot_{i}": rotation[i] for i in rotation},
      **{f"apple_{i}": apple[i] for i in apple},
      **{f"acorn_{i}": acorn[i] for i in acorn},
      **{f"catch_{i}": catch[i] for i in catch},
      **{f"death_{i}": death_cnt[i] for i in death_cnt},
      **{f"stuck_{i}": stuck_rate[i] for i in stuck_rate},
      **{f"on_grass_{i}": on_t[i] for i in on_t},
      **{f"off_grass_{i}": off_t[i] for i in off_t},
      **{f"frac_off_{i}": frac_off[i] for i in frac_off},
      **pair_dist,
    }
    rows.append(row)

  df = pd.DataFrame(rows)
  df.to_pickle(out_pkl)


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
