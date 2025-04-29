#!/usr/bin/env python3
"""
plot_combined_all_pairs.py

This script combines cumulative results from both cross‐rollout evaluations
(as saved in a “combined” folder) and non‐cross rollout evaluations (loaded
from training arena directories). It then creates a single pivot table of a chosen
metric (e.g. mean_pred_stuck_rate) with predator training backgrounds as rows and
prey training backgrounds as columns, and displays a heatmap.
"""

import os
import glob
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import re
from pandas import pivot, pivot_table

# ----- Mapping for non-cross pairs -----
training_arena_name_list = [
  'PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962',
  'PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026',
  'PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357',
  'PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651',
  'PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441',
  'PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-01-11_11:10:52.115495',
  'PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092',
  'PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274',
]
mix_arena_abbr_list = [
  'AH20250107',
  'AH20250210',
  'OP20241126ckp7357',
  'OP20241126ckp9651',
  'OP20250224',
  'OR20250111',
  'OR20250210',
  'OR20250305',
]


def load_cumulative_results(results_dir):
  """
  Loads all cumulative CSV files from results_dir.
  Assumes each file is named like "<pair>_cumulative_results.csv".
  Returns a combined DataFrame with a new column "pair" extracted from the filename.
  """
  csv_files = glob.glob(os.path.join(results_dir, "*_cumulative_results.csv"))
  df_list = []
  for f in csv_files:
    try:
      df = pd.read_csv(f)
      pair_name = os.path.basename(f).replace("_cumulative_results.csv", "")
      # df["pair"] = pair_name
      # if a df columns contain more than one non-null value, we pass it to tmp_df as a list
      # Create a dictionary for a single row
      row_data = {}
      for col in df.columns:
        if df[col].notnull().sum() > 1:
          # If you want to store the entire list in one cell, keep it as is.
          row_data[col] = df[col].tolist()
        else:
          # If there's only one non-null value, grab that value.
          row_data[col] = df[col].iloc[0]

      # Create a one-row DataFrame using the dictionary
      tmp_df = pd.DataFrame([row_data])
      tmp_df["pair"] = pair_name

      df_list.append(tmp_df)
    except Exception as e:
      print(f"Error reading file {f}: {e}")
  if df_list:
    return pd.concat(df_list, ignore_index=True)
  else:
    return pd.DataFrame()  # Return empty DataFrame if none found


def get_dim_from_abbr(abbr):
  """
  Given an arena abbreviation string (e.g. "AH20250107"), extract the date
  (assumed to be the 8 digits following the two-letter prefix) and return 128
  if the date is before 20250201 and 256 otherwise.
  For abbreviations that include extra tokens (like "ckp7357"), we ignore those
  for date comparison.
  """
  try:
    date_str = abbr[2:10]  # e.g. "20250107"
    # Compare as integer or string (YYYYMMDD)
    if int(date_str) < 20250201:
      return "128"
    else:
      return "256"
  except Exception:
    return "NA"


def load_cross_results(cross_dir):
  """
  Load the cross-rollout cumulative results CSV files from the combined folder.
  Each file is expected to have a "pair" column with a title like:
    "AH20250107_agent0_dim128_vs_AH20250210_agent5_dim256"
  Returns a DataFrame.
  """
  csv_files = glob.glob(os.path.join(cross_dir, "*_cumulative_results.csv"))
  dfs = []
  for f in csv_files:
    try:
      df = pd.read_csv(f)
      pair_name = os.path.basename(f).replace("_cumulative_results.csv", "")
      df["pair"] = pair_name
      dfs.append(df)
    except Exception as e:
      print(f"Error reading {f}: {e}")
  if dfs:
    return pd.concat(dfs, ignore_index=True)
  else:
    return pd.DataFrame()


def load_non_cross_results():
  """
  For each training arena in the mapping, load the non-cross cumulative results.
  These files are stored in:
     /home/mikan/e/Documents/GitHub/social-agents-JAX/results/<arena_name>/pickles/cumulative_results_dict.pkl
  The loaded object is a dict with keys like "0_10" (pred 0 vs prey 10).
  For each such key, build a row with a new pair label.
  Returns a DataFrame with one row per non-cross pair.
  """
  base_path = "/home/mikan/Documents/GitHub/social-agents-JAX/results"
  rows = []
  for arena_name, mix_abbr in zip(training_arena_name_list, mix_arena_abbr_list):
    file_path = os.path.join(base_path, arena_name, "pickles", "cumulative_results_dict.pkl")
    if not os.path.exists(file_path):
      print(f"File not found: {file_path}")
      continue
    try:
      with open(file_path, 'rb') as f:
        arena_dict = pickle.load(f)
    except Exception as e:
      print(f"Error loading {file_path}: {e}")
      continue
    # Determine dimension from mix_abbr.
    dim = get_dim_from_abbr(mix_abbr)
    # For each pair key (e.g., "0_10")
    for key, metrics in arena_dict.items():
      try:
        pred_idx_str, prey_idx_str = key.split("_")
        pred_idx = int(pred_idx_str)
        prey_idx = int(prey_idx_str)
      except Exception as e:
        print(f"Error parsing key '{key}' in {file_path}: {e}")
        continue
      # Construct a new pair label.
      new_pair = f"{mix_abbr}_agent{pred_idx}_dim{dim}_vs_{mix_abbr}_agent{prey_idx}_dim{dim}"
      # Assume metrics dict already contains cumulative summary values.
      # We'll create a row with all key-value pairs and add the new label.
      row = {"pair": new_pair,
             "arena": mix_abbr,
             "pred_index": pred_idx,
             "prey_index": prey_idx,
             "dim": dim}
      # Copy over all keys from metrics (e.g., mean_time_per_round, mean_pred_stuck_rate, etc.)
      row.update(metrics)
      rows.append(row)
  if rows:
    return pd.DataFrame(rows)
  else:
    return pd.DataFrame()


def combine_results(cross_df, non_cross_df):
  """
  Combine the two DataFrames (from cross and non-cross evaluations).
  For plotting, we need to extract predator and prey descriptive labels.
  We assume that in cross_df, the "pair" column is of the form:
     "<pred_arena>_agent<p>_dim<d1>_vs_<prey_arena>_agent<q>_dim<d2>"
  We then extract pred_desc and prey_desc directly.
  For non_cross_df, the "pair" column is already in the form:
     "<mix_abbr>_pred<p>_dim<d>_vs_<mix_abbr>_prey<q>_dim<d>"
  We then return the concatenated DataFrame.
  """
  if cross_df is None:
    combined = non_cross_df
  elif non_cross_df.empty:
    combined = cross_df
  else:
    combined = pd.concat([cross_df, non_cross_df], ignore_index=True)

  # Parse predator and prey labels from the "pair" column.
  def parse_labels(pair):
    try:
      pred_part, prey_part = pair.split("_vs_")
      # For cross pairs, we expect format: "<arena>_agent<p>_dim<d>"
      # We can convert "agent" to "pred" and "prey" accordingly.
      # For simplicity, here we use pred_part and prey_part as-is.
      return pred_part, prey_part
    except Exception:
      return None, None

  parsed = combined["pair"].apply(parse_labels)
  combined[['pred_desc', 'prey_desc']] = pd.DataFrame(parsed.tolist(), index=combined.index)
  return combined


def plot_combined_heatmap(combined_df, metric="pred_stuck_rate", fmt=".0%"):
  """
  Loads cross and non-cross results, combines them, pivots by predator and prey labels,
  and plots a heatmap for the chosen metric.
  """
  # Path to cross results CSV folder (assumed from previous processing)

  if combined_df.empty:
    print("No results found!")
    return

  # For readability, you might want to sort by predator and prey labels.
  # combined_df = combined_df.sort_values(by="pair")
  pred_rewards, prey_rewards = compute_reward_metrics(combined_df)


  # Now, we want to pivot the data to form a matrix where:
  # - Rows: predator descriptive label
  # - Columns: prey descriptive label
  # We need to extract predator and prey from pred_desc and prey_desc.
  def extract_index(label, role="pred"):
    # label examples:
    # "AH20250107_agent0_dim128" or "AH20250107_pred0_dim128"
    # We'll search for "pred" or "agent" and then get the number.
    try:
      if role == "pred":
        for token in label.split("_"):
          if token.startswith("pred"):
            return int(token.replace("pred", ""))
          if token.startswith("agent"):
            return int(token.replace("agent", ""))
      else:
        for token in label.split("_"):
          if token.startswith("prey"):
            return int(token.replace("prey", ""))
          if token.startswith("agent"):
            return int(token.replace("agent", ""))
    except Exception:
      return None

  combined_df["pred_index"] = combined_df["pred_desc"].apply(lambda x: extract_index(x, role="pred"))
  combined_df["prey_index"] = combined_df["prey_desc"].apply(lambda x: extract_index(x, role="prey"))
  if 'mean' not in metric:
    mean_metric = "mean_" + metric
    combined_df[mean_metric] = combined_df[metric].apply(lambda x: np.nanmean(np.array(x)))
    pivot_table = combined_df.pivot_table(index='pred_desc', columns='prey_desc', values=mean_metric, aggfunc='mean')
    pivot_table = pivot_table.sort_index().sort_index(axis=1)
  else:
    # Create a pivot table using the chosen metric.
    pivot_table = combined_df.pivot_table(index='pred_desc', columns='prey_desc', values=metric, aggfunc='mean')
    pivot_table = pivot_table.sort_index().sort_index(axis=1)


  # sorted_pred = sorted(pivot_table.index, key=lambda x: pred_rewards.get(x, -np.inf), reverse=True)
  # sorted_prey = sorted(pivot_table.columns, key=lambda x: prey_rewards.get(x, -np.inf), reverse=True)
  # pivot_table = pivot_table.loc[sorted_pred, sorted_prey]

  plt.figure(figsize=(30, 15))
  medium = pivot_table.mean().mean()
  std = pivot_table.std().mean()
  sns.heatmap(pivot_table, annot=True, fmt=fmt, cmap="RdBu_r", cbar_kws={'label': metric}, cbar=False,
              vmax=medium +std, vmin=medium - std)
  plt.title(f"Combined Heatmap of {metric}\n(All Pairs: Cross and Non-Cross)")
  plt.xlabel("Prey Training Background")
  plt.ylabel("Predator Training Background")
  plt.tight_layout()
  # Save the figure
  plt.savefig(os.path.join('./round_result_figures', f"combined_heatmap_{metric}.png"), dpi=300)
  plt.show()


def compute_reward_metrics(df):
  """
    For each unique predator and prey training background (pred_desc and prey_desc),
    compute a reward metric as follows:
      - Predator reward = round_count (number of rounds, assumed to be available in column 'round_count')
      - Prey reward = (mean_apple_collected_per_round) + 6*(mean_acorn_collected_per_round)
    If the required columns are missing, we default to NaN.
    Returns two dictionaries: pred_rewards and prey_rewards.
    """
  pred_rewards = {}
  prey_rewards = {}
  # For each unique predator background:
  for pred in df['pred_desc'].unique():
    sub = df[df['pred_desc'] == pred]
    # If 'round_count' exists, use its mean as reward.
    if 'round_count' in sub.columns:
      reward = len(np.hstack(sub['round_time']))
    else:
      reward = np.nan
    pred_rewards[pred] = reward
  for prey in df['prey_desc'].unique():
    sub = df[df['prey_desc'] == prey]
    # Compute prey reward = mean_apple + 6*mean_acorn
    try:
      apple = np.nanmean(np.hstack(sub['num_apple_collected_per_round']))
    except:
      apple = 0
    try:
      acorn = np.nanmean(np.hstack(sub['num_acorn_collected_per_round']))
    except:
      acorn = 0
    reward = apple + 6 * acorn
    prey_rewards[prey] = reward
  return pred_rewards, prey_rewards



if __name__ == '__main__':
  start_time = time.time()
  load_old = False

  # If exisitng cross results are not found, we will load the non-cross results.
  if (os.path.exists('./round_result_figures/combined_cumulative_results.pkl')) and load_old:
    combined_df = pd.read_pickle('./round_result_figures/combined_cumulative_results.pkl')
  else:
    cross_dir = os.path.expanduser("~/Documents/GitHub/social-agents-JAX/results/mix/analysis_results")
    cross_df = load_cumulative_results(cross_dir)
    non_cross_df = load_non_cross_results()
    combined_df = combine_results(cross_df, non_cross_df)

    # Compute reward metrics per row
    pred_reward_list, prey_reward_list = [], []
    for idx, row in combined_df.iterrows():
      apple = np.nanmean(row['num_apple_collected_per_round']) if isinstance(row['num_apple_collected_per_round'], list) else row['num_apple_collected_per_round']
      acorn = np.nanmean(row['num_acorn_collected_per_round']) if isinstance(row['num_acorn_collected_per_round'], list) else row['num_acorn_collected_per_round']
      if isinstance(row['time_per_round'], list) and (len(row['time_per_round'])):
        if 'episode_' in row['round'][0]:
          num_episodes = len(np.unique([int(re.search(r'episode_\d+', val).group(0).split('_')[1]) for val in row['round']]))
        else:
          num_episodes = len(
            np.unique([int(re.search(r'\d+_round', val).group(0).split('_')[0]) for val in row['round']]))

        num_catch = len(row['time_per_round'])
        num_catch_per_episode = num_catch / num_episodes if (np.isfinite(num_catch) and np.isfinite(num_episodes)) else np.nan
      elif isinstance(row['time_per_round'], float):
        num_catch_per_episode = 1
      else:
        num_catch_per_episode = 0
      pred_reward_list.append(num_catch_per_episode)
      prey_reward_list.append(apple + 6 * acorn)
    combined_df['mean_pred_reward'] = pred_reward_list
    combined_df['mean_prey_reward'] = prey_reward_list

    if not os.path.exists('./round_result_figures'):
      os.makedirs('./round_result_figures')
    combined_df.to_pickle('./round_result_figures/combined_cumulative_results.pkl', )

  # You can choose which metric to plot.
  metrics = [
    'pred_stuck_rate',
    'prey_stuck_rate',
    'num_acorn_collected_per_round',
    'num_apple_collected_per_round',
    'prey_move_distances_per_round',
    'pred_move_distances_per_round',
    'time_on_grass_per_round',
    'time_off_grass_per_round',
    'frac_off_grass_per_round',
    'frac_time_in_3_steps',
    'frac_time_in_5_steps',
    'predator_rotate_per_round',
    'prey_rotate_per_round',
    'time_per_round',
  ]
  for metric in metrics:
    if ('frac' in metric) or ('rate' in metric):
      fmt = ".0%"
    else:
      fmt = ".0f"
    try:
      plot_combined_heatmap(combined_df, metric, fmt=fmt)
    except Exception as e:
      print(f"Error plotting metric {metric}: {e}")
      continue

  print("--- %s seconds ---" % (time.time() - start_time))