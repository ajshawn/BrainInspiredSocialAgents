#!/usr/bin/env python3
"""
plot_combined_all_pairs_ordered.py

This script combines cumulative results from both cross‐rollout and non‐cross evaluations,
computes a reward metric for each training background (for predators and prey), sorts them
accordingly, and then plots a single heatmap for a chosen metric (e.g. mean_pred_stuck_rate)
with rows ordered by predator reward and columns ordered by prey reward.

Reward definitions:
  - Prey reward = (mean_apple_collected_per_round) + 6*(mean_acorn_collected_per_round)
  - Predator reward = round_count (i.e. number of rounds)

Assumes that the combined CSV files (from previous processing) are available in a specified folder.
"""

import os
import glob
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
      # If the CSV has multiple rows (e.g., a list was stored), we aggregate into one row.
      # Here we assume that if a column has more than one non-null value, we store the list.
      row_data = {}
      for col in df.columns:
        if df[col].notnull().sum() > 1:
          row_data[col] = df[col].tolist()
        else:
          row_data[col] = df[col].iloc[0]
      tmp_df = pd.DataFrame([row_data])
      tmp_df["pair"] = pair_name
      df_list.append(tmp_df)
    except Exception as e:
      print(f"Error reading file {f}: {e}")
  if df_list:
    return pd.concat(df_list, ignore_index=True)
  else:
    return pd.DataFrame()


def load_non_cross_results():
  """
    For each training arena, load the non-cross cumulative results from:
      /home/mikan/e/Documents/GitHub/social-agents-JAX/results/<arena_name>/pickles/cumulative_results_dict.pkl
    For each key (like "0_10"), construct a new row with a descriptive pair label.
    Returns a DataFrame.
    """
  base_path = "/home/mikan/e/Documents/GitHub/social-agents-JAX/results"
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
    dim = get_dim_from_abbr(mix_abbr)
    for key, metrics in arena_dict.items():
      try:
        pred_idx_str, prey_idx_str = key.split("_")
        pred_idx = int(pred_idx_str)
        prey_idx = int(prey_idx_str)
      except Exception as e:
        print(f"Error parsing key '{key}' in {file_path}: {e}")
        continue
      new_pair = f"{mix_abbr}_agent{pred_idx}_dim{dim}_vs_{mix_abbr}_agent{prey_idx}_dim{dim}"
      row = {"pair": new_pair,
             "arena": mix_abbr,
             "pred_index": pred_idx,
             "prey_index": prey_idx,
             "dim": dim}
      # Assume metrics dict already contains cumulative summary values.
      row.update(metrics)
      rows.append(row)
  if rows:
    return pd.DataFrame(rows)
  else:
    return pd.DataFrame()


def get_dim_from_abbr(abbr):
  try:
    date_str = abbr[2:10]  # e.g. "20250107"
    if int(date_str) < 20250201:
      return "128"
    else:
      return "256"
  except Exception:
    return "NA"


def combine_results(cross_df, non_cross_df):
  if cross_df is None:
    combined = non_cross_df
  elif non_cross_df.empty:
    combined = cross_df
  else:
    combined = pd.concat([cross_df, non_cross_df], ignore_index=True)

  def parse_labels(pair):
    try:
      pred_part, prey_part = pair.split("_vs_")
      return pred_part, prey_part
    except Exception:
      return None, None

  parsed = combined["pair"].apply(parse_labels)
  combined[['pred_desc', 'prey_desc']] = pd.DataFrame(parsed.tolist(), index=combined.index)
  return combined


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
      reward = pd.to_numeric(sub['round_count'], errors='coerce').mean()
    else:
      reward = np.nan
    pred_rewards[pred] = reward
  for prey in df['prey_desc'].unique():
    sub = df[df['prey_desc'] == prey]
    # Compute prey reward = mean_apple + 6*mean_acorn
    try:
      apple = pd.to_numeric(sub['apple_collected_per_round'], errors='coerce').mean()
    except:
      apple = 0
    try:
      acorn = pd.to_numeric(sub['acorn_collected_per_round'], errors='coerce').mean()
    except:
      acorn = 0
    reward = apple + 6 * acorn
    prey_rewards[prey] = reward
  return pred_rewards, prey_rewards


def plot_combined_heatmap(metric="mean_pred_stuck_rate", fmt=".2f"):
  """
    Loads cross and non-cross results, combines them, sorts by total reward,
    pivots by predator and prey descriptive labels, and plots a heatmap for the chosen metric.
    """
  # Load cross results (CSV files)
  cross_dir = os.path.expanduser("~/Documents/GitHub/social-agents-JAX/results/mix/analysis_results/")
  cross_df = load_cumulative_results(cross_dir)
  non_cross_df = load_non_cross_results()
  combined_df = combine_results(cross_df, non_cross_df)
  if combined_df.empty:
    print("No results found!")
    return

  # Sort the combined dataframe by a common key for later pivoting.
  combined_df = combined_df.sort_values(by="pair")

  # Compute reward metrics for ordering.
  pred_rewards, prey_rewards = compute_reward_metrics(combined_df)

  # Create pivot table for the chosen metric.
  pivot_table = combined_df.pivot_table(index='pred_desc', columns='prey_desc', values=metric, aggfunc='mean')

  # Reorder rows and columns based on the reward dictionaries.
  # We'll sort in descending order (higher reward on top/left).
  sorted_pred = sorted(pivot_table.index, key=lambda x: pred_rewards.get(x, -np.inf), reverse=True)
  sorted_prey = sorted(pivot_table.columns, key=lambda x: prey_rewards.get(x, -np.inf), reverse=True)
  pivot_table = pivot_table.loc[sorted_pred, sorted_prey]

  plt.figure(figsize=(18, 12))
  sns.heatmap(pivot_table, annot=True, fmt=fmt, cmap="RdBu_r", cbar_kws={'label': metric})
  plt.title(f"Combined Heatmap of {metric}\nSorted by Reward (Predator & Prey)", fontsize=16)
  plt.xlabel("Prey Training Background", fontsize=14)
  plt.ylabel("Predator Training Background", fontsize=14)
  plt.tight_layout()
  out_fig = os.path.join('./cross_figures', f"combined_heatmap_{metric}_sorted.png")
  plt.savefig(out_fig, dpi=300)
  plt.show()
  print(f"Saved heatmap to {out_fig}")


def main():
  if not os.path.exists('./cross_figures'):
    os.makedirs('./cross_figures')
  # You can choose which metric to plot.
  # For example, we plot mean_pred_stuck_rate (if available).
  plot_combined_heatmap(metric="pred_stuck_rate", fmt=".2f")
  # You can also plot other metrics by calling plot_combined_heatmap with different metric names.


if __name__ == '__main__':
  main()
