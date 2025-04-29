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

def plot_combined_heatmap(combined_df, metric, display_metric_name, fmt=".0%",
                          pred_clusters=None, prey_clusters=None, output_dir="."):
  """
  Loads cross and non-cross results, combines them, pivots by predator and prey labels,
  sorts by cluster (if available) and reward, and plots a heatmap for the chosen metric.

  Args:
      combined_df (pd.DataFrame): DataFrame with combined results.
      metric (str): The column name of the metric to plot (should be numeric).
      display_metric_name (str): The metric name to use for titles and labels.
      fmt (str): Formatting string for annotations.
      pred_clusters (dict, optional): Mapping {pred_desc: cluster_label}. Defaults to None.
      prey_clusters (dict, optional): Mapping {prey_desc: cluster_label}. Defaults to None.
      output_dir (str): Directory to save the heatmap image.
  """
  if combined_df.empty:
    print("No results found to plot!")
    return
  if metric not in combined_df.columns:
    print(f"Metric column '{metric}' not found in DataFrame. Skipping plot.")
    return
  if combined_df[metric].isna().all():
    print(f"Metric column '{metric}' contains only NaN values. Skipping plot.")
    return

  # Ensure pred_desc and prey_desc columns exist (should be created by combine_results)
  if 'pred_desc' not in combined_df.columns or 'prey_desc' not in combined_df.columns:
    print("Error: 'pred_desc' or 'prey_desc' columns missing. Cannot create pivot table.")
    # Try to regenerate them if pair column exists
    if 'pair' in combined_df.columns:
      print("Attempting to parse 'pred_desc', 'prey_desc' from 'pair' column...")

      def parse_labels(pair):
        try:
          return pair.split("_vs_")
        except:
          return None, None

      parsed = combined_df["pair"].apply(parse_labels)
      combined_df[['pred_desc', 'prey_desc']] = pd.DataFrame(parsed.tolist(), index=combined_df.index)
      if 'pred_desc' not in combined_df.columns or 'prey_desc' not in combined_df.columns:
        print("Failed to parse descriptions. Aborting plot.")
        return
    else:
      print("Aborting plot.")
      return

  # --- Create Pivot Table ---
  print(f"Creating pivot table for metric: {display_metric_name} (using column: {metric})")
  if 'mean_' not in metric:
    combined_df['mean_' + metric] = combined_df[metric].apply(lambda x: np.mean(x))
    metric = 'mean_' + metric
  try:
    # Use mean aggregation, as one pred/prey pair might appear multiple times if input data had duplicates
    pivot_table = combined_df.pivot_table(index='pred_desc', columns='prey_desc',
                                          values=metric, aggfunc='mean')
  except Exception as e:
    print(f"Error creating pivot table: {e}")
    return

  # Remove rows/columns that are all NaN (can happen if some agents only appeared in pairs not present in the final df)
  pivot_table.dropna(axis=0, how='all', inplace=True)
  pivot_table.dropna(axis=1, how='all', inplace=True)

  if pivot_table.empty:
    print(f"Pivot table for metric '{display_metric_name}' is empty after dropping NaNs. Skipping plot.")
    return

  # --- Compute Reward Metrics for Sorting ---
  # Use the original combined_df before pivoting
  pred_rewards, prey_rewards = compute_reward_metrics(combined_df)

  # --- Sorting Logic (Cluster + Reward) ---
  print("Sorting axes by cluster (if available) and reward...")

  # Helper function for grouped sorting
  def sort_agents(agent_list, cluster_map, reward_map):
    if cluster_map is None:
      # No cluster info, just sort by reward
      print("No cluster information provided, sorting by reward only.")
      return sorted(agent_list, key=lambda x: reward_map.get(x, -np.inf), reverse=True)
    else:
      # Group by cluster, sort clusters by avg reward, sort agents within cluster by individual reward
      agent_data = pd.DataFrame({'agent': agent_list})
      agent_data['cluster'] = agent_data['agent'].map(cluster_map).fillna(-1).astype(
        int)  # Handle agents not in cluster map
      agent_data['reward'] = agent_data['agent'].map(reward_map).fillna(-np.inf)

      # Calculate average reward per cluster
      cluster_avg_reward = agent_data.groupby('cluster')['reward'].mean().sort_values(ascending=False)

      # Sort agents: first by cluster order (based on avg reward), then by individual reward within cluster
      agent_data['cluster_order'] = agent_data['cluster'].map(
        lambda c: cluster_avg_reward.index.get_loc(c) if c in cluster_avg_reward.index else np.inf)
      sorted_agents = agent_data.sort_values(by=['cluster_order', 'reward'], ascending=[True, False])[
        'agent'].tolist()

      # Get cluster boundaries for adding lines later
      cluster_boundaries = []
      if not sorted_agents: return [], []  # Handle empty list
      current_cluster = cluster_map.get(sorted_agents[0], -1)
      for i, agent in enumerate(sorted_agents[1:], 1):
        next_cluster = cluster_map.get(agent, -1)
        if next_cluster != current_cluster:
          cluster_boundaries.append(i)
          current_cluster = next_cluster

      return sorted_agents, cluster_boundaries

  # Sort predator rows (index)
  valid_predators = [p for p in pivot_table.index if
                     p in pred_rewards]  # Filter for predators actually in the pivot table and reward list
  sorted_pred, pred_boundaries = sort_agents(valid_predators, pred_clusters, pred_rewards)

  # Sort prey columns
  valid_prey = [p for p in pivot_table.columns if
                p in prey_rewards]  # Filter for prey actually in the pivot table and reward list
  sorted_prey, prey_boundaries = sort_agents(valid_prey, prey_clusters, prey_rewards)

  # Reindex the pivot table
  try:
    pivot_table = pivot_table.loc[sorted_pred, sorted_prey]
  except KeyError as e:
    print(f"KeyError during reindexing. Some agents might be missing from pivot table or rewards. Error: {e}")
    # Fallback to unsorted pivot table or handle differently? For now, we'll continue with potentially partial data.
    # To debug, check which keys in sorted_pred/sorted_prey are not in pivot_table.index/columns respectively
    missing_preds = [p for p in sorted_pred if p not in pivot_table.index]
    missing_prey = [p for p in sorted_prey if p not in pivot_table.columns]
    if missing_preds: print("Missing predators from pivot index:", missing_preds)
    if missing_prey: print("Missing prey from pivot columns:", missing_prey)
    # Attempt reindex ignoring missing keys (might result in smaller table)
    valid_sorted_pred = [p for p in sorted_pred if p in pivot_table.index]
    valid_sorted_prey = [p for p in sorted_prey if p in pivot_table.columns]
    if not valid_sorted_pred or not valid_sorted_prey:
      print("Cannot reindex, not enough valid agents remain. Skipping plot.")
      return
    pivot_table = pivot_table.loc[valid_sorted_pred, valid_sorted_prey]
    # Adjust boundaries if agents were removed (this part is tricky, might skip boundaries if errors occur)
    print("Warning: Reindexed with potentially fewer agents due to KeyErrors.")
    # Reset boundaries as they might be incorrect now
    # pred_boundaries, prey_boundaries = [], []

  # --- Plotting Heatmap ---
  plt.figure(
    figsize=(max(15, len(pivot_table.columns) * 0.5), max(10, len(pivot_table.index) * 0.4)))  # Dynamic figsize

  # Calculate vmin/vmax based on data percentile to avoid outlier skew
  vmin = np.nanpercentile(pivot_table.values, 5) if not np.isnan(pivot_table.values).all() else None
  vmax = np.nanpercentile(pivot_table.values, 95) if not np.isnan(pivot_table.values).all() else None

  sns.heatmap(pivot_table, annot=True, fmt=fmt, cmap="RdBu_r",  # Or choose another cmap like "viridis"
              linewidths=.5, linecolor='lightgray',  # Add grid lines
              # cbar_kws={'label': display_metric_name},
              cbar=False,
              vmin=vmin, vmax=vmax)  # Use percentile-based limits

  # Add cluster separator lines if boundaries exist
  for boundary in pred_boundaries:
    plt.axhline(boundary, color='black', lw=2.5)
  for boundary in prey_boundaries:
    plt.axvline(boundary, color='black', lw=2.5)

  plt.title(f"Combined Heatmap of {display_metric_name}\n(Sorted by Cluster and Reward)", fontsize=16)
  plt.xlabel("Prey Training Background (Cluster | Reward)", fontsize=12)
  plt.ylabel("Predator Training Background (Cluster | Reward)", fontsize=12)
  plt.xticks(rotation=90, fontsize=8)
  plt.yticks(rotation=0, fontsize=8)
  plt.tight_layout()

  # Save the figure
  save_filename = os.path.join(output_dir, f"clustered_heatmap_{display_metric_name.replace('/', '_')}.png")
  plt.savefig(save_filename, dpi=300, bbox_inches='tight')
  print(f"Saved heatmap to {save_filename}")
  # plt.show() # Optional: display plot immediately
  plt.close()  # Close the figure to free memory

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
  load_old = True # Set to True to load previously saved combined_df

  # Define the metrics for plotting heatmaps
  metrics = [
    'pred_stuck_rate',
    'prey_stuck_rate',
    'mean_num_acorn_collected_per_round',  # Assuming mean was saved
    'mean_num_apple_collected_per_round',  # Assuming mean was saved
    'mean_prey_move_distances_per_round',  # Assuming mean was saved
    'mean_predator_move_distances_per_round',  # Assuming mean was saved
    'mean_time_on_grass_per_round',  # Assuming mean was saved
    'mean_time_off_grass_per_round',  # Assuming mean was saved
    'frac_off_grass_per_round',  # Fractions/Rates likely don't need 'mean_'
    'frac_time_in_3_steps',
    'frac_time_in_5_steps',
    'mean_predator_rotate_per_round',  # Assuming mean was saved
    'mean_prey_rotate_per_round',  # Assuming mean was saved
    'mean_time_per_round',  # Assuming mean was saved
    'mean_pred_reward',
    'mean_prey_reward',
  ]

  # --- Configuration for loading/saving data ---
  combined_df_path = './round_result_figures/combined_cumulative_results.pkl'
  figures_dir = './round_result_figures' # Main dir for figures
  clustering_results_dir = os.path.join(figures_dir, 'clustering_results_10_10') # Subdir where cluster CSVs are
  heatmap_output_dir = os.path.join(figures_dir, 'clustered_heatmaps_10_10') # New subdir for clustered heatmaps

  # --- Create directories if they don't exist ---
  if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
  if not os.path.exists(clustering_results_dir):
    # If clustering dir doesn't exist, clustering likely hasn't run
    print(f"Warning: Clustering results directory not found at {clustering_results_dir}")
    # We can proceed without clustering info, the plot will just be sorted by reward only
  if not os.path.exists(heatmap_output_dir):
    os.makedirs(heatmap_output_dir)


  # --- Load or Generate Combined DataFrame ---
  if os.path.exists(combined_df_path) and load_old:
    print(f"Loading existing combined DataFrame from {combined_df_path}")
    combined_df = pd.read_pickle(combined_df_path)
  else:
    print("Loading and combining cross and non-cross results...")
    cross_dir = os.path.expanduser("~/Documents/GitHub/social-agents-JAX/results/mix/analysis_results")
    # Ensure load_cumulative_results and load_non_cross_results produce numeric means where needed
    # or add ensure_numeric_metrics logic here if necessary
    cross_df = load_cumulative_results(cross_dir)
    non_cross_df = load_non_cross_results()
    combined_df = combine_results(cross_df, non_cross_df)
    # Optional: Ensure numeric metrics right after combining if needed
    # combined_df = ensure_numeric_metrics(combined_df, [m for m in metrics if 'mean_' in m or 'frac' in m or 'rate' in m])

    if not combined_df.empty:
      print(f"Saving combined DataFrame to {combined_df_path}")
      combined_df.to_pickle(combined_df_path)
    else:
      print("Error: Combined DataFrame is empty after loading. Exiting.")
      exit()

  # --- Load Cluster Assignments ---
  pred_cluster_file = os.path.join(clustering_results_dir, 'pred_clustered_metrics.csv')
  prey_cluster_file = os.path.join(clustering_results_dir, 'prey_clustered_metrics.csv')
  pred_clusters = None
  prey_clusters = None

  if os.path.exists(pred_cluster_file):
    print(f"Loading predator clusters from {pred_cluster_file}")
    try:
      pred_cluster_df = pd.read_csv(pred_cluster_file, index_col=0) # Assuming index is agent desc
      # Find the cluster column (handle potential variations in naming)
      cluster_col_pred = next((col for col in pred_cluster_df.columns if 'pred_cluster' in col), None)
      if cluster_col_pred:
        pred_clusters = pred_cluster_df[cluster_col_pred].to_dict()
        print(f"Loaded {len(pred_clusters)} predator cluster assignments.")
      else:
        print("Warning: Could not find 'pred_cluster' column in predator cluster file.")
    except Exception as e:
      print(f"Error loading predator cluster file: {e}")
  else:
    print("Predator cluster file not found. Heatmap rows will only be sorted by reward.")

  if os.path.exists(prey_cluster_file):
    print(f"Loading prey clusters from {prey_cluster_file}")
    try:
      prey_cluster_df = pd.read_csv(prey_cluster_file, index_col=0)
      cluster_col_prey = next((col for col in prey_cluster_df.columns if 'prey_cluster' in col), None)
      if cluster_col_prey:
        prey_clusters = prey_cluster_df[cluster_col_prey].to_dict()
        print(f"Loaded {len(prey_clusters)} prey cluster assignments.")
      else:
        print("Warning: Could not find 'prey_cluster' column in prey cluster file.")
    except Exception as e:
      print(f"Error loading prey cluster file: {e}")
  else:
    print("Prey cluster file not found. Heatmap columns will only be sorted by reward.")


  # --- Plotting Section ---
  print("\nGenerating heatmaps (sorted by cluster and reward)...")

  for metric in metrics:
    # Check if metric actually exists in the dataframe before plotting
    metric_to_plot = metric
    if metric not in combined_df.columns:
      print(f"  Metric '{metric}' not found in DataFrame, skipping heatmap.")
      continue
    # Handle potential list data if not pre-averaged - calculate mean on the fly for pivot
    if combined_df[metric].dtype == 'object':
      print(f"Metric '{metric}' is object type, attempting to calculate mean for plotting...")
      try:
        # Define a temporary column with means for pivoting
        mean_metric_col = f"temp_mean_{metric}"
        combined_df[mean_metric_col] = combined_df[metric].apply(
          lambda x: np.nanmean([float(i) for i in x if i is not None and not np.isnan(float(i))]) if isinstance(x, (list, np.ndarray, pd.Series)) and len(x)>0 else (float(x) if pd.notna(x) else np.nan)
        )
        metric_to_plot = mean_metric_col
      except Exception as e:
        print(f"  Could not calculate mean for '{metric}', skipping heatmap. Error: {e}")
        continue
    elif not pd.api.types.is_numeric_dtype(combined_df[metric]):
      print(f"Metric '{metric}' is not numeric, attempting conversion...")
      combined_df[metric_to_plot] = pd.to_numeric(combined_df[metric_to_plot], errors='coerce')
      if combined_df[metric_to_plot].isna().all():
        print(f"  Could not convert '{metric}' to numeric, skipping heatmap.")
        continue

    if metric_to_plot not in combined_df.columns or combined_df[metric_to_plot].isna().all():
      print(f"Skipping heatmap for '{metric}': Column not found or all NaN.")
      if metric_to_plot.startswith("temp_mean_"): # Clean up temporary column
        combined_df.drop(columns=[metric_to_plot], inplace=True)
      continue


    # Determine formatting based on original metric name
    base_metric_name = metric # Use original name for format check
    if ('frac' in base_metric_name) or ('rate' in base_metric_name):
      fmt = ".0%"
    else:
      fmt = ".0f"

    try:
      plot_combined_heatmap(
        combined_df.copy(), # Pass a copy to avoid side effects
        metric=metric_to_plot, # Use potentially temporary mean column
        display_metric_name=metric, # Use original name for title/labels
        fmt=fmt,
        pred_clusters=pred_clusters, # Pass cluster dict
        prey_clusters=prey_clusters, # Pass cluster dict
        output_dir=heatmap_output_dir # Pass output dir
      )
    except Exception as e:
      print(f"Error plotting heatmap for metric {metric}: {e}")
      # import traceback
      # traceback.print_exc() # Uncomment for detailed traceback
      continue
    finally:
      # Clean up temporary column if it was created
      if metric_to_plot.startswith("temp_mean_"):
        combined_df.drop(columns=[metric_to_plot], inplace=True)


  print("\nHeatmap generation finished.")
  print("--- Total time: %s seconds ---" % (time.time() - start_time))