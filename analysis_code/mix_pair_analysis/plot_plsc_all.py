import numpy as np
import pandas as pd
import os
import pickle
import scipy as sp
import matplotlib.pyplot as plt
import glob
from plot_round_all import combine_results, plot_combined_heatmap
from plot_round_all_based_on_clustering import plot_combined_heatmap as plot_combined_heatmap_clustered
import time

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

def load_non_cross_results(select_metrics=None):
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
    file_path = os.path.join(base_path, arena_name, "pickles", "PLSC_results_dict.pkl")
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
      if select_metrics is None:
        row.update(metrics)
      else:
        for key in select_metrics:
          if key in metrics:
            row[key] = metrics[key]
      rows.append(row)
  if rows:
    return pd.DataFrame(rows)
  else:
    return pd.DataFrame()


def load_cross_results(cross_dir, select_metrics=None):
  """
  Load the cross-rollout cumulative results CSV files from the combined folder.
  Each file is expected to have a "pair" column with a title like:
    "AH20250107_agent0_dim128_vs_AH20250210_agent5_dim256"
  Returns a DataFrame.
  """
  csv_files = glob.glob(os.path.join(cross_dir, "*PLSC_results.pkl"))
  dfs = []
  for f in csv_files:
    try:
      dct = pd.read_pickle(f)
      pair_name = os.path.basename(f).replace("_PLSC_results.pkl", "")
      tmp = {"pair": pair_name}
      if select_metrics is not None:
        for key in select_metrics:
          tmp[key] = dct[key]
      dfs.append(tmp)
    except Exception as e:
      print(f"Error reading {f}: {e}")
  if dfs:
    return pd.DataFrame(dfs)
  else:
    return pd.DataFrame()

if __name__ == '__main__':

  start_time = time.time()
  load_old = True
  select_metrics = ['rank', 'cov', 'cor']
  fmts = ['.0f', '.2f', '.2f']
  # If exisitng cross results are not found, we will load the non-cross results.
  if os.path.exists('./plsc_result_figures/combined_plsc_results.pkl') and load_old:
    combined_df = pd.read_pickle('./plsc_result_figures/combined_plsc_results.pkl')
  else:
    if not os.path.exists('./plsc_result_figures'):
      os.makedirs('./plsc_result_figures')
    # Load cross results from the combined folder.
    cross_dir = "/home/mikan/Documents/GitHub/social-agents-JAX/results/mix/analysis_results/"
    cross_results_df = load_cross_results(cross_dir, select_metrics=select_metrics)
    # Load non-cross results.
    non_cross_results_df = load_non_cross_results(select_metrics=select_metrics)
    # Combine non-cross and cross results.
    combined_df = combine_results(cross_results_df, non_cross_results_df)
    # Save the combined results to a CSV file.
    # combined_df.to_csv('./plsc_result_figures/combined_plsc_results.csv', index=False)
    combined_df.to_pickle('./plsc_result_figures/combined_plsc_results.pkl')

  round_df = pd.read_pickle('./round_result_figures/combined_cumulative_results.pkl')
  combined_df = pd.merge(combined_df, round_df, how='left', on='pair')

  # calculate the mean_cov and mean_cor of shared dim (rank)
  mean_cov_shared_dim, mean_cor_shared_dim = [], []
  for rid, row in combined_df.iterrows():
    # Get the rank of the shared dimension.
    rank = row['rank']
    # Get the cov and cor values for the shared dimension.
    cov = row['cov']
    cor = row['cor']
    # Append to the lists.
    mean_cov_shared_dim.append(cov)
    mean_cor_shared_dim.append(cor)
    break

  # Plot the combined results.
  for metric, fmt in zip(select_metrics, fmts):
    # Create a heatmap for the current metric.
    plot_combined_heatmap(combined_df, metric=metric, fmt=fmt)
    # Save the figure.
    plt.savefig(f'./plsc_result_figures/combined_plsc_results_{metric}.png')
    plt.close()
  end_time = time.time()

  ## Now include the cluster
  combined_df_path = './round_result_figures/combined_cumulative_results.pkl'
  figures_dir = './round_result_figures'
  clustering_results_dir = os.path.join(figures_dir, 'clustering_results_10_10') # Subdir where cluster CSVs are
  heatmap_output_dir = os.path.join('./plsc_result_figures', 'clustered_heatmaps_10_10') # New subdir for clustered heatmaps



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


  # Create the heatmap output directory if it doesn't exist.
  if not os.path.exists(heatmap_output_dir):
    os.makedirs(heatmap_output_dir)

  for metric, fmt in zip(select_metrics, fmts):

    try:
      plot_combined_heatmap_clustered(
        combined_df.copy(), # Pass a copy to avoid side effects
        metric=metric, # Use potentially temporary mean column
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


