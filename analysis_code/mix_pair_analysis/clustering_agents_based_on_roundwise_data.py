#!/usr/bin/env python3
"""
cluster_agents_from_pickle.py

Loads pre-computed combined results from a pickle file and performs
clustering on predator and prey agents based on their behavioral metrics.
It determines an optimal number of clusters using Silhouette scores,
applies K-Means, saves the results, and generates PCA visualizations.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Keep for potential future use or consistency
import time
import warnings

# Scikit-learn imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ----- Helper Functions for Clustering -----

def ensure_numeric_metrics(df, metric_cols):
  """
  Ensures that specified metric columns in the DataFrame contain single numeric values.
  If a column contains lists or arrays, it calculates the mean. Handles potential errors.
  (Useful check even when loading from pickle).

  Args:
      df (pd.DataFrame): The input DataFrame.
      metric_cols (list): A list of column names to process.

  Returns:
      pd.DataFrame: The DataFrame with processed metric columns.
  """
  processed_df = df.copy()
  print("Verifying numeric types for metric columns...")
  for col in metric_cols:
    if col in processed_df.columns:
      # Check if the column dtype is object, suggesting potential lists/mixed types
      if processed_df[col].dtype == 'object':
        # Attempt to apply robust mean calculation only if needed
        try:
          # Check if the first non-null element looks list-like
          first_val = processed_df[col].dropna().iloc[0] if not processed_df[col].dropna().empty else None
          is_list_like = isinstance(first_val, (list, np.ndarray, pd.Series))
          if is_list_like:
            print(f"Column '{col}' contains list-like objects. Calculating mean...")
            def robust_mean(x):
              if isinstance(x, (list, np.ndarray, pd.Series)):
                if len(x) > 0:
                  try:
                    numeric_elements = [float(i) for i in x if i is not None and not (isinstance(i, float) and np.isnan(i))]
                    return np.mean(numeric_elements) if numeric_elements else np.nan
                  except (ValueError, TypeError): return np.nan
                else: return np.nan
              elif pd.isna(x): return np.nan
              else:
                try: return float(x)
                except (ValueError, TypeError): return np.nan
            processed_df[col] = processed_df[col].apply(robust_mean)
          else:
            # If not list-like but still object, try direct numeric conversion
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

        except Exception as e:
          print(f"  Warning: Could not process object column '{col}'. Attempting direct numeric conversion. Error: {e}")
          processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

      # If not object, still ensure it's numeric (handles int -> float etc.)
      elif not pd.api.types.is_numeric_dtype(processed_df[col]):
        print(f"  Column '{col}' is not numeric ({processed_df[col].dtype}). Attempting conversion...")
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

    else:
      print(f"Warning: Metric column '{col}' not found in loaded DataFrame.")
      # processed_df[col] = np.nan # Add NaN column if needed downstream

  print("Numeric verification complete.")
  return processed_df

def prepare_agent_data(df, role, metric_cols):
  """
  Prepares agent-level data for clustering by averaging metrics.

  Args:
      df (pd.DataFrame): The combined results DataFrame (pre-processed).
      role (str): 'pred' or 'prey', indicating which agents to extract.
      metric_cols (list): List of metric column names to use as features.

  Returns:
      pd.DataFrame: DataFrame with unique agents as index and averaged metrics as columns.
                    Returns an empty DataFrame if errors occur.
  """
  if role == 'pred':
    desc_col = 'pred_desc'
  elif role == 'prey':
    desc_col = 'prey_desc'
  else:
    raise ValueError("Role must be 'pred' or 'prey'")

  if desc_col not in df.columns:
    print(f"Error: Description column '{desc_col}' not found in DataFrame.")
    return pd.DataFrame()

  # Select only necessary columns (agent description + metrics)
  available_metrics = [m for m in metric_cols if m in df.columns]
  if not available_metrics:
    print(f"Error: None of the specified metric columns found for role '{role}'.")
    return pd.DataFrame()

  relevant_cols = [desc_col] + available_metrics
  agent_df = df[relevant_cols].copy()

  # Ensure metric columns are numeric after selection
  for col in available_metrics:
    if not pd.api.types.is_numeric_dtype(agent_df[col]):
      agent_df[col] = pd.to_numeric(agent_df[col], errors='coerce')

  # Group by agent description and calculate the mean for each metric
  try:
    # Exclude non-numeric columns explicitly if any crept in (though shouldn't happen)
    numeric_cols_only = agent_df.select_dtypes(include=np.number).columns
    agent_agg = agent_df.groupby(desc_col)[numeric_cols_only].mean()
  except Exception as e:
    print(f"Error during aggregation for role '{role}': {e}")
    return pd.DataFrame()

  return agent_agg

def find_optimal_k(data, max_k=10, random_state=42):
  """
  Calculates inertia (WCSS) and silhouette scores for different values of k.

  Args:
      data (np.ndarray): Scaled data for clustering.
      max_k (int): Maximum number of clusters to test.
      random_state (int): Random state for KMeans reproducibility.

  Returns:
      tuple: (range_k, inertias, silhouette_scores)
  """
  inertias = []
  silhouette_scores = []
  # Ensure max_k is reasonable given the number of samples
  actual_max_k = min(max_k, data.shape[0] - 1)
  if actual_max_k < 2:
    print("Warning: Not enough samples to test multiple k values.")
    return [], [], []
  range_k = range(2, actual_max_k + 1) # Start from 2 clusters for silhouette

  print(f"Testing k from 2 to {actual_max_k}...")
  for k in range_k:
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)
    try:
      score = silhouette_score(data, kmeans.labels_)
      silhouette_scores.append(score)
    except ValueError as e:
      print(f"  Could not calculate silhouette score for k={k}: {e}")
      # This can happen if a cluster ends up with only one member
      silhouette_scores.append(np.nan)


  return range_k, inertias, silhouette_scores

def plot_elbow_silhouette(range_k, inertias, silhouette_scores, role, save_dir):
  """Plots the Elbow method and Silhouette score graphs."""
  if not range_k: # Handle case where no k was tested
    print("Cannot plot Elbow/Silhouette: No k values were tested.")
    return

  fig, ax1 = plt.subplots(figsize=(10, 5))

  # Plot Elbow (Inertia)
  color = 'tab:blue'
  ax1.set_xlabel('Number of clusters (k)')
  ax1.set_ylabel('Inertia (WCSS)', color=color)
  ax1.plot(range_k, inertias, marker='o', linestyle='-', color=color)
  ax1.tick_params(axis='y', labelcolor=color)
  ax1.set_title(f'Elbow Method & Silhouette Score for {role.capitalize()} Clustering')
  ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # Ensure integer ticks for k

  # Plot Silhouette Score on secondary y-axis
  ax2 = ax1.twinx()
  color = 'tab:red'
  ax2.set_ylabel('Silhouette Score', color=color)
  # Filter out NaN scores before plotting
  valid_k = [k for k, s in zip(range_k, silhouette_scores) if not np.isnan(s)]
  valid_scores = [s for s in silhouette_scores if not np.isnan(s)]
  if valid_k:
    ax2.plot(valid_k, valid_scores, marker='x', linestyle='--', color=color)
  else:
    print("Warning: No valid silhouette scores to plot.")
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()
  plt.grid(True)
  save_path = os.path.join(save_dir, f"{role}_clustering_elbow_silhouette.png")
  plt.savefig(save_path, dpi=300)
  print(f"Saved Elbow/Silhouette plot to {save_path}")
  # plt.show() # Optionally display plot immediately
  plt.close(fig) # Close figure to free memory

def perform_kmeans_clustering(data, n_clusters, random_state=42):
  """
  Performs K-Means clustering on the data.

  Args:
      data (np.ndarray): Scaled and imputed data.
      n_clusters (int): The desired number of clusters (k).
      random_state (int): Random state for reproducibility.

  Returns:
      sklearn.cluster.KMeans: The fitted KMeans object.
  """
  if n_clusters <= 0:
    raise ValueError("Number of clusters (n_clusters) must be positive.")
  if n_clusters > data.shape[0]:
    raise ValueError(f"Number of clusters ({n_clusters}) cannot be greater than the number of samples ({data.shape[0]}).")

  kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
  kmeans.fit(data)
  return kmeans

def plot_clusters_pca(data_scaled, labels, role, n_clusters, agent_names, save_dir):
  """
  Visualizes clusters using PCA.

  Args:
      data_scaled (np.ndarray): Scaled data used for clustering.
      labels (np.ndarray): Cluster labels for each data point.
      role (str): 'pred' or 'prey'.
      n_clusters (int): The number of clusters used.
      agent_names (pd.Index): Index containing agent names/descriptions.
      save_dir (str): Directory to save the plot.
  """
  if data_scaled.shape[1] < 2:
    print("Warning: Cannot perform PCA with less than 2 features. Skipping PCA plot.")
    return

  pca = PCA(n_components=2)
  try:
    data_pca = pca.fit_transform(data_scaled)
  except Exception as e:
    print(f"Error during PCA transformation for {role}: {e}")
    return

  plt.figure(figsize=(12, 8))
  scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50) # Increased size
  plt.title(f'PCA Visualization of {role.capitalize()} Clusters (k={n_clusters})')
  plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
  plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')

  # Add legend
  try:
    legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend1)
  except ValueError as e:
    print(f"Warning: Could not create legend for {role} PCA plot: {e}")


  # Optional: Annotate points (use sparingly)
  # num_points_to_label = 20 # Limit number of labels
  # if len(agent_names) <= num_points_to_label:
  #      for i, name in enumerate(agent_names):
  #          plt.annotate(name.split('_')[0], (data_pca[i, 0], data_pca[i, 1]), fontsize=7, alpha=0.6)


  plt.grid(True)
  save_path = os.path.join(save_dir, f"{role}_cluster_pca_visualization.png")
  plt.savefig(save_path, dpi=300)
  print(f"Saved PCA visualization to {save_path}")
  # plt.show() # Optionally display plot immediately
  plt.close() # Close figure

# ----- Main Execution -----

if __name__ == '__main__':
  start_time = time.time()
  warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress some sklearn warnings

  # --- Configuration ---
  combined_df_path = './round_result_figures/combined_cumulative_results.pkl'
  figures_dir = './round_result_figures/clustering_results_10_10'  # Separate subdir for clustering output
  random_state_seed = 42  # For reproducibility

  # Define the metrics to use for clustering (should match those saved in the pickle)
  # Make sure these column names EXIST in your loaded combined_df
  behavioral_metrics = [
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
    'dim',
    # Add/remove metrics as needed based on the columns in your .pkl file
  ]
  # --- End Configuration ---

  print(f"Starting clustering analysis from: {combined_df_path}")
  if not os.path.exists(figures_dir):
    print(f"Creating output directory: {figures_dir}")
    os.makedirs(figures_dir)

  # 1. Load Data
  if not os.path.exists(combined_df_path):
    print(f"Error: Input file not found at {combined_df_path}")
    exit()

  try:
    print("Loading combined DataFrame from pickle...")
    with open(combined_df_path, 'rb') as f:
      combined_df = pickle.load(f)
    print(f"Successfully loaded DataFrame with shape: {combined_df.shape}")
    # print("Columns in loaded DataFrame:", combined_df.columns.tolist()) # Useful for debugging
  except Exception as e:
    print(f"Error loading pickle file: {e}")
    exit()

  # 2. Pre-process: Ensure metrics are numeric
  # Identify columns actually present in the loaded df
  clustering_feature_cols = [m for m in behavioral_metrics if m in combined_df.columns]
  if not clustering_feature_cols:
    print("Error: None of the specified behavioral_metrics columns found in the loaded DataFrame.")
    print("Available columns:", combined_df.columns.tolist())
    exit()
  print(f"Using these {len(clustering_feature_cols)} columns for clustering: {clustering_feature_cols}")

  # Run the check/conversion function
  combined_df = ensure_numeric_metrics(combined_df, clustering_feature_cols)

  # --- Loop through Predator and Prey Clustering ---
  observed_optinal_k = {'pred': 10, 'prey': 10}
  overriding_optimal_k = True
  for role in ['pred', 'prey']:
    print(f"\n{'='*20} Processing {role.capitalize()} Agents {'='*20}")

    # 3. Prepare Agent-Level Data
    agent_metrics = prepare_agent_data(combined_df, role, clustering_feature_cols)

    if agent_metrics.empty:
      print(f"Could not prepare data for {role} agents. Skipping.")
      continue
    if agent_metrics.shape[0] < 2:  # Need at least 2 agents to cluster
      print(f"Only {agent_metrics.shape[0]} unique {role} agent(s) found. Cannot perform clustering.")
      continue

    print(f"Prepared data for {len(agent_metrics)} unique {role} agents.")
    print(f"Features: {list(agent_metrics.columns)}")

    # 4. Handle Missing Data (Imputation)
    print("Handling missing values using mean imputation...")
    # Check for NaNs before imputation
    nan_counts_before = agent_metrics.isna().sum()
    if nan_counts_before.sum() > 0:
      print("NaNs found before imputation:\n", nan_counts_before[nan_counts_before > 0])
    else:
      print("No NaNs found before imputation.")

    imputer = SimpleImputer(strategy='mean')
    try:
      # Fit on data that might have NaNs, transform creates numpy array
      agent_metrics_imputed = imputer.fit_transform(agent_metrics)
      # Convert back to DataFrame, preserving index and columns
      agent_metrics_imputed_df = pd.DataFrame(agent_metrics_imputed,
                                              index=agent_metrics.index,
                                              columns=agent_metrics.columns)
      # Check for NaNs after imputation (should be none if imputer worked)
      if agent_metrics_imputed_df.isna().sum().sum() > 0:
        print("Warning: NaNs still present after imputation! Check input data or consider different imputation.")
        print(agent_metrics_imputed_df.isna().sum())
      else:
        print("Imputation successful.")

    except Exception as e:
      print(f"Error during imputation for {role}: {e}")
      continue  # Skip to next role if imputation fails

    # 5. Scale Features
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    try:
      # Scale the imputed data
      agent_metrics_scaled = scaler.fit_transform(agent_metrics_imputed_df)
      print("Scaling successful.")
    except Exception as e:
      print(f"Error during scaling for {role}: {e}")
      continue

    # 6. Find Optimal K & Plot Elbow/Silhouette
    print(f"Calculating Elbow and Silhouette scores to find optimal k for {role}...")
    max_k_test = min(20, agent_metrics_scaled.shape[0] - 1)  # Test up to 10 clusters or n_samples-1
    range_k, inertias, silhouette_scores = find_optimal_k(agent_metrics_scaled,
                                                          max_k=max_k_test,
                                                          random_state=random_state_seed)

    if range_k:  # Check if any k values were tested
      plot_elbow_silhouette(range_k, inertias, silhouette_scores, role, figures_dir)

      # Automatically determine k based on max silhouette score
      valid_scores = [s for s in silhouette_scores if not np.isnan(s)]
      if valid_scores:
        optimal_k = range_k[np.nanargmax(silhouette_scores)]  # Use nanargmax to handle potential NaNs
        print(f"-> Automatically selected k = {optimal_k} for {role} (based on max Silhouette score).")
        print("  (Review the Elbow/Silhouette plot to confirm this choice)")
      else:
        optimal_k = 3  # Default if no valid scores
        print(f"Warning: No valid Silhouette scores found. Defaulting to k = {optimal_k}.")
    else:
      optimal_k = 0  # Indicate failure to find k
      print("Could not determine optimal k. Skipping K-Means.")

    if overriding_optimal_k:
      optimal_k = observed_optinal_k[role]
      print(f"Overriding the automatically determined k = {optimal_k} for {role}.")

    # 7. Perform K-Means Clustering (if optimal_k was found)
    if optimal_k >= 2:  # Need at least 2 clusters
      print(f"Performing K-Means clustering for {role} with k={optimal_k}...")
      try:
        kmeans_model = perform_kmeans_clustering(agent_metrics_scaled,
                                                 n_clusters=optimal_k,
                                                 random_state=random_state_seed)
        cluster_labels = kmeans_model.labels_
        print("K-Means clustering successful.")

        # 8. Analyze Results & Save
        # Add cluster labels back to the original (non-scaled, pre-imputation) data for inspection
        agent_metrics_with_clusters = agent_metrics.copy()  # Start from original averages
        agent_metrics_with_clusters[f'{role}_cluster'] = cluster_labels

        print(f"\n{role.capitalize()} Agent Cluster Assignments (k={optimal_k}):")
        print(agent_metrics_with_clusters[f'{role}_cluster'].value_counts().sort_index())

        # Display average metrics per cluster
        print(f"\nAverage metrics per {role.capitalize()} cluster:")
        cluster_summary = agent_metrics_with_clusters.groupby(f'{role}_cluster').mean()
        print(cluster_summary.to_string())  # Print full summary

        # Change cluster labels by descending rewards
        reward_col = 'mean_pred_reward' if role == 'pred' else 'mean_prey_reward'
        if reward_col in cluster_summary.columns:
          cluster_avg_rewards = cluster_summary.sort_values(by=reward_col, ascending=False).index
          cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_avg_rewards)}
          agent_metrics_with_clusters[f'{role}_cluster'] = agent_metrics_with_clusters[f'{role}_cluster'].map(cluster_mapping)

          print(f"\n{role.capitalize()} Agent Cluster Assignments after reward re-labeling:")
          print(agent_metrics_with_clusters[f'{role}_cluster'].value_counts().sort_index())

          print(f"\nAverage metrics per {role.capitalize()} cluster after reward re-labeling:")
          cluster_summary = agent_metrics_with_clusters.groupby(f'{role}_cluster').mean()
          print(cluster_summary.to_string())

        # Save the clustered data (original metrics + cluster label)
        clustered_output_path = os.path.join(figures_dir, f"{role}_clustered_metrics.csv")
        agent_metrics_with_clusters.to_csv(clustered_output_path)
        print(f"Saved clustered {role} metrics to {clustered_output_path}")

        # 9. Visualize Clusters (PCA)
        print(f"Generating PCA visualization for {role} clusters...")
        plot_clusters_pca(agent_metrics_scaled, cluster_labels, role,
                          optimal_k, agent_metrics.index, figures_dir)

      except Exception as e:
        print(f"Error during K-Means or subsequent analysis for {role}: {e}")
    else:
      print(f"Skipping K-Means for {role} as optimal k ({optimal_k}) is less than 2.")

  # --- End Loop ---

  print(f"\n{'='*20} Clustering analysis finished {'='*20}")
  print("--- Total time: %s seconds ---" % (time.time() - start_time))
  print(f"Outputs saved in: {figures_dir}")