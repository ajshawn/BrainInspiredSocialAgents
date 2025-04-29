#!/usr/bin/env python3
"""
Combined and parallel-processed acorn collection and classification.

This script performs the following steps:
  1. Finds all cross-results pickle files for individual pairs.
  2. In parallel, for each pair:
      - Reads the raw data pkl file.
      - Runs the acorn-event detection (using the 'rewards' array).
      - For each detected event, extracts a behavior trace (from a fixed window around the event)
        and appends onset/offset information.
  3. Combines all the traces into a single list (acorn_collection_bv_list).
  4. Extracts features from each trace to produce a DataFrame (acorn_collection_bv_feature_df).

Large intermediate dictionaries are no longer created.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
from joblib import Parallel, delayed
from math import ceil
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import concurrent.futures
import math

# ---------------------------
# Utility Functions
# ---------------------------

def analyze_and_plot_preys(prey_network_states_dict, min_events=50, n_components=3,
                           figure_path='./serial_clustering_results/'):
  # For each prey with enough datapoints, analyze the neural dynamics.
  for prey, data in prey_network_states_dict.items():
    network_states_list = data['network_states']  # list of arrays; each array shape: (T, hidden_dim)
    clusters = data['cluster_labels']
    categories = data['acorn_event_category']

    if len(network_states_list) < min_events:
      print(f"Skipping prey {prey}: only {len(network_states_list)} events (< {min_events}).")
      continue

    # Option A: Flatten each event to create one vector per event.
    flattened_events = [event.flatten() for event in network_states_list]
    X_flat = np.vstack(flattened_events)

    # Run PCA on the flattened events.
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_flat)

    # For visualization, plot the first two principal components.
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
    plt.title(f"Prey {prey}: PCA of time-averaged neural responses")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Acorn event cluster")
    plt.savefig(f'{figure_path}{prey}_pca_flattened.png')
    plt.show()

    # Option B: Time-resolved PCA (per event trajectory)
    T, hidden_dim = network_states_list[0].shape  # assume consistent T and hidden_dim
    events_array = np.stack(network_states_list)  # shape: (n_events, T, hidden_dim)
    n_events = events_array.shape[0]
    reshaped = events_array.reshape(n_events * T, hidden_dim)
    pca_time = PCA(n_components=2)
    pca_scores = pca_time.fit_transform(reshaped)
    trajectories = pca_scores.reshape(n_events, T, 2)

    plt.figure(figsize=(10, 8))
    for i in range(min(20, n_events)):
      traj = trajectories[i]
      sc = plt.scatter(traj[:, 0], traj[:, 1], c=np.linspace(0, 1, T), cmap='viridis', s=15)
      plt.plot(traj[:, 0], traj[:, 1], alpha=0.5)
    plt.title(f"Prey {prey}: Time-resolved PCA trajectories (first 20 events)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(sc, label="Normalized time")
    plt.savefig(f"{figure_path}{prey}_pca_trajectories.png")
    plt.show()

    df = pd.DataFrame({
      'cluster': clusters,
      'category': categories
    })
    print(f"Prey {prey} event composition by category:")
    print(df['category'].value_counts())
    print(f"Prey {prey} event composition by cluster:")
    print(df['cluster'].value_counts())


def process_pair_network(pair, sub_df, mix_data_directory):
  """
  Process one unique pair:
    - Loads the network states pickle.
    - Iterates through the sub-dataframe rows for this pair.
    - Computes network state slices based on the row and returns two dictionaries:
        one for the predator and one for the prey.
  """
  file_path = os.path.join(mix_data_directory, pair + "_network_states.pkl")
  network_states_df = pd.read_pickle(file_path)
  pred, prey = pair.split('_vs_')
  pred_results = {
    'network_states': [],
    'cluster_labels': [],
    'acorn_event_category': []
  }
  prey_results = {
    'network_states': [],
    'cluster_labels': [],
    'acorn_event_category': []
  }
  for _, row in sub_df.iterrows():
    t_start = row['episode'] * 1000 + row['onset'] - 20
    t_end = row['episode'] * 1000 + row['onset'] + 20  # 20 timesteps before and after
    network_states = network_states_df.iloc[t_start:t_end]
    pred_network_states = network_states.loc[:, network_states.columns.str.contains('hidden_0')].to_numpy()
    prey_network_states = network_states.loc[:, network_states.columns.str.contains('hidden_1')].to_numpy()
    prey_results['network_states'].append(prey_network_states)
    prey_results['cluster_labels'].append(row['cluster'])
    prey_results['acorn_event_category'].append(row['category'])
    pred_results['network_states'].append(pred_network_states)
    pred_results['cluster_labels'].append(row['cluster'])
    pred_results['acorn_event_category'].append(row['category'])
  return (pred, pred_results), (prey, prey_results)


def parallel_processing(bv_list_df, mix_data_directory):
  """
  Parallelizes processing over each unique 'pair' in bv_list_df.
  Aggregates results into two dictionaries for predators and prey.
  """
  pred_network_states_dict = {}
  prey_network_states_dict = {}
  unique_pairs = bv_list_df['pair'].unique()
  with concurrent.futures.ProcessPoolExecutor() as executor:
    future_to_pair = {
      executor.submit(process_pair_network, pair, bv_list_df[bv_list_df['pair'] == pair], mix_data_directory): pair
      for pair in unique_pairs
    }
    for future in concurrent.futures.as_completed(future_to_pair):
      try:
        (pred_key, pred_data), (prey_key, prey_data) = future.result()
        if pred_key not in pred_network_states_dict:
          pred_network_states_dict[pred_key] = pred_data
        else:
          pred_network_states_dict[pred_key]['network_states'].extend(pred_data['network_states'])
          pred_network_states_dict[pred_key]['cluster_labels'].extend(pred_data['cluster_labels'])
          pred_network_states_dict[pred_key]['acorn_event_category'].extend(pred_data['acorn_event_category'])
        if prey_key not in prey_network_states_dict:
          prey_network_states_dict[prey_key] = prey_data
        else:
          prey_network_states_dict[prey_key]['network_states'].extend(prey_data['network_states'])
          prey_network_states_dict[prey_key]['cluster_labels'].extend(prey_data['cluster_labels'])
          prey_network_states_dict[prey_key]['acorn_event_category'].extend(prey_data['acorn_event_category'])
      except Exception as e:
        print(f"Error processing pair {future_to_pair[future]}: {e}")
  return pred_network_states_dict, prey_network_states_dict

def plot_colored_line(ax, pos, cmap='cool'):
  points = pos.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  lc = LineCollection(segments, cmap=plt.get_cmap(cmap), linewidth=2)
  lc.set_array(np.linspace(0, 1, len(pos)))
  ax.add_collection(lc)
  # Mark the start, mid, and end points with different shape
  ax.scatter(pos[0, 0], pos[0, 1], color='blue', s=100, label='Start')
  ax.scatter(pos[len(pos) // 2, 0], pos[len(pos) // 2, 1], color='green', s=100, label='Mid')
  ax.scatter(pos[-1, 0], pos[-1, 1], color='red', s=100, label='End')

def plot_mean_routes_with_temporal_color_and_probability_arrows(
    acorn_collection_list, cluster_labels, bg_img_path,
    save_csv_name="dominant_routes.csv", n_clusters=15, n_cols=5, figure_path='./serial_clustering_results/',
):
  import matplotlib.pyplot as plt
  from matplotlib.collections import LineCollection
  from matplotlib.patches import Patch
  from math import ceil
  from collections import Counter
  import numpy as np
  import pandas as pd

  bg_img = plt.imread(bg_img_path)
  grouped = {i: [] for i in range(n_clusters)}
  for trace, label in zip(acorn_collection_list, cluster_labels):
    grouped[label].append(trace)

  cluster_patterns = {}
  for i in range(n_clusters):
    if len(grouped[i]) > 0:
      cat_list = [trace.get("category", "unknown") for trace in grouped[i]]
      cnt = Counter(cat_list)
      total = len(cat_list)
      major_patterns = []
      for cat, freq in cnt.items():
        prop = freq / total
        if prop > 0.2:
          major_patterns.append(f"{cat}: {prop:.2f}")
      cluster_patterns[i] = ", ".join(major_patterns) if major_patterns else ""
    else:
      cluster_patterns[i] = ""

  cluster_routes = []
  for i in range(n_clusters):
    if len(grouped[i]) == 0:
      continue
    prey_positions = [np.array(trace['prey_position']) for trace in grouped[i]]
    predator_positions = [np.array(trace['predator_position']) for trace in grouped[i]]
    prey_stack = np.stack(prey_positions)
    predator_stack = np.stack(predator_positions)
    mean_prey = np.mean(prey_stack, axis=0) * 60
    mean_predator = np.mean(predator_stack, axis=0) * 60
    prey_distance = np.sum(np.linalg.norm(np.diff(mean_prey, axis=0), axis=1))
    predator_distance = np.sum(np.linalg.norm(np.diff(mean_predator, axis=0), axis=1))
    overall_distance = prey_distance + predator_distance
    prey_stack_scaled = prey_stack * 60
    predator_stack_scaled = predator_stack * 60
    prey_directions = np.diff(prey_stack_scaled, axis=1)
    prey_magnitudes = np.linalg.norm(prey_directions, axis=2)
    prey_prob = np.mean(prey_magnitudes, axis=0)
    predator_directions = np.diff(predator_stack_scaled, axis=1)
    predator_magnitudes = np.linalg.norm(predator_directions, axis=2)
    predator_prob = np.mean(predator_magnitudes, axis=0)

    cluster_routes.append({
      "cluster": i,
      "mean_prey": mean_prey,
      "mean_predator": mean_predator,
      "prey_prob": prey_prob,
      "predator_prob": predator_prob,
      "overall_distance": overall_distance,
      "cluster_size": len(grouped[i])
    })

  cluster_routes.sort(key=lambda x: x['overall_distance'])
  n_rows = ceil(len(cluster_routes) / n_cols)
  fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 5), sharex=True, sharey=True)
  axs = np.array(axs).reshape(-1)

  for idx, route in enumerate(cluster_routes):
    ax = axs[idx]
    ax.imshow(bg_img, extent=[0, 720, 0, 720], aspect='auto', origin='lower')
    prey = route['mean_prey']
    predator = route['mean_predator']
    prey_prob = route['prey_prob']
    predator_prob = route['predator_prob']
    plot_colored_line(ax, prey, cmap='PuBuGn')
    plot_colored_line(ax, predator, cmap='YlOrRd')
    for i in range(len(prey) - 1):
      ax.arrow(prey[i, 0], prey[i, 1],
               prey[i + 1, 0] - prey[i, 0],
               prey[i + 1, 1] - prey[i, 1],
               head_width=5, alpha=0.6, color='blue',
               length_includes_head=True,
               width=0.01 * prey_prob[i])
      ax.arrow(predator[i, 0], predator[i, 1],
               predator[i + 1, 0] - predator[i, 0],
               predator[i + 1, 1] - predator[i, 1],
               head_width=5, alpha=0.6, color='red',
               length_includes_head=True,
               width=0.01 * predator_prob[i])
    base_title = (f"Cluster {route['cluster']} (n = {route['cluster_size']}), "
                  f"distance: {route['overall_distance']:.2f}")
    major_pattern = cluster_patterns.get(route['cluster'], "")
    full_title = base_title + ("\nMajor pattern: " + major_pattern if major_pattern else "")
    ax.set_title(full_title)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 720)
    ax.set_ylim(0, 720)

  for i in range(len(cluster_routes), len(axs)):
    axs[i].axis('off')

  legend_elements = [
    Patch(facecolor='red', edgecolor='red', label='Predator'),
    Patch(facecolor='blue', edgecolor='blue', label='Prey')
  ]
  fig.legend(handles=legend_elements, loc='upper right')
  plt.tight_layout()
  plt.savefig(f"{figure_path}mean_routes_by_cluster_with_bg.png")
  plt.show()

  pair_cluster_df = []
  for i, trace in enumerate(acorn_collection_list):
    pair_id = trace.get('pair', 'unknown')
    cluster = cluster_labels[i]
    pair_cluster_df.append((pair_id, cluster))
  df = pd.DataFrame(pair_cluster_df, columns=['pair', 'cluster'])
  dominant_routes = df.groupby(['pair', 'cluster']).size().reset_index(name='count')
  dominant_routes['prob'] = dominant_routes.groupby('pair')['count'].transform(lambda x: x / x.sum())
  dominant_summary = []
  for pair, group in dominant_routes.groupby('pair'):
    sorted_group = group.sort_values(by='prob', ascending=False)
    cumulative = 0
    for _, row in sorted_group.iterrows():
      if cumulative >= 0.8:
        break
      dominant_summary.append(row)
      cumulative += row['prob']
  result_df = pd.DataFrame(dominant_summary)
  result_df.to_csv(os.path.join(save_csv_name), index=False)
  return result_df, cluster_routes


def detect_acorn_events(rewards, pair):
  """
  Detect acorn events in a rewards array with shape (episodes, timesteps, agents).
  Returns a DataFrame with one row per detected event plus additional columns.
  """
  rewards = np.array(rewards)
  episodes, timesteps, agents = rewards.shape
  assert agents == 2, "Expected 2 agents: predator and prey"
  results = []
  for ep in range(episodes):
    prey_rewards = rewards[ep, :, 1]
    predator_rewards = rewards[ep, :, 0]
    acorn_times = np.where(prey_rewards == 6)[0]
    used_times = set()
    lives = []
    death_times = np.where(predator_rewards == 1)[0]
    life_start = 0
    for dt in death_times:
      lives.append((life_start, dt))
      life_start = dt + 20
    if life_start < timesteps:
      lives.append((life_start, timesteps - 1))
    life_index = 0
    for (life_start, life_end) in lives:
      life_acorn_times = [t for t in acorn_times if life_start <= t <= life_end]
      i = 0
      while i < len(life_acorn_times):
        t = life_acorn_times[i]
        if t in used_times:
          i += 1
          continue
        t2, t3 = t + 5, t + 10
        if t2 in life_acorn_times and t3 in life_acorn_times:
          if t2 not in used_times and t3 not in used_times:
            results.append({
              "category": "3_bite",
              "onset": t,
              "offset": t3,
              "killed": False,
              "episode": ep,
              "life": life_index
            })
            used_times.update([t, t2, t3])
            i += 1
            continue
        if t + 5 in life_acorn_times and (t + 5) not in used_times:
          predator_window = predator_rewards[t + 6: t + 11]
          if 1 in predator_window:
            results.append({
              "category": "2_bite_killed",
              "onset": t,
              "offset": t + 5,
              "killed": True,
              "episode": ep,
              "life": life_index
            })
            used_times.update([t, t + 5])
            i += 1
            continue
        predator_window = predator_rewards[t + 1: t + 6]
        if 1 in predator_window:
          results.append({
            "category": "1_bite_killed",
            "onset": t,
            "offset": t,
            "killed": True,
            "episode": ep,
            "life": life_index
          })
          used_times.add(t)
          i += 1
          continue
        results.append({
          "category": "unclassified",
          "onset": t,
          "offset": t,
          "killed": False,
          "episode": ep,
          "life": life_index
        })
        used_times.add(t)
        i += 1
      life_index += 1
  if len(results) == 0:
    return pd.DataFrame()
  else:
    df = pd.DataFrame(results)
    df['pair'] = [pair] * len(results)
  return df


def extract_features_from_trace(trace_list):
  """
  Given a list of trace dictionaries, extract a set of features from each trace.
  Returns a DataFrame with one row per trace.
  """
  feature_rows = []

  def total_distance(positions):
    return np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

  def action_entropy(actions):
    values, counts = np.unique(actions, return_counts=True)
    return entropy(counts, base=2)

  for trace in trace_list:
    prey_stamina = trace['prey_stamina']
    prey_position = trace['prey_position']
    prey_action = trace['prey_action']
    prey_stuck = trace['prey_stuck']
    predator_stamina = trace['predator_stamina']
    predator_position = trace['predator_position']
    predator_action = trace['predator_action']
    predator_stuck = trace['predator_stuck']
    caught = 0
    if np.any(trace['predator_reward']):
      t_death = np.where(trace['predator_reward'] == 1)[0][0]
      prey_stamina = prey_stamina[:t_death]
      prey_position = prey_position[:t_death]
      prey_action = prey_action[:t_death]
      prey_stuck = prey_stuck[:t_death]
      predator_stamina = predator_stamina[:t_death]
      predator_position = predator_position[:t_death]
      predator_action = predator_action[:t_death]
      predator_stuck = predator_stuck[:t_death]
      caught = 1

    inter_distance = np.linalg.norm(prey_position - predator_position, axis=1)
    feature_rows.append({
      "prey_mean_stamina": np.mean(prey_stamina),
      "prey_min_stamina": np.min(prey_stamina),
      "prey_max_stamina": np.max(prey_stamina),
      "prey_std_stamina": np.std(prey_stamina),
      "prey_stamina_delta": prey_stamina[-1] - prey_stamina[0],
      "prey_total_distance": total_distance(prey_position),
      "prey_stuck_ratio": np.mean(prey_stuck),
      "prey_action_entropy": action_entropy(prey_action),
      "predator_mean_stamina": np.mean(predator_stamina),
      "predator_min_stamina": np.min(predator_stamina),
      "predator_max_stamina": np.max(predator_stamina),
      "predator_std_stamina": np.std(predator_stamina),
      "predator_stamina_delta": predator_stamina[-1] - predator_stamina[0],
      "predator_total_distance": total_distance(predator_position),
      "predator_stuck_ratio": np.mean(predator_stuck),
      "predator_action_entropy": action_entropy(predator_action),
      "avg_inter_distance": np.mean(inter_distance),
      "min_inter_distance": np.min(inter_distance),
      "max_inter_distance": np.max(inter_distance),
      "std_inter_distance": np.std(inter_distance),
      "caught": caught,
    })
  return pd.DataFrame(feature_rows)


# ---------------------------
# Improved Analysis Functions
# ---------------------------
def plot_avg_trajectories_by_cluster(network_states_dict, species_label, n_components=2,
                                     figure_path='./serial_clustering_results/'):
  """
  For each species key in the network states dictionary (e.g., for prey),
  group the events by cluster, compute the averaged network state trajectory for each cluster,
  apply PCA (to reduce hidden state dimensions to 2D), and plot the averaged trajectories.
  """
  for key, data in network_states_dict.items():
    states = data['network_states']
    clusters = np.array(data['cluster_labels'])
    if len(states) == 0:
      continue
    events_array = np.stack(states)  # shape: (n_events, T, hidden_dim)
    unique_clusters = np.unique(clusters)
    # Fit PCA on all states in this key (flattening across time)
    reshaped_all = events_array.reshape(-1, events_array.shape[-1])
    pca = PCA(n_components=n_components)
    pca.fit(reshaped_all)
    plt.figure(figsize=(10, 8))
    for clust in unique_clusters:
      cluster_events = events_array[clusters == clust]
      if len(cluster_events) == 0:
        continue
      avg_traj = np.mean(cluster_events, axis=0)  # shape: (T, hidden_dim)
      avg_traj_pca = pca.transform(avg_traj)
      plt.plot(avg_traj_pca[:, 0], avg_traj_pca[:, 1], label=f"Cluster {clust}")
    plt.title(f"Averaged Trajectories by Cluster for {species_label}: {key}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.savefig(f"{figure_path}{key}_{species_label}_avg_cluster_trajectories.png")
    plt.show()

def plot_agent_clusters_by_key(network_states_dict, agent_type, n_components=2, figure_path='./serial_clustering_results/'):
  """
  For each unique agent (key) in the network states dictionary (e.g., predator or prey),
  group events by cluster, compute the averaged network state trajectory for each cluster,
  reduce the trajectory to 2D using PCA, and plot each cluster in its own subplot within one figure.

  Parameters:
    network_states_dict : dict
        Dictionary keyed by agent identity containing:
          - 'network_states': list of event arrays (shape: (T, hidden_dim))
          - 'cluster_labels': list/array of cluster labels for each event.
    agent_type : str
        A label (e.g., "Predator" or "Prey") used in the figure title and filename.
    n_components : int, optional
        Number of PCA components to compute (default is 2).
  """

  # For each agent (unique key) in the dictionary:
  for agent_key, data in network_states_dict.items():
    states = data['network_states']
    clusters = np.array(data['cluster_labels'])
    if len(states) == 0:
      continue
    # Stack events: shape becomes (n_events, T, hidden_dim)
    events_array = np.stack(states)

    # Identify unique clusters for this agent key.
    unique_clusters = np.unique(clusters)
    num_clusters = len(unique_clusters)
    print(f"{agent_type} {agent_key}: found {num_clusters} clusters.")

    # Fit PCA on all events for this agent key.
    reshaped_all = events_array.reshape(-1, events_array.shape[-1])
    pca = PCA(n_components=n_components)
    pca.fit(reshaped_all)

    # Dynamically set grid: e.g., try to form a nearly square grid.
    n_cols = int(math.ceil(math.sqrt(num_clusters)))
    n_rows = int(math.ceil(num_clusters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False,
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    # Loop over unique clusters to compute and plot the averaged trajectory.
    for i, clust in enumerate(unique_clusters):
      cluster_events = events_array[clusters == clust]
      if cluster_events.shape[0] == 0:
        continue
      # Compute the mean trajectory over events belonging to the same cluster.
      avg_traj = np.mean(cluster_events, axis=0)  # shape: (T, hidden_dim)
      # Apply PCA on the averaged trajectory.
      avg_traj_pca = pca.transform(avg_traj)
      ax = axes_flat[i]
      # ax.plot(avg_traj_pca[:, 0], avg_traj_pca[:, 1], marker='o')
      plot_colored_line(ax, avg_traj_pca, cmap='YlOrRd')
      ax.set_title(f"Cluster {clust}\nn = {np.sum(clusters == clust)}")
      ax.set_xlabel("PC1")
      ax.set_ylabel("PC2")

    # Turn off any unused subplots.
    for j in range(num_clusters, len(axes_flat)):
      axes_flat[j].axis('off')

    # Set a suptitle and adjust layout.
    fig.suptitle(f"{agent_type} {agent_key} - Averaged Trajectories by Cluster", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_filename = f"{figure_path}{agent_key}_{agent_type}_clusters.png"
    plt.savefig(save_filename)
    print(f"Saved {save_filename}")
    plt.show()


# =============================================================================
# In your main processing code, after generating/loading the network states dictionaries,
# simply call the function for each agent type. For example:
# =============================================================================



def compare_predator_and_prey(pred_network_states_dict, prey_network_states_dict, n_components=2):
  """
  Computes an averaged trajectory for each species in both predator and prey dictionaries,
  projects them to 2D using a common PCA, and plots them together for comparison.
  Predator trajectories are plotted with dashed lines and prey with solid lines.
  """
  predator_means = {}
  for key, data in pred_network_states_dict.items():
    if len(data['network_states']) == 0:
      continue
    arr = np.stack(data['network_states'])  # shape: (n_events, T, hidden_dim)
    mean_traj = np.mean(arr, axis=0)
    predator_means[key] = mean_traj
  prey_means = {}
  for key, data in prey_network_states_dict.items():
    if len(data['network_states']) == 0:
      continue
    arr = np.stack(data['network_states'])
    mean_traj = np.mean(arr, axis=0)
    prey_means[key] = mean_traj
  # Combine all trajectories to fit PCA
  all_trajs = list(predator_means.values()) + list(prey_means.values())
  all_data = np.concatenate(all_trajs, axis=0)
  pca = PCA(n_components=n_components)
  pca.fit(all_data)
  plt.figure(figsize=(10, 8))
  for key, traj in predator_means.items():
    traj_pca = pca.transform(traj)
    plt.plot(traj_pca[:, 0], traj_pca[:, 1], linestyle='--', label=f"Predator {key}")
  for key, traj in prey_means.items():
    traj_pca = pca.transform(traj)
    plt.plot(traj_pca[:, 0], traj_pca[:, 1], linestyle='-', label=f"Prey {key}")
  plt.title("Comparison of Averaged Trajectories: Predator vs Prey")
  plt.xlabel("PC1")
  plt.ylabel("PC2")
  plt.legend()
  plt.savefig("comparison_predator_vs_prey.png")
  plt.show()


# ---------------------------
# Parallel Processing Function
# ---------------------------

def process_pair(file_path, window_size=20):
  """
  Given a file path for a cross-results pickle file corresponding to a single pair:
    - Loads the raw data from the file.
    - Runs the acorn event detector.
    - For each detected event, extracts a fixed window of the behavior trace
      (window_size timesteps before and after the event's onset).
    - Returns a list of trace dictionaries (with added onset/offset information).
  """
  pair_name = os.path.basename(file_path).replace("_serial_results.pkl", "")
  if pair_name.count('dim128') >= 2:
    return []
  try:
    with open(file_path, 'rb') as f:
      pair_data = pickle.load(f)
      pair_data = pair_data[pair_name]
  except Exception as e:
    print(f"Error loading file {file_path}: {e}")
    return []
  events_df = detect_acorn_events(pair_data['rewards'], pair_name)
  if events_df.empty:
    return []
  traces = []
  for idx, row in events_df.iterrows():
    episode = row['episode']
    event_onset = row['onset']
    on = event_onset - window_size
    off = event_onset + window_size
    if on < 0 or off > len(pair_data['STAMINA'][episode]):
      print(f"Skipping episode {episode} for pair {pair_name} due to invalid window range.")
      continue
    stamina_trace = np.array(pair_data['STAMINA'][episode][on:off])
    position_trace = np.array(pair_data['POSITION'][episode][on:off])
    action_trace = np.array(pair_data['actions'][episode][on:off])
    stuck_trace = np.array(pair_data['stuck_indicator'][episode][on:off])
    reward_trace = np.array(pair_data['rewards'][episode][on:off])
    if len(stuck_trace) == 0:
      stuck_trace = np.zeros_like(stamina_trace)
    predator_stamina_trace = stamina_trace[:, 0]
    predator_position_trace = position_trace[:, 0]
    predator_action_trace = action_trace[:, 0]
    predator_stuck_trace = stuck_trace[:, 0]
    predator_reward_trace = reward_trace[:, 0]
    prey_stamina_trace = stamina_trace[:, 1]
    prey_position_trace = position_trace[:, 1]
    prey_action_trace = action_trace[:, 1]
    prey_stuck_trace = stuck_trace[:, 1]
    prey_reward_trace = reward_trace[:, 1]
    trace_dict = {
      'predator_stamina': predator_stamina_trace,
      'predator_position': predator_position_trace,
      'predator_action': predator_action_trace,
      'predator_stuck': predator_stuck_trace,
      'prey_stamina': prey_stamina_trace,
      'prey_position': prey_position_trace,
      'prey_action': prey_action_trace,
      'prey_stuck': prey_stuck_trace,
      'category': row['category'],
      'pair': pair_name,
      'episode': episode,
      'onset': on,
      'offset': off,
      'predator_reward': predator_reward_trace,
      'prey_reward': prey_reward_trace,
    }
    traces.append(trace_dict)
  return traces


# ---------------------------
# Main Processing Function
# ---------------------------

def main():
  # Define the directory where cross-results pickle files are stored.
  serial_results_dir = os.path.expanduser(
    "~/Documents/GitHub/social-agents-JAX/results/mix/analysis_results/"
  )
  file_list = glob.glob(os.path.join(serial_results_dir, "*_serial_results.pkl"))
  print(f"Found {len(file_list)} pair files to process.")
  all_traces_nested = Parallel(n_jobs=50)(
    delayed(process_pair)(file_path) for file_path in file_list
  )
  acorn_collection_bv_list = [trace for traces in all_traces_nested for trace in traces]
  print("Total extracted traces:", len(acorn_collection_bv_list))
  acorn_collection_bv_feature_df = extract_features_from_trace(acorn_collection_bv_list)
  print("acorn_collection_bv_feature_df shape:", acorn_collection_bv_feature_df.shape)
  return acorn_collection_bv_list, acorn_collection_bv_feature_df


if __name__ == '__main__':
  reprocessing = False
  base_directory = './serial_clustering_results/'
  mix_data_directory = os.path.expanduser('~/Documents/GitHub/social-agents-JAX/results/mix/analysis_results/')
  if reprocessing:
    bv_list, bv_feature_df = main()
    if not os.path.exists(base_directory):
      os.makedirs(base_directory)
    with open(os.path.join(base_directory, 'acorn_collection_bv_list.pkl'), 'wb') as f:
      pickle.dump(bv_list, f)
    bv_feature_df.to_pickle(os.path.join(base_directory, 'acorn_collection_bv_feature_df.pkl'))
    bv_feature_df.to_csv(os.path.join(base_directory, 'acorn_collection_bv_feature_df.csv'), index=False)
    print("Saved acorn_collection_bv_list and acorn_collection_bv_feature_df to disk.")
  else:
    with open(os.path.join(base_directory, 'acorn_collection_bv_list.pkl'), 'rb') as f:
      bv_list = pickle.load(f)
    bv_feature_df = pd.read_pickle(os.path.join(base_directory, 'acorn_collection_bv_feature_df.pkl'))

  scaler = StandardScaler()
  norm_bv_feature_df = scaler.fit_transform(bv_feature_df)
  n_clusters = 30
  kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
  cluster_labels = kmeans.fit_predict(norm_bv_feature_df)
  for trace, label in zip(bv_list, cluster_labels):
    trace['cluster'] = label
  cluster_labels = kmeans.labels_
  dominant_df, cluster_route = plot_mean_routes_with_temporal_color_and_probability_arrows(
    bv_list, cluster_labels, n_clusters=n_clusters,
    bg_img_path=f'./{base_directory}/background_template.png',
    save_csv_name=os.path.join(base_directory, 'dominant_routes.csv'),
  )
  cluster_labels_reordered = np.zeros_like(cluster_labels)
  for i, route in enumerate(cluster_route):
    cluster_labels_reordered[cluster_labels == route['cluster']] = i
  bv_list_df = pd.DataFrame(bv_list)
  bv_list_df['cluster'] = cluster_labels_reordered


  dominant_df, cluster_route = plot_mean_routes_with_temporal_color_and_probability_arrows(
    bv_list, cluster_labels_reordered, n_clusters=n_clusters,
    bg_img_path=f'./{base_directory}/background_template.png',
    save_csv_name=os.path.join(base_directory, 'dominant_routes.csv'),
  )
  # Plot the cdf of cluster_routes by pair by prob
  for pair, group in dominant_df.groupby('pair'):
    plt.plot(group['prob'].cumsum(), label=pair)
  plt.xlabel('Cluster')
  plt.ylabel('Cumulative Probability')
  plt.title('CDF of Cluster Routes by Pair')
  plt.legend()
  plt.savefig(os.path.join(base_directory, 'cdf_cluster_routes_by_pair.png'))
  plt.show()

  # Now split the pair by predator and prey

  # Split by pair and process network states in parallel.
  pred_network_states_dict, prey_network_states_dict = parallel_processing(bv_list_df, mix_data_directory)
  with open(os.path.join(base_directory, 'pred_network_states_dict.pkl'), 'wb') as f:
    pickle.dump(pred_network_states_dict, f)
  with open(os.path.join(base_directory, 'prey_network_states_dict.pkl'), 'wb') as f:
    pickle.dump(prey_network_states_dict, f)
  analyze_and_plot_preys(
    prey_network_states_dict, min_events=50,
    n_components=3,
  )

  ## Improved Analysis and Plotting of Prey Network States ##
  # 1. Plot the averaged trajectory of each cluster in prey's network states.
  # plot_avg_trajectories_by_cluster(prey_network_states_dict, species_label="Prey", n_components=2)

  # 2. Compare the averaged network state trajectories for predator vs prey.
  # compare_predator_and_prey(pred_network_states_dict, prey_network_states_dict, n_components=2)

  # For prey:
  plot_agent_clusters_by_key(prey_network_states_dict, agent_type="Prey", n_components=2)

  # For predator:
  plot_agent_clusters_by_key(pred_network_states_dict, agent_type="Predator", n_components=2)
