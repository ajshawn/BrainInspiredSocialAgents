import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap
import os
# import concurrent.futures


def load_scenario_pickle_files(base_path):
  """Helper function to load pickle files for a single scenario."""
  serial_path = os.path.join(base_path, 'serial_results_dict.pkl')
  plsc_path = os.path.join(base_path, 'PLSC_results_dict.pkl')
  cross_path = os.path.join(base_path, 'PLSC_results_cross_rollout_dict_NP.pkl')

  return {
    'serial': load_pickle(serial_path),
    'plsc': load_pickle(plsc_path),
    'cross': load_pickle(cross_path)
  }

def load_pickle(file_path: str):
  """Load a pickle file and return its contents."""
  with open(file_path, 'rb') as f:
    return pickle.load(f)


def compute_rewards(
        results_dict: dict,
        predator_ids: list,
        prey_ids: list
):
  """
  Compute mean apple, acorn, and catch rewards for given predator/prey ID pairs
  from a results dictionary.

  Parameters
  ----------
  results_dict : dict
      A dictionary of the form: results_dict[title]['rewards'] => a nested array
      that stores reward information.
  predator_ids : list
      A list of predator IDs to compute the metrics for.
  prey_ids : list
      A list of prey IDs to compute the metrics for.

  Returns
  -------
  tuple of np.ndarrays
      (apple_means, acorn_means, catch_means) all in 1D arrays with length =
      len(predator_ids) * len(prey_ids)
  """
  apple_means, acorn_means, catch_means = [], [], []
  print(predator_ids, prey_ids, results_dict.keys())
  for predator_id in predator_ids:
    for prey_id in prey_ids:
      title = f'{predator_id}_{prey_id}'
      rewards = np.array(results_dict[title]['rewards'])
      # Rewards dimensions assumed: (num_rollouts, T, reward_dim)
      # Based on your usage:
      #   - predator reward is at index 0
      #   - prey reward is at index 1
      # Apple = exactly 1 in prey reward
      # Acorn = > 1 in prey reward
      # Catch = exactly 1 in predator reward
      apple_mean = (rewards[:, :, 1] == 1).mean()
      acorn_mean = (rewards[:, :, 1] > 1).mean()
      catch_mean = (rewards[:, :, 0] == 1).mean()

      apple_means.append(apple_mean)
      acorn_means.append(acorn_mean)
      catch_means.append(catch_mean)

  return (
    np.array(apple_means),
    np.array(acorn_means),
    np.array(catch_means)
  )


def compute_plsc_data(
        plsc_dict: dict,
        predator_ids: list,
        prey_ids: list
):
  """
  Compute the mean of 'rank' (shared dimension), 'cor' (correlation),
  and 'cor_perm_array' (permutation correlation) from a PLSC dictionary.

  Returns
  -------
  tuple of 1D np.ndarrays
      (plsc_rank, plsc_cor, plsc_cor_perm)
  """
  ranks, cors, cors_perm = [], [], []
  for predator_id in predator_ids:
    for prey_id in prey_ids:
      title = f'{predator_id}_{prey_id}'
      ranks.append(plsc_dict[title]['rank'])
      cors.append(plsc_dict[title]['cor'])
      cors_perm.append(plsc_dict[title]['cor_perm_array'])

  # Each entry is a list-of-lists or array-of-arrays
  # We take the nanmean across the 2nd dimension
  # Adjust indexing as needed based on your data structure
  ranks = np.nanmean(ranks, axis=1)  # shape [n, ...] => mean of dimension 1
  cors = np.nanmean(np.array(cors)[:, :, 0], axis=1)
  cors_perm = np.nanmean(np.array(cors_perm)[:, :, :, 0], axis=(1, 2))

  return ranks, cors, cors_perm


def compute_cross_plsc_data(
        cross_results_dict: dict,
        predator_ids: list,
        prey_ids: list
):
  """
  Compute cross-rollout PLSC data (rank, cor, cor_perm_median)
  by searching for keys that match each predator_id + '_' + prey_id pattern.
  Also compute the per-title means.

  Returns
  -------
  tuple of 1D np.ndarrays
      (plsc_rank, plsc_cor, plsc_cor_perm,
       plsc_rank_mean, plsc_cor_mean, plsc_cor_perm_mean)
  """
  all_ranks, all_cors, all_cors_perm = [], [], []
  mean_ranks, mean_cors, mean_cors_perm = [], [], []

  used_titles = set()
  for pred_id in predator_ids:
    for pry_id in prey_ids:
      base_title = f'{pred_id}_{pry_id}'
      # Collect all cross-keys that contain the base_title
      matched_keys = [key for key in cross_results_dict.keys() if base_title in key]

      tmp_ranks, tmp_cors, tmp_cors_perm = [], [], []
      for mk in matched_keys:
        if mk not in used_titles:
          used_titles.add(mk)
          all_ranks.append(cross_results_dict[mk]['rank'])
          all_cors.append(cross_results_dict[mk]['cor'])
          all_cors_perm.append(cross_results_dict[mk]['cor_perm_median'])

        tmp_ranks.append(cross_results_dict[mk]['rank'])
        tmp_cors.append(cross_results_dict[mk]['cor'])
        tmp_cors_perm.append(cross_results_dict[mk]['cor_perm_median'])

      # Compute per-title mean
      mean_ranks.append(np.nanmean(tmp_ranks))
      try:
        mean_cors.append(np.nanmean(np.array(tmp_cors)[:, :, 0]))
      except:
        print('Error in mean_cors')

      mean_cors_perm.append(np.nanmean(np.array(tmp_cors_perm)[:, :, 0]))

  # Convert lists to arrays
  all_ranks = np.mean(all_ranks, axis=1)
  all_cors = np.mean(np.array(all_cors)[:, :, 0], axis=1)
  all_cors_perm = np.mean(np.array(all_cors_perm)[:, :, 0], axis=1)

  mean_ranks = np.array(mean_ranks)
  mean_cors = np.array(mean_cors)
  mean_cors_perm = np.array(mean_cors_perm)

  return (
    all_ranks, all_cors, all_cors_perm,
    mean_ranks, mean_cors, mean_cors_perm
  )


def plot_box_swarm(
        df: pd.DataFrame,
        part_key: str,
        output_dir: str,
        title_prefix: str = "comparison between naive and trained"
):
  """
  Creates and saves a box + swarm plot for the columns in df that contain part_key.

  Parameters
  ----------
  df : pd.DataFrame
      DataFrame containing columns of interest.
  part_key : str
      Substring to match columns in df.
  output_dir : str
      Where to save the output figure.
  title_prefix : str
      Title text to display on the plot.
  """
  plt.figure(figsize=(0.15 * len(df.columns), 6))
  keys = [key for key in df.columns if part_key in key]

  sns.boxplot(data=df[keys])
  sns.swarmplot(data=df[keys], color='black')
  medians_of_selection = df[keys].median()

  # Label each box with its median value
  for i in range(medians_of_selection.shape[0]):
    plt.text(i, medians_of_selection[i], f"{medians_of_selection[i]:.2f}",
             ha='center', va='bottom', color='red')

  # Rotate and wrap x-tick labels
  labels = [textwrap.fill(label, 12) for label in keys]
  plt.xticks(range(len(labels)), labels, rotation=90)
  plt.tick_params(axis='x', pad=0)

  plt.title(f'{part_key} {title_prefix}')
  plt.tight_layout()

  # Save
  filename = os.path.join(output_dir, f'{part_key}_comparison.png')
  plt.savefig(filename)
  plt.show()


def plot_heatmaps(
        df: pd.DataFrame,
        part_keys: list,
        group_keys: list,
        ids_shape=(5, 5),
        output_dir=".",
        file_prefix="heatmap"
):
  """
  Create and save heatmaps for the given part_keys and group_keys. The data is
  reshaped based on ids_shape (predator x prey).

  Parameters
  ----------
  df : pd.DataFrame
      DataFrame containing columns to plot.
  part_keys : list of str
      e.g. ['apple', 'acorn', 'catch']
  group_keys : list of str
      e.g. ['npnp', 'tptp', 'tpnp', 'tptp perturb-predator']
  ids_shape : tuple
      The shape to reshape data into, typically (# prey, # predator).
      NOTE: Adjust as needed.
  output_dir : str
      Directory path to store the saved figures.
  file_prefix : str
      Prefix for the saved figure file name.
  """
  fig, axs = plt.subplots(len(part_keys), len(group_keys),
                          figsize=(4 * len(group_keys), 3.5 * len(part_keys)),
                          sharex=True, sharey=True)
  if axs.ndim == 1:
    # Add one more dim
    axs = axs.reshape(-1, 1)
  for i, pk in enumerate(part_keys):
    for j, gk in enumerate(group_keys):
      col_key = f'{pk} {gk}'
      data = df[col_key].dropna().values.reshape(*ids_shape).T
      sns.heatmap(data, ax=axs[i, j], cmap='coolwarm',
                  cbar=False, annot=True, fmt=".2f")
      axs[i, j].set_title(col_key)
      if i == len(part_keys) - 1:
        axs[i, j].set_xlabel('predator')
      if j == 0:
        axs[i, j].set_ylabel('prey')

  plt.tight_layout()
  filename = os.path.join(output_dir, f'{file_prefix}.png')
  plt.savefig(filename)
  plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import os


def plot_heatmaps_all_scenarios(
    df,
    scenario_configs,
    part_keys,
    output_dir=".",
    file_prefix="heatmap"
):
  """
  Create heatmaps for each 'part_key' (row) × 'scenario' (column) in subplots.

  Parameters
  ----------
  df : pd.DataFrame
      DataFrame containing columns named like '{part_key} {scenario_name}'.
  scenario_configs : dict
      A dict where each entry has the structure:
          scenario_name: {
              'path': <some_path>,
              'predator_ids': [...],
              'prey_ids': [...]
          }
  part_keys : list of str
      e.g. ['apple', 'acorn', 'catch'] or ['# PLSC shared dim', 'delta PLSC1'].
  output_dir : str
      Where to save the resulting plot image.
  file_prefix : str
      Prefix for the saved figure file name.
  """
  # 1) Prepare subplots: rows = part_keys, cols = scenarios
  scenario_names = list(scenario_configs.keys())
  n_rows = len(part_keys)
  n_cols = len(scenario_names)

  fig, axs = plt.subplots(
    n_rows, n_cols,
    figsize=(4 * n_cols, 3.5 * n_rows),
    sharex=False,  # set False or True as you prefer
    sharey=False
  )

  # In case we only have 1 row or 1 col
  if axs.ndim == 1:
    if n_rows == 1:
      axs = axs.reshape(1, -1)  # shape = (1, n_cols)
    else:
      axs = axs.reshape(-1, 1)  # shape = (n_rows, 1)

  # 2) Loop over each row (part_key) and each column (scenario)
  for i, pk in enumerate(part_keys):
    for j, scenario_name in enumerate(scenario_names):
      ax = axs[i, j]

      # 3) Grab relevant predator/prey info
      predator_ids = scenario_configs[scenario_name]['predator_ids']
      prey_ids = scenario_configs[scenario_name]['prey_ids']
      expected_size = len(predator_ids) * len(prey_ids)

      # 4) Build column name for the DataFrame
      col_key = f"{pk} {scenario_name}"
      if col_key not in df.columns:
        print(f"Warning: Missing column '{col_key}' in DataFrame.")
        continue

      # 5) Extract data and reshape
      arr = df[col_key].dropna().values
      if arr.size != expected_size:
        print(f"Warning: Data for '{col_key}' has size {arr.size}, "
              f"but expected {expected_size} (pred:{len(predator_ids)}, "
              f"prey:{len(prey_ids)}). Skipping.")
        continue

      # reshape => (predators, prey)
      data = arr.reshape(len(predator_ids), len(prey_ids))
      # transpose => rows = prey, cols = predator
      data = data.T

      # 6) Plot heatmap
      sns.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        # Provide IDs as tick labels
        xticklabels=predator_ids,
        yticklabels=prey_ids
      )

      # 7) Title / labeling
      wrapped_title = textwrap.fill(f"{pk} - {scenario_name}", 18)
      ax.set_title(wrapped_title)
      if i == (n_rows - 1):
        ax.set_xlabel("Predator ID")
      if j == 0:
        ax.set_ylabel("Prey ID")

  plt.tight_layout()

  # 8) Save and show
  filename = os.path.join(output_dir, f"{file_prefix}.png")
  plt.savefig(filename)
  plt.show()
  print(f"Heatmap figure saved to {filename}")


def main():
  """
  Main script to load data, compute metrics, store them in a DataFrame,
  export to CSV, and generate visualizations.
  """

  # =============== 1) Define constants and paths ===============
  trained_predator_ids = list(range(3))
  trained_prey_ids = list(range(3, 13))

  # Paths
  cp7357_path = f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles/'
  cp9651_path = f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles/'
  AH_path = f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles/'
  cp7357_perturb_pred_path = f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles_perturb_predator/'
  cp9651_perturb_pred_path = f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles_perturb_predator/'
  AH_perturb_pred_path = f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_predator/'
  cp7357_perturb_prey_path = f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles_perturb_prey/'
  cp9651_perturb_prey_path = f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles_perturb_prey/'
  AH_perturb_prey_path = f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_prey/'
  cp7357_perturb_both_path = f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles_perturb_both/'
  cp9651_perturb_both_path = f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles_perturb_both/'
  AH_perturb_both_path = f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_both/'

  # Output CSV path
  output_csv_path = (
    './results/behavior_metrics.csv'
  )
  output_plot_dir = os.path.dirname(output_csv_path)
  os.makedirs(output_plot_dir, exist_ok=True)

  # =============== 2) Define scenario configurations ===============
  # Each scenario will have a path, predator IDs, and prey IDs
  scenario_configs = {
    'cp7357': {
      'path': cp7357_path,
      'predator_ids': trained_predator_ids,
      'prey_ids': trained_prey_ids
    },
    'cp9651': {
      'path': cp9651_path,
      'predator_ids': trained_predator_ids,
      'prey_ids': trained_prey_ids
    },
    'AH': {
      'path': AH_path,
      'predator_ids': list(range(5)),
      'prey_ids': list(range(5, 13)),
    },
    'cp7357_perturb_pred': {
      'path': cp7357_perturb_pred_path,
      'predator_ids': trained_predator_ids,
      'prey_ids': trained_prey_ids
    },
    'cp9651_perturb_pred': {
      'path': cp9651_perturb_pred_path,
      'predator_ids': trained_predator_ids,
      'prey_ids': trained_prey_ids
    },
    'AH_perturb_pred': {
      'path': AH_perturb_pred_path,
      'predator_ids': list(range(5)),
      'prey_ids': list(range(5, 13)),
    },
    'cp7357_perturb_prey': {
      'path': cp7357_perturb_prey_path,
      'predator_ids': trained_predator_ids,
      'prey_ids': trained_prey_ids
    },
    'cp9651_perturb_prey': {
      'path': cp9651_perturb_prey_path,
      'predator_ids': trained_predator_ids,
      'prey_ids': trained_prey_ids
    },
    'AH_perturb_prey': {
      'path': AH_perturb_prey_path,
      'predator_ids': list(range(5)),
      'prey_ids': list(range(5, 13)),
    },
    'cp7357_perturb_both': {
      'path': cp7357_perturb_both_path,
      'predator_ids': trained_predator_ids,
      'prey_ids': trained_prey_ids
    },
    'cp9651_perturb_both': {
      'path': cp9651_perturb_both_path,
      'predator_ids': trained_predator_ids,
      'prey_ids': trained_prey_ids
    },
    'AH_perturb_both': {
      'path': AH_perturb_both_path,
      'predator_ids': list(range(5)),
      'prey_ids': list(range(5, 13)),
    }

  }

  # =============== 3) Load all data for each scenario ===============
  scenario_data = {}

  # # Use a ProcessPoolExecutor for parallel CPU-bound tasks
  # # or ThreadPoolExecutor for I/O-bound tasks
  # with concurrent.futures.ProcessPoolExecutor() as executor:
  #   # Submit jobs for each scenario
  #   future_to_scenario = {
  #     executor.submit(load_scenario_pickle_files, cfg['path']): scenario_name
  #     for scenario_name, cfg in scenario_configs.items()
  #   }
  #
  #   # Collect results as they complete
  #   for future in concurrent.futures.as_completed(future_to_scenario):
  #     scenario_name = future_to_scenario[future]
  #     try:
  #       scenario_data[scenario_name] = future.result()
  #     except Exception as exc:
  #       print(f'{scenario_name} generated an exception: {exc}')

  for scenario_name, cfg in scenario_configs.items():
    scenario_data[scenario_name] = load_scenario_pickle_files(cfg['path'])


  # =============== 4) Compute metrics for each scenario ===============
  for scenario_name, cfg in scenario_configs.items():
    results_serial = scenario_data[scenario_name]['serial']
    results_plsc = scenario_data[scenario_name]['plsc']
    results_cross = scenario_data[scenario_name]['cross']

    predator_ids = cfg['predator_ids']
    prey_ids = cfg['prey_ids']

    # -- Compute Reward metrics
    apple, acorn, catch = compute_rewards(
      results_serial, predator_ids, prey_ids
    )

    # -- Compute normal PLSC
    plsc_rank, plsc_cor, plsc_cor_perm = compute_plsc_data(
      results_plsc, predator_ids, prey_ids
    )
    plsc_delta = plsc_cor - plsc_cor_perm

    # -- Compute cross-rollout PLSC
    (plsc_cross_rank, plsc_cross_cor, plsc_cross_cor_perm,
     plsc_cross_rank_mean, plsc_cross_cor_mean, plsc_cross_cor_perm_mean
     ) = compute_cross_plsc_data(
      results_cross, predator_ids, prey_ids
    )
    plsc_cross_delta = plsc_cross_cor - plsc_cross_cor_perm

    # -- Store them in scenario_data
    scenario_data[scenario_name].update({
      'apple': apple,
      'acorn': acorn,
      'catch': catch,
      'plsc_rank': plsc_rank,
      'plsc_delta': plsc_delta,
      'plsc_cross_rank': plsc_cross_rank,
      'plsc_cross_delta': plsc_cross_delta
    })

  # =============== 5) Build a consolidated DataFrame ===============
  # We'll create a dictionary of lists/arrays, then transpose at the end
  data_dict = {}
  for scenario_name in scenario_configs:
    # scenario_data[scenario_name]['apple'] is your array of apples, etc.
    data_dict[f'apple {scenario_name}'] = scenario_data[scenario_name]['apple']
    data_dict[f'acorn {scenario_name}'] = scenario_data[scenario_name]['acorn']
    data_dict[f'catch {scenario_name}'] = scenario_data[scenario_name]['catch']
    data_dict[f'# PLSC shared dim {scenario_name}'] = scenario_data[scenario_name]['plsc_rank']
    data_dict[f'delta PLSC1 {scenario_name}'] = scenario_data[scenario_name]['plsc_delta']
    data_dict[f'# PLSC shared dim cross {scenario_name}'] = scenario_data[scenario_name]['plsc_cross_rank']
    data_dict[f'delta PLSC1 cross {scenario_name}'] = scenario_data[scenario_name]['plsc_cross_delta']

  df = pd.DataFrame.from_dict(data_dict, orient='index').T

  # =============== 6) Save to CSV ===============
  df.to_csv(output_csv_path, index=False)
  print(f"Data saved to {output_csv_path}.")

  # =============== 7) Plotting ===============
  # 7a) Box & swarm plots
  for part_key in ['apple', 'acorn', 'catch', '# PLSC', 'delta PLSC1']:
    if part_key in ['apple', 'acorn', 'catch']:
      # Multiply by 1000 if needed, as in your original code
      plot_box_swarm(df * 1000, part_key, output_plot_dir)
    else:
      plot_box_swarm(df, part_key, output_plot_dir)

  # # 7b) Heatmaps for rewards
  # plot_heatmaps(
  #   df * 1000,
  #   part_keys=['apple', 'acorn', 'catch'],
  #   group_keys=list(scenario_configs.keys()),  # e.g. ['npnp', 'tptp', ...]
  #   ids_shape=(3, 10),  # Adjust if needed
  #   output_dir=output_plot_dir,
  #   file_prefix='reward per 1k step (mean)'
  # )
  #
  # # 7c) Heatmaps for # PLSC and delta PLSC1
  # plot_heatmaps(
  #   df,
  #   part_keys=['# PLSC shared dim', 'delta PLSC1'],
  #   group_keys=list(scenario_configs.keys()),
  #   ids_shape=(3, 10),
  #   output_dir=output_plot_dir,
  #   file_prefix='plsc'
  # )
  part_keys = ['apple', 'acorn', 'catch']
  plot_heatmaps_all_scenarios(
      df=df * 1000,
      scenario_configs=scenario_configs,
      part_keys=part_keys,
      output_dir=output_plot_dir,
      file_prefix="all_scenarios_rewards"
  )
  part_keys = ['# PLSC shared dim', 'delta PLSC1']
  plot_heatmaps_all_scenarios(
      df=df,
      scenario_configs=scenario_configs,
      part_keys=part_keys,
      output_dir=output_plot_dir,
      file_prefix="all_scenarios_plsc"
  )

if __name__ == "__main__":
  main()
