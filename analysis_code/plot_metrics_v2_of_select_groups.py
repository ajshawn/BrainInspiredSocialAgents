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
    plsc_path   = os.path.join(base_path, 'PLSC_results_dict.pkl')
    cross_path  = os.path.join(base_path, 'PLSC_results_cross_rollout_dict_NP.pkl')
    return {
        'serial': load_pickle(serial_path),
        'plsc':   load_pickle(plsc_path),
        'cross':  load_pickle(cross_path)
    }

def load_pickle(file_path: str):
    """Load a pickle file and return its contents."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def compute_rewards(results_dict, predator_ids, prey_ids):
    """
    Compute mean apple, acorn, and catch rewards for given predator/prey IDs.
    """
    apple_means, acorn_means, catch_means = [], [], []
    for predator_id in predator_ids:
        for prey_id in prey_ids:
            title = f'{predator_id}_{prey_id}'
            if title not in results_dict:
                # If missing, skip or append np.nan to keep indexing aligned
                apple_means.append(np.nan)
                acorn_means.append(np.nan)
                catch_means.append(np.nan)
                continue

            rewards = np.array(results_dict[title]['rewards'])
            # Rewards dimensions assumed: (num_rollouts, T, reward_dim)
            # Apple = exactly 1 in prey reward => (rewards[:, :, 1] == 1)
            # Acorn = > 1 in prey reward       => (rewards[:, :, 1] > 1)
            # Catch = exactly 1 in predator reward => (rewards[:, :, 0] == 1)

            apple_mean = (rewards[:, :, 1] == 1).mean()
            acorn_mean = (rewards[:, :, 1] > 1).mean()
            catch_mean = (rewards[:, :, 0] == 1).mean()

            apple_means.append(apple_mean)
            acorn_means.append(acorn_mean)
            catch_means.append(catch_mean)

    return (np.array(apple_means), np.array(acorn_means), np.array(catch_means))

def compute_plsc_data(plsc_dict, predator_ids, prey_ids):
    """
    Compute the mean of 'rank' (shared dimension), 'cor' (correlation),
    and 'cor_perm_array' (permutation correlation) from a PLSC dictionary,
    for the requested predator/prey IDs.
    """
    ranks, cors, cors_perm = [], [], []
    for predator_id in predator_ids:
        for prey_id in prey_ids:
            title = f'{predator_id}_{prey_id}'
            if title not in plsc_dict:
                ranks.append(np.nan)
                cors.append(np.nan)
                cors_perm.append(np.nan)
                continue

            data = plsc_dict[title]
            # Each is typically a list-of-lists or array-of-arrays
            # We'll do np.nanmean across them
            rank_val = np.nanmean(data['rank'])
            cor_val  = np.nanmean(np.array(data['cor'])[:, 0])       # e.g. shape (rollouts, 2)?
            cperm    = np.nanmean(np.array(data['cor_perm_array'])[:, :, 0])

            ranks.append(rank_val)
            cors.append(cor_val)
            cors_perm.append(cperm)

    ranks     = np.array(ranks)
    cors      = np.array(cors)
    cors_perm = np.array(cors_perm)
    return (ranks, cors, cors_perm)

def compute_cross_plsc_data(cross_results_dict, predator_ids, prey_ids):
    """
    Compute cross-rollout PLSC data (rank, cor, cor_perm_median)
    by searching for keys that match each predator_id + '_' + prey_id pattern.
    Also compute the per-title means if you like.
    """
    all_ranks, all_cors, all_cors_perm = [], [], []
    used_titles = set()

    # We'll also compute a "mean" across the matched keys per (pred, prey),
    # as in your example code
    mean_ranks_per_pair = []
    mean_cors_per_pair  = []
    mean_cperm_per_pair = []

    for pred_id in predator_ids:
        for pry_id in prey_ids:
            base_title = f'{pred_id}_{pry_id}'
            # Collect all cross-keys that contain the base_title
            matched_keys = [key for key in cross_results_dict.keys() if base_title in key]
            if not matched_keys:
                # No data => append NaN
                all_ranks.append(np.nan)
                all_cors.append(np.nan)
                all_cors_perm.append(np.nan)

                mean_ranks_per_pair.append(np.nan)
                mean_cors_per_pair.append(np.nan)
                mean_cperm_per_pair.append(np.nan)
                continue

            tmp_ranks, tmp_cors, tmp_cors_perm = [], [], []
            for mk in matched_keys:
                used_titles.add(mk)
                cdata = cross_results_dict[mk]

                r_val = np.nanmean(cdata['rank'])
                c_val = np.nanmean(np.array(cdata['cor'])[:, 0])
                p_val = np.nanmean(np.array(cdata['cor_perm_median'])[:, 0])

                tmp_ranks.append(r_val)
                tmp_cors.append(c_val)
                tmp_cors_perm.append(p_val)

            # For the "all_" arrays, let's just append the average
            # across these matched keys.
            # Alternatively, you could keep them individually.
            mean_r = np.nanmean(tmp_ranks)
            mean_c = np.nanmean(tmp_cors)
            mean_p = np.nanmean(tmp_cors_perm)

            all_ranks.append(mean_r)
            all_cors.append(mean_c)
            all_cors_perm.append(mean_p)

            # Also store them in "mean" arrays if you like
            mean_ranks_per_pair.append(mean_r)
            mean_cors_per_pair.append(mean_c)
            mean_cperm_per_pair.append(mean_p)

    # Convert lists to arrays
    all_ranks     = np.array(all_ranks)
    all_cors      = np.array(all_cors)
    all_cors_perm = np.array(all_cors_perm)

    return (
        all_ranks,
        all_cors,
        all_cors_perm,
        np.array(mean_ranks_per_pair),
        np.array(mean_cors_per_pair),
        np.array(mean_cperm_per_pair),
    )

def plot_box_swarm(df, part_key, output_dir, title_prefix="comparison"):
    """
    Creates and saves a box + swarm plot for the columns in df that contain part_key.
    """
    plt.figure(figsize=(0.15 * len(df.columns), 6))
    keys = [key for key in df.columns if part_key in key]
    if not keys:
        print(f"No columns found for part_key='{part_key}'. Skipping box+swarm plot.")
        return

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

    plt.title(f"{part_key} {title_prefix}")
    plt.tight_layout()

    filename = os.path.join(output_dir, f"{part_key}_comparison.png")
    plt.savefig(filename)
    plt.show()

def plot_heatmaps(df, part_keys, group_keys, ids_shape=(5, 5), output_dir=".", file_prefix="heatmap"):
    """
    Create and save heatmaps for the given part_keys and group_keys.
    The data is reshaped based on ids_shape (num_predators, num_preys) or vice versa.
    """
    fig, axs = plt.subplots(len(part_keys), len(group_keys),
                            figsize=(4 * len(group_keys), 3.5 * len(part_keys)),
                            sharex=True, sharey=True)

    if axs.ndim == 1:
        axs = axs.reshape(len(part_keys), -1)

    for i, pk in enumerate(part_keys):
        for j, gk in enumerate(group_keys):
            col_key = f'{pk} {gk}'
            if col_key not in df.columns:
                # skip if not found
                axs[i, j].set_title(f"Missing: {col_key}")
                axs[i, j].axis('off')
                continue
            data = df[col_key].values
            if data.size != (ids_shape[0] * ids_shape[1]):
                # shape mismatch => skip or attempt fallback
                axs[i, j].set_title(f"Size mismatch in {col_key}")
                axs[i, j].axis('off')
                continue

            data_reshaped = data.reshape(ids_shape)
            # You might want to transpose if you prefer rows=prey, cols=pred
            # data_reshaped = data_reshaped.T

            sns.heatmap(
                data_reshaped,
                ax=axs[i, j],
                cmap='coolwarm',
                cbar=False,
                annot=True,
                fmt=".2f"
            )
            axs[i, j].set_title(col_key)

    plt.tight_layout()
    filename = os.path.join(output_dir, f"{file_prefix}.png")
    plt.savefig(filename)
    plt.show()

def plot_heatmaps_all_scenarios(df, scenario_configs, part_keys, output_dir=".", file_prefix="heatmap"):
    """
    Create heatmaps for each 'part_key' (row) × 'scenario' (column) in subplots,
    using the shape implied by scenario_configs[scenario_name]['predator_ids']
    and scenario_configs[scenario_name]['prey_ids'].
    """
    scenario_names = list(scenario_configs.keys())
    n_rows = len(part_keys)
    n_cols = len(scenario_names)

    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        sharex=False,
        sharey=False
    )

    if axs.ndim == 1:
        if n_rows == 1:
            axs = axs.reshape(1, -1)
        else:
            axs = axs.reshape(-1, 1)

    for i, pk in enumerate(part_keys):
        for j, scenario_name in enumerate(scenario_names):
            ax = axs[i, j]
            predator_ids = scenario_configs[scenario_name]['predator_ids']
            prey_ids     = scenario_configs[scenario_name]['prey_ids']
            expected_size = len(predator_ids) * len(prey_ids)

            col_key = f"{pk} {scenario_name}"
            if col_key not in df.columns:
                ax.set_title(f"Missing: {col_key}")
                ax.axis('off')
                continue

            arr = df[col_key].dropna().values
            if arr.size != expected_size:
                ax.set_title(f"Size mismatch in {col_key} (arr.size={arr.size}, expected={expected_size})")
                ax.axis('off')
                continue

            data = arr.reshape(len(predator_ids), len(prey_ids))
            # Optionally transpose => rows=prey, cols=pred
            data = data.T

            sns.heatmap(
                data,
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap='coolwarm',
                xticklabels=predator_ids,
                yticklabels=prey_ids
            )
            ax.set_title(f"{pk} - {scenario_name}")
            if i == (n_rows - 1):
                ax.set_xlabel("Predator ID")
            if j == 0:
                ax.set_ylabel("Prey ID")

    plt.tight_layout()
    filename = os.path.join(output_dir, f"{file_prefix}.png")
    plt.savefig(filename)
    plt.show()
    print(f"Heatmap figure saved to {filename}")

def compute_subgroup_metrics(scenario_data, scenario_name, predator_ids, prey_ids):
    """
    Recompute the metrics for a particular subset of predator_ids, prey_ids
    within one scenario. Returns a DataFrame with the columns of interest.
    """
    results_serial = scenario_data[scenario_name]['serial']
    results_plsc   = scenario_data[scenario_name]['plsc']
    results_cross  = scenario_data[scenario_name]['cross']

    apple, acorn, catch = compute_rewards(results_serial, predator_ids, prey_ids)

    plsc_rank, plsc_cor, plsc_cor_perm = compute_plsc_data(results_plsc, predator_ids, prey_ids)
    plsc_delta = plsc_cor - plsc_cor_perm

    (plsc_cross_rank, plsc_cross_cor, plsc_cross_cor_perm,
     plsc_cross_rank_mean, plsc_cross_cor_mean, plsc_cross_cor_perm_mean
    ) = compute_cross_plsc_data(results_cross, predator_ids, prey_ids)
    plsc_cross_delta = plsc_cross_cor - plsc_cross_cor_perm

    data_dict = {
        f'apple {scenario_name}': apple,
        f'acorn {scenario_name}': acorn,
        f'catch {scenario_name}': catch,
        f'# PLSC shared dim {scenario_name}': plsc_rank,
        f'delta PLSC1 {scenario_name}': plsc_delta,
        f'# PLSC shared dim cross {scenario_name}': plsc_cross_rank,
        f'delta PLSC1 cross {scenario_name}': plsc_cross_delta,
    }
    return pd.DataFrame.from_dict(data_dict, orient='index').T

def plot_box_swarm_connect_lines(
    df,
    part_key,
    output_dir,
    title_prefix="comparison",
    line_color="gray",
    line_alpha=0.5
):
    """
    Creates and saves a box+swarm plot for columns in df that contain part_key.
    Additionally, draws lines to connect the same row across columns
    (i.e., the same (predator, prey) pair) to highlight perturbation changes.

    Parameters
    ----------
    df : pd.DataFrame
        Each row represents a single (pred_id, prey_id) pair.
    part_key : str
        Substring to match columns in df (e.g., "apple", "acorn", "catch").
    output_dir : str
        Directory to store the resulting figure.
    title_prefix : str
        A string to add to the figure title, describing the context.
    line_color : str
        Color for the connecting lines.
    line_alpha : float
        Transparency for the connecting lines.
    """
    # 1) Identify columns containing part_key
    keys = [col for col in df.columns if part_key in col]
    if not keys:
        print(f"[plot_box_swarm_connect_lines] No columns found for part_key='{part_key}'. Skipping.")
        return

    # 2) Subset DataFrame
    sub_df = df[keys]

    # 3) Create figure
    plt.figure(figsize=(0.15 * len(df.columns), 6))

    # 4) Boxplot and Swarmplot
    sns.boxplot(data=sub_df, showfliers=False)  # showfliers=False to reduce overlap
    sns.swarmplot(data=sub_df, color='black')

    # 5) Optionally label medians
    medians_of_selection = sub_df.median()
    for i in range(len(medians_of_selection)):
        plt.text(i, medians_of_selection[i],
                 f"{medians_of_selection[i]:.2f}",
                 ha='center', va='bottom', color='red')

    # 6) Connect the same row (same pair) across columns
    #    sub_df.shape[0] = number of pairs, sub_df.shape[1] = number of scenario columns
    for row_i in range(sub_df.shape[0]):
        yvals = sub_df.iloc[row_i, :].values
        xvals = np.arange(len(yvals))
        # If all NaN, skip
        if np.isnan(yvals).all():
            continue
        # Plot line across the columns
        plt.plot(xvals, yvals, color=line_color, alpha=line_alpha, marker='o')

    # 7) X-axis ticks, labels, title
    #    Wrap long column names, rotate them
    wrapped_labels = [textwrap.fill(k, 12) for k in keys]
    plt.xticks(range(len(keys)), wrapped_labels, rotation=90)
    plt.tick_params(axis='x', pad=0)

    plt.title(f"{part_key} {title_prefix}")
    plt.tight_layout()

    # 8) Save
    filename = os.path.join(output_dir, f"{part_key}_{title_prefix}_connected.png")
    plt.savefig(filename)
    plt.show()
    print(f"[plot_box_swarm_connect_lines] Figure saved => {filename}")

def main():
    """
    Main script to:
     1) Define scenarios/paths
     2) Load data
     3) Compute *full* DataFrame over the entire set of predator/prey IDs
     4) Do your standard plots
     5) Then do sub-group analysis for selected predator/prey ID sets
    """

    # =============== 1) Define constants and paths ===============
    trained_predator_ids = list(range(3))
    trained_prey_ids     = list(range(3, 13))

    cp7357_path = '/path/to/cp7357/pickles/'
    cp9651_path = '/path/to/cp9651/pickles/'
    AH_path     = '/path/to/AH/pickles/'
    # ... and so on, adjust these paths accordingly
    cp7357_perturb_pred_path  = '/path/to/cp7357/pickles_perturb_predator/'
    cp9651_perturb_pred_path  = '/path/to/cp9651/pickles_perturb_predator/'
    AH_perturb_pred_path      = '/path/to/AH/pickles_perturb_predator/'
    cp7357_perturb_prey_path  = '/path/to/cp7357/pickles_perturb_prey/'
    cp9651_perturb_prey_path  = '/path/to/cp9651/pickles_perturb_prey/'
    AH_perturb_prey_path      = '/path/to/AH/pickles_perturb_prey/'
    cp7357_perturb_both_path  = '/path/to/cp7357/pickles_perturb_both/'
    cp9651_perturb_both_path  = '/path/to/cp9651/pickles_perturb_both/'
    AH_perturb_both_path      = '/path/to/AH/pickles_perturb_both/'


    output_csv_path = './results/behavior_metrics.csv'
    output_plot_dir = os.path.dirname(output_csv_path)
    os.makedirs(output_plot_dir, exist_ok=True)

    # =============== 2) Define scenario configurations ===============
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
    for scenario_name, cfg in scenario_configs.items():
        scenario_data[scenario_name] = load_scenario_pickle_files(cfg['path'])

    # =============== 4) Compute full metrics for each scenario ===============
    #    (i.e. no sub-group filter, just the entire set of predator_ids/prey_ids)
    data_dict = {}
    for scenario_name, cfg in scenario_configs.items():
        predator_ids = cfg['predator_ids']
        prey_ids     = cfg['prey_ids']

        results_serial = scenario_data[scenario_name]['serial']
        results_plsc   = scenario_data[scenario_name]['plsc']
        results_cross  = scenario_data[scenario_name]['cross']

        apple, acorn, catch = compute_rewards(results_serial, predator_ids, prey_ids)
        plsc_rank, plsc_cor, plsc_cor_perm = compute_plsc_data(results_plsc, predator_ids, prey_ids)
        plsc_delta = plsc_cor - plsc_cor_perm

        (plsc_cross_rank, plsc_cross_cor, plsc_cross_cor_perm,
         plsc_cross_rank_mean, plsc_cross_cor_mean, plsc_cross_cor_perm_mean
        ) = compute_cross_plsc_data(results_cross, predator_ids, prey_ids)
        plsc_cross_delta = plsc_cross_cor - plsc_cross_cor_perm

        data_dict[f'apple {scenario_name}']                   = apple
        data_dict[f'acorn {scenario_name}']                   = acorn
        data_dict[f'catch {scenario_name}']                   = catch
        data_dict[f'# PLSC shared dim {scenario_name}']       = plsc_rank
        data_dict[f'delta PLSC1 {scenario_name}']             = plsc_delta
        data_dict[f'# PLSC shared dim cross {scenario_name}'] = plsc_cross_rank
        data_dict[f'delta PLSC1 cross {scenario_name}']       = plsc_cross_delta

    df = pd.DataFrame.from_dict(data_dict, orient='index').T
    df.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}.")

    # =============== 5) Standard Box+Swarm Plots (all data) ===============
    for part_key in ['apple', 'acorn', 'catch', '# PLSC shared dim', 'delta PLSC1']:
        plot_box_swarm(df, part_key, output_plot_dir, title_prefix="All scenarios")

    # =============== 6) Example Heatmaps (all data) ===============
    # You can choose an appropriate shape for each scenario individually.
    # Here's an example for plotting them all in a single figure:
    # (Each scenario may have a different shape, so the universal call may or may not be feasible.)
    part_keys = ['apple', 'acorn', 'catch']
    plot_heatmaps_all_scenarios(
        df=df,
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

    # === 9) Additional sub-group analysis ===
    #
    # For example, suppose we want:
    #   - “good predator” = [0,1,2]
    #   - “apple-prey”    = [3,8,10,11,12] (cp9651) or [8,9,10,11,12] (AH)
    #   - “acorn-prey”    = [5,6,7] (cp9651) (none in AH)
    #   - “failed-prey”   = [4,9] (cp9651) or [5,6,7] (AH)
    #
    # We then pick which scenario set we want to examine.

    subgroups = {
        # Good predator vs. apple (cp9651 + its perturbations)
        'goodpred_apple_cp9651': {
            'scenarios': ['cp9651','cp9651_perturb_pred','cp9651_perturb_prey','cp9651_perturb_both'],
            'predator_ids': [0,1,2],
            'prey_ids': [3,8,10,11,12]
        },
        # Good predator vs. apple (AH + its perturbations)
        'goodpred_apple_AH': {
            'scenarios': ['AH','AH_perturb_pred','AH_perturb_prey','AH_perturb_both'],
            'predator_ids': [0,1,2],
            'prey_ids': [8,9,10,11,12]
        },
        # Good predator vs. acorn (cp9651 + perturbations)
        'goodpred_acorn_cp9651': {
            'scenarios': ['cp9651','cp9651_perturb_pred','cp9651_perturb_prey','cp9651_perturb_both'],
            'predator_ids': [0,1,2],
            'prey_ids': [5,6,7]
        },
        # Good predator vs. failed (cp9651 + perturbations)
        'goodpred_failed_cp9651': {
            'scenarios': ['cp9651','cp9651_perturb_pred','cp9651_perturb_prey','cp9651_perturb_both'],
            'predator_ids': [0,1,2],
            'prey_ids': [4,9]
        },
        # Good predator vs. failed (AH + perturbations)
        'goodpred_failed_AH': {
            'scenarios': ['AH','AH_perturb_pred','AH_perturb_prey','AH_perturb_both'],
            'predator_ids': [0,1,2],
            'prey_ids': [5,6,7]
        },
    }

    # We’ll create a sub-DataFrame and do the same box/swarm + heatmap for each subgroup scenario
    for subgroup_name, subgroup_cfg in subgroups.items():
        # Build an accumulator so that we combine columns from all scenarios
        # in the same sub-DataFrame. This will let us do a boxplot across them.
        combined_data_dict = {}

        for scenario_name in subgroup_cfg['scenarios']:
            if scenario_name not in scenario_data:
                print(f"Warning: scenario '{scenario_name}' not found.")
                continue

            pred_ids = subgroup_cfg['predator_ids']
            prey_ids = subgroup_cfg['prey_ids']

            df_sub = compute_subgroup_metrics(
                scenario_data,
                scenario_name,
                pred_ids,
                prey_ids
            )
            # df_sub has columns like "apple cp9651", "acorn cp9651", etc.
            # We'll merge them into combined_data_dict
            for col in df_sub.columns:
                combined_data_dict[col] = df_sub[col].values  # keep it 1D

        if not combined_data_dict:
            print(f"No data for subgroup '{subgroup_name}'")
            continue

        df_combined = pd.DataFrame.from_dict(combined_data_dict, orient='index').T

        # For convenience, store to CSV
        csv_name = os.path.join(output_plot_dir, f"{subgroup_name}_metrics.csv")
        df_combined.to_csv(csv_name, index=False)
        print(f"Subgroup '{subgroup_name}': data saved => {csv_name}")

        # Now do the same style of box+swarm or heatmaps
        # Box+swarm for selected metrics:
        for part_key in ['apple', 'acorn', 'catch', '# PLSC shared dim', 'delta PLSC1']:
            if part_key in ['apple', 'acorn', 'catch']:

                # plot_box_swarm(
                #     df_combined * 1000,
                #     part_key,
                #     output_plot_dir,
                #     title_prefix=f"{subgroup_name}"
                # )
                plot_box_swarm_connect_lines(
                    df=df_combined * 1000,
                    part_key=part_key,  # or "acorn", "catch", etc.
                    output_dir=output_plot_dir,
                    title_prefix=subgroup_name  # name of this sub-group
                )
            else:
                plot_box_swarm(
                    df_combined,
                    part_key,
                    output_plot_dir,
                    title_prefix=f"{subgroup_name}"
                )

        # If you want a heatmap, you need to pick one scenario at a time OR pick a shape.
        # Example: we do a heatmap for each scenario in the subgroup:
        for scenario_name in subgroup_cfg['scenarios']:
            # We'll just do a single scenario’s columns.
            # Need the shape: (#predators, #preys).
            # (If your predator_ids × prey_ids for this scenario is e.g. 3 × 5 => (3,5).)

            pred_ids = subgroup_cfg['predator_ids']
            prey_ids = subgroup_cfg['prey_ids']
            shape_ = (len(pred_ids), len(prey_ids))

            # We'll filter columns for that scenario:
            scenario_cols = [c for c in df_combined.columns if scenario_name in c]

            # Just an example call:
            plot_heatmaps(
                df_combined[scenario_cols],
                part_keys=['apple', 'acorn', 'catch'],  # or whichever
                group_keys=[scenario_name],  # single scenario
                ids_shape=shape_,
                output_dir=output_plot_dir,
                file_prefix=f"{subgroup_name}_{scenario_name}_heatmap"
            )
            # Similarly, you could do a second call for # PLSC shared dim, delta PLSC1, etc.

if __name__ == "__main__":
    main()
