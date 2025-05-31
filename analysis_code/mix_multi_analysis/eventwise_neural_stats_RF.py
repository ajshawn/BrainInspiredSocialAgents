import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Any, Optional
from joblib import Parallel, delayed
import numpy as np
import argparse  # Added for command-line arguments


# --- Generic Plotting Function (Corrected & Maintained) ---
def plot_box_swarm(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str],
    role: str,
    out_dir: Path,
    title: str,
    palette: Any = "Set2"
) -> None:
  """Generic box+swarm plot of y vs x, with optional hue, for one role."""
  sub = df[df["role"] == role].copy()
  out_dir.mkdir(parents=True, exist_ok=True)

  if x not in sub.columns:
    print(f"Warning: Column '{x}' not found for role '{role}'. Skipping plot for {title}.")
    return

  sub = sub.dropna(subset=[x, y])
  if hue and hue in sub.columns:
    sub = sub.dropna(subset=[hue])
  elif hue and hue not in sub.columns:
    print(f"Warning: Hue column '{hue}' not found for role '{role}'. Plotting without hue.")
    hue = None

  if sub.empty:
    print(f"No data to plot for role '{role}' and metric '{y}' for category '{x}'. Skipping plot for {title}.")
    return

  order = sorted(sub[x].unique())

  fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.25), 5))

  boxplot_palette = palette
  if hue is None and isinstance(palette, (list, tuple)):
    boxplot_palette = palette[0] if palette else "gray"

  sns.boxplot(x=x, y=y, hue=hue, data=sub, ax=ax, palette=boxplot_palette, order=order)

  if hue:
    sns.swarmplot(x=x, y=y, hue=hue, data=sub, palette=palette, size=2, ax=ax, order=order, dodge=True)
  else:
    sns.swarmplot(x=x, y=y, data=sub, color=".25", size=2, ax=ax, order=order)

  if ax.get_legend() is not None:
    ax.get_legend().remove()

  ax.set_title(f"{title} ({role.capitalize()})")
  ax.set_xlabel(x.replace("_", " ").capitalize())
  ax.set_ylabel(y.replace("_", " ").capitalize())
  ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
  ax.grid(axis='y', linestyle='--', alpha=0.7)
  plt.tight_layout()

  filename = f"{title.replace(' ', '_')}_{y}_by_{x}_{role}.png".replace("/", "_").replace("\\", "_")
  fig.savefig(out_dir / filename, dpi=150)
  plt.close(fig)


# --- 1. Fraction of each event to lifespan/reward ---
def plot_event_fractions(
    df: pd.DataFrame,
    events: List[str],
    figure_dir: Path,
    parallelize: bool = False,
    n_jobs: int = -1
) -> None:
  """
  Box+swarm of fraction of lifespan and fraction of reward for each event in `events`.
  """
  roles = ["predator", "prey"]
  tasks = []

  # 1) lifespan fractions
  lifespan_vars = [f"{ev}_lifespan" for ev in events]
  cols_to_check_life = ["agent", "role", "lifespan_total"] + lifespan_vars
  if not all(col in df.columns for col in cols_to_check_life):
    print(
      f"Warning: Missing one or more lifespan columns for fraction calculation. Expected: {cols_to_check_life}. Skipping lifespan fraction plot.")
  else:
    frac_life = df[cols_to_check_life].copy()
    for v in lifespan_vars:
      frac_life[v] = frac_life[v] / frac_life["lifespan_total"].replace(0, np.nan)
    m1 = frac_life.melt(
      id_vars=["agent", "role"],
      value_vars=lifespan_vars,
      var_name="event",
      value_name="fraction"
    ).dropna(subset=["fraction"])
    m1["event"] = m1["event"].str.replace("_lifespan", "")

    for role in roles:
      out_dir = figure_dir / "event_fractions" / "lifespan"
      tasks.append(delayed(plot_box_swarm)(
        m1, "event", "fraction", None, role,
        out_dir,
        title="Fraction of lifespan"
      ))

  # 2) reward fractions
  reward_vars = [f"{ev}_reward" for ev in events]
  cols_to_check_reward = ["agent", "role",
                          "total_reward"] + reward_vars  # Assuming 'total_reward' column exists for overall reward
  # If 'total_reward' isn't available, sum of event rewards can be used, but check if it's appropriate.
  if 'total_reward' not in df.columns:
    print(
      "Warning: 'total_reward' column not found for reward fraction calculation. Using sum of event rewards as denominator.")
    df['total_reward_calculated_from_events'] = df[reward_vars].sum(axis=1)
    reward_denominator_col = 'total_reward_calculated_from_events'
  else:
    reward_denominator_col = 'total_reward'

  if not all(col in df.columns for col in ["agent", "role"] + reward_vars + [reward_denominator_col]):
    print(
      f"Warning: Missing one or more reward columns for fraction calculation. Expected: {cols_to_check_reward}. Skipping reward fraction plot.")
  else:
    frac_rew = df[["agent", "role"] + reward_vars + [reward_denominator_col]].copy()
    for v in reward_vars:
      frac_rew[v] = frac_rew[v] / frac_rew[reward_denominator_col].replace(0, np.nan)
    m2 = frac_rew.melt(
      id_vars=["agent", "role"],
      value_vars=reward_vars,
      var_name="event",
      value_name="fraction"
    ).dropna(subset=["fraction"])
    m2["event"] = m2["event"].str.replace("_reward", "")

    for role in roles:
      out_dir = figure_dir / "event_fractions" / "reward"
      tasks.append(delayed(plot_box_swarm)(
        m2, "event", "fraction", None, role,
        out_dir,
        title="Fraction of reward"
      ))

  if 'total_reward_calculated_from_events' in df.columns:  # Clean up temporary column
    df.drop(columns=['total_reward_calculated_from_events'], inplace=True)

  if tasks:
    print(f"Plotting event fractions with {'parallel' if parallelize else 'sequential'} execution...")
    if parallelize:
      Parallel(n_jobs=n_jobs, backend='loky')(tasks)
    else:
      for task in tasks:
        task()


# --- 2. Metrics by Category and Role ---
def plot_metrics_by_category_and_role(
    df: pd.DataFrame,
    metrics: List[str],
    category_col: str,
    base_figure_dir: Path,
    plot_title_suffix: str,
    parallelize: bool = False,
    n_jobs: int = -1
) -> None:
  """
  Generic function to plot metrics by a given category column.
  """
  roles = ['predator', 'prey']
  tasks = []

  if category_col not in df.columns:
    print(f"Error: Category column '{category_col}' not found in DataFrame. Skipping plots for {plot_title_suffix}.")
    return

  for role in roles:
    for m in metrics:
      if m not in df.columns:
        print(f"Warning: Metric column '{m}' not found. Skipping plot for {m} by {plot_title_suffix}, role {role}.")
        continue
      out_dir = base_figure_dir / f"metrics_by_{category_col}" / role
      tasks.append(delayed(plot_box_swarm)(
        df, category_col, m, None, role,
        out_dir,
        title=f"{m.replace('_', ' ').title()} by {plot_title_suffix}"
      ))

  if tasks:
    print(f"Plotting metrics by {category_col} with {'parallel' if parallelize else 'sequential'} execution...")
    if parallelize:
      Parallel(n_jobs=n_jobs, backend='loky')(tasks)
    else:
      for task in tasks:
        task()


# --- 3 & 4. Combined AUC plots ---
def plot_combined_aucs(
    df: pd.DataFrame,
    base_events: List[str],
    modes: List[str],
    figure_dir: Path,
    parallelize: bool = False,
    n_jobs: int = -1
) -> None:
  """
  Creates box+swarm plots for combined AUCs.
  """
  roles = ["predator", "prey"]
  tasks = []

  for mode in modes:
    cols_to_melt = [f"{ev.split('_')[0]}_vs_{mode}" for ev in base_events]

    actual_cols_to_melt = [col for col in cols_to_melt if col in df.columns]
    if not actual_cols_to_melt:
      print(f"Warning: No AUC columns found for mode 'vs_{mode}' with base events {base_events}. Skipping plot.")
      continue

    missing_auc_cols = set(cols_to_melt) - set(actual_cols_to_melt)
    if missing_auc_cols:
      print(
        f"Warning: Missing AUC columns for mode 'vs_{mode}': {missing_auc_cols}. Proceeding with available columns.")

    melt = df.melt(
      id_vars=["agent", "role", "source"],  # Ensure these id_vars exist
      value_vars=actual_cols_to_melt,
      var_name="event_vs",
      value_name="auc"
    ).dropna(subset=["auc"])

    if melt.empty:
      print(f"No data to plot for AUCs vs {mode} after melting and dropping NaNs.")
      continue

    melt["event"] = melt["event_vs"].str.replace(f"_vs_{mode}", "")

    for role in roles:
      out_dir = figure_dir / "combined_auc" / mode
      tasks.append(delayed(plot_box_swarm)(
        melt, "event", "auc", None, role,
        out_dir,
        title=f"ROC AUC vs {mode.capitalize()}"
      ))

  if tasks:
    print(f"Plotting combined AUCs with {'parallel' if parallelize else 'sequential'} execution...")
    if parallelize:
      Parallel(n_jobs=n_jobs, backend='loky')(tasks)
    else:
      for task in tasks:
        task()


# --- Data Loading Function (Improved Robustness) ---
def load_neural_summary(summary_dir: Path) -> pd.DataFrame:
  """Concatenate all '*_event_stats_and_decode.pkl' into one DataFrame."""
  if not summary_dir.is_dir():
    print(f"Error: Summary directory does not exist or is not a directory: {summary_dir}")
    return pd.DataFrame()

  dfs = []
  pickle_files = list(summary_dir.glob("*_event_stats_and_decode.pkl"))

  if not pickle_files:
    print(f"Warning: No '*_event_stats_and_decode.pkl' files found in {summary_dir}")
    return pd.DataFrame()

  for p in pickle_files:
    try:
      df_loaded = pd.read_pickle(p)
      dfs.append(df_loaded)
    except Exception as e:
      print(f"Error loading pickle file {p}: {e}. Skipping.")

  if not dfs:
    print(f"Warning: No data successfully loaded from pickle files in {summary_dir}")
    return pd.DataFrame()

  return pd.concat(dfs, ignore_index=True)




# --- Main Execution Block ---
if __name__ == "__main__":
  # --- Argument Parsing ---
  ap = argparse.ArgumentParser(description="Generate plots from neural summary statistics.")
  ap.add_argument('--summary_input_dir', type=str, default='../../results/mix_RF3/pipeline_summaries',
                  help="Path to the directory containing '*_event_stats_and_decode.pkl' summary files.")
  ap.add_argument('--figures_output_dir', type=str, default="./figures_neural_output",
                  help="Path to the directory where output figures will be saved.")
  ap.add_argument('--parallel', action='store_true', help="Enable parallel processing for plotting.")
  ap.add_argument('--no_parallel', dest='parallel', action='store_false', help="Disable parallel processing.")
  ap.set_defaults(parallel=True)
  ap.add_argument('--n_jobs', type=int, default=-1, help="Number of jobs for parallel processing (-1 for all cores).")

  args = ap.parse_args()

  neural_summary_dir = Path(args.summary_input_dir)
  figs_dir = Path(args.figures_output_dir)
  figs_dir.mkdir(parents=True, exist_ok=True)

  ENABLE_PARALLELISM = args.parallel
  N_JOBS = args.n_jobs
  # --- End Argument Parsing ---

  print(f"Loading neural summary data from: {neural_summary_dir}")
  summary_df = load_neural_summary(neural_summary_dir)

  if summary_df.empty:
    print(f"No data loaded from {neural_summary_dir}. Exiting.")
    exit()  # Exit if no data is loaded

  print(f"Data loaded. Shape: {summary_df.shape}. Columns: {summary_df.columns.tolist()}")

  # Add derived columns (with checks)
  if 'top_neu_cor_pred_score' in summary_df.columns:
    summary_df['top_neu_cor_dist_pred_score'] = summary_df['top_neu_cor_pred_score'].abs()
  else:
    print("Warning: 'top_neu_cor_pred_score' column not found. 'top_neu_cor_dist_pred_score' will not be created.")
    summary_df['top_neu_cor_dist_pred_score'] = np.nan

  if 'top_neu_cor_prey_score' in summary_df.columns:
    summary_df['top_neu_cor_dist_prey_score'] = summary_df['top_neu_cor_prey_score'].abs()
  else:
    print("Warning: 'top_neu_cor_prey_score' column not found. 'top_neu_cor_dist_prey_score' will not be created.")
    summary_df['top_neu_cor_dist_prey_score'] = np.nan

  # Define all event variants for fraction calculations
  events_all_variants = [
    "apple_cooperation_events", "distraction_events_helper",
    "distraction_events_beneficiary", "fence_events_helper",
    "fence_events_beneficiary", "none"
  ]

  # Define base events for combined AUC plots
  base_events_for_auc = [
    "apple_cooperation_events", "distraction_events", "fence_events"
  ]

  # Define primary metrics for boxplots
  # Note: "_vs_event" was changed to "_vs_targetevent" in some conventions, check your column names
  neural_metrics = [
    "apple_vs_none", "distraction_vs_none", "fence_vs_none",
    "apple_vs_event", "distraction_vs_event", "fence_vs_event",  # Or "_vs_targetevent"
    "apple_vs_rest", "distraction_vs_rest", "fence_vs_rest",
    "lifespan_total", "total_reward",  # Added total_reward for consistency
    "top_neu_cor_dist_pred_score", "top_neu_cor_dist_prey_score"
  ]
  for event in events_all_variants:  # Ensure these event-specific columns exist if used as metrics
    neural_metrics.append(f"{event}_lifespan")
    neural_metrics.append(f"{event}_reward")

  existing_neural_metrics = [m for m in neural_metrics if m in summary_df.columns]
  missing_metrics = list(set(neural_metrics) - set(existing_neural_metrics))  # Use set for unique missing
  if missing_metrics:
    print(
      f"Warning: The following requested metrics are not in the DataFrame and will be skipped for direct plotting: {sorted(missing_metrics)}")

  # --- Plotting Calls ---

  # 1. Plot metrics by 'source'
  print("\n--- Plotting metrics by Source ---")
  if 'source' in summary_df.columns:
    plot_metrics_by_category_and_role(
      summary_df, existing_neural_metrics, 'source', figs_dir, "Source",
      parallelize=ENABLE_PARALLELISM, n_jobs=N_JOBS
    )
  else:
    print("Warning: 'source' column not found. Skipping plots by Source.")

  # 2. Plot metrics by 'source_type'
  print("\n--- Plotting metrics by Source Type ---")
  if 'source' in summary_df.columns:
    summary_df_copy_for_source_type = summary_df.copy()
    summary_df_copy_for_source_type['source_type'] = summary_df_copy_for_source_type['source'].str.replace(
      r'_pre[y|d]_\d+$', '', regex=True)  # Ensure this regex matches your 'source' format
    plot_metrics_by_category_and_role(
      summary_df_copy_for_source_type, existing_neural_metrics, 'source_type', figs_dir, "Source Type",
      parallelize=ENABLE_PARALLELISM, n_jobs=N_JOBS
    )
  else:
    print("Warning: 'source' column not found. Cannot derive 'source_type'. Skipping plots by Source Type.")

  # 3. Fraction of each event to lifespan/reward
  print("\n--- Plotting Event Fractions ---")
  plot_event_fractions(
    summary_df, events_all_variants, figs_dir,
    parallelize=ENABLE_PARALLELISM, n_jobs=N_JOBS
  )

  # 4. Combined AUC plots
  print("\n--- Plotting Combined AUCs ---")
  auc_modes_to_plot = ["none", "rest", "event"]  # Or "targetevent"
  plot_combined_aucs(
    summary_df, base_events_for_auc, auc_modes_to_plot, figs_dir,
    parallelize=ENABLE_PARALLELISM, n_jobs=N_JOBS
  )

  print("\nAll plotting tasks completed.")