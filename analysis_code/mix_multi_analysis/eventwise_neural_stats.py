import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Any, Optional
from joblib import Parallel, delayed
import numpy as np  # Needed for np.nan


# --- Generic Plotting Function (Re-introduced & Corrected) ---
def plot_box_swarm(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str],  # Can be None
    role: str,
    out_dir: Path,
    title: str,
    palette: Any = "Set2"  # Use Any for flexibility, as it can be string or list
) -> None:
  """Generic box+swarm plot of y vs x, with optional hue, for one role."""
  # Use .copy() to avoid SettingWithCopyWarning if 'sub' is modified later
  sub = df[df["role"] == role].copy()

  out_dir.mkdir(parents=True, exist_ok=True)

  # Ensure 'x' column exists before sorting unique values
  if x not in sub.columns:
    print(f"Warning: Column '{x}' not found for role '{role}'. Skipping plot for {title}.")
    return

  # Drop NaNs in relevant columns to avoid issues with plotting functions
  # Especially important for 'y' variable
  sub = sub.dropna(subset=[x, y])
  if hue and hue in sub.columns:
    sub = sub.dropna(subset=[hue])
  elif hue and hue not in sub.columns:
    print(f"Warning: Hue column '{hue}' not found. Plotting without hue.")
    hue = None  # Fallback to no hue if column doesn't exist

  # Determine the category order once
  if not sub.empty:  # Only sort if there's data
    order = sorted(sub[x].unique())
  else:
    print(f"No data to plot for role '{role}' and metric '{y}' for category '{x}'. Skipping plot.")
    return

  # Create a new figure for each plot (crucial for parallelism)
  fig, ax = plt.subplots(
    figsize=(max(6, len(sub[x].unique()) * 0.25), 5)  # Dynamic width based on number of categories
  )

  # Determine palette for boxplot and swarmplot
  # If hue is None, seaborn expects a single color or named palette for the overall plot.
  # If palette is a list/tuple, use the first color for the single boxplot.
  boxplot_palette = palette
  if hue is None and isinstance(palette, (list, tuple)):
    # If no hue, and palette is a list, use the first color for the single boxplot
    boxplot_palette = palette[0] if palette else "gray"  # Fallback to a default color if list is empty

  # Plot boxplot
  sns.boxplot(x=x, y=y, hue=hue, data=sub, ax=ax, palette=boxplot_palette, order=order)

  # Plot swarmplot
  # If hue is present, swarmplot should also use the palette to match boxplot colors.
  # If hue is None, swarmplot should use a single fixed color.
  if hue:
    sns.swarmplot(x=x, y=y, hue=hue, data=sub, palette=palette, size=2, ax=ax, order=order, dodge=True)
  else:
    sns.swarmplot(x=x, y=y, data=sub, color=".25", size=2, ax=ax, order=order)

  # Remove legend consistently after both plots, if it exists.
  if ax.get_legend() is not None:
    ax.get_legend().remove()

  ax.set_title(f"{title} ({role.capitalize()})")
  ax.set_xlabel(x.replace("_", " ").capitalize())
  ax.set_ylabel(y.replace("_", " ").capitalize())
  # Rotate x-axis labels for readability if they are long
  ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
  ax.grid(axis='y', linestyle='--', alpha=0.7)
  plt.tight_layout()

  # Save the figure
  # Ensure a valid filename, replacing characters that might cause issues
  filename = f"{y}_by_{x}.png".replace("/", "_").replace("\\", "_")
  fig.savefig(out_dir / filename, dpi=150)

  plt.close(fig)  # Crucial for freeing memory in parallel processing


# --- 1. Fraction of each event to lifespan/reward (New Function) ---
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
    print(f"Warning: Missing one or more lifespan columns for fraction calculation. Skipping lifespan fraction plot.")
    print(f"Expected: {cols_to_check_life}, Found: {df.columns.tolist()}")
  else:
    frac_life = df[cols_to_check_life].copy()
    for v in lifespan_vars:
      # Handle division by zero or NaN lifespan_total
      frac_life[v] = frac_life[v] / frac_life["lifespan_total"].replace(0, np.nan)
    m1 = frac_life.melt(
      id_vars=["agent", "role"],
      value_vars=lifespan_vars,
      var_name="event",
      value_name="fraction"
    ).dropna(subset=["fraction"])
    m1["event"] = m1["event"].str.replace("_lifespan", "")

    for role in roles:
      out_dir = figure_dir / "fractions" / "lifespan" / role
      tasks.append(delayed(plot_box_swarm)(
        m1, "event", "fraction", None, role,
        out_dir,
        title="Fraction of lifespan"
      ))

  # 2) reward fractions
  reward_vars = [f"{ev}_reward" for ev in events]
  cols_to_check_reward = ["agent", "role"] + reward_vars
  if not all(col in df.columns for col in cols_to_check_reward):
    print(f"Warning: Missing one or more reward columns for fraction calculation. Skipping reward fraction plot.")
    print(f"Expected: {cols_to_check_reward}, Found: {df.columns.tolist()}")
  else:
    frac_rew = df[cols_to_check_reward].copy()
    frac_rew["total"] = frac_rew[reward_vars].sum(axis=1)
    for v in reward_vars:
      # Handle division by zero or NaN total reward
      frac_rew[v] = frac_rew[v] / frac_rew["total"].replace(0, np.nan)
    m2 = frac_rew.melt(
      id_vars=["agent", "role"],
      value_vars=reward_vars,
      var_name="event",
      value_name="fraction"
    ).dropna(subset=["fraction"])
    m2["event"] = m2["event"].str.replace("_reward", "")

    for role in roles:
      out_dir = figure_dir / "fractions" / "reward" / role
      tasks.append(delayed(plot_box_swarm)(
        m2, "event", "fraction", None, role,
        out_dir,
        title="Fraction of reward"
      ))

  if tasks:
    print(f"Plotting event fractions with {'parallel' if parallelize else 'sequential'} execution...")
    if parallelize:
      Parallel(n_jobs=n_jobs, backend='loky')(tasks)
    else:
      for task in tasks:
        task()


# --- 2. Swarmplot added to all current plots (Refactoring your existing functions) ---
# We will use the new plot_box_swarm function to replace the pandas boxplot
def plot_metrics_by_category_and_role(
    df: pd.DataFrame,
    metrics: List[str],
    category_col: str,  # 'source' or 'source_type'
    base_figure_dir: Path,
    plot_title_suffix: str,
    parallelize: bool = False,
    n_jobs: int = -1
) -> None:
  """
  Generic function to plot metrics by a given category column ('source' or 'source_type').
  Uses plot_box_swarm internally.
  """
  roles = ['predator', 'prey']
  tasks = []

  # Ensure category_col exists
  if category_col not in df.columns:
    print(f"Error: Category column '{category_col}' not found in DataFrame. Skipping plots.")
    return

  for role in roles:
    for m in metrics:
      out_dir = base_figure_dir / role / plot_title_suffix.lower().replace(" ",
                                                                           "_")  # e.g., 'by_source', 'by_source_type'
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


# --- 3 & 4. Combined AUC plots (New Function for "vs_event", "vs_none", "vs_rest") ---
def plot_combined_aucs(
    df: pd.DataFrame,
    base_events: List[str],  # e.g., ["apple", "distraction", "fence"]
    modes: List[str],  # e.g., ["none", "rest", "event"]
    figure_dir: Path,
    parallelize: bool = False,
    n_jobs: int = -1
) -> None:
  """
  Creates box+swarm plots for combined AUCs (e.g., apple_vs_mode, distraction_vs_mode).
  One plot per mode (none, rest, event) for each role, with event types as categories.
  """
  roles = ["predator", "prey"]
  tasks = []

  for mode in modes:
    # Construct actual column names from the DataFrame
    cols_to_melt = [f"{ev.split('_')[0]}_vs_{mode}" for ev in
                    base_events]  # Assuming base_events might contain full names like "apple_cooperation_events"

    # Verify columns exist before melting
    if not all(col in df.columns for col in cols_to_melt):
      print(f"Warning: Missing one or more AUC columns for mode 'vs_{mode}'. Skipping plot.")
      print(f"Expected: {cols_to_melt}, Found: {df.columns.tolist()}")
      continue

    melt = df.melt(
      id_vars=["agent", "role", "source"],
      value_vars=cols_to_melt,
      var_name="event_vs",
      value_name="auc"
    ).dropna(subset=["auc"])  # Drop rows where AUC is NaN

    # Extract base event name (e.g., "apple" from "apple_vs_none")
    melt["event"] = melt["event_vs"].str.replace(f"_vs_{mode}", "")

    for role in roles:
      out_dir = figure_dir / "combined_auc" / mode / role
      tasks.append(delayed(plot_box_swarm)(
        melt, "event", "auc", None, role,  # No 'hue' for event types
        out_dir,
        title=f"ROC AUC vs {mode}"
      ))

  if tasks:
    print(f"Plotting combined AUCs with {'parallel' if parallelize else 'sequential'} execution...")
    if parallelize:
      Parallel(n_jobs=n_jobs, backend='loky')(tasks)
    else:
      for task in tasks:
        task()

def load_neural_summary(summary_dir: Path) -> pd.DataFrame:
  """Concatenate all <session>_summary.pkl into one DataFrame."""
  dfs = []
  for p in summary_dir.glob("*_event_stats_and_decode.pkl"):
    df = pd.read_pickle(p)
    dfs.append(df)
  return pd.concat(dfs, ignore_index=True)


# --- Main Execution Block ---
if __name__ == "__main__":
  neural_summary_dir = Path("/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/analysis_results_event_stats/")
  figs_dir = Path("./figures_neural")
  figs_dir.mkdir(exist_ok=True)  # Ensure figures_neural directory exists

  # --- Parallelism Control ---
  ENABLE_PARALLELISM = True  # Set to False to run sequentially
  N_JOBS = -1  # -1 means use all available CPU cores. Adjust as needed.
  # --- End Parallelism Control ---

  print("Loading neural summary data...")
  summary_df = load_neural_summary(neural_summary_dir)
  print(f"Data loaded. Shape: {summary_df.shape}")

  # Add derived columns (with checks)
  if 'top_neu_cor_pred_score' in summary_df.columns:
    summary_df['top_neu_cor_dist_pred_score'] = summary_df['top_neu_cor_pred_score'].abs()
  else:
    print("Warning: 'top_neu_cor_pred_score' column not found. 'top_neu_cor_dist_pred_score' will not be created.")
    summary_df['top_neu_cor_dist_pred_score'] = np.nan  # Add as NaN to avoid KeyError later if metric is used

  if 'top_neu_cor_prey_score' in summary_df.columns:
    summary_df['top_neu_cor_dist_prey_score'] = summary_df['top_neu_cor_prey_score'].abs()
  else:
    print("Warning: 'top_neu_cor_prey_score' column not found. 'top_neu_cor_dist_prey_score' will not be created.")
    summary_df['top_neu_cor_dist_prey_score'] = np.nan  # Add as NaN

  # Define all event variants for fraction calculations
  events_all_variants = [
    "apple_cooperation_events",
    "distraction_events_helper",
    "distraction_events_beneficiary",
    "fence_events_helper",
    "fence_events_beneficiary",
    "none"
  ]

  # Define base events for combined AUC plots
  # These should map to the prefixes of your AUC columns (e.g., "apple_vs_none")
  base_events_for_auc = [
    "apple_cooperation_events",  # plot_combined_aucs will split to "apple"
    "distraction_events",  # assumes this is base, not helper/beneficiary
    "fence_events"  # assumes this is base
  ]

  # Define primary metrics for boxplots
  neural_metrics = [
    # leave-one-out AUCs (assuming these are actual column names)
    "apple_vs_none", "distraction_vs_none", "fence_vs_none",
    "apple_vs_event", "distraction_vs_event", "fence_vs_event",
    "apple_vs_rest", "distraction_vs_rest", "fence_vs_rest",  # Added vs_rest for completeness
    # other metrics
    "lifespan_total",
    "top_neu_cor_dist_pred_score",
    "top_neu_cor_dist_prey_score"
  ]
  # Add lifespan and reward metrics for all event variants
  for event in events_all_variants:
    neural_metrics.append(f"{event}_lifespan")
    neural_metrics.append(f"{event}_reward")

  # Filter neural_metrics to only include columns that actually exist in summary_df
  # This prevents errors if some metrics are missing from the loaded data
  existing_neural_metrics = [m for m in neural_metrics if m in summary_df.columns]
  missing_metrics = [m for m in neural_metrics if m not in summary_df.columns]
  if missing_metrics:
    print(f"Warning: The following requested metrics are not in the DataFrame and will be skipped: {missing_metrics}")

  # --- Plotting Calls ---

  # 1. Plot per-agent (grouped by 'source') with swarmplots
  print("\n--- Plotting metrics by Source ---")
  plot_metrics_by_category_and_role(
    summary_df, existing_neural_metrics, 'source', figs_dir, "Source",
    parallelize=ENABLE_PARALLELISM, n_jobs=N_JOBS
  )

  # 2. Plot per-source-type (strip off _agentID) with swarmplots
  print("\n--- Plotting metrics by Source Type ---")
  summary_df_copy_for_source_type = summary_df.copy()  # Operate on a copy to add 'source_type'
  summary_df_copy_for_source_type['source_type'] = summary_df_copy_for_source_type['source'].str.replace(
    r'_pre[y|d]_\d+$', '', regex=True)

  plot_metrics_by_category_and_role(
    summary_df_copy_for_source_type, existing_neural_metrics, 'source_type', figs_dir, "Source Type",
    parallelize=ENABLE_PARALLELISM, n_jobs=N_JOBS
  )

  # 3. Fraction of each event to lifespan/reward (new requirement)
  print("\n--- Plotting Event Fractions ---")
  plot_event_fractions(
    summary_df, events_all_variants, figs_dir,
    parallelize=ENABLE_PARALLELISM, n_jobs=N_JOBS
  )

  # 4. Combined AUC plots for "vs_none", "vs_rest", "vs_event" (new requirement)
  print("\n--- Plotting Combined AUCs ---")
  auc_modes_to_plot = ["none", "rest", "event"]  # Suffixes in column names (e.g., apple_vs_none)
  plot_combined_aucs(
    summary_df, base_events_for_auc, auc_modes_to_plot, figs_dir,
    parallelize=ENABLE_PARALLELISM, n_jobs=N_JOBS
  )

  print("\nAll plotting tasks completed.")