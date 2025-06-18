import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.ticker as mticker # For better tick formatting

# Helper function to extract checkpoint number from its name
# This function is duplicated from match_ckpts_archived.py for self-containment
def get_ckpt_from_name(ckpt_name_str: str) -> int:
    """Extracts the numerical ID from a checkpoint name like 'ckpt-123'."""
    try:
        # Handles names like "ckpt-123" or "checkpoint_name/ckpt-123"
        name_part = ckpt_name_str.split('/')[-1]
        return int(name_part.split('-')[-1])
    except Exception:
        # Return -1 or another indicator for invalid names, which will be filtered out
        return -1

def format_y_axis_episodes(y, pos):
    """
    Formats the y-axis tick labels for episode counts.
    Converts numbers like 1,000,000 to 1M, 1,000 to 1K.
    """
    if y >= 1_000_000:
        return f'{y / 1_000_000:.1f}M'
    elif y >= 1_000:
        return f'{y / 1_000:.0f}K'
    return f'{int(y)}'

def main():
    """
    Plots checkpoint training times, episode counts, and other metrics
    from matched CSV files across multiple experiment directories.
    """
    parser = argparse.ArgumentParser(
        description="Plot checkpoint training times and episode counts from matched CSVs."
    )
    parser.add_argument(
        "--exp_dirs",
        type=Path,
        nargs="+",  # Allows passing multiple directory paths
        default=[
            Path('results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__open_2025-06-12_15:49:53.201394'),
            Path('results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__orchard_2025-06-15_10:11:40.382778'),
            Path('results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__alley_hunt_2025-06-15_10:11:40.406930'),
            Path('results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__random_forest_2025-06-12_15:49:53.295564'),
        ],
        help="List of experiment directories. Each directory should contain "
             "a 'csv_logs/checkpoint_episodes.csv' file."
    )
    args = parser.parse_args()

    all_combined_dfs = [] # List to store DataFrames from all experiments

    # Iterate through each provided experiment directory
    for exp_dir in args.exp_dirs:
        matched_csv_path = exp_dir / "csv_logs" / "checkpoint_episodes.csv"

        # Check if the expected CSV file exists
        if not matched_csv_path.is_file():
            print(f"Warning: Matched CSV not found at {matched_csv_path}. Skipping directory.")
            continue

        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(matched_csv_path)

            # Check if the DataFrame is empty after reading
            if df.empty:
                print(f"Warning: Matched CSV at {matched_csv_path} is empty. Skipping directory.")
                continue

            # Ensure essential columns exist
            required_columns = ["ckpt", "timestamp", "actor_episodes", "learner_time_elapsed", "learner_steps"]
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: CSV at {matched_csv_path} is missing one or more required columns ({required_columns}). Skipping directory.")
                continue

            # Add an 'exp_name' column to identify data from different experiments
            df["exp_name"] = exp_dir.name

            # Fill NaN values for ckpt=1 (assuming these are initial values)
            row_ckpt_1_mask = df["ckpt"] == 1
            if row_ckpt_1_mask.any():
                df.loc[row_ckpt_1_mask, :] = \
                    df.loc[row_ckpt_1_mask, :].fillna(0)

            # Calculate training time from timestamp if not already present
            if 'timestamp' in df.columns:
                df['training_time'] = df['timestamp'] - df['timestamp'].min()
            else:
                print(f"Warning: 'timestamp' column not found in {matched_csv_path}. Cannot calculate training_time. Skipping.")
                continue


            all_combined_dfs.append(df) # Add processed DataFrame to the list

        except Exception as e:
            print(f"Error processing {matched_csv_path}: {e}. Skipping directory.")
            continue

    # If no valid data was found across all directories, exit
    if not all_combined_dfs:
        print("No valid data found to plot from any of the specified directories. Exiting.")
        return

    # Concatenate all DataFrames into a single DataFrame for plotting
    final_df = pd.concat(all_combined_dfs, ignore_index=True)
    final_df['source'] = 'counter'  # Add a source column to indicate the data source (you might want to make this dynamic)

    # Create the output directory for plots
    output_dir = Path("figure_ckpt_check")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Plotting Setup ---
    # Create a figure with five subplots stacked vertically, sharing the x-axis for the first two
    fig, axes = plt.subplots(5, 1, figsize=(10, 20), sharex=False) # Changed figsize for better readability

    # Get unique combinations of experiment name and source (counter/learner)
    unique_combinations = final_df[['exp_name', 'source']].drop_duplicates().to_records(index=False)

    # Use a colormap to get distinct colors for each unique combination
    colors = plt.cm.get_cmap('tab10', len(unique_combinations))
    # Create a mapping from combination (tuple) to color
    color_map = {tuple(combo): colors(i) for i, combo in enumerate(unique_combinations)}

    plot_configs = [
        {"y_col": "learner_time_elapsed", "x_col": "training_time", "x_unit_conversion": 3600, "x_unit_label": "hours",
         "y_unit_conversion": 3600, 'y_unit_label': 'hours', "title": "Training Time vs. Learner Time Elapsed",
         "ylabel": "Learner Time Elapsed", "xlabel": "Training Time", },
        {"y_col": "actor_episodes", "x_col": "learner_time_elapsed", "x_unit_conversion": 3600, "x_unit_label": "hours",
         "title": "Learner Time Elapsed vs. Number of Episodes", "ylabel": "Number of Episodes",
         "xlabel": "Learner Time Elapsed", },
        {"y_col": "training_time", "x_col": "ckpt", "title": "Checkpoint Number vs. Training Time",
         "ylabel": "Training Time (seconds)", "xlabel": "Checkpoint Number"},
        {"y_col": "actor_episodes", "x_col": "ckpt", "title": "Checkpoint Number vs. Number of Episodes",
         "ylabel": "Number of Episodes", "xlabel": "Checkpoint Number"},
        {"y_col": "actor_episodes", "x_col": "training_time", "x_unit_conversion": 3600, "x_unit_label": "hours",
         "title": "Training Time vs. Number of Episodes", "ylabel": "Number of Episodes", "xlabel": "Training Time"},
    ]

    for i, config in enumerate(plot_configs):
        ax = axes[i]
        x_col = config["x_col"]
        y_col = config["y_col"]
        x_unit_conversion = config.get("x_unit_conversion", 1)
        x_unit_label = config.get("x_unit_label", "")
        y_unit_conversion = config.get("y_unit_conversion", 1)
        y_unit_label = config.get("y_unit_label", "")

        for exp_name, source in unique_combinations:
            subset_df = final_df[(final_df["exp_name"] == exp_name) & (final_df["source"] == source)].copy()
            label = f"{exp_name} ({source})"
            color = color_map[(exp_name, source)]

            # Sort by the x-axis column to ensure proper line plotting
            subset_df_sorted = subset_df.sort_values(by=x_col)

            ax.plot(
                subset_df_sorted[x_col] / x_unit_conversion, subset_df_sorted[y_col] / y_unit_conversion,
                label=label, color=color, marker='o', linestyle='-', markersize=4, linewidth=1
            )

            # Annotation for "Checkpoint Number vs. Number of Episodes" plot
            if config["title"] == "Checkpoint Number vs. Number of Episodes" and not subset_df_sorted.empty:
                last_point = subset_df_sorted.iloc[-1]
                last_ckpt = last_point["ckpt"]
                last_episodes = last_point["actor_episodes"]

                if last_episodes >= 1_000_000:
                    ep_text = f"{last_episodes / 1_000_000:.1f}M"
                elif last_episodes >= 1_000:
                    ep_text = f"{last_episodes / 1_000:.0f}K"
                else:
                    ep_text = f"{int(last_episodes)}"

                ax.annotate(
                    ep_text,
                    xy=(last_ckpt, last_episodes),
                    xytext=(8, 0),
                    textcoords="offset points",
                    ha='left',
                    va='center',
                    fontsize=9,
                    color=color,
                    fontweight='bold'
                )

        ax.set_title(config["title"], fontsize=14) # Reduced title font size slightly
        ax.set_xlabel(f"{config['xlabel']} ({x_unit_label})" if x_unit_label else config['xlabel'], fontsize=12)
        ax.set_ylabel(f"{config['ylabel']} ({y_unit_label})" if y_unit_label else config['ylabel'], fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=0) # Removed rotation for general plots, keep if needed

        # Apply y-axis formatter for episode plots
        if y_col == "actor_episodes":
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_y_axis_episodes))


    # --- Shared Legend ---
    # Collect handles and labels from the first subplot to create a single legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Determine number of columns for the legend dynamically
    ncol = min(int(np.ceil(len(labels)/3)), 4) # Max 4 columns, or fewer if not many labels
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02), # Position legend further below the plot area
        ncol=ncol,
        title="Experiment & Source",
        fontsize=10,
        frameon=False # Optional: remove legend frame
    )

    # Adjust layout to prevent titles/labels from overlapping and make space for the legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.98]) # Adjusted rect to make space for legend at bottom

    # Save the combined plot to the specified directory
    output_plot_path = output_dir / "ckpt_metrics_combined_plot.png" # More descriptive filename
    plt.savefig(output_plot_path, bbox_inches='tight', dpi=300) # Increased DPI for higher quality
    print(f"Plots saved successfully to: {output_plot_path}")

if __name__ == "__main__":
    main()