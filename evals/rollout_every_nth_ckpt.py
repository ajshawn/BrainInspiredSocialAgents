import argparse
import pandas as pd
from pathlib import Path
import subprocess
import sys
import numpy as np
import os
from typing import List, Dict, Tuple

# --- Configuration ---
# Define the explicit mapping from experiment directory full paths
# to their desired output root names. This replicates your bash script's logic.
EXPERIMENT_DIR_MAPPING = {
  '/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__random_forest_2025-06-12_15:49:53.295564': 'RandomForestR0_20250612_2v4_rollouts',
  '/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__open_2025-06-12_15:49:53.201394': 'OpenR0_20250612_2v4_rollouts',
  '/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__orchard_2025-06-15_10:11:40.382778': 'OrchardR0_20250615_10:11:40.382778_2v4_rollouts',
  '/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__alley_hunt_2025-06-15_10:11:40.406930': 'AlleyHuntR0_20250615_10:11:40.406930_2v4_rollouts',
}

# The base path where all your experiment results are located
# This is crucial for resolving the full paths from the provided directory names.
BASE_RESULTS_PATH = Path("results/")

# Default list of target episode counts
DEFAULT_EPISODE_TARGETS = [
  0, 500, 1000, 2000, 3000, 5000, 7500, 10000, 12500, 15000, 20000,
  30000, 50000, 75000, 100000, 150000, 200000
]


# --- Helper Functions ---
def find_closest_ckpt(df: pd.DataFrame, target_episodes: int) -> tuple[int]:
  """
  Finds the checkpoint number (ckpt) in the DataFrame whose 'actor_episodes'
  value is closest to the given target_episodes.

  Args:
      df: A pandas DataFrame containing 'ckpt' and 'actor_episodes' columns.
      target_episodes: The target number of episodes to find the closest checkpoint for.

  Returns:
      The checkpoint number (int) closest to the target_episodes, or None if no valid
      checkpoint data is found.
  """
  if df.empty:
    return None, None, None  # No data to compare against

  # Handle the special case for target_episodes = 0.
  # We assume ckpt-1 corresponds to the start (0 episodes).
  if target_episodes == 0:
    ckpt_1_row = df[df['ckpt'] == 1]
    if not ckpt_1_row.empty:
      # If ckpt-1 exists, it's our target for 0 episodes
      return 1, 0, 0
    else:
      # If ckpt-1 doesn't exist, we can't map 0 episodes reliably
      return None, None, None

  # Ensure 'actor_episodes' column is numeric, coercing errors to NaN
  # Then, drop rows where 'actor_episodes' is NaN as they can't be compared numerically.
  df_valid_episodes = df.copy()
  df_valid_episodes['actor_episodes_numeric'] = pd.to_numeric(df_valid_episodes['actor_episodes'], errors='coerce')
  df_valid_episodes = df_valid_episodes.dropna(subset=['actor_episodes_numeric'])

  if df_valid_episodes.empty:
    return None, None, None  # No valid episode data to compare after filtering

  # Calculate absolute difference between actor_episodes and the target
  df_valid_episodes['diff'] = abs(df_valid_episodes['actor_episodes_numeric'] - target_episodes)

  # Find the row(s) with the minimum difference.
  # If multiple checkpoints have the same minimum difference,
  # we select the one with the smallest checkpoint number for consistency.
  min_diff = df_valid_episodes['diff'].min()
  closest_rows = df_valid_episodes[df_valid_episodes['diff'] == min_diff]

  # Sort by 'ckpt' to get the earliest checkpoint in case of a tie
  closest_ckpt = closest_rows.sort_values(by='ckpt').iloc[0]['ckpt']

  return int(closest_ckpt), target_episodes, closest_rows.iloc[0]['actor_episodes_numeric']


def get_experiment_paths_and_outputs(arg_exp_dirs: List[Path]) -> Dict[Path, Path]:
  """
  Maps provided (potentially partial) experiment directory paths to their full absolute
  paths and their corresponding output roots, based on the predefined mapping.

  Args:
      arg_exp_dirs: List of Path objects for experiment directories. These can be
                    just the directory names (e.g., 'PopArtIMPALA_..._random_forest_...')

  Returns:
      A dictionary mapping full absolute experiment Path objects to their
      corresponding full absolute output root Path objects.
  """
  resolved_mapping = {}
  for arg_exp_dir in arg_exp_dirs:
    found_match = False
    for full_exp_dir_str, output_root_name in EXPERIMENT_DIR_MAPPING.items():
      full_exp_dir_path = Path(full_exp_dir_str)
      # Check if the provided argument's name matches the name of a known full path
      print('Checking:', arg_exp_dir.name, '\nagainst', full_exp_dir_path.name)
      if arg_exp_dir.name == full_exp_dir_path.name:
        # Reconstruct the expected full path for the argument using BASE_RESULTS_PATH
        reconstructed_full_arg_path = BASE_RESULTS_PATH / arg_exp_dir.name

        # Verify that the reconstructed path matches the full path in our mapping
        if Path(os.path.abspath(reconstructed_full_arg_path)) == full_exp_dir_path:
          # Map the full experiment directory path to its full output root path
          resolved_mapping[full_exp_dir_path] = BASE_RESULTS_PATH / output_root_name
          found_match = True
          break
    if not found_match:
      print(f"Warning: No predefined output root mapping found for '{arg_exp_dir.name}'. Skipping.", file=sys.stderr)
  return resolved_mapping


# --- Main Script Logic ---
def main():
  parser = argparse.ArgumentParser(
    description="Orchestrate running evaluate_rollout_every_nth_ckpt.py for selected episode counts."
  )
  parser.add_argument(
    "--exp_dirs",
    type=Path,
    nargs="+",
    default=[
      Path('PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__open_2025-06-12_15:49:53.201394'),
      Path('PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__orchard_2025-06-15_10:11:40.382778'),
      Path('PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__alley_hunt_2025-06-15_10:11:40.406930'),
      Path('PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__random_forest_2025-06-12_15:49:53.295564'),
    ],
    help="List of experiment directories (provide just the last folder name). "
         "The script will use a predefined mapping to find full paths and output roots."
  )
  parser.add_argument(
    "--episode_targets",
    type=int,
    nargs="+",
    default=DEFAULT_EPISODE_TARGETS,
    help="List of target episode counts to select checkpoints for. "
         "The closest available checkpoint will be chosen. Default includes 0 (for ckpt-1)."
  )
  parser.add_argument(
    "--eval_script_path",
    type=Path,
    default="/home/mikan/e/GitHub/social-agents-JAX/evals/evaluate_rollout_every_nth_ckpt.py",
    help="Path to the evaluate_rollout_every_nth_ckpt.py script."
  )
  parser.add_argument(
    "--max_workers_eval_script",
    type=int,
    default=90,  # A reasonable default for k_samples 16, adjust as needed
    help="Number of workers to pass to the inner evaluate_rollout_every_nth_ckpt.py script (--max_workers)."
  )
  parser.add_argument(
    "--k_samples_eval_script",
    type=int,
    default=16,
    help="Number of diverse combinations (k_samples) to pass to the inner evaluate_rollout_every_nth_ckpt.py script."
  )

  args = parser.parse_args()

  # Resolve full paths for experiment directories and their corresponding output roots
  exp_dir_to_out_root_map = get_experiment_paths_and_outputs(args.exp_dirs)

  if not exp_dir_to_out_root_map:
    print("No valid experiment directories found or mapped. Exiting.", file=sys.stderr)
    return

  # Check if the evaluation script exists and is executable
  if not args.eval_script_path.is_file():
    sys.exit(f"Error: Evaluation script not found at {args.eval_script_path}. Please provide a correct path.")
  if not os.access(args.eval_script_path, os.X_OK):
    print(
      f"Warning: Evaluation script {args.eval_script_path} is not executable. Attempting to run anyway, but consider adding execute permissions.",
      file=sys.stderr)

  # Main loop: Process each experiment directory sequentially
  for exp_dir, out_root in exp_dir_to_out_root_map.items():
    print(f"\n--- Processing Experiment: {exp_dir.name} ---")
    print(f"Full Experiment Path: {exp_dir}")
    print(f"Rollouts Output To:   {out_root}")

    # Construct the path to the checkpoint_episodes.csv file
    csv_path = exp_dir / "csv_logs" / "checkpoint_episodes.csv"
    if not csv_path.is_file():
      print(f"Warning: CSV log not found at {csv_path}. Skipping this experiment.", file=sys.stderr)
      continue

    try:
      df = pd.read_csv(csv_path)
      if df.empty:
        print(f"Warning: CSV at {csv_path} is empty. Skipping this experiment.", file=sys.stderr)
        continue

      # Verify essential columns are present
      if 'ckpt' not in df.columns or 'actor_episodes' not in df.columns:
        print(f"Warning: CSV at {csv_path} is missing 'ckpt' or 'actor_episodes' columns. Skipping this experiment.",
              file=sys.stderr)
        continue

    except Exception as e:
      print(f"Error reading CSV for {exp_dir.name}: {e}. Skipping this experiment.", file=sys.stderr)
      continue

    selected_ckpts = set()  # Use a set to store unique checkpoint numbers for this experiment

    print("\nDetermining closest checkpoints for target episodes:")
    for target_episodes in args.episode_targets:
      ckpt_num, _, matched_episodes = find_closest_ckpt(df, target_episodes)

      if ckpt_num is not None:
        selected_ckpts.add(ckpt_num)
        print(f"  Target: {target_episodes:<8} episodes -> Selected Checkpoint: {ckpt_num}, Matched Episodes: {matched_episodes}")
      else:
        print(
          f"  Warning: Could not find a suitable checkpoint for target episodes: {target_episodes} in {exp_dir.name}.",
          file=sys.stderr)

    if not selected_ckpts:
      print(f"No unique checkpoints selected for {exp_dir.name}. Skipping rollout execution.", file=sys.stderr)
      continue

    # Convert the set to a sorted list for consistent command line arguments
    sorted_selected_ckpts = sorted(list(selected_ckpts))
    print(f"\nFinal unique checkpoints to evaluate for {exp_dir.name}: {sorted_selected_ckpts}")

    # Ensure the output root directory for rollouts exists
    out_root.mkdir(parents=True, exist_ok=True)
    # Define the path for the log file (placed next to the output_root directory)
    log_file_path = out_root.parent / f"{out_root.name}.log"

    # Prepare the command to run evaluate_rollout_every_nth_ckpt.py
    # We use sys.executable to ensure the script runs with the same python interpreter
    # as this orchestrator script.
    cmd = [
            sys.executable,
            str(args.eval_script_path),
            "--experiment_dir", str(exp_dir),
            "--max_workers", str(args.max_workers_eval_script),
            "--out_root", str(out_root),
            "--k_samples", str(args.k_samples_eval_script),
            "--selected_checkpoints",  # This flag takes a list of integers
          ] + [str(c) for c in sorted_selected_ckpts]  # Add each checkpoint number as a separate argument

    print(f"\nExecuting command for {exp_dir.name}:")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output and errors will be logged to: {log_file_path}")

    try:
      # Use subprocess.run to execute the command.
      # stdout and stderr are redirected to the log file.
      # `text=True` decodes stdout/stderr as text.
      # `check=True` raises CalledProcessError if the command returns a non-zero exit code.
      with open(log_file_path, 'w') as log_file:
        process = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=True)
        print(f"Rollout for {exp_dir.name} completed successfully. (Return Code: {process.returncode})")
    except subprocess.CalledProcessError as e:
      print(f"Error during rollout for {exp_dir.name}: Command failed with exit code {e.returncode}.", file=sys.stderr)
      print(f"Please check the log file for details: {log_file_path}", file=sys.stderr)
      # print(f"Command output (if captured by error): {e.stdout}", file=sys.stderr) # Uncomment for debug
    except Exception as e:
      print(f"An unexpected error occurred during rollout for {exp_dir.name}: {e}", file=sys.stderr)
      print(f"Please check the log file for details: {log_file_path}", file=sys.stderr)

  print("\n--- All evaluations orchestrated. ---")


if __name__ == "__main__":
  main()