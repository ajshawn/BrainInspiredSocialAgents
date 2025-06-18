#!/bin/bash
# This script will mix and match checkpoint pairs for predator and prey,
# and run the evaluations in parallel with a maximum of 25 concurrent jobs.
# Please adjust the checkpoint list and index ranges as needed.

# Define the list of checkpoints.
checkpoints=(
  "results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962"
  "results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026"
#  "results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp2263"
  "results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357"
  "results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17:36:18.023323_ckp9651"
  "results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651"
  "results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441"
  "results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-01-11_11:10:52.115495"
  "results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092"
#  "results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306"
#  "results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274"
)

#checkpoints2=(
#  "results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306"
#  "results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274"
#)
checkpoints2=${checkpoints}
# Function to extract date (YYYY-MM-DD) from a checkpoint path.
extract_date() {
  # Extract the first occurrence of a date pattern
  echo "$1" | grep -oP '\d{4}-\d{2}-\d{2}'
}

# Function to determine recurrent dimension from a date.
# Returns 128 if the date is before 2025-02, else 256.
get_recurrent_dim() {
  date_str=$(extract_date "$1")
  year=$(echo "$date_str" | cut -d'-' -f1)
  month=$(echo "$date_str" | cut -d'-' -f2)
  if [ "$year" -lt 2025 ] || { [ "$year" -eq 2025 ] && [ "$month" -lt 2 ]; }; then
    echo "128"
  else
    echo "256"
  fi
}

# To verify the get_recurrent_dim function.
for ckpt in "${checkpoints[@]}"; do
  echo "Recurrent dim for $ckpt: $(get_recurrent_dim "$ckpt")"
done
# Maximum number of parallel jobs.
max_workers=30
current_jobs=0

# Loop over all pairs (predator and prey)
for pred_ckpt in "${checkpoints2[@]}"; do
  for prey_ckpt in "${checkpoints[@]}"; do
    # Skip if both checkpoints are the same.
#    if [ "$pred_ckpt" = "$prey_ckpt" ]; then
#      continue
#    fi

    # For predator checkpoint: decide allowed indices and recurrent_dim.
    if [[ "$pred_ckpt" == *"__open_"* ]]; then
      pred_indices=(0 1 2)
    else
      pred_indices=(0 1 2 3 4)
    fi
    pred_rec_dim=$(get_recurrent_dim "$pred_ckpt")

    # For prey checkpoint: decide allowed indices and recurrent_dim.
    if [[ "$prey_ckpt" == *"__open_"* ]]; then
      prey_indices=(3 4 5 6 7 8 9 10 11 12)
    else
      prey_indices=(5 6 7 8 9 10 11 12)
    fi
    prey_rec_dim=$(get_recurrent_dim "$prey_ckpt")

    # Iterate over all allowed index combinations.
    for pred_idx in "${pred_indices[@]}"; do
      for prey_idx in "${prey_indices[@]}"; do

        # Build the command line.
        cmd="export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:/home/mikan/miniconda3/envs/kilosort/lib/python3.9/site-packages/nvidia/cudnn/lib/\"; "
        cmd+="export CUDA_VISIBLE_DEVICES=-1; "
        cmd+="export JAX_PLATFORM_NAME=cpu; "
        cmd+="python evaluate_cross_trial.py --async_distributed "
        cmd+="--available_gpus -1 "
        cmd+="--num_actors 2 "
        cmd+="--algo_name PopArtIMPALA "
        cmd+="--env_name meltingpot "
        cmd+="--map_name predator_prey__simplified10x10_OneVsOne "
        cmd+="--record_video true "
        cmd+="--cross_checkpoint_paths \"${pred_ckpt},${prey_ckpt}\" "
        cmd+="--agent_roles \"predator, prey\" "
        cmd+="--agent_param_indices \"${pred_idx},${prey_idx}\" "
        cmd+="--recurrent_dims \"${pred_rec_dim},${prey_rec_dim}\" "
        cmd+="--num_episodes 40"

        # Print the command for logging.
        echo "Running: $cmd"
#        continue
        # Launch the command in the background.
#        TODO remove this comment
        eval "$cmd" &

        # Increment job counter.
        ((current_jobs++))
        # If we have reached the max_workers, wait for any job to finish.
        if [ "$current_jobs" -ge "$max_workers" ]; then
          wait -n   # Wait for any single job to finish.
          ((current_jobs--))
        fi

      done
    continue
    done
  done
done


# Loop over all pairs (predator and prey)
for pred_ckpt in "${checkpoints[@]}"; do
  for prey_ckpt in "${checkpoints2[@]}"; do
    # Skip if both checkpoints are the same.
    if [ "$pred_ckpt" != "$prey_ckpt" ]; then
      continue
    fi

    # For predator checkpoint: decide allowed indices and recurrent_dim.
    if [[ "$pred_ckpt" == *"__open_"* ]]; then
      pred_indices=(0 1 2)
    else
      pred_indices=(0 1 2 3 4)
    fi
    pred_rec_dim=$(get_recurrent_dim "$pred_ckpt")

    # For prey checkpoint: decide allowed indices and recurrent_dim.
    if [[ "$prey_ckpt" == *"__open_"* ]]; then
      prey_indices=(3 4 5 6 7 8 9 10 11 12)
    else
      prey_indices=(5 6 7 8 9 10 11 12)
    fi
    prey_rec_dim=$(get_recurrent_dim "$prey_ckpt")

    # Iterate over all allowed index combinations.
    for pred_idx in "${pred_indices[@]}"; do
      for prey_idx in "${prey_indices[@]}"; do

        # Build the command line.
        cmd="export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:/home/mikan/miniconda3/envs/kilosort/lib/python3.9/site-packages/nvidia/cudnn/lib/\"; "
        cmd+="export CUDA_VISIBLE_DEVICES=-1; "
        cmd+="export JAX_PLATFORM_NAME=cpu; "
        cmd+="python evaluate_cross_trial.py --async_distributed "
        cmd+="--available_gpus -1 "
        cmd+="--num_actors 2 "
        cmd+="--algo_name PopArtIMPALA "
        cmd+="--env_name meltingpot "
        cmd+="--map_name predator_prey__simplified10x10_OneVsOne "
        cmd+="--record_video true "
        cmd+="--cross_checkpoint_paths \"${pred_ckpt},${prey_ckpt}\" "
        cmd+="--agent_roles \"predator, prey\" "
        cmd+="--agent_param_indices \"${pred_idx},${prey_idx}\" "
        cmd+="--recurrent_dims \"${pred_rec_dim},${prey_rec_dim}\" "
        cmd+="--num_episodes 40"

        # Print the command for logging.
        echo "Running: $cmd"
#        continue
        # Launch the command in the background.
#        TODO remove this comment
        eval "$cmd" &

        # Increment job counter.
        ((current_jobs++))
        # If we have reached the max_workers, wait for any job to finish.
        if [ "$current_jobs" -ge "$max_workers" ]; then
          wait -n   # Wait for any single job to finish.
          ((current_jobs--))
        fi

      done
    continue
    done
  done
done

# Wait for all background jobs to complete.
wait
echo "All evaluations completed."
