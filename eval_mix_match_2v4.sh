#!/bin/bash

# Function to extract date (YYYY-MM-DD) from a checkpoint path.
extract_date() {
  echo "$1" | grep -oP '\d{4}-\d{2}-\d{2}'
}

# Function to determine recurrent dimension from a date.
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

# Predator pool with correct paths and indices
predator_checkpoints=(
  "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306/"
  "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/"
  "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274/"
)

# Generate predator combinations
predator_combinations=()

# For the first checkpoint, the predators are indexed 0, 1, 2
predator_combinations+=("./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306/ predator 0 $(get_recurrent_dim "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306/")")
predator_combinations+=("./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306/ predator 1 $(get_recurrent_dim "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306/")")
predator_combinations+=("./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306/ predator 2 $(get_recurrent_dim "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306/")")

# For the second checkpoint, only predator index 0 is valid
predator_combinations+=("./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/ predator 0 $(get_recurrent_dim "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/")")

# For the third checkpoint, only predator index 0 is valid
predator_combinations+=("./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274/ predator 0 $(get_recurrent_dim "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274/")")

# Prey pool with correct paths and indices
prey_checkpoints=(
  "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/"
  "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651"
  "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306"
)

# Generate prey combinations
prey_combinations=()

# For the first prey checkpoint, the valid prey indices are 5, 10
prey_combinations+=("./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/ prey 5 $(get_recurrent_dim "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/")")
prey_combinations+=("./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/ prey 10 $(get_recurrent_dim "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/")")

# For the second prey checkpoint, the valid prey indices are 5, 6, 7
prey_combinations+=("./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651 prey 5 $(get_recurrent_dim "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651")")
prey_combinations+=("./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651 prey 6 $(get_recurrent_dim "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651")")
prey_combinations+=("./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651 prey 7 $(get_recurrent_dim "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651")")

# For the third prey checkpoint, the valid prey index is 6
prey_combinations+=("./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306/ prey 6 $(get_recurrent_dim "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306/")")

# Define combinations: 1vs1, 1vs2, 2vs3, 2vs4
combinations=("1vs1" "1vs2" "2vs3" "2vs4")

# Function to randomly sample combinations without replacement
sample_without_replacement() {
  # Number of agents: $1 for predator, $2 for prey
  pred_count=$1
  prey_count=$2
  sample_predators=$(shuf -e "${predator_combinations[@]}" -n $pred_count)
  sample_preys=$(shuf -e "${prey_combinations[@]}" -n $prey_count)

  # Generate the full command
  for pred_combo in $sample_predators; do
    IFS=" " read -r pred_ckpt pred_role pred_idx pred_dim <<< "$pred_combo"

    for prey_combo in $sample_preys; do
      IFS=" " read -r prey_ckpt prey_role prey_idx prey_dim <<< "$prey_combo"

      # Build cross_checkpoint_paths, agent_roles, indices, and recurrent_dims
      cross_checkpoint_paths="${pred_ckpt},${prey_ckpt}"
      agent_roles="predator,prey"
      agent_param_indices="${pred_idx},${prey_idx}"
      recurrent_dims="${pred_dim},${prey_dim}"

      # Build and print the command
      cmd="python evaluate_cross_trial.py --async_distributed "
      cmd+="--available_gpus -1 "
      cmd+="--num_actors 16 "
      cmd+="--algo_name PopArtIMPALA "
      cmd+="--env_name meltingpot "
      cmd+="--map_name predator_prey__simplified10x10_OneVsOne "
      cmd+="--record_video true "
      cmd+="--cross_checkpoint_paths \"$cross_checkpoint_paths\" "
      cmd+="--agent_roles \"$agent_roles\" "
      cmd+="--agent_param_indices \"$agent_param_indices\" "
      cmd+="--recurrent_dims \"$recurrent_dims\" "
      cmd+="--num_episodes 20"

      # Print the command for logging
      echo "Running: $cmd"

      # Uncomment to run the command
      # eval "$cmd" &
    done
  done
}

# Loop over combinations and sample appropriately
for combo in "${combinations[@]}"; do
  case $combo in
    "1vs1")
      sample_without_replacement 1 1
      ;;
    "1vs2")
      sample_without_replacement 1 2
      ;;
    "2vs3")
      sample_without_replacement 2 3
      ;;
    "2vs4")
      sample_without_replacement 2 4
      ;;
  esac
done
