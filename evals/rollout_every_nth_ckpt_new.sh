##!/bin/bash
### We are working on the following experiments:
##/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__random_forest_2025-06-12_15:49:53.295564
##/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__open_2025-06-12_15:49:53.201394
##/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__orchard_2025-06-15_10:11:40.382778
##/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__alley_hunt_2025-06-15_10:11:40.406930
##
## We set the k_samples to 16 to save spaces
## We store their name in this format:
##RandomForestR0_20250612_2v4_rollouts
## OpenR0_20250612_2v4_rollouts
## OrchardR0_20250615_2v4_rollouts
## AlleyHuntR0_20250615_2v4_rollouts
#directories=(
#'PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__random_forest_2025-06-12_15:49:53.295564'
#'PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__open_2025-06-12_15:49:53.201394'
#'PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__orchard_2025-06-15_10:11:40.382778'
#'PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__alley_hunt_2025-06-15_10:11:40.406930'
#)
#output_roots=(
#'RandomForestR0_20250612_2v4_rollouts'
#'OpenR0_20250612_2v4_rollouts'
#'OrchardR0_20250615_2v4_rollouts'
#'AlleyHuntR0_20250615_2v4_rollouts'
#)
### This script runs a command every N checkpoints.
## first check if all evaluate_rollout_every_nth_ckpt.py finishes, if not, wait for them to finish
#  # first chunk
## wait for any old runs to finish
#while pgrep -f "evaluate_rollout_every_nth_ckpt.py" > /dev/null; do
#  echo "Waiting for previous evaluations…"
#  sleep 10
#done
#
#for i in "${!directories[@]}"; do
#  exp_dir="/home/mikan/e/GitHub/social-agents-JAX/results/${directories[$i]}/"
#  out_root="/home/mikan/e/GitHub/social-agents-JAX/results/${output_roots[$i]}/"
#
#  echo "Starting rollouts for ${directories[$i]}…"
#
#  # first chunk
#  python evaluate_rollout_every_nth_ckpt.py \
#    --experiment_dir "$exp_dir" \
#    --max_workers 30 \
#    --out_root "$out_root" \
#    --starting_ckpt 1 \
#    --ending_ckpt 12 \
#    --stride 2 \
#    --k_samples 16 \
#    > "${output_roots[$i]}.log" 2>&1 &
#  pid1=$!
#
#  # second chunk
#  python evaluate_rollout_every_nth_ckpt.py \
#    --experiment_dir "$exp_dir" \
#    --max_workers 30 \
#    --out_root "$out_root" \
#    --starting_ckpt 12 \
#    --ending_ckpt 72 \
#    --stride 6 \
#    --k_samples 16 \
#    >> "${output_roots[$i]}.log" 2>&1 &
#  pid2=$!
#
#  # wait if you want to serialize per‐directory
#  wait $pid1 $pid2
#done

#!/usr/bin/env bash

# List of experiment directories (just the last path component)
directories=(
  'PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__random_forest_2025-06-12_15:49:53.295564'
  'PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__open_2025-06-12_15:49:53.201394'
  'PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__orchard_2025-06-15_10:11:40.382778'
  'PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__alley_hunt_2025-06-15_10:11:40.406930'
)

# Corresponding output roots
output_roots=(
  'RandomForestR0_20250612_2v4_rollouts'
  'OpenR0_20250612_2v4_rollouts'
  'OrchardR0_20250615_2v4_rollouts'
  'AlleyHuntR0_20250615_2v4_rollouts'
)

# Wait for any existing evaluations to finish
while pgrep -f "evaluate_rollout_every_nth_ckpt.py" > /dev/null; do
  echo "Waiting for existing evaluations to finish..."
  sleep 10
done

# Loop through each experiment
for i in "${!directories[@]}"; do
  exp_dir="/home/mikan/e/GitHub/social-agents-JAX/results/${directories[$i]}/"
  out_root="/home/mikan/e/GitHub/social-agents-JAX/results/${output_roots[$i]}/"
  log_file="${output_roots[$i]}.log"

  echo "=== Starting evaluations for ${directories[$i]} ==="
  echo "Experiment dir: $exp_dir"
  echo "Output root:    $out_root"
  echo "Logging to:     $log_file"
  echo
  # We are going to skip the first part of i==1
  if [[ $i -eq 1 ]]; then
    echo "Skipping directory ${directories[$i]} as per request."
    continue
  fi
  # First pass: checkpoints 1–12, stride 2
  python evaluate_rollout_every_nth_ckpt.py \
    --experiment_dir "$exp_dir" \
    --max_workers 90 \
    --out_root "$out_root" \
    --starting_ckpt 1 \
    --ending_ckpt 12 \
    --stride 2 \
    --k_samples 16 \
    > "$log_file" 2>&1

  echo "-- Completed first pass for ${directories[$i]} --"
  echo

  # Second pass: checkpoints 12–72, stride 6
  python evaluate_rollout_every_nth_ckpt.py \
    --experiment_dir "$exp_dir" \
    --max_workers 90 \
    --out_root "$out_root" \
    --starting_ckpt 12 \
    --ending_ckpt 72 \
    --stride 6 \
    --k_samples 16 \
    >> "$log_file" 2>&1

  echo "-- Completed second pass for ${directories[$i]} --"
  echo
done

echo "All evaluations done."
