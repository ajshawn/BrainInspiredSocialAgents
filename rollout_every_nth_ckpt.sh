#!/bin/bash

# This script runs a command every N checkpoints.
#python evaluate_rollout_every_nth_ckpt.py \
#  --experiment_dir "/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__random_forest_2025-04-24_17:23:56.924098/" \
#  --stride 10 \
#  --max_workers 50 \
#  --out_root "/home/mikan/e/GitHub/social-agents-JAX/results/RandomForest20250424_2v4_rollouts/"


python evaluate_rollout_every_nth_ckpt.py \
  --experiment_dir "/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274/" \
  --stride 10 \
  --max_workers 80 \
  --out_root "/home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/"