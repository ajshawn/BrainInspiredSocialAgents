#!/bin/bash

## This script runs a command every N checkpoints.
## Run at Jun 17th 2025
python evaluate_rollout_every_nth_ckpt.py \
  --experiment_dir "/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__random_forest_2025-04-24_17:23:56.924098/" \
  --max_workers 30 \
  --out_root "/home/mikan/e/GitHub/social-agents-JAX/results/RandomForest20250424_2v4_rollouts/" \
  --starting_ckpt 2 \
  --ending_ckpt 8 \
  --stride 1 \
  --skip_ckpts 6

python evaluate_rollout_every_nth_ckpt.py \
  --experiment_dir "/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274/" \
  --max_workers 30 \
  --out_root "/home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/"\
  --starting_ckpt 2 \
  --ending_ckpt 8 \
  --stride 1 \
  --skip_ckpts 6

##python evaluate_rollout_every_nth_ckpt.py \
#  --experiment_dir "/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__random_forest_2025-04-24_17:23:56.924098/" \
#  --max_workers 80 \
#  --out_root "/home/mikan/e/GitHub/social-agents-JAX/results/RandomForest20250424_2v4_rollouts/" \
#  --starting_ckpt 6 \
#  --ending_ckpt 50 \
#  --stride 5 \
#  --skip_ckpts 1 11 21 31 41 51
#
#
#
#python evaluate_rollout_every_nth_ckpt.py \
#  --experiment_dir "/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274/" \
#  --max_workers 80 \
#  --out_root "/home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/"\
#  --starting_ckpt 6 \
#  --ending_ckpt 50 \
#  --stride 5 \
#  --skip_ckpts 1 11 21 31 41 51
#
#
#python evaluate_rollout_every_nth_ckpt.py \
#  --experiment_dir "/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_no_group__open_2025-05-29_13:13:09.229184/" \
#  --max_workers 80 \
#  --starting_ckpt 1 \
#  --out_root "/home/mikan/e/GitHub/social-agents-JAX/results/open_no_group_20250529_2v4_rollouts/"\
#  --starting_ckpt 2 \
#  --ending_ckpt 50 \
#  --stride 5

#python evaluate_rollout_every_nth_ckpt.py \
#  --experiment_dir "/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_no_group__orchard_2025-05-29_13:14:34.637462/" \
#  --max_workers 80 \
#  --starting_ckpt 1 \
#  --out_root "/home/mikan/e/GitHub/social-agents-JAX/results/orchard_no_group_20250529_2v4_rollouts/"\
#  --ending_ckpt 50 \
#  --stride 5