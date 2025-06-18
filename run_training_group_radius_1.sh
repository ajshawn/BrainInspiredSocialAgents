#!/usr/bin/env bash
set -euo pipefail

# 1) cuDNN library path
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/mikan/miniconda3/envs/kilosort/lib/python3.9/site-packages/nvidia/cudnn/lib/"

# 2) experiment directories (still exported, but no --experiment_dir passed)
#OPEN_EXP="/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_1__open_2025-05-29_13:13:09.229184"
#RF_EXP="/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_1__random_forest_2025-05-29_13:14:34.637462"
#ORCHARD_EXP="/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_1__orchard_2025-05-29_13:14:34.637462"
#ALLEY_EXP="/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_1__alley_hunt_2025-01-07_12:11:32.926962"

# 3) Per-job settings
maps=(open random_forest orchard alley_hunt)
gpus=(0 1 0 1)
actors=(52 52 52 52)
# tie them together:
#exp_dirs=("$OPEN_EXP" "$RF_EXP" "$ORCHARD_EXP" "$ALLEY_EXP")

# 4) Run two at a time, each capped at 50 h
for ((i=0; i<${#maps[@]}; i+=2)); do
  for offset in 0 1; do
    idx=$((i+offset))
    # guard against odd length
    if (( idx >= ${#maps[@]} )); then
      continue
    fi

    MAP="${maps[idx]}"
    GPU="${gpus[idx]}"
    NA="${actors[idx]}"
#    EXP_DIR="${exp_dirs[idx]}"

    echo "Launching: GPU${GPU} → ${MAP} (${NA} actors)…"
    timeout 50h \
      env CUDA_VISIBLE_DEVICES="${GPU}" \
      python train.py \
        --async_distributed \
        --available_gpus "${GPU}" \
        --num_actors "${NA}" \
        --algo_name PopArtIMPALA \
        --env_name meltingpot \
        --map_name "predator_prey_group_radius_1__${MAP}" \
        --seed 42 \
        --use_wandb=False \
        --recurrent_dim 256 \
      &> "${MAP}.log" &
  done
  wait   # wait for both to finish before next pair
done

echo "All jobs launched in pairs, each with a 50 h timeout."
