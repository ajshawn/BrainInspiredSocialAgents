#!/usr/bin/env bash
set -euo pipefail

# 1) cuDNN library path
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/mikan/miniconda3/envs/kilosort/lib/python3.9/site-packages/nvidia/cudnn/lib/"

# 2) experiment directories (still exported, but no --experiment_dir passed)
OPEN_EXP="/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__open_2025-06-12_15:49:53.201394"
RF_EXP="/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__random_forest_2025-06-12_15:49:53.295564"
#RF_EXP="/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_1__random_forest_2025-05-29_13:14:34.637462"
ORCHARD_EXP="/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__orchard_2025-06-15_10:11:40.382778"
ALLEY_EXP="/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__alley_hunt_2025-06-15_10:11:40.406930"

# 3) Per-job settings
maps=(open random_forest orchard alley_hunt)
gpus=(0 1 0 1)
actors=(52 52 52 52)
#actors=(65 65 65 65)
# tie them together:
exp_dirs=("$OPEN_EXP" "$RF_EXP" "$ORCHARD_EXP" "$ALLEY_EXP")

# 4) Run two at a time, each capped at 50 h
#for ((i=0; i<${#maps[@]}; i+=2)); do
for ((i=2; i<${#maps[@]}; i+=2)); do
  for offset in 0 1; do
    idx=$((i+offset))
    # guard against odd length
    if (( idx >= ${#maps[@]} )); then
      continue
    fi

    MAP="${maps[idx]}"
    GPU="${gpus[idx]}"
    NA="${actors[idx]}"
    EXP_DIR="${exp_dirs[idx]}"

    echo "Launching: GPU${GPU} → ${MAP} (${NA} actors)…"
    timeout 24h \
      env CUDA_VISIBLE_DEVICES="${GPU}" \
      python train.py \
        --async_distributed \
        --all_parallel \
        --available_gpus "${GPU}" \
        --num_actors "${NA}" \
        --algo_name PopArtIMPALA \
        --env_name meltingpot \
        --map_name "predator_prey_group_radius_0__${MAP}" \
        --seed 42 \
        --use_wandb=False \
        --wandb_entity=qinli2021 \
        --wandb_project=marl-jax \
        --wandb_tags="PopArtIMPALA,group_radius_0,${MAP}" \
        --recurrent_dim 256 \
        --model_time_delta_minutes 60 \
        --experiment_dir "${EXP_DIR}" \
      &> "${MAP}.log" &
  done
  wait   # wait for both to finish before next pair
done

echo "All jobs launched in pairs, each with a 50 h timeout."
