#!/usr/bin/env bash

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/mikan/miniconda3/envs/kilosort/lib/python3.9/site-packages/nvidia/cudnn/lib/"
export CUDA_VISIBLE_DEVICES=-1
export JAX_PLATFORM_NAME=cpu

EXP_DIR_PREFIX="results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp"
EVN_NAME="predator_prey__simplified10x10_OneVsOne"
MAP_NAME=""
ALGORITHM_NAME="PopArtIMPALA"
TIME_STAMP="2024-11-26_17:36:18.023323"
GPUS="0"

# Maximum number of parallel jobs in the first block.
MAX_JOBS=12
current_jobs=0

# ---------------------------------------------------------------------------
# 1) First block: run baseline jobs with no PLSC perturbation
# ---------------------------------------------------------------------------
for ckp in 7357 9651
do
  for predator_param_index in {0..2}
  do
    for prey_param_index in {3..12}
    do
      # 1) Throttle concurrency: if at max, wait for one job to finish
      if [ "${current_jobs}" -ge "${MAX_JOBS}" ]; then
        wait -n
        ((current_jobs--))
      fi

      # 2) Launch job in background
      (
        echo "Starting job: ckp=${ckp}, predator=${predator_param_index}, prey=${prey_param_index}"

        CUDA_VISIBLE_DEVICES="${GPUS}" python evaluate.py \
          --async_distributed \
          --available_gpus "${GPUS}" \
          --num_actors 16 \
          --algo_name "${ALGORITHM_NAME}" \
          --env_name meltingpot \
          --map_name "${EVN_NAME}" \
          --record_video true \
          --experiment_dir "${EXP_DIR_PREFIX}${ckp}" \
          --agent_roles "predator, prey" \
          --agent_param_indices "${predator_param_index}, ${prey_param_index}" \
          --num_episodes 100

        echo "Job finished: ckp=${ckp}, predator=${predator_param_index}, prey=${prey_param_index}"
      ) &

      # 3) Sleep 60s before the next job
      echo "Sleeping 60 seconds before launching the next job..."
      sleep 60

      # 4) Increment the concurrency count
      ((current_jobs++))
    done
  done
done

# ---------------------------------------------------------------------------
# 2) Second block: run code with PLSC perturbations
# ---------------------------------------------------------------------------
# OPTIONAL: re-use concurrency for second block
MAX_JOBS_2=12
current_jobs_2=0

for ckp in 7357 9651
do
  for predator_param_index in {0..2}
  do
    for prey_param_index in {3..12}
    do
      for perturb in "predator" "prey" "predator, prey"
      do
        # (A) Throttle concurrency if desired
        if [ "${current_jobs_2}" -ge "${MAX_JOBS_2}" ]; then
          wait -n
          ((current_jobs_2--))
        fi

        # (B) Launch job in background
        (
          echo "Starting job: ckp=${ckp}, predator=${predator_param_index}, prey=${prey_param_index}, perturb=${perturb}"

          CUDA_VISIBLE_DEVICES="${GPUS}" python evaluate.py \
            --async_distributed \
            --available_gpus "${GPUS}" \
            --num_actors 16 \
            --algo_name "${ALGORITHM_NAME}" \
            --env_name meltingpot \
            --map_name "${EVN_NAME}" \
            --record_video true \
            --experiment_dir "${EXP_DIR_PREFIX}${ckp}" \
            --agent_roles "predator, prey" \
            --agent_param_indices "${predator_param_index}, ${prey_param_index}" \
            --num_episodes 100 \
            --plsc_dim_to_perturb 10 \
            --agent_to_perturb "${perturb}" \
            --plsc_decomposition_dict_path "${EXP_DIR_PREFIX}${ckp}/pickles/PLSC_usv_dict.pkl"

          echo "Job finished: ckp=${ckp}, predator=${predator_param_index}, prey=${prey_param_index}, perturb=${perturb}"
        ) &

        # (C) Optional sleep if you want a gap between launches
        echo "Sleeping 30 seconds before next PLSC job..."
        sleep 30

        # (D) Increment concurrency
        ((current_jobs_2++))
      done
    done
  done
done

# ---------------------------------------------------------------------------
# 3) Wait for all background jobs to finish
# ---------------------------------------------------------------------------
wait
echo "All parallel jobs finished!"
