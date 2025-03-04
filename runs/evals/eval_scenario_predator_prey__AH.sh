#!/usr/bin/env bash

EXP_DIR_PREFIX="results/PopArtIMPALA_meltingpot_predator_prey__alley_hunt_ckp"
EVN_NAME="predator_prey__alley_hunt"
ALGORITHM_NAME="PopArtIMPALA"
TIME_STAMP="2025-02-26"
GPUS="6,7"
N_SCENES=4

# We still keep the array for scenes 0,1 and possibly 4..6,
# but for scenes 2 and 3 we will override it in code.
AGENT_PARAM_INDICES_LIST=(
  "5,6,7,8,9,10,11,12" # Scene 0
  "0,1,2,3,4"          # Scene 1
  ""                   # Scene 2 (overridden below)
  ""                   # Scene 3 (overridden below)
)

# Make a directory to store logs (if it doesn't exist).
mkdir -p logs

for ckp in 2419
do
    # 1. Loop through each scenario
    for (( i=0; i < N_SCENES; i++ )); do

        # For scenario 2 (the "third" scene),
        # evaluate each predator index [0..4] one-by-one
        if [ "$i" -eq 2 ]; then
            for p in 0 1 2 3 4
            do
                OUTPUT_FILE="logs/${EVN_NAME}_${i}_param${p}_ckp${ckp}.txt"
                echo "Evaluating scenario: ${EVN_NAME}_${i}, checkpoint: ${ckp}, predator index: ${p}"
                echo "Logging to: ${OUTPUT_FILE}"

                CUDA_VISIBLE_DEVICES="${GPUS}" python evaluate.py \
                    --available_gpus "${GPUS}" \
                    --num_actors 16 \
                    --algo_name "${ALGORITHM_NAME}" \
                    --env_name meltingpot \
                    --map_name "${EVN_NAME}" \
                    --record_video true \
                    --experiment_dir "${EXP_DIR_PREFIX}${ckp}" \
                    --run_scenario_by_name "${EVN_NAME}_${i}" \
                    --agent_param_indices "${p}" \
                    > "${OUTPUT_FILE}" 2>&1
            done

        # For scenario 3 (the "fourth" scene),
        # evaluate each prey index [5..12] one-by-one
        elif [ "$i" -eq 3 ]; then
            for p in 5 6 7 8 9 10 11 12
            do
                OUTPUT_FILE="logs/${EVN_NAME}_${i}_param${p}_ckp${ckp}.txt"
                echo "Evaluating scenario: ${EVN_NAME}_${i}, checkpoint: ${ckp}, prey index: ${p}"
                echo "Logging to: ${OUTPUT_FILE}"

                CUDA_VISIBLE_DEVICES="${GPUS}" python evaluate.py \
                    --available_gpus "${GPUS}" \
                    --num_actors 16 \
                    --algo_name "${ALGORITHM_NAME}" \
                    --env_name meltingpot \
                    --map_name "${EVN_NAME}" \
                    --record_video true \
                    --experiment_dir "${EXP_DIR_PREFIX}${ckp}" \
                    --run_scenario_by_name "${EVN_NAME}_${i}" \
                    --agent_param_indices "${p}" \
                    > "${OUTPUT_FILE}" 2>&1
            done

        else
            # For other scenes, just use whatever is in AGENT_PARAM_INDICES_LIST
            OUTPUT_FILE="logs/${EVN_NAME}_${i}_ckp${ckp}.txt"
            echo "Evaluating scenario: ${EVN_NAME}_${i}, checkpoint: ${ckp}"
            echo "Logging to: ${OUTPUT_FILE}"

            CUDA_VISIBLE_DEVICES="${GPUS}" python evaluate.py \
                --available_gpus "${GPUS}" \
                --num_actors 16 \
                --algo_name "${ALGORITHM_NAME}" \
                --env_name meltingpot \
                --map_name "${EVN_NAME}" \
                --record_video true \
                --experiment_dir "${EXP_DIR_PREFIX}${ckp}" \
                --run_scenario_by_name "${EVN_NAME}_${i}" \
                --agent_param_indices "${AGENT_PARAM_INDICES_LIST[$i]}" \
                > "${OUTPUT_FILE}" 2>&1
        fi
    done

    # 2. Rename directories for all scenarios
    new_recording_name="${ALGORITHM_NAME}_${EVN_NAME}_${TIME_STAMP}_ckp${ckp}"

    for ((i = 0; i < N_SCENES; i++)); do
        recording_dir="recordings/meltingpot/${EVN_NAME}_${i}"

        echo "Deleting empty directories in $recording_dir:"
        find "$recording_dir" -type d -empty -print -delete

        echo "Renaming non-empty directories in $recording_dir that start with a number:"
        count=1
        for dir in "$recording_dir"/*; do
            if [ -d "$dir" ] && [ "$(ls -A "$dir")" ] && [[ "$(basename "$dir")" =~ ^[0-9] ]]; then
                new_name="$recording_dir/${new_recording_name}_$count"
                mv "$dir" "$new_name"
                echo "Renamed $dir to $new_name"
                ((count++))
            else
                echo "Skipping $dir (does not start with a number or is empty)"
            fi
        done
    done
done
