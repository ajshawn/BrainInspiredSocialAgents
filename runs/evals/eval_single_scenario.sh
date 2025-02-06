#!/usr/bin/env bash

EXP_DIR_PREFIX="results/PopArtIMPALA_2_meltingpot_predator_prey__open_0_2025-02-01_19:23:39.789296_ckp"
EVN_NAME="predator_prey__open"
ALGORITHM_NAME="PopArtIMPALA"
TIME_STAMP="2025-02-01_19:23:39.789296"
GPUS="1,2"
N_SCENES=1

# Make a directory to store logs (if it doesn't exist).
mkdir -p logs

i=0

for ckp in 1166
do
    # Construct an output file name that includes the scenario index and checkpoint.
    OUTPUT_FILE="logs/${EVN_NAME}_${i}_ckp${ckp}.txt"

    echo "Evaluating scenario: ${EVN_NAME}_${i}, checkpoint: ${ckp}"
    echo "Logging to: ${OUTPUT_FILE}"

    # Run the python script and redirect both stdout and stderr to the text file.
    CUDA_VISIBLE_DEVICES="${GPUS}" python evaluate.py \
        --available_gpus "${GPUS}" \
        --num_actors 16 \
        --algo_name "${ALGORITHM_NAME}" \
        --env_name meltingpot \
        --map_name "${EVN_NAME}" \
        --record_video true \
        --experiment_dir "${EXP_DIR_PREFIX}${ckp}" \
        --run_scenario_by_name "${EVN_NAME}_${i}" \
        --agent_param_indices "0,1,2,3,4,5,6,7,8,9" \
        > "${OUTPUT_FILE}" 2>&1

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
