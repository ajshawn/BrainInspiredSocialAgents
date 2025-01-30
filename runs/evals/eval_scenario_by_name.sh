#!/usr/bin/env bash

EXP_DIR_PREFIX="results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17:36:18.023323_ckp"
EVN_NAME="predator_prey__open"
ALGORITHM_NAME="PopArtIMPALA"
TIME_STAMP="2024-11-26_17:36:18.023323"
GPUS="1,2"
N_SCENES=7

# Define a list of agent_param_indices strings – one per scenario.
AGENT_PARAM_INDICES_LIST=(
  "3,4,5,6,7,8,9,10,11,12" # Focal prey visited by background predators
  "0,1,2" # Focal predators aim to eat basic resident prey
  "1" # A focal predator competes with background predators to eat prey
  "7" # One focal prey ad hoc cooperates with background prey to avoid predation
  "0,1,2" # Focal predators hunt smarter resident prey
  "1" # A focal predator competes with background predators to hunt smarter prey
  "7" # One focal prey ad hoc cooperates with background smart prey to avoid predation
)

# Make a directory to store logs (if it doesn't exist).
mkdir -p logs

for ckp in 10684
do
    # 1. Loop through each scenario
    for (( i=0; i < N_SCENES; i++ )); do
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
            --agent_param_indices "${AGENT_PARAM_INDICES_LIST[$i]}" \
            > "${OUTPUT_FILE}" 2>&1

        # Optionally, you might prefer appending (`>> "${OUTPUT_FILE}" 2>&1`) 
        # if you want to accumulate logs rather than overwrite.
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
