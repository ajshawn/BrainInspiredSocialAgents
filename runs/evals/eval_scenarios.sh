EXP_DIR_PREFIX="results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17:36:18.023323_ckp"
EVN_NAME="predator_prey__open"
ALGORITHM_NAME="PopArtIMPALA"
TIME_STAMP="2024-11-26_17:36:18.023323"
GPUS="1,2"
N_SCENES=8

for ckp in 10684
do
    CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py \
        --available_gpus ${GPUS} \
        --num_actors 16 \
        --algo_name ${ALGORITHM_NAME} \
        --env_name meltingpot \
        --map_name ${EVN_NAME} \
        --record_video true \
        --experiment_dir ${EXP_DIR_PREFIX}${ckp} \
        --run_eval_on_scenarios true \
        --agent_param_indices "3,4,5,6,7,8,9,10,11,12" \

    new_recording_name="${ALGORITHM_NAME}_${EVN_NAME}_${TIME_STAMP}_ckp${ckp}"

    # Rename all scenarios' corresponding directories
    for ((i = 0; i < N_SCENES; i++)); do
        recording_dir="recordings/meltingpot/${EVN_NAME}_${i}"
        #Delete empty directories in recording_dir
        echo "Deleting empty directories in $recording_dir:"
        find "$recording_dir" -type d -empty -print -delete

        # 3. Rename non-empty directories in recording_dir if their name starts with a number
        echo "Renaming non-empty directories in $recording_dir that start with a number:"
        count=1
        for dir in "$recording_dir"/*; do
            # Check if it's a directory and its name starts with a number
            if [ -d "$dir" ] && [ "$(ls -A "$dir")" ] && [[ "$(basename "$dir")" =~ ^[0-9] ]]; then
                new_name="$recording_dir/${new_recording_name}_$count"
                mv "$dir" "$new_name"
                echo "Renamed $dir to $new_name"
                ((count++))
            else
                echo "Skipping $dir (does not start with a number)"
            fi
        done
    done
done
