EXP_DIR_PREFIX="results/PopArtIMPALA_1_meltingpot_coop_mining_2024-12-08_17:16:42.390336_ckp"
EVN_NAME="coop_mining"
ALGORITHM_NAME="PopArtIMPALA"
TIME_STAMP="2024-12-08_17:16:42.390336"
GPUS="5, 6"

for ckp in 513
do
    CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py \
        --async_distributed \
        --available_gpus ${GPUS} \
        --num_actors 8 \
        --algo_name ${ALGORITHM_NAME} \
        --env_name meltingpot \
        --map_name ${EVN_NAME} \
        --record_video true \
        --experiment_dir ${EXP_DIR_PREFIX}${ckp} \
        --conservative_mine_beam true \
        --dense_ore_regrow true \
        --agent_roles 'default, default, default'

    recording_dir="recordings/meltingpot/${EVN_NAME}"
    new_recording_name="${ALGORITHM_NAME}_${EVN_NAME}_${TIME_STAMP}_ckp${ckp}"

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
