EXP_DIR_PREFIX="./results/PopArtIMPALA_attention_spatial_1_meltingpot_coop_mining_2025-05-29_22:34:03.544807" 
EVN_NAME="coop_mining"
ALGORITHM_NAME="PopArtIMPALA_attention_spatial"
TIME_STAMP="2025-05-29_22:34:03.544807"
GPUS="1"

export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"
# comment out --record video to suppress video recording 
# comment out --log_timesteps to suppress timestep log 

for ckp in 16  ;do # {2..195} #{4,20,73,99,123} {4,40,70,100,135}
    CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py \
        --async_distributed \
        --available_gpus ${GPUS} \
        --num_actors 16 \
        --algo_name ${ALGORITHM_NAME} \
        --env_name meltingpot \
        --map_name ${EVN_NAME} \
        --map_layout small_map \
        --experiment_dir ${EXP_DIR_PREFIX} \
        --agent_roles 'default, default, default' \
        --dense_ore_regrow True \
        --iron_rate 0.00012 \
        --gold_rate 0.00008 \
        --conservative_mine_beam True \
        --iron_reward 1 \
        --gold_reward 6 \
        --mining_reward 0 \
        --ckp ${ckp} \
        --n_episodes 2 \
        --record_video True \
        --positional_embedding learnable
        #--log_timesteps True \
        

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
        #else
            #echo "Skipping $dir (does not start with a number)"
        fi
    done

done
