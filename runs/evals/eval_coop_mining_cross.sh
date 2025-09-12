#EXP_DIR_PREFIX="./results/PopArtIMPALA_attention_1_meltingpot_coop_mining_2025-09-04_01:34:46.303068,./results/PopArtIMPALA_attention_1_meltingpot_coop_mining_2025-06-10_21:32:15.740274,./results/PopArtIMPALA_attention_1_meltingpot_coop_mining_2025-05-28_00:34:00.259639,./results/PopArtIMPALA_attention_1_meltingpot_coop_mining_2025-05-28_00:34:00.259639,./results/PopArtIMPALA_attention_1_meltingpot_coop_mining_2025-05-28_00:34:00.259639,./results/PopArtIMPALA_attention_1_meltingpot_coop_mining_2025-05-28_00:34:00.259639" 
#ckp_map="0:180-0,1:60-1,2:60-2,3:60-3,4:40-0"

save_dir="./cross_eval_results/attention-alone"
EXP_DIR_PREFIX=".,./results/PopArtIMPALA_attention_1_meltingpot_coop_mining_2025-06-10_21:32:15.740274"
ckp_map="0:180-1"
EVN_NAME="coop_mining"
ALGORITHM_NAME="PopArtIMPALA_attention"
TIME_STAMP="2025-06-10_21:32:15.740274"
GPUS="0"

export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"
# comment out --record video to suppress video recording 
# comment out --log_timesteps to suppress timestep log 


CUDA_VISIBLE_DEVICES=${GPUS} python cross_evaluate.py \
    --async_distributed \
    --available_gpus ${GPUS} \
    --num_actors 16 \
    --algo_name ${ALGORITHM_NAME} \
    --positional_embedding learnable \
    --env_name meltingpot \
    --map_name ${EVN_NAME} \
    --map_layout original \
    --max_episode_length 500 \
    --experiment_dir ${EXP_DIR_PREFIX} \
    --save_dir ${save_dir} \
    --agent_roles 'default' \
    --dense_ore_regrow True \
    --iron_rate 0.0002 \
    --gold_rate 0.0001 \
    --conservative_mine_beam True \
    --iron_reward 1 \
    --gold_reward 6 \
    --mining_reward 0 \
    --ckp_map ${ckp_map} \
    --n_episodes 10 \
    --agent_param_indices '0'\
    #--record_video True \
    #--log_timesteps True \
    #--add_selection_vector True \
        
        

    recording_dir="recordings/meltingpot/${EVN_NAME}"
    new_recording_name="${ALGORITHM_NAME}_${EVN_NAME}_${TIME_STAMP}_ckp${ckp_map}"

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


