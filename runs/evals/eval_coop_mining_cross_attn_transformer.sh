#EXP_DIR_PREFIX="./results/PopArtIMPALA_1_meltingpot_coop_mining_2025-03-17_14:26:51.204572,./results/PopArtIMPALA_1_meltingpot_coop_mining_2025-03-17_14:26:51.204572,./results/PopArtIMPALA_1_meltingpot_coop_mining_2025-03-14_14:39:21.229188,./results/PopArtIMPALA_1_meltingpot_coop_mining_2025-03-14_14:39:21.229188,./results/PopArtIMPALA_1_meltingpot_coop_mining_2025-03-14_14:39:21.229188,./results/PopArtIMPALA_1_meltingpot_coop_mining_2025-03-17_14:26:51.204572" 
#ckp_map="0:120-2,1:40-1,2:40-2,3:40-3,4:30-0"

# save_dir="./cross_eval_results/envstep300/attn-transformer-alone-5ag"
# EXP_DIR_PREFIX=".,./results/simple_transformer_attention_1_meltingpot_coop_mining_2025-09-29_13:02:18.340053"
# ckp_map="0:133-0"
# EVN_NAME="coop_mining"
# ALGORITHM_NAME="simple_transformer_attention"
# TIME_STAMP="2025-09-29_13:02:18.340053"
# GPUS="1"

# save_dir="./cross_eval_results/envstep300/attn-transformer-2in5"
# EXP_DIR_PREFIX=".,./results/simple_transformer_attention_1_meltingpot_coop_mining_2025-09-29_13:02:18.340053,./results/simple_transformer_attention_1_meltingpot_coop_mining_2025-09-29_13:02:18.340053"
# EVN_NAME="coop_mining"
# ALGORITHM_NAME="simple_transformer_attention"
# TIME_STAMP="2025-09-29_13:02:18.340053"
# GPUS="1"

# ckp_maps=(
#     "0:133-0,1:133-1"
#     "0:133-2,1:133-3"
#     "0:133-4,1:130-0"
# )

# save_dir="./cross_eval_results/attn-transformer-2coop3single"
# EXP_DIR_PREFIX=".,./results/simple_transformer_attention_1_meltingpot_coop_mining_2025-09-29_13:02:18.340053,./results/simple_transformer_attention_1_meltingpot_coop_mining_2025-09-30_13:22:33.666745,./results/simple_transformer_attention_1_meltingpot_coop_mining_2025-09-30_13:22:33.666745,./results/simple_transformer_attention_1_meltingpot_coop_mining_2025-09-30_13:22:33.666745,./results/simple_transformer_attention_1_meltingpot_coop_mining_2025-09-29_13:02:18.340053"
# ckp_map="0:49-0,1:23-0,2:23-1,3:23-2,4:49-2"
# EVN_NAME="coop_mining"
# ALGORITHM_NAME="simple_transformer_attention"
# TIME_STAMP="2025-09-29_13:02:18.340053"
# GPUS="1"

save_dir="./cross_eval_results/attn-transformer-alone-hidden0.5"
EXP_DIR_PREFIX=".,./results/simple_transformer_attention_1_meltingpot_coop_mining_2025-10-15_04:32:48.824037"
ckp_map="0:133-0"
EVN_NAME="coop_mining"
ALGORITHM_NAME="simple_transformer_attention"
TIME_STAMP="2025-10-15_04:32:48.824037"
GPUS="1"


export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"
# comment out --record video to suppress video recording 
# comment out --log_timesteps to suppress timestep log 
#for ckp_map in "${ckp_maps[@]}"; do
for i in {0..2}; do 
   ckp_map="0:49-$i"
    CUDA_VISIBLE_DEVICES=${GPUS} python cross_evaluate.py \
        --async_distributed \
        --available_gpus ${GPUS} \
        --num_actors 16 \
        --algo_name ${ALGORITHM_NAME} \
        --env_name meltingpot \
        --map_name ${EVN_NAME} \
        --map_layout original \
        --max_episode_length 500 \
        --experiment_dir ${EXP_DIR_PREFIX} \
        --save_dir ${save_dir} \
        --agent_roles 'default' \
        --hidden_scale 0 \
        --dense_ore_regrow True \
        --iron_rate 0.0002 \
        --gold_rate 0.0001 \
        --conservative_mine_beam True \
        --iron_reward 1 \
        --gold_reward 6 \
        --mining_reward 0 \
        --ckp_map ${ckp_map} \
        --n_episodes 4 \
        --agent_param_indices '0'\
        --positional_embedding learnable \
        #--record_video True \
        #--log_timesteps True \
                
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
done


