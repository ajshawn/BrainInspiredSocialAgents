EXP_DIR_PREFIX="results/PopArtIMPALA_attention_multihead_1_meltingpot_coop_mining_2025-06-26_18:40:32.248065"
EVN_NAME="coop_mining"
ALGORITHM_NAME="PopArtIMPALA_attention_multihead"
TIME_STAMP="2025-06-26_18:40:32.248065"
LOG_INTERVAL=1
N_AGENTS=3
GPUS="5"
N_HEADS=4

export PYTHONPATH=".:./gits/meltingpot:gits/acme:${PYTHONPATH}"

for ckp in 60 ;do # {2..195} #{4,20,73,99,123} {4,40,70,100,135}

    obs_out_dir="data/${ALGORITHM_NAME}_${EVN_NAME}_${TIME_STAMP}_ckp${ckp}"
    log_filename="${obs_out_dir}/observations.jsonl"
    log_img_dir="${obs_out_dir}/agent_view_images"

    # CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py \
    #     --async_distributed \
    #     --available_gpus ${GPUS} \
    #     --num_actors 16 \
    #     --algo_name ${ALGORITHM_NAME} \
    #     --env_name meltingpot \
    #     --map_name ${EVN_NAME} \
    #     --map_layout original \
    #     --experiment_dir ${EXP_DIR_PREFIX} \
    #     --agent_roles 'default, default, default' \
    #     --dense_ore_regrow True \
    #     --iron_rate 0.0001 \
    #     --gold_rate 0.00008 \
    #     --conservative_mine_beam True \
    #     --iron_reward 1 \
    #     --gold_reward 6 \
    #     --mining_reward 0 \
    #     --ckp ${ckp} \
    #     --n_episodes 1 \
    #     --record_video True \
    #     --positional_embedding learnable \
    #     --log_timesteps True \
    #     --log_obs True \
    #     --log_filename ${log_filename} \
    #     --log_img_dir ${log_img_dir} \
    #     --log_interval ${LOG_INTERVAL} \
    #     --agent_param_indices "0,1,2"
        
    recording_dir="recordings/meltingpot/${EVN_NAME}"
    new_recording_name="${ALGORITHM_NAME}_${EVN_NAME}_${TIME_STAMP}_ckp${ckp}"

    # Delete empty directories in recording_dir
    echo "Deleting empty directories in $recording_dir:"
    find "$recording_dir" -type d -empty -print -delete

    # Rename non-empty directories in recording_dir if their name starts with a number
    echo "Renaming non-empty directories in $recording_dir that start with a number:"
    count=1
    for dir in "$recording_dir"/*; do
        # Check if it's a directory and its name starts with a number
        if [ -d "$dir" ] && [ "$(ls -A "$dir")" ] && [[ "$(basename "$dir")" =~ ^[0-9] ]]; then
            new_name="$recording_dir/${new_recording_name}_$count"
            mv "$dir" "$new_name"
            echo "Renamed $dir to $new_name" 
            ((count++))
        fi
    done

    # Visualize attention
    echo "Visualizing attention for checkpoint ${ckp}..."
    
    attn_vis_out_dir="${obs_out_dir}/attention_vis"
    attn_src_csv="${EXP_DIR_PREFIX}/csv_logs/coop_mining${ckp}-timesteps.csv"
    attn_src_img_dir="${log_img_dir}"

    python marl/utils/visualize_attention_global.py \
        --attn_weights_csv ${attn_src_csv} \
        --image_dir ${attn_src_img_dir} \
        --log_jsonl_path ${log_filename} \
        --save_dir ${attn_vis_out_dir} \
        --n_agents ${N_AGENTS} \
        --n_heads ${N_HEADS}         

    echo "Attention visualization saved to ${attn_vis_out_dir}"

done
