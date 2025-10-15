EXP_DIR_PREFIX="results/simple_transformer_attention_1_meltingpot_coop_mining_2025-10-10_18:01:51.444848"
EVN_NAME="coop_mining"
ALGORITHM_NAME="simple_transformer_attention"
TIME_STAMP="2025-10-10_18:01:51.444848"
LOG_INTERVAL=1
N_AGENTS=3
GPUS="0"
N_HEADS=1

export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"

for ckp in 33 ;do # {2..195}

    obs_out_dir="data/${ALGORITHM_NAME}_${EVN_NAME}_${TIME_STAMP}_ckp${ckp}"
    log_filename="${obs_out_dir}/observations.jsonl"
    log_img_dir="${obs_out_dir}/agent_view_images"

    CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py \
        --async_distributed \
        --available_gpus ${GPUS} \
        --num_actors 1 \
        --algo_name ${ALGORITHM_NAME} \
        --env_name meltingpot \
        --map_name ${EVN_NAME} \
        --experiment_dir ${EXP_DIR_PREFIX} \
        --ckp ${ckp} \
        --n_episodes 1 \
        --record_video True \
        --log_timesteps True \
        --log_obs True \
        --log_filename ${log_filename} \
        --log_img_dir ${log_img_dir} \
        --log_interval ${LOG_INTERVAL} \
        --agent_param_indices "0,1,2"
        
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
    attn_src_csv="${EXP_DIR_PREFIX}/csv_logs/${EVN_NAME}${ckp}-timesteps.csv"
    attn_src_img_dir="${log_img_dir}"

    python marl/utils/visualize_attention.py \
        --csv_path ${attn_src_csv} \
        --image_dir ${attn_src_img_dir} \
        --save_dir ${attn_vis_out_dir} \
        --n_agents ${N_AGENTS} \
        --n_heads ${N_HEADS}

    echo "CNN visualization saved to ${attn_vis_out_dir}"

done
