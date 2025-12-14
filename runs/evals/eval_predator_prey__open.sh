EXP_DIR_PREFIX="results/PopArtIMPALA_attention_multihead_1_meltingpot_predator_prey__open_2025-12-12_00:25:34.057868"
EVN_NAME="predator_prey__open"
map_layout="smaller_16x16"
ALGORITHM_NAME="PopArtIMPALA_attention_multihead"
TIME_STAMP="2025-12-12_00:25:34.057868"
LOG_INTERVAL=1
N_AGENTS=3
GPUS="0"

export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"

for ckp in 64
do
    CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py \
        --async_distributed \
        --available_gpus ${GPUS} \
        --num_actors 16 \
        --algo_name ${ALGORITHM_NAME} \
        --env_name meltingpot \
        --map_name ${EVN_NAME} \
        --map_layout ${map_layout} \
        --experiment_dir ${EXP_DIR_PREFIX} \
        --ckp ${ckp} \
        --agent_roles "predator, prey, prey" \
        --agent_param_indices "0,1,2" \
        --n_episodes 1 \
        --positional_embedding learnable \
        --num_heads 1 \
        --record_video True \

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
