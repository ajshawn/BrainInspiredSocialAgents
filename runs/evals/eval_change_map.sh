GPUS="1"
EVN_NAME="coop_mining"

# Define arrays for each experiment setting
# save_dirs=(
#     "./map_eval_results/attn-transformer-hidden0.5"
#     "./map_eval_results/transformer"
#     "./map_eval_results/baseline"
#     "./map_eval_results/attn-lstm-supervised"
#     "./map_eval_results/attn-lstm-self-supervised"
# )

# EXP_DIR_PREFIXES=(
#     "./results/results/simple_transformer_attention_1_meltingpot_coop_mining_2025-10-15_04:32:45.743946"
#     "./results/results/simple_transformer_1_meltingpot_coop_mining_2025-09-17_16:27:05.143749"
#     "./results/results/PopArtIMPALA_1_meltingpot_coop_mining_2025-07-21_16:00:58.916298"
#     "./results/results/PopArtIMPALA_attention_multihead_item_aware_1_meltingpot_coop_mining_2025-07-23_16:38:01.516419"
#     "./results/results/PopArtIMPALA_attention_multihead_self_supervision_1_meltingpot_coop_mining_2025-09-14_11:15:56.921430"
# )

# ckps=(61 22 58 101 128)

# ALGORITHM_NAMES=(
#     "simple_transformer_attention"
#     "simple_transformer"
#     "PopArtIMPALA"
#     "PopArtIMPALA_attention_multihead_item_aware"
#     "PopArtIMPALA_attention_multihead_self_supervision"
# )

# TIME_STAMPS=(
#     "2025-10-15_04:32:45.743946"
#     "2025-09-17_16:27:05.143749"
#     "2025-07-21_16:00:58.916298"
#     "2025-07-23_16:38:01.516419"
#     "2025-09-14_11:15:56.921430"
# )

TIME_STAMPS=(
    # "2025-07-21_16:00:58.916298"
    # "2025-08-18_20:07:21.933169"
    # "2025-07-23_16:38:01.516419"
    # "2025-08-14_16:19:12.187931"
    # "2025-08-18_20:05:07.151340"
    "2025-09-14_11:15:56.921430"
    "2025-10-17_20:06:05.608232"
    "2025-10-20_10:24:52.040748"
)

save_dirs=(
    # "./map_eval_results/seeds/baseline"
    # "./map_eval_results/seeds/baseline"
    # "./map_eval_results/seeds/attn-lstm-supervised"
    # "./map_eval_results/seeds/attn-lstm-supervised"
    # "./map_eval_results/seeds/attn-lstm-supervised"
    "./map_eval_results/seeds/attn-lstm-self-supervised"
    "./map_eval_results/seeds/attn-lstm-self-supervised"
    "./map_eval_results/seeds/attn-lstm-self-supervised"
)

EXP_DIR_PREFIXES=(
    # "./results/results/PopArtIMPALA_1_meltingpot_coop_mining_2025-07-21_16:00:58.916298"
    # "./results/results/PopArtIMPALA_3_meltingpot_coop_mining_2025-08-18_20:07:21.933169"
    # "./results/results/PopArtIMPALA_attention_multihead_item_aware_1_meltingpot_coop_mining_2025-07-23_16:38:01.516419"
    # "./results/results/PopArtIMPALA_attention_multihead_item_aware_2_meltingpot_coop_mining_2025-08-14_16:19:12.187931"
    # "./results/results/PopArtIMPALA_attention_multihead_item_aware_3_meltingpot_coop_mining_2025-08-18_20:05:07.151340"
    "./results/results/PopArtIMPALA_attention_multihead_self_supervision_1_meltingpot_coop_mining_2025-09-14_11:15:56.921430"
    "./results/results/PopArtIMPALA_attention_multihead_self_supervision_2_meltingpot_coop_mining_2025-10-17_20:06:05.608232"
    "./results/results/PopArtIMPALA_attention_multihead_self_supervision_3_meltingpot_coop_mining_2025-10-20_10:24:52.040748"
)

#ckps=(58 62 101 101 92 128 141 99)
ckps=(128 141 99)

ALGORITHM_NAMES=(
    # "PopArtIMPALA"
    # "PopArtIMPALA"
    # "PopArtIMPALA_attention_multihead_item_aware"
    # "PopArtIMPALA_attention_multihead_item_aware"
    # "PopArtIMPALA_attention_multihead_item_aware"
    "PopArtIMPALA_attention_multihead_self_supervision"
    "PopArtIMPALA_attention_multihead_self_supervision"
    "PopArtIMPALA_attention_multihead_self_supervision"
)



export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"

# Outer loop: iterate over 5 experiment configurations
for i in {0..4}; do
    save_dir=${save_dirs[$i]}
    EXP_DIR_PREFIX=${EXP_DIR_PREFIXES[$i]}
    ckp=${ckps[$i]}
    ALGORITHM_NAME=${ALGORITHM_NAMES[$i]}
    TIME_STAMP=${TIME_STAMPS[$i]}

    echo "=============================="
    echo "Running ${ALGORITHM_NAME} (ckp ${ckp})"
    echo "=============================="

    # Inner loop: iterate over 5 map layouts
    for map_layout in random2 random3 random4 random5; do
        echo "========== Evaluating map layout: ${map_layout} =========="

        CUDA_VISIBLE_DEVICES=${GPUS} python evaluate.py \
            --async_distributed \
            --available_gpus ${GPUS} \
            --num_actors 16 \
            --algo_name ${ALGORITHM_NAME} \
            --num_heads 1 \
            --env_name meltingpot \
            --map_name ${EVN_NAME} \
            --map_layout ${map_layout} \
            --experiment_dir ${EXP_DIR_PREFIX} \
            --save_dir ${save_dir} \
            --max_episode_length 500 \
            --agent_roles 'default,default,default' \
            --dense_ore_regrow True \
            --iron_rate 0.0002 \
            --gold_rate 0.0001 \
            --conservative_mine_beam True \
            --iron_reward 1 \
            --gold_reward 6 \
            --mining_reward 0 \
            --ckp ${ckp} \
            --n_episodes 10 \
            --positional_embedding learnable \
            --agent_param_indices '0,1,2' \
            --hidden_scale 0.5
            # --record_video True  # uncomment if needed

        recording_dir="recordings/meltingpot/${EVN_NAME}"
        mkdir -p "$recording_dir"
        new_recording_name="${ALGORITHM_NAME}_${EVN_NAME}_${TIME_STAMP}_ckp${ckp}"

        # Delete empty directories
        echo "Deleting empty directories in $recording_dir:"
        find "$recording_dir" -type d -empty -print -delete

        # Rename non-empty directories that start with a number
        echo "Renaming non-empty directories in $recording_dir that start with a number:"
        count=1
        for dir in "$recording_dir"/*; do
            if [ -d "$dir" ] && [ "$(ls -A "$dir")" ] && [[ "$(basename "$dir")" =~ ^[0-9] ]]; then
                new_name="$recording_dir/${new_recording_name}_$count"
                mv "$dir" "$new_name"
                echo "Renamed $dir to $new_name"
                ((count++))
            fi
        done
    done
done