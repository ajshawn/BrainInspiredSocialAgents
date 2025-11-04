GPUS="0"
EVN_NAME="coop_mining"

TIME_STAMP="2025-07-21_16:00:58.916298"

save_dir="./map_eval_results/agent_eval_results/baseline"

RUNS=(
  "./results/results/PopArtIMPALA_1_meltingpot_coop_mining_2025-07-21_16:00:58.916298"
  "./results/results/PopArtIMPALA_1_meltingpot_coop_mining_2025-06-13_11:31:14.677056"
  "./results/results/PopArtIMPALA_3_meltingpot_coop_mining_2025-08-18_20:07:21.933169"
)
STEPS=(58 50 62)
run_of ()   { local k="$1"; echo $(( k / 3 )); }   # 0..2
agent_of () { local k="$1"; echo $(( k % 3 )); }   # 0..2

ALGORITHM_NAME="PopArtIMPALA"

export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"

# Outer loop: iterate over 5 experiment configurations
#for ckp_map in "${ckp_maps[@]}"; do

for i in {0..8}; do
    ri=$(run_of "$i")
    ai=$(agent_of "$i")
    step_i=${STEPS[$ri]}
    run_i=${RUNS[$ri]}
    for j in {0..8}; do
        rj=$(run_of "$j")
        aj=$(agent_of "$j")
        step_j=${STEPS[$rj]}
        run_j=${RUNS[$rj]}

        EXP_DIR_PREFIX=".,${run_i},${run_j}"
        ckp_map="0:${step_i}-${ai},1:${step_j}-${aj}"

        echo "=============================="
        echo "Running agent ${ckp_map})"
        echo "=============================="

        CUDA_VISIBLE_DEVICES=${GPUS} python cross_evaluate.py \
            --async_distributed \
            --available_gpus ${GPUS} \
            --num_actors 16 \
            --algo_name ${ALGORITHM_NAME} \
            --num_heads 1 \
            --env_name meltingpot \
            --map_name ${EVN_NAME} \
            --map_layout original \
            --experiment_dir ${EXP_DIR_PREFIX} \
            --save_dir ${save_dir} \
            --max_episode_length 500 \
            --agent_roles 'default,default' \
            --dense_ore_regrow True \
            --iron_rate 0.0002 \
            --gold_rate 0.0001 \
            --conservative_mine_beam True \
            --iron_reward 1 \
            --gold_reward 6 \
            --mining_reward 0 \
            --ckp_map ${ckp_map} \
            --n_episodes 10 \
            --positional_embedding learnable \
            --agent_param_indices '0,1' \
            # --hidden_scale 0.5
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