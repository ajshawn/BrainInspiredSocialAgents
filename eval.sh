EXP_DIR="/local/shawn/marl-jax/results/PopArtIMPALA_0_meltingpot_predator_prey__open_2024-10-21_20:05:04.446458"

CUDA_VISIBLE_DEVICES="4,5" python evaluate.py \
    --async_distributed \
    --available_gpus "4, 5" \
    --num_actors 16 \
    --algo_name PopArtIMPALA \
    --env_name meltingpot \
    --map_name predator_prey__open \
    --experiment_dir ${EXP_DIR} \
    --record_video true