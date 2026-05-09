export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"

GPUS="2"
CUDA_VISIBLE_DEVICES=${GPUS} python cross_evaluate.py \
    --cross_play_config_path "runs/evals/cross_eval_config.yaml" \
    --env_name meltingpot \
    --map_name predator_prey__open \
    --map_layout smaller_16x16_rand \
    --n_episodes 20 \
    --run_config_sweep True