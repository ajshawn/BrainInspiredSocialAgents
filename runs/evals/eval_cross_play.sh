export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"

GPUS="1"
CUDA_VISIBLE_DEVICES=${GPUS} python cross_evaluate.py \
    --cross_play_config_path "runs/evals/cross_eval_config.yaml" \
    --env_name meltingpot \
    --map_name coop_mining \
    --map_layout original \
    --dense_ore_regrow True \
    --iron_rate 0.00012 \
    --gold_rate 0.00008 \
    --conservative_mine_beam True \
    --iron_reward 1 \
    --gold_reward 6 \
    --mining_reward 0 