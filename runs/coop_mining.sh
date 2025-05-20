GPUS="0"

export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"
#exp_dir='results/PopArtIMPALA_1_meltingpot_coop_mining_2025-03-28_19:45:56.562736'
CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
  --algo_name PopArtIMPALA_attention --positional_embedding True --env_name meltingpot --map_name coop_mining --seed 1 --agent_roles "default, default" \
  --dense_ore_regrow True --iron_rate 0.0001 --gold_rate 0.00008 \
  --conservative_mine_beam True --iron_reward 1 --gold_reward 6 --mining_reward 0 \
  #--experiment_dir ${exp_dir}

