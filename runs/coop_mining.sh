GPUS="6,7"

export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"
#exp_dir='results/PopArtIMPALA_attention_1_meltingpot_coop_mining_2025-05-22_16:22:39.109747'
CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
  --algo_name PopArtIMPALA_attention --positional_embedding learnable --env_name meltingpot --map_name coop_mining --map_layout small_map --seed 1 --agent_roles "default, default, default" \
  --dense_ore_regrow True --iron_rate 0.00012 --gold_rate 0.00008 \
  --conservative_mine_beam True --iron_reward 1 --gold_reward 6 --mining_reward 0 \
  #--experiment_dir ${exp_dir} \ 
#--map_layout small_map 

