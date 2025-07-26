GPUS="4,5,6,7"

export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"
exp_dir='results/PopArtIMPALA_1_meltingpot_coop_mining_2025-07-21_16:00:58.916298'
# CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
#   --algo_name PopArtIMPALA_attention_multihead \
#   --env_name meltingpot --map_name coop_mining --map_layout original --seed 1 --agent_roles "default, default, default" \
#   --dense_ore_regrow True --iron_rate 0.00012 --gold_rate 0.00008 \
#   --conservative_mine_beam True --iron_reward 1 --gold_reward 6 --mining_reward 0
  # --experiment_dir ${exp_dir}
  #--max_episode_length 500 \


  # attention-item CE loss on Baseline Impala with CNN
  CUDA_VISIBLE_DEVICES=${GPUS} python train.py  \
    --async_distributed \
    --available_gpus ${GPUS} \
    --num_actors 16 \
    --algo_name PopArtIMPALA_CNN_visualization \
    --env_name meltingpot \
    --map_name coop_mining \
    --map_layout original \
    --seed 1 \
    --agent_roles "default, default, default" \
    --dense_ore_regrow True \
    --iron_rate 0.00012 \
    --gold_rate 0.00008 \
    --conservative_mine_beam True \
    --iron_reward 1 \
    --gold_reward 6 \
    --mining_reward 0 \
    --head_cross_entropy_cost 0.05

  



