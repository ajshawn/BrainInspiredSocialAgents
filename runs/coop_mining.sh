GPUS="0"
export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"

# #exp_dir='results/PopArtIMPALA_attention_multihead_item_aware_1_meltingpot_coop_mining_2025-08-08_18:30:42.296262'
# CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
#   --algo_name PopArtIMPALA_attention --positional_embedding frequency \
#   --env_name meltingpot --map_name coop_mining --map_layout original --seed 1 --agent_roles "default, default, default,default" \
#   --dense_ore_regrow True --iron_rate 0 --gold_rate 0.0001 --max_episode_length 500 \
#   --conservative_mine_beam True --iron_reward 1 --gold_reward 6 --mining_reward 0 - --num_heads 1 \
 # --experiment_dir ${exp_dir} \

  #--max_episode_length 500 \
# --head_cross_entropy_cost 0.05

# # attention-item CE loss on Baseline Impala with CNN
# CUDA_VISIBLE_DEVICES=${GPUS} python train.py  \
#   --async_distributed \
#   --available_gpus ${GPUS} \
#   --num_actors 16 \
#   --algo_name PopArtIMPALA_CNN_visualization \
#   --env_name meltingpot \
#   --map_name coop_mining \
#   --map_layout original \
#   --seed 1 \
#   --agent_roles "default, default, default" \
#   --dense_ore_regrow True \
#   --iron_rate 0.00012 \
#   --gold_rate 0.00008 \
#   --conservative_mine_beam True \
#   --iron_reward 1 \
#   --gold_reward 6 \
#   --mining_reward 0 \
#   --head_cross_entropy_cost 0.05

# Multihead attention with self-supervised attention map
# CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
#   --algo_name PopArtIMPALA_attention_multihead_self_supervision --positional_embedding learnable \
#   --env_name meltingpot --map_name coop_mining --map_layout original --seed 1 --agent_roles "default, default, default" \
#   --dense_ore_regrow True --iron_rate 0.00012 --gold_rate 0.00008 \
#   --conservative_mine_beam True --iron_reward 1 --gold_reward 6 --mining_reward 0 --num_heads 1 --head_cross_entropy_cost 0.05

# Simple Transformer
CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 1 \
  --algo_name simple_transformer \
  --env_name meltingpot --map_name coop_mining --map_layout original --seed 1 --agent_roles "default, default, default" \
  --dense_ore_regrow True --iron_rate 0.00012 --gold_rate 0.00008 \
  --conservative_mine_beam True --iron_reward 1 --gold_reward 6 --mining_reward 0



