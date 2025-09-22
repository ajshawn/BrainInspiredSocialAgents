GPUS="0"
export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"
exp_dir='results/PopArtIMPALA_attention_multihead_1_meltingpot_coop_mining_2025-09-19_15:33:50.206404'


# CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
#   --algo_name PopArtIMPALA_attention_multihead_self_supervision --positional_embedding learnable \
#   --head_cross_entropy_cost 0 \
#   --env_name meltingpot --map_name coop_mining --map_layout original --seed 1 --agent_roles "default, default, default" \
#   --dense_ore_regrow True --iron_rate 0.00012 --gold_rate 0.00008 \
#   --conservative_mine_beam True --iron_reward 1 --gold_reward 6 --mining_reward 0 --num_heads 1 \
#   --experiment_dir ${exp_dir} \

  #--max_episode_length 500 \


# attention
# CUDA_VISIBLE_DEVICES=${GPUS} python train.py  \
#   --async_distributed \
#   --available_gpus ${GPUS} \
#   --num_actors 16 \
#   --algo_name PopArtIMPALA_attention_multihead \
#   --num_heads 1 \
#   --positional_embedding learnable \
#   --env_name meltingpot \
#   --map_name coop_mining \
#   --map_layout original \
#   --seed 1 \
#   --agent_roles "default, default, default,default, default, default, default, default" \
#   --dense_ore_regrow True \
#   --iron_rate 0.00012 \
#   --gold_rate 0.00008 \
#   --conservative_mine_beam True \
#   --iron_reward 1 \
#   --gold_reward 6 \
#   --mining_reward 0 \
#   --experiment_dir ${exp_dir} \

# Multihead attention with self-supervised attention map
# CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
#   --algo_name PopArtIMPALA_attention_multihead_self_supervision --positional_embedding learnable \
#   --env_name meltingpot --map_name coop_mining --map_layout original --seed 1 --agent_roles "default, default, default" \
#   --dense_ore_regrow True --iron_rate 0.00012 --gold_rate 0.00008 \
#   --conservative_mine_beam True --iron_reward 1 --gold_reward 6 --mining_reward 0 --num_heads 1 --head_cross_entropy_cost 0 \
#   --experiment_dir ${exp_dir} \

# # Simple Transformer
# CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
#   --algo_name simple_transformer \
#   --env_name meltingpot --map_name coop_mining --map_layout original --seed 1 --agent_roles "default, default, default" \
#   --dense_ore_regrow True --iron_rate 0.00012 --gold_rate 0.00008 \
#   --conservative_mine_beam True --iron_reward 1 --gold_reward 6 --mining_reward 0

# # # Simple Transformer with attention 
# CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
#   --algo_name simple_transformer_attention \
#   --positional_embedding learnable --attn_entropy_cost 0.01\
#   --env_name meltingpot --map_name coop_mining --map_layout original --seed 1 --agent_roles "default, default, default, default" \
#   --dense_ore_regrow True --iron_rate 0.00012 --gold_rate 0.00008 \
#   --conservative_mine_beam True --iron_reward 1 --gold_reward 6 --mining_reward 0

#multihead with entropy cost
CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
  --algo_name PopArtIMPALA_attention_multihead --positional_embedding learnable \
  --num_heads 1 --head_cross_entropy_cost 0 --attn_entropy_cost 0.015 \
  --env_name meltingpot --map_name coop_mining --map_layout original --seed 1 --agent_roles "default, default, default,default" \
  --dense_ore_regrow True --iron_rate 0.00012 --gold_rate 0.00008 \
  --conservative_mine_beam True --iron_reward 1 --gold_reward 6 --mining_reward 0 \
  