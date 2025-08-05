GPUS="4,5,6,7"
EXP_DIR="results/PopArtIMPALA_attention_item_aware_1_meltingpot_predator_prey__open_2025-06-21_13:24:43.145268"

# 1 head, 64dim, preys with acorn CE 0.05
# CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
#   --async_distributed \
#   --available_gpus ${GPUS} \
#   --num_actors 16 \
#   --algo_name PopArtIMPALA_attention_multihead_item_aware \
#   --env_name meltingpot \
#   --map_name predator_prey__open \
#   --seed 1 \
#   --positional_embedding learnable \
#   --head_cross_entropy_cost 0.05 \
#   --num_heads 1 \
#   --attn_enhance_agent_skip_indices '0,1,2'

# # CNN "attention", preys with acorn CE 0.05
# CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
#   --async_distributed \
#   --available_gpus ${GPUS} \
#   --num_actors 16 \
#   --algo_name PopArtIMPALA_CNN_visualization \
#   --env_name meltingpot \
#   --map_name predator_prey__open \
#   --seed 1 \
#   --head_cross_entropy_cost 0.05 \
#   --attn_enhance_agent_skip_indices '0,1,2'

# 1 head, 64dim, preys with acorn CE 0.05, frequecy-based positional embedding
CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
  --async_distributed \
  --available_gpus ${GPUS} \
  --num_actors 16 \
  --algo_name PopArtIMPALA_attention_multihead_item_aware \
  --env_name meltingpot \
  --map_name predator_prey__open \
  --seed 1 \
  --positional_embedding frequency \
  --head_cross_entropy_cost 0.05 \
  --num_heads 1 \
  --attn_enhance_agent_skip_indices '0,1,2'