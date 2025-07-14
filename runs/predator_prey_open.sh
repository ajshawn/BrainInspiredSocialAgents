GPUS="0"

CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
  --async_distributed \
  --available_gpus ${GPUS} \
  --num_actors 13 \
  --algo_name PopArtIMPALA_attention_multihead_item_aware \
  --env_name meltingpot \
  --map_name predator_prey__open \
  --seed 1 \
  --positional_embedding learnable \
  --head_cross_entropy_cost 0.01