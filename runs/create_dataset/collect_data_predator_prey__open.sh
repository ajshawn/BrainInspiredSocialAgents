GPUS="4,5,6,7"

CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
  --available_gpus ${GPUS} \
  --num_actors 2 \
  --algo_name PopArtIMPALA \
  --env_name meltingpot \
  --map_name predator_prey__open \
  --seed 1 \
  --log_agent_views true