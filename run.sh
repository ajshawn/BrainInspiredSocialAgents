GPUS="4,5,6,7"
FROZEN_AGENTS="3,4,5,6,7,8,9,10,11,12"

CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
  --algo_name PopArtIMPALA --env_name meltingpot --map_name predator_prey__open --frozen_agents ${FROZEN_AGENTS}