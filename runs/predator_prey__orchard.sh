GPUS="4,5,6,7"

CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
  --async_distributed \
  --available_gpus ${GPUS} \
  --num_actors 26 \
  --algo_name PopArtIMPALA \
  --env_name meltingpot \
  --map_name predator_prey__orchard \
  --seed 1 \
  --experiment_dir ${EXP_DIR}