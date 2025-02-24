GPUS="1"

CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
  --async_distributed \
  --available_gpus ${GPUS} \
  --num_actors 26 \
  --algo_name PopArtIMPALA \
  --env_name meltingpot \
  --map_name predator_prey__open \
  --seed 1 \
  --use_wandb=False \
  --recurrent_dim 256 \
#  --experiment_dir ${EXP_DIR}