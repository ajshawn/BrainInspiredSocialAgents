GPUS="4,5,6,7"
EXP_DIR="/local/shawn/marl-jax/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17:36:18.023323"

CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
  --async_distributed \
  --available_gpus ${GPUS} \
  --num_actors 26 \
  --algo_name PopArtIMPALA \
  --env_name meltingpot \
  --map_name predator_prey__open_0 \
  --seed 2
  # --experiment_dir ${EXP_DIR}