GPUS="0,1,2,3"
EXP_DIR="/local/shawn/marl-jax/results/PopArtIMPALA_2_meltingpot_predator_prey__open_4_2025-02-05_20:54:56.012060"

CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
  --async_distributed \
  --available_gpus ${GPUS} \
  --num_actors 26 \
  --algo_name PopArtIMPALA \
  --env_name meltingpot \
  --map_name predator_prey__open_4 \
  --seed 2 \
  --experiment_dir ${EXP_DIR}