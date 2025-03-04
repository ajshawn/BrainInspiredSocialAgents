GPUS="4,5,6,7"
EXP_DIR="results/PopArtIMPALA_2_meltingpot_predator_prey__open_0_2025-02-06_10:08:31.711418"

CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
  --async_distributed \
  --available_gpus ${GPUS} \
  --num_actors 26 \
  --algo_name PopArtIMPALA \
  --env_name meltingpot \
  --map_name predator_prey__open_0 \
  --seed 2 \
  --recurrent_dim 256 \
  --experiment_dir ${EXP_DIR}