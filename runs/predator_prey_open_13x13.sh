GPUS="0,1,2,3"
exp_dir='results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-01-08_11:36:42.879145'

CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
  --async_distributed \
  --available_gpus ${GPUS} \
  --num_actors 26 \
  --algo_name PopArtIMPALA \
  --env_name meltingpot \
  --map_name predator_prey__open \
  --seed 1 \
  --map_layout smaller_13x13 \
  --agent_roles "predator, prey, prey, prey" \
  --experiment_dir ${exp_dir}