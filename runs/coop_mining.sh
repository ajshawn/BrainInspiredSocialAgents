GPUS="0,1,2,3"

CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 24 \
  --algo_name PopArtIMPALA --env_name meltingpot --map_name coop_mining --seed 1 \
  --conservative_mine_beam true --dense_ore_regrow true --agent_roles "default, default, default"