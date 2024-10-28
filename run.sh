CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py  --async_distributed --available_gpus "0,1,2,3" --num_actors 26 \
  --algo_name PopArtIMPALA --env_name meltingpot --map_name predator_prey__open