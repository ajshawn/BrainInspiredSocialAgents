GPUS="1"
export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"
#export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib/:/root/miniconda3/pkgs/cudatoolkit-11.8.0-h6a678d5_0/lib:$LD_LIBRARY_PATH"

# baseline 
CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
  --algo_name simple_transformer \
  --env_name meltingpot --map_name clean_up --seed 1 --agent_roles "default, default, default, default, default" \
  #--experiment_dir ${exp_dir} \

# CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
#   --algo_name PopArtIMPALA_attention_multihead --positional_embedding learnable \
#   --head_cross_entropy_cost 0 --attn_entropy_cost 0 \
#   --env_name meltingpot --map_name coop_mining --map_layout original --seed 1 --agent_roles "default, default, default, default, default" \
#   --dense_ore_regrow True --iron_rate 0.00012 --gold_rate 0.00008 \
#   --conservative_mine_beam True --iron_reward 1 --gold_reward 6 --mining_reward 0 --num_heads 1 \
#   #--experiment_dir ${exp_dir} \

  #--max_episode_length 500 \

