GPUS="0,1"
export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib/:/root/miniconda3/pkgs/cudatoolkit-11.8.0-h6a678d5_0/lib:$LD_LIBRARY_PATH"
#exp_dir='results/simple_transformer_attention_1_meltingpot_coop_mining_2025-10-08_13:36:13.677366'


# #Simple Transformer with attention 
CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
  --algo_name simple_transformer_attention \
  --positional_embedding learnable --hidden_scale 0 --reward_pred_cost 0 \
  --env_name meltingpot --map_name coop_mining --map_layout original --seed 1 --agent_roles "default, default, default" \
  --dense_ore_regrow True --iron_rate 0.00012 --gold_rate 0.00008 \
  --conservative_mine_beam True --iron_reward 1 --gold_reward 10 --mining_reward 0 \
  #--experiment_dir ${exp_dir}

# # # Simple Transformer with cnn feedback
# CUDA_VISIBLE_DEVICES=${GPUS} python train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
#   --algo_name simple_transformer_cnnfeedback \
#   --positional_embedding learnable --reward_pred_cost 0 \
#   --env_name meltingpot --map_name coop_mining --map_layout original --seed 1 --agent_roles "default, default, default" \
#   --dense_ore_regrow True --iron_rate 0.00012 --gold_rate 0.00008 \
#   --conservative_mine_beam True --iron_reward 1 --gold_reward 8 --mining_reward 0 \
# #  --experiment_dir ${exp_dir}