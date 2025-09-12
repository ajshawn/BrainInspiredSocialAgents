
EXP_DIR_PREFIX="./results/PopArtIMPALA_attention_1_meltingpot_coop_mining_2025-05-28_00:34:00.259639,./results/PopArtIMPALA_attention_1_meltingpot_coop_mining_2025-06-10_21:32:15.740274,./results/PopArtIMPALA_attention_1_meltingpot_coop_mining_2025-05-28_00:34:00.259639,./results/PopArtIMPALA_attention_1_meltingpot_coop_mining_2025-06-10_21:32:15.740274" 
ckp_map="0:180-0,1:60-1,2:160-1"

GPUS="1"

export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"
#exp_dir='results/PopArtIMPALA_attention_multihead_item_aware_1_meltingpot_coop_mining_2025-08-08_18:30:42.296262'
CUDA_VISIBLE_DEVICES=${GPUS} python cross_train.py  --async_distributed --available_gpus ${GPUS} --num_actors 16 \
  --algo_name PopArtIMPALA_attention --positional_embedding learnable\
  --env_name meltingpot --map_name coop_mining --map_layout original --seed 1 --agent_roles "default, default, default" \
  --dense_ore_regrow True --iron_rate 0.0001 --gold_rate 0.0001 --max_episode_length 500 \
  --conservative_mine_beam True --iron_reward 1 --gold_reward 6 --mining_reward 0 --num_heads 1 \
  --experiment_dir ${EXP_DIR_PREFIX} --ckp_map ${ckp_map} \
  --frozen_agents "1" \
  #--max_episode_length 500 \
# --head_cross_entropy_cost 0.05

# # attention-item CE loss on Baseline Impala with CNN
# CUDA_VISIBLE_DEVICES=${GPUS} python train.py  \
#   --async_distributed \
#   --available_gpus ${GPUS} \
#   --num_actors 16 \
#   --algo_name PopArtIMPALA_CNN_visualization \
#   --env_name meltingpot \
#   --map_name coop_mining \
#   --map_layout original \
#   --seed 1 \
#   --agent_roles "default, default, default" \
#   --dense_ore_regrow True \
#   --iron_rate 0.00012 \
#   --gold_rate 0.00008 \
#   --conservative_mine_beam True \
#   --iron_reward 1 \
#   --gold_reward 6 \
#   --mining_reward 0 \
#   --head_cross_entropy_cost 0.05
