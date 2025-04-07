GPUS="6,7"
experiment_dir="results/predator_prey__open_1B_step/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17:36:18.023323_ckp10684"
external_cnn_dir="results/predator_prey__open_1B_step/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17:36:18.023323_ckp10684"
external_cnn_finetune_dir="results/PopArtIMPALA_1_meltingpot_predator_prey__open_external_cnn_from_ckp10684_2"
replace_cnn_agent_idx_lhs="7,9"
replace_cnn_agent_idx_rhs="9,7"
frozen_agents="0,1,2,3,4,5,6,8,10,11,12"

CUDA_VISIBLE_DEVICES=$GPUS python train.py \
  --available_gpus $GPUS \
  --num_actors 16 \
  --algo_name PopArtIMPALA \
  --env_name meltingpot \
  --map_name predator_prey__open \
  --experiment_dir $experiment_dir \
  --external_cnn_dir $external_cnn_dir \
  --external_cnn_finetune_dir $external_cnn_finetune_dir \
  --replace_cnn_agent_idx_lhs $replace_cnn_agent_idx_lhs \
  --replace_cnn_agent_idx_rhs $replace_cnn_agent_idx_rhs \
  --seed 1 \
  --frozen_agents $frozen_agents
