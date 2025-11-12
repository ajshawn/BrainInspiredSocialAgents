GPUS="0"
export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"
#exp_dir='results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-01-08_11:36:42.879145'

CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
  --async_distributed \
  --available_gpus ${GPUS} \
  --num_actors 16 \
  --algo_name PopArtIMPALA_attention_multihead \
  --num_heads 1 \
  --positional_embedding learnable \
  --env_name meltingpot \
  --map_name predator_prey__open \
  --seed 1 \
  --map_layout smaller_16x16 \
  --agent_roles "predator, prey, prey" \
  #--experiment_dir ${exp_dir}