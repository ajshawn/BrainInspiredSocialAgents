GPUS="0"
export PYTHONPATH="./gits/meltingpot:gits/acme:${PYTHONPATH}"
#exp_dir='results/PopArtIMPALA_attention_multihead_1_meltingpot_predator_prey__open_2026-01-08_02:40:43.111077'

CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
  --async_distributed \
  --available_gpus ${GPUS} \
  --num_actors 16 \
  --algo_name PopArtIMPALA_attention_multihead \
  --num_heads 2 \
  --positional_embedding learnable \
  --env_name meltingpot \
  --map_name predator_prey__open \
  --seed 1 \
  --map_layout smaller_16x16 \
  --agent_roles "predator, prey, prey" \
  #--experiment_dir ${exp_dir}