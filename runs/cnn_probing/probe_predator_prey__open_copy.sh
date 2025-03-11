export PYTHONPATH="."

GPUS="0,1"
EXP_DIR_PREFIX="results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17:36:18.023323_ckp"
AGENT_IDX=7

for ckp in 2953
do
   CUDA_VISIBLE_DEVICES=${GPUS} python ./marl/cnn_study/probe_cnn.py \
        --available_gpus ${GPUS} \
        --num_actors 1 \
        --algo_name PopArtIMPALA \
        --env_name meltingpot \
        --map_name predator_prey__open \
        --experiment_dir ${EXP_DIR_PREFIX}${ckp} \
        --ckp_idx $ckp \
        --agent_idx $AGENT_IDX \
        --random_baseline true
done

