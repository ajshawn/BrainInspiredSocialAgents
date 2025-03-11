export CUDA_VISIBLE_DEVICES="2,3"
export PYTHONPATH="."

EXP_DIR_PREFIX="results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17:36:18.023323_ckp"
AGENT_IDX=9

for ckp in 2953 5465 7357 9651 10505
do
    python ./marl/cnn_study/probe_cnn.py \
        --available_gpus 2,3 \
        --num_actors 1 \
        --algo_name PopArtIMPALA \
        --env_name meltingpot \
        --map_name predator_prey__open \
        --experiment_dir ${EXP_DIR_PREFIX}${ckp} \
        --ckp_idx $ckp \
        --agent_idx $AGENT_IDX
done

