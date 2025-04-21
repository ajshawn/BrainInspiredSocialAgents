export CUDA_VISIBLE_DEVICES="2,3"
export PYTHONPATH="."

EXP_DIR_PREFIX="results/predator_prey__open_1B_step/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17:36:18.023323_ckp"
AGENT_IDX=7

for ckp in 10684
do
    python ./marl/cnn_study/acorn_cnn.py \
        --available_gpus 2,3 \
        --num_actors 1 \
        --algo_name PopArtIMPALA \
        --env_name meltingpot \
        --map_name predator_prey__open \
        --experiment_dir ${EXP_DIR_PREFIX}${ckp} \
        --ckp_idx $ckp \
        --agent_idx $AGENT_IDX \
        --mode cnn_inference \
        --labels_path data/predator_prey_acorn_pairs/labels.jsonl \
        --random_baseline true
done

