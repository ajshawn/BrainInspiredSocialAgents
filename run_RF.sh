export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mikan/miniconda3/envs/kilosort/lib/python3.9/site-packages/nvidia/cudnn/lib/
#EXP_DIR="/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-01-07_14:12:58.801978/"
#EXP_DIR="/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/"
#EXP_DIR="/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274/"
CUDA_VISIBLE_DEVICES="1" python train.py \
	  --async_distributed \
	    --available_gpus "1" \
	      --num_actors 52 \
	        --algo_name PopArtIMPALA \
		  --env_name meltingpot \
		    --map_name predator_prey__random_forest \
		      --seed 1 \
		      --use_wandb=False \
		      --recurrent_dim 256
#		      --experiment_dir $EXP_DIR
