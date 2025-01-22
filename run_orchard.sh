export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mikan/miniconda3/envs/kilosort/lib/python3.9/site-packages/nvidia/cudnn/lib/
#EXP_DIR="/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-01-07_14:12:58.801978/"
CUDA_VISIBLE_DEVICES="1" python train.py \
	  --async_distributed \
	    --available_gpus "1" \
	      --num_actors 39 \
	        --algo_name PopArtIMPALA \
		  --env_name meltingpot \
		    --map_name predator_prey__orchard \
		      --seed 1 #\
#		      --experiment_dir $EXP_DIR
