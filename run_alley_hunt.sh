export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mikan/miniconda3/envs/kilosort/lib/python3.9/site-packages/nvidia/cudnn/lib/
#EXP_DIR="/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/"
CUDA_VISIBLE_DEVICES="0" python train.py \
	  --async_distributed \
	    --available_gpus "0" \
	      --num_actors 39 \
	        --algo_name PopArtIMPALA \
		  --env_name meltingpot \
		    --map_name predator_prey__alley_hunt \
		      --seed 1  #\
		       #--experiment_dir $EXP_DIR