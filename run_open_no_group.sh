export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mikan/miniconda3/envs/kilosort/lib/python3.9/site-packages/nvidia/cudnn/lib/
experiment_dir='/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_no_group__open_2025-05-29_13:13:09.229184/'
CUDA_VISIBLE_DEVICES="0" python train.py \
	  --async_distributed \
	    --available_gpus "0" \
	      --num_actors 52 \
	        --algo_name PopArtIMPALA \
		  --env_name meltingpot \
		    --map_name predator_prey_no_group__open \
		      --seed 42 \
		      --use_wandb=False \
		      --recurrent_dim 256 \
		      --experiment_dir $experiment_dir
