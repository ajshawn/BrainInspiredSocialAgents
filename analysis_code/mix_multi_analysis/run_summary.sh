#!/bin/bash
#python summarize_basic_behavior_info.py
#python summarize_higher_level_behavior_info.py
#python summarize_behavior_netState_df_mix.py
python summarize_basic_behavior_info.py \
--base_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/ \
--out_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/analysis_results/ \
--jobs 30
python summarize_higher_level_behavior_info.py \
--base_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/ \
--out_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/analysis_results_extended/ \
--jobs 30
python summarize_behavior_netState_df_mix.py \
--base_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/ \
--out_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/analysis_results/ \
--jobs 30