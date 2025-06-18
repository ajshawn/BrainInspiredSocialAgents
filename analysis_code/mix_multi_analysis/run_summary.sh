#!/bin/bash
#python summarize_basic_behavior_info.py
#python summarize_higher_level_behavior_info.py
#python summarize_behavior_netState_df_mix.py

#python summarize_basic_behavior_info.py \
#--base_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/ \
#--out_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/analysis_results/ \
#--jobs 90
#python summarize_higher_level_behavior_info.py \
#--base_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/ \
#--out_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/analysis_results_extended/ \
#--jobs 90
#python summarize_behavior_netState_df_mix.py \
#--base_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/ \
#--out_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/analysis_results/ \
#--jobs 90

folders=("/home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/" \
"/home/mikan/e/GitHub/social-agents-JAX/results/RandomForest20250424_2v4_rollouts/" \
#"/home/mikan/e/GitHub/social-agents-JAX/results/open_no_group_20250529_2v4_rollouts/" \
#"/home/mikan/e/GitHub/social-agents-JAX/results/orchard_no_group_20250529_2v4_rollouts/" \
)
for folder in "${folders[@]}"; do
  echo "Processing folder: $folder"
  python summarize_basic_behavior_info.py \
    --base_dir "$folder" \
    --out_dir "${folder}analysis_results/" \
    --jobs 90

  python summarize_higher_level_behavior_info.py \
    --base_dir "$folder" \
    --out_dir "${folder}analysis_results_extended/" \
    --jobs 90

  python summarize_behavior_netState_df_mix.py \
    --base_dir "$folder" \
    --out_dir "${folder}analysis_results/" \
    --jobs 90
done

python merge_behavior_df.py
python eventwise_decode_binned.py

bash run_plot_neural_summary.sh