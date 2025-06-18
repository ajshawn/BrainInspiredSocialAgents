#!/bin/bash

python plot_eventwise_neural_stats_training_ckpts.py \
--summary_input_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/analysis_results_event_stats/ \
--bv_input_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_20250305_2v4_rollouts/analysis_results_merged/ \
--figures_output_dir ./figure_neural_output_orchard_50_bin10 \
--x_lim_min 0 --x_lim_max 50

python plot_eventwise_neural_stats_training_ckpts.py \
--summary_input_dir /home/mikan/e/GitHub/social-agents-JAX/results/RandomForest20250424_2v4_rollouts/analysis_results_event_stats_bin10/ \
--bv_input_dir /home/mikan/e/GitHub/social-agents-JAX/results/RandomForest20250424_2v4_rollouts/analysis_results_merged/ \
--figures_output_dir ./figure_neural_output_random_forest_50_bin10 \
--x_lim_min 0 --x_lim_max 50

python plot_eventwise_neural_stats_training_ckpts.py \
--summary_input_dir /home/mikan/e/GitHub/social-agents-JAX/results/open_no_group_20250529_2v4_rollouts/analysis_results_event_stats_bin10/ \
--bv_input_dir /home/mikan/e/GitHub/social-agents-JAX/results/open_no_group_20250529_2v4_rollouts/analysis_results_merged/ \
--figures_output_dir ./figure_neural_output_open_no_group_50_bin10 \
--x_lim_min 0 --x_lim_max 50
python plot_eventwise_neural_stats_training_ckpts.py \
--summary_input_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_no_group_20250529_2v4_rollouts/analysis_results_event_stats_bin10/ \
--bv_input_dir /home/mikan/e/GitHub/social-agents-JAX/results/orchard_no_group_20250529_2v4_rollouts/analysis_results_merged/ \
--figures_output_dir ./figure_neural_output_orchard_no_group_50_bin10 \
--x_lim_min 0 --x_lim_max 50