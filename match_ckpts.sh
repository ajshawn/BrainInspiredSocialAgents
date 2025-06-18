#!/bin/bash
python_file='./marl/utils/extract_episodes_to_csv.py'
checkpoints_dir=('results/PopArtIMPALA_42_meltingpot_predator_prey__open_2025-06-10_14:50:02.935391')
checkpoints_dir+=('results/PopArtIMPALA_1_meltingpot_predator_prey__random_forest_2025-04-24_17:23:56.924098'
                 'results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274')
for exp in "${checkpoints_dir[@]}"; do
  if [ ! -d "$exp" ]; then
    echo "Directory $exp does not exist." >&2
    continue
  fi

  counter_dir="$exp/checkpoints/counter"
  actor_csv="$exp/csv_logs/actor.csv"

  if [[ -d "$counter_dir" && -f "$actor_csv" ]]; then
    echo "→ Processing experiment: $exp"
    python "$python_file" --exp_dir "$exp"
  else
    echo "⚠️ Skipping $exp: missing checkpoints/counter or csv_logs/actor.csv" >&2
  fi
done


for exp in results/*radius_0*; do
  if [ ! -d "$exp" ]; then
    # Handle cases where the glob doesn't match anything and returns the pattern itself
    if [[ "$exp" == "results/*radius_0*" ]]; then
        echo "No directories found matching 'results/*radius_0*'" >&2
        break
    fi
  fi

  counter_dir="$exp/checkpoints/counter"
  actor_csv="$exp/csv_logs/actor.csv"

  if [[ -d "$counter_dir" && -f "$actor_csv" ]];
  then # <--- 'then' on a new line
    echo "→ Processing experiment: $exp"
    python "$python_file" --exp_dir "$exp"
  else
    echo "⚠️ Skipping $exp: missing checkpoints/counter or csv_logs/actor.csv" >&2
  fi
done

python match_ckpts_plots.py --exp_dirs \
results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__alley_hunt_2025-06-15_10:11:40.406930 \
results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__open_2025-06-12_15:49:53.201394 \
results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__orchard_2025-06-15_10:11:40.382778 \
results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__random_forest_2025-06-12_15:49:53.295564 \
results/PopArtIMPALA_1_meltingpot_predator_prey__random_forest_2025-04-24_17:23:56.924098 \
results/PopArtIMPALA_42_meltingpot_predator_prey__open_2025-06-10_14:50:02.935391 \
results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274