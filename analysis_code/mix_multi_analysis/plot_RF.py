 #!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Add your util path so imports work
sys.path.append("group_analysis_utils")

from plot_behavior_info_mix_higher_level import load_and_process_metrics

# ─── Configuration ───────────────────────────────────────────────────────────
MERGED_DIR = "../../results/mix_RF2/analysis_results_merged"
figure_dir = './RF2_figures'
METRICS = {
    'prey': [
        'move', 'rot', 'acorn', 'apple', 'frac_off', 'death',
        # 'good_gathering',  # old alias
        'gathering_rates',  # NEW
        'sync_to_same_role', 'sync_to_other_role',  # NEW
        'shapley_eff', 'shapley_sus',  # NEW
        'apple_coop_cnt',  # NEW
        'help_in_fence_cnt', 'helped_in_fence_cnt',  # NEW
        'help_in_distraction_cnt', 'helped_in_distraction_cnt',  # NEW
        'fence_count', 'dist_to_pred'
    ],
    'predator': [
        'move', 'rot', 'catch', 'death',
        'good_gathering',
        'gathering_rates',  # NEW
        'sync_to_same_role', 'sync_to_other_role',  # NEW
        'shapley_eff', 'shapley_sus',  # NEW
        'interact_count', 'invalid_interact_count'
    ]
}
# ─────────────────────────────────────────────────────────────────────────────

def extract_ckpt_from_source(source: str) -> int:
    """Extract checkpoint number from the 'source' string, e.g., '..._ckpt42'."""
    m = re.search(r"ckpt_(\d+)", source)
    return int(m.group(1)) if m else -1

def main():
    # 1. Load all metrics into a DataFrame with 'ckpt' column
    df = load_and_process_metrics(MERGED_DIR, parallel_loading=True)
    # zap “ckpt” plus the digits that follow it
    df['source'] = df['source'].str.replace(r'ckpt\d+', '', regex=True)

    os.makedirs(figure_dir, exist_ok=True)
    if 'ckpt' not in df.columns:
        df['ckpt'] = df['source'].apply(extract_ckpt_from_source)

    # 2. For each role and each metric, plot one line per agent
    for role, metrics in METRICS.items():
        df_role = df[df['role'] == role]
        sources = sorted(df_role['source'].unique())

        for metric in metrics:
            plt.figure(figsize=(8, 5))
            for src in sources:
                sub = df_role[df_role['source'] == src].sort_values('ckpt')
                if metric not in sub.columns:
                    continue
                plt.plot(sub['ckpt'], sub[metric], marker='o', label=src)

            plt.title(f"{role.capitalize()} — {metric} per Agent over Checkpoints")
            plt.xlabel("Checkpoint")
            plt.ylabel(metric)
            plt.legend(loc='best', fontsize='small', ncol=2)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(figure_dir, f"{role}_{metric}_per_agent.png"))
            plt.show()

if __name__ == "__main__":
    main()
