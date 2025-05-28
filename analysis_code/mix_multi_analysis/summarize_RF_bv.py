#!/usr/bin/env python3
import os
import re
import logging
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

# ————————————————————————————————————————————————————————————— #
# Assume these functions are imported or defined above:
# parse_agent_roles, mark_death_periods, get_safe_grass_locations,
# compute_agent_move_distance, compute_num_rotation,
# compute_collect_counts, compute_death_counts, compute_stuck_rate_all,
# compute_grass_time_per_life, compute_pairwise_distance_mean
# ————————————————————————————————————————————————————————————— #
from analysis_code.mix_multi_analysis.group_analysis_utils.basic_behavioral_metrics import *

def process_pair_folder(folder_path: str) -> pd.DataFrame:
    folder = Path(folder_path)
    base = folder.name.split('predator_prey__')[0]
    role, src = parse_agent_roles(base)
    predators = [i for i, r in role.items() if r == 'predator']
    preys     = [i for i, r in role.items() if r == 'prey']

    pkl_dir = folder / "episode_pickles"
    files = sorted(pkl_dir.glob("*.pkl"))
    if not files:
        logging.warning("No pickle files in %s", pkl_dir)
        return None

    # load safe grass once
    safe = get_safe_tiles(base, map="smaller_13x13")

    rows = []
    for pkl_file in files:
        with open(pkl_file, "rb") as fp:
            episode = pickle.load(fp)

        pos = np.array([d["POSITION"]    for d in episode])
        ori = np.array([d["ORIENTATION"] for d in episode])
        act = np.array([d["actions"]     for d in episode])
        rew = np.array([d["rewards"]     for d in episode])
        sta = np.array([d["STAMINA"]     for d in episode])

        death_lbl = {i: mark_death_periods(sta[:, i]) for i in range(sta.shape[1])}

        # compute per-episode metrics
        move_dist = compute_agent_move_distance(pos)
        rotation  = compute_num_rotation(ori)
        apple, acorn, catch = compute_collect_counts(rew, predators, preys)
        death_cnt = compute_death_counts(sta)
        stuck_rate = compute_stuck_rate_all(pos, death_labels=death_lbl)

        on_t, off_t, frac_off = compute_grass_time_per_life(
            pos, death_lbl, preys, safe
        )
        pair_dist = compute_pairwise_distance_mean(pos, death_lbl)

        # assemble one row
        row = {
            "trial_name": base,
            "role": role,
            "source": src,
            "episode": pkl_file.stem,
            **{f"move_{i}": move_dist[i] for i in move_dist},
            **{f"rot_{i}": rotation[i]   for i in rotation},
            **{f"apple_{i}": apple[i]     for i in apple},
            **{f"acorn_{i}": acorn[i]     for i in acorn},
            **{f"catch_{i}": catch[i]     for i in catch},
            **{f"death_{i}": death_cnt[i] for i in death_cnt},
            **{f"stuck_{i}": stuck_rate[i] for i in stuck_rate},
            **{f"on_grass_{i}": on_t[i]   for i in on_t},
            **{f"off_grass_{i}": off_t[i] for i in off_t},
            **{f"frac_off_{i}": frac_off[i] for i in frac_off},
            **pair_dist,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def process_and_save(folder_path: str, out_dir: str):
    df = process_pair_folder(folder_path)
    if df is None or df.empty:
        return

    # Extract checkpoint number N from parent dir name "mix_RF_ckpt<N>"
    parent = Path(folder_path).parent.name
    m = re.match(r"mix_RF_ckpt(\d+)", parent)
    ckpt_num = m.group(1) if m else "unknown"

    # Build output filenames
    csv_path = Path(out_dir) / f"ckpt_{ckpt_num}_metrics.csv"
    pkl_path = Path(out_dir) / f"ckpt_{ckpt_num}_metrics.pkl"

    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    print(f"Saved: {csv_path.name}, {pkl_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_dir",
        default="../../results/mix_RF3",
        help="Root dir containing mix_RF_ckpt*/<run>/episode_pickles"
    )
    ap.add_argument(
        "--jobs", "-j",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel jobs"
    )
    args = ap.parse_args()

    base = Path(args.base_dir)
    out_dir = base / "analysis_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # discover all run subfolders under mix_RF_ckpt*
    run_folders = []
    for ckpt_dir in base.glob("mix_RF_ckpt*"):
        for run_sub in ckpt_dir.iterdir():
            if (run_sub / "episode_pickles").is_dir():
                run_folders.append(str(run_sub))

    if not run_folders:
        print("No valid run folders found under", base)
        return

    Parallel(n_jobs=args.jobs)(
        delayed(process_and_save)(rf, str(out_dir))
        for rf in sorted(run_folders)
    )
    # dfs = []
    # for rf in sorted(run_folders):
    #     try:
    #         df = process_pair_folder(rf)
    #         if df is not None and not df.empty:
    #             dfs.append(df)
    #             process_and_save(rf, str(out_dir))
    #     except Exception as e:
    #         logging.error(f"Error processing folder {rf}: {e}")

    print("All folders processed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
