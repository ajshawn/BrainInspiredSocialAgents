from summarize_higher_level_behavior_info_2 import *

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
# your existing analysis imports (assumed available)
# from your_module import (
#   parse_agent_roles,
#   get_safe_tiles, get_apple_tiles, get_acorn_tiles,
#   mark_death_periods, segment_active_phases,
#   apple_periods_segmented, detect_distraction_events,
#   compute_good_gathering_segmented,
#   compute_successful_fencing_and_helpers_segmented,
#   compute_invalid_interactions_segmented,
#   analyse_altruism_events,
#   analyse_events_with_sync,
#   compute_shapley_values_for_segments,
#   compute_episode_sync, compute_segment_sync,
#   PARAMS
# )
# ————————————————————————————————————————————————————————————— #

def process_folder(run_folder: str, out_dir: str):
    run_folder = Path(run_folder)
    pkl_dir = run_folder / "episode_pickles"
    if not pkl_dir.exists():
        logging.error("Pickle directory not found: %s", pkl_dir)
        return None

    files = sorted(pkl_dir.glob("*.pkl"))
    if not files:
        logging.error("No pickle files found in %s", pkl_dir)
        return None

    # infer ckpt number N from the parent of run_folder
    # e.g. run_folder = .../mix_RF_ckpt42/<run-subdir>/episode_pickles
    ckpt_dir = run_folder.parent  # .../mix_RF_ckpt42/<run-subdir>
    m = re.search(r"mix_RF_ckpt(\d+)", ckpt_dir.name)
    if not m:
        logging.error("Can't parse checkpoint number from %s", ckpt_dir)
        return None
    ckpt_num = m.group(1)

    # parse roles & source from the run_subdir name
    role, src = parse_agent_roles(run_folder.name)
    predators = [i for i, r in role.items() if r == 'predator']
    preys = [i for i, r in role.items() if r == 'prey']

    # get map‐specific arrays once
    grass  = get_safe_tiles(run_folder.name, map="smaller_13x13")
    apples = get_apple_tiles(run_folder.name, map="smaller_13x13")
    acorns = get_acorn_tiles(run_folder.name, map="smaller_13x13")  # if implemented

    rows = []
    for pkl in files:
        epi = pickle.load(open(pkl, "rb"))
        pos = np.array([d["POSITION"]    for d in epi])
        ori = np.array([d["ORIENTATION"] for d in epi])
        act = np.array([d["actions"]     for d in epi])
        rew = np.array([d["rewards"]     for d in epi])
        sta = np.array([d["STAMINA"]     for d in epi])
        death = {i: mark_death_periods(sta[:, i]) for i in range(pos.shape[1])}

    # Segment an episode into active phases, where at least one prey is alive
    active_segments = segment_active_phases({i: death[i] for i in preys},
                                            min_prey_alive=PARAMS.min_prey_alive,
                                            min_all_dead_duration=PARAMS.min_all_dead_duration)

    apple_cooperation_events = apple_periods_segmented(
      pos, rew, death, preys, grass,
      active_segments, cluster_radius=3, apple_period_min_len=5, apple_period_gap=5,
      participation_threshold=0.4
    )

    distraction_events = detect_distraction_events(
      pos, ori, death, grass, predators, preys,
      active_segments, window_away=5, shift_window=5, distraction_period_gap=2,
      radius=3.0, move_thresh=0.5,
      max_period_duration=30,  # Changed hard_cap -> max_period_duration
      scenario='grass',  # or 'chase'
    ) + detect_distraction_events(
      pos, ori, death, grass, predators, preys,
      active_segments, window_away=5, shift_window=5, distraction_period_gap=2,
      radius=3.0, move_thresh=0.5,
      max_period_duration=30,  # Changed hard_cap -> max_period_duration
      scenario='chase',  # or 'grass'
    )

    gathering_rates_ep = compute_good_gathering_segmented(
      pos, predators, preys, [(0, 1000)], radius=3, death_labels=death
    )

    gathering_rates_seg = compute_good_gathering_segmented(
      pos, predators, preys, active_segments, radius=3, death_labels=death
    )

    fence_events = mark_events_successful_fencing_and_helpers_segmented(
      pos, ori, act, rew, predators, preys, death,
      active_segments=active_segments, radius=3,
    )

    invalid_events = mark_events_compute_invalid_interactions_segmented(
      act, rew, pos, ori, sta,
      predators, preys, active_segments=active_segments,
    )

    for ev in fence_events:
      ev["type"] = "fence"
      # helpers already present, prey being helped:
      ev["beneficiaries"] = [ev["beneficiary"]] if type(ev["beneficiary"]) == int else ev["beneficiary"]
      ev["helpers"] = [ev["helper"]] if type(ev["helper"]) == int else ev["helper"]
      ev["participants"] = sorted(np.hstack([ev['helper'], ev['beneficiary'], ev['predator']]).astype(int).tolist())
      ev["time_start"] = max(0, ev["time"] - 10)
      ev["time_end"] = min(len(pos), ev["time"] + 10)

    for ev in distraction_events:
      ev["type"] = "distraction"
      ev["helpers"] = [ev["helper"]]  if type(ev["helper"]) == int else ev["helper"]
      ev["beneficiaries"] = [ev["beneficiary"]]  if type(ev["beneficiary"]) == int else ev["beneficiary"]
      ev["participants"] = sorted(np.hstack([ev['helper'], ev['beneficiary'], ev['predator']]).astype(int).tolist())
      # time_start/shift/end already there

    for ev in apple_cooperation_events:
      ev["type"] = "apple_cooperation"

    # 3) Add altruism metrics to distraction and fence events
    distraction_events, _ = analyse_altruism_events(
      distraction_events, positions=pos, stamina=sta, death_labels=death, safe_grass=grass,
    )
    fence_events, _ = analyse_altruism_events(
      fence_events, positions=pos, stamina=sta, death_labels=death, safe_grass=grass,
    )

    #4) Add sync metrics to apple cooperation, distraction and fence events
    apple_cooperation_events = analyse_events_with_sync(
      apple_cooperation_events, pos, act, death, action_smooth_window=3, weight_spatial=0.5,
    )
    distraction_events = analyse_events_with_sync(
      distraction_events, pos, act, death, action_smooth_window=3, weight_spatial=0.5,
    )
    fence_events = analyse_events_with_sync(
      fence_events, pos, act, death, action_smooth_window=3, weight_spatial=0.5,
    )

    # 5) compute Shapley values
    shapley_metrics_ep = compute_shapley_values_for_segments(
      episode_rewards=rew,
      role_map=role,
      active_segments=[(0, 1000)],
      death_labels=death,
      MCPerm=50,
    )

    shapley_mertrics_seg = compute_shapley_values_for_segments(
      episode_rewards=rew,
      role_map=role,
      active_segments=active_segments,
      death_labels=death,
      MCPerm=50,
    )

    # 6) compute sync
    sync_score_ep = compute_episode_sync(
      pos, act, death,
      action_smooth_window=3,
      weight_spatial=0.5,
    )

    sync_score_seg = compute_segment_sync(
      pos, act, death,
      active_segments=active_segments,
      action_smooth_window=3,
      weight_spatial=0.5,
    )

    # Merge all metrics together
    metrics = {
      'trial_name': os.path.basename(run_folder),
      'role': role,
      'source': src,
      'episode': pkl.name.replace('.pkl', '').split('_')[-1],
      # First start with episode-level metrics
      'gathering_rates': gathering_rates_ep,
      'sync_score': sync_score_ep,
      'shapley_metrics': shapley_metrics_ep,
      # Then add segment-level metrics
      'segment_metrics':{
        'active_segments': active_segments,
        'gathering_rates': gathering_rates_seg,
        'sync_score': sync_score_seg,
        'shapley_metrics': shapley_mertrics_seg,
      },
      'apple_cooperation_events': apple_cooperation_events,
      'distraction_events': distraction_events,
      'fence_events': fence_events,
      'invalid_events': invalid_events,
    }
    rows.append(metrics)

    # write out a per-ckpt pickle
    out_pkl = Path(out_dir) / f"ckpt_{ckpt_num}_higher_level_metrics.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(rows, f)

    # return a DataFrame for optional aggregation
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_dir",
        default="../../results/mix_RF3",
        help="Root dir containing mix_RF_ckpt*/…/episode_pickles"
    )
    ap.add_argument(
        "--jobs", "-j", type=int, default=8,
        help="Number of parallel jobs"
    )
    args = ap.parse_args()

    base = Path(args.base_dir)
    out_dir = base / "analysis_results_extended"
    out_dir.mkdir(exist_ok=True, parents=True)

    # find all run folders that contain episode_pickles
    run_folders = []
    for ckpt_dir in base.glob("mix_RF_ckpt*"):
        for run_sub in ckpt_dir.iterdir():
            if (run_sub / "episode_pickles").is_dir():
                run_folders.append(str(run_sub))

    if not run_folders:
        print("No run folders found under", base)
        return

    # parallel processing
    dfs = Parallel(n_jobs=args.jobs)(
        delayed(process_folder)(rf, str(out_dir))
        for rf in sorted(run_folders)
    )
    # dfs = []
    # for rf in sorted(run_folders):
    #     df = process_folder(rf, str(out_dir))
    #     if df is not None:
    #         dfs.append(df)

    # optional: aggregate all into one CSV
    all_dfs = [df for df in dfs if df is not None]
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(out_dir / "higher_level_metrics_combined.csv", index=False)
        print("Wrote combined CSV with", len(combined), "rows to", out_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
