import os
import pandas as pd
import numpy as np
import pickle
import argparse
import logging
from joblib import Parallel, delayed
from group_analysis_utils.helper import parse_agent_roles
from group_analysis_utils.constants import PARAMS
from group_analysis_utils.map_utils import get_safe_tiles, get_apple_tiles, get_acorn_tiles
from group_analysis_utils.segmentation import segment_active_phases, mark_death_periods
from group_analysis_utils.cooperation import apple_periods_segmented
from group_analysis_utils.distraction import detect_distraction_events
from group_analysis_utils.gather_and_fence import (compute_good_gathering_segmented,
                                                   mark_events_successful_fencing_and_helpers_segmented,
                                                   mark_events_compute_invalid_interactions_segmented)
from group_analysis_utils.sync import compute_episode_sync, compute_segment_sync, analyse_events_with_sync
from group_analysis_utils.altruism import analyse_altruism_events
from group_analysis_utils.shapley_coalition import compute_shapley_values_for_segments



# + other imports from summarize_behavior_info_mix and altruism modules …
# from summarize_behavior_info_mix import *

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


def process_folder(folder, out_dir='../../results/mix_2_4/analysis_results_extended'):
  # Check the existence of the episode_pickles directory
  try:
    pkl_dir = os.path.join(folder, 'episode_pickles')
    assert os.path.exists(pkl_dir), f"Pickle directory not found: {pkl_dir}"
    files = sorted(f for f in os.listdir(pkl_dir) if f.endswith('.pkl'))
    assert files, f"No pickle files found in {pkl_dir}"
  except AssertionError as e:
    logging.error(e)
    return None

  role, src = parse_agent_roles(os.path.basename(folder))
  predators = [i for i, r in role.items() if r == 'predator']
  preys = [i for i, r in role.items() if r == 'prey']

  grass = get_safe_tiles(folder, map='smaller_13x13')
  apples = get_apple_tiles(folder, map='smaller_13x13')
  acorns = get_acorn_tiles(folder, map='smaller_13x13') # TODO: we did not yet implement acorn related analysis

  rows = []
  for pkl in files:
    epi = pickle.load(open(os.path.join(folder, 'episode_pickles', pkl), 'rb'))
    pos = np.array([d['POSITION'] for d in epi])
    ori = np.array([d['ORIENTATION'] for d in epi])
    act = np.array([d['actions'] for d in epi])
    rew = np.array([d['rewards'] for d in epi])
    sta = np.array([d['STAMINA'] for d in epi])
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
      'trial_name': os.path.basename(folder),
      'role': role,
      'source': src,
      'episode': pkl.replace('.pkl', ''),
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


  # Pickle output
  base = os.path.basename(folder).split('predator_prey__')[0]
  out_pkl = os.path.join(out_dir, base + '_higher_level_metrics.pkl')
  with open(out_pkl, 'wb') as f:
    pickle.dump(rows, f)

  return pd.DataFrame(rows) if rows else None


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--base_dir', default='/home/mikan/e/GitHub/social-agents-JAX/results/RandomForest20250424_2v4_rollouts/',)
  p.add_argument('--out_dir', default='/home/mikan/e/GitHub/social-agents-JAX/results/RandomForest20250424_2v4_rollouts/analysis_results_extended/',
                  help='Output directory for the analysis results')
  p.add_argument('--jobs', type=int, default=50)
  args = p.parse_args()

  out = args.out_dir
  os.makedirs(out, exist_ok=True)
  folders = [os.path.join(args.base_dir, d) for d in os.listdir(args.base_dir)
             if os.path.isdir(os.path.join(args.base_dir, d)) and 'episode_pickles' in os.listdir(
      os.path.join(args.base_dir, d))]
  if not folders:
    folders = [
      os.path.join(args.base_dir, d1, d2)  # <-- full path we keep
      for d1 in os.listdir(args.base_dir)  # level‑1 dirs
      for d2 in os.listdir(os.path.join(args.base_dir, d1))  # level‑2 dirs
      if os.path.isdir(os.path.join(args.base_dir, d1, d2)) and
         'episode_pickles' in os.listdir(os.path.join(args.base_dir, d1, d2))
    ]

  folders = sorted(folders)
  process = delayed(process_folder)
  dfs = Parallel(n_jobs=args.jobs)(
    process(folder, out) for folder in folders
  )
  # dfs = []
  # for folder in folders:
  #   try:
  #     df = process_folder(folder, out)
  #     if df is not None:
  #       dfs.append(df)
  #   except Exception as e:
  #     logging.error(f"Error processing folder {folder}: {e}")

  if dfs:
    final = pd.concat(dfs, ignore_index=True)
    final.to_csv(os.path.join(out, 'higher_level_metrics_combined.csv'), index=False)
    logging.info("Saved aggregated CSV with %d segment rows", len(final))
  else:
    logging.warning("No data extracted!")


if __name__ == "__main__":
  main()
