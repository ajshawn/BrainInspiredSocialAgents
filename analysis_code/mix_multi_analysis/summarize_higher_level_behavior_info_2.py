import os
import pandas as pd
import numpy as np
import pickle
import argparse
import logging
from joblib import Parallel, delayed

from group_analysis_utils.constants import PARAMS
from group_analysis_utils.map_utils import get_safe_tiles, get_apple_tiles
from group_analysis_utils.segmentation import segment_active_phases
from group_analysis_utils.cooperation import apple_frames, apple_periods
from group_analysis_utils.sync import add_event_sync, pairwise_cosine
# + other imports from summarize_behavior_info_mix and altruism modules …
from summarize_behavior_info_mix import *

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


def process_folder(folder):
  role, _ = parse_agent_roles(os.path.basename(folder))
  preds = [i for i, r in role.items() if r == 'predator']
  preys = [i for i, r in role.items() if r == 'prey']

  grass = get_safe_tiles(folder);
  apples = get_apple_tiles(folder)
  rows = []
  for pkl in sorted(os.listdir(os.path.join(folder, 'episode_pickles'))):
    epi = pickle.load(open(os.path.join(folder, 'episode_pickles', pkl), 'rb'))
    pos = np.array([d['POSITION'] for d in epi])
    ori = np.array([d['ORIENTATION'] for d in epi])
    rew = np.array([d['rewards'] for d in epi])
    sta = np.array([d['STAMINA'] for d in epi])
    death = {i: mark_death_periods(sta[:, i]) for i in range(pos.shape[1])}

    segs = segment_active_phases({i: death[i] for i in preys})
    for seg_i, (s, e) in enumerate(segs):
      # coalition – apples
      ap_frames = apple_frames(pos, rew, death, preys, grass, s, e)
      ap_period = apple_periods(pos, rew, preys, s, e)
      add_event_sync(ap_period, pos)

      baseline_sync = pairwise_cosine(
        np.diff(pos[s:e, preys], axis=0, prepend=pos[s:s + 1, preys]))

      rows.append(dict(
        folder=os.path.basename(folder), episode=pkl.replace('.pkl', ''),
        seg=seg_i, t0=s, t1=e,
        n_apple_frames=len(ap_frames),
        n_apple_periods=len(ap_period),
        mean_period_sync=np.nanmean([ev['sync'] for ev in ap_period]) \
          if ap_period else np.nan,
        baseline_sync=baseline_sync
      ))
  return pd.DataFrame(rows) if rows else None


def main():
  ap = argparse.ArgumentParser();
  ap.add_argument('--base_dir', required=True);
  ap.add_argument('--jobs', type=int, default=1)
  args = ap.parse_args()

  out = os.path.join(args.base_dir, 'analysis_results_extended');
  os.makedirs(out, exist_ok=True)
  folders = [os.path.join(args.base_dir, d) for d in os.listdir(args.base_dir)
             if os.path.isdir(os.path.join(args.base_dir, d)) and 'episode_pickles' in os.listdir(
      os.path.join(args.base_dir, d))]
  process = delayed(process_folder)
  dfs = Parallel(args.jobs)(process(f) for f in folders) if args.jobs > 1 else [process_folder(f) for f in folders]
  dfs = [d for d in dfs if d is not None]
  if dfs:
    final = pd.concat(dfs, ignore_index=True)
    final.to_csv(os.path.join(out, 'all_segments.csv'), index=False)
    logging.info("Saved aggregated CSV with %d segment rows", len(final))
  else:
    logging.warning("No data extracted!")


if __name__ == "__main__":
  main()
