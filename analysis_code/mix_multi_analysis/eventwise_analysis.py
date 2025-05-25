import os
import glob
import re
from typing import Dict, List, Tuple
import concurrent.futures
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from joblib import Parallel, delayed
import json
import pickle
from group_analysis_utils.helper import parse_agent_roles
from summarize_RF_bv import parse_agent_roles
def process_metrics_file(session_name: str,
                         event_dir: str,
                         timesteps_dir: str,
                         network_dir: str,
                         ) -> List[Dict]:
  """
  Load one *_metrics.pkl (output of `process_folder`), compute per-agent
  means / counts for **all** requested metrics and return a list of
  records (one per agent).
  """
  roles, sources = parse_agent_roles(session_name)
  ev_df = pd.read_pickle(os.path.join(event_dir, session_name + '_merged.pkl'))          # ← list-of-dict rows
  ts_path = os.path.join(timesteps_dir, session_name + '*_info.csv')
  assert glob.glob(ts_path), f"Missing timesteps file for {session_name}"
  ts_df = pd.read_csv(glob.glob(ts_path)[0])
  # load network states
  ns_path = os.path.join(network_dir, session_name + '*_network_states.pkl')
  assert glob.glob(ns_path), f"Missing network states file for {session_name}"
  ns_df = pd.read_pickle(glob.glob(ns_path)[0])
  print(f"Loaded {session_name}")
  # Now, check the event mean of distance and the mean of distance per agents
  events = [ev for ev in ev_df.columns if 'event' in ev]
  # events = ['apple_cooperation_events', 'distraction_events', 'fence_events',
  #  'invalid_events', 'fencing_events']
  apple_ev = 'apple_cooperation_events'
  apple_event = ev_df[apple_ev].values

  agent_positions_list = []
  agent_death_mark = []

  for agent_id in range(len(roles)):
    agent_death_mark.append(ts_df[f'death_{agent_id}'].values)
    x_col = f'POSITION_x_{agent_id}'
    y_col = f'POSITION_y_{agent_id}'

    # Ensure the columns exist before attempting to select them
    if x_col in ts_df.columns and y_col in ts_df.columns:
      # Stack the x and y columns side-by-side for this agent
      agent_pos = ts_df[[x_col, y_col]].values  # .values converts to NumPy array
      agent_positions_list.append(agent_pos)
    else:
      print(f"Warning: Missing position columns for agent {agent_id}. Skipping.")
      # Handle missing data as appropriate, e.g., append zeros or NaNs
      agent_positions_list.append(np.full((len(ts_df), 2), np.nan))  # Append NaNs
  agent_positions = np.array(agent_positions_list)
  agent_death_mark = np.array(agent_death_mark).T

  episode_length = 1000
  # apple_coop_ev_mask = np.zeros((max(agent_death_mark.shape), len(roles))).astype(bool)
  # distraction_ev_mask = np.zeros((max(agent_death_mark.shape), len(roles))).astype(bool)
  # fence_ev_mask = np.zeros((max(agent_death_mark.shape), len(roles))).astype(bool)
  #
  # ---------- pre‑amble ----------
  T, N_agents = agent_death_mark.shape
  episode_length = 1000  # keep if you really have fixed‑length episodes

  # create an empty mask for every event you’re interested in
  events_of_interest = ['apple_cooperation_events', 'distraction_events', 'fence_events']
  event_mask_dict = {ev: np.zeros((T, N_agents), dtype=bool) for ev in events_of_interest}

  # ---------- build the masks ----------
  for event in events_of_interest:
    for ep, data_ep in enumerate(ev_df[event].values):
      if not data_ep:  # empty list
        continue
      offset = ep * episode_length  # =0 if your times are absolute
      for entry in data_ep:
        participants = entry['participants']
        if 'time_start' in entry:  # time span
          s, e = entry['time_start'], entry['time_end']
          event_mask_dict[event][offset + s: offset + e, participants] = True
        else:  # single‑time event
          t = entry['time']
          event_mask_dict[event][offset + t, participants] = True

  # ---------- pairwise distances ----------
  dist_dict = {}
  for i in range(N_agents):
    for j in range(i + 1, N_agents):
      dij = np.linalg.norm(agent_positions[i] - agent_positions[j], axis=1)
      # knock out dead periods
      alive = (agent_death_mark[:, i] == 1) & (agent_death_mark[:, j] == 1)
      dij[~alive] = np.nan
      dist_dict[(i, j)] = dij

  # ---------- plotting ----------
  fig, axes = plt.subplots(N_agents, 1, figsize=(10, 3 * N_agents),
                           sharex=True, sharey=True)
  axes = np.atleast_1d(axes)  # ensures axes is always an array

  pair_colors = {'predator': 'red', 'prey': 'blue'}
  event_shades = {'apple_cooperation_events': 'limegreen',
                  'distraction_events': 'orange',
                  'fence_events': 'purple'}

  x_lim = (1000,2000)  # or (0, T) (1000, 2000) if you want a window

  for i, ax in enumerate(axes):
    ax.set_title(f"Distances for agent {i} ({roles[i]})")

    # plot every pair that involves agent i
    for (a, b), dist in dist_dict.items():
      if i in (a, b):
        other = b if i == a else a
        color = pair_colors[roles[other]]
        ax.plot(dist, label=f"{roles[a]}–{roles[b]}", color=color, lw=0.8)

    # raster‑style shading for each event
    for ev, mask in event_mask_dict.items():
      # True → False transitions give segment ends, False → True give starts
      starts = np.where(np.diff(mask[:, i].astype(int), prepend=0) == 1)[0]
      ends = np.where(np.diff(mask[:, i].astype(int), prepend=0) == -1)[0]

      # if an event runs all the way to T it has no trailing -1 diff; fix that
      if len(ends) < len(starts):
        ends = np.append(ends, T)

      for s, e in zip(starts, ends):
        ax.axvspan(s, e, color=event_shades[ev], alpha=0.25, zorder=0)

    ax.set_xlim(x_lim)
    ax.set_ylabel("Distance (px)")
    ax.legend(fontsize='small', ncol=2, loc='upper right')

  axes[-1].set_xlabel("Timestep")
  plt.tight_layout()

  from matplotlib.patches import Patch

  # ---------- one legend for all raster shades ----------
  from matplotlib.patches import Patch

  event_handles = [
    Patch(facecolor=event_shades[ev], alpha=0.25,
          label=ev.replace('_', ' ').title())
    for ev in events_of_interest
  ]

  fig.legend(
    handles=event_handles,  # <-- say these are handles
    labels=[h.get_label() for h in event_handles],  # and these are labels
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),  # adjust vertical gap as needed
    ncol=len(event_handles),
    frameon=False
  )

  plt.show()

  # We first verify the difference

def load_and_process_metrics(event_dir: str,
                            timesteps_dir: str,
                            network_dir: str,
                            parallel_loading=False) -> pd.DataFrame:
  """
  Parallel load and preprocess all *_metrics.pkl in event_dir.
  """
  paths = glob.glob(os.path.join(event_dir, '*.pkl'))
  paths = sorted(paths)
  session_names = [re.sub(r'.*/', '', path).replace('_merged.pkl', '') for path in paths]

  fn = partial(process_metrics_file, event_dir=event_dir, timesteps_dir=timesteps_dir, network_dir=network_dir)
  all_events: List[Dict] = []
  if parallel_loading:
    with concurrent.futures.ProcessPoolExecutor() as executor:
      for recs in executor.map(fn, session_names):
        pass
        # all_events.extend(recs)
  else:
    for session in session_names:
      recs = fn(session)
      # all_events.extend(recs)

  agent_df = pd.DataFrame(all_events)

  agent_df['reward_ref'] = agent_df.apply(
    lambda row: row['reward'] / (row['group_size'][1] if row['role']=='predator' else 1),
    axis=1
  )
  agent_df['group_label'] = agent_df['group_size'].apply(lambda g: f"{g[0]}p,{g[1]}r")
  return agent_df


event_metric_dir = '/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/analysis_results_merged/'
timesteps_dir = '/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/analysis_results/'
network_dir = '/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/analysis_results/'
METRICS = {
  'prey': [
    # 'move', 'rot', 'acorn', 'apple', 'frac_off', 'death',
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
    # 'move', 'rot', 'catch', 'death',
    'good_gathering',
    'gathering_rates',  # NEW
    'sync_to_same_role', 'sync_to_other_role',  # NEW
    'shapley_eff', 'shapley_sus',  # NEW
    'interact_count', 'invalid_interact_count'
  ]
}

if __name__ == "__main__":
  # Load and process metrics
  agent_df = load_and_process_metrics(event_metric_dir, timesteps_dir, network_dir, parallel_loading=False)


