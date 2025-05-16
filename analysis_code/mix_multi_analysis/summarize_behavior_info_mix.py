#!/usr/bin/env python3
"""

analyze_mixed_results_flexible.py

General-purpose analysis for *N* predators vs *M* preys.

Defines helper functions for per-agent and per-pair metrics, and
`process_pair_folder` to load episodes, compute all metrics, and return
a DataFrame. `main()` scans a base directory of pair folders and
outputs one CSV per pair.
"""
import os
import re
import pickle
import numpy as np
import pandas as pd
import importlib
from joblib import Parallel, delayed

# --- Helper functions ---
def parse_agent_roles(base_name: str):
  # find all '<model>_pred_<idx>' and '<model>_prey_<idx>' in order
  agent_specs = re.findall(r'([A-Za-z0-9]+_pre[a-z]_\d+)', base_name)
  role, source = {}, {}
  for i, spec in enumerate(agent_specs):
    role[i] = "predator" if "pred" in spec else "prey"
    source[i] = spec
  return role, source


def mark_death_periods(data, respawn_cooldown=100):
  data = np.asarray(data, dtype=float)
  n = len(data)

  # Initialize all as 'living' (1)
  labels = np.ones(n, dtype=int)

  # Create a boolean mask of zeros
  zero_mask = (data == 0.0)

  # Convolve this boolean mask with a window of length 20
  # conv[i] -> number of zeros in data[i : i+20]
  conv = np.convolve(zero_mask, np.ones(respawn_cooldown, dtype=int), mode='valid')

  # Indices i for which data[i : i+20] are all zeros => conv[i] == 20
  potential_starts = np.where(conv == respawn_cooldown)[0]

  # Check if the element right after those 20 zeros is 1.0
  for start_idx in potential_starts:
    # The index of the element that should be 1.0 after 20 zeros
    check_idx = start_idx + respawn_cooldown
    if check_idx < n and data[check_idx] == 1.0:
      # Mark the entire 20-element region as death (0)
      labels[start_idx: check_idx] = 0

  return labels


def compute_agent_move_distance(positions):
  _, A, _ = positions.shape
  return {i: float(np.sum(np.any(np.diff(positions[:, i], axis=0), axis=1)))
          for i in range(A)}


def compute_num_rotation(orientations):
  _, A = orientations.shape
  return {i: int(np.sum(np.abs(np.diff(orientations[:, i])))) for i in range(A)}


def compute_collect_counts(rewards, predators, preys):
  num_apple = {i: int(np.sum(rewards[:, i] == 1)) for i in preys}
  num_acorn = {i: int(np.sum(rewards[:, i] > 1)) for i in preys}
  num_catch = {i: int(np.sum(rewards[:, i] == 1)) for i in predators}
  return num_apple, num_acorn, num_catch

def compute_stuck_rate_all(positions, min_duration=50, death_labels=None):
  stuck_dict = {}
  for i in range(positions.shape[1]):
    stuck_rate, stuck_t = compute_stuck_rate(positions[:, i], min_duration)
    valid_stuck_t = stuck_t[death_labels[i] == 1]
    stuck_dict[i] = np.sum(valid_stuck_t) / len(valid_stuck_t) if len(valid_stuck_t) > 0 else np.nan
  return stuck_dict

def compute_stuck_rate(pos, min_duration=50):
  """
  Given a 2D position array pos of shape (T,2), compute the fraction of timesteps that are "stuck".
  A contiguous block of timesteps is marked as stuck if the agent's position does not change
  for at least min_duration timesteps. The entire duration of that block is then marked as stuck.
  Returns a float between 0 and 1 (the stuck rate).
  """
  T = pos.shape[0]
  stuck_indicator = np.zeros(T, dtype=int)
  i = 0
  while i < T - 1:
    # If current and next positions are identical (allowing for floating-point equality)
    if np.allclose(pos[i], pos[i + 1]):
      j = i
      while j < T - 1 and np.allclose(pos[j], pos[j + 1]):
        j += 1
      # Now, positions from index i to j (inclusive) did not change.
      if (j - i + 1) >= min_duration:
        stuck_indicator[i:j + 1] = 1
      i = j + 1
    else:
      i += 1
  return np.mean(stuck_indicator), stuck_indicator


def compute_death_counts(stamina, duration=100):
  deaths = {}
  for i in range(stamina.shape[1]):
    labels = mark_death_periods(stamina[:, i], duration)
    deaths[i] = int(np.sum((labels[:-1] == 1) & (labels[1:] == 0)))
  return deaths


def compute_grass_time_per_life(positions, death_labels, preys, safe_grass):
  on_time, off_time, frac_off = {}, {}, {}
  T = positions.shape[0]
  for i in preys:
    segments, alive = [], False
    for t in range(T):
      if death_labels[i][t] == 1 and not alive:
        start, alive = t, True
      if death_labels[i][t] == 0 and alive:
        segments.append((start, t));
        alive = False
    if alive:
      segments.append((start, T))
    on_list, off_list, frac_list = [], [], []
    for s, e in segments:
      on = sum(tuple(positions[t, i]) in safe_grass for t in range(s, e))
      total = e - s
      off = total - on
      on_list.append(on)
      off_list.append(off)
      frac_list.append(off / total if total > 0 else np.nan)
    on_time[i], off_time[i], frac_off[i] = on_list, off_list, frac_list
  return on_time, off_time, frac_off

def get_safe_grass_locations(folder_path: str):
  """
  Dynamically load the Meltingpot substrate config matching folder_path
  and extract all coordinates of safe grass tiles from its ASCII map.
  """
  # Infer substrate module name, e.g. 'predator_prey__open_debug'
  m = re.search(r'(predator_prey__[^_]+)', folder_path)
  substrate = m.group(1) if m else 'predator_prey__open_debug'
  module_name = f"meltingpot.python.configs.substrates.{substrate}"
  try:
    mod = importlib.import_module(module_name)
  except ImportError:
    raise ImportError(f"Could not import substrate config {module_name}")
  # Decide map size: if 'open' substrate, use default; else smaller_13x13
  kwargs = {} if 'open' in substrate else {'smaller_13x13': True}
  cfg = mod.get_config(**kwargs)
  ascii_map = cfg.layout.ascii_map.strip('\n').splitlines()
  char_map = cfg.layout.char_prefab_map

  safe_positions = []
  for y, row in enumerate(ascii_map):
    for x, ch in enumerate(row):
      val = char_map.get(ch)
      # string mapping
      if isinstance(val, str) and val.startswith('safe_grass'):
        safe_positions.append((x, y))
      # dict mapping with list of prefabs
      elif isinstance(val, dict) and 'list' in val and 'safe_grass' in val['list']:
        safe_positions.append((x, y))
  return safe_positions

def compute_pairwise_distance_mean(positions, death_labels):
  _, A, _ = positions.shape
  dist = {}
  for i in range(A):
    for j in range(i + 1, A):
      dists = np.sum(abs(positions[:, i] - positions[:, j]), axis=1)
      dists = dists[(death_labels[i] == 1) & (death_labels[j] == 1)]
      dist[f"dist_{i}to{j}"] = float(np.mean(dists))
  return dist


def ori_position(A_pos, A_orient, B_pos):
  orientation_transform = {
    0: lambda x, y: (x, y),
    1: lambda x, y: (y, -x),
    2: lambda x, y: (-x, -y),
    3: lambda x, y: (-y, x),
  }
  dx = B_pos[0] - A_pos[0]
  dy = B_pos[1] - A_pos[1]
  return orientation_transform[int(A_orient)](dx, dy)


def compute_good_gathering(positions, predators, preys, radius=3, death_labels=None):
  T = positions.shape[0]
  flags = {f"in_gather_{q}_to_{p}": [] for p in predators for q in preys}
  for t in range(T):
    for p in predators:
      # If the predator is dead, skip it
      if death_labels is not None and death_labels[p][t] == 0:
        continue
      posp = positions[t, p]
      nearby_prey = [q for q in preys if np.linalg.norm(positions[t, q] - posp) <= radius]
      if len(nearby_prey) > 0 and (death_labels is not None):
        nearby_prey = [q for q in nearby_prey if death_labels[q][t] == 1]
      nearby_pred = [r for r in predators if np.linalg.norm(positions[t, r] - posp) <= radius]
      if len(nearby_pred) > 0 and (death_labels is not None):
        nearby_pred = [r for r in nearby_pred if death_labels[r][t] == 1]

      for q in preys:
        if q in nearby_prey:
          flags[f"in_gather_{q}_to_{p}"].append(int(len(nearby_prey) > len(nearby_pred)))
        else:
          flags[f"in_gather_{q}_to_{p}"].append(np.nan)
  return {k: float(np.nanmean(v)) for k, v in flags.items()}



def compute_successful_fencing_and_helpers(
    positions, orientations, actions, rewards,
    predators, preys, death_labels, radius=3, seg_start = None, seg_end = None):
  """
  Returns fencing events with alive helpers only.
  Each event: {'time','predator','prey','helpers','helpers_distance'}
  In default situation, the segment is the whole episode. If not, we are passing in a different value of seg_start such
  that we can return the correct time in the episode instead of counting from seg_start.
  """
  events = []
  T = positions.shape[0]
  for t in range(T):
    for p in predators:
      if actions[t, p] == 7 and rewards[t, p] == 0:
        for q in preys:
          # If the current prey is dead, skip it
          if death_labels.get(q, np.zeros(T, dtype=int))[t] == 0:
            continue
          # Check if the prey is within the predator's range
          if ori_position(positions[t, p], orientations[t, p], positions[t, q]) == (0, 1):
            helper_list = []
            helper_dist = []
            for k in preys:
              if k != q and death_labels.get(k, np.zeros(T, dtype=int))[t] == 1:
                d = np.linalg.norm(positions[t, k] - positions[t, p])
                if d <= radius:
                  helper_list.append(k)
                  helper_dist.append(float(d))
            if seg_start is not None:
              events.append({'time': t + seg_start, 'segment_time': t, 'predator': p, 'prey': q,
                             'helpers': helper_list, 'helpers_distance': helper_dist})
            else:
              events.append({'time': t, 'predator': p, 'prey': q, 'helpers': helper_list,
                             'helpers_distance': helper_dist})
  return events




def compute_invalid_interactions(
    actions, rewards, positions, orientations,
    stamina, predators, preys,
    cooldown=5, radius2=2, radius3=3, interact_code=7):
  last = {p:-cooldown for p in predators}
  events=[]
  death_lbl = {i:mark_death_periods(stamina[:,i]) for i in preys}
  for t in range(actions.shape[0]):
    if t==18161:
      print('debug')
    for p in predators:
      if actions[t,p]==interact_code and rewards[t,p]==0:
        delta = t-last[p]
        if delta<cooldown:
          reason=f"repeat in {delta} timesteps"
        else:
          last[p]=t
          alive=[q for q in preys if death_lbl[q][t]==1]
          if not alive:
            reason='no prey alive'
          else:
            # group defense
            near_pred=[r for r in predators if np.linalg.norm(positions[t,r]-positions[t,p])<=radius3]
            fenced=False
            for q in alive:
              if ori_position(positions[t,p],orientations[t,p],positions[t,q])==(0,1):
                defenders=[k for k in alive if k!=q and np.linalg.norm(positions[t,k]-positions[t,p])<=radius3]
                if len(defenders) >= len(near_pred): fenced=True; break
            if fenced:
              reason='fenced'
            else:
              # distance to closest alive prey
              dists={q:np.linalg.norm(positions[t,q]-positions[t,p]) for q in alive}
              qmin=min(dists,key=dists.get); d=dists[qmin]
              if d<=radius2:
                # find last movement
                last_move=None
                for tau in range(t-1,-1,-1):
                  if not np.allclose(positions[tau,qmin],positions[tau+1,qmin]):
                    last_move=tau+1; break
                if last_move is not None:
                  rel=ori_position(positions[last_move,p],orientations[last_move,p],positions[last_move,qmin])
                  if rel==(0,1) and np.linalg.norm(positions[last_move,qmin]-positions[last_move,p])<=radius2:
                    reason='late shot'
                  else:
                    reason='miss prediction'
                else:
                  reason='miss prediction'
              elif d<=radius3:
                reason='off target r3'
              else:
                reason='exception'
        events.append({'time':t,'predator':p,'invalid_interact':reason})
  return events

# --- Core processing ---
#
# def process_pair_folder(folder_path):
#   base=os.path.basename(folder_path).split('predator_prey__')[0]
#   role,src=parse_agent_roles(base)
#   preds=[i for i,r in role.items() if r=='predator']
#   preys_idx=[i for i,r in role.items() if r=='prey']
#   pkl=os.path.join(folder_path,'episode_pickles')
#   files=sorted(f for f in os.listdir(pkl) if f.endswith('.pkl'))
#   if not files: return None
#   data=[]
#   for f in files:
#     data.extend(pickle.load(open(os.path.join(pkl,f),'rb')))
#   pos=np.array([d['POSITION'] for d in data])
#   ori=np.array([d['ORIENTATION'] for d in data])
#   act=np.array([d['actions'] for d in data])
#   rew=np.array([d['rewards'] for d in data])
#   sta=np.array([d['STAMINA'] for d in data])
#   # metrics
#   move=compute_agent_move_distance(pos)
#   rot =compute_num_rotation(ori)
#   apple,acorn,catch=compute_collect_counts(rew,preds,preys_idx)
#   death_cnt=compute_death_counts(sta)
#   death_lbl={i:mark_death_periods(sta[:,i]) for i in range(sta.shape[1])}
#   safe={(x,y) for x in [8,9,10] for y in [4,5,6]}
#   on,off,fo=compute_grass_time_per_life(pos,death_lbl,preys_idx,safe)
#   pair_dist_mean=compute_pairwise_distance_mean(pos)
#   good=compute_good_gathering(pos,preds,preys_idx)
#   fence=compute_successful_fencing_and_helpers(pos,ori,act,rew,preds,preys_idx,death_lbl)
#   invalid=compute_invalid_interactions(act,rew,pos,ori,sta,preds,preys_idx)
#   # Quickly verify invalid interactions
#   merged = pd.DataFrame(fence).merge(
#     pd.DataFrame(invalid)[['time', 'predator', 'invalid_interact']],
#     on=['time', 'predator'],
#     how='left'
#   )
#   merged['recognized_as_fence'] = merged['invalid_interact'] == 'fenced'
#   unrecognized_fences = merged[~merged['recognized_as_fence']]
#   print(unrecognized_fences)
#
#   row={'pair':base,'fencing_events':merged,'invalid_events':invalid}
#   row.update({f'move_{i}':move[i] for i in move})
#   row.update({f'rot_{i}':rot[i] for i in rot})
#   row.update({f'apple_{i}':apple[i] for i in apple})
#   row.update({f'acorn_{i}':acorn[i] for i in acorn})
#   row.update({f'catch_{i}':catch[i] for i in catch})
#   row.update({f'death_{i}':death_cnt[i] for i in death_cnt})
#   row.update(pair_dist_mean)
#   row.update(good)
#   return pd.DataFrame([row])
def process_pair_folder(folder_path):
  base = os.path.basename(folder_path).split('predator_prey__')[0]
  role, src = parse_agent_roles(base)
  predators = [i for i, r in role.items() if r == 'predator']
  preys = [i for i, r in role.items() if r == 'prey']

  pkl_dir = os.path.join(folder_path, 'episode_pickles')
  files = sorted(f for f in os.listdir(pkl_dir) if f.endswith('.pkl'))
  if not files:
    return None

  all_episode_dfs = []
  for fname in files:
    # load one episode
    with open(os.path.join(pkl_dir, fname), 'rb') as fp:
      episode_data = pickle.load(fp)
    # build arrays just for this episode
    pos = np.array([d['POSITION'] for d in episode_data])
    ori = np.array([d['ORIENTATION'] for d in episode_data])
    act = np.array([d['actions'] for d in episode_data])
    rew = np.array([d['rewards'] for d in episode_data])
    sta = np.array([d['STAMINA'] for d in episode_data])

    # compute death_labels for this episode
    death_lbl = {i: mark_death_periods(sta[:, i]) for i in range(sta.shape[1])}

    # compute your metrics exactly as before, but now per-episode
    move_dist = compute_agent_move_distance(pos)
    rotation = compute_num_rotation(ori)
    apple, acorn, catch = compute_collect_counts(rew, predators, preys)
    death_cnt = compute_death_counts(sta)  # counts per agent
    stuck_rate = compute_stuck_rate_all(pos, death_labels=death_lbl)

    # load safe grass from substrate
    safe = set(get_safe_grass_locations(folder_path))
    on_t, off_t, fo = compute_grass_time_per_life(pos, death_lbl, preys, safe)
    pair_dist = compute_pairwise_distance_mean(pos, death_lbl)
    good = compute_good_gathering(pos, predators, preys, death_labels=death_lbl)
    fence = compute_successful_fencing_and_helpers(pos, ori, act, rew,
                                                   predators, preys, death_lbl)
    invalid = compute_invalid_interactions(act, rew, pos, ori, sta,
                                           predators, preys)

    # assemble one DataFrame row for this episode
    row = {
      'trial_name': base,
      'role': role,
      'source': src,
      'episode': fname.replace('.pkl', ''),
      'fencing_events': fence,
      'invalid_events': invalid,
      # flatten the per-agent / per-pair scalars:
      **{f"move_{i}": move_dist[i] for i in move_dist},
      **{f"rot_{i}": rotation[i] for i in rotation},
      **{f"apple_{i}": apple[i] for i in apple},
      **{f"acorn_{i}": acorn[i] for i in acorn},
      **{f"catch_{i}": catch[i] for i in catch},
      **{f"death_{i}": death_cnt[i] for i in death_cnt},
      **{f"on_grass_{i}": on_t[i] for i in on_t},
      **{f"off_grass_{i}": off_t[i] for i in off_t},
      **{f"frac_off_{i}": fo[i] for i in fo},
      **pair_dist,
      **good
    }
    all_episode_dfs.append(pd.DataFrame([row]))

  # now concatenate all episodes for this predator–prey folder
  return pd.concat(all_episode_dfs, ignore_index=True)

def process_and_save(folder_path, out_dir):
  """
  Wrapper to process one folder and save its metrics CSV.
  """
  df = process_pair_folder(folder_path)
  if df is None:
    return
  # Derive base filename from folder
  base = os.path.basename(folder_path).split('predator_prey__')[0]
  out_path = os.path.join(out_dir, f"{base}_metrics.csv")
  df.to_csv(out_path, index=False)
  pkl_path = os.path.join(out_dir, f"{base}_metrics.pkl")
  df.to_pickle(pkl_path)
  print(f"Saved: {os.path.basename(out_path)}")


def main():
  base_dir = "/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/"
  out_dir = os.path.join(base_dir, "analysis_results")
  os.makedirs(out_dir, exist_ok=True)

  # Find all valid pair folders
  folders = [os.path.join(base_dir, d)
             for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d))
             and 'episode_pickles' in os.listdir(os.path.join(base_dir, d))
             and 'simplified10x10' not in d]

  # Process in parallel using all available CPUs
  Parallel(n_jobs=50)(
    delayed(process_and_save)(folder, out_dir)
    for folder in folders
  )
  # for folder in folders:
  #   process_and_save(folder, out_dir)
if __name__ == '__main__':
  main()
