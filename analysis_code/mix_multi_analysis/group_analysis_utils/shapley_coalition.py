#!/usr/bin/env python3
"""
shapley_coalition.py
Compute per‐segment and per‐episode Shapley values for Efficiency,
Equality, and Sustainability in predator–prey rollouts.
"""

import os
import pickle
import random
from typing import List, Dict, Tuple, Sequence, Optional
import numpy as np

# -----------------------------------------------------------------------------
# Metric computation for a single segment
# -----------------------------------------------------------------------------
def compute_metrics_for_segment(
    rewards_seq: List[List[float]],
    roles: List[str],
    death_labels: Optional[Dict[int, np.ndarray]] = None,
    segment_start: int = 0,
) -> Dict[str, float]:
  """
  Compute Efficiency, Equality, and Sustainability for one segment.

  Args:
      rewards_seq: list of length T of lists of length N:
          rewards_seq[t][i] = reward for agent i at time t.
      roles: length-N list of 'predator' or 'prey'.
      death_labels: optional dict mapping agent index -> array of 0/1 over ENTIRE episode.
      segment_start: absolute time index of the first frame in this segment.

  Returns:
      metrics dict with keys 'efficiency', 'equality', 'sustainability'.
  """
  num_agents = len(roles)
  T = len(rewards_seq)
  # Empty segment => no reward, perfect equality, sustainability=segment_start
  if T == 0 or num_agents == 0:
    return {
      "efficiency": 0.0,
      "equality": 1.0,
      "sustainability": float(segment_start),
    }

  # 1) Efficiency: total team reward over segment
  #    flatten and sum per-agent totals
  total_rewards = [sum(col) for col in zip(*rewards_seq)]
  efficiency = sum(total_rewards)

  # 2) Equality: 1 - (Gini coefficient) on total_rewards
  if efficiency <= 1e-9:
    # no team reward => if all agents equal, equality=1, else 0
    all_eq = all(abs(r - total_rewards[0]) < 1e-9 for r in total_rewards)
    equality = 1.0 if all_eq else 0.0
  else:
    # standard Gini: sum |ri - rj| / (2 N * sum ri)
    diff_sum = sum(
      abs(total_rewards[i] - total_rewards[j])
      for i in range(num_agents)
      for j in range(num_agents)
    )
    gini = diff_sum / (2.0 * num_agents * efficiency)
    equality = max(0.0, 1.0 - gini)

  # 3) Sustainability: average *absolute* timestep of reward collection
  # For each agent, collect all times t+segment_start at which reward>0
  sustain_times = []
  for i in range(num_agents):
    times_i = [t + segment_start for t in range(T) if rewards_seq[t][i] > 1e-9]
    if times_i:
      mean_time = sum(times_i) / len(times_i)
      # if death_labels given, clamp to last alive time
      if death_labels and i in death_labels:
        arr = death_labels[i]
        end_idx = min(len(arr)-1, segment_start + T - 1)
        # if agent died before end
        if arr[end_idx] == 0:
          # find last alive
          last_alive = segment_start
          for tau in range(end_idx, segment_start-1, -1):
            if arr[tau] == 1:
              last_alive = tau
              break
          mean_time = min(mean_time, last_alive)
      sustain_times.append(mean_time)
    else:
      sustain_times.append(float(segment_start + T))
  sustainability = sum(sustain_times) / num_agents

  return {
    "efficiency": efficiency,
    "equality": equality,
    "sustainability": sustainability,
  }


# -----------------------------------------------------------------------------
# Shapley value via Monte Carlo for one segment
# -----------------------------------------------------------------------------
def calculate_segment_shapley_values(
    episode_rewards: Sequence[Sequence[float]],
    roles: List[str],
    death_labels: Optional[Dict[int, np.ndarray]],
    segment_start: int,
    segment_end: int,
    MCPerm: int,
) -> Dict[str, List[float]]:
  """
  Monte Carlo approximate Shapley values for a given segment.

  Args:
      episode_rewards: shape (episode_length, num_agents)
      roles: length-N list of agent roles
      death_labels: optional dict agent_idx-> alive(1)/dead(0) array
      segment_start: segment first index (inclusive)
      segment_end:   segment end index (exclusive)
      MCPerm: number of random permutations

  Returns:
      dict with keys:
        'shapley_efficiency': [Φ_eff_i],
        'shapley_equality':   [Φ_eq_i],
        'shapley_sustainability':[Φ_sus_i]
      each list of length num_agents.
  """
  # slice the segment
  segment_data = episode_rewards[segment_start:segment_end]
  T = len(segment_data)
  N = len(roles)

  # helper: value of coalition C (list of agent indices)
  def coalition_value(coalition: Sequence[int]) -> Dict[str, float]:
    if not coalition:
      # empty coalition base case
      return {
        "efficiency": 0.0,
        "equality": 1.0,
        "sustainability": float(segment_start),
      }
    # build reward subsequence for only those agents
    sub_rewards = [[row[i] for i in coalition] for row in segment_data]
    # build roles subset
    sub_roles = [roles[i] for i in coalition]
    # build death_labels subset mapping original indices
    sub_death = None
    if death_labels:
      sub_death = {i: death_labels[i] for i in coalition}
    return compute_metrics_for_segment(
      sub_rewards, sub_roles, sub_death, segment_start
    )

  # accumulate marginal contributions
  shap_sums = {
    "eff": [0.0]*N,
    "eq":  [0.0]*N,
    "sus": [0.0]*N,
  }

  agents = list(range(N))
  for _ in range(MCPerm):
    perm = agents.copy()
    random.shuffle(perm)
    current_coal = []
    current_val = coalition_value([])

    for idx in perm:
      new_coal = current_coal + [idx]
      new_val = coalition_value(new_coal)
      shap_sums["eff"][idx] += new_val["efficiency"] - current_val["efficiency"]
      shap_sums["eq"][idx]  += new_val["equality"]   - current_val["equality"]
      shap_sums["sus"][idx] += new_val["sustainability"] - current_val["sustainability"]
      current_coal = new_coal
      current_val = new_val

  # average
  return {
    "shapley_efficiency":   [s / MCPerm for s in shap_sums["eff"]],
    "shapley_equality":     [s / MCPerm for s in shap_sums["eq"]],
    "shapley_sustainability":[s / MCPerm for s in shap_sums["sus"]],
  }


# -----------------------------------------------------------------------------
# Full-episode and per-segment Shapley evaluation
# -----------------------------------------------------------------------------
def compute_shapley_values_for_segments(
    episode_rewards: Sequence[Sequence[float]],
    role_map: Dict[int, str],
    active_segments: List[Tuple[int, int]],
    death_labels: Optional[Dict[int, np.ndarray]] = None,
    MCPerm: int = 500,
) -> Dict:
  """
  Compute Shapley values for the entire episode and each active segment.

  Args:
      episode_rewards: (T x N) rewards sequence for the episode.
      role_map: map agent_idx -> 'predator'/'prey'
      active_segments: list of (start, end) indices
      death_labels: optional agent_idx -> alive/dead
      MCPerm: number of MC samples per segment

  Returns:
      dict with keys:
        'episode_shapley': dict of 3 lists
        'segments': [
           {
             'time_start': s, 'time_end': e,
             'shapley_efficiency': [...],
             'shapley_equality':   [...],
             'shapley_sustainability': [...],
           }, ...
        ]
  """
  # prepare roles list
  N = len(role_map)
  roles = [role_map[i] for i in range(N)]

  # entire episode
  episode_vals = calculate_segment_shapley_values(
    episode_rewards, roles, death_labels, 0, len(episode_rewards), MCPerm
  )

  # per-segment
  seg_results = []
  for s,e in active_segments:
    seg_vals = calculate_segment_shapley_values(
      episode_rewards, roles, death_labels, s, e, MCPerm
    )
    seg_results.append({
      "time_start": s,
      "time_end": e,
      **seg_vals
    })

  return {
    "episode_shapley": episode_vals,
    "segments": seg_results
  }



if __name__ == "__main__":
  # Example usage (replace with your actual data)
  episode_name = (
    "mix_AH20250107_pred_0_AH20250107_pred_1_AH20250107_prey_3_AH20250107_prey_4_"
    "AH20250107_prey_5_AH20250107_prey_6predator_prey__open_debug_agent_0_1_3_4_5_6"
  )
  folder = f'/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/{episode_name}/episode_pickles/'

  import os

  base = os.path.basename(episode_name).split("predator_prey__")[0]
  from helper import parse_agent_roles

  role_map, src_map = parse_agent_roles(base)  # role_map[idx] = "predator" or "prey"
  predator_ids = [i for i, r in role_map.items() if r == "predator"]
  prey_ids = [i for i, r in role_map.items() if r == "prey"]
  agent_roles = [role_map[i] for i in range(len(role_map))]

  # Load the episode data
  import pickle

  files = os.listdir(folder)
  files = sorted(f for f in files if f.endswith(".pkl"))
  for fname in files:
    with open(os.path.join(folder, fname), "rb") as fp:
      episode_data = pickle.load(fp)

    pos_ep = np.array([d["POSITION"] for d in episode_data])
    ori_ep = np.array([d["ORIENTATION"] for d in episode_data])
    act_ep = np.array([d["actions"] for d in episode_data])
    rew_ep = np.array([d["rewards"] for d in episode_data])
    sta_ep = np.array([d["STAMINA"] for d in episode_data])

    from segmentation import mark_death_periods, segment_active_phases

    death_labels_ep = {
      i: mark_death_periods(sta_ep[:, i]) for i in range(sta_ep.shape[1])
    }
    death_labels_all_prey_ep = {
      pi: death_labels_ep[pi] for pi in prey_ids if pi in death_labels_ep
    }
    active_segments = segment_active_phases(death_labels_all_prey_ep)

    # Compute Shapley values for each segment
    results = compute_shapley_values_for_segments(
      episode_rewards=rew_ep,
      role_map=role_map,
      active_segments=active_segments,
      death_labels=death_labels_ep,
      MCPerm=50,
    )

    import pandas as pd
    pd.DataFrame(results).to_csv("example/shapley_results_example.csv", index=False)
    # Print the results in the desired format
    print(f"Shapley values for episode: {fname}")
    print(results)
