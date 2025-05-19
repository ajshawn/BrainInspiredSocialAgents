# altruism.py
"""
Unified altruism analysis across *any* social-interaction events.

An *event* is a dictionary that MUST contain:
    - 'type' : str              (e.g. 'gather', 'fence', 'distraction')
    - 'time_start' : int
    - 'time_end'   : int        (inclusive, episode-time indices)
    - 'helpers'    : List[int]  (agent indices)
    - 'beneficiaries' : List[int]  (agents who are being helped)
It MAY contain:
    - 'predator' : int
    - anything else the detector already records (kept & returned)

All time indices are in **absolute episode frames** – no segment-relative
indices here; convert before calling if needed.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Any, Sequence

# ---------------------------------------------------------------------
#  Core helper functions
# ---------------------------------------------------------------------

def _sum_stamina_loss(
    agent_ids: Sequence[int],
    t0: int,
    t1: int,
    stamina: np.ndarray,
) -> float:
    """Sum (stamina[t0] – stamina[t1]) for listed agents (clamped to 0)."""
    T = stamina.shape[0]
    t0 = max(0, min(t0, T - 1))
    t1 = max(0, min(t1, T - 1))
    loss = 0.0
    for a in agent_ids:
        loss += max(0.0, stamina[t0, a] - stamina[t1, a])
    return loss

def _any_agent_caught(
    agent_ids: Sequence[int],
    t0: int,
    t1: int,
    death_labels: Dict[int, np.ndarray]
) -> bool:
    """True if *any* of the given agents dies within (t0, t1] inclusive."""
    for a in agent_ids:
        arr = death_labels[a]
        window = arr[t0+1 : t1+1] if t1+1 <= len(arr) else arr[t0+1 :]
        if np.any(window == 0):
            return True
    return False

def _benefit_escape_or_survive(
    target_ids: Sequence[int],
    positions: np.ndarray,
    death_labels: Dict[int, np.ndarray],
    safe_tiles: set[Tuple[int,int]],
    t0: int,
    t1: int,
) -> bool:
    """
    Return True if *any* target reaches grass OR none die within [t0,t1].
    """
    for b in target_ids:
        alive_series = death_labels[b][t0:t1+1]
        if np.all(alive_series == 1):
            return True  # survived whole window
        # else check escape frame-by-frame
        for tau in range(t0, t1+1):
            if death_labels[b][tau] == 0:
                break  # died, cannot escape later
            if tuple(positions[tau, b]) in safe_tiles:
                return True
    return False

# ---------------------------------------------------------------------
#  Main public API
# ---------------------------------------------------------------------

def analyse_altruism_events(
    events: List[Dict[str,Any]],
    positions: np.ndarray,
    stamina: np.ndarray,
    death_labels: Dict[int, np.ndarray],
    safe_grass: Sequence[Tuple[int,int]],
    *,
    cost_pre_window: int  = 5,
    cost_post_window: int = 20,
    benefit_window: int   = 20
) -> Tuple[List[Dict[str,Any]], Dict[int,Dict[str,float]]]:
    """
    Parameters
    ----------
    events      : list produced by your detectors (gather/fence/distraction…)
    positions   : (T, N, 2) float array
    stamina     : (T, N) float array
    death_labels: {agent→(T,) int array}
    safe_grass  : set of grass-tile tuples
    Returns
    -------
    event_metrics   : same list size as *events*, with extra fields:
                        • helper_cost
                        • helper_caught
                        • beneficiary_gain
                        • altruistic (bool)
    agent_metrics   : {agent_id : {'num_events', 'total_cost', 'total_gain'}}
    """
    safe_tiles = set(safe_grass)
    T = positions.shape[0]
    # -----------------------------------------------------------------
    # Event-level pass
    # -----------------------------------------------------------------
    enriched: List[Dict[str,Any]] = []
    for ev in events:
        t_mid  = (ev["time_start"] + ev["time_end"]) // 2  # pivot time
        helpers   = ev.get("helpers",       [])
        targets   = ev.get("beneficiaries", [])
        # ----- cost to helpers -----
        t0_cost = max(0, t_mid - cost_pre_window)
        t1_cost = min(T-1, t_mid + cost_post_window)
        cost_sum   = _sum_stamina_loss(helpers, t0_cost, t1_cost, stamina)
        caught_any = _any_agent_caught(helpers, t0_cost, t1_cost, death_labels)
        # ----- benefit to targets ---
        t1_benef = min(T-1, t_mid + benefit_window)
        gained = _benefit_escape_or_survive(
            targets, positions, death_labels, safe_tiles,
            t_mid, t1_benef
        )
        # ----- final decision -------
        altruistic = bool(helpers) and bool(targets) and (cost_sum > 0 or caught_any) and gained
        enriched.append({
            **ev,
            "helper_stamina_cost": cost_sum,
            "helper_caught": caught_any,
            "beneficiary_gain": gained,
            "is_altruistic": altruistic
        })

    # -----------------------------------------------------------------
    # Aggregate to agent-level summary
    # -----------------------------------------------------------------
    agent_stats: Dict[int,Dict[str,float]] = {}
    for ev in enriched:
        if not ev["is_altruistic"]:
            continue
        helpers = ev["helpers"]
        gain    = 1.0 if ev["beneficiary_gain"] else 0.0
        cost    = ev["helper_cost"] / max(len(helpers), 1)
        for h in helpers:
            s = agent_stats.setdefault(h, {"num_events":0, "total_cost":0.0, "total_gain":0.0})
            s["num_events"]   += 1
            s["total_cost"]   += cost
            s["total_gain"]   += gain

    return enriched, agent_stats


if __name__ == "__main__":
  episode_name = (
    'mix_AH20250107_pred_0_AH20250107_pred_1_AH20250107_prey_3_AH20250107_prey_4_'
    'AH20250107_prey_5_AH20250107_prey_6predator_prey__open_debug_agent_0_1_3_4_5_6'
  )
  folder = f'/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/{episode_name}/episode_pickles/'

  import os
  base = os.path.basename(episode_name).split('predator_prey__')[0]
  from helper import parse_agent_roles

  role_map, src_map = parse_agent_roles(base)  # role_map[idx] = "predator" or "prey"
  predator_ids = [i for i, r in role_map.items() if r == 'predator']
  prey_ids = [i for i, r in role_map.items() if r == 'prey']

  episode_segment_dfs = []  # Store rows for each active segment of each episode
  # Load the episode data
  import pickle

  files = os.listdir(folder)
  files = sorted(f for f in files if f.endswith('.pkl'))

  for fname in files:
    with open(os.path.join(folder, fname), 'rb') as fp:
      episode_data = pickle.load(fp)

    pos_ep = np.array([d['POSITION'] for d in episode_data])
    ori_ep = np.array([d['ORIENTATION'] for d in episode_data])
    act_ep = np.array([d['actions'] for d in episode_data])
    rew_ep = np.array([d['rewards'] for d in episode_data])
    sta_ep = np.array([d['STAMINA'] for d in episode_data])

    from segmentation import mark_death_periods, segment_active_phases
    from map_utils import get_safe_tiles

    death_labels_ep = {i: mark_death_periods(sta_ep[:, i]) for i in range(sta_ep.shape[1])}
    death_labels_all_prey_ep = {pi: death_labels_ep[pi] for pi in prey_ids if pi in death_labels_ep}
    active_segments = segment_active_phases(death_labels_all_prey_ep)
    safe_grass_locs = get_safe_tiles(folder, map='smaller_13x13')

    from gather_and_fence import compute_good_gathering_segmented, compute_successful_fencing_and_helpers_segmented
    from cooperation import apple_periods_segmented
    from distraction import detect_distraction_events

    gather_events = compute_good_gathering_segmented(pos_ep, predator_ids, prey_ids, active_segments)
    fence_events = compute_successful_fencing_and_helpers_segmented(
      pos_ep, ori_ep, act_ep, rew_ep, predator_ids,
      prey_ids, death_labels_ep, active_segments=active_segments, radius=3,
    )
    apple_cooperation_events = apple_periods_segmented(
      pos_ep, rew_ep, death_labels_ep, prey_ids, safe_grass_locs,
      active_segments, cluster_radius=3,apple_period_min_len=5, apple_period_gap=5,
      participation_threshold=0.4
    )

    distraction_events = detect_distraction_events(
      positions=pos_ep,
      orientations=ori_ep,
      death_labels=death_labels_ep,
      safe_grass=safe_grass_locs,
      predators=predator_ids,
      preys=prey_ids,
      active_segments=active_segments,
      window_away=5,
      shift_window=5,
      distraction_period_gap=2,
      radius=3.0,
      move_thresh=0.5,
      max_period_duration=30,  # Changed hard_cap -> max_period_duration
      scenario='grass',  # or 'chase'
    )

    distraction_events += detect_distraction_events(
      positions=pos_ep,
      orientations=ori_ep,
      death_labels=death_labels_ep,
      safe_grass=safe_grass_locs,
      predators=predator_ids,
      preys=prey_ids,
      active_segments=active_segments,
      window_away=5,
      shift_window=5,
      distraction_period_gap=2,
      radius=3.0,
      move_thresh=0.5,
      max_period_duration=30,  # Changed hard_cap -> max_period_duration
      scenario='chase',  # or 'chase'
    )



    episode_segment_dfs.append(
      {
        "folder": base,
        "episode": fname.replace(".pkl", ""),
        "seg_idx": [(start, end) for start, end in active_segments],
        "good_gathering": gather_events,
        "fencing_events": fence_events,
        "apple_cooperation_events": apple_cooperation_events,
        "distraction_events": distraction_events,
      }
    )

    # Now, lets start the analysis based on the events we have by adding the altruism analysis
    for ev in fence_events:
      ev["type"] = "fence"
      # helpers already present, prey being helped:
      ev["beneficiaries"] = [ev["beneficiary"]] if type(ev["beneficiary"]) == int else ev["beneficiary"]
      ev["helpers"] = [ev["helper"]] if type(ev["helper"]) == int else ev["helper"]
      ev["time_start"] = ev["time"] - 10
      ev["time_end"] = ev["time"] + 20

    for ev in distraction_events:
      ev["type"] = "distraction"
      ev["helpers"] = [ev["helper"]]  if type(ev["helper"]) == int else ev["helper"]
      ev["beneficiaries"] = [ev["beneficiary"]]  if type(ev["beneficiary"]) == int else ev["beneficiary"]
      # time_start/shift/end already there


    # 3) concatenate and call
    all_events = fence_events + distraction_events
    event_metrics, agent_metrics = analyse_altruism_events(
      all_events,
      positions=pos_ep,
      stamina=sta_ep,
      death_labels=death_labels_ep,
      safe_grass=safe_grass_locs,
    )

    print("=== Event-level ===")
    for ev in event_metrics:
      print(ev)

    print("=== Agent summary ===")
    for aid, s in agent_metrics.items():
      print(aid, s)

  # Convert to DataFrame
  import pandas as pd

  df = pd.DataFrame(episode_segment_dfs)
  print(df)
  # Save to CSV
  if not os.path.exists("example"):
    os.makedirs("example")
  df.to_csv("example/gather_and_fence_analysis_results_example.csv", index=False)
