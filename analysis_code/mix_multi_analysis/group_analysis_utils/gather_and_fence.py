import numpy as np
from typing import List, Dict, Tuple, Sequence, Optional
from analysis_code.mix_multi_analysis.group_analysis_utils.position import ori_position
from analysis_code.mix_multi_analysis.group_analysis_utils.segmentation import mark_death_periods




def compute_good_gathering_segmented(
    positions: np.ndarray,
    predators: Sequence[int],
    preys: Sequence[int],
    active_segments: List[Tuple[int, int]]=[(0, 1000)], # Default segment
    radius: float = 3.0,
    death_labels: Optional[Dict[int, np.ndarray]] = None,
) -> List[Dict[str, object]]:
  """
  Computes a metric indicating whether prey are gathered around a predator
  with more nearby prey than predators, considering active segments.

  Args:
      positions: A NumPy array of shape (T, N, 2) representing the (x, y)
          coordinates of all agents over T timesteps.
      predators: A sequence of integer indices representing the predator agents.
      preys: A sequence of integer indices representing the prey agents.
      active_segments: A list of tuples, where each tuple (start, end)
                       defines an active segment in the episode.
      radius: The radius within which to consider nearby agents. Defaults to 3.
      death_labels: An optional dictionary mapping agent indices to a NumPy
          array of shape (T,) where 1 indicates the agent is alive and 0
          indicates it is dead. Defaults to None.

  Returns:
      A dictionary where keys are strings in the format
      "in_gather_{prey_index}_to_{predator_index}" and values are the
      mean over time (ignoring NaN values) of whether that prey was in a
      "good gathering" state relative to that predator, across all active
      segments. NaN values indicate the prey was not within the predator's
      radius at that timestep within the segment.
  """
  flags = []
  T = positions.shape[0]

  for t0, t1 in active_segments:
    seg_flags: Dict[str, List[Optional[float]]] = {
        f"in_gather_{q}_to_{p}": [] for p in predators for q in preys
    }
    for t in range(t0, t1):
      for p in predators:
        is_predator_alive = True
        if death_labels is not None and death_labels.get(p, np.ones(T, dtype=int))[t] == 0:
          is_predator_alive = False

        if is_predator_alive:
          posp = positions[t, p]
          nearby_prey = [
            q for q in preys if np.linalg.norm(positions[t, q] - posp) <= radius
          ]
          if death_labels is not None:
            nearby_prey = [
              q for q in nearby_prey if death_labels.get(q, np.ones(T, dtype=int))[t] == 1
            ]

          nearby_pred = [
            r for r in predators if np.linalg.norm(positions[t, r] - posp) <= radius
          ]
          if death_labels is not None:
            nearby_pred = [
              r for r in nearby_pred if death_labels.get(r, np.ones(T, dtype=int))[t] == 1
            ]

          is_good_gathering = len(nearby_prey) > len(nearby_pred)

          for q in preys:
            if q in nearby_prey:
              seg_flags[f"in_gather_{q}_to_{p}"].append(float(is_good_gathering))
            else:
              seg_flags[f"in_gather_{q}_to_{p}"].append(np.nan)
        else:
          for q in preys:
            seg_flags[f"in_gather_{q}_to_{p}"].append(np.nan)  # Predator is dead, no gathering
    seg_flags = {k: np.nanmean(v) for k, v in seg_flags.items()}
  flags.append(seg_flags)
  # return {k: float(np.nanmean(v)) for k, v in flags.items()}
  return flags


def mark_events_successful_fencing_and_helpers_segmented(
    positions: np.ndarray,
    orientations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    predators: Sequence[int],
    preys: Sequence[int],
    death_labels: Dict[int, np.ndarray],
    active_segments: List[Tuple[int, int]]=[(0, 1000)],  # Default segment
    radius: float = 3.0,
) -> List[Dict[str, object]]:
  """
  Detects successful fencing events with alive helpers only, considering
  active segments.

  Args:
      positions: A NumPy array of shape (T, N, 2) representing agent positions.
      orientations: A NumPy array of shape (T, N, 2) representing agent orientations.
      actions: A NumPy array of shape (T, N) representing agent actions.
      rewards: A NumPy array of shape (T, N) representing agent rewards.
      predators: A sequence of integer indices for predator agents.
      preys: A sequence of integer indices for prey agents.
      death_labels: A dictionary mapping agent indices to a NumPy array of
          shape (T,) indicating aliveness (1) or deadness (0).
      active_segments: A list of tuples, where each tuple (start, end)
                       defines an active segment in the episode.
      radius: The radius to consider other prey as helpers. Defaults to 3.

  Returns:
      A list of dictionaries, where each dictionary represents a successful
      fencing event with alive helpers within any of the active segments.
      Each event dictionary contains:
      - 'time': The timestep of the event (aligned to the episode time).
      - 'segment_time': The timestep of the event within the segment.
      - 'predator': The index of the predator involved.
      - 'prey': The index of the prey being fenced.
      - 'helpers': A list of indices of the alive prey agents acting as helpers.
      - 'helpers_distance': A list of the distances of the helpers to the predator.
  """
  events: List[Dict[str, object]] = []
  T = positions.shape[0]
  interact_action = 7  # Assuming 7 is the code for the fencing action
  no_reward_on_interact = 0  # Assuming 0 reward indicates a failed/unsuccessful interact

  for t0, t1 in active_segments:
    for t in range(t0, t1):
      for p in predators:
        if actions[t, p] == interact_action and rewards[t, p] == no_reward_on_interact:
          for q in preys:
            is_prey_alive = death_labels.get(q, np.zeros(T, dtype=int))[t] == 1
            if is_prey_alive:
              # Check if the prey is in front of the predator
              if ori_position(positions[t, p], orientations[t, p], positions[t, q]) == (0, 1):
                helper_list: List[int] = []
                helper_dist: List[float] = []
                for k in preys:
                  if k != q and death_labels.get(k, np.zeros(T, dtype=int))[t] == 1:
                    distance = np.linalg.norm(positions[t, k] - positions[t, p])
                    if distance <= radius:
                      helper_list.append(k)
                      helper_dist.append(float(distance))

                events.append({
                  'time': t,
                  'segment_time': t - t0,
                  'predator': p,
                  'beneficiary': q,
                  'helper': helper_list,
                  'helper_distance': helper_dist,
                })
  return events


def mark_events_compute_invalid_interactions_segmented(
    actions: np.ndarray,
    rewards: np.ndarray,
    positions: np.ndarray,
    orientations: np.ndarray,
    stamina: np.ndarray,
    predators: Sequence[int],
    preys: Sequence[int],
    active_segments: List[Tuple[int, int]] = [(0, 1000)],  # Default segment
    cooldown: int = 5,
    radius2: float = 2.0,
    radius3: float = 3.0,
    interact_code: int = 7,
) -> List[Dict[str, object]]:
  """
  Detects and categorizes invalid interaction attempts by predators within
  active segments.

  Args:
      actions: A NumPy array of shape (T, N) representing agent actions.
      rewards: A NumPy array of shape (T, N) representing agent rewards.
      positions: A NumPy array of shape (T, N, 2) representing agent positions.
      orientations: A NumPy array of shape (T, N, 2) representing agent orientations.
      stamina: A NumPy array of shape (T, N) representing agent stamina.
      predators: A sequence of integer indices for predator agents.
      preys: A sequence of integer indices for prey agents.
      active_segments: A list of tuples, where each tuple (start, end)
                       defines an active segment in the episode.
      cooldown: The minimum number of timesteps between interaction attempts
          by the same predator. Defaults to 5.
      radius2: A smaller radius used for checking close proximity to prey.
          Defaults to 2.
      radius3: A larger radius used for checking nearby predators and defenders.
          Defaults to 3.
      interact_code: The integer code representing the interaction action.
          Defaults to 7.

  Returns:
      A list of dictionaries, where each dictionary represents an invalid
      interaction event within any of the active segments. Each event
      dictionary contains:
      - 'time': The timestep of the invalid interaction (aligned to episode time).
      - 'segment_time': The timestep of the event within the segment.
      - 'predator': The index of the predator that attempted the interaction.
      - 'invalid_interact': A string describing the reason for the invalid
        interaction.
  """
  last_interact_time: Dict[int, int] = {p: -cooldown for p in predators}
  events: List[Dict[str, object]] = []
  T = actions.shape[0]

  for t0, t1 in active_segments:
    death_labels: Dict[int, np.ndarray] = {
      i: mark_death_periods(stamina[t0:t1, i]) for i in preys
    }
    for t_seg in range(t1 - t0):
      t_ep = t0 + t_seg  # Episode-aligned timestep
      for p in predators:
        if actions[t_ep, p] == interact_code and rewards[t_ep, p] == 0:
          delta_time = t_ep - last_interact_time[p]
          if delta_time < cooldown:
            reason = f"repeat in {delta_time} timesteps"
          else:
            last_interact_time[p] = t_ep
            alive_prey = [q for q in preys if death_labels[q][t_seg] == 1]
            if not alive_prey:
              reason = 'no prey alive'
            else:
              # group defense
              nearby_predators = [
                r for r in predators if np.linalg.norm(positions[t_ep, r] - positions[t_ep, p]) <= radius3
              ]
              is_fenced = False
              for q in alive_prey:
                if ori_position(positions[t_ep, p], orientations[t_ep, p], positions[t_ep, q]) == (0, 1):
                  defenders = [
                    k for k in alive_prey
                    if k != q and np.linalg.norm(positions[t_ep, k] - positions[t_ep, p]) <= radius3
                  ]
                  if len(defenders) >= len(nearby_predators):
                    is_fenced = True
                    break
              if is_fenced:
                reason = 'fenced'
              else:
                # distance to closest alive prey
                distances_to_alive: Dict[int, float] = {
                  q: np.linalg.norm(positions[t_ep, q] - positions[t_ep, p]) for q in alive_prey
                }
                closest_prey = min(distances_to_alive, key=distances_to_alive.get) if distances_to_alive else None
                min_distance = distances_to_alive.get(closest_prey) if closest_prey is not None else float('inf')

                if min_distance <= radius2:
                  # find last movement
                  last_move_time = None
                  if closest_prey is not None:
                    for tau_seg in range(t_seg - 1, -1, -1):
                      tau_ep = t0 + tau_seg
                      if not np.allclose(positions[tau_ep, closest_prey], positions[tau_ep + 1, closest_prey]):
                        last_move_time = tau_ep + 1
                        break
                  if last_move_time is not None:
                    relative_pos_last_move = ori_position(
                      positions[last_move_time, p],
                      orientations[last_move_time, p],
                      positions[last_move_time, closest_prey],
                    )
                    distance_last_move = np.linalg.norm(
                      positions[last_move_time, closest_prey] - positions[last_move_time, p]
                    )
                    if relative_pos_last_move == (0, 1) and distance_last_move <= radius2:
                      reason = 'late shot'
                    else:
                      reason = 'miss prediction'
                  else:
                    reason = 'miss prediction'
                elif min_distance <= radius3:
                  reason = 'off target r3'
                else:
                  reason = 'exception'
          events.append({
            'time': t_ep,
            'segment_time': t_seg,
            'predator': p,
            'invalid_interact': reason,
          })
  return events


if __name__ == '__main__':
  # Example usage (replace with your actual data)
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

    death_labels_ep = {i: mark_death_periods(sta_ep[:, i]) for i in range(sta_ep.shape[1])}
    death_labels_all_prey_ep = {pi: death_labels_ep[pi] for pi in prey_ids if pi in death_labels_ep}
    active_segments = segment_active_phases(death_labels_all_prey_ep)

    all_good_gathering = []
    all_fencing_events = []
    all_invalid_interactions = []
    for seg_idx, (start, end) in enumerate(active_segments):
      # compute_good_gathering
      good_gathering = compute_good_gathering_segmented(
        pos_ep, predator_ids, prey_ids, active_segments=[(start, end)], death_labels=death_labels_ep
      )
      all_good_gathering.extend(good_gathering)
      # compute_successful_fencing_and_helpers
      fencing_events = mark_events_successful_fencing_and_helpers_segmented(
        pos_ep,
        ori_ep,
        act_ep,
        rew_ep,
        predator_ids,
        prey_ids,
        death_labels_ep,
        active_segments=[(start, end)],
        radius=3,
      )
      all_fencing_events.extend(fencing_events)

      # compute_invalid_interactions
      invalid_interactions = mark_events_compute_invalid_interactions_segmented(
        act_ep,
        rew_ep,
        pos_ep,
        ori_ep,
        sta_ep,
        predator_ids,
        prey_ids,
        active_segments=[(start, end)],
        cooldown=5,
        radius2=2,
        radius3=3,
        interact_code=7,
      )
      all_invalid_interactions.extend(invalid_interactions)

    episode_segment_dfs.append(
      {
        "folder": base,
        "episode": fname.replace(".pkl", ""),
        "seg_idx": [(start, end) for start, end in active_segments],
        "t0": active_segments[0][0] if active_segments else 0,
        "t1": active_segments[-1][1] if active_segments else len(pos_ep) if pos_ep.any() else 0,
        "good_gathering": all_good_gathering,
        "fencing_events": all_fencing_events,
        "invalid_interactions": all_invalid_interactions,
      }
    )

  # Convert to DataFrame
  import pandas as pd

  df = pd.DataFrame(episode_segment_dfs)
  print(df)
  # Save to CSV
  if not os.path.exists("example"):
    os.makedirs("example")
  df.to_csv("example/gather_and_fence_analysis_results_example.csv", index=False)
