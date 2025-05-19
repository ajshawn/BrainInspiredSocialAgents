import numpy as np
from typing import List, Dict, Tuple, Optional, Sequence


def _is_moving_away(
    prev_pred_pos: np.ndarray,
    prev_prey_pos: np.ndarray,
    curr_pred_pos: np.ndarray,
    curr_prey_pos: np.ndarray,
    move_thresh: float,
) -> bool:
  """
  Check if the prey is moving away from the predator.

  Args:
      prev_pred_pos: Previous predator position (NumPy array, shape (2,)).
      prev_prey_pos: Previous prey position (NumPy array, shape (2,)).
      curr_pred_pos: Current predator position (NumPy array, shape (2,)).
      curr_prey_pos: Current prey position (NumPy array, shape (2,)).
      move_thresh: Distance threshold for considering movement "away".

  Returns:
      True if the predator is within a certain radius and the prey
      has moved away from it by more than the threshold.
  """
  d_prev = np.linalg.norm(prev_pred_pos - prev_prey_pos)
  d_curr = np.linalg.norm(curr_pred_pos - curr_prey_pos)
  return d_prev <= 3.0 and (d_curr - d_prev) > move_thresh  # Use radius here



def _find_distraction_t_end(
    positions: np.ndarray,
    orientations: np.ndarray,
    death_labels: Dict[int, np.ndarray],
    safe_grass: List[Tuple[int, int]],
    predator: int,
    prey_B: int,
    start_t: int,
    max_t: int,
    radius: float,
    window_away: int,
    max_period_duration: int,
) -> Tuple[int, str]:
  """
  Find the end timestep and reason for the end of a distraction event.

  Args:
      positions: Agent positions over the episode (NumPy array, shape (episode_length, num_agents, 2)).
      orientations: Agent orientations over the episode (NumPy array, shape (episode_length, num_agents, 2)).
      death_labels: Dictionary of agent death labels for the episode.
          Dict: {agent_index: np.array of aliveness (1) / death (0) flags}.
      safe_grass: List of safe grass tile coordinates.
      predator: Index of the predator.
      prey_B: Index of the prey being distracted to.
      start_t: Starting timestep of the distraction event.
      max_t: Maximum timestep of the episode.
      radius: Maximum distance for the predator to be considered "near".
      window_away: Window size for checking if the predator moved away.
      max_period_duration: Maximum duration of the distraction event.

  Returns:
      A tuple containing the ending timestep and a string indicating the reason
      for the end of the event.  End reasons include:
      - 'helper_on_grass': Prey B reached the grass.
      - 'helper_dead': Prey B died.
      - 'dist_ge_4': Distance between predator and prey B >= 4.0 (now >= radius).
      - 'pred_move_away': Predator moved away from prey B.
      - 'hard_cap': Hard cap on event duration reached (now max_period_duration).
      - 'predator_dead': The predator died.
  """

  def _on_grass(agent_index: int, timestep: int) -> bool:
    """Check if an agent is on the grass at a given timestep."""
    return tuple(positions[timestep, agent_index]) in safe_grass

  dist_hist = []
  for offset in range(max_period_duration + 1): # changed hard_cap
    tau = start_t + offset
    if (tau > max_t) or (tau >= len(positions)):
      break
    # print(tau, max_t)
    # 1) B reaches grass
    if _on_grass(prey_B, tau):
      return tau, "helper_on_grass"
    # 2) B is eaten
    if death_labels[prey_B][tau] == 0:
      return tau, "helper_dead"
    # 3) predator dead?
    if death_labels[predator][tau] == 0:
      return tau, "predator_dead"

    # 3) distance >= radius
    d = np.linalg.norm(positions[tau, predator] - positions[tau, prey_B])
    if d >= radius: # changed from 4.0
      return tau, "dist_ge_4"

    # 4) predator moved away 3/5
    dist_hist.append(d)
    if len(dist_hist) == window_away:
      if np.sum(np.diff(dist_hist) > 0) >= 3:
        return tau, "pred_move_away"
      dist_hist.pop(0)

  # 5) hard cap
  t_cap = min(max_t, start_t + max_period_duration) # changed hard_cap
  return t_cap, "hard_cap"



def detect_distraction_events(
    positions: np.ndarray,
    orientations: np.ndarray,
    death_labels: Dict[int, np.ndarray],
    safe_grass: List[Tuple[int, int]],
    predators: Sequence[int], # Changed predator_ids
    preys: Sequence[int], # Changed prey_ids
    active_segments: List[Tuple[int, int]], # Added active_segments
    window_away: int=5, # Added window_away
    shift_window: int=5, # Added shift_window
    distraction_period_gap: int=2, # Added
    radius: float=3, # Added
    move_thresh: float=0.5, # Added
    max_period_duration: int=30, # Changed hard_cap -> max_period_duration
    scenario: str = 'grass', # Added scenario
) -> List[Dict]:
  """
  Detect distraction events (either "grass" or "chase" scenario) within specified segments of an episode.

  Args:
      positions: Agent positions over the episode (NumPy array, shape (episode_length, num_agents, 2)).
      orientations: Agent orientations over the episode (NumPy array, shape (episode_length, num_agents, 2)).
      death_labels: Dictionary of agent death labels for the episode.
          Dict: {agent_index: np.array of aliveness (1) / death (0) flags}.
      safe_grass: List of safe grass tile coordinates.
      predators: Sequence of predator agent IDs.
      preys: Sequence of prey agent IDs.
      active_segments:  List of tuples, where each tuple (start, end) defines a segment.
      window_away: Window size for checking if the predator moved away.
      shift_window:  Window size for checking if the predator can shift to prey B.
      distraction_period_gap:  Minimum time gap required between consecutive distraction events.
      radius: Maximum distance for the predator to be considered "near" a prey.
      move_thresh: Distance threshold for considering movement "away".
      max_period_duration: Maximum duration of a distraction event.
      scenario: Type of distraction scenario to detect ("grass" or "chase").

  Returns:
      A list of dictionaries, where each dictionary describes a detected
      distraction event.  Each event dictionary contains:
      - 'segment_start': Start timestep of the segment.
      - 'segment_end': End timestep of the segment.
      - 'events': A list of distraction events within the segment, where each event is a dictionary containing:
          - 'scenario':  "grass" or "chase".
          - 'time_start': Timestep when the initial phase (lurking/chasing) began.
          - 'time_shift': Timestep when the predator switched to prey B.
          - 'time_end': Timestep when the distraction event ended.
          - 'predator':  Predator agent ID.
          - 'prey_A':  Prey A agent ID (the initial prey).
          - 'prey_B':  Prey B agent ID (the prey being distracted to).
          - 'end_reason':  Reason why the event ended.
  """
  all_segment_events: List[Dict] = [] # Changed the output

  for seg_start, seg_end in active_segments:
    events = []
    t = seg_start
    last_event_end_time = -distraction_period_gap # Initialize

    P, A, B_id = None, None, None # Initialize
    while t <= seg_end - window_away: # Changed from min_window to window_away
      for P in predators:
        if death_labels[P][t] == 0:
          continue
        for A in preys:
          if scenario == "grass":
            if tuple(positions[t, A]) not in safe_grass:
              continue
            window_ok = True
            for dt in range(window_away): # Changed from min_window to window_away
              tau = t + dt
              if (
                  tau >= seg_end
                  or death_labels[A][tau] == 0
                  or death_labels[P][tau] == 0
              ):
                window_ok = False
                break
              if (
                  np.linalg.norm(positions[tau, P] - positions[tau, A])
                  > radius # use radius
              ):
                window_ok = False
                break
            if not window_ok:
              continue
          elif scenario == "chase":
            if tuple(positions[t, A]) in safe_grass:
              continue
            away_count = 0
            valid = True
            for dt in range(window_away): # Changed from min_window to window_away
              tau = t + dt
              if (
                  tau >= seg_end
                  or death_labels[A][tau] == 0
                  or death_labels[P][tau] == 0
              ):
                valid = False
                break
              if (
                  np.linalg.norm(positions[tau, P] - positions[tau, A])
                  > radius # Use radius
              ):
                valid = False
                break
              if (
                  dt > 0
                  and _is_moving_away(
                positions[tau - 1, P],
                positions[tau - 1, A],
                positions[tau, P],
                positions[tau, A],
                move_thresh, # Use move_thresh
              )
              ):
                away_count += 1
            if not valid or away_count < 3:
              continue
          else:
            raise ValueError(
              f"Invalid scenario: {scenario}. Must be 'grass' or 'chase'."
            )

          # -- shift to B --
          shift_found, B_id, shift_t = False, None, None
          for dt in range(1, shift_window + 1):
            tau = t + window_away - 1 + dt # changed from min_window
            if tau >= seg_end:
              break
            candidates = [
              q
              for q in preys
              if q != A
                 and death_labels[q][tau] == 1
                 and np.linalg.norm(
                positions[tau, P] - positions[tau, q]
              )
                 <= radius # Use radius
            ]
            if not candidates:
              continue
            B = min(
              candidates,
              key=lambda q: np.linalg.norm(
                positions[tau, P] - positions[tau, q]
              ),
            )
            if (
                np.linalg.norm(positions[tau, P] - positions[tau, B])
                < np.linalg.norm(positions[tau, P] - positions[tau, A])
            ):
              shift_found, B_id, shift_t = True, B, tau
              break
          if not shift_found:
            continue

          # -- termination --
          t_end, end_note = _find_distraction_t_end(
            positions,
            orientations,
            death_labels,
            safe_grass,
            predator=P,
            prey_B=B_id,
            start_t=shift_t,
            max_t=seg_end, # Use seg_end
            radius=radius, # Pass radius
            window_away=window_away, # Pass window_away
            max_period_duration=max_period_duration, # Pass max_period_duration
          )
          # Check for gap between events
          if t_end <= last_event_end_time + distraction_period_gap:
            continue

          events.append(
            dict(
              scenario=scenario,
              time_start=t,
              time_shift=shift_t,
              time_end=t_end,
              predator=P,
              beneficiary=A,
              helper=B_id,
              end_reason=end_note,
            )
          )
          t = t_end
          last_event_end_time = t_end # Update last_event_end_time
          break
        else:
          continue
        break
      else:
        t += 1
    if events:
      all_segment_events.extend(events)
  return all_segment_events


if __name__ == "__main__":

  # Example usage (replace with your actual data)
  episode_name = ('mix_AH20250107_pred_0_AH20250107_pred_1_AH20250107_prey_3_AH20250107_prey_4_'
                  'AH20250107_prey_5_AH20250107_prey_6predator_prey__open_debug_agent_0_1_3_4_5_6')
  folder = f'/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/{episode_name}/episode_pickles/'
  from map_utils import get_safe_tiles, get_apple_tiles
  safe_tiles = get_safe_tiles(episode_name, map='smaller_13x13')
  apple_tiles = get_apple_tiles(episode_name, map='smaller_13x13')

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

    # Detect distraction events
    distraction_events = detect_distraction_events(
      positions=pos_ep,
      orientations=ori_ep,
      death_labels=death_labels_ep,
      safe_grass=safe_tiles,
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
      safe_grass=safe_tiles,
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

    episode_segment_dfs.append({
      "episode_name": episode_name,
      "active_segments": active_segments,
      "distraction_events": distraction_events,
    })
    print(distraction_events)
  import pandas as pd
  df = pd.DataFrame(episode_segment_dfs)
  df.to_csv("example/distraction_events_example.csv", index=False)
  print(f"Distraction events saved to example/distraction_events_example.csv")
