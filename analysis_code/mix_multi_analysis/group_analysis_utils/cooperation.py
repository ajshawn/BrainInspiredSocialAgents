'''
In this file, we define functions to analyze collective apple collection behavior in a simulation environment.
The acorn collection behavior is not yet implemented.
'''

import numpy as np
from typing import List, Dict, Tuple, Optional
from map_utils import get_safe_tiles, get_apple_tiles, get_acorn_tiles
# TODO: replicate apple_periods but:
#   • take only events whose centre lies within `acorn_tiles`
#   • stop the period once ANY participant reaches safe grass carrying acorn
# Return structure identical to apple_periods.
def acorn_periods(*args, **kwargs):
  """Not implemented yet."""
  return None


def apple_frames(pos: np.ndarray, rew: np.ndarray, death: Dict[int, List[int]],
                 prey: List[int], safe_grass_locs: set, t0: int, t1: int,
                 cluster_radius: int = 3, apple_min_prey: int = 2) -> List[Dict]:
  """
  Detects frames where at least `apple_min_prey` prey are within
  `cluster_radius` of each other while collecting an apple, not on safe grass.
  This function is currently not used in the main analysis but is kept.

  Args:
      pos: Agent positions (T, A, 2).
      rew: Agent rewards (T, A).
      death: Dictionary of death labels {agent_id: [0 or 1 for each timestep]}.
      prey: List of prey agent IDs.
      safe_grass_locs: Set of (x, y) coordinates of safe grass tiles.
      t0: Start timestep.
      t1: End timestep.
      cluster_radius: Maximum distance between prey for them to be considered
          in a cluster.
      apple_min_prey: Minimum number of prey required to define a collective
          apple collection event.

  Returns:
      A list of dictionaries, where each dictionary represents a single frame
      meeting the criteria.  Each dictionary has the following keys:
      - 'time': The timestep of the event.
      - 'participants': A list of prey agent IDs participating in the event.
  """
  events = []
  for t in range(t0, t1):
    collectors = [q for q in prey if death[q][t] == 1 and rew[t, q] == 1 and tuple(pos[t, q]) not in safe_grass_locs]
    if len(collectors) >= apple_min_prey:
      group_positions = pos[t, collectors]
      distances = np.linalg.norm(group_positions[:, None] - group_positions[None, :], axis=2)
      if np.max(distances) <= cluster_radius * 2:
        events.append({
          'time': t,
          'participants': collectors,
        })
  return events


import numpy as np
from typing import List, Dict, Tuple, Optional


def apple_frames(pos: np.ndarray, rew: np.ndarray, death: Dict[int, List[int]],
                 prey: List[int], safe_grass_locs: set, t0: int, t1: int,
                 cluster_radius: int = 3, apple_min_prey: int = 2) -> List[Dict]:
  """
  Detects frames where at least `apple_min_prey` prey are within
  `cluster_radius` of each other while collecting an apple, not on safe grass.

  Args:
      pos: Agent positions (T, A, 2).
      rew: Agent rewards (T, A).
      death: Dictionary of death labels {agent_id: [0 or 1 for each timestep]}.
      prey: List of prey agent IDs.
      safe_grass_locs: Set of (x, y) coordinates of safe grass tiles.
      t0: Start timestep.
      t1: End timestep.
      cluster_radius: Maximum distance between prey for them to be considered
          in a cluster.
      apple_min_prey: Minimum number of prey required to define a collective
          apple collection event.

  Returns:
      A list of dictionaries, where each dictionary represents a single frame
      meeting the criteria.  Each dictionary has the following keys:
      - 'time': The timestep of the event.
      - 'participants': A list of prey agent IDs participating in the event.
  """
  events = []
  for t in range(t0, t1):
    collectors = [q for q in prey if death[q][t] == 1 and rew[t, q] == 1 and tuple(pos[t, q]) not in safe_grass_locs]
    if len(collectors) >= apple_min_prey:
      group_positions = pos[t, collectors]
      distances = np.linalg.norm(group_positions[:, None] - group_positions[None, :], axis=2)
      if np.max(distances) <= cluster_radius * 2:
        events.append({
          'time': t,
          'participants': collectors,
        })
  return events


def apple_periods(pos: np.ndarray, rew: np.ndarray, death: Dict[int, List[int]],
                  prey: List[int], safe_grass_locs: set, t0: int, t1: int,
                  cluster_radius: int = 3, apple_period_min_len: int = 5,
                  apple_period_gap: int = 2,
                  participation_threshold: float = 0.4,
                  ) -> List[Dict]:
  """
  Detects periods of collective apple collection, where prey are within
  `cluster_radius` of each other while collecting apples, not on safe grass,
  for at least `apple_period_min_len` frames, with gaps no larger than
  `apple_period_gap`.  Detects multiple periods.

  Args:
      pos: Agent positions (T, A, 2).
      rew: Agent rewards (T, A).
      death: Dictionary of death labels {agent_id: [0 or 1 for each timestep]}.
      prey: List of prey agent IDs.
      safe_grass_locs: Set of (x, y) coordinates of safe grass tiles.
      t0: Start timestep.
      t1: End timestep.
      cluster_radius: Maximum distance between prey for them to be considered
          in a cluster.
      apple_period_min_len: Minimum length of a collective apple collection
          period (in timesteps).
      apple_period_gap: Maximum gap (in timesteps) between frames to be
          considered part of the same period.
      participation_threshold: The minimum fraction of the event duration

  Returns:
      A list of dictionaries, where each dictionary represents a collective
      apple collection period.  Each dictionary has the following keys:
      - 'time_start': The start timestep of the period.
      - 'time_end': The end timestep of the period.
      - 'participants': A list of prey agent IDs participating in the period.
      - 'center_positions': A list of (x,y) positions of the cluster center
          at each timestep of the period
  """
  events = []
  t = t0
  while t < t1:
    seeds = [q for q in prey if rew[t, q] == 1 and death[q][t] == 1 and tuple(pos[t,q]) not in safe_grass_locs]
    if not seeds:
      t += 1
      continue
    center = pos[t, seeds[0]]
    group = {seeds[0]}
    last_apple_time = t
    duration = 1
    center_route = [center] # record the center positions
    agent_participation_count = {seed: 1 for seed in seeds} # count participation
    for tau in range(t + 1, t1):
      if tau - last_apple_time > apple_period_gap:
        break
      still_close = {q for q in group if
                     (np.linalg.norm(pos[tau, q] - center) <= cluster_radius) and death[q][tau] == 1}
      new_joiners = [q for q in prey if
                     rew[tau, q] == 1 and np.linalg.norm(pos[tau, q] - center) <= cluster_radius and death[q][tau] == 1]
      if new_joiners:
        last_apple_time = tau
        still_close.update(new_joiners)
        for joiner in new_joiners:
          agent_participation_count[joiner] = agent_participation_count.get(joiner, 0) + 1
      if not still_close:
        break
      group = still_close
      for q in group:
        agent_participation_count[q] = agent_participation_count.get(q, 0) + 1
      center = np.median(pos[tau, list(group)], axis=0)
      center_route.append(center)
      duration += 1

    if duration >= apple_period_min_len:
      # Filter participants based on the new criteria
      core_participants = [agent for agent, count in agent_participation_count.items() if count >= 5]
      qualified_participants = [
        agent for agent, count in agent_participation_count.items()
        if count >= 5 and (count >= duration * participation_threshold)
      ]
      if len(core_participants) >= 2:
        events.append({
          'time_start': t,
          'time_end': t + duration -1, # inclusive
          'participants': qualified_participants,
          'center_positions': center_route,
        })
    t += duration # Important:  Move t past the current event.
  return events



def apple_periods_with_apple_maps(pos: np.ndarray, rew: np.ndarray,
                                  death: Dict[int, List[int]], prey: List[int],
                                  safe_grass_locs: set, t0: int, t1: int,
                                  apple_tiles: set,
                                  cluster_radius: int = 3,
                                  apple_period_min_len: int = 5,
                                  apple_period_gap: int = 2,
                                  participation_threshold: float = 0.4) -> List[Dict]:
  """
  Detects periods of collective apple collection, considering proximity to
  apple tiles.  This version allows for dynamic participant changes and
  reports collective events.  Detects multiple periods.

  Args:
      pos: Agent positions (T, A, 2).
      rew: Agent rewards (T, A).
      death: Dictionary of death labels {agent_id: [0 or 1 for each timestep]}.
      prey: List of prey agent IDs.
      safe_grass_locs: Set of (x, y) coordinates of safe grass tiles.
      t0: Start timestep.
      t1: End timestep.
      apple_tiles: Set of (x, y) coordinates of apple tiles.
      cluster_radius: Maximum distance between prey and apple tile.
      apple_period_min_len: Minimum length of a collective apple collection
          period (in timesteps).
      apple_period_gap: Maximum gap (in timesteps) between frames to be
          considered part of the same period.
      participation_threshold: The minimum fraction of the event duration

  Returns:
      A list of dictionaries, where each dictionary represents a collective
      apple collection period.  Each dictionary has the following keys:
      - 'time_start': The start timestep of the period.
      - 'time_end': The end timestep of the period.
      - 'apple_tile': The (x, y) coordinate of the apple tile.
      - 'participants': A list of prey agent IDs participating in the period.
      - 'center_positions': List of center positions
  """
  events = []
  apple_tiles_arr = np.array(list(apple_tiles))
  t = t0
  while t < t1:
    seeds = [(q, tile_idx) for q in prey
             for tile_idx, tile_pos in enumerate(apple_tiles_arr)
             if rew[t, q] == 1 and death[q][t] == 1 and
             np.linalg.norm(pos[t, q] - tile_pos) <= cluster_radius and
             tuple(pos[t,q]) not in safe_grass_locs]
    if not seeds:
      t += 1
      continue

    # Use the first seed as a starting point.
    seed_prey, seed_tile_idx = seeds[0]
    center = pos[t, seed_prey]
    group = {seed_prey}
    last_apple_time = t
    duration = 1
    apple_tile = tuple(apple_tiles_arr[seed_tile_idx])
    center_route = [center]
    agent_participation_count = {seed_prey: 1}

    for tau in range(t + 1, t1):
      if tau - last_apple_time > apple_period_gap:
        break

      still_close = {q for q in group if death[q][tau] == 1 and np.linalg.norm(pos[tau,q] - center) <= cluster_radius}

      new_joiners = [(q, tile_idx) for q in prey
                     for tile_idx, tile_pos in enumerate(apple_tiles_arr)
                     if rew[tau, q] == 1 and death[q][tau] == 1 and
                     np.linalg.norm(pos[tau, q] - tile_pos) <= cluster_radius and tuple(pos[tau,q]) not in safe_grass_locs]
      new_joiners_same_tile = [q for q, tile_idx in new_joiners if tuple(apple_tiles_arr[tile_idx]) == apple_tile]
      if new_joiners_same_tile:
        last_apple_time = tau
        still_close.update(new_joiners_same_tile)
        for joiner in new_joiners_same_tile:
          agent_participation_count[joiner] = agent_participation_count.get(joiner, 0) + 1

      if not still_close:
        break
      group = still_close
      for q in group:
        agent_participation_count[q] = agent_participation_count.get(q, 0) + 1
      center = np.median(pos[tau, list(group)], axis=0)
      center_route.append(center)
      duration += 1

    if duration >= apple_period_min_len:
      events.append({
        'time_start': t,
        'time_end': t + duration - 1,
        'apple_tile': apple_tile,
        'participants': list(agent_participation_count.keys()),
        'center_positions': center_route
      })
    t += duration  # Move t past the current event
  return events


def analyze_collective_apple_behavior(pos: np.ndarray, rew: np.ndarray,
                                      death: Dict[int, List[int]], prey: List[int],
                                      safe_grass_locs: set, t0: int, t1: int,
                                      apple_tiles: set,
                                      cluster_radius: int = 3,
                                      apple_period_min_len: int = 5,
                                      apple_period_gap: int = 2,
                                      participation_threshold: float = 0.4) -> Dict:
  """
  Analyzes collective apple collection behavior, unifying the logic of
  `apple_frames` and `apple_periods` and allowing for multiple event detections.

  Args:
      pos: Agent positions (T, A, 2).
      rew: Agent rewards (T, A).
      death: Dictionary of death labels {agent_id: [0 or 1 for each timestep]}.
      prey: List of prey agent IDs.
      safe_grass_locs: Set of (x, y) coordinates of safe grass tiles.
      t0: Start timestep.
      t1: End timestep.
      apple_tiles: Set of (x, y) coordinates of apple tiles.
      cluster_radius: Maximum distance between prey for them to be considered
          in a cluster.
      apple_period_min_len: Minimum length of a collective apple collection
          period (in timesteps).
      apple_period_gap: Maximum gap (in timesteps) between frames to be
          considered part of the same period.
      participation_threshold: The minimum fraction of the event duration
          an agent must participate to be included in the final participants list.

  Returns:
      A dictionary containing the analysis results, with the following keys:
      - 'apple_frames': A list of dictionaries, where each dictionary
          represents a single frame meeting the criteria (as in the original
          `apple_frames` output).
      - 'apple_periods': A list of dictionaries, where each dictionary
          represents a collective apple collection period (as in the
          original `apple_periods` output).
      - 'apple_periods_with_maps': A list of dictionaries, where each dictionary
          represents a collective apple collection period, considering
          apple tiles.
  """
  apple_frame_events = apple_frames(pos, rew, death, prey, safe_grass_locs, t0, t1, cluster_radius)
  apple_period_events = apple_periods(pos, rew, death, prey, safe_grass_locs, t0, t1, cluster_radius,
                                      apple_period_min_len, apple_period_gap, participation_threshold)
  apple_period_map_events = apple_periods_with_apple_maps(pos, rew, death, prey, safe_grass_locs, t0, t1, apple_tiles,
                                                          cluster_radius, apple_period_min_len, apple_period_gap,
                                                          participation_threshold)
  return {
    'apple_frames': apple_frame_events,
    'apple_periods': apple_period_events,
    'apple_periods_with_maps': apple_period_map_events
  }


if __name__ == '__main__':
  # Example usage (replace with your actual data)
  # Create dummy data for demonstration
  episode_name = ('mix_AH20250107_pred_0_AH20250107_pred_1_AH20250107_prey_3_AH20250107_prey_4_'
                  'AH20250107_prey_5_AH20250107_prey_6predator_prey__open_debug_agent_0_1_3_4_5_6')
  folder = f'/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/{episode_name}/episode_pickles/'

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

    for seg_idx, (start, end) in enumerate(active_segments):
      # Analyze the segment
      apple_events = analyze_collective_apple_behavior(
        pos_ep, rew_ep, death_labels_ep, prey_ids, safe_tiles, start, end,
        apple_tiles, cluster_radius=3, apple_period_min_len=5,
        apple_period_gap=2, participation_threshold=1 / 3
      )
      # Store the results for this segment
      episode_segment_dfs.append({
        'folder': base,
        'episode': fname.replace('.pkl', ''),
        'seg_idx': seg_idx,
        't0': start,
        't1': end,
        **apple_events
      })

  # Convert to DataFrame
  import pandas as pd
  df = pd.DataFrame(episode_segment_dfs)
  print(df)
  # Save to CSV
  if not os.path.exists('example'):
    os.makedirs('example')
  df.to_csv('example/apple_analysis_results_example.csv', index=False)


