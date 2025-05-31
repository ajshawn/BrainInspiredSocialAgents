#!/usr/bin/env python3
"""
analyze_mixed_results_extended.py

Extends analyze_mixed_results_flexible.py to include metrics for
coalition and altruism in N predators vs M preys environments.
"""
import os
import re
import pickle
import numpy as np
import pandas as pd
import importlib

# Attempt to import from the user's existing script
# If this script is in the same directory, this should work.
# Otherwise, ensure basic_behavioral_metrics.py is in your PYTHONPATH.
try:
  from analysis_code.mix_multi_analysis.group_analysis_utils.basic_behavioral_metrics import (
    parse_agent_roles,
    mark_death_periods,
    compute_agent_move_distance,
    compute_num_rotation,
    compute_collect_counts,
    compute_stuck_rate_all,  # Assuming this now uses death_labels correctly
    compute_death_counts,
    compute_grass_time_per_life,
    get_safe_grass_locations,
    compute_pairwise_distance_mean,
    ori_position,  # Assuming this is the same as used for fencing
    compute_good_gathering,
    compute_successful_fencing_and_helpers,
    compute_invalid_interactions
  )
except ImportError:
  print("ERROR: Could not import from basic_behavioral_metrics.py.")
  print("Please ensure it's in the same directory or your PYTHONPATH.")
  # Define stubs or raise error if critical functions are missing
  # For this example, I'll assume the import will succeed.
  # If not, you'd need to copy those function definitions here.
  raise

# --- Stamina and Action Configuration ---
PREDATOR_STAMINA_CONFIG = {
  "max_stamina": 18,  # 0 to 18 are 19 states
  "levels": {
    "invisible": {"range": (13, 18), "freeze_time": 0},  # 6 states
    "green": {"range": (7, 12), "freeze_time": 0},  # 6 states
    "yellow": {"range": (1, 6), "freeze_time": 1},  # 6 states
    "red": {"range": (0, 0), "freeze_time": 6}  # 1 state
  }
}

PREY_STAMINA_CONFIG = {
  "max_stamina": 18,
  "levels": {
    "invisible": {"range": (13, 18), "freeze_time": 1},
    "green": {"range": (7, 12), "freeze_time": 1},
    "yellow": {"range": (1, 6), "freeze_time": 2},
    "red": {"range": (0, 0), "freeze_time": 4}
  }
}

# --- Parameters for New Metrics ---
# TODO: Tune these parameters based on your environment's scale and dynamics
PARAMS = {
  "episode_activity_min_prey_alive": 1,
  "episode_activity_min_dead_duration": 5,  # Min steps all prey must be dead to end a segment
  "post_fencing_focus_radius": 2,  # How close predator stays to focused prey
  "post_fencing_disengage_steps": 8,  # Steps predator looks away to be "disengaged"
  "collective_apple_min_prey": 2,
  "collective_apple_radius": 3,  # Radius around apple for considering collection
  "collective_apple_time_window": 5,  # Timesteps for simultaneous targeting
  "bodyguard_lurking_radius_A": 5,  # Predator near Prey A (at grass)
  "bodyguard_P_reaction_dist_B": 10,  # Max distance P reacts to B
  "bodyguard_fov_max_dist": 10,  # Max observation distance for FOV
  "bodyguard_delta_t_reaction": 5,  # Time for P to react to B
  "bodyguard_delta_t_A_acts": 10,  # Time for A to benefit
  "bodyguard_cost_B_window": 20,  # Window to check cost for B
  "altruistic_fencing_risk_window": 30,  # Window to check if fencer caught after event
  "movement_sync_default_window": 5,  # Default window for synchronization
  "min_meaningful_move_dist": 1,  # Min distance change to be considered movement
}


# --- New Helper Functions ---

def _load_substrate_cfg(folder_path: str):
  """Infer the MeltingPot substrate module from *folder_path* and return its
  config object (calling get_config() with the right kwargs)."""
  m = re.search(r'(predator_prey__[^_/]+)', folder_path)
  substrate = m.group(1) if m else 'predator_prey__open'
  module_name = f"meltingpot.python.configs.substrates.{substrate}"
  mod = importlib.import_module(module_name)

  # choose map size flag from folder name if present
  kwargs = {}
  if 'smaller_13x13' in folder_path:
    kwargs['smaller_13x13'] = True
  elif 'smaller_10x10' in folder_path:
    kwargs['smaller_10x10'] = True
  return mod.get_config(**kwargs)


def get_apple_locations(folder_path: str):
  """Return a *set* of (x,y) tuples for all tiles that can spawn an APPLE or
  its waiting state."""
  cfg = _load_substrate_cfg(folder_path)
  amap = cfg.layout.ascii_map.strip('\n').splitlines()
  prefabs = cfg.layout.char_prefab_map

  apple_tiles = set()
  for y, row in enumerate(amap):
    for x, ch in enumerate(row):
      prefab_entry = prefabs.get(ch)
      if prefab_entry is None:
        continue
      # prefab_entry can be str or dict{'type':'all','list':[...]}
      prefab_list = ([prefab_entry] if isinstance(prefab_entry, str)
                     else prefab_entry.get('list', []))
      if any(p.startswith('apple') for p in prefab_list):
        apple_tiles.add((x, y))
  return apple_tiles

def get_acorn_locations(folder_path: str):
  """Return a set of (x,y) tiles that spawn an ACORN."""
  cfg = _load_substrate_cfg(folder_path)
  amap = cfg.layout.ascii_map.strip('\n').splitlines()
  prefabs = cfg.layout.char_prefab_map

  acorn_tiles = set()
  for y, row in enumerate(amap):
    for x, ch in enumerate(row):
      prefab_entry = prefabs.get(ch)
      if prefab_entry is None:
        continue
      prefab_list = ([prefab_entry] if isinstance(prefab_entry, str)
                     else prefab_entry.get('list', []))
      if any(p.startswith('floor_acorn') for p in prefab_list):
        acorn_tiles.add((x, y))
  return acorn_tiles

def get_stamina_level_and_freeze(stamina_value, config):
  for level_name, details in config["levels"].items():
    if details["range"][0] <= stamina_value <= details["range"][1]:
      return level_name, details["freeze_time"]
  return "unknown", 0


def get_agent_freeze_time(stamina_value, role, predator_config, prey_config):
  if role == "predator":
    _, freeze_time = get_stamina_level_and_freeze(stamina_value, predator_config)
  elif role == "prey":
    _, freeze_time = get_stamina_level_and_freeze(stamina_value, prey_config)
  else:
    freeze_time = 0
  return freeze_time


def segment_episode_by_activity(death_labels_all_prey, min_prey_alive=1, min_all_dead_duration=5):
  """
  Segments an episode into active phases where at least min_prey_alive are active.
  An active phase ends when all prey are dead for min_all_dead_duration, or episode ends.
  death_labels_all_prey: dict {prey_idx: np.array_of_death_labels (1=alive, 0=dead)}
  """
  if not death_labels_all_prey:
    return []

  prey_indices = list(death_labels_all_prey.keys())
  if not prey_indices:  # No prey in this setup
    # Get episode length from an arbitrary agent if available, or assume fixed length
    # This part might need adjustment based on how death_labels_all_prey is structured if empty
    # For now, if no prey, return no active segments.
    return []

  # Use length of first prey's death label array to determine episode length
  episode_len = len(death_labels_all_prey[prey_indices[0]])

  num_alive_prey_at_t = np.zeros(episode_len, dtype=int)
  for prey_idx in prey_indices:
    num_alive_prey_at_t += death_labels_all_prey[prey_idx]

  active_segments = []
  in_segment = False
  current_segment_start = 0

  consecutive_all_dead_count = 0

  for t in range(episode_len):
    if num_alive_prey_at_t[t] >= min_prey_alive:
      consecutive_all_dead_count = 0
      if not in_segment:
        in_segment = True
        current_segment_start = t
    else:  # num_alive_prey_at_t[t] < min_prey_alive
      if in_segment:  # Potentially end of segment
        if num_alive_prey_at_t[t] == 0:
          consecutive_all_dead_count += 1
        else:  # Some prey alive, but fewer than min_prey_alive threshold, reset counter
          consecutive_all_dead_count = 0

        if consecutive_all_dead_count >= min_all_dead_duration:
          active_segments.append((current_segment_start, t - min_all_dead_duration + 1))
          in_segment = False
          consecutive_all_dead_count = 0

  if in_segment:  # Segment was active until the end of the episode
    active_segments.append((current_segment_start, episode_len))

  return active_segments


def is_in_predator_fov(pred_pos, pred_ori, target_pos,
                       fov_front=9, fov_back=1, fov_sides=5, max_dist=10):
  """
  True iff *target_pos* lies inside the rectangular FOV of a predator.

  We rotate the world‑space vector into the predator’s egocentric frame
  via the existing `ori_position` helper:

      (dx_local, dy_local) = ori_position(pred_pos, pred_ori, target_pos)

  In this local frame:
      •  +y  = straight ahead
      •  -y  = straight behind
      •  +/‑x = left / right
  """
  dx_local, dy_local = ori_position(pred_pos, pred_ori, target_pos)

  # # Fast distance gate
  # if (dx_local * dx_local + dy_local * dy_local) > max_dist * max_dist:
  #   return False

  # Width & depth checks
  if -fov_sides <= dx_local <= fov_sides:
    if -fov_back < dy_local <= fov_front:
      return True
  return False


def get_closest_agents_in_fov(observer_pos, observer_ori, observer_idx,
                              all_agent_pos_at_t, all_agent_death_labels_at_t,
                              agent_indices_to_check,
                              fov_params):
  """Finds agents in FOV, returns sorted by distance."""
  agents_in_fov = []
  for agent_idx in agent_indices_to_check:
    if agent_idx == observer_idx or all_agent_death_labels_at_t.get(agent_idx, 0) == 0:
      continue
    target_pos = all_agent_pos_at_t[agent_idx]
    if is_in_predator_fov(observer_pos, observer_ori, target_pos, **fov_params):
      dist = np.linalg.norm(np.array(observer_pos) - np.array(target_pos))
      agents_in_fov.append({'id': agent_idx, 'dist': dist, 'pos': target_pos})

  return sorted(agents_in_fov, key=lambda x: x['dist'])


def is_agent_moving_towards(agent_pos_series_t, target_pos_series_t, agent_id, agent_role,
                            stamina_series_t, actions_series_t, orientations_series_t,
                            pred_config, prey_config,
                            min_dist_change=PARAMS["min_meaningful_move_dist"], look_ahead_steps=5):
  """
  Determines if an agent is "intending" or effectively moving towards a target,
  considering stamina-based freezes.
  This is a simplified heuristic.
  agent_pos_series_t: positions of the agent over time (window)
  target_pos_series_t: positions of the target over time (window)
  TODO: Refine this logic. Current version checks if orientation matches and some movement occurs.
  """
  if len(agent_pos_series_t) < 2 or len(target_pos_series_t) < 2:
    return False

  current_pos = agent_pos_series_t[-1]
  current_target_pos = target_pos_series_t[-1]
  current_stamina = stamina_series_t[-1]
  current_orientation = orientations_series_t[-1]

  # Check orientation: Is agent generally facing the target?
  # This uses the logic from ori_position to see if target is in front
  # (0,1) means target is 1 unit in front in agent's local coords.
  # We care if target is generally in front, not just exactly (0,1).
  # So check if y-component of relative position in agent's frame is positive.

  # Get relative position in agent's frame
  # This part re-uses logic from ori_position, ensure it's consistent
  # dx_local, dy_local = ori_position(current_pos, current_orientation, current_target_pos)
  # For simplicity, let's assume if the agent is moving, its orientation is a good indicator.
  # A more robust check of "facing target" would be needed.

  # Check position change: Did the agent get closer or attempt to?
  # Consider freeze time. If frozen for long, might not move.
  freeze_time = get_agent_freeze_time(current_stamina, agent_role, pred_config, prey_config)

  # Look at movement over a few steps, considering potential freeze
  # Find last non-frozen move opportunity
  # For this simplified version, we just check if the agent moved recently
  # and if its current orientation makes sense for moving towards target.

  # Heuristic: if agent is oriented towards target, and it's not in a long freeze,
  # and it has moved in the recent past in that general direction, consider it "moving towards".

  # Calculate vector to target
  vec_to_target = np.array(current_target_pos) - np.array(current_pos)
  dist_to_target = np.linalg.norm(vec_to_target)
  if dist_to_target < 0.5:  # Already there
    return True

  # Check if orientation aligns with vector to target
  # 0: Up (-Y), 1: Left (-X), 2: Down (+Y), 3: Right (+X)
  oriented_towards = False
  if current_orientation == 0 and vec_to_target[1] < 0:
    oriented_towards = True  # Moving -Y
  elif current_orientation == 1 and vec_to_target[0] < 0:
    oriented_towards = True  # Moving -X
  elif current_orientation == 2 and vec_to_target[1] > 0:
    oriented_towards = True  # Moving +Y
  elif current_orientation == 3 and vec_to_target[0] > 0:
    oriented_towards = True  # Moving +X

  if not oriented_towards:
    return False

  # Check if agent moved recently and reduced distance
  # This needs to be more sophisticated due to freeze.
  # For now, simple check: did distance decrease in last few steps if not frozen?
  if freeze_time < 2:  # Agent is relatively mobile
    # Check if distance decreased in the last `look_ahead_steps` if possible
    # This needs to be trajectory based, not just instantaneous action.
    # The definition given "action of choice at time (t) ... leads to a closer distance"
    # implies we need to know the *outcome* of an action.
    # Let's assume if oriented towards and not deeply frozen, it's "trying".

    # Check if actual position changed in the last few steps in a way that reduced distance
    # This part is tricky because an action is chosen, then freeze occurs.
    # The true check should be: if action chosen at t was move_forward, and agent is oriented to target,
    # then it is "moving_towards" intent-wise.
    # For now, if oriented and not in long freeze, we'll be lenient.
    # TODO: This is a placeholder for a more robust "moving towards" based on actual game action codes
    # and handling of freeze delays in position updates.
    # If you have 'intended_action' data, that would be best.
    # Lacking that, we rely on orientation and not being in a long freeze.
    if np.linalg.norm(np.array(agent_pos_series_t[-1]) - np.array(agent_pos_series_t[0])) > min_dist_change:
      # check if distance to target reduced over the series
      initial_dist_to_target = np.linalg.norm(np.array(agent_pos_series_t[0]) - np.array(target_pos_series_t[0]))
      final_dist_to_target = np.linalg.norm(np.array(agent_pos_series_t[-1]) - np.array(target_pos_series_t[-1]))
      if final_dist_to_target < initial_dist_to_target - min_dist_change:
        return True
    elif freeze_time == 0:  # Not frozen and oriented but didn't move much? Maybe stuck or target moved.
      return True  # Assuming intent if oriented and not frozen.

  return False


def compute_post_fencing_predator_metrics(fencing_events, positions, stamina, rewards, death_labels,
                                          predator_roles, segment_start_time, segment_end_time, predators_in_segment):
  """
  Analyzes predator behavior after a successful fencing event.
  Returns a list of dicts, each describing a post-fencing pursuit.
  """
  # TODO: This function needs access to roles (predator/prey) for each agent index.
  # For now, assuming predator_roles is a dict {agent_idx: "predator" or "prey"}
  # And predators_in_segment is a list of predator agent_indices active in this segment.

  pursuit_metrics = []
  if not fencing_events:
    return pursuit_metrics

  for fence_event in fencing_events:
    pred_idx = fence_event['predator']
    target_prey_idx = fence_event['prey']
    fence_time = fence_event['time']  # This time is relative to episode, adjust if needed for sliced data
    if 'segment_time' in fence_event:
      seg_time = fence_event['segment_time']
    else:
      seg_time = fence_time - segment_start_time

    # 1) segment bounds
    assert segment_start_time <= fence_time < segment_end_time, (
      f"Fencing event time {fence_time} not in segment [{segment_start_time}, {segment_end_time})"
    )

    # 2) predator: must be in segment and alive (1 == alive)
    assert pred_idx in predators_in_segment, f"Predator {pred_idx} not in segment {predators_in_segment}"
    assert death_labels[pred_idx][fence_time] == 1, f"Predator {pred_idx} not alive at fencing time {fence_time}"

    # 3) prey: must have role “prey” and be alive
    assert predator_roles.get(target_prey_idx) == "prey", f"Target {target_prey_idx} is not designated as prey"
    assert death_labels[target_prey_idx][fence_time] == 1, f"Prey {target_prey_idx} not alive at fencing time {fence_time}"

    # Slice data from fence_time to end of segment for this pursuit
    # Times need to be relative to the start of the main episode data
    start_slice_abs = fence_time
    end_slice_abs = segment_end_time

    # Relative indices for sliced data
    start_slice_rel = fence_time - segment_start_time
    end_slice_rel = segment_end_time - segment_start_time

    pred_pos_pursuit = positions[start_slice_abs:end_slice_abs, pred_idx]
    target_prey_pos_pursuit = positions[start_slice_abs:end_slice_abs, target_prey_idx]
    pred_stamina_pursuit = stamina[start_slice_abs:end_slice_abs, pred_idx]

    # For rewards and death_labels, ensure using relative indexing if data is already sliced for segment
    # If data is not pre-sliced for segment, then use absolute indexing.
    # Assuming rewards, death_labels passed to this function are for the full episode.

    pred_rewards_pursuit = rewards[start_slice_abs:end_slice_abs, pred_idx]
    target_prey_death_pursuit = death_labels[target_prey_idx][start_slice_abs:end_slice_abs]
    pred_death_pursuit = death_labels[pred_idx][start_slice_abs:end_slice_abs]

    stamina_at_fence = stamina[fence_time, pred_idx]
    pursuit_duration = 0
    focused = True  # Assume focus starts
    outcome = "unknown"

    for t_pursuit in range(len(pred_pos_pursuit)):
      current_time_abs = start_slice_abs + t_pursuit
      pursuit_duration += 1

      if pred_death_pursuit[t_pursuit] == 0:  # Predator died
        outcome = "predator_died"
        break
      if target_prey_death_pursuit[t_pursuit] == 0:  # Prey died
        # Check if this predator made the catch
        if t_pursuit > 0 and rewards[current_time_abs - 1, pred_idx] == 1 and \
            np.linalg.norm(positions[current_time_abs - 1, pred_idx] - positions[
              current_time_abs - 1, target_prey_idx]) < 2.0:  # crude catch check
          outcome = "catch_by_hunter"
        else:
          outcome = "prey_died_other"
        break

      # Check if focus is maintained
      # TODO: Implement a more robust "is_focused" check (e.g. using is_agent_moving_towards, consistent proximity)
      dist_to_target = np.linalg.norm(pred_pos_pursuit[t_pursuit] - target_prey_pos_pursuit[t_pursuit])
      if dist_to_target > PARAMS["post_fencing_focus_radius"]:
        # Check if disengaged for several steps
        # This needs more state; for now, simple radius check
        # outcome = "disengaged"
        # focused = False; break
        pass  # Keep it simple for now

      if pursuit_duration > PARAMS["post_fencing_disengage_steps"] * 5:  # Timeout
        outcome = "pursuit_timeout"
        break

    stamina_at_end = pred_stamina_pursuit[pursuit_duration - 1]
    stamina_cost = stamina_at_fence - stamina_at_end

    pursuit_metrics.append({
      "fence_time_abs": fence_time,
      "predator_id": pred_idx,
      "target_prey_id": target_prey_idx,
      "duration": pursuit_duration,
      "stamina_cost": stamina_cost,
      "outcome": outcome
    })
  return pursuit_metrics

def compute_collective_apple_events(positions, rewards, death_labels,
                                    prey_indices, safe_grass_locs,
                                    segment_start_time, segment_end_time,
                                    radius=3, min_prey=2,
                                    cooldown=5,
                                    apple_locations=None
                                    ):
  """
  Detects frames where ≥`min_prey`alive prey simultaneously receive an
  *apple* reward **and** form a tight cluster (max pairwise dist ≤2×radius)
  *off* safe grass.

  Because the precise apple‑reward ID can vary across configs, we infer
  it on‑the‑fly: the most common **positive** reward given to prey that
  are *not* tagged in this segment.
  """
  # ------------------------------------------------------------------
  # 1) infer apple reward value for this segment (catch/acorn excluded)
  # ------------------------------------------------------------------
  # segment_rew = rewards[segment_start_time:segment_end_time, :]
  # candidate_vals, counts = np.unique(segment_rew[segment_rew > 0], return_counts=True)
  # if len(candidate_vals) == 0:
  #   return []  # nothing positive collected
  # apple_val = float(candidate_vals[np.argmax(counts)])
  apple_val = 1

  events = []
  t = segment_start_time
  while t < segment_end_time:
    # prey that just got (apple) reward this frame, alive & off grass
    collectors = [q for q in prey_indices
                  if death_labels[q][t] == 1
                  and rewards[t, q] == apple_val
                  and tuple(positions[t, q]) not in safe_grass_locs]

    if len(collectors) >= min_prey:
      # check spatial cohesion
      group_pos = positions[t, collectors, :]
      if np.max(np.linalg.norm(group_pos[:, None, :] - group_pos[None, :, :], axis=2)) <= radius * 2:
        events.append(
          {"time": t,
           "participating_prey": collectors,
           "num_participants": len(collectors)}
        )
        t += cooldown  # skip ahead to avoid double counts
        continue
    t += 1
  return events

# ---------------------------------------------------------------------
#  Collective apple collection detector (spawn‑aware)
# ---------------------------------------------------------------------
def compute_collective_apple_events_wMap(positions, rewards, death_labels, prey_indices,
                                          safe_grass_locs, segment_start_time, segment_end_time,
                                          apple_tiles,              # <- output of get_apple_locations
                                          radius=2, min_prey=2, cooldown=5):
  """
  Flags frames where ≥`min_prey` prey simultaneously harvest apples *from the
  same spawn tile (±radius)*.

  Logic:
    • A prey q is a **collector** at t if
        – q is alive,
        – rewards[t,q] == 1 (apple),
        – distance(pos[t,q], nearest apple tile) ≤ radius,
        – q is not on grass.
    • For each apple tile, count collectors within its radius at t.
    • If any tile gathers ≥ min_prey collectors, record one event then
      advance `t` by `cooldown` to avoid duplicates.
  """
  events = []
  apple_tiles_arr = np.array(list(apple_tiles))  # (N,2) for fast broadcast
  t = segment_start_time
  while t < segment_end_time:
    # candidate collectors this frame
    collectors = []
    for q in prey_indices:
      if death_labels[q][t] == 0:
        continue
      if rewards[t, q] != 1:              # apple reward for prey
        continue
      if tuple(positions[t, q]) in safe_grass_locs:
        continue
      collectors.append(q)

    if collectors:
      # map each collector to nearest apple tile
      pos_q = positions[t, collectors, :]          # (K,2)
      dists = np.linalg.norm(pos_q[:, None, :] - apple_tiles_arr[None, :, :], axis=2)
      nearest_idx = np.argmin(dists, axis=1)
      for tile_idx in np.unique(nearest_idx):
        group = [collectors[i] for i, idx in enumerate(nearest_idx) if idx == tile_idx
                 and dists[i, idx] <= radius]
        if len(group) >= min_prey:
          events.append(
            {"time": t,
             "apple_tile": tuple(apple_tiles_arr[tile_idx]),
             "participating_prey": group,
             "num_participants": len(group)}
          )
          break          # one event per frame is enough
      if events and events[-1]["time"] == t:
        t += cooldown      # skip ahead
        continue
    t += 1
  return events


## Apple collection detector (period_wise)
def find_collective_apple_events(segment_start, segment_end,
                                 positions, rewards, prey_ids,
                                 apple_reward=1,  # value already confirmed
                                 radius=3, min_duration=5, max_gap=2):
  events = []
  t = segment_start
  while t < segment_end:
    # Step 1 ─ seed: any prey that just got an apple at t
    seeders = [q for q in prey_ids if rewards[t, q] == apple_reward]
    if not seeders:
      t += 1;
      continue

    seed_q = seeders[0]
    participants = {seed_q}
    centre = positions[t, seed_q]
    last_bite = t
    duration = 1

    # Step 2 ─ grow the episode forward
    for tau in range(t + 1, segment_end):
      # (a) bail if gap too long
      if tau - last_bite > max_gap:
        break

      # (b) who’s still within radius?
      still_close = {q for q in participants
                     if np.linalg.norm(positions[tau, q] - centre) <= radius}

      # (c) add new joiners that bit an apple this frame and are close
      joiners = [q for q in prey_ids
                 if rewards[tau, q] == apple_reward
                 and np.linalg.norm(positions[tau, q] - centre) <= radius]

      if joiners:
        last_bite = tau
        still_close.update(joiners)

      # (d) update and decide whether we’re still in an event
      if not still_close:
        break
      participants = still_close
      centre = np.median(positions[tau, list(participants), :], axis=0)
      duration += 1

    if duration >= min_duration:
      events.append(dict(time_start=t,
                         time_end=t + duration,
                         participants=list(participants)))
    t += duration  # resume search after the episode
  return events


def compute_movement_synchronization(positions_group, time_window_indices):
  """
  Computes movement synchronization for a group of agents over a time window.
  positions_group: NumSteps x NumAgentsInGroup x 2
  """
  if positions_group.shape[0] < 2 or positions_group.shape[1] < 2:
    return np.nan

  velocities = np.diff(positions_group, axis=0)  # T-1 x NumAgents x 2

  sync_scores = []
  for t in range(velocities.shape[0]):  # Iterate over time steps with velocities
    num_agents_in_group = velocities.shape[1]
    pairwise_cos_sim = []
    for i in range(num_agents_in_group):
      for j in range(i + 1, num_agents_in_group):
        v_i = velocities[t, i]
        v_j = velocities[t, j]
        norm_i = np.linalg.norm(v_i)
        norm_j = np.linalg.norm(v_j)
        if norm_i > 0 and norm_j > 0:
          cos_sim = np.dot(v_i, v_j) / (norm_i * norm_j)
          pairwise_cos_sim.append(cos_sim)
    if pairwise_cos_sim:
      sync_scores.append(np.mean(pairwise_cos_sim))

  return np.nanmean(sync_scores) if sync_scores else np.nan


## The function below will be aborted. Let's use the newer one.
def compute_distraction_events(positions, orientations, rewards, stamina, death_labels,
                               predator_indices, prey_indices, role_map, safe_grass_locs,
                               segment_start_time, segment_end_time,
                               lurking_radius=PARAMS["bodyguard_lurking_radius_A"],
                               reaction_dist=PARAMS["bodyguard_P_reaction_dist_B"],
                               delta_react=PARAMS["bodyguard_delta_t_reaction"],
                               delta_A_acts=PARAMS["bodyguard_delta_t_A_acts"],
                               cost_B_window=PARAMS["bodyguard_cost_B_window"]):
  """
  Flags “body‑guard” episodes:

      1. Predator P lurks ≤`lurking_radius` from preyA on grass.
      2. A different preyB enters P’s FOV within `reaction_dist`.
      3. ≤`delta_react` steps later P is closer to B than to A  ⟹ switch.
      4. Within `delta_A_acts` steps A leaves grass *and* gains any reward.
  """

  fov_kwargs = dict(fov_front=9, fov_back=1, fov_sides=5, max_dist=reaction_dist)
  events = []

  for t in range(segment_start_time, segment_end_time):
    # ------------------------------------------------------------------
    # loop predators
    # ------------------------------------------------------------------
    for P in predator_indices:
      if death_labels[P][t] == 0:  # predator dead
        continue
      P_pos = positions[t, P]
      P_ori = orientations[t, P]

      # ---------- phase1: choose a candidate preyA (on grass) ----------
      grass_prey = [A for A in prey_indices
                    if death_labels[A][t] == 1
                    and tuple(positions[t, A]) in safe_grass_locs
                    and np.linalg.norm(P_pos - positions[t, A]) <= lurking_radius]

      if not grass_prey:
        continue
      # pick nearest
      A = min(grass_prey, key=lambda p: np.linalg.norm(P_pos - positions[t, p]))

      # ---------- phase2: did any preyB enter FOV recently? ------------
      B = None
      for dt in range(1, delta_react + 1):
        tau = t - dt
        if tau < segment_start_time:
          break
        if death_labels[P][tau] == 0:
          break  # predator died earlier

        P_tau_pos = positions[tau, P]
        P_tau_ori = orientations[tau, P]
        # alive, not A, in FOV
        cand_B = [q for q in prey_indices
                  if q != A
                  and death_labels[q][tau] == 1
                  and is_in_predator_fov(P_tau_pos, P_tau_ori,
                                         positions[tau, q], **fov_kwargs)]
        if cand_B:
          # take the first (closest) for simplicity
          B = min(cand_B, key=lambda q: np.linalg.norm(P_tau_pos - positions[tau, q]))
          break  # found a B
      if B is None:
        continue  # no bodyguard

      # ---------- phase3: confirm switch at t ---------------------------
      dist_PA = np.linalg.norm(positions[t, P] - positions[t, A])
      dist_PB = np.linalg.norm(positions[t, P] - positions[t, B])
      if not (dist_PB < dist_PA):  # P still cares more about A
        continue

      # ---------- phase4: benefit to A ---------------------------------
      benefit = False
      for dt in range(delta_A_acts):
        tb = t + dt
        if tb >= segment_end_time or death_labels[A][tb] == 0:
          break
        off_grass_now = tuple(positions[tb, A]) not in safe_grass_locs
        if off_grass_now and rewards[tb, A] > 0:
          benefit = True
          break
      if not benefit:
        continue  # no altruistic outcome

      # ---------- phase5: cost to B ------------------------------------
      stamina_loss = stamina[t, B] - stamina[min(segment_end_time - 1, t + cost_B_window), B]
      caught = any(death_labels[B][tb] == 0
                   for tb in range(t, min(segment_end_time, t + cost_B_window)))

      events.append(dict(
        time_switch=t, predator_id=P,
        prey_A_id=A, prey_B_id=B,
        benefit_A_resource=True,
        cost_B_stamina=stamina_loss,
        cost_B_caught=caught
      ))
  return events


def analyze_altruistic_fencing_aspects(fencing_events, positions, stamina, death_labels,
                                       predator_indices, prey_indices, role_map, safe_grass_locs,
                                       segment_start_time, segment_end_time):
  """
  Analyzes fencing events for altruistic aspects.
  Augments fencing_events with cost/benefit analysis.
  """
  altruistic_fencing_analysis = []
  if not fencing_events: return altruistic_fencing_analysis

  for fence_event in fencing_events:
    pred_id = fence_event['predator']
    fenced_prey_id = fence_event['prey']  # Prey P was trying to tag
    helper_ids = fence_event['helpers']  # Prey that were nearby P (and not Q)
    event_time_abs = fence_event['time']

    if not (segment_start_time <= event_time_abs < segment_end_time): continue
    if death_labels[pred_id][event_time_abs - segment_start_time] == 0: continue
    if death_labels[fenced_prey_id][event_time_abs - segment_start_time] == 0: continue

    # Potential fencers are the helpers + the fenced_prey IF it actively defended
    # For simplicity, let's focus on helpers as altruists.
    # Cost to helpers, benefit to fenced_prey

    total_helper_stamina_cost = 0
    any_helper_caught_soon = False

    for helper_idx in helper_ids:
      if death_labels[helper_idx][event_time_abs - segment_start_time] == 0: continue

      stamina_helper_before = stamina[max(segment_start_time, event_time_abs - 5), helper_idx]  # Stamina shortly before
      stamina_helper_after = stamina[min(segment_end_time - 1, event_time_abs + 5), helper_idx]  # Stamina shortly after
      total_helper_stamina_cost += (stamina_helper_before - stamina_helper_after)

      # Check if helper caught within risk window
      for t_risk_rel in range(PARAMS["altruistic_fencing_risk_window"]):
        t_risk_abs = event_time_abs + t_risk_rel
        if t_risk_abs >= segment_end_time: break
        if death_labels[helper_idx][t_risk_abs - segment_start_time] == 0:
          # TODO: Attribute catch if possible
          any_helper_caught_soon = True
          break

    fenced_prey_escaped_to_grass = False
    fenced_prey_survived_long = True  # Survived risk window

    for t_benefit_rel in range(PARAMS["altruistic_fencing_risk_window"]):
      t_benefit_abs = event_time_abs + t_benefit_rel
      if t_benefit_abs >= segment_end_time: break
      if death_labels[fenced_prey_id][t_benefit_abs - segment_start_time] == 0:
        fenced_prey_survived_long = False
        break
      current_pos_fenced_prey = tuple(positions[t_benefit_abs, fenced_prey_id])
      if current_pos_fenced_prey in safe_grass_locs:
        fenced_prey_escaped_to_grass = True
        break  # Count as escaped

    is_altruistic_candidate = (total_helper_stamina_cost > 0 or any_helper_caught_soon) and \
                              (fenced_prey_escaped_to_grass or fenced_prey_survived_long) and \
                              len(helper_ids) > 0

    analysis = fence_event.copy()  # Start with original fence event data
    analysis.update({
      "helper_stamina_cost_sum": total_helper_stamina_cost,
      "helper_caught_in_risk_window": any_helper_caught_soon,
      "fenced_prey_escaped_to_grass": fenced_prey_escaped_to_grass,
      "fenced_prey_survived_risk_window": fenced_prey_survived_long,
      "is_altruistic_candidate": is_altruistic_candidate
    })
    altruistic_fencing_analysis.append(analysis)

  return altruistic_fencing_analysis


## Add more helping functions
def _pairwise_cosine_sync(vel: np.ndarray) -> float:
  """
  vel : (T, N, 2)  ‑‑ per‑frame velocity vectors for N agents.
  Returns the mean pairwise cosine similarity averaged over time.
  Frames where **all** agents are stationary are ignored.
  """
  if vel.shape[0] < 1 or vel.shape[1] < 2:
    return np.nan

  # normalise, avoid /0
  norms = np.linalg.norm(vel, axis=2, keepdims=True) + 1e-9
  vhat = vel / norms

  sims_per_t = []
  for t in range(vhat.shape[0]):
    if np.all(norms[t] < 1e-6):
      continue  # skip all‑stationary frame
    sims = []
    for i in range(vhat.shape[1]):
      for j in range(i + 1, vhat.shape[1]):
        sims.append(np.dot(vhat[t, i], vhat[t, j]))
    if sims:
      sims_per_t.append(np.mean(sims))
  return np.nanmean(sims_per_t) if sims_per_t else np.nan

def movement_sync_for_events(events, positions):
  """
  Adds a `sync_score` field to each event dict in *events* (in‑place).
  events[i] must contain:
      'time_start', 'time_end', 'participants'
  """
  for ev in events:
    t0, t1 = ev["time_start"], ev["time_end"]
    ids = ev["participants"]
    # slice positions: (ΔT, |ids|, 2)
    pos_seg = positions[t0:t1, ids, :]
    vel_seg = np.diff(pos_seg, axis=0, prepend=pos_seg[:1])
    ev["sync_score"] = _pairwise_cosine_sync(vel_seg)

  return events

def baseline_movement_sync(segment_start, segment_end,
                           positions, death_labels, agent_ids):
  """
  Computes synchrony for *agent_ids* over all frames where **every** agent
  is alive (1).  Useful as a null‑model reference.
  """
  # frames where every agent in the set is alive
  alive_mask = np.ones(segment_end - segment_start, dtype=bool)
  for q in agent_ids:
    alive_mask &= death_labels[q][segment_start:segment_end] == 1
  if not np.any(alive_mask):
    return np.nan

  pos_live = positions[segment_start:segment_end][alive_mask][:, agent_ids, :]
  vel_live = np.diff(pos_live, axis=0, prepend=pos_live[:1])
  return _pairwise_cosine_sync(vel_live)

def _is_moving_away(prev_pred_pos, prev_prey_pos,
                    curr_pred_pos, curr_prey_pos, thresh=0.5):
  """True iff predator stayed ≤3 and prey increased distance by > thresh."""
  d_prev = np.linalg.norm(prev_pred_pos - prev_prey_pos)
  d_curr = np.linalg.norm(curr_pred_pos - curr_prey_pos)
  return d_prev <= 3.0 and (d_curr - d_prev) > thresh


def _find_distraction_t_end(positions, orientation, death_labels, safe_grass,
                            predator, prey_B, start_t, max_t,
                            window_away=5, hard_cap=30):
  """
  Returns (t_end, end_reason) where end_reason ∈ {
    'helper_on_grass', 'helper_dead', 'dist_ge_4', 'pred_move_away', 'hard_cap', 'predator_dead'
  }
  """

  def _on_grass(q, t):
    return tuple(positions[t, q]) in safe_grass

  dist_hist = []
  for offset in range(hard_cap + 1):
    tau = start_t + offset
    if tau > max_t:
      break

    # 1) B reaches grass
    if _on_grass(prey_B, tau):
      return tau, 'helper_on_grass'
    # 2) B is eaten
    if death_labels[prey_B][tau] == 0:
      return tau, 'helper_dead'
    # predator dead?
    if death_labels[predator][tau] == 0:
      return tau, 'predator_dead'

    # 3) distance ≥ 4
    d = np.linalg.norm(positions[tau, predator] - positions[tau, prey_B])
    if d >= 4.0:
      return tau, 'dist_ge_4'

    # 4) predator moved away 3/5
    dist_hist.append(d)
    if len(dist_hist) == window_away:
      if np.sum(np.diff(dist_hist) > 0) >= 3:
        return tau, 'pred_move_away'
      dist_hist.pop(0)

  # 5) hard cap
  t_cap = min(max_t, start_t + hard_cap)
  return t_cap, 'hard_cap'


def detect_distraction_grass(positions, orientations, death_labels,
                             predator_ids, prey_ids, safe_grass,
                             start_t, end_t,
                             min_window=5, shift_window=5):
  """
  Prey A sits on grass. Predator P lurks ≤3 for ≥5 ts, then switches to B.
  """
  events = []
  t = start_t
  while t <= end_t - min_window:
    for P in predator_ids:
      if death_labels[P][t] == 0:  # predator dead
        continue
      for A in prey_ids:
        if tuple(positions[t, A]) not in safe_grass:  # A must start on grass
          continue
        # check 5‑frame lurking window
        window_ok = True
        for dt in range(min_window):
          tau = t + dt
          if tau >= end_t or death_labels[A][tau] == 0 or death_labels[P][tau] == 0:
            window_ok = False;
            break
          if np.linalg.norm(positions[tau, P] - positions[tau, A]) > 3.0:
            window_ok = False;
            break
        if not window_ok:
          continue

        # ---------------- shift phase ----------------
        shift_found, B_id, shift_t = False, None, None
        for dt in range(1, shift_window + 1):
          tau = t + min_window - 1 + dt
          if tau >= end_t: break
          # candidate B: alive prey within 3 of P & not A
          candidates = [q for q in prey_ids if q != A and
                        death_labels[q][tau] == 1 and
                        np.linalg.norm(positions[tau, P] - positions[tau, q]) <= 3.0]
          if not candidates:
            continue
          # choose closest B
          B = min(candidates, key=lambda q: np.linalg.norm(positions[tau, P] - positions[tau, q]))
          if np.linalg.norm(positions[tau, P] - positions[tau, B]) < \
              np.linalg.norm(positions[tau, P] - positions[tau, A]):
            shift_found, B_id, shift_t = True, B, tau
            break
        if not shift_found:
          continue

        # ---------------- termination ----------------
        t_end, end_note = _find_distraction_t_end(positions, orientations, death_labels, safe_grass,
          predator=P, prey_B=B_id, start_t=shift_t, max_t=end_t)

        events.append(dict(
          scenario="grass",
          time_start=t,
          time_shift=shift_t,
          time_end=t_end,
          predator=P, prey_A=A, prey_B=B_id,
          end_reason=end_note
        ))
        t = t_end  # skip past event
        break  # break A loop
      else:
        continue  # continue predator loop
      break  # break predator loop
    else:
      t += 1  # no event at this t
  return events

def detect_distraction_chase(positions, orientations, actions, death_labels,
                             predator_ids, prey_ids, safe_grass,
                             start_t, end_t,
                             min_window=5, shift_window=5):
  """
  Predator P chases prey A (off grass) for ≥5 ts.  Prey moves away ≥3/5 frames.
  Then P switches to B.
  """
  events = []
  t = start_t
  while t <= end_t - min_window:
    for P in predator_ids:
      if death_labels[P][t] == 0: continue
      for A in prey_ids:
        if tuple(positions[t, A]) in safe_grass: continue  # A must start off grass
        # -- 5‑frame chase check --
        away_count = 0
        valid = True
        for dt in range(min_window):
          tau = t + dt
          if tau >= end_t or death_labels[A][tau] == 0 or death_labels[P][tau] == 0:
            valid = False;
            break
          if np.linalg.norm(positions[tau, P] - positions[tau, A]) > 3.0:
            valid = False;
            break
          if dt > 0 and _is_moving_away(
              positions[tau - 1, P], positions[tau - 1, A],
              positions[tau, P], positions[tau, A]):
            away_count += 1
          if not valid or away_count < 3:
            continue

        # -- shift to B  --
        shift_found, B_id, shift_t = False, None, None
        for dt in range(1, shift_window + 1):
          tau = t + min_window - 1 + dt
          if tau >= end_t: break
          candidates = [q for q in prey_ids if q != A and
                        death_labels[q][tau] == 1 and
                        np.linalg.norm(positions[tau, P] - positions[tau, q]) <= 3.0]
          if not candidates: continue
          B = min(candidates, key=lambda q: np.linalg.norm(positions[tau, P] - positions[tau, q]))
          if np.linalg.norm(positions[tau, P] - positions[tau, B]) < \
              np.linalg.norm(positions[tau, P] - positions[tau, A]):
            shift_found, B_id, shift_t = True, B, tau
            break
        if not shift_found: continue

        # -- termination --
        t_end, end_note = _find_distraction_t_end(
          positions, orientations, death_labels, safe_grass,
          predator=P, prey_B=B_id,
          start_t=shift_t, max_t=end_t)

        events.append(dict(
          scenario="chase",
          time_start=t,
          time_shift=shift_t,
          time_end=t_end,
          predator=P, prey_A=A, prey_B=B_id,
        end_reason=end_note))
        t = t_end
        break
      else:
        continue
      break
    else:
      t += 1
  return events


# --- Modified Core Processing ---
def process_pair_folder(folder_path):
  base = os.path.basename(folder_path).split('predator_prey__')[0]
  role_map, src_map = parse_agent_roles(base)  # role_map[idx] = "predator" or "prey"
  predator_ids = [i for i, r in role_map.items() if r == 'predator']
  prey_ids = [i for i, r in role_map.items() if r == 'prey']

  pkl_dir = os.path.join(folder_path, 'episode_pickles')
  if not os.path.exists(pkl_dir):
    print(f"Pickle directory not found: {pkl_dir}")
    return None

  files = sorted(f for f in os.listdir(pkl_dir) if f.endswith('.pkl'))
  if not files:
    print(f"No pickle files found in: {pkl_dir}")
    return None

  all_episode_segment_dfs = []  # Store rows for each active segment of each episode

  # Load safe grass locations once per folder
  safe_grass = set(get_safe_grass_locations(folder_path))

  for fname in files:
    try:
      with open(os.path.join(pkl_dir, fname), 'rb') as fp:
        episode_data_list = pickle.load(fp)  # Assuming this is a list of dicts as in original
    except Exception as e:
      print(f"Error loading or processing {fname}: {e}")
      continue

    if not isinstance(episode_data_list, list) or not episode_data_list:
      print(f"Empty or invalid data in {fname}")
      continue

    # Build arrays for the full episode first
    try:
      pos_ep = np.array([d['POSITION'] for d in episode_data_list])
      ori_ep = np.array([d['ORIENTATION'] for d in episode_data_list])
      act_ep = np.array([d['actions'] for d in episode_data_list])
      rew_ep = np.array([d['rewards'] for d in episode_data_list])
      sta_ep = np.array([d['STAMINA'] for d in episode_data_list])
    except KeyError as e:
      print(f"Missing key {e} in episode data for {fname}. Skipping episode.")
      continue
    except ValueError as e:  # Handles issues with array creation if shapes mismatch etc
      print(f"ValueError creating arrays for {fname}: {e}. Skipping episode.")
      continue

    num_agents = pos_ep.shape[1]
    death_labels_ep = {i: mark_death_periods(sta_ep[:, i]) for i in range(num_agents)}

    death_labels_all_prey_ep = {pi: death_labels_ep[pi] for pi in prey_ids if pi in death_labels_ep}

    active_segments = segment_episode_by_activity(
      death_labels_all_prey_ep,
      PARAMS["episode_activity_min_prey_alive"],
      PARAMS["episode_activity_min_dead_duration"]
    )
    if not active_segments:  # No prey activity in this episode
      # TODO: decide if to log episodes with no prey activity, or skip.
      # For now, skipping segments if none are active.
      # You might still want to calculate some full-episode metrics here.
      # print(f"No active prey segments in {fname}")
      pass  # continue to next file if no active segments

    for seg_idx, (seg_start, seg_end) in enumerate(active_segments):
      if seg_end <= seg_start: continue  # Skip empty segments

      # Slice data for the current segment
      pos = pos_ep[seg_start:seg_end]
      ori = ori_ep[seg_start:seg_end]
      act = act_ep[seg_start:seg_end]
      rew = rew_ep[seg_start:seg_end]
      sta = sta_ep[seg_start:seg_end]

      # Create death_labels for the segment for convenience, or pass full and use seg_start/end
      # For functions expecting full episode death_labels, pass death_labels_ep and seg_start/end
      # For functions that might operate on sliced data, slice death_labels too:
      death_labels_segment = {i: death_labels_ep[i][seg_start:seg_end] for i in range(num_agents)}

      # --- Standard Metrics (from original script, applied to segment) ---
      # TODO: Review if all original metrics make sense at segment level or should be full episode
      move_dist = compute_agent_move_distance(pos)
      rotation = compute_num_rotation(ori)
      # Add the above two as the total state changes
      total_executions = {i: move_dist[i] + rotation[i] for i in range(num_agents)}

      # For collect counts, need to specify which agents are pred/prey in this segment
      # Pass predator_ids, prey_ids lists.
      apple_counts, acorn_counts, catch_counts = compute_collect_counts(rew, predator_ids, prey_ids)

      # Death counts within segment (might be less meaningful than full episode)
      # For full episode deaths, calculate outside segment loop
      # death_cnt_segment = compute_death_counts(sta) # TODO: Review this metric for segments

      # Stuck rate within segment
      stuck_rate = compute_stuck_rate_all(pos, death_labels=death_labels_segment)  # Pass sliced death_labels

      # Grass time within segment
      on_t, off_t, fo = compute_grass_time_per_life(pos, death_labels_segment, prey_ids, safe_grass)

      # Pairwise distance within segment
      pair_dist = compute_pairwise_distance_mean(pos, death_labels_segment)

      # Good gathering within segment
      good_gathering = compute_good_gathering(pos, predator_ids, prey_ids, death_labels=death_labels_segment)

      # Fencing events within segment
      # These functions take full episode data and then internally slice or should be adapted
      # For compute_successful_fencing_and_helpers, it might be easier to run on full episode data
      # and then filter events that fall within the segment. Or adapt it to take sliced data.
      # Let's assume we run on full episode and filter here, for now.
      # Or, better, adapt them if they don't rely on history outside the segment.
      # For simplicity, let's assume they can take sliced data for now.
      fencing_events_segment = compute_successful_fencing_and_helpers(pos, ori, act, rew,
                                                                      predator_ids, prey_ids, death_labels_segment,
                                                                      seg_start=seg_start)

      # We pause here if we detect a successful fencing event
      if len(fencing_events_segment) > 0 and fencing_events_segment[-1]['helpers']:
        print(f"Fencing events detected in segment {seg_idx} of {fname}: {fencing_events_segment}")
      # invalid_interactions_segment = compute_invalid_interactions(act, rew, pos, ori, sta,
      #                                                 predator_ids, prey_ids) # Pass sliced data
      # TODO: compute_invalid_interactions needs careful check if it can use sliced `sta` or needs full `sta_ep`

      # --- New Coalition & Altruism Metrics for the Segment ---
      # Predator Post-Fencing
      # Need predator_roles for the full agent set
      post_fencing_pursuits = compute_post_fencing_predator_metrics(
        fencing_events_segment, pos_ep, sta_ep, rew_ep, death_labels_ep,  # Pass full episode data here for context
        role_map, seg_start, seg_end, predator_ids  # But specify segment bounds
      )
      # Aggregate post_fencing_pursuits for summary
      num_post_fence_pursuits = len(post_fencing_pursuits)
      avg_post_fence_stamina_cost = np.nanmean(
        [p['stamina_cost'] for p in post_fencing_pursuits]) if post_fencing_pursuits else 0
      post_fence_catches = sum(1 for p in post_fencing_pursuits if p['outcome'] == 'catch_by_hunter')

      # Prey Collective Apple Collection
      # TODO: Need apple_locations for compute_collective_apple_events.
      # This is a major dependency. For now, it will return empty or rely on dummy data.
      collective_apple_events = compute_collective_apple_events(
        pos_ep, rew_ep, death_labels_ep, prey_ids, safe_grass, seg_start, seg_end, apple_locations=None
      )
      apple_tiles = get_apple_locations(folder_path)  # Assuming this is a function that returns apple tile locations
      collective_apple_events_wMap = compute_collective_apple_events_wMap(
        pos_ep, rew_ep, death_labels_ep, prey_ids, safe_grass, seg_start, seg_end,
        apple_tiles=apple_tiles
      )
      collective_apple_events_period = find_collective_apple_events(
        seg_start, seg_end, pos_ep, rew_ep, prey_ids,
        apple_reward=1,  # Assuming this is the apple reward value
        radius=3, min_duration=5, max_gap=2
      )

      num_collective_apple_events = len(collective_apple_events)
      num_collective_apple_events_wMap = len(collective_apple_events_wMap)
      num_collective_apple_events_period = len(collective_apple_events_period)

      # For debug
      if num_collective_apple_events > 0:
        print(f"Collective apple events detected in segment {seg_idx} of {fname}: {collective_apple_events}")
      if num_collective_apple_events_wMap > 0:
        print(f"Collective apple events with map detected in segment {seg_idx} of {fname}: {collective_apple_events_wMap}")
      if num_collective_apple_events_period > 0:
        print(f"Collective apple events (period-wise) detected in segment {seg_idx} of {fname}: {collective_apple_events_period}")

      '''
      # Movement Synchronization (Example: for successful collective apple events)
      # TODO: Define when to calculate this. Example for prey during collective apple collection.
      avg_sync_collective_apple = np.nan
      if collective_apple_events:
        sync_scores_list = []
        for ca_event in collective_apple_events:
          event_time = ca_event['time']  # This is absolute time
          event_duration = PARAMS["collective_apple_time_window"]  # Example duration
          if event_time >= seg_start and (event_time + event_duration) <= seg_end:
            group_prey_ids = ca_event['participating_prey']
            if len(group_prey_ids) >= 2:
              # Slice positions for these prey during the event window
              # Ensure indices are valid for pos (which is already sliced for segment)
              start_rel = event_time - seg_start
              end_rel = start_rel + event_duration
              if end_rel <= pos.shape[0]:  # Check bounds for sliced pos
                pos_group_event = pos[start_rel:end_rel, group_prey_ids, :]
                sync_score = compute_movement_synchronization(pos_group_event,
                                                              None)  # time_window_indices not used currently
                if not np.isnan(sync_score):
                  sync_scores_list.append(sync_score)
        if sync_scores_list: avg_sync_collective_apple = np.nanmean(sync_scores_list)
      '''
      baseline_sync_all_prey = baseline_movement_sync(
        seg_start, seg_end, pos_ep, death_labels_ep, prey_ids)

      collective_apple_events_period = movement_sync_for_events(collective_apple_events_period, pos_ep)

      avg_sync_collective_apple = np.nanmean([e['sync_score'] for e in collective_apple_events_period])

      # Prey Bodyguard/Distraction
      distraction_events = compute_distraction_events(
        pos_ep, ori_ep, rew_ep, sta_ep, death_labels_ep,
        predator_ids, prey_ids, role_map, safe_grass, seg_start, seg_end
      )
      num_distraction_events = len(distraction_events)
      num_distraction_A_benefit = sum(1 for d in distraction_events if d['benefit_A_resource'])
      num_distraction_B_caught = sum(1 for d in distraction_events if d['cost_B_caught'])

      if distraction_events:
        print(f"Distraction events detected in segment {seg_idx} of {fname}: {distraction_events}")

      # Another two distractions
      grass_events = detect_distraction_grass(pos_ep, ori_ep, death_labels_ep,
                                              predator_ids, prey_ids, safe_grass,
                                              seg_start, seg_end)

      chase_events = detect_distraction_chase(pos_ep, ori_ep, act_ep, death_labels_ep,
                                              predator_ids, prey_ids, safe_grass,
                                              seg_start, seg_end)

      all_distractions = grass_events + chase_events

      num_distraction_events = len(all_distractions)
      if all_distractions:
        print(f"Distraction events detected in segment {seg_idx} of {fname}: {all_distractions}")

      # Altruistic Fencing Analysis
      altruistic_fencing_details = analyze_altruistic_fencing_aspects(
        fencing_events_segment, pos_ep, sta_ep, death_labels_ep,
        predator_ids, prey_ids, role_map, safe_grass, seg_start, seg_end
      )
      num_altruistic_fencing_candidates = sum(1 for afd in altruistic_fencing_details if afd['is_altruistic_candidate'])

      # --- Assemble one DataFrame row for this segment ---
      row = {
        'trial_name': base,
        'episode': fname.replace('.pkl', ''),
        'segment_idx': seg_idx,
        'segment_start_time': seg_start,
        'segment_end_time': seg_end,
        'role_map': role_map,  # For reference in pkl
        'source_map': src_map,  # For reference in pkl

        # Original metrics (can be prefixed with seg_ if also calculated for full episode)
        # **{f"move_{i}": move_dist[i] for i in move_dist if i in role_map},
        # **{f"rot_{i}": rotation[i] for i in rotation if i in role_map},
        # **{f"apple_{i}": apple_counts.get(i, 0) for i in prey_ids},  # .get for safety
        # **{f"acorn_{i}": acorn_counts.get(i, 0) for i in prey_ids},
        # **{f"catch_{i}": catch_counts.get(i, 0) for i in predator_ids},
        # **{f"stuck_{i}": stuck_rate[i] for i in stuck_rate if i in role_map},
        # **{f"on_grass_{i}": np.sum(on_t[i]) if i in on_t and on_t[i] is not None else np.nan for i in prey_ids},
        # # Sum list of times
        # **{f"off_grass_{i}": np.sum(off_t[i]) if i in off_t and off_t[i] is not None else np.nan for i in prey_ids},
        # **{f"frac_off_{i}": np.nanmean(fo[i]) if i in fo and fo[i] is not None and len(fo[i]) > 0 else np.nan for i in
        #    prey_ids},  # Avg list of fracs
        # **pair_dist,
        # **good_gathering,

        # New Coalition Metrics
        "num_fencing_events_segment": len(fencing_events_segment),
        "num_post_fence_pursuits": num_post_fence_pursuits,
        "avg_post_fence_stamina_cost": avg_post_fence_stamina_cost,
        "post_fence_catches": post_fence_catches,
        "num_collective_apple_events": num_collective_apple_events,
        "avg_sync_collective_apple": avg_sync_collective_apple,

        # New Altruism Metrics
        "num_distraction_events": num_distraction_events,
        "num_distraction_A_benefit": num_distraction_A_benefit,
        "num_distraction_B_caught": num_distraction_B_caught,
        "num_altruistic_fencing_candidates": num_altruistic_fencing_candidates,

        # Full event lists for pkl (optional, can make files large)
        # 'fencing_events_list': fencing_events_segment,
        # 'post_fencing_pursuits_list': post_fencing_pursuits,
        # 'distraction_events_list': distraction_events,
        # 'altruistic_fencing_details_list': altruistic_fencing_details
      }
      all_episode_segment_dfs.append(pd.DataFrame([row]))

  if not all_episode_segment_dfs:
    print(f"No data processed for folder {folder_path}")
    return None

  return pd.concat(all_episode_segment_dfs, ignore_index=True)


def process_and_save(folder_path, out_dir):
  """
  Wrapper to process one folder and save its metrics CSV & PKL.
  """
  print(f"Processing folder: {folder_path}")
  df = process_pair_folder(folder_path)
  if df is None or df.empty:
    print(f"No dataframe generated for {folder_path}")
    return

  base_name = os.path.basename(folder_path)  # Keep full folder name for clarity

  # Ensure trial_name is consistent if base was split before
  # trial_name_from_df = df['trial_name'].iloc[0] # Get from df if generated

  out_path_csv = os.path.join(out_dir, f"{base_name}_extended_metrics.csv")
  out_path_pkl = os.path.join(out_dir, f"{base_name}_extended_metrics.pkl")

  try:
    df.to_csv(out_path_csv, index=False)
    df.to_pickle(out_path_pkl)
    print(f"Saved: {os.path.basename(out_path_csv)} and {os.path.basename(out_path_pkl)}")
  except Exception as e:
    print(f"Error saving files for {base_name}: {e}")


def main():
  base_dir = "/results/mix_2_4/"
  # base_dir = "path/to/your/experiment_folders_parent_directory/"
  out_dir = os.path.join(base_dir, "analysis_results_extended")
  os.makedirs(out_dir, exist_ok=True)

  folders = []
  for d in os.listdir(base_dir):
    path = os.path.join(base_dir, d)
    if os.path.isdir(path) and 'episode_pickles' in os.listdir(path):
      # Add any other folder filters if needed
      # e.g. 'simplified10x10' not in d
      folders.append(path)

  print(f"Found {len(folders)} folders to process.")
  folders = sorted(folders)
  # Process in parallel or serially
  # Parallel processing (adjust n_jobs as needed)
  # n_jobs=-1 means use all available CPUs. Can be memory intensive. Start with 1 or a small number.
  # if folders:
  #   Parallel(n_jobs=4)(  # TODO: Adjust n_jobs based on your system
  #     delayed(process_and_save)(folder, out_dir)
  #     for folder in folders
  #   )
  # else:
  #   print("No folders found to process.")
  # Serial processing for debugging:
  for folder in folders:
    process_and_save(folder, out_dir)


if __name__ == '__main__':
  main()
