"""social_metrics_archive.py
Helper functions for coalition and altruism analysis in predator–prey
experiments.  These are **independent** of the environment‑specific
`analyze_mixed_results_flexible.py` script so that you can test /
iterate quickly and import them where needed.

All functions assume NumPy arrays with the same shapes you already use:
  positions:  (T, A, 2)
  orientations: (T, A)
  actions:     (T, A)
  rewards:     (T, A)
  stamina:     (T, A)

The new behavioural events are returned as *lists of dicts* so you can
either keep them raw (for case‑by‑case inspection) or aggregate counts
/ summary stats per episode.
"""
import numpy as np
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------
#  Episode segmentation
# ---------------------------------------------------------------------

def segment_active_phases(stamina: np.ndarray,
                          prey_idx: List[int],
                          min_quiet: int = 50) -> List[Tuple[int, int]]:
  """Return [(start, end)] intervals where *at least one* prey is alive.

  An *active phase* starts the first frame any prey becomes alive after
  a quiet period where **all** prey were dead for `min_quiet` frames.
  It ends as soon as all prey are dead again or the episode ends.
  """
  T = stamina.shape[0]
  alive_counts = np.sum(stamina[:, prey_idx] > 0, axis=1)

  segments = []
  in_seg = False
  start = 0
  for t in range(T):
    if not in_seg and alive_counts[t] > 0:
      in_seg = True
      start = t
    elif in_seg and alive_counts[t] == 0:
      segments.append((start, t))
      in_seg = False
  if in_seg:
    segments.append((start, T))
  return segments


# ---------------------------------------------------------------------
#  Geometry helpers
# ---------------------------------------------------------------------
_FOV_DEFAULT = {'front': 9, 'back': 1, 'side': 5}

def _rotate(dx: int, dy: int, ori: int) -> Tuple[int, int]:
  """Rotate vector (dx,dy) so that ori==0 faces +y (classic Meltingpot)."""
  if ori % 4 == 0:
    return dx, dy
  if ori % 4 == 1:
    return dy, -dx
  if ori % 4 == 2:
    return -dx, -dy
  return -dy, dx


def is_in_predator_fov(pred_pos: np.ndarray,
                       pred_ori: int,
                       prey_pos: np.ndarray,
                       fov: Dict[str, int] = _FOV_DEFAULT) -> bool:
  """Return True if *prey_pos* lies inside predator's FOV rectangle."""
  dx, dy = prey_pos - pred_pos
  rx, ry = _rotate(dx, dy, pred_ori)
  if ry >= 0:
    return (abs(rx) <= fov['side']) and (0 < ry <= fov['front'])
  return (abs(rx) <= fov['side']) and (-fov['back'] <= ry < 0)


def _pairwise_dir_similarity(vecs: np.ndarray) -> float:
  """Return mean pairwise cosine similarity of a (N,2) array."""
  if len(vecs) < 2:
    return 1.0
  v = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
  sims = []
  for i in range(len(v)):
    for j in range(i+1, len(v)):
      sims.append(float(np.dot(v[i], v[j])))
  return float(np.mean(sims))

# ---------------------------------------------------------------------
#  Coalition metrics
# ---------------------------------------------------------------------

def compute_predator_coalition_hunts(positions: np.ndarray,
                                     orientations: np.ndarray,
                                     actions: np.ndarray,
                                     rewards: np.ndarray,
                                     predators: List[int],
                                     preys: List[int],
                                     hunt_radius: int = 5,
                                     delta_t: int = 10) -> List[Dict]:
  """Detect "joint hunts" where >1 predator go after the same prey.

  Returns list of dicts: time, prey, hunter, partners, partner_sync.
  """
  T = positions.shape[0]
  events = []
  vel = np.diff(positions, axis=0, prepend=positions[:1])
  for t in range(T):
    catchers = [p for p in predators if rewards[t, p] == 1]
    if not catchers:
      continue
    died = [q for q in preys if (t+1 < T) and (rewards[t+1, q] < rewards[t, q])]
    for hunter in catchers:
      for q in died:
        partners = []
        for p2 in predators:
          if p2 == hunter:
            continue
          if np.linalg.norm(positions[t, p2] - positions[t, q]) > hunt_radius:
            continue
          t0 = max(0, t - delta_t)
          if np.linalg.norm(positions[t0, p2] - positions[t0, q]) <= np.linalg.norm(positions[t, p2] - positions[t, q]):
            continue
        partners.append(p2)
      if partners:
        vecs = np.vstack([vel[t-1, hunter], *[vel[t-1, pp] for pp in partners]])
        score = _pairwise_dir_similarity(vecs)
        events.append({'time': t, 'prey': q,
                       'hunter': hunter,
                       'partners': partners,
                       'partner_sync': score})
  return events


def compute_collective_apple_collection(positions: np.ndarray,
                                        rewards: np.ndarray,
                                        preys: List[int],
                                        apple_locs: List[Tuple[int, int]],
                                        radius: int = 2,
                                        delta_t: int = 6,
                                        min_group: int = 2) -> List[Dict]:
  """Detect when >=min_group prey converge on the same apple patch."""
  T = positions.shape[0]
  events = []
  patch_map = {loc: idx for idx, loc in enumerate(apple_locs)}
  active = {idx: None for idx in patch_map.values()}
  for t in range(T):
    for loc, pid in patch_map.items():
      near = [q for q in preys if np.sum(abs(positions[t,q] - loc)) <= radius]
      if len(near) >= min_group and active[pid] is None:
        active[pid] = {'start': t, 'agents': set(near)}
      if active[pid] is not None and len(near) < min_group:
        ev = active[pid]
        ev['end'] = t
        ev['duration'] = ev['end'] - ev['start']
        ev['agents'] = list(ev['agents'])
        ev['apples'] = sum(1 for q in preys
                           if 1 in rewards[ev['start']:ev['end'], q])
        events.append(ev)
        active[pid] = None
      if active.get(pid) is not None:
        active[pid]['agents'].update(near)
  return events

# ---------------------------------------------------------------------
#  Altruism metrics
# ---------------------------------------------------------------------

def compute_bodyguard_events(positions: np.ndarray,
                             orientations: np.ndarray,
                             stamina: np.ndarray,
                             predators: List[int],
                             preys: List[int],
                             safe_grass: set,
                             fov: Dict[str,int] = _FOV_DEFAULT,
                             lurking_radius: int = 4,
                             reaction_dist: int = 6,
                             delta_react: int = 6) -> List[Dict]:
  """Detect distraction events: prey B draws predator P from A. """
  T = positions.shape[0]
  events = []
  prev_observed = { (P,B): False for P in predators for B in preys }
  for t in range(T):
    for P in predators:
      for A in preys:
        # A on safe grass
        if tuple(positions[t,A]) not in safe_grass:
          continue
        # P near A
        if np.linalg.norm(positions[t,P]-positions[t,A]) > lurking_radius:
          continue
        # P facing A
        if not is_in_predator_fov(positions[t,P], orientations[t,P], positions[t,A], fov):
          continue
        # look for any B != A entering FOV
        for B in preys:
          if B == A:
            continue
          in_fov = (np.linalg.norm(positions[t,B]-positions[t,P]) <= reaction_dist
                    and is_in_predator_fov(positions[t,P], orientations[t,P], positions[t,B], fov))
          if in_fov and not prev_observed[(P,B)]:
            # predator should switch
            # check within delta_react frames
            end = min(T, t+delta_react)
            switched = False
            for tau in range(t, end):
              if not is_in_predator_fov(positions[tau,P], orientations[tau,P], positions[tau,A], fov):
                switched = True
                break
            if switched:
              # record cost/benefit
              cost = stamina[t,A] - stamina[end-1,A]
              caught_B = any(np.linalg.norm(positions[tau,P]-positions[tau,B])<1
                             for tau in range(t, end))
              events.append({'time_switch': t,
                             'predator': P,
                             'prey_A': A,
                             'bodyguard_B': B,
                             'benefit_to_A': True,
                             'stamina_cost_B': cost,
                             'caught_B': caught_B})
          prev_observed[(P,B)] = in_fov
  return events

def compute_collective_apple_events(positions, rewards, death_labels, prey_indices,
                                    safe_grass_locs,  # Needed to ensure prey are off-grass for apples
                                    segment_start_time, segment_end_time,
                                    apple_locations=None):  # TODO: Need apple locations
  """
  Identifies collective apple collection by prey.
  apple_locations: list of (x,y) tuples for current apple spawns. This might change per step.
  For simplicity, this example might need a fixed set or needs dynamic apple_locs per step.
  """
  # TODO: This function is highly dependent on how apple_locations are provided.
  # If apple_locations change per step, this needs to be passed in as T x NumApples x 2 or similar.
  # Assuming for now `apple_locations` is a static list for the episode for simplicity.
  # A robust implementation would require knowing which rewards are apples (e.g. rewards value > 1 and not catch)
  # and the locations of those apples when they were collected.

  # This is a placeholder, as apple collection details are not fully specified.
  # Key idea: find N_prey_collective simultaneously near the same apple and getting apple rewards.
  # Requires knowing reward type (apple vs acorn vs catch) and apple locations at time of collection.

  # Simplified: Count instances where multiple prey get non-catch rewards simultaneously
  # while off grass and near each other.

  collective_events = []
  # Example: if rewards[:, prey_idx] == 2 indicates apple for prey_idx
  # This needs to be defined based on your reward structure.
  APPLE_REWARD_VALUE = 2  # Placeholder

  # Iterate through time steps in the segment
  for t_rel in range(segment_end_time - segment_start_time - PARAMS["collective_apple_time_window"]):
    t_abs = segment_start_time + t_rel

    active_prey_at_t_window = []
    for prey_idx in prey_indices:
      if death_labels[prey_idx][t_abs:t_abs + PARAMS["collective_apple_time_window"]].all() and \
          not any(tuple(pos) in safe_grass_locs for pos in
                  positions[t_abs:t_abs + PARAMS["collective_apple_time_window"], prey_idx]):
        # Check for apple reward in window
        if np.any(rewards[t_abs:t_abs + PARAMS["collective_apple_time_window"], prey_idx] == APPLE_REWARD_VALUE):
          active_prey_at_t_window.append(prey_idx)

    if len(active_prey_at_t_window) >= PARAMS["collective_apple_min_prey"]:
      # Check if these prey are close to each other (potential same apple patch)
      group_positions = positions[t_abs + PARAMS["collective_apple_time_window"] // 2, active_prey_at_t_window, :]
      pairwise_distances = np.linalg.norm(group_positions[:, np.newaxis, :] - group_positions[np.newaxis, :, :], axis=2)

      # Check if they form a close group (e.g., all within a certain max distance of each other, or connected component)
      # Simplified: check if max distance in the group is below a threshold
      if np.max(pairwise_distances) < PARAMS["collective_apple_radius"] * 2:  # Crude check
        event = {
          "time": t_abs,
          "participating_prey": active_prey_at_t_window,
          "num_participants": len(active_prey_at_t_window),
          # TODO: Add more details like specific apples collected if possible
        }
        collective_events.append(event)
        # To avoid double counting, you might want to advance t_rel past this window
        # t_rel += PARAMS["collective_apple_time_window"]
  return collective_events


def compute_distraction_events(positions, orientations, rewards, stamina, death_labels,
                               predator_indices, prey_indices, role_map, safe_grass_locs,
                               segment_start_time, segment_end_time):
  """
  Identifies prey bodyguard/distraction behavior.
  """
  distraction_events = []
  fov_params = {
    'fov_front': 9, 'fov_back': 1, 'fov_sides': 5,  # From user
    'max_dist': PARAMS["bodyguard_fov_max_dist"]
  }

  for t_rel in range(PARAMS["bodyguard_delta_t_reaction"],
                     segment_end_time - segment_start_time - PARAMS["bodyguard_delta_t_A_acts"]):
    t_abs = segment_start_time + t_rel

    for pred_idx in predator_indices:
      if death_labels[pred_idx][t_abs] == 0: continue

      pred_pos_at_t = positions[t_abs, pred_idx]
      pred_ori_at_t = orientations[t_abs, pred_idx]

      # Phase 1: Identify P lurking/targeting A
      # TODO: Need a robust way to define "targeting".
      # Simplified: Closest prey in FOV near grass, or one P is moving towards.
      # My plan: find the prey that satisfies: 1. the predator is moving toward for 3 consecutive position changes
      # 2. If there are multiple, pick the one closest to predator.

      # Get all prey alive at t_abs
      current_alive_prey = [pi for pi in prey_indices if death_labels[pi][t_abs] == 1]
      if not current_alive_prey: continue

      # Get prey in FOV for predator P
      # Create per-timestep death labels for get_closest_agents_in_fov
      death_labels_at_t_abs_dict = {agent_idx: death_labels[agent_idx][t_abs] for agent_idx in death_labels}

      prey_in_fov = get_closest_agents_in_fov(pred_pos_at_t, pred_ori_at_t, pred_idx,
                                              positions[t_abs], death_labels_at_t_abs_dict,
                                              current_alive_prey, fov_params)
      if not prey_in_fov: continue

      # Let's assume predator is initially "interested" in the closest prey in FOV (Prey A)
      # if that prey is near grass. A more complex "targeting" state would be better.
      prey_A_candidate = None
      for p_fov in prey_in_fov:
        prey_A_idx = p_fov['id']
        prey_A_pos = p_fov['pos']
        # Check if Prey A is on/near grass
        is_A_on_grass = any(
          np.allclose(prey_A_pos, sg_pos) for sg_pos in safe_grass_locs)  # TODO: Use a radius for "near grass"
        if is_A_on_grass and np.linalg.norm(pred_pos_at_t - prey_A_pos) < PARAMS["bodyguard_lurking_radius_A"]:
          prey_A_candidate = prey_A_idx
          break  # Found a potential Prey A

      if prey_A_candidate is None: continue
      prey_A_id = prey_A_candidate

      # Store predator's initial target/focus (simplified to prey_A_id)
      initial_pred_target_vector = positions[t_abs, prey_A_id] - pred_pos_at_t

      # Phase 2 & 3: Prey B intervenes & P switches attention
      # Look in a window *before* t_abs for B's appearance and P's reaction
      # For this event at t_abs (P already switched, A acts), B's action was in (t_abs - delta_t_reaction, t_abs)

      prey_B_id = None
      pred_switched_due_to_B = False

      for t_intervene_rel in range(max(0, t_rel - PARAMS["bodyguard_delta_t_reaction"]), t_rel):
        t_intervene_abs = segment_start_time + t_intervene_rel
        if death_labels[pred_idx][t_intervene_abs] == 0: continue

        # Re-evaluate FOV at t_intervene_abs for potential Prey B
        death_labels_at_t_intervene_dict = {agent_idx: death_labels[agent_idx][t_intervene_abs] for agent_idx in
                                            death_labels}
        intervene_prey_in_fov = get_closest_agents_in_fov(positions[t_intervene_abs, pred_idx],
                                                          orientations[t_intervene_abs, pred_idx],
                                                          pred_idx, positions[t_intervene_abs],
                                                          death_labels_at_t_intervene_dict,
                                                          current_alive_prey, fov_params)
        for p_B_fov_candidate in intervene_prey_in_fov:
          if p_B_fov_candidate['id'] == prey_A_id: continue  # B must be different from A

          # Check if P's orientation/movement changed towards B *after* B seen, *before* t_abs
          # This requires comparing pred movement from t_intervene_abs to t_abs
          pred_pos_series = positions[t_intervene_abs: t_abs + 1, pred_idx]
          prey_B_pos_series = positions[t_intervene_abs: t_abs + 1, p_B_fov_candidate['id']]

          # Is predator now moving towards B?
          # TODO: Use the more robust is_agent_moving_towards if it takes series
          # Simplified check: did predator's vector change to align more with B?
          current_pred_target_vector = positions[t_abs, p_B_fov_candidate['id']] - positions[t_abs, pred_idx]
          dot_product_initial = np.dot(initial_pred_target_vector,
                                       orientations[t_abs, pred_idx])  # Assuming orientation vector
          dot_product_current = np.dot(current_pred_target_vector, orientations[t_abs, pred_idx])
          # This orientation dot product is not quite right. Need to compare change in predator's actual movement vector.

          # Simpler: Did predator get closer to B and further/same from A between t_intervene and t_abs?
          dist_P_A_intervene = np.linalg.norm(
            positions[t_intervene_abs, pred_idx] - positions[t_intervene_abs, prey_A_id])
          dist_P_B_intervene = np.linalg.norm(
            positions[t_intervene_abs, pred_idx] - positions[t_intervene_abs, p_B_fov_candidate['id']])
          dist_P_A_current = np.linalg.norm(positions[t_abs, pred_idx] - positions[t_abs, prey_A_id])
          dist_P_B_current = np.linalg.norm(positions[t_abs, pred_idx] - positions[t_abs, p_B_fov_candidate['id']])

          if dist_P_B_current < dist_P_B_intervene and \
              (dist_P_A_current >= dist_P_A_intervene or dist_P_A_current > PARAMS[
                "bodyguard_lurking_radius_A"] * 1.5):  # P moved away from A or A is no longer focus
            prey_B_id = p_B_fov_candidate['id']
            pred_switched_due_to_B = True
            break
        if pred_switched_due_to_B: break

      if not pred_switched_due_to_B or prey_B_id is None: continue

      # Phase 4: Prey A acts (Benefit)
      benefit_A_achieved = False
      # Check if A moves off grass and collects reward in window [t_abs, t_abs + delta_t_A_acts]
      for t_A_act_rel in range(PARAMS["bodyguard_delta_t_A_acts"]):
        t_A_act_abs = t_abs + t_A_act_rel
        if t_A_act_abs >= segment_end_time or death_labels[prey_A_id][t_A_act_abs] == 0: break

        is_A_on_grass_now = any(np.allclose(positions[t_A_act_abs, prey_A_id], sg_pos) for sg_pos in safe_grass_locs)
        if not is_A_on_grass_now:  # A moved off grass
          # Check for apple/acorn reward (assuming reward value > 0 and not a catch)
          # TODO: Clarify reward values for apples/acorns
          if rewards[t_A_act_abs, prey_A_id] > 0 and rewards[
            t_A_act_abs, prey_A_id] < 0.9:  # Assuming catch reward is 1.0
            benefit_A_achieved = True
            break

      if not benefit_A_achieved: continue

      # Phase 5: Cost to Prey B
      # Check stamina loss or capture for B in window [t_abs, t_abs + bodyguard_cost_B_window]
      cost_B_stamina = 0
      cost_B_caught = False
      stamina_B_at_switch = stamina[t_abs, prey_B_id]
      final_stamina_B = stamina_B_at_switch

      for t_B_cost_rel in range(PARAMS["bodyguard_cost_B_window"]):
        t_B_cost_abs = t_abs + t_B_cost_rel
        if t_B_cost_abs >= segment_end_time: break
        if death_labels[prey_B_id][t_B_cost_abs] == 0:
          # Check if caught by this predator P
          # TODO: Need robust catch attribution
          cost_B_caught = True
          final_stamina_B = 0
          break
        final_stamina_B = stamina[t_B_cost_abs, prey_B_id]

      cost_B_stamina = stamina_B_at_switch - final_stamina_B

      distraction_events.append({
        "time_A_acts_abs": t_abs, "predator_id": pred_idx, "prey_A_id": prey_A_id, "prey_B_id": prey_B_id,
        "cost_B_stamina": cost_B_stamina, "cost_B_caught": cost_B_caught, "benefit_A_resource": benefit_A_achieved
      })

  return distraction_events
