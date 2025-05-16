from constants import PARAMS

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

# Idea: if prey H is caught within `block_radius` along LOS between
# predator P and low‑stamina prey B, and B survives ≥ k timesteps,
# count as sacrifice.  Requires scanning catch frames.
def detect_sacrifice(*args, **kwargs):
    """Prototype not implemented – returns None for now."""
    return None


