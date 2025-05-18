import numpy as np

def mark_death_periods(stamina_trace, respawn_cooldown=100):
  stamina_trace = np.asarray(stamina_trace, dtype=float)
  n = len(stamina_trace)

  # Initialize all as 'living' (1)
  labels = np.ones(n, dtype=int)

  # Create a boolean mask of zeros
  zero_mask = (stamina_trace == 0.0)

  # Convolve this boolean mask with a window of length 20
  # conv[i] -> number of zeros in data[i : i+20]
  conv = np.convolve(zero_mask, np.ones(respawn_cooldown, dtype=int), mode='valid')

  # Indices i for which data[i : i+20] are all zeros => conv[i] == 20
  potential_starts = np.where(conv == respawn_cooldown)[0]

  # Check if the element right after those 20 zeros is 1.0
  for start_idx in potential_starts:
    # The index of the element that should be 1.0 after 20 zeros
    check_idx = start_idx + respawn_cooldown
    if check_idx < n and stamina_trace[check_idx] == 1.0:
      # Mark the entire 20-element region as death (0)
      labels[start_idx: check_idx] = 0

  return labels


def segment_active_phases(death_labels_all_prey, min_prey_alive=1, min_all_dead_duration=5):
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
