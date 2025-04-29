import numpy as np

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


def mark_death_periods(data):
  """
  Mark each element with 0 if it is part of a 'death period'
  (20 consecutive zeros followed immediately by a 1.0), otherwise 1.

  Uses NumPy vectorization and convolution to more efficiently detect
  consecutive zeros than a purely Python-based loop.

  :param data: A list-like or NumPy array of floats in [0,1].
  :return: A NumPy array of the same length, with 0 for 'death' and 1 for 'living'.
  """
  data = np.asarray(data, dtype=float)
  n = len(data)

  # Initialize all as 'living' (1)
  labels = np.ones(n, dtype=int)

  # Create a boolean mask of zeros
  zero_mask = (data == 0.0)

  # Convolve this boolean mask with a window of length 20
  # conv[i] -> number of zeros in data[i : i+20]
  conv = np.convolve(zero_mask, np.ones(20, dtype=int), mode='valid')

  # Indices i for which data[i : i+20] are all zeros => conv[i] == 20
  potential_starts = np.where(conv == 20)[0]

  # Check if the element right after those 20 zeros is 1.0
  for start_idx in potential_starts:
    # The index of the element that should be 1.0 after 20 zeros
    check_idx = start_idx + 20
    if check_idx < n and data[check_idx] == 1.0:
      # Mark the entire 20-element region as death (0)
      labels[start_idx: check_idx] = 0

  return labels