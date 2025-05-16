import numpy as np

from constants import PARAMS


def apple_frames(pos, rew, death, prey, grass, t0, t1):
  """Point‑based detector copied from working code."""
  ev = []
  for t in range(t0, t1):
    col = [q for q in prey if death[q][t] == 1 and rew[t, q] == 1 and tuple(pos[t, q]) not in grass]
    if len(col) >= PARAMS.apple_min_prey:
      g = pos[t, col]
      d = np.linalg.norm(g[:, None] - g[None, :], 2, 2)
      if np.max(d) <= PARAMS.apple_radius * 2:
        ev.append(dict(time=t, participants=col))
  return ev


def apple_periods(pos, rew, prey, t0, t1):
  """Period‑based detector copied & slightly cleaned."""
  radius = PARAMS.apple_period_radius
  ev = []
  t = t0
  while t < t1:
    seeds = [q for q in prey if rew[t, q] == 1]
    if not seeds:
      t += 1
      continue
    centre = pos[t, seeds[0]]
    grp = {seeds[0]}
    last = t
    dur = 1
    for tau in range(t + 1, t1):
      if tau - last > PARAMS.apple_period_gap: break
      close = {q for q in grp if np.linalg.norm(pos[tau, q] - centre) <= radius}
      join = [q for q in prey if rew[tau, q] == 1 and np.linalg.norm(pos[tau, q] - centre) <= radius]
      if join:
        last = tau
        close.update(join)
      if not close:
        break
      grp = close
      centre = np.median(pos[tau, list(grp)], 0)
      dur += 1
    if dur >= PARAMS.apple_period_min_len:
      ev.append(dict(time_start=t, time_end=t + dur, participants=list(grp)))
    t += dur
  return ev

# TODO: replicate apple_periods but:
#   • take only events whose centre lies within `acorn_tiles`
#   • stop the period once ANY participant reaches safe grass carrying acorn
# Return structure identical to apple_periods.
def acorn_periods(*args, **kwargs):
    """Not implemented yet."""
    return None
