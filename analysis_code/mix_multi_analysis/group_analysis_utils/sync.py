import numpy as np

def pairwise_cosine(vel: np.ndarray) -> float:
    if vel.shape[1]<2:
      return np.nan
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


def add_event_sync(events, pos):
    for e in events:
        seg=pos[e['time_start']:e['time_end'], e['participants']]
        e['sync']=pairwise_cosine(np.diff(seg,axis=0,prepend=seg[:1]))
    return events
