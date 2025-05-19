import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Sequence
from sklearn.preprocessing import StandardScaler  # Import StandardScaler

EPS = 1e-9


# ----------------------------------------------------------------------
# 1) ACTION PREPROCESSING: one-hot → smoothing → z-score
# ----------------------------------------------------------------------
def _one_hot_actions(actions: np.ndarray, num_actions: int = 8) -> np.ndarray:
    """
    Converts integer-coded actions to one-hot representation.

    Args:
        actions: (T, N) int codes in [0..num_actions-1].
        num_actions: The number of possible actions.

    Returns:
        one_hot: (T, N, num_actions) floats.
    """
    T, N = actions.shape
    oh = np.zeros((T, N, num_actions), dtype=float)
    valid = (actions >= 0) & (actions < num_actions)
    t_idx, n_idx = np.nonzero(valid)
    a_idx = actions[t_idx, n_idx]
    oh[t_idx, n_idx, a_idx] = 1.0
    return oh


def _smooth_actions(one_hot: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Simple moving average along time for each (agent, action) channel.

    Args:
        one_hot: (T, N, A) one-hot action representation.
        window: Size of the smoothing window.

    Returns:
        smoothed: (T, N, A) smoothed actions.
    """
    T, N, A = one_hot.shape
    pad = np.zeros((window - 1, N, A), dtype=float)
    cum = np.concatenate([pad, one_hot], axis=0).cumsum(axis=0)
    # smoothed[t] = (cum[t+1] - cum[t+1-window]) / window
    sm = (cum[window:] - cum[:-window]) / float(window)
    return sm



def _zscore_actions_with_death_mask(
    actions: np.ndarray, death: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Z-scores action features, handling dead periods by assigning 0, using sklearn.

    Args:
        actions: (T, N, A) Smoothed action features.
        death: {agent_idx: (T,) bool or 1/0} Agent death labels.

    Returns:
        out: (T, N, A) Z-scored actions, with 0s for death periods.
    """
    T, N, A = actions.shape
    out = np.zeros_like(actions)
    scaler = StandardScaler()

    for n in range(N):
        for a in range(A):
            ts = actions[:, n, a]
            alive_mask = death[n] == 1
            alive_ts = ts[alive_mask].reshape(-1, 1)  # Reshape for StandardScaler
            if alive_ts.size > 0:
                scaler.fit(alive_ts)
                z = scaler.transform(ts.reshape(-1, 1)).flatten()  # Flatten the result
                out[:, n, a] = z
            else:
                out[:, n, a] = 0.0
    return out



# ----------------------------------------------------------------------
# 2) SPATIAL SYNC helper
# ----------------------------------------------------------------------
def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors.

    Args:
        u: First vector (NumPy array).
        v: Second vector (NumPy array).

    Returns:
        Cosine similarity (float) or NaN if either vector has zero norm.
    """
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < EPS or nv < EPS:
        return np.nan
    return float(np.dot(u, v) / (nu * nv))



# ----------------------------------------------------------------------
# 3) MAIN per-event sync
# ----------------------------------------------------------------------
def compute_sync_score(
    positions: np.ndarray,
    actions: np.ndarray,
    death: Dict[int, np.ndarray],
    participants: Sequence[int],
    time_start: int,
    time_end: int,
    action_smooth_window: int = 3,
    weight_spatial: float = 0.5,
) -> float:
    """
    Computes a synchronization score for a given time window and set of participants.

    Args:
        positions: (T, N, 2) Agent positions.
        actions: (T, N) Integer-coded actions.
        death: {agent_idx: (T,) bool or 1/0} Agent death labels.
        participants: List of agent IDs to consider for synchronization.
        time_start: Start timestep of the window.
        time_end: End timestep of the window (inclusive).
        action_smooth_window: Size of the action smoothing window.
        weight_spatial: Weight for the spatial component of the sync score.

    Returns:
        A single synchronization score in [0, 1] for the specified window and participants,
        or NaN if the window is invalid or there are fewer than 2 alive participants.
    """
    if time_end <= time_start or len(participants) < 2:
        return np.nan

    # 1) compute velocities
    pos_seg = positions[time_start : time_end + 1, :, :]  # (L, N, 2)
    vel = np.diff(pos_seg, axis=0, prepend=pos_seg[:1])  # (L, N, 2)

    # 2) prepare smoothed & z-scored action features once, for the segment
    oh = _one_hot_actions(actions, num_actions=8)  # (T,N,8)
    smooth = _smooth_actions(oh, window=action_smooth_window)  # (T,N,8)
    zscored = _zscore_actions_with_death_mask(smooth, death)  # (T,N,8)

    vel_scores = []
    act_scores = []

    for t in range(vel.shape[0]):
        alive_agents = [p for p in participants if death[p][time_start + t] == 1]
        if len(alive_agents) < 2:
            continue

        v_t = vel[t, alive_agents, :]  # (K,2)
        f_t = zscored[time_start + t, alive_agents, :]  # (K,8)

        K = v_t.shape[0]

        # spatial: mean pairwise cosine on v_t
        cos_vals = []
        for i in range(K):
            for j in range(i + 1, K):
                c = _cosine(v_t[i], v_t[j])
                if not np.isnan(c):
                    cos_vals.append(c)
        if cos_vals:
            vel_scores.append(np.mean(cos_vals))

        # action: mean pairwise cosine on f_t
        act_vals = []
        for i in range(K):
            for j in range(i + 1, K):
                c = _cosine(f_t[i], f_t[j])
                if not np.isnan(c):
                    act_vals.append(c)
        if act_vals:
            act_scores.append(np.mean(act_vals))

    if not vel_scores and not act_scores:
        return np.nan

    mv = np.nanmean(vel_scores) if vel_scores else np.nan
    ma = np.nanmean(act_scores) if act_scores else np.nan

    # blend, ignoring NaNs
    w_s = weight_spatial if not np.isnan(mv) else 0.0
    w_a = (1.0 - weight_spatial) if not np.isnan(ma) else 0.0
    denom = w_s + w_a
    if denom < EPS:
        return np.nan
    return ((w_s * (mv if not np.isnan(mv) else 0.0)) + (w_a * (ma if not np.isnan(ma) else 0.0))) / denom



def compute_all_sync_scores(
    positions: np.ndarray,
    actions: np.ndarray,
    death: Dict[int, np.ndarray],
    active_segments: List[Tuple[int, int]],
    events: Optional[List[Dict[str, Any]]] = None,
    action_smooth_window: int = 3,
    weight_spatial: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes synchronization scores for the entire episode, active segments, and provided events.

    Args:
        positions: (T, N, 2) Agent positions.
        actions: (T, N) Integer-coded actions.
        death: Dictionary of agent death labels.
        active_segments: List of (start, end) tuples defining active segments.
        events: (Optional) List of event dictionaries. If provided, sync scores are added to each event.
        action_smooth_window: Size of the action smoothing window.
        weight_spatial: Weight for the spatial component of the sync score.

    Returns:
        A dictionary containing the synchronization scores:
        {
            "episode_sync": float,  # Sync score for the entire episode
            "segment_sync": List[float],  # Sync scores for each active segment
            "event_sync": List[float],  # Sync scores for each event (if events provided)
        }
        Each score is a float between 0 and 1, or NaN if not computable.
    """
    T = positions.shape[0]
    N = positions.shape[1]
    all_participants = list(range(N))

    results = {}

    # 1. Episode-wise synchronization
    results["episode_sync"] = compute_sync_score(
        positions=positions,
        actions=actions,
        death=death,
        participants=all_participants,
        time_start=0,
        time_end=T - 1,
        action_smooth_window=action_smooth_window,
        weight_spatial=weight_spatial,
    )

    # 2. Segment-wise synchronization
    results["segment_sync"] = [
        compute_sync_score(
            positions=positions,
            actions=actions,
            death=death,
            participants=all_participants,  # Or should this be limited to agents alive in the segment?
            time_start=start,
            time_end=end,
            action_smooth_window=action_smooth_window,
            weight_spatial=weight_spatial,
        )
        for start, end in active_segments
    ]

    # 3. Event synchronization (if events are provided)
    if events:
        event_sync_scores = []
        for event in events:
            event_sync_score = compute_sync_score(
                positions=positions,
                actions=actions,
                death=death,
                participants=event["participants"],
                time_start=event["time_start"],
                time_end=event["time_end"],
                action_smooth_window=action_smooth_window,
                weight_spatial=weight_spatial,
            )
            event["sync"] = event_sync_score  # Mutate events in place as requested.
            event_sync_scores.append(event_sync_score)
        results["event_sync"] = event_sync_scores

    return results
