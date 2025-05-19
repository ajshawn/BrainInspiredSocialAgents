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
    smoothed = np.zeros_like(one_hot, dtype=float)
    for n in range(N):
        for a in range(A):
            smoothed[:, n, a] = np.convolve(one_hot[:, n, a], np.ones(window) / window, mode='same')
    return smoothed


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
def _compute_pairwise_sync_score(
    positions: np.ndarray,
    actions: np.ndarray,
    death: Dict[int, np.ndarray],
    participants: Sequence[int],
    time_start: int,
    time_end: int,
    action_smooth_window: int = 3,
    weight_spatial: float = 0.5,
    position_smooth_window: int = 3,  # New parameter for position smoothing
) -> Dict[Tuple[int, int], float]:
    """
    Computes synchronization scores for all pairs of participants over a given time window.

    Args:
        positions: (T, N, 2) Agent positions.
        actions: (T, N) Integer-coded actions.
        death: {agent_idx: (T,) bool or 1/0} Agent death labels.
        participants: List of agent IDs to consider for synchronization.
        time_start: Start timestep of the window.
        time_end: End timestep of the window (inclusive).
        action_smooth_window: Size of the action smoothing window.
        weight_spatial: Weight for the spatial component of the sync score.
        position_smooth_window: Size of the smoothing window for positions.

    Returns:
        A dictionary where keys are tuples (agent_id1, agent_id2) representing pairs of
        participants, and values are the corresponding synchronization scores
        (float in [0, 1] or NaN).
    """
    if time_end <= time_start or len(participants) < 2:
        return {}  # Return empty dict for invalid input

    # 1) compute velocities
    pos_seg = positions[time_start : time_end + 1, :, :]  # (L, N, 2)

    # Apply smoothing to positions
    if position_smooth_window > 1:
        smoothed_pos_seg = np.zeros_like(pos_seg, dtype=float)
        for n in range(pos_seg.shape[1]):
            for d in range(pos_seg.shape[2]):
              try:
                smoothed_pos_seg[:, n, d] = np.convolve(
                    pos_seg[:, n, d], np.ones(position_smooth_window) / position_smooth_window, mode='same'
                )
              except:
                print(f"Error smoothing position for agent {n} in dimension {d}")
                smoothed_pos_seg[:, n, d] = pos_seg[:, n, d]
        pos_seg = smoothed_pos_seg

    vel = np.diff(pos_seg, axis=0, prepend=pos_seg[:1])  # (L, N, 2)

    # 2) prepare smoothed & z-scored action features once, for the segment
    oh = _one_hot_actions(actions, num_actions=8)  # (T,N,8)
    smooth = _smooth_actions(oh, window=action_smooth_window)  # (T,N,8)
    zscored = _zscore_actions_with_death_mask(smooth, death)  # (T,N,8)

    pair_scores = {}  # Store results

    for p1_idx, p1 in enumerate(participants):
        for p2_idx in range(p1_idx + 1, len(participants)):
            p2 = participants[p2_idx]
            pair = (p1, p2)
            vel_scores = []
            act_scores = []

            for t in range(vel.shape[0]):
                if death[p1][time_start + t] == 0 or death[p2][time_start + t] == 0:
                    continue  # Skip if either agent is dead
                try:
                  v_t = vel[t, [p1, p2], :]  # (2, 2)
                except:
                  print(f"Error accessing velocity for agents {p1} and {p2} at time {t}")
                  continue
                f_t = zscored[time_start + t, [p1, p2], :]  # (2, 8)

                # spatial: cosine similarity
                cos_v = _cosine(v_t[0], v_t[1])
                if not np.isnan(cos_v):
                    vel_scores.append(cos_v)

                # action: cosine similarity
                cos_a = _cosine(f_t[0], f_t[1])
                if not np.isnan(cos_a):
                    act_scores.append(cos_a)

            if not vel_scores and not act_scores:
                pair_scores[pair] = np.nan
                continue

            mv = np.nanmean(vel_scores) if vel_scores else np.nan
            ma = np.nanmean(act_scores) if act_scores else np.nan

            # blend, ignoring NaNs
            w_s = weight_spatial if not np.isnan(mv) else 0.0
            w_a = (1.0 - weight_spatial) if not np.isnan(ma) else 0.0
            denom = w_s + w_a
            if denom < EPS:
                pair_scores[pair] = np.nan
            else:
                pair_scores[pair] = (
                    (w_s * (mv if not np.isnan(mv) else 0.0)) + (w_a * (ma if not np.isnan(ma) else 0.0))
                ) / denom

    return pair_scores



def compute_episode_sync(
    positions: np.ndarray,
    actions: np.ndarray,
    death: Dict[int, np.ndarray],
    action_smooth_window: int = 3,
    weight_spatial: float = 0.5,
    position_smooth_window: int = 3,  # New parameter
) -> Dict[Tuple[int, int], float]:
    """
    Computes pairwise synchronization scores for the entire episode.

    Args:
        positions: (T, N, 2) Agent positions.
        actions: (T, N) Integer-coded actions.
        death: Dictionary of agent death labels.
        action_smooth_window: Size of the action smoothing window.
        weight_spatial: Weight for the spatial component of the sync score.
        position_smooth_window: Size of the smoothing window for positions.

    Returns:
        A dictionary of pairwise synchronization scores for the episode.
    """
    T = positions.shape[0]
    N = positions.shape[1]
    all_participants = list(range(N))
    return _compute_pairwise_sync_score(
        positions=positions,
        actions=actions,
        death=death,
        participants=all_participants,
        time_start=0,
        time_end=T - 1,
        action_smooth_window=action_smooth_window,
        weight_spatial=weight_spatial,
        position_smooth_window=position_smooth_window,
    )



def compute_segment_sync(
    positions: np.ndarray,
    actions: np.ndarray,
    death: Dict[int, np.ndarray],
    active_segments: List[Tuple[int, int]],
    action_smooth_window: int = 3,
    weight_spatial: float = 0.5,
    position_smooth_window: int = 3,  # New parameter
) -> List[Dict[Tuple[int, int], float]]:
    """
    Computes pairwise synchronization scores for each active segment.

    Args:
        positions: (T, N, 2) Agent positions.
        actions: (T, N) Integer-coded actions.
        death: Dictionary of agent death labels.
        active_segments: List of (start, end) tuples defining active segments.
        action_smooth_window: Size of the action smoothing window.
        weight_spatial: Weight for the spatial component of the sync score.
        position_smooth_window: Size of the smoothing window for positions.

    Returns:
        A list of dictionaries, where each dictionary contains the pairwise
        synchronization scores for the corresponding segment.
    """
    return [
        _compute_pairwise_sync_score(
            positions=positions,
            actions=actions,
            death=death,
            participants=list(range(positions.shape[1])),  # Consider all agents in the segment
            time_start=start,
            time_end=end,
            action_smooth_window=action_smooth_window,
            weight_spatial=weight_spatial,
            position_smooth_window=position_smooth_window,
        )
        for start, end in active_segments
    ]



def compute_event_sync(
    positions: np.ndarray,
    actions: np.ndarray,
    death: Dict[int, np.ndarray],
    event: Dict[str, Any],
    action_smooth_window: int = 3,
    weight_spatial: float = 0.5,
    position_smooth_window: int = 3,  # New parameter
) -> Dict[Tuple[int, int], float]:
    """
    Computes pairwise synchronization scores for a given event.

    Args:
        positions: (T, N, 2) Agent positions.
        actions: (T, N) Integer-coded actions.
        death: Dictionary of agent death labels.
        event: A dictionary representing the event, containing 'participants',
               'time_start', and 'time_end' keys.
        action_smooth_window: Size of the action smoothing window.
        weight_spatial: Weight for the spatial component of the sync score.
        position_smooth_window: Size of the smoothing window for positions.

    Returns:
        A dictionary of pairwise synchronization scores for the event.
    """
    return _compute_pairwise_sync_score(
        positions=positions,
        actions=actions,
        death=death,
        participants=event["participants"],
        time_start=event["time_start"],
        time_end=event["time_end"],
        action_smooth_window=action_smooth_window,
        weight_spatial=weight_spatial,
        position_smooth_window=position_smooth_window,
    )



def compute_all_sync_scores(
    positions: np.ndarray,
    actions: np.ndarray,
    death: Dict[int, np.ndarray],
    active_segments: List[Tuple[int, int]],
    events: Optional[List[Dict[str, Any]]] = None,
    action_smooth_window: int = 3,
    weight_spatial: float = 0.5,
    position_smooth_window: int = 3,  # New parameter
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
        position_smooth_window: Size of the smoothing window for positions.

    Returns:
        A dictionary containing the synchronization scores:
        {
            "episode_sync": Dict[Tuple[int,int], float],  # Pairwise sync scores for the entire episode
            "segment_sync": List[Dict[Tuple[int,int], float]],  # Pairwise sync scores for each segment
            "event_sync": List[Dict[Tuple[int,int], float]],  # Pairwise sync scores for each event (if events provided)
        }
        Each score is a float between 0 and 1, or NaN if not computable.
    """
    results = {}

    # 1. Episode-wise synchronization
    results["episode_sync"] = compute_episode_sync(
        positions=positions,
        actions=actions,
        death=death,
        action_smooth_window=action_smooth_window,
        weight_spatial=weight_spatial,
        position_smooth_window=position_smooth_window,
    )

    # 2. Segment-wise synchronization
    results["segment_sync"] = [
        compute_segment_sync(
            positions=positions,
            actions=actions,
            death=death,
            active_segments= [(start, end) for start, end in active_segments], # pass the active segments
            action_smooth_window=action_smooth_window,
            weight_spatial=weight_spatial,
            position_smooth_window=position_smooth_window,
        )
        for start, end in active_segments
    ]

    # 3. Event synchronization (if events are provided)
    if events:
        event_sync_scores = []
        for event in events:
            event_sync_score = compute_event_sync(
                positions=positions,
                actions=actions,
                death=death,
                event=event,
                action_smooth_window=action_smooth_window,
                weight_spatial=weight_spatial,
                position_smooth_window=position_smooth_window,
            )
            event["sync_pairs"] = event_sync_score  # Store pairwise scores in the event dict
            event_sync_scores.append(event_sync_score)
        results["event_sync"] = event_sync_scores

    return results



def analyse_events_with_sync(
    events: List[Dict[str, Any]],
    positions: np.ndarray,
    actions: np.ndarray,
    death: Dict[int, np.ndarray],
    *,
    action_smooth_window: int = 3,
    weight_spatial: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Enriches a list of events with pairwise synchronization scores.

    Args:
        events: List of event dictionaries.  Each event dict must contain
            'time_start', 'time_end', and 'participants' keys.
        positions: (T, N, 2) Agent positions.
        actions: (T, N) Integer-coded actions.
        death: Dictionary of agent death labels.
        action_smooth_window: Size of the action smoothing window.
        weight_spatial: Weight for the spatial component of the sync score.

    Returns:
        A new list of event dictionaries, where each event dictionary has
        been augmented with a 'sync_pairs' key.  The value of 'sync_pairs'
        is a dictionary of pairwise synchronization scores for that event.
    """
    enriched_events = []
    for event in events:
        event_sync_scores = compute_event_sync(
            positions=positions,
            actions=actions,
            death=death,
            event=event,
            action_smooth_window=action_smooth_window,
            weight_spatial=weight_spatial,
        )
        enriched_event = {
            **event,
            "sync_pairs": event_sync_scores,
        }
        enriched_events.append(enriched_event)
    return enriched_events

if __name__ == "__main__":
    episode_name = (
        'mix_AH20250107_pred_0_AH20250107_pred_1_AH20250107_prey_3_AH20250107_prey_4_'
        'AH20250107_prey_5_AH20250107_prey_6predator_prey__open_debug_agent_0_1_3_4_5_6'
    )
    folder = f'/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/{episode_name}/episode_pickles/'

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
        from map_utils import get_safe_tiles

        death_labels_ep = {i: mark_death_periods(sta_ep[:, i]) for i in range(sta_ep.shape[1])}
        death_labels_all_prey_ep = {pi: death_labels_ep[pi] for pi in prey_ids if pi in death_labels_ep}
        active_segments = segment_active_phases(death_labels_all_prey_ep)
        safe_grass_locs = get_safe_tiles(folder, map='smaller_13x13')

        from gather_and_fence import compute_good_gathering_segmented, compute_successful_fencing_and_helpers_segmented
        from cooperation import apple_periods_segmented
        from distraction import detect_distraction_events

        gather_events = compute_good_gathering_segmented(pos_ep, predator_ids, prey_ids, active_segments)
        fence_events = compute_successful_fencing_and_helpers_segmented(
            pos_ep, ori_ep, act_ep, rew_ep, predator_ids,
            prey_ids, death_labels_ep, active_segments=active_segments, radius=3,
        )
        apple_cooperation_events = apple_periods_segmented(
            pos_ep, rew_ep, death_labels_ep, prey_ids, safe_grass_locs,
            active_segments, cluster_radius=3, apple_period_min_len=5, apple_period_gap=5,
            participation_threshold=0.4
        )

        distraction_events = detect_distraction_events(
            positions=pos_ep,
            orientations=ori_ep,
            death_labels=death_labels_ep,
            safe_grass=safe_grass_locs,
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
            safe_grass=safe_grass_locs,
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



        episode_segment_dfs.append(
            {
                "folder": base,
                "episode": fname.replace(".pkl", ""),
                "seg_idx": [(start, end) for start, end in active_segments],
                "good_gathering": gather_events,
                "fencing_events": fence_events,
                "apple_cooperation_events": apple_cooperation_events,
                "distraction_events": distraction_events,
            }
        )

        # Now, lets start the analysis based on the events we have by adding the altruism analysis
        for ev in fence_events:
            ev["type"] = "fence"
            # we add participants
            ev["participants"] = sorted(np.hstack([ev['helper'], ev['beneficiary'], ev['predator']]).astype(int).tolist())
            ev["time_start"] = max(0, ev["time"] - 10)
            ev["time_end"] = min(len(pos_ep) - 1, ev["time"] + 20)

        for ev in distraction_events:
            ev["type"] = "distraction"
            ev["helpers"] = [ev["helper"]] if type(ev["helper"]) == int else ev["helper"]
            ev["participants"] = sorted(np.hstack([ev['helper'], ev['beneficiary'], ev['predator']]).astype(int).tolist())
            # time_start/shift/end already there

        # 3) concatenate and call
        all_events = fence_events + distraction_events

        # 4) add sync scores to events
        episode_sync_scores = compute_episode_sync(
            positions=pos_ep,
            actions=act_ep,
            death=death_labels_ep,
            action_smooth_window=3,
            weight_spatial=0.5,
        )

        segment_sync_scores = compute_segment_sync(
            positions=pos_ep,
            actions=act_ep,
            death=death_labels_ep,
            active_segments=active_segments,
            action_smooth_window=3,
            weight_spatial=0.5,
        )
        event_sync_scores = []
        for event in all_events:
            event_sync_score = compute_event_sync(
                positions=pos_ep,
                actions=act_ep,
                death=death_labels_ep,
                event=event,
                action_smooth_window=3,
                weight_spatial=0.5,
            )
            event["sync_pairs"] = event_sync_score  # Store pairwise scores in the event dict
            event_sync_scores.append(event_sync_score)

        print("Episode Sync", episode_sync_scores)
        print("Segment Sync", segment_sync_scores)
        print("Event Sync", event_sync_scores)