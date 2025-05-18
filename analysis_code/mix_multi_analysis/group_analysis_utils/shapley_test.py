import pickle
import math
import random
from typing import List, Dict, Tuple, Sequence, Optional
import numpy as np  # Added import for NumPy


def compute_metrics_for_segment(
    rewards_seq: List[List[float]],
    roles: List[str],
    death_labels: Optional[Dict[int, np.ndarray]] = None,
    segment_start: int = 0,
) -> Dict[str, float]:
  """
  Compute Efficiency, Equality, and Sustainability for a single segment of an episode.

  Args:
      rewards_seq: Per-step reward lists for each agent within the segment.
          Shape: (segment_length, num_agents).
      roles: Roles for each agent (e.g., ['predator','prey',...]).
          Length should match the number of agents in rewards_seq.
      death_labels: (Optional) Agent death labels for the *entire episode*.
          Dict: {agent_index: np.array of aliveness (1) / death (0) flags}.
      segment_start: Starting timestep of the segment within the *entire episode*.

  Returns:
      Dictionary containing the calculated metrics:
      - 'efficiency': Total team reward.
      - 'equality': 1 - (Gini coefficient).
      - 'sustainability': Average *absolute* timestep of reward collection.
  """
  num_agents = len(roles)
  segment_length = len(rewards_seq)

  # Handle empty segment
  if segment_length == 0:
    return {
      "efficiency": 0.0,
      "equality": 1.0,
      "sustainability": float(segment_start),
    }

  # Calculate total rewards for each agent
  total_rewards = [sum(agent_rewards) for agent_rewards in zip(*rewards_seq)]

  # Efficiency: total team reward
  efficiency = sum(total_rewards)

  # Equality: 1 - (Gini coefficient)
  if efficiency <= 1e-9:
    equality = 1.0 if all(abs(r - total_rewards[0]) < 1e-9 for r in total_rewards) else 0.0
  else:
    diff_sum = sum(
      abs(total_rewards[i] - total_rewards[j])
      for i in range(num_agents)
      for j in range(num_agents)
    )
    inequality = diff_sum / (2 * num_agents * efficiency)
    equality = 1 - inequality
    equality = max(0.0, equality)

  # Sustainability: average absolute timestep of reward collection
  sustainability_times = []
  for agent_index in range(num_agents):
    agent_reward_times = [
      t + segment_start
      for t, reward in enumerate(rewards_seq)
      if reward[agent_index] > 1e-9
    ]

    if agent_reward_times:
      agent_sustainability = sum(agent_reward_times) / len(agent_reward_times)
      if death_labels is not None:
        episode_end = segment_start + segment_length - 1
        agent_dead_at_end = death_labels[agent_index][episode_end] == 0
        if agent_dead_at_end:
          last_alive_time = episode_end
          for t_rev in range(episode_end, segment_start - 1, -1):
            if death_labels[agent_index][t_rev] == 1:
              last_alive_time = t_rev
              break
          agent_sustainability = min(
            agent_sustainability, last_alive_time
          )  # Use min of reward and death
      sustainability_times.append(agent_sustainability)
    else:
      sustainability_times.append(float(segment_start + segment_length))

  sustainability = (
    sum(sustainability_times) / num_agents if num_agents > 0 else float(segment_start + segment_length)
  )

  return {
    "efficiency": efficiency,
    "equality": equality,
    "sustainability": sustainability,
  }


def calculate_segment_shapley_values(
    episode_data: List[List[float]],
    roles: List[str],
    death_labels: Optional[Dict[int, np.ndarray]],
    segment_start: int,
    segment_end: int,
    M: int,
) -> Dict:
  """
  Calculate Shapley values for a single segment.

  Args:
      episode_data:  Reward data for the entire episode.
      roles:  List of agent roles.
      death_labels: Death labels for the entire episode.
      segment_start: Start of the segment.
      segment_end: End of the segment.
      M:  Monte Carlo iterations.

  Returns:
       Dictionary of Shapley values for the segment.
  """
  segment_data = episode_data[segment_start:segment_end]
  num_agents = len(roles)
  segment_length = len(segment_data)

  def coalition_value(agent_indices: Sequence[int]) -> Dict[str, float]:
    """
    Calculates the value (metrics) of a coalition within the current segment.

    Args:
        agent_indices: List of agent indices in the coalition.

    Returns:
        Dictionary of metrics ('efficiency', 'equality', 'sustainability').
    """
    coalition_rewards = [
      [segment_data[t][i] for i in agent_indices] for t in range(segment_length)
    ]
    coalition_roles = [roles[i] for i in agent_indices]
    coalition_death_labels: Optional[Dict[int, np.ndarray]] = None
    if death_labels is not None:
      coalition_death_labels = {
        agent_indices.index(original_index): death_labels[original_index]
        for original_index in agent_indices
      }
    return compute_metrics_for_segment(
      coalition_rewards, coalition_roles, coalition_death_labels, segment_start
    )

  shapley_sums = {
    "eff": [0.0] * num_agents,
    "eq": [0.0] * num_agents,
    "sus": [0.0] * num_agents,
  }

  for _ in range(M):
    perm = list(range(num_agents))
    random.shuffle(perm)
    current_coalition = []
    current_val = coalition_value([])
    for agent_idx in perm:
      new_coalition = current_coalition + [agent_idx]
      new_val = coalition_value(new_coalition)
      shapley_sums["eff"][agent_idx] += (
          new_val["efficiency"] - current_val["efficiency"]
      )
      shapley_sums["eq"][agent_idx] += (
          new_val["equality"] - current_val["equality"]
      )
      shapley_sums["sus"][agent_idx] += (
          new_val["sustainability"] - current_val["sustainability"]
      )
      current_coalition = new_coalition
      current_val = new_val

  return {
    "shapley_efficiency": [s / M for s in shapley_sums["eff"]],
    "shapley_equality": [s / M for s in shapley_sums["eq"]],
    "shapley_sustainability": [s / M for s in shapley_sums["sus"]],
  }



def compute_shapley_values_for_segments(
    episode_data: List[List[float]],
    role_map: Dict[int, str],
    active_segments: List[Tuple[int, int]],
    death_labels: Optional[Dict[int, np.ndarray]] = None,
    positions: Optional[np.ndarray] = None,
    orientations: Optional[np.ndarray] = None,
    actions: Optional[np.ndarray] = None,
    rewards: Optional[np.ndarray] = None,  # Redundant with episode_data, but kept for consistency
    stamina: Optional[np.ndarray] = None,
    max_predators: Optional[int] = None,
    max_prey: Optional[int] = None,
    M: int = 1000,
) -> Dict:
  """
  Compute Shapley values for each active segment of an episode, and for the episode as a whole.

  Args:
      episode_data:  Per-step reward lists for the entire episode.
          Shape: (episode_length, num_agents).
      role_map: Dictionary mapping agent index to role (e.g., {0: 'predator', 1: 'prey', ...}).
      active_segments: List of (start, end) tuples defining active segments in the episode.
      death_labels: (Optional) Agent death labels for the entire episode.
          Dict: {agent_index: np.array of aliveness (1) / death (0) flags}.
      positions: (Optional) Agent positions over the entire episode.
          Shape: (episode_length, num_agents, 2).
      orientations: (Optional) Agent orientations.
          Shape: (episode_length, num_agents, 2).
      actions: (Optional) Agent actions.
          Shape: (episode_length, num_agents).
      rewards: (Optional) Agent rewards.  Redundant with episode_data.
          Shape: (episode_length, num_agents).
      stamina: (Optional) Agent stamina.
          Shape: (episode_length, num_agents).
      max_predators: (Optional) Maximum number of predator agents. If None, inferred from roles.
      max_prey: (Optional) Maximum number of prey agents. If None, inferred from roles.
      M: Number of random permutations for Monte Carlo approximation.

  Returns:
      A dictionary containing:
      - 'shapley_efficiency': Shapley values for episode-wise efficiency.
      - 'shapley_equality': Shapley values  for episode-wise equality.
      - 'shapley_sustainability': Shapley values for episode-wise sustainability.
      - 'segmentations': A list of dictionaries, one for each active segment, containing:
          - 'time_start': Start timestep of the segment in the episode.
          - 'time_end': End timestep of the segment in the episode.
          - 'shapley_efficiency': Shapley values for the segment.
          - 'shapley_equality':  Shapley values for the segment.
          - 'shapley_sustainability': Shapley values for the segment.
  """
  episode_length = len(episode_data)
  roles = [role_map[i] for i in range(len(role_map))]
  num_agents = len(roles)
  P = max_predators if max_predators is not None else roles.count("predator")
  Q = max_prey if max_prey is not None else roles.count("prey")
  total_agents = P + Q

  # Calculate shapley values for the entire episode
  episode_shapley_values = calculate_segment_shapley_values(
    episode_data, roles, death_labels, 0, episode_length, M
  )

  # Calculate shapley values for each segment
  segment_results = []
  for seg_idx, (seg_start, seg_end) in enumerate(active_segments):
    segment_shapley_values = calculate_segment_shapley_values(
      episode_data, roles, death_labels, seg_start, seg_end, M
    )
    segment_results.append(
      {
        "time_start": seg_start,
        "time_end": seg_end,
        "shapley_efficiency": segment_shapley_values["shapley_efficiency"],
        "shapley_equality": segment_shapley_values["shapley_equality"],
        "shapley_sustainability": segment_shapley_values["shapley_sustainability"],
      }
    )

  return {
    "shapley_efficiency": episode_shapley_values["shapley_efficiency"],
    "shapley_equality": episode_shapley_values["shapley_equality"],
    "shapley_sustainability": episode_shapley_values["shapley_sustainability"],
    "segmentations": segment_results,
  }


if __name__ == "__main__":
  # Example usage (replace with your actual data)
  episode_name = (
    "mix_AH20250107_pred_0_AH20250107_pred_1_AH20250107_prey_3_AH20250107_prey_4_"
    "AH20250107_prey_5_AH20250107_prey_6predator_prey__open_debug_agent_0_1_3_4_5_6"
  )
  folder = f'/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/{episode_name}/episode_pickles/'

  import os

  base = os.path.basename(episode_name).split("predator_prey__")[0]
  from helper import parse_agent_roles

  role_map, src_map = parse_agent_roles(base)  # role_map[idx] = "predator" or "prey"
  predator_ids = [i for i, r in role_map.items() if r == "predator"]
  prey_ids = [i for i, r in role_map.items() if r == "prey"]
  agent_roles = [role_map[i] for i in range(len(role_map))]

  # Load the episode data
  import pickle

  files = os.listdir(folder)
  files = sorted(f for f in files if f.endswith(".pkl"))
  for fname in files:
    with open(os.path.join(folder, fname), "rb") as fp:
      episode_data = pickle.load(fp)

    pos_ep = np.array([d["POSITION"] for d in episode_data])
    ori_ep = np.array([d["ORIENTATION"] for d in episode_data])
    act_ep = np.array([d["actions"] for d in episode_data])
    rew_ep = np.array([d["rewards"] for d in episode_data])
    sta_ep = np.array([d["STAMINA"] for d in episode_data])

    from segmentation import mark_death_periods, segment_active_phases

    death_labels_ep = {
      i: mark_death_periods(sta_ep[:, i]) for i in range(sta_ep.shape[1])
    }
    death_labels_all_prey_ep = {
      pi: death_labels_ep[pi] for pi in prey_ids if pi in death_labels_ep
    }
    active_segments = segment_active_phases(death_labels_all_prey_ep)

    # Compute Shapley values for each segment
    results = compute_shapley_values_for_segments(
      episode_data=rew_ep,
      role_map=role_map,
      active_segments=active_segments,
      death_labels=death_labels_ep,
      positions=pos_ep,
      orientations=ori_ep,
      actions=act_ep,
      rewards=rew_ep,
      stamina=sta_ep,
      max_predators=len(predator_ids),
      max_prey=len(prey_ids),
      M=100,
    )

    # Print the results in the desired format
    print(f"Shapley values for episode: {fname}")
    print(results)
