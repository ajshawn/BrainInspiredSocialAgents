import pickle, math, random


def compute_metrics_for_episode(episode):
  """Compute Efficiency, Equality, and Sustainability for a single episode."""
  roles = episode['roles']  # list of roles for each agent (e.g. ['predator','prey',...])
  rewards_seq = episode['rewards']  # list of per-step reward lists for each agent
  N = len(roles)
  # Sum total reward for each agent
  if len(rewards_seq) == 0:
    total_rewards = [0.0] * N
  else:
    # Transpose the list of lists to sum rewards of each agent over all time steps
    total_rewards = [sum(agent_rewards) for agent_rewards in zip(*rewards_seq)]
  # Efficiency: total team reward (sum of all agent returns)
  total_team_reward = sum(total_rewards)
  # Equality: 1 - (Gini coefficient) based on final returns
  if total_team_reward <= 1e-9:
    # If no total reward (or very tiny), define equality as 1 if all got equal reward, else 0
    all_equal = all(abs(r - total_rewards[0]) < 1e-9 for r in total_rewards)
    equality = 1.0 if all_equal else 0.0
  else:
    # Compute Gini-based inequality: sum of pairwise differences
    diff_sum = 0.0
    for i in range(N):
      for j in range(N):
        diff_sum += abs(total_rewards[i] - total_rewards[j])
    inequality = diff_sum / (2 * N * total_team_reward)  # Gini coefficient
    equality = 1 - inequality
    if equality < 0: equality = 0.0
  # Sustainability: average time step at which rewards are collected.
  # For each agent, find the average time of its reward events (if none, use episode length as the time).
  T = len(rewards_seq)
  times = []
  for i in range(N):
    # All time steps where agent i got a positive reward
    reward_times = [t for t, rew in enumerate(zip(*rewards_seq)[i]) if rew > 1e-9]
    if reward_times:
      times.append(sum(reward_times) / len(reward_times))
    else:
      # No reward for this agent; assume reward collected at end (T) for sustainability calc
      times.append(float(T))
  sustainability = sum(times) / N if N > 0 else 0.0
  return total_team_reward, equality, sustainability


def compute_shapley_values_for_roles(episodes, max_predators=None, max_prey=None, M=1000):
  """
  Compute Shapley values for each agent role (predator or prey) with respect to Efficiency, Equality, and Sustainability.
  - episodes: list of episode data dicts, each with 'roles' (list of agent roles) and 'rewards' (list of rewards per step per agent).
  - max_predators, max_prey: total number of predator and prey agents in the grand coalition (if not provided, inferred from data).
  - M: number of random permutations to sample for Monte Carlo approximation of Shapley values.
  Returns: (shapley_values, (predator_indices, prey_indices)):
           shapley_values is a dict with keys 'eff', 'eq', 'sus', each a list of Shapley values for agents [0..P+Q-1].
           predator_indices and prey_indices are lists of indices corresponding to predator and prey roles respectively.
  """
  # 1. Compute average Efficiency, Equality, Sustainability for each (pred_count, prey_count) combination
  combo_metrics = {}  # (p,q) -> {'eff': avg_eff, 'eq': avg_eq, 'sus': avg_sus}
  max_p = 0
  max_q = 0
  total_len = 0
  count_eps = 0
  for ep in episodes:
    roles = ep['roles']
    p = roles.count('predator')
    q = roles.count('prey')
    max_p = max(max_p, p)
    max_q = max(max_q, q)
    eff, eq, sus = compute_metrics_for_episode(ep)
    # accumulate metrics for this combination
    combo_metrics.setdefault((p, q), {'eff': [], 'eq': [], 'sus': []})
    combo_metrics[(p, q)]['eff'].append(eff)
    combo_metrics[(p, q)]['eq'].append(eq)
    combo_metrics[(p, q)]['sus'].append(sus)
    total_len += len(ep.get('rewards', []))
    count_eps += 1
  # Average the metrics for each combination
  for combo, metric_lists in combo_metrics.items():
    combo_metrics[combo] = {
      'eff': sum(metric_lists['eff']) / len(metric_lists['eff']),
      'eq': sum(metric_lists['eq']) / len(metric_lists['eq']),
      'sus': sum(metric_lists['sus']) / len(metric_lists['sus'])
    }
  # Determine grand coalition size (maximum predators P and prey Q present in any scenario)
  P = max_predators if max_predators is not None else max_p
  Q = max_prey if max_prey is not None else max_q
  total_agents = P + Q
  predator_indices = list(range(P))
  prey_indices = list(range(P, P + Q))
  # Representative episode length (average length) for fallback in sustainability
  rep_length = int(total_len / count_eps) if count_eps > 0 else 0

  # Helper to get metric values for a coalition defined by (pred_count, prey_count)
  def coalition_value(pred_count, prey_count):
    if (pred_count, prey_count) in combo_metrics:
      return combo_metrics[(pred_count, prey_count)]
    # If no data for this combo, define default outcome:
    if pred_count == 0 and prey_count == 0:
      # No agents: no reward, perfectly equal (trivial), no time (0)
      return {'eff': 0.0, 'eq': 1.0, 'sus': 0.0}
    if pred_count == 0 or prey_count == 0:
      # One side absent: assume no interactions -> zero reward, equality = 1 (all present agents get 0),
      # and sustainability = full episode length (no reward collected until end).
      return {'eff': 0.0, 'eq': 1.0, 'sus': float(rep_length)}
    # For any other missing combo, assume zero reward and neutral metrics (as a conservative default).
    return {'eff': 0.0, 'eq': 1.0, 'sus': float(rep_length)}

  # 2. Monte Carlo approximation of Shapley values
  shapley_sums = {'eff': [0.0] * total_agents, 'eq': [0.0] * total_agents, 'sus': [0.0] * total_agents}
  for _ in range(M):
    # sample a random permutation of all agents (by index)
    perm = list(range(total_agents))
    random.shuffle(perm)
    current_pred = current_pre = 0
    current_val = coalition_value(0, 0)  # start with empty coalition
    # Add agents one by one in this random order
    for idx in perm:
      # Determine role of idx (predator or prey) and update counts
      if idx < P:  # predator agent
        new_pred = current_pred + 1
        new_pre = current_pre
      else:  # prey agent
        new_pred = current_pred
        new_pre = current_pre + 1
      # Coalition value with this agent added
      new_val = coalition_value(new_pred, new_pre)
      # Marginal contribution = new_val - current_val for each metric
      shapley_sums['eff'][idx] += (new_val['eff'] - current_val['eff'])
      shapley_sums['eq'][idx] += (new_val['eq'] - current_val['eq'])
      shapley_sums['sus'][idx] += (new_val['sus'] - current_val['sus'])
      # Update current coalition state
      current_pred, current_pre = new_pred, new_pre
      current_val = new_val
  # Average contributions over M samples to get Shapley values
  shapley_values = {
    'eff': [s / M for s in shapley_sums['eff']],
    'eq': [s / M for s in shapley_sums['eq']],
    'sus': [s / M for s in shapley_sums['sus']]
  }
  return shapley_values, (predator_indices, prey_indices)

# Example usage (assuming episodes_data is a list of loaded episode dicts):
# episodes_data = [pickle.load(open(file, 'rb')) for file in episode_files]
# shapley_vals, (pred_indices, prey_indices) = compute_shapley_values_for_roles(episodes_data, M=1000)
# print("Predator Shapley (Efficiency):", [shapley_vals['eff'][i] for i in pred_indices])
# print("Prey Shapley (Efficiency):", [shapley_vals['eff'][j] for j in prey_indices])
