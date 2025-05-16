import os
import glob
import re
from typing import Dict, List, Tuple
import concurrent.futures
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from joblib import Parallel, delayed
import json

def parse_agent_roles(filename: str) -> Tuple[Dict[int, str], Dict[int, str]]:
  """
  Extract agent roles and source names from a file path.
  """
  base = os.path.basename(filename).replace('_metrics.pkl', '')
  specs = re.findall(r'([A-Za-z0-9]+_pre[a-z]_\d+)', base)
  roles = {idx: ('predator' if '_pred_' in spec else 'prey')
           for idx, spec in enumerate(specs)}
  sources = {idx: spec for idx, spec in enumerate(specs)}
  return roles, sources


def compute_reward_vectorized(df: pd.DataFrame) -> pd.Series:
  """
  Vectorized reward: predator uses 'catch', prey uses 'apple + 6 * acorn'.
  """
  return np.where(df['role']=='predator', df['catch'], df['apple'] + 6 * df['acorn'])


def process_metrics_file(path: str, move_threshold: float) -> List[Dict]:
  """
  Load one metrics.pkl, filter by move threshold, and extract per-agent records.
  """
  df = pd.read_pickle(path)
  roles, sources = parse_agent_roles(path)
  total_eps = len(df)
  # filter episodes where average move across agents < threshold
  move_cols = [f'move_{i}' for i in roles]
  df = df[df[move_cols].mean(axis=1) >= move_threshold]
  keep_rate = len(df) / total_eps

  records: List[Dict] = []
  for i, role in roles.items():
    prefix = f"_{i}"
    if f'move{prefix}' not in df.columns:
      continue
    rec = {
      'source': sources[i],
      'role': role,
      'group_size': (
        sum(r=='predator' for r in roles.values()),
        sum(r=='prey' for r in roles.values())
      ),
      'episode_keep_rate': keep_rate
    }
    # base metrics
    for m in ['move','rot','death']:
      rec[m] = np.nanmean(df[f'{m}{prefix}'])
    if role == 'prey':
      rec.update({
        'acorn': np.nanmean(df[f'acorn_{i}']),
        'apple': np.nanmean(df[f'apple_{i}']),
        'frac_off': np.nanmean(np.hstack(df[f'frac_off_{i}'])),
        # per-episode good_gathering, then average
        'good_gathering': np.nanmean([
          np.nanmean([
            row[f'in_gather_{i}_to_{p}']
            for p, r in roles.items() if r == 'predator'
            if f'in_gather_{i}_to_{p}' in row.index
          ])
          for _, row in df.iterrows()
        ]),
        'fence_count': np.nanmean([
          sum(1 for e in row['fencing_events'] if e['prey'] == i)
          for _, row in df.iterrows()
        ]),
        'help_count': np.nanmean([
          sum(len(e['helpers']) for e in row['fencing_events'] if e['prey'] == i)
          for _, row in df.iterrows()
        ]),
        'dist_to_pred': np.nanmean([
          np.nanmean([
            row[f'dist_{p}to{i}']
            for p, r in roles.items() if r == 'predator'
            if f'dist_{p}to{i}' in row.index
          ])
          for _, row in df.iterrows()
        ])
      })
    else:
      rec.update({
        'catch': np.nanmean(df[f'catch_{i}']),
        'invalid_interact_count': np.nanmean([
          sum(1 for e in row['invalid_events'] if e['predator'] == i)
          for _, row in df.iterrows()
        ]),
        'interact_count': np.nanmean([
          len(row['invalid_events'])
          for _, row in df.iterrows()
        ])
      })
    records.append(rec)
  return records


def load_and_process_metrics(data_dir: str, move_threshold: float = 50) -> pd.DataFrame:
  """
  Parallel load and preprocess all *_metrics.pkl in data_dir.
  """
  paths = glob.glob(os.path.join(data_dir, '*_metrics.pkl'))
  fn = partial(process_metrics_file, move_threshold=move_threshold)
  all_records: List[Dict] = []
  with concurrent.futures.ProcessPoolExecutor() as executor:
    for recs in executor.map(fn, paths):
      all_records.extend(recs)

  agent_df = pd.DataFrame(all_records)
  agent_df['reward'] = compute_reward_vectorized(agent_df)
  agent_df['reward_ref'] = agent_df.apply(
    lambda row: row['reward'] / (row['group_size'][1] if row['role']=='predator' else 1),
    axis=1
  )
  agent_df['group_label'] = agent_df['group_size'].apply(lambda g: f"{g[0]}p,{g[1]}r")
  return agent_df


def normalize_metrics(df: pd.DataFrame, metrics: Dict[str, List[str]], quantile: float = 0.95) -> pd.DataFrame:
  """
  Create _norm columns by dividing each metric by its role-specific quantile.
  """
  normed = df.copy()
  for role, mlist in metrics.items():
    subset = normed[normed['role']==role]
    for m in mlist:
      if m not in normed:
        continue
      q = subset[m].quantile(quantile)
      denom = q if q>0 else 1
      normed[f'{m}_norm'] = normed[m] / denom
      print(f"Normalized {m} by {quantile*100:.0f}th percentile: {denom:.3f}")
  return normed


def pivot_rewards(df: pd.DataFrame) -> pd.DataFrame:
  """
  Pivot mean reward_ref by source and group_label.
  """
  return df.pivot_table(
    index='source',
    columns='group_label',
    values='reward_ref',
    aggfunc='mean',
    fill_value=0
  )

def generate_sorted_sources(
    df: pd.DataFrame,
    reward_pivot: pd.DataFrame,
    sort_option: str = 'sort_by_reward',
    role: str = None
) -> List[str]:
    """
    Flattens sources into a single list according to sort_option:
      * 'sort_by_reward'
      * 'sort_by_source'
      * 'sort_by_source_and_reward'
      * 'sorted_by_source_and_reward_for_better_agents'
    Optionally filters to a specific role before sorting.
    """
    # optionally filter to one role
    sub = df[df['role'] == role] if role else df
    agent_rewards = sub.groupby('source')['reward_ref'].mean()

    def split(src: str) -> Tuple[str,int]:
        m = re.match(r'(.+?_pre[ay])_(\d+)$', src)
        return (m.group(1), int(m.group(2))) if m else (src, 0)

    # special case: only “better” agents
    if sort_option == 'sorted_by_source_and_reward_for_better_agents':
        _, better = get_better_agents_from_pivot(reward_pivot, role)
        # build per‐prefix lists
        groups = {
            prefix: [f"{prefix}_{i}" for i in idxs]
            for prefix, idxs in better.items()
        }
        # filter out any missing sources
        groups = {g: [s for s in sl if s in agent_rewards.index] for g, sl in groups.items()}
        # order prefixes by cumulative better‐agent reward
        cum = {g: agent_rewards.loc[sl].sum() for g, sl in groups.items()}
        prefixes = sorted(groups, key=lambda g: cum[g], reverse=True)

        out = []
        for g in prefixes:
            # within‐group sort by descending individual reward
            out.extend(sorted(groups[g], key=lambda s: agent_rewards[s], reverse=True))
        return out

    # the other three options all group by prefix
    if sort_option == 'sort_by_reward':
        return agent_rewards.sort_values(ascending=False).index.tolist()

    # build prefix→sources map
    groups: Dict[str, List[str]] = {}
    for src in agent_rewards.index:
        prefix, _ = split(src)
        groups.setdefault(prefix, []).append(src)

    if sort_option == 'sort_by_source':
        prefixes = sorted(groups.keys())
    else:  # sort_by_source_and_reward
        cum = {g: agent_rewards.loc[groups[g]].sum() for g in groups}
        prefixes = sorted(groups.keys(), key=lambda g: cum[g], reverse=True)

    out = []
    for g in prefixes:
        sl = groups[g]
        if sort_option == 'sort_by_source':
            out.extend(sorted(sl, key=lambda s: split(s)[1]))
        else:
            out.extend(sorted(sl, key=lambda s: agent_rewards[s], reverse=True))
    return out


def generate_source_grid(df: pd.DataFrame, reward_pivot: pd.DataFrame, sort_option: str = 'sort_by_reward', role: str = None) -> List[Tuple[str,...]]:
  """Build a grid of source tuples for a specific role."""
  sorted_list = generate_sorted_sources(df, reward_pivot, sort_option, role)
  # group into rows by prefix
  def split(src: str): m = re.match(r'(.+?_pre[dy])_(\d+)$', src); return (m.group(1), int(m.group(2))) if m else (src,0)
  if 'source' in sort_option:
    groups: Dict[str,List[str]] = {}
    for src in sorted_list:
      prefix, _ = split(src)
      groups.setdefault(prefix, []).append(src)
  else:
    # We simply arrange them by rows and columns
    groups: Dict[str,List[str]] = {}
    n_cols = np.ceil(np.sqrt(len(sorted_list))).astype(int)
    n_rows = np.ceil(len(sorted_list) / n_cols).astype(int)
    for i, src in enumerate(sorted_list):
      row = i // n_cols
      col = i % n_cols
      groups.setdefault(row, []).append(src)
  grid = [tuple(groups[p]) for p in groups]
  return grid


def plot_boxplots(df: pd.DataFrame, metrics: Dict[str,List[str]], reward_pivot: pd.DataFrame,
                  sort_option: str = 'sort_by_reward', figure_dir: str = './'):
  """
  Parallel boxplots: one job per (role,metric)
  """
  tasks = []
  for role, mlist in metrics.items():
    tasks.extend((role, m) for m in ['reward'] + [x for x in mlist if x in df.columns])

  def _plot_task(role: str, m: str):
    grid = generate_source_grid(df, reward_pivot, sort_option, role)
    n_rows = len(grid)
    n_cols = max(len(r) for r in grid)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols,4*n_rows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(n_rows, n_cols)
    for i, row in enumerate(grid):
      full_labels = sorted(set(df[df['source'].isin(row)]['group_label']), key=lambda s: tuple(map(int, re.findall(r'(\d+)', s))))
      for j, src in enumerate(row):
        ax = axes[i,j]
        sub = df[(df['role']==role)&(df['source']==src)]
        data = [sub[sub['group_label']==lbl][m].dropna().values for lbl in full_labels]
        counts = [len(d) for d in data]
        labels_n = [f"{lbl}\nn={cnt}" for lbl,cnt in zip(full_labels,counts)]
        ax.boxplot(data, labels=labels_n)
        r1 = reward_pivot.get('1p,1r',{}).get(src,0)
        r2 = reward_pivot.get('2p,4r',{}).get(src,0)
        if r1 >= 2 * r2:
          label, color = "better at single", "blue"
        elif r2 >= 2 * r1:
          label, color = "better at group", "orange"
        else:
          label, color = "", "black"
        title = src + (f"\n{label}" if label else "")
        ax.set_title(title, color=color)
        ax.set_xlabel('Group size (predator, prey)')
        ax.set_ylabel(m)
      for j in range(len(row), n_cols):
        axes[i,j].axis('off')
    fig.suptitle(f'Distribution of {role} {m}', fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(os.path.join(figure_dir, f'{role}_{m}_{sort_option}.png'), dpi=300, bbox_inches='tight')
    plt.show()

  Parallel(n_jobs=-1)(delayed(_plot_task)(role, m) for role, m in tasks)


def plot_radars(df: pd.DataFrame, metrics: Dict[str,List[str]], reward_pivot: pd.DataFrame,
                sort_option: str = 'sort_by_reward', figure_dir: str = './'):
  for role, mlist in metrics.items():
    grid = generate_source_grid(df, reward_pivot, sort_option, role)
    n_rows, n_cols = len(grid), max(len(r) for r in grid)
    fig, axes = plt.subplots(n_rows, n_cols, subplot_kw={'projection':'polar'}, figsize=(3.5*n_cols,4*n_rows))
    axes = np.array(axes).reshape(n_rows,n_cols)
    for i,row in enumerate(grid):
      for j,src in enumerate(row):
        ax=axes[i,j]
        sub=df[(df['role']==role)&(df['source']==src)]
        norm_cols=[f'{m}_norm' for m in mlist if f'{m}_norm' in sub]
        if not norm_cols: ax.axis('off'); continue
        vals=sub[norm_cols].mean().fillna(0).tolist()
        angs=np.linspace(0,2*pi,len(vals),endpoint=False).tolist()
        vals+=vals[:1]; angs+=angs[:1]
        ax.plot(angs,vals,marker='o'); ax.fill(angs,vals,alpha=0.25)
        ax.set_xticks(angs[:-1]); ax.set_xticklabels([c.replace('_norm','') for c in norm_cols])
        r1 = reward_pivot.get('1p,1r', {}).get(src, 0)
        r2 = reward_pivot.get('2p,4r', {}).get(src, 0)
        if r1 >= 2 * r2:
          label, color = "better at single", "blue"
        elif r2 >= 2 * r1:
          label, color = "better at group", "orange"
        else:
          label, color = "", "black"
        title = src + (f"\n{label}" if label else "")
        ax.set_title(title, color=color)

      for j in range(len(row),n_cols): axes[i,j].axis('off')
    fig.suptitle(f'{role.capitalize()} Normalized Metrics Radar', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(figure_dir, f'{role}_radar_{sort_option}.png'), dpi=300, bbox_inches='tight')
    plt.show()


# def get_better_agents(df: pd.DataFrame) -> Dict[str, List[int]]:
#   """
#   Identify “better” agents (both predators and preys) whose reward_ref is
#   at least the global median for their role OR at least the median of their
#   own source‐group. Returns a dict mapping each source‐prefix to a sorted
#   list of agent indices.
#   """
#   better: Dict[str, List[int]] = {}
#   for role in ['predator', 'prey']:
#     sub = df[df['role'] == role]
#     if sub.empty:
#       continue
#     global_med = sub['reward_ref'].mean()
#
#     def split(src: str) -> Tuple[str, int]:
#       m = re.match(r'(.+?_pre[dy])_(\d+)$', src)
#       return (m.group(1), int(m.group(2))) if m else (src, -1)
#
#     # group by common prefix
#     for prefix, group in sub.groupby(sub['source'].apply(lambda s: split(s)[0])):
#       # group_med = np.nanpercentile(group['reward_ref'], 75)
#       group_med = group['reward_ref'].median()
#       # agent is better if ≥ global or ≥ group median
#       cond = (group['reward_ref'] >= global_med) | (group['reward_ref'] >= group_med)
#
#       idxs = []
#       for src in group.loc[cond, 'source']:
#         _, idx = split(src)
#         if idx >= 0:
#           idxs.append(idx)
#
#       better[prefix] = sorted(list(set(idxs)))
#
#   return better

def get_better_agents_from_pivot(
    reward_pivot: pd.DataFrame,
    role: str
) -> (List, Dict[str, List[int]]):
  """
  Using the reward_pivot (index=source strings, columns=group_labels),
  identify “better” agents of a given role whose overall mean reward_ref
  (across all group_labels) is ≥ the global median for that role OR ≥
  the median within their source‐prefix group.

  Returns a dict mapping each source‐prefix (e.g. "OP..._pred") to a sorted
  list of adopted agent indices.
  """
  # 1) Filter sources by role
  #    assume source strings contain "_pred_" or "_prey_"
  mask = reward_pivot.index.str.contains(f"_pre{'d' if role == 'predator' else 'y'}_")
  sub = reward_pivot.loc[mask]

  # 2) compute each source's overall mean
  overall = sub.mean(axis=1)

  # 3) global median for this role
  global_med = overall.median()

  # 4) group by prefix: split "PREFIX_i"
  def split(src: str) -> Tuple[str, int]:
    m = re.match(r"(.+?_pre[dy])_(\d+)$", src)
    return (m.group(1), int(m.group(2))) if m else (src, -1)

  groups: Dict[str, List[str]] = {}
  for src in overall.index:
    prefix, _ = split(src)
    groups.setdefault(prefix, []).append(src)

  # 5) per‐prefix median
  prefix_med: Dict[str, float] = {
    p: overall.loc[srcs].median()
    for p, srcs in groups.items()
  }

  # 6) select adopted
  adopted: Dict[str, List[int]] = {}
  for prefix, srcs in groups.items():
    idxs: List[int] = []
    for src in srcs:
      val = overall.at[src]
      if val >= global_med or val >= prefix_med[prefix]:
        _, i = split(src)
        if i >= 0:
          idxs.append(i)
    adopted[prefix] = sorted(idxs)

  adopted_list = [key + '_' + str(i) for key, idxs in adopted.items() for i in idxs]
  return adopted_list, adopted


if __name__ == '__main__':
  DATA_DIR = '/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/analysis_results'
  METRICS = {
    'prey': ['move', 'rot', 'acorn', 'apple', 'frac_off', 'death',
             'good_gathering', 'fence_count', 'help_count', 'dist_to_pred'],
    'predator': ['move', 'rot', 'catch', 'death',
                 'interact_count', 'invalid_interact_count']
  }

  agent_df = load_and_process_metrics(DATA_DIR)
  agent_df = normalize_metrics(agent_df, METRICS, quantile=0.8)
  reward_pivot = pivot_rewards(agent_df)
  # Now, we limit the size of normed such that it will not exceed 1.5, only for columns with '_norm'
  normed = agent_df.copy()
  for col in normed.columns:
    if '_norm' in col:
      normed[col] = np.clip(normed[col], 0, 1.1)


  if not os.path.exists('./figures_sort_by_reward'):
    os.makedirs('./figures_sort_by_reward')
  plot_radars(normed, METRICS, reward_pivot, figure_dir='./figures_sort_by_reward')
  plot_boxplots(agent_df, METRICS, reward_pivot, figure_dir='./figures_sort_by_reward')

  # Plot in the other two ways
  if not os.path.exists('./figures_sort_by_source_and_reward'):
    os.makedirs('./figures_sort_by_source_and_reward')
  plot_boxplots(agent_df, METRICS, reward_pivot, sort_option='sort_by_source_and_reward', figure_dir='./figures_sort_by_source_and_reward')
  plot_radars(agent_df, METRICS, reward_pivot, sort_option='sort_by_source_and_reward', figure_dir='./figures_sort_by_source_and_reward')

  if not os.path.exists('./figures_sort_by_source'):
    os.makedirs('./figures_sort_by_source')
  plot_boxplots(agent_df, METRICS, reward_pivot, sort_option='sort_by_source', figure_dir='./figures_sort_by_source')
  plot_radars(agent_df, METRICS, reward_pivot, sort_option='sort_by_source', figure_dir='./figures_sort_by_source')

  # if not os.path.exists('./figures_sorted_by_source_and_reward_for_better_agents'):
  #   os.makedirs('./figures_sorted_by_source_and_reward_for_better_agents')
  # better_agents = get_better_agents(agent_df)
  # plot_radars(agent_df, METRICS, reward_pivot, sort_option='sorted_by_source_and_reward_for_better_agents', figure_dir='./figures_sorted_by_source_and_reward_for_better_agents')
  # plot_boxplots(agent_df, METRICS, reward_pivot, sort_option='sorted_by_source_and_reward_for_better_agents', figure_dir='./figures_sorted_by_source_and_reward_for_better_agents')

  if not os.path.exists('./figures_sorted_by_reward_for_better_agents'):
    os.makedirs('./figures_sorted_by_reward_for_better_agents')
  better_pred_list, better_pred = get_better_agents_from_pivot(reward_pivot, 'predator')
  better_prey_list, better_prey = get_better_agents_from_pivot(reward_pivot, 'prey')
  better_agents = sorted(set(better_pred_list + better_prey_list))
  better_agent_df = agent_df[agent_df['source'].isin(better_agents)]
  better_reward_pivot = reward_pivot[reward_pivot.index.isin(better_agents)]
  plot_radars(better_agent_df, METRICS, better_reward_pivot, sort_option='sorted_by_reward', figure_dir='./figures_sorted_by_reward_for_better_agents')
  plot_boxplots(better_agent_df, METRICS, better_reward_pivot, sort_option='sorted_by_reward', figure_dir='./figures_sorted_by_reward_for_better_agents')


