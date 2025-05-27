#!/usr/bin/env python3
"""
analysis_pipeline.py

1) Loads per-session event, timestep, and network-state data
2) Builds Boolean event masks & computes pairwise distances
3) Plots distance traces with event rasters
4) Computes and bar‐plots distance stats, prey lifespans, and prey rewards
5) Builds and applies PCA to neural activities, then plots PCs with rasters
6) (Optionally) processes all sessions in parallel into a DataFrame
"""

import json
import logging
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

from group_analysis_utils.helper import parse_agent_roles

# ─── configure logging ───────────────────────────────────────────────────────

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ─── I) DATA LOADING UTILITIES ────────────────────────────────────────────────

def load_data(session: str,
              event_dir: Path,
              ts_dir: Path,
              net_dir: Path
              ) -> Tuple[Dict[int,str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Return (roles, ev_df, ts_df, ns_df) for one session."""
  roles, sources = parse_agent_roles(session)

  ev_path = event_dir / f"{session}_merged.pkl"
  ts_path = next(ts_dir.glob(f"{session}*_info.csv"), None)
  ns_path = next(net_dir.glob(f"{session}*_network_states.pkl"), None)

  if not ev_path.exists():
    log.error(f"Missing events file for {session}: {ev_path}")
    raise FileNotFoundError(ev_path)
  if ts_path is None:
    log.error(f"Missing timesteps file for {session}")
    raise FileNotFoundError(ts_dir)
  if ns_path is None:
    log.error(f"Missing network-states file for {session}")
    raise FileNotFoundError(net_dir)

  ev_df = pd.read_pickle(ev_path)
  ts_df = pd.read_csv(ts_path)
  ns_df = pd.read_pickle(ns_path)

  log.info(f"Loaded session {session}: "
           f"{ev_path.name}, {ts_path.name}, {ns_path.name}")
  return roles, ev_df, ts_df, ns_df


# ─── II) EVENT MASKS & DISTANCES ─────────────────────────────────────────────

def build_event_masks(ev_df: pd.DataFrame,
                      n_steps: int,
                      n_agents: int,
                      events: List[str],
                      episode_length: int = 1000
                      ) -> Dict[str, np.ndarray]:
  """Build Boolean masks of shape (T, n_agents) for each event type."""
  masks = {ev: np.zeros((n_steps, n_agents), dtype=bool) for ev in events}
  for ev in events:
    for ep, entries in enumerate(ev_df[ev].values):
      if not entries:
        continue
      offset = ep * episode_length
      for d in entries:
        try:
          parts = d['participants']
        except KeyError:
          log.warning(f"Missing 'participants' in event {ev} at episode {ep}")
          continue
        if 'time_start' in d:
          s, e = d['time_start'], d['time_end']
          masks[ev][offset+s:offset+e, parts] = True
        else:
          t = d['time']
          masks[ev][offset+t, parts] = True
  return masks


def compute_agent_positions(ts_df: pd.DataFrame,
                            roles: Dict[int,str]
                            ) -> Tuple[np.ndarray, np.ndarray]:
  """Return (positions, death_mask) arrays of shapes (n_agents, T,2) and (T, n_agents)."""
  n_agents = len(roles)
  T = len(ts_df)
  pos_list = []
  death_mat = np.zeros((T, n_agents), dtype=int)

  for aid in range(n_agents):
    death_mat[:, aid] = ts_df[f"death_{aid}"].values
    x_col = f"POSITION_x_{aid}"
    y_col = f"POSITION_y_{aid}"
    if x_col in ts_df and y_col in ts_df:
      pos = ts_df[[x_col, y_col]].values  # (T,2)
    else:
      log.warning(f"Missing position cols for agent {aid} → using NaNs")
      pos = np.full((T,2), np.nan)
    pos_list.append(pos)

  positions = np.stack(pos_list, axis=0)  # (n_agents, T,2)
  return positions, death_mat


def compute_pairwise_distances(positions: np.ndarray,
                               death_mask: np.ndarray
                               ) -> Dict[Tuple[int,int], np.ndarray]:
  """Return dict[(i,j)] → distance time series (length T), with dead masked to NaN."""
  n_agents, T, _ = positions.shape
  out = {}
  for i in range(n_agents):
    for j in range(i+1, n_agents):
      dij = np.linalg.norm(positions[i] - positions[j], axis=1)
      alive = (death_mask[:, i] == 1) & (death_mask[:, j] == 1)
      dij[~alive] = np.nan
      out[(i,j)] = dij
  return out


# ─── III) PLOTTING FUNCTIONS ────────────────────────────────────────────────

def plot_distance_rasters(dist_dict: Dict[Tuple[int,int], np.ndarray],
                          masks: Dict[str, np.ndarray],
                          roles: Dict[int,str],
                          x_lim: Tuple[int,int] = (0,1000),
                          figsize=(10,6)
                          ) -> plt.Figure:
  """One subplot per agent: pair-wise distance traces + event rasters."""
  n_agents = len(roles)
  fig, axes = plt.subplots(n_agents, 1,
                           sharex=True, sharey=True,
                           figsize=(figsize[0], figsize[1]+2*n_agents))
  axes = np.atleast_1d(axes)

  pair_colors = {'predator': 'red', 'prey': 'blue'}
  event_shades = {ev: c for ev, c in zip(masks.keys(),
                                         ['limegreen','orange','purple'])}

  for i, ax in enumerate(axes):
    ax.set_title(f"Agent {i} ({roles[i]})")
    # plot each pair that includes agent i
    for (a,b), dist in dist_dict.items():
      if i in (a,b):
        other = b if i==a else a
        ax.plot(dist, label=f"{roles[a]}–{roles[b]}",
                color=pair_colors[roles[other]], lw=0.8)

    # overlay event rasters
    for ev, mask in masks.items():
      series = mask[:,i].astype(int)
      starts = np.where(np.diff(series, prepend=0)==1)[0]
      ends   = np.where(np.diff(series, prepend=0)==-1)[0]
      if len(ends)<len(starts):
        ends = np.append(ends, len(series))
      for s,e in zip(starts, ends):
        ax.axvspan(s, e,
                   color=event_shades[ev],
                   alpha=0.25,
                   zorder=0)

    ax.set_xlim(x_lim)
    ax.set_ylabel("Distance (px)")
    ax.legend(fontsize='small', ncol=2, loc='upper right')

  axes[-1].set_xlabel("Timestep")
  # single legend for events
  handles = [Patch(facecolor=event_shades[ev], alpha=0.25,
                   label=ev.replace('_',' ').title())
             for ev in masks]
  fig.legend(handles=handles,
             labels=[h.get_label() for h in handles],
             loc='lower center', ncol=len(handles),
             frameon=False)
  plt.tight_layout(rect=[0,0.05,1,1])
  return fig


def plot_distance_stats(dist_dict: Dict[Tuple[int,int], np.ndarray],
                        ev_df: pd.DataFrame,
                        masks: Dict[str, np.ndarray],
                        roles: Dict[int,str],
                        episode_length: int = 1000
                        ) -> plt.Figure:
  """Bar‐plot of mean distances by category during each event."""
  def categorize(i,j,parts):
    r_i,r_j = roles[i], roles[j]
    in_i, in_j = (i in parts),(j in parts)
    pair = (r_i, in_i, r_j, in_j)
    if pair in {('predator',True,'prey',True),('prey',True,'predator',True)}:
      return 'predP–preyP'
    if r_i==r_j=='prey' and in_i and in_j:
      return 'preyP–preyP'
    if r_i==r_j=='prey' and in_i!=in_j:
      return 'preyP–prey¬P'
    if pair in {('predator',False,'prey',False),('prey',False,'predator',False)}:
      return 'pred¬P–prey¬P'
    return None

  rows = []
  for ev in masks:
    for ep, entries in enumerate(ev_df[ev].values):
      if not entries: continue
      for d in entries:
        parts = set(d['participants'])
        if 'time_start' in d:
          s,e = d['time_start'], d['time_end']
          idx = slice(ep*episode_length+s, ep*episode_length+e)
        else:
          t = d['time']; idx = ep*episode_length+t
        for (i,j), dist in dist_dict.items():
          cat = categorize(i,j,parts)
          if cat is None: continue
          rows.append({
            'event': ev,
            'category': cat,
            'distance': np.nanmean(dist[idx])
          })

  df = pd.DataFrame(rows).dropna()
  pivot = (df.groupby(['event','category'])['distance']
           .mean().unstack('category'))
  fig, ax = plt.subplots(figsize=(8,4))
  pivot.plot(kind='bar', ax=ax)
  ax.set_ylabel("Mean distance (px)")
  ax.set_title("Pairwise distances during events")
  ax.legend(title="")
  plt.tight_layout()
  return fig


def plot_lifespan_stats(masks: Dict[str, np.ndarray],
                        death_mask: np.ndarray,
                        roles: Dict[int,str]
                        ) -> plt.Figure:
  """Stacked bar of prey lifespans split by event participation."""
  prey_ids = [i for i,r in roles.items() if r=='prey']
  T = death_mask.shape[0]
  spans = []
  for a in prey_ids:
    alive = (death_mask[:,a]==0)  # 0==alive, 1==dead?
    starts = np.where(np.diff(alive.astype(int), prepend=0)==1)[0]
    ends   = np.where(np.diff(alive.astype(int), prepend=0)==-1)[0]
    if len(ends)<len(starts): ends = np.append(ends,T)
    for k,(s,e) in enumerate(zip(starts,ends)):
      label = 'none'
      for ev,mask in masks.items():
        if mask[s:e,a].any():
          label = ev; break
      spans.append((a,k,label,e-s))

  df = pd.DataFrame(spans, columns=['agent','span','event','length'])
  pivot = df.groupby(['agent','event'])['length'].sum().unstack('event').fillna(0)
  # add gray for none if missing
  if 'none' not in pivot: pivot['none'] = 0
  colors = {**{ev:c for ev,c in zip(masks, ['limegreen','orange','purple'])},
            'none':'lightgray'}
  fig, ax = plt.subplots(figsize=(8,3))
  pivot.plot(kind='bar', stacked=True, ax=ax,
             color=[colors[c] for c in pivot.columns])
  ax.set_ylabel("Timestep count")
  ax.set_title("Prey lifespan by participation")
  ax.legend(title="")
  plt.tight_layout()
  return fig


def plot_reward_stats(ts_df: pd.DataFrame,
                      death_mask: np.ndarray,
                      spans_df: pd.DataFrame,
                      roles: Dict[int,str]
                      ) -> plt.Figure:
  """Mean reward per step for each prey span & event."""
  # assume ts_df has columns reward_{id}
  prey_ids = spans_df['agent'].unique()
  records = []
  for a in prey_ids:
    rewards = ts_df[f"reward_{a}"].values
    alive   = (death_mask[:,a]==0)
    df_a    = spans_df[spans_df['agent']==a]
    for _,r in df_a.iterrows():
      s = r['span']  # not perfect but illustrative
      length = r['length']
      # find segment in alive; naive mapping:
      idx = np.where(alive)[0][s:s+length]
      records.append({
        'agent': a,
        'event': r['event'],
        'reward': rewards[idx].sum()/length
      })
  df = pd.DataFrame(records).dropna()
  pivot = df.groupby(['agent','event'])['reward'].mean().unstack('event').fillna(0)
  fig, ax = plt.subplots(figsize=(8,3))
  pivot.plot(kind='bar', ax=ax,
             color=[ 'limegreen','orange','purple','lightgray'])
  ax.set_ylabel("Mean reward / step")
  ax.set_title("Prey rewards by participation")
  ax.legend(title="")
  plt.tight_layout()
  return fig


def plot_pca_rasters(ns_df: pd.DataFrame,
                               death_mask: np.ndarray,
                               masks: Dict[str, np.ndarray],
                               roles: Dict[int, str],
                               n_components: int = 3,
                               x_lim: Tuple[int, int] = (0, 1000),
                               ) -> plt.Figure:
  """
  For each agent i:
    - Pull columns "hidden_i_*"
    - Fit a PCA(n_components) on that agent's alive‐period data
    - Project all timesteps into PC space
    - Plot PC1…PCn with event raster shading
  """
  T = len(ns_df)
  agents = sorted(roles.keys())
  fig, axes = plt.subplots(len(agents), 1,
                           sharex=True,
                           figsize=(10, 3 * len(agents)))
  axes = np.atleast_1d(axes)

  event_shades = {ev: c for ev, c in zip(masks.keys(),
                                         ['limegreen','orange','purple'])}
  # pick a palette for PCs
  pc_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

  for idx, agent in enumerate(agents):
    ax = axes[idx]
    ax.set_title(f"Agent {agent} PCs ({roles[agent]})")

    # 1) find this agent's hidden cols & build array (T x F_i)
    cols = [c for c in ns_df.columns if c.startswith(f"hidden_{agent}_")]
    # sort by the neuron index
    cols = sorted(cols, key=lambda c: int(c.split('_')[2]))
    X = ns_df[cols].values  # shape (T, F_i)

    # 2) fit PCA on alive rows only
    alive = death_mask[:, agent] == 1  # adjust if 0=death,1=live
    pca = PCA(n_components=min(n_components, X.shape[1]))
    pca.fit(X[alive])

    # 3) project entire trace
    scores = pca.transform(X)  # (T, n_components_agent)

    # 4) plot each PC, mask dead as NaN
    for pc in range(pca.n_components_):
      trace = scores[:, pc].copy()
      trace[~alive] = np.nan
      ax.plot(trace,
              label=f"PC{pc + 1}",
              color=pc_colors[pc],
              lw=0.8)

    # 5) raster-style event shading
    for ev, mask in masks.items():
      series = mask[:, agent].astype(int)
      starts = np.where(np.diff(series, prepend=0) == 1)[0]
      ends = np.where(np.diff(series, prepend=0) == -1)[0]
      if len(ends) < len(starts):
        ends = np.append(ends, T)
      for s, e in zip(starts, ends):
        ax.axvspan(s, e,
                   color=event_shades.get(ev, 'gray'),
                   alpha=0.25,
                   zorder=0)

    ax.set_ylabel("PC value")
    ax.legend(fontsize='small', loc='upper right')

  ax.set_xlim(x_lim)
  axes[-1].set_xlabel("Timestep")
  plt.tight_layout()
  return fig


# ─── IV) SESSION PROCESSOR ─────────────────────────────────────────────────

def process_one(session: str,
                event_dir: Path,
                ts_dir: Path,
                net_dir: Path,
                output_dir: Path
                ) -> None:
  """Load data, compute & save all figures for a single session."""
  roles, ev_df, ts_df, ns_df = load_data(session, event_dir, ts_dir, net_dir)
  positions, death_mask = compute_agent_positions(ts_df, roles)
  T, n_agents = death_mask.shape
  # events = [c for c in ev_df if 'event' in c]
  events = ['apple_cooperation_events', 'distraction_events', 'fence_events']

  masks = build_event_masks(ev_df, T, n_agents, events)

  # distances + rasters
  dist_dict = compute_pairwise_distances(positions, death_mask)
  figs = []
  figs.append(plot_distance_rasters(dist_dict, masks, roles, x_lim=(1000,2000)))
  figs.append(plot_distance_stats(dist_dict, ev_df, masks, roles))
  # lifespans & rewards
  # note: plot_lifespan_stats returns a fig but also relies on internal df
  lifespan_fig = plot_lifespan_stats(masks, death_mask, roles)
  figs.append(lifespan_fig)
  # you could extract spans_df from inside or recompute to pass to plot_reward_stats
  # figs.append(plot_reward_stats(ts_df, death_mask, spans_df, roles))

  # PCA
  figs.append(plot_pca_rasters(ns_df, death_mask, masks, roles, n_components=5,
                               x_lim=(1000,2000)))


  for i, fig in enumerate(figs):
    out = output_dir / f"{session}_fig{i}.png"
    fig.savefig(out, dpi=150)
    plt.show()
    plt.close(fig)
    log.info(f"Wrote {out}")


# ─── V) MAIN & PARALLEL ENTRY ────────────────────────────────────────────────

def main():
  # adjust these paths:
  # event_metric_dir = '/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/analysis_results_merged/'
  # timesteps_dir = '/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/analysis_results/'
  # network_dir = '/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/analysis_results/'

  event_dir = Path("/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/analysis_results_merged/")
  ts_dir    = Path("/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/analysis_results/")
  net_dir   = ts_dir
  output_dir= Path("./figures")
  output_dir.mkdir(exist_ok=True)

  sessions = [p.name.replace("_merged.pkl","")
              for p in event_dir.glob("*.pkl")]

  sessions = sorted(sessions)
  # run in parallel or sequentially:
  use_parallel = False
  if use_parallel:
    Parallel(n_jobs=8)(
      delayed(process_one)(sess, event_dir, ts_dir, net_dir, output_dir)
      for sess in sessions
    )
  else:
    for sess in sessions:
      process_one(sess, event_dir, ts_dir, net_dir, output_dir)


if __name__ == "__main__":
  main()
