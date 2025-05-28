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

import warnings
warnings.filterwarnings("ignore")

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# ─── configure logging ───────────────────────────────────────────────────────

logging.basicConfig(
    filename="analysis.log",    # <— log file
    filemode="w",               # overwrite each run
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger(__name__)


# ─── I) DATA LOADING UTILITIES ────────────────────────────────────────────────

def load_data(session: str,
              event_dir: Path,
              ts_dir: Path,
              net_dir: Path,
              agent_roles: Dict[int, str] = None
              ) -> Tuple[Dict[int,str], Dict[int,str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Return (roles, ev_df, ts_df, ns_df) for one session."""
  if agent_roles is None:
    roles, sources = parse_agent_roles(session)
  else:
    roles = agent_roles
    sources = {i: f"{session}_agent_{i}" for i in roles}
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
  return roles, sources, ev_df, ts_df, ns_df


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


def plot_lifespan_stats(ev_df: pd.DataFrame,
                        death_mask: np.ndarray,
                        roles: Dict[int,str],
                        events: List[str],
                        episode_length: int = 1000
                        ) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Compute & plot total alive‐duration (lifespan) per agent × event variant,
    where variants are:
      - 'apple_cooperation_events'
      - 'distraction_events_helper', 'distraction_events_beneficiary'
      - 'fence_events_helper', 'fence_events_beneficiary'
      - 'none' (no event)
    Returns:
      fig     – bar‐plot Figure
      span_df – DataFrame with columns [agent, event, length]
    """
    prey_ids = [i for i,r in roles.items() if r=='prey']
    T = death_mask.shape[0]
    variants = [
        'apple_cooperation_events',
        'distraction_events_helper','distraction_events_beneficiary',
        'fence_events_helper','fence_events_beneficiary',
        'none'
    ]

    # first, build a per‐agent/event boolean mask
    masks = {agent: {var: np.zeros(T, bool) for var in variants}
             for agent in prey_ids}

    for ev in events:
        for ep, entries in enumerate(ev_df[ev].values):
            offset = ep*episode_length
            for d in entries:
                t0 = d.get('time_start', d.get('time'))
                t1 = d.get('time_end',   t0+1)
                idx = slice(offset+t0, offset+t1)
                # apple
                if ev=='apple_cooperation_events':
                    for a in d['participants']:
                        masks[a][ev][idx] = True
                else:
                    # distraction/fence: helper vs beneficiary
                    for role_key in ('helper','beneficiary'):
                        inds = d.get(role_key, [])
                        if isinstance(inds,int):
                            inds = [inds]
                        for a in inds:
                            masks[a][f"{ev}_{role_key}"][idx] = True

    records = []
    for a in prey_ids:
        alive = death_mask[:,a] == 1
        any_mask = np.zeros(T,bool)
        # accumulate any event
        for var in variants:
            if var!='none':
                any_mask |= masks[a][var]

        for var in variants:
            if var=='none':
                m = (~any_mask) & alive
            else:
                m = masks[a][var] & alive
            total_len = int(m.sum())
            records.append({'agent':a, 'event':var, 'length': total_len})

    span_df = pd.DataFrame(records)

    # pivot & plot
    pivot = span_df.pivot(index='agent', columns='event', values='length').fillna(0)
    fig, ax = plt.subplots(figsize=(10,4))
    pivot.plot(kind='bar', ax=ax)
    ax.set_ylabel("Total timesteps alive")
    ax.set_title("Prey lifespan by event variant")
    ax.legend(title="", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()

    return fig, span_df


def plot_reward_stats(ts_df: pd.DataFrame,
                      death_mask: np.ndarray,
                      ev_df: pd.DataFrame,
                      roles: Dict[int,str],
                      events: List[str],
                      reward_prefix: str = 'reward_',
                      episode_length: int = 1000
                      ) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    For each agent and each event variant, compute mean reward per alive step:
      • apple_cooperation_events
      • distraction_events_helper / _beneficiary
      • fence_events_helper / _beneficiary
      • plus 'none' (no event)
    Returns the bar‐plot Figure and a long‐form DataFrame [agent,event,reward].
    """
    T, n_agents = death_mask.shape

    # precompute global any‐event mask (1D per agent)
    any_event = {agent: np.zeros(T, bool) for agent in roles}
    for ev in events:
        for ep, entries in enumerate(ev_df[ev].values):
            offset = ep * episode_length
            for d in entries:
                t0 = d.get('time_start', d.get('time'))
                t1 = d.get('time_end',   t0+1)
                for a in d['participants']:
                    idx = slice(offset+t0, offset+t1)
                    any_event[a][idx] = True

    records = []
    for agent in sorted(roles):
        alive = (death_mask[:, agent] == 1)
        col   = f"{reward_prefix}{agent}"
        if col not in ts_df.columns:
            continue
        rewards = ts_df[col].values

        # 1) apple_coop base
        ev = 'apple_cooperation_events'
        mask = np.zeros(T, bool)
        for ep, entries in enumerate(ev_df[ev].values):
            offset = ep * episode_length
            for d in entries:
                if agent not in d['participants']:
                    continue
                t0 = d.get('time_start', d.get('time'))
                t1 = d.get('time_end',   t0+1)
                mask[offset+t0 : offset+t1] = True
        mask &= alive
        records.append({'agent':agent, 'event':ev, 'reward': np.nanmean(rewards[mask]) if mask.any() else np.nan})

        # 2) distraction & fence, helper/beneficiary
        for ev in ['distraction_events','fence_events']:
            for role_key in ['helper','beneficiary']:
                mask = np.zeros(T, bool)
                for ep, entries in enumerate(ev_df[ev].values):
                    offset = ep * episode_length
                    for d in entries:
                        inds = d.get(role_key, [])
                        if isinstance(inds, int):
                            inds = [inds]
                        if agent not in inds:
                            continue
                        t0 = d.get('time_start', d.get('time'))
                        t1 = d.get('time_end',   t0+1)
                        mask[offset+t0 : offset+t1] = True
                mask &= alive
                records.append({
                    'agent': agent,
                    'event': f"{ev}_{role_key}",
                    'reward': np.nanmean(rewards[mask]) if mask.any() else np.nan
                })

        # 3) none
        none_mask = (~any_event[agent]) & alive
        records.append({'agent':agent, 'event':'none',
                        'reward': np.nanmean(rewards[none_mask]) if none_mask.any() else np.nan})

    reward_df = pd.DataFrame(records)

    # pivot & plot
    pivot = reward_df.pivot(index='agent', columns='event', values='reward').fillna(0)
    fig, ax = plt.subplots(figsize=(10,4))
    pivot.plot(kind='bar', ax=ax)
    ax.set_ylabel("Mean reward / step")
    ax.set_title("Reward per agent by event variant")
    ax.legend(title="", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()

    return fig, reward_df


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

def classify_multiple_events_per_agent(
    ns_df: pd.DataFrame,
    death_mask: np.ndarray,
    ev_df: pd.DataFrame,
    events: List[str],
    roles: Dict[int, str],
    mode: str = 'event_vs_rest',
    episode_length: int = 1000,
    n_components: int = 3
) -> pd.DataFrame:
  """
  For each agent and each event in `events`, do:
    - build y[t] = 1 at event onset (time_start or time)
    - negatives:
       * mode='event_vs_none': times when no event is happening for that agent
       * mode='event_vs_rest': times when neither *other* events or none happen (i.e. ¬this_event)
       * mode='event_vs_event': times when other events happen (¬this_event)
    - leave-one-episode-out: for each fold leave one ep out for test
    - within each fold:
        • fit PCA(n_components) on TRAINING *alive* data
        • project both train/test into PC space
        • train logistic regression on train & score ROC AUC on test
    - return mean ROC AUC for each (agent, event)
  """
  T = len(ns_df)
  n_eps = len(ev_df[events[0]])
  # 1D group labels for LOGO
  groups = np.repeat(np.arange(n_eps), episode_length)[:T]

  # precompute event‐onset masks & any‐event mask
  onset_masks = {}
  any_event = np.zeros(T, dtype=bool)
  for ev in events:
    m = np.zeros(T, dtype=bool)
    for ep, entries in enumerate(ev_df[ev].values):
      for d in entries:
        t0 = d.get('time_start', d.get('time'))
        idx = ep * episode_length + t0
        if idx < T:
          m[idx] = True
    onset_masks[ev] = m
    any_event |= m

  logo = LeaveOneGroupOut()
  records = []

  for agent in sorted(roles):
    # grab this agent's neural data
    cols = sorted([c for c in ns_df.columns if c.startswith(f"hidden_{agent}_")],
                  key=lambda c: int(c.split('_')[2]))
    X_full = ns_df[cols].values  # (T, F_i)
    alive = death_mask[:, agent] == 1  # True = alive

    for ev in events:
      y = onset_masks[ev].astype(int)
      # define negative mask
      if mode == 'event_vs_none':
        neg_mask = (~any_event) & alive
      elif mode == 'event_vs_rest':
        neg_mask = (~onset_masks[ev]) & alive
      elif mode == 'event_vs_event':
        # only onset times of other events
        others = any_event & (~onset_masks[ev])
        neg_mask = others & alive
      else:
        raise ValueError("mode must be 'event_vs_none', 'event_vs_rest' or 'event_vs_event'")

      aucs = []
      # LOGO folds
      for train_idx, test_idx in logo.split(X_full, y, groups):
        # pick only alive+labelled timepoints
        tr_sel = train_idx[(y[train_idx] == 1) | (neg_mask[train_idx])]
        te_sel = test_idx[(y[test_idx] == 1) | (neg_mask[test_idx])]

        # need at least one positive and one negative in train & test
        if y[te_sel].sum() == 0 or len(np.unique(y[tr_sel])) < 2:
          continue

        # fit PCA on train-alive
        pca = PCA(n_components=min(n_components, X_full.shape[1]))
        pca.fit(X_full[tr_sel])

        scores = pca.transform(X_full)  # (T, n_comp)
        X_tr, X_te = scores[tr_sel], scores[te_sel]
        y_tr, y_te = y[tr_sel], y[te_sel]

        clf = LogisticRegression(solver='liblinear')
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, probs))

      records.append({
        'agent': agent,
        'event': ev,
        'mode': mode,
        'roc_auc': float(np.nanmean(aucs)) if aucs else np.nan
      })

  return pd.DataFrame(records).pivot_table(
    index='agent', columns=['event', 'mode'], values='roc_auc'
  )


def classify_event_onset(scores: np.ndarray,
                         ev_df: pd.DataFrame,
                         event: str,
                         episode_length: int = 1000
                         ) -> float:
  """
  Leave-one-episode-out ROC AUC for predicting the *onset* of `event`
  from PCA scores.

  scores: (T, n_components) array for one agent.
  ev_df[event]: list-of-lists of event dicts, one entry per episode.
  """
  T, n_comp = scores.shape
  # build y (1 at each time_start, else 0)
  y = np.zeros(T, dtype=int)
  n_eps = len(ev_df[event])
  for ep, entries in enumerate(ev_df[event].values):
    for d in entries:
      t = d.get('time_start', d.get('time'))
      y[ep * episode_length + t] = 1

  # group labels for leave-one-out
  groups = np.repeat(np.arange(n_eps), episode_length)

  logo = LeaveOneGroupOut()
  aucs = []
  for train, test in logo.split(scores, y, groups):
    # skip folds with no positives in test
    if y[test].sum() == 0:
      continue
    clf = LogisticRegression(solver='liblinear')
    clf.fit(scores[train], y[train])
    probs = clf.predict_proba(scores[test])[:, 1]
    aucs.append(roc_auc_score(y[test], probs))

  return float(np.mean(aucs)) if aucs else np.nan


def find_distance_responsive_neurons(ns_df: pd.DataFrame,
                                     positions: np.ndarray,
                                     death_mask: np.ndarray,
                                     roles: Dict[int, str]
                                     ) -> pd.DataFrame:
  """
  For each agent, compute:
    - distance to nearest predator (d_pred[t])
    - distance to nearest other prey    (d_prey[t])
  Then for each hidden_{agent}_{neuron} time series,
  correlate (alive periods only) with d_pred and with d_prey.
  Returns a df with, per agent:
    best_neuron_pred, corr_pred, best_neuron_prey, corr_prey
  """
  results = {}
  T = positions.shape[1]
  agents = sorted(roles.keys())
  scaler = StandardScaler()

  for agent in agents:
    # build distance time-series
    pos_a = positions[agent]  # (T,2)
    # all predators / other preys
    preds = [j for j, r in roles.items() if r == 'predator' and j != agent]
    other_preys = [j for j, r in roles.items() if r == 'prey' and j != agent]

    # stack distances: shape (len(preds),T)

    if preds:
      d_pred_mat = np.stack([np.linalg.norm(pos_a - positions[p], axis=1)
                             for p in preds], axis=0)
      d_pred = d_pred_mat.min(axis=0)
    else:
      d_pred = np.full(T, np.nan)

    if other_preys:
      d_prey_mat = np.stack([np.linalg.norm(pos_a - positions[p], axis=1)
                             for p in other_preys], axis=0)
      d_prey = d_prey_mat.min(axis=0)
    else:
      d_prey = np.full(T, np.nan)

    # get this agent's neural data
    cols = sorted([c for c in ns_df if c.startswith(f"hidden_{agent}_")],
                  key=lambda c: int(c.split('_')[2]))
    X = ns_df[cols].values  # (T, F_i)
    alive = death_mask[:, agent] == 1  # assume 0=alive,1=dead

    # correlate
    corrs_pred = []
    corrs_prey = []
    for i in range(X.shape[1]):
      xi = X[:, i]
      # only where alive & finite
      mask = alive & np.isfinite(d_pred) & np.isfinite(xi)

      if mask.sum() > 2:
        xi_zscore = scaler.fit_transform(xi[mask].reshape(-1, 1)).flatten()
        d_pred_zscore = scaler.transform(d_pred[mask].reshape(-1, 1)).flatten()
        corrs_pred.append(np.corrcoef(xi_zscore, d_pred_zscore)[0, 1])
      else:
        corrs_pred.append(0)
      mask2 = alive & np.isfinite(d_prey) & np.isfinite(xi)
      if mask2.sum() > 2:
        xi_zscore = scaler.fit_transform(xi[mask2].reshape(-1, 1)).flatten()
        d_prey_zscore = scaler.transform(d_prey[mask2].reshape(-1, 1)).flatten()
        # compute correlation
        corrs_prey.append(np.corrcoef(xi_zscore, d_prey_zscore)[0, 1])
        # corrs_prey.append(np.corrcoef(xi[mask2], d_prey[mask2])[0, 1])
      else:
        corrs_prey.append(0)

    # pick best by absolute corr
    best_i_pred = int(np.nanargmax(np.abs(corrs_pred)))
    best_i_prey = int(np.nanargmax(np.abs(corrs_prey)))

    results[agent] = {
      'best_neuron_pred': cols[best_i_pred],
      'corr_pred': corrs_pred[best_i_pred],
      'best_neuron_prey': cols[best_i_prey],
      'corr_prey': corrs_prey[best_i_prey]
    }

  return pd.DataFrame.from_dict(results, orient='index')

def process_one(session: str,
                event_dir: Path,
                ts_dir: Path,
                net_dir: Path,
                figure_dir: Path,
                output_dir: Path,
                ignore_existing: bool = False
                ) -> None:
  """Load data, compute & save all figures and a per-agent summary for one session."""
  # ─── I) LOAD & PREPARE ───────────────────────────────────────────
  roles, sources, ev_df, ts_df, ns_df = load_data(session,
                                                  event_dir,
                                                  ts_dir,
                                                  net_dir)

  summary_out = output_dir / f"{session}_event_stats_and_decode.pkl"
  if ignore_existing and summary_out.exists():
    log.info(f"Skipping {session}, summary already exists: {summary_out}")
    return
  try:
    positions, death_mask = compute_agent_positions(ts_df, roles)
    T, n_agents = death_mask.shape
    episode_length = 1000
    events = ['apple_cooperation_events',
              'distraction_events',
              'fence_events']

    # boolean masks & distances
    ev_masks = build_event_masks(ev_df, T, n_agents, events, episode_length)
    dist_dict = compute_pairwise_distances(positions, death_mask)

    # ─── II) GENERATE & SAVE FIGURES ────────────────────────────────
    figs = []
    figs.append(plot_distance_rasters(dist_dict, ev_masks, roles,
                                      x_lim=(1000, 2000)))
    figs.append(plot_distance_stats(dist_dict, ev_df, ev_masks, roles))

    life_fig, span_df = plot_lifespan_stats(
      ev_df=ev_df,
      death_mask=death_mask,
      roles=roles,
      events=events,
      episode_length=episode_length
    )
    figs.append(life_fig)

    reward_fig, reward_df = plot_reward_stats(
      ts_df=ts_df,
      death_mask=death_mask,
      ev_df=ev_df,
      roles=roles,
      events=events,
      reward_prefix='rewards_',
      episode_length=episode_length
    )
    figs.append(reward_fig)

    pca_fig = plot_pca_rasters(ns_df, death_mask, ev_masks, roles,
                               n_components=5,
                               x_lim=(1000, 2000))
    figs.append(pca_fig)

    for idx, fig in enumerate(figs):
      outpath = figure_dir / f"{session}_fig{idx}.png"
      fig.savefig(outpath, dpi=150)
      plt.close(fig)

    # ─── III) CLASSIFIERS & CORRELATIONS ────────────────────────────
    auc_vs_rest = classify_multiple_events_per_agent(
      ns_df, death_mask, ev_df, events, roles,
      mode='event_vs_rest',
      episode_length=episode_length,
      n_components=3
    )
    auc_vs_event = classify_multiple_events_per_agent(
      ns_df, death_mask, ev_df, events, roles,
      mode='event_vs_event',
      episode_length=episode_length,
      n_components=3
    )
    auc_vs_none = classify_multiple_events_per_agent(
      ns_df, death_mask, ev_df, events, roles,
      mode='event_vs_none',
      episode_length=episode_length,
      n_components=3
    )
    corr_df = find_distance_responsive_neurons(ns_df,
                                               positions,
                                               death_mask,
                                               roles)

    # ─── IV) BUILD & SAVE SUMMARY ────────────────────────────────────
    rows = []
    # total life (alive timesteps)
    total_life = {a: int((death_mask[:, a] == 1).sum()) for a in roles}

    # Determine all variants present in span_df / reward_df
    variants = sorted(span_df['event'].unique())

    for a in sorted(roles):
      row = {
        'trial_name': session,
        'agent': a,
        'role': roles[a],
        'source': sources[a],
        'lifespan_total': total_life[a],
        'top_neu_cor_pred_dist': corr_df.loc[a, 'best_neuron_pred'],
        'top_neu_cor_prey_dist': corr_df.loc[a, 'best_neuron_prey'],
        'top_neu_cor_pred_score': corr_df.loc[a, 'corr_pred'],
        'top_neu_cor_prey_score': corr_df.loc[a, 'corr_prey']
      }

      # ROC AUCs
      for ev in events:
        short = ev.split('_')[0]
        row[f"{short}_vs_rest"] = auc_vs_rest.loc[a, (ev, 'event_vs_rest')]
        row[f"{short}_vs_event"] = auc_vs_event.loc[a, (ev, 'event_vs_event')]
        row[f"{short}_vs_none"] = auc_vs_none.loc[a, (ev, 'event_vs_none')]

      # per-variant lifespan (from span_df)
      df_sp = span_df[span_df['agent'] == a]
      for var in variants:
        length = int(df_sp.loc[df_sp['event'] == var, 'length'].sum())
        row[f"{var}_lifespan"] = length

      # per-variant reward (from reward_df)
      df_rw = reward_df[reward_df['agent'] == a]
      for var in variants:
        # reward_df may use NaN for missing
        val = df_rw.loc[df_rw['event'] == var, 'reward']
        row[f"{var}_reward"] = float(val.values[0]) if not val.empty else np.nan

      rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_pickle(summary_out)
    print(f"Wrote summary to {summary_out}")
  except Exception as e:
    log.error(f"Error processing session {session}: {e}")
    return


# ─── V) MAIN & PARALLEL ENTRY ────────────────────────────────────────────────

def main():
  # Result_paths
  top_dir = '/home/mikan/e/GitHub/social-agents-JAX/results/mix_RF2/'
  # top_dir = '/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/'
  event_dir = Path(f"{top_dir}analysis_results_merged/")
  ts_dir    = Path(f"{top_dir}analysis_results/")
  net_dir   = ts_dir
  figure_dir= Path("./figures")
  figure_dir.mkdir(exist_ok=True)
  output_dir= Path(f"{top_dir}analysis_results_event_stats/")
  output_dir.mkdir(exist_ok=True)

  sessions = [p.name.replace("_merged.pkl","")
              for p in event_dir.glob("*.pkl")]

  sessions = sorted(sessions)
  # run in parallel or sequentially:
  use_parallel = True
  if use_parallel:
    Parallel(n_jobs=60)(
      delayed(process_one)(sess, event_dir, ts_dir, net_dir, figure_dir, output_dir)
      for sess in tqdm(sessions)
    )
  else:
    for sess in sessions:
      process_one(sess, event_dir, ts_dir, net_dir, figure_dir, output_dir)


if __name__ == "__main__":
  main()
