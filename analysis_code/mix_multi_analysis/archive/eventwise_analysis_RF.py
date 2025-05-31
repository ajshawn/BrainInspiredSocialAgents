#!/usr/bin/env python3
"""
analysis_pipeline.py

1) Loads per-run event, timestep, and network-state data, organized by checkpoints.
2) Builds Boolean event masks & computes pairwise distances.
3) Plots distance traces with event rasters.
4) Computes and bar‐plots distance stats, prey lifespans, and prey rewards.
5) Builds and applies PCA to neural activities, then plots PCs with rasters.
6) Performs event classification using neural data.
7) (Optionally) processes all runs in parallel into a DataFrame.
"""

import warnings

warnings.filterwarnings("ignore")

import json
import logging
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse  # Added
import re  # Added

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from tqdm import tqdm  # Added

from group_analysis_utils.helper import parse_agent_roles
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ─── configure logging ───────────────────────────────────────────────────────
# Configure logging to write to a file and also to console
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("../analysis_pipeline.log", mode="w")  # Log file
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
log.addHandler(console_handler)


# ─── I) DATA LOADING UTILITIES ────────────────────────────────────────────────

def load_data(run_name: str,
              ckpt_num_str: str,
              event_input_dir: Path,
              ts_input_dir: Path,
              net_input_dir: Path,
              type_2_naming = False,
              ) -> Tuple[Dict[int, str], Dict[int, str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Return (roles, sources, ev_df, ts_df, ns_df) for one run."""
  roles, sources = parse_agent_roles(run_name)  # run_name is the original folder name like "predator_prey_..."

  # Construct file paths based on the new structure
  # Assumes event files are named by run_name directly in event_input_dir
  if not type_2_naming:
    ev_path = event_input_dir / f"{run_name}_higher_level_metrics.pkl"
    # Assumes ts and net files are named with ckpt_num and run_name in their respective dirs
    ts_path = ts_input_dir / f"ckpt_{ckpt_num_str}_{run_name}_info.csv"
    ns_path = net_input_dir / f"ckpt_{ckpt_num_str}_{run_name}_network_states.pkl"
  else:

    ev_path = event_input_dir / f"ckpt_{ckpt_num_str}_higher_level_metrics.pkl"
    ts_path = ts_input_dir / f"ckpt{ckpt_num_str}_{run_name}_info.csv"
    ns_path = net_input_dir / f"ckpt{ckpt_num_str}_{run_name}_network_states.pkl"

  if not ev_path.exists():
    log.error(f"Missing events file for run {run_name} (ckpt {ckpt_num_str}): {ev_path}")
    raise FileNotFoundError(f"Event file not found: {ev_path}")
  if not ts_path.exists():
    log.error(f"Missing timesteps file for run {run_name} (ckpt {ckpt_num_str}): {ts_path}")
    raise FileNotFoundError(f"Timestep file not found: {ts_path}")
  if not ns_path.exists():
    log.error(f"Missing network-states file for run {run_name} (ckpt {ckpt_num_str}): {ns_path}")
    raise FileNotFoundError(f"Network state file not found: {ns_path}")

  ev_df = pd.read_pickle(ev_path)
  ts_df = pd.read_csv(ts_path)
  ns_df = pd.read_pickle(ns_path)
  ev_df = pd.DataFrame(ev_df)
  ts_df = pd.DataFrame(ts_df)
  ev_df = pd.DataFrame(ev_df)

  log.info(f"Loaded data for run {run_name} (ckpt {ckpt_num_str}): "
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
    # ev_df is a DataFrame where each row corresponds to an episode.
    # ev_df[ev] is a Series where each element is a list of event dicts for that episode.
    for ep, entries_for_episode in enumerate(ev_df[ev].values):
      if not isinstance(entries_for_episode, list):  # Check if it's a list of events
        # log.warning(f"No event entries or unexpected format for event {ev} in episode {ep}. Skipping.")
        continue
      if not entries_for_episode:  # Empty list of events for this episode
        continue

      offset = ep * episode_length
      for d in entries_for_episode:
        if not isinstance(d, dict):
          log.warning(f"Event entry is not a dict for event {ev}, episode {ep}. Entry: {d}. Skipping this entry.")
          continue
        try:
          parts = d.get('participants')
          if parts is None:  # Handle if 'participants' key is missing
            log.warning(f"Missing 'participants' in event {ev} at episode {ep}, entry: {d}. Skipping this entry.")
            continue
          # Ensure participants are integers and within agent bounds
          valid_parts = [p for p in parts if isinstance(p, int) and 0 <= p < n_agents]
          if not valid_parts:
            log.warning(f"No valid participants for event {ev}, episode {ep}, entry: {d}. Skipping this entry.")
            continue

        except KeyError:
          log.warning(f"Missing 'participants' in event {ev} at episode {ep}, entry: {d}. Skipping this entry.")
          continue

        time_start = d.get('time_start', d.get('time'))  # Prioritize time_start
        time_end = d.get('time_end', time_start + 1 if time_start is not None else None)

        if time_start is None or time_end is None:
          log.warning(f"Missing time information in event {ev} at episode {ep}, entry: {d}. Skipping this entry.")
          continue

        s, e = int(time_start), int(time_end)

        # Boundary checks for time and offset
        if offset + e > n_steps:  # If event extends beyond total steps, cap it or warn
          # log.warning(f"Event {ev} in ep {ep} (entry {d}) extends beyond n_steps. Capping at {n_steps}.")
          e = n_steps - offset
          if s >= e: continue  # Skip if start is already beyond or at the new end

        masks[ev][offset + s:offset + e, valid_parts] = True
  return masks


def compute_agent_positions(ts_df: pd.DataFrame,
                            roles: Dict[int, str]
                            ) -> Tuple[np.ndarray, np.ndarray]:
  """Return (positions, death_mask) arrays of shapes (n_agents, T,2) and (T, n_agents)."""
  n_agents = len(roles)
  if ts_df.empty:
    log.warning("Timestep DataFrame (ts_df) is empty. Cannot compute agent positions.")
    # Return appropriately shaped empty/NaN arrays if possible, or raise error
    dummy_T = 1  # Cannot determine T if ts_df is empty
    return np.full((n_agents, dummy_T, 2), np.nan), np.full((dummy_T, n_agents), 0, dtype=int)

  T = len(ts_df)
  pos_list = []
  death_mat = np.zeros((T, n_agents), dtype=int)  # Assuming 0 is alive, 1 is dead from mark_death_periods

  for aid in range(n_agents):
    death_col_name = f"death_{aid}"
    if death_col_name in ts_df:
      death_mat[:, aid] = ts_df[death_col_name].values
    else:
      log.warning(f"Death column {death_col_name} not found for agent {aid}. Assuming agent is always alive.")
      death_mat[:, aid] = 0  # Default to alive if no info

    x_col = f"POSITION_x_{aid}"
    y_col = f"POSITION_y_{aid}"
    if x_col in ts_df and y_col in ts_df:
      pos = ts_df[[x_col, y_col]].values  # (T,2)
    else:
      log.warning(f"Missing position cols {x_col} or {y_col} for agent {aid} → using NaNs")
      pos = np.full((T, 2), np.nan)
    pos_list.append(pos)

  positions = np.stack(pos_list, axis=0)  # (n_agents, T,2)
  # In this pipeline, death_mask: 1 means alive.
  # mark_death_periods from group_analysis_utils typically returns 0 for alive, 1 for dead.
  # Let's ensure consistency: death_mask should be True for alive.
  # If death_mat has 0=alive, 1=dead, then alive_mask = (death_mat == 0)
  # The original pipeline's `compute_pairwise_distances` used `alive = (death_mask[:, i] == 1) & (death_mask[:, j] == 1)`
  # This implies death_mask had 1 for alive.
  # Let's adjust: if `ts_df[f"death_{aid}"]` stores 0 for alive and 1 for dead (common from `mark_death_periods`),
  # then we need to flip it for this pipeline's convention or adjust usage.
  # The `plot_lifespan_stats` uses `death_mask[:,a] == 1` for alive.
  # `find_distance_responsive_neurons` also uses `death_mask[:, agent] == 1` for alive.
  # So, let's assume `ts_df[f"death_{aid}"]` has 1 for alive, 0 for dead.
  # If it's the other way (0=alive, 1=dead from `mark_death_periods`), then it needs conversion here.
  # `mark_death_periods` from `group_analysis_utils.segmentation` returns 0 for alive, 1 for dead.
  # So, we need to invert it if that's the source.
  # Let's assume the `death_{aid}` columns in `_info.csv` are already 1 for alive, 0 for dead.
  # If not, this is a point of potential error.
  # For now, proceed assuming `death_mat` as loaded (0 or 1) is used consistently by downstream functions.
  # The functions `plot_lifespan_stats` and `find_distance_responsive_neurons` expect 1=alive.
  # `compute_pairwise_distances` also expects 1=alive in its `alive` boolean mask.
  # Let's ensure death_mask passed around has 1 for alive.
  # If `ts_df[f"death_{aid}"]` is 0 for alive, 1 for dead:
  # death_mask_final = (death_mat == 0).astype(int) # Now 1 is alive, 0 is dead
  # However, the original code directly used `death_mask[:, i] == 1`.
  # This implies the `death_{aid}` columns in `_info.csv` should store 1 for alive.
  # Let's assume this convention is met by the input `_info.csv`.

  return positions, death_mat  # death_mat where 1 means alive


def compute_pairwise_distances(positions: np.ndarray,
                               death_mask: np.ndarray  # Expects 1 for alive
                               ) -> Dict[Tuple[int, int], np.ndarray]:
  """Return dict[(i,j)] → distance time series (length T), with dead masked to NaN."""
  n_agents, T, _ = positions.shape
  out = {}
  for i in range(n_agents):
    for j in range(i + 1, n_agents):
      dij = np.linalg.norm(positions[i] - positions[j], axis=1)
      # True if both agents i and j are alive
      alive_i = death_mask[:, i] == 1
      alive_j = death_mask[:, j] == 1
      both_alive = alive_i & alive_j
      dij[~both_alive] = np.nan  # Mask distance to NaN if either is dead
      out[(i, j)] = dij
  return out


# ─── III) PLOTTING FUNCTIONS ────────────────────────────────────────────────

def plot_distance_rasters(dist_dict: Dict[Tuple[int, int], np.ndarray],
                          masks: Dict[str, np.ndarray],  # event masks
                          roles: Dict[int, str],
                          episode_length: int = 1000,  # Added for x_lim
                          figsize=(10, 6)
                          ) -> plt.Figure:
  """One subplot per agent: pair-wise distance traces + event rasters."""
  n_agents = len(roles)
  fig, axes = plt.subplots(n_agents, 1,
                           sharex=True, sharey=True,
                           figsize=(figsize[0], figsize[1] + 1.5 * n_agents))  # Adjusted height factor
  axes = np.atleast_1d(axes)  # Ensure axes is always iterable

  pair_colors = {'predator': 'crimson', 'prey': 'royalblue'}
  event_shades = {ev: c for ev, c in zip(masks.keys(),
                                         ['limegreen', 'orange', 'mediumpurple', 'gold', 'turquoise',
                                          'lightcoral'])}  # Added more colors

  for i, ax in enumerate(axes):
    ax.set_title(f"Agent {i} ({roles[i]})")
    # plot each pair that includes agent i
    for (a, b), dist_trace in dist_dict.items():
      if i in (a, b):
        other_agent_idx = b if i == a else a
        # Determine color based on the role of the *other* agent in the pair
        color_role = roles.get(other_agent_idx, 'prey')  # Default to prey if role not found
        ax.plot(dist_trace, label=f"vs Agent {other_agent_idx} ({roles[other_agent_idx]})",
                color=pair_colors.get(color_role, 'gray'), lw=0.8)

    # overlay event rasters
    y_shade_pos = 0.95  # Start shading from top of plot
    shade_height = 0.05  # Relative height for each event band

    for ev_idx, (ev, event_mask_for_all_agents) in enumerate(masks.items()):
      # series is the event mask for the current agent 'i'
      series = event_mask_for_all_agents[:, i].astype(int)
      starts = np.where(np.diff(series, prepend=0) == 1)[0]
      ends = np.where(np.diff(series, prepend=0) == -1)[0]
      if len(ends) < len(starts):  # If last event continues to end of trace
        ends = np.append(ends, len(series))

      for s, e in zip(starts, ends):
        # Use axvspan for shading background regions
        ax.axvspan(s, e,
                   color=event_shades.get(ev, 'lightgray'),  # Use .get for safety
                   alpha=0.3,  # Slightly increased alpha for visibility
                   zorder=0)  # Ensure it's behind lines

    ax.set_xlim(0, episode_length)  # Use dynamic episode_length
    ax.set_ylabel("Distance (px)")
    ax.legend(fontsize='x-small', ncol=max(1, n_agents // 2), loc='upper right')  # Dynamic columns for legend
    ax.grid(True, linestyle=':', alpha=0.6)

  axes[-1].set_xlabel("Timestep")
  # Single legend for event types (colors)
  handles = [Patch(facecolor=event_shades.get(ev, 'lightgray'), alpha=0.3,
                   label=ev.replace('_', ' ').title())
             for ev in masks]
  fig.legend(handles=handles,
             labels=[h.get_label() for h in handles],
             loc='lower center', ncol=len(handles),
             frameon=False, fontsize='small')
  plt.tight_layout(rect=[0, 0.06, 1, 0.98])  # Adjust rect to make space for fig legend
  return fig


def plot_distance_stats(dist_dict: Dict[Tuple[int, int], np.ndarray],
                        ev_df: pd.DataFrame,  # event data, one row per episode
                        masks: Dict[str, np.ndarray],  # precomputed event masks (T, n_agents)
                        roles: Dict[int, str],
                        episode_length: int = 1000
                        ) -> plt.Figure:
  """Bar‐plot of mean distances by category during each event."""

  def categorize(agent_i_idx, agent_j_idx, event_participants_set, roles_map):
    role_i, role_j = roles_map[agent_i_idx], roles_map[agent_j_idx]
    i_is_participant = agent_i_idx in event_participants_set
    j_is_participant = agent_j_idx in event_participants_set

    # Predator-Prey interactions
    if (role_i == 'predator' and role_j == 'prey') or \
        (role_i == 'prey' and role_j == 'predator'):
      if i_is_participant and j_is_participant: return 'Pred(P)-Prey(P)'  # Both participating
      if i_is_participant and not j_is_participant: return f"{role_i.capitalize()}(P)-{role_j.capitalize()}(¬P)"
      if not i_is_participant and j_is_participant: return f"{role_i.capitalize()}(¬P)-{role_j.capitalize()}(P)"
      return 'Pred(¬P)-Prey(¬P)'  # Neither participating

    # Prey-Prey interactions
    if role_i == 'prey' and role_j == 'prey':
      if i_is_participant and j_is_participant: return 'Prey(P)-Prey(P)'
      if i_is_participant and not j_is_participant: return 'Prey(P)-Prey(¬P)'
      if not i_is_participant and j_is_participant: return 'Prey(¬P)-Prey(P)'
      return 'Prey(¬P)-Prey(¬P)'

    # Predator-Predator (if applicable)
    if role_i == 'predator' and role_j == 'predator':
      if i_is_participant and j_is_participant: return 'Pred(P)-Pred(P)'
      # Add more specific predator-predator categories if needed
      return 'Pred(¬P)-Pred(¬P)'

    return None  # Default for unhandled pairs

  rows = []
  for event_name in masks.keys():  # Iterate through event types using the precomputed masks
    # event_mask_for_event_type has shape (T, n_agents)
    event_mask_for_event_type = masks[event_name]

    # Need to iterate through episodes to get participants for specific event instances
    for ep_idx, entries_for_episode in enumerate(ev_df[event_name].values):
      if not isinstance(entries_for_episode, list): continue

      offset = ep_idx * episode_length

      for event_instance_dict in entries_for_episode:
        if not isinstance(event_instance_dict, dict): continue

        participants_list = event_instance_dict.get('participants', [])
        if not participants_list: continue
        participants_set = set(participants_list)

        time_start = event_instance_dict.get('time_start', event_instance_dict.get('time'))
        time_end = event_instance_dict.get('time_end', time_start + 1 if time_start is not None else None)

        if time_start is None or time_end is None: continue

        # Global time indices for this specific event instance
        s_global, e_global = offset + int(time_start), offset + int(time_end)
        if e_global > event_mask_for_event_type.shape[0]:  # Cap at total timesteps
          e_global = event_mask_for_event_type.shape[0]
        if s_global >= e_global: continue

        # Iterate over agent pairs for distances
        for (i, j), dist_trace in dist_dict.items():
          category = categorize(i, j, participants_set, roles)
          if category is None: continue

          # Extract distances during this specific event instance
          # dist_trace is (T_total_duration)
          # We need dist_trace[s_global:e_global]
          event_instance_distances = dist_trace[s_global:e_global]
          mean_dist_for_instance = np.nanmean(event_instance_distances)

          if not np.isnan(mean_dist_for_instance):
            rows.append({
              'event': event_name,
              'category': category,
              'distance': mean_dist_for_instance
            })

  if not rows:
    log.warning("No distance stats data generated. Plot will be empty.")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, "No data for distance stats", ha='center', va='center')
    return fig

  df = pd.DataFrame(rows).dropna()
  if df.empty:
    log.warning("Distance stats DataFrame is empty after dropping NaNs. Plot will be empty.")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, "No data for distance stats after NaN drop", ha='center', va='center')
    return fig

  pivot = (df.groupby(['event', 'category'])['distance']
           .mean().unstack('category'))

  fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), 5))  # Dynamic width
  pivot.plot(kind='bar', ax=ax, width=0.8)
  ax.set_ylabel("Mean distance (px)")
  ax.set_title("Pairwise Distances During Events")
  ax.legend(title="Pair Category", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
  ax.tick_params(axis='x', rotation=45)
  ax.grid(axis='y', linestyle='--', alpha=0.7)
  plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust for legend
  return fig


def plot_lifespan_stats(ev_df: pd.DataFrame,
                        death_mask: np.ndarray,  # 1 for alive
                        roles: Dict[int, str],
                        events: List[str],  # Base event names: 'apple_cooperation_events', etc.
                        episode_length: int = 1000
                        ) -> Tuple[plt.Figure, pd.DataFrame]:
  """
  Compute & plot total alive‐duration (lifespan) per agent × event variant.
  """
  prey_ids = [i for i, r in roles.items() if r == 'prey']
  if not prey_ids:
    log.warning("No prey agents found. Lifespan stats plot will be empty.")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, "No prey agents for lifespan stats", ha='center', va='center')
    return fig, pd.DataFrame()

  T = death_mask.shape[0]
  # Define variants based on how events are structured (e.g., with helper/beneficiary roles)
  variants = ['apple_cooperation_events',
              'distraction_events_helper', 'distraction_events_beneficiary',
              'fence_events_helper', 'fence_events_beneficiary',
              'none']  # 'none' means not in any of the listed specific event types

  # Per-agent, per-variant boolean masks (T_total_duration)
  variant_masks_for_agent = {agent_id: {var_name: np.zeros(T, dtype=bool) for var_name in variants}
                             for agent_id in prey_ids}

  # Populate masks for specific event variants
  for base_event_name in events:  # e.g., 'apple_cooperation_events', 'distraction_events', 'fence_events'
    if base_event_name not in ev_df.columns:
      log.warning(f"Base event '{base_event_name}' not in ev_df columns. Skipping for lifespan stats.")
      continue

    for ep_idx, entries_for_episode in enumerate(ev_df[base_event_name].values):
      if not isinstance(entries_for_episode, list): continue
      offset = ep_idx * episode_length

      for event_instance_dict in entries_for_episode:
        if not isinstance(event_instance_dict, dict): continue

        time_s = event_instance_dict.get('time_start', event_instance_dict.get('time'))
        time_e = event_instance_dict.get('time_end', time_s + 1 if time_s is not None else None)
        if time_s is None or time_e is None: continue

        idx_slice = slice(offset + int(time_s), min(offset + int(time_e), T))  # Ensure slice is within bounds

        if base_event_name == 'apple_cooperation_events':
          participants = event_instance_dict.get('participants', [])
          for agent_id in participants:
            if agent_id in prey_ids:
              variant_masks_for_agent[agent_id][base_event_name][idx_slice] = True
        elif base_event_name in ['distraction_events', 'fence_events']:
          for role_key in ('helper', 'beneficiary'):
            variant_name = f"{base_event_name}_{role_key}"
            role_participants = event_instance_dict.get(role_key, [])
            if isinstance(role_participants, int): role_participants = [role_participants]
            for agent_id in role_participants:
              if agent_id in prey_ids:
                variant_masks_for_agent[agent_id][variant_name][idx_slice] = True

  # Calculate lifespan for each variant
  records = []
  for agent_id in prey_ids:
    agent_is_alive_mask = (death_mask[:, agent_id] == 1)  # Mask where this agent is alive

    # Create a mask for 'any' of the defined specific events for this agent
    any_specific_event_for_agent = np.zeros(T, dtype=bool)
    for var_name in variants:
      if var_name != 'none':  # Exclude 'none' itself from this aggregation
        any_specific_event_for_agent |= variant_masks_for_agent[agent_id][var_name]

    for var_name in variants:
      if var_name == 'none':
        # 'none' is when agent is alive AND not in any_specific_event
        current_variant_active_mask = (~any_specific_event_for_agent) & agent_is_alive_mask
      else:
        # For specific events, it's when the event variant is active AND agent is alive
        current_variant_active_mask = variant_masks_for_agent[agent_id][var_name] & agent_is_alive_mask

      total_lifespan_in_variant = int(current_variant_active_mask.sum())
      records.append({'agent': agent_id, 'event_variant': var_name, 'lifespan': total_lifespan_in_variant})

  span_df = pd.DataFrame(records)

  if span_df.empty:
    log.warning("Lifespan DataFrame is empty. Plot will be empty.")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, "No data for lifespan stats", ha='center', va='center')
    return fig, span_df

  pivot = span_df.pivot(index='agent', columns='event_variant', values='lifespan').fillna(0)
  fig, ax = plt.subplots(figsize=(max(8, len(variants) * 0.8), 5))  # Dynamic width
  pivot.plot(kind='bar', ax=ax, width=0.8)
  ax.set_ylabel("Total Timesteps Alive")
  ax.set_title("Prey Lifespan by Event Variant")
  ax.legend(title="Event Variant", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
  ax.tick_params(axis='x', rotation=0)  # Agents on x-axis, no rotation needed if few
  ax.grid(axis='y', linestyle='--', alpha=0.7)
  plt.tight_layout(rect=[0, 0, 0.85, 1])
  return fig, span_df


def plot_reward_stats(ts_df: pd.DataFrame,
                      death_mask: np.ndarray,  # 1 for alive
                      ev_df: pd.DataFrame,
                      roles: Dict[int, str],
                      events: List[str],  # Base event names
                      reward_prefix: str = 'rewards_',  # Adjusted from 'reward_'
                      episode_length: int = 1000
                      ) -> Tuple[plt.Figure, pd.DataFrame]:
  """
  For each agent and each event variant, compute mean reward per alive step.
  """
  T, n_agents = death_mask.shape
  variants = ['apple_cooperation_events',
              'distraction_events_helper', 'distraction_events_beneficiary',
              'fence_events_helper', 'fence_events_beneficiary',
              'none']

  # Per-agent, per-variant boolean masks (T_total_duration)
  variant_masks_for_agent = {agent_id: {var_name: np.zeros(T, dtype=bool) for var_name in variants}
                             for agent_id in roles.keys()}  # For all agents

  # Populate masks (similar to lifespan stats)
  for base_event_name in events:
    if base_event_name not in ev_df.columns: continue
    for ep_idx, entries_for_episode in enumerate(ev_df[base_event_name].values):
      if not isinstance(entries_for_episode, list): continue
      offset = ep_idx * episode_length
      for event_instance_dict in entries_for_episode:
        if not isinstance(event_instance_dict, dict): continue
        time_s = event_instance_dict.get('time_start', event_instance_dict.get('time'))
        time_e = event_instance_dict.get('time_end', time_s + 1 if time_s is not None else None)
        if time_s is None or time_e is None: continue
        idx_slice = slice(offset + int(time_s), min(offset + int(time_e), T))

        if base_event_name == 'apple_cooperation_events':
          participants = event_instance_dict.get('participants', [])
          for agent_id in participants:
            if agent_id in roles:  # Check if agent_id is valid
              variant_masks_for_agent[agent_id][base_event_name][idx_slice] = True
        elif base_event_name in ['distraction_events', 'fence_events']:
          for role_key in ('helper', 'beneficiary'):
            variant_name = f"{base_event_name}_{role_key}"
            role_participants = event_instance_dict.get(role_key, [])
            if isinstance(role_participants, int): role_participants = [role_participants]
            for agent_id in role_participants:
              if agent_id in roles:
                variant_masks_for_agent[agent_id][variant_name][idx_slice] = True

  records = []
  for agent_id in sorted(roles.keys()):
    agent_is_alive_mask = (death_mask[:, agent_id] == 1)
    reward_col_name = f"{reward_prefix}{agent_id}"
    if reward_col_name not in ts_df.columns:
      log.warning(
        f"Reward column {reward_col_name} not found for agent {agent_id}. Skipping reward stats for this agent.")
      continue
    agent_rewards_trace = ts_df[reward_col_name].values

    any_specific_event_for_agent = np.zeros(T, dtype=bool)
    for var_name in variants:
      if var_name != 'none':
        any_specific_event_for_agent |= variant_masks_for_agent[agent_id][var_name]

    for var_name in variants:
      if var_name == 'none':
        current_variant_active_mask = (~any_specific_event_for_agent) & agent_is_alive_mask
      else:
        current_variant_active_mask = variant_masks_for_agent[agent_id][var_name] & agent_is_alive_mask

      if current_variant_active_mask.any():  # If agent was active in this variant at all
        mean_reward_in_variant = np.nanmean(agent_rewards_trace[current_variant_active_mask])
      else:
        mean_reward_in_variant = np.nan

      records.append({'agent': agent_id, 'event_variant': var_name, 'mean_reward_per_step': mean_reward_in_variant})

  reward_df = pd.DataFrame(records)

  if reward_df.empty:
    log.warning("Reward DataFrame is empty. Plot will be empty.")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, "No data for reward stats", ha='center', va='center')
    return fig, reward_df

  pivot = reward_df.pivot(index='agent', columns='event_variant', values='mean_reward_per_step').fillna(
    0)  # Fill NaN with 0 for plotting
  fig, ax = plt.subplots(figsize=(max(8, len(variants) * 0.8), 5))
  pivot.plot(kind='bar', ax=ax, width=0.8)
  ax.set_ylabel("Mean Reward / Step")
  ax.set_title("Agent Mean Reward per Step by Event Variant")
  ax.legend(title="Event Variant", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
  ax.tick_params(axis='x', rotation=0)
  ax.grid(axis='y', linestyle='--', alpha=0.7)
  plt.tight_layout(rect=[0, 0, 0.85, 1])
  return fig, reward_df


def plot_pca_rasters(ns_df: pd.DataFrame,
                     death_mask: np.ndarray,  # 1 for alive
                     masks: Dict[str, np.ndarray],  # event masks
                     roles: Dict[int, str],
                     n_components: int = 3,
                     episode_length: int = 1000,  # For x_lim
                     ) -> plt.Figure:
  """
  For each agent i: PCA on alive‐period hidden states, plot PCs with event rasters.
  """
  T = death_mask.shape[0]  # ns_df might be shorter if not all episodes processed, use death_mask T
  agents = sorted(roles.keys())
  fig, axes = plt.subplots(len(agents), 1,
                           sharex=True,
                           figsize=(10, 2.5 * len(agents)))  # Adjusted height factor
  axes = np.atleast_1d(axes)

  event_shades = {ev: c for ev, c in zip(masks.keys(),
                                         ['limegreen', 'orange', 'mediumpurple', 'gold', 'turquoise', 'lightcoral'])}
  pc_colors = plt.cm.viridis(np.linspace(0, 0.8, n_components))  # Use a colormap for PCs

  for idx, agent_id in enumerate(agents):
    ax = axes[idx]
    ax.set_title(f"Agent {agent_id} PCs ({roles[agent_id]})")

    agent_hidden_cols = sorted([c for c in ns_df.columns if c.startswith(f"hidden_{agent_id}_")],
                               key=lambda c: int(c.split('_')[-1]))  # Sort by neuron index part
    if not agent_hidden_cols:
      log.warning(f"No hidden state columns found for agent {agent_id}. Skipping PCA plot for this agent.")
      ax.text(0.5, 0.5, "No hidden state data", ha='center', va='center')
      continue

    X_agent_all_steps = ns_df[agent_hidden_cols].values  # (T_ns_df, F_agent)

    # Ensure X_agent_all_steps has same T as death_mask, if ns_df was shorter (e.g. from single episode)
    if X_agent_all_steps.shape[0] != T:
      log.warning(
        f"Mismatch in length between ns_df ({X_agent_all_steps.shape[0]}) and death_mask ({T}) for agent {agent_id}. Truncating/padding ns_df for PCA plot.")
      # This case needs careful handling; for now, let's assume they match or ns_df is a subset
      # If ns_df is shorter, we can only plot up to its length.
      # If ns_df is longer (should not happen if T is from death_mask of full run), it's an issue.
      # Simplest for now: use min_len if they differ, but this might hide issues.
      # Let's assume T from death_mask is the true full length.
      # If X_agent_all_steps is shorter, it implies ns_df didn't cover all episodes.
      # The PCA should be fit on available data, but plotted against full timeline if possible.
      # This part is tricky if lengths don't align. The original code assumes ns_df has T rows.
      # For now, proceed assuming ns_df covers T, or this will error/misalign.

    agent_is_alive_mask = (death_mask[:, agent_id] == 1)  # Mask where this agent is alive

    # Fit PCA only on steps where agent is alive and data is finite
    X_agent_alive = X_agent_all_steps[agent_is_alive_mask & np.all(np.isfinite(X_agent_all_steps), axis=1)]

    if X_agent_alive.shape[0] < n_components or X_agent_alive.shape[
      1] < n_components:  # Check if enough data/features for PCA
      log.warning(
        f"Not enough alive samples or features for PCA for agent {agent_id} (samples: {X_agent_alive.shape[0]}, features: {X_agent_alive.shape[1]}). Skipping PCA.")
      ax.text(0.5, 0.5, "PCA skipped (insufficient data)", ha='center', va='center')
      continue

    pca = PCA(n_components=min(n_components, X_agent_alive.shape[1]))  # Ensure n_components <= n_features
    try:
      pca.fit(X_agent_alive)
    except ValueError as e:
      log.error(f"PCA fit failed for agent {agent_id}: {e}. Skipping PCA for this agent.")
      ax.text(0.5, 0.5, f"PCA fit error: {e}", ha='center', va='center', fontsize='small', wrap=True)
      continue

    projected_scores_all_steps = pca.transform(X_agent_all_steps)  # (T_ns_df, n_components_actual)

    for pc_idx in range(pca.n_components_):
      pc_trace = projected_scores_all_steps[:, pc_idx].copy()
      pc_trace[~agent_is_alive_mask[:len(pc_trace)]] = np.nan  # Mask dead steps, ensure consistent length
      ax.plot(pc_trace,
              label=f"PC{pc_idx + 1}",
              color=pc_colors[pc_idx % len(pc_colors)],  # Cycle through colors if n_components > len(pc_colors)
              lw=0.9)

    for ev, event_mask_for_all_agents in masks.items():
      series = event_mask_for_all_agents[:, agent_id].astype(int)
      starts = np.where(np.diff(series, prepend=0) == 1)[0]
      ends = np.where(np.diff(series, prepend=0) == -1)[0]
      if len(ends) < len(starts): ends = np.append(ends, T)
      for s, e in zip(starts, ends):
        ax.axvspan(s, e, color=event_shades.get(ev, 'lightgray'), alpha=0.3, zorder=0)

    ax.set_ylabel("PC Value")
    ax.legend(fontsize='small', loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)

  axes[-1].set_xlim(0, episode_length)  # Show one typical episode length
  axes[-1].set_xlabel("Timestep")
  plt.tight_layout()
  return fig


def classify_multiple_events_per_agent(
    ns_df: pd.DataFrame,
    death_mask: np.ndarray,  # 1 for alive
    ev_df: pd.DataFrame,  # event data, one row per episode
    events: List[str],  # Base event names
    roles: Dict[int, str],
    mode: str = 'event_vs_rest',  # 'event_vs_none', 'event_vs_event'
    episode_length: int = 1000,
    n_components: int = 3  # For PCA pre-processing
) -> pd.DataFrame:
  """
  Leave-one-episode-out classification of event onsets using PCA-reduced neural states.
  """
  T_total = death_mask.shape[0]
  if T_total == 0: return pd.DataFrame()  # No data to process

  # Determine number of episodes from ev_df (assuming all event lists have same length)
  # Or, if T_total and episode_length are reliable:
  n_episodes = T_total // episode_length
  if T_total % episode_length != 0:
    log.warning(
      f"Total timesteps {T_total} not a multiple of episode_length {episode_length}. Grouping for LOGO might be inexact.")
    # Fallback: use length of one of the event lists in ev_df if available
    if events and events[0] in ev_df.columns and len(ev_df[events[0]]) > 0:
      n_episodes = len(ev_df[events[0]])
    else:  # If ev_df is also problematic for n_episodes
      log.error("Cannot reliably determine n_episodes for LOGO. Aborting classification.")
      return pd.DataFrame()

  # Group labels for LeaveOneGroupOut (LOGO)
  groups_for_logo = np.repeat(np.arange(n_episodes), episode_length)[:T_total]  # Ensure groups match T_total

  # Precompute event onset masks (1 at onset, 0 otherwise) for each base event type
  event_onset_masks = {}  # Key: event_name, Value: np.array (T_total_duration)
  any_event_onset_overall = np.zeros(T_total, dtype=bool)  # Mask for any event onset across all types

  for event_name in events:
    if event_name not in ev_df.columns:
      log.warning(f"Event type '{event_name}' not in ev_df. Skipping for classification.")
      continue
    current_event_onset_mask = np.zeros(T_total, dtype=bool)
    for ep_idx, entries_for_episode in enumerate(ev_df[event_name].values):
      if not isinstance(entries_for_episode, list): continue
      offset = ep_idx * episode_length
      for event_instance_dict in entries_for_episode:
        if not isinstance(event_instance_dict, dict): continue
        # Use 'time_start' if available, else 'time' for onset
        onset_time_in_episode = event_instance_dict.get('time_start', event_instance_dict.get('time'))
        if onset_time_in_episode is not None:
          global_onset_idx = offset + int(onset_time_in_episode)
          if 0 <= global_onset_idx < T_total:
            current_event_onset_mask[global_onset_idx] = True
    event_onset_masks[event_name] = current_event_onset_mask
    any_event_onset_overall |= current_event_onset_mask

  logo_cv_splitter = LeaveOneGroupOut()
  classification_results = []

  for agent_id in sorted(roles.keys()):
    agent_is_alive_mask = (death_mask[:, agent_id] == 1)

    agent_hidden_cols = sorted([c for c in ns_df.columns if c.startswith(f"hidden_{agent_id}_")],
                               key=lambda c: int(c.split('_')[-1]))
    if not agent_hidden_cols:
      log.warning(f"No hidden state columns for agent {agent_id}. Skipping classification.")
      continue
    X_agent_all_steps = ns_df[agent_hidden_cols].values

    for target_event_name, y_target_event_onsets in event_onset_masks.items():
      # Define negative class mask based on classification mode
      if mode == 'event_vs_none':
        # Negatives: alive AND no event onset of ANY type is happening
        negative_class_mask = agent_is_alive_mask & (~any_event_onset_overall)
      elif mode == 'event_vs_rest':
        # Negatives: alive AND this specific target_event_name is NOT an onset
        negative_class_mask = agent_is_alive_mask & (~y_target_event_onsets)
      elif mode == 'event_vs_event':
        # Negatives: alive AND some OTHER event is an onset (but not target_event_name)
        other_event_onsets = any_event_onset_overall & (~y_target_event_onsets)
        negative_class_mask = agent_is_alive_mask & other_event_onsets
      else:
        log.error(f"Invalid classification mode: {mode}")
        continue

      fold_aucs = []
      for train_indices, test_indices in logo_cv_splitter.split(X_agent_all_steps, y_target_event_onsets,
                                                                groups_for_logo):
        # Select timepoints for training: must be (target event onset OR in negative class) AND agent alive
        train_selection_mask = (y_target_event_onsets[train_indices] | negative_class_mask[train_indices]) & \
                               agent_is_alive_mask[train_indices]
        X_train_fold_selected = X_agent_all_steps[train_indices][train_selection_mask]
        y_train_fold_selected = y_target_event_onsets[train_indices][train_selection_mask]

        # Select timepoints for testing (similar logic)
        test_selection_mask = (y_target_event_onsets[test_indices] | negative_class_mask[test_indices]) & \
                              agent_is_alive_mask[test_indices]
        X_test_fold_selected = X_agent_all_steps[test_indices][test_selection_mask]
        y_test_fold_selected = y_target_event_onsets[test_indices][test_selection_mask]

        if len(np.unique(y_train_fold_selected)) < 2 or X_train_fold_selected.shape[
          0] < 2: continue  # Need both classes in train
        if len(np.unique(y_test_fold_selected)) < 2 or X_test_fold_selected.shape[
          0] < 2: continue  # And in test for AUC

        # PCA fitting and transformation
        num_pca_components = min(n_components, X_train_fold_selected.shape[1],
                                 X_train_fold_selected.shape[0] - 1 if X_train_fold_selected.shape[0] > 1 else 1)
        if num_pca_components <= 0: continue

        pca = PCA(n_components=num_pca_components)
        try:
          X_train_pca = pca.fit_transform(X_train_fold_selected)
          X_test_pca = pca.transform(X_test_fold_selected)
        except ValueError:  # PCA can fail if variance is zero etc.
          continue

        # Logistic Regression
        clf = LogisticRegression(solver='liblinear', class_weight='balanced')
        try:
          clf.fit(X_train_pca, y_train_fold_selected)
          if hasattr(clf, "predict_proba"):
            test_pred_probs = clf.predict_proba(X_test_pca)[:, 1]
            fold_aucs.append(roc_auc_score(y_test_fold_selected, test_pred_probs))
        except ValueError:  # e.g. if only one class present after some filtering
          continue

      mean_auc_for_event = float(np.nanmean(fold_aucs)) if fold_aucs else np.nan
      classification_results.append({
        'agent': agent_id,
        'event': target_event_name,  # This is the target event
        'mode': mode,  # The mode string like 'event_vs_rest'
        'roc_auc': mean_auc_for_event
      })

  if not classification_results: return pd.DataFrame()

  # Pivot the table for easier access in summary
  results_df = pd.DataFrame(classification_results)
  try:
    # Pivoting might fail if there are duplicate (agent, event, mode) entries or all NaNs
    pivot_df = results_df.pivot_table(
      index='agent', columns=['event', 'mode'], values='roc_auc'
    )
    return pivot_df
  except Exception as e:
    log.error(f"Could not pivot classification results: {e}. Returning unpivoted DataFrame.")
    return results_df


def find_distance_responsive_neurons(ns_df: pd.DataFrame,
                                     positions: np.ndarray,  # (n_agents, T, 2)
                                     death_mask: np.ndarray,  # (T, n_agents), 1 for alive
                                     roles: Dict[int, str]
                                     ) -> pd.DataFrame:
  """
  Correlate each neuron's activity with distance to nearest predator and nearest other prey.
  """
  results_list = []  # Store results as a list of dicts for easier DataFrame creation
  T_total = positions.shape[1]
  scaler = StandardScaler()

  for agent_id in sorted(roles.keys()):
    agent_pos_trace = positions[agent_id]  # (T_total, 2)
    agent_is_alive_mask = (death_mask[:, agent_id] == 1)

    # Distances to nearest predator
    predator_ids = [p_id for p_id, r in roles.items() if r == 'predator' and p_id != agent_id]
    if predator_ids:
      dist_to_preds_matrix = np.array(
        [np.linalg.norm(agent_pos_trace - positions[p_id], axis=1) for p_id in predator_ids])
      dist_to_nearest_pred = np.nanmin(dist_to_preds_matrix, axis=0)
    else:
      dist_to_nearest_pred = np.full(T_total, np.nan)

    # Distances to nearest other prey
    other_prey_ids = [op_id for op_id, r in roles.items() if r == 'prey' and op_id != agent_id]
    if other_prey_ids:
      dist_to_other_preys_matrix = np.array(
        [np.linalg.norm(agent_pos_trace - positions[op_id], axis=1) for op_id in other_prey_ids])
      dist_to_nearest_other_prey = np.nanmin(dist_to_other_preys_matrix, axis=0)
    else:
      dist_to_nearest_other_prey = np.full(T_total, np.nan)

    agent_hidden_cols = sorted([c for c in ns_df.columns if c.startswith(f"hidden_{agent_id}_")],
                               key=lambda c: int(c.split('_')[-1]))
    if not agent_hidden_cols:
      log.warning(f"No hidden state columns for agent {agent_id} for distance correlation.")
      results_list.append({'agent': agent_id, 'best_neuron_idx_pred': np.nan, 'corr_pred': np.nan,
                           'best_neuron_idx_prey': np.nan, 'corr_prey': np.nan})
      continue

    X_agent_all_steps = ns_df[agent_hidden_cols].values

    corrs_with_pred_dist = []
    corrs_with_prey_dist = []

    for neuron_idx in range(X_agent_all_steps.shape[1]):
      neuron_activity = X_agent_all_steps[:, neuron_idx]

      # Correlation with predator distance
      valid_mask_pred = agent_is_alive_mask & np.isfinite(dist_to_nearest_pred) & np.isfinite(neuron_activity)
      if valid_mask_pred.sum() > 2:  # Need at least 2 points for correlation
        # Standardize before correlating to mitigate scale effects (optional but good practice)
        act_scaled = scaler.fit_transform(neuron_activity[valid_mask_pred].reshape(-1, 1)).flatten()
        dist_scaled = scaler.fit_transform(dist_to_nearest_pred[valid_mask_pred].reshape(-1, 1)).flatten()
        corrs_with_pred_dist.append(np.corrcoef(act_scaled, dist_scaled)[0, 1])
      else:
        corrs_with_pred_dist.append(np.nan)

      # Correlation with other prey distance
      valid_mask_prey = agent_is_alive_mask & np.isfinite(dist_to_nearest_other_prey) & np.isfinite(neuron_activity)
      if valid_mask_prey.sum() > 2:
        act_scaled = scaler.fit_transform(neuron_activity[valid_mask_prey].reshape(-1, 1)).flatten()
        dist_scaled = scaler.fit_transform(dist_to_nearest_other_prey[valid_mask_prey].reshape(-1, 1)).flatten()
        corrs_with_prey_dist.append(np.corrcoef(act_scaled, dist_scaled)[0, 1])
      else:
        corrs_with_prey_dist.append(np.nan)

    # Find neuron with max absolute correlation for each distance type
    best_idx_pred = np.nanargmax(np.abs(corrs_with_pred_dist)) if not all(
      np.isnan(c) for c in corrs_with_pred_dist) else np.nan
    best_idx_prey = np.nanargmax(np.abs(corrs_with_prey_dist)) if not all(
      np.isnan(c) for c in corrs_with_prey_dist) else np.nan

    results_list.append({
      'agent': agent_id,  # Store agent_id for merging
      'best_neuron_idx_pred': int(best_idx_pred) if pd.notna(best_idx_pred) else np.nan,
      'corr_pred': corrs_with_pred_dist[int(best_idx_pred)] if pd.notna(
        best_idx_pred) and corrs_with_pred_dist else np.nan,
      'best_neuron_idx_prey': int(best_idx_prey) if pd.notna(best_idx_prey) else np.nan,
      'corr_prey': corrs_with_prey_dist[int(best_idx_prey)] if pd.notna(
        best_idx_prey) and corrs_with_prey_dist else np.nan
    })

  return pd.DataFrame(results_list).set_index('agent')


def process_one(run_name: str,
                ckpt_num_str: str,
                event_input_dir: Path,
                ts_input_dir: Path,
                net_input_dir: Path,
                figure_output_root: Path,
                summary_output_root: Path,
                ignore_existing: bool = False,
                episode_length_config: int = 1000,  # Make episode length configurable
                type_2_naming=False,
                ) -> None:
  """Load data, compute & save all figures and a per-agent summary for one run."""

  current_figure_dir = figure_output_root / f"ckpt_{ckpt_num_str}" / run_name
  current_figure_dir.mkdir(parents=True, exist_ok=True)

  current_summary_dir = summary_output_root / f"ckpt_{ckpt_num_str}"
  current_summary_dir.mkdir(parents=True, exist_ok=True)
  summary_out_path = current_summary_dir / f"{run_name}_event_stats_and_decode.pkl"

  if ignore_existing and summary_out_path.exists():
    log.info(f"Skipping run {run_name} (ckpt {ckpt_num_str}), summary already exists: {summary_out_path}")
    return

  try:
    log.info(f"Processing run: {run_name} (ckpt: {ckpt_num_str})")
    roles, sources, ev_df, ts_df, ns_df = load_data(run_name, ckpt_num_str,
                                                    event_input_dir, ts_input_dir, net_input_dir, type_2_naming=type_2_naming)

    # if ts_df.empty:  # Critical check
    #   log.error(f"Timestep data (ts_df) is empty for run {run_name}. Cannot proceed.")
    #   return
    # if ns_df.empty:
    #   log.warning(f"Network state data (ns_df) is empty for run {run_name}. Some analyses might fail or be empty.")
    # if ev_df.empty:
    #   log.warning(f"Event data (ev_df) is empty for run {run_name}. Event-related analyses will be empty.")

    positions, death_mask = compute_agent_positions(ts_df, roles)
    T_total_duration, n_agents = death_mask.shape

    # Define base events for analysis
    events_to_analyze = ['apple_cooperation_events', 'distraction_events', 'fence_events']
    # Filter events_to_analyze to those present in ev_df to prevent KeyErrors
    available_events = [ev for ev in events_to_analyze if ev in ev_df.columns]
    if not available_events:
      log.warning(
        f"No specified events ({events_to_analyze}) found in ev_df columns for run {run_name}. Event-specific plots will be empty.")

    # Build event masks only for available events
    event_masks = build_event_masks(ev_df, T_total_duration, n_agents, available_events, episode_length_config)
    pair_distances = compute_pairwise_distances(positions, death_mask)

    # --- II) GENERATE & SAVE FIGURES ---
    figures_to_save = []
    figure_names = []

    if available_events:  # Only plot if there are events to show
      figures_to_save.append(plot_distance_rasters(pair_distances, event_masks, roles, episode_length_config))
      figure_names.append("distance_rasters")
      figures_to_save.append(plot_distance_stats(pair_distances, ev_df, event_masks, roles, episode_length_config))
      figure_names.append("distance_stats")

    life_fig, lifespan_summary_df = plot_lifespan_stats(ev_df, death_mask, roles, available_events,
                                                        episode_length_config)
    figures_to_save.append(life_fig)
    figure_names.append("lifespan_stats")

    reward_fig, reward_summary_df = plot_reward_stats(ts_df, death_mask, ev_df, roles, available_events,
                                                      reward_prefix='rewards_', episode_length=episode_length_config)
    figures_to_save.append(reward_fig)
    figure_names.append("reward_stats")

    if not ns_df.empty and available_events:  # PCA plot needs network states and events
      pca_rasters_fig = plot_pca_rasters(ns_df, death_mask, event_masks, roles, n_components=3,
                                         episode_length=episode_length_config)
      figures_to_save.append(pca_rasters_fig)
      figure_names.append("pca_rasters")

    for fig_name, fig_object in zip(figure_names, figures_to_save):
      output_path = current_figure_dir / f"{run_name}_{fig_name}.png"
      fig_object.savefig(output_path, dpi=150, bbox_inches='tight')
      plt.close(fig_object)
    log.info(f"Saved all figures for run {run_name} (ckpt {ckpt_num_str}) to {current_figure_dir}")

    # --- III) CLASSIFIERS & CORRELATIONS ---
    # Ensure dataframes are not empty before proceeding
    auc_results_dfs = {}
    if not ns_df.empty and not ev_df.empty and available_events:
      for mode in ['event_vs_rest', 'event_vs_event', 'event_vs_none']:
        auc_df = classify_multiple_events_per_agent(
          ns_df, death_mask, ev_df, available_events, roles, mode=mode,
          episode_length=episode_length_config, n_components=3)
        auc_results_dfs[mode] = auc_df
    else:
      log.warning(f"Skipping event classification for {run_name} due to empty ns_df, ev_df or no available_events.")

    distance_corr_df = pd.DataFrame()  # Initialize empty
    if not ns_df.empty:
      distance_corr_df = find_distance_responsive_neurons(ns_df, positions, death_mask, roles)
    else:
      log.warning(f"Skipping distance correlation for {run_name} due to empty ns_df.")

    # --- IV) BUILD & SAVE SUMMARY ---
    summary_rows = []
    total_lifespan_per_agent = {aid: int((death_mask[:, aid] == 1).sum()) for aid in roles}

    # Get all unique event variants from lifespan_summary_df and reward_summary_df for summary columns
    # These variants include helper/beneficiary specifics and 'none'.
    all_variants_in_data = set()
    if not lifespan_summary_df.empty and 'event_variant' in lifespan_summary_df.columns:
      all_variants_in_data.update(lifespan_summary_df['event_variant'].unique())
    if not reward_summary_df.empty and 'event_variant' in reward_summary_df.columns:
      all_variants_in_data.update(reward_summary_df['event_variant'].unique())
    sorted_variants = sorted(list(all_variants_in_data))

    for agent_id in sorted(roles.keys()):
      row_data = {
        'trial_name': run_name, 'agent': agent_id, 'role': roles[agent_id],
        'source': sources.get(agent_id, f"{run_name}_{roles[agent_id]}_{agent_id}"),  # Use .get for safety
        'lifespan_total': total_lifespan_per_agent.get(agent_id, 0)
      }
      # Add distance correlation scores
      if not distance_corr_df.empty and agent_id in distance_corr_df.index:
        row_data['top_neu_idx_pred_dist'] = distance_corr_df.loc[agent_id, 'best_neuron_idx_pred']
        row_data['top_neu_idx_prey_dist'] = distance_corr_df.loc[agent_id, 'best_neuron_idx_prey']
        row_data['top_neu_corr_pred_score'] = distance_corr_df.loc[agent_id, 'corr_pred']
        row_data['top_neu_corr_prey_score'] = distance_corr_df.loc[agent_id, 'corr_prey']
      else:  # Fill with NaNs if no data
        for k_corr in ['top_neu_idx_pred_dist', 'top_neu_idx_prey_dist', 'top_neu_corr_pred_score',
                       'top_neu_corr_prey_score']:
          row_data[k_corr] = np.nan

      # Add ROC AUC scores
      for mode, auc_df in auc_results_dfs.items():
        if not auc_df.empty and agent_id in auc_df.index:
          for event_name in available_events:  # Iterate through base events used for classification
            short_event_name = event_name.split('_')[0]  # e.g., "apple" from "apple_cooperation_events"
            # Column name in auc_df is a tuple: (event_name, mode_string_from_classification)
            # mode_string_from_classification is 'event_vs_rest', 'event_vs_event', 'event_vs_none'
            column_key_in_auc_df = (event_name, mode)
            summary_col_name = f"{short_event_name}_vs_{mode.split('_')[-1]}"  # e.g. apple_vs_rest

            if column_key_in_auc_df in auc_df.columns:
              row_data[summary_col_name] = auc_df.loc[agent_id, column_key_in_auc_df]
            else:  # If specific event was not classified for this agent/mode
              row_data[summary_col_name] = np.nan
        else:  # Fill with NaNs if no AUC data for this agent/mode
          for event_name in available_events:
            short_event_name = event_name.split('_')[0]
            summary_col_name = f"{short_event_name}_vs_{mode.split('_')[-1]}"
            row_data[summary_col_name] = np.nan

      # Add per-variant lifespan and reward
      for variant_name in sorted_variants:
        # Lifespan
        lifespan_val = np.nan
        if not lifespan_summary_df.empty and agent_id in lifespan_summary_df['agent'].values:
          agent_lifespan_data = lifespan_summary_df[
            (lifespan_summary_df['agent'] == agent_id) & (lifespan_summary_df['event_variant'] == variant_name)]
          if not agent_lifespan_data.empty:
            lifespan_val = int(agent_lifespan_data['lifespan'].iloc[0])
        row_data[f"{variant_name}_lifespan"] = lifespan_val

        # Reward
        reward_val = np.nan
        if not reward_summary_df.empty and agent_id in reward_summary_df['agent'].values:
          agent_reward_data = reward_summary_df[
            (reward_summary_df['agent'] == agent_id) & (reward_summary_df['event_variant'] == variant_name)]
          if not agent_reward_data.empty:
            reward_val = float(agent_reward_data['mean_reward_per_step'].iloc[0])
        row_data[f"{variant_name}_reward"] = reward_val

      summary_rows.append(row_data)

    final_summary_df = pd.DataFrame(summary_rows)
    final_summary_df.to_pickle(summary_out_path)
    log.info(f"Wrote summary for run {run_name} (ckpt {ckpt_num_str}) to {summary_out_path}")

  except FileNotFoundError as e:
    log.error(f"Skipping run {run_name} (ckpt {ckpt_num_str}) due to MISSING FILE: {e}")
  except Exception as e:
    log.error(f"Error processing run {run_name} (ckpt {ckpt_num_str}): {e}", exc_info=True)
    # Optionally, re-raise or save error state if needed for pipeline management
    # raise e
    return  # Stop processing this run on other errors


# ─── V) MAIN & PARALLEL ENTRY ────────────────────────────────────────────────

def main():
  ap = argparse.ArgumentParser(description="Run analysis pipeline on processed experiment data.")
  ap.add_argument('--base_dir', type=str, default='../../results/mix_RF3/',
                  help="Base directory containing checkpoint folders (e.g., '../../results/mix_RF3').")
  ap.add_argument('--ckpt_pattern', type=str, default="mix_RF_ckpt*",
                  help="Pattern to find checkpoint directories within base_dir (e.g., 'ckpt_*', 'model_run_*').")
  ap.add_argument('--event_input_subdir', type=str, default="analysis_results_extended",
                  help="Subdirectory within base_dir for event files (e.g., output of summarize_higher_level_behavior_info.py).")
  ap.add_argument('--ts_input_subdir', type=str, default="analysis_aggregated_states",
                  help="Subdirectory within base_dir for timestep info files (e.g., output of summarize_timestep_data.py).")
  ap.add_argument('--net_input_subdir', type=str, default="analysis_aggregated_states",
                  help="Subdirectory within base_dir for network state files.")
  ap.add_argument('--figure_output_subdir', type=str, default="pipeline_figures",
                  help="Subdirectory within base_dir to save generated figures.")
  ap.add_argument('--summary_output_subdir', type=str, default="pipeline_summaries",
                  help="Subdirectory within base_dir to save generated summary DataFrames.")
  ap.add_argument('--jobs', type=int, default=50, help="Number of parallel jobs for processing runs.")
  ap.add_argument('--ignore_existing', action='store_true', help="Skip processing if summary output already exists.")
  ap.add_argument('--episode_length', type=int, default=1000, help="Default length of an episode for calculations.")

  args = ap.parse_args()

  base_dir_path = Path(args.base_dir)
  event_files_input_dir = base_dir_path / args.event_input_subdir
  timestep_files_input_dir = base_dir_path / args.ts_input_subdir
  network_state_files_input_dir = base_dir_path / args.net_input_subdir

  figure_output_root_dir = base_dir_path / args.figure_output_subdir
  summary_output_root_dir = base_dir_path / args.summary_output_subdir

  figure_output_root_dir.mkdir(parents=True, exist_ok=True)
  summary_output_root_dir.mkdir(parents=True, exist_ok=True)

  processing_tasks = []
  log.info(f"Scanning for runs in {base_dir_path} with pattern {args.ckpt_pattern}...")

  for ckpt_dir in sorted(base_dir_path.glob(args.ckpt_pattern)):
    if not ckpt_dir.is_dir():
      continue

    ckpt_match = re.search(r"ckpt_?(\d+)", ckpt_dir.name)  # Flexible ckpt number extraction
    if not ckpt_match:
      log.warning(f"Could not parse checkpoint number from directory {ckpt_dir.name}. Skipping.")
      continue
    ckpt_number_str = ckpt_match.group(1)

    for run_dir in sorted(ckpt_dir.iterdir()):
      if run_dir.is_dir():
        current_run_name = run_dir.name

        # Basic check for existence of key input files before adding to task list
        # This makes assumptions about naming conventions from previous scripts
        # expected_event_f = event_files_input_dir / f"{current_run_name}_higher_level_metrics.pkl"
        # expected_ts_f = timestep_files_input_dir / f"ckpt_{ckpt_number_str}_{current_run_name}_info.csv"
        # expected_net_f = network_state_files_input_dir / f"ckpt_{ckpt_number_str}_{current_run_name}_network_states.pkl"

        type_2_naming = True
        if type_2_naming:
          expected_event_f = event_files_input_dir / f"ckpt_{ckpt_number_str}_higher_level_metrics.pkl"
          expected_ts_f = timestep_files_input_dir / f"ckpt{ckpt_number_str}_{current_run_name}_info.csv"
          expected_net_f = network_state_files_input_dir / f"ckpt{ckpt_number_str}_{current_run_name}_network_states.pkl"

        if not expected_event_f.exists():
          log.debug(f"Event file missing for {current_run_name} (ckpt {ckpt_number_str}): {expected_event_f}")
          continue
        if not expected_ts_f.exists():
          log.debug(f"Timestep file missing for {current_run_name} (ckpt {ckpt_number_str}): {expected_ts_f}")
          continue
        if not expected_net_f.exists():
          log.debug(f"Network state file missing for {current_run_name} (ckpt {ckpt_number_str}): {expected_net_f}")
          continue

        processing_tasks.append(
          delayed(process_one)(
            current_run_name,
            ckpt_number_str,
            event_files_input_dir,
            timestep_files_input_dir,
            network_state_files_input_dir,
            figure_output_root_dir,
            summary_output_root_dir,
            args.ignore_existing,
            args.episode_length,
            type_2_naming=type_2_naming,
          )
        )

  if not processing_tasks:
    log.info("No valid runs found to process based on existing input files.")
    return

  log.info(f"Collected {len(processing_tasks)} runs to process.")
  if args.jobs > 1 and len(processing_tasks) > 1:
    log.info(f"Starting parallel processing with {args.jobs} jobs...")
    Parallel(n_jobs=args.jobs)(processing_tasks)
  else:
    log.info("Starting sequential processing...")
    for task_fn in tqdm(processing_tasks, desc="Processing runs"):
      task_fn()  # Execute the delayed function directly

  log.info("Analysis pipeline finished.")


if __name__ == "__main__":
  main()
