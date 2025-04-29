#!/usr/bin/env python3
"""
analyze_mixed_pair_plsc_parallel.py

This script processes mixed rollout pair folders in parallel.
Each folder in
  ~/Documents/GitHub/social-agents-JAX/results/mix/analysis_results
that ends with "_episode_pickles" corresponds to a unique predator–prey pair.
For each folder, all episode pickle files are loaded and concatenated (up to a fixed number of timesteps),
the hidden states for each agent are normalized, and a PLSC decomposition is computed.
The resulting dictionary (keyed by pair folder title) is saved as a pickle file.
"""

import os
import pickle
import numpy as np
from numba import jit
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

scaler = StandardScaler()


@jit(nopython=True)
def PLSC(h1, h2):
  h1_cont = np.ascontiguousarray(h1)
  h2_cont = np.ascontiguousarray(h2)
  n_samples = h1.shape[0]
  covMat = np.dot(h1_cont.T, h2_cont) / (n_samples - 1)
  U, s, Vh = np.linalg.svd(covMat, full_matrices=False)
  A = np.dot(h1_cont, U)
  B = np.dot(h2_cont, Vh.T)
  return A, B


@jit(nopython=True)
def PLSC_decom(h1, h2):
  h1_cont = np.ascontiguousarray(h1)
  h2_cont = np.ascontiguousarray(h2)
  n_samples = h1.shape[0]
  covMat = np.dot(h1_cont.T, h2_cont) / (n_samples - 1)
  U, s, Vh = np.linalg.svd(covMat, full_matrices=False)
  return U, s, Vh


@jit(nopython=True)
def compute_diagonal_covariance_and_correlation(A, B):
  if A.shape != B.shape:
    raise ValueError("Matrices A and B must have the same dimensions.")
  n, d = A.shape
  covariance = np.zeros(d)
  std_A = np.zeros(d)
  std_B = np.zeros(d)
  for i in range(d):
    for j in range(n):
      covariance[i] += A[j, i] * B[j, i]
      std_A[i] += A[j, i] ** 2
      std_B[i] += B[j, i] ** 2
    covariance[i] /= (n - 1)
    std_A[i] = np.sqrt(std_A[i] / n)
    std_B[i] = np.sqrt(std_B[i] / n)
  correlation = covariance / (std_A * std_B)
  return covariance, correlation


def get_plsc_from_concat_episodes_pair(pair_folder, timesteps=1000, num_perm=10, scaler=scaler):
  """
  Processes one pair folder (which contains episode pickle files named like "<pred>_<prey>_<ep>.pkl").
  Loads all episodes (up to the specified number of timesteps), concatenates the hidden state arrays,
  normalizes them, and computes a PLSC decomposition using PLSC_decom.

  Returns: U, s, Vh, pair_title.
  """
  pkl_files = sorted([f for f in os.listdir(pair_folder) if f.endswith('.pkl')])
  if len(pkl_files) == 0:
    print(f"No pickle files found in {pair_folder}")
    return None, None, None, os.path.basename(pair_folder).replace("_episode_pickles", "")

  h1_list = []
  h2_list = []
  for pkl in pkl_files:
    file_path = os.path.join(pair_folder, pkl)
    try:
      with open(file_path, 'rb') as f:
        data = pickle.load(f)
    except Exception as e:
      print(f"Error loading {file_path}: {e}")
      continue
    # Extract hidden states for agent 0 and agent 1 for this episode.
    h1 = [tmp['hidden'][0] for tmp in data]
    h2 = [tmp['hidden'][1] for tmp in data]
    h1 = np.array(h1)[:timesteps]
    h2 = np.array(h2)[:timesteps]
    h1_list.append(h1)
    h2_list.append(h2)

  if len(h1_list) == 0:
    print(f"No valid episodes found in {pair_folder}")
    return None, None, None, os.path.basename(pair_folder).replace("_episode_pickles", "")

  h1_concat = np.concatenate(h1_list, axis=0)
  h2_concat = np.concatenate(h2_list, axis=0)
  h1_concat = scaler.fit_transform(h1_concat)
  h2_concat = scaler.fit_transform(h2_concat)

  try:
    U, s, Vh = PLSC_decom(h1_concat, h2_concat)
  except Exception as e:
    print(f"Error in PLSC_decom for folder {pair_folder}: {e}")
    nan_array = np.full((h1_concat.shape[1],), np.nan)
    return nan_array, nan_array, nan_array, os.path.basename(pair_folder).replace("_episode_pickles", "")

  pair_title = os.path.basename(pair_folder).replace("_episode_pickles", "")
  return U, s, Vh, pair_title


def main():
  # Base directory where reorganized pair folders reside.
  base_dir = os.path.expanduser("~/Documents/GitHub/social-agents-JAX/results/mix")
  # List all folders ending with "_episode_pickles".
  pair_folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
                  if os.path.isdir(os.path.join(base_dir, d)) and d.endswith("_episode_pickles")]

  if not pair_folders:
    print("No pair folders found in", base_dir)
    return

  # Process each pair folder in parallel.
  results = Parallel(n_jobs=-1)(delayed(get_plsc_from_concat_episodes_pair)(folder)
                                for folder in pair_folders)

  plsc_dict = {}
  for res in results:
    U, s, Vh, pair_title = res
    if U is None:
      continue
    plsc_dict[pair_title] = {'U': U, 's': s, 'Vh': Vh}

  out_file = os.path.join(base_dir, "PLSC_usv_dict_mixed.pkl")
  with open(out_file, 'wb') as f:
    pickle.dump(plsc_dict, f)
  print(f"Saved PLSC dictionary to {out_file}")


if __name__ == '__main__':
  main()
