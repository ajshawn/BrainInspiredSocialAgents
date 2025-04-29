import os
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

from typing import List

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU only
# =========================================================
# 1) Helper: defineBehaviorCols (replicating your MATLAB logic)
# =========================================================
def define_behavior_cols(bv_type: str):
  """
  Return (predator_bv_cols, prey_bv_cols) as lists of column names,
  depending on the requested bv_type.
  """
  if bv_type == 'act+ori+sta':
    predator_bv_cols = [f"actions_0_{i}" for i in range(8)] \
                       + [f"orientations_0_{i}" for i in range(4)] \
                       + ["STAMINA_0"]
    prey_bv_cols     = [f"actions_1_{i}" for i in range(8)] \
                       + [f"orientations_1_{i}" for i in range(4)] \
                       + ["STAMINA_1"]
  elif bv_type == 'act_prob':
    predator_bv_cols = [f"actions_prob_0_{i}" for i in range(8)]
    prey_bv_cols     = [f"actions_prob_1_{i}" for i in range(8)]
  else:
    raise ValueError(f"Unrecognized bv_type: {bv_type}")
  return predator_bv_cols, prey_bv_cols


# =========================================================
# 2) mark_death_periods in Python
# =========================================================
def mark_death_periods(data: np.ndarray) -> np.ndarray:
  """
  Python version of your MATLAB function:
    - data is 1D array of 0/1 floats.
    - find 20 consecutive zeros followed by a 1 => mark them as "death" (0).
    - else "living" (1).
  Returns a 1D array of the same shape, with 0 for death, 1 for living.
  """
  data = data.astype(float).ravel()  # ensure 1D
  n = len(data)
  labels = np.ones(n, dtype=int)

  zero_mask = (data == 0.0)
  # Convolve to find 20 consecutive zeros
  window = np.ones(20, dtype=int)
  conv_result = np.convolve(zero_mask.astype(int), window, mode='valid')  # length n-19
  potential_starts = np.where(conv_result == 20)[0]  # these are start indices

  for start_idx in potential_starts:
    check_idx = start_idx + 20
    if check_idx < n and data[check_idx] == 1.0:
      labels[start_idx:check_idx] = 0
  return labels


# =========================================================
# 3) filling_missing_orientation_one_hot_vec in Python
# =========================================================
def filling_missing_orientation_one_hot_vec(df: pd.DataFrame) -> pd.DataFrame:
  """
  If "orientations_X_0..3" columns are missing, but we have an
  "ORIENTATION_X" column, decode it into four binary columns.
  Adjust as needed for your actual column names.
  """
  for agent_id in [0, 1]:
    prefix = f"orientations_{agent_id}_"
    needed_cols = [f"{prefix}{i}" for i in range(4)]
    if not all(col in df.columns for col in needed_cols):
      # maybe we have "ORIENTATION_X"? If so, decode.
      big_col = f"ORIENTATION_{agent_id}"
      if big_col in df.columns:
        orientation_values = df[big_col].values
        orientation_binary = np.zeros((len(df), 4), dtype=int)
        for i in range(4):
          orientation_binary[:, i] = (orientation_values == i).astype(int)

        # create new columns
        new_cols = pd.DataFrame(orientation_binary, columns=needed_cols)
        df = pd.concat([df, new_cols], axis=1)
      else:
        # no orientation data at all => your logic
        pass
  return df


# =========================================================
# 4) tempShift for random/circular shifts
#    We do a direct Python version.
# =========================================================
def temp_shift(df: jnp.ndarray, lag: int) -> jnp.ndarray:
  """
  Circulary shift the rows of df by a random amount between [lag..(n-lag)].
  'df' is a JAX array of shape (time, features).
  Returns the shifted array (same shape).
  """
  n = df.shape[0]
  # pick random shift in [lag..(n-lag)]
  # We'll use jax.random for reproducibility if desired
  # but for simplicity, do a normal python random here:
  import random
  shift_amount = random.randint(lag, n - 1)
  # Use jnp.roll
  return jnp.roll(df, shift_amount, axis=0)


# =========================================================
# 5) JAX-based partial least squares (very rough).
#    We can't replicate MATLAB's plsregress exactly, but we do something similar:
#    - We'll do an SVD-based approach to see how well X's top components
#      can linearly reconstruct Y. Then measure % var explained.
# =========================================================

def jax_pls_var_explained(X: jnp.ndarray, Y: jnp.ndarray) -> float:
  """
  1) We'll do SVD of X^T Y => get top components => project X => see how much variance
     that reconstruction can explain in Y. This is a simplified approach.

  2) Return the sum of variance explained (like sum of pctvar(2,:)) in MATLAB's plsregress.

  Note: This is an approximation to replicate your "pctVar" logic from MATLAB.
  """
  # SVD of Cov(X, Y)
  n = X.shape[0]
  # ensure mean-zero
  Xc = X - X.mean(axis=0, keepdims=True)
  Yc = Y - Y.mean(axis=0, keepdims=True)

  covMat = (Xc.T @ Yc) / (n - 1)  # shape (Dx, Dy)
  U, S, Vt = jnp.linalg.svd(covMat, full_matrices=False)

  # dU = Xc @ U  -> shape (N, rank)
  # We can measure how well that can reconstruct Y:
  # For each dimension, see how correlated it is with Y. This is a simplified approach.
  # We'll do something naive to get a single "var explained" value:

  # Project X onto the left singular vectors
  scores = Xc @ U  # shape (N, rank)
  # We'll do a linear regression: scores => Yc
  # Then measure R^2. Let's do a simple normal eq:
  #   W = (scores^T scores)^-1 scores^T Yc
  #   Yhat = scores @ W
  #   varExplained = 1 - MSE(Yhat, Yc)/Var(Yc)
  # For consistent shape, we do it for each Y column and average.

  # pseudo-inverse part
  XT_X = scores.T @ scores  # shape (rank, rank)
  XT_Y = scores.T @ Yc      # shape (rank, Dy)
  W = jnp.linalg.inv(XT_X) @ XT_Y  # shape (rank, Dy)

  Yhat = scores @ W  # shape (N, Dy)
  # total variance of Yc (like sum of squares)
  SSE = jnp.sum((Yc - Yhat)**2)
  SST = jnp.sum(Yc**2)

  var_expl = 1.0 - (SSE / SST)
  return float(var_expl)


# =========================================================
# 6) computeNonRedundantVar in a JAX style
#    We'll replicate your logic with time shifts for chance level
# =========================================================

def compute_non_redundant_var(X_list: List[np.ndarray],
                              Y: np.ndarray,
                              numVarPermute: int,
                              numShuffle: int) -> List[float]:
  """
  X_list is a list of 2 sets of variables, e.g.:
    X_list[0] -> prey columns
    X_list[1] -> predator columns
  We'll combine them into full X, measure var explained in Y,
  then timeShift each group individually, measure difference, etc.

  Returns: [nonRedundantVar_Group1, nonRedundantVar_Group2]
  """
  # Convert to jax arrays
  X_all = jnp.array( np.hstack(X_list), dtype=jnp.float32 )
  Y_jax = jnp.array(Y, dtype=jnp.float32)

  # 1) "Full" model
  full_obs = jax_pls_var_explained(X_all, Y_jax)

  # Chance for full model
  # We'll do 'numShuffle' time shifts, average them

  def single_shuffle(_):
    X_shifted = temp_shift(X_all, lag=60)
    return jax_pls_var_explained(X_shifted, Y_jax)

  # We can use jax.vmap or a Python loop:
  full_chance_vals = [single_shuffle(i) for i in range(numShuffle)]
  full_chance_avg = float(np.mean(full_chance_vals))

  # 2) For each group, do the "permuted" model
  # In MATLAB, you're doing XTemp{i} = tempShift(XTemp{i}, 60).
  # We'll do that in python:
  nGroups = len(X_list)
  results = []

  for group_idx in range(nGroups):
    # We do "numVarPermute" permutations in MATLAB logic,
    # though your code was a bit condensed. We'll just do 1 or more:
    # but let's do an average across 'numVarPermute' permutations:
    sum_obs = 0.0
    sum_chance = 0.0

    for _ in range(numVarPermute):
      # permute group i
      X_temp = []
      for j, mat in enumerate(X_list):
        mat_jax = jnp.array(mat, dtype=jnp.float32)
        if j == group_idx:
          # shift
          mat_jax = temp_shift(mat_jax, lag=60)
        X_temp.append(mat_jax)
      X_fit = jnp.hstack(X_temp)  # shape (N, D1+D2)

      obs = jax_pls_var_explained(X_fit, Y_jax)
      # chance
      chance_vals = []
      for __ in range(numShuffle):
        X_perm = temp_shift(X_fit, lag=60)
        chance_vals.append(jax_pls_var_explained(X_perm, Y_jax))
      chance_avg = float(np.mean(chance_vals))

      sum_obs += (obs - chance_avg)

    avg_obs_minus_chance = sum_obs / float(numVarPermute)
    # "full model" is also "obs - chance" for the full set:
    full_obs_minus_chance = full_obs - full_chance_avg

    # "non-redundant" = (full obs-chance) - (permuted obs-chance)
    # i.e. how much unique variance group i provides beyond the other group
    non_redundant = full_obs_minus_chance - avg_obs_minus_chance
    results.append(non_redundant)

  return results

# Convert the dataset to CPU for JAX
def convert_to_cpu(structure):
  # Recursively calls np.array on every leaf
  return jax.tree_map(lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, structure)


# =========================================================
# 7) Main run logic replicating your MATLAB structure
# =========================================================
def main():
  video_paths = [
    '/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles/',
    '/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles/',
    '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles/',
    '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles/',
    '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles/',

  ]
  suffices = [
    "cp7357", "cp9651", "AH", "AH256", "Orchard256"
  ]
  bv_types = [
    "act+ori+sta",
    "act_prob"
  ]
  applying_living_period = True

  # If we want to rename suffices to "no_death", just as example:
  if applying_living_period:
    suffices = [s + "_no_death" for s in suffices]

  # Example ID ranges:
  # If 'open'
  #    predator_ids = 0..2, prey_ids = 3..12
  # else
  #    predator_ids = 0..4, prey_ids = 5..12

  for bv_type in bv_types:
    print(f"[INFO] Processing bv_type={bv_type}")

    predator_bv_cols, prey_bv_cols = define_behavior_cols(bv_type)

    for (vid_path, suffix) in zip(video_paths, suffices):
      print(f"==> Video path={vid_path}, suffix={suffix}")

      if "open" in vid_path:
        predator_ids = range(3)   # 0..2
        prey_ids     = range(3, 13)
      else:
        predator_ids = range(5)  # 0..4
        prey_ids     = range(5, 13)

      results = {}

      # Pairwise loop
      for prey_id in prey_ids:
        for predator_id in predator_ids:
          bv_file  = os.path.join(vid_path, f"{predator_id}_{prey_id}_info.csv")
          net_file = os.path.join(vid_path, f"{predator_id}_{prey_id}_network_states.csv")

          if not (os.path.isfile(bv_file) and os.path.isfile(net_file)):
            continue

          # read CSV
          bv_data  = pd.read_csv(bv_file)
          net_data = pd.read_csv(net_file)

          if applying_living_period:
            # mark death from "STAMINA_1"
            if "STAMINA_1" not in bv_data.columns:
              continue
            living_period = mark_death_periods(bv_data["STAMINA_1"].values)
            mask = (living_period == 1)
            bv_data  = bv_data.loc[mask].reset_index(drop=True)
            net_data = net_data.loc[mask].reset_index(drop=True)

          # fill orientation
          bv_data = filling_missing_orientation_one_hot_vec(bv_data)

          # zscore
          net_data_z = (net_data - net_data.mean()) / net_data.std(ddof=1)

          # slice predator / prey hidden states
          predator_cols = [c for c in net_data_z.columns if "hidden_0" in c]
          prey_cols     = [c for c in net_data_z.columns if "hidden_1" in c]
          predator_net_data = net_data_z[predator_cols].values  # shape (N, Dp)
          prey_net_data     = net_data_z[prey_cols].values      # shape (N, Dq)

          # Subset the bv_data
          # In Python, missing columns would cause an error:
          all_bv_cols = predator_bv_cols + prey_bv_cols
          for c in all_bv_cols:
            if c not in bv_data.columns:
              # skip if missing columns
              pass

          bv_data_z = (bv_data[all_bv_cols] - bv_data[all_bv_cols].mean()) / bv_data[all_bv_cols].std(ddof=1)

          # Prey perspective
          Xprey = bv_data_z[prey_bv_cols].values
          Xpred = bv_data_z[predator_bv_cols].values
          # "xx{1}=prey, xx{2}=predator => we measure var in prey_net_data"
          xx = [Xprey, Xpred]
          prey_score = compute_non_redundant_var(xx, prey_net_data, numVarPermute=10, numShuffle=50)

          # Predator perspective
          xx = [Xpred, Xprey]
          predator_score = compute_non_redundant_var(xx, predator_net_data, numVarPermute=10, numShuffle=50)

          # Store
          key_prey     = f"pair_{predator_id}_{prey_id}_prey"
          key_predator = f"pair_{predator_id}_{prey_id}_predator"
          results[key_prey]     = prey_score
          results[key_predator] = predator_score

          print(f"... Done pair pred={predator_id}, prey={prey_id} => {suffix}, {bv_type}")

      # similarly you can do your "combined data" logic (concatenate across multiple predators or preys),
      # then store them in results as well.

      # At the end, you can write out "results" to disk, or produce heatmaps using e.g. matplotlib
      out_file = f"partner_rep_result_{suffix}_{bv_type}.pkl"
      import pickle
      with open(out_file, "wb") as f:
        results = convert_to_cpu(results)
        pickle.dump(results, f)
      print(f"[INFO] Wrote results to {out_file}")

  print("[INFO] All done!")


if __name__ == "__main__":
  main()
