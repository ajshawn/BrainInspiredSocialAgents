import os
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

def remove_top_k_plscs(X, U, k=10):
  """
  Removes the top k PLS components from X using U[:, :k].
  If X has shape (n_samples, d), U has shape (d, rank).
  Then:
      X_null = X - (X @ U_k) @ U_k.T
  """
  # handle edge case if rank < k
  dU = U.shape[1]
  k = min(k, dU)
  U_k = U[:, :k]  # top k columns
  X_recon_topk = (X @ U_k) @ U_k.T
  X_null = X - X_recon_topk
  return X_null


def time_permute(df: np.ndarray) -> np.ndarray:
  """
  Replicates the MATLAB timePermute(df) function:
    1) 'df' is shape (time, cells).
    2) If time < cells:
         frameShift = randi(time, cells)
       else:
         frameShift = randperm(time, cells)
    3) For each column, perform a circular shift by frameShift[col].

  Returns a new array 'shiftDf' with the same shape as 'df'.
  """

  time, cells = df.shape
  shiftDf = np.zeros_like(df)

  if time < cells:
    # In MATLAB: frameShift = randi(time, cells)
    # => random integers in [1..time], one per column
    frameShift = np.random.randint(1, time + 1, size=cells)
  else:
    # In MATLAB: frameShift = randperm(time, cells)
    # => 'cells' distinct integers in [1..time]
    frameShift = np.random.choice(np.arange(1, time + 1), size=cells, replace=False)

  # Circularly shift each column
  for col in range(cells):
    shiftDf[:, col] = np.roll(df[:, col], shift=frameShift[col], axis=0)

  return shiftDf

def plsc_decomposition(h1, h2):
  """
  Basic PLSC via SVD of Cov(h1,h2).
  Returns:
    U_shared: shape (D1, rank)
    s: singular values
    V_shared: shape (D2, rank)
  """
  n_samples = h1.shape[0]
  covMat = (h1.T @ h2) / (n_samples - 1)
  U, s, Vt = np.linalg.svd(covMat, full_matrices=False)
  return U, s, Vt

def get_Urand_Vrand_from_concat(
    trial_directory, title, episodes, timesteps=1000, topPLSdim=10
):
  """
  1) Concatenate h1/h2 from all episodes into h1_concat, h2_concat.
  2) Z-score.
  3) PLS => U_shared, V_shared. Remove top 'topPLSdim' from each.
  4) timePermute h1_null, h2_null, do PCA => U_rand, V_rand.
  Return U_rand, V_rand.
  """

  # 1) Concatenate
  h1_concat = []
  h2_concat = []
  for eId in episodes:
    file_path = os.path.join(trial_directory, f"{title}_{eId}.pkl")
    if not os.path.isfile(file_path):
      continue
    with open(file_path, 'rb') as f:
      data = pickle.load(f)
    # each data is presumably a list of dicts: data[t]['hidden'][0 or 1]
    h1 = [d['hidden'][0] for d in data]
    h2 = [d['hidden'][1] for d in data]

    h1 = np.array(h1)[:timesteps]
    h2 = np.array(h2)[:timesteps]

    h1_concat.append(h1)
    h2_concat.append(h2)

  if len(h1_concat) == 0:
    print(f"No data found for {title}")
    return None, None

  h1_concat = np.vstack(h1_concat)
  h2_concat = np.vstack(h2_concat)

  # 2) Z-score each
  scaler1 = StandardScaler()
  scaler2 = StandardScaler()
  h1_concat = scaler1.fit_transform(h1_concat)
  h2_concat = scaler2.fit_transform(h2_concat)

  # 3) PLS => remove top dims
  U, s, Vt = plsc_decomposition(h1_concat, h2_concat)  # shapes: (d1, r), (r,), (d2, r)
  V = Vt.T  # shape (d2, r)

  # remove top k from h1, h2
  h1_null = remove_top_k_plscs(h1_concat, U, topPLSdim)
  h2_null = remove_top_k_plscs(h2_concat, V, topPLSdim)

  # 4) timePermute => PCA
  h1_perm = time_permute(h1_null)
  h2_perm = time_permute(h2_null)

  # We'll do PCA via sklearn
  pca1 = PCA()
  pca2 = PCA()

  pca1.fit(h1_perm)  # shape (n_samples, d1)
  pca2.fit(h2_perm)  # shape (n_samples, d2)

  U_rand = pca1.components_.T  # shape (d1, d1) typically
  V_rand = pca2.components_.T  # shape (d2, d2)

  return U_rand, V_rand

# -------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
  num_top_plscs_to_remove = 10
  run_dict = {
    # 'open_cp9651':{
    #   'video_path': '/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles/',
    #   'file_name': f'randPC_remTop{num_top_plscs_to_remove}PLSCs.pkl',
    #   'predator_ids': list(range(3)),
    #   'prey_ids': list(range(3, 13)),
    # },
    # 'open_cp7357':{
    #   'video_path': '/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles/',
    #   'file_name': f'randPC_remTop{num_top_plscs_to_remove}PLSCs.pkl',
    #   'predator_ids': list(range(3)),
    #   'prey_ids': list(range(3, 13)),
    # },
    # 'alley_hunt':{
    #   'video_path': '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles/',
    #   'file_name': f'randPC_remTop{num_top_plscs_to_remove}PLSCs.pkl',
    #   'predator_ids': list(range(5)),
    #   'prey_ids': list(range(5, 13)),
    # },
    # 'alley_hunt256':{
    #   'video_path': '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles/',
    #   'file_name': f'randPC_remTop{num_top_plscs_to_remove}PLSCs.pkl',
    #   'predator_ids': list(range(5)),
    #   'prey_ids': list(range(5, 13)),
    # },
    # "orchard256": {
    #   'video_path': '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles/',
    #   'file_name': f'randPC_remTop{num_top_plscs_to_remove}PLSCs.pkl',
    #   'predator_ids': list(range(5)),
    #   'prey_ids': list(range(5, 13)),
    # },
    "open256": {
      'video_path': '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles/',
      'file_name': f'randPC_remTop{num_top_plscs_to_remove}PLSCs.pkl',
      'predator_ids': list(range(3)),
      'prey_ids': list(range(3, 13)),
    },
  }
  for run_name, run_info in run_dict.items():
    video_path = run_info["video_path"]
    file_name = run_info["file_name"]
    predator_ids = run_info["predator_ids"]
    prey_ids = run_info["prey_ids"]

    titles = [
      f"{predator_id}_{prey_id}"
      for predator_id in predator_ids
      for prey_id in prey_ids
    ]

    # PARALLEL process
    results = Parallel(n_jobs=25)(
      delayed(get_Urand_Vrand_from_concat)(
        trial_directory=video_path,
        title=title,
        episodes=range(1, 100 + 1),
        timesteps=1000,
        topPLSdim=10,  # remove top 10 PLS dims
      )
      for title in titles
    )

    # Gather into a dict: {title: {"U_rand":..., "V_rand":...}}
    result_dict = {}
    for title, (U_rand, V_rand) in zip(titles, results):
      if U_rand is None or V_rand is None:
        continue
      result_dict[title] = {
        "U": U_rand,
        "V": V_rand
      }

    # Save to disk
    out_file = os.path.join(video_path, file_name)
    with open(out_file, "wb") as f:
      pickle.dump(result_dict, f)

    print(f"[{run_name}] => Saved {len(result_dict)} pairs to {out_file}")