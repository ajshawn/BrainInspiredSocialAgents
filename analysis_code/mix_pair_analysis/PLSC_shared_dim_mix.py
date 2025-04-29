import os
import numpy as np
import pickle
from numba import jit
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
sys.path.append('../')
from helper import mark_death_periods

scaler = StandardScaler()


@jit(nopython=True)
def PLSC(h1, h2):
  # Calculate covariance matrix and perform SVD.
  h1_cont = np.ascontiguousarray(h1)
  h2_cont = np.ascontiguousarray(h2)
  n_samples = h1.shape[0]
  covMat = np.dot(h1_cont.T, h2_cont) / (n_samples - 1)
  U, s, Vh = np.linalg.svd(covMat, full_matrices=False)
  A = np.dot(h1_cont, U)
  B = np.dot(h2_cont, Vh.T)
  return A, B


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


def isExceedingConfidence_linear_percentile(shuffle, pt, confidence=0.95):
  low = (1 - confidence) / 2 * 100
  low_bound, high_bound = np.nanpercentile(shuffle, [low, 100 - low], axis=1)
  pos_exceed_confidence = pt >= high_bound
  neg_exceed_confidence = pt <= low_bound
  significant_pts = 1 * pos_exceed_confidence - 1 * neg_exceed_confidence
  return significant_pts


def process_episode(trial_directory, title, eId, timesteps=1000, num_perm=10, permutations=None, scaler=scaler,
                    removing_death_period=False, **kwargs):
  """Process a single episode."""
  file_path = os.path.join(trial_directory, f'{title}_{eId}.pkl')
  with open(file_path, 'rb') as f:
    data = pickle.load(f)
  h1 = [tmp['hidden'][0] for tmp in data]
  h2 = [tmp['hidden'][1] for tmp in data]
  h1 = np.array(h1)[:timesteps]
  h2 = np.array(h2)[:timesteps]
  if removing_death_period:
    stamina = np.array([tmp['STAMINA'][1] for tmp in data])[:timesteps]
    living_period = mark_death_periods(stamina).astype(bool)
    h1 = h1[living_period]
    h2 = h2[living_period]
  h1 = scaler.fit_transform(h1)
  h2 = scaler.fit_transform(h2)
  try:
    A, B = PLSC(h1, h2)
    cov_diag, cor_diag = compute_diagonal_covariance_and_correlation(A, B)
  except Exception:
    nan_array = np.full(h1.shape[-1], np.nan)
    perm_nan_array = np.full((num_perm, h1.shape[-1]), np.nan)
    return np.nan, nan_array, nan_array, perm_nan_array, perm_nan_array, nan_array, nan_array
  if permutations is None:
    permutations = np.random.randint(0, A.shape[0], size=num_perm)
  cov_perm_array = np.full((num_perm, A.shape[1]), np.nan)
  cor_perm_array = np.full((num_perm, A.shape[1]), np.nan)
  for pi, perm in enumerate(permutations):
    h2_perm = np.roll(h2, perm, axis=0)
    try:
      A_perm, B_perm = PLSC(h1, h2_perm)
      cov_diag_perm, cor_diag_perm = compute_diagonal_covariance_and_correlation(A_perm, B_perm)
      cov_perm_array[pi] = cov_diag_perm
      cor_perm_array[pi] = cor_diag_perm
    except Exception:
      continue
  cov_sig = isExceedingConfidence_linear_percentile(cov_perm_array.T, cov_diag) > 0
  cor_sig = isExceedingConfidence_linear_percentile(cor_perm_array.T, cor_diag) > 0
  rank_cov = np.where(cov_sig == 0)[0][0] if (cov_sig == 0).any() else len(cov_sig)
  rank_cor = np.where(cor_sig == 0)[0][0] if (cor_sig == 0).any() else len(cor_sig)
  rank = np.min([rank_cov, rank_cor])
  return rank, cov_diag, cor_diag, cov_perm_array, cor_perm_array, cov_sig, cor_sig


def processing_episode_wrapper(**kwargs):
  try:
    res = process_episode(**kwargs)
    return res + (kwargs['title'], kwargs['eId'])
  except Exception as e:
    print(f"Error processing episode {kwargs['eId']} for {kwargs['title']}: {str(e)}")
    return (None,) * 7 + (kwargs['title'], kwargs['eId'])


if __name__ == '__main__':
  # For current mixed rollout, assume each pair folder contains episode pickle files.
  # We'll process each pair folder separately.
  base_dir = os.path.expanduser("~/Documents/GitHub/social-agents-JAX/results/mix")
  # Find all pair folders (each ending with "_episode_pickles")
  pair_folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
                  if os.path.isdir(os.path.join(base_dir, d)) and d.endswith("_episode_pickles")]
  # Only process the folder containing OR20250305 or OP20250224ckp6306
  pair_folders = [f for f in pair_folders if 'OR20250305' in f or 'OP20250224ckp6306' in f]
  if not pair_folders:
    print("No pair folders found in", base_dir)
    exit(1)

  # Process for both conditions: with and without removing death periods.
  for removing_death in [True, False]:
    remove_death_str = '_remove_death' if removing_death else ''
    for folder in tqdm(pair_folders, desc=f"Processing all pairs {remove_death_str}"):
      # The common title for episodes is taken from the folder name (without the suffix)
      # title = os.path.basename(folder).replace("_episode_pickles", "")
      title = 'predator_prey__simplified10x10_OneVsOne_episode'
      # Get list of episode files.
      episode_files = sorted([f for f in os.listdir(folder) if f.endswith('.pkl')])
      episode_ids = []
      for fname in episode_files:
        try:
          # Assuming filename format: "<title>_<eId>.pkl"
          eId = int(fname.split('_')[-1].replace('.pkl', ''))
          episode_ids.append(eId)
        except Exception:
          continue
      episode_ids = sorted(episode_ids)
      if len(episode_ids) == 0:
        print(f"No valid episodes found in folder {folder}")
        continue
      # Process episodes in parallel.
      results = Parallel(n_jobs=22)(delayed(processing_episode_wrapper)(
        trial_directory=folder,
        title=title,
        eId=eId,
        timesteps=1000,
        num_perm=200,
        permutations=None,
        scaler=scaler,
        removing_death_period=removing_death
      ) for eId in episode_ids)
      # results = []
      # for eId in tqdm(episode_ids, desc=f"Processing {title}{remove_death_str}"):
      #   res = processing_episode_wrapper(
      #     trial_directory=folder,
      #     title=title,
      #     eId=eId,
      #     timesteps=1000,
      #     num_perm=200,
      #     permutations=None,
      #     scaler=scaler,
      #     removing_death_period=removing_death
      #   )
      #   results.append(res)


      # Aggregate results for this pair folder (each title separately).
      # Instead of a big dict for all pairs, we create one dict per title.
      result_dict = {}
      for res in results:
        rank, cov_diag, cor_diag, cov_perm_array, cor_perm_array, cov_sig, cor_sig, t, eId = res
        if len(result_dict) == 0:
          result_dict = {'rank': [], 'cov': [], 'cor': [], 'cov_sig': [], 'cor_sig': [],
                            'cov_perm_array': [], 'cor_perm_array': []}
        result_dict['rank'].append(rank)
        result_dict['cov'].append(cov_diag)
        result_dict['cor'].append(cor_diag)
        result_dict['cov_sig'].append(cov_sig)
        result_dict['cor_sig'].append(cor_sig)
        result_dict['cov_perm_array'].append(cov_perm_array)
        result_dict['cor_perm_array'].append(cor_perm_array)
      # Now save one file per title in this folder.
      file_name = folder.split('/')[-1].replace('_episode_pickles', '')
      # for t, data in result_dict.items():
      out_file = os.path.join(base_dir, 'analysis_results', f'{file_name}_PLSC_results{remove_death_str}.pkl')
      with open(out_file, 'wb') as f:
        pickle.dump(result_dict, f)
      print(f"Saved results for {t} to {out_file}")
