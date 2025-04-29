import os
import numpy as np
import pickle
from numba import jit
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

@jit(nopython=True)  # Use Numba decorator to compile this function to machine code
def PLSC(h1, h2):
  # Calculate covariance matrix
  h1_cont = np.ascontiguousarray(h1)
  h2_cont = np.ascontiguousarray(h2)
  n_samples = h1.shape[0]
  covMat = np.dot(h1_cont.T, h2_cont) / (n_samples - 1)
  U, s, Vh = np.linalg.svd(covMat, full_matrices=False)
  # Compute transformed matrices
  A = np.dot(h1_cont, U)
  B = np.dot(h2_cont, Vh.T)  # Transpose Vh to get V
  return A, B

@jit(nopython=True)  # Use Numba decorator to compile this function to machine code
def PLSC_decom(h1, h2):
  # Calculate covariance matrix
  h1_cont = np.ascontiguousarray(h1)
  h2_cont = np.ascontiguousarray(h2)
  n_samples = h1.shape[0]
  covMat = np.dot(h1_cont.T, h2_cont) / (n_samples - 1)
  U, s, Vh = np.linalg.svd(covMat, full_matrices=False)
  return U, s, Vh

@jit(nopython=True)  # Use Numba decorator to compile this function to machine code
def compute_diagonal_covariance_and_correlation(A, B):
  if A.shape != B.shape:
    raise ValueError("Matrices A and B must have the same dimensions.")

  n, d = A.shape
  covariance = np.zeros(d)
  std_A = np.zeros(d)
  std_B = np.zeros(d)

  # Compute covariance and standard deviations manually
  for i in range(d):
    for j in range(n):
      covariance[i] += (A[j, i] * B[j, i])
      std_A[i] += A[j, i] ** 2
      std_B[i] += B[j, i] ** 2

    covariance[i] /= (n - 1)
    std_A[i] = np.sqrt(std_A[i] / n)
    std_B[i] = np.sqrt(std_B[i] / n)

  correlation = covariance / (std_A * std_B)

  return covariance, correlation

def isExceedingConfidence_linear_percentile(shuffle, pt, confidence=0.95):
  # data is in the shape of (nNeurons, n_shuffles)
  # pt is in the shape of (nNeurons)
  # return 1 for > right outreach, -1 for < left outreach, 0 for within bounds
  low = (1-confidence) / 2 * 100 # get the percentile required by the confidence interval
  low_bound, high_bound = np.nanpercentile(shuffle, [low, 100 - low], axis=1)
  pos_exceed_confidence = pt >= high_bound
  neg_exceed_confidence = pt <= low_bound
  significant_pts = 1 * pos_exceed_confidence - 1 * neg_exceed_confidence
  return significant_pts


scaler = StandardScaler()
def process_episode(trial_directory, title, eId, timesteps=1000, num_perm=10, permutations=None, scaler=scaler,
                    **kwargs):
  """Function to process a single episode."""

  with open(os.path.join(trial_directory, f'{title}_{eId}.pkl'), 'rb') as f:
    data = pickle.load(f)
  h1 = [tmp['hidden'][0] for tmp in data]
  h2 = [tmp['hidden'][1] for tmp in data]
  h1 = np.array(h1)[:timesteps]
  h2 = np.array(h2)[:timesteps]
  # data_path = f'{trial_directory}out_files/behavior_output_{checkpoint}iters_test_{eId + 1}{file_suffix}'
  # data = sio.loadmat(data_path)
  # h1 = data['h1'][:timesteps]
  # h2 = data['h2'][:timesteps]

  # Normalize h1 and h2
  h1 = scaler.fit_transform(h1)
  h2 = scaler.fit_transform(h2)
  # Below is less flexible with exception like std=0
  # h1 = (h1 - np.mean(h1, axis=0)) / (np.std(h1, axis=0))
  # h2 = (h2 - np.mean(h2, axis=0)) / (np.std(h2, axis=0))

  # A, B = pls.fit_transform(h1, h2)
  try:
    A,B = PLSC(h1, h2)
    cov_diag, cor_diag = compute_diagonal_covariance_and_correlation(A, B)
  except:
    nan_array = np.zeros(h1.shape[-1]) * np.nan
    perm_nan_array = np.zeros((num_perm, h1.shape[-1])) * np.nan
    return np.nan, nan_array, nan_array, perm_nan_array, perm_nan_array, nan_array, nan_array
  if permutations is None:
    permutations = np.random.randint(low=0, high=A.shape[0], size=num_perm)
  cov_perm_array = np.zeros((num_perm, A.shape[1])) * np.nan
  cor_perm_array = np.zeros((num_perm, A.shape[1])) * np.nan

  for pi, perm in enumerate(permutations):
    h2_perm = np.roll(h2, perm, axis=0)
    # _, B_perm = pls.fit_transform(h1, h2_perm)
    # cov_diag_perm, cor_diag_perm = compute_diagonal_covariance_and_correlation(A, B_perm)
    # A_perm, B_perm = pls.fit_transform(h1, h2_perm)
    try:
      A_perm, B_perm = PLSC(h1, h2_perm)
      cov_diag_perm, cor_diag_perm = compute_diagonal_covariance_and_correlation(A_perm, B_perm)
      cov_perm_array[pi] = cov_diag_perm
      cor_perm_array[pi] = cor_diag_perm
    except:
      continue
  cov_sig = isExceedingConfidence_linear_percentile(cov_perm_array.T, cov_diag) > 0
  cor_sig = isExceedingConfidence_linear_percentile(cor_perm_array.T, cor_diag) > 0
  rank_cov = np.where(cov_sig == 0)[0][0] if (cov_sig == 0).any() else len(cov_sig)
  rank_cor = np.where(cor_sig == 0)[0][0] if (cor_sig == 0).any() else len(cor_sig)
  rank = np.min([rank_cov, rank_cor])

  return rank, cov_diag, cor_diag, cov_perm_array, cor_perm_array, cov_sig, cor_sig


def processing_episode_wrapper(**kwargs):
  try:
    # Extract the data needed from process_episode
    rank, cov_diag, cor_diag, cov_perm_array, cor_perm_array, cov_sig, cor_sig = process_episode(**kwargs)
    # Return the unpacked results along with the title and episode ID for easy indexing
    return rank, cov_diag, cor_diag, cov_perm_array, cor_perm_array, cov_sig, cor_sig, kwargs['title'], kwargs['eId']
  except Exception as e:
    # Handle potential errors gracefully
    print(f"Error processing episode {kwargs['eId']} for {kwargs['title']}: {str(e)}")
    # Return None or a default value set to maintain structure
    return None, None, None, None, None, None, None, kwargs['title'], kwargs['eId']

def get_plsc_from_concat_episodes(trial_directory, title, episodes, timesteps=1000, scaler=scaler,
                    **kwargs):
  h1_concat = []
  h2_concat = []
  for eId in episodes:
    with open(os.path.join(trial_directory, f'{title}_{eId}.pkl'), 'rb') as f:
      data = pickle.load(f)
    h1 = [tmp['hidden'][0] for tmp in data]
    h2 = [tmp['hidden'][1] for tmp in data]
    h1 = np.array(h1)[:timesteps]
    h2 = np.array(h2)[:timesteps]
    h1_concat.extend(h1)
    h2_concat.extend(h2)
  h1_concat = np.array(h1_concat)
  h2_concat = np.array(h2_concat)
  # Normalize h1 and h2
  h1_concat = scaler.fit_transform(h1_concat)
  h2_concat = scaler.fit_transform(h2_concat)

  try:
    U, s, Vh = PLSC_decom(h1_concat, h2_concat)

  except:
    nan_array = np.zeros(h1_concat.shape[-1], h1_concat.shape[-1]) * np.nan
    return nan_array, nan_array
  return U, s, Vh, title



if __name__ == '__main__':

  run_dict = {
    # 'open_cp9651':{
    #   'video_path': '/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles/',
    #   'file_name': 'PLSC_usv_dict.pkl',
    #   'predator_ids': list(range(3)),
    #   'prey_ids': list(range(3, 13)),
    # },
    # 'open_cp7357':{
    #   'video_path': '/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles/',
    #   'file_name': 'PLSC_usv_dict.pkl',
    #   'predator_ids': list(range(3)),
    #   'prey_ids': list(range(3, 13)),
    # },
    # 'alley_hunt':{
    #   'video_path': '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles/',
    #   'file_name': 'PLSC_usv_dict.pkl',
    #   'predator_ids': list(range(5)),
    #   'prey_ids': list(range(5, 13)),
    # },
    # 'alley_hunt256':{
    #   'video_path': '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles/',
    #   'file_name': 'PLSC_usv_dict.pkl',
    #   'predator_ids': list(range(5)),
    #   'prey_ids': list(range(5, 13)),
    # },
    # "orchard256": {
    #   'video_path': '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles/',
    #   'file_name': 'PLSC_usv_dict.pkl',
    #   'predator_ids': list(range(5)),
    #   'prey_ids': list(range(5, 13)),
    # },
    "open256":{
      'video_path': '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles/',
      'file_name': 'PLSC_usv_dict.pkl',
      'predator_ids': list(range(3)),
      'prey_ids': list(range(3, 13)),
    }
  }
  for run_name, run_info in run_dict.items():
    video_path = run_info['video_path']
    file_name = run_info['file_name']
    predator_ids = run_info['predator_ids']
    prey_ids = run_info['prey_ids']
    titles = [f'{predator_id}_{prey_id}' for predator_id in predator_ids for prey_id in prey_ids]
    results = Parallel(n_jobs=50)(delayed(get_plsc_from_concat_episodes)(
      trial_directory=video_path, video_path=video_path, title=title,
      episodes=list(range(1,101)), timesteps=1000, num_perm=200, permutations=None, scaler=scaler
    ) for title in titles)

    result_dict = {}
    for result in results:
      U, s, Vh, title = result
      if title not in result_dict:
        result_dict[title] = {title: [] for title in ['U', 's', 'Vh']}
      result_dict[title]['U'] = U
      result_dict[title]['s'] = s
      result_dict[title]['Vh'] = Vh


    with open(f'{video_path}{file_name}', 'wb') as f:
      pickle.dump(result_dict, f)