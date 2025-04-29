import os
import jax.numpy as jnp
import jax.random as jrandom
import pickle
from jax import jit, vmap, lax
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from tqdm import tqdm
from helper import mark_death_periods
from jax.config import config

config.update("jax_enable_x64", False)


@jit  # JAX JIT compilation
def PLSC(h1, h2):
  h1_cont = jnp.asarray(h1)
  h2_cont = jnp.asarray(h2)
  n_samples = h1.shape[0]
  covMat = jnp.dot(h1_cont.T, h2_cont) / (n_samples - 1)
  U, s, Vh = jnp.linalg.svd(covMat, full_matrices=False)
  A = jnp.dot(h1_cont, U)
  B = jnp.dot(h2_cont, Vh.T)  # Transpose Vh to get V
  return A, B


@jit  # JAX JIT compilation
def compute_diagonal_covariance_and_correlation(A, B):
  if A.shape != B.shape:
    raise ValueError("Matrices A and B must have the same dimensions.")

  covariance = jnp.mean(A * B, axis=0)  # Element-wise multiplication and mean
  std_A = jnp.std(A, axis=0)
  std_B = jnp.std(B, axis=0)
  correlation = covariance / (std_A * std_B)
  return covariance, correlation


def isExceedingConfidence_linear_percentile(shuffle, pt, confidence=0.95):
  shuffle = jnp.asarray(shuffle)  # Convert to JAX array
  pt = jnp.asarray(pt)  # Ensure pt is a JAX array

  low = (1 - confidence) / 2 * 100
  low_bound, high_bound = jnp.nanpercentile(shuffle, jnp.array([low, 100 - low]), axis=1)

  pos_exceed_confidence = pt >= high_bound
  neg_exceed_confidence = pt <= low_bound
  significant_pts = 1 * pos_exceed_confidence - 1 * neg_exceed_confidence

  return significant_pts


scaler = StandardScaler()


def process_episode(trial_directory, title, eId, timesteps=1000, num_perm=10, permutations=None, scaler=scaler,
                    removing_death_period=None, **kwargs):
  file_path = os.path.join(trial_directory, f'{title}_{eId}.pkl')
  if not os.path.exists(file_path):
    return None

  with open(file_path, 'rb') as f:
    data = pickle.load(f)

  h1 = jnp.array([tmp['hidden'][0] for tmp in data])[:timesteps]
  h2 = jnp.array([tmp['hidden'][1] for tmp in data])[:timesteps]

  if removing_death_period:
    stamina = jnp.array([tmp['STAMINA'][1] for tmp in data])[:timesteps]
    living_period = jnp.array(mark_death_periods(stamina)).astype(bool)
    h1 = h1[living_period]
    h2 = h2[living_period]

  h1 = scaler.fit_transform(h1)
  h2 = scaler.fit_transform(h2)

  try:
    A, B = PLSC(h1, h2)
    cov_diag, cor_diag = compute_diagonal_covariance_and_correlation(A, B)
  except:
    return None

  if permutations is None:
    key = jrandom.PRNGKey(42)  # Seed for reproducibility
    keys = jrandom.split(key, num_perm)
    permutations = jrandom.randint(keys[0], (num_perm,), 0, A.shape[0])

  @jit
  def permuted_PLSC(perm):
    h2_perm = jnp.roll(h2, perm, axis=0)
    A_perm, B_perm = PLSC(h1, h2_perm)
    return compute_diagonal_covariance_and_correlation(A_perm, B_perm)

  cov_perm_array, cor_perm_array = vmap(permuted_PLSC)(permutations)

  cov_sig = isExceedingConfidence_linear_percentile(cov_perm_array.T, cov_diag) > 0
  cor_sig = isExceedingConfidence_linear_percentile(cor_perm_array.T, cor_diag) > 0
  rank_cov = jnp.where(cov_sig == 0, jnp.arange(len(cov_sig)), len(cov_sig)).min()
  rank_cor = jnp.where(cor_sig == 0, jnp.arange(len(cor_sig)), len(cor_sig)).min()
  rank = jnp.min(jnp.array([rank_cov, rank_cor]))

  return rank, cov_diag, cor_diag, cov_perm_array, cor_perm_array, cov_sig, cor_sig


if __name__ == '__main__':
  video_paths = [
    # f'/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles/',
    # f'/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles_perturb_predator/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles_perturb_predator/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_predator/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles_perturb_prey/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles_perturb_prey/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_prey/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles_perturb_both/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles_perturb_both/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_both/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles_perturb_prey/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles_perturb_predator/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles_perturb_both/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles/',
    '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles_perturb_prey/',
    '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles_perturb_predator/',
    '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles_perturb_both/',
    f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_predator_25randPC_remTop10PLSCs/',
    f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_prey_25randPC_remTop10PLSCs/',
    f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_both_25randPC_remTop10PLSCs/',
  ]
  removing_death_period = True

  for video_path in video_paths:
    titles = [f'{predator_id}_{prey_id}' for predator_id in range(3) for prey_id in range(3, 13)]

    results = []
    for eId in tqdm(range(1, 101)):
      for title in titles:
        result = process_episode(
          trial_directory=video_path, title=title, eId=eId, timesteps=1000, num_perm=100, scaler=scaler,
          removing_death_period=removing_death_period
        )
        if result is not None:
          results.append((title, eId, result))

    result_dict = {title: [] for title in titles}
    for title, eId, result in results:
      rank, cov_diag, cor_diag, cov_perm_array, cor_perm_array, cov_sig, cor_sig = result
      result_dict[title].append((rank, cov_diag, cor_diag, cov_perm_array, cor_perm_array, cov_sig, cor_sig))

    with open(f'{video_path}PLSC_results_dict.pkl', 'wb') as f:
      pickle.dump(result_dict, f)
