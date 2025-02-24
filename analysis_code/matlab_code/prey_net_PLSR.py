import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, r2_score
from sklearn.cross_decomposition import PLSRegression
import pickle
import tempfile
import time
from numba import jit
# suppress warning
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['PYTHONWARNINGS'] = 'ignore'


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

if __name__ == '__main__':
  video_path = os.path.join(os.environ["HOME"], "Documents", "GitHub", "meltingpot-2.2.0", "examples", "videos", "open_field_1_1")

  predator_ids = range(5)
  prey_ids = range(9)
  # kf = KFold(n_splits=10, shuffle=True, random_state=42)
  kf = KFold(n_splits=10)
  num_perm = 50
  # permutations = [0]

  for prey_id in prey_ids:
    bv_prey_df, net_prey_df = pd.DataFrame(), pd.DataFrame()

    # Concatenate data from multiple files
    for predator_id in predator_ids:
      bv_path = os.path.join(video_path, f"{predator_id}_{prey_id}_info.csv")
      net_path = os.path.join(video_path, f"{predator_id}_{prey_id}_network_states.csv")
      bv_prey_df = pd.concat([bv_prey_df, pd.read_csv(bv_path)])
      net_prey_df = pd.concat([net_prey_df, pd.read_csv(net_path)])
    bv_prey_df.reset_index(drop=True, inplace=True)
    net_prey_df.reset_index(drop=True, inplace=True)

    # Simple action_cols, orientation_cols, and stamina_cols
    predator_bv_cols = [f'actions_0_{i}' for i in range(8)] + [f'orientations_0_{i}' for i in range(4)] + ['STAMINA_0',]
    prey_bv_cols = [f'actions_1_{i}' for i in range(8)] + [f'orientations_1_{i}' for i in range(4)] + ['STAMINA_1',]

    # Normalize network state data
    scaler = StandardScaler()
    net_prey_df = pd.DataFrame(scaler.fit_transform(net_prey_df), columns=net_prey_df.columns)
    net_prey_df = net_prey_df.filter(regex='lstmMemory_1')
    bv_prey_df = bv_prey_df.filter(predator_bv_cols + prey_bv_cols)
    bv_prey_df = pd.DataFrame(scaler.fit_transform(bv_prey_df), columns=bv_prey_df.columns)

    # PLSR regression to fit the net_prey_df
    pls = PLSRegression(n_components=26)
    pls.fit(bv_prey_df, net_prey_df)
    net_pred = pls.predict(bv_prey_df)
    # Calculate only the diagonal correlation
    covariance, correlation = compute_diagonal_covariance_and_correlation(net_prey_df.to_numpy(), net_pred)

    # Calculate explained variance for X and Y
    X_scaled = bv_prey_df.to_numpy()
    Y_scaled = net_prey_df.to_numpy()
    X_scores = pls.transform(X_scaled)
    Y_scores = pls.predict(X_scaled)

    explained_variance_X = np.var(X_scores, axis=0) / np.var(X_scaled, axis=0).sum()
    explained_variance_Y = np.var(Y_scores, axis=0) / np.var(Y_scaled, axis=0).sum()

    print("Explained variance in X by each component:", explained_variance_X)
    print("Explained variance in Y by each component:", explained_variance_Y)
    # r2 = pls.score(bv_prey_df, net_prey_df)
    r2 = r2_score(net_prey_df, net_pred)
    # Get the weights
    weights = pls.x_weights_
