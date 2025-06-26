import argparse
import pickle
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
def load_data(data_path):
  return pd.read_pickle(data_path)

def zscore_in_time(hidden):
  """Z-score each neuron across time."""
  return StandardScaler().fit_transform(hidden)

def build_regressors(data, keys):
  """Build design matrix for regression with given behavior keys."""
  regs, names = [], []
  for key in keys:
    arr = np.stack([d[key] for d in data], axis=0)
    if key == 'actions':
      # Rebuild digits to onehot
      oh_pred = np.eye(8, dtype=int)[arr[:,0]]  # [T,8]
      oh_prey = np.eye(8, dtype=int)[arr[:,1]]  # [T,8]
      arr = zscore_in_time(np.concatenate([oh_pred, oh_prey], axis=1))

    elif arr.ndim == 1:
      arr = arr.reshape(-1,1)
    elif arr.ndim > 2:
      arr = arr.reshape(arr.shape[0], -1)
    regs.append(arr)
    dim = arr.shape[1]
    for j in range(dim):
      names.append(key if dim==1 else f"{key}_{j}")
  design = np.concatenate(regs, axis=1)
  return design, names

def compute_tuning_weights(hidden, design):
  """Linear regression per neuron: y = design * w + c"""
  T, D = hidden.shape
  K = design.shape[1]
  weights = np.zeros((D, K))
  model = LinearRegression()
  for i in range(D):
    model.fit(design, hidden[:, i])
    weights[i, :] = model.coef_
  return weights

def select_top_neurons(weights, names, behavior_keys, top_n=100):
  """Select top_n neurons by max |weight| across specified behavior keys."""
  # find indices of columns matching any key prefix
  cols = [i for i, nm in enumerate(names) if any(nm.startswith(key) for key in behavior_keys)]
  # compute max abs weight for each neuron over those cols
  scores = np.max(np.abs(weights[:, cols]), axis=1)
  # sort descending and take top_n
  order = np.argsort(-scores)[:top_n]
  return order

def compute_clusters(z_hidden, order, n_clusters):
  """Cluster only the selected neurons, ordering them by cluster."""
  # features: selected_neurons × time
  feat = z_hidden[:, order].T
  clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
  labels = clusterer.fit_predict(feat)
  # order within selected by cluster label
  intra = np.argsort(labels)
  return labels, order[intra]

def overlay_video(
    video_path,
    pred_z, prey_z,
    order_pred, order_prey,
    actions_pred, actions_prey,
    win, output_path,
    neuron_map_width=10,
    margin_width=60,
):
  """
  Overlay per-frame: 16×16 grids for pred/prey, PC1 trace, action raster,
  laid out via subplot_mosaic.
  """
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS) * 2
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (w*2+margin_width, h))

  T, D = pred_z.shape
  width = 2*win + 1

  # PCA for PC1
  pca_pred = PCA(n_components=1).fit(pred_z)
  pca_prey = PCA(n_components=1).fit(prey_z)

  action_names = ['NOOP','Forward','STEP_RIGHT','BACKWARD',
                  'STEP_LEFT','TURN_LEFT','TURN_RIGHT','INTERACT']

  t=0
  while t<200:
    t+=1
    ret, frame = cap.read()
    if not ret:
      break

    # current index
    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    idx = min(max(idx, 0), T-1)

    # grid data
    vals_pred = pred_z[idx, order_pred]
    vals_prey = prey_z[idx, order_prey]
    grid_pred = vals_pred.reshape(neuron_map_width,-1)
    grid_prey = vals_prey.reshape(neuron_map_width,-1)

    # PC1 window
    lo = max(0, idx - win)
    hi = min(T, idx + win + 1)
    x = np.arange(lo, hi) - idx
    pc1_pred = pca_pred.transform(pred_z[lo:hi]).flatten()
    pc1_prey = pca_prey.transform(prey_z[lo:hi]).flatten()

    # actions
    act_p = actions_pred[lo:hi]
    act_pre = actions_prey[lo:hi]

    # create mosaic
    mosaic = [
      ['pred', 'prey'],
      ['pred', 'prey'],
      ['pc', 'pc'],
      ['act', 'act']
    ]
    fig, axs = plt.subplot_mosaic(mosaic, figsize=(8, 8), constrained_layout=True, dpi=100)

    # pred grid
    ax = axs['pred']
    im = ax.pcolor(grid_pred, cmap='gray', vmin=-2, vmax=2)
    ax.set_title('Pred Neurons')
    ax.axis('off')

    # prey grid
    ax = axs['prey']
    im = ax.pcolor(grid_prey, cmap='gray', vmin=-2, vmax=2)
    ax.set_title('Prey Neurons')
    ax.axis('off')

    # PC1 trace
    ax = axs['pc']
    ax.plot(x, pc1_pred, label='Pred PC1')
    ax.plot(x, pc1_prey, label='Prey PC1')
    ax.axvline(0, color='k', linestyle='--')
    ax.set_xlim(-win, win)
    ax.set_ylabel('PC1')
    ax.legend(fontsize='x-small', loc='upper right')
    ax.set_xticks([])

    # action raster
    ax = axs['act']
    ax.scatter(x, act_p, marker='|', label='Pred')
    ax.scatter(x, act_pre, marker='o', label='Prey')
    ax.axvline(0, color='k', linestyle='--')
    ax.set_xlim(-win, win)
    ax.set_ylim(-0.5, len(action_names)-0.5)
    ax.set_yticks(range(len(action_names)))
    ax.set_yticklabels(action_names, fontsize='xx-small')
    ax.set_xlabel('Time Offset')
    ax.legend(fontsize='x-small', loc='upper right')

    # render
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    ih, iw = fig.canvas.get_width_height()[::-1]
    overlay = buf.reshape((ih, iw, 3))
    plt.close(fig)

    # resize to video
    overlay = cv2.resize(overlay, (w, h))
    # White margin
    margin = np.ones((h, margin_width, 3), dtype=np.uint8) * 255
    combo = np.hstack((overlay, margin, frame))
    out.write(combo)

  cap.release()
  out.release()

def main():
  p = argparse.ArgumentParser(description="Overlay multi-panel neural and behavior plots on video")
  p.add_argument('--video_path', type=str, required=False,
                 default='/home/mikan/e/GitHub/social-agents-JAX/recordings/meltingpot_old/mix_OR20250305_pred_0_OP20241126ckp9651_prey_7_predator_prey__simplified10x10_OneVsOne_None/17.mp4')
  p.add_argument('--data_path', type=str, required=False,
                 default='/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4_old/mix_OR20250305_pred_0_OP20241126ckp9651_prey_7predator_prey__simplified10x10_OneVsOne_agent_0_7/episode_pickles/predator_prey__simplified10x10_OneVsOne_episode_17.pkl')
  p.add_argument('--output_path', default='snapshot_overlay.mp4')
  p.add_argument('--win', type=int, default=50)
  p.add_argument('--n_clusters', type=int, default=8)
  args = p.parse_args()

  data = load_data(args.data_path)
  pred_hidden = np.array([d['hidden'][0] for d in data])
  prey_hidden = np.array([d['hidden'][1] for d in data])
  actions = np.array([d['actions'] for d in data])
  actions_pred = actions[:,0]
  actions_prey = actions[:,1]

  # z-score
  pred_z = zscore_in_time(pred_hidden)
  prey_z = zscore_in_time(prey_hidden)

  # clustering
  # labels_pred, order_pred = compute_clusters(pred_z, args.n_clusters)
  # labels_prey, order_prey = compute_clusters(prey_z, args.n_clusters)


  # build regressors and weights
  behavior_keys = ['STAMINA','POSITION','ORIENTATION','actions', 'rewards']
  design, names = build_regressors(data, behavior_keys)
  w_pred = compute_tuning_weights(pred_z, design)
  w_prey = compute_tuning_weights(prey_z, design)

  # select top100 and cluster within
  top_pred = select_top_neurons(w_pred, names, behavior_keys, top_n=121)
  _, order_pred = compute_clusters(pred_z, top_pred, args.n_clusters)
  top_prey = select_top_neurons(w_prey, names, behavior_keys, top_n=121)
  _, order_prey = compute_clusters(prey_z, top_prey, args.n_clusters)

  overlay_video(
    args.video_path,
    pred_z, prey_z,
    order_pred, order_prey,
    actions_pred, actions_prey,
    args.win, args.output_path,
    neuron_map_width=11,
  )
  print(f"Wrote {args.output_path}")

if __name__=='__main__':
  main()
