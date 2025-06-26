import argparse
import pickle
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_data(data_path):
  """Load list of per-timestep dicts from pickle."""
  return pd.read_pickle(data_path)


def build_regressors(data, keys):
  """
  Build design matrix for regression.
  Returns:
    design: np.ndarray of shape [T, K]
    names: list of length K, names for each regressor column
  """
  regs, names = [], []
  for key in keys:
    arrs = [d[key] for d in data]
    arr = np.stack(arrs, axis=0)
    if arr.ndim == 1:
      arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
      arr = arr.reshape(arr.shape[0], -1)
    regs.append(arr)
    dim = arr.shape[1]
    for j in range(dim):
      names.append(key if dim == 1 else f"{key}_{j}")
  design = np.concatenate(regs, axis=1)
  return design, names


def compute_tuning_weights(hidden, design):
  """
  Fit a linear regression for each neuron: y = design * w + c.
  Returns weights: [D, K]
  """
  T, D = hidden.shape
  K = design.shape[1]
  weights = np.zeros((D, K))
  model = LinearRegression()
  for i in range(D):
    model.fit(design, hidden[:, i])
    weights[i, :] = model.coef_
  return weights


def assign_preferred(weights, names):
  """
  For each neuron (row), find the regressor with max |weight|.
  Returns pref_idx, pref_name, pref_w
  """
  pref_idx = np.argmax(np.abs(weights), axis=1)
  pref_w = weights[np.arange(weights.shape[0]), pref_idx]
  pref_name = [names[idx] for idx in pref_idx]
  return pref_idx, pref_name, pref_w


def sort_neurons(pref_idx, pref_w):
  """
  Sort neurons by preferred regressor group, then descending |weight|.
  Returns neuron ordering indices.
  """
  return np.lexsort((-np.abs(pref_w), pref_idx))

def overlay_video(video_path, pred_hidden, prey_hidden,
                  order_pred, order_prey,
                  actions_pred, actions_prey,
                  win, output_path,
                  margin_width=60,):
  """
  Create 5-panel overlay: video, predator heatmap, prey heatmap,
  PC1 traces, and action raster, using RdBu_r colormap.
  Heatmaps are padded so the current time stays centered.
  """
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS) * 2
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (w*2+margin_width, h))

  T = pred_hidden.shape[0]
  frame_idx = 0
  width = 2 * win + 1

  # PCA for PC1
  pca_pred = PCA(n_components=1).fit(pred_hidden)
  pca_prey = PCA(n_components=1).fit(prey_hidden)

  action_names = ['NOOP', 'Forward', 'STEP_RIGHT', 'BACKWARD',
                  'STEP_LEFT', 'TURN_LEFT', 'TURN_RIGHT', 'INTERACT']

  t = 0
  while t<200:
    t+=1
    ret, frame = cap.read()
    if not ret:
      break

    lo = max(0, frame_idx - win)
    hi = min(T, frame_idx + win + 1)
    L = hi - lo

    # raw snippets
    raw_pred = pred_hidden[lo:hi][:, order_pred].T  # [D, L]
    raw_prey = prey_hidden[lo:hi][:, order_prey].T  # [D, L]

    # pad with NaNs to keep center fixed
    pad_pred = np.full((raw_pred.shape[0], width), np.nan)
    pad_prey = np.full((raw_prey.shape[0], width), np.nan)
    start = max(0, win - frame_idx)
    pad_pred[:, start:start + L] = raw_pred
    pad_prey[:, start:start + L] = raw_prey

    # PC1 traces
    pc1_pred = pca_pred.transform(pred_hidden[lo:hi]).flatten()
    pc1_prey = pca_prey.transform(prey_hidden[lo:hi]).flatten()

    # actions
    act_p = actions_pred[lo:hi]
    act_pre = actions_prey[lo:hi]

    # plot panels
    fig, axs = plt.subplots(4, 1, figsize=(8, 8), dpi=100)
    ax1, ax2, ax3, ax4 = axs

    # # 1. video frame
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # ax0.imshow(frame_rgb)
    # ax0.axis('off')

    # 2. predator heatmap
    xedges = np.arange(width + 1) - win
    yedges_pred = np.arange(raw_pred.shape[0] + 1)
    ax1.pcolormesh(xedges, yedges_pred, pad_pred,
                   cmap='RdBu_r', shading='auto')
    ax1.axvline(0, color='k', linestyle='--', linewidth=1)
    ax1.set_ylabel('Pred Neurons')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # 3. prey heatmap
    yedges_prey = np.arange(raw_prey.shape[0] + 1)
    ax2.pcolormesh(xedges, yedges_prey, pad_prey,
                   cmap='RdBu_r', shading='auto')
    ax2.axvline(0, color='k', linestyle='--', linewidth=1)
    ax2.set_ylabel('Prey Neurons')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # 4. PC1 traces
    x = np.arange(L) - (frame_idx - lo)
    ax3.plot(x, pc1_pred, label='Pred PC1')
    ax3.plot(x, pc1_prey, label='Prey PC1')
    ax3.axvline(0, color='k', linestyle='--', linewidth=1)
    ax3.set_xlim(-win, win)
    ax3.set_ylabel('PC1')
    ax3.legend(fontsize='small', ncol=2)
    ax3.set_xticks([])

    # 5. action raster
    ax4.scatter(x, act_p, marker='|', label='Pred')
    ax4.scatter(x, act_pre, marker='o', label='Prey')
    ax4.axvline(0, color='k', linestyle='--', linewidth=1)
    ax4.set_xlim(-win, win)
    ax4.set_ylim(-0.5, len(action_names) - 0.5)
    ax4.set_yticks(range(len(action_names)))
    ax4.set_yticklabels(action_names)
    ax4.set_ylabel('Action')
    ax4.set_xlabel('Time Offset')
    ax4.legend(fontsize='small', loc='upper right')

    fig.tight_layout(pad=0.5, rect=(0,0,0.95,1))
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    ih, iw = fig.canvas.get_width_height()[::-1]
    overlay = buf.reshape((ih, iw, 3))
    plt.close(fig)

    # combine and write
    overlay = cv2.resize(overlay, (w, h))
    margin = np.ones((h, margin_width, 3), dtype=np.uint8) * 255
    combo = np.hstack((overlay, margin, frame))
    out.write(combo)

    frame_idx += 1

  cap.release()
  out.release()


def main():
  parser = argparse.ArgumentParser(description="Overlay multi-panel neural and behavior plots on video")
  parser.add_argument('--video_path', type=str, required=False,
                      default='/home/mikan/e/GitHub/social-agents-JAX/recordings/meltingpot_old/mix_OR20250305_pred_0_OP20241126ckp9651_prey_7_predator_prey__simplified10x10_OneVsOne_None/17.mp4')
  parser.add_argument('--data_path', type=str, required=False,
                      default='/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4_old/mix_OR20250305_pred_0_OP20241126ckp9651_prey_7predator_prey__simplified10x10_OneVsOne_agent_0_7/episode_pickles/predator_prey__simplified10x10_OneVsOne_episode_17.pkl')

  parser.add_argument('--output_path', type=str, default='tuning_overlay_multi.mp4')
  parser.add_argument('--win', type=int, default=50,
                      help='half-window size for snippets')
  args = parser.parse_args()

  data = load_data(args.data_path)
  pred_hidden = np.array([d['hidden'][0] for d in data])
  prey_hidden = np.array([d['hidden'][1] for d in data])
  actions = np.array([d['actions'] for d in data])  # [T, 2]
  actions_pred = actions[:,0]
  actions_prey = actions[:,1]

  behavior_keys = ['STAMINA', 'POSITION', 'ORIENTATION', 'actions', 'rewards']
  design, names = build_regressors(data, behavior_keys)

  # predator tuning order
  w_pred = compute_tuning_weights(pred_hidden, design)
  idx_pred, _, wts_pred = assign_preferred(w_pred, names)
  order_pred = sort_neurons(idx_pred, wts_pred)

  # prey tuning order
  w_prey = compute_tuning_weights(prey_hidden, design)
  idx_prey, _, wts_prey = assign_preferred(w_prey, names)
  order_prey = sort_neurons(idx_prey, wts_prey)

  overlay_video(args.video_path, pred_hidden, prey_hidden,
                order_pred, order_prey,
                actions_pred, actions_prey,
                args.win, args.output_path)
  print(f"Wrote {args.output_path}")

if __name__ == '__main__':
  main()
