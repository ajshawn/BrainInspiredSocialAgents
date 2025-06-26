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
import re


def parse_agent_roles(base_name: str):
  agent_specs = re.findall(r'([A-Za-z0-9]+_pre[a-z]_\d+)', base_name)
  role, source = {}, {}
  for i, spec in enumerate(agent_specs):
    role[i] = "predator" if "pred" in spec else "prey"
    source[i] = spec
  return role, source
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
    z_all,
    actions_all,
    roles,
    orderings,
    win,
    output_path,
    event_list,
    neuron_map_width=10,
    margin_width=60,
    T_max=100,
):
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2 + margin_width, h))

  T, A, D = z_all.shape
  T = min(T, T_max)
  width = 2 * win + 1

  pcas = [PCA(n_components=1).fit(z_all[:, i]) for i in range(A)]
  action_names = ['NOOP', 'Forward', 'STEP_RIGHT', 'BACKWARD',
                  'STEP_LEFT', 'TURN_LEFT', 'TURN_RIGHT', 'INTERACT']
  t=0
  while t < T_max:
    t += 1
    ret, frame = cap.read()
    if not ret:
      break

    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    idx = min(max(idx, 0), T - 1)

    lo = max(0, idx - win)
    hi = min(T, idx + win + 1)
    x = np.arange(lo, hi) - idx

    agent_panels = [f'agent{i}' for i in range(A)]
    grid_rows = (A + 2 + 2) // 3  # +2 for PC + ACT
    rows = [agent_panels[i:i + 3] for i in range(0, A, 3)]
    rows += [['pc'] * 3, ['act'] * 3]
    fig, axs = plt.subplot_mosaic(rows, figsize=(8, 8), constrained_layout=True, dpi=100)

    # 1. Detect active events
    active = [e for e in event_list if e['start'] <= idx <= e['end']]
    active_participants = set(p for e in active for p in e['participants'])
    active_texts = [e['text'] for e in active]

    # 2. Agent neuron maps
    for i in range(A):
      z = z_all[idx, i][orderings[i]]
      grid = z.reshape(neuron_map_width, -1)
      ax = axs[f'agent{i}']
      ax.pcolor(grid, cmap='gray', vmin=-2, vmax=2)
      title_color = 'red' if i in active_participants else 'black'
      ax.set_title(f"{roles[i].capitalize()} {i}", fontsize=8, color=title_color)
      ax.axis('off')

    # PC1
    ax_pc = axs['pc']
    for i in range(A):
      pc1 = pcas[i].transform(z_all[lo:hi, i]).flatten()
      ax_pc.plot(x, pc1, label=f'{roles[i]} {i}')
    ax_pc.axvline(0, color='k', linestyle='--')
    ax_pc.set_xlim(-win, win)
    ax_pc.set_ylabel('PC1')
    ax_pc.legend(fontsize='x-small', ncol=3)
    ax_pc.set_xticks([])

    # Actions
    ax_act = axs['act']
    for i in range(A):
      a = actions_all[lo:hi, i]
      marker = '|' if roles[i] == 'predator' else 'o'
      ax_act.scatter(x, a, marker=marker, label=f'{roles[i]} {i}', alpha=0.7)
    ax_act.axvline(0, color='k', linestyle='--')
    ax_act.set_xlim(-win, win)
    ax_act.set_ylim(-0.5, len(action_names) - 0.5)
    ax_act.set_yticks(range(len(action_names)))
    ax_act.set_yticklabels(action_names, fontsize='xx-small')
    ax_act.set_xlabel('Time Offset')
    ax_act.legend(fontsize='x-small', loc='upper right')
    # 4. Add event text
    # if active_texts:
    #   fig.text(0.5, 0.01, " | ".join(active_texts), ha='center', va='bottom', fontsize=8, color='blue')

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    ih, iw = fig.canvas.get_width_height()[::-1]
    overlay = buf.reshape((ih, iw, 3))
    plt.close(fig)

    # Resize overlay to video frame size
    overlay = cv2.resize(overlay, (w, h))

    # Draw active event text directly on the frame
    if active_texts:
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 0.8
      thickness = 1
      text = " | ".join(active_texts)
      lines = wrap_text_cv(text, font, font_scale, thickness, frame.shape[1] - 40)

      line_height = 18  # pixels between lines
      start_y = frame.shape[0] - 10 - line_height * (len(lines) - 1)

      for i, line in enumerate(lines):
        text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_x = int((frame.shape[1] - text_size[0]) / 2)
        text_y = start_y + i * line_height
        cv2.putText(frame, line, (text_x, text_y), font,
                    font_scale, (0, 0, 255), thickness, lineType=cv2.LINE_AA)

    margin = np.ones((h, margin_width, 3), dtype=np.uint8) * 255
    combo = np.hstack((overlay, margin, frame))
    out.write(combo)

  cap.release()
  out.release()

def wrap_text_cv(text, font, font_scale, thickness, max_width):
  """
  Wrap text for OpenCV based on pixel width using cv2.getTextSize.
  Returns a list of wrapped lines.
  """
  words = text.split()
  lines = []
  current_line = ""

  for word in words:
    test_line = current_line + (" " if current_line else "") + word
    size, _ = cv2.getTextSize(test_line, font, font_scale, thickness)
    if size[0] > max_width and current_line:
      lines.append(current_line)
      current_line = word
    else:
      current_line = test_line

  if current_line:
    lines.append(current_line)

  return lines


def parse_events(raw_events, display_window=5):
  active_events = []
  for e in raw_events:
    if 'time_start' in e and 'time_end' in e:
      t_start, t_end = e['time_start'], e['time_end']
    elif 'time' in e:
      t_start = max(0, e['time'] - display_window)
      t_end = t_start + display_window
    else:
      continue
    event_text = f"{e.get('type', 'event')}: participants {', '.join(map(str, e.get('participants', [])))}"
    participants = e.get('participants', [])
    active_events.append({'start': t_start, 'end': t_end, 'text': event_text, 'participants': participants})
  return active_events


def pad_hidden_dim(data):
  T = len(data)
  A = len(data[0]['hidden'])
  max_dim = max(d['hidden'][i].shape[0] for d in data for i in range(A))

  hidden_padded = np.zeros((T, A, max_dim))
  for t in range(T):
    for a in range(A):
      h = data[t]['hidden'][a]
      hidden_padded[t, a, :len(h)] = h
  return hidden_padded


def main():
  parser = argparse.ArgumentParser(description="Overlay multi-panel neural and behavior plots on video")
  parser.add_argument('--video_path', type=str, required=False,
                      default='/home/mikan/e/GitHub/social-agents-JAX/recordings/meltingpot/mix_AH20250107_pred_0_AH20250107_pred_1_AH20250107_prey_3_AH20250107_prey_4_AH20250107_prey_5_AH20250107_prey_8_predator_prey__open_debug_smaller_13x13/1.mp4')
                      # default=  '/home/mikan/e/GitHub/social-agents-JAX/recordings/meltingpot/mix_OR20250210_pred_0_OP20241126ckp9651_prey_7_OP20250224ckp6306_prey_6_predator_prey__open_debug_smaller_13x13/6.mp4')
  parser.add_argument('--data_path', type=str, required=False,
                      default='/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/mix_AH20250107_pred_0_AH20250107_pred_1_AH20250107_prey_3_AH20250107_prey_4_AH20250107_prey_5_AH20250107_prey_8predator_prey__open_debug_agent_0_1_3_4_5_8/episode_pickles/predator_prey__open_debug_episode_1.pkl')
                      # default=  '/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/mix_OR20250210_pred_0_OP20241126ckp9651_prey_7_OP20250224ckp6306_prey_6predator_prey__open_debug_agent_0_7_6/episode_pickles/predator_prey__open_debug_episode_6.pkl')
  parser.add_argument('--output_path', type=str, default='./example_AH_2_4_overlay_mosaic.mp4')
  parser.add_argument("--event_path", type=str,
                      default='/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/analysis_results_merged/mix_AH20250107_pred_0_AH20250107_pred_1_AH20250107_prey_3_AH20250107_prey_4_AH20250107_prey_5_AH20250107_prey_8_merged.pkl')
                      # default='/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/analysis_results_merged/mix_OR20250210_pred_0_OP20241126ckp9651_prey_7_OP20250224ckp6306_prey_6_merged.pkl')
  parser.add_argument('--win', type=int, default=50)
  parser.add_argument('--n_clusters', type=int, default=10)
  args = parser.parse_args()
  args.episode = args.video_path.split('/')[-1].split('.')[0]

  role_dict, _ = parse_agent_roles(args.data_path)
  n_agents = len(role_dict)
  roles = [role_dict[i] for i in range(n_agents)]

  data = load_data(args.data_path)
  event = pd.read_pickle(args.event_path).loc[int(args.episode), ['apple_cooperation_events', 'distraction_events', 'fence_events']]
  event = np.hstack(event)
  event_list = parse_events(event)


  T_max = 200
  A = len(roles)

  # hidden_all = np.array([d['hidden'] for d in data])  # [T, A, D]
  hidden_all = pad_hidden_dim(data)
  actions_all = np.array([d['actions'] for d in data])  # [T, A]

  z_all = np.stack([zscore_in_time(hidden_all[:, i]) for i in range(A)], axis=1)

  behavior_keys = ['STAMINA', 'POSITION', 'ORIENTATION', 'actions', 'rewards']
  design, names = build_regressors(data, behavior_keys)

  orderings = []
  for i in range(A):
    w = compute_tuning_weights(z_all[:, i], design)
    top = select_top_neurons(w, names, behavior_keys, top_n=121)
    _, order = compute_clusters(z_all[:, i], top, args.n_clusters)
    orderings.append(order)

  overlay_video(
    args.video_path,
    z_all,
    actions_all,
    roles,
    orderings,
    args.win,
    args.output_path,
    event_list,
    neuron_map_width=11,
    T_max=T_max,
  )

if __name__ == '__main__':
  main()
