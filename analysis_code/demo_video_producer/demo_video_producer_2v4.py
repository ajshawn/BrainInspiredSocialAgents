import argparse
import pickle
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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


def build_regressors(data, keys):
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
    T, D = hidden.shape
    K = design.shape[1]
    weights = np.zeros((D, K))
    model = LinearRegression()
    for i in range(D):
        model.fit(design, hidden[:, i])
        weights[i, :] = model.coef_
    return weights


def assign_preferred(weights, names):
    pref_idx = np.argmax(np.abs(weights), axis=1)
    pref_w = weights[np.arange(weights.shape[0]), pref_idx]
    pref_name = [names[idx] for idx in pref_idx]
    return pref_idx, pref_name, pref_w


def sort_neurons(pref_idx, pref_w):
    return np.lexsort((-np.abs(pref_w), pref_idx))


def overlay_video(video_path, hidden_all, actions_all, roles, orderings, win, output_path, margin_width=60):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) * 2
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2 + margin_width, h))

    T, A, D = hidden_all.shape
    frame_idx = 0
    width = 2 * win + 1

    pcas = [PCA(n_components=1).fit(hidden_all[:, i]) for i in range(A)]
    action_names = ['NOOP', 'Forward', 'STEP_RIGHT', 'BACKWARD', 'STEP_LEFT', 'TURN_LEFT', 'TURN_RIGHT', 'INTERACT']

    while frame_idx < 200:
        ret, frame = cap.read()
        if not ret:
            break

        lo = max(0, frame_idx - win)
        hi = min(T, frame_idx + win + 1)
        L = hi - lo
        x = np.arange(L) - (frame_idx - lo)

        fig, axs = plt.subplots(A + 2, 1, figsize=(8, 8), dpi=100)

        for i in range(A):
            role = roles[i]
            order = orderings[i]
            raw = hidden_all[lo:hi, i][:, order].T
            pad = np.full((raw.shape[0], width), np.nan)
            start = max(0, win - frame_idx)
            pad[:, start:start + L] = raw
            xedges = np.arange(width + 1) - win
            yedges = np.arange(raw.shape[0] + 1)
            axs[i].pcolormesh(xedges, yedges, pad, cmap='RdBu_r', shading='auto')
            axs[i].axvline(0, color='k', linestyle='--')
            axs[i].set_ylabel(f'{role.capitalize()} {i}')
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        # PC1 subplot
        ax_pc1 = axs[A]
        for i in range(A):
            pc1 = pcas[i].transform(hidden_all[lo:hi, i]).flatten()
            ax_pc1.plot(x, pc1, label=f'{roles[i]} {i}')
        ax_pc1.axvline(0, color='k', linestyle='--')
        ax_pc1.set_xlim(-win, win)
        ax_pc1.set_ylabel('PC1')
        ax_pc1.legend(fontsize='small', ncol=3)
        ax_pc1.set_xticks([])

        # Action raster
        ax_act = axs[A + 1]
        for i in range(A):
            a = actions_all[lo:hi, i]
            ax_act.scatter(x, a, marker='|' if roles[i] == 'predator' else 'o', label=f'{roles[i]} {i}', alpha=0.7)
        ax_act.axvline(0, color='k', linestyle='--')
        ax_act.set_xlim(-win, win)
        ax_act.set_ylim(-0.5, len(action_names) - 0.5)
        ax_act.set_yticks(range(len(action_names)))
        ax_act.set_yticklabels(action_names)
        ax_act.set_ylabel('Action')
        ax_act.set_xlabel('Time Offset')
        ax_act.legend(fontsize='small', ncol=3, loc='upper right')

        fig.tight_layout(pad=0.5, rect=(0, 0, 0.95, 1))
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        ih, iw = fig.canvas.get_width_height()[::-1]
        overlay = buf.reshape((ih, iw, 3))
        plt.close(fig)

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
                        default='/home/mikan/e/GitHub/social-agents-JAX/recordings/meltingpot/mix_AH20250107_pred_0_AH20250107_pred_1_AH20250107_prey_3_AH20250107_prey_4_AH20250107_prey_5_AH20250107_prey_8_predator_prey__open_debug_smaller_13x13/1.mp4')
    parser.add_argument('--data_path', type=str, required=False,
                        default='/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/mix_AH20250107_pred_0_AH20250107_pred_1_AH20250107_prey_3_AH20250107_prey_4_AH20250107_prey_5_AH20250107_prey_8predator_prey__open_debug_agent_0_1_3_4_5_8/episode_pickles/predator_prey__open_debug_episode_1.pkl')
    parser.add_argument('--output_path', type=str, default='./example_AH_2_4_overlay.mp4')
    parser.add_argument('--win', type=int, default=50)
    args = parser.parse_args()

    role_dict, _ = parse_agent_roles(args.data_path)
    n_agents = len(role_dict)
    roles = [role_dict[i] for i in range(n_agents)]

    data = load_data(args.data_path)
    T = len(data)
    hidden_all = np.array([d['hidden'] for d in data])  # [T, A, D]
    actions_all = np.array([d['actions'] for d in data])  # [T, A]

    behavior_keys = ['STAMINA', 'POSITION', 'ORIENTATION', 'actions', 'rewards']
    design, names = build_regressors(data, behavior_keys)

    orderings = []
    for i in range(n_agents):
        h = hidden_all[:, i]
        w = compute_tuning_weights(h, design)
        idx, _, wts = assign_preferred(w, names)
        orderings.append(sort_neurons(idx, wts))

    overlay_video(args.video_path, hidden_all, actions_all,
                  roles, orderings, args.win, args.output_path)
    print(f"Wrote {args.output_path}")


if __name__ == '__main__':
    main()
