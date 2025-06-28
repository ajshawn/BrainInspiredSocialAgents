import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2
import matplotlib.cm as cm

events = []
# for i in range(1,11):
i=1
with open(
    f'/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/mix_OP20250224ckp6306_pred_1_OR20250210_pred_0_OR20250210_prey_5_OR20250210_prey_10_OP20241126ckp9651_prey_5_OP20241126ckp9651_prey_6predator_prey__open_debug_agent_1_0_5_10_5_6/episode_pickles/predator_prey__open_debug_episode_{i}.pkl',
    'rb') as f:
  data = pickle.load(f)
  events.extend([data_i['events'] for data_i in data if data_i['events']])
  tile_codes  = np.array([data_i['tile_code'] for data_i in data])
  world_tile_codes = [data_i['world_tile_code'] for data_i in data]
  positions = [data_i['POSITION'] for data_i in data]
  orientations = [data_i['ORIENTATION'] for data_i in data]
  RGB = np.array([data_i['RGB'] for data_i in data])
  # Now check if the FOV tile code matches the world tile code for each individual
n_steps      = len(world_tile_codes)
n_agents     = tile_codes[0].shape[0]
h_world, w_world = world_tile_codes[0].shape
Hf, Wf       = tile_codes[0].shape[1:]  # e.g. (11,11)

# your FOV params
LEFT, RIGHT, FORWARD, BACKWARD = 5, 5, 9, 1

def ori_position(A_pos, A_orient, B_pos):
  orientation_transform = {
    0: lambda x, y: (x, y),  # UP
    1: lambda x, y: (y, -x),  # RIGHT
    2: lambda x, y: (-x, -y),  # DOWN
    3: lambda x, y: (-y, x)  # LEFT
  }
mismatches = []
for t in range(n_steps):
    world_map = world_tile_codes[t]
    pos_list  = positions[t]
    ori_list  = orientations[t]
    fov_list  = tile_codes[t]

    for a in range(n_agents):
        px, py = pos_list[a]
        ori    = int(ori_list[a])
        fov    = fov_list[a]

        # build expected slice
        expected = np.zeros_like(fov)
        for i in range(Hf):
            for j in range(Wf):
                # compute relative offsets in agent frame
                dx = j - LEFT
                dy = FORWARD - i

                # rotate into world coords
                if ori == 0:       # UP
                    wx = px + dx
                    wy = py + dy
                elif ori == 1:     # RIGHT
                    wx = px + dy
                    wy = py - dx
                elif ori == 2:     # DOWN
                    wx = px - dx
                    wy = py - dy
                elif ori == 3:     # LEFT
                    wx = px - dy
                    wy = py + dx
                else:
                    wx, wy = -1, -1

                # sample if in bounds
                if 0 <= wx < w_world and 0 <= wy < h_world:
                    expected[i, j] = world_map[wy, wx]

        if not np.array_equal(fov, expected):
            mismatches.append((t, a))

print(f"Checked {n_steps}×{n_agents} = {n_steps*n_agents} observations.")
print(f"Found {len(mismatches)} mismatches.")
if mismatches:
    t0, a0 = mismatches[0]
    print(f"First mismatch at time {t0}, agent {a0}:")
    print("Observed FOV:\n", tile_codes[t0][a0])
    print("Expected slice:\n", expected)  # from last loop


# Extract arrays: RGB (1001,6,88,88,3), tile_codes (1001,6,11,11)
RGB         = np.stack([d['RGB'] for d in data], axis=0)              # shape: (T, 6, H, W, 3)
tile_codes  = np.stack([d['tile_code'] for d in data], axis=0)        # shape: (T, 6, hf, wf)

# --- 2) Prepare video writer ---
T, n_agents, H, W, _ = RGB.shape
hf, wf = tile_codes.shape[2:]
cell_h, cell_w = H//hf, W//wf

# Compute a colormap for codes
max_code = tile_codes.max()
cmap = cm.get_cmap('tab20', max_code+1)

fps = 30        # or your environment's FPS
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('TILE_CODE_demos/overlay_agents.mp4', fourcc, fps, (W*3, H*2))

# --- 3) Build and write each frame ---
for t in range(T):
    agent_frames = []
    for a in range(n_agents):
        # Convert RGB to BGR for OpenCV
        frame_rgb = (RGB[t, a] * 255).astype(np.uint8) if RGB.max() <= 1 else RGB[t, a].astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Create a transparent overlay
        overlay = np.zeros_like(frame_bgr)

        code_map = tile_codes[t, a]
        for y in range(hf):
            for x in range(wf):
                val = code_map[y, x]
                if val > 0:
                    # colormap returns RGBA in [0,1]; convert to BGR [0,255]
                    rgb_col = np.array(cmap(val)[:3]) * 255
                    bgr_col = tuple(int(c) for c in rgb_col[::-1])
                    top_left     = (x*cell_w, y*cell_h)
                    bottom_right = ((x+1)*cell_w, (y+1)*cell_h)
                    cv2.rectangle(overlay, top_left, bottom_right, bgr_col, thickness=-1)

        # Blend and collect
        blended = cv2.addWeighted(frame_bgr, 0.7, overlay, 0.3, 0)
        agent_frames.append(blended)

    # Arrange 6 views in a 2×3 grid
    top    = np.hstack(agent_frames[:3])
    bottom = np.hstack(agent_frames[3:])
    mosaic = np.vstack([top, bottom])

    out.write(mosaic)

out.release()
print("Overlay video saved to overlay_agents.mp4")