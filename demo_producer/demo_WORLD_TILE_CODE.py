import cv2
import numpy as np
import pickle
from matplotlib import cm

# Load the pickle file and extract tile codes
with open('TILE_CODE_demos/predator_prey__open_debug_episode_1.pkl', 'rb') as f:
    data = pickle.load(f)

# Each entry’s 'tile_code' is already shaped (H_cells, W_cells)
tile_codes = [d['tile_code'] for d in data]

# Open the original video
cap = cv2.VideoCapture('TILE_CODE_demos/1.mp4')
fps      = cap.get(cv2.CAP_PROP_FPS)
frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('TILE_CODE_demos/overlay.mp4', fourcc, fps, (frame_w, frame_h))

# Build a colormap for codes (up to max_code)
max_code = max(tc.max() for tc in tile_codes)
cmap     = cm.get_cmap('tab20', max_code + 1)

# Compute cell size in pixels
h_cells, w_cells = tile_codes[0].shape
cell_h = frame_h // h_cells
cell_w = frame_w // w_cells

# Text settings
font      = cv2.FONT_HERSHEY_SIMPLEX
font_scale = min(cell_w, cell_h) / 50.0   # tweak as needed
thickness  = 1

for i, code_map in enumerate(tile_codes):
    ret, frame = cap.read()
    if not ret:
        break

    # Create transparent overlay
    overlay = np.zeros_like(frame, dtype=np.uint8)

    for y in range(h_cells):
        for x in range(w_cells):
            val = int(code_map[y, x])
            if val > 0:
                # fill rectangle
                color = (np.array(cmap(val)[:3]) * 255).astype(np.uint8).tolist()
                tl = (x * cell_w, y * cell_h)
                br = ((x + 1) * cell_w, (y + 1) * cell_h)
                cv2.rectangle(overlay, tl, br, color, thickness=-1)

                # draw number in center
                text = str(val)
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = tl[0] + (cell_w - text_w) // 2
                text_y = tl[1] + (cell_h + text_h) // 2
                cv2.putText(
                    overlay, text,
                    (text_x, text_y),
                    font, font_scale,
                    (255, 255, 255),  # white text
                    thickness,
                    lineType=cv2.LINE_AA
                )

    # Blend and write out
    blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    out.write(blended)

cap.release()
out.release()
print("Overlay video saved to TILE_CODE_demos/overlay.mp4")
