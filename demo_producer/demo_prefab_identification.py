import cv2
import numpy as np
import pickle
from matplotlib import cm

# Load the pickle file and extract tile codes
with open('prefab_demo/predator_prey__open_debug_episode_1.pkl', 'rb') as f:
    data = pickle.load(f)

# Reshape and transpose so that code_map has shape (rows, cols) = (height_cells, width_cells)
tile_codes = [d['tile_code'] for d in data]

# Open the original video
cap = cv2.VideoCapture('prefab_demo/1.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./prefab_demo/overlay.mp4', fourcc, fps, (frame_w, frame_h))

# Build a colormap for codes (up to max_code)
max_code = max(tc.max() for tc in tile_codes)
cmap = cm.get_cmap('tab20', max_code + 1)

# Compute cell dimensions in pixels
h_cells, w_cells = tile_codes[0].shape
cell_h = frame_h // h_cells
cell_w = frame_w // w_cells

# Process each frame
for i, code_map in enumerate(tile_codes):
    ret, frame = cap.read()
    if not ret:
        break

    # Create an overlay image
    overlay = np.zeros_like(frame, dtype=np.uint8)

    # Draw each cell
    for y in range(h_cells):
        for x in range(w_cells):
            val = code_map[y, x]
            if val > 0:
                color = (np.array(cmap(val)[:3]) * 255).astype(np.uint8)
                top_left = (x * cell_w, y * cell_h)
                bottom_right = ((x + 1) * cell_w, (y + 1) * cell_h)
                cv2.rectangle(overlay, top_left, bottom_right,
                              color.tolist(), thickness=-1)

    # Blend original frame and overlay
    blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Write to output
    out.write(blended)

# Clean up
cap.release()
out.release()

print("Overlay video saved to /prefab_demo/overlay.mp4")
