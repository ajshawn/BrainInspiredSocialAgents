import csv
import ast
import os
import re
from collections import defaultdict
import json
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import argparse

ATTEN_WEIGHTS_KEY = "embedding"
TIME_STEP_KEY = "timestep"
POSITION_KEY = "POSITION"
ORIENTATION_KEY = "ORIENTATION"

vis_left = 5
vis_right = 5
vis_top = 9
vis_bottom = 1

orientation2overlay_offsets = {
    0: [-vis_left, vis_right + 1, -vis_top, vis_bottom + 1],  # North
    1: [-vis_bottom, vis_top + 1, -vis_left, vis_right + 1],  # East
    2: [-vis_right, vis_left + 1, -vis_bottom, vis_top + 1],  # South
    3: [-vis_top, vis_bottom + 1, -vis_right, vis_left + 1],  # West
}

def prepare_global_images(image_dir):
    all_image_paths = [
        os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
        if fname.lower().endswith(".png") and 'world' in fname
    ]
    all_image_paths = sorted(
        all_image_paths,
        key=lambda x: int(re.search(r'step_(\d+)_world', x).group(1))
    )
    return all_image_paths

def read_position_and_orientation(jsonl_path):
    agent2position = defaultdict(list)
    agent2orientation = defaultdict(list)
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            for agent_id, value in data.items():
                agent2position[agent_id].append(value[POSITION_KEY])
                agent2orientation[agent_id].append(value[ORIENTATION_KEY])
    agent2position = {agent_id: np.array(positions) for agent_id, positions in agent2position.items()}
    agent2orientation = {agent_id: np.array(orientations) for agent_id, orientations in agent2orientation.items()}
    return agent2position, agent2orientation

def parse_step(filename: str) -> int:
    filename = os.path.basename(filename)
    pattern = r'^step_(\d+)_world\.png$'
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match the expected pattern.")
    step_str = match.group(1)
    return int(step_str)

def read_attn_weights(
    csv_path: str, 
    n_agents: int,
    n_heads: int = 1,
) -> np.ndarray:
    """Reads attention weights from a CSV file."""
    with open(csv_path, newline='\n', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Skip the header row
        next(reader)
        # Read the data into a list of lists
        for row in reader:
            data = [ast.literal_eval(cell) for cell in row if cell]
            keys = data[0].keys()
            log_dict = {key: np.vstack([entry[key] for entry in data]) for key in keys}
    attn_weights = log_dict[ATTEN_WEIGHTS_KEY]
    if n_heads > 1:
        attn_weights = rearrange(attn_weights, '(t n_agent) 1 n_head h -> t n_agent n_head h', n_agent=n_agents, n_head=n_heads)
    elif attn_weights.ndim == 3:
        attn_weights = rearrange(attn_weights, '(t n) 1 h -> t n h', n=n_agents)
    elif attn_weights.ndim == 4:
        attn_weights = rearrange(attn_weights, '(t n) 1 h 1 -> t n h', n=n_agents)
    else:
        raise ValueError(f"Unexpected attention weights shape: {attn_weights.shape}")
    sorted_indices = np.argsort(log_dict[TIME_STEP_KEY].reshape(-1))
    attn_weights = attn_weights[sorted_indices]
    if n_heads > 1:
        attn_weights = rearrange(attn_weights, 't n_agent n_head h -> n_agent t n_head h')
    else:
        attn_weights = rearrange(attn_weights, 't n h -> n t 1 h')
    return attn_weights

def visualize_attention_weights(
    attn_weights: np.ndarray,
    agent2position: dict,
    agent2orientation: dict,
    all_image_paths: list,
    n_agents: int = 3,
    save_path: str = None,
):
    """Visualizes attention weights by aligning in tile coordinates first,
    then converting to pixel, with a white border around the global map.
    """
    if save_path:       
        save_path = os.path.join(save_path, "global_attn")
        os.makedirs(save_path, exist_ok=True)

    TILE_SIZE = 8
    RIM_TILES = 11
    RIM_PX = RIM_TILES * TILE_SIZE

    for image_path in tqdm(all_image_paths):
        global_image = Image.open(image_path).convert("RGB")
        global_image_np = np.array(global_image).astype("float32") / 255

        # === Add white rim ===
        H, W, C = global_image_np.shape
        padded_H = H + 2 * RIM_PX
        padded_W = W + 2 * RIM_PX

        # Start with all white
        padded_image = np.ones((padded_H, padded_W, C), dtype=global_image_np.dtype)

        # Insert original image in the center
        padded_image[RIM_PX:RIM_PX + H, RIM_PX:RIM_PX + W] = global_image_np

        step = parse_step(image_path)
        overlay_image = padded_image.copy()
        
        total_steps = attn_weights.shape[1]
        if step >= total_steps:
            print(f"Warning: Step {step} exceeds the number of steps in attention weights ({total_steps}). Skipping this step.")
            continue

        for agent_id in range(n_agents):
            agent_key = f"agent_{agent_id}"
            if agent_key not in agent2position:
                continue

            position_tile = agent2position[agent_key][step]  # [Y_tile, X_tile]
            orientation = agent2orientation[agent_key][step]

            attn_weight = attn_weights[agent_id, step, 0, :]
            alpha = attn_weight.reshape(11, 11)
            alpha = np.kron(alpha, np.ones((TILE_SIZE, TILE_SIZE)))  # upscale to 88×88
            alpha_norm = (alpha - alpha.min()) / (alpha.ptp() + 1e-8)
            heatmap = plt.cm.jet(alpha_norm)[..., :3]
            # Rotate the heatmap based on orientation
            heatmap = np.rot90(heatmap, k=-orientation)
            
            overlay_offsets = orientation2overlay_offsets[orientation]
            overlay_area = [
                position_tile[0] + overlay_offsets[0],
                position_tile[0] + overlay_offsets[1],
                position_tile[1] + overlay_offsets[2],
                position_tile[1] + overlay_offsets[3],
            ]
            
            # Convert overlay area to pixel coordinates, include the rim offset
            Y1, Y2, X1, X2 = [ele * TILE_SIZE for ele in overlay_area]
            Y1 += RIM_PX
            Y2 += RIM_PX
            X1 += RIM_PX
            X2 += RIM_PX

            overlay_image[X1:X2, Y1:Y2] = (
                0.5 * overlay_image[X1:X2, Y1:Y2]
                + 0.5 * heatmap
            )

        overlay_image = (overlay_image * 255).astype("uint8")
        overlay_pil = Image.fromarray(overlay_image)

        if save_path:
            output_fp = f"{save_path}/step_{step}_overlay.png"
            overlay_pil.save(output_fp)
        else:
            overlay_pil.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize attention weights on global images.")
    parser.add_argument("--attn_weights_csv", type=str, required=True, help="Path to the attention weights file.")
    parser.add_argument("--log_jsonl_path", type=str, required=True, help="Path to the JSONL file with agent positions and orientations.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing global images.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the overlay images. If None, will display them.")
    parser.add_argument("--n_agents", type=int, default=3, help="Number of agents in the environment.")
    parser.add_argument("--n_heads", type=int, default=1, help="Number of attention heads to visualize.")
    
    args = parser.parse_args()

    attn_weights = read_attn_weights(args.attn_weights_csv, n_agents=args.n_agents, n_heads=args.n_heads)
    agent2position, agent2orientation = read_position_and_orientation(args.log_jsonl_path)
    all_image_paths = prepare_global_images(args.image_dir)

    visualize_attention_weights(
        attn_weights,
        agent2position,
        agent2orientation,
        all_image_paths,
        n_agents=args.n_agents,
        save_path=args.save_dir
    )
    
if __name__ == "__main__":
    main()