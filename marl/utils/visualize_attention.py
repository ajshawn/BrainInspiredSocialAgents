import csv
import ast
import os
import re
from typing import Tuple
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from tqdm import tqdm
import imageio.v2 as imageio
import argparse

ATTEN_WEIGHTS_KEY = "embedding"
TIME_STEP_KEY = "timestep"

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

def parse_step_and_agent(filename: str) -> Tuple[int, int]:
    """
    Given a filename of the form: step_0_agent_agent_0.png
    extracts the step number (0) and agent number (0).
    Returns them as integers: (step, agent).
    """
    filename = os.path.basename(filename)
    pattern = r'^step_(\d+)_agent_agent_(\d+)\.png$'
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match the expected pattern.")
    step_str, agent_str = match.groups()
    return int(step_str), int(agent_str)

def create_video_from_images(
    image_dir: str,
    output_path: str,
    fps: int = 3,
    pattern: str = '.png'
):
    """
    Creates an MP4 video from PNG images in a directory.

    Args:
        image_dir (str): Path to the directory containing PNG images.
        output_path (str): Path to save the generated MP4 video.
        fps (int): Frames per second of the output video.
        pattern (str): File extension to match (default is '.png').
    """
    # Get sorted list of image file paths
    files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith('.png')],
        key=lambda f: int(re.search(r'atten_step_(\d+)_agent', f).group(1))
    )

    if not files:
        raise ValueError(f"No '{pattern}' files found in directory: {image_dir}")

    # Read images and write video
    with imageio.get_writer(output_path, format='FFMPEG', mode='I', fps=fps, codec='libx264') as writer:
        for filename in files:
            image = imageio.imread(os.path.join(image_dir, filename))
            writer.append_data(image)

    print(f"Video saved to {output_path}")


def visualize_attn(csv_path, image_dir, n_agents, save_dir=None, n_heads=1):
    """
    Visualizes attention weights as heatmaps over 88x88 images.

    Args:
        csv_path (str): Path to the CSV file containing attention weights.
        image_dir (str): Path to directory containing observation images.
        n_agents (int): Number of agents.
        save_dir (str): If provided, saves visualizations to this directory instead of displaying them.
    """
    for i in range(n_agents):
        os.makedirs(os.path.join(save_dir, f"agent_{i}"), exist_ok=True)

    attn_weights = read_attn_weights(csv_path, n_agents, n_heads=n_heads)

    all_image_paths = [
        os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
        if fname.lower().endswith(".png") and 'global' not in fname
    ]
    all_image_paths = sorted(
        all_image_paths,
        key=lambda x: int(re.search(r'step_(\d+)_agent_agent_(\d+)', x).group(1))
    )

    for image_path in tqdm(all_image_paths):
        step, agent_idx = parse_step_and_agent(image_path)
        try:
            attn = attn_weights[agent_idx, step]  # shape (n_heads, 121)
        except IndexError:
            print(f"Skipping image {image_path} for agent {agent_idx} at step {step}: out of bounds in attention weights.")
            continue    

        # Load and normalize the image
        img = Image.open(image_path).convert("RGB")
        img = img.resize((88, 88), Image.BICUBIC)
        img_np = np.array(img).astype("float32") / 255

        # Keep original PIL image for final concat
        concat_images = [img]

        for head_idx in range(n_heads):
            attn_head = attn[head_idx]
            alpha = attn_head.reshape(11, 11)
            alpha = np.kron(alpha, np.ones((8, 8)))  # upscale to 88x88

            # Plot with overlay and save to a temporary buffer
            fig, ax = plt.subplots()
            ax.imshow(img_np)
            ax.imshow(alpha, alpha=0.5, cmap=cm.jet)
            ax.axis("off")

            tmp_path = f"temp_attn_{step}_{agent_idx}_{head_idx}.png"
            plt.savefig(tmp_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            # Load back the overlay as PIL
            attn_img = Image.open(tmp_path).resize((88, 88), Image.BICUBIC)
            concat_images.append(attn_img)

            os.remove(tmp_path)

        # Concatenate original + all heads horizontally
        total_width = 88 * (1 + n_heads)
        concat_result = Image.new('RGB', (total_width, 88))
        for i, img_part in enumerate(concat_images):
            concat_result.paste(img_part, (i * 88, 0))

        if save_dir:
            save_path = os.path.join(
                save_dir, f"agent_{agent_idx}", f"step_{step}_agent_{agent_idx}.png"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            concat_result.save(save_path)
        else:
            concat_result.show()

    # Optional: Save video for each agent
    if save_dir:
        for i in range(n_agents):
            video_path = os.path.join(save_dir, f"agent_{i}", "attn_video.mp4")
            create_video_from_images(os.path.join(save_dir, f"agent_{i}"), video_path)         
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize attention weights from CSV logs.")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file containing attention weights.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing observation images.')
    parser.add_argument('--n_agents', type=int, required=True, help='Number of agents.')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save visualizations. If not provided, displays them.')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads to visualize.')
    args = parser.parse_args()

    visualize_attn(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        n_agents=args.n_agents,
        save_dir=args.save_dir,
        n_heads=args.n_heads
    )
