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

def read_attn_weights(csv_path: str, n_agents: int) -> np.ndarray:
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
    attn_weights = rearrange(attn_weights, '(t n) 1 h -> t n h', n=n_agents)
    sorted_indices = np.argsort(log_dict[TIME_STEP_KEY].reshape(-1))
    attn_weights = attn_weights[sorted_indices]
    attn_weights = rearrange(attn_weights, 't n h -> n t h')
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


def visualize_attn(csv_path, image_dir, n_agents, save_dir=None):
    """
    Visualizes attention weights as heatmaps over 88x88 images.

    Args:
        csv_path (str): Path to the CSV file containing attention weights.
        image_dir (str): Path to directory containing observation images.
        n_agents (int): Number of agents.
        smooth (bool): Whether to apply smoothing to attention heatmaps.
        save_dir (str): If provided, saves visualizations to this directory instead of displaying them.
    """
    for i in range(n_agents):
        os.makedirs(os.path.join(save_dir, f"agent_{i}"), exist_ok=True)

    attn_weights = read_attn_weights(csv_path, n_agents)

    all_image_paths = [
        os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
        if fname.lower().endswith(".png")
    ]

    all_image_paths = sorted(all_image_paths, key=lambda x: int(re.search(r'step_(\d+)_agent_agent_(\d+)', x).group(1)))

    for image_path in tqdm(all_image_paths):
        step, agent_idx = parse_step_and_agent(image_path)
        try:
            attn = attn_weights[agent_idx, step]  # shape (121,)
        except IndexError:
            print(f"Skipping image {image_path} for agent {agent_idx} at step {step}: out of bounds in attention weights.")
            continue    

        # Load and normalize the image
        img = Image.open(image_path).convert("RGB")
        img = img.resize((88, 88), Image.BICUBIC)
        img_np = np.array(img).astype("float32") / 255

        # Reshape and optionally smooth attention map
        alpha = attn.reshape(11, 11)
        # alpha = np.fliplr(alpha)
        alpha = np.kron(alpha, np.ones((8, 8)))

        # Plot the image and overlay attention
        fig, ax = plt.subplots()
        ax.imshow(img_np)
        ax.imshow(alpha, alpha=0.5, cmap=cm.jet)
        ax.axis("off")

        

        if save_dir:
            attn_path = os.path.join(save_dir, f"agent_{agent_idx}", f"atten_step_{step}_agent_{agent_idx}_attn.png")
            concat_path = os.path.join(save_dir, f"agent_{agent_idx}", f"atten_step_{step}_agent_{agent_idx}.png")

            # Save attention-only image
            plt.savefig(attn_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            # Load the saved attention image
            attn_img = Image.open(attn_path).resize((88, 88), Image.BICUBIC)

            # Concatenate: original on left, attention on right
            concat_img = Image.new('RGB', (88 * 2, 88))
            concat_img.paste(img, (0, 0))
            concat_img.paste(attn_img, (88, 0))

            # Save combined image and optionally remove the temporary attention image
            concat_img.save(concat_path)
            os.remove(attn_path)
        else:
            plt.show()

            plt.close(fig)
            
    # Save the video
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
    args = parser.parse_args()

    visualize_attn(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        n_agents=args.n_agents,
        save_dir=args.save_dir
    )
