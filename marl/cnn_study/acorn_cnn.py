import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import haiku as hk
import jax
import jax.numpy as jnp
from absl import flags
from absl import app

from marl.cnn_study.make_dataset import _parse_step_and_agent, _parse_objs_in_view, \
    _decode_image, OBJECTS_IN_VIEW, ACORN
from marl.agents.networks import VisualFeatures
from marl.cnn_study.probe_cnn import build_standalone_model_from_checkpoint

ACORN_IMAGE_PATH = "data/acorn.png"  # Path to the acorn image
FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "create_pairs", "Mode to run: create_pairs or cnn_inference")
flags.DEFINE_string("labels_path", "data/predator_prey_acorn_pairs/labels.jsonl", "Path to the labels file")

def create_acorn_pairs(
    image_dir: str,
    label_file: str,
    step_interval: int = 50,
    n_agents: int = 13,
    n_acorns: int = 1,
    out_image_w_acorn_path: str = "data/predator_prey_acorn_pairs/with_acorn",
    out_image_wo_acorn_path: str = "data/predator_prey_acorn_pairs/without_acorn",
    out_labels_path: str = "data/predator_prey_acorn_pairs/labels.jsonl",
    acorn_image_path: str = ACORN_IMAGE_PATH,
):
    '''
    Based on a set of images that do not contain acorns, add acorn(s) to the images at random locations.
    '''

    # Load the acorn image
    acorn_image, _ = _decode_image(acorn_image_path, None)

    # List of all images in the directory
    all_image_paths = [
        os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
        if fname.lower().endswith(".png")
    ]

    # Load labels from JSON lines file
    with open(label_file, "r") as f:
        lines = f.readlines()
    labels = [json.loads(line) for line in lines]

    # Check that we have exactly n_agents images per label to align the label matching below
    assert len(labels) * n_agents == len(all_image_paths), \
        f"Number of labels ({len(labels)}) does not match number of images ({len(all_image_paths)})"

    result_labels = []

    for filename in tqdm(all_image_paths):
        step, agent_idx = _parse_step_and_agent(filename)
        label_dict = labels[step // step_interval]

        # Objects string for this agent
        objects_str = label_dict[f"agent_{agent_idx}"][OBJECTS_IN_VIEW]
        objects_dict = _parse_objs_in_view(objects_str, agent_idx)

        if ACORN in objects_dict:
            continue  # Skip images that already contain acorns

        # Load the image
        image, _ = _decode_image(filename, None)
        image = np.array(image)

        # Create a copy of the image for the acorn version
        image_without_acorn = image.copy()
        image_with_acorn = replace_random_block(image.copy(), acorn_image)

        # Check if the acorn was successfully placed
        if image_with_acorn is None:
            continue

        # Save the modified images as PNG files
        image_with_acorn_path = os.path.join(out_image_w_acorn_path, os.path.basename(filename))
        image_without_acorn_path = os.path.join(out_image_wo_acorn_path, os.path.basename(filename))

        Image.fromarray(image_with_acorn).resize((88, 88), Image.BICUBIC).save(image_with_acorn_path)
        Image.fromarray(image_without_acorn).resize((88, 88), Image.BICUBIC).save(image_without_acorn_path)

        # Save the labels
        objects_dict["acorn"] = n_acorns
        objects_dict["acorn_image"] = image_with_acorn_path
        objects_dict["no_acorn_image"] = image_without_acorn_path
        objects_dict["agent_idx"] = agent_idx

        result_labels.append(objects_dict)

    # Save the labels to a JSON file
    with open(out_labels_path, "w") as f:
        for label in result_labels:
            f.write(json.dumps(label) + "\n")


def replace_random_block(source_img: np.ndarray, target_img: np.ndarray, K: int = 10) -> np.ndarray:
    """
    Replace a random 8x8 block in the source image with the target image, 
    satisfying the conditions:
    1. Block must not be all black.
    2. Block must not be at grid position (10, 5) (agent location).
    
    Args:
        source_img: np.ndarray of shape (88, 88, 3)
        target_img: np.ndarray of shape (8, 8, 3)
        K: int, number of attempts to find a valid location
    
    Returns:
        Modified source image or None if no valid location found.
    """
    assert source_img.shape == (88, 88, 3)
    assert target_img.shape == (8, 8, 3)

    grid_h, grid_w = 88 // 8, 88 // 8
    forbidden_pos = (10, 5)

    # Create list of valid grid positions (excluding forbidden one)
    valid_positions = [
        (i, j) for i in range(grid_h) for j in range(grid_w)
        if (i, j) != forbidden_pos
    ]

    for _ in range(K):
        # Randomly pick a grid position
        grid_i, grid_j = valid_positions[np.random.randint(len(valid_positions))]
        h_start, w_start = grid_i * 8, grid_j * 8

        block = source_img[h_start:h_start+8, w_start:w_start+8]

        if not np.all(block == 0):  # Check if block is not all black
            # Replace block with target
            source_img[h_start:h_start+8, w_start:w_start+8] = target_img
            return source_img

    # If all K attempts failed, skip modification
    print(f"Failed to place acorn after {K} attempts.")
    return None


class CNNProbe(hk.Module):

  def __init__(self):
    super().__init__("meltingpot_cnn_probe")
    self._visual_torso = VisualFeatures()

  def __call__(self, obs):
    # extract visual features form RGB observation
    ip_img = obs.astype(jnp.float32) / 255
    vis_op = self._visual_torso(ip_img)
    return vis_op
  

def cnn_inference_on_acorn_pairs(
    labels_path: str = "data/predator_prey_acorn_pairs/labels.jsonl",
):
    import wandb
    import numpy as np
    from PIL import Image
    import jax.numpy as jnp
    import jax
    import haiku as hk
    import json
    from tqdm import tqdm

    def forward_fn(images):
        probe = CNNProbe()
        return probe(images)

    # Transform and load CNN parameters
    transformed_forward = hk.transform(forward_fn)
    params = build_standalone_model_from_checkpoint(
        transformed_forward,
        agent_idx=FLAGS.agent_idx,
        random_baseline=FLAGS.random_baseline,
    )
    rng = jax.random.PRNGKey(0)

    # Initialize W&B for analysis logging
    wandb.init(
        project="cnn_probe",
        name=f"acorn_eval_agent{FLAGS.agent_idx}_ckpt{FLAGS.ckp_idx}" if not FLAGS.random_baseline else "acorn_eval_random_baseline",
        config={
            "agent_idx": FLAGS.agent_idx,
            "ckpt_idx": FLAGS.ckp_idx,
        }
    )

    # Load labels.jsonl
    with open(labels_path, "r") as f:
        lines = [json.loads(line.strip()) for line in f.readlines()]

    distances = []

    for entry in tqdm(lines):
        path_with = entry["acorn_image"]
        path_without = entry["no_acorn_image"]

        try:
            # Load and preprocess both images
            img_with = _decode_image(path_with, None)[0]
            img_wo = _decode_image(path_without, None)[0]

            # Add batch dimension
            img_with = jnp.expand_dims(jnp.array(img_with), axis=0)
            img_wo = jnp.expand_dims(jnp.array(img_wo), axis=0)

            # Run CNN forward pass
            feat_with = transformed_forward.apply(params, rng, img_with)[0]
            feat_wo = transformed_forward.apply(params, rng, img_wo)[0]

            # Normalize to unit length
            feat_with = feat_with / jnp.linalg.norm(feat_with)
            feat_wo = feat_wo / jnp.linalg.norm(feat_wo)

            # Compute L2 distance
            dist = float(jnp.linalg.norm(feat_with - feat_wo))
            distances.append(dist)

            wandb.log({"acorn_pair_distance": dist})

        except Exception as e:
            print(f"Error processing pair {path_with} / {path_without}: {e}")
            continue

    # Log aggregate statistics
    distances_np = np.array(distances)
    wandb.log({
        "distance_mean": float(np.mean(distances_np)),
        "distance_std": float(np.std(distances_np)),
        "distance_hist": wandb.Histogram(np_histogram=np.histogram(distances_np, bins=100)),
    })


    print(f"[INFO] Logged {len(distances_np)} distances. Mean={np.mean(distances_np):.4f}, Std={np.std(distances_np):.4f}")

    wandb.finish()
    return distances_np


def main(_):
    if FLAGS.mode == "create_pairs":
        create_acorn_pairs(
            image_dir="data/predator_prey__open_random/agent_view_images",
            label_file="data/predator_prey__open_random/observations.jsonl",
            step_interval=50,
            n_agents=13,
            n_acorns=1,
            out_image_w_acorn_path="data/predator_prey_acorn_pairs/with_acorn",
            out_image_wo_acorn_path="data/predator_prey_acorn_pairs/without_acorn",
            out_labels_path="data/predator_prey_acorn_pairs/labels.jsonl",
        )
    elif FLAGS.mode == "cnn_inference":
        cnn_inference_on_acorn_pairs(
            labels_path=FLAGS.labels_path,
        )
    else:
        raise ValueError("Invalid mode. Choose 'create_pairs' or 'cnn_inference'.")


if __name__ == "__main__":
    app.run(main)
    


