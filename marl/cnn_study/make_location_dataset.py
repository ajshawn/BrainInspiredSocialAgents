import os
import re
import json
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple
from collections import defaultdict, Counter

OBJECTS_IN_VIEW_TENSOR = "OBJECTS_IN_VIEW_TENSOR"
EMPTY_CELL = 0
PLAYER = 1
IRONRAW = 2
GOLDRAW = 3
GOLDPARTIAL = 4
APPLE = 5
ACORN = 6

PREY_KEYS = [f'player{i}' for i in range(4, 14)]
PREDATOR_KEYS = [f'player{i}' for i in range(1, 4)]
PREYS = "preys"
PREDATORS = "predators"
APPLE = "apple"
ACORN = "floorAcorn"

def build_tf_dataset(
    label_key: str,
    image_dir: str,
    label_file: str,
    batch_size: int,
    task: str,
    num_epochs: int = None,
    shuffle: bool = True,
    step_interval: int = 50,
    n_agents: int = 13,
    split_ratio: float = 0.8,
    balance_data: bool = True
):
    """
    Builds two datasets (train_ds, val_ds) from image + label pairs.
    """
    if task not in ["agent_loc"]:
        raise ValueError(f"Invalid task: {task}")
    
    # List of all images in the directory
    all_image_paths = [
        os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
        if fname.lower().endswith(".png") and 'world' not in fname
    ]
    
    # Load labels from JSON lines file
    with open(label_file, "r") as f:
        lines = f.readlines()
    labels = [json.loads(line) for line in lines]
    
    # Check that we have exactly n_agents images per label
    assert len(labels) * n_agents == len(all_image_paths), \
        f"Number of labels ({len(labels)}) does not match number of images ({len(all_image_paths)})"
    
    # Gather (filename, label) pairs
    filenames = []
    out_labels = []
    for filename in all_image_paths:
        step, agent_idx = _parse_step_and_agent(filename)
        label_dict = labels[step // step_interval]

        # Objects string for this agent
        objects_tensor = label_dict[f"agent_{agent_idx}"][OBJECTS_IN_VIEW_TENSOR]
        objects_loc_dict = _parse_objs_in_view_tensor(objects_tensor)

        # Binary classification => label is 0/1
        # Regression => label is integer count
        if task == "agent_loc":
            label_val = objects_loc_dict.get(label_key, [])
        else: 
            raise ValueError(f"Unknown task: {task}")

        # Skip if object of interest not present
        if not label_val:
            continue

        filenames.append(filename)
        out_labels.append(label_val)

    # Convert to np arrays for easy indexing
    filenames = np.array(filenames)
    out_labels = np.array(out_labels)

    # Shuffle if required (do this once before splitting)
    if shuffle:
        indices = np.random.permutation(len(filenames))
        filenames = filenames[indices]
        out_labels = out_labels[indices]
    
    # Compute split index
    split_index = int(len(filenames) * split_ratio)

    # Split into train / val
    train_filenames = filenames[:split_index]
    train_labels = out_labels[:split_index]
    val_filenames = filenames[split_index:]
    val_labels = out_labels[split_index:]
            
    print(f"Training label distribution: {Counter(train_labels)}")
    print(f"Validation label distribution: {Counter(val_labels)}")

    # Build the tf.data.Dataset objects
    train_ds = _build_single_dataset(
        train_filenames,
        train_labels,
        batch_size,
        num_epochs,
        shuffle=True
    )
    val_ds = _build_single_dataset(
        val_filenames,
        val_labels,
        batch_size,
        num_epochs,
        shuffle=False  # Typically do not shuffle validation.
    )

    return train_ds, val_ds


def _build_single_dataset(filenames, labels, batch_size, num_epochs, shuffle=False):
    """
    Helper to build a single dataset (either train or val).
    """
    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # (Re)Shuffle if needed (usually you'd shuffle only training here)
    if shuffle:
        ds = ds.shuffle(len(filenames))

    # Decode images
    ds = ds.map(_decode_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Repeat for given epochs
    if num_epochs is not None:
        ds = ds.repeat(num_epochs)

    # Batch & prefetch
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds.as_numpy_iterator()


def _decode_image(filename, label):
    """
    Reads and decodes the PNG image from file.
    """
    image_contents = tf.io.read_file(filename)
    image = tf.image.decode_image(image_contents, channels=3)
    return image, label


def _parse_step_and_agent(filename: str) -> Tuple[int, int]:
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


def _parse_objs_in_view_tensor(objects_tensor: str) -> Dict[int, Tuple[int, int]]:
    loc_dict = defaultdict(list)
    for info in objects_tensor:
        obj_type = info[0]
        x_loc = info[1]
        y_loc = info[2]
        # Skip empty cell and agent itself
        if obj_type == EMPTY_CELL or (x_loc == 0 and y_loc == 0):
            continue
        loc_dict[obj_type].append((x_loc, y_loc))
    return loc_dict



if __name__ == "__main__":
    train_ds, val_ds = build_tf_dataset(
        label_key=PLAYER,
        image_dir="data/coop_mining/2025-08-14_16:12:58.502508/agent_view_images",
        label_file="data/coop_mining/2025-08-14_16:12:58.502508/observations.jsonl",
        batch_size=16,
        task="agent_loc",
        num_epochs=2,
        shuffle=True,
        step_interval=1,
        n_agents=2,
        split_ratio=0.8
    )
    
    # Example: iterate over training batches
    print("TRAINING BATCHES:")
    for batch_images, batch_labels in train_ds:
        print("Train batch images shape:", batch_images.shape)
        print("Train batch labels shape:", batch_labels.shape)
        break  # Just one example batch
    
    # Example: iterate over validation batches
    print("\nVALIDATION BATCHES:")
    for batch_images, batch_labels in val_ds:
        print("Val batch images shape:", batch_images.shape)
        print("Val batch labels shape:", batch_labels.shape)
        break  # Just one example batch