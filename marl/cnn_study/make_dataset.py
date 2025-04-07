import os
import re
import json
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple
from collections import defaultdict, Counter

OBJECTS_IN_VIEW = "OBJECTS_IN_VIEW"
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
    if task not in ["binary_cls", "regression"]:
        raise ValueError(f"Invalid task: {task}")
    
    # List of all images in the directory
    all_image_paths = [
        os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
        if fname.lower().endswith(".png")
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
        objects_str = label_dict[f"agent_{agent_idx}"][OBJECTS_IN_VIEW]
        objects_dict = _parse_objs_in_view(objects_str, agent_idx)

        # Binary classification => label is 0/1
        # Regression => label is integer count
        if task == "binary_cls":
            label_val = int(objects_dict.get(label_key, 0) > 0)
        else:  # "regression"
            label_val = objects_dict.get(label_key, 0)

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
        
    # Optionally balance each split if this is a binary classification task
    if balance_data:
        if task == "binary_cls":    
            train_filenames, train_labels = _balance_dataset_cls(train_filenames, train_labels)
            val_filenames, val_labels = _balance_dataset_cls(val_filenames, val_labels)
        else:
            train_filenames, train_labels = _balance_dataset_regression(train_filenames, train_labels)
            val_filenames, val_labels = _balance_dataset_regression(val_filenames, val_labels)
    
    # print(f"Training label distribution: {Counter(train_labels)}")
    # print(f"Validation label distribution: {Counter(val_labels)}")

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

def _balance_dataset_cls(filenames: np.ndarray, labels: np.ndarray, seed: int = 42):
    """
    Simple undersampling to balance 0/1 labels.
    Returns balanced filenames, labels arrays.
    """
    np.random.seed(seed)
    
    labels = labels.astype(int)  # Ensure integer
    class0_indices = np.where(labels == 0)[0]
    class1_indices = np.where(labels == 1)[0]

    # Undersample: pick the same number of items from each class
    n_class0 = len(class0_indices)
    n_class1 = len(class1_indices)
    n_to_sample = min(n_class0, n_class1)
    
    sampled_class0 = np.random.choice(class0_indices, n_to_sample, replace=False)
    sampled_class1 = np.random.choice(class1_indices, n_to_sample, replace=False)
    
    new_indices = np.concatenate([sampled_class0, sampled_class1])
    np.random.shuffle(new_indices)

    return filenames[new_indices], labels[new_indices]

def _balance_dataset_regression(filenames: np.ndarray, labels: np.ndarray, max_per_label: int = 300, seed: int = 42):
    """
    Caps the maximum number of samples for each unique label to `max_per_label`.
    Suitable for regression tasks.
    
    Returns capped filenames and labels arrays.
    """
    np.random.seed(seed)

    # Convert labels to float (if not already)
    labels = labels.astype(float)
    
    # Group by unique labels (if labels are floats, you may want to round them first)
    # Optional: adjust rounding depending on your task
    rounded_labels = np.round(labels, decimals=2)  # adjust decimals if needed
    
    unique_labels = np.unique(rounded_labels)
    selected_indices = []

    for lbl in unique_labels:
        indices = np.where(rounded_labels == lbl)[0]
        if len(indices) > max_per_label:
            sampled = np.random.choice(indices, max_per_label, replace=False)
        else:
            sampled = indices
        selected_indices.extend(sampled)

    selected_indices = np.array(selected_indices)
    
    return filenames[selected_indices], labels[selected_indices]


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


def _parse_objs_in_view(input_str: str, agent_idx: int) -> Dict[str, int]:
    parts = input_str.split(",")
    parts = [p.strip() for p in parts]
    if len(parts) % 2 != 0:
        raise ValueError(f"Invalid input string: {input_str}")
    result = defaultdict(int)
    for i in range(0, len(parts), 2):
        key = parts[i]
        value = int(parts[i + 1])
        # Skip the player if it's the agent itself
        if key == f"player{agent_idx+1}":
            continue
        if key in PREY_KEYS:
            result[PREYS] += value
        elif key in PREDATOR_KEYS:
            result[PREDATORS] += value
        else:
            result[key] += value
    return result


if __name__ == "__main__":
    train_ds, val_ds = build_tf_dataset(
        label_key=APPLE,
        image_dir="data/predator_prey__open_random/agent_view_images",
        label_file="data/predator_prey__open_random/observations.jsonl",
        batch_size=16,
        task="binary_cls",
        num_epochs=2,
        shuffle=True,
        step_interval=50,
        n_agents=13,
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
