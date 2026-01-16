from collections import Counter

import numpy as np
import csv
import ast
import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split
from lightning.pytorch.loggers import WandbLogger
from sklearn.preprocessing import LabelEncoder
import torchmetrics

PLAYER = 1
IRONRAW = 2
GOLDRAW = 3
GOLDPARTIAL = 4

def preprocess_csv(csv_file):
    episodes = []
    with open(csv_file, newline='\n', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data = [ast.literal_eval(cell) for cell in row if cell]
            keys = data[0].keys()
            episodes.append({key: np.vstack([entry[key] for entry in data]) for key in keys})
    return episodes


def prepare_decode_loc_data(
        episodes, 
        agent_idx, 
        n_agents, 
        decode_loc,
        feature_key='embedding', 
        label_key='obj_in_view',
        random_features=False
    ):
    """
    Prepares features and labels for a specific agent for encoder probing with location decoding.

    Args:
        episodes (list): List of episode dictionaries containing features and labels.
        agent_idx (int): Index of the agent to extract data for.
        n_agents (int): Total number of agents in the environment.
        decode_loc (str): Location to decode, x or y.
        feature_key (str): Key to access features in the episode dictionary.
        label_key (str): Key to access labels in the episode dictionary.
    """
    features = []
    labels = []

    assert agent_idx < n_agents, "agent_idx must be less than n_agents"

    for epi in episodes:
        all_features = epi[feature_key]
        all_objects = epi[label_key]

        # Group by agent
        agent_features = all_features[agent_idx::n_agents]
        agent_objects = all_objects[agent_idx::n_agents]

        # Extract labels based on decode_loc
        for idx, object_list in enumerate(agent_objects):                
            for info_triplets in object_list:
                obj_type, x_pos, y_pos = info_triplets
                # Only consider other agents for location decoding
                if obj_type == PLAYER and (abs(x_pos) + abs(y_pos)) > 0:
                    if decode_loc == 'x':
                        labels.append(x_pos)
                    elif decode_loc == 'y':
                        labels.append(y_pos)
                    else:
                        raise ValueError("decode_loc must be 'x' or 'y'")
                    features.append(agent_features[idx][10:])
                    break # Found the player, move to next timestep

    print(f"Label distribution for decode_loc '{decode_loc}': {Counter(labels)}")
    
    X = np.array(features).squeeze()
    y = np.array(labels)

    if random_features:
        X = np.random.randn(*X.shape)

    return X, y
        

class LinearProbe(L.LightningModule):
    def __init__(self, input_dim, num_classes, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Linear(input_dim, num_classes)
        # hidden_dim = 128
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, num_classes)
        # )
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Use torchmetrics for cleaner accuracy tracking
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Update and log validation accuracy
        self.val_acc(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def train(X, y, num_classes=None, batch_size=32, epochs=10, train_val_split=0.8, run_name=""):
    """
    X: np.array of shape (N, dim)
    y: np.array of shape (N,) with arbitrary discrete labels
    """
    # Map arbitrary labels to [0, num_classes - 1]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Update num_classes based on actual data if not provided
    if num_classes is None:
        num_classes = len(le.classes_)
    
    print(f"Mapped labels: {le.classes_} to {np.unique(y_encoded)}")

    # Convert NumPy arrays to PyTorch Tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y_encoded).long()

    # Create Train/Val Split
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(len(dataset) * train_val_split)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # Get validation label distribution and chance levels
    val_labels = [y.item() for _, y in val_set]
    total_val_samples = len(val_labels)

    val_distribution = Counter(val_labels)

    # Majority Class (Zero-Rule)
    max_class_count = max(val_distribution.values())
    majority_baseline = (max_class_count / total_val_samples) * 100

    # Proportional Random Guessing
    # Sum of squared proportions: P(class1)^2 + P(class2)^2 + ...
    proportional_baseline = sum([(count / total_val_samples)**2 
                                for count in val_distribution.values()]) * 100

    print("-" * 30)
    print(f"VALIDATION SET STATS (N={total_val_samples})")
    print("-" * 30)

    for label, count in sorted(val_distribution.items()):
        pct = (count / total_val_samples) * 100
        print(f"Class {label}: {count:4d} samples ({pct:.2f}%)")

    print("-" * 30)
    print(f"Majority Class Baseline: {majority_baseline:.2f}%")
    print(f"Proportional Chance Level: {proportional_baseline:.2f}%")
    print("-" * 30)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Initialize Model and Logger
    input_dim = X.shape[1]
    model = LinearProbe(input_dim=input_dim, num_classes=num_classes)
    
    wandb_logger = WandbLogger(project="marl-linear-probe", name=run_name)

    # Train Model
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        check_val_every_n_epoch=1  # Validates every epoch
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    return model, le

if __name__ == "__main__":
    file_path = 'data/coop_mining179-timesteps-attnvalue.csv'
    decode_loc = 'x'  # or 'y'
    episodes = preprocess_csv(file_path)
    agent_idx = 0
    random_features = True

    feature = file_path.split('-')[-1].replace('.csv', '')
    name = f"probe_agent{agent_idx}_decode_{decode_loc}_{feature}"
    if random_features:
        name += "_random_baseline"

    X, y = prepare_decode_loc_data(
        episodes, 
        agent_idx=agent_idx,
        n_agents=2, 
        decode_loc=decode_loc,
        feature_key='embedding', 
        label_key='obj_in_view',        
        random_features=random_features
    )

    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)

    model, label_encoder = train(X, y, batch_size=32, epochs=200, run_name=name)