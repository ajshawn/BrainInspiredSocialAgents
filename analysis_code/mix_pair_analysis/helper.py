import numpy as np
import pandas as pd
import os
import pickle
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import cv2

def check_pair_data():
  # Directory where the pickle files are located.
  serial_results_dir = os.path.expanduser("~/Documents/GitHub/social-agents-JAX/results/mix/analysis_results/")

  # Get the list of all pickle files in that directory.
  file_list = glob.glob(os.path.join(serial_results_dir, "*_serial_results.pkl"))

  # Extract the pair names (remove the directory and the suffix).
  pair_names = [os.path.basename(f).replace("_serial_results.pkl", "") for f in file_list]

  # Extract predator and prey from each pair.
  # Assuming the format "predator_vs_prey"
  predators = [pair.split('_vs_')[0] for pair in pair_names]
  preys = [pair.split('_vs_')[1] for pair in pair_names]

  # Create a DataFrame with the information.
  df_pairs = pd.DataFrame({'predator': predators, 'prey': preys})

  # Create a pivot table (or crosstab) that indicates the presence (1) or absence (0) of each predator-prey pair.
  # The crosstab counts the occurrence; we convert any nonzero count to 1.
  pivot = pd.crosstab(df_pairs['predator'], df_pairs['prey'])
  binary_pivot = (pivot > 0).astype(int)

  # Plot the binary matrix as a heatmap.
  plt.figure(figsize=(10, 8))
  sns.heatmap(binary_pivot, annot=True, fmt="d", cmap='coolwarm', cbar=False)
  plt.title("Predator-Prey Pair Existence (1 = exists, 0 = missing)")
  plt.xlabel("Prey")
  plt.ylabel("Predator")
  plt.tight_layout()
  plt.show()

# Now produce a template background figure extracted from one of the rollout video
def plot_background_template():
  # Load the template image (assuming it's in the same directory).
  template_video_path = os.path.expanduser("/home/mikan/e/Documents/GitHub/social-agents-JAX/recordings/meltingpot/predator_prey__simplified10x10_OneVsOne/2025-04-14 15:23:11/1.mp4")
  # Open the video file.

  cap = cv2.VideoCapture(template_video_path)
  # Read the first frame.
  ret, frame = cap.read()
  # If the frame was read successfully, display it.
  if ret:
    plt.imshow(frame)
    plt.axis('off')  # Hide axis
    plt.title("Background Template")
    plt.tight_layout()
    plt.show()
  else:
    print("Failed to read video frame.")
    return
  # if the frame successfully read, save it as a png file
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
  cv2.imwrite("./serial_clustering_results/background_template.png", frame)

