import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV, skipping bad lines
table_raw = pd.read_csv(
    "/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__random_forest_2025-04-24_17:23:56.924098/csv_logs/learner.csv",
    on_bad_lines='skip'
)

# table_raw = pd.read_csv(
#   "/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274/csv_logs/learner.csv",
#   on_bad_lines='skip'
# )

select_col = [[
'agent_0/extrinsic_reward',
 'agent_1/extrinsic_reward',
 'agent_2/extrinsic_reward',
 'agent_3/extrinsic_reward',
 'agent_4/extrinsic_reward',
],
  [
 'agent_5/extrinsic_reward',
 'agent_6/extrinsic_reward',
 'agent_7/extrinsic_reward',
 'agent_8/extrinsic_reward',
 'agent_9/extrinsic_reward',
 'agent_10/extrinsic_reward',
 'agent_11/extrinsic_reward',
 'agent_12/extrinsic_reward',
  ]
  ]

# for name, col in zip(["predator", "prey"], select_col):
#   # Select only the relevant columns
#   table = table_raw[col]
#
#   # Apply 100-row moving average to numeric columns
#   smoothed = table.select_dtypes(include='number').rolling(window=100).mean()
#
#   # Downsample to every 1000th row
#   downsampled = smoothed.iloc[::1000]
#
#   # Plot each column in its own subplot
#   num_cols = downsampled.shape[1]
#   fig, axs = plt.subplots(num_cols, 1, figsize=(10, 2 * num_cols), sharex=True, sharey=True)
#
#   if num_cols == 1:
#       axs = [axs]
#
#   for ax, col in zip(axs, downsampled.columns):
#       ax.plot(downsampled.index, downsampled[col])
#       ax.set_title(col)
#
#   plt.tight_layout()
#   # plt.savefig(f"plot_RF_training_rew_{name}.png", dpi=300)
#   plt.savefig(f"plot_OR_training_rew_{name}.png", dpi=300)
#   plt.show()

# Instead of plotting each column in its own subplot, plot all columns in a single plot and legend them
for name, col in zip(["predator", "prey"], select_col):
  # Select only the relevant columns
  table = table_raw[col]

  # Apply 100-row moving average to numeric columns
  smoothed = table.select_dtypes(include='number').rolling(window=100).mean()

  # Downsample to every 1000th row
  downsampled = smoothed.iloc[::1000]

  # Plot all columns in a single plot
  plt.figure(figsize=(10, 6))
  downsampled.plot(ax=plt.gca(), legend=True)
  plt.title(f"{name} extrinsic reward")
  plt.xlabel("Episode")
  plt.ylabel("Reward")
  # plt.legend()
  plt.tight_layout()
  plt.savefig(f"plot_RF_training_rew_{name}.png", dpi=300)
  # plt.savefig(f"plot_OR_training_rew_{name}.png", dpi=300)
  plt.show()
