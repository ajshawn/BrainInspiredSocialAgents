import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

N_TOTAL_AGENTS = 13
window_size = 20  # Choose a window size that suits your data

parser = argparse.ArgumentParser(description='Plot agent rewards with moving average smoothing.')
parser.add_argument('--prefix', '-p', type=str, required=True, help='Prefix for the results directory')
args = parser.parse_args()

csv_file = f'{args.prefix}/csv_logs/actor.csv'

df = pd.read_csv(csv_file)

# Remove rows with string values
df = df[~df['actor_episodes'].str.contains('actor_episodes', na=False)]

print(df.head())

for agent_idx in range(N_TOTAL_AGENTS):
    # Ensure Y contains only numeric values by dropping NaNs
    Y = pd.to_numeric(df[f'agent_{agent_idx}/episode_return'], errors='coerce').dropna()
    X = np.arange(len(Y))
    
    # Calculate moving average
    Y_smoothed = Y.rolling(window=window_size, min_periods=1).mean()
    
    # Set a larger figure size (width, height)
    plt.figure(figsize=(16, 8))

    # Plot original data
    plt.plot(X, Y, label='Original', alpha=0.3)  # Lower opacity for original data
    # Plot moving average
    plt.plot(X, Y_smoothed, label='Moving Average', linewidth=2)

    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'Agent {agent_idx} Reward (Smoothed)')
    plt.legend()

    # Save the plot
    plt.savefig(f'{args.prefix}/agent_{agent_idx}_reward_smoothed.png')

    # Clear the plot
    plt.clf()
