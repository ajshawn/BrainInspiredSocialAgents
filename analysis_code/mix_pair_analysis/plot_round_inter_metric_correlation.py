import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

def calculate_and_plot_correlation(df, metric_cols, save_dir, filename="correlation_matrix.png"):
    """
    Calculates the correlation matrix, handles lists/NaNs,
    clusters/sorts by reward correlation, and plots it.
    """

    available_metrics = [col for col in metric_cols if col in df.columns]
    if not available_metrics:
        print("Error: None of the specified metric columns found in the DataFrame.")
        return

    metric_df = df[available_metrics].copy()
    for col in metric_df.columns:
        metric_df[col] = metric_df[col].apply(lambda x: np.nanmean(x) if isinstance(x, list) else (-1 if pd.isna(x) else x))

    correlation_matrix = metric_df.corr()

    # Cluster/sort the correlation matrix
    linkage_matrix = linkage(correlation_matrix, method='ward')
    dendrogram_data = dendrogram(linkage_matrix, no_plot=True)
    ordered_indices = dendrogram_data['leaves']
    clustered_corr = correlation_matrix.iloc[ordered_indices, ordered_indices]

    # Sort rows and columns by correlation with reward metrics
    reward_corr = clustered_corr[['mean_prey_reward']].copy()
    reward_corr['reward_avg'] = reward_corr.mean(axis=1)
    sorted_indices_reward = reward_corr.sort_values(by='reward_avg', ascending=False).index

    clustered_corr = clustered_corr.loc[sorted_indices_reward, sorted_indices_reward]

    # Plot the clustered/sorted correlation matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(clustered_corr, annot=True, cmap='RdBu_r', fmt=".2f", cbar=False)
    plt.title("Reward-Sorted Correlation Matrix of Behavioral Metrics")
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Reward-sorted correlation matrix saved to {save_path}")
    plt.close()

# Example usage
if __name__ == '__main__':
    combined_df_path = './round_result_figures/combined_cumulative_results.pkl'
    figures_dir = './round_result_figures/correlation_results'

    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    try:
        with open(combined_df_path, 'rb') as f:
            combined_df = pickle.load(f)
        print(f"Successfully loaded DataFrame with shape: {combined_df.shape}")

        behavioral_metrics = [
            'pred_stuck_rate',
            'prey_stuck_rate',
            'mean_num_acorn_collected_per_round',
            'mean_num_apple_collected_per_round',
            'mean_prey_move_distances_per_round',
            'mean_predator_move_distances_per_round',
            'mean_time_on_grass_per_round',
            'mean_time_off_grass_per_round',
            'frac_off_grass_per_round',
            'frac_time_in_3_steps',
            'frac_time_in_5_steps',
            'mean_predator_rotate_per_round',
            'mean_prey_rotate_per_round',
            'mean_time_per_round',
            'mean_pred_reward',
            'mean_prey_reward',
            'dim',
        ]

        calculate_and_plot_correlation(combined_df, behavioral_metrics, figures_dir)

    except FileNotFoundError:
        print(f"Error: File not found at {combined_df_path}")
    except Exception as e:
        print(f"An error occurred: {e}")