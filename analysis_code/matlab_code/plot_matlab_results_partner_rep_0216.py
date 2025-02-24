import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_histograms_seaborn(ax, results, xlabel=None, title=None, legends=None):
  """
  Plots histograms for positive, negative, and significant indices using Seaborn on a given subplot.

  Parameters:
      ax (matplotlib.axes._subplots.AxesSubplot): Axes object to plot on.
      results (dict): A dictionary with fields `mAll`, `posIdxAll`, `negIdxAll`, and `sigIdxAll`.
      legends (list): List of legend labels for the histograms, e.g., ['PLSC1', 'U1', 'Both'].
      xlabel (str): Label for the x-axis.
      title (str): Title of the subplot.
  """
  mAll = np.array(results['mAll'])
  posIdxAll = np.array(results['posIdxAll'], dtype=bool)
  negIdxAll = np.array(results['negIdxAll'], dtype=bool)
  sigIdxAll = np.array(results['sigIdxAll'], dtype=bool)



  # Plot histograms with Seaborn
  if legends is None:
    legends = ['1', '2', '3']
  # Create a DataFrame for Seaborn
  data = pd.DataFrame({
    'Value': np.concatenate([mAll[posIdxAll], mAll[negIdxAll], mAll[sigIdxAll]]),
    'Category': ([legends[0]] * sum(posIdxAll) +
                 [legends[1]] * sum(negIdxAll) +
                 [legends[2]] * sum(sigIdxAll))
  })

  sns.histplot(data=data, x='Value', hue='Category', ax=ax,
               bins=np.arange(min(mAll), max(mAll) + 0.01, 0.01),
               palette={legends[0]: (0.8, 0, 0), legends[1]: (0, 0, 0.8), legends[2]: (0.5, 0, 0.5)},
               alpha=0.6, kde=True, stat='percent')

  # Add labels and title
  ax.set_ylabel('Cells Percent')
  ax.set_xlabel(xlabel)
  ax.set_title(f"Total Cells: {len(mAll)}\n{title}")

  if legends is not None:
    ax.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)


if __name__ == '__main__':
  result_path = './results/'
  figure_path = './figures_partner_rep_0216/'
  if not os.path.exists(figure_path):
    os.makedirs(figure_path)
  rollout_pairs = ['cp9651', 'AH',]
  bv_types = ['act+ori+sta', 'act','ori+sta','act_prob+ori+sta', 'act_prob']

  rollout_pairs = [name + '_' + bv_type for name in rollout_pairs for bv_type in bv_types]
  result_dict = {}
  for rollout_pair in rollout_pairs:
    # tmp = sio.loadmat(f'{result_path}plsc1_vs_2_{rollout_pair}.mat', squeeze_me=True, simplify_cells=True)
    # result_dict['plsc1_vs_2_' + rollout_pair] = tmp['results']
    # tmp = sio.loadmat(f'{result_path}plsc1_vs_u1_{rollout_pair}.mat', squeeze_me=True, simplify_cells=True)
    # result_dict['plsc1_vs_u1_' + rollout_pair] = tmp['results']
    tmp = sio.loadmat(f'{result_path}partner_rep_result_{rollout_pair}.mat', squeeze_me=True, simplify_cells=True)
    result_dict['partner_rep_' + rollout_pair] = tmp['result_dict']
    # tmp = sio.loadmat(f'{result_path}non_selective_partner_rep_result_{rollout_pair}.mat', squeeze_me=True, simplify_cells=True)
    # result_dict['redundant_partner_rep_' + rollout_pair] = tmp['result_dict']

  ## heatmap and Boxplot the neural variance explained by partner
  column_min = []
  column_max = []

  for rollout_pair in rollout_pairs:
    partner_rep = result_dict['partner_rep_' + rollout_pair]
    for key in ['prey_self_rep_matrix', 'prey_partner_rep_matrix', 'predator_self_rep_matrix',
                'predator_partner_rep_matrix']:
      column_min.append(np.min(partner_rep[key]))
      column_max.append(np.max(partner_rep[key]))
  column_min = np.reshape(column_min, (len(rollout_pairs), 4)).min(axis=0)
  column_max = np.reshape(column_max, (len(rollout_pairs), 4)).max(axis=0)
  # fig, axs = plt.subplots(len(rollout_pairs), 4, figsize=(16, 12), sharex=True, sharey=True)
  fig, axs = plt.subplots(len(rollout_pairs), 4, figsize=(12, 20))

  for i, rollout_pair in enumerate(rollout_pairs):
    partner_rep = result_dict['partner_rep_' + rollout_pair]
    for j, key in enumerate(['prey_self_rep_matrix', 'prey_partner_rep_matrix', 'predator_self_rep_matrix',
                             'predator_partner_rep_matrix']):

      if rollout_pair in ['cp7357', 'cp9651']:
        sns.heatmap(partner_rep[key][:3,3:13] * 100, ax=axs[i, j], cmap='RdBu_r', cbar=False, annot=True,
                    vmin=column_min[j] *100, vmax=column_max[j] *100)
      else:
        predator_ids = list(range(5))
        prey_ids = list(range(5, 13))
        sns.heatmap(partner_rep[key][:5, 5:13] * 100, ax=axs[i, j], cmap='RdBu_r', cbar=False, annot=True,
                    vmin=column_min[j] *100, vmax=column_max[j] *100)
      axs[i, j].set_title(key) if i == 0 else None

      # Only hide axis ticks and spine for aesthetics
      axs[i, j].set_xticks([])
      axs[i, j].set_yticks([])
      if j == 0:
        axs[i, j].set_ylabel('predator_id')
      if i == len(rollout_pairs) - 1:
        axs[i, j].set_xlabel('prey_id')


    # Set y-label for the first column
    axs[i, 0].set_ylabel(f'{rollout_pair} (%)\npredator_id')

  plt.suptitle('Neural variance explained by partner (in %)')
  plt.tight_layout()
  plt.savefig(f'{figure_path}partner_rep_heatmap.png')
  plt.show()

  ## Boxplot the neural variance explained by partner
  fig, axs = plt.subplots(1,4, figsize=(20,8), sharey=True, sharex=True)
  for j, key in enumerate(['prey_self_rep_matrix', 'prey_partner_rep_matrix', 'predator_self_rep_matrix',
                           'predator_partner_rep_matrix']):
    data = []
    for i, rollout_pair in enumerate(rollout_pairs):
      partner_rep = result_dict['partner_rep_' + rollout_pair]
      if rollout_pair == 'tptp_perturb_predator':
        data.append(partner_rep[key][:,[3, 4, 14, 15, 16]].flatten() * 100)
      else:
        data.append(partner_rep[key].flatten() * 100)
    sns.boxplot(data=data, ax=axs[j], palette='Set3')
    sns.swarmplot(data=data, ax=axs[j], color=".25")
    axs[j].set_title(key)
    axs[j].set_xticklabels(rollout_pairs, rotation=90)
    axs[j].set_ylabel('Variance explained (%)')
  plt.tight_layout()
  plt.savefig(f'{figure_path}partner_rep_boxplot.png')
  plt.show()

