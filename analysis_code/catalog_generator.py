import pandas as pd
import os

from plot_metrics_v2 import load_scenario_pickle_files
result_path = '/home/mikan/e/Documents/GitHub/social-agents-JAX/results/'
cp7357_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles/'
cp9651_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles/'
AH_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles/'
cp7357_perturb_pred_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles_perturb_predator/'
cp9651_perturb_pred_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles_perturb_predator/'
AH_perturb_pred_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_predator/'
cp7357_perturb_prey_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles_perturb_prey/'
cp9651_perturb_prey_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles_perturb_prey/'
AH_perturb_prey_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_prey/'
cp7357_perturb_both_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles_perturb_both/'
cp9651_perturb_both_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles_perturb_both/'
AH_perturb_both_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_both/'
AH_perturb_pred_25randPC_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_predator_25randPC_remTop10PLSCs/'
AH_perturb_prey_25randPC_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_prey_25randPC_remTop10PLSCs/'
AH_perturb_both_25randPC_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_both_25randPC_remTop10PLSCs/'

AH256_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles/'
Orchard256_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles/'
AH256_perturb_pred_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles_perturb_predator/'
Orchard256_perturb_pred_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles_perturb_predator/'
AH256_perturb_prey_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles_perturb_prey/'
Orchard256_perturb_prey_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles_perturb_prey/'
AH256_perturb_both_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles_perturb_both/'
Orchard256_perturb_both_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles_perturb_both/'

AH256_perturb_prey_30randPC_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles_perturb_prey_30randPC_remTop10PLSCs/'
AH256_perturb_pred_30randPC_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles_perturb_predator_30randPC_remTop10PLSCs/'
AH256_perturb_both_30randPC_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles_perturb_both_30randPC_remTop10PLSCs/'

Orchard256_perturb_prey_30randPC_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles_perturb_prey_30randPC_remTop10PLSCs/'
Orchard256_perturb_pred_30randPC_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles_perturb_predator_30randPC_remTop10PLSCs/'
Orchard256_perturb_both_30randPC_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles_perturb_both_30randPC_remTop10PLSCs/'

open256_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles/'
open256_perturb_prey_30randPC_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles_perturb_prey_30randPC_remTop10PLSCs/'
open256_perturb_pred_30randPC_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles_perturb_predator_30randPC_remTop10PLSCs/'
open256_perturb_both_30randPC_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles_perturb_both_30randPC_remTop10PLSCs/'
open256_perturb_prey_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles_perturb_prey/'
open256_perturb_pred_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles_perturb_predator/'
open256_perturb_both_path = f'{result_path}PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles_perturb_both/'

scenario_configs = {
  'open_cp7357': {
    'path': cp7357_path,
    'predator_ids': list(range(3)),
    'prey_ids': list(range(3, 13)),
  },
  'open_cp9651': {
    'path': cp9651_path,
    'predator_ids': list(range(3)),
    'prey_ids': list(range(3, 13)),
  },
  'AH': {
    'path': AH_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'open_cp7357_perturb_pred': {
    'path': cp7357_perturb_pred_path,
    'predator_ids': list(range(3)),
    'prey_ids': list(range(3, 13)),
  },
  'open_cp9651_perturb_pred': {
    'path': cp9651_perturb_pred_path,
    'predator_ids': list(range(3)),
    'prey_ids': list(range(3, 13)),
  },
  'AH_perturb_pred': {
    'path': AH_perturb_pred_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'open_cp7357_perturb_prey': {
    'path': cp7357_perturb_prey_path,
    'predator_ids': list(range(3)),
    'prey_ids': list(range(3, 13)),
  },
  'open_cp9651_perturb_prey': {
    'path': cp9651_perturb_prey_path,
    'predator_ids': list(range(3)),
    'prey_ids': list(range(3, 13)),
  },
  'AH_perturb_prey': {
    'path': AH_perturb_prey_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'open_cp7357_perturb_both': {
    'path': cp7357_perturb_both_path,
    'predator_ids': list(range(3)),
    'prey_ids': list(range(3, 13)),
  },
  'open_cp9651_perturb_both': {
    'path': cp9651_perturb_both_path,
    'predator_ids': list(range(3)),
    'prey_ids': list(range(3, 13)),
  },
  'AH_perturb_both': {
    'path': AH_perturb_both_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'AH_perturb_pred_25randPC': {
    'path': AH_perturb_pred_25randPC_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'AH_perturb_prey_25randPC': {
    'path': AH_perturb_prey_25randPC_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'AH_perturb_both_25randPC': {
    'path': AH_perturb_both_25randPC_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },

  'AH256': {
    'path': AH256_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  "Orchard256": {
    'path': Orchard256_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'AH256_perturb_pred': {
    'path': AH256_perturb_pred_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'Orchard256_perturb_pred': {
    'path': Orchard256_perturb_pred_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'AH256_perturb_prey': {
    'path': AH256_perturb_prey_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'Orchard256_perturb_prey': {
    'path': Orchard256_perturb_prey_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'AH256_perturb_both': {
    'path': AH256_perturb_both_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'Orchard256_perturb_both': {
    'path': Orchard256_perturb_both_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'AH256_perturb_prey_30randPC': {
    'path': AH256_perturb_prey_30randPC_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'AH256_perturb_pred_30randPC': {
    'path': AH256_perturb_pred_30randPC_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'AH256_perturb_both_30randPC': {
    'path': AH256_perturb_both_30randPC_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'Orchard256_perturb_prey_30randPC': {
    'path': Orchard256_perturb_prey_30randPC_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'Orchard256_perturb_pred_30randPC': {
    'path': Orchard256_perturb_pred_30randPC_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'Orchard256_perturb_both_30randPC': {
    'path': Orchard256_perturb_both_30randPC_path,
    'predator_ids': list(range(5)),
    'prey_ids': list(range(5, 13)),
  },
  'open256': {
    'path': open256_path,
    'predator_ids': list(range(3)),
    'prey_ids': list(range(3, 13)),
  },
  'open256_perturb_prey_30randPC': {
    'path': open256_perturb_prey_30randPC_path,
    'predator_ids': list(range(3)),
    'prey_ids': list(range(3, 13)),
  },
  'open256_perturb_pred_30randPC': {
    'path': open256_perturb_pred_30randPC_path,
    'predator_ids': list(range(3)),
    'prey_ids': list(range(3, 13)),
  },
  'open256_perturb_both_30randPC': {
    'path': open256_perturb_both_30randPC_path,
    'predator_ids': list(range(3)),
    'prey_ids': list(range(3, 13)),
  },
  'open256_perturb_prey': {
    'path': open256_perturb_prey_path,
    'predator_ids': list(range(3)),
    'prey_ids': list(range(3, 13)),
  },
  'open256_perturb_pred': {
    'path': open256_perturb_pred_path,
    'predator_ids': list(range(3)),
    'prey_ids': list(range(3, 13)),
  },
  'open256_perturb_both': {
    'path': open256_perturb_both_path,
    'predator_ids': list(range(3)),
    'prey_ids': list(range(3, 13)),
  },

}
df = pd.DataFrame.from_dict(scenario_configs, orient='index')

# Verify if all the paths are valid and print if we are missing any files
df['valid'] = df['path'].apply(lambda x: os.path.exists(x))
for scenario, config in scenario_configs.items():
  try:
    os.listdir(config['path'])
  except:
    print(f"Missing path for {scenario}")

  # # Load the data
  # try:
  #   data = load_scenario_pickle_files(config['path'], config['predator_ids'], config['prey_ids'])
  #   print(f"Loaded {scenario}")
  # except:
  #   print(f"Failed to load {scenario}")

df = df.sort_index()
df.to_csv('./catalog.csv')


