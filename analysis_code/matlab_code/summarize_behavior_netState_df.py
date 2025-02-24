import pickle
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def ori_position(A_pos, A_orient, B_pos):
  orientation_transform = {
    0: lambda x, y: (x, y),  # UP
    1: lambda x, y: (y, -x),  # RIGHT
    2: lambda x, y: (-x, -y),  # DOWN
    3: lambda x, y: (-y, x)  # LEFT
  }
  transform = orientation_transform[A_orient]
  delta_x = B_pos[0] - A_pos[0]
  delta_y = B_pos[1] - A_pos[1]
  relative_x, relative_y = transform(delta_x, delta_y)
  return (relative_x, relative_y)


def process_pair(predator_id, prey_id, video_path):
  info_dict = {}
  network_states_dict = {}
  serial_data_dict = {}
  title = f'{predator_id}_{prey_id}'
  print(f'Processing {title}')
  for eId in range(1, 101):
    try:
      with open(f'{video_path}{title}_{eId}.pkl', 'rb') as f:
        results = pickle.load(f)
    except:
      print(f'Error loading {video_path}{title}_{eId}.pkl')
      continue
    if title not in serial_data_dict:
      serial_data_dict[title] = {k: [] for k in
                                 ['STAMINA', 'POSITION', 'ORIENTATION', 'rewards', 'actions', 'distances']}

    serial_data_dict[title]['STAMINA'].append([info['STAMINA'] for info in results])
    serial_data_dict[title]['POSITION'].append([info['POSITION'] for info in results])
    serial_data_dict[title]['ORIENTATION'].append([info['ORIENTATION'] for info in results])
    serial_data_dict[title]['rewards'].append([info['rewards'] for info in results])
    serial_data_dict[title]['actions'].append([info['actions'] for info in results])

    duration = len(results)
    if 'predator_id' not in info_dict:
      info_dict['predator_id'] = []
    info_dict['predator_id'].extend([predator_id] * duration)
    if 'prey_id' not in info_dict:
      info_dict['prey_id'] = []
    info_dict['prey_id'].extend([prey_id] * duration)
    if 'eId' not in info_dict:
      info_dict['eId'] = []
    info_dict['eId'].extend([eId] * duration)

    for key in ['STAMINA', 'POSITION', 'ORIENTATION', 'rewards', 'actions']:
      for agent_id in range(2):
        key_id = key + f'_{agent_id}'
        if key_id not in info_dict:
          info_dict[key_id] = []
        info_dict[key_id].extend([info[key][agent_id] for info in results])

    positions = np.array([info['POSITION'] for info in results])
    orientations = np.array([info['ORIENTATION'] for info in results]).astype(int)
    for agent_id in range(2):
      ori_positions = []
      for i in range(len(orientations)):
        opponent_id = 1 - agent_id
        ori_positions.append(ori_position(positions[i, agent_id], orientations[i, agent_id], positions[i, opponent_id]))
      ori_positions = np.array(ori_positions).astype(int)
      for ii, xyname in enumerate(['x', 'y']):
        if f'rel_position_{agent_id}_{xyname}' not in info_dict:
          info_dict[f'rel_position_{agent_id}_{xyname}'] = []
        info_dict[f'rel_position_{agent_id}_{xyname}'].extend(ori_positions[:, ii])

    for state_names in ['hidden', 'cell']:
      for agent_id in range(2):
        network_states = [info[state_names][agent_id] for info in results]
        network_states = np.array(network_states)
        for i in range(network_states.shape[1]):
          if f'{state_names}_{agent_id}_{i}' not in network_states_dict:
            network_states_dict[f'{state_names}_{agent_id}_{i}'] = []
          network_states_dict[f'{state_names}_{agent_id}_{i}'].extend(network_states[:, i])

    actions = np.array([info['actions'] for info in results])
    for agent_id, tmp in enumerate([actions[:, 0], actions[:, 1]]):
      action_one_hot = np.zeros((len(tmp), 8))
      action_one_hot[np.arange(len(tmp)), tmp] = 1
      for i in range(8):
        if f'actions_{agent_id}_{i}' not in info_dict:
          info_dict[f'actions_{agent_id}_{i}'] = []
        info_dict[f'actions_{agent_id}_{i}'].extend(action_one_hot[:, i])

    distances = []
    for i in range(2):
      position = positions[:, i]
      distance = np.linalg.norm(position[1:] - position[:-1], axis=1)
      distance = np.concatenate([[0], distance])
      distances.append(distance)
    serial_data_dict[title]['distances'].append(distances)

  info_df = pd.DataFrame(info_dict)
  network_states_df = pd.DataFrame(network_states_dict)
  info_df.to_csv(f'{video_path}{title}_info.csv', index=False)
  network_states_df.to_csv(f'{video_path}{title}_network_states.csv', index=False)


if __name__ == '__main__':

  video_paths = [
    f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles/',
    f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles/',
    f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles/',
  ]
  for video_path in video_paths:
    if 'open' in video_path:
      predator_ids = [0, 1, 2]
      prey_ids = list(range(3, 13))
    else:
      predator_ids = [0, 1, 2, 3, 4]
      prey_ids = list(range(5, 13))
    # Using joblib's Parallel and delayed to parallelize the process_pair function
    Parallel(n_jobs=20)(delayed(process_pair)(predator_id, prey_id, video_path) for predator_id in predator_ids for prey_id in prey_ids)
