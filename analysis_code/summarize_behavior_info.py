import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from helper import compute_stuck_rate

if __name__ == '__main__':
  video_paths = [
    f'/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles/',
    f'/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles/',
    f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles_perturb_predator/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles_perturb_predator/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_predator/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles_perturb_prey/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles_perturb_prey/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_prey/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles_perturb_both/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles_perturb_both/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_both/',
    '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles_perturb_prey/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles_perturb_predator/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles_perturb_both/',
    '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles_perturb_prey/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles_perturb_predator/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles_perturb_both/',
  #   f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_predator_25randPC_remTop10PLSCs/',
  #   f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_prey_25randPC_remTop10PLSCs/',
  #   f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles_perturb_both_25randPC_remTop10PLSCs/',
    f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles/',
  #   f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles_perturb_predator/',
  #   f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles_perturb_prey/',
  #   f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles_perturb_both/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles_perturb_predator_30randPC_remTop10PLSCs/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles_perturb_prey_30randPC_remTop10PLSCs/',
    # f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles_perturb_both_30randPC_remTop10PLSCs/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles_perturb_prey_30randPC_remTop10PLSCs/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles_perturb_predator_30randPC_remTop10PLSCs/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles_perturb_both_30randPC_remTop10PLSCs/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles_perturb_prey_30randPC_remTop10PLSCs/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles_perturb_predator_30randPC_remTop10PLSCs/',
    # '/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles_perturb_both_30randPC_remTop10PLSCs/',

  ]
  for video_path in video_paths:
    if 'open' in video_path:
      predator_ids = [0, 1, 2]
      prey_ids = list(range(3, 13))
    else:
      predator_ids = [0, 1, 2, 3, 4]
      prey_ids = list(range(5, 13))
    serial_data_dict = {}
    cumulative_dict = {}
    for predator_id in predator_ids:
      for prey_id in prey_ids:
        title = f'{predator_id}_{prey_id}'
        for eId in range(1, 101):
          try:
            with open(f'{video_path}{title}_{eId}.pkl', 'rb') as f:
              results = pickle.load(f)
          except:
            print(f'Error loading {video_path}{title}_{eId}.pkl')
            continue
          if title not in serial_data_dict:
            serial_data_dict[title] = {title: [] for title in [
              'INVENTORY', 'READY_TO_SHOOT', 'STAMINA', 'POSITION', 'ORIENTATION', 'rewards', 'actions','distances',
              'stuck_indicator']}
          serial_data_dict[title]['STAMINA'].append([info['STAMINA'] for info in results])
          serial_data_dict[title]['POSITION'].append([info['POSITION'] for info in results])
          serial_data_dict[title]['ORIENTATION'].append([info['ORIENTATION'] for info in results])
          serial_data_dict[title]['rewards'].append([info['rewards'] for info in results])
          serial_data_dict[title]['actions'].append([info['actions'] for info in results])

          # Now, calculate the distance moved by each agent in each episode
          distances = []
          positions = np.array([info['POSITION'] for info in results])
          num_agents = 2
          for i in range(num_agents):
            position = positions[:, i]
            distance = np.linalg.norm(position[1:] - position[:-1], axis=1)
            distance = np.concatenate([[0], distance])
            distances.append(distance)
          serial_data_dict[title]['distances'].append(distances)

          # Compute stuck rates for each agent over the entire episode.
          pred_pos = positions[:, 0, :]
          prey_pos = positions[:, 1, :]
          pred_stuck, pred_stuck_ind = compute_stuck_rate(pred_pos, min_duration=20)
          prey_stuck, prey_stuck_ind = compute_stuck_rate(prey_pos, min_duration=20)
          serial_data_dict[title]['stuck_indicator'].append(
            [pred_stuck_ind.astype(bool), prey_stuck_ind.astype(bool)])

          # Now, calculate the time till catch
          rewards = np.array([info['rewards'] for info in results])
          # Processing logic for safe area, catching, and apple collection
          safe_grass = [[i, j] for i in [8, 9, 10] for j in [4, 5, 6]]
          t_catch = np.where(rewards[:, 0] == 1)[0]
          t_apple = np.where(rewards[:, 1] == 1)[0]
          t_acorn = np.where(rewards[:, 1] > 1)[0]
          if len(t_acorn) > 0:
            print(f'Acorn collected at {t_acorn[0]}')
            pass
          t_respawn = t_catch + 21
          t_respawn = np.insert(t_respawn, 0, 0)

          if title not in cumulative_dict:
            summary_titles = ['round', 'time_per_round', 'prey_move_distances_per_round',
                              'predator_move_distances_per_round', 'num_apple_collected_per_round',
                              'num_acorn_collected_per_round',
                              'prey_rotate_per_round', 'predator_rotate_per_round',
                              'time_on_grass_per_round', 'time_off_grass_per_round',
                              'frac_off_grass_per_round',
                              'frac_moving_away_per_round', 'percent_time_in_3_steps',
                              'percent_time_in_5_steps',
                              'pred_stuck_rate', 'prey_stuck_rate',]
            cumulative_dict[title] = {title: [] for title in summary_titles}

          for round, t_start in enumerate(t_respawn[:-1]):
            t_end = np.min([t_respawn[round + 1], len(rewards)])
            t_leave_safe = np.nan
            for ti in range(t_start, t_end):
              if list(positions[ti, 1, :]) not in safe_grass:
                t_leave_safe = ti
                break

            if np.isnan(t_leave_safe):
              continue

            t_catch_i = t_catch[t_catch >= t_leave_safe]
            t_catch_i = t_catch_i[0] if len(t_catch_i) > 0 else np.nan
            time_per_round = t_catch_i - t_leave_safe
            t_apple_i = t_apple[(t_leave_safe <= t_apple) & (t_apple < t_end)]
            t_acorn_i = t_acorn[(t_leave_safe <= t_acorn) & (t_acorn < t_end)]
            num_apple = len(t_apple_i)
            num_acorn = len(t_acorn_i)

            # Compute stuck rates for each agent over the current round.
            pred_stuck, pred_stuck_ind = compute_stuck_rate(pred_pos[t_start:t_end], min_duration=20)
            prey_stuck, prey_stuck_ind = compute_stuck_rate(prey_pos[t_start:t_end], min_duration=20)
            cumulative_dict[title]['pred_stuck_rate'].append(pred_stuck)
            cumulative_dict[title]['prey_stuck_rate'].append(prey_stuck)

            cumulative_dict[title]['round'].append(f'{eId}_round{round}')
            cumulative_dict[title]['time_per_round'].append(time_per_round)
            cumulative_dict[title]['num_apple_collected_per_round'].append(num_apple)
            cumulative_dict[title]['num_acorn_collected_per_round'].append(num_acorn)

            # First, get total distances of move
            total_distances = np.sum(np.abs(distances)==1, axis=1)
            cumulative_dict[title]['total_distances'] = total_distances
            prey_move_distances_per_round = np.sum(distances[1][t_start:t_end])
            predator_move_distances_per_round = np.sum(distances[0][t_start:t_end])
            cumulative_dict[title]['prey_move_distances_per_round'].append(prey_move_distances_per_round)
            cumulative_dict[title]['predator_move_distances_per_round'].append(predator_move_distances_per_round)

            # second, get the rotation, which is change in orientation
            orientations = np.array([info['ORIENTATION'] for info in results])
            total_rotations = np.sum(np.abs(orientations[1:] - orientations[:-1]), axis=0)
            cumulative_dict[title]['total_rotations'] = total_rotations
            prey_rotate_per_round = np.sum(np.abs(orientations[1][t_start:t_end] - orientations[1][t_start-1:t_end-1]))
            predator_rotate_per_round = np.sum(np.abs(orientations[0][t_start:t_end] - orientations[0][t_start-1:t_end-1]))
            cumulative_dict[title]['prey_rotate_per_round'].append(prey_rotate_per_round)
            cumulative_dict[title]['predator_rotate_per_round'].append(predator_rotate_per_round)

            ## Now, check the time on grass per round and the time off the grass per round
            time_off_grass_per_round = 0
            time_on_grass_per_round = 0
            if not np.isfinite(t_catch_i):
              t_catch_i = len(rewards)
            for ti in range(t_start, t_catch_i):
              if list(positions[ti, 1, :]) not in safe_grass:
                time_off_grass_per_round += 1
              else:
                time_on_grass_per_round += 1
            if time_on_grass_per_round < 0:
              raise ValueError('Time on grass is negative')
            cumulative_dict[title]['time_on_grass_per_round'].append(time_on_grass_per_round)
            cumulative_dict[title]['time_off_grass_per_round'].append(time_off_grass_per_round)
            cumulative_dict[title]['frac_off_grass_per_round'].append(time_off_grass_per_round / (time_on_grass_per_round + time_off_grass_per_round))

            # Now check for each time step in this round if the prey's position change cause a longer distance with the predator
            if t_leave_safe > 0 and t_catch_i < len(rewards):
              distance_to_predator = np.linalg.norm(positions[t_leave_safe-1:t_catch_i-1, 0] - positions[t_leave_safe:t_catch_i, 1], axis=1)
              t_moved = [i - t_leave_safe for i in range(t_leave_safe, t_catch_i) if (positions[i, 1] != positions[i-1, 1]).any()]
              distance_to_predator = distance_to_predator[t_moved]
              t_moving_away = np.sum([distance_to_predator[i] > distance_to_predator[i-1] for i in range(1, len(distance_to_predator))])
              frac_moving_away = t_moving_away / len(t_moved)
              cumulative_dict[title]['frac_moving_away_per_round'] = frac_moving_away
              manhattan_distance = np.sum(np.abs(positions[t_leave_safe:t_catch_i, 0] - positions[t_leave_safe:t_catch_i, 1]), axis=1)
              percent_time_in_3_steps = np.sum(manhattan_distance < 3) / len(manhattan_distance)
              cumulative_dict[title]['percent_time_in_3_steps'] = percent_time_in_3_steps
              percent_time_in_5_steps = np.sum(manhattan_distance < 5) / len(manhattan_distance)
              cumulative_dict[title]['percent_time_in_5_steps'] = percent_time_in_5_steps

          cumulative_dict[title]['mean_time_per_round'] = np.nanmean(cumulative_dict[title]['time_per_round'])
          cumulative_dict[title]['mean_apple_per_round'] = np.nanmean(cumulative_dict[title]['num_apple_collected_per_round'])
          cumulative_dict[title]['mean_acorn_per_round'] = np.nanmean(cumulative_dict[title]['num_acorn_collected_per_round'])
          cumulative_dict[title]['mean_prey_move_distances_per_round'] = np.nanmean(cumulative_dict[title]['prey_move_distances_per_round'])
          cumulative_dict[title]['mean_predator_move_distances_per_round'] = np.nanmean(cumulative_dict[title]['predator_move_distances_per_round'])
          cumulative_dict[title]['mean_prey_rotate_per_round'] = np.nanmean(cumulative_dict[title]['prey_rotate_per_round'])
          cumulative_dict[title]['mean_predator_rotate_per_round'] = np.nanmean(cumulative_dict[title]['predator_rotate_per_round'])

    with open(f'{video_path}serial_results_dict.pkl', 'wb') as f:
      pickle.dump(serial_data_dict, f)
    # TODO: uncomment this line to save the cumulative results
    # with open(f'{video_path}cumulative_results_dict.pkl', 'wb') as f:
    #   pickle.dump(cumulative_dict, f)
