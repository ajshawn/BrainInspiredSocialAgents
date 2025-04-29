import os
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def compute_stuck_rate(pos, min_duration=20):
    T = pos.shape[0]
    stuck_indicator = np.zeros(T, dtype=int)
    i = 0
    while i < T - 1:
        if np.allclose(pos[i], pos[i + 1]):
            j = i
            while j < T - 1 and np.allclose(pos[j], pos[j + 1]):
                j += 1
            if (j - i + 1) >= min_duration:
                stuck_indicator[i:j + 1] = 1
            i = j + 1
        else:
            i += 1
    return np.mean(stuck_indicator), stuck_indicator

def process_pair_folder(folder_path, predator_id, prey_id):
    """
    Process one predator-prey pair within a given folder.
    """
    title = f'{predator_id}_{prey_id}'
    print(f"Processing {title} in folder: {folder_path}")

    episode_data_dict = {title: {key: [] for key in [
        'num_rounds', 'mean_time_per_round',
        'prey_move_distances_per_episode', 'predator_move_distances_per_episode',
        'num_apple_collected_per_episode', 'num_acorn_collected_per_episode',
        'prey_reward_per_episode', 'predator_reward_per_episode',
        'prey_rotate_per_episode', 'predator_rotate_per_episode',
        'time_on_grass_per_episode', 'time_off_grass_per_episode',
        'time_on_grass_nonstuck_per_episode', 'time_off_grass_nonstuck_per_episode',
        'frac_off_grass_per_episode', 'frac_off_grass_nonstuck_per_episode',
        'frac_moving_away_per_episode', 'frac_time_within_3_steps', 'frac_time_within_5_steps',
        'pred_stuck_rate', 'prey_stuck_rate',
        'mean_predator_stamina', 'mean_prey_stamina',
        'mean_predator_stamina_nonstuck', 'mean_prey_stamina_nonstuck'
    ]}}

    episode_results = {}
    for eId in range(1, 101):
        file_path = os.path.join(folder_path, f'{title}_{eId}.pkl')
        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
            episode_results[eId] = results
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

    if not episode_results:
        print(f"No pickle files found for {title} in {folder_path}")
        return None

    for eId, results in episode_results.items():
        positions = np.array([info['POSITION'] for info in results])
        rewards_arr = np.array([info['rewards'] for info in results])
        orientations_arr = np.array([info['ORIENTATION'] for info in results])
        stamina_arr = np.array([info['STAMINA'] for info in results])

        num_agents = 2
        distances = [np.linalg.norm(np.diff(positions[:, i, :], axis=0), axis=1) for i in range(num_agents)]
        distances = [np.concatenate(([0], dist)) for dist in distances]

        safe_grass = [[i, j] for i in [8, 9, 10] for j in [4, 5, 6]]
        t_catch = np.where(rewards_arr[:, 0] == 1)[0]
        t_respawn = np.insert(t_catch + 21, 0, 0)

        round_time_list = []
        prey_move_list = []
        predator_move_list = []
        apple_count_list = []
        acorn_count_list = []
        prey_rotate_list = []
        predator_rotate_list = []
        time_on_list = []
        time_off_list = []
        time_on_nonstuck_list = []
        time_off_nonstuck_list = []
        time_moving_away_list = []
        time_moved_list = []
        time_within_3_steps_list = []
        time_within_5_steps_list = []
        pred_stuck_list = []
        prey_stuck_list = []
        prey_rewards_list = []
        pred_rewards_list = []
        pred_stamina_list = []
        prey_stamina_list = []
        pred_stamina_nonstuck_list = []
        prey_stamina_nonstuck_list = []

        for round_idx, t_start in enumerate(t_respawn[:-1]):
            t_end = min(t_respawn[round_idx + 1], len(rewards_arr))
            t_leave_safe = next((ti for ti in range(t_start, t_end) if list(positions[ti, 1]) not in safe_grass), None)
            if t_leave_safe is None:
                continue
            t_catch_i = next((ti for ti in t_catch if ti >= t_leave_safe), None)
            if t_catch_i is None:
                continue
            time_per_round = t_catch_i - t_leave_safe
            round_time_list.append(time_per_round)
            #rest of the code is the same.
            apple_count_list.append(np.sum((rewards_arr[t_leave_safe:t_end, 1] == 1)))
            acorn_count_list.append(np.sum((rewards_arr[t_leave_safe:t_end, 1] > 1)))
            prey_move_list.append(np.sum(distances[1][t_start:t_end] == 1))
            predator_move_list.append(np.sum(distances[0][t_start:t_end] == 1))
            if t_start > 0:
                prey_rotate_list.append(np.sum(np.abs(np.diff(orientations_arr[t_start:t_end, 1]))))
                predator_rotate_list.append(np.sum(np.abs(np.diff(orientations_arr[t_start:t_end, 0]))))

            pred_stuck, pred_stuck_time_points = compute_stuck_rate(positions[t_start:t_catch_i, 0], min_duration=20)
            prey_stuck, prey_stuck_time_points = compute_stuck_rate(positions[t_start:t_catch_i, 1], min_duration=20)
            pred_stuck_list.append(pred_stuck)
            prey_stuck_list.append(prey_stuck)
            prey_rewards_list.append(np.sum(rewards_arr[t_start:t_catch_i, 1]))
            pred_rewards_list.append(np.sum(rewards_arr[t_start:t_catch_i, 0]))
            pred_stamina_list.extend(stamina_arr[t_start:t_catch_i, 0])
            prey_stamina_list.extend(stamina_arr[t_start:t_catch_i, 1])
            pred_stamina_nonstuck_list.extend(stamina_arr[t_start:t_catch_i, 0][pred_stuck_time_points == 0])
            prey_stamina_nonstuck_list.extend(stamina_arr[t_start:t_catch_i, 1][prey_stuck_time_points == 0])

            time_on = np.sum([1 for ti in range(t_start, t_catch_i) if list(positions[ti, 1]) in safe_grass])
            time_off = time_per_round - time_on
            time_on_nonstuck = np.sum([1 for ti in range(t_start, t_catch_i) if (list(positions[ti, 1]) in safe_grass) and (0 == prey_stuck_time_points[ti - t_start])])
            time_off_nonstuck = np.sum([1 for ti in range(t_start, t_catch_i) if (list(positions[ti, 1]) not in safe_grass) and (0 == pred_stuck_time_points[ti - t_start])])

            time_on_list.append(time_on)
            time_off_list.append(time_off)
            time_on_nonstuck_list.append(time_on_nonstuck)
            time_off_nonstuck_list.append(time_off_nonstuck)

            if t_leave_safe >= 0 and t_catch_i <= t_end:
                if t_leave_safe == 0:
                    t_leave_safe = 1
                distance_to_predator = np.linalg.norm(positions[t_leave_safe - 1:t_catch_i - 1, 0] - positions[t_leave_safe:t_catch_i, 1], axis=1)
                moved_indices = [i - t_leave_safe for i in range(t_leave_safe, t_catch_i) if (positions[i, 1] != positions[i - 1, 1]).any()]
                distance_to_predator_moved = distance_to_predator[[i - t_leave_safe for i in range(t_leave_safe, t_catch_i) if (positions[i, 1] != positions[i - 1, 1]).any()]]
                time_moving_away_list.append(np.sum([1 for i in range(1, len(distance_to_predator_moved)) if distance_to_predator_moved[i] > distance_to_predator_moved[i - 1]]))
                time_moved_list.append(len(moved_indices))
                manhattan_distance = np.sum(np.abs(positions[t_leave_safe:t_catch_i, 0] - positions[t_leave_safe:t_catch_i, 1]), axis=1)
                time_within_3_steps_list.append(np.sum(manhattan_distance < 3))
                time_within_5_steps_list.append(np.sum(manhattan_distance < 5))
            else:
                time_moving_away_list.append(np.nan)
                time_moved_list.append(np.nan)
                time_within_3_steps_list.append(np.nan)
                time_within_5_steps_list.append(np.nan)
        episode_data_dict[title]['num_rounds'].append(len(round_time_list))
        episode_data_dict[title]['mean_time_per_round'].append(np.nanmean(round_time_list) if round_time_list else np.nan)
        #rest of the code is the same.
        episode_data_dict[title]['prey_move_distances_per_episode'].append(np.nansum(prey_move_list))
        episode_data_dict[title]['predator_move_distances_per_episode'].append(np.nansum(predator_move_list))
        episode_data_dict[title]['num_apple_collected_per_episode'].append(np.nansum(apple_count_list))
        episode_data_dict[title]['num_acorn_collected_per_episode'].append(np.nansum(acorn_count_list))
        episode_data_dict[title]['prey_rotate_per_episode'].append(np.nansum(prey_rotate_list))
        episode_data_dict[title]['predator_rotate_per_episode'].append(np.nansum(predator_rotate_list))
        episode_data_dict[title]['time_on_grass_per_episode'].append(np.nansum(time_on_list))
        episode_data_dict[title]['time_off_grass_per_episode'].append(np.sum(time_off_list))
        episode_data_dict[title]['time_on_grass_nonstuck_per_episode'].append(np.nansum(time_on_nonstuck_list))
        episode_data_dict[title]['time_off_grass_nonstuck_per_episode'].append(np.nansum(time_off_nonstuck_list))
        episode_data_dict[title]['frac_off_grass_per_episode'].append(np.sum(time_off_list) / (np.nansum(time_on_list) + np.nansum(time_off_list)) if (np.nansum(time_on_list) + np.nansum(time_off_list)) > 0 else np.nan)
        episode_data_dict[title]['frac_off_grass_nonstuck_per_episode'].append(np.nansum(time_off_nonstuck_list) / (np.nansum(time_on_nonstuck_list) + np.nansum(time_off_nonstuck_list)) if (np.sum(time_on_nonstuck_list) + np.sum(time_off_nonstuck_list)) > 0 else np.nan)
        episode_data_dict[title]['frac_moving_away_per_episode'].append(np.nansum(time_moving_away_list) / np.nansum(time_moved_list) if time_moved_list else np.nan)
        episode_data_dict[title]['frac_time_within_3_steps'].append(np.nansum(time_within_3_steps_list) / np.nansum(round_time_list) if round_time_list else np.nan)
        episode_data_dict[title]['frac_time_within_5_steps'].append(np.nansum(time_within_5_steps_list) / np.nansum(round_time_list) if round_time_list else np.nan)
        episode_data_dict[title]['pred_stuck_rate'].append(np.nanmean(pred_stuck_list) if pred_stuck_list else np.nan)
        episode_data_dict[title]['prey_stuck_rate'].append(np.nanmean(prey_stuck_list) if prey_stuck_list else np.nan)
        episode_data_dict[title]['prey_reward_per_episode'].append(np.nansum(prey_rewards_list))
        episode_data_dict[title]['predator_reward_per_episode'].append(np.nansum(pred_rewards_list))
        episode_data_dict[title]['mean_predator_stamina'].append(np.nanmean(pred_stamina_list))
        episode_data_dict[title]['mean_prey_stamina'].append(np.nanmean(prey_stamina_list))
        episode_data_dict[title]['mean_predator_stamina_nonstuck'].append(np.nanmean(pred_stamina_nonstuck_list))
        episode_data_dict[title]['mean_prey_stamina_nonstuck'].append(np.nanmean(prey_stamina_nonstuck_list))

    return episode_data_dict

def process_and_save(folder_path, predator_id, prey_id, out_dir):
    result = process_pair_folder(folder_path, predator_id, prey_id)
    if result is None:
        return
    pair_key = list(result.keys())[0]
    episode_out = os.path.join(out_dir, f"{pair_key}_episode_results.csv")
    episode_df = pd.DataFrame().from_dict(result[pair_key], orient='index').T
    episode_df.to_csv(episode_out, index=False)
    print(f"Saved analysis for {pair_key} in {folder_path}")

def main():
    video_paths = [
        f'/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles/',
        f'/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651/pickles/',
        f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962/pickles/',
        f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/pickles/',
        f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026/pickles/',
        f'/home/mikan/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441/pickles/',
    ]
    out_dir = os.path.join(os.path.dirname(video_paths[0]), "analysis_results")
    os.makedirs(out_dir, exist_ok=True)

    for video_path in video_paths:
        if 'open' in video_path:
            predator_ids = [0, 1, 2]
            prey_ids = list(range(3, 13))
        else:
            predator_ids = [0, 1, 2, 3, 4]
            prey_ids = list(range(5, 13))
        tasks = []
        for predator_id in predator_ids:
            for prey_id in prey_ids:
                tasks.append(delayed(process_and_save)(video_path, predator_id, prey_id, out_dir))
        Parallel(n_jobs=50)(tasks)
    print("All analysis files have been generated.")

if __name__ == '__main__':
    main()