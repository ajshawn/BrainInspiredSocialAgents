import os
import re
import itertools
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parse command-line arguments for parallelism
parser = argparse.ArgumentParser(description="Run predator-prey evaluations with limited parallel jobs.")
parser.add_argument("--n_jobs", type=int, default=40,
                    help="Maximum number of concurrent evaluation jobs.")
args = parser.parse_args()
N_JOBS = args.n_jobs

# Function to extract date (YYYY-MM-DD) from a checkpoint path.
def extract_date(ckpt_path):
    base = os.path.basename(os.path.normpath(ckpt_path))
    match = re.search(r"(\d{4}-\d{2}-\d{2})", base)
    if not match:
        raise ValueError(f"No date found in checkpoint path: {ckpt_path}")
    return match.group(1)

# Function to determine recurrent dimension from a date.
def get_recurrent_dim(ckpt_path):
    year, month, _ = map(int, extract_date(ckpt_path).split('-'))
    return 128 if (year < 2025 or (year == 2025 and month < 2)) else 256

# Build combinations list from checkpoint dictionary
def build_combinations(pool_dict, role_name):
    combos = []
    for path, indices in pool_dict.items():
        rec_dim = get_recurrent_dim(path)
        for idx in indices:
            combos.append((path, role_name, idx, rec_dim))
    return combos

# Greedy diverse sampler: picks combos maximizing new coverage
def sample_diverse(all_combos, limit):
    remaining = all_combos.copy()
    combo_sets = [set((c[0], c[2]) for c in combo) for combo in remaining]
    selected = []
    used = set()
    while len(selected) < limit and remaining:
        best_i = max(range(len(remaining)),
                     key=lambda i: len(combo_sets[i] - used))
        selected.append(remaining.pop(best_i))
        used |= combo_sets.pop(best_i)
    return selected

# Build shell commands and envs for a given list of combos
def build_commands_for_combos(combo_list, map_name='predator_prey__simplified10x10_OneVsOne', map_layout=None):
    commands = []
    for combos in combo_list:
        paths   = [c[0] for c in combos]
        roles   = [c[1] for c in combos]
        indices = [str(c[2]) for c in combos]
        dims    = [str(c[3]) for c in combos]

        # prepare environment
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":/home/mikan/miniconda3/envs/kilosort/lib/python3.9/site-packages/nvidia/cudnn/lib/"
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["JAX_PLATFORM_NAME"]    = "cpu"

        # build command string
        cmd_str = (
            "python evaluate_cross_trial.py --async_distributed --available_gpus -1 "
            f"--num_actors 16 --algo_name PopArtIMPALA --env_name meltingpot "
            f"--map_name {map_name} --record_video true "
            f"--cross_checkpoint_paths {','.join(paths)} "
            f"--agent_roles {','.join(roles)} --agent_param_indices {','.join(indices)} "
            f"--recurrent_dims {','.join(dims)} --num_episodes 50"
        )
        if map_layout:
            cmd_str += f" --map_layout {map_layout}"
        commands.append((cmd_str, env))
    return commands

# Execute a list of shell commands with a limit on parallel jobs
def run_commands(commands, max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for cmd_str, env in commands:
            futures.append(executor.submit(subprocess.run, cmd_str, shell=True, env=env))
        for future in as_completed(futures):
            try:
                future.result()
                print(f"Completed: {future}")
            except Exception as e:
                print(f"Error: {e}")

# Main execution
def main(scenarios, predator_combinations, prey_combinations):
    for name, (p, q) in scenarios.items():
        print(f"\n=== Scenario: {name} ({p} predators vs {q} preys) ===")
        # generate all possible combos
        all_combos = [list(pred) + list(prey)
                      for pred in itertools.combinations(predator_combinations, p)
                      for prey in itertools.combinations(prey_combinations, q)]
        # sample up to 10 most diverse
        sampled = sample_diverse(all_combos, 20)
        # commands = build_commands_for_combos(sampled, map_name='predator_prey__open_debug', map_layout='smaller_13x13')
        commands = build_commands_for_combos(sampled, map_name='predator_prey__simplified10x10_OneVsOne', map_layout=None)
        run_commands(commands, N_JOBS)

if __name__ == '__main__':
    # Scenarios: number of predators vs number of preys
    scenarios = {
        '1vs1': (1, 1),
        '1vs2': (1, 2),
        # '2vs3': (2, 3),
        # '2vs4': (2, 4)
    }
    #
    # # first_combo
    # # Define predator and prey checkpoint pools with explicit indices
    # predator_checkpoints = {
    #     "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306/": [0, 1, 2],
    #     "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/": [0],
    #     "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274/": [0]
    # }
    # prey_checkpoints = {
    #     # Acorn collector
    #     "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/": [5, 10],
    #     "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651": [5, 6, 7],
    #     "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306/": [6]
    # }
    #
    # # Build predator and prey combinations
    # predator_combinations = build_combinations(predator_checkpoints, 'predator')
    # prey_combinations = build_combinations(prey_checkpoints, 'prey')
    # main(scenarios, predator_combinations, prey_combinations)
    #
    # # second_combo
    # # Define predator and prey checkpoint pools with explicit indices
    # predator_checkpoints = {
    #     "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306/": [0, 1, 2],
    #     "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/": [0],
    #     "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274/": [0]
    # }
    # prey_checkpoints = {
    #     # Apple collector
    #     # "./results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026": [5,6,7,9],
    #     "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092": [6, 8],
    #     "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306/": [4, 10],
    # }
    #
    # # Build predator and prey combinations
    # predator_combinations = build_combinations(predator_checkpoints, 'predator')
    # prey_combinations = build_combinations(prey_checkpoints, 'prey')
    # main(scenarios, predator_combinations, prey_combinations)
    #
    # # third_combo
    # # Define predator and prey checkpoint pools with explicit indices
    # predator_checkpoints = {
    #     "./results/PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441_ckp6306/": [0, 1, 2],
    #     "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/": [0],
    #     "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274/": [0]
    # }
    # prey_checkpoints = {
    #     # Apple collector
    #     "./results/PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026": [5,6,7,9],
    # }
    #
    # # Build predator and prey combinations
    # predator_combinations = build_combinations(predator_checkpoints, 'predator')
    # prey_combinations = build_combinations(prey_checkpoints, 'prey')
    # main(scenarios, predator_combinations, prey_combinations)

    # fourth_combo
    # Define predator and prey checkpoint pools with explicit indices
    predator_checkpoints = {
        "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274/": [0,2,3]
    }
    prey_checkpoints = {
        # Apple collector
            "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092/": [0],
            "./results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274/": [0]
    }

    # Build predator and prey combinations
    predator_combinations = build_combinations(predator_checkpoints, 'predator')
    prey_combinations = build_combinations(prey_checkpoints, 'prey')
    main(scenarios, predator_combinations, prey_combinations)