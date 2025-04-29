import os
import shutil

# List of training arena folder names and corresponding mix abbreviations.
training_arena_name_list = [
  'PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-01-07_12:11:32.926962',
  'PopArtIMPALA_1_meltingpot_predator_prey__alley_hunt_2025-02-10_21:44:27.355026',
  'PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357',
  'PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp9651',
  'PopArtIMPALA_1_meltingpot_predator_prey__open_2025-02-24_16:09:49.096441',
  'PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-01-11_11:10:52.115495',
  'PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-02-10_21:45:28.296092',
  'PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274',
]

mix_arena_abbr_list = [
  'AH20250107',
  'AH20250210',
  'OP20241126ckp7357',
  'OP20241126ckp9651',
  'OP20250224',
  'OR20250111',
  'OR20250210',
  'OR20250305',
]


def get_dim_from_abbr(abbr):
  """
  Given an arena abbreviation string (e.g. "AH20250107"),
  extract the date (assumed to be the 8 digits following the two-letter prefix)
  and return "128" if the date is before 20250201 and "256" otherwise.
  For abbreviations that include extra tokens (like "ckp7357"), we ignore those
  for the date comparison.
  """
  try:
    date_str = abbr[2:10]  # e.g. "20250107"
    if int(date_str) < 20250201:
      return "128"
    else:
      return "256"
  except Exception:
    return "NA"


# Base directory where your results are stored.
results_dir = "/home/mikan/e/Documents/GitHub/social-agents-JAX/results"

# Loop over each training folder and its corresponding mix arena abbreviation.
for training_arena, mix_abbr in zip(training_arena_name_list, mix_arena_abbr_list):
  dim = get_dim_from_abbr(mix_abbr)
  training_folder = os.path.join(results_dir, training_arena)
  pickles_folder = os.path.join(training_folder, "pickles")

  # Ensure the source pickles folder exists; if not, skip this arena.
  if not os.path.isdir(pickles_folder):
    print(f"Pickles folder does not exist: {pickles_folder}")
    continue

  # Iterate over predator indices, prey indices and episodes.
  for pred in range(5):  # Predator indices 0,1,2,3,4
    for prey in range(3, 13):  # Prey indices 3,4,...,12
      # Use a flag to check if any file was found for this pred-prey combination.
      copied_any_file = False
      for episode in range(1, 101):  # Episodes 1 through 100
        src_file = os.path.join(pickles_folder, f"{pred}_{prey}_{episode}.pkl")
        if os.path.exists(src_file):
          # Construct the destination folder for this (pred, prey) pair.
          dest_folder = os.path.join(
            results_dir,
            "mix",
            f"{mix_abbr}_agent{pred}_dim{dim}_vs_{mix_abbr}_agent{prey}_dim{dim}_episode_pickles"
          )
          # Create destination folder if not already created.
          if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
          # Destination file name format.
          dest_file = os.path.join(
            dest_folder,
            f"predator_prey__simplified10x10_OneVsOne_episode_{episode}.pkl"
          )
          # Copy the file.
          shutil.copy(src_file, dest_file)
          print(f"Copied: {src_file} -> {dest_file}")
          copied_any_file = True

      # If no source file for this pred/prey combination is found,
      # no dest folder will be created (or it remains empty).
      if not copied_any_file:
        print(f"No files found for {training_arena}, predator {pred} vs prey {prey}.")
