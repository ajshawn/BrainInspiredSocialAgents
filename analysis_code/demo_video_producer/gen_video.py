import subprocess
from pathlib import Path


def run_overlay_batch(video_list, data_list, event_data_list, output_dir, overlay_script, episode=0):
  """
  Runs the overlay script for each triplet of (video, data, event_data).

  Args:
      video_list: List of paths to video files.
      data_list: List of paths to data pickle files (matching videos).
      event_data_list: List of paths to event pickle files (one per session).
      output_dir: Directory to save outputs.
      overlay_script: Path to the Python script that performs overlay.
      episode: Episode index to extract from event file (default 0).
  """
  assert len(video_list) == len(data_list) == len(event_data_list), "All lists must be of same length"
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  for i, (vpath, dpath, epath) in enumerate(zip(video_list, data_list, event_data_list)):
    output_path = output_dir / f"overlay_{i}.mp4"

    cmd = [
      "python", overlay_script,
      "--video_path", str(vpath),
      "--data_path", str(dpath),
      "--event_path", str(epath),
      "--output_path", str(output_path)
    ]

    print(f"Running overlay for video {i + 1}/{len(video_list)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
      print(f"[ERROR] Overlay failed for index {i}:\n{result.stderr}")
    else:
      print(f"[DONE] Saved to {output_path}")


video_list = [
  '/home/mikan/e/GitHub/social-agents-JAX/recordings/meltingpot/mix_OR20250210_pred_1_OR20250210_prey_10_OR20250210_prey_8_predator_prey__open_debug_smaller_13x13/1.mp4',
  '/home/mikan/e/GitHub/social-agents-JAX/recordings/meltingpot/mix_OP20250224ckp6306_pred_0_OP20250224ckp6306_pred_1_OR20250210_prey_6_OR20250210_prey_8_OP20250224ckp6306_prey_4_OP20250224ckp6306_prey_10_predator_prey__open_debug_smaller_13x13/16.mp4',
  '/home/mikan/e/GitHub/social-agents-JAX/recordings/meltingpot/mix_OR20250210_pred_0_OP20241126ckp9651_prey_7_OP20250224ckp6306_prey_6_predator_prey__open_debug_smaller_13x13/6.mp4'
]

data_list = [
  '/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/mix_OR20250210_pred_1_OR20250210_prey_10_OR20250210_prey_8predator_prey__open_debug_agent_1_10_8/episode_pickles/predator_prey__open_debug_episode_1.pkl',
  '/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/mix_OP20250224ckp6306_pred_0_OP20250224ckp6306_pred_1_OR20250210_prey_6_OR20250210_prey_8_OP20250224ckp6306_prey_4_OP20250224ckp6306_prey_10predator_prey__open_debug_agent_0_1_6_8_4_10/episode_pickles/predator_prey__open_debug_episode_16.pkl',
  '/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/mix_OR20250210_pred_0_OP20241126ckp9651_prey_7_OP20250224ckp6306_prey_6predator_prey__open_debug_agent_0_7_6/episode_pickles/predator_prey__open_debug_episode_6.pkl'
]

event_data_list = [
  '/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/analysis_results_merged/mix_OR20250210_pred_1_OR20250210_prey_10_OR20250210_prey_8_merged.pkl',
  '/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/analysis_results_merged/mix_OP20250224ckp6306_pred_0_OP20250224ckp6306_pred_1_OR20250210_prey_6_OR20250210_prey_8_OP20250224ckp6306_prey_4_OP20250224ckp6306_prey_10_merged.pkl',
  '/home/mikan/e/GitHub/social-agents-JAX/results/mix_2_4/analysis_results_merged/mix_OR20250210_pred_0_OP20241126ckp9651_prey_7_OP20250224ckp6306_prey_6_merged.pkl'
]

run_overlay_batch(
  video_list,
  data_list,
  event_data_list,
  output_dir='/home/mikan/e/GitHub/social-agents-JAX/recordings/meltingpot/overlay_videos',
  overlay_script='/home/mikan/e/GitHub/social-agents-JAX/analysis_code/demo_video_producer/demo_video_producer_neural_grid_2v4.py',
)
