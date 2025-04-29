import os

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
import sys

cwd = os.getcwd()
sys.path.append(cwd)
# os.environ[
#     "XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# Option 1: Tell JAX to pretend there is no GPU:
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Option 2: Explicitly choose the CPU platform:
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import functools

from absl import app
from absl import flags
from meltingpot.python import scenario

from marl import experiments
from marl.experiments import config as ma_config
from marl.utils import helpers
from marl.utils.experiment_utils import make_experiment_logger
import train

FLAGS = flags.FLAGS
flags.DEFINE_bool("run_eval_on_scenarios", False, "Whether to run evaluation on meltingpot scenarios.")
flags.DEFINE_bool("using_my_custom_environment_loop", True, "Whether to use my custom environment loop.")
flags.DEFINE_string("agent_param_indices", None, "Comma separated list of agent param indices.")
flags.DEFINE_integer("num_episodes", 5, "Number of episodes to run evaluation for.")
# experiment.plsc_dim_to_perturb = 10
# experiment.agent_to_perturb = 'predator, prey'
# experiment.plsc_decomposition_dict_path = '/home/mikan/e/Documents/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__open_2024-11-26_17_36_18.023323_ckp7357/pickles/PLSC_usv_dict.pkl'
flags.DEFINE_integer("plsc_dim_to_perturb", None, "Number of dimensions to perturb.")
flags.DEFINE_string("agent_to_perturb", None, "Agent to perturb.")
flags.DEFINE_string("plsc_decomposition_dict_path", None, "Path to PLSC decomposition dict.")
# flags.DEFINE_bool("run_eval", True, "Whether to run evaluation.")
def main(_):
  if FLAGS.experiment_dir is None:
    raise ValueError("experiment_dir must be specified")

  if FLAGS.agent_param_indices is not None:
    FLAGS.agent_param_indices = [int(idx) for idx in FLAGS.agent_param_indices.split(",")]

  config, experiment_dir = train.build_experiment_config()

  ckpt_config = ma_config.CheckpointingConfig(
    max_to_keep=3, directory=experiment_dir, add_uid=False)

  log_dir_append = ""
  if FLAGS.agent_param_indices is not None:
    log_dir_append = FLAGS.get_flag_value("map_name", "") + "_agent_" + "_".join(map(str, FLAGS.agent_param_indices))
  if (FLAGS.agent_to_perturb is not None) and ('PLSC_usv_dict' in FLAGS.plsc_decomposition_dict_path):
    if ('predator' in FLAGS.agent_to_perturb) and ('prey' in FLAGS.agent_to_perturb):
      log_dir_append += "_perturb_both"
    elif 'predator' in FLAGS.agent_to_perturb:
      log_dir_append += "_perturb_predator"
    elif 'prey' in FLAGS.agent_to_perturb:
      log_dir_append += "_perturb_prey"
  elif (FLAGS.agent_to_perturb is not None) and ('randPC' in FLAGS.plsc_decomposition_dict_path):
    tag = str(FLAGS.plsc_dim_to_perturb) + FLAGS.plsc_decomposition_dict_path.split('/')[-1].split('.pkl')[0]
    if ('predator' in FLAGS.agent_to_perturb) and ('prey' in FLAGS.agent_to_perturb):
      log_dir_append += f"_perturb_both_{tag}"
    elif 'predator' in FLAGS.agent_to_perturb:
      log_dir_append += f"_perturb_predator_{tag}"
    elif 'prey' in FLAGS.agent_to_perturb:
      log_dir_append += f"_perturb_prey_{tag}"

  config.logger_factory = functools.partial(
    make_experiment_logger, log_dir=os.path.join(experiment_dir, "logs", log_dir_append), use_tb=False)
  config.agent_param_indices = FLAGS.agent_param_indices
  config.using_my_custom_environment_loop = FLAGS.using_my_custom_environment_loop

  if FLAGS.plsc_decomposition_dict_path is not None:
    config.plsc_decomposition_dict_path = FLAGS.plsc_decomposition_dict_path
    config.plsc_dim_to_perturb = FLAGS.plsc_dim_to_perturb
    config.agent_to_perturb = FLAGS.agent_to_perturb

  if FLAGS.env_name == "meltingpot":

    if FLAGS.run_eval_on_scenarios:
      scenarios_for_substrate = sorted(list(scenario.SCENARIOS_BY_SUBSTRATE[FLAGS.map_name]))

      print(f"Running evaluation on scenarios: {scenarios_for_substrate}")

      for scenario_name in scenarios_for_substrate:
        env_factory = functools.partial(
          helpers.make_meltingpot_scenario, scenario_name=scenario_name, record=True)
        env_factory(0)
        config.environment_factory = env_factory
        experiments.run_evaluation(
          config, ckpt_config, environment_name=scenario_name, num_eval_episodes=FLAGS.num_episodes, )
    else:
      # running evaluation on substrate
      experiments.run_evaluation(
        config, ckpt_config, environment_name=FLAGS.map_name, num_eval_episodes=FLAGS.num_episodes, )

  else:
    # running evaluation on substrate
    experiments.run_evaluation(
      config,
      ckpt_config,
      environment_name=f"{FLAGS.env_name}_{FLAGS.map_name}",
      num_eval_episodes=FLAGS.num_episodes)


if __name__ == "__main__":
  app.run(main)
