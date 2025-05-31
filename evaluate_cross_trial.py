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
import jax
import re
from absl import app
from absl import flags
from meltingpot.python import scenario

from marl import experiments
from marl import specs as ma_specs
from marl.agents import impala, opre
from marl.agents.networks import ArrayFE, ImageFE, MeltingpotFE
from marl.experiments import config as ma_config
from marl.experiments import inference_server
from marl.utils import helpers
from marl.utils import lp_utils as ma_lp_utils
from marl.utils.experiment_utils import make_experiment_logger
FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "async_distributed",
    False,
    "Should an agent be executed in an off-policy distributed way",
)
flags.DEFINE_bool("run_eval", False, "Whether to run evaluation.")
flags.DEFINE_bool(
    "all_parallel", False,
    "Flag to run all agents in parallel using vmap. Only use if GPU with large memory is available."
)
flags.DEFINE_enum(
    "env_name",
    "overcooked",
    ["meltingpot", "overcooked"],
    "Environment to train on",
)
flags.DEFINE_string(
    "map_name", "cramped_room",
    "Meltingpot/Overcooked Map to train on. Only used when 'env_name' is 'meltingpot' or 'overcooked'"
)
flags.DEFINE_string("frozen_agents", None,
                    "Comma separated list of frozen agents.")
flags.DEFINE_enum("algo_name", "IMPALA",
                  ["IMPALA", "PopArtIMPALA", "OPRE", "PopArtOPRE"],
                  "Algorithm to train")
flags.DEFINE_bool("record_video", False,
                  "Whether to record videos. (Only use during evaluation)")
flags.DEFINE_integer("reward_scale", 1, "Reward scale factor.")
flags.DEFINE_bool("prosocial", False,
                  "Whether to use shared reward for prosocial training.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_steps", 200_000_000, "Number of env steps to run.")
flags.DEFINE_string("exp_log_dir", "./results/mix_2_4/",
                    "Directory to store experiment logs in.")
flags.DEFINE_bool("use_tb", True, "Flag to enable tensorboard logging.")
flags.DEFINE_bool("use_wandb", True, "Flag to enable wandb.ai logging.")
flags.DEFINE_string("wandb_entity", "ajshawn723", "Entity name for wandb account.")
flags.DEFINE_string("wandb_project", "marl-jax",
                    "Project name for wandb logging.")
flags.DEFINE_string("wandb_tags", "", "Comma separated list of tags for wandb.")
flags.DEFINE_string("available_gpus", "0", "Comma separated list of GPU ids.")
flags.DEFINE_integer(
    "num_actors", 2,
    "Number of actors to use (should be less than total number of CPU cores).")
flags.DEFINE_integer("actors_per_node", 1, "Number of actors per thread.")
flags.DEFINE_bool("inference_server", False, "Whether to run inference server.")
flags.DEFINE_string("agent_roles", None,
                    "Comma separated list of agent roles.")

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
flags.DEFINE_string("cross_checkpoint_paths", None, "Comma separated list of checkpoint paths.")
flags.DEFINE_string("recurrent_dims", None, "Comma separated list of recurrent dims.")
flags.DEFINE_string("map_layout", None, "Custom map layout for meltingpot maps")


def abbreviate_ckpt(ckpt):
  # Remove a known prefix if present.
  prefix = "PopArtIMPALA_1_meltingpot_predator_prey__"
  if ckpt.startswith(prefix):
    ckpt = ckpt[len(prefix):]
  # Decide abbreviation based on keyword.
  if "alley_hunt" in ckpt:
    abbr = "AH"
  elif "open" in ckpt:
    abbr = "OP"
  elif "orchard" in ckpt:
    abbr = "OR"
  elif "random_forest" in ckpt:
    abbr = "RF"
  else:
    abbr = "XX"
  # Extract the first date (YYYY-MM-DD) and remove dashes.
  m = re.search(r'(\d{4}-\d{2}-\d{2})', ckpt)
  if m:
    date_str = m.group(1).replace("-", "")
  else:
    date_str = "00000000"
  match_ckp = re.search(r'ckp\d+', ckpt)
  # if not found, try ckpt
  if not match_ckp:
    match_ckp = re.search(r'ckpt\d+', ckpt)
  if match_ckp:
    return f"{abbr}{date_str}{match_ckp.group(0)}"
  return f"{abbr}{date_str}"

def _get_custom_env_configs():
  result = {}
  if FLAGS.env_name == "meltingpot" and FLAGS.map_name == "coop_mining":
    if FLAGS.conservative_mine_beam:
      result["conservative_mine_beam"] = True
    if FLAGS.dense_ore_regrow:
      result["dense_ore_regrow"] = True
  if FLAGS.map_layout:
    result[FLAGS.map_layout] = True
  return result

def build_experiment_config(experiment_dir, num_agents=None, recurrent_dim=None, video_path=None):
  """Builds experiment config which can be executed in different ways."""
  # Create an environment, grab the spec, and use it to create networks.

  # creating the following values so that FLAGS doesn't need to be pickled
  map_name = FLAGS.map_name
  reward_scale = FLAGS.reward_scale
  autoreset = False
  prosocial = FLAGS.prosocial
  record = FLAGS.record_video
  memory_efficient = not FLAGS.all_parallel
  frozen_agents = set([int(agent) for agent in FLAGS.frozen_agents.split(",")] if FLAGS.frozen_agents else [])
  agent_roles = [role.strip() for role in FLAGS.agent_roles.split(",")] if FLAGS.agent_roles else None

  if experiment_dir:
    assert FLAGS.algo_name in experiment_dir, f"experiment_dir must be a {FLAGS.algo_name} experiment"
    assert FLAGS.env_name in experiment_dir, f"experiment_dir must be a {FLAGS.env_name} experiment"
    experiment_name = experiment_dir.split("/")[-1]
  else:
    ## TODO: Change the checkpoint dir
    raise ValueError("experiment_dir must be specified")

  wandb_config = {
    "project": FLAGS.wandb_project,
    "entity": FLAGS.wandb_entity,
    "name": experiment_name,
    "group": experiment_name,
    "resume": True,
    "tags": [st for st in FLAGS.wandb_tags.split(",") if st],
  }

  feature_extractor = ArrayFE

  # Create environment factory
  if FLAGS.env_name == "meltingpot":
    custom_env_configs = _get_custom_env_configs()
    env_factory = lambda seed: helpers.env_factory(
      seed,
      map_name,
      autoreset=autoreset,
      shared_reward=prosocial,
      reward_scale=reward_scale,
      shared_obs=False,
      record=record,
      agent_roles=agent_roles,
      video_path=video_path,
      **custom_env_configs,)
    feature_extractor = MeltingpotFE
    num_options = 16
  else:
    raise ValueError(f"Unknown env_name {FLAGS.env_name}")

  environment_specs = ma_specs.MAEnvironmentSpec(env_factory(0))

  if FLAGS.algo_name == "PopArtIMPALA":
    # Create network
    network_factory = functools.partial(
      impala.make_network_2,
      feature_extractor=feature_extractor,
      recurrent_dim=recurrent_dim
    )
    network = network_factory(
      environment_specs.get_single_agent_environment_specs())
    # Construct the agent.
    config = impala.IMPALAConfig(
      n_agents=environment_specs.num_agents if num_agents is None else num_agents,
      memory_efficient=memory_efficient)
    core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
    builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)

  else:
    raise ValueError(f"Unknown algo_name {FLAGS.algo_name}")

  # Add frozen agents
  builder._config.frozen_agents = frozen_agents

  return (
    experiments.MAExperimentConfig(
      builder=builder,
      environment_factory=env_factory,
      network_factory=network_factory,
      logger_factory=functools.partial(
        make_experiment_logger,
        log_dir=experiment_dir,
        use_tb=FLAGS.use_tb,
        use_wandb=FLAGS.use_wandb,
        wandb_config=wandb_config,
      ),
      environment_spec=environment_specs,
      evaluator_env_factories=None,
      seed=FLAGS.seed,
      max_num_actor_steps=None,
      resume_training=True,
    ),
    experiment_dir,
  )

def main(_):
  if (FLAGS.cross_checkpoint_paths is None):
    raise ValueError("experiment_dir must be specified")

  if FLAGS.agent_param_indices is not None:
    FLAGS.agent_param_indices = [int(idx) for idx in FLAGS.agent_param_indices.split(",")]


  ckpt_paths = FLAGS.cross_checkpoint_paths.split(",")
  abbrs = [abbreviate_ckpt(ckpt) for ckpt in ckpt_paths]
  agent_roles = [role.strip() for role in FLAGS.agent_roles.split(",")] if FLAGS.agent_roles else None
  agent_ids = FLAGS.agent_param_indices
  abbrs = [f"{abbr}_{role[:4]}_{agent_id}" for abbr, role, agent_id in zip(abbrs, agent_roles, agent_ids)]
  log_dir = os.path.join(FLAGS.exp_log_dir, f"mix_{'_'.join(abbrs)}", )

  configs, ckpt_configs = [], []
  recurrent_dims = FLAGS.recurrent_dims.split(",")
  base_ckpt_path = ckpt_paths[0]
  base_recurrent_dim = int(recurrent_dims[0])
  param_dict = {
    'base': {
      "experiment_dir": base_ckpt_path,
      "num_agents": 2,
      "recurrent_dim": base_recurrent_dim,
    }
  }
  param_dict = {**param_dict, **{f"agent_{i}": {
    "experiment_dir": ckpt_paths[i],
    "num_agents": 1,
    "recurrent_dim": int(recurrent_dims[i]),
  } for i in range(len(ckpt_paths))}}

  for name, param in param_dict.items():
    ckpt_path = param["experiment_dir"]
    num_agents = param["num_agents"]
    recurrent_dim = param["recurrent_dim"]
    assert os.path.exists(ckpt_path), f"Checkpoint path {ckpt_path} does not exist"
    config, experiment_dir = build_experiment_config(ckpt_path, num_agents=num_agents, recurrent_dim=int(recurrent_dim),
                                                     video_path=f"mix_{'_'.join(abbrs)}_{FLAGS.map_name}_{FLAGS.map_layout}")

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
      make_experiment_logger, log_dir=log_dir + log_dir_append, use_tb=False)
    config.agent_param_indices = FLAGS.agent_param_indices
    config.using_my_custom_environment_loop = FLAGS.using_my_custom_environment_loop

    if FLAGS.plsc_decomposition_dict_path is not None:
      config.plsc_decomposition_dict_path = FLAGS.plsc_decomposition_dict_path
      config.plsc_dim_to_perturb = FLAGS.plsc_dim_to_perturb
      config.agent_to_perturb = FLAGS.agent_to_perturb
    configs.append(config)
    ckpt_configs.append(ckpt_config)
  if FLAGS.env_name == "meltingpot":
    if FLAGS.cross_checkpoint_paths is not None:
      # running evaluation on substrate
      experiments.run_cross_evaluation(
        configs, ckpt_configs, environment_name=FLAGS.map_name, num_eval_episodes=FLAGS.num_episodes, )
    else:
      experiments.run_evaluation(
        config, ckpt_config, environment_name=FLAGS.map_name, num_eval_episodes=FLAGS.num_episodes
      )
  else:
    raise NotImplementedError("Only meltingpot evaluation implemented")


if __name__ == "__main__":
  app.run(main)
