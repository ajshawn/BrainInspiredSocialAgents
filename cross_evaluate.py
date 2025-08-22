import os

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
import sys

cwd = os.getcwd()
sys.path.append(cwd)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = (
    "0.6"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991
)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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

flags.DEFINE_string("agent_param_indices", None, "Comma separated list of agent param indices.") 

flags.DEFINE_bool(
    "log_timesteps", False, "Whether to log each timestep's activations, locations, actions, rewards."
)
flags.DEFINE_integer(
    "n_episodes", 1, "The number of roll out episode to run"
)
flags.DEFINE_string(
    "ckp_map", None, "map which agent comes from which checkpoint"
) # example: 0:50, 1:50, 2:50

def main(_):
    if FLAGS.experiment_dir is None:
        raise ValueError("experiment_dir must be specified")

    config, experiment_dir = train.build_experiment_config()

    ckpt_config = ma_config.CheckpointingConfig(
        max_to_keep=3, directory=experiment_dir, add_uid=False
    )

    config.logger_factory = functools.partial(
        make_experiment_logger, log_dir=experiment_dir, use_tb=False
    )

    agent_param_indices = [int(idx) for idx in FLAGS.agent_param_indices.split(",")]
    config.agent_param_indices = agent_param_indices

    if FLAGS.ckp_map is not None:
        #print(FLAGS.ckp_map)
        #print(FLAGS.ckp_map.split(","))
        ckp_map = {
            int(agent): int(ckpt)
            for agent, ckpt in (pair.split(":") for pair in FLAGS.ckp_map.split(","))
        }
    else:
        ckp_map = None
    
    # running evaluation on substrate
    experiments.run_cross_evaluation(
        config,
        ckpt_config,
        environment_name=f"{FLAGS.env_name}_{FLAGS.map_name}",
        ckp_map=ckp_map,
        num_eval_episodes = FLAGS.n_episodes,
        log_timesteps = FLAGS.log_timesteps,
    )


if __name__ == "__main__":
    app.run(main)
