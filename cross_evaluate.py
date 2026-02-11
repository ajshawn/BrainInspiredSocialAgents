import os
import sys

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = (
    "0.6"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991
)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

cwd = os.getcwd()
sys.path.append(cwd)

from absl import app
from absl import flags
import yaml

import config
import train
from marl import experiments

FLAGS = flags.FLAGS

flags.DEFINE_string("agent_param_indices", None, "Comma separated list of agent param indices.") 

flags.DEFINE_bool(
    "log_timesteps", False, "Whether to log each timestep's activations, locations, actions, rewards."
)
flags.DEFINE_integer(
    "n_episodes", 1, "The number of roll out episode to run"
)
flags.DEFINE_string(
    "cross_play_config_path", None, "Path to YAML config file specifying the checkpoint mapping for cross evaluation."
) 

flags.DEFINE_string(
    "save_dir", None, "save directory for cross evaluation results"
) 

def main(_):
    cross_play_config_path = FLAGS.cross_play_config_path
    with open(cross_play_config_path, "r") as f:
        cross_play_config = yaml.safe_load(f)
    
    if FLAGS.save_dir is None:
        FLAGS.save_dir = FLAGS.experiment_dir

    base_exp_config, _ = train.build_experiment_config(
        override_config_args=cross_play_config['env']
    )

    # running evaluation
    experiments.run_cross_evaluation( 
        base_exp_config = base_exp_config,    
        cross_play_config = cross_play_config,
        environment_name=f"{FLAGS.env_name}_{FLAGS.map_name}",        
        num_eval_episodes = FLAGS.n_episodes,
        log_timesteps = FLAGS.log_timesteps,
    )


if __name__ == "__main__":
    app.run(main)
