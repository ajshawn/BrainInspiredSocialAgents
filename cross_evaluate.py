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

flags.DEFINE_string(
    "save_dir", None, "save directory for cross evaluation results"
) 

def create_config_sweep():
    env = {
        'name': 'meltingpot', 
        'map_name': 'predator_prey__open', 
        'agent_roles': ['predator', 'prey', 'prey', 'prey']
    }

    # values of (ckp_dir, ckp_num) tuples for each agent
    ckps = {
        'cnn_0': ("results/cross_play/PopArtIMPALA_1_meltingpot_predator_prey__open_2026-04-12_00_46_30.783807", 65),
        'cnn_1': ("results/cross_play/PopArtIMPALA_0_meltingpot_predator_prey__open_2026-04-07_17_10_45.655515", 86),
        'cnn_2': ("results/cross_play/PopArtIMPALA_2_meltingpot_predator_prey__open_2026-04-12_00_47_22.704925", 65),
        'cnn_3': ("results/cross_play/PopArtIMPALA_3_meltingpot_predator_prey__open_2026-04-15_14_10_32.876746", 54),
        'cnn_4': ("results/cross_play/PopArtIMPALA_4_meltingpot_predator_prey__open_2026-04-15_14_10_58.033055", 65),
        'attn_0': ("results/cross_play/PopArtIMPALA_attention_multihead_0_meltingpot_predator_prey__open_2026-04-02_16_21_03.610948", 105),
        'attn_1': ("results/cross_play/PopArtIMPALA_attention_multihead_2_meltingpot_predator_prey__open_2026-04-02_16_38_57.810965", 106),
        'attn_2': ("results/cross_play/PopArtIMPALA_attention_multihead_1_meltingpot_predator_prey__open_2026-04-23_15_48_45.124070", 76),
        'attn_3': ("results/cross_play/PopArtIMPALA_attention_multihead_3_meltingpot_predator_prey__open_2026-04-30_13_32_28.699609", 43),
        'attn_4': ("results/cross_play/PopArtIMPALA_attention_multihead_4_meltingpot_predator_prey__open_2026-04-17_21_17_32.406172", 85),
    }

    cnn_template = {
        'agent_idx': None,
        'algo_name': "PopArtIMPALA",
        'ckp_dir': None,
        'ckp_num': None,
        'source_agent_idx': None,
        'agent_roles': ['predator', 'prey', 'prey', 'prey']
    }
    attn_template = {
        'agent_idx': None,
        'algo_name': "PopArtIMPALA_attention_multihead",
        'ckp_dir': None,
        'ckp_num': None,
        'source_agent_idx': None,
        'num_heads': 2,
        'positional_embedding': "learnable",
        'agent_roles': ['predator', 'prey', 'prey', 'prey']
    }

    name_to_config = {}
    PREDATOR_IDX = 0
    PREY_IDX_START = 1
    N_PREYS = 3
    # Choose predator from current ckp, sweep through preys from all ckps
    for pred_ckp_key, (pred_ckp_dir, pred_ckp_num) in ckps.items():       
        for prey_ckp_key, (prey_ckp_dir, prey_ckp_num) in ckps.items():  
            agents = []

            predator_config = deepcopy(cnn_template) if "cnn" in pred_ckp_key else deepcopy(attn_template)
            predator_config.update({
                'agent_idx': PREDATOR_IDX,
                'ckp_dir': pred_ckp_dir,
                'ckp_num': pred_ckp_num,
                'source_agent_idx': PREDATOR_IDX,
            })
            agents.append(predator_config)

            for prey_idx in range(PREY_IDX_START, PREY_IDX_START + N_PREYS):
                prey_config = deepcopy(cnn_template) if "cnn" in prey_ckp_key else deepcopy(attn_template)
                prey_config.update({
                    'agent_idx': prey_idx,
                    'ckp_dir': prey_ckp_dir,
                    'ckp_num': prey_ckp_num,
                    'source_agent_idx': prey_idx,
                })
                agents.append(prey_config)
            
            config = {
                'env': deepcopy(env),
                'agents': agents
            }
            name_to_config[f"pred_{pred_ckp_key}__prey_{prey_ckp_key}"] = config   

    return name_to_config 

def main(_):
    if FLAGS.experiment_dir is None:
        raise ValueError("experiment_dir must be specified")
    
        base_exp_config, _ = train.build_experiment_config(
            override_config_args=cross_play_config['env']
        )
        
        save_dir = f"results/cross_play/{name}"
        os.makedirs(save_dir, exist_ok=True)
        base_exp_config.logger_factory = functools.partial(
            make_experiment_logger, log_dir=save_dir, use_tb=False
        )

    # OBJECTS_IN_VIEW_TENSOR is only consumed by the timestep log
    config, experiment_dir = train.build_experiment_config(
        objects_in_view_tensor=FLAGS.log_timesteps)

    ckpt_config = ma_config.CheckpointingConfig(
        max_to_keep=3, directory=experiment_dir, add_uid=False
    )

    config.logger_factory = functools.partial(
        make_experiment_logger, log_dir=FLAGS.save_dir, use_tb=False
    )

    agent_param_indices = [int(idx) for idx in FLAGS.agent_param_indices.split(",")]
    config.agent_param_indices = agent_param_indices

    if FLAGS.ckp_map is not None:
        #print(FLAGS.ckp_map)
        #print(FLAGS.ckp_map.split(","))
        ckp_map = {
            int(target): {
                "ckpt_num": int(ckpt_num),
                "ckpt_agent": int(ckpt_agent)
            }
            for target, pair in (entry.split(":") for entry in FLAGS.ckp_map.split(","))
            for ckpt_num, ckpt_agent in [pair.split("-")]
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
