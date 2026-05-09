import os
import sys
import functools

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

from copy import deepcopy
import train
from marl import experiments
from marl.utils.experiment_utils import make_experiment_logger

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
flags.DEFINE_bool(
    "run_config_sweep", False, "Whether to run cross evaluation on a sweep of configurations defined in create_config_sweep(). If True, cross_play_config_path will be ignored."
)

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
    if FLAGS.run_config_sweep:
        name_to_config = create_config_sweep()
    else:
        cross_play_config_path = FLAGS.cross_play_config_path
        with open(cross_play_config_path, "r") as f:
            cross_play_config = yaml.safe_load(f)
        name_to_config = {cross_play_config_path: cross_play_config}

    for name, cross_play_config in name_to_config.items():
        print(f"Running cross evaluation for config: {name}")
    
        base_exp_config, _ = train.build_experiment_config(
            override_config_args=cross_play_config['env']
        )
        
        save_dir = f"results/cross_play/{name}"
        os.makedirs(save_dir, exist_ok=True)
        base_exp_config.logger_factory = functools.partial(
            make_experiment_logger, log_dir=save_dir, use_tb=False
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
