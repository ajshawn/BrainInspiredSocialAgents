import os

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = (
    "0.2"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991
)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import datetime
import functools

from absl import app
from absl import flags
import jax
import launchpad as lp
from types import SimpleNamespace

import config
from marl import experiments
from marl import specs as ma_specs
from marl.agents import impala
from marl.agents import opre
from marl.agents.networks import ArrayFE
from marl.agents.networks import ImageFE
from marl.agents.networks import MeltingpotFE, MeltingpotFECNNVis
from marl.agents.networks import AttentionCNN_FE, AttentionSpatialCNN_FE, AttentionCNN_FE_SelfSupervise, MeltingpotFE_feedback
from marl.experiments import config as ma_config
from marl.experiments import inference_server
from marl.utils import helpers
from marl.utils import lp_utils as ma_lp_utils
from marl.utils.experiment_utils import make_experiment_logger

FLAGS = flags.FLAGS


def parse_list(flag_val, item_type=int, default=None):
    """Helper to parse comma-separated strings from FLAGS."""
    if not flag_val:
        return default if default is not None else []
    return [item_type(x.strip()) for x in flag_val.split(",")]


def _get_custom_env_configs():
    result = {}
    if FLAGS.env_name == "meltingpot" and FLAGS.map_name == "coop_mining":
        if FLAGS.conservative_mine_beam:
            result["conservative_mine_beam"] = True
            result["mining_reward"] = FLAGS.mining_reward
            result["iron_reward"] = FLAGS.iron_reward
            result["gold_reward"] = FLAGS.gold_reward
        if FLAGS.dense_ore_regrow:
            result["dense_ore_regrow"] = True
            result["iron_rate"] = FLAGS.iron_rate
            result["gold_rate"] = FLAGS.gold_rate
    if FLAGS.map_layout:
        result[FLAGS.map_layout] = True
    if FLAGS.max_episode_length:
        result["max_episode_length"] = FLAGS.max_episode_length
    return result


def build_experiment_config(override_config_args=None):
    """Builds experiment config which can be executed in different ways."""
    config_args = dict(        
        # Forward direct flags
        **{k: getattr(FLAGS, k) for k in [
            'env_name', 'map_name', 'reward_scale', 'prosocial', 'num_heads',
            'record_video', 'log_obs', 'log_filename', 'log_img_dir', 'log_interval',             
            'algo_name', 'positional_embedding', 'attn_enhance_multiplier', 'hidden_scale',
            'attn_enhance_item_idx', 'head_entropy_cost', 'attn_entropy_cost',
            'head_cross_entropy_cost', 'head_mse_cost', 'reward_pred_cost', 'attn_key_size'
        ]},
        # Parsed fields
        add_selection_vector=(FLAGS.add_selection_vector == "True"),
        attn_enhance_head_indices=parse_list(FLAGS.attn_enhance_head_indices),
        attn_enhance_agent_skip_indices=parse_list(FLAGS.attn_enhance_agent_skip_indices),
        disturb_heads=parse_list(FLAGS.disturb_heads),
        agent_roles=parse_list(FLAGS.agent_roles, str, None),
        frozen_agents=set(parse_list(FLAGS.frozen_agents)),    
        memory_efficient=not FLAGS.all_parallel,
    )
    
    if override_config_args:
        config_args.update(override_config_args)
    c = SimpleNamespace(**config_args)

    # Assert at most one of the attention head auxiliary losses is used
    assert sum([c.head_entropy_cost > 0, c.head_cross_entropy_cost > 0, c.head_mse_cost > 0]) <= 1

    autoreset = False

    if FLAGS.experiment_dir:
        assert (
            c.algo_name in FLAGS.experiment_dir or \
            any(key in c.algo_name for key in ["disturb", "visualization", "enhance"])                
        ), f"experiment_dir must be a {c.algo_name} experiment"
        assert (
            c.env_name in FLAGS.experiment_dir
        ), f"experiment_dir must be a {c.env_name} experiment"        
        experiment_dir = FLAGS.experiment_dir
        experiment_name = experiment_dir.split("/")[-1]
    else:
        experiment_name = f"{c.algo_name}_{FLAGS.seed}_{c.env_name}"
        experiment_name += f"_{c.map_name}"
        experiment_name += f"_{datetime.datetime.now()}"
        experiment_name = experiment_name.replace(" ", "_")
        experiment_dir = os.path.join(FLAGS.exp_log_dir, experiment_name)

    wandb_config = {
        "project": FLAGS.wandb_project,
        "entity": FLAGS.wandb_entity,
        "name": experiment_name,
        "group": experiment_name,
        "resume": True if FLAGS.experiment_dir else False,
        "tags": [st for st in FLAGS.wandb_tags.split(",") if st],
    }

    feature_extractor = ArrayFE

    # Create environment factory
    if c.env_name == "overcooked":
        env_factory = lambda seed: helpers.make_overcooked_environment(
            seed,
            c.map_name,
            autoreset=autoreset,
            reward_scale=c.reward_scale,
            global_observation_sharing=True,
            record=c.record_video,
        )
        num_options = 8
    elif c.env_name == "ssd":
        env_factory = lambda seed: helpers.make_ssd_environment(
            seed,
            c.map_name,
            autoreset=autoreset,
            reward_scale=c.reward_scale,
            team_reward=c.prosocial,
            global_observation_sharing=True,
            record=c.record_video,
        )
        feature_extractor = ImageFE
        num_options = 8
    elif c.env_name == "meltingpot":
        custom_env_configs = _get_custom_env_configs()
        env_factory = lambda seed: helpers.env_factory(
            seed,
            c.map_name,
            autoreset=autoreset,
            shared_reward=c.prosocial,
            reward_scale=c.reward_scale,
            shared_obs=False,
            record=c.record_video,
            agent_roles=c.agent_roles,
            log_obs=c.log_obs,
            log_filename=c.log_filename,
            log_img_dir=c.log_img_dir,
            log_interval=c.log_interval,    
            attn_enhance_agent_skip_indices=c.attn_enhance_agent_skip_indices,    
            **custom_env_configs,
        )
        feature_extractor = MeltingpotFE
        #feature_extractor = AttentionCNN_FE
        num_options = 16
    else:
        raise ValueError(f"Unknown env_name {c.env_name}")

    environment_specs = ma_specs.MAEnvironmentSpec(env_factory(0))

    if c.algo_name == "IMPALA":
        # Create network
        network_factory = functools.partial(
            impala.make_network, feature_extractor=feature_extractor
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, memory_efficient=c.memory_efficient
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.IMPALABuilder(config, core_state_spec=core_spec)
    
    elif c.algo_name == "PopArtIMPALA":
        # Create network
        network_factory = functools.partial(
            impala.make_network_2, feature_extractor=feature_extractor
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, memory_efficient=c.memory_efficient
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)

    elif c.algo_name == "PopArtIMPALA_CNN_visualization":
        # Create network
        network_factory = functools.partial(
            impala.make_network_impala_cnn_visualization, feature_extractor=MeltingpotFECNNVis
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, 
            memory_efficient=c.memory_efficient,
            head_cross_entropy_cost=c.head_cross_entropy_cost, 
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)
    
    elif c.algo_name == "PopArtIMPALA_attention":
        # Create network
        network_factory = functools.partial(
            impala.make_network_attention, feature_extractor=AttentionCNN_FE, positional_embedding=c.positional_embedding, add_selection_vec = c.add_selection_vector
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, memory_efficient=c.memory_efficient
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)
        
    elif c.algo_name == "PopArtIMPALA_attention_spatial":
        # Create network
        network_factory = functools.partial(
            impala.make_network_attention_spatial, feature_extractor=AttentionSpatialCNN_FE, add_selection_vec = c.add_selection_vector
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, memory_efficient=c.memory_efficient
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)

    elif c.algo_name == "PopArtIMPALA_attention_item_aware":
        # Create network
        network_factory = functools.partial(
            impala.make_network_attention_item_aware, 
            feature_extractor=AttentionCNN_FE, 
            positional_embedding=c.positional_embedding,
            attn_enhance_multiplier=c.attn_enhance_multiplier,
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, memory_efficient=c.memory_efficient
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)

    elif c.algo_name == "PopArtIMPALA_attention_tanh":
        # Create network
        network_factory = functools.partial(
            impala.make_network_attention_tanh, feature_extractor=AttentionCNN_FE, positional_embedding=c.positional_embedding
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, memory_efficient=c.memory_efficient
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)

    elif c.algo_name == "PopArtIMPALA_attention_multihead":
        # Create network
        network_factory = functools.partial(
            impala.make_network_attention_multihead, 
            feature_extractor=AttentionCNN_FE, 
            positional_embedding=c.positional_embedding,
            add_selection_vec=c.add_selection_vector,
            attn_enhance_multiplier=c.attn_enhance_multiplier,
            num_heads=c.num_heads,
            key_size=c.attn_key_size,
            hidden_scale = c.hidden_scale,
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, memory_efficient=c.memory_efficient, head_entropy_cost=c.head_entropy_cost, attn_entropy_cost=c.attn_entropy_cost,
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)
    
    elif c.algo_name == "PopArtIMPALA_attention_multihead_ff":
        # Create network
        network_factory = functools.partial(
            impala.make_network_attention_multihead_ff, 
            feature_extractor=AttentionCNN_FE, 
            positional_embedding=c.positional_embedding,
            num_heads=c.num_heads,
            key_size=c.attn_key_size,
            hidden_scale = c.hidden_scale,
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, memory_efficient=c.memory_efficient, head_entropy_cost=c.head_entropy_cost, attn_entropy_cost=c.attn_entropy_cost,
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)

    elif c.algo_name == "PopArtIMPALA_attention_multihead_gated":
        # Create network
        network_factory = functools.partial(
            impala.make_network_attention_multihead_gated, 
            feature_extractor=AttentionCNN_FE, 
            positional_embedding=c.positional_embedding,
            num_heads=c.num_heads,
            key_size=c.attn_key_size,
            hidden_scale = c.hidden_scale,
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, memory_efficient=c.memory_efficient, head_entropy_cost=c.head_entropy_cost, attn_entropy_cost=c.attn_entropy_cost,
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)

    elif c.algo_name == "PopArtIMPALA_attention_multihead_disturb":
        # Create network
        network_factory = functools.partial(
            impala.make_network_attention_multihead_disturb, 
            feature_extractor=AttentionCNN_FE, 
            positional_embedding=c.positional_embedding,
            add_selection_vec=c.add_selection_vector,
            attn_enhance_multiplier=c.attn_enhance_multiplier,
            num_heads=c.num_heads,
            disturb_heads=c.disturb_heads,
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, memory_efficient=c.memory_efficient, head_entropy_cost=c.head_entropy_cost,
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)
        
    elif c.algo_name == "PopArtIMPALA_attention_multihead_enhance":
        # Create network
        network_factory = functools.partial(
            impala.make_network_attention_multihead_enhance, 
            feature_extractor=AttentionCNN_FE, 
            positional_embedding=c.positional_embedding,
            add_selection_vec=c.add_selection_vector,            
            num_heads=c.num_heads,
            key_size=c.attn_key_size,
            attn_enhance_multiplier=c.attn_enhance_multiplier,
            attn_enhance_head_indices=c.attn_enhance_head_indices,
            attn_enhance_item_idx=c.attn_enhance_item_idx,
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, 
            memory_efficient=c.memory_efficient, 
            head_entropy_cost=c.head_entropy_cost,
            head_cross_entropy_cost=c.head_cross_entropy_cost, 
            head_mse_cost=c.head_mse_cost
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)
                
    elif c.algo_name == "PopArtIMPALA_attention_multihead_item_aware":
        # Create network
        network_factory = functools.partial(
            impala.make_network_attention_multihead_item_aware, 
            feature_extractor=AttentionCNN_FE, 
            positional_embedding=c.positional_embedding,
            add_selection_vec=c.add_selection_vector,
            num_heads=c.num_heads,
            key_size=c.attn_key_size,
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, 
            memory_efficient=c.memory_efficient, 
            head_entropy_cost=c.head_entropy_cost,
            head_cross_entropy_cost=c.head_cross_entropy_cost,
            head_mse_cost=c.head_mse_cost,
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)

    elif c.algo_name == "PopArtIMPALA_attention_multihead_self_supervision":
        # Create network
        network_factory = functools.partial(
            impala.make_network_attention_multihead_self_supervision, 
            feature_extractor=AttentionCNN_FE_SelfSupervise,
            positional_embedding=c.positional_embedding,
            add_selection_vec=c.add_selection_vector,
            num_heads=c.num_heads,
            key_size=c.attn_key_size,
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, 
            memory_efficient=c.memory_efficient, 
            head_entropy_cost=c.head_entropy_cost,
            head_cross_entropy_cost=c.head_cross_entropy_cost,
            head_mse_cost=c.head_mse_cost,
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)

    elif c.algo_name == "simple_transformer":
        # Create network
        network_factory = functools.partial(
            impala.make_network_simple_transformer,
            feature_extractor=MeltingpotFE,
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, 
            memory_efficient=c.memory_efficient, 
            head_entropy_cost=c.head_entropy_cost,
            head_cross_entropy_cost=c.head_cross_entropy_cost,
            head_mse_cost=c.head_mse_cost,
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)
    
    elif c.algo_name == "simple_transformer_attention":
        # Create network
        network_factory = functools.partial(
            impala.make_network_transformer_attention,
            feature_extractor=AttentionCNN_FE,
            positional_embedding=c.positional_embedding,
            hidden_scale = c.hidden_scale
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, 
            memory_efficient=c.memory_efficient, 
            head_entropy_cost=c.head_entropy_cost,
            head_cross_entropy_cost=c.head_cross_entropy_cost,
            attn_entropy_cost=c.attn_entropy_cost,
            head_mse_cost=c.head_mse_cost,
            reward_pred_cost = c.reward_pred_cost,
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)
    elif c.algo_name == "simple_transformer_cnnfeedback":
        # Create network
        network_factory = functools.partial(
            impala.make_network_transformer_cnnfeedback,
            feature_extractor=MeltingpotFE_feedback,
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = impala.IMPALAConfig(
            n_agents=environment_specs.num_agents, 
            memory_efficient=c.memory_efficient, 
            head_entropy_cost=c.head_entropy_cost,
            head_cross_entropy_cost=c.head_cross_entropy_cost,
            attn_entropy_cost=c.attn_entropy_cost,
            head_mse_cost=c.head_mse_cost,
            reward_pred_cost = c.reward_pred_cost
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = impala.PopArtIMPALABuilder(config, core_state_spec=core_spec)

    elif c.algo_name == "OPRE":
        # Create network
        network_factory = functools.partial(
            opre.make_network,
            num_options=num_options,
            feature_extractor=feature_extractor,
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = opre.OPREConfig(
            n_agents=environment_specs.num_agents,
            num_options=num_options,
            memory_efficient=c.memory_efficient,
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = opre.OPREBuilder(config, core_state_spec=core_spec)
    
    elif c.algo_name == "PopArtOPRE":
        # Create network
        network_factory = functools.partial(
            opre.make_network_2,
            num_options=num_options,
            feature_extractor=feature_extractor,
        )
        network = network_factory(
            environment_specs.get_single_agent_environment_specs()
        )
        # Construct the agent.
        config = opre.OPREConfig(
            n_agents=environment_specs.num_agents,
            num_options=num_options,
            memory_efficient=c.memory_efficient,
        )
        core_spec = network.initial_state_fn(jax.random.PRNGKey(0))
        builder = opre.PopArtOPREBuilder(config, core_state_spec=core_spec)
    
    else:
        raise ValueError(f"Unknown algo_name {c.algo_name}")

    # Add frozen agents
    builder._config.frozen_agents = c.frozen_agents

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
            max_num_actor_steps=FLAGS.num_steps,
            resume_training=True if FLAGS.experiment_dir else False,
        ),
        experiment_dir,
    )


def main(_):
    assert not FLAGS.record_video, "Video recording is not supported during training"
    config, experiment_dir = build_experiment_config()
    ckpt_config = ma_config.CheckpointingConfig(
        max_to_keep=500, directory=experiment_dir, add_uid=False
    )
    if FLAGS.async_distributed:

        nodes_on_gpu = helpers.node_allocation(
            FLAGS.available_gpus, FLAGS.inference_server
        )
        program = experiments.make_distributed_experiment(
            experiment=config,
            num_actors=FLAGS.num_actors * FLAGS.actors_per_node,
            inference_server_config=(
                inference_server.InferenceServerConfig(
                    batch_size=min(8, FLAGS.num_actors // 2),
                    update_period=1,
                    timeout=datetime.timedelta(
                        seconds=1, milliseconds=0, microseconds=0
                    ),
                )
                if FLAGS.inference_server
                else None
            ),
            num_actors_per_node=FLAGS.actors_per_node,
            checkpointing_config=ckpt_config,
        )
        local_resources = ma_lp_utils.to_device(
            program_nodes=program.groups.keys(), nodes_on_gpu=nodes_on_gpu
        )

        lp.launch(
            program,
            launch_type="local_mp",
            terminal="current_terminal",
            local_resources=local_resources,
        )
    else:
        experiments.run_experiment(
            experiment=config, checkpointing_config=ckpt_config, num_eval_episodes=0
        )


if __name__ == "__main__":
    app.run(main)
