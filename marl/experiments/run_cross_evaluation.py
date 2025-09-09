"""
This is a fork of
  https://github.com/deepmind/acme/blob/master/acme/jax/experiments/run_experiment.py
with some modifications to work with MARL setup.
"""

"""Runner used for executing local MARL agent."""

import acme
from acme import core
from acme.jax import variable_utils
from acme.tf import savers as tf_savers
from acme.utils import counting
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
from typing import Optional
from marl import specs as ma_specs
from marl.experiments import config as ma_config
from marl.utils import experiment_utils as ma_utils
from marl.utils.scenario2agent_idx import SCENARIO_2_AGENT_IDX_OFFSET
from marl.experiments.environment_loop_event_log import EnvironmentLoopEvents


def run_cross_evaluation(
    experiment: ma_config.MAExperimentConfig,
    checkpointing_config: ma_config.CheckpointingConfig,
    environment_name: str,
    num_eval_episodes: int = 1,
    ckp_map: Optional[dict[int, int]] = None,   # agent_id → checkpoint mapping
    log_timesteps: bool = False,
):
    """
    Runs a simple evaluation loop where agents can come from different checkpoints.

    Arguments:
      experiment: Definition and configuration of the agent to run.
      checkpointing_config: Configuration for checkpointing to load checkpoint from.
      environment_name: The scenario to evaluate.
      num_eval_episodes: Number of episodes to run.
      ckp_map: Mapping {agent_idx: checkpoint_number} telling which checkpoint
               each agent should load params from.
      log_timesteps: Whether to log timesteps in addition to episodes.
    """

    key = jax.random.PRNGKey(experiment.seed)

    # Environment setup
    environment = experiment.environment_factory(experiment.seed)
    environment_specs: ma_specs.MAEnvironmentSpec = experiment.environment_spec

    network = experiment.network_factory(
        environment_specs.get_single_agent_environment_specs()
    )

    parent_counter = counting.Counter(time_delta=0.0)
    dataset = None

    learner_key, key = jax.random.split(key)
    learner = experiment.builder.make_learner(
        random_key=learner_key,
        networks=network,
        dataset=dataset,
        logger_fn=experiment.logger_factory,
        environment_spec=environment_specs,
    )

    template = learner._combined_states.params
    combined_params = jax.tree_map(lambda x: jnp.zeros_like(x), template)

    # Load params from each dir and checkpoint needed 
    jax.debug.print(f"ckpdir{checkpointing_config.directory}")
    dirs = checkpointing_config.directory.split(",")
    dummy = dirs[0]
    dirs = dirs[1:]
    jax.debug.print(dirs[0])

    for target_id, info in ckp_map.items():
        ckpt_num = info["ckpt_num"]
        ckpt_agent = info["ckpt_agent"]
        checkpointer = tf_savers.Checkpointer(
            objects_to_save={"learner": learner},
            directory=dirs[target_id],
            subdirectory="learner",
            time_delta_minutes=checkpointing_config.model_time_delta_minutes,
            add_uid=checkpointing_config.add_uid,
            max_to_keep=checkpointing_config.max_to_keep,
        )
        checkpointer.restore(ckp=ckpt_num)
        cache = learner._combined_states.params.copy()
        for k in combined_params.keys():
            for k_ in combined_params[k].keys():
                combined_params[k][k_] = combined_params[k][k_].at[target_id].set(
                    cache[k][k_][ckpt_agent]
                )
    print("Combined parameters from checkpoints:", ckp_map)

    # Variable client as usual 
    variable_client = variable_utils.VariableClient(
        client=learner, key="network", update_period=int(1), device="cpu"
    )

    # Create evaluation actor
    agent_idx_offset = SCENARIO_2_AGENT_IDX_OFFSET.get(environment_name, 0)
    eval_actor = Evaluate(
        network.forward_fn,
        network.initial_state_fn,
        n_agents=environment.num_agents,
        n_params=environment_specs.num_agents,
        agent_idx_offset=agent_idx_offset,
        variable_client=variable_client,
        rng=hk.PRNGSequence(experiment.seed),
        agent_param_indices=experiment.agent_param_indices,
    )

    # Inject merged params into actor
    eval_actor.params = combined_params

    # Evaluation loop 
    eval_counter = counting.Counter(
        parent_counter, prefix=environment_name, time_delta=0.0
    )
    eval_logger = experiment.logger_factory(
        label=f"{environment_name}", steps_key=eval_counter.get_steps_key(), task_instance=0
    )
    timestep_logger = experiment.logger_factory(
        label=f"{environment_name}-timesteps", steps_key=eval_counter.get_steps_key(), task_instance=0
    ) if log_timesteps else None

    eval_loop = EnvironmentLoopEvents(
        environment,
        eval_actor,
        counter=eval_counter,
        logger=eval_logger,
        should_update=False,
        log_timesteps=log_timesteps,
        timestep_logger=timestep_logger,
    )

    eval_loop.run(num_episodes=num_eval_episodes)

class Evaluate(core.Actor):

    def __init__(
        self,
        forward_fn,
        initial_state_fn,
        n_agents,
        n_params,
        agent_idx_offset,
        variable_client,
        rng,
        agent_param_indices,
    ):
        self.forward_fn = forward_fn
        self.n_agents = n_agents
        self.n_params = n_params
        self.agent_idx_offset = agent_idx_offset
        self._rng = rng
        self._variable_client = variable_client
        self.agent_param_indices = agent_param_indices

        # Params storage
        self._merged_params = None   # will be injected externally
        self._initial_states = ma_utils.merge_data(
            [initial_state_fn(next(rng)) for _ in range(self.n_agents)]
        )
        self._p_forward = jax.vmap(self.forward_fn)

        self._states = None
        self._embedding = None

    @property
    def params(self) -> hk.Params:
        """Prefer externally injected merged params, otherwise fall back to variable client."""
        if self._merged_params is not None:
            return self._merged_params
        return self._variable_client.params

    @params.setter
    def params(self, value: hk.Params):
        """Allow external code to inject merged params directly."""
        self._merged_params = value

    def select_action(self, observations):
        if self._states is None:
            self._states = self._initial_states
            self.update(True)

            # Deterministic agent selection
            selected_params = jnp.array(self.agent_param_indices) + self.agent_idx_offset
            self.episode_params = ma_utils.select_idx(self.params, selected_params)

        (logits, _, _, embedding), new_states = self._p_forward(
            self.episode_params, observations, self._states
        )

        # Greedy action selection
        actions = jnp.argmax(logits, axis=-1)

        self._embedding = embedding
        self._states = new_states
        return jax.tree_util.tree_map(lambda a: [*a], actions)

    def observe_first(self, timestep: dm_env.TimeStep):
        self._states = None

    def observe(self, action, next_timestep: dm_env.TimeStep):
        pass

    def update(self, wait: bool = True):
        self._variable_client.update(wait)