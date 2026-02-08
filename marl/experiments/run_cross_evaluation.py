"""
This is a fork of
  https://github.com/deepmind/acme/blob/master/acme/jax/experiments/run_experiment.py
with some modifications to work with MARL setup.
"""

"""Runner used for executing local MARL agent."""
from acme import core
from acme.jax import variable_utils
from acme.tf import savers as tf_savers
from acme.utils import counting
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
from typing import List, Tuple, Callable

import train
from marl import specs as ma_specs
from marl.experiments import config as ma_config
from marl.utils import experiment_utils as ma_utils
from marl.experiments.environment_loop_event_log import EnvironmentLoopEvents


def run_cross_evaluation(
    base_exp_config: ma_config.MAExperimentConfig,
    cross_play_config: dict,
    environment_name: str,
    num_eval_episodes: int = 1,    
    log_timesteps: bool = False,
):
    """
    Runs a simple evaluation loop where agents can come from different checkpoints.

    Arguments:
      cross_play_config: Configuration dict specifying the checkpoint mapping for cross evaluation.
      environment_name: The scenario to evaluate.
      num_eval_episodes: Number of episodes to run.
      log_timesteps: Whether to log timesteps in addition to episodes.
    """

    key = jax.random.PRNGKey(base_exp_config.seed)

    # Environment setup
    environment = base_exp_config.environment_factory(base_exp_config.seed)
    environment_specs: ma_specs.MAEnvironmentSpec = base_exp_config.environment_spec
    parent_counter = counting.Counter(time_delta=0.0)

    cross_data = []

    for agent_info in cross_play_config["agents"]:
        exp, _ = train.build_experiment_config(override_config_args=agent_info)
        ckpt_config = ma_config.CheckpointingConfig(directory=agent_info["ckp_dir"], add_uid=False)
        # Build the network and learner to load the checkpoint parameters.
        network_i = exp.network_factory(exp.environment_spec.get_single_agent_environment_specs())
        tmp_env_spec = exp.environment_factory(exp.seed)
        learner_key, key = jax.random.split(key)
        learner = exp.builder.make_learner(
            random_key=learner_key,
            networks=network_i,
            dataset=None,
            logger_fn=exp.logger_factory,
            environment_spec=tmp_env_spec,
        )
        
        s0 = learner._combined_states.params.copy()
        checkpointer = tf_savers.Checkpointer(
            objects_to_save={"learner": learner},
            directory=ckpt_config.directory,
            subdirectory="learner",
            time_delta_minutes=ckpt_config.model_time_delta_minutes,
            add_uid=ckpt_config.add_uid,
            max_to_keep=ckpt_config.max_to_keep,
        )
        checkpointer.restore()
        s1 = learner._combined_states.params.copy()        

        # testing that the learner parameters are actually loaded
        for k, v in s0.items():
            for k_, v_ in v.items():
                assert (s0[k][k_] - s1[k][k_]
                        ).sum() != 0, f'New parameters are the same as old {k}.{k_}'
        print(f'Learner parameters successfully updated for agent {agent_info["agent_idx"]} from checkpoint {agent_info["ckp_dir"]}.')

        variable_client = variable_utils.VariableClient(
            client=learner, key="network", update_period=int(1), device="cpu")
        params = learner._combined_states.params.copy()
        params = ma_utils.select_idx(params, jnp.array(agent_info["source_agent_idx"])).copy()
        cross_data.append((network_i.forward_fn, network_i.initial_state_fn, params, variable_client))

    # Create an evaluation actor that uses the separate network data.
    eval_actor = EvaluateCross(
        cross_data=cross_data,
        rng=hk.PRNGSequence(base_exp_config.seed),        
    )

    # Set up evaluation logging.
    eval_counter = counting.Counter(
        parent_counter, prefix=environment_name, time_delta=0.0
    )
    eval_logger = base_exp_config.logger_factory(
        label=f"{environment_name}", steps_key=eval_counter.get_steps_key(), task_instance=0
    )
    timestep_logger = base_exp_config.logger_factory(
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
    

class EvaluateCross(core.Actor):
  """Evaluation actor for cross-checkpoint rollouts with separate network factories.

  This actor receives a list of tuples containing the forward function,
  initial state function, and parameters for each agent.
  """
  def __init__(
      self,
      cross_data: List[Tuple[Callable, Callable, hk.Params, variable_utils.VariableClient]],
      rng,      
  ):
    self.cross_data = cross_data  # List of (forward_fn, initial_state_fn, params)
    self.n_agents = len(cross_data)
    self._rng = rng    
    
    self._initial_states = [
      init_fn(next(self._rng)) for (_, init_fn, _, _) in cross_data # No need to expand dims
    ]
    self._variable_client = [client for (_, _, _, client) in cross_data]
    self._states = None
    self._logits = None
    self._embedding = None

  def select_action(self, observations):
    if self._states is None:
        self._states = self._initial_states
        self.update(True)
    actions = []
    new_states = []
    embeddings = []
    for agent_idx in range(self.n_agents):
        forward_fn, _, params, _ = self.cross_data[agent_idx]
        # obs = observations[agent_idx]
        obs = {}
        for k, v in observations.items():
            if k == 'observation':
                obs[k] = {}
                for k_, v_ in v.items():
                    # obs[k][k_] = jnp.expand_dims(v_[agent_idx], axis=0)
                    obs[k][k_] = v_[agent_idx]
            else:
                # obs[k] = jnp.expand_dims(v[agent_idx], axis=0)
                obs[k] = v[agent_idx]
        state = self._states[agent_idx]
        # if state.
        (logits, _, _, embedding), new_state = forward_fn(params, obs, state)      

        action = int(jnp.argmax(logits, axis=-1))
        actions.append(action)
        new_states.append(new_state)
        embeddings.append(embedding)

    self._states = new_states
    self._logits = logits
    self._embedding = embeddings

    return [action for action in actions]

  def observe_first(self, timestep: dm_env.TimeStep):
    self._states = None

  def observe(self, action, next_timestep: dm_env.TimeStep):
    pass

  def update(self, wait: bool = True):
    # self._variable_client.update(wait)
    # pass
    for client in self._variable_client:
      client.update(wait)

  def _params(self) -> hk.Params:
    # return self._variable_client.params
    # return None
    return [client.params for client in self._variable_client]
