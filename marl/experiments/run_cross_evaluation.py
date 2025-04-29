"""
This is a fork of
  https://github.com/deepmind/acme/blob/master/acme/jax/experiments/run_experiment.py
with modifications to work with a cross-checkpoint MARL setup using separate network factories.
"""
"""Runner used for executing local MARL agent cross-evaluation with separate network factories."""

import acme
from acme import core
from acme.jax import variable_utils
from acme.tf import savers as tf_savers
from acme.utils import counting
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import pickle
from typing import Optional, Tuple, List, Callable
from marl import specs as ma_specs
from marl.experiments import config as ma_config
from marl.utils import experiment_utils as ma_utils
from marl.utils.scenario2agent_idx import SCENARIO_2_AGENT_IDX_OFFSET
from marl.experiments.my_custom_environment_loop import MyCustomEnvironmentLoop

from marl.experiments.run_evaluation import build_plsc_perturb_matrices

def run_cross_evaluation(
    experiment_configs: List[ma_config.MAExperimentConfig],
    checkpointing_configs: List[ma_config.CheckpointingConfig],
    environment_name: str,
    num_eval_episodes: int = 5,
    run_eval_on_scenario: bool = False,
):
  """Runs a cross-checkpoint evaluation loop with separate network factories.

  Each experiment and its checkpoint configuration is used to load an agent's
  parameters using its own network factory.
  """
  # Use the first experiment as the base for environment and spec.
  base_experiment = experiment_configs[0]
  key = jax.random.PRNGKey(base_experiment.seed)

  # Create the environment and environment spec using the base experiment.
  environment = base_experiment.environment_factory(base_experiment.seed)
  environment_specs: ma_specs.MAEnvironmentSpec = base_experiment.environment_spec

  # Create a parent counter for evaluation steps.
  parent_counter = counting.Counter(time_delta=0.0)

  # Loop over experiments to load separate networks and checkpoints.
  # Each entry in cross_data is a tuple: (forward_fn, initial_state_fn, params)
  cross_data = []
  for agent_id, (exp, ckpt_config) in enumerate(zip(experiment_configs[1:], checkpointing_configs[1:])):
    # Use the experiment's own network factory.
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
    s1 = ma_utils.select_idx(s1, jnp.array(exp.agent_param_indices[agent_id])).copy()

    # testing that the learner parameters are actually loaded
    for k, v in s0.items():
      for k_, v_ in v.items():
        assert (s0[k][k_] - s1[k][k_]
                ).sum() != 0, f'New parameters are the same as old {k}.{k_}'
    print(f'Learner parameters successfully updated!')

    variable_client = variable_utils.VariableClient(
      client=learner, key="network", update_period=int(1), device="cpu")
    params = learner._combined_states.params.copy()

    if not run_eval_on_scenario:
      assert environment_specs.num_agents == len(exp.agent_param_indices), (
        f'Number of agents in the environment ({environment_specs.num_agents}) does not match '
        f'the number of agent param indices {exp.agent_param_indices}'
      )
      params = ma_utils.select_idx(params, jnp.array(exp.agent_param_indices[agent_id])).copy()
      # params = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), params)
    cross_data.append((network_i.forward_fn, network_i.initial_state_fn, params, variable_client))

  # Build perturbation matrices from the base experiment.
  pred_perturb_matrix, prey_perturb_matrix = build_plsc_perturb_matrices(base_experiment, environment_specs)

  # Create an evaluation actor that uses the separate network data.
  eval_actor = EvaluateCross(
    cross_data=cross_data,
    rng=hk.PRNGSequence(base_experiment.seed),
    pred_perturb_matrix=pred_perturb_matrix,
    prey_perturb_matrix=prey_perturb_matrix,
  )

  # Set up evaluation logging.
  eval_counter = counting.Counter(
    parent_counter, prefix=environment_name, time_delta=0.0)
  eval_logger = base_experiment.logger_factory(
    label=environment_name,
    steps_key=eval_counter.get_steps_key(),
    task_instance=0)

  if hasattr(base_experiment, "using_my_custom_environment_loop") and base_experiment.using_my_custom_environment_loop:
    eval_loop = MyCustomEnvironmentLoop(
      environment,
      eval_actor,
      counter=eval_counter,
      logger=eval_logger,
      should_update=False,
    )
  else:
    eval_loop = acme.EnvironmentLoop(
      environment,
      eval_actor,
      counter=eval_counter,
      logger=eval_logger,
      should_update=False,
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
      pred_perturb_matrix: Optional[jnp.ndarray] = None,
      prey_perturb_matrix: Optional[jnp.ndarray] = None,
  ):
    self.cross_data = cross_data  # List of (forward_fn, initial_state_fn, params)
    self.n_agents = len(cross_data)
    self._rng = rng
    self.pred_perturb_matrix = pred_perturb_matrix
    self.prey_perturb_matrix = prey_perturb_matrix
    # Initialize recurrent states for each agent and ensure they have a batch dimension.
    # self._initial_states = [
    #   jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), init_fn(next(self._rng)))
    #   for (_, init_fn, _, _) in cross_data
    # ]
    self._initial_states = [
      init_fn(next(self._rng)) for (_, init_fn, _, _) in cross_data # No need to expand dims
    ]
    self._variable_client = [client for (_, _, _, client) in cross_data]
    self._states = None
    self._logits = None

  def select_action(self, observations):
    if self._states is None:
      self._states = self._initial_states
      self.update(True)
    actions = []
    new_states = []
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
      (logits, _), new_state = forward_fn(params, obs, state)

      # Optionally apply hidden-state perturbations for 2-agent scenarios.
      if self.n_agents == 2:
        action_branch_name = next((k for k in params if "policy" in k), None)
        policy_params = params[action_branch_name] if action_branch_name else None
        if policy_params is None:
          raise ValueError("No policy parameters found in params. Check naming.")
        if agent_idx == 0 and self.pred_perturb_matrix is not None:
          hidden = new_state.hidden[0]
          hidden = jnp.dot(self.pred_perturb_matrix, hidden)
          logits = jnp.matmul(hidden, policy_params["w"][0]) + policy_params["b"][0]
        elif agent_idx == 1 and self.prey_perturb_matrix is not None:
          hidden = new_state.hidden[1]
          hidden = jnp.dot(self.prey_perturb_matrix, hidden)
          logits = jnp.matmul(hidden, policy_params["w"][1]) + policy_params["b"][1]

      action = int(jnp.argmax(logits, axis=-1))
      actions.append(action)
      new_states.append(new_state)
    self._states = new_states
    self._logits = logits


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

