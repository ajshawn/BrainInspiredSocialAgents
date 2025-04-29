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
import pickle
from marl import specs as ma_specs
from marl.experiments import config as ma_config
from marl.utils import experiment_utils as ma_utils
from marl.utils.scenario2agent_idx import SCENARIO_2_AGENT_IDX_OFFSET
from marl.experiments.my_custom_environment_loop import MyCustomEnvironmentLoop

from typing import Optional, Callable, Tuple


def build_plsc_perturb_matrices(
    experiment,
    environment_specs,
) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
  """Builds predator/prey perturbation matrices if the experiment is set up
  with plsc_decomposition_dict_path, plsc_dim_to_perturb, and agent_to_perturb.

  Returns:
      A tuple (pred_perturb_matrix, prey_perturb_matrix). Either or both
      may be None if no matrix was created or no agent is perturbed.
  """
  pred_perturb_matrix = None
  prey_perturb_matrix = None

  # 1) Check if we have a decomposition path
  plsc_path = getattr(experiment, 'plsc_decomposition_dict_path', None)
  if not plsc_path:
    return pred_perturb_matrix, prey_perturb_matrix  # nothing to do

  # 2) We currently only support 2-agent scenarios for PLSC
  if environment_specs.num_agents != 2:
    raise ValueError(
      f"PLSC is only supported for 2-agent rollouts, but got "
      f"{environment_specs.num_agents} agents."
    )

  # 3) Check if plsc_dim_to_perturb is set
  dim = getattr(experiment, 'plsc_dim_to_perturb', None)
  if dim is None:
    raise ValueError("experiment.plsc_dim_to_perturb must be set.")

  # 4) Load decomposition and retrieve U, s, V for the relevant agent indices
  with open(plsc_path, 'rb') as f:
    decomposition_dict = pickle.load(f)

  agent_idx_str = '_'.join(str(i) for i in experiment.agent_param_indices)
  if agent_idx_str not in decomposition_dict:
    raise KeyError(
      f"Did not find key {agent_idx_str!r} in PLSC decomposition file. "
      f"Available keys: {list(decomposition_dict.keys())}"
    )

  plsc_USV = decomposition_dict[agent_idx_str]
  U = plsc_USV['U'][:, :dim]  # shape [hidden_dim, dim]
  # s = plsc_USV['s'][0][:dim]   # only needed if you explicitly use 's'
  if 'Vh' in plsc_USV:
    V = plsc_USV['Vh'].T[:, :dim]
  elif 'V' in plsc_USV:
    V = plsc_USV['V'][:, :dim]
  else:
    raise ValueError("PLSC decomposition must contain 'V' or 'Vh'.")

  UUT = jnp.dot(U, U.T)  # shape [hidden_dim, hidden_dim]
  ImUUT = jnp.eye(UUT.shape[0]) - UUT
  VVT = jnp.dot(V, V.T)
  ImVVT = jnp.eye(VVT.shape[0]) - VVT

  # 5) Decide which matrices to apply, based on experiment.agent_to_perturb
  agent_roles = getattr(experiment, 'agent_to_perturb', None)
  if agent_roles:
    if 'predator' in agent_roles:
      pred_perturb_matrix = ImUUT
    if 'prey' in agent_roles:
      prey_perturb_matrix = ImVVT

  return pred_perturb_matrix, prey_perturb_matrix

def run_evaluation(
    experiment: ma_config.MAExperimentConfig,
    checkpointing_config: ma_config.CheckpointingConfig,
    environment_name: str,
    num_eval_episodes: int = 5,
    run_eval_on_scenario: bool = False,
):
  """Runs a simple, single-threaded evaluation loop using the default evaluators.

    Arguments:
      experiment: Definition and configuration of the agent to run.
      checkpointing_config: Configuration for checkpointing to load checkpoint from.
    """

  key = jax.random.PRNGKey(experiment.seed)

  # Create the environment and get its spec.
  environment = experiment.environment_factory(experiment.seed)
  environment_specs: ma_specs.MAEnvironmentSpec = experiment.environment_spec
  scenario_spec: ma_specs.MAEnvironmentSpec = ma_specs.MAEnvironmentSpec(
    environment)

  # Create the networks and policy.
  network = experiment.network_factory(
    environment_specs.get_single_agent_environment_specs())

  # Parent counter allows to share step counts between train and eval loops and
  # the learner, so that it is possible to plot for example evaluator's return
  # value as a function of the number of training episodes.
  parent_counter = counting.Counter(time_delta=0.0)

  # Create actor, and learner for generating, storing, and consuming
  # data respectively.
  dataset = None  # fakes.transition_dataset_from_spec(environment_specs.get_agent_environment_specs())

  learner_key, key = jax.random.split(key)
  learner = experiment.builder.make_learner(
    random_key=learner_key,
    networks=network,
    dataset=dataset,
    logger_fn=experiment.logger_factory,
    environment_spec=environment_specs,
  )

  # learner2 = experiment.builder.make_learner(
  # )

  s0 = learner._combined_states.params.copy()
  checkpointer = tf_savers.Checkpointer(
    objects_to_save={"learner": learner},
    directory=checkpointing_config.directory,
    subdirectory="learner",
    time_delta_minutes=checkpointing_config.model_time_delta_minutes,
    add_uid=checkpointing_config.add_uid,
    max_to_keep=checkpointing_config.max_to_keep,
  )
  checkpointer.restore()
  s1 = learner._combined_states.params.copy()

  if not run_eval_on_scenario:
    assert environment_specs.num_agents == len(experiment.agent_param_indices), \
      f'Number of agents in the environment ({environment_specs.num_agents}) does not match the number of agent param indices {experiment.agent_param_indices}'

    # Select the agent param indices to evaluate from s1
    s1 = ma_utils.select_idx(s1, jnp.array(experiment.agent_param_indices)).copy()

  # testing that the learner parameters are actually loaded
  for k, v in s0.items():
    for k_, v_ in v.items():
      assert (s0[k][k_] - s1[k][k_]
              ).sum() != 0, f'New parameters are the same as old {k}.{k_}'
  print(f'Learner parameters successfully updated!')

  variable_client = variable_utils.VariableClient(
    client=learner, key="network", update_period=int(1), device="cpu")

  # Create the evaluation actor and loop.
  eval_counter = counting.Counter(
    parent_counter, prefix=environment_name, time_delta=0.0)
  eval_logger = experiment.logger_factory(
    label=environment_name,
    steps_key=eval_counter.get_steps_key(),
    task_instance=0)

  # Add perturbation matrices from PLSC results
  pred_perturb_matrix, prey_perturb_matrix = build_plsc_perturb_matrices(experiment, environment_specs)

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
    pred_perturb_matrix=pred_perturb_matrix,
    prey_perturb_matrix=prey_perturb_matrix,
  )
  if hasattr(experiment, "using_my_custom_environment_loop") and experiment.using_my_custom_environment_loop:
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
      pred_perturb_matrix: Optional[jnp.ndarray] = None,
      prey_perturb_matrix: Optional[jnp.ndarray] = None,
  ):
    self.forward_fn = forward_fn
    self.n_agents = n_agents
    self.n_params = n_params
    self.agent_idx_offset = agent_idx_offset
    self._rng = rng
    self._variable_client = variable_client
    self._agent_param_indices = agent_param_indices
    self.pred_perturb_matrix = pred_perturb_matrix
    self.prey_perturb_matrix = prey_perturb_matrix
    """
    If you have a custom function plsc_transform_fn(states: List[hk.LSTMState]) -> List[hk.LSTMState],
    pass it in. Otherwise, the states won't be perturbed.
    """
    # self._plsc_transform_fn = plsc_transform_fn  # new, optional

    def initialize_states(rng_sequence: hk.PRNGSequence,) -> list[hk.LSTMState]:
      """Initialize the recurrent states of the actor."""
      states = list()
      for _ in range(self.n_agents):
        states.append(initial_state_fn(next(rng_sequence)))
      return states

    self._initial_states = ma_utils.merge_data(initialize_states(self._rng))
    self._p_forward = jax.vmap(self.forward_fn)
    self._states = None
    self.loaded_params = None
    self.selected_params = None
    self.episode_params = None
    self._logits = None


  def select_action(self, observations):
    if self._states is None:
      self._states = self._initial_states
      self.update(True)
      self.loaded_params = self._params
      # Replace the random parameter selection with deterministic sequential selection
      # Add agent_idx_offset to the selected_params to ensure the correct roles in scenario evaluations
      self.selected_params = (jnp.array(self._agent_param_indices))
      self.episode_params = ma_utils.select_idx(self.loaded_params, self.selected_params)

      # self.selected_params = jax.random.choice(
      #     next(self._rng), self.n_params, (self.n_agents,), replace=False)
      # self.episode_params = ma_utils.select_idx(self.loaded_params,
      #                                           self.selected_params)

    # Qin: Optionally apply PLSC transform to the states BEFORE forward pass
    (logits, _), new_states = self._p_forward(self.episode_params, observations,
                                              self._states)

    # 4) Possibly apply hidden-state perturbations for agent 0 (pred) and agent 1 (prey).
    #    Then rebuild new logits for each agent.
    if self.n_agents == 2 \
        and (self.pred_perturb_matrix is not None or self.prey_perturb_matrix is not None):
      # probability based action selection
      action_branch_name = next((k for k in self.episode_params if "policy" in k), None)
      policy_params = self.episode_params[action_branch_name] if action_branch_name else None
      if policy_params is None:
        raise ValueError("No policy parameters found in episode_params. Double check the naming.")

      # Extract agent 0 (predator) hidden
      hidden_0 = new_states.hidden[0]  # shape [hidden_dim]
      # Possibly apply pred_perturb_matrix
      if self.pred_perturb_matrix is not None:
        hidden_0 = jnp.dot(self.pred_perturb_matrix, hidden_0)

      # Extract agent 1 (prey) hidden
      hidden_1 = new_states.hidden[1]
      # Possibly apply prey_perturb_matrix
      if self.prey_perturb_matrix is not None:
        hidden_1 = jnp.dot(self.prey_perturb_matrix, hidden_1)

      # Rebuild final logits for each agent from the new hidden states
      # We assume policy_params["w"] shape is [n_agents, hidden_dim, action_dim]
      # and policy_params["b"] shape is [n_agents, action_dim].
      w = policy_params["w"]  # shape [2, hidden_dim, action_dim]
      b = policy_params["b"]  # shape [2, action_dim]
      logits_0 = jnp.matmul(hidden_0, w[0]) + b[0]  # shape [action_dim]
      logits_1 = jnp.matmul(hidden_1, w[1]) + b[1]  # shape [action_dim]
      logits = jnp.stack([logits_0, logits_1], axis=0)

      new_hidden = jnp.stack([hidden_0, hidden_1], axis=0)  # shape [2, hidden_dim]
      new_states = new_states._replace(hidden=new_hidden)

    # 5) Store the final logits for debugging (optional).
    self._logits = logits

    # 6) Select greedy actions from the final logits.
    actions = jnp.argmax(logits, axis=-1)  # shape [n_agents]

    # 7) Save updated LSTM states for the next step
    self._states = new_states

    # 8) Return the actions in the shape the environment expects
    #    e.g. a list for each agent
    return jax.tree_util.tree_map(lambda a: [*a], actions)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._states = None

  def observe(self, action, next_timestep: dm_env.TimeStep):
    pass

  def update(self, wait: bool = True):
    self._variable_client.update(wait)

  @property
  def _params(self) -> hk.Params:
    return self._variable_client.params
