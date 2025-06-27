# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple agent-environment training loop."""

import operator
import time
from typing import List, Optional, Sequence

from acme import core
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
from acme.utils import signals

import dm_env
from dm_env import specs
import numpy as np
import tree
import logging

class MyCustomEnvironmentLoop(core.Worker):
  """A simple RL environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. Agent is updated if `should_update=True`. This can be used as:

    loop = EnvironmentLoop(environment, actor)
    loop.run(num_episodes)

  A `Counter` instance can optionally be given in order to maintain counts
  between different Acme components. If not given a local Counter will be
  created to maintain counts between calls to the `run` method.

  A `Logger` instance can also be passed in order to control the output of the
  loop. If not given a platform-specific default logger will be used as defined
  by utils.loggers.make_default_logger. A string `label` can be passed to easily
  change the label associated with the default logger; this is ignored if a
  `Logger` instance is given.

  A list of 'Observer' instances can be specified to generate additional metrics
  to be logged by the logger. They have access to the 'Environment' instance,
  the current timestep datastruct and the current action.
  """

  def __init__(
      self,
      environment: dm_env.Environment,
      actor: core.Actor,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      should_update: bool = True,
      label: str = 'environment_loop',
      observers: Sequence[observers_lib.EnvLoopObserver] = (),
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._actor = actor
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
      label, steps_key=self._counter.get_steps_key())
    self._should_update = should_update
    self._observers = observers

  def _get_time_step_data(self,
                          timestep: dm_env.TimeStep,
                          action: Optional[np.ndarray] = None):
    """Builds the dict of everything you want to log for a single timestep."""
    data = {
      "step": len(self._episode_data),
      "STAMINA": timestep.observation['observation'].get('STAMINA', []),
      "POSITION": timestep.observation['observation'].get('POSITION', []),
      "ORIENTATION": timestep.observation['observation'].get('ORIENTATION', []),
      "actions": action,
      "logits": getattr(self._actor, '_logits', None),
      "rewards": timestep.reward,
      "events": [],
    }
    ## Currently, the inventory is always 0. I excluded it.
    # if 'INVENTORY' in timestep.observation['observation']:
    #   data['INVENTORY'] = timestep.observation['observation']['INVENTORY']
    #   print(data['INVENTORY'])
    # collect events
    # Collect events
    # Collect events
    raw_events = self._environment.events() or []
    for event_tuple in raw_events:
      # Each event_tuple is expected to be (event_name_str, list_of_data)
      # e.g., ('attack_attempt', [b'dict', b'predator_index', 2.0, b'prey_index', 4.0])

      try:
        # Basic validation of the event_tuple structure
        if not (isinstance(event_tuple, tuple) and len(event_tuple) == 2 and
                isinstance(event_tuple[0], str) and isinstance(event_tuple[1], list)):
          logging.warning(f"Skipping malformed event_tuple: {event_tuple}")
          continue  # Skip to the next event if format is completely off

        event_name = event_tuple[0]
        event_data_list = event_tuple[1]

        parsed_event = {'name': event_name}  # Always include the event name

        # Determine the starting index for key-value pairs
        # If the first element is b'dict', start from index 1.
        # Otherwise, assume key-value pairs start from index 0.
        start_idx = 0
        if event_data_list and event_data_list[0] == b'dict':
          start_idx = 1

        # Iterate through the list, taking two elements at a time (key and value)
        current_idx = start_idx
        while current_idx < len(event_data_list) - 1:  # Ensure there's a key AND a value
          key_raw = event_data_list[current_idx]
          value_raw = int(event_data_list[current_idx + 1])

          # Attempt to decode key if it's bytes
          key_str = None
          if isinstance(key_raw, bytes):
            try:
              key_str = key_raw.decode('utf-8')
            except UnicodeDecodeError:
              logging.warning(f"Could not decode event key '{key_raw}' for event '{event_name}'. Skipping pair.")
          elif isinstance(key_raw, str):
            key_str = key_raw  # Already a string
          else:
            logging.warning(
              f"Unexpected type for event key '{key_raw}' (type: {type(key_raw)}) for event '{event_name}'. Skipping pair.")

          if key_str:  # Only add if key was successfully processed
            parsed_event[key_str] = value_raw

          current_idx += 2  # Move to the next potential key
        print(f"Parsed event: {parsed_event}")
        data['events'].append(parsed_event)

      except Exception as e:
        # Catch any other unexpected errors during parsing of a single event
        logging.error(f"Error parsing event {event_tuple}: {e}")
        # Optionally, you might append a partially parsed event or a placeholder
        # to indicate a problem, or just skip it as done above.
        continue  # Continue to the next raw event

    # collect tile codes
    obs = timestep.observation['observation']
    if 'WORLD.TILE_CODES' in obs:
      flat = obs['WORLD.TILE_CODES'][0]
      spec = self._environment.observation_spec()['observation']['WORLD.TILE_CODES']
      data['tile_code'] = flat.reshape(spec.shape[2], spec.shape[1])
      # print(data['tile_code'])
    # optional RNN state
    if hasattr(self._actor, '_states') and self._actor._states is not None:
      if isinstance(self._actor._states, list):
        data['hidden'] = [s.hidden.copy() for s in self._actor._states]
        data['cell']   = [s.cell.copy()   for s in self._actor._states]
      else:
        data['hidden'] = self._actor._states.hidden.copy()
        data['cell']   = self._actor._states.cell.copy()
    return data


  def run_episode(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    episode_start_time = time.time()
    select_action_durations: List[float] = []
    env_step_durations: List[float] = []
    episode_steps: int = 0

    ## Here is my modification to add the logging of the episode data

    # For evaluation, this keeps track of the total undiscounted reward
    # accumulated during the episode.
    episode_return = tree.map_structure(_generate_zeros_from_spec,
                                        self._environment.reward_spec())
    env_reset_start = time.time()
    timestep = self._environment.reset()
    env_reset_duration = time.time() - env_reset_start
    # Make the first observation.
    self._actor.observe_first(timestep)
    for observer in self._observers:
      # Initialize the observer with the current state of the env after reset
      # and the initial timestep.
      observer.observe_first(self._environment, timestep)

    # prepare the per‐episode buffer
    self._episode_data = []
    # A quick check of the first frame after observation
    time_step_dict = self._get_time_step_data(timestep, action=None)
    self._episode_data.append(time_step_dict)

    while not timestep.last():
      # Book-keeping.
      episode_steps += 1

      # Generate an action from the agent's policy.
      select_action_start = time.time()
      action = self._actor.select_action(timestep.observation)
      select_action_durations.append(time.time() - select_action_start)

      # Step the environment with the agent's selected action.
      env_step_start = time.time()
      timestep = self._environment.step(action)
      env_step_durations.append(time.time() - env_step_start)
      time_step_dict = self._get_time_step_data(timestep, action)
      self._episode_data.append(time_step_dict)

      # Have the agent and observers observe the timestep.
      self._actor.observe(action, next_timestep=timestep)
      for observer in self._observers:
        # One environment step was completed. Observe the current state of the
        # environment, the current timestep and the action.
        observer.observe(self._environment, timestep, action)

      # Give the actor the opportunity to update itself.
      if self._should_update:
        self._actor.update()

      # Equivalent to: episode_return += timestep.reward
      # We capture the return value because if timestep.reward is a JAX
      # DeviceArray, episode_return will not be mutated in-place. (In all other
      # cases, the returned episode_return will be the same object as the
      # argument episode_return.)
      episode_return = tree.map_structure(operator.iadd,
                                          episode_return,
                                          timestep.reward)

    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - episode_start_time)

    result = {
      'episode_length': episode_steps,
      'episode_return': episode_return,
      'steps_per_second': steps_per_second,
      'env_reset_duration_sec': env_reset_duration,
      'select_action_duration_sec': np.mean(select_action_durations),
      'env_step_duration_sec': np.mean(env_step_durations),
      'episode_data': self._episode_data,
    }
    result.update(counts)
    for observer in self._observers:
      result.update(observer.get_metrics())
    return result

  def run(
      self,
      num_episodes: Optional[int] = None,
      num_steps: Optional[int] = None,
  ) -> int:
    """Perform the run loop.

    Run the environment loop either for `num_episodes` episodes or for at
    least `num_steps` steps (the last episode is always run until completion,
    so the total number of steps may be slightly more than `num_steps`).
    At least one of these two arguments has to be None.

    Upon termination of an episode a new episode will be started. If the number
    of episodes and the number of steps are not given then this will interact
    with the environment infinitely.

    Args:
      num_episodes: number of episodes to run the loop for.
      num_steps: minimal number of steps to run the loop for.

    Returns:
      Actual number of steps the loop executed.

    Raises:
      ValueError: If both 'num_episodes' and 'num_steps' are not None.
    """

    if not (num_episodes is None or num_steps is None):
      raise ValueError('Either "num_episodes" or "num_steps" should be None.')

    def should_terminate(episode_count: int, step_count: int) -> bool:
      return ((num_episodes is not None and episode_count >= num_episodes) or
              (num_steps is not None and step_count >= num_steps))

    episode_count: int = 0
    step_count: int = 0
    with signals.runtime_terminator():
      while not should_terminate(episode_count, step_count):
        episode_start = time.time()
        result = self.run_episode()
        result = {**result, **{'episode_duration': time.time() - episode_start}}
        episode_data = result['episode_data']
        episode_count += 1
        step_count += int(result['episode_length'])
        # Log the given episode results.
        self._logger.write(result)

        # print(f"Saved episode {episode_count} data to {filename}")

    return step_count


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)
