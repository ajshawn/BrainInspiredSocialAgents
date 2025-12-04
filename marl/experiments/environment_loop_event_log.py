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
from collections import defaultdict
import operator
import time
from typing import List, Optional, Sequence
import json

from acme import core
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
from acme.utils import signals

import dm_env
from dm_env import specs
import numpy as np
import tree


class EnvironmentLoopEvents(core.Worker):
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
        label: str = "environment_loop",
        observers: Sequence[observers_lib.EnvLoopObserver] = (),
        log_timesteps: bool = False,
        timestep_logger: Optional[loggers.Logger] = None,
    ):
        # Internalize agent and environment.
        self._environment = environment
        self._actor = actor
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            label, steps_key=self._counter.get_steps_key()
        )
        self._should_update = should_update
        self._observers = observers
        self._log_timesteps = log_timesteps
        self._timestep_logger = timestep_logger

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

        # log custom events
        episode_data = {
            "mine_iron": defaultdict(int),
            "mine_gold": defaultdict(int),
            "extract_gold": defaultdict(int),
            "follower_mining": defaultdict(int),
            "acorn_collected": defaultdict(int),
            "acorn_consumed": defaultdict(int),
            "acorn_consumed_grass": defaultdict(int),
            "apple": defaultdict(int),
            "prey_caught": defaultdict(int),
        }
        timestep_data = []
        # For evaluation, this keeps track of the total undiscounted reward
        # accumulated during the episode.

        episode_return = tree.map_structure(
            _generate_zeros_from_spec, self._environment.reward_spec()
        )
        env_reset_start = time.time()
        timestep = self._environment.reset()
        env_reset_duration = time.time() - env_reset_start
        # Make the first observation.
        self._actor.observe_first(timestep)
        for observer in self._observers:
            # Initialize the observer with the current state of the env after reset
            # and the initial timestep.
            observer.observe_first(self._environment, timestep)
        
        # Run an episode.
        while not timestep.last():
            # Book-keeping.
            episode_steps += 1
            
            # current observation (bef action)

            position = timestep.observation['observation'].get('POSITION', np.array([])).tolist(),
            orientation = timestep.observation['observation'].get('ORIENTATION', np.array([])).tolist(),
            obj_in_view = timestep.observation['observation'].get('OBJECTS_IN_VIEW', np.array([])).tolist()
            
            # Generate an action from the agent's policy.
            select_action_start = time.time()
            action = self._actor.select_action(timestep.observation)
            select_action_durations.append(time.time() - select_action_start)

            # Step the environment with the agent's selected action.
            env_step_start = time.time()
            timestep = self._environment.step(action)
            env_step_durations.append(time.time() - env_step_start)

            rewards = timestep.reward
            #OBS = self._environment.observation_spec()
            
            events = self._environment.environment.events()
            # add a small punishment for the zapping action 
            # for i, x in enumerate(action):
            #     if x == 7:
            #         timestep.reward[i] -= 0.1
            # f = open("temp_obs.json", "w")
            
            # episode_steps
            # log events - training 
            mine_iron = [0] * len(rewards)
            mine_gold = [0] * len(rewards)
            extract_gold = [0] * len(rewards)
            acorn_collected = [0] * len(rewards)
            acorn_consumed = [0] * len(rewards)
            acorn_consumed_grass = [0] * len(rewards)
            apple = [0] * len(rewards)
            prey_caught = [0] * len(rewards)
            for event in events:
                event_type = event[0]
                if event_type == "mining" and event[1][4] == 1:
                    episode_data["mine_iron"][int(event[1][2])] += 1
                    mine_iron[int(event[1][2])-1] = 1
                elif event_type == "mining" and event[1][4] == 2: # mined gold 
                    episode_data["mine_gold"][int(event[1][2])] += 1
                    mine_gold[int(event[1][2])-1] = 1
                    if rewards[int(event[1][2])-1]>=2: # mine and extract at the same time - follower 
                        episode_data["follower_mining"][int(event[1][2])] += 1
                elif event_type == "extraction" and event[1][4] == 2: # extracted gold 
                    episode_data["extract_gold"][int(event[1][2])] += 1
                    extract_gold[int(event[1][2])-1] = 1
                # predator-prey 
                elif event_type == "acorn_collected":
                    episode_data["acorn_collected"][int(event[1][2])] += 1
                    acorn_collected[int(event[1][2])-1] = 1
                elif event_type == "acorn_consumed":
                    episode_data["acorn_consumed"][int(event[1][2])] += 1
                    acorn_consumed[int(event[1][2])-1] = 1
                elif event_type == "acorn_consumed_safely":
                    episode_data["acorn_consumed_grass"][int(event[1][2])] += 1
                    acorn_consumed_grass[int(event[1][2])-1] = 1
                elif event_type == "apple_consumed":
                    episode_data["apple"][int(event[1][2])] += 1
                    apple[int(event[1][2])-1] = 1
                elif event_type == "prey_consumed":
                    episode_data["prey_caught"][int(event[1][4])] += 1
                    prey_caught[int(event[1][4])-1] = 1
                # elif event_type == "overextration" or any([r == -0.5 for r in rewards]):
                #     print(rewards)
                #     f.write(json.dump(timestep.observation.to_list()))
                # log events - each time step  action and rewards
            if self._log_timesteps:
                timestep_data.append({"action":[int(arr.item()) for arr in action],
                "mine_iron":mine_iron,
                "mine_gold":mine_gold,
                "extract_gold":extract_gold,
                "acorn_collected":acorn_collected,
                "acorn_consumed":acorn_consumed,
                "acorn_consumed_grass":acorn_consumed_grass,
                "apple":apple,
                "prey_caught":prey_caught,
                "reward":list(map(float, rewards)),
                "timestep":int(episode_steps),
                "position": position,
                "orientation": orientation,
                "obj_in_view": obj_in_view,
                "hidden": self._actor._states.hidden.tolist(),
                "embedding": self._actor._embedding[0].tolist() if isinstance(self._actor._embedding, tuple) else self._actor._embedding.tolist(),
                "cell_states": self._actor._states.cell.tolist(),
                })

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
            episode_return = tree.map_structure(
                operator.iadd, episode_return, timestep.reward
            )

        # Record counts.
        counts = self._counter.increment(episodes=1, steps=episode_steps)

        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - episode_start_time)
        
        result = {
            "episode_length": episode_steps,
            "episode_return": episode_return,
            "steps_per_second": steps_per_second,
            "env_reset_duration_sec": env_reset_duration,
            "select_action_duration_sec": np.mean(select_action_durations),
            "env_step_duration_sec": np.mean(env_step_durations),
            #"timestep_data":timestep_data,
        }
        # Add episode custom data
        update_dict = {}
        for agent_id in range(1,len(rewards)+1):
            update_dict.update(
                {
                    f"mine_iron_{agent_id}": episode_data["mine_iron"].get(agent_id, 0),
                    f"mine_gold_{agent_id}": episode_data["mine_gold"].get(agent_id, 0),
                    f"extract_gold_{agent_id}": episode_data["extract_gold"].get(agent_id, 0),
                    f"follower_mining_{agent_id}": episode_data["follower_mining"].get(agent_id, 0),
                    f"acorn_collected_{agent_id}": episode_data["acorn_collected"].get(agent_id, 0),
                    f"acorn_consumed_{agent_id}": episode_data["acorn_consumed"].get(agent_id, 0),
                    f"acorn_consumed_grass_{agent_id}": episode_data["acorn_consumed_grass"].get(agent_id, 0),
                    f"apple_{agent_id}": episode_data["apple"].get(agent_id, 0),
                    f"prey_caught_{agent_id}": episode_data["prey_caught"].get(agent_id, 0),
                }
            )

        result.update(update_dict)
        result.update(counts)
        if self._log_timesteps:
            result.update({"timestep_data":timestep_data})
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
            return (num_episodes is not None and episode_count >= num_episodes) or (
                num_steps is not None and step_count >= num_steps
            )

        episode_count: int = 0
        step_count: int = 0
        with signals.runtime_terminator():
            while not should_terminate(episode_count, step_count):
                episode_start = time.time()
                result = self.run_episode()
                result = {**result, **{"episode_duration": time.time() - episode_start}}
                episode_count += 1
                step_count += int(result["episode_length"])
                # Log the given episode results.
                if self._log_timesteps:
                    self._timestep_logger.write({"timestep_data": result.get("timestep_data", {})})
                    result.pop("timestep_data")
                    self._logger.write(result)

                else:
                    self._logger.write(result)

        return step_count


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)
