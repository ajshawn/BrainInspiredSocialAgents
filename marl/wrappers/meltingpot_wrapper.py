"""Wraps an Melting Pot RL environment to be used as a dm_env environment."""

from typing import Union

import os
import json
from acme import specs
from acme import types
import dm_env
import dmlab2d
from meltingpot.python.utils.scenarios.scenario import Scenario
from meltingpot.python.utils.substrates.substrate import Substrate
import numpy as np
from PIL import Image

from marl import types as marl_types

USED_OBS_KEYS = {"global", "RGB", "INVENTORY", "READY_TO_SHOOT", "OBJECTS_IN_VIEW", "OBJECTS_IN_VIEW_TENSOR", "POSITION", "ORIENTATION"}

def obs_to_json_dict(data: dm_env.TimeStep.observation) -> dict:
    """Converts a dm_env observation to a dict for JSON serialization."""
    result = {}
    for i, obs in enumerate(data):
      for key, value in obs.items():
        if isinstance(value, np.ndarray):
          value = value.tolist()
          if isinstance(value, bytes):
            value = value.decode("utf-8")
        obs[key] = value
      result[f"agent_{i}"] = obs
    return result
          
class MeltingPotWrapper(dmlab2d.Environment):
  """Environment wrapper for MeltingPot RL environments."""

  # Note: we don't inherit from base.EnvironmentWrapper because that class
  # assumes that the wrapped environment is a dm_env.Environment.

  def __init__(self,
               environment: Union[Substrate, Scenario],
               shared_reward: bool = False,
               reward_scale: float = 1.0,
               log_obs: bool = False,
               log_filename: str = "observations.jsonl",
               log_img_dir: str = "agent_view_images",
               log_interval: int = 1,
               attn_enhance_agent_skip_indices: list[int] = None,
    ):
    self._environment = environment
    self.reward_scale = reward_scale
    self._reset_next_step = True
    self._shared_reward = shared_reward
    self.is_turn_based = False
    self.num_agents = len(self._environment.action_spec())
    self.num_actions = self._environment.action_spec()[0].num_values
    self.agents = list(range(self.num_agents))
    self.obs_spec = [
        self._remove_unwanted_observations(obs_spec)
        for obs_spec in self._environment.observation_spec()
    ]
    self.log_obs = log_obs
    self.log_interval = log_interval
    self.attn_enhance_agent_skip_indices = attn_enhance_agent_skip_indices or []
    self.steps = 0
    
    # Set up observaiton logging
    self.log_filename = log_filename
    self.log_img_dir = log_img_dir
    self.log_item_dir = log_img_dir + "_items"
    if self.log_obs:
      if not os.path.exists(os.path.dirname(self.log_filename)):
        os.makedirs(os.path.dirname(self.log_filename))
      if not os.path.exists(self.log_img_dir):
        os.makedirs(self.log_img_dir)
      if not os.path.exists(self.log_item_dir):
        os.makedirs(self.log_item_dir)
      self.log_file = open(self.log_filename, "a", encoding="utf-8")
      
  def _show_rgb_image(self, obs_dict, step, output_dir):
    for agent_id, agent_obs in obs_dict.items():
        rgb_array = np.array(agent_obs['RGB'], dtype=np.uint8)
        # Convert the NumPy array to a PIL Image and resize to 88x88
        pil_img = Image.fromarray(rgb_array)
        pil_img = pil_img.resize((88, 88), Image.BICUBIC)  # or Image.ANTIALIAS for older PIL versions
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"step_{step}_agent_{agent_id}.png")
        pil_img.save(save_path)
        del agent_obs['RGB']
        if agent_id == 'agent_0':
          # Save the world RGB image for the first agent
          world_rgb_array = np.array(agent_obs['WORLD.RGB'], dtype=np.uint8)
          world_pil_img = Image.fromarray(world_rgb_array)
          world_pil_img = world_pil_img.resize(world_rgb_array.shape[:-1], Image.BICUBIC)
          world_save_path = os.path.join(output_dir, f"step_{step}_world.png")
          world_pil_img.save(world_save_path)
        del agent_obs['WORLD.RGB']
  
  def _show_item_coord(self, obs_dict, step, output_dir):
    for agent_id, agent_obs in obs_dict.items():
      items_array = np.array(agent_obs['OBJECTS_IN_VIEW'], dtype=np.uint8)
      items_array *= 255
      
      if items_array.ndim == 2:
        items_array = np.expand_dims(items_array, axis=0)  # Add a channel dimension if missing
      
      for i in range(items_array.shape[0]):
        pil_img = Image.fromarray(items_array[i])
        pil_img = pil_img.resize((11, 11), Image.BICUBIC)
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"step_{step}_agent_{agent_id}_items_{i}.png")
        pil_img.save(save_path)
      
      del agent_obs['OBJECTS_IN_VIEW']

  def _remove_unwanted_observations(self, observation: marl_types.Observation):
    """Removes unwanted observations from a marl observation."""
    return {
        key: value for key, value in observation.items() if key in USED_OBS_KEYS
    }

  def _mask_items_for_skipped_agents(self, observation: marl_types.Observation):
    """Masks items in the observation for agents that are skipped."""
    for i, obs in enumerate(observation):
      if i in self.attn_enhance_agent_skip_indices:
        obs["OBJECTS_IN_VIEW"] = np.zeros_like(obs["OBJECTS_IN_VIEW"])
    
    return observation

  def _refine_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Refines a dm_env.TimeStep to use dict instead of list for data of multiple-agents."""
    reward = [reward * self.reward_scale for reward in timestep.reward]
    if self._shared_reward:
      reward = [np.mean(reward)] * self.num_agents
    discount = [timestep.discount] * self.num_agents
    observation = [
        self._remove_unwanted_observations(agent_obs)
        for agent_obs in timestep.observation
    ]

    # Mask items for skipped agents
    observation = self._mask_items_for_skipped_agents(observation)
    
    # Log observation
    if self.log_obs:
      obs_dict = obs_to_json_dict(timestep.observation)
      if self.steps % self.log_interval == 0:
        self._show_rgb_image(obs_dict, self.steps, self.log_img_dir)
        self._show_item_coord(obs_dict, self.steps, self.log_item_dir)
        self.log_file.write(json.dumps(obs_dict) + "\n")
      self.steps += 1
    return dm_env.TimeStep(timestep.step_type, reward, discount, observation)

  def reset(self) -> dm_env.TimeStep:
    """Resets the episode."""
    self._reset_next_step = False
    timestep = self._environment.reset()
    timestep = self._refine_timestep(timestep)
    return timestep

  def step(self, actions: types.NestedArray) -> dm_env.TimeStep:
    """Steps the environment."""
    if self._reset_next_step:
      return self.reset()
    # actions = [actions[f"agent_{i}"] for i in range(self.num_agents)]
    timestep = self._environment.step(actions)
    timestep = self._refine_timestep(timestep)
    if timestep.last():
      self._reset_next_step = True
      self._env_done = True
    return timestep

  def env_done(self) -> bool:
    """Check if env is done.
        Returns:
            bool: bool indicating if env is done.
        """
    done = not self.agents or self._env_done
    return done

  def observation_spec(self) -> list[marl_types.Observation]:
    return self.obs_spec

  def action_spec(self,) -> list[specs.DiscreteArray]:
    return self._environment.action_spec()

  def reward_spec(self) -> list[specs.Array]:
    return self._environment.reward_spec()

  def discount_spec(self) -> list[specs.BoundedArray]:
    return [self._environment.discount_spec()] * self.num_agents

  def extras_spec(self) -> list[specs.BoundedArray]:
    """Extra data spec.
        Returns:
            List[specs.BoundedArray]: spec for extra data.
        """
    return list()

  @property
  def environment(self) -> dmlab2d.Environment:
    """Returns the wrapped environment."""
    return self._environment

  def __getattr__(self, name: str):
    """Expose any other attributes of the underlying environment."""
    return getattr(self._environment, name)