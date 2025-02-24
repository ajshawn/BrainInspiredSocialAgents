# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration for predator_prey__alley_hunt.

Example video: https://youtu.be/ctVjhn7VYgo

See predator_prey.py for a detailed description applicable to all predator_prey
substrates.sw

In this variant prey must forage for apples in a maze with many dangerous
dead-end corridors where they could easily be trapped by predators.
"""

from ml_collections import config_dict

# from meltingpot.python.configs.substrates import predator_prey as base_config
# from meltingpot.python.utils.substrates import map_helpers
# from meltingpot.python.utils.substrates import specs
from meltingpot.python.configs.substrates import predator_prey_position_visible as base_config
from meltingpot.python.utils.substrates import specs
import numpy as np
build = base_config.build

ASCII_MAP = """
;__________,
!'''a''''a'|
!'a''''''''|
!''''''''''|
!XXa''''<**|
!AAXa'''<**|
!XX''a''<**|
!''''''''''|
!'''a''''''|
!'''a''''a'|
!a'''''a'''|
L~~~~~~~~~~J
"""

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    "*": {"type": "all", "list": ["safe_grass", "spawn_point_prey"]},
    "X": {"type": "all", "list": ["tiled_floor", "spawn_point_predator"]},
    "a": {"type": "all", "list": ["tiled_floor", "apple"]},
    "A": {"type": "all", "list": ["tiled_floor", "floor_acorn"]},
    ";": "nw_wall_corner",
    ",": "ne_wall_corner",
    "J": "se_wall_corner",
    "L": "sw_wall_corner",
    "_": "wall_north",
    "|": "wall_east",
    "~": "wall_south",
    "!": "wall_west",
    "=": "nw_inner_wall_corner",
    "+": "ne_inner_wall_corner",
    "]": "se_inner_wall_corner",
    "[": "sw_inner_wall_corner",
    "'": "tiled_floor",
    "<": "safe_grass_w_edge",
    ">": "safe_grass",
    "/": "fill",
}


def get_config():
  """Default configuration."""
  config = base_config.get_config()

  # Override the map layout settings.
  config.layout = config_dict.ConfigDict()
  config.layout.ascii_map = ASCII_MAP
  config.layout.char_prefab_map = CHAR_PREFAB_MAP

  # The specs of the environment (from a single-agent perspective).
  config.timestep_spec = specs.timestep({
      "RGB": specs.OBSERVATION["RGB"],
      "STAMINA": specs.float64(),
      # Debug only (do not use the following observations in policies).
      "WORLD.RGB": specs.rgb(12*8, 12*8),
      "POSITION": specs.OBSERVATION["POSITION"],
      "ORIENTATION": specs.OBSERVATION["ORIENTATION"],
      "AGENT_INDEX": specs.int32(name="AGENT_INDEX"),
      # "POSITIONS": specs.int32(6, 2, name="POSITIONS"),
  })

  # The roles assigned to each player.
  config.default_player_roles = ("predator",) * 1 + ("prey",) * 1

  return config
