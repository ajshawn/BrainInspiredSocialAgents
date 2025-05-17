# This file is to define the helper functions used in the analysis code,
import re
import numpy as np

def parse_agent_roles(base_name: str):
  # find all '<model>_pred_<idx>' and '<model>_prey_<idx>' in order
  agent_specs = re.findall(r'([A-Za-z0-9]+_pre[a-z]_\d+)', base_name)
  role, source = {}, {}
  for i, spec in enumerate(agent_specs):
    role[i] = "predator" if "pred" in spec else "prey"
    source[i] = spec
  return role, source

