import re, importlib
from typing import Set, Tuple
import os
import glob


def _load_cfg(folder: str, map: str = None):
    substrates_dir = os.path.dirname(importlib.util.find_spec("meltingpot.python.configs.substrates").origin)
    pattern = os.path.join(substrates_dir, "predator_prey__*.py")
    potential_modules = glob.glob(pattern)

    module_name = None
    normalized_folder = folder.lower().replace("_", "")

    # Sort potential modules by length in descending order
    potential_modules.sort(key=lambda path: len(os.path.splitext(os.path.basename(path))[0]), reverse=True)

    for module_path in potential_modules:
        module_base = os.path.splitext(os.path.basename(module_path))[0]
        normalized_module_base = module_base.lower().replace("_", "")

        # First, try to find the entire normalized folder as a substring
        if normalized_folder in normalized_module_base:
            module_name = f"meltingpot.python.configs.substrates.{module_base}"
            break
        else:
            # If not found, try to find a module base that is a substring of the normalized folder
            if normalized_module_base in normalized_folder:
                module_name = f"meltingpot.python.configs.substrates.{module_base}"
                break
            else:
                # As a last resort, look for the 'predator_prey__' prefix followed by as many
                # valid module name characters as possible
                match = re.search(r"(predator_prey__[a-z0-9_]+)", normalized_folder)
                if match:
                    potential_base = match.group(1)
                    normalized_potential_base = potential_base.replace("_", "")
                    if normalized_potential_base == normalized_module_base:
                        module_name = f"meltingpot.python.configs.substrates.{module_base}"
                        break

    if module_name is None:
        module_name = "meltingpot.python.configs.substrates.predator_prey__open"
        print(f"Warning: Could not find a specific module for folder '{folder}'. Using default: '{module_name}'.")

    mod = importlib.import_module(module_name)
    kw = {}
    if map is None:
        if '13x13' in folder:
            kw['smaller_13x13'] = True
        elif '10x10' in folder:
            kw['smaller_10x10'] = True
        else:
            kw['default'] = True
    else:
        kw[map] = True
    return mod.get_config(**kw)

def _tiles_with_prefab(folder: str, target: str, map: str=None) -> Set[Tuple[int, int]]:
  """
  Dynamically load the Meltingpot substrate config matching folder_path
  :param folder:     The folder path containing the substrate configuration.
  :param target:    The target prefab name to search for in the ASCII map.
  :param map:      The specific map size to load (e.g., 'smaller_13x13').
  :return:     A set of (x, y) coordinates where the target prefab is located in the ASCII map.
  """
  cfg = _load_cfg(folder, map)
  amap, cpm = cfg.layout.ascii_map.strip('\n').splitlines(), cfg.layout.char_prefab_map
  out = set()
  for y, row in enumerate(amap):
    for x, ch in enumerate(row):
      entry = cpm.get(ch)
      lst = [entry] if isinstance(entry, str) else entry.get('list', []) if entry else []
      if any(p.startswith(target) for p in lst): out.add((x, y))
  return out


get_apple_tiles = lambda folder, map=None: _tiles_with_prefab(folder, "apple", map)
get_acorn_tiles = lambda folder, map=None: _tiles_with_prefab(folder, "floor_acorn", map)
get_safe_tiles = lambda folder, map=None: _tiles_with_prefab(folder, "safe_grass", map)
