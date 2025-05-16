import re, importlib
from typing import Set, Tuple


def _load_cfg(folder: str):
  m = re.search(r'(predator_prey__[^_/]+)', folder)
  mod = importlib.import_module(
    f"meltingpot.python.configs.substrates.{m.group(1) if m else 'predator_prey__open'}")
  kw = {'smaller_13x13': True} if '13x13' in folder else \
    {'smaller_10x10': True} if '10x10' in folder else {}
  return mod.get_config(**kw)


def _tiles_with_prefab(folder: str, prefix: str) -> Set[Tuple[int, int]]:
  cfg = _load_cfg(folder)
  amap, cpm = cfg.layout.ascii_map.strip('\n').splitlines(), cfg.layout.char_prefab_map
  out = set()
  for y, row in enumerate(amap):
    for x, ch in enumerate(row):
      entry = cpm.get(ch)
      lst = [entry] if isinstance(entry, str) else entry.get('list', []) if entry else []
      if any(p.startswith(prefix) for p in lst): out.add((x, y))
  return out


get_apple_tiles = lambda folder: _tiles_with_prefab(folder, "apple")
get_acorn_tiles = lambda folder: _tiles_with_prefab(folder, "floor_acorn")
get_safe_tiles = lambda folder: _tiles_with_prefab(folder, "safe_grass")
