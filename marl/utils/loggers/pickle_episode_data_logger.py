# file: marl/utils/loggers/pickle_episode_data_logger.py

import os
import pickle
import numpy as np
import tree  # This is from `dm_tree` or `tree` in the DeepMind repos; ensure it's installed
from acme.utils.loggers import base
from typing import Optional
import jax

class PickleEpisodeDataLogger(base.Logger):
    """
    A logger wrapper that intercepts 'episode_data' in the log record, converts
    any JAX arrays to CPU-based NumPy arrays, pickles them to a file, and
    removes that field before passing the rest to an underlying logger.
    """

    def __init__(self, to_logger: base.Logger, log_dir: str, label: str = "default"):
        self._logger = to_logger
        self._label = label
        # We'll store pickles in <log_dir>/episode_pickles/
        self._pickle_dir = os.path.join(os.path.expanduser(log_dir), "episode_pickles")
        os.makedirs(self._pickle_dir, exist_ok=True)

        self._episode_counter = 0

    def _convert_jax_to_numpy(self, data):
        """Recursively convert any JAX arrays in `data` to CPU-based NumPy arrays."""
        if jax is None:
            # If JAX isn't installed, just return the data as-is
            return data

        def convert_fn(x):
            # If it's a JAX array, block, then convert to numpy
            # (Handles both old jaxlib.xla_extension.Array and new jax.Array)
            if isinstance(x, jax.Array) or "jaxlib.xla_extension.Array" in str(type(x)):
                x.block_until_ready()
                return np.asarray(x)
            return x

        return tree.map_structure(convert_fn, data)

    def write(self, data: base.LoggingData):
        # 1) Pull out the episode_data if present
        episode_data = data.pop("episode_data", None)
        if episode_data is not None:
            # 2) Convert JAX arrays to numpy
            episode_data = self._convert_jax_to_numpy(episode_data)

            # 3) Pickle to a separate file
            self._episode_counter += 1
            pickle_path = os.path.join(
                self._pickle_dir, f"{self._label}_episode_{self._episode_counter}.pkl"
            )
            with open(pickle_path, "wb") as f:
                pickle.dump(episode_data, f)
            print(f"Pickled episode_data (CPU arrays) to {pickle_path}")

        # 4) Pass the rest of the data to the underlying logger
        self._logger.write(data)

    def close(self):
        # Make sure to close the underlying logger
        self._logger.close()
