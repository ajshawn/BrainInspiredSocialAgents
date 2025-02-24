"""Default logger."""

from collections.abc import Mapping
import logging
import os
from typing import Any, Callable, Optional

from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import csv
from acme.utils.loggers import filters
from acme.utils.loggers import terminal
from acme.utils.loggers import tf_summary

from marl.utils.loggers.ma_filter import MAFilter
from marl.utils.loggers.pickle_episode_data_logger import PickleEpisodeDataLogger

try:
  import wandb
except ImportError:
  wandb = None


def make_default_logger(
    label: str,
    log_dir: str = "~/marl-jax",
    save_data: bool = True,
    use_tb: bool = True,
    use_wandb: bool = False,
    wandb_config: Mapping[str, Any] = None,
    time_delta: float = 1.0,
    asynchronous: bool = False,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = base.to_numpy,
    steps_key: str = "steps",
) -> base.Logger:
  """Makes a default Acme logger.

    Args:
      label: Name to give to the logger.
      save_data: Whether to persist data.
      time_delta: Time (in seconds) between logging events.
      asynchronous: Whether the write function should block or not.
      print_fn: How to print to terminal (defaults to print).
      serialize_fn: An optional function to apply to the write inputs before
        passing them to the various loggers.
      steps_key: Ignored.

    Returns:
      A logger object that responds to logger.write(some_dict).
    """
  # Remove unused steps_key for now.
  del steps_key
  if not print_fn:
    print_fn = logging.info

    # 1) Terminal
    terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)
    sub_loggers = [terminal_logger]

    # 2) Optionally CSV
    if save_data:
      csv_dir = os.path.join(os.path.expanduser(log_dir), "csv_logs")
      os.makedirs(csv_dir, exist_ok=True)
      csv_file = os.path.join(csv_dir, label + ".csv")
      sub_loggers.append(csv.CSVLogger(directory_or_file=open(csv_file, mode="a")))

    # 3) Optional TF summary
    if use_tb:
      tb_dir = os.path.join(os.path.expanduser(log_dir), "tb_logs")
      os.makedirs(tb_dir, exist_ok=True)
      sub_loggers.append(tf_summary.TFSummaryLogger(logdir=tb_dir, label=label))

    # 4) Optional Weights & Biases
    if use_wandb:
      sub_loggers.append(WandbLogger(label=label, **wandb_config))

    # 5) Aggregate them
    logger = aggregators.Dispatcher(sub_loggers, serialize_fn)
    logger = filters.NoneFilter(logger)  # Filter out None-valued keys
    logger = MAFilter(logger)            # Your custom multi-agent filter

    # 6) Possibly async
    if asynchronous:
      logger = async_logger.AsyncLogger(logger)

    # 7) Add time-based filter
    logger = filters.TimeFilter(logger, time_delta)

    # 8) **Wrap** with our pickling logger
    logger = PickleEpisodeDataLogger(to_logger=logger, log_dir=log_dir, label=label)

    return logger


class WandbLogger(base.Logger):
  """Logging results to weights and biases"""

  def __init__(
      self,
      label: Optional[str] = None,
      steps_key: Optional[str] = None,
      *,
      project: Optional[str] = None,
      entity: Optional[str] = None,
      dir: Optional[str] = None,  # pylint: disable=redefined-builtin
      name: Optional[str] = None,
      group: Optional[str] = None,
      config: Optional[Any] = None,
      **wandb_kwargs,
  ):
    if wandb is None:
      raise ImportError(
        'Logger not supported as `wandb` logger is not installed yet, '
        'install it with `pip install wandb`.'
      )
    self._label = label
    self._iter = 0
    self._steps_key = steps_key
    if wandb.run is None:
      self._run = wandb.init(
        project=project,
        dir=dir,
        entity=entity,
        name=name,
        group=group,
        config=config,
        reinit=True,
        **wandb_kwargs,
      )
    else:
      self._run = wandb.run
    # define default x-axis (for latest wandb versions)
    if steps_key and getattr(self._run, 'define_metric', None):
      prefix = f'{self._label}/*' if self._label else '*'
      self._run.define_metric(prefix, step_metric=f'{self._label}/{self._steps_key}')

@property
def run(self):
  """Return the current wandb run."""
  return self._run

  def write(self, data: base.LoggingData):
    data = base.to_numpy(data)
    if self._steps_key is not None and self._steps_key not in data:
      logging.warning('steps key %s not found. Skip logging.', self._steps_key)
      return
    if self._label:
      stats = {f'{self._label}/{k}': v for k, v in data.items()}
    else:
      stats = data
    self._run.log(stats)
    self._iter += 1

def close(self):
  wandb.finish()
