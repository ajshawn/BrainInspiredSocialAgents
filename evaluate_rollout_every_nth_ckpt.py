#!/usr/bin/env python3
"""
Roll-out every Nth Random-Forest checkpoint by copying it into
a standalone experiment dir, then evaluating from there.

Additionally, sample 30 diverse combinations of 2 predators and 4 prey.
"""

import argparse, os, re, shutil, subprocess, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import random


def find_ckpt_prefixes(learner_dir: Path):
  # grab all ckpt-<n>.index files
  prefixes = []
  for idx in learner_dir.glob("ckpt-*.index"):
    m = re.match(r"ckpt-(\d+)\.index$", idx.name)
    if m:
      prefixes.append(int(m.group(1)))
  return sorted(prefixes)


def sample_diverse_combinations(pred_idx_range, prey_idx_range, k_samples):
  """
  Generate all combinations of 2 predators and 4 prey,
  then select k_samples combinations maximizing diversity
  via a greedy symmetric-difference heuristic.
  """
  # all possible pairs of predator indices and quadruples of prey indices
  all_combs = [(list(p), list(q))
               for p in itertools.combinations(pred_idx_range, 2)
               for q in itertools.combinations(prey_idx_range, 4)]
  selected = []
  # seed with a random combination
  selected.append(random.choice(all_combs))
  while len(selected) < k_samples and len(selected) < len(all_combs):
    def dist(a, b):
      sa = set(a[0] + a[1])
      sb = set(b[0] + b[1])
      return len(sa.symmetric_difference(sb))
    best = None
    best_min = -1
    for cand in all_combs:
      if cand in selected:
        continue
      # compute min distance to any already selected
      dmin = min(dist(cand, s) for s in selected)
      if dmin > best_min:
        best_min = dmin
        best = cand
    if best is None:
      break
    selected.append(best)
  return selected


def prepare_experiment_clone(root: Path, ckpt_num: int, out_root: Path) -> Path:
  """
  Create a clone directory under out_root named
  <orig_name>_ckpt<ckpt_num>, containing only:
    - ckpt-<n>.index
    - ckpt-<n>.data-*-of-*
    - checkpoint metadata pointing at ckpt-<n>
  Returns the Path to that clone.
  """
  base = root.name
  clone = out_root / f"{base}_ckp{ckpt_num}"
  clone_ckpt = clone / "checkpoints" / "learner"
  if clone_ckpt.exists():
    return clone

  # Fresh clone: copy just the learner checkpoint files
  clone_ckpt = clone / 'checkpoints' / 'learner'
  clone_ckpt.mkdir(parents=True, exist_ok=True)

  learner = root / "checkpoints" / "learner"
  # copy the 3 TF checkpoint files
  for ext in ["index", "data-00000-of-00002", "data-00001-of-00002"]:
    src = learner / f"ckpt-{ckpt_num}.{ext}"
    if not src.exists():
      raise FileNotFoundError(f"Missing {src}")
    shutil.copy2(src, clone_ckpt / f"ckpt-{ckpt_num}.{ext}")

  # write the TF checkpoint metadata
  ckpt_meta = clone_ckpt / "checkpoint"
  ckpt_meta.write_text(
    f'model_checkpoint_path: "ckpt-{ckpt_num}"\n'
    f'all_model_checkpoint_paths: "ckpt-{ckpt_num}"\n'
  )

  return clone


def build_cmd(clone: Path, log_root: Path, combo):
  """
  Build the CLI for a single evaluation job using the given combo.
  combo: (predator_list, prey_list)
  """
  preds, preys = combo
  roles_list = ["predator"] * len(preds) + ["prey"] * len(preys)
  indices_list = preds + preys
  dims_list = [256] * len(indices_list)

  roles = ",".join(roles_list)
  indices = ",".join(str(i) for i in indices_list)
  dims = ",".join(str(d) for d in dims_list)

  exp_id = clone.name.split('_')[-1]
  combo_tag = "_".join(str(i) for i in indices_list)
  exp_log = str(log_root / f"mix_{exp_id}_comb{combo_tag}")

  cmd = [
    "python", "evaluate_cross_trial.py",
    "--async_distributed",
    "--available_gpus", "-1",
    "--num_actors", "16",
    "--algo_name", "PopArtIMPALA",
    "--env_name", "meltingpot",
    "--map_name", "predator_prey__open_debug",
    "--map_layout", "smaller_13x13",
    "--record_video", "true",
    "--agent_roles", roles,
    "--agent_param_indices", indices,
    "--recurrent_dims", dims,
    "--num_episodes", "50",
    # use the clone_dir for each agent
    "--cross_checkpoint_paths", ",".join([str(clone)] * len(indices_list)),
    "--exp_log_dir", exp_log,
  ]
  return cmd


def worker(root, ckpt_num, combo, out_root):
    try:
        root = Path(root)
        out_root = Path(out_root)
        clone = prepare_experiment_clone(root, ckpt_num, out_root)
    except FileNotFoundError as e:
        print(f'SKIPPING ckpt-{ckpt_num}: {e}', file=sys.stderr)
        return ckpt_num, combo, 0
    cmd = build_cmd(clone, out_root, combo)
    proc = subprocess.run(cmd)
    return ckpt_num, combo, proc.returncode


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--experiment_dir", required=False,
                 default="/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__orchard_2025-03-05_18:04:37.638274/",
                 help="RF experiment directory")
  p.add_argument("--stride", type=int, default=10,
                 help="evaluate every K-th ckpt")
  p.add_argument("--max_workers", type=int,
                 default=os.cpu_count() or 4,
                 help="Parallel jobs")
  p.add_argument("--out_root", default="./results",
                 help="Where to put logs & clones")
  args = p.parse_args()

  root = Path(args.experiment_dir).expanduser().resolve()
  learner = root / "checkpoints" / "learner"
  if not learner.is_dir():
    sys.exit(f"No learner checkpoint folder at {learner}")

  # 1) sample 30 diverse combos of 2 predator and 4 prey
  pred_range = list(range(0, 5))
  prey_range = list(range(5, 13))
  combos = sample_diverse_combinations(pred_range, prey_range, 30)

  # 2) find checkpoints every stride
  prefixes = find_ckpt_prefixes(learner)
  selected_ckpts = [n for i, n in enumerate(prefixes) if i % args.stride == 0]
  print(f"Found {len(prefixes)} ckpts, evaluating {len(selected_ckpts)} "
        f"(every {args.stride}) across {len(combos)} combos with {args.max_workers} workers")

  out_root = Path(args.out_root)
  out_root.mkdir(parents=True, exist_ok=True)
  with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
    futures = {}
    for ckpt in selected_ckpts:
      for combo in combos:
        combo_tag = '_'.join(str(i) for i in (combo[0]+combo[1]))
        exp_log_dir = out_root / f'mix_{ckpt}_comb{combo_tag}'
        if exp_log_dir.exists():
          print(f'Skipping ckpt-{ckpt} combo {combo_tag}, results already exist')
          continue
        futures[pool.submit(worker, root, ckpt, combo, out_root)] = (ckpt, combo)
    for fut in as_completed(futures):
      ckpt, combo = futures[fut]
      num, cmb, rc = fut.result()
      print(f"ckpt-{num:>3} combo {cmb}  {'✓' if rc == 0 else '✗'}")

if __name__ == "__main__":
  main()
