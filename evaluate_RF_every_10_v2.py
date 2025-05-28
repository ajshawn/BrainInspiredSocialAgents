#!/usr/bin/env python3
"""
Roll-out every Nth Random-Forest checkpoint by copying it into
a standalone experiment dir, then evaluating from there.

Usage:
  chmod +x rollout_rf_copy_ckpt.py
  ./rollout_rf_copy_ckpt.py \
      --root /home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__random_forest_2025-04-24_17:23:56.924098 \
      --stride 10 \
      --max_workers 4 \
      --out_root ./results
"""
import argparse, os, re, shutil, subprocess, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def find_ckpt_prefixes(learner_dir: Path):
    # grab all ckpt-<n>.index files
    prefixes = []
    for idx in learner_dir.glob("ckpt-*.index"):
        m = re.match(r"ckpt-(\d+)\.index$", idx.name)
        if m:
            prefixes.append(int(m.group(1)))
    return sorted(prefixes)

def prepare_experiment_clone(root: Path, ckpt_num: int, out_root: Path) -> Path:
    """
    Create a clone directory under out_root named
    <orig_name>_ckpt<ckpt_num>, containing only:
      - ckpt-<n>.index
      - ckpt-<n>.data-*-of-*
      - checkpoint   ← metadata pointing at ckpt-<n>
    Returns the Path to that clone.
    """
    base = root.name
    clone = out_root / f"{base}_ckpt{ckpt_num}"
    clone_ckpt = clone / "checkpoints" / "learner"
    if clone_ckpt.exists():
        shutil.rmtree(clone_ckpt)
    clone_ckpt.mkdir(parents=True)

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

def build_cmd(clone_dir: Path, log_root: Path):
    """Build the CLI for a single evaluation job."""
    roles   = "predator,predator,prey,prey,prey,prey"
    # indices = "1,2,5,6,7,11"
    indices = "1,2,7,8,10,11"
    dims    = "256,256,256,256,256,256"
    # logs go under: log_root/mix_RF_ckpt<NUM>/
    exp_log = str(log_root / f"mix_RF_{clone_dir.name.split('_')[-1]}")

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
        # the clone_dir *is* the experiment_dir for checkpoint loading:
        "--cross_checkpoint_paths", ",".join([str(clone_dir)] * 6),
        "--exp_log_dir", exp_log,
    ]
    return cmd

def worker(root, ckpt_num, out_root):
    root = Path(root)
    out_root = Path(out_root)
    clone = prepare_experiment_clone(root, ckpt_num, out_root)
    cmd = build_cmd(clone, out_root)
    proc = subprocess.run(cmd)
    return ckpt_num, proc.returncode

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=False, help="RF experiment directory",
                    default="/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__random_forest_2025-04-24_17:23:56.924098")
    p.add_argument("--stride", type=int, default=10, help="evaluate every K‑th ckpt")
    p.add_argument("--max_workers", type=int, default=os.cpu_count() or 4,
                   help="Parallel jobs")
    p.add_argument("--out_root", default="./results",
                   help="Where to put logs & clones")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    learner = root / "checkpoints" / "learner"
    if not learner.is_dir():
        sys.exit(f"No learner checkpoint folder at {learner}")

    prefixes = find_ckpt_prefixes(learner)
    selected = [n for i,n in enumerate(prefixes) if i % args.stride == 0]
    print(f"Found {len(prefixes)} ckpts, evaluating {len(selected)} "
          f"(every {args.stride}) with {args.max_workers} workers")

    out_root = Path(args.out_root)
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(worker, root, n, out_root): n for n in selected}
        for fut in as_completed(futures):
            num, rc = fut.result()
            print(f"ckpt-{num:>3}  {'✓' if rc==0 else '✗'}")

    # for num in selected:
    # num=1
    # clone = prepare_experiment_clone(root, num, out_root)
    # cmd = build_cmd(clone, out_root)
    # print(f"Running: {' '.join(cmd)}")
    # proc = subprocess.run(cmd)
    # print(f"ckpt-{num:>3}  {'✓' if proc.returncode==0 else '✗'}")
if __name__ == "__main__":
    main()
