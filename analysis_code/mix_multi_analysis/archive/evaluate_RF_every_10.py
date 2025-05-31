#!/usr/bin/env python3
"""
Parallel roll-out of every N-th Random-Forest checkpoint
   (predators 0,4 ; preys 7,8,10,11)
   for TF-style checkpoints in a single directory.

Example
-------
chmod +x rollout_rf_parallel_tfckpt.py
./rollout_rf_parallel_tfckpt.py \
    --root /home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__random_forest_2025-04-24_17:23:56.924098 \
    --stride 10 \
    --max_workers 4
"""
import argparse, os, re, subprocess, sys, fcntl
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def find_ckpt_prefixes(learner_dir: Path) -> list[str]:
    """
    Look for files named 'ckpt-<n>.index' and return
    sorted list of prefixes ['ckpt-1', 'ckpt-2', ...].
    """
    prefixes = set()
    for idx_file in learner_dir.glob("ckpt-*.index"):
        m = re.match(r"(ckpt-\d+)\.index$", idx_file.name)
        if m:
            prefixes.add(m.group(1))
    # sort by numeric suffix
    return sorted(prefixes, key=lambda p: int(p.split("-")[1]))

def patch_checkpoint_file(learner_dir: Path, prefix: str):
    """
    Overwrite the 'checkpoint' file to point to prefix.
    Uses a lock so parallel jobs don't clash.
    """
    ckpt_meta = learner_dir / "checkpoint"
    lockfile  = learner_dir / ".ckptlock"

    text = (
        f'model_checkpoint_path: "{prefix}"\n'
        f'all_model_checkpoint_paths: "{prefix}"\n'
    )

    # write under exclusive lock
    with open(lockfile, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        ckpt_meta.write_text(text)
        fcntl.flock(lf, fcntl.LOCK_UN)

def build_cmd(exp_dir: Path, ckpt_num: int) -> list[str]:
    """
    Build the evaluate_cross_trial.py command as a list.
    """
    roles   = "predator,predator,prey,prey,prey,prey"
    indices = "0,4,7,8,10,11"
    dims    = "256,256,256,256,256,256"
    log_dir = f"./results/mix_RF_ckpt{ckpt_num}"

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
        "--exp_log_dir", log_dir,
        # cross_checkpoint_paths expects the experiment dir six times
        "--cross_checkpoint_paths", ",".join([str(exp_dir)] * 6),
    ]
    return cmd

def run_ckpt(exp_dir: str, prefix: str):
    exp_dir     = Path(exp_dir)
    learner_dir = exp_dir / "checkpoints" / "learner"
    ckpt_num    = int(prefix.split("-")[1])

    # 1) patch the metadata to point at this prefix
    patch_checkpoint_file(learner_dir, prefix)

    # 2) launch the evaluation
    cmd = build_cmd(exp_dir, ckpt_num)
    result = subprocess.run(cmd)
    return prefix, result.returncode

def main():
    ap = argparse.ArgumentParser()

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=False, help="RF experiment directory",
                    default="/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__random_forest_2025-04-24_17:23:56.924098")
    ap.add_argument("--stride", type=int, default=10, help="evaluate every K‑th ckpt")
    ap.add_argument("--max_workers", type=int, default=50)
    args = ap.parse_args()

    exp_dir     = Path(args.root).expanduser().resolve()
    learner_dir = exp_dir / "checkpoints" / "learner"

    prefixes = find_ckpt_prefixes(learner_dir)
    if not prefixes:
        sys.exit(f"No 'ckpt-<n>.index' files in {learner_dir!r}")
    selected = [p for i, p in enumerate(prefixes) if i % args.stride == 0]

    print(f"Found {len(prefixes)} checkpoints, evaluating {len(selected)} "
          f"(every {args.stride}) with {args.max_workers} workers\n")

    # with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
    #     futures = {pool.submit(run_ckpt, str(exp_dir), p): p for p in selected}
    #     for fut in as_completed(futures):
    #         prefix, rc = fut.result()
    #         status = "✓" if rc == 0 else "✗"
    #         print(f"[{prefix}] {status}")

    for i, prefix in enumerate(selected):
        print(f"[{i+1}/{len(selected)}] {prefix}")
        rc = run_ckpt(str(exp_dir), prefix)
        status = "✓" if rc == 0 else "✗"
        print(f"[{prefix}] {status}")

if __name__ == "__main__":
    main()


"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=False, help="RF experiment directory",
                    default="/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_1_meltingpot_predator_prey__random_forest_2025-04-24_17:23:56.924098")
    ap.add_argument("--stride", type=int, default=10, help="evaluate every K‑th ckpt")
    ap.add_argument("--max_workers", type=int, default=50)
    args = ap.parse_args()
"""