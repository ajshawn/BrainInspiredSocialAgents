#!/usr/bin/env python3
"""
extract_counter_to_csv.py

Restore your Acme Counter at each saved checkpoint and write out
the episode & step counts alongside the checkpoint timestamps.

Usage:
  python extract_episodes_to_csv.py --exp_dir /path/to/experiment
"""
import argparse
import re
import shutil
import filecmp
from pathlib import Path

import pandas as pd
from acme.utils.counting import Counter
from acme.tf.savers import Checkpointer

def get_ckpt_num(name: str) -> int:
    """Extract the integer N from a checkpoint name 'ckpt-N'."""
    m = re.search(r"ckpt-(\d+)$", name)
    if not m:
        raise ValueError(f"Unexpected checkpoint name: {name}")
    return int(m.group(1))

def parse_checkpoint_manifest(manifest: Path):
    """
    Read a checkpoint file (the TF-style manifest) and return two
    parallel lists: [ 'ckpt-1', 'ckpt-2', … ], [ ts1, ts2, … ].
    """
    names, stamps = [], []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if line.startswith("all_model_checkpoint_paths:"):
            names.append(line.split(":",1)[1].strip().strip('"'))
        elif line.startswith("all_model_checkpoint_timestamps:"):
            stamps.append(float(line.split(":",1)[1].strip()))
    if len(names) != len(stamps):
        raise RuntimeError(
            f"{manifest.name}: found {len(names)} names but {len(stamps)} timestamps"
        )
    return names, stamps


def sanitize_manifest(manifest_path: Path, ckpt_dir: Path):
    """
    Ensure the main 'checkpoint' manifest only references existing checkpoint files.
    If any referenced ckpt-<N> lacks the backing .index or .data shards, rewrite
    the manifest to include only the valid ones and update model_checkpoint_path
    and last_preserved_timestamp accordingly.
    """
    if not manifest_path.exists():
        return
    lines = manifest_path.read_text().splitlines()
    model_ckpt = None
    all_names, all_stamps = [], []
    # parse existing entries
    for line in lines:
        line = line.strip()
        if line.startswith("model_checkpoint_path:"):
            model_ckpt = line.split(":",1)[1].strip().strip('"')
        elif line.startswith("all_model_checkpoint_paths:"):
            all_names.append(line.split(":",1)[1].strip().strip('"'))
        elif line.startswith("all_model_checkpoint_timestamps:"):
            all_stamps.append(float(line.split(":",1)[1].strip()))
    # filter to only names with actual files
    valid_names, valid_stamps = [], []
    for name, stamp in zip(all_names, all_stamps):
        # a checkpoint is valid if we find any .index or .data shard
        idx_file = ckpt_dir / f"{name}.index"
        data_files = list(ckpt_dir.glob(f"{name}.data-*"))
        if idx_file.exists() or data_files:
            valid_names.append(name)
            valid_stamps.append(stamp)
    # if all were valid, nothing to do
    if len(valid_names) == len(all_names):
        return
    if not valid_names:
        # no valid ckpts; do not overwrite
        print(f"Warning: no valid checkpoints found for {manifest_path.name}")
        return
    # choose latest
    latest_name = valid_names[-1]
    latest_stamp = valid_stamps[-1]
    # rebuild manifest
    out = []
    out.append(f'model_checkpoint_path: "{latest_name}"')
    for nm in valid_names:
        out.append(f'all_model_checkpoint_paths: "{nm}"')
    for st in valid_stamps:
        out.append(f'all_model_checkpoint_timestamps: {st}')
    out.append(f'last_preserved_timestamp: {latest_stamp}')
    manifest_path.write_text("\n".join(out))
    print(f"Sanitized manifest '{manifest_path.name}': now points to {latest_name}")


def backup_manifest(manifest: Path):
    """
    Back up a live 'checkpoint' manifest into 'checkpoint_<start>_<end>'
    whenever it differs from existing backup.
    """
    if not manifest.exists() or manifest.name != 'checkpoint':
        return
    try:
        names, _ = parse_checkpoint_manifest(manifest)
    except Exception:
        return
    if not names:
        return
    nums = sorted(get_ckpt_num(n) for n in names)
    start, end = nums[0], nums[-1]
    backup = manifest.parent / f"checkpoint_{start}_{end}"
    if backup.exists() and filecmp.cmp(str(manifest), str(backup), shallow=False):
        return
    shutil.copy2(str(manifest), str(backup))
    print(f"Backed up manifest to {backup.name}")


def merge_fragments(main_manifest: Path, ckpt_dir: Path):
    """
    Merge all checkpoint_* backups into the main 'checkpoint' file:
      • parse each backup and the main manifest
      • union names → latest stamps
      • sort by numeric index
      • sanitize by requiring shards exist
      • rewrite main manifest with merged entries
    """
    # collect all entries from main and backups
    merged = {}
    for mf in ckpt_dir.glob('checkpoint*'):
        if not mf.is_file():
            continue
        try:
            names, stamps = parse_checkpoint_manifest(mf)
        except Exception:
            continue
        for name, ts in zip(names, stamps):
            # prefer newest timestamp
            if name not in merged or ts > merged[name]:
                merged[name] = ts
    if not merged:
        return
    # Check if the main manifest is the same as the merged one, if so, skip
    if main_manifest.exists():
        try:
            main_names, main_stamps = parse_checkpoint_manifest(main_manifest)
        except Exception:
            main_names, main_stamps = [], []
        if len(main_names) == len(merged) and all(
            merged.get(n) == ts for n, ts in zip(main_names, main_stamps)
        ):
            print(f"Main manifest {main_manifest.name} is already up-to-date.")
            return

    # sort by ckpt number
    items = sorted(merged.items(), key=lambda kv: get_ckpt_num(kv[0]))
    # filter out entries missing shards
    valid = []
    for name, ts in items:
        idx = ckpt_dir / f"{name}.index"
        data = list(ckpt_dir.glob(f"{name}.data-*"))
        if idx.exists() or data:
            valid.append((name, ts))
        else:
            print(f"Dropping missing shard {name}@{ts}")
    if not valid:
        return
    # rewrite main manifest
    out = []
    last_name, last_ts = valid[-1]
    out.append(f'model_checkpoint_path: "{last_name}"')
    for nm, _ in valid:
        out.append(f'all_model_checkpoint_paths: "{nm}"')
    for _, st in valid:
        out.append(f'all_model_checkpoint_timestamps: {st}')
    out.append(f'last_preserved_timestamp: {last_ts}')
    main_manifest.write_text("\n".join(out))
    print(f"Merged {len(valid)} entries into {main_manifest.name}")


class CheckpointerWithTarget(Checkpointer):
    """Extend Acme's Checkpointer so we can restore arbitrary ckpt numbers."""
    def restore_checkpoint_number(self, checkpoint_number: int):
    #     # Look up the full path in the manager, this no longer works if the checkpoint file is modified
    #     candidates = [
    #         p for p in self._checkpoint_manager.checkpoints
    #         if p.endswith(f"ckpt-{checkpoint_number}")
    #     ]
    #     if not candidates:
    #         raise ValueError(f"No checkpoint 'ckpt-{checkpoint_number}' found")
        ckpt_dir = Path(self._checkpoint_manager.directory)
        # Find the .index file for this checkpoint
        idx = ckpt_dir / f"ckpt-{checkpoint_number}.index"
        if not idx.exists():
            raise ValueError(f"No shard for ckpt-{checkpoint_number} in {ckpt_dir}")
        # The ckpt_n is the same path without extension
        ckpt_n = str(idx.with_suffix(''))
        # Directly restore from that prefix
        self._checkpoint.restore(ckpt_n)
        return ckpt_n

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--exp_dir",
        type=Path,
        # required=True,
        default=Path(
            "/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__open_2025-06-12_15:49:53.201394/"),
        help="Experiment root (must contain csv_logs/ and checkpoints/)",
    )
    args = p.parse_args()

    exp = args.exp_dir
    counter_dir = exp / 'checkpoints' / 'counter'
    learner_dir = exp / 'checkpoints' / 'learner'
    if not counter_dir.exists():
        raise FileNotFoundError(f"Cannot find counter dir at {counter_dir}")

    # 1) sanitize main manifest, backup, and merge fragments
    sanitize_manifest(counter_dir / 'checkpoint', counter_dir)
    backup_manifest(counter_dir / 'checkpoint')
    merge_fragments(counter_dir / 'checkpoint', counter_dir)
    sanitize_manifest(counter_dir / 'checkpoint', counter_dir)

    sanitizing_learner = False
    if sanitizing_learner:
        sanitize_manifest(learner_dir / 'checkpoint', learner_dir)
        backup_manifest(learner_dir / 'checkpoint')
        merge_fragments(learner_dir / 'checkpoint', learner_dir)
        sanitize_manifest(learner_dir / 'checkpoint', learner_dir)

    # 1) parse the manifest
    ckpt_names, ckpt_stamps = parse_checkpoint_manifest(counter_dir / 'checkpoint')
    if not ckpt_names:
        print("No checkpoints found to process.")
        return
    # 2) build a fresh Counter and Checkpointer pointed at your 'counter' subdir
    counter = Counter()
    cp = CheckpointerWithTarget(
        objects_to_save={"counter": counter},
        directory=str(exp),
        subdirectory="counter",
        add_uid=False,
    )

    # 3) gather rows
    rows = []

    # # TODO: delete this after test
    # for i in range(1, 113):
    #     cp._checkpoint_manager.checkpoints.append(
    #         '/home/mikan/e/GitHub/social-agents-JAX/results/PopArtIMPALA_42_meltingpot_predator_prey_group_radius_0__open_2025-06-12_15:49:53.201394/checkpoints/counter/ckpt-' + str(i)
    #     )

    for name, ts in zip(ckpt_names, ckpt_stamps):
        num = get_ckpt_num(name)
        # restore that exact checkpoint
        try:
            cp.restore_checkpoint_number(num)
        except ValueError as e:
            print(f"Skipping ckpt-{num}: {e}")
            continue
        counts = counter.get_counts()
        rows.append({
            "ckpt":           num,
            "timestamp":      ts,
            "actor_episodes": counts.get("actor_episodes"),
            "actor_steps":    counts.get("actor_steps"),
            "learner_steps":  counts.get("learner_steps"),
            "learner_time_elapsed": counts.get("learner_time_elapsed"),
        })

    # 4) write out
    df = pd.DataFrame(rows)
    out_csv = exp / "csv_logs" / "checkpoint_episodes.csv"
    df.to_csv(out_csv, index=False, float_format="%.3f")
    print(f"Wrote {len(df)} rows to {out_csv}")

if __name__ == "__main__":
    main()
