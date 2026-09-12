#!/usr/bin/env python3
"""Run one N=58 worker's assigned tasks sequentially."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

WORKLOADS = {
    "pgsui": {
        "worker_manifest": "manifests/pgsui_n58_workers.tsv",
        "task_manifest": "manifests/pgsui_n58_tasks.tsv",
        "runner": "scripts/run_pgsui_canonical_task.py",
    },
    "sumstats": {
        "worker_manifest": "manifests/pgsui_n58_sumstats_workers.tsv",
        "task_manifest": "manifests/pgsui_n58_sumstats_tasks.tsv",
        "runner": "scripts/run_pgsui_n58_sumstats_task.py",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse worker-runner arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workload", choices=tuple(WORKLOADS), required=True)
    parser.add_argument(
        "--work-root",
        type=Path,
        help="Task-private working root for PG-SUI tasks.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_worker_tasks(path: Path, worker_index: int) -> list[int]:
    """Return the ordered task IDs assigned to one worker."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    matching = [row for row in rows if int(row["worker_id"]) == worker_index]
    if len(matching) != 1:
        raise ValueError(
            f"Expected one worker row for index {worker_index}, found {len(matching)}"
        )
    task_ids = [int(value) for value in matching[0]["task_ids"].split()]
    if len(task_ids) != int(matching[0]["task_count"]):
        raise ValueError(f"Worker {worker_index} task count does not match its IDs")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"Worker {worker_index} contains duplicate task IDs")
    return task_ids


def task_command(
    *,
    root: Path,
    workload: str,
    task_id: int,
    work_root: Path | None,
    dry_run: bool,
) -> list[str]:
    """Build the command for one assigned task."""
    config = WORKLOADS[workload]
    command = [
        sys.executable,
        str(root / config["runner"]),
        "--bundle-root",
        str(root),
        "--manifest",
        config["task_manifest"],
        "--task-index",
        str(task_id),
    ]
    if workload == "pgsui" and work_root is not None:
        command.extend(("--work-dir", str(work_root / f"task_{task_id:03d}")))
    if dry_run:
        command.append("--dry-run")
    return command


def main() -> int:
    """Execute all tasks assigned to one worker, stopping at first failure."""
    args = parse_args()
    root = args.bundle_root.expanduser().resolve()
    config = WORKLOADS[args.workload]
    worker_manifest = root / config["worker_manifest"]
    task_ids = read_worker_tasks(worker_manifest, args.worker_index)
    work_root = args.work_root.expanduser().resolve() if args.work_root else None
    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Worker {args.worker_index} will run {len(task_ids)} "
        f"{args.workload} tasks sequentially: {task_ids}"
    )
    for sequence_index, task_id in enumerate(task_ids, start=1):
        print(
            f"Worker {args.worker_index}: task {sequence_index}/{len(task_ids)} "
            f"(manifest task {task_id})"
        )
        command = task_command(
            root=root,
            workload=args.workload,
            task_id=task_id,
            work_root=work_root,
            dry_run=args.dry_run,
        )
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
