#!/usr/bin/env python3
"""Run and verify one manifest row from the canonical PG-SUI benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pgsui.utils.canonical_benchmark import audit_task_reports, read_task_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pgsui_gpu_tasks.tsv"),
    )
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--pgsui-executable", default="pg-sui")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def installed_pgsui_version() -> str:
    try:
        return version("pg-sui")
    except PackageNotFoundError:
        return "not-installed"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    bundle_root = args.bundle_root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = bundle_root / manifest_path
    tasks = read_task_manifest(manifest_path)
    if not 0 <= args.task_index < len(tasks):
        raise IndexError(
            f"task-index {args.task_index} is outside [0, {len(tasks) - 1}]"
        )
    task = tasks[args.task_index]
    if task.task_id != args.task_index:
        raise ValueError(
            f"Manifest task_id={task.task_id} does not match index={args.task_index}"
        )

    missing = [
        path for path in task.required_input_paths(bundle_root) if not path.is_file()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required task inputs are missing:\n{formatted}")

    output_directory = task.output_directory(bundle_root)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    status_dir = bundle_root / "status"
    success_path = status_dir / f"task_{task.task_id:02d}.success.json"
    failure_path = status_dir / f"task_{task.task_id:02d}.failure.json"

    if success_path.is_file() and not args.force:
        print(f"Task {task.task_id} already passed: {success_path}")
        return 0

    command = task.command(bundle_root, executable=args.pgsui_executable)
    print(task.command_text(bundle_root, executable=args.pgsui_executable))
    if args.dry_run:
        return 0

    installed_version = installed_pgsui_version()
    if installed_version != task.expected_pgsui_version:
        raise RuntimeError(
            "Canonical task requires PG-SUI "
            f"{task.expected_pgsui_version}, but {installed_version} is installed."
        )

    metadata: dict[str, Any] = {
        "task_id": task.task_id,
        "dataset_id": task.dataset_id,
        "strategy": task.strategy,
        "command": command,
        "started_at_utc": utc_now(),
        "hostname": platform.node(),
        "pgsui_version": installed_version,
        "python_version": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    completed = subprocess.run(command, check=False)
    metadata["finished_at_utc"] = utc_now()
    metadata["returncode"] = completed.returncode
    if completed.returncode != 0:
        write_json(failure_path, metadata)
        return completed.returncode

    audit_rows = audit_task_reports(task, bundle_root)
    failures = [row for row in audit_rows if row["status"] != "ok"]
    metadata["support_audit"] = audit_rows
    if failures:
        metadata["returncode"] = 3
        metadata["error"] = "PG-SUI completed, but canonical support audit failed"
        write_json(failure_path, metadata)
        return 3

    failure_path.unlink(missing_ok=True)
    write_json(success_path, metadata)
    print(f"Task {task.task_id} completed and passed the mask-support audit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
