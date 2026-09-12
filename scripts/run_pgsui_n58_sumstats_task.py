#!/usr/bin/env python3
"""Run and verify one SNPio summary-statistics task from the N=58 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pgsui_n58_sumstats_tasks.tsv"),
    )
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 digest using bounded memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable JSON status record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read and validate the fixed 58-task summary-statistics manifest."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 58:
        raise ValueError(f"Expected 58 summary-statistics tasks, found {len(rows)}")
    task_ids = [int(row["task_id"]) for row in rows]
    if task_ids != list(range(58)):
        raise ValueError("Summary-statistics task IDs must be ordered 0 through 57")
    if len({row["dataset_id"] for row in rows}) != 58:
        raise ValueError("Summary-statistics manifest contains duplicate datasets")
    return rows


def task_fingerprint(row: dict[str, str]) -> str:
    """Return a stable fingerprint of a manifest row."""
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def installed_version(distribution: str) -> str:
    """Return an installed distribution version or an explicit marker."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def resolve(root: Path, value: str) -> Path:
    """Resolve one manifest path relative to its bundle root."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def output_is_complete(
    path: Path,
    expected_snpio: str,
    expected_ploidy: int | None = None,
) -> bool:
    """Validate reusable outputs and embedded SNPio/ploidy provenance."""
    required = (
        path / "Total_LocusStats.csv",
        path / "Total_Summary.json",
        path / "all_summaries.json",
    )
    if not all(item.is_file() for item in required):
        return False
    try:
        summary = json.loads(required[1].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return summary.get("SNPio_Version") == expected_snpio and (
        expected_ploidy is None or summary.get("Ploidy") == expected_ploidy
    )


def main() -> int:
    """Execute one task in a temporary directory and atomically stage results."""
    args = parse_args()
    root = args.bundle_root.expanduser().resolve()
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = root / manifest
    rows = read_manifest(manifest)
    if not 0 <= args.task_index < len(rows):
        raise IndexError(f"task-index must be between 0 and {len(rows) - 1}")
    row = rows[args.task_index]
    task_id = int(row["task_id"])
    if task_id != args.task_index:
        raise ValueError("Task index differs from manifest task_id")
    n_jobs = int(row["n_jobs"])
    if n_jobs != 1:
        raise ValueError("N=58 summary-statistics tasks require n_jobs=1")
    ploidy = int(row["ploidy"])
    if ploidy != 2:
        raise ValueError("N=58 summary-statistics tasks require diploid VCFs")

    source = resolve(root, row["input_vcf"])
    output = resolve(root, row["output_dir"])
    if not source.is_file():
        raise FileNotFoundError(source)
    observed_digest = sha256_file(source)
    if observed_digest != row["input_sha256"]:
        raise ValueError(f"Input VCF SHA-256 mismatch: {source}")
    current_snpio = installed_version("snpio")
    if current_snpio != row["expected_snpio_version"]:
        raise RuntimeError(
            f"Task requires SNPio {row['expected_snpio_version']}, "
            f"but {current_snpio} is installed"
        )

    status_dir = root / "status"
    success = status_dir / f"sumstats_task_{task_id:03d}.success.json"
    failure = status_dir / f"sumstats_task_{task_id:03d}.failure.json"
    fingerprint = task_fingerprint(row)
    if success.is_file():
        previous = json.loads(success.read_text(encoding="utf-8"))
        if previous.get("task_fingerprint") == fingerprint and output_is_complete(
            output, current_snpio, ploidy
        ):
            print(f"Summary-statistics task {task_id} already passed: {success}")
            return 0
        raise RuntimeError(
            f"Existing task status or outputs failed validation: {success}"
        )
    if output.exists():
        raise FileExistsError(
            f"Output directory exists without a reusable success status: {output}"
        )

    command_prefix = [
        sys.executable,
        str(root / "scripts" / "run_snpiosp.py"),
        "--input",
        str(source),
        "--format",
        "vcf",
        "--n-jobs",
        "1",
        "--ploidy",
        str(ploidy),
    ]
    if args.dry_run:
        print(" ".join((*command_prefix, "--outdir", str(output))))
        return 0

    work_root = root / "work" / "sumstats"
    work_root.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "task_id": task_id,
        "dataset_id": row["dataset_id"],
        "input_vcf": str(source),
        "input_sha256": observed_digest,
        "output_dir": str(output),
        "snpio_version": current_snpio,
        "ploidy": ploidy,
        "task_fingerprint": fingerprint,
        "started_at_utc": utc_now(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    task_environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        task_environment[name] = "1"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=f"task_{task_id:03d}_", dir=work_root
    ) as temporary:
        temporary_output = Path(temporary) / "output"
        command = [*command_prefix, "--outdir", str(temporary_output)]
        metadata["command"] = command
        completed = subprocess.run(command, check=False, env=task_environment)
        metadata["returncode"] = completed.returncode
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["finished_at_utc"] = utc_now()
        if completed.returncode != 0:
            write_json(failure, metadata)
            return completed.returncode
        if not output_is_complete(temporary_output, current_snpio, ploidy):
            metadata["returncode"] = 3
            metadata["error"] = "SNPio outputs failed completeness/version audit"
            write_json(failure, metadata)
            return 3
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temporary_output), output)

    failure.unlink(missing_ok=True)
    write_json(success, metadata)
    print(f"Summary-statistics task {task_id} completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
