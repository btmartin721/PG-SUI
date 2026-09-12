#!/usr/bin/env python3
"""Audit all 58 SNPio summary-statistics tasks for the N=58 workflow."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from run_pgsui_n58_sumstats_task import (
        output_is_complete,
        read_manifest,
        resolve,
        sha256_file,
        task_fingerprint,
    )
except ModuleNotFoundError as exc:
    if exc.name != "run_pgsui_n58_sumstats_task":
        raise
    from scripts.run_pgsui_n58_sumstats_task import (
        output_is_complete,
        read_manifest,
        resolve,
        sha256_file,
        task_fingerprint,
    )

EXPECTED_TASKS = 58


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pgsui_n58_sumstats_tasks.tsv"),
    )
    return parser.parse_args()


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write stable audit rows as a tab-delimited table."""
    if not rows:
        raise ValueError("Cannot write an empty summary-statistics audit")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sumstats_status_row(row: dict[str, str], root: Path) -> dict[str, Any]:
    """Validate one SNPio status, input, and output against its manifest row."""
    task_id = int(row["task_id"])
    status_path = root / "status" / f"sumstats_task_{task_id:03d}.success.json"
    source = resolve(root, row["input_vcf"])
    output = resolve(root, row["output_dir"])
    base = {
        "task_id": task_id,
        "dataset_id": row["dataset_id"],
        "status_path": str(status_path),
        "output_dir": str(output),
    }
    if not status_path.is_file():
        return {**base, "status": "missing_success_status"}
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {**base, "status": "invalid_success_status", "error": str(exc)}
    checks = {
        "task_fingerprint": payload.get("task_fingerprint") == task_fingerprint(row),
        "task_id": payload.get("task_id") == task_id,
        "dataset_id": payload.get("dataset_id") == row["dataset_id"],
        "returncode": payload.get("returncode") == 0,
        "snpio_version": payload.get("snpio_version") == row["expected_snpio_version"],
        "ploidy": payload.get("ploidy") == int(row["ploidy"]),
        "input_exists": source.is_file(),
        "input_sha256": source.is_file() and sha256_file(source) == row["input_sha256"],
        "output_complete": output_is_complete(
            output,
            row["expected_snpio_version"],
            int(row["ploidy"]),
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        **base,
        "status": "ok" if not failed else "status_mismatch",
        "failed_checks": " ".join(failed),
        "snpio_version": payload.get("snpio_version", ""),
        "ploidy": payload.get("ploidy", ""),
        "task_fingerprint": payload.get("task_fingerprint", ""),
        "elapsed_seconds": payload.get("elapsed_seconds", ""),
        "finished_at_utc": payload.get("finished_at_utc", ""),
        "slurm_job_id": payload.get("slurm_job_id", ""),
        "slurm_array_job_id": payload.get("slurm_array_job_id", ""),
    }


def main() -> int:
    """Audit the complete summary-statistics grid and write results."""
    args = parse_args()
    root = args.bundle_root.expanduser().resolve()
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = root / manifest
    rows = read_manifest(manifest)
    audited = [sumstats_status_row(row, root) for row in rows]
    valid = sum(row["status"] == "ok" for row in audited)
    summary = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "complete": valid == EXPECTED_TASKS,
        "manifest": str(manifest),
        "expected_tasks": EXPECTED_TASKS,
        "valid_tasks": valid,
        "invalid_tasks": EXPECTED_TASKS - valid,
        "status_counts": {
            status: sum(row["status"] == status for row in audited)
            for status in sorted({str(row["status"]) for row in audited})
        },
        "snpio_versions": sorted(
            {str(row["snpio_version"]) for row in audited if row.get("snpio_version")}
        ),
    }
    analysis = root / "analysis"
    write_tsv(analysis / "tables" / "n58_sumstats_audit.tsv", audited)
    (analysis / "n58_sumstats_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
