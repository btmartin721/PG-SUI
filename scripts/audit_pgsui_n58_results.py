#!/usr/bin/env python3
"""Audit all N=58 PG-SUI results and emit analysis-ready metric tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from _canonical_benchmark_support import (
        MODEL_ORDER,
        audit_task_reports,
        metric_rows,
        read_task_manifest,
        report_classes,
        task_fingerprint,
    )
except ModuleNotFoundError as exc:
    if exc.name != "_canonical_benchmark_support":
        raise
    from pgsui.utils.canonical_benchmark import (
        MODEL_ORDER,
        audit_task_reports,
        metric_rows,
        read_task_manifest,
        report_classes,
        task_fingerprint,
    )

EXPECTED_TASKS = 290


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pgsui_n58_tasks.tsv"),
    )
    return parser.parse_args()


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write dictionaries as a tab-delimited table with stable columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def task_status_row(task: Any, root: Path) -> dict[str, Any]:
    """Validate one task success record against its manifest fingerprint."""
    status_path = root / "status" / f"task_{task.task_id:03d}.success.json"
    base = {
        "task_id": task.task_id,
        "dataset_id": task.dataset_id,
        "strategy": task.strategy,
        "status_path": str(status_path),
    }
    if not status_path.is_file():
        return {**base, "status": "missing_success_status"}
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {**base, "status": "invalid_success_status", "error": str(exc)}

    expected_fingerprint = task_fingerprint(task)
    checks = {
        "task_fingerprint": payload.get("task_fingerprint") == expected_fingerprint,
        "task_id": payload.get("task_id") == task.task_id,
        "dataset_id": payload.get("dataset_id") == task.dataset_id,
        "strategy": payload.get("strategy") == task.strategy,
        "returncode": payload.get("returncode") == 0,
        "pgsui_version": payload.get("pgsui_version") == task.expected_pgsui_version,
        "snpio_version": payload.get("snpio_version") == task.expected_snpio_version,
        "pgsui_source_sha256": payload.get("pgsui_source_sha256")
        == task.expected_pgsui_source_sha256,
        "benchmark_profile": payload.get("benchmark_profile") == task.benchmark_profile,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        **base,
        "status": "ok" if not failed_checks else "status_mismatch",
        "failed_checks": " ".join(failed_checks),
        "expected_task_fingerprint": expected_fingerprint,
        "observed_task_fingerprint": payload.get("task_fingerprint", ""),
        "pgsui_version": payload.get("pgsui_version", ""),
        "snpio_version": payload.get("snpio_version", ""),
        "pgsui_source_sha256": payload.get("pgsui_source_sha256", ""),
        "benchmark_profile": payload.get("benchmark_profile", ""),
        "elapsed_seconds": payload.get("elapsed_seconds", ""),
        "finished_at_utc": payload.get("finished_at_utc", ""),
        "slurm_job_id": payload.get("slurm_job_id", ""),
        "slurm_array_job_id": payload.get("slurm_array_job_id", ""),
    }


def main() -> int:
    """Audit the complete task grid and write tabular outputs."""
    args = parse_args()
    root = args.bundle_root.expanduser().resolve()
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = root / manifest
    tasks = read_task_manifest(manifest)
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError(f"Expected {EXPECTED_TASKS} tasks, found {len(tasks)}")

    task_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for task in tasks:
        status_row = task_status_row(task, root)
        task_rows.append(status_row)
        reports = audit_task_reports(task, root)
        model_rows.extend(reports)
        if status_row["status"] == "ok" and all(
            row["status"] == "ok" for row in reports
        ):
            metrics.extend(metric_rows(task, root))

    analysis_dir = root / "analysis"
    tables_dir = analysis_dir / "tables"
    write_tsv(tables_dir / "n58_task_audit.tsv", task_rows)
    write_tsv(tables_dir / "n58_model_audit.tsv", model_rows)
    write_tsv(tables_dir / "n58_metrics_long.tsv", metrics)

    valid_tasks = sum(row["status"] == "ok" for row in task_rows)
    valid_models = sum(row["status"] == "ok" for row in model_rows)
    expected_models = EXPECTED_TASKS * len(MODEL_ORDER)
    expected_metric_rows = sum(
        len(task.models) * (len(report_classes(task.ploidy)) + 2) for task in tasks
    )
    complete = (
        valid_tasks == EXPECTED_TASKS
        and valid_models == expected_models
        and len(metrics) == expected_metric_rows
    )
    summary = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "complete": complete,
        "manifest": str(manifest),
        "expected_tasks": EXPECTED_TASKS,
        "valid_tasks": valid_tasks,
        "invalid_tasks": EXPECTED_TASKS - valid_tasks,
        "expected_model_reports": expected_models,
        "valid_model_reports": valid_models,
        "invalid_model_reports": expected_models - valid_models,
        "expected_metric_rows": expected_metric_rows,
        "metric_rows": len(metrics),
        "task_status_counts": {
            status: sum(row["status"] == status for row in task_rows)
            for status in sorted({str(row["status"]) for row in task_rows})
        },
        "model_status_counts": {
            status: sum(row["status"] == status for row in model_rows)
            for status in sorted({str(row["status"]) for row in model_rows})
        },
        "pgsui_versions": sorted(
            {str(row["pgsui_version"]) for row in task_rows if row["pgsui_version"]}
        ),
        "snpio_versions": sorted(
            {str(row["snpio_version"]) for row in task_rows if row["snpio_version"]}
        ),
        "pgsui_source_sha256": sorted(
            {
                str(row["pgsui_source_sha256"])
                for row in task_rows
                if row["pgsui_source_sha256"]
            }
        ),
        "benchmark_profiles": sorted(
            {
                str(row["benchmark_profile"])
                for row in task_rows
                if row["benchmark_profile"]
            }
        ),
    }
    summary_path = analysis_dir / "n58_audit_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
