#!/usr/bin/env python3
"""Audit canonical PG-SUI results and summarize genotype-class performance."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pgsui.utils.canonical_benchmark import (
    GENOTYPE_CLASSES,
    audit_task_reports,
    metric_rows,
    read_task_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pgsui_gpu_tasks.tsv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write partial summaries instead of failing on missing/mismatched reports.",
    )
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def summarize_alternate_genotypes(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create one row per task/model emphasizing HET and ALT performance."""
    selected = metrics[metrics["class"].isin(("HET", "ALT", "macro avg"))]
    wide = selected.pivot_table(
        index=[
            "task_id",
            "dataset_id",
            "strategy",
            "model",
            "model_family",
            "backend",
        ],
        columns="class",
        values=["f1", "average_precision", "recall", "support"],
        aggfunc="first",
    )
    wide.columns = [
        f"{metric}_{label.lower().replace(' ', '_')}" for metric, label in wide.columns
    ]
    wide = wide.reset_index()
    wide["alternate_genotype_macro_f1"] = wide[["f1_het", "f1_alt"]].mean(axis=1)
    wide["alternate_genotype_macro_average_precision"] = wide[
        ["average_precision_het", "average_precision_alt"]
    ].mean(axis=1)
    return wide


def task_runtime_rows(bundle_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((bundle_root / "status").glob("task_*.success.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        started = datetime.fromisoformat(data["started_at_utc"])
        finished = datetime.fromisoformat(data["finished_at_utc"])
        rows.append(
            {
                "task_id": data["task_id"],
                "dataset_id": data["dataset_id"],
                "strategy": data["strategy"],
                "runtime_seconds": (finished - started).total_seconds(),
                "hostname": data.get("hostname"),
                "slurm_job_id": data.get("slurm_job_id"),
                "slurm_array_job_id": data.get("slurm_array_job_id"),
                "slurm_array_task_id": data.get("slurm_array_task_id"),
                "pgsui_version": data.get("pgsui_version"),
                "status_path": str(path),
            }
        )
    return rows


def create_plots(
    metrics: pd.DataFrame, alternate: pd.DataFrame, plots_dir: Path
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="ticks", context="talk")

    class_metrics = metrics[metrics["class"].isin(GENOTYPE_CLASSES)].copy()
    fig, ax = plt.subplots(figsize=(18, 9))
    sns.boxplot(
        data=class_metrics,
        x="model",
        y="f1",
        hue="class",
        hue_order=GENOTYPE_CLASSES,
        ax=ax,
    )
    ax.set(
        xlabel="PG-SUI model",
        ylabel="F1 score",
        title="Test-mask genotype-class performance",
    )
    ax.tick_params(axis="x", rotation=25)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "pgsui_genotype_class_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(18, 9))
    sns.boxplot(
        data=alternate,
        x="model",
        y="alternate_genotype_macro_f1",
        hue="strategy",
        ax=ax,
    )
    ax.set(
        xlabel="PG-SUI model",
        ylabel="Mean HET/ALT F1 score",
        title="Alternate-genotype performance by missingness strategy",
    )
    ax.tick_params(axis="x", rotation=25)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(
        plots_dir / "pgsui_alternate_genotype_f1_by_strategy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    bundle_root = args.bundle_root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = bundle_root / manifest_path
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = bundle_root / output_dir
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"

    tasks = read_task_manifest(manifest_path)
    audit_rows = [
        row for task in tasks for row in audit_task_reports(task, bundle_root)
    ]
    audit = pd.DataFrame(audit_rows)
    write_csv(audit, tables_dir / "pgsui_test_mask_support_audit.csv")

    failures = audit[audit["status"] != "ok"]
    if not failures.empty and not args.allow_incomplete:
        print(
            f"Canonical support audit failed for {len(failures)} model reports. "
            f"See {tables_dir / 'pgsui_test_mask_support_audit.csv'}.",
            file=sys.stderr,
        )
        return 2

    metrics = pd.DataFrame(
        row for task in tasks for row in metric_rows(task, bundle_root)
    )
    write_csv(metrics, tables_dir / "pgsui_classification_metrics_long.csv")

    if metrics.empty:
        print(
            "No PG-SUI classification reports were available for analysis.",
            file=sys.stderr,
        )
        return 2

    alternate = summarize_alternate_genotypes(metrics)
    write_csv(alternate, tables_dir / "pgsui_alternate_genotype_summary.csv")

    runtime = pd.DataFrame(task_runtime_rows(bundle_root))
    write_csv(runtime, tables_dir / "pgsui_task_runtime_summary.csv")
    create_plots(metrics, alternate, plots_dir)

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "pgsui_version": package_version("pg-sui"),
        "snpio_version": package_version("snpio"),
        "n_tasks_expected": len(tasks),
        "n_model_reports_expected": len(tasks) * 6,
        "n_model_reports_ok": int((audit["status"] == "ok").sum()),
        "n_model_reports_failed": int((audit["status"] != "ok").sum()),
        "manifest": str(manifest_path),
        "support_audit": str(tables_dir / "pgsui_test_mask_support_audit.csv"),
        "alternate_genotype_summary": str(
            tables_dir / "pgsui_alternate_genotype_summary.csv"
        ),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Validated {manifest['n_model_reports_ok']} model reports across "
        f"{len(tasks)} canonical tasks."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
