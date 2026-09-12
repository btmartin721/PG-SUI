#!/usr/bin/env python3
"""Build a checksummed reviewer package from a complete N=58 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from _canonical_benchmark_support import sha256_python_tree
except ModuleNotFoundError as exc:
    if exc.name != "_canonical_benchmark_support":
        raise
    from pgsui.utils.canonical_benchmark import sha256_python_tree

EXPECTED_DATASETS = 58
EXPECTED_TASKS = 290
EXPECTED_MODEL_REPORTS = 1_740
PACKAGE_DIRECTORIES: tuple[tuple[str, str], ...] = (
    ("canonical_vcfs", "canonical_vcfs"),
    ("masked_vcfs", "masked_vcfs"),
    ("masks", "masks"),
    ("splits", "splits"),
    ("inputs", "inputs"),
    ("manifests", "manifests"),
    ("provenance", "provenance"),
    ("results", "results"),
    ("analysis", "analysis"),
    ("logs", "logs"),
    ("status", "status"),
    ("scripts", "scripts"),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-metadata-dir",
        type=Path,
        help="Optional manuscript metadata directory copied verbatim.",
    )
    parser.add_argument(
        "--tar-gz",
        action="store_true",
        help="Also create <output>.tar.gz after verifying the directory package.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 digest using bounded memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_complete_run(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require a complete audit and internally consistent run provenance."""
    audit_path = root / "analysis" / "n58_audit_summary.json"
    sumstats_audit_path = root / "analysis" / "n58_sumstats_audit_summary.json"
    provenance_path = root / "provenance" / "n58_run_manifest.json"
    audit = read_json(audit_path)
    sumstats_audit = read_json(sumstats_audit_path)
    provenance = read_json(provenance_path)
    if not audit.get("complete"):
        raise ValueError(f"N=58 audit is not complete: {audit_path}")
    expected = {
        "dataset_count": EXPECTED_DATASETS,
        "original_vcf_count": EXPECTED_DATASETS,
        "task_count": EXPECTED_TASKS,
        "sumstats_task_count": EXPECTED_DATASETS,
        "expected_model_reports": EXPECTED_MODEL_REPORTS,
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Run provenance is incomplete: {mismatches}")
    ploidy_counts = (
        provenance.get("haploid_dataset_count"),
        provenance.get("diploid_dataset_count"),
        provenance.get("tetraploid_dataset_count"),
    )
    if not all(isinstance(value, int) for value in ploidy_counts):
        raise ValueError("Run provenance is missing integer source-ploidy counts")
    if ploidy_counts != (0, EXPECTED_DATASETS, 0):
        raise ValueError("N=58 provenance must contain only diploid source VCFs")
    if provenance.get("categorically_diploidized_dataset_count") != 0:
        raise ValueError("N=58 provenance must not contain diploidized datasets")
    if provenance.get("excluded_dataset_ids") != ["results131", "results576"]:
        raise ValueError("N=58 provenance must record both non-diploid exclusions")
    if provenance.get("conda_environment") != "pgsui-val":
        raise ValueError("N=58 provenance must record Conda environment pgsui-val")
    if provenance.get("partition") != "shu-hpc-biocpu":
        raise ValueError("N=58 provenance must record the shu-hpc-biocpu partition")
    if provenance.get("worker_assignment") != (
        "largest_processing_time_by_matrix_cell_count"
    ):
        raise ValueError("N=58 provenance must record cell-count load balancing")
    worker_loads = provenance.get("worker_estimated_cell_loads")
    if (
        not isinstance(worker_loads, list)
        or len(worker_loads) != 8
        or not all(isinstance(value, int) and value > 0 for value in worker_loads)
    ):
        raise ValueError("N=58 provenance has invalid worker load estimates")
    if not isinstance(provenance.get("pgsui_git_dirty"), bool):
        raise ValueError("Run provenance is missing the PG-SUI dirty-tree state")
    if provenance["pgsui_git_dirty"]:
        raise ValueError(
            "Reviewer packages require a clean, released PG-SUI source tree"
        )
    completion_tasks = provenance.get("nonrandom_completion_tasks")
    completion_count = provenance.get("nonrandom_completion_task_count")
    if (
        not isinstance(completion_tasks, list)
        or not isinstance(completion_count, int)
        or completion_count != len(completion_tasks)
    ):
        raise ValueError("Run provenance has invalid nonrandom-completion records")
    for record in completion_tasks:
        if (
            not isinstance(record, dict)
            or record.get("strategy") not in {"nonrandom", "nonrandom_weighted"}
            or record.get("mode")
            not in {
                "uniform_node_marginal_tip_clades",
                "branch_length_weighted_tip_clades",
            }
            or not isinstance(record.get("count"), int)
            or record["count"] <= 0
            or not isinstance(record.get("fraction_of_simulated_mask"), float)
            or not 0.0 < record["fraction_of_simulated_mask"] <= 1.0
        ):
            raise ValueError(
                "Run provenance contains an invalid nonrandom-completion task"
            )
    source_digest = str(provenance.get("pgsui_source_sha256", ""))
    source_root = root / "inputs" / "software" / "pgsui"
    if len(source_digest) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in source_digest
    ):
        raise ValueError("Run provenance has an invalid PG-SUI source digest")
    if sha256_python_tree(source_root) != source_digest:
        raise ValueError("Bundled PG-SUI source differs from run provenance")
    if audit.get("valid_tasks") != EXPECTED_TASKS:
        raise ValueError("Audit does not contain 290 valid task records")
    if audit.get("valid_model_reports") != EXPECTED_MODEL_REPORTS:
        raise ValueError("Audit does not contain 1,740 valid model reports")
    if not sumstats_audit.get("complete") or sumstats_audit.get("valid_tasks") != 58:
        raise ValueError("SNPio audit does not contain 58 valid task records")
    if audit.get("pgsui_versions") != [provenance.get("pgsui_version")]:
        raise ValueError("Audited PG-SUI version differs from run provenance")
    if audit.get("snpio_versions") != [provenance.get("snpio_version")]:
        raise ValueError("Audited SNPio version differs from run provenance")
    if audit.get("pgsui_source_sha256") != [provenance.get("pgsui_source_sha256")]:
        raise ValueError("Audited PG-SUI source digest differs from run provenance")
    if sumstats_audit.get("snpio_versions") != [provenance.get("snpio_version")]:
        raise ValueError("Summary-statistics SNPio version differs from provenance")
    required_posthoc = (
        root / "analysis" / "n58_posthoc_manifest.json",
        root / "analysis" / "popgen_stats" / "comparison" / "n58_popgen_manifest.json",
        root / "analysis" / "feature_effects" / "n58_feature_effects_manifest.json",
    )
    missing_posthoc = [path for path in required_posthoc if not path.is_file()]
    if missing_posthoc:
        raise FileNotFoundError(
            f"Required post-hoc manifests are missing: {missing_posthoc}"
        )
    return audit, provenance


def ensure_safe_output(root: Path, output: Path) -> None:
    """Reject existing or recursively nested package destinations."""
    if output.exists():
        raise FileExistsError(f"Reviewer package destination exists: {output}")
    if output == root or root in output.parents:
        raise ValueError("Reviewer package output must be outside the run bundle")


def copy_directory(source: Path, destination: Path) -> None:
    """Copy one required directory without following external symlinks."""
    if not source.is_dir():
        raise FileNotFoundError(f"Required package directory is missing: {source}")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Reviewer package inputs must not be symlinks: {path}")
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def write_readme(path: Path, provenance: dict[str, Any]) -> None:
    """Write a concise package entry point with immutable run identifiers."""
    text = f"""# PG-SUI N=58 reviewer package

This package contains the complete inputs, manifests, execution records,
PG-SUI outputs, and post-hoc analyses for the N=58 manuscript validation.
Both the checksum-verified raw Dryad VCFs and the SNPio-normalized run VCFs
are included.

- PG-SUI: {provenance["pgsui_version"]}
- SNPio: {provenance["snpio_version"]}
- PG-SUI Git revision: {provenance["pgsui_git_revision"]}
- PG-SUI Git tree dirty at bundle creation: {provenance["pgsui_git_dirty"]}
- PG-SUI source SHA-256: {provenance["pgsui_source_sha256"]}
- Profile: {provenance["profile"]}
- Models: ImputeAutoencoder, ImputeVAE, ImputeNLPCA, ImputeUBP,
  ImputeMostFrequent, ImputeRefAllele
- Grid: 58 diploid datasets x 5 simulation strategies x 6 models
- Source ploidy: {provenance["haploid_dataset_count"]} haploid,
  {provenance["diploid_dataset_count"]} diploid, and
  {provenance["tetraploid_dataset_count"]} tetraploid datasets
- Excluded non-diploid datasets: results131 (haploid), results576 (tetraploid)
- Device and parallelism: CPU, one thread per task, eight sequential workers
- Worker assignment: dataset-preserving load balance by matrix cell count
- Conda environment: {provenance["conda_environment"]}
- SLURM partition: {provenance["partition"]}
- Tuning: {provenance["tune_n_trials"]} trials, {provenance["preset"]} preset
- Nonrandom tasks requiring seeded phylogenetic tip-clade completion:
  {provenance["nonrandom_completion_task_count"]}

Start with `analysis/n58_audit_summary.json` and
`analysis/n58_sumstats_audit_summary.json`. Every packaged file
except the checksum table itself is recorded in
`REVIEWER_PACKAGE_SHA256SUMS.tsv`.
"""
    path.write_text(text, encoding="utf-8")


def package_files(output: Path) -> list[Path]:
    """List regular package files except the checksum manifest itself."""
    checksum_path = output / "REVIEWER_PACKAGE_SHA256SUMS.tsv"
    return sorted(
        path for path in output.rglob("*") if path.is_file() and path != checksum_path
    )


def write_checksum_manifest(output: Path, files: list[Path]) -> None:
    """Write and independently verify the complete package checksum manifest."""
    manifest = output / "REVIEWER_PACKAGE_SHA256SUMS.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("relative_path", "size_bytes", "sha256"),
            delimiter="\t",
        )
        writer.writeheader()
        for path in files:
            writer.writerow(
                {
                    "relative_path": path.relative_to(output),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            path = output / row["relative_path"]
            if path.stat().st_size != int(row["size_bytes"]):
                raise AssertionError(f"Packaged file size changed: {path}")
            if sha256_file(path) != row["sha256"]:
                raise AssertionError(f"Packaged file checksum changed: {path}")


def main() -> int:
    """Create, checksum, and optionally archive a complete reviewer package."""
    args = parse_args()
    root = args.bundle_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    ensure_safe_output(root, output)
    audit, provenance = validate_complete_run(root)
    archive_path = Path(f"{output}.tar.gz") if args.tar_gz else None
    if archive_path is not None and archive_path.exists():
        raise FileExistsError(f"Reviewer archive exists: {archive_path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        for source_name, destination_name in PACKAGE_DIRECTORIES:
            copy_directory(root / source_name, staging / destination_name)
        if args.source_metadata_dir is not None:
            metadata = args.source_metadata_dir.expanduser().resolve()
            copy_directory(metadata, staging / "source_metadata")
        source_readme = root / "README.md"
        if source_readme.is_file():
            shutil.copy2(source_readme, staging / "RUN_REPRODUCIBILITY.md")

        write_readme(staging / "README.md", provenance)
        pre_summary_files = package_files(staging)
        summary = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_bundle_root": str(root),
            "package_root": str(output),
            "pgsui_version": provenance["pgsui_version"],
            "snpio_version": provenance["snpio_version"],
            "pgsui_git_revision": provenance["pgsui_git_revision"],
            "pgsui_git_dirty": provenance["pgsui_git_dirty"],
            "pgsui_source_sha256": provenance["pgsui_source_sha256"],
            "profile": provenance["profile"],
            "dataset_count": provenance["dataset_count"],
            "task_count": provenance["task_count"],
            "model_report_count": audit["valid_model_reports"],
            "pre_summary_file_count": len(pre_summary_files),
            "pre_summary_size_bytes": sum(
                path.stat().st_size for path in pre_summary_files
            ),
        }
        (staging / "REVIEWER_PACKAGE_MANIFEST.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        files = package_files(staging)
        write_checksum_manifest(staging, files)
        os.replace(staging, output)

    if archive_path is not None:
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(output, arcname=output.name, recursive=True)

    result = {
        **summary,
        "checksummed_file_count": len(files),
        "archive": "" if archive_path is None else str(archive_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
