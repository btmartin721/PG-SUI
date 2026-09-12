#!/usr/bin/env python3
"""Create the verified 290-task CPU run manifest for the N=58 manuscript."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pgsui.utils.canonical_benchmark import (  # noqa: E402
    MODEL_ORDER,
    SIMULATION_STRATEGIES,
    CanonicalBenchmarkTask,
    python_source_files,
    read_task_manifest,
    sha256_file,
    sha256_python_tree,
)

SOURCE_CATALOG_DATASET_COUNT = 60
EXCLUDED_DATASET_IDS = frozenset({"results131", "results576"})
EXPECTED_DATASET_COUNT = 58
EXPECTED_TASK_COUNT = EXPECTED_DATASET_COUNT * len(SIMULATION_STRATEGIES)
WORKER_COUNT = 8
EXPECTED_PGSUI_VERSION = "1.8.6"
EXPECTED_SNPIO_VERSION = "1.7.4"
TUNE_N_TRIALS = 50
SIM_MAX_TRIES = 100_000
PRESET = "fast"
PROFILE = "n58-manuscript-fast-50"
CONDA_ENVIRONMENT = "pgsui-val"
SLURM_PARTITION = "shu-hpc-biocpu"
PORTABLE_SUPPORT_NAME = "_canonical_benchmark_support.py"
SCRIPT_NAMES = (
    "run_pgsui_canonical_task.py",
    "run_pgsui_n58_worker.py",
    "run_pgsui_n58_sumstats_task.py",
    "pgsui_n58_cpu_workers.slurm",
    "pgsui_n58_sumstats_workers.slurm",
    "submit_pgsui_n58_cpu_workers.zsh",
    "submit_pgsui_n58_sumstats_workers.zsh",
    "audit_pgsui_n58_results.py",
    "audit_pgsui_n58_sumstats.py",
    "analyze_pgsui_n58_results.py",
    "analyze_pgsui_n58_feature_effects.py",
    "run_pgsui_n58_posthoc.zsh",
    "build_pgsui_n58_reviewer_package.py",
    "PGSUI_N58_REPRODUCIBILITY.md",
    "pgsui_n58_dataset_ids.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--vcf-source-manifest", type=Path, required=True)
    parser.add_argument(
        "--scripts-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_rows(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return (
        path.resolve()
        if path.is_absolute()
        else (manifest_path.parent / path).resolve()
    )


def portable_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"Run input is outside the simulation root: {path}") from exc


def stage_shared_input(source: Path, destination: Path) -> Path:
    """Copy a shared immutable input once and reject conflicting content."""
    if destination.is_file():
        if sha256_file(source) != sha256_file(destination):
            raise FileExistsError(
                "Refusing to replace a staged input with different content: "
                f"{destination}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def vcf_suffix(path: Path) -> str:
    """Return the complete supported VCF suffix for a staged source file."""
    return ".vcf.gz" if path.name.endswith(".vcf.gz") else ".vcf"


def discard_stale_vcf_indexes(vcf: Path) -> list[Path]:
    """Remove generated indexes that predate their compressed VCF."""
    removed: list[Path] = []
    for suffix in (".tbi", ".csi"):
        index = Path(f"{vcf}{suffix}")
        if index.is_file() and index.stat().st_mtime < vcf.stat().st_mtime:
            index.unlink()
            removed.append(index)
    return removed


def stage_source_vcfs(
    source_rows: Sequence[Mapping[str, str]],
    simulation_rows: Sequence[Mapping[str, str]],
    simulation_manifest_path: Path,
    simulation_root: Path,
) -> list[dict[str, Any]]:
    """Stage raw VCFs and write portable references to run-canonical VCFs."""
    canonical_by_dataset: dict[str, tuple[Path, str, str]] = {}
    for row in simulation_rows:
        dataset_id = row["dataset_id"]
        canonical = resolve_manifest_path(simulation_manifest_path, row["input_vcf"])
        source_digest = row.get("source_input_sha256", "")
        canonical_digest = row.get("input_sha256", "")
        if len(source_digest) != 64 or len(canonical_digest) != 64:
            raise ValueError(
                f"Simulation checksums are missing or invalid for {dataset_id}"
            )
        record = (canonical, source_digest, canonical_digest)
        existing = canonical_by_dataset.setdefault(dataset_id, record)
        if existing != record:
            raise ValueError(
                f"{dataset_id} strategies do not share one canonical input VCF "
                "and checksum chain"
            )

    staged_rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        row = dict(source_row)
        dataset_id = row["dataset_id"]
        raw_source = Path(row["local_vcf"]).expanduser().resolve()
        canonical, simulation_source_digest, expected_canonical_digest = (
            canonical_by_dataset[dataset_id]
        )
        if not canonical.is_file():
            raise FileNotFoundError(
                f"Run-canonical VCF is missing for {dataset_id}: {canonical}"
            )
        prepared_digest = row.get("benchmark_vcf_sha256", "")
        canonical_digest = sha256_file(canonical)
        if prepared_digest != simulation_source_digest:
            raise ValueError(
                "Simulation source differs from the checksum-verified benchmark "
                f"VCF for {dataset_id}"
            )
        if canonical_digest != expected_canonical_digest:
            raise ValueError(
                f"Run-canonical VCF checksum mismatch for {dataset_id}: {canonical}"
            )

        raw_destination = (
            simulation_root
            / "inputs"
            / "original_vcfs"
            / f"{dataset_id}{vcf_suffix(raw_source)}"
        )
        stage_shared_input(raw_source, raw_destination)
        raw_index = Path(f"{raw_source}.tbi")
        if raw_index.is_file():
            stage_shared_input(raw_index, Path(f"{raw_destination}.tbi"))

        row["prepared_benchmark_vcf_sha256"] = prepared_digest
        row["simulation_source_input_sha256"] = simulation_source_digest
        row["simulation_input_sha256"] = expected_canonical_digest
        row["local_vcf"] = str(
            Path("..") / raw_destination.relative_to(simulation_root)
        )
        row["local_sha256"] = sha256_file(raw_destination)
        row["benchmark_vcf"] = str(Path("..") / canonical.relative_to(simulation_root))
        row["benchmark_vcf_sha256"] = canonical_digest
        staged_rows.append(row)
    return staged_rows


def git_revision(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    revision = completed.stdout.strip()
    if len(revision) != 40:
        raise ValueError(f"Unexpected Git revision: {revision!r}")
    return revision


def git_is_dirty(project_root: Path) -> bool:
    """Return whether the repository has release-relevant local changes.

    The repository-local ``AGENTS.md`` file contains development-agent
    instructions rather than runtime or validation source. It is intentionally
    left untracked and therefore does not make a staged reviewer run dirty.
    """
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    status_lines = completed.stdout.splitlines()
    relevant_lines = [line for line in status_lines if line != "?? AGENTS.md"]
    return bool(relevant_lines)


def stage_python_source_tree(source: Path, destination: Path) -> str:
    """Stage an exact, lightweight snapshot of all PG-SUI Python sources."""
    source_paths = python_source_files(source)
    if not source_paths:
        raise ValueError(f"No Python sources found below {source}")
    for source_path in source_paths:
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    source_digest = sha256_python_tree(source)
    staged_digest = sha256_python_tree(destination)
    if staged_digest != source_digest:
        raise ValueError(
            "Staged PG-SUI source snapshot differs from the working source tree"
        )
    return staged_digest


def validate_source_manifest(path: Path) -> list[dict[str, str]]:
    rows = read_rows(path, delimiter="\t")
    if len(rows) != SOURCE_CATALOG_DATASET_COUNT:
        raise ValueError(
            f"Expected {SOURCE_CATALOG_DATASET_COUNT} VCF source rows, "
            f"found {len(rows)}"
        )
    invalid = [
        row.get("dataset_id", "")
        for row in rows
        if row.get("validation_status") != "verified"
    ]
    if invalid:
        raise ValueError(
            "Every source VCF must pass Dryad checksum and PHYLIP equivalence "
            f"checks; invalid datasets: {invalid}"
        )
    dataset_ids = [row.get("dataset_id", "") for row in rows]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("VCF source manifest contains duplicate dataset IDs")
    dois = [row.get("doi", "") for row in rows]
    if any(not doi for doi in dois) or len(dois) != len(set(dois)):
        raise ValueError("VCF source catalog requires 60 unique Dryad DOIs")
    for row in rows:
        source = Path(row.get("local_vcf", "")).expanduser()
        if not source.is_file():
            raise FileNotFoundError(
                f"Verified source VCF is missing for {row['dataset_id']}: {source}"
            )
        expected = row.get("local_sha256", "")
        if len(expected) != 64 or sha256_file(source) != expected:
            raise ValueError(
                f"Source VCF SHA-256 mismatch for {row['dataset_id']}: {source}"
            )
        benchmark = Path(row.get("benchmark_vcf", "")).expanduser()
        if not benchmark.is_file():
            raise FileNotFoundError(
                f"Verified benchmark VCF is missing for {row['dataset_id']}: "
                f"{benchmark}"
            )
        benchmark_digest = row.get("benchmark_vcf_sha256", "")
        if len(benchmark_digest) != 64 or sha256_file(benchmark) != benchmark_digest:
            raise ValueError(
                f"Benchmark VCF SHA-256 mismatch for {row['dataset_id']}: {benchmark}"
            )
        try:
            ploidy = int(row.get("gt_ploidy", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Source VCF ploidy is missing for {row['dataset_id']}"
            ) from exc
        if ploidy not in {1, 2}:
            raise ValueError(
                "Benchmark-analysis VCF ploidy is unsupported for "
                f"{row['dataset_id']}: {ploidy}"
            )
        source_ploidy = int(row.get("source_gt_ploidy", ploidy))
        if source_ploidy not in {1, 2, 4}:
            raise ValueError(
                f"Raw source VCF ploidy is unsupported for {row['dataset_id']}: "
                f"{source_ploidy}"
            )
    return rows


def validate_simulation_grid(rows: Sequence[Mapping[str, str]]) -> list[str]:
    if len(rows) != EXPECTED_TASK_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TASK_COUNT} simulation rows, found {len(rows)}"
        )
    pairs = {(row["dataset_id"], row["strategy"]) for row in rows}
    if len(pairs) != EXPECTED_TASK_COUNT:
        raise ValueError("Simulation manifest has duplicate dataset/strategy pairs")
    datasets = sorted({row["dataset_id"] for row in rows}, key=dataset_sort_key)
    if len(datasets) != EXPECTED_DATASET_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DATASET_COUNT} datasets, found {len(datasets)}"
        )
    for dataset_id in datasets:
        observed = {row["strategy"] for row in rows if row["dataset_id"] == dataset_id}
        if observed != set(SIMULATION_STRATEGIES):
            raise ValueError(f"{dataset_id} strategy grid differs: {sorted(observed)}")
    incorrect_protocol: list[str] = []
    for row in rows:
        label = f"{row['dataset_id']}/{row['strategy']}"
        try:
            max_tries = int(row.get("sim_max_tries", 0) or 0)
            achieved_rate = float(row["eligible_simulated_missing_rate"])
            requested_rate = float(row["sim_prop"])
            ploidy = int(row["ploidy"])
            n_cells = int(row["n_cells"])
            n_original_missing = int(row["n_original_missing"])
            n_simulated_missing = int(row["n_simulated_missing"])
            n_written_missing = int(row["n_written_missing"])
            n_dropped = int(row["n_dropped_mask_positions"])
            n_test_evaluation = int(row["n_test_evaluation"])
            completion_count = int(row.get("nonrandom_completion_count", 0) or 0)
            completion_mode = row.get("nonrandom_completion_mode", "")
        except (KeyError, TypeError, ValueError):
            incorrect_protocol.append(label)
            continue
        n_eligible = n_cells - n_original_missing
        target = round(requested_rate * n_eligible)
        expected_rate = n_simulated_missing / n_eligible if n_eligible else math.nan
        if (
            max_tries != SIM_MAX_TRIES
            or ploidy != 2
            or not math.isfinite(achieved_rate)
            or not math.isfinite(requested_rate)
            or n_eligible <= 0
            or n_simulated_missing != target
            or n_written_missing != n_original_missing + n_simulated_missing
            or n_dropped != 0
            or n_test_evaluation <= 0
            or completion_count < 0
            or completion_count > n_simulated_missing
            or not math.isclose(achieved_rate, expected_rate, abs_tol=1e-12)
        ):
            incorrect_protocol.append(label)
            continue
        expected_completion_mode = (
            "branch_length_weighted_tip_clades"
            if row["strategy"] == "nonrandom_weighted"
            else "uniform_node_marginal_tip_clades"
        )
        if completion_count > 0:
            if (
                not row["strategy"].startswith("nonrandom")
                or completion_mode != expected_completion_mode
            ):
                incorrect_protocol.append(label)
        elif completion_mode != "none":
            incorrect_protocol.append(label)
    if incorrect_protocol:
        raise ValueError(
            "Simulation rows violate the N=58 exact-mask, max-tries, or "
            "test-evaluation "
            f"protocol: {incorrect_protocol}"
        )
    return datasets


def dataset_sort_key(dataset_id: str) -> tuple[int, str]:
    suffix = dataset_id.removeprefix("results")
    return (int(suffix), dataset_id) if suffix.isdigit() else (10**9, dataset_id)


def task_mapping(task: CanonicalBenchmarkTask) -> dict[str, Any]:
    row = asdict(task)
    row["tune_metrics"] = " ".join(task.tune_metrics)
    row["models"] = " ".join(task.models)
    return row


def build_tasks(
    rows: Sequence[Mapping[str, str]],
    manifest_path: Path,
    simulation_root: Path,
    revision: str,
    source_sha256: str,
) -> list[CanonicalBenchmarkTask]:
    order = {strategy: index for index, strategy in enumerate(SIMULATION_STRATEGIES)}
    sorted_rows = sorted(
        rows,
        key=lambda row: (dataset_sort_key(row["dataset_id"]), order[row["strategy"]]),
    )
    tasks: list[CanonicalBenchmarkTask] = []
    for task_id, row in enumerate(sorted_rows):
        strategy = row["strategy"]
        input_vcf = resolve_manifest_path(manifest_path, row["input_vcf"])
        discard_stale_vcf_indexes(input_vcf)
        mask_npz = resolve_manifest_path(manifest_path, row["mask_npz"])
        split_tsv = resolve_manifest_path(manifest_path, row["split_tsv"])
        evaluation_tsv = resolve_manifest_path(
            manifest_path, row["evaluation_mask_tsv"]
        )
        treefile = (
            resolve_manifest_path(manifest_path, row["treefile"])
            if strategy.startswith("nonrandom")
            else None
        )
        qmatrix = (
            resolve_manifest_path(manifest_path, row["qmatrix"])
            if strategy.startswith("nonrandom")
            else None
        )
        siterates = (
            resolve_manifest_path(manifest_path, row["siterates"])
            if strategy.startswith("nonrandom")
            else None
        )
        required = [input_vcf, mask_npz, split_tsv, evaluation_tsv]
        required.extend(path for path in (treefile, qmatrix, siterates) if path)
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing prepared inputs for {row['dataset_id']}/{strategy}: {missing}"
            )
        if treefile is not None and qmatrix is not None and siterates is not None:
            iqtree_dir = simulation_root / "inputs" / "iqtree"
            treefile = stage_shared_input(
                treefile, iqtree_dir / f"{row['dataset_id']}.treefile"
            )
            qmatrix = stage_shared_input(
                qmatrix, iqtree_dir / f"{row['dataset_id']}.iqtree"
            )
            siterates = stage_shared_input(
                siterates, iqtree_dir / f"{row['dataset_id']}.rate"
            )
        tasks.append(
            CanonicalBenchmarkTask(
                task_id=task_id,
                dataset_id=row["dataset_id"],
                strategy=strategy,
                input_vcf=portable_path(simulation_root, input_vcf),
                evaluation_mask_tsv=portable_path(simulation_root, evaluation_tsv),
                mask_npz=portable_path(simulation_root, mask_npz),
                split_tsv=portable_path(simulation_root, split_tsv),
                output_prefix=(
                    f"results/pgsui/{row['dataset_id']}/{strategy}/"
                    f"{row['dataset_id']}_{strategy}_cpu"
                ),
                seed=int(row["seed"]),
                sim_prop=float(row["sim_prop"]),
                sim_max_tries=int(row["sim_max_tries"]),
                validation_split=float(row["validation_split"]),
                ploidy=int(row["ploidy"]),
                device="cpu",
                n_jobs=1,
                tune_n_trials=TUNE_N_TRIALS,
                tune_metrics=("f1", "mcc", "average_precision"),
                models=MODEL_ORDER,
                preset=PRESET,
                batch_size=128,
                treefile=(
                    "" if treefile is None else portable_path(simulation_root, treefile)
                ),
                qmatrix=(
                    "" if qmatrix is None else portable_path(simulation_root, qmatrix)
                ),
                siterates=(
                    ""
                    if siterates is None
                    else portable_path(simulation_root, siterates)
                ),
                expected_pgsui_version=EXPECTED_PGSUI_VERSION,
                expected_snpio_version=EXPECTED_SNPIO_VERSION,
                expected_pgsui_git_revision=revision,
                expected_pgsui_source_sha256=source_sha256,
                benchmark_profile=PROFILE,
                input_sha256=sha256_file(input_vcf),
                evaluation_mask_sha256=sha256_file(evaluation_tsv),
                mask_sha256=sha256_file(mask_npz),
                split_sha256=sha256_file(split_tsv),
                treefile_sha256="" if treefile is None else sha256_file(treefile),
                qmatrix_sha256="" if qmatrix is None else sha256_file(qmatrix),
                siterates_sha256="" if siterates is None else sha256_file(siterates),
            )
        )
    for task in tasks:
        task.validate_values()
    return tasks


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty task manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_worker_rows(
    task_rows: Sequence[Mapping[str, Any]],
    *,
    worker_count: int = WORKER_COUNT,
    dataset_weights: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Load-balance complete datasets across sequential workers.

    Largest-processing-time assignment uses a reproducible per-dataset weight
    while keeping every strategy for a dataset in one worker. Unit weights are
    used when no workload estimate is supplied.
    """
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    dataset_ids = sorted(
        {str(row["dataset_id"]) for row in task_rows}, key=dataset_sort_key
    )
    if worker_count > len(dataset_ids):
        raise ValueError("worker_count cannot exceed the number of datasets")
    if dataset_weights is None:
        weights = {dataset_id: 1 for dataset_id in dataset_ids}
    else:
        if set(dataset_weights) != set(dataset_ids):
            raise ValueError("dataset_weights must contain every dataset exactly once")
        weights = {
            dataset_id: int(dataset_weights[dataset_id]) for dataset_id in dataset_ids
        }
        if any(weight <= 0 for weight in weights.values()):
            raise ValueError("dataset_weights must be positive integers")

    worker_loads = [0] * worker_count
    worker_datasets: list[list[str]] = [[] for _ in range(worker_count)]
    ordered_datasets = sorted(
        dataset_ids,
        key=lambda dataset_id: (-weights[dataset_id], dataset_sort_key(dataset_id)),
    )
    for dataset_id in ordered_datasets:
        worker_id = min(
            range(worker_count), key=lambda index: (worker_loads[index], index)
        )
        worker_datasets[worker_id].append(dataset_id)
        worker_loads[worker_id] += weights[dataset_id]

    rows: list[dict[str, Any]] = []
    for worker_id in range(worker_count):
        assigned_datasets = worker_datasets[worker_id]
        worker_dataset_set = set(assigned_datasets)
        worker_tasks = [
            row for row in task_rows if str(row["dataset_id"]) in worker_dataset_set
        ]
        rows.append(
            {
                "worker_id": worker_id,
                "dataset_count": len(assigned_datasets),
                "task_count": len(worker_tasks),
                "estimated_cell_load": worker_loads[worker_id],
                "dataset_ids": " ".join(assigned_datasets),
                "task_ids": " ".join(str(row["task_id"]) for row in worker_tasks),
            }
        )
    assigned_task_ids = [
        task_id for row in rows for task_id in str(row["task_ids"]).split()
    ]
    expected_task_ids = [str(row["task_id"]) for row in task_rows]
    if sorted(assigned_task_ids, key=int) != sorted(expected_task_ids, key=int):
        raise AssertionError("Worker assignment does not cover every task exactly once")
    return rows


def build_sumstats_rows(
    simulation_rows: Sequence[Mapping[str, str]],
    simulation_manifest_path: Path,
    simulation_root: Path,
    *,
    source_rows: Sequence[Mapping[str, Any]] | None = None,
    source_manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Build one SNPio summary-statistics task per raw source VCF."""
    raw_by_dataset = (
        {row["dataset_id"]: row for row in source_rows}
        if source_rows is not None
        else {}
    )
    if raw_by_dataset and source_manifest_path is None:
        raise ValueError("source_manifest_path is required with source_rows")
    rows: list[dict[str, Any]] = []
    for task_id, dataset_id in enumerate(
        sorted({row["dataset_id"] for row in simulation_rows}, key=dataset_sort_key)
    ):
        dataset_rows = [
            row for row in simulation_rows if row["dataset_id"] == dataset_id
        ]
        input_paths = {
            resolve_manifest_path(simulation_manifest_path, row["input_vcf"])
            for row in dataset_rows
        }
        if len(input_paths) != 1:
            raise ValueError(
                f"{dataset_id} strategies do not share one canonical input VCF"
            )
        canonical_input = input_paths.pop()
        ploidies = {int(row["ploidy"]) for row in dataset_rows}
        if len(ploidies) != 1:
            raise ValueError(f"{dataset_id} strategies do not share one ploidy")
        analysis_ploidy = ploidies.pop()
        source_row = raw_by_dataset.get(dataset_id)
        if source_row is None:
            input_path = canonical_input
            source_ploidy = analysis_ploidy
        else:
            assert source_manifest_path is not None
            input_path = resolve_manifest_path(
                source_manifest_path, str(source_row["local_vcf"])
            )
            source_ploidy = int(
                source_row.get("source_gt_ploidy", source_row["gt_ploidy"])
            )
        if not input_path.is_file():
            raise FileNotFoundError(f"Summary-statistics VCF is missing: {input_path}")
        rows.append(
            {
                "task_id": task_id,
                "dataset_id": dataset_id,
                "input_vcf": portable_path(simulation_root, input_path),
                "input_sha256": sha256_file(input_path),
                "expected_snpio_version": EXPECTED_SNPIO_VERSION,
                "ploidy": source_ploidy,
                "analysis_ploidy": analysis_ploidy,
                "input_layer": "raw_source" if source_row is not None else "canonical",
                "n_jobs": 1,
                "output_dir": f"analysis/popgen_stats/by_dataset/{dataset_id}",
            }
        )
    if len(rows) != EXPECTED_DATASET_COUNT:
        raise AssertionError("Summary-statistics task grid is incomplete")
    return rows


def write_input_checksums(root: Path) -> int:
    checksum_path = root / "provenance" / "N58_INPUT_SHA256SUMS"
    roots = [
        root / name
        for name in ("canonical_vcfs", "masks", "splits", "inputs", "scripts")
    ]
    roots.extend([root / "manifests", root / "provenance"])
    files = sorted(
        path
        for source_root in roots
        if source_root.exists()
        for path in source_root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.relative_to(root)}\n")
    return len(files)


def main() -> int:
    args = parse_args()
    simulation_root = args.simulation_root.expanduser().resolve()
    scripts_dir = args.scripts_dir.expanduser().resolve()
    source_manifest_path = args.vcf_source_manifest.expanduser().resolve()
    simulation_manifest_path = simulation_root / "manifests" / "simulation_manifest.csv"
    source_catalog_rows = validate_source_manifest(source_manifest_path)
    simulation_rows = read_rows(simulation_manifest_path)
    datasets = validate_simulation_grid(simulation_rows)
    source_catalog_datasets = {row["dataset_id"] for row in source_catalog_rows}
    simulation_datasets = set(datasets)
    excluded_datasets = source_catalog_datasets.difference(simulation_datasets)
    if excluded_datasets != EXCLUDED_DATASET_IDS:
        raise ValueError(
            "N=58 workflow must exclude exactly the haploid and tetraploid "
            f"datasets {sorted(EXCLUDED_DATASET_IDS)}; observed exclusions were "
            f"{sorted(excluded_datasets)}"
        )
    unexpected_simulation_datasets = simulation_datasets.difference(
        source_catalog_datasets
    )
    if unexpected_simulation_datasets:
        raise ValueError(
            "Simulation manifest contains datasets absent from the verified "
            f"source catalog: {sorted(unexpected_simulation_datasets)}"
        )
    source_rows = [
        row for row in source_catalog_rows if row["dataset_id"] in simulation_datasets
    ]
    analysis_ploidies = {
        row["dataset_id"]: int(row["gt_ploidy"]) for row in source_rows
    }
    source_ploidies = {
        row["dataset_id"]: int(row.get("source_gt_ploidy", row["gt_ploidy"]))
        for row in source_rows
    }
    simulation_ploidies = {
        row["dataset_id"]: int(row["ploidy"]) for row in simulation_rows
    }
    if analysis_ploidies != simulation_ploidies:
        raise ValueError("Source VCF and simulation manifests disagree on ploidy")
    if set(source_ploidies.values()) != {2}:
        raise ValueError("Every included N=58 source VCF must be diploid")
    revision = git_revision(scripts_dir.parent)
    source_snapshot_dir = simulation_root / "inputs" / "software" / "pgsui"
    source_sha256 = stage_python_source_tree(
        scripts_dir.parent / "pgsui", source_snapshot_dir
    )
    shutil.copy2(
        scripts_dir.parent / "pyproject.toml",
        simulation_root / "inputs" / "software" / "pyproject.toml",
    )
    tasks = build_tasks(
        simulation_rows,
        simulation_manifest_path,
        simulation_root,
        revision,
        source_sha256,
    )

    task_manifest = simulation_root / "manifests" / "pgsui_n58_tasks.tsv"
    if task_manifest.exists() and not args.force:
        raise FileExistsError(f"Task manifest exists; use --force: {task_manifest}")
    task_rows = [task_mapping(task) for task in tasks]
    dataset_cell_weights: dict[str, int] = {}
    for row in simulation_rows:
        dataset_id = row["dataset_id"]
        n_cells = int(row["n_cells"])
        existing = dataset_cell_weights.setdefault(dataset_id, n_cells)
        if existing != n_cells:
            raise ValueError(
                f"{dataset_id} strategies do not share one matrix cell count"
            )
    write_tsv(task_manifest, task_rows)
    portable_tasks = read_task_manifest(task_manifest)
    if len(portable_tasks) != EXPECTED_TASK_COUNT:
        raise AssertionError("Written task manifest failed its round-trip validation")
    worker_rows = build_worker_rows(task_rows, dataset_weights=dataset_cell_weights)
    write_tsv(simulation_root / "manifests" / "pgsui_n58_workers.tsv", worker_rows)
    bundle_source_manifest = (
        simulation_root / "manifests" / "n58_vcf_source_manifest.tsv"
    )
    staged_source_rows = stage_source_vcfs(
        source_rows,
        simulation_rows,
        simulation_manifest_path,
        simulation_root,
    )
    write_tsv(bundle_source_manifest, staged_source_rows)
    sumstats_manifest = simulation_root / "manifests" / "pgsui_n58_sumstats_tasks.tsv"
    sumstats_rows = build_sumstats_rows(
        simulation_rows,
        simulation_manifest_path,
        simulation_root,
        source_rows=staged_source_rows,
        source_manifest_path=bundle_source_manifest,
    )
    write_tsv(sumstats_manifest, sumstats_rows)
    sumstats_worker_rows = build_worker_rows(
        sumstats_rows, dataset_weights=dataset_cell_weights
    )
    write_tsv(
        simulation_root / "manifests" / "pgsui_n58_sumstats_workers.tsv",
        sumstats_worker_rows,
    )
    bundle_scripts = simulation_root / "scripts"
    bundle_scripts.mkdir(parents=True, exist_ok=True)
    for name in SCRIPT_NAMES:
        source = scripts_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Required N=58 script is missing: {source}")
        shutil.copy2(source, bundle_scripts / name)
    shutil.copy2(
        scripts_dir.parent / "pgsui" / "utils" / "canonical_benchmark.py",
        bundle_scripts / PORTABLE_SUPPORT_NAME,
    )
    for name in ("snpiosp.py", "run_snpiosp.py", "compare_dataset_stats.py"):
        shutil.copy2(scripts_dir.parent / "snpiosp" / name, bundle_scripts / name)
    shutil.copy2(
        scripts_dir / "PGSUI_N58_REPRODUCIBILITY.md",
        simulation_root / "README.md",
    )

    for relative in ("results/pgsui", "logs", "status", "work", "provenance"):
        (simulation_root / relative).mkdir(parents=True, exist_ok=True)
    completion_tasks = [
        {
            "dataset_id": row["dataset_id"],
            "strategy": row["strategy"],
            "count": int(row["nonrandom_completion_count"]),
            "fraction_of_simulated_mask": (
                int(row["nonrandom_completion_count"]) / int(row["n_simulated_missing"])
            ),
            "mode": row["nonrandom_completion_mode"],
        }
        for row in simulation_rows
        if int(row.get("nonrandom_completion_count", 0) or 0) > 0
    ]
    provenance = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "profile": PROFILE,
        "pgsui_version": EXPECTED_PGSUI_VERSION,
        "snpio_version": EXPECTED_SNPIO_VERSION,
        "pgsui_git_revision": revision,
        "pgsui_git_dirty": git_is_dirty(scripts_dir.parent),
        "pgsui_source_sha256": source_sha256,
        "pgsui_source_snapshot": "inputs/software/pgsui",
        "device": "cpu",
        "n_jobs": 1,
        "tune_n_trials": TUNE_N_TRIALS,
        "sim_max_tries": SIM_MAX_TRIES,
        "simulated_mask_target_rule": (
            "round(sim_prop * (n_cells - n_original_missing))"
        ),
        "nonrandom_completion_task_count": len(completion_tasks),
        "nonrandom_completion_tasks": completion_tasks,
        "preset": PRESET,
        "conda_environment": CONDA_ENVIRONMENT,
        "partition": SLURM_PARTITION,
        "worker_count": WORKER_COUNT,
        "worker_assignment": "largest_processing_time_by_matrix_cell_count",
        "worker_estimated_cell_loads": [
            int(row["estimated_cell_load"]) for row in worker_rows
        ],
        "dataset_count": len(datasets),
        "excluded_dataset_count": len(EXCLUDED_DATASET_IDS),
        "excluded_dataset_ids": sorted(EXCLUDED_DATASET_IDS, key=dataset_sort_key),
        "exclusion_reason": "non-diploid source genotype ploidy",
        "strategy_count": len(SIMULATION_STRATEGIES),
        "task_count": len(tasks),
        "sumstats_task_count": EXPECTED_DATASET_COUNT,
        "model_count": len(MODEL_ORDER),
        "models": list(MODEL_ORDER),
        "expected_model_reports": len(tasks) * len(MODEL_ORDER),
        "input_format": "vcf",
        "genotype_encoding_source": "SNPio GenotypeEncoder.genotypes_012",
        "simulation_rate_denominator": "genotypes observed before simulation",
        "source_vcf_manifest": "manifests/n58_vcf_source_manifest.tsv",
        "simulation_manifest": "manifests/simulation_manifest.csv",
        "task_manifest": "manifests/pgsui_n58_tasks.tsv",
        "worker_manifest": "manifests/pgsui_n58_workers.tsv",
        "sumstats_task_manifest": "manifests/pgsui_n58_sumstats_tasks.tsv",
        "sumstats_worker_manifest": "manifests/pgsui_n58_sumstats_workers.tsv",
        "dryad_doi_count": len({row["doi"] for row in source_rows}),
        "original_vcf_count": len(source_rows),
        "haploid_dataset_count": sum(value == 1 for value in source_ploidies.values()),
        "diploid_dataset_count": sum(value == 2 for value in source_ploidies.values()),
        "tetraploid_dataset_count": sum(
            value == 4 for value in source_ploidies.values()
        ),
        "categorically_diploidized_dataset_count": sum(
            source_ploidies[dataset_id] == 4 and analysis_ploidies[dataset_id] == 2
            for dataset_id in source_ploidies
        ),
    }
    provenance_path = simulation_root / "provenance" / "n58_run_manifest.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance["input_checksum_count"] = write_input_checksums(simulation_root)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_input_checksums(simulation_root)
    print(
        f"Prepared {len(tasks)} CPU tasks for {len(datasets)} diploid datasets "
        f"across {WORKER_COUNT} sequential workers at {simulation_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
