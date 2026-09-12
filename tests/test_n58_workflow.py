"""Tests for N=58 manifest, status, and reviewer-package safeguards."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pgsui.utils.canonical_benchmark import (
    SIMULATION_STRATEGIES,
    CanonicalBenchmarkTask,
    sha256_python_tree,
    task_fingerprint,
)
from scripts.audit_pgsui_n58_results import task_status_row
from scripts.audit_pgsui_n58_sumstats import sumstats_status_row
from scripts.build_pgsui_n58_reviewer_package import (
    package_files,
    validate_complete_run,
    write_checksum_manifest,
)
from scripts.build_pgsui_n58_run import (
    build_sumstats_rows,
    build_worker_rows,
    discard_stale_vcf_indexes,
    git_is_dirty,
    sha256_file,
    stage_shared_input,
    stage_source_vcfs,
    validate_simulation_grid,
    validate_source_manifest,
)
from scripts.run_pgsui_n58_sumstats_task import output_is_complete
from scripts.run_pgsui_n58_sumstats_task import (
    task_fingerprint as sumstats_fingerprint,
)
from scripts.run_pgsui_n58_worker import read_worker_tasks, task_command


def test_git_is_dirty_ignores_only_untracked_agent_instructions(
    monkeypatch,
) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="?? AGENTS.md\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert not git_is_dirty(Path("."))

    def fake_dirty_run(*args, **kwargs):
        return SimpleNamespace(stdout="?? AGENTS.md\n M pgsui/example.py\n")

    monkeypatch.setattr(subprocess, "run", fake_dirty_run)
    assert git_is_dirty(Path("."))


def test_stage_shared_input_reuses_only_identical_content(tmp_path) -> None:
    source = tmp_path / "source.rate"
    destination = tmp_path / "bundle" / "results2.rate"
    source.write_text("Site Rate\n1 1.0\n", encoding="utf-8")
    assert stage_shared_input(source, destination) == destination
    assert stage_shared_input(source, destination) == destination
    source.write_text("Site Rate\n1 2.0\n", encoding="utf-8")
    try:
        stage_shared_input(source, destination)
    except FileExistsError as exc:
        assert "different content" in str(exc)
    else:
        raise AssertionError("Conflicting staged input was not rejected")


def test_discard_stale_vcf_indexes_keeps_only_current_sidecars(tmp_path) -> None:
    vcf = tmp_path / "input.vcf.gz"
    stale = tmp_path / "input.vcf.gz.tbi"
    current = tmp_path / "input.vcf.gz.csi"
    stale.write_text("stale\n", encoding="utf-8")
    vcf.write_text("vcf\n", encoding="utf-8")
    current.write_text("current\n", encoding="utf-8")

    removed = discard_stale_vcf_indexes(vcf)

    assert removed == [stale]
    assert not stale.exists()
    assert current.is_file()


def test_stage_source_vcfs_preserves_raw_and_uses_run_canonical_paths(
    tmp_path,
) -> None:
    root = tmp_path / "bundle"
    manifest = root / "manifests" / "simulation_manifest.csv"
    manifest.parent.mkdir(parents=True)
    raw = tmp_path / "downloads" / "results1.vcf.gz"
    raw.parent.mkdir()
    raw.write_text("raw\n", encoding="utf-8")
    canonical = root / "canonical_vcfs" / "results1.vcf"
    canonical.parent.mkdir()
    canonical.write_text("canonical\n", encoding="utf-8")
    prepared = tmp_path / "prepared.vcf"
    prepared.write_text("prepared\n", encoding="utf-8")
    source_rows = [
        {
            "dataset_id": "results1",
            "local_vcf": str(raw),
            "local_sha256": sha256_file(raw),
            "benchmark_vcf": str(prepared),
            "benchmark_vcf_sha256": sha256_file(prepared),
        }
    ]
    simulation_rows = [
        {
            "dataset_id": "results1",
            "input_vcf": "../canonical_vcfs/results1.vcf",
            "source_input_sha256": sha256_file(prepared),
            "input_sha256": sha256_file(canonical),
        }
    ]

    rows = stage_source_vcfs(source_rows, simulation_rows, manifest, root)

    assert (root / "inputs" / "original_vcfs" / "results1.vcf.gz").is_file()
    assert rows[0]["local_vcf"] == "../inputs/original_vcfs/results1.vcf.gz"
    assert rows[0]["benchmark_vcf"] == "../canonical_vcfs/results1.vcf"
    assert rows[0]["prepared_benchmark_vcf_sha256"] == sha256_file(prepared)
    assert rows[0]["simulation_source_input_sha256"] == sha256_file(prepared)
    assert rows[0]["simulation_input_sha256"] == sha256_file(canonical)
    assert rows[0]["benchmark_vcf_sha256"] == sha256_file(canonical)


def test_stage_source_vcfs_rejects_broken_checksum_chain(tmp_path) -> None:
    root = tmp_path / "bundle"
    manifest = root / "manifests" / "simulation_manifest.csv"
    manifest.parent.mkdir(parents=True)
    raw = tmp_path / "raw.vcf"
    prepared = tmp_path / "prepared.vcf"
    canonical = root / "canonical.vcf.gz"
    raw.write_text("raw\n", encoding="utf-8")
    prepared.write_text("prepared\n", encoding="utf-8")
    canonical.write_text("canonical\n", encoding="utf-8")
    source_rows = [
        {
            "dataset_id": "results1",
            "local_vcf": str(raw),
            "benchmark_vcf_sha256": sha256_file(prepared),
        }
    ]
    simulation_rows = [
        {
            "dataset_id": "results1",
            "input_vcf": "../canonical.vcf.gz",
            "source_input_sha256": "0" * 64,
            "input_sha256": sha256_file(canonical),
        }
    ]

    try:
        stage_source_vcfs(source_rows, simulation_rows, manifest, root)
    except ValueError as exc:
        assert "checksum-verified benchmark" in str(exc)
    else:
        raise AssertionError("Broken source checksum chain was not rejected")
    assert not (root / "inputs" / "original_vcfs" / "results1.vcf").exists()


def test_build_sumstats_rows_uses_one_canonical_vcf_per_dataset(tmp_path) -> None:
    root = tmp_path / "bundle"
    manifest = root / "manifests" / "simulation_manifest.csv"
    manifest.parent.mkdir(parents=True)
    simulation_rows = []
    for index in range(58):
        vcf = root / "canonical_vcfs" / f"results{index + 1}.vcf"
        vcf.parent.mkdir(exist_ok=True)
        vcf.write_text(f"results{index + 1}\n", encoding="utf-8")
        simulation_rows.append(
            {
                "dataset_id": f"results{index + 1}",
                "input_vcf": f"../canonical_vcfs/results{index + 1}.vcf",
                "ploidy": "2",
            }
        )
    rows = build_sumstats_rows(simulation_rows, manifest, root)
    assert len(rows) == 58
    assert rows[0]["n_jobs"] == 1
    assert rows[0]["expected_snpio_version"] == "1.7.4"
    assert rows[0]["input_vcf"] == "canonical_vcfs/results1.vcf"


def test_build_sumstats_rows_uses_raw_source_vcf_when_available(tmp_path) -> None:
    root = tmp_path / "bundle"
    simulation_manifest = root / "manifests" / "simulation_manifest.csv"
    source_manifest = root / "manifests" / "sources.tsv"
    simulation_manifest.parent.mkdir(parents=True)
    simulation_rows = []
    source_rows = []
    for index in range(58):
        dataset_id = f"results{index + 1}"
        canonical = root / "canonical_vcfs" / f"{dataset_id}.vcf"
        raw = root / "inputs" / "original_vcfs" / f"{dataset_id}.vcf"
        canonical.parent.mkdir(exist_ok=True)
        raw.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text(f"canonical {dataset_id}\n", encoding="utf-8")
        raw.write_text(f"raw {dataset_id}\n", encoding="utf-8")
        simulation_rows.append(
            {
                "dataset_id": dataset_id,
                "input_vcf": f"../canonical_vcfs/{dataset_id}.vcf",
                "ploidy": "2",
            }
        )
        source_rows.append(
            {
                "dataset_id": dataset_id,
                "local_vcf": f"../inputs/original_vcfs/{dataset_id}.vcf",
                "gt_ploidy": "2",
                "source_gt_ploidy": "2",
            }
        )

    rows = build_sumstats_rows(
        simulation_rows,
        simulation_manifest,
        root,
        source_rows=source_rows,
        source_manifest_path=source_manifest,
    )

    assert rows[-1]["input_vcf"] == "inputs/original_vcfs/results58.vcf"
    assert rows[-1]["ploidy"] == 2
    assert rows[-1]["analysis_ploidy"] == 2
    assert rows[-1]["input_layer"] == "raw_source"


def test_source_manifest_rechecks_verified_vcf_hashes(tmp_path) -> None:
    path = tmp_path / "sources.tsv"
    rows = []
    for index in range(60):
        vcf = tmp_path / f"results{index + 1}.vcf"
        vcf.write_text(f"results{index + 1}\n", encoding="utf-8")
        rows.append(
            {
                "dataset_id": f"results{index + 1}",
                "doi": f"10.5061/dryad.example{index + 1}",
                "local_vcf": str(vcf),
                "local_sha256": sha256_file(vcf),
                "benchmark_vcf": str(vcf),
                "benchmark_vcf_sha256": sha256_file(vcf),
                "validation_status": "verified",
                "gt_ploidy": "2",
                "source_gt_ploidy": "4" if index == 59 else "2",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    assert len(validate_source_manifest(path)) == 60
    rows[-1]["local_sha256"] = "0" * 64
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    try:
        validate_source_manifest(path)
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("Tampered VCF source hash was not rejected")


def test_simulation_grid_requires_exact_mask_target_and_search_limit() -> None:
    rows = [
        {
            "dataset_id": f"results{dataset_index}",
            "strategy": strategy,
            "sim_prop": "0.3",
            "sim_max_tries": "100000",
            "eligible_simulated_missing_rate": "0.3",
            "ploidy": "2",
            "n_cells": "100",
            "n_original_missing": "0",
            "n_simulated_missing": "30",
            "n_written_missing": "30",
            "n_dropped_mask_positions": "0",
            "n_test_evaluation": "10",
            "nonrandom_completion_count": "0",
            "nonrandom_completion_mode": "none",
        }
        for dataset_index in range(1, 59)
        for strategy in SIMULATION_STRATEGIES
    ]

    assert len(validate_simulation_grid(rows)) == 58
    rows[0]["eligible_simulated_missing_rate"] = "nan"
    try:
        validate_simulation_grid(rows)
    except ValueError as exc:
        assert "results1/random" in str(exc)
    else:
        raise AssertionError("A non-finite achieved simulation rate was accepted")

    rows[0]["eligible_simulated_missing_rate"] = "0.3"
    rows[0]["n_simulated_missing"] = "29"
    try:
        validate_simulation_grid(rows)
    except ValueError as exc:
        assert "results1/random" in str(exc)
    else:
        raise AssertionError("An inexact simulated-mask count was accepted")


def test_worker_rows_keep_complete_datasets_in_eight_balanced_workers() -> None:
    task_rows = [
        {
            "task_id": task_id,
            "dataset_id": f"results{dataset_index}",
            "strategy": strategy,
        }
        for task_id, (dataset_index, strategy) in enumerate(
            (dataset_index, strategy)
            for dataset_index in range(1, 59)
            for strategy in SIMULATION_STRATEGIES
        )
    ]

    weights = {f"results{index}": index for index in range(1, 59)}
    workers = build_worker_rows(task_rows, dataset_weights=weights)

    assert len(workers) == 8
    assert sorted(int(row["dataset_count"]) for row in workers) == [
        7,
        7,
        7,
        7,
        7,
        7,
        8,
        8,
    ]
    assert sorted(int(row["task_count"]) for row in workers) == [
        35,
        35,
        35,
        35,
        35,
        35,
        40,
        40,
    ]
    dataset_workers: dict[str, set[int]] = {}
    for worker in workers:
        worker_id = int(worker["worker_id"])
        for dataset_id in str(worker["dataset_ids"]).split():
            dataset_workers.setdefault(dataset_id, set()).add(worker_id)
    assert len(dataset_workers) == 58
    assert all(len(worker_ids) == 1 for worker_ids in dataset_workers.values())
    loads = [int(worker["estimated_cell_load"]) for worker in workers]
    assert max(loads) - min(loads) <= max(weights.values())


def test_worker_rows_reject_incomplete_workload_weights() -> None:
    task_rows = [
        {"task_id": 0, "dataset_id": "results1", "strategy": "random"},
        {"task_id": 1, "dataset_id": "results2", "strategy": "random"},
    ]

    with pytest.raises(ValueError, match="every dataset exactly once"):
        build_worker_rows(
            task_rows,
            worker_count=2,
            dataset_weights={"results1": 10},
        )


def test_worker_runner_reads_one_assignment_and_builds_task_private_commands(
    tmp_path,
) -> None:
    worker_manifest = tmp_path / "workers.tsv"
    worker_manifest.write_text(
        "worker_id\tdataset_count\ttask_count\tdataset_ids\ttask_ids\n"
        "0\t1\t2\tresults2\t0 1\n",
        encoding="utf-8",
    )

    assert read_worker_tasks(worker_manifest, 0) == [0, 1]
    root = tmp_path / "bundle"
    work_root = tmp_path / "scratch"
    command = task_command(
        root=root,
        workload="pgsui",
        task_id=1,
        work_root=work_root,
        dry_run=True,
    )

    assert command == [
        sys.executable,
        str(root / "scripts" / "run_pgsui_canonical_task.py"),
        "--bundle-root",
        str(root),
        "--manifest",
        "manifests/pgsui_n58_tasks.tsv",
        "--task-index",
        "1",
        "--work-dir",
        str(work_root / "task_001"),
        "--dry-run",
    ]


def test_task_status_requires_fingerprint_and_current_versions(tmp_path) -> None:
    task = CanonicalBenchmarkTask(
        task_id=0,
        dataset_id="results2",
        strategy="random",
        input_vcf="canonical_vcfs/results2.vcf",
        evaluation_mask_tsv="masks/results2.tsv",
        mask_npz="masks/results2.npz",
        split_tsv="splits/results2.tsv",
        output_prefix="results/results2",
        tune_n_trials=50,
        preset="fast",
        benchmark_profile="n58-manuscript-fast-50",
        expected_pgsui_source_sha256="a" * 64,
    )
    status = tmp_path / "status" / "task_000.success.json"
    status.parent.mkdir()
    status.write_text(
        json.dumps(
            {
                "task_id": 0,
                "dataset_id": "results2",
                "strategy": "random",
                "returncode": 0,
                "pgsui_version": "1.8.6",
                "snpio_version": "1.7.4",
                "pgsui_source_sha256": "a" * 64,
                "benchmark_profile": "n58-manuscript-fast-50",
                "task_fingerprint": task_fingerprint(task),
            }
        ),
        encoding="utf-8",
    )
    assert task_status_row(task, tmp_path)["status"] == "ok"


def test_sumstats_output_requires_embedded_snpio_version(tmp_path) -> None:
    for name in ("Total_LocusStats.csv", "all_summaries.json"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    (tmp_path / "Total_Summary.json").write_text(
        '{"SNPio_Version": "1.7.4"}\n', encoding="utf-8"
    )
    assert output_is_complete(tmp_path, "1.7.4")
    assert not output_is_complete(tmp_path, "1.7.3")


def test_reviewer_validation_and_checksum_manifest(tmp_path) -> None:
    root = tmp_path / "run"
    (root / "analysis").mkdir(parents=True)
    (root / "provenance").mkdir()
    source = root / "inputs" / "software" / "pgsui"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    source_digest = sha256_python_tree(source)
    (root / "analysis" / "n58_audit_summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "valid_tasks": 290,
                "valid_model_reports": 1740,
                "pgsui_versions": ["1.8.6"],
                "snpio_versions": ["1.7.4"],
                "pgsui_source_sha256": [source_digest],
            }
        ),
        encoding="utf-8",
    )
    (root / "analysis" / "n58_sumstats_audit_summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "valid_tasks": 58,
                "snpio_versions": ["1.7.4"],
            }
        ),
        encoding="utf-8",
    )
    for relative in (
        "analysis/n58_posthoc_manifest.json",
        "analysis/popgen_stats/comparison/n58_popgen_manifest.json",
        "analysis/feature_effects/n58_feature_effects_manifest.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (root / "provenance" / "n58_run_manifest.json").write_text(
        json.dumps(
            {
                "dataset_count": 58,
                "original_vcf_count": 58,
                "task_count": 290,
                "sumstats_task_count": 58,
                "haploid_dataset_count": 0,
                "diploid_dataset_count": 58,
                "tetraploid_dataset_count": 0,
                "categorically_diploidized_dataset_count": 0,
                "excluded_dataset_ids": ["results131", "results576"],
                "conda_environment": "pgsui-val",
                "partition": "shu-hpc-biocpu",
                "worker_assignment": ("largest_processing_time_by_matrix_cell_count"),
                "worker_estimated_cell_loads": [100] * 8,
                "expected_model_reports": 1740,
                "pgsui_version": "1.8.6",
                "snpio_version": "1.7.4",
                "pgsui_source_sha256": source_digest,
                "pgsui_git_dirty": False,
                "pgsui_git_revision": "b" * 40,
                "nonrandom_completion_task_count": 1,
                "nonrandom_completion_tasks": [
                    {
                        "dataset_id": "results105",
                        "strategy": "nonrandom_weighted",
                        "count": 1,
                        "fraction_of_simulated_mask": 0.1,
                        "mode": "branch_length_weighted_tip_clades",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _, provenance = validate_complete_run(root)
    assert provenance["pgsui_version"] == "1.8.6"

    provenance_path = root / "provenance" / "n58_run_manifest.json"
    dirty_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    dirty_provenance["pgsui_git_dirty"] = True
    provenance_path.write_text(json.dumps(dirty_provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="clean, released"):
        validate_complete_run(root)

    package = tmp_path / "package"
    package.mkdir()
    (package / "artifact.txt").write_text("verified\n", encoding="utf-8")
    files = package_files(package)
    write_checksum_manifest(package, files)
    with (package / "REVIEWER_PACKAGE_SHA256SUMS.tsv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows[0]["relative_path"] == "artifact.txt"
    assert len(rows[0]["sha256"]) == 64


def test_sumstats_audit_requires_matching_status_input_and_outputs(tmp_path) -> None:
    source = tmp_path / "canonical_vcfs" / "results2.vcf"
    source.parent.mkdir()
    source.write_text("input\n", encoding="utf-8")
    output = tmp_path / "analysis" / "popgen_stats" / "results2"
    output.mkdir(parents=True)
    (output / "Total_LocusStats.csv").write_text("Locus\n", encoding="utf-8")
    (output / "Total_Summary.json").write_text(
        '{"Ploidy": 2, "SNPio_Version": "1.7.4"}\n', encoding="utf-8"
    )
    (output / "all_summaries.json").write_text("{}\n", encoding="utf-8")
    row = {
        "task_id": "0",
        "dataset_id": "results2",
        "input_vcf": "canonical_vcfs/results2.vcf",
        "input_sha256": sha256_file(source),
        "expected_snpio_version": "1.7.4",
        "ploidy": "2",
        "n_jobs": "1",
        "output_dir": "analysis/popgen_stats/results2",
    }
    status = tmp_path / "status" / "sumstats_task_000.success.json"
    status.parent.mkdir()
    status.write_text(
        json.dumps(
            {
                "task_id": 0,
                "dataset_id": "results2",
                "returncode": 0,
                "snpio_version": "1.7.4",
                "ploidy": 2,
                "task_fingerprint": sumstats_fingerprint(row),
            }
        ),
        encoding="utf-8",
    )

    assert sumstats_status_row(row, tmp_path)["status"] == "ok"
    source.write_text("changed\n", encoding="utf-8")
    observed = sumstats_status_row(row, tmp_path)
    assert observed["status"] == "status_mismatch"
    assert "input_sha256" in observed["failed_checks"]
