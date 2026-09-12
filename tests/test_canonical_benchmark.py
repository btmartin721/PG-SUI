from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pgsui.utils.canonical_benchmark import (
    MODEL_ORDER,
    CanonicalBenchmarkTask,
    audit_task_reports,
    expected_support,
    read_task_manifest,
    sha256_python_tree,
)


def make_task(strategy: str = "random") -> CanonicalBenchmarkTask:
    treefile = (
        "inputs/tree/example.treefile" if strategy.startswith("nonrandom") else ""
    )
    qmatrix = "inputs/tree/example.iqtree" if strategy.startswith("nonrandom") else ""
    siterates = "inputs/tree/example.rate" if strategy.startswith("nonrandom") else ""
    return CanonicalBenchmarkTask(
        task_id=0,
        dataset_id="example",
        strategy=strategy,
        input_vcf="inputs/example.vcf",
        evaluation_mask_tsv="masks/example.evaluation_mask.tsv",
        mask_npz="masks/example.mask.npz",
        split_tsv="splits/example.split.tsv",
        output_prefix="results/pgsui/example/random/example_random_cpu",
        treefile=treefile,
        qmatrix=qmatrix,
        siterates=siterates,
    )


def write_evaluation_mask(path: Path, *, ploidy: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_index", "locus_index", "truth_class"),
            delimiter="\t",
        )
        writer.writeheader()
        classes = (
            ("REF", "REF", "ALT", "ALT")
            if ploidy == 1
            else (
                "REF",
                "HET",
                "ALT",
                "ALT",
            )
        )
        writer.writerows(
            {
                "sample_index": index // 2,
                "locus_index": index % 2,
                "truth_class": genotype_class,
            }
            for index, genotype_class in enumerate(classes)
        )


def write_report(path: Path, alt_support: int = 2, *, ploidy: int = 2) -> None:
    support = (
        {"REF": 2, "ALT": alt_support}
        if ploidy == 1
        else {"REF": 1, "HET": 1, "ALT": alt_support}
    )
    report = {
        label: {
            "precision": 0.5,
            "recall": 0.5,
            "f1-score": 0.5,
            "support": count,
            "average-precision": 0.5,
            "jaccard": 0.5,
        }
        for label, count in support.items()
    }
    report["macro avg"] = {
        "precision": 0.5,
        "recall": 0.5,
        "f1-score": 0.5,
        "support": sum(support.values()),
        "average-precision": 0.5,
        "jaccard": 0.5,
    }
    report["weighted avg"] = dict(report["macro avg"])
    report["mcc"] = 0.25
    report["accuracy"] = 0.5
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")


def write_mask_artifacts(
    task: CanonicalBenchmarkTask,
    root: Path,
    model: str | None = None,
    *,
    change_coordinate: bool = False,
) -> None:
    test_idx = np.array([0, 1], dtype=np.int64)
    evaluation_mask = np.array([[True, True], [True, True], [False, False]], dtype=bool)
    canonical_path = root / task.mask_npz
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        canonical_path,
        evaluation_mask=evaluation_mask,
        test_idx=test_idx,
    )
    if model is None:
        return
    observed = evaluation_mask[test_idx].copy()
    if change_coordinate:
        observed[0, 0] = False
    artifact_path = task.evaluation_artifact_path(root, model)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        evaluation_mask_test=observed,
        test_idx=test_idx,
        full_shape=np.asarray(evaluation_mask.shape, dtype=np.int64),
        seed=np.asarray(task.seed, dtype=np.int64),
        strategy=np.asarray(task.strategy),
        validation_split=np.asarray(task.validation_split),
    )


@pytest.mark.parametrize("strategy", ("nonrandom", "nonrandom_weighted"))
def test_task_command_contains_canonical_settings(
    tmp_path: Path, strategy: str
) -> None:
    task = make_task(strategy)

    command = task.command(tmp_path)

    assert command[0] == "pg-sui"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--ploidy") + 1] == "2"
    assert "--verbose" in command
    assert "--disable-plotting" in command
    assert command[command.index("--tune-n-trials") + 1] == "100"
    assert command[command.index("--n-jobs") + 1] == "1"
    assert command[command.index("--set") + 1] == "train.validation_split=0.3"
    metric_start = command.index("--tune-metrics") + 1
    model_start = command.index("--models")
    assert tuple(command[metric_start:model_start]) == (
        "f1",
        "mcc",
        "average_precision",
    )
    expected_treefile = tmp_path / "inputs/tree/example.treefile"
    expected_iqtree = tmp_path / "inputs/tree/example.iqtree"
    expected_siterates = tmp_path / "inputs/tree/example.rate"
    assert command[command.index("--treefile") + 1] == str(expected_treefile)
    assert command[command.index("--qmatrix") + 1] == str(expected_iqtree)
    assert command[command.index("--siterates") + 1] == str(expected_siterates)


def test_task_mapping_defaults_to_cpu() -> None:
    task = CanonicalBenchmarkTask.from_mapping(
        {
            "task_id": "0",
            "dataset_id": "example",
            "strategy": "random",
            "input_vcf": "inputs/example.vcf",
            "evaluation_mask_tsv": "masks/example.evaluation.tsv",
            "mask_npz": "masks/example.npz",
            "split_tsv": "splits/example.tsv",
            "output_prefix": "results/pgsui/example_cpu",
        }
    )

    assert task.device == "cpu"
    assert task.ploidy == 2


def test_haploid_report_audit_uses_ref_alt_support(tmp_path: Path) -> None:
    task = replace(make_task(), ploidy=1)
    write_evaluation_mask(tmp_path / task.evaluation_mask_tsv, ploidy=1)
    write_mask_artifacts(task, tmp_path)
    for model in MODEL_ORDER:
        write_report(task.report_path(tmp_path, model), ploidy=1)
        write_mask_artifacts(task, tmp_path, model)

    rows = audit_task_reports(task, tmp_path)

    assert all(row["status"] == "ok" for row in rows)
    assert all(row["expected_het_support"] == 0 for row in rows)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_task_validation_accepts_supported_device(device: str) -> None:
    task = make_task()
    replace(task, device=device).validate_values()


def test_task_validation_rejects_unknown_device() -> None:
    task = make_task()
    with pytest.raises(ValueError, match="device='cpu' or device='cuda'"):
        replace(task, device="mps").validate_values()


def test_random_task_omits_tree_arguments(tmp_path: Path) -> None:
    command = make_task("random").command(tmp_path)
    assert "--treefile" not in command
    assert "--qmatrix" not in command
    assert "--siterates" not in command


def test_task_command_propagates_simulation_search_limit(tmp_path: Path) -> None:
    command = replace(make_task("nonrandom"), sim_max_tries=100_000).command(tmp_path)

    settings = [
        command[index + 1] for index, value in enumerate(command) if value == "--set"
    ]
    assert "sim.sim_kwargs={'max_tries': 100000}" in settings


def test_task_uses_family_specific_report_names(tmp_path: Path) -> None:
    task = make_task()

    assert task.report_path(tmp_path, "ImputeVAE").name == "zygosity_report.json"
    assert (
        task.report_path(tmp_path, "ImputeRefAllele").name
        == "classification_report_zygosity.json"
    )


def test_task_command_accepts_private_staged_input(tmp_path: Path) -> None:
    task = make_task()
    staged = tmp_path / "scratch" / "example.vcf"

    command = task.command(tmp_path, input_path=staged)

    assert command[command.index("--input") + 1] == str(staged.resolve())


def test_expected_support_counts_ref_het_alt(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.tsv"
    write_evaluation_mask(path)

    assert expected_support(path) == {"REF": 1, "HET": 1, "ALT": 2, "TOTAL": 4}


def test_report_audit_requires_exact_class_support(tmp_path: Path) -> None:
    task = make_task()
    write_evaluation_mask(tmp_path / task.evaluation_mask_tsv)
    write_mask_artifacts(task, tmp_path)
    for model in MODEL_ORDER:
        write_report(task.report_path(tmp_path, model))
        write_mask_artifacts(task, tmp_path, model)

    rows = audit_task_reports(task, tmp_path)
    assert {row["status"] for row in rows} == {"ok"}

    write_report(task.report_path(tmp_path, "ImputeRefAllele"), alt_support=1)
    rows = audit_task_reports(task, tmp_path)
    status = {row["model"]: row["status"] for row in rows}
    assert status["ImputeRefAllele"] == "support_mismatch"


def test_report_audit_rejects_coordinate_mismatch_with_same_support(
    tmp_path: Path,
) -> None:
    task = make_task()
    write_evaluation_mask(tmp_path / task.evaluation_mask_tsv)
    write_mask_artifacts(task, tmp_path)
    for model in MODEL_ORDER:
        write_report(task.report_path(tmp_path, model))
        write_mask_artifacts(
            task,
            tmp_path,
            model,
            change_coordinate=model == "ImputeVAE",
        )

    rows = audit_task_reports(task, tmp_path)
    status = {row["model"]: row["status"] for row in rows}
    assert status["ImputeVAE"] == "mask_mismatch"
    assert status["ImputeAutoencoder"] == "ok"


def test_report_audit_rejects_nonfinite_scoring_metrics(tmp_path: Path) -> None:
    task = make_task()
    write_evaluation_mask(tmp_path / task.evaluation_mask_tsv)
    write_mask_artifacts(task, tmp_path)
    for model in MODEL_ORDER:
        report_path = task.report_path(tmp_path, model)
        write_report(report_path)
        write_mask_artifacts(task, tmp_path, model)
    report_path = task.report_path(tmp_path, "ImputeVAE")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["REF"]["f1-score"] = float("nan")
    report_path.write_text(json.dumps(report), encoding="utf-8")

    rows = audit_task_reports(task, tmp_path)
    status = {row["model"]: row["status"] for row in rows}
    assert status["ImputeVAE"] == "invalid_report"


def test_read_task_manifest_rejects_noncontiguous_ids(tmp_path: Path) -> None:
    path = tmp_path / "tasks.tsv"
    row = {
        "task_id": 2,
        "dataset_id": "example",
        "strategy": "random",
        "input_vcf": "inputs/example.vcf",
        "evaluation_mask_tsv": "masks/example.tsv",
        "mask_npz": "masks/example.npz",
        "split_tsv": "splits/example.tsv",
        "output_prefix": "results/example",
        "models": " ".join(MODEL_ORDER),
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match="contiguous"):
        read_task_manifest(path)


def test_python_source_digest_includes_paths_and_contents(tmp_path: Path) -> None:
    package = tmp_path / "package"
    (package / "nested").mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "nested" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    first = sha256_python_tree(package)

    (package / "nested" / "module.py").write_text("VALUE = 3\n", encoding="utf-8")

    assert sha256_python_tree(package) != first


def test_python_source_digest_excludes_bundled_gui_dependencies(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    first = sha256_python_tree(package)
    gui_dependency = package / "electron" / "app" / "node_modules" / "tool.py"
    gui_dependency.parent.mkdir(parents=True)
    gui_dependency.write_text("VALUE = 2\n", encoding="utf-8")

    assert sha256_python_tree(package) == first


def test_task_requires_all_six_models() -> None:
    task = make_task()
    invalid = {**task.__dict__, "models": MODEL_ORDER[:-1]}
    with pytest.raises(ValueError, match="model order"):
        CanonicalBenchmarkTask(**invalid).validate_values()


def test_n58_profile_requires_input_hashes() -> None:
    task = replace(
        make_task(),
        benchmark_profile="n58-manuscript-fast-50",
        tune_n_trials=50,
        preset="fast",
        sim_max_tries=100_000,
        expected_pgsui_git_revision="a" * 40,
        expected_pgsui_source_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="require all applicable input hashes"):
        task.validate_values()


def test_n58_profile_rejects_protocol_drift() -> None:
    task = replace(
        make_task(),
        benchmark_profile="n58-manuscript-fast-50",
        tune_n_trials=50,
        preset="fast",
        sim_max_tries=100_000,
        expected_pgsui_git_revision="a" * 40,
        expected_pgsui_source_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="fixed execution profile"):
        replace(task, device="cuda").validate_values()
