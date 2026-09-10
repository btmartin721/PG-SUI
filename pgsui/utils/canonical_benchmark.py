"""Reusable support for the canonical PG-SUI/GTImputation benchmark."""

from __future__ import annotations

import csv
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

MODEL_ORDER: tuple[str, ...] = (
    "ImputeAutoencoder",
    "ImputeVAE",
    "ImputeNLPCA",
    "ImputeUBP",
    "ImputeMostFrequent",
    "ImputeRefAllele",
)
DEEP_MODELS: frozenset[str] = frozenset(MODEL_ORDER[:4])
DETERMINISTIC_MODELS: frozenset[str] = frozenset(MODEL_ORDER[4:])
GENOTYPE_CLASSES: tuple[str, ...] = ("REF", "HET", "ALT")
SIMULATION_STRATEGIES: tuple[str, ...] = (
    "random",
    "random_weighted",
    "random_weighted_inv",
    "nonrandom",
    "nonrandom_weighted",
)
SUPPORTED_BENCHMARK_DEVICES: frozenset[str] = frozenset({"cpu", "cuda"})


@dataclass(frozen=True)
class CanonicalBenchmarkTask:
    """One dataset/strategy PG-SUI execution unit."""

    task_id: int
    dataset_id: str
    strategy: str
    input_vcf: str
    evaluation_mask_tsv: str
    mask_npz: str
    split_tsv: str
    output_prefix: str
    seed: int = 42
    sim_prop: float = 0.3
    validation_split: float = 0.3
    device: str = "cpu"
    n_jobs: int = 1
    tune_n_trials: int = 100
    tune_metrics: tuple[str, ...] = ("f1", "mcc", "average_precision")
    models: tuple[str, ...] = MODEL_ORDER
    preset: str = "balanced"
    batch_size: int = 128
    treefile: str = ""
    qmatrix: str = ""
    siterates: str = ""
    expected_pgsui_version: str = "1.8.5"

    @classmethod
    def from_mapping(cls, row: Mapping[str, str]) -> CanonicalBenchmarkTask:
        """Construct and validate a task from a delimited-text row."""

        def words(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            value = row.get(key, "").strip()
            return tuple(value.split()) if value else default

        task = cls(
            task_id=int(row["task_id"]),
            dataset_id=row["dataset_id"].strip(),
            strategy=row["strategy"].strip(),
            input_vcf=row["input_vcf"].strip(),
            evaluation_mask_tsv=row["evaluation_mask_tsv"].strip(),
            mask_npz=row["mask_npz"].strip(),
            split_tsv=row["split_tsv"].strip(),
            output_prefix=row["output_prefix"].strip(),
            seed=int(row.get("seed", 42)),
            sim_prop=float(row.get("sim_prop", 0.3)),
            validation_split=float(row.get("validation_split", 0.3)),
            device=row.get("device", "cpu").strip() or "cpu",
            n_jobs=int(row.get("n_jobs", 1)),
            tune_n_trials=int(row.get("tune_n_trials", 100)),
            tune_metrics=words("tune_metrics", ("f1", "mcc", "average_precision")),
            models=words("models", MODEL_ORDER),
            preset=row.get("preset", "balanced").strip() or "balanced",
            batch_size=int(row.get("batch_size", 128)),
            treefile=row.get("treefile", "").strip(),
            qmatrix=row.get("qmatrix", "").strip(),
            siterates=row.get("siterates", "").strip(),
            expected_pgsui_version=row.get("expected_pgsui_version", "1.8.5").strip(),
        )
        task.validate_values()
        return task

    def validate_values(self) -> None:
        """Validate scientific and execution invariants for the task."""
        if self.strategy not in SIMULATION_STRATEGIES:
            raise ValueError(f"Unknown simulation strategy: {self.strategy}")
        if self.device not in SUPPORTED_BENCHMARK_DEVICES:
            raise ValueError(
                "Canonical benchmark task requires device='cpu' or device='cuda', "
                f"got {self.device!r}"
            )
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")
        if self.tune_n_trials != 100:
            raise ValueError(
                "Canonical reviewer runs require exactly 100 tuning trials"
            )
        if not 0.0 < self.validation_split < 1.0:
            raise ValueError("validation_split must be in (0, 1)")
        if not 0.0 < self.sim_prop < 1.0:
            raise ValueError("sim_prop must be in (0, 1)")
        if tuple(self.models) != MODEL_ORDER:
            raise ValueError(
                "Canonical task model order must include the four neural models "
                "and two deterministic baselines exactly once"
            )
        if len(self.tune_metrics) < 2:
            raise ValueError("Canonical runs require multi-objective tuning metrics")
        if self.strategy.startswith("nonrandom") and not all(
            (self.treefile, self.qmatrix, self.siterates)
        ):
            raise ValueError(
                f"{self.strategy} requires treefile, qmatrix, and siterates"
            )

    def resolve(self, bundle_root: Path, value: str) -> Path:
        """Resolve one task path relative to the portable bundle root."""
        path = Path(value)
        return path if path.is_absolute() else bundle_root / path

    def required_input_paths(self, bundle_root: Path) -> tuple[Path, ...]:
        """Return files that must exist before this task can run."""
        values = (
            self.input_vcf,
            self.evaluation_mask_tsv,
            self.mask_npz,
            self.split_tsv,
        )
        if self.strategy.startswith("nonrandom"):
            values += (self.treefile, self.qmatrix, self.siterates)
        return tuple(self.resolve(bundle_root, value) for value in values)

    def output_prefix_path(self, bundle_root: Path) -> Path:
        """Return the absolute PG-SUI output prefix."""
        return self.resolve(bundle_root, self.output_prefix)

    def output_directory(self, bundle_root: Path) -> Path:
        """Return the output directory created by the PG-SUI CLI."""
        prefix = self.output_prefix_path(bundle_root)
        return prefix.with_name(f"{prefix.name}_output")

    def report_path(self, bundle_root: Path, model: str) -> Path:
        """Return the expected zygosity classification-report path."""
        family = "Unsupervised" if model in DEEP_MODELS else "Deterministic"
        report_name = (
            "zygosity_report.json"
            if model in DEEP_MODELS
            else "classification_report_zygosity.json"
        )
        return (
            self.output_directory(bundle_root)
            / family
            / "metrics"
            / model
            / report_name
        )

    def evaluation_artifact_path(self, bundle_root: Path, model: str) -> Path:
        """Return the model's exact test-evaluation-mask artifact path."""
        return self.report_path(bundle_root, model).parent / "evaluation_mask_test.npz"

    def command(
        self,
        bundle_root: Path,
        executable: str = "pg-sui",
        *,
        input_path: Path | None = None,
    ) -> list[str]:
        """Build the canonical PG-SUI command for this task."""
        output_prefix = self.output_prefix_path(bundle_root)
        resolved_input = (
            input_path.resolve()
            if input_path is not None
            else self.resolve(bundle_root, self.input_vcf)
        )
        command = [
            executable,
            "--input",
            str(resolved_input),
            "--format",
            "vcf",
            "--preset",
            self.preset,
            "--prefix",
            str(output_prefix),
            "--seed",
            str(self.seed),
            "--n-jobs",
            str(self.n_jobs),
            "--device",
            self.device,
            "--tune",
            "--tune-n-trials",
            str(self.tune_n_trials),
            "--tune-metrics",
            *self.tune_metrics,
            "--models",
            *self.models,
            "--sim-strategy",
            self.strategy,
            "--sim-prop",
            str(self.sim_prop),
            "--batch-size",
            str(self.batch_size),
            "--set",
            f"train.validation_split={self.validation_split}",
            "--disable-plotting",
            "--disable-multiqc",
            "--verbose",
        ]
        if self.strategy.startswith("nonrandom"):
            command.extend(
                [
                    "--treefile",
                    str(self.resolve(bundle_root, self.treefile)),
                    "--qmatrix",
                    str(self.resolve(bundle_root, self.qmatrix)),
                    "--siterates",
                    str(self.resolve(bundle_root, self.siterates)),
                ]
            )
        return command

    def command_text(
        self,
        bundle_root: Path,
        executable: str = "pg-sui",
        *,
        input_path: Path | None = None,
    ) -> str:
        """Return a shell-escaped rendering of :meth:`command`."""
        return shlex.join(
            self.command(
                bundle_root,
                executable=executable,
                input_path=input_path,
            )
        )


def read_task_manifest(path: Path) -> list[CanonicalBenchmarkTask]:
    """Read canonical benchmark tasks from CSV or TSV."""
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open(newline="", encoding="utf-8") as handle:
        tasks = [
            CanonicalBenchmarkTask.from_mapping(row)
            for row in csv.DictReader(handle, delimiter=delimiter)
        ]
    if not tasks:
        raise ValueError(f"Task manifest is empty: {path}")
    task_ids = [task.task_id for task in tasks]
    if task_ids != list(range(len(tasks))):
        raise ValueError("Task IDs must be contiguous, ordered, and zero-based")
    pairs = {(task.dataset_id, task.strategy) for task in tasks}
    if len(pairs) != len(tasks):
        raise ValueError("Task manifest contains duplicate dataset/strategy pairs")
    return tasks


def expected_support(evaluation_mask_tsv: Path) -> dict[str, int]:
    """Count canonical test-mask truth classes without loading genotype data."""
    counts = {label: 0 for label in GENOTYPE_CLASSES}
    with evaluation_mask_tsv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "truth_class" not in (reader.fieldnames or ()):
            raise ValueError(f"Missing truth_class column: {evaluation_mask_tsv}")
        for row in reader:
            label = row["truth_class"].strip().upper()
            if label not in counts:
                raise ValueError(
                    f"Unexpected truth_class={label!r} in {evaluation_mask_tsv}"
                )
            counts[label] += 1
    counts["TOTAL"] = sum(counts.values())
    if counts["TOTAL"] == 0:
        raise ValueError(f"Evaluation mask has no coordinates: {evaluation_mask_tsv}")
    return counts


def read_classification_report(path: Path) -> dict[str, Any]:
    """Read and minimally validate a PG-SUI zygosity report."""
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    missing = [
        label for label in (*GENOTYPE_CLASSES, "macro avg") if label not in report
    ]
    if missing:
        raise ValueError(f"Report {path} is missing sections: {missing}")
    return report


def report_support(report: Mapping[str, Any]) -> dict[str, int]:
    """Extract integer class and total support from a classification report."""
    support = {
        label: int(round(float(report[label]["support"]))) for label in GENOTYPE_CLASSES
    }
    support["TOTAL"] = int(round(float(report["macro avg"]["support"])))
    return support


def compare_evaluation_artifact(
    task: CanonicalBenchmarkTask,
    bundle_root: Path,
    model: str,
) -> tuple[bool, str]:
    """Compare a model's exact evaluation coordinates with the canonical mask."""
    canonical_path = task.resolve(bundle_root, task.mask_npz)
    model_path = task.evaluation_artifact_path(bundle_root, model)
    if not model_path.is_file():
        return False, f"missing evaluation-mask artifact: {model_path}"

    with (
        np.load(canonical_path, allow_pickle=False) as canonical,
        np.load(model_path, allow_pickle=False) as observed,
    ):
        required_canonical = {"evaluation_mask", "test_idx"}
        required_observed = {
            "evaluation_mask_test",
            "test_idx",
            "full_shape",
            "seed",
            "strategy",
            "validation_split",
        }
        if not required_canonical.issubset(canonical.files):
            missing = sorted(required_canonical.difference(canonical.files))
            return False, f"canonical mask is missing arrays: {missing}"
        if not required_observed.issubset(observed.files):
            missing = sorted(required_observed.difference(observed.files))
            return False, f"model mask artifact is missing arrays: {missing}"

        canonical_mask = np.asarray(canonical["evaluation_mask"], dtype=bool)
        canonical_test_idx = np.asarray(canonical["test_idx"], dtype=np.int64)
        observed_test_idx = np.asarray(observed["test_idx"], dtype=np.int64)
        observed_mask = np.asarray(observed["evaluation_mask_test"], dtype=bool)
        observed_shape = tuple(np.asarray(observed["full_shape"], dtype=int).tolist())

        if observed_shape != canonical_mask.shape:
            return False, (
                f"full shape differs: observed={observed_shape}, "
                f"canonical={canonical_mask.shape}"
            )
        if not np.array_equal(observed_test_idx, canonical_test_idx):
            return False, "test row indices differ from the canonical split"
        expected_test_mask = canonical_mask[canonical_test_idx]
        if not np.array_equal(observed_mask, expected_test_mask):
            differences = int(np.count_nonzero(observed_mask ^ expected_test_mask))
            return False, f"evaluation coordinates differ at {differences} cells"
        if int(observed["seed"]) != task.seed:
            return (
                False,
                f"seed differs: observed={int(observed['seed'])}, expected={task.seed}",
            )
        if str(observed["strategy"]) != task.strategy:
            return False, (
                f"strategy differs: observed={str(observed['strategy'])}, "
                f"expected={task.strategy}"
            )
        if not np.isclose(float(observed["validation_split"]), task.validation_split):
            return False, "validation_split differs from the task manifest"
    return True, "exact_match"


def audit_task_reports(
    task: CanonicalBenchmarkTask,
    bundle_root: Path,
) -> list[dict[str, Any]]:
    """Audit all model reports for one task against its canonical test mask."""
    expected = expected_support(task.resolve(bundle_root, task.evaluation_mask_tsv))
    rows: list[dict[str, Any]] = []
    for model in task.models:
        path = task.report_path(bundle_root, model)
        if not path.is_file():
            rows.append(
                {
                    "task_id": task.task_id,
                    "dataset_id": task.dataset_id,
                    "strategy": task.strategy,
                    "model": model,
                    "status": "missing_report",
                    "report_path": str(path),
                }
            )
            continue
        try:
            report = read_classification_report(path)
            observed = report_support(report)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "task_id": task.task_id,
                    "dataset_id": task.dataset_id,
                    "strategy": task.strategy,
                    "model": model,
                    "status": "invalid_report",
                    "error": str(exc),
                    "report_path": str(path),
                }
            )
            continue

        support_matches = all(
            observed[label] == expected[label] for label in (*GENOTYPE_CLASSES, "TOTAL")
        )
        mask_matches, mask_message = compare_evaluation_artifact(
            task, bundle_root, model
        )
        row: dict[str, Any] = {
            "task_id": task.task_id,
            "dataset_id": task.dataset_id,
            "strategy": task.strategy,
            "model": model,
            "status": (
                "ok"
                if support_matches and mask_matches
                else "support_mismatch"
                if not support_matches
                else "mask_mismatch"
            ),
            "exact_mask_match": mask_matches,
            "mask_audit_message": mask_message,
            "report_path": str(path),
            "evaluation_artifact_path": str(
                task.evaluation_artifact_path(bundle_root, model)
            ),
        }
        for label in (*GENOTYPE_CLASSES, "TOTAL"):
            key = label.lower()
            row[f"expected_{key}_support"] = expected[label]
            row[f"observed_{key}_support"] = observed[label]
        rows.append(row)
    return rows


def metric_rows(
    task: CanonicalBenchmarkTask,
    bundle_root: Path,
) -> list[dict[str, Any]]:
    """Flatten one task's valid model reports into analysis-ready rows."""
    rows: list[dict[str, Any]] = []
    for model in task.models:
        path = task.report_path(bundle_root, model)
        if not path.is_file():
            continue
        report = read_classification_report(path)
        base = {
            "task_id": task.task_id,
            "dataset_id": task.dataset_id,
            "strategy": task.strategy,
            "model": model,
            "model_family": "deep_learning"
            if model in DEEP_MODELS
            else "deterministic",
            "backend": task.device if model in DEEP_MODELS else "deterministic",
            "report_path": str(path),
        }
        for label in (*GENOTYPE_CLASSES, "macro avg", "weighted avg"):
            values = report[label]
            rows.append(
                {
                    **base,
                    "class": label,
                    "precision": float(values.get("precision", float("nan"))),
                    "recall": float(values.get("recall", float("nan"))),
                    "f1": float(values.get("f1-score", float("nan"))),
                    "average_precision": float(
                        values.get("average-precision", float("nan"))
                    ),
                    "jaccard": float(values.get("jaccard", float("nan"))),
                    "support": int(round(float(values["support"]))),
                    "mcc": float(report.get("mcc", float("nan"))),
                    "accuracy": float(report.get("accuracy", float("nan"))),
                }
            )
    return rows
