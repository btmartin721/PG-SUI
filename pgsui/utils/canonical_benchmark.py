"""Reusable support for the canonical PG-SUI/GTImputation benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shlex
from dataclasses import asdict, dataclass
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
    sim_max_tries: int = 0
    validation_split: float = 0.3
    ploidy: int = 2
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
    expected_pgsui_version: str = "1.8.6"
    expected_snpio_version: str = "1.7.4"
    expected_pgsui_git_revision: str = ""
    expected_pgsui_source_sha256: str = ""
    benchmark_profile: str = "canonical-balanced-100"
    input_sha256: str = ""
    evaluation_mask_sha256: str = ""
    mask_sha256: str = ""
    split_sha256: str = ""
    treefile_sha256: str = ""
    qmatrix_sha256: str = ""
    siterates_sha256: str = ""

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
            sim_max_tries=int(row.get("sim_max_tries", 0) or 0),
            validation_split=float(row.get("validation_split", 0.3)),
            ploidy=int(row.get("ploidy", 2)),
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
            expected_pgsui_version=row.get("expected_pgsui_version", "1.8.6").strip(),
            expected_snpio_version=row.get("expected_snpio_version", "1.7.4").strip(),
            expected_pgsui_git_revision=row.get(
                "expected_pgsui_git_revision", ""
            ).strip(),
            expected_pgsui_source_sha256=row.get(
                "expected_pgsui_source_sha256", ""
            ).strip(),
            benchmark_profile=row.get(
                "benchmark_profile", "canonical-balanced-100"
            ).strip(),
            input_sha256=row.get("input_sha256", "").strip(),
            evaluation_mask_sha256=row.get("evaluation_mask_sha256", "").strip(),
            mask_sha256=row.get("mask_sha256", "").strip(),
            split_sha256=row.get("split_sha256", "").strip(),
            treefile_sha256=row.get("treefile_sha256", "").strip(),
            qmatrix_sha256=row.get("qmatrix_sha256", "").strip(),
            siterates_sha256=row.get("siterates_sha256", "").strip(),
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
        if self.tune_n_trials < 1:
            raise ValueError("tune_n_trials must be at least 1")
        if not 0.0 < self.validation_split < 1.0:
            raise ValueError("validation_split must be in (0, 1)")
        if self.ploidy not in {1, 2}:
            raise ValueError("ploidy must be 1 or 2")
        if not 0.0 < self.sim_prop < 1.0:
            raise ValueError("sim_prop must be in (0, 1)")
        if self.sim_max_tries < 0:
            raise ValueError("sim_max_tries must be nonnegative")
        if tuple(self.models) != MODEL_ORDER:
            raise ValueError(
                "Canonical task model order must include the four neural models "
                "and two deterministic baselines exactly once"
            )
        if len(self.tune_metrics) < 2:
            raise ValueError("Canonical runs require multi-objective tuning metrics")
        if not self.benchmark_profile:
            raise ValueError("benchmark_profile must not be empty")
        if self.expected_pgsui_git_revision and not re.fullmatch(
            r"[0-9a-fA-F]{40}", self.expected_pgsui_git_revision
        ):
            raise ValueError(
                "expected_pgsui_git_revision must be a 40-character Git SHA"
            )
        if self.expected_pgsui_source_sha256 and not re.fullmatch(
            r"[0-9a-fA-F]{64}", self.expected_pgsui_source_sha256
        ):
            raise ValueError(
                "expected_pgsui_source_sha256 must be a SHA-256 hexadecimal digest"
            )
        for name in (
            "input_sha256",
            "evaluation_mask_sha256",
            "mask_sha256",
            "split_sha256",
            "treefile_sha256",
            "qmatrix_sha256",
            "siterates_sha256",
        ):
            value = getattr(self, name)
            if value and not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
        if self.benchmark_profile == "n58-manuscript-fast-50":
            protocol = {
                "device": (self.device, "cpu"),
                "n_jobs": (self.n_jobs, 1),
                "tune_n_trials": (self.tune_n_trials, 50),
                "preset": (self.preset, "fast"),
                "sim_max_tries": (self.sim_max_tries, 100_000),
                "expected_pgsui_version": (self.expected_pgsui_version, "1.8.6"),
                "expected_snpio_version": (self.expected_snpio_version, "1.7.4"),
            }
            protocol_mismatches = {
                name: values
                for name, values in protocol.items()
                if values[0] != values[1]
            }
            if protocol_mismatches:
                raise ValueError(
                    "N=58 manuscript task violates the fixed execution profile: "
                    f"{protocol_mismatches}"
                )
            if not self.expected_pgsui_git_revision:
                raise ValueError("N=58 manuscript tasks require a PG-SUI Git revision")
            if not self.expected_pgsui_source_sha256:
                raise ValueError(
                    "N=58 manuscript tasks require an exact PG-SUI source digest"
                )
            required_hashes = {
                "input_sha256": self.input_sha256,
                "evaluation_mask_sha256": self.evaluation_mask_sha256,
                "mask_sha256": self.mask_sha256,
                "split_sha256": self.split_sha256,
            }
            if self.strategy.startswith("nonrandom"):
                required_hashes.update(
                    {
                        "treefile_sha256": self.treefile_sha256,
                        "qmatrix_sha256": self.qmatrix_sha256,
                        "siterates_sha256": self.siterates_sha256,
                    }
                )
            missing_hashes = [
                name for name, value in required_hashes.items() if not value
            ]
            if missing_hashes:
                raise ValueError(
                    "N=58 manuscript tasks require all applicable input hashes; "
                    f"missing {missing_hashes}"
                )
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
            "--ploidy",
            str(self.ploidy),
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
        if self.sim_max_tries:
            command.extend(
                ["--set", f"sim.sim_kwargs={{'max_tries': {self.sim_max_tries}}}"]
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


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 digest using bounded memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def python_source_files(root: Path) -> tuple[Path, ...]:
    """Return runtime Python sources while excluding bundled GUI dependencies."""
    return tuple(
        sorted(
            path
            for path in root.rglob("*.py")
            if path.is_file()
            and "__pycache__" not in path.relative_to(root).parts
            and path.relative_to(root).parts[:2] != ("electron", "app")
        )
    )


def sha256_python_tree(root: Path) -> str:
    """Hash Python source paths and contents below a package directory.

    Args:
        root (Path): Root directory containing the Python package source.

    Returns:
        str: Stable SHA-256 digest of every ``.py`` relative path and file.

    Raises:
        FileNotFoundError: If ``root`` is not a directory.
        ValueError: If no Python source files are present.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Python source directory is missing: {root}")
    paths = python_source_files(root)
    if not paths:
        raise ValueError(f"Python source directory contains no .py files: {root}")
    hasher = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        hasher.update(len(relative).to_bytes(8, byteorder="big"))
        hasher.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
    return hasher.hexdigest()


def task_fingerprint(task: CanonicalBenchmarkTask) -> str:
    """Return a stable digest of every task configuration field."""
    payload = json.dumps(asdict(task), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_task_input_hashes(
    task: CanonicalBenchmarkTask,
    bundle_root: Path,
) -> dict[str, dict[str, str | bool]]:
    """Verify task inputs carrying expected SHA-256 values in the manifest."""
    pairs = {
        "input_vcf": (task.input_vcf, task.input_sha256),
        "evaluation_mask_tsv": (
            task.evaluation_mask_tsv,
            task.evaluation_mask_sha256,
        ),
        "mask_npz": (task.mask_npz, task.mask_sha256),
        "split_tsv": (task.split_tsv, task.split_sha256),
        "treefile": (task.treefile, task.treefile_sha256),
        "qmatrix": (task.qmatrix, task.qmatrix_sha256),
        "siterates": (task.siterates, task.siterates_sha256),
    }
    results: dict[str, dict[str, str | bool]] = {}
    for name, (raw_path, expected) in pairs.items():
        if not raw_path or not expected:
            continue
        path = task.resolve(bundle_root, raw_path)
        observed = sha256_file(path)
        results[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "match": observed == expected,
        }
    failures = [name for name, result in results.items() if not result["match"]]
    if failures:
        raise ValueError(f"Task input SHA-256 mismatch: {', '.join(failures)}")
    return results


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


def report_classes(ploidy: int) -> tuple[str, ...]:
    """Return the zygosity report classes for a supported ploidy."""
    if ploidy == 1:
        return ("REF", "ALT")
    if ploidy == 2:
        return GENOTYPE_CLASSES
    raise ValueError("ploidy must be 1 or 2")


def read_classification_report(path: Path, *, ploidy: int = 2) -> dict[str, Any]:
    """Read and validate all manuscript metrics in a PG-SUI zygosity report."""
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    classes = report_classes(ploidy)
    missing = [
        label
        for label in (*classes, "macro avg", "weighted avg")
        if label not in report
    ]
    if missing:
        raise ValueError(f"Report {path} is missing sections: {missing}")
    metric_names = (
        "precision",
        "recall",
        "f1-score",
        "support",
        "average-precision",
        "jaccard",
    )
    for label in (*classes, "macro avg", "weighted avg"):
        section = report[label]
        if not isinstance(section, Mapping):
            raise TypeError(f"Report section {label!r} is not a mapping")
        missing_metrics = [name for name in metric_names if name not in section]
        if missing_metrics:
            raise ValueError(
                f"Report section {label!r} is missing metrics: {missing_metrics}"
            )
        values = np.asarray([float(section[name]) for name in metric_names])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Report section {label!r} has non-finite metrics")
        support = float(section["support"])
        if support < 0 or not support.is_integer():
            raise ValueError(f"Report section {label!r} has invalid support")
    for name in ("mcc", "accuracy"):
        if name not in report or not np.isfinite(float(report[name])):
            raise ValueError(f"Report has invalid or missing {name}")
    expected_total = sum(int(report[label]["support"]) for label in classes)
    for label in ("macro avg", "weighted avg"):
        if int(report[label]["support"]) != expected_total:
            raise ValueError(f"Report section {label!r} support is inconsistent")
    return report


def report_support(report: Mapping[str, Any], *, ploidy: int = 2) -> dict[str, int]:
    """Extract integer class and total support from a classification report."""
    support = {label: 0 for label in GENOTYPE_CLASSES}
    support.update(
        {
            label: int(round(float(report[label]["support"])))
            for label in report_classes(ploidy)
        }
    )
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
            report = read_classification_report(path, ploidy=task.ploidy)
            observed = report_support(report, ploidy=task.ploidy)
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
        report = read_classification_report(path, ploidy=task.ploidy)
        base = {
            "task_id": task.task_id,
            "dataset_id": task.dataset_id,
            "strategy": task.strategy,
            "model": model,
            "model_family": "deep_learning"
            if model in DEEP_MODELS
            else "deterministic",
            "backend": task.device if model in DEEP_MODELS else "deterministic",
            "ploidy": task.ploidy,
            "report_path": str(path),
        }
        for label in (*report_classes(task.ploidy), "macro avg", "weighted avg"):
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
