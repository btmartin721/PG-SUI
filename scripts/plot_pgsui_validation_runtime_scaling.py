#!/usr/bin/env python3
"""Plot PG-SUI validation runtime scaling for the final N=60 datasets.

The workflow intentionally ignores the focused GTImputation comparison bundle.
It reads the final-60 PG-SUI validation outputs, attaches exact dataset sizes
from the staged PHYLIP headers, estimates one runtime per dataset/model by
aggregating strategy runs, and fits per-model regressions against total
genotype count.

For deep PG-SUI models, the plotted/regressed runtime is the logged per-trial
Optuna duration, not the full 100-trial tuning wall time. Raw fit/imputation
segment timings are retained in the row-level table for auditability.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
(SCRIPT_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import lines as mlines
from matplotlib import ticker as mticker
from scipy import stats

PGSUI_MODELS = (
    "ImputeRefAllele",
    "ImputeMostFrequent",
    "ImputeAutoencoder",
    "ImputeVAE",
    "ImputeNLPCA",
    "ImputeUBP",
)

DEEP_MODELS = (
    "ImputeAutoencoder",
    "ImputeVAE",
    "ImputeNLPCA",
    "ImputeUBP",
)

MODEL_LABELS = {
    "ImputeRefAllele": "RefAllele",
    "ImputeMostFrequent": "MostFrequent",
    "ImputeAutoencoder": "Autoencoder",
    "ImputeVAE": "VAE",
    "ImputeNLPCA": "NLPCA",
    "ImputeUBP": "UBP",
}

MODEL_COLORS = {
    "ImputeRefAllele": "#18E9EE",
    "ImputeMostFrequent": "#5AAE00",
    "ImputeAutoencoder": "#FE019A",
    "ImputeVAE": "#811BF3",
    "ImputeNLPCA": "#D69E00",
    "ImputeUBP": "#FA4224",
}

DETERMINISTIC_LOGS = {
    "ImputeRefAllele": "pgsui.impute.deterministic.imputers.ref_allele.log",
    "ImputeMostFrequent": "pgsui.impute.deterministic.imputers.mode.log",
}

STRATEGY_ORDER = (
    "random",
    "random_weighted",
    "random_weighted_inv",
    "nonrandom",
    "nonrandom_weighted",
)

TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

SAMPLES_LOCI_RE = re.compile(
    r"(?P<samples>[0-9][0-9,]*)\s+samples?\s*;\s*"
    r"(?P<loci>[0-9][0-9,]*)\s+(?:SNPs?/)?loci",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RuntimeSegment:
    model: str
    start_at: datetime
    end_at: datetime
    tuning_wall_seconds: float
    mean_trial_seconds: float
    trial_count: float
    end_event: str

    @property
    def runtime_seconds(self) -> float:
        return max((self.end_at - self.start_at).total_seconds(), 0.0)

    @property
    def per_trial_runtime_seconds(self) -> float:
        if np.isfinite(self.mean_trial_seconds) and self.mean_trial_seconds > 0:
            return float(self.mean_trial_seconds)
        if (
            np.isfinite(self.tuning_wall_seconds)
            and self.tuning_wall_seconds > 0
            and np.isfinite(self.trial_count)
            and self.trial_count > 0
        ):
            return float(self.tuning_wall_seconds / self.trial_count)
        return math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create PG-SUI-only runtime scaling plots and regressions for the  final N=60 validation datasets."
        )
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=PROJECT_ROOT / "pgsui_osf_reviewer_package",
        help="Root of the staged reviewer package.",
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=None,
        help=(
            "PG-SUI by-dataset validation root. Defaults to "
            "<package-root>/01_validation_results/pgsui/by_dataset."
        ),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=None,
        help=(
            "Final dataset manifest with input PHYLIP paths. Defaults to "
            "<package-root>/manifests/final_dataset_manifest.csv."
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Final dataset metadata CSV used only as a sample/locus fallback.  Defaults to the staged final_analysis_dataset_metadata.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for tables and plots. Defaults to <package-root>/01_validation_results/pgsui/runtime_scaling."
        ),
    )
    parser.add_argument(
        "--strategy-aggregation",
        choices=("mean", "median"),
        default="mean",
        help="Dataset-model runtime statistic used for regressions and plotting.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Plot export DPI.")
    return parser.parse_args()


def round_up_nearest_magnitude(number):
    if number <= 0:
        return 0  # Handles zero or negative numbers gracefully

    # 1. Determine the magnitude scale (e.g., 4500 is in the 1000s scale)
    scale = 10 ** math.floor(math.log10(number))

    # 2. Divide by the scale, round up to the next integer, and multiply back
    return int(math.ceil(number / scale) * scale)


def make_ticks(
    minimum: float,
    maximum: float,
    n_intermediate: int = 3,
) -> np.ndarray:
    """Create integer tick positions spanning a positive logarithmic range.

    The minimum and maximum values are always included, with geometrically spaced intermediate ticks.

    Args:
        minimum: Minimum positive axis value.
        maximum: Maximum positive axis value.
        n_intermediate: Number of ticks between the minimum and maximum.

    Returns:
        Sorted array of unique integer tick positions.

    Raises:
        ValueError: If either bound is nonpositive or maximum is less than
            minimum.
    """
    if minimum <= 0 or maximum <= 0:
        raise ValueError("Log-scaled axis bounds must be greater than zero.")

    if maximum < minimum:
        raise ValueError("maximum must be greater than or equal to minimum.")

    minimum_int = int(round(minimum))
    maximum_int = int(round(maximum))

    if minimum_int == maximum_int:
        return np.asarray([minimum_int], dtype=np.int64)

    ticks = np.geomspace(minimum, maximum, num=n_intermediate + 2)

    ticks = np.rint(ticks).astype(np.int64)
    ticks[0] = minimum_int
    ticks[-1] = maximum_int

    return np.unique(ticks)


def resolve_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    package_root = args.package_root.expanduser().resolve()
    args.package_root = package_root
    if args.validation_root is None:
        args.validation_root = (
            package_root / "01_validation_results" / "pgsui" / "by_dataset"
        )
    if args.dataset_manifest is None:
        args.dataset_manifest = (
            package_root / "manifests" / "final_dataset_manifest.csv"
        )
    if args.metadata is None:
        args.metadata = (
            package_root
            / "02_inputs_metadata"
            / "final_dataset_metadata_review"
            / "data"
            / "final_analysis_dataset_metadata.csv"
        )
    if args.output_dir is None:
        args.output_dir = (
            package_root / "01_validation_results" / "pgsui" / "runtime_scaling"
        )
    args.validation_root = args.validation_root.expanduser().resolve()
    args.dataset_manifest = args.dataset_manifest.expanduser().resolve()
    args.metadata = args.metadata.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def parse_phylip_header(path: Path) -> tuple[int, int]:
    """Return sample and locus counts from a PHYLIP header."""
    with path.open("rt", errors="replace") as handle:
        header = handle.readline().strip()
    parts = header.split()
    if len(parts) < 2:
        raise ValueError(f"PHYLIP header has fewer than two fields: {path}")
    try:
        sample_size = int(parts[0])
        locus_count = int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"PHYLIP header does not start with integer counts: {path}"
        ) from exc
    if sample_size <= 0 or locus_count <= 0:
        raise ValueError(f"PHYLIP header counts must be positive: {path}")
    return sample_size, locus_count


def parse_samples_loci_text(value: object) -> tuple[int, int] | None:
    """Parse '<N> samples; <L> SNPs/loci' metadata text."""
    match = SAMPLES_LOCI_RE.search(str(value))
    if not match:
        return None
    sample_size = int(match.group("samples").replace(",", ""))
    locus_count = int(match.group("loci").replace(",", ""))
    return sample_size, locus_count


def load_metadata_size_fallback(metadata_path: Path) -> dict[str, tuple[int, int]]:
    if not metadata_path.exists():
        return {}

    metadata = pd.read_csv(metadata_path)
    if "Dataset" not in metadata.columns or "Samples and loci" not in metadata.columns:
        return {}

    fallback: dict[str, tuple[int, int]] = {}
    for _, row in metadata.iterrows():
        parsed = parse_samples_loci_text(row["Samples and loci"])
        if parsed is not None:
            fallback[str(row["Dataset"])] = parsed
    return fallback


def resolve_package_path(package_root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return package_root / path


def load_dataset_sizes(
    *,
    package_root: Path,
    dataset_manifest_path: Path,
    metadata_path: Path,
) -> pd.DataFrame:
    """Load exact sample/locus counts for the final-60 validation datasets."""
    if not dataset_manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {dataset_manifest_path}")

    manifest = pd.read_csv(dataset_manifest_path)
    required = {"dataset_id", "dataset_folder", "input_phylip"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Dataset manifest missing required columns: {sorted(missing)}"
        )

    metadata_fallback = load_metadata_size_fallback(metadata_path)
    rows: list[dict[str, object]] = []

    for _, row in manifest.iterrows():
        input_phylip = resolve_package_path(package_root, row["input_phylip"])
        sample_size = math.nan
        locus_count = math.nan
        source = ""

        if input_phylip.exists():
            sample_size, locus_count = parse_phylip_header(input_phylip)
            source = "phylip_header"
        else:
            fallback = metadata_fallback.get(str(row["dataset_id"]))
            if fallback is not None:
                sample_size, locus_count = fallback
                source = "metadata_samples_and_loci"

        rows.append(
            {
                "dataset_id": str(row["dataset_id"]),
                "dataset_folder": str(row["dataset_folder"]),
                "input_phylip": str(input_phylip),
                "sample_size": sample_size,
                "locus_count": locus_count,
                "total_genotypes": (
                    float(sample_size) * float(locus_count)
                    if np.isfinite(sample_size) and np.isfinite(locus_count)
                    else math.nan
                ),
                "size_source": source or "missing",
            }
        )

    sizes = pd.DataFrame(rows)
    if sizes.empty:
        raise ValueError("No dataset-size rows loaded.")
    return sizes


def parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def parse_duration_text(value: object) -> float | None:
    """Parse HH:MM:SS[.frac] or compact h/m/s durations to seconds."""
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)

    hms_match = re.fullmatch(
        r"(?:(\d+)\s+days?,\s*)?(?:(\d+):)?(\d{1,2}):(\d{2}(?:\.\d+)?)",
        text,
    )
    if hms_match:
        days = float(hms_match.group(1) or 0)
        hours = float(hms_match.group(2) or 0)
        minutes = float(hms_match.group(3))
        seconds = float(hms_match.group(4))
        return days * 86400.0 + hours * 3600.0 + minutes * 60.0 + seconds

    total = 0.0
    matched = False
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", text.lower()):
        matched = True
        total += float(amount) * {"h": 3600.0, "m": 60.0, "s": 1.0}[unit]
    return total if matched else None


def parse_deep_runtime_segments(base_log: Path, model: str) -> list[RuntimeSegment]:
    """Parse complete fit-to-transform runtime segments for one deep model."""
    if not base_log.exists():
        return []

    segments: list[RuntimeSegment] = []
    current_start: datetime | None = None
    current_fit_complete: datetime | None = None
    current_tuning = math.nan
    current_mean_trial = math.nan
    current_trial_count = math.nan

    tuning_re = re.compile(
        r"Tuning completed in:\s*([^\(]+).*?\bfor\s+(\d+)\s+trials?",
        flags=re.IGNORECASE,
    )
    mean_trial_res = (
        re.compile(r"Average trial duration:\s*(.+)$", flags=re.IGNORECASE),
        re.compile(r"Avg trial \(s\)\s*=\s*([0-9]+(?:\.[0-9]+)?)"),
        re.compile(r"mean trial time\s*:\s*([^\n\r|]+)", flags=re.IGNORECASE),
    )

    with base_log.open("rt", errors="replace") as handle:
        for line in handle:
            timestamp = parse_timestamp(line)
            if timestamp is None:
                continue

            if f"Fitting {model} model" in line:
                if current_start is not None and current_fit_complete is not None:
                    segments.append(
                        RuntimeSegment(
                            model=model,
                            start_at=current_start,
                            end_at=current_fit_complete,
                            tuning_wall_seconds=current_tuning,
                            mean_trial_seconds=current_mean_trial,
                            trial_count=current_trial_count,
                            end_event="fit_complete",
                        )
                    )
                current_start = timestamp
                current_fit_complete = None
                current_tuning = math.nan
                current_mean_trial = math.nan
                current_trial_count = math.nan
                continue

            if current_start is None:
                continue

            if f"{model} fitting complete!" in line:
                current_fit_complete = timestamp
                continue

            tuning_match = tuning_re.search(line)
            if tuning_match:
                parsed = parse_duration_text(tuning_match.group(1).strip())
                if parsed is not None:
                    current_tuning = parsed
                current_trial_count = float(tuning_match.group(2))
                continue

            for mean_trial_re in mean_trial_res:
                mean_trial_match = mean_trial_re.search(line)
                if mean_trial_match:
                    parsed = parse_duration_text(mean_trial_match.group(1).strip())
                    if parsed is not None:
                        current_mean_trial = parsed
                    break
            else:
                mean_trial_match = None

            if mean_trial_match:
                continue

            if f"{model} Imputation complete!" in line:
                segments.append(
                    RuntimeSegment(
                        model=model,
                        start_at=current_start,
                        end_at=timestamp,
                        tuning_wall_seconds=current_tuning,
                        mean_trial_seconds=current_mean_trial,
                        trial_count=current_trial_count,
                        end_event="imputation_complete",
                    )
                )
                current_start = None
                current_fit_complete = None
                current_tuning = math.nan
                current_mean_trial = math.nan
                current_trial_count = math.nan

    if current_start is not None and current_fit_complete is not None:
        segments.append(
            RuntimeSegment(
                model=model,
                start_at=current_start,
                end_at=current_fit_complete,
                tuning_wall_seconds=current_tuning,
                mean_trial_seconds=current_mean_trial,
                trial_count=current_trial_count,
                end_event="fit_complete",
            )
        )

    return segments


def select_deep_runtime(base_log: Path, model: str) -> dict[str, object]:
    segments = parse_deep_runtime_segments(base_log, model)
    if not segments:
        return {
            "runtime_seconds": math.nan,
            "runtime_source": "missing_complete_deep_log_segment",
            "runtime_start_at": "",
            "runtime_end_at": "",
            "tuning_wall_seconds": math.nan,
            "mean_trial_seconds": math.nan,
            "per_trial_runtime_seconds": math.nan,
            "trial_count": math.nan,
            "runtime_segment_count": 0,
        }

    selected = max(segments, key=lambda segment: segment.runtime_seconds)
    source = (
        "base_log_longest_complete_fit_transform_segment"
        if selected.end_event == "imputation_complete"
        else "base_log_longest_fit_complete_segment_missing_transform_marker"
    )
    return {
        "runtime_seconds": selected.runtime_seconds,
        "runtime_source": source,
        "runtime_start_at": selected.start_at.isoformat(sep=" "),
        "runtime_end_at": selected.end_at.isoformat(sep=" "),
        "tuning_wall_seconds": selected.tuning_wall_seconds,
        "mean_trial_seconds": selected.mean_trial_seconds,
        "per_trial_runtime_seconds": selected.per_trial_runtime_seconds,
        "trial_count": selected.trial_count,
        "runtime_segment_count": len(segments),
    }


def parse_deterministic_runtime(log_path: Path) -> dict[str, object]:
    if not log_path.exists():
        return {
            "runtime_seconds": math.nan,
            "runtime_source": "missing_deterministic_log",
            "runtime_start_at": "",
            "runtime_end_at": "",
            "tuning_wall_seconds": math.nan,
            "mean_trial_seconds": math.nan,
            "per_trial_runtime_seconds": math.nan,
            "trial_count": math.nan,
            "runtime_segment_count": 0,
        }

    timestamps: list[datetime] = []
    with log_path.open("rt", errors="replace") as handle:
        for line in handle:
            timestamp = parse_timestamp(line)
            if timestamp is not None:
                timestamps.append(timestamp)

    if len(timestamps) < 2:
        return {
            "runtime_seconds": math.nan,
            "runtime_source": "unparsed_deterministic_log",
            "runtime_start_at": "",
            "runtime_end_at": "",
            "tuning_wall_seconds": math.nan,
            "mean_trial_seconds": math.nan,
            "per_trial_runtime_seconds": math.nan,
            "trial_count": math.nan,
            "runtime_segment_count": 0,
        }

    start_at = min(timestamps)
    end_at = max(timestamps)
    return {
        "runtime_seconds": max((end_at - start_at).total_seconds(), 0.0),
        "runtime_source": "deterministic_log_timestamp_span_lower_bound",
        "runtime_start_at": start_at.isoformat(sep=" "),
        "runtime_end_at": end_at.isoformat(sep=" "),
        "tuning_wall_seconds": math.nan,
        "mean_trial_seconds": math.nan,
        "per_trial_runtime_seconds": math.nan,
        "trial_count": math.nan,
        "runtime_segment_count": 1,
    }


def model_artifact_paths(
    output_dir: Path, model: str
) -> tuple[Path | None, Path | None]:
    if model in DEEP_MODELS:
        report_path = (
            output_dir / "Unsupervised" / "metrics" / model / "zygosity_report.json"
        )
        imputed_dir = output_dir / "Unsupervised" / "imputed" / model
    else:
        report_path = (
            output_dir
            / "Deterministic"
            / "metrics"
            / model
            / "classification_report_zygosity.json"
        )
        imputed_dir = output_dir / "Deterministic" / "imputed" / model

    imputed_paths = sorted(
        path
        for path in imputed_dir.glob("*")
        if path.is_file() and path.name.endswith((".phy", ".vcf", ".vcf.gz"))
    )
    return (report_path if report_path.exists() else None), (
        imputed_paths[0] if imputed_paths else None
    )


def iter_output_dirs(validation_root: Path) -> Iterable[tuple[str, str, Path]]:
    """Yield dataset folder, strategy, and PG-SUI output directory paths."""
    for dataset_dir in sorted(
        path for path in validation_root.iterdir() if path.is_dir()
    ):
        if dataset_dir.name.startswith("."):
            continue
        for strategy_dir in sorted(
            path for path in dataset_dir.iterdir() if path.is_dir()
        ):
            if strategy_dir.name not in STRATEGY_ORDER:
                continue
            output_dirs = sorted(
                path
                for path in strategy_dir.iterdir()
                if path.is_dir() and path.name.endswith("_output")
            )
            for output_dir in output_dirs:
                yield dataset_dir.name, strategy_dir.name, output_dir


def collect_model_runtimes(validation_root: Path, sizes: pd.DataFrame) -> pd.DataFrame:
    if not validation_root.exists():
        raise FileNotFoundError(f"Validation root not found: {validation_root}")

    size_by_folder = sizes.set_index("dataset_folder").to_dict(orient="index")
    rows: list[dict[str, object]] = []

    for dataset_folder, strategy, output_dir in iter_output_dirs(validation_root):
        size_row = size_by_folder.get(dataset_folder, {})
        base_log = output_dir / "logs" / "pgsui.impute.unsupervised.base.log"

        for model in PGSUI_MODELS:
            if model in DEEP_MODELS:
                runtime = select_deep_runtime(base_log, model)
            else:
                runtime = parse_deterministic_runtime(
                    output_dir / "logs" / DETERMINISTIC_LOGS[model]
                )

            report_path, imputed_path = model_artifact_paths(output_dir, model)

            rows.append(
                {
                    "dataset_id": size_row.get("dataset_id", ""),
                    "dataset_folder": dataset_folder,
                    "strategy": strategy,
                    "output_dir": str(output_dir),
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "sample_size": size_row.get("sample_size", math.nan),
                    "locus_count": size_row.get("locus_count", math.nan),
                    "total_genotypes": size_row.get("total_genotypes", math.nan),
                    "size_source": size_row.get("size_source", ""),
                    "report_path": str(report_path) if report_path else "",
                    "imputed_path": str(imputed_path) if imputed_path else "",
                    "report_complete": bool(report_path),
                    "imputed_complete": bool(imputed_path),
                    "artifact_complete": bool(report_path and imputed_path),
                    **runtime,
                }
            )

    runtime_df = pd.DataFrame(rows)
    if runtime_df.empty:
        raise ValueError(f"No PG-SUI output directories found under {validation_root}")
    return runtime_df


def coerce_runtime_rows(runtime_df: pd.DataFrame) -> pd.DataFrame:
    df = runtime_df.copy()
    for column in (
        "runtime_seconds",
        "sample_size",
        "locus_count",
        "total_genotypes",
        "tuning_wall_seconds",
        "mean_trial_seconds",
        "per_trial_runtime_seconds",
        "trial_count",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    deep_mask = df["model"].isin(DEEP_MODELS) if "model" in df.columns else False
    per_trial = (
        df["per_trial_runtime_seconds"]
        if "per_trial_runtime_seconds" in df.columns
        else pd.Series(math.nan, index=df.index)
    )
    raw_runtime = (
        df["runtime_seconds"]
        if "runtime_seconds" in df.columns
        else pd.Series(math.nan, index=df.index)
    )

    df["runtime_metric_seconds"] = math.nan
    df.loc[deep_mask & per_trial.gt(0), "runtime_metric_seconds"] = per_trial.loc[
        deep_mask & per_trial.gt(0)
    ]
    df.loc[~deep_mask & raw_runtime.gt(0), "runtime_metric_seconds"] = raw_runtime.loc[
        ~deep_mask & raw_runtime.gt(0)
    ]
    df["runtime_metric_source"] = np.where(
        deep_mask & per_trial.gt(0),
        "deep_log_per_trial_seconds",
        np.where(
            deep_mask,
            "deep_missing_per_trial_seconds",
            "deterministic_log_runtime_seconds",
        ),
    )
    return df


def summarize_dataset_model_runtimes(
    runtime_df: pd.DataFrame,
    *,
    strategy_aggregation: str,
) -> pd.DataFrame:
    """Summarize dataset-model runtimes for regression and plotting."""
    df = coerce_runtime_rows(runtime_df)
    df = df.loc[
        df["report_complete"]
        & df["runtime_metric_seconds"].gt(0)
        & df["total_genotypes"].gt(0)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    group_cols = [
        "dataset_id",
        "dataset_folder",
        "model",
        "model_label",
        "sample_size",
        "locus_count",
        "total_genotypes",
    ]

    summary = (
        df.groupby(group_cols, observed=True)
        .agg(
            n_strategy_runs=("strategy", "nunique"),
            strategies=(
                "strategy",
                lambda values: ";".join(sorted(set(map(str, values)))),
            ),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            median_runtime_seconds=("runtime_seconds", "median"),
            sd_runtime_seconds=("runtime_seconds", "std"),
            min_runtime_seconds=("runtime_seconds", "min"),
            max_runtime_seconds=("runtime_seconds", "max"),
            mean_tuning_wall_seconds=("tuning_wall_seconds", "mean"),
            mean_trial_seconds=("mean_trial_seconds", "mean"),
            mean_per_trial_runtime_seconds=("per_trial_runtime_seconds", "mean"),
            mean_runtime_metric_seconds=("runtime_metric_seconds", "mean"),
            median_runtime_metric_seconds=("runtime_metric_seconds", "median"),
            sd_runtime_metric_seconds=("runtime_metric_seconds", "std"),
            min_runtime_metric_seconds=("runtime_metric_seconds", "min"),
            max_runtime_metric_seconds=("runtime_metric_seconds", "max"),
            runtime_metric_sources=(
                "runtime_metric_source",
                lambda values: ";".join(sorted(set(map(str, values)))),
            ),
        )
        .reset_index()
    )

    y_col = f"{strategy_aggregation}_runtime_metric_seconds"
    summary["runtime_seconds_for_regression"] = summary[y_col]
    summary["runtime_metric_seconds_for_regression"] = summary[y_col]
    summary["runtime_hours_for_regression"] = (
        summary["runtime_seconds_for_regression"] / 3600.0
    )
    summary["_model_rank"] = summary["model"].map(
        {model: index for index, model in enumerate(PGSUI_MODELS)}
    )
    return (
        summary.sort_values(["_model_rank", "total_genotypes", "dataset_id"])
        .drop(columns="_model_rank")
        .reset_index(drop=True)
    )


def linregress_or_nan(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if x_values.size < 2 or np.unique(x_values).size < 2:
        return {
            "slope": math.nan,
            "intercept": math.nan,
            "r_value": math.nan,
            "r_squared": math.nan,
            "p_value": math.nan,
            "std_error": math.nan,
        }
    fit = stats.linregress(x_values, y_values)
    return {
        "slope": float(fit.slope),  # type: ignore
        "intercept": float(fit.intercept),  # type: ignore
        "r_value": float(fit.rvalue),  # type: ignore
        "r_squared": float(fit.rvalue**2),  # type: ignore
        "p_value": float(fit.pvalue),  # type: ignore
        "std_error": float(fit.stderr),  # type: ignore
    }


def regression_summary(dataset_model_summary: pd.DataFrame) -> pd.DataFrame:
    if dataset_model_summary.empty:
        return pd.DataFrame()

    df = dataset_model_summary.copy()
    df["total_genotypes"] = pd.to_numeric(df["total_genotypes"], errors="coerce")
    df["runtime_seconds_for_regression"] = pd.to_numeric(
        df["runtime_seconds_for_regression"], errors="coerce"
    )

    rows: list[dict[str, object]] = []
    for model in PGSUI_MODELS:
        model_df = df.loc[df["model"].eq(model)].dropna(
            subset=["total_genotypes", "runtime_seconds_for_regression"]
        )
        model_df = model_df.loc[
            model_df["total_genotypes"].gt(0)
            & model_df["runtime_seconds_for_regression"].gt(0)
        ]

        x_values = model_df["total_genotypes"].to_numpy(dtype=float)
        y_values = model_df["runtime_seconds_for_regression"].to_numpy(dtype=float)
        linear = linregress_or_nan(x_values, y_values)
        log_fit = linregress_or_nan(np.log10(x_values), np.log10(y_values))

        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n_datasets": int(model_df["dataset_id"].nunique()),
                "n_strategy_runs": (
                    int(model_df["n_strategy_runs"].sum())
                    if "n_strategy_runs" in model_df.columns
                    else 0
                ),
                "total_genotypes_min": (
                    float(np.nanmin(x_values)) if x_values.size else math.nan
                ),
                "total_genotypes_max": (
                    float(np.nanmax(x_values)) if x_values.size else math.nan
                ),
                "runtime_seconds_min": (
                    float(np.nanmin(y_values)) if y_values.size else math.nan
                ),
                "runtime_seconds_max": (
                    float(np.nanmax(y_values)) if y_values.size else math.nan
                ),
                "runtime_metric_seconds_min": (
                    float(np.nanmin(y_values)) if y_values.size else math.nan
                ),
                "runtime_metric_seconds_max": (
                    float(np.nanmax(y_values)) if y_values.size else math.nan
                ),
                "slope_seconds_per_genotype": linear["slope"],
                "slope_seconds_per_million_genotypes": (
                    linear["slope"] * 1_000_000.0
                    if np.isfinite(linear["slope"])
                    else math.nan
                ),
                "slope_hours_per_million_genotypes": (
                    linear["slope"] * 1_000_000.0 / 3600.0
                    if np.isfinite(linear["slope"])
                    else math.nan
                ),
                "intercept_seconds": linear["intercept"],
                "r_squared_linear": linear["r_squared"],
                "p_value_linear": linear["p_value"],
                "std_error_linear": linear["std_error"],
                "log10_runtime_vs_log10_genotypes_slope": log_fit["slope"],
                "log10_runtime_vs_log10_genotypes_intercept": log_fit["intercept"],
                "r_squared_log10": log_fit["r_squared"],
                "p_value_log10": log_fit["p_value"],
                "std_error_log10": log_fit["std_error"],
            }
        )

    return pd.DataFrame(rows)


def format_duration_tick(seconds: float, _pos: int | None = None) -> str:
    """Format seconds for log-scaled y-axis ticks."""
    if not np.isfinite(seconds) or seconds <= 0:
        return ""
    return f"{seconds:.1f}s"


def format_effect_per_million(hours_per_million: float) -> str:
    """Format the effect per million genotypes."""
    if not np.isfinite(hours_per_million):
        return ""
    seconds_per_million = hours_per_million * 3600.0
    return f"{seconds_per_million:.1f} s / M genotypes"


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.65,
            "font.family": "Arial",
            "font.size": 16,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_runtime_scaling(
    dataset_model_summary: pd.DataFrame,
    regressions: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int,
) -> None:
    """Plot runtime scaling for PG-SUI models across datasets."""

    if dataset_model_summary.empty or regressions.empty:
        print("Skipping runtime scaling plot: no summarized runtime rows.")
        return

    plot_df = dataset_model_summary.copy()
    plot_df = plot_df.loc[
        plot_df["total_genotypes"].gt(0)
        & plot_df["runtime_seconds_for_regression"].gt(0)
    ]
    if plot_df.empty:
        return

    configure_plot_style()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=((18, 12)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    x_min = float(plot_df["total_genotypes"].min())
    x_max = float(plot_df["total_genotypes"].max())

    y_min = float(plot_df["runtime_seconds_for_regression"].min())
    y_max = float(plot_df["runtime_seconds_for_regression"].max())

    x_line = np.logspace(np.log10(x_min), np.log10(x_max), 200)

    handles: list = []
    for i, (ax, model) in enumerate(zip(axes.ravel(), PGSUI_MODELS)):
        model_df = plot_df.loc[plot_df["model"].eq(model)]
        color = MODEL_COLORS[model]

        if model_df.empty:
            ax.set_axis_off()
            continue

        ax.scatter(
            model_df["total_genotypes"],
            model_df["runtime_seconds_for_regression"],
            s=56,
            color=color,
            edgecolor="black",
            linewidth=0.55,
            alpha=0.78,
            zorder=4,
        )

        fit = regressions.loc[regressions["model"].eq(model)]

        if not fit.empty:
            fit_row = fit.iloc[0]
            slope = fit_row["log10_runtime_vs_log10_genotypes_slope"]
            intercept = fit_row["log10_runtime_vs_log10_genotypes_intercept"]

            if pd.notna(slope) and pd.notna(intercept):
                y_line = 10 ** (float(intercept) + float(slope) * np.log10(x_line))

                ax.plot(
                    x_line,
                    y_line,
                    color="#1F1F24",
                    linewidth=2.0,
                    zorder=3,
                )

            linear_effect = fit_row["slope_hours_per_million_genotypes"]
            r2 = fit_row["r_squared_log10"]

            label_lines = []

            if pd.notna(linear_effect):
                label_lines.append(format_effect_per_million(float(linear_effect)))

            if pd.notna(r2):
                label_lines.append(f"R²={float(r2):.2f}")

            if label_lines:
                ax.text(
                    0.03,
                    0.95,
                    "\n".join(label_lines),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize="xx-large",
                    color="#1F1F24",
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "#BBBBBB",
                        "alpha": 0.92,
                    },
                )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlim((x_min * 0.85, x_max * 1.15))
        ax.set_ylim((y_min * 0.85, y_max * 1.15))

        if i >= len(PGSUI_MODELS) // 2:
            ax.set_xlabel("log10(Total Genotypes)", fontsize="xx-large")
        if i % 3 == 0:
            ax.set_ylabel("log10(Execution Time [s])", fontsize="xx-large")

        x_ticks = make_ticks(x_min, x_max, n_intermediate=3)
        x_ticks = [round_up_nearest_magnitude(tick) for tick in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(tick)) for tick in x_ticks], fontsize="xx-large")

        ax.xaxis.set_major_formatter(mticker.EngFormatter(places=0, sep=""))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_duration_tick))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        ax.tick_params(axis="both", labelsize="xx-large")

        legend_handles = mlines.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=20,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.65,
            alpha=0.82,
            label=MODEL_LABELS[model],
        )
        handles.append(legend_handles)
        ax.grid(visible=False, which="both", axis="both")

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        fontsize="xx-large",
        title_fontsize="xx-large",
        shadow=True,
        fancybox=True,
        columnspacing=1.5,
        handletextpad=0.5,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out_dir.mkdir(parents=True, exist_ok=True)

    for suffix in ("png", "pdf"):
        fig.savefig(
            out_dir / f"pgsui_validation_runtime_total_genotypes.{suffix}",
            dpi=dpi if suffix == "png" else None,
            bbox_inches="tight",
        )

    plt.close(fig)


def write_readme(
    out_dir: Path,
    *,
    validation_root: Path,
    dataset_manifest: Path,
    strategy_aggregation: str,
    runtime_df: pd.DataFrame,
    dataset_summary: pd.DataFrame,
) -> None:
    """Write a README.md file summarizing the PG-SUI runtime scaling outputs."""
    report_rows = runtime_df.loc[runtime_df["report_complete"]].copy()
    artifact_rows = runtime_df.loc[runtime_df["artifact_complete"]].copy()
    runtime_rows = coerce_runtime_rows(runtime_df)
    finite_rows = runtime_rows.loc[
        runtime_rows["report_complete"]
        & pd.to_numeric(runtime_rows["runtime_metric_seconds"], errors="coerce").gt(0)
    ]
    lines = [
        "# PG-SUI validation runtime scaling",
        "",
        f"- Validation root: `{validation_root}`",
        f"- Dataset manifest: `{dataset_manifest}`",
        f"- Strategy aggregation used for regression: `{strategy_aggregation}`",
        f"- Runtime rows with complete zygosity reports: {len(report_rows)}",
        f"- Runtime rows with complete report and imputed artifacts: {len(artifact_rows)}",
        f"- Runtime rows with parsed positive scaled execution time: {len(finite_rows)}",
        f"- Dataset-model rows used for regressions: {len(dataset_summary)}",
        f"- Datasets represented: {dataset_summary['dataset_id'].nunique() if not dataset_summary.empty else 0}",
        "",
        "Outputs:",
        "",
        "- `pgsui_validation_runtime_total_genotypes.png`",
        "- `pgsui_validation_runtime_total_genotypes.pdf`",
        "- `pgsui_validation_model_runtime_rows.csv`",
        "- `pgsui_validation_dataset_model_runtime_summary.csv`",
        "- `pgsui_validation_runtime_regression_summary.csv`",
        "",
        "Notes:",
        "",
        "- Total genotypes are `sample_size * locus_count` from the final-60 PHYLIP headers.",
        "- Deep-model plotted/regressed runtimes use logged per-trial Optuna duration (`Average trial duration` or `Avg trial (s)`), not the full 100-trial tuning wall time.",
        "- Raw deep-model fit/imputation segment times and tuning wall times are retained in `pgsui_validation_model_runtime_rows.csv` for auditability.",
        "- Deterministic runtimes are timestamp-span lower bounds from their model-specific logs because those archived logs start at fit completion.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def write_outputs(
    runtime_df: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    regressions: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Write CSV outputs for PG-SUI validation runtime scaling."""
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_df.to_csv(out_dir / "pgsui_validation_model_runtime_rows.csv", index=False)
    dataset_summary.to_csv(
        out_dir / "pgsui_validation_dataset_model_runtime_summary.csv", index=False
    )
    regressions.to_csv(
        out_dir / "pgsui_validation_runtime_regression_summary.csv", index=False
    )


def main() -> None:
    args = resolve_default_paths(parse_args())
    sizes = load_dataset_sizes(
        package_root=args.package_root,
        dataset_manifest_path=args.dataset_manifest,
        metadata_path=args.metadata,
    )
    runtime_df = collect_model_runtimes(args.validation_root, sizes)
    dataset_summary = summarize_dataset_model_runtimes(
        runtime_df,
        strategy_aggregation=args.strategy_aggregation,
    )
    regressions = regression_summary(dataset_summary)
    write_outputs(runtime_df, dataset_summary, regressions, args.output_dir)
    plot_runtime_scaling(dataset_summary, regressions, args.output_dir, dpi=args.dpi)
    write_readme(
        args.output_dir,
        validation_root=args.validation_root,
        dataset_manifest=args.dataset_manifest,
        strategy_aggregation=args.strategy_aggregation,
        runtime_df=runtime_df,
        dataset_summary=dataset_summary,
    )
    print(f"Wrote PG-SUI runtime scaling outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
