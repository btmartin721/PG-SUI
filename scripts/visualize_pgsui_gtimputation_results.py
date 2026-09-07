#!/usr/bin/env python3
"""Visualize PG-SUI and GTImputation F1 scores and execution times.

This script scores the current GTImputation bundle against the PG-SUI-generated
masked-cell truth masks, reads PG-SUI classification reports, filters incomplete
comparison grids, and writes publication-friendly plots and summary tables.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence, cast

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
(SCRIPT_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib import lines as mlines
from matplotlib import ticker as mticker
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

GENOTYPE_CLASS_COLUMN_MAP = {
    "0": "ref_f1",
    "REF": "ref_f1",
    "REFERENCE": "ref_f1",
    "HOM_REF": "ref_f1",
    "HOMOZYGOUS_REFERENCE": "ref_f1",
    "1": "het_f1",
    "HET": "het_f1",
    "HETEROZYGOUS": "het_f1",
    "2": "alt_f1",
    "ALT": "alt_f1",
    "ALTERNATE": "alt_f1",
    "HOM_ALT": "alt_f1",
    "HOMOZYGOUS_ALTERNATE": "alt_f1",
}

PGSUI_MODELS = (
    "ImputeRefAllele",
    "ImputeMostFrequent",
    "ImputeAutoencoder",
    "ImputeVAE",
    "ImputeNLPCA",
    "ImputeUBP",
)

PGSUI_DEEP_MODELS = ("ImputeAutoencoder", "ImputeVAE", "ImputeNLPCA", "ImputeUBP")

PGSUI_DETERMINISTIC_MODELS = ("ImputeRefAllele", "ImputeMostFrequent")

PGSUI_MODEL_LABELS = {
    "ImputeRefAllele": "RefAllele",
    "ImputeMostFrequent": "MostFrequent",
    "ImputeAutoencoder": "Autoencoder",
    "ImputeVAE": "VAE",
    "ImputeNLPCA": "NLPCA",
    "ImputeUBP": "UBP",
}

PGSUI_LOG_NAMES = {
    "ImputeRefAllele": "pgsui.impute.deterministic.imputers.ref_allele.log",
    "ImputeMostFrequent": "pgsui.impute.deterministic.imputers.mode.log",
    "ImputeAutoencoder": "pgsui.impute.unsupervised.imputers.ImputeAutoencoder.log",
    "ImputeVAE": "pgsui.impute.unsupervised.imputers.ImputeVAE.log",
    "ImputeNLPCA": "pgsui.impute.unsupervised.imputers.ImputeNLPCA.log",
    "ImputeUBP": "pgsui.impute.unsupervised.imputers.ImputeUBP.log",
}

GTI_METHODS = ("som", "naive")

GTI_MODEL_LABELS = {"som": "SOM (GTImputation)", "naive": "Naive (GTImputation)"}

MODEL_ORDER = (
    "Naive (GTImputation)",
    "SOM (GTImputation)",
    "RefAllele",
    "MostFrequent",
    "Autoencoder",
    "VAE",
    "NLPCA",
    "UBP",
)

STRATEGY_ORDER = (
    "random",
    "random_weighted",
    "random_weighted_inv",
    "nonrandom",
    "nonrandom_weighted",
)

STRATEGY_LABELS = {
    "random": "Random",
    "random_weighted": "Random Weighted",
    "random_weighted_inv": "Random Weighted Inverted",
    "nonrandom": "Nonrandom",
    "nonrandom_weighted": "Nonrandom Weighted",
}

STRATEGY_ABBREVIATIONS = {
    "random": "R",
    "random_weighted": "RW",
    "random_weighted_inv": "RWI",
    "nonrandom": "N",
    "nonrandom_weighted": "NW",
}

STRATEGY_ABBR_ORDER = tuple(
    STRATEGY_ABBREVIATIONS[strategy] for strategy in STRATEGY_ORDER
)

CLASS_LABELS = ("REF", "HET", "ALT")

CLASS_IDS = (0, 1, 2)

GENOTYPE_CLASS_LABELS = {
    "REF": "Homozygous Reference",
    "HET": "Heterozygous",
    "ALT": "Homozygous Alternate",
}

GENOTYPE_CLASS_ORDER = tuple(GENOTYPE_CLASS_LABELS.values())

GENOTYPE_CLASS_SHORT_LABELS = {
    "Homozygous Reference": "REF",
    "Heterozygous": "HET",
    "Homozygous Alternate": "ALT",
}

REPORT_FILENAMES = {
    "zygosity": ("classification_report_zygosity.json", "zygosity_report.json"),
    "iupac": ("classification_report_iupac.json", "iupac_report.json"),
}

RETRO_90S_PALETTE = ("#FE019A", "#18E9EE", "#5AAE00", "#FA4224", "#811BF3", "#D69E00")

MODEL_PALETTE = {
    "Naive (GTImputation)": "#9E9E9E",
    "SOM (GTImputation)": "#4D4D4D",
    **dict(
        zip((PGSUI_MODEL_LABELS[model] for model in PGSUI_MODELS), RETRO_90S_PALETTE)
    ),
}

# Four analysis families used for the head-to-head comparison.
# Grouping the eight models this way keeps the PG-SUI
# (deterministic and deep learning) versus
# GTImputation (naive and SOM deep learning) contrast legible in every figure.
MODEL_CATEGORY = {
    "Naive (GTImputation)": "GTImputation: Naive",
    "SOM (GTImputation)": "GTImputation: SOM (deep learning)",
    "RefAllele": "PG-SUI: Deterministic",
    "MostFrequent": "PG-SUI: Deterministic",
    "Autoencoder": "PG-SUI: Deep learning",
    "VAE": "PG-SUI: Deep learning",
    "NLPCA": "PG-SUI: Deep learning",
    "UBP": "PG-SUI: Deep learning",
}

CATEGORY_ORDER = (
    "GTImputation: Naive",
    "GTImputation: SOM (deep learning)",
    "PG-SUI: Deterministic",
    "PG-SUI: Deep learning",
)

CATEGORY_COLORS = {
    "GTImputation: Naive": "#B0B0B0",
    "GTImputation: SOM (deep learning)": "#4D4D4D",
    "PG-SUI: Deterministic": "#18E9EE",
    "PG-SUI: Deep learning": "#811BF3",
}

CATEGORY_SHORT_LABELS = {
    "GTImputation: Naive": "Naive",
    "GTImputation: SOM (deep learning)": "SOM",
    "PG-SUI: Deterministic": "PG-SUI Det.",
    "PG-SUI: Deep learning": "PG-SUI DL",
}

BACKEND_PRIORITY = ("cuda", "mps", "cpu", "none")

RUNTIME_PGSUI_BACKENDS = ("cpu", "cuda")

PGSUI_BACKEND_LABELS = {
    "cuda": "GPU",
    "cpu": "CPU",
    "mps": "MPS",
    "none": "No Device",
    "deterministic": "Deterministic",
    "not_applicable": "Not Applicable",
}

MISSING_GT = {"", ".", "./.", ".|.", "-", "?", "NA", "NAN", "NONE"}

GTI_METRIC_COLUMNS = (
    "dataset_id",
    "strategy",
    "strategy_label",
    "software",
    "model",
    "model_label",
    "run_id",
    "report_type",
    "runtime_seconds",
    "runtime_source",
    "optuna_elapsed_seconds",
    "optuna_rolling_avg_trial_seconds",
    "optuna_mean_trial_seconds",
    "run_mode",
    "tuning_trials_planned",
    "tuning_trials_completed",
    "parallel_jobs",
    "runtime_wall_seconds",
    "parallel_adjusted_runtime_seconds",
    "runtime_accounting",
    "report_path",
    "input_path",
    "mask_tsv",
    "truth_vcf",
    "masked_input_vcf",
    "mask_rows",
    "scored_masked_sites",
    "dropped_unmasked_mask_rows",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "mcc",
    "support",
    "n_pred_missing",
    "ref_f1",
    "ref_support",
    "het_f1",
    "het_support",
    "alt_f1",
    "alt_support",
)

GTI_CLASS_REPORT_COLUMNS = (
    "dataset_id",
    "strategy",
    "strategy_label",
    "software",
    "model",
    "model_label",
    "run_id",
    "report_type",
    "runtime_seconds",
    "runtime_source",
    "optuna_elapsed_seconds",
    "optuna_rolling_avg_trial_seconds",
    "optuna_mean_trial_seconds",
    "run_mode",
    "tuning_trials_planned",
    "tuning_trials_completed",
    "parallel_jobs",
    "runtime_wall_seconds",
    "parallel_adjusted_runtime_seconds",
    "runtime_accounting",
    "report_path",
    "input_path",
    "mask_tsv",
    "truth_vcf",
    "masked_input_vcf",
    "mask_rows",
    "scored_masked_sites",
    "dropped_unmasked_mask_rows",
    "class_label",
    "genotype_class",
    "precision",
    "recall",
    "f1",
    "support",
)

GTI_ERROR_COLUMNS = ("dataset_id", "strategy", "method", "run_id", "error")

PGSUI_METRIC_COLUMNS = (
    "dataset_id",
    "strategy",
    "strategy_label",
    "software",
    "model",
    "model_label",
    "backend",
    "output_dir",
    "report_type",
    "runtime_seconds",
    "runtime_source",
    "optuna_elapsed_seconds",
    "optuna_rolling_avg_trial_seconds",
    "optuna_mean_trial_seconds",
    "run_mode",
    "tuning_trials_planned",
    "tuning_trials_completed",
    "parallel_jobs",
    "runtime_wall_seconds",
    "parallel_adjusted_runtime_seconds",
    "runtime_accounting",
    "report_path",
    "input_path",
    "mask_tsv",
    "truth_vcf",
    "masked_input_vcf",
    "mask_rows",
    "scored_masked_sites",
    "dropped_unmasked_mask_rows",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "mcc",
    "support",
    "n_pred_missing",
    "ref_f1",
    "ref_support",
    "het_f1",
    "het_support",
    "alt_f1",
    "alt_support",
)

PGSUI_CLASS_REPORT_COLUMNS = (
    "dataset_id",
    "strategy",
    "strategy_label",
    "software",
    "model",
    "model_label",
    "backend",
    "output_dir",
    "report_type",
    "runtime_seconds",
    "runtime_source",
    "optuna_elapsed_seconds",
    "optuna_rolling_avg_trial_seconds",
    "optuna_mean_trial_seconds",
    "run_mode",
    "tuning_trials_planned",
    "tuning_trials_completed",
    "parallel_jobs",
    "runtime_wall_seconds",
    "parallel_adjusted_runtime_seconds",
    "runtime_accounting",
    "report_path",
    "input_path",
    "mask_tsv",
    "truth_vcf",
    "masked_input_vcf",
    "mask_rows",
    "scored_masked_sites",
    "dropped_unmasked_mask_rows",
    "class_label",
    "genotype_class",
    "precision",
    "recall",
    "f1",
    "support",
)

PGSUI_ERROR_COLUMNS = (
    "dataset_id",
    "strategy",
    "backend",
    "model",
    "output_dir",
    "error",
)


@dataclass(frozen=True)
class VCFMatrix:
    path: Path
    samples: list[str]
    variant_keys: list[tuple[str, str, str, str]]
    zygosity: np.ndarray
    sample_index: Mapping[str, int]
    variant_index: Mapping[tuple[str, str, str, str], int]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for visualizing PG-SUI and GTImputation results."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize complete PG-SUI versus GTImputation grids using zygosity macro F1 and execution-time plots."
        )
    )
    parser.add_argument(
        "--pgsui-dir",
        type=Path,
        default=Path("/Users/btm002/PG-SUI/results-pgsui"),
        help="Root containing PG-SUI *_output directories.",
    )
    parser.add_argument(
        "--gti-dir",
        type=Path,
        default=Path("/Users/btm002/PG-SUI/results-gti"),
        help="Root containing GTImputation result bundle.",
    )
    parser.add_argument(
        "--sim-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "gtimputation-results"
        / "zygosity_missingness_simulations"
        / "manifest.csv",
        help="Manifest produced by simulate_gtimputation_missingness.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "gtimputation-results"
        / "pgsui_gtimputation_current_comparison",
    )
    parser.add_argument(
        "--report-type",
        choices=("zygosity",),
        default="zygosity",
        help="PG-SUI report type to compare. GTImputation scoring is zygosity-based.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGY_ORDER,
        default=list(STRATEGY_ORDER),
        help="Simulation strategies required for complete grids.",
    )
    parser.add_argument(
        "--pgsui-models",
        nargs="+",
        choices=PGSUI_MODELS,
        default=list(PGSUI_MODELS),
        help="PG-SUI models required for a complete PG-SUI run.",
    )
    parser.add_argument(
        "--gti-methods",
        nargs="+",
        choices=GTI_METHODS,
        default=list(GTI_METHODS),
        help="GTImputation methods required for a complete GTImputation run.",
    )
    parser.add_argument(
        "--pgsui-backend",
        choices=("auto", "cuda", "mps", "cpu", "none"),
        default="cuda",
        help=(
            "PG-SUI backend used for F1-score comparisons. Default cuda prevents "
            "CPU/GPU duplicate F1 rows."
        ),
    )
    parser.add_argument(
        "--runtime-pgsui-backends",
        nargs="+",
        choices=("cuda", "mps", "cpu", "none"),
        default=list(RUNTIME_PGSUI_BACKENDS),
        help="PG-SUI backends to include for deep-model runtime comparisons.",
    )
    parser.add_argument(
        "--backend-priority",
        nargs="+",
        default=list(BACKEND_PRIORITY),
        help="Backend preference used when --pgsui-backend auto.",
    )
    parser.add_argument(
        "--allow-partial-grid",
        action="store_true",
        help="Include complete dataset-strategy pairs even if not all strategies are complete.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Write tables only.")
    parser.add_argument(
        "--skip-gti-scoring",
        action="store_true",
        help="Skip VCF scoring and write completion tables only.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Plot export DPI.")
    return parser.parse_args()


def _get_contrasting_text_color(
    rgba: tuple[float, float, float, float] | np.ndarray,
) -> str:
    """Return black or white based on contrast with an RGBA background.

    Uses WCAG-style relative luminance after converting sRGB values to linear RGB.

    Args:
        rgba: Background color containing red, green, blue, and optionally alpha components, with values in the range [0, 1].

    Returns:
        Either "black" or "white", whichever provides greater contrast.
    """
    rgb = np.asarray(rgba[:3], dtype=float)

    linear_rgb = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    luminance = np.dot(linear_rgb, [0.2126, 0.7152, 0.0722])

    black_contrast = (luminance + 0.05) / 0.05
    white_contrast = 1.05 / (luminance + 0.05)

    return "white" if white_contrast >= black_contrast else "black"


def _canonicalize_genotype_class(genotype_class: pd.Series) -> pd.Series:
    """Map genotype-class labels to their output F1-score columns.

    Args:
        genotype_class: Genotype-class labels from a classification report.

    Returns:
        Series containing ``ref_f1``, ``het_f1``, or ``alt_f1`` for
        recognized genotype classes. Unrecognized classes are returned as
        missing values.
    """
    normalized = (
        genotype_class.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return normalized.map(GENOTYPE_CLASS_COLUMN_MAP)


def _first_common_column(
    left: pd.DataFrame,
    right: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str:
    """Return the first candidate column shared by two DataFrames.

    Args:
        left: First DataFrame.
        right: Second DataFrame.
        candidates: Candidate column names in priority order.

    Returns:
        The first column present in both DataFrames.

    Raises:
        ValueError: If no candidate column is shared.
    """
    for column in candidates:
        if column in left.columns and column in right.columns:
            return column

    raise ValueError(
        f"No common column found. Expected one of: {', '.join(candidates)}"
    )


def resolve_gti_dir(gti_dir: Path) -> Path:
    gti_dir = gti_dir.expanduser().resolve()
    if gti_dir.exists():
        return gti_dir
    raise FileNotFoundError(f"GTImputation directory not found: {gti_dir}")


def canonical_strategy(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if normalized in STRATEGY_ORDER:
        return normalized
    compact = normalized.replace("_", "")
    if "nonrandom" in compact and "weighted" in compact:
        return "nonrandom_weighted"
    if "nonrandom" in compact:
        return "nonrandom"
    if "random" in compact and "weighted" in compact and "inv" in compact:
        return "random_weighted_inv"
    if "random" in compact and "weighted" in compact:
        return "random_weighted"
    if "random" in compact:
        return "random"
    return normalized


def parse_pgsui_output_name(path: Path) -> tuple[str, str, str] | None:
    if not path.name.endswith("_output"):
        return None
    stem = path.name[: -len("_output")]
    for strategy in sorted(STRATEGY_ORDER, key=len, reverse=True):
        pattern = rf"^(.+)_({re.escape(strategy)})(?:_(.+))?$"
        match = re.match(pattern, stem)
        if match:
            backend = match.group(3) or "none"
            return match.group(1), strategy, backend
    return None


def as_float(value) -> float:
    if value is None:
        return math.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_pgsui_report(path: Path) -> dict[str, float]:
    """Load a PG-SUI classification report JSON and return a flattened metric dictionary."""
    report = json.loads(path.read_text())
    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})
    row = {
        "accuracy": as_float(report.get("accuracy")),
        "macro_precision": as_float(macro.get("precision")),
        "macro_recall": as_float(macro.get("recall")),
        "macro_f1": as_float(macro.get("f1-score")),
        "weighted_f1": as_float(weighted.get("f1-score")),
        "support": as_float(macro.get("support")),
        "mcc": as_float(report.get("mcc")),
    }
    for class_label in CLASS_LABELS:
        class_values = report.get(class_label, {})
        row[f"{class_label.lower()}_f1"] = as_float(class_values.get("f1-score"))
        row[f"{class_label.lower()}_support"] = as_float(class_values.get("support"))
    return row


def load_pgsui_class_report(path: Path) -> list[dict[str, object]]:
    """Load a PG-SUI classification report JSON and return a list of class-specific metric dictionaries."""
    report = json.loads(path.read_text())
    class_rows: list[dict[str, object]] = []
    for class_label in CLASS_LABELS:
        class_values = report.get(class_label, {})
        class_rows.append(
            {
                "class_label": class_label,
                "genotype_class": GENOTYPE_CLASS_LABELS[class_label],
                "precision": as_float(class_values.get("precision")),
                "recall": as_float(class_values.get("recall")),
                "f1": as_float(class_values.get("f1-score")),
                "support": as_float(class_values.get("support")),
            }
        )
    return class_rows


def find_model_report(output_dir: Path, model: str, report_type: str) -> Path | None:
    """Find the PG-SUI classification report JSON for a given model and report type."""
    wanted = set(REPORT_FILENAMES[report_type])
    matches = [
        path
        for path in output_dir.glob(f"*/metrics/{model}/*.json")
        if path.name in wanted
    ]
    return sorted(matches)[0] if matches else None


def find_model_imputed_vcf(output_dir: Path, model: str) -> Path | None:
    """Find the imputed VCF file for a given model."""
    matches = sorted(
        path
        for path in output_dir.glob(f"*/imputed/{model}/*")
        if path.name.endswith((".vcf", ".vcf.gz"))
    )
    if matches:
        return matches[0]

    model_slug = model.lower()
    fallback_matches = sorted(
        path
        for path in output_dir.rglob(f"{model_slug}*imputed*.vcf*")
        if path.is_file()
    )
    return fallback_matches[0] if fallback_matches else None


def parse_duration_text(value: str) -> float | None:
    """Parse a duration string and return the total duration in seconds."""
    value = value.strip()
    hms_match = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{2}(?:\.\d+)?)", value)
    if hms_match:
        hours = float(hms_match.group(1) or 0)
        minutes = float(hms_match.group(2))
        seconds = float(hms_match.group(3))
        return hours * 3600.0 + minutes * 60.0 + seconds

    total = 0.0
    matched = False
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", value.lower()):
        matched = True
        scalar = {"h": 3600.0, "m": 60.0, "s": 1.0}[unit]
        total += float(amount) * scalar
    return total if matched else None


def parse_log_timestamps(text: str) -> list[datetime]:
    """Parse log timestamps from a text string and return a list of datetime objects."""
    timestamps: list[datetime] = []
    for match in re.finditer(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text, re.MULTILINE
    ):
        try:
            timestamps.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            continue
    return timestamps


def parse_pgsui_runtime(output_dir: Path, model: str) -> dict[str, object]:
    """Parse the PG-SUI runtime information from the log file for a given model."""
    log_path = output_dir / "logs" / PGSUI_LOG_NAMES[model]
    if not log_path.exists():
        return {
            "runtime_seconds": math.nan,
            "runtime_source": "missing_log",
            "optuna_elapsed_seconds": math.nan,
            "optuna_rolling_avg_trial_seconds": math.nan,
            "optuna_mean_trial_seconds": math.nan,
            "optuna_completed_trials": math.nan,
            "optuna_planned_trials": math.nan,
            "optuna_parallel_jobs": math.nan,
            "log_path": "",
        }

    text = log_path.read_text(errors="replace")
    elapsed_matches = re.findall(r"elapsed_s=([0-9]+(?:\.[0-9]+)?)", text)
    rolling_avg_matches = re.findall(
        r"Avg trial \(s\)\s*=\s*([0-9]+(?:\.[0-9]+)?)", text
    )
    mean_trial_matches = re.findall(
        r"mean trial time\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE
    )
    completed_matches = re.findall(r"completed=(\d+)/(\d+)", text)
    planned_matches = re.findall(
        r"planned trials\s*:\s*(\d+)", text, flags=re.IGNORECASE
    )
    n_jobs_matches = re.findall(r"n_jobs\s*:\s*(\d+)", text, flags=re.IGNORECASE)
    if elapsed_matches:
        completed, planned = (math.nan, math.nan)
        if completed_matches:
            completed, planned = completed_matches[-1]
        elif planned_matches:
            planned = planned_matches[-1]
        elapsed_seconds = float(elapsed_matches[-1])
        rolling_avg_trial_seconds = (
            float(rolling_avg_matches[-1]) if rolling_avg_matches else math.nan
        )
        mean_trial_seconds = (
            parse_duration_text(mean_trial_matches[-1]) if mean_trial_matches else None
        )
        if model in PGSUI_DEEP_MODELS and np.isfinite(rolling_avg_trial_seconds):
            runtime_seconds = rolling_avg_trial_seconds
            runtime_source = "optuna_rolling_avg_trial_s"
        elif model in PGSUI_DEEP_MODELS and mean_trial_seconds is not None:
            runtime_seconds = mean_trial_seconds
            runtime_source = "optuna_mean_trial_time"
        else:
            runtime_seconds = elapsed_seconds
            runtime_source = "optuna_elapsed_s"
        return {
            "runtime_seconds": runtime_seconds,
            "runtime_source": runtime_source,
            "optuna_elapsed_seconds": elapsed_seconds,
            "optuna_rolling_avg_trial_seconds": rolling_avg_trial_seconds,
            "optuna_mean_trial_seconds": (
                mean_trial_seconds if mean_trial_seconds is not None else math.nan
            ),
            "optuna_completed_trials": as_float(completed),
            "optuna_planned_trials": as_float(planned),
            "optuna_parallel_jobs": (
                as_float(n_jobs_matches[-1]) if n_jobs_matches else math.nan
            ),
            "log_path": str(log_path),
        }

    wall_matches = re.findall(r"wall time\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if wall_matches:
        parsed = parse_duration_text(wall_matches[-1])
        if parsed is not None:
            return {
                "runtime_seconds": parsed,
                "runtime_source": "wall_time",
                "optuna_elapsed_seconds": math.nan,
                "optuna_rolling_avg_trial_seconds": math.nan,
                "optuna_mean_trial_seconds": math.nan,
                "optuna_completed_trials": math.nan,
                "optuna_planned_trials": math.nan,
                "optuna_parallel_jobs": math.nan,
                "log_path": str(log_path),
            }

    timestamps = parse_log_timestamps(text)
    if len(timestamps) >= 2:
        elapsed = max((timestamps[-1] - timestamps[0]).total_seconds(), 0.0)
        return {
            "runtime_seconds": elapsed,
            "runtime_source": "log_span",
            "optuna_elapsed_seconds": math.nan,
            "optuna_rolling_avg_trial_seconds": math.nan,
            "optuna_mean_trial_seconds": math.nan,
            "optuna_completed_trials": math.nan,
            "optuna_planned_trials": math.nan,
            "optuna_parallel_jobs": math.nan,
            "log_path": str(log_path),
        }

    return {
        "runtime_seconds": math.nan,
        "runtime_source": "unparsed_log",
        "optuna_elapsed_seconds": math.nan,
        "optuna_rolling_avg_trial_seconds": math.nan,
        "optuna_mean_trial_seconds": math.nan,
        "optuna_completed_trials": math.nan,
        "optuna_planned_trials": math.nan,
        "optuna_parallel_jobs": math.nan,
        "log_path": str(log_path),
    }


def finite_or_default(value: object, default: float) -> float:
    """Return the value if it is finite, otherwise return the default."""
    parsed = as_float(value)
    return parsed if np.isfinite(parsed) else default


def pgsui_runtime_metadata(
    model: str, runtime: Mapping[str, object]
) -> dict[str, object]:
    """Return a dictionary of runtime metadata for a given PG-SUI model and runtime information.

    Args:
        model (str): The PG-SUI model name.
        runtime (Mapping[str, object]): A mapping containing runtime information.

    Returns:
        dict[str, object]: A dictionary containing runtime metadata, including run mode, tuning trials, parallel jobs, wall-clock time, and runtime accounting.
    """
    runtime_seconds = as_float(runtime.get("runtime_seconds"))
    if model in PGSUI_DEEP_MODELS:
        planned_trials = finite_or_default(runtime.get("optuna_planned_trials"), 100.0)
        completed_trials = finite_or_default(
            runtime.get("optuna_completed_trials"), planned_trials
        )
        parallel_jobs = finite_or_default(runtime.get("optuna_parallel_jobs"), 2.0)
        wall_seconds = finite_or_default(
            runtime.get("optuna_elapsed_seconds"), runtime_seconds
        )
        return {
            "run_mode": "optuna_tuned",
            "tuning_trials_planned": planned_trials,
            "tuning_trials_completed": completed_trials,
            "parallel_jobs": parallel_jobs,
            "runtime_wall_seconds": wall_seconds,
            "parallel_adjusted_runtime_seconds": (
                wall_seconds * parallel_jobs if np.isfinite(wall_seconds) else math.nan
            ),
            "runtime_accounting": (
                "Primary runtime is the latest logged Optuna rolling average per trial; "
                "runtime_wall_seconds retains the full Optuna tuning wall-clock time. "
                "Parallel-adjusted runtime is wall time multiplied by logged n_jobs, "
                "with a fallback of 2 if n_jobs is absent."
            ),
        }

    return {
        "run_mode": "single_run_no_tuning",
        "tuning_trials_planned": 1.0,
        "tuning_trials_completed": 1.0,
        "parallel_jobs": 1.0,
        "runtime_wall_seconds": runtime_seconds,
        "parallel_adjusted_runtime_seconds": runtime_seconds,
        "runtime_accounting": "Single deterministic imputation run; no Optuna tuning.",
    }


def gti_runtime_metadata(
    row: pd.Series, runtime_seconds: float, runtime_source: str
) -> dict[str, object]:
    """Return a dictionary of runtime metadata for a given GTImputation run.

    Args:
        row (pd.Series): A pandas Series containing GTImputation run information.
        runtime_seconds (float): The runtime in seconds for the GTImputation run.
        runtime_source (str): The source of the runtime information (e.g., 'log', 'wall_time').

    Returns:
        dict[str, object]: A dictionary containing runtime metadata, including run mode, tuning trials, parallel jobs, wall-clock time, and runtime accounting.
    """
    method = str(row.get("method", "")).lower()
    if method == "som":
        accounting = (
            "Single SOM run; runtime includes genotype-database build plus imputation."
        )
    else:
        accounting = "Single Naive run; runtime is imputation phase only."
    return {
        "run_mode": "single_run_no_tuning",
        "tuning_trials_planned": 1.0,
        "tuning_trials_completed": 1.0,
        "parallel_jobs": 1.0,
        "runtime_wall_seconds": runtime_seconds,
        "parallel_adjusted_runtime_seconds": runtime_seconds,
        "runtime_accounting": f"{accounting} Source={runtime_source}.",
    }


def collect_pgsui_outputs(
    pgsui_dir: Path,
    models: Sequence[str],
    strategies: Sequence[str],
    report_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect PG-SUI output directories and parse classification reports.

    Args:
        pgsui_dir (Path): The root directory containing PG-SUI output directories.
        models (Sequence[str]): A sequence of PG-SUI model names to include.
        strategies (Sequence[str]): A sequence of simulation strategies to include.
        report_type (str): The type of PG-SUI report to parse (e.g., 'zygosity').

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas DataFrames:
            - status_df: A DataFrame with the status of each PG-SUI run.
            - metrics_df: A DataFrame with the metrics from each PG-SUI report.
    """
    pgsui_dir = pgsui_dir.expanduser().resolve()

    if not pgsui_dir.exists():
        raise FileNotFoundError(f"PG-SUI results directory not found: {pgsui_dir}")

    status_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    strategy_set = set(strategies)

    for output_dir in sorted(path for path in pgsui_dir.iterdir() if path.is_dir()):
        parsed = parse_pgsui_output_name(output_dir)
        if parsed is None:
            continue
        dataset_id, strategy, backend = parsed
        if strategy not in strategy_set:
            continue

        for model in models:
            report_path = find_model_report(output_dir, model, report_type)
            imputed_vcf_path = find_model_imputed_vcf(output_dir, model)
            runtime = parse_pgsui_runtime(output_dir, model)
            row_base = {
                "dataset_id": dataset_id,
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                "backend": backend,
                "output_dir": str(output_dir),
                "model": model,
                "model_label": PGSUI_MODEL_LABELS.get(model, model),
                "report_type": report_type,
                "software": "PG-SUI",
                "input_path": str(imputed_vcf_path) if imputed_vcf_path else "",
                "scoring_source": "pgsui_classification_report",
                **runtime,
                **pgsui_runtime_metadata(model, runtime),
            }
            status_rows.append(
                {
                    **row_base,
                    "report_path": str(report_path) if report_path else "",
                    "imputed_vcf_path": (
                        str(imputed_vcf_path) if imputed_vcf_path else ""
                    ),
                    "model_complete": bool(report_path and imputed_vcf_path),
                }
            )
            if report_path is None:
                continue
            metric_rows.append(
                {
                    **row_base,
                    "report_path": str(report_path),
                    **load_pgsui_report(report_path),
                }
            )

    status_df = pd.DataFrame(status_rows)
    metrics_df = pd.DataFrame(metric_rows)
    if status_df.empty:
        raise FileNotFoundError(f"No PG-SUI output directories found under {pgsui_dir}")
    return status_df, metrics_df


def backend_rank(backend: str, priority: Sequence[str]) -> int:
    try:
        return list(priority).index(backend)
    except ValueError:
        return len(priority) + 1


def complete_pgsui_runs(status_df: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    required_model_count = len(set(models))
    complete_runs = (
        status_df.groupby(
            ["dataset_id", "strategy", "backend", "output_dir"], as_index=False
        )
        .agg(
            complete_model_count=("model_complete", "sum"),
            model_count=("model", "nunique"),
            missing_models=(
                "model",
                lambda values: ",".join(sorted(set(models) - set(values))),
            ),
        )
        .assign(
            run_complete=lambda frame: frame["complete_model_count"].eq(
                required_model_count
            )
        )
    )
    return complete_runs.loc[complete_runs["run_complete"]].copy()


def require_all_strategies(
    selected: pd.DataFrame,
    strategies: Sequence[str],
    *,
    group_cols: Sequence[str] = ("dataset_id",),
) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    strategy_set = set(strategies)
    complete_datasets = (
        selected.groupby(list(group_cols))["strategy"]
        .agg(lambda values: strategy_set.issubset(set(values)))
        .rename("has_all_strategies")
        .reset_index()
    )
    selected = selected.merge(complete_datasets, on=list(group_cols), how="left")
    return selected.loc[selected["has_all_strategies"]].drop(
        columns=["has_all_strategies"]
    )


def select_complete_pgsui_runs(
    status_df: pd.DataFrame,
    *,
    models: Sequence[str],
    strategies: Sequence[str],
    backend: str,
    backend_priority: Sequence[str],
    allow_partial_grid: bool,
) -> pd.DataFrame:
    complete_runs = complete_pgsui_runs(status_df, models)
    if backend != "auto":
        complete_runs = complete_runs.loc[complete_runs["backend"].eq(backend)].copy()
    complete_runs["backend_rank"] = complete_runs["backend"].map(
        lambda value: backend_rank(str(value), backend_priority)
    )
    complete_runs = complete_runs.sort_values(
        ["dataset_id", "strategy", "backend_rank", "output_dir"]
    )
    selected = complete_runs.groupby(["dataset_id", "strategy"], as_index=False).first()

    if allow_partial_grid:
        return selected

    return require_all_strategies(selected, strategies)


def select_complete_pgsui_backend_runs(
    status_df: pd.DataFrame,
    *,
    models: Sequence[str],
    strategies: Sequence[str],
    backends: Sequence[str],
    allow_partial_grid: bool,
) -> pd.DataFrame:
    complete_runs = complete_pgsui_runs(status_df, models)
    complete_runs = complete_runs.loc[
        complete_runs["backend"].isin(set(backends))
    ].copy()
    complete_runs = complete_runs.sort_values(
        ["dataset_id", "strategy", "backend", "output_dir"]
    )
    if allow_partial_grid:
        return complete_runs
    return require_all_strategies(
        complete_runs, strategies, group_cols=("dataset_id", "backend")
    )


def load_gti_manifest(
    gti_dir: Path, methods: Sequence[str], strategies: Sequence[str]
) -> pd.DataFrame:
    manifest_path = gti_dir / "metadata" / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"GTImputation manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {
        "run_id",
        "method",
        "dataset_name",
        "simulation_strategy",
        "imputation_log_status",
        "copied_vcf",
        "candidate_vcf_filename",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"GTImputation manifest missing required columns: {sorted(missing)}"
        )

    manifest = manifest.copy()
    manifest["method"] = manifest["method"].astype(str).str.lower()
    manifest["dataset_id"] = manifest["dataset_name"].astype(str)
    manifest["strategy"] = manifest["simulation_strategy"].map(canonical_strategy)
    manifest = manifest.loc[
        manifest["method"].isin(methods)
        & manifest["strategy"].isin(strategies)
        & manifest["imputation_log_status"].astype(str).str.upper().eq("OK")
    ].copy()
    if "gtdb_log_status" in manifest.columns:
        som_has_ok_gtdb = (
            manifest["gtdb_log_status"].fillna("OK").astype(str).str.upper().eq("OK")
        )
        manifest = manifest.loc[manifest["method"].ne("som") | som_has_ok_gtdb].copy()
    return manifest


def complete_gti_keys(
    gti_manifest: pd.DataFrame, methods: Sequence[str]
) -> pd.DataFrame:
    required_methods = set(methods)
    grouped = (
        gti_manifest.groupby(["dataset_id", "strategy"], as_index=False)["method"]
        .agg(lambda values: required_methods.issubset(set(values)))
        .rename(columns={"method": "gti_complete"})  # type: ignore
    )
    return grouped.loc[grouped["gti_complete"]].copy()


def select_retained_keys(
    selected_pgsui: pd.DataFrame,
    gti_manifest: pd.DataFrame,
    strategies: Sequence[str],
    methods: Sequence[str],
    allow_partial_grid: bool,
) -> pd.DataFrame:
    gti_complete = complete_gti_keys(gti_manifest, methods)
    retained = selected_pgsui.merge(
        gti_complete, on=["dataset_id", "strategy"], how="inner"
    )
    if allow_partial_grid:
        return retained

    strategy_set = set(strategies)
    complete_datasets = (
        retained.groupby("dataset_id")["strategy"]
        .agg(lambda values: strategy_set.issubset(set(values)))
        .rename("has_all_strategies")
        .reset_index()
    )
    retained = retained.merge(complete_datasets, on="dataset_id", how="left")
    return retained.loc[retained["has_all_strategies"]].drop(
        columns=["has_all_strategies"]
    )


def open_text(path: Path):
    with path.open("rb") as handle:
        is_gzip = handle.read(2) == b"\x1f\x8b"
    if is_gzip:
        return gzip.open(path, "rt")
    return path.open()


def gt_to_zygosity(value: str) -> int:
    genotype = str(value).split(":", 1)[0].strip()
    if genotype.upper() in MISSING_GT:
        return -1
    alleles = re.split(r"[\/|]", genotype)
    if not alleles or any(allele == "." for allele in alleles):
        return -1
    if len(set(alleles)) > 1:
        return 1
    return 0 if alleles[0] == "0" else 2


def load_vcf_matrix(path: Path) -> VCFMatrix:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"VCF not found: {path}")

    samples: list[str] = []
    variant_keys: list[tuple[str, str, str, str]] = []
    rows: list[list[int]] = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("##") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                samples = fields[9:]
                continue
            if not samples:
                raise ValueError(
                    f"VCF header with sample columns not found before records: {path}"
                )
            if len(fields) < 10:
                continue
            variant_keys.append((fields[0], fields[1], fields[3], fields[4]))
            rows.append([gt_to_zygosity(value) for value in fields[9:]])

    if not samples or not rows:
        raise ValueError(f"No genotype records loaded from VCF: {path}")
    matrix = np.asarray(rows, dtype=np.int8)
    return VCFMatrix(
        path=path,
        samples=samples,
        variant_keys=variant_keys,
        zygosity=matrix,
        sample_index={sample: idx for idx, sample in enumerate(samples)},
        variant_index={key: idx for idx, key in enumerate(variant_keys)},
    )


def resolve_manifest_path(raw_path, fallback: Path | None = None) -> Path:
    path_text = "" if pd.isna(raw_path) else str(raw_path)
    path = Path(path_text).expanduser()
    if path.exists():
        return path.resolve()
    if fallback is not None and fallback.exists():
        return fallback.resolve()
    return path


def resolve_gti_vcf(row: pd.Series, gti_dir: Path) -> Path:
    copied = resolve_manifest_path(row.get("copied_vcf", ""))
    if copied.exists():
        return copied
    candidate = str(row.get("candidate_vcf_filename", ""))
    fallback = gti_dir / "vcfs" / str(row["method"]) / candidate
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(
        f"Could not resolve GTImputation VCF for run {row['run_id']}"
    )


def resolve_truth_vcf(sim_row: pd.Series, sim_manifest_path: Path) -> Path:
    input_vcf = resolve_manifest_path(sim_row.get("input_vcf", ""))
    if input_vcf.exists():
        return input_vcf

    dataset_id = str(sim_row["dataset_id"])
    cache_dir = sim_manifest_path.parent / ".snpio_cache" / dataset_id
    candidates = (
        cache_dir / f"{dataset_id}.nremover.vcf_sorted.vcf.gz",
        cache_dir / f"{dataset_id}.nremover.vcf.gz",
        cache_dir / f"{dataset_id}.vcf.gz",
        cache_dir / f"{dataset_id}.vcf",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Truth VCF missing for {dataset_id}. Tried manifest input_vcf and {cache_dir}"
    )


def resolve_mask_tsv(sim_row: pd.Series, sim_manifest_path: Path) -> Path:
    mask = resolve_manifest_path(sim_row.get("mask_tsv", ""))
    if mask.exists():
        return mask
    dataset_id = str(sim_row["dataset_id"])
    strategy = str(sim_row["strategy"])
    fallback = (
        sim_manifest_path.parent
        / dataset_id
        / strategy
        / "masks"
        / f"{dataset_id}__sim30__{strategy}__seed42.mask.tsv"
    )
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"Mask TSV missing for {dataset_id}/{strategy}: {mask}")


def resolve_masked_input_vcf(sim_row: pd.Series, sim_manifest_path: Path) -> Path:
    output_vcf = resolve_manifest_path(sim_row.get("output_vcf", ""))
    if output_vcf.exists():
        return output_vcf
    dataset_id = str(sim_row["dataset_id"])
    strategy = str(sim_row["strategy"])
    fallback = (
        sim_manifest_path.parent
        / dataset_id
        / strategy
        / f"{dataset_id}__sim30__{strategy}__seed42.vcf"
    )
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(
        f"Masked input VCF missing for {dataset_id}/{strategy}: {output_vcf}"
    )


def load_sim_manifest(path: Path, strategies: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Simulation manifest not found: {path}")
    manifest = pd.read_csv(path)
    required = {"dataset_id", "strategy", "input_vcf", "output_vcf", "mask_tsv"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Simulation manifest missing required columns: {sorted(missing)}"
        )
    manifest = manifest.copy()
    manifest["strategy"] = manifest["strategy"].map(canonical_strategy)
    return manifest.loc[manifest["strategy"].isin(strategies)].copy()


def attach_simulation_size_metadata(
    df: pd.DataFrame, sim_manifest: pd.DataFrame
) -> pd.DataFrame:
    """Attach per-dataset sample and locus counts from the simulation manifest."""
    if df.empty or sim_manifest.empty:
        return df.copy()

    key_cols = ["dataset_id", "strategy"]
    metadata_cols = [
        column
        for column in (
            "n_samples",
            "n_loci",
            "n_cells",
            "n_original_missing",
            "n_simulated_missing",
            "simulated_missing_rate",
        )
        if column in sim_manifest.columns
    ]
    if not metadata_cols or not set(key_cols).issubset(df.columns):
        return df.copy()

    metadata = (
        sim_manifest[key_cols + metadata_cols]
        .drop_duplicates(subset=key_cols)
        .reset_index(drop=True)
    )
    out = df.drop(columns=[column for column in metadata_cols if column in df.columns])
    return out.merge(metadata, on=key_cols, how="left")


def vcf_indices_from_mask(
    mask: pd.DataFrame, vcf: VCFMatrix
) -> tuple[np.ndarray, np.ndarray]:
    sample_indices = mask["sample_id"].map(vcf.sample_index)
    if sample_indices.isna().any():
        missing_samples = sorted(
            mask.loc[sample_indices.isna(), "sample_id"].astype(str).unique()
        )
        raise ValueError(f"VCF {vcf.path} missing mask samples: {missing_samples[:8]}")

    locus_indices = mask["locus_index"].astype(int).to_numpy()
    if locus_indices.size and int(locus_indices.max()) < len(vcf.variant_keys):
        sample_keys = mask[["chrom", "pos", "ref", "alt", "locus_index"]].head(25)
        index_matches = True
        for _, row in sample_keys.iterrows():
            key = (str(row["chrom"]), str(row["pos"]), str(row["ref"]), str(row["alt"]))
            if vcf.variant_keys[int(row["locus_index"])] != key:
                index_matches = False
                break
        if index_matches:
            return locus_indices, sample_indices.astype(int).to_numpy()

    variant_keys = list(
        zip(
            mask["chrom"].astype(str),
            mask["pos"].astype(str),
            mask["ref"].astype(str),
            mask["alt"].astype(str),
        )
    )
    mapped = pd.Series(variant_keys).map(vcf.variant_index)
    if mapped.isna().any():
        missing_count = int(mapped.isna().sum())
        raise ValueError(f"VCF {vcf.path} missing {missing_count} masked variants")
    return mapped.astype(int).to_numpy(), sample_indices.astype(int).to_numpy()


def mask_actual_missing_sites(
    mask: pd.DataFrame,
    sim_row: pd.Series,
    *,
    sim_manifest_path: Path,
    masked_cache: dict[Path, VCFMatrix],
) -> tuple[pd.DataFrame, dict[str, object]]:
    masked_input_path = resolve_masked_input_vcf(sim_row, sim_manifest_path)
    if masked_input_path not in masked_cache:
        masked_cache[masked_input_path] = load_vcf_matrix(masked_input_path)
    masked_vcf = masked_cache[masked_input_path]
    mask_loci, mask_samples = vcf_indices_from_mask(mask, masked_vcf)
    actual_missing = masked_vcf.zygosity[mask_loci, mask_samples] < 0
    filtered_mask = mask.loc[actual_missing].reset_index(drop=True)
    if filtered_mask.empty:
        raise ValueError(
            f"No actually missing genotypes found in masked input VCF for "
            f"{sim_row['dataset_id']}/{sim_row['strategy']}"
        )
    stats = {
        "masked_input_vcf": str(masked_input_path),
        "mask_rows": int(len(mask)),
        "scored_masked_sites": int(len(filtered_mask)),
        "dropped_unmasked_mask_rows": int(len(mask) - len(filtered_mask)),
    }
    return filtered_mask, stats


def score_vectors(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[dict[str, object], list[dict[str, object]]]:
    valid = y_true >= 0
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_true.size == 0:
        raise ValueError("No non-missing truth genotypes available for scoring")

    precision, recall, f1_score, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(CLASS_IDS),
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(CLASS_IDS),
            average="weighted",
            zero_division=0,
        )
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1_score),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "support": float(y_true.size),
        "n_pred_missing": int(np.count_nonzero(y_pred < 0)),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=list(CLASS_IDS),
        target_names=list(CLASS_LABELS),
        output_dict=True,
        zero_division=0,
    )

    if not isinstance(report, dict):
        raise ValueError(
            f"Expected classification_report to return a dict, got {type(report)}"
        )

    report = cast(dict, report)

    class_rows: list[dict[str, object]] = []
    for class_label in CLASS_LABELS:
        class_values = report.get(class_label, {})

        metrics[f"{class_label.lower()}_f1"] = as_float(class_values.get("f1-score"))
        metrics[f"{class_label.lower()}_support"] = as_float(
            class_values.get("support")
        )
        class_rows.append(
            {
                "class_label": class_label,
                "genotype_class": GENOTYPE_CLASS_LABELS[class_label],
                "precision": as_float(class_values.get("precision")),
                "recall": as_float(class_values.get("recall")),
                "f1": as_float(class_values.get("f1-score")),
                "support": as_float(class_values.get("support")),
            }
        )
    return metrics, class_rows


def gti_runtime_seconds(row: pd.Series) -> tuple[float, str]:
    method = str(row.get("method", "")).lower()
    imputation_seconds = as_float(row.get("imputation_phase_real_seconds"))
    gtdb_seconds = as_float(row.get("gtdb_build_real_seconds"))

    if (
        method == "som"
        and np.isfinite(imputation_seconds)
        and np.isfinite(gtdb_seconds)
    ):
        return (
            imputation_seconds + gtdb_seconds,
            "gtdb_build_real_seconds+imputation_phase_real_seconds",
        )

    if method != "som" and np.isfinite(imputation_seconds):
        return imputation_seconds, "imputation_phase_real_seconds"

    total_seconds = as_float(row.get("total_gtimputation_real_seconds"))
    if np.isfinite(total_seconds):
        return total_seconds, "total_gtimputation_real_seconds"

    script_seconds = as_float(row.get("imputation_script_duration_seconds"))
    if np.isfinite(script_seconds):
        return script_seconds, "imputation_script_duration_seconds"

    return math.nan, "missing_runtime"


def score_gti_run(
    row: pd.Series,
    sim_row: pd.Series,
    *,
    gti_dir: Path,
    sim_manifest_path: Path,
    truth_cache: dict[Path, VCFMatrix],
    masked_cache: dict[Path, VCFMatrix],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    truth_path = resolve_truth_vcf(sim_row, sim_manifest_path)
    mask_path = resolve_mask_tsv(sim_row, sim_manifest_path)
    pred_path = resolve_gti_vcf(row, gti_dir)

    if truth_path not in truth_cache:
        truth_cache[truth_path] = load_vcf_matrix(truth_path)
    truth_vcf = truth_cache[truth_path]
    pred_vcf = load_vcf_matrix(pred_path)

    mask = pd.read_csv(
        mask_path,
        sep="\t",
        usecols=["sample_id", "locus_index", "chrom", "pos", "ref", "alt"],
        dtype={"sample_id": str, "chrom": str, "pos": str, "ref": str, "alt": str},
    )
    mask, mask_stats = mask_actual_missing_sites(
        mask,
        sim_row,
        sim_manifest_path=sim_manifest_path,
        masked_cache=masked_cache,
    )
    truth_loci, truth_samples = vcf_indices_from_mask(mask, truth_vcf)
    pred_loci, pred_samples = vcf_indices_from_mask(mask, pred_vcf)
    y_true = truth_vcf.zygosity[truth_loci, truth_samples]
    y_pred = pred_vcf.zygosity[pred_loci, pred_samples]
    validate_nonmissing_truth(
        y_true, f"GTImputation {row['dataset_id']}/{row['strategy']}/{row['run_id']}"
    )
    metrics, class_rows = score_vectors(y_true, y_pred)

    method = str(row["method"]).lower()
    runtime_seconds, runtime_source = gti_runtime_seconds(row)

    base = {
        "dataset_id": str(row["dataset_id"]),
        "strategy": str(row["strategy"]),
        "strategy_label": STRATEGY_LABELS.get(
            str(row["strategy"]), str(row["strategy"])
        ),
        "software": "GTImputation",
        "model": method,
        "model_label": GTI_MODEL_LABELS.get(method, method),
        "run_id": str(row["run_id"]),
        "report_type": "zygosity",
        "runtime_seconds": runtime_seconds,
        "runtime_source": runtime_source,
        **gti_runtime_metadata(row, runtime_seconds, runtime_source),
        "report_path": "",
        "input_path": str(pred_path),
        "mask_tsv": str(mask_path),
        "truth_vcf": str(truth_path),
        **mask_stats,
    }
    class_rows = [{**base, **class_row} for class_row in class_rows]
    return {**base, **metrics}, class_rows


def add_key_columns_from_index(
    sim_row: pd.Series, key_cols: Sequence[str], key: tuple[object, ...]
) -> pd.Series:
    sim_row = sim_row.copy()
    for key_col, key_value in zip(key_cols, key):
        if key_col not in sim_row.index:
            sim_row[key_col] = key_value
    return sim_row


def score_pgsui_run(
    row: pd.Series,
    sim_row: pd.Series,
    *,
    sim_manifest_path: Path,
    truth_cache: dict[Path, VCFMatrix],
    masked_cache: dict[Path, VCFMatrix],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    truth_path = resolve_truth_vcf(sim_row, sim_manifest_path)
    mask_path = resolve_mask_tsv(sim_row, sim_manifest_path)
    pred_path = Path(str(row["imputed_vcf_path"])).expanduser().resolve()
    if not pred_path.exists():
        raise FileNotFoundError(f"PG-SUI imputed VCF not found: {pred_path}")

    if truth_path not in truth_cache:
        truth_cache[truth_path] = load_vcf_matrix(truth_path)
    truth_vcf = truth_cache[truth_path]
    pred_vcf = load_vcf_matrix(pred_path)

    mask = pd.read_csv(
        mask_path,
        sep="\t",
        usecols=["sample_id", "locus_index", "chrom", "pos", "ref", "alt"],
        dtype={"sample_id": str, "chrom": str, "pos": str, "ref": str, "alt": str},
    )
    mask, mask_stats = mask_actual_missing_sites(
        mask,
        sim_row,
        sim_manifest_path=sim_manifest_path,
        masked_cache=masked_cache,
    )
    truth_loci, truth_samples = vcf_indices_from_mask(mask, truth_vcf)
    pred_loci, pred_samples = vcf_indices_from_mask(mask, pred_vcf)
    y_true = truth_vcf.zygosity[truth_loci, truth_samples]
    y_pred = pred_vcf.zygosity[pred_loci, pred_samples]
    validate_nonmissing_truth(
        y_true, f"PG-SUI {row['dataset_id']}/{row['strategy']}/{row['model']}"
    )
    metrics, class_rows = score_vectors(y_true, y_pred)

    base = {
        "dataset_id": str(row["dataset_id"]),
        "strategy": str(row["strategy"]),
        "strategy_label": STRATEGY_LABELS.get(
            str(row["strategy"]), str(row["strategy"])
        ),
        "software": "PG-SUI",
        "model": str(row["model"]),
        "model_label": str(row["model_label"]),
        "backend": str(row["backend"]),
        "output_dir": str(row["output_dir"]),
        "report_type": "zygosity_vcf_masked_sites",
        "runtime_seconds": as_float(row.get("runtime_seconds")),
        "runtime_source": str(row.get("runtime_source", "")),
        "optuna_elapsed_seconds": as_float(row.get("optuna_elapsed_seconds")),
        "optuna_rolling_avg_trial_seconds": as_float(
            row.get("optuna_rolling_avg_trial_seconds")
        ),
        "optuna_mean_trial_seconds": as_float(row.get("optuna_mean_trial_seconds")),
        "run_mode": row.get("run_mode"),
        "tuning_trials_planned": row.get("tuning_trials_planned"),
        "tuning_trials_completed": row.get("tuning_trials_completed"),
        "parallel_jobs": row.get("parallel_jobs"),
        "runtime_wall_seconds": row.get("runtime_wall_seconds"),
        "parallel_adjusted_runtime_seconds": row.get(
            "parallel_adjusted_runtime_seconds"
        ),
        "runtime_accounting": row.get("runtime_accounting"),
        "report_path": str(row.get("report_path", "")),
        "input_path": str(pred_path),
        "mask_tsv": str(mask_path),
        "truth_vcf": str(truth_path),
        **mask_stats,
    }
    class_rows = [{**base, **class_row} for class_row in class_rows]
    return {**base, **metrics}, class_rows


def score_pgsui_metrics(
    pgsui_status: pd.DataFrame,
    sim_manifest: pd.DataFrame,
    retained_keys: pd.DataFrame,
    *,
    models: Sequence[str],
    sim_manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key_cols = ["dataset_id", "strategy"]
    selected = retained_keys[key_cols + ["output_dir"]].drop_duplicates()
    scoring_status = pgsui_status.merge(
        selected, on=key_cols + ["output_dir"], how="inner"
    )
    scoring_status = scoring_status.loc[
        scoring_status["model"].isin(models)
        & scoring_status["model_complete"].astype(bool)
    ].copy()

    sim_lookup = sim_manifest.set_index(key_cols)
    truth_cache: dict[Path, VCFMatrix] = {}
    masked_cache: dict[Path, VCFMatrix] = {}
    metric_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []

    total = len(scoring_status)
    for index, (_, row) in enumerate(scoring_status.iterrows(), start=1):
        key = (row["dataset_id"], row["strategy"])
        print(
            f"Scoring PG-SUI {index}/{total}: "
            f"{row['dataset_id']} {row['strategy']} {row['model']}",
            file=sys.stderr,
        )
        try:
            sim_row = sim_lookup.loc[key]
            if isinstance(sim_row, pd.DataFrame):
                sim_row = sim_row.iloc[0]
            sim_row = add_key_columns_from_index(sim_row, key_cols, key)  # type: ignore
            metric_row, class_report_rows = score_pgsui_run(
                row,
                sim_row,
                sim_manifest_path=sim_manifest_path,
                truth_cache=truth_cache,
                masked_cache=masked_cache,
            )
        except Exception as exc:
            error_rows.append(
                {
                    "dataset_id": row.get("dataset_id"),
                    "strategy": row.get("strategy"),
                    "backend": row.get("backend"),
                    "model": row.get("model"),
                    "output_dir": row.get("output_dir"),
                    "error": str(exc),
                }
            )
            continue
        metric_rows.append(metric_row)
        class_rows.extend(class_report_rows)

    return (
        pd.DataFrame(metric_rows, columns=PGSUI_METRIC_COLUMNS),
        pd.DataFrame(class_rows, columns=PGSUI_CLASS_REPORT_COLUMNS),
        pd.DataFrame(error_rows, columns=PGSUI_ERROR_COLUMNS),
    )


def score_gti_metrics(
    gti_manifest: pd.DataFrame,
    sim_manifest: pd.DataFrame,
    retained_keys: pd.DataFrame,
    *,
    gti_dir: Path,
    sim_manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key_cols = ["dataset_id", "strategy"]
    wanted = retained_keys[key_cols].drop_duplicates()
    scoring_manifest = gti_manifest.merge(wanted, on=key_cols, how="inner")
    sim_lookup = sim_manifest.set_index(key_cols)
    truth_cache: dict[Path, VCFMatrix] = {}
    masked_cache: dict[Path, VCFMatrix] = {}
    metric_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []

    total = len(scoring_manifest)
    for index, (_, row) in enumerate(scoring_manifest.iterrows(), start=1):
        key = (row["dataset_id"], row["strategy"])
        print(
            f"Scoring GTImputation {index}/{total}: "
            f"{row['dataset_id']} {row['strategy']} {row['method']}",
            file=sys.stderr,
        )
        try:
            sim_row = sim_lookup.loc[key]
            if isinstance(sim_row, pd.DataFrame):
                sim_row = sim_row.iloc[0]
            sim_row = add_key_columns_from_index(sim_row, key_cols, key)  # type: ignore
            metric_row, class_report_rows = score_gti_run(
                row,
                sim_row,
                gti_dir=gti_dir,
                sim_manifest_path=sim_manifest_path,
                truth_cache=truth_cache,
                masked_cache=masked_cache,
            )
        except Exception as exc:
            error_rows.append(
                {
                    "dataset_id": row.get("dataset_id"),
                    "strategy": row.get("strategy"),
                    "method": row.get("method"),
                    "run_id": row.get("run_id"),
                    "error": str(exc),
                }
            )
            continue
        metric_rows.append(metric_row)
        class_rows.extend(class_report_rows)

    return (
        pd.DataFrame(metric_rows, columns=GTI_METRIC_COLUMNS),
        pd.DataFrame(class_rows, columns=GTI_CLASS_REPORT_COLUMNS),
        pd.DataFrame(error_rows, columns=GTI_ERROR_COLUMNS),
    )


def filter_after_pgsui_scoring(
    retained_keys: pd.DataFrame,
    pgsui_metrics: pd.DataFrame,
    models: Sequence[str],
    strategies: Sequence[str],
    allow_partial_grid: bool,
) -> pd.DataFrame:
    if pgsui_metrics.empty:
        return retained_keys.iloc[0:0].copy()
    model_labels = {PGSUI_MODEL_LABELS[model] for model in models}
    complete = (
        pgsui_metrics.groupby(["dataset_id", "strategy", "output_dir"], as_index=False)[
            "model_label"
        ]
        .agg(lambda values: model_labels.issubset(set(values)))
        .rename(columns={"model_label": "pgsui_scored_complete"})  # type: ignore
    )
    retained = retained_keys.merge(
        complete.loc[complete["pgsui_scored_complete"]],
        on=["dataset_id", "strategy", "output_dir"],
        how="inner",
    )
    if allow_partial_grid:
        return retained

    strategy_set = set(strategies)
    complete_datasets = (
        retained.groupby("dataset_id")["strategy"]
        .agg(lambda values: strategy_set.issubset(set(values)))
        .rename("has_all_strategies")
        .reset_index()
    )
    retained = retained.merge(complete_datasets, on="dataset_id", how="left")
    return retained.loc[retained["has_all_strategies"]].drop(
        columns=["has_all_strategies"]
    )


def filter_after_gti_scoring(
    retained_keys: pd.DataFrame,
    gti_metrics: pd.DataFrame,
    methods: Sequence[str],
    strategies: Sequence[str],
    allow_partial_grid: bool,
) -> pd.DataFrame:
    if gti_metrics.empty:
        return retained_keys.iloc[0:0].copy()
    method_labels = {GTI_MODEL_LABELS[method] for method in methods}
    complete = (
        gti_metrics.groupby(["dataset_id", "strategy"], as_index=False)["model_label"]
        .agg(lambda values: method_labels.issubset(set(values)))
        .rename(columns={"model_label": "gti_scored_complete"})  # type: ignore
    )
    retained = retained_keys.merge(
        complete.loc[complete["gti_scored_complete"]],
        on=["dataset_id", "strategy"],
        how="inner",
    )
    if allow_partial_grid:
        return retained

    strategy_set = set(strategies)
    complete_datasets = (
        retained.groupby("dataset_id")["strategy"]
        .agg(lambda values: strategy_set.issubset(set(values)))
        .rename("has_all_strategies")
        .reset_index()
    )
    retained = retained.merge(complete_datasets, on="dataset_id", how="left")
    return retained.loc[retained["has_all_strategies"]].drop(
        columns=["has_all_strategies"]
    )


def select_pgsui_metrics_for_runs(
    pgsui_metrics: pd.DataFrame, selected_runs: pd.DataFrame
) -> pd.DataFrame:
    if pgsui_metrics.empty or selected_runs.empty:
        return pgsui_metrics.iloc[0:0].copy()
    selected_outputs = selected_runs[
        ["dataset_id", "strategy", "output_dir"]
    ].drop_duplicates()
    return pgsui_metrics.merge(
        selected_outputs, on=["dataset_id", "strategy", "output_dir"], how="inner"
    )


def backend_display_label(backend: object) -> str:
    backend_text = str(backend)
    return PGSUI_BACKEND_LABELS.get(backend_text, backend_text.upper())


def pgsui_runtime_model_label(
    model: object, model_label: object, backend: object
) -> str:
    if str(model) in PGSUI_DEEP_MODELS:
        return f"{model_label} ({backend_display_label(backend)})"
    return str(model_label)


def add_runtime_labels(runtime_df: pd.DataFrame) -> pd.DataFrame:
    runtime_df = runtime_df.copy()
    if runtime_df.empty:
        runtime_df["runtime_backend"] = pd.Series(dtype="object")
        runtime_df["runtime_backend_label"] = pd.Series(dtype="object")
        runtime_df["runtime_model_label"] = pd.Series(dtype="object")
        return runtime_df

    runtime_backend: list[str] = []
    runtime_backend_label: list[str] = []
    runtime_model_label: list[str] = []
    for _, row in runtime_df.iterrows():
        software = str(row.get("software", ""))
        model = str(row.get("model", ""))
        backend = str(row.get("backend", "not_applicable"))
        model_label = str(row.get("model_label", model))
        if software == "PG-SUI" and model in PGSUI_DEEP_MODELS:
            runtime_backend.append(backend)
            runtime_backend_label.append(backend_display_label(backend))
            runtime_model_label.append(
                pgsui_runtime_model_label(model, model_label, backend)
            )
        elif software == "PG-SUI":
            runtime_backend.append("deterministic")
            runtime_backend_label.append(backend_display_label("deterministic"))
            runtime_model_label.append(model_label)
        else:
            runtime_backend.append("not_applicable")
            runtime_backend_label.append("GTImputation")
            runtime_model_label.append(model_label)
    runtime_df["runtime_backend"] = runtime_backend
    runtime_df["runtime_backend_label"] = runtime_backend_label
    runtime_df["runtime_model_label"] = runtime_model_label
    return runtime_df


def build_runtime_comparison(
    pgsui_metrics: pd.DataFrame,
    gti_metrics: pd.DataFrame,
    retained_keys: pd.DataFrame,
    selected_pgsui_f1: pd.DataFrame,
    selected_pgsui_runtime: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = ["dataset_id", "strategy"]
    if retained_keys.empty:
        return pd.DataFrame()

    retained = retained_keys[key_cols].drop_duplicates()
    deterministic_runs = selected_pgsui_f1[key_cols + ["output_dir"]].drop_duplicates()
    runtime_runs = selected_pgsui_runtime[
        key_cols + ["backend", "output_dir"]
    ].drop_duplicates()

    deterministic = (
        pgsui_metrics.loc[pgsui_metrics["model"].isin(PGSUI_DETERMINISTIC_MODELS)]
        .merge(deterministic_runs, on=key_cols + ["output_dir"], how="inner")
        .merge(retained, on=key_cols, how="inner")
    )
    deep = (
        pgsui_metrics.loc[pgsui_metrics["model"].isin(PGSUI_DEEP_MODELS)]
        .merge(runtime_runs, on=key_cols + ["backend", "output_dir"], how="inner")
        .merge(retained, on=key_cols, how="inner")
    )
    gti = gti_metrics.merge(retained, on=key_cols, how="inner").copy()

    common_cols = [
        "dataset_id",
        "strategy",
        "strategy_label",
        "software",
        "model",
        "model_label",
        "backend",
        "report_type",
        "macro_f1",
        "runtime_seconds",
        "runtime_source",
        "optuna_elapsed_seconds",
        "optuna_rolling_avg_trial_seconds",
        "optuna_mean_trial_seconds",
        "run_mode",
        "tuning_trials_planned",
        "tuning_trials_completed",
        "parallel_jobs",
        "runtime_wall_seconds",
        "parallel_adjusted_runtime_seconds",
        "runtime_accounting",
        "report_path",
        "input_path",
    ]
    for frame in (deterministic, deep, gti):
        for column in common_cols:
            if column not in frame.columns:
                frame[column] = np.nan

    runtime_df = pd.concat(
        [deterministic[common_cols], deep[common_cols], gti[common_cols]],
        ignore_index=True,
    )
    runtime_df = add_runtime_labels(runtime_df)
    runtime_df["dataset_strategy"] = (
        runtime_df["dataset_id"] + "\n" + runtime_df["strategy_label"].astype(str)
    )
    runtime_df["strategy_label"] = pd.Categorical(
        runtime_df["strategy_label"],
        categories=[STRATEGY_LABELS[strategy] for strategy in STRATEGY_ORDER],
        ordered=True,
    )
    order = runtime_model_order(runtime_df)
    runtime_df["runtime_model_label"] = pd.Categorical(
        runtime_df["runtime_model_label"],
        categories=order,
        ordered=True,
    )
    return runtime_df.sort_values(
        ["dataset_id", "strategy", "software", "runtime_model_label"]
    ).reset_index(drop=True)


def build_combined_metrics(
    pgsui_metrics: pd.DataFrame,
    gti_metrics: pd.DataFrame,
    retained_keys: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = ["dataset_id", "strategy"]
    selected_outputs = retained_keys[key_cols + ["output_dir"]].drop_duplicates()
    pgsui = pgsui_metrics.merge(
        selected_outputs, on=key_cols + ["output_dir"], how="inner"
    )
    common_cols = [
        "dataset_id",
        "strategy",
        "strategy_label",
        "software",
        "model",
        "model_label",
        "backend",
        "report_type",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "support",
        "mcc",
        "ref_f1",
        "het_f1",
        "alt_f1",
        "runtime_seconds",
        "runtime_source",
        "optuna_elapsed_seconds",
        "optuna_rolling_avg_trial_seconds",
        "optuna_mean_trial_seconds",
        "run_mode",
        "tuning_trials_planned",
        "tuning_trials_completed",
        "parallel_jobs",
        "runtime_wall_seconds",
        "parallel_adjusted_runtime_seconds",
        "runtime_accounting",
        "report_path",
        "input_path",
        "mask_tsv",
        "truth_vcf",
        "masked_input_vcf",
        "mask_rows",
        "scored_masked_sites",
        "dropped_unmasked_mask_rows",
    ]
    for column in common_cols:
        if column not in pgsui.columns:
            pgsui[column] = np.nan
        if column not in gti_metrics.columns:
            gti_metrics[column] = np.nan
    combined = pd.concat(
        [pgsui[common_cols], gti_metrics[common_cols]], ignore_index=True
    )
    combined = combined.merge(
        retained_keys[key_cols].drop_duplicates(), on=key_cols, how="inner"
    )
    combined["dataset_strategy"] = (
        combined["dataset_id"] + "\n" + combined["strategy_label"]
    )
    combined["model_label"] = pd.Categorical(
        combined["model_label"], categories=MODEL_ORDER, ordered=True
    )
    combined["strategy_label"] = pd.Categorical(
        combined["strategy_label"],
        categories=[STRATEGY_LABELS[strategy] for strategy in STRATEGY_ORDER],
        ordered=True,
    )
    return combined.sort_values(
        ["dataset_id", "strategy", "software", "model_label"]
    ).reset_index(drop=True)


def build_pgsui_class_report_from_reports(pgsui_metrics: pd.DataFrame) -> pd.DataFrame:
    class_rows: list[dict[str, object]] = []
    for _, row in pgsui_metrics.iterrows():
        report_path = Path(str(row.get("report_path", "")))
        if not report_path.exists():
            continue
        base = {
            "dataset_id": row.get("dataset_id"),
            "strategy": row.get("strategy"),
            "strategy_label": row.get("strategy_label"),
            "software": "PG-SUI",
            "model": row.get("model"),
            "model_label": row.get("model_label"),
            "backend": row.get("backend"),
            "output_dir": row.get("output_dir"),
            "report_type": row.get("report_type"),
            "runtime_seconds": row.get("runtime_seconds"),
            "runtime_source": row.get("runtime_source"),
            "optuna_elapsed_seconds": row.get("optuna_elapsed_seconds"),
            "optuna_rolling_avg_trial_seconds": row.get(
                "optuna_rolling_avg_trial_seconds"
            ),
            "optuna_mean_trial_seconds": row.get("optuna_mean_trial_seconds"),
            "run_mode": row.get("run_mode"),
            "tuning_trials_planned": row.get("tuning_trials_planned"),
            "tuning_trials_completed": row.get("tuning_trials_completed"),
            "parallel_jobs": row.get("parallel_jobs"),
            "runtime_wall_seconds": row.get("runtime_wall_seconds"),
            "parallel_adjusted_runtime_seconds": row.get(
                "parallel_adjusted_runtime_seconds"
            ),
            "runtime_accounting": row.get("runtime_accounting"),
            "report_path": str(report_path),
            "input_path": row.get("input_path", ""),
            "mask_tsv": "",
            "truth_vcf": "",
            "masked_input_vcf": "",
            "mask_rows": np.nan,
            "scored_masked_sites": np.nan,
            "dropped_unmasked_mask_rows": np.nan,
        }
        class_rows.extend(
            {**base, **class_row} for class_row in load_pgsui_class_report(report_path)
        )
    return pd.DataFrame(class_rows, columns=PGSUI_CLASS_REPORT_COLUMNS)


def build_combined_class_report(
    pgsui_class_report: pd.DataFrame,
    gti_class_report: pd.DataFrame,
    retained_keys: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = ["dataset_id", "strategy"]
    selected_outputs = retained_keys[key_cols + ["output_dir"]].drop_duplicates()
    pgsui = pgsui_class_report.merge(
        selected_outputs, on=key_cols + ["output_dir"], how="inner"
    )
    common_cols = [
        "dataset_id",
        "strategy",
        "strategy_label",
        "software",
        "model",
        "model_label",
        "backend",
        "report_type",
        "class_label",
        "genotype_class",
        "precision",
        "recall",
        "f1",
        "support",
        "runtime_seconds",
        "runtime_source",
        "optuna_elapsed_seconds",
        "optuna_rolling_avg_trial_seconds",
        "optuna_mean_trial_seconds",
        "run_mode",
        "tuning_trials_planned",
        "tuning_trials_completed",
        "parallel_jobs",
        "runtime_wall_seconds",
        "parallel_adjusted_runtime_seconds",
        "runtime_accounting",
        "report_path",
        "input_path",
        "mask_tsv",
        "truth_vcf",
        "masked_input_vcf",
        "mask_rows",
        "scored_masked_sites",
        "dropped_unmasked_mask_rows",
    ]
    for column in common_cols:
        if column not in pgsui.columns:
            pgsui[column] = np.nan
        if column not in gti_class_report.columns:
            gti_class_report[column] = np.nan
    combined = pd.concat(
        [pgsui[common_cols], gti_class_report[common_cols]], ignore_index=True
    )
    combined = combined.merge(
        retained_keys[key_cols].drop_duplicates(), on=key_cols, how="inner"
    )
    combined["dataset_strategy"] = (
        combined["dataset_id"] + "\n" + combined["strategy_label"]
    )
    combined["model_label"] = pd.Categorical(
        combined["model_label"], categories=MODEL_ORDER, ordered=True
    )
    combined["strategy_label"] = pd.Categorical(
        combined["strategy_label"],
        categories=[STRATEGY_LABELS[strategy] for strategy in STRATEGY_ORDER],
        ordered=True,
    )
    combined["genotype_class"] = pd.Categorical(
        combined["genotype_class"],
        categories=GENOTYPE_CLASS_ORDER,
        ordered=True,
    )
    return combined.sort_values(
        ["dataset_id", "strategy", "software", "model_label", "genotype_class"]
    ).reset_index(drop=True)


def best_model_summary(
    combined: pd.DataFrame,
    combined_class_report: pd.DataFrame | None = None,
    *,
    pgsui_software: str = "PG-SUI",
    gti_software: str = "GTImputation",
) -> pd.DataFrame:
    """Select the best model per dataset, strategy, and software package.

    The function selects the model with the highest macro F1 score separately
    for PG-SUI and GTImputation. It then attaches class-specific F1 scores,
    total support, and the PG-SUI-minus-GTImputation macro-F1 difference.

    Args:
        combined: Model-level metrics containing at least dataset, strategy,
            software, model, and macro-F1 columns.
        combined_class_report: Optional long-form classification report
            containing genotype-class F1 scores and support.
        pgsui_software: Software label used for PG-SUI rows.
        gti_software: Software label used for GTImputation rows.

    Returns:
        Long-form DataFrame with one winning model per dataset, strategy, and
        software package. The output includes ``support``, ``ref_f1``,
        ``het_f1``, ``alt_f1``, and ``delta_macro_f1``.

    Raises:
        ValueError: If required columns are unavailable.
    """
    metrics = combined.copy()

    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "strategy",
                "strategy_label",
                "software",
                "model",
                "model_label",
                "macro_f1",
                "delta_macro_f1",
                "support",
                "ref_f1",
                "het_f1",
                "alt_f1",
            ]
        )

    required_columns = {"dataset_id", "software", "macro_f1"}
    missing_columns = required_columns.difference(metrics.columns)
    if missing_columns:
        raise ValueError(
            f"combined is missing required columns: {sorted(missing_columns)}"
        )

    strategy_column = next(
        (
            column
            for column in ("strategy", "strategy_label")
            if column in metrics.columns
        ),
        None,
    )
    model_column = next(
        (column for column in ("model", "model_label") if column in metrics.columns),
        None,
    )

    if strategy_column is None:
        raise ValueError("combined must contain either 'strategy' or 'strategy_label'.")
    if model_column is None:
        raise ValueError("combined must contain either 'model' or 'model_label'.")

    # Guarantee that the user-facing label columns are present.
    if "strategy_label" not in metrics.columns:
        metrics["strategy_label"] = metrics[strategy_column].astype("string")

    if "model_label" not in metrics.columns:
        metrics["model_label"] = metrics[model_column].astype("string")

    pair_columns = ["dataset_id", strategy_column]
    selection_columns = [*pair_columns, "software"]

    # Stable sorting gives deterministic model selection when macro-F1 ties occur.
    best = (
        metrics.dropna(subset=["macro_f1"])
        .sort_values(
            by=[*selection_columns, "macro_f1", "model_label"],
            ascending=[
                *([True] * len(selection_columns)),
                False,
                True,
            ],
            kind="mergesort",
        )
        .drop_duplicates(subset=selection_columns, keep="first")
        .copy()
    )

    if combined_class_report is not None and not combined_class_report.empty:
        class_report = combined_class_report.copy()

        required_class_columns = {
            "dataset_id",
            "software",
            "genotype_class",
            "f1",
        }
        missing_class_columns = required_class_columns.difference(class_report.columns)
        if missing_class_columns:
            raise ValueError(
                "combined_class_report is missing required columns: "
                f"{sorted(missing_class_columns)}"
            )

        class_strategy_column = _first_common_column(
            best,
            class_report,
            ("strategy", "strategy_label"),
        )
        class_model_column = _first_common_column(
            best,
            class_report,
            ("model", "model_label"),
        )

        class_join_columns = [
            "dataset_id",
            class_strategy_column,
            "software",
            class_model_column,
        ]

        class_report["_f1_output_column"] = _canonicalize_genotype_class(
            class_report["genotype_class"]
        )
        class_report = class_report.dropna(subset=["_f1_output_column"]).copy()

        class_f1_wide = (
            class_report.pivot_table(
                index=class_join_columns,
                columns="_f1_output_column",
                values="f1",
                aggfunc="first",
            )
            .rename_axis(columns=None)
            .reset_index()
        )

        class_metric_columns = [
            column
            for column in ("ref_f1", "het_f1", "alt_f1")
            if column in class_f1_wide.columns
        ]
        class_f1_wide = class_f1_wide.rename(
            columns={column: f"__class_{column}" for column in class_metric_columns}
        )

        best = best.merge(
            class_f1_wide,
            on=class_join_columns,
            how="left",
            validate="one_to_one",
        )

        for column in ("ref_f1", "het_f1", "alt_f1"):
            class_column = f"__class_{column}"

            if class_column not in best.columns:
                continue

            if column in best.columns:
                best[column] = best[class_column].combine_first(best[column])
            else:
                best[column] = best[class_column]

            best = best.drop(columns=class_column)

        if "support" in class_report.columns:
            support_summary = (
                class_report.groupby(
                    class_join_columns,
                    dropna=False,
                    as_index=False,
                )["support"]
                .sum(min_count=1)
                .rename(columns={"support": "__class_support"})  # type: ignore
            )

            best = best.merge(
                support_summary,
                on=class_join_columns,
                how="left",
                validate="one_to_one",
            )

            if "support" in best.columns:
                best["support"] = best["support"].combine_first(best["__class_support"])
            else:
                best["support"] = best["__class_support"]

            best = best.drop(columns="__class_support")

    # Construct paired PG-SUI and GTImputation results while preserving the
    # original long-form best-model summary rows used by tables and README output.
    paired_models = (
        best.pivot_table(
            index=pair_columns,
            columns="software",
            values="model_label",
            aggfunc="first",
        )
        .reindex(columns=[pgsui_software, gti_software])
        .rename(
            columns={
                pgsui_software: "pgsui_model",
                gti_software: "gti_model",
            }
        )
    )

    paired_scores = (
        best.pivot_table(
            index=pair_columns,
            columns="software",
            values="macro_f1",
            aggfunc="first",
        )
        .reindex(columns=[pgsui_software, gti_software])
        .rename(
            columns={
                pgsui_software: "pgsui_macro_f1",
                gti_software: "gti_macro_f1",
            }
        )
    )

    paired_summary = (
        paired_models.join(paired_scores, how="outer")
        .rename_axis(columns=None)
        .reset_index()
    )

    paired_summary["delta_macro_f1"] = (
        paired_summary["pgsui_macro_f1"] - paired_summary["gti_macro_f1"]
    )

    best = best.merge(
        paired_summary,
        on=pair_columns,
        how="left",
        validate="many_to_one",
    )

    # Guarantee the requested output schema even when an input field is absent.
    for column in ("support", "ref_f1", "het_f1", "alt_f1"):
        if column not in best.columns:
            best[column] = pd.NA

    preferred_order = [
        "dataset_id",
        "strategy",
        "strategy_label",
        "software",
        "model",
        "model_label",
        "macro_f1",
        "pgsui_model",
        "gti_model",
        "pgsui_macro_f1",
        "gti_macro_f1",
        "delta_macro_f1",
        "support",
        "ref_f1",
        "het_f1",
        "alt_f1",
    ]

    ordered_columns = [column for column in preferred_order if column in best.columns]
    remaining_columns = [
        column for column in best.columns if column not in ordered_columns
    ]

    return (
        best.loc[:, [*ordered_columns, *remaining_columns]]
        .sort_values(
            by=[*pair_columns, "software"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def summarize_numeric(
    df: pd.DataFrame, group_cols: list[str], value_col: str
) -> pd.DataFrame:
    return (
        df.dropna(subset=[value_col])
        .groupby(group_cols, observed=True)[value_col]
        .agg(n="count", mean="mean", median="median", sd="std", min="min", max="max")
        .reset_index()
    )


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-oriented copy of a table while preserving source CSVs."""
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4g}"
            )
        elif pd.api.types.is_integer_dtype(out[column]):
            out[column] = out[column].map(
                lambda value: "" if pd.isna(value) else str(int(value))
            )
        else:
            out[column] = out[column].astype("object").where(out[column].notna(), "")
    return out


def formatted_table_dirs(tables_dir: Path) -> dict[str, Path]:
    base = tables_dir / "formatted"
    return {
        "html": base / "html",
        "latex": base / "latex",
        "rtf": base / "rtf",
    }


def write_html_table_view(df: pd.DataFrame, path: Path, title: str) -> None:
    display = display_table(df)
    table_html = display.to_html(index=False, escape=True, classes="data-table")
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      margin: 24px;
      color: #1f1f24;
      background: #fafafc;
    }}
    h1 {{
      font-size: 22px;
      margin: 0 0 6px;
    }}
    .meta {{
      color: #555;
      margin-bottom: 16px;
    }}
    .table-wrap {{
      max-height: 82vh;
      overflow: auto;
      border: 1px solid #d2d2da;
      background: white;
      box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    }}
    table.data-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
      line-height: 1.35;
    }}
    table.data-table th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: #22222a;
      color: white;
      text-align: left;
      border-bottom: 2px solid #000;
    }}
    table.data-table th,
    table.data-table td {{
      padding: 6px 8px;
      border: 1px solid #dedee6;
      white-space: nowrap;
    }}
    table.data-table tbody tr:nth-child(even) {{
      background: #f4f4f8;
    }}
    table.data-table tbody tr:hover {{
      background: #fff7c2;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="meta">{len(df):,} rows x {len(df.columns):,} columns</div>
  <div class="table-wrap">
    {table_html}
  </div>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document)


def write_latex_table_view(df: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_latex(
            path,
            index=False,
            escape=True,
            longtable=True,
            caption=title.replace("_", "\\_"),
            float_format=lambda value: f"{value:.4g}",
        )
    except Exception as exc:
        path.with_suffix(".latex_error.txt").write_text(
            f"Could not write LaTeX table for {title}: {exc}\n"
        )


def rtf_escape(value) -> str:
    text = "" if pd.isna(value) else str(value)
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\line ")
    )


def write_rtf_table_view(
    df: pd.DataFrame,
    path: Path,
    title: str,
    *,
    max_rows: int = 250,
    max_cols: int = 12,
) -> None:
    """Write a compact Word-readable RTF table for summary-sized outputs."""
    if df.shape[0] > max_rows or df.shape[1] > max_cols:
        path.unlink(missing_ok=True)
        return

    display = display_table(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    column_width = max(1200, min(2800, int(14000 / max(1, len(display.columns)))))
    cell_edges = [column_width * (index + 1) for index in range(len(display.columns))]

    def row_rtf(values: Sequence[object], *, bold: bool = False) -> str:
        parts = [r"\trowd\trgaph108\trleft0"]
        parts.extend(rf"\cellx{edge}" for edge in cell_edges)
        for value in values:
            text = rtf_escape(value)
            if bold:
                text = rf"\b {text}\b0"
            parts.append(rf"\intbl {text}\cell")
        parts.append(r"\row")
        return "\n".join(parts)

    rows = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0 Arial;}}",
        r"\fs20",
        rf"\b {rtf_escape(title)}\b0\par",
        rf"{len(df):,} rows x {len(df.columns):,} columns\par\par",
        row_rtf(display.columns.tolist(), bold=True),
    ]
    rows.extend(row_rtf(row) for row in display.itertuples(index=False, name=None))
    rows.append("}")
    path.write_text("\n".join(rows))


def write_table_views(df: pd.DataFrame, csv_path: Path) -> None:
    stem = csv_path.stem
    dirs = formatted_table_dirs(csv_path.parent)
    write_html_table_view(df, dirs["html"] / f"{stem}.html", stem)
    write_latex_table_view(df, dirs["latex"] / f"{stem}.tex", stem)
    write_rtf_table_view(df, dirs["rtf"] / f"{stem}.rtf", stem)


def write_table_view_index(tables_dir: Path) -> None:
    dirs = formatted_table_dirs(tables_dir)
    csv_paths = sorted(tables_dir.glob("*.csv"))
    rows: list[str] = []
    for csv_path in csv_paths:
        stem = csv_path.stem
        links = [f'<a href="../{html.escape(csv_path.name)}">CSV</a>']
        generated_links = [
            ("HTML", dirs["html"] / f"{stem}.html", f"html/{stem}.html"),
            ("LaTeX", dirs["latex"] / f"{stem}.tex", f"latex/{stem}.tex"),
            ("RTF", dirs["rtf"] / f"{stem}.rtf", f"rtf/{stem}.rtf"),
        ]
        links.extend(
            f'<a href="{html.escape(href)}">{label}</a>'
            for label, output_path, href in generated_links
            if output_path.exists()
        )
        rows.append(f"<tr><td>{html.escape(stem)}</td><td>{', '.join(links)}</td></tr>")

    index_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PG-SUI GTImputation Table Views</title>
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      margin: 24px;
      background: #fafafc;
      color: #1f1f24;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
      box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    }}
    th, td {{
      padding: 8px 10px;
      border: 1px solid #dedee6;
      text-align: left;
    }}
    th {{
      background: #22222a;
      color: white;
    }}
    tr:nth-child(even) {{
      background: #f4f4f8;
    }}
    a {{
      color: #0755a3;
      text-decoration: none;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <h1>PG-SUI GTImputation Table Views</h1>
  <p>CSV files remain the source tables. HTML and LaTeX views are generated for every table; RTF is generated for compact summary tables that are reasonable to open in Word.</p>
  <table>
    <thead><tr><th>Table</th><th>Formats</th></tr></thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    index_path = tables_dir / "formatted" / "index.html"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(index_html)


def cpu_gpu_runtime_delta_df(runtime_combined: pd.DataFrame) -> pd.DataFrame:
    """Return paired CPU/GPU runtime deltas for PG-SUI deep models."""
    required = {
        "software",
        "model",
        "model_label",
        "runtime_backend",
        "runtime_seconds",
    }

    if runtime_combined.empty or not required.issubset(runtime_combined.columns):
        return pd.DataFrame()

    key_cols = ["dataset_id", "strategy", "model", "model_label"]

    plot_df = runtime_combined.loc[
        runtime_combined["software"].eq("PG-SUI")
        & runtime_combined["model"].isin(PGSUI_DEEP_MODELS)
        & runtime_combined["runtime_backend"].isin(("cpu", "cuda"))
    ].copy()

    plot_df["runtime_seconds"] = pd.to_numeric(
        plot_df["runtime_seconds"], errors="coerce"
    )

    plot_df = plot_df.loc[plot_df["runtime_seconds"].gt(0)].dropna(
        subset=key_cols + ["runtime_backend", "runtime_seconds"]
    )

    if plot_df.empty:
        return pd.DataFrame()

    paired = (
        plot_df.pivot_table(
            index=key_cols,
            columns="runtime_backend",
            values="runtime_seconds",
            aggfunc="mean",
            observed=True,
        )
        .rename_axis(columns=None)
        .reset_index()
    )

    if not {"cpu", "cuda"}.issubset(paired.columns):
        return pd.DataFrame()

    paired = paired.dropna(subset=["cpu", "cuda"]).copy()
    paired["cpu_minus_gpu_seconds"] = paired["cpu"] - paired["cuda"]
    paired["cpu_gpu_speedup"] = paired["cpu"] / paired["cuda"]

    return paired.sort_values(["model_label", "dataset_id", "strategy"]).reset_index(
        drop=True
    )


def f1_class_balance_summary_df(combined: pd.DataFrame) -> pd.DataFrame:
    """Summarize class-balanced F1 behavior for each model."""
    class_cols = ["ref_f1", "het_f1", "alt_f1"]
    required = {"software", "model_label", "macro_f1", *class_cols}

    if combined.empty or not required.issubset(combined.columns):
        return pd.DataFrame()

    plot_df = combined.dropna(subset=["macro_f1", *class_cols]).copy()

    for column in ["macro_f1", "mcc", *class_cols]:
        if column in plot_df.columns:
            plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")

    plot_df = plot_df.dropna(subset=["macro_f1", *class_cols])

    if plot_df.empty:
        return pd.DataFrame()

    plot_df["worst_class_f1"] = plot_df[class_cols].min(axis=1)
    plot_df["class_f1_spread"] = plot_df[class_cols].max(axis=1) - plot_df[
        class_cols
    ].min(axis=1)

    agg_spec: dict[str, tuple[str, str]] = {
        "n": ("macro_f1", "count"),
        "mean_macro_f1": ("macro_f1", "mean"),
        "mean_worst_class_f1": ("worst_class_f1", "mean"),
        "mean_class_f1_spread": ("class_f1_spread", "mean"),
    }
    if "mcc" in plot_df.columns:
        agg_spec["mean_mcc"] = ("mcc", "mean")

    return (
        plot_df.groupby(["software", "model_label"], observed=True)
        .agg(**agg_spec)
        .reset_index()
        .sort_values(["software", "model_label"])
    )


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    write_table_views(df, path)


_SAVEFIG_DPI = 600
_SANS_FALLBACK = ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"]


def apply_house_style() -> None:
    """Apply the shared publication style.

    Seaborn's ``set_theme`` resets font-related rcParams, so any function that
    calls it must re-apply the house rcParams afterwards. Centralizing both steps
    here keeps fonts and sizes identical across every figure.
    """
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _SANS_FALLBACK,
            "figure.dpi": 300,
            "savefig.dpi": _SAVEFIG_DPI,
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "legend.title_fontsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def configure_plot_style(dpi: int) -> None:
    global _SAVEFIG_DPI
    _SAVEFIG_DPI = dpi
    apply_house_style()


def format_decimal_ticks(
    ax, axis: str = "both", decimals: int = 2, labelsize: object = "xx-large"
) -> None:
    """Format tick labels to a fixed number of decimals without moving ticks.

    Using a formatter avoids the ``set_ticks(get_ticks())`` idiom, which silently pulls in off-view ticks (for example at -0.2 and 1.2) and expands the axis limits past an intended ``ylim`` such as (-0.02, 1.02).
    """
    formatter = mticker.FormatStrFormatter(f"%.{decimals}f")
    if axis in {"y", "both"}:
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis="y", labelsize=labelsize)

    if axis in {"x", "both"}:
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", labelsize=labelsize)


def add_strategy_key(fig, y: float = -0.03) -> None:
    """Add a footnote decoding the R/RW/RWI/N/NW strategy abbreviations.

    Every strategy plot shows compact abbreviations on the x-axis; without this
    key a reader cannot tell what, for example, ``RWI`` means.
    """
    key = "   ".join(
        f"{STRATEGY_ABBREVIATIONS[strategy]} = {STRATEGY_LABELS[strategy]}"
        for strategy in STRATEGY_ORDER
    )
    fig.text(
        0.5,
        y,
        f"Simulation strategy:   {key}",
        ha="center",
        va="bottom",
        fontsize="x-large",
        color="#444444",
    )


def save_figure(fig, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def dataset_strategy_order(df: pd.DataFrame) -> list[str]:
    labels = df[["dataset_id", "strategy", "dataset_strategy"]].drop_duplicates()
    labels["strategy"] = pd.Categorical(
        labels["strategy"], categories=STRATEGY_ORDER, ordered=True
    )
    labels = labels.sort_values(["dataset_id", "strategy"])
    return labels["dataset_strategy"].tolist()


def lighten_color(color: str, amount: float = 0.55) -> str:
    rgb = np.array(mcolors.to_rgb(color))
    lightened = rgb + (1.0 - rgb) * amount
    return mcolors.to_hex(np.clip(lightened, 0, 1))  # type: ignore


def repel_point_labels(
    ax,
    xy_points: Sequence[tuple[float, float]],
    labels: Sequence[str],
    *,
    fontsize: float | str = 11.0,
    fontweight: str = "normal",
    text_color: str = "#1f1f24",
    arrow_color: str = "#555555",
    pad_frac: float = 0.14,
    n_iter: int = 320,
) -> None:
    """Annotate scatter points with non-overlapping labels and leader lines.

    A lightweight, dependency-free force-directed placement: label boxes repel
    each other and the data markers, then relax back toward their anchor point.
    Positions are solved in data coordinates using label extents measured from
    the renderer, so the result survives ``savefig`` at any DPI.
    """
    labels = list(labels)
    points = np.asarray(xy_points, dtype=float)

    if points.size == 0:
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def data_size(width_px: float, height_px: float) -> tuple[float, float]:
        origin = inv.transform((0.0, 0.0))
        far = inv.transform((width_px, height_px))
        return abs(far[0] - origin[0]), abs(far[1] - origin[1])

    half_w = np.zeros(len(labels))
    half_h = np.zeros(len(labels))

    for index, label in enumerate(labels):
        probe = ax.text(0, 0, label, fontsize=fontsize, fontweight=fontweight)
        extent = probe.get_window_extent(renderer)
        probe.remove()
        width_data, height_data = data_size(extent.width, extent.height)
        half_w[index] = width_data / 2.0 + width_data * pad_frac
        half_h[index] = height_data / 2.0 + height_data * pad_frac

    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    log_x = ax.get_xscale() == "log"
    span_x = abs(np.log10(x_hi / x_lo)) if log_x else abs(x_hi - x_lo)
    span_y = abs(y_hi - y_lo)

    def to_axis(value: float, low: float, is_log: bool) -> float:
        return np.log10(value / low) if is_log else value - low

    def from_axis(value: float, low: float, is_log: bool) -> float:
        return low * (10.0**value) if is_log else low + value

    anchors = np.column_stack(
        [
            [to_axis(px, x_lo, log_x) for px in points[:, 0]],
            [to_axis(py, y_lo, False) for py in points[:, 1]],
        ]
    )

    if log_x:
        hw = np.array(
            [
                abs(np.log10((points[i, 0] + half_w[i]) / max(points[i, 0], 1e-12)))
                for i in range(len(labels))
            ]
        )

    else:
        hw = half_w.copy()

    hh = half_h.copy()

    # Fan the initial label positions out in different directions so labels for
    # near-coincident markers do not all start stacked on top of each other.
    angles = np.linspace(0.0, 2.0 * np.pi, len(labels), endpoint=False) + 0.5

    pos = anchors + np.column_stack(
        [np.cos(angles) * hw * 1.6, np.sin(angles) * hh * 2.2]
    )

    marker_clear = np.maximum(hh, 0.02 * span_y)

    for _ in range(n_iter):
        shift = np.zeros_like(pos)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                overlap_x = (hw[i] + hw[j]) - abs(dx)
                overlap_y = (hh[i] + hh[j]) - abs(dy)

                if overlap_x > 0 and overlap_y > 0:
                    if overlap_x / (hw[i] + hw[j]) < overlap_y / (hh[i] + hh[j]):
                        push = (overlap_x + 1e-9) / 2.0
                        direction = 1.0 if dx >= 0 else -1.0
                        shift[i, 0] += direction * push
                        shift[j, 0] -= direction * push
                    else:
                        push = (overlap_y + 1e-9) / 2.0
                        direction = 1.0 if dy >= 0 else -1.0
                        shift[i, 1] += direction * push
                        shift[j, 1] -= direction * push

        for i in range(len(labels)):
            # Repel the label away from every marker so it clears the points.
            for k in range(len(labels)):
                dx = pos[i, 0] - anchors[k, 0]
                dy = pos[i, 1] - anchors[k, 1]

                if abs(dx) < hw[i] and abs(dy) < marker_clear[i]:
                    if abs(dy) >= abs(dx):
                        shift[i, 1] += (marker_clear[i] - abs(dy) + 1e-9) * (
                            1.0 if dy >= 0 else -1.0
                        )

                    else:
                        shift[i, 0] += (hw[i] - abs(dx) + 1e-9) * (
                            1.0 if dx >= 0 else -1.0
                        )

            # A weak tether keeps each label near its own anchor.
            shift[i, 0] -= 0.006 * (pos[i, 0] - anchors[i, 0])
            shift[i, 1] -= 0.006 * (pos[i, 1] - anchors[i, 1])

        pos += shift
        pos[:, 0] = np.clip(pos[:, 0], hw, span_x - hw)
        pos[:, 1] = np.clip(pos[:, 1], hh, span_y - hh)

    for i, label in enumerate(labels):
        label_x = from_axis(pos[i, 0], x_lo, log_x)
        label_y = from_axis(pos[i, 1], y_lo, False)

        ax.annotate(
            label,
            xy=(points[i, 0], points[i, 1]),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize=fontsize,
            fontweight=fontweight,
            color=text_color,
            ha="center",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": arrow_color,
                "linewidth": 0.8,
                "alpha": 0.8,
                "shrinkA": 2,
                "shrinkB": 3,
            },
            path_effects=[
                path_effects.withStroke(linewidth=3.0, foreground="white", alpha=0.9)
            ],
            zorder=6,
        )


def add_strategy_abbreviations(df: pd.DataFrame) -> pd.DataFrame:
    """Attach compact strategy labels used by dense strategy comparison plots."""
    result = df.copy()
    strategy_label_map = {
        STRATEGY_LABELS[strategy]: abbreviation
        for strategy, abbreviation in STRATEGY_ABBREVIATIONS.items()
    }

    if "strategy" in result.columns:
        result["strategy_label_abbr"] = result["strategy"].map(STRATEGY_ABBREVIATIONS)
        if "strategy_label" in result.columns:
            result["strategy_label_abbr"] = result["strategy_label_abbr"].fillna(
                result["strategy_label"].map(strategy_label_map)
            )
    elif "strategy_label" in result.columns:
        result["strategy_label_abbr"] = result["strategy_label"].map(strategy_label_map)
    else:
        result["strategy_label_abbr"] = np.nan

    result["strategy_label_abbr"] = pd.Categorical(
        result["strategy_label_abbr"],
        categories=list(STRATEGY_ABBR_ORDER),
        ordered=True,
    )

    return result


def present_model_order(df: pd.DataFrame, label_col: str = "model_label") -> list[str]:
    present = {str(value) for value in df[label_col].dropna().astype(str).unique()}
    order = [model for model in MODEL_ORDER if model in present]
    order.extend(sorted(present - set(order)))
    return order


def present_model_palette(order: Sequence[str]) -> dict[str, str]:
    return {str(label): MODEL_PALETTE.get(str(label), "#777777") for label in order}


def validate_nonmissing_truth(y_true: np.ndarray, context: str) -> None:
    """Require scored truth genotypes to be observed ground-truth calls."""
    missing_truth = int(np.count_nonzero(np.asarray(y_true) < 0))

    if missing_truth:
        raise ValueError(
            f"{context}: truth VCF has {missing_truth} missing genotypes at "
            "scored simulated-missing sites."
        )


def runtime_model_order(df: pd.DataFrame) -> list[str]:
    if "runtime_model_label" not in df.columns:
        return list(MODEL_ORDER)

    present = {
        str(value) for value in df["runtime_model_label"].dropna().astype(str).unique()
    }

    order = [
        label
        for label in (
            "Naive (GTImputation)",
            "SOM (GTImputation)",
            "RefAllele",
            "MostFrequent",
        )
        if label in present
    ]
    for model in PGSUI_DEEP_MODELS:
        model_label = PGSUI_MODEL_LABELS[model]
        for backend in RUNTIME_PGSUI_BACKENDS:
            label = f"{model_label} ({backend_display_label(backend)})"

            if label in present:
                order.append(label)

    order.extend(sorted(present - set(order)))
    return order


def runtime_model_palette(order: Sequence[str]) -> dict[str, str]:
    palette: dict[str, str] = {}
    for label in order:
        base_label = str(label).split(" (", 1)[0]
        base_color = MODEL_PALETTE.get(base_label, "#777777")

        if str(label).endswith("(CPU)"):
            palette[str(label)] = lighten_color(base_color)
        elif str(label).endswith("(GPU)"):
            palette[str(label)] = base_color
        else:
            palette[str(label)] = MODEL_PALETTE.get(str(label), base_color)

    return palette


def runtime_panel_backend_labels(df: pd.DataFrame) -> list[str] | list[str | None]:
    """Return PG-SUI runtime backend panels to plot, or one un-faceted panel."""
    if "runtime_backend_label" not in df.columns:
        return [None]

    present = set(df["runtime_backend_label"].dropna().astype(str))

    panel_labels = [
        backend_display_label(backend)
        for backend in RUNTIME_PGSUI_BACKENDS
        if backend_display_label(backend) in present
    ]

    return panel_labels or [None]


def runtime_panel_subset(df: pd.DataFrame, panel_label: str | None) -> pd.DataFrame:
    if panel_label is None or "runtime_backend_label" not in df.columns:
        return df.copy()

    baseline_labels = {"Deterministic"}
    if panel_label == backend_display_label("cpu"):
        baseline_labels.add("GTImputation")

    return df.loc[
        df["runtime_backend_label"].astype(str).isin({panel_label, *baseline_labels})
    ].copy()


def runtime_panel_title(panel_label: str | None) -> str:
    if panel_label is None:
        return "Runtime by Model"
    return f"Imputations ({panel_label})"


def runtime_panel_model_order(
    df: pd.DataFrame, label_col: str = "model_label"
) -> list[str]:
    present = {str(value) for value in df[label_col].dropna().astype(str).unique()}

    order = [label for label in MODEL_ORDER if label in present]
    order.extend(sorted(present - set(order)))
    return order


def runtime_panel_palette(order: Sequence[str]) -> dict[str, str]:
    return {str(label): MODEL_PALETTE.get(str(label), "#777777") for label in order}


def summarize_mean_ci(
    data: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
) -> pd.DataFrame:
    """Summarize values as means with normal-approximation 95% CI bounds."""
    summary = (
        data.dropna(subset=[value_col])
        .groupby(list(group_cols), observed=True)[value_col]
        .agg(n="count", mean="mean", sd="std")
        .reset_index()
    )

    if summary.empty:
        summary["ci95"] = pd.Series(dtype=float)
        summary["ci95_min"] = pd.Series(dtype=float)
        summary["ci95_max"] = pd.Series(dtype=float)
        return summary

    summary["sd"] = summary["sd"].fillna(0.0)
    summary["ci95"] = 1.96 * summary["sd"] / np.sqrt(summary["n"].clip(lower=1))
    summary["ci95_min"] = summary["mean"] - summary["ci95"]
    summary["ci95_max"] = summary["mean"] + summary["ci95"]
    return summary


def add_ci_errorbars(
    ax,
    summary: pd.DataFrame,
    *,
    x_col: str,
    x_order: Sequence[str],
    hue_col: str | None = None,
    hue_order: Sequence[str] | None = None,
    mean_col: str = "mean",
    lower_col: str = "ci95_min",
    upper_col: str = "ci95_max",
    color: str = "black",
    linewidth: float = 1.5,
    capsize: float = 5.0,
) -> None:
    """Overlay explicit 95% CI error bars on seaborn bar containers."""
    bar_containers = [
        container for container in ax.containers if hasattr(container, "patches")
    ]

    if not bar_containers or summary.empty:
        return

    if hue_col is None:
        lookup = summary.set_index(x_col)
        for bar, x_value in zip(bar_containers[0].patches, x_order):
            if x_value not in lookup.index:
                continue

            row = lookup.loc[x_value]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            mean = float(row[mean_col])
            lower = float(row[lower_col])
            upper = float(row[upper_col])

            ax.errorbar(
                bar.get_x() + bar.get_width() / 2.0,
                mean,
                yerr=[[max(mean - lower, 0.0)], [max(upper - mean, 0.0)]],
                fmt="none",
                ecolor=color,
                elinewidth=linewidth,
                capsize=capsize,
                capthick=linewidth,
                zorder=5,
            )

        return

    if hue_order is None:
        hue_order = list(summary[hue_col].dropna().unique())

    lookup = summary.set_index([x_col, hue_col])

    for container, hue_value in zip(bar_containers, hue_order):
        for bar, x_value in zip(container.patches, x_order):
            key = (x_value, hue_value)

            if key not in lookup.index:
                continue

            row = lookup.loc[key]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            mean = float(row[mean_col])  # type: ignore[no-untyped-call]
            lower = float(row[lower_col])  # type: ignore[no-untyped-call]
            upper = float(row[upper_col])  # type: ignore[no-untyped-call]

            ax.errorbar(
                bar.get_x() + bar.get_width() / 2.0,
                mean,
                yerr=[[max(mean - lower, 0.0)], [max(upper - mean, 0.0)]],
                fmt="none",
                ecolor=color,
                elinewidth=linewidth,
                capsize=capsize,
                capthick=linewidth,
                zorder=5,
            )


def draw_grouped_barplot_with_ci(
    summary: pd.DataFrame,
    *,
    ax,
    x_col: str,
    hue_col: str,
    x_order: Sequence[str],
    hue_order: Sequence[str],
    palette: Mapping[str, str] | Sequence[str],
    ylabel: str,
    xlabel: str = "",
    ylim: tuple[float, float] | None = None,
) -> None:
    """Draw a seaborn grouped mean barplot with explicit 95% CI error bars."""
    sns.barplot(
        data=summary,
        x=x_col,
        y="mean",
        hue=hue_col,
        order=x_order,
        hue_order=hue_order,
        palette=palette,
        errorbar=None,
        edgecolor="black",
        ax=ax,
    )

    add_ci_errorbars(
        ax,
        summary,
        x_col=x_col,
        x_order=x_order,
        hue_col=hue_col,
        hue_order=hue_order,
    )

    ax.set_xlabel(xlabel, fontsize="x-large")
    ax.set_ylabel(ylabel, fontsize="x-large")

    if ylim is not None:
        ax.set_ylim(*ylim)


def draw_single_barplot_with_ci(
    summary: pd.DataFrame,
    *,
    ax,
    x_col: str,
    x_order: Sequence[str],
    palette: Mapping[str, str] | Sequence[str],
    ylabel: str,
    xlabel: str = "",
    ylim: tuple[float, float] | None = None,
    step: float = 0.25,
    use_step: bool = False,
) -> None:
    """Draw a single-series mean barplot with explicit 95% CI error bars."""
    lookup = summary.set_index(x_col)
    positions = np.arange(len(x_order))
    means: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    colors: list[str] = []

    for index, x_value in enumerate(x_order):
        if x_value not in lookup.index:
            means.append(np.nan)
            lower_errors.append(0.0)
            upper_errors.append(0.0)

        else:
            row = lookup.loc[x_value]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            mean = float(row["mean"])
            lower = float(row["ci95_min"])
            upper = float(row["ci95_max"])
            means.append(mean)
            lower_errors.append(max(mean - lower, 0.0))
            upper_errors.append(max(upper - mean, 0.0))

        if isinstance(palette, Mapping):
            colors.append(
                palette.get(
                    x_value,
                    RETRO_90S_PALETTE[index % len(RETRO_90S_PALETTE)],
                )
            )
        else:
            colors.append(palette[index % len(palette)])

    ax.bar(
        positions,
        means,
        yerr=[lower_errors, upper_errors],
        color=colors,
        edgecolor="black",
        error_kw={
            "ecolor": "black",
            "elinewidth": 1.5,
            "capsize": 5.0,
            "capthick": 1.5,
        },
    )

    if use_step:
        y_min = np.floor(min(ax.get_ylim()[0], 0) / step) * step
        y_max = np.ceil(ax.get_ylim()[1] / step) * step
        y_ticks = np.arange(y_min, y_max + step / 2, step).astype(float).tolist()
    else:
        y_ticks = ax.get_yticks()

    ax.set_xticks(positions)
    ax.set_xticklabels(x_order, rotation=30, ha="right", fontsize="xx-large")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{tick:.2f}" for tick in y_ticks], fontsize="xx-large")
    ax.set_xlabel(xlabel, fontsize="xx-large")
    ax.set_ylabel(ylabel, fontsize="xx-large")

    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_macro_f1_by_model(combined: pd.DataFrame, out_dir: Path) -> None:
    """Plot macro F1-score by model.

    Args:
        combined (pd.DataFrame): Combined metrics DataFrame containing macro F1 scores.
        out_dir (Path): Output directory to save the plot.
    """
    plot_df = combined.dropna(subset=["macro_f1"]).copy()

    apply_house_style()

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.despine(fig=fig)

    model_order = [
        model
        for model in MODEL_ORDER
        if model in set(plot_df["model_label"].astype(str))
    ]

    summary = summarize_mean_ci(plot_df, ["model_label"], "macro_f1")

    draw_single_barplot_with_ci(
        summary,
        ax=ax,
        x_col="model_label",
        x_order=model_order,
        palette=MODEL_PALETTE,
        ylabel="Macro F1-score",
        xlabel="Model",
        ylim=(-0.02, 1.02),
        use_step=False,
    )

    format_decimal_ticks(ax, axis="y", decimals=2)
    save_figure(fig, out_dir / "macro_f1_by_model")


def plot_macro_f1_by_strategy(combined: pd.DataFrame, out_dir: Path) -> None:
    """Plot macro F1 by missingness strategy.

    Args:
        combined (pd.DataFrame): Combined metrics DataFrame containing macro F1 scores.
        out_dir (Path): Output directory to save the plot.
    """
    plot_df = combined.dropna(subset=["macro_f1"]).copy()
    df = plot_df.loc[plot_df["model_label"].ne("RefAllele")].copy()

    if df.empty:
        print("Skipping macro-F1 strategy plot: no non-RefAllele rows.")
        return

    strat_abbr = ["R", "RW", "RWI", "N", "NW"]

    if "strategy" in df.columns:
        strat_map = dict(zip(STRATEGY_ORDER, strat_abbr))
        df["strategy_label_abbr"] = df["strategy"].map(strat_map)

        if "strategy_label" in df.columns:
            label_map = {
                STRATEGY_LABELS[strategy]: abbreviation
                for strategy, abbreviation in zip(STRATEGY_ORDER, strat_abbr)
            }

            df["strategy_label_abbr"] = df["strategy_label_abbr"].fillna(
                df["strategy_label"].map(label_map)
            )
    else:
        strat_map = {
            STRATEGY_LABELS[strategy]: abbreviation
            for strategy, abbreviation in zip(STRATEGY_ORDER, strat_abbr)
        }

        df["strategy_label_abbr"] = df["strategy_label"].map(strat_map)

    df = df.dropna(subset=["strategy_label_abbr"])

    model_order = [
        model
        for model in MODEL_ORDER
        if model != "RefAllele" and model in set(df["model_label"].astype(str))
    ]

    model_palette = {
        model: MODEL_PALETTE[model] for model in model_order if model in MODEL_PALETTE
    }

    apply_house_style()

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    sns.despine(fig=fig)

    summary = summarize_mean_ci(
        df,
        ["strategy_label_abbr", "model_label"],
        "macro_f1",
    )

    draw_grouped_barplot_with_ci(
        summary,
        ax=ax,
        x_col="strategy_label_abbr",
        hue_col="model_label",
        x_order=strat_abbr,
        hue_order=model_order,
        palette=model_palette,
        xlabel="Simulation Strategy",
        ylabel="Macro F1-score",
        ylim=(-0.02, 1.02),
    )

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), fontsize="xx-large")
    format_decimal_ticks(ax, axis="y", decimals=2)

    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        fontsize="x-large",
        title="Model",
        title_fontsize="x-large",
        shadow=True,
        fancybox=True,
    )

    add_strategy_key(fig)
    save_figure(fig, out_dir / "macro_f1_by_strategy")


def plot_best_deltas(combined: pd.DataFrame, out_dir: Path) -> None:
    """Plot PG-SUI model F1 deltas against GTImputation methods.

    Args:
        combined: Long-form model metrics containing PG-SUI and GTImputation
            macro-F1 and genotype-specific F1 scores.
        out_dir: Output directory in which to save the plot.

    Raises:
        ValueError: If required plotting columns are unavailable.
    """
    strategy_column = next(
        (column for column in ("strategy", "strategy_label") if column in combined),
        None,
    )

    if strategy_column is None:
        raise ValueError("combined must contain either 'strategy' or 'strategy_label'.")

    required_columns = {
        "dataset_id",
        strategy_column,
        "software",
        "model_label",
        "macro_f1",
    }

    missing_columns = required_columns.difference(combined.columns)
    if missing_columns:
        raise ValueError(
            f"combined is missing required columns: {sorted(missing_columns)}"
        )

    genotype_metric_labels = {
        "ref_f1": "Homozygous Ref",
        "het_f1": "Heterozygous",
        "alt_f1": "Homozygous Alt",
    }

    genotype_metric_columns = [
        column for column in genotype_metric_labels if column in combined.columns
    ]

    metric_columns = ["macro_f1", *genotype_metric_columns]
    pair_columns = ["dataset_id", strategy_column]

    plot_df = combined.dropna(
        subset=[
            "dataset_id",
            strategy_column,
            "software",
            "model_label",
            "macro_f1",
        ]
    ).copy()

    for column in metric_columns:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")

    plot_df = plot_df.dropna(subset=["macro_f1"])

    pgsui_present = set(
        plot_df.loc[plot_df["software"].eq("PG-SUI"), "model_label"].astype(str)
    )

    pgsui_present.discard(PGSUI_MODEL_LABELS["ImputeRefAllele"])
    gti_present = set(
        plot_df.loc[plot_df["software"].eq("GTImputation"), "model_label"].astype(str)
    )

    pgsui_order = [
        PGSUI_MODEL_LABELS[model]
        for model in PGSUI_MODELS
        if PGSUI_MODEL_LABELS[model] in pgsui_present
    ]

    pgsui_order.extend(sorted(pgsui_present - set(pgsui_order)))

    gti_order = [
        GTI_MODEL_LABELS[method]
        for method in GTI_METHODS
        if GTI_MODEL_LABELS[method] in gti_present
    ]

    gti_order.extend(sorted(gti_present - set(gti_order)))

    pgsui_df = (
        plot_df.loc[
            plot_df["software"].eq("PG-SUI")
            & plot_df["model_label"].astype(str).isin(pgsui_order),
            [*pair_columns, "model_label", *metric_columns],
        ]
        .rename(
            columns={
                "model_label": "pgsui_model",
                **{column: f"pgsui_{column}" for column in metric_columns},
            }
        )
        .drop_duplicates(subset=[*pair_columns, "pgsui_model"])
    )

    gti_df = (
        plot_df.loc[
            plot_df["software"].eq("GTImputation")
            & plot_df["model_label"].astype(str).isin(gti_order),
            [*pair_columns, "model_label", *metric_columns],
        ]
        .rename(
            columns={
                "model_label": "gti_method",
                **{column: f"gti_{column}" for column in metric_columns},
            }
        )
        .drop_duplicates(subset=[*pair_columns, "gti_method"])
    )

    plot_df = pgsui_df.merge(
        gti_df, on=pair_columns, how="inner", validate="many_to_many"
    )

    plot_df["delta_macro_f1"] = plot_df["pgsui_macro_f1"] - plot_df["gti_macro_f1"]

    if plot_df.empty:
        print("Skipping PG-SUI vs GTImputation delta plot: no paired scores.")
        return

    summary = (
        plot_df.groupby(["pgsui_model", "gti_method"], observed=True)["delta_macro_f1"]
        .agg(n="count", mean="mean", sd="std")
        .reset_index()
    )

    summary["ci95"] = (
        1.96 * summary["sd"].fillna(0.0) / np.sqrt(summary["n"].clip(lower=1))
    )

    summary["ci95_min"] = summary["mean"] - summary["ci95"]
    summary["ci95_max"] = summary["mean"] + summary["ci95"]
    summary["gti_method_label"] = summary["gti_method"]

    gti_hue_order = [method for method in gti_order]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    sns.despine(fig=fig)

    sns.barplot(
        data=summary,
        x="pgsui_model",
        y="mean",
        hue="gti_method_label",
        order=pgsui_order,
        hue_order=gti_hue_order,
        palette=["#4C72B0", "#DD8452"],
        errorbar=None,
        edgecolor="black",
        ax=ax,
    )

    summary_lookup = summary.set_index(["pgsui_model", "gti_method_label"])
    observed_values = summary[["ci95_min", "ci95_max"]].to_numpy().ravel().tolist()

    for container, gti_method_label in zip(ax.containers, gti_hue_order):
        for bar, pgsui_model in zip(container, pgsui_order):
            if (pgsui_model, gti_method_label) not in summary_lookup.index:
                continue

            row = summary_lookup.loc[(pgsui_model, gti_method_label)]

            mean = float(row["mean"])  # type: ignore[no-untyped-call]
            ci95_min = float(row["ci95_min"])  # type: ignore[no-untyped-call]
            ci95_max = float(row["ci95_max"])  # type: ignore[no-untyped-call]
            x_position = bar.get_x() + bar.get_width() / 2

            ax.errorbar(
                x_position,
                mean,
                yerr=[
                    [mean - ci95_min],
                    [ci95_max - mean],
                ],
                fmt="none",
                ecolor="black",
                elinewidth=1.5,
                capsize=5.0,
                capthick=1.5,
                zorder=3,
            )

    y_padding = max(0.025, (max(observed_values) - min(observed_values)) * 0.10)

    ax.set_ylim(
        min(min(observed_values) - y_padding, -y_padding),
        max(observed_values) + y_padding,
    )

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xticks(np.arange(len(pgsui_order)))
    ax.set_xticklabels(pgsui_order, rotation=35, ha="right", fontsize="xx-large")

    format_decimal_ticks(ax, axis="both", decimals=2)
    ax.set_xlabel("PG-SUI Model", fontsize="xx-large")
    ax.set_ylabel("Δ Macro F1-score", fontsize="xx-large")

    ax.legend(
        title="Baseline (PG-SUI − GTImputation)",
        fontsize="xx-large",
        title_fontsize="xx-large",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=len(gti_hue_order),
        shadow=True,
        fancybox=True,
    )

    fig.tight_layout()
    save_figure(fig, out_dir / "best_macro_f1_delta")

    genotype_delta_frames = []
    for metric_column, genotype_label in genotype_metric_labels.items():
        if metric_column not in genotype_metric_columns:
            continue

        pgsui_column = f"pgsui_{metric_column}"
        gti_column = f"gti_{metric_column}"
        genotype_df = plot_df.dropna(subset=[pgsui_column, gti_column]).copy()

        if genotype_df.empty:
            continue

        genotype_df["genotype_class"] = genotype_label
        genotype_df["delta_genotype_f1"] = (
            genotype_df[pgsui_column] - genotype_df[gti_column]
        )

        genotype_delta_frames.append(
            genotype_df[
                [
                    "pgsui_model",
                    "gti_method",
                    "genotype_class",
                    "delta_genotype_f1",
                ]
            ]
        )

    if not genotype_delta_frames:
        print("Skipping genotype-specific delta plot: no paired genotype F1 scores.")
        return

    genotype_delta_df = pd.concat(genotype_delta_frames, ignore_index=True)

    genotype_summary = (
        genotype_delta_df.groupby(
            ["pgsui_model", "gti_method", "genotype_class"],
            observed=True,
        )["delta_genotype_f1"]
        .agg(n="count", mean="mean", sd="std")
        .reset_index()
    )

    genotype_summary["ci95"] = (
        1.96
        * genotype_summary["sd"].fillna(0.0)
        / np.sqrt(genotype_summary["n"].clip(lower=1))
    )

    genotype_summary["ci95_min"] = genotype_summary["mean"] - genotype_summary["ci95"]

    genotype_summary["ci95_max"] = genotype_summary["mean"] + genotype_summary["ci95"]

    genotype_summary["gti_method_label"] = genotype_summary["gti_method"]

    genotype_hue_order = [
        label
        for label in genotype_metric_labels.values()
        if label in set(genotype_summary["genotype_class"])
    ]

    genotype_palette = {
        "Homozygous Ref": RETRO_90S_PALETTE[0],
        "Heterozygous": RETRO_90S_PALETTE[4],
        "Homozygous Alt": RETRO_90S_PALETTE[1],
    }

    genotype_observed_values = (
        genotype_summary[["ci95_min", "ci95_max"]].to_numpy().ravel().tolist()
    )

    genotype_y_padding = max(
        0.025,
        (max(genotype_observed_values) - min(genotype_observed_values)) * 0.10,
    )

    genotype_y_limits = (
        min(min(genotype_observed_values) - genotype_y_padding, -genotype_y_padding),
        max(genotype_observed_values) + genotype_y_padding,
    )

    gti_panel_order = list(reversed(gti_hue_order))

    fig, axes = plt.subplots(
        1,
        len(gti_panel_order),
        figsize=(6.8 * len(gti_panel_order), 5.2),
        sharey=True,
    )

    axes = np.atleast_1d(axes)
    sns.despine(fig=fig)

    gt_pal_map = {
        "Homozygous Ref": "Homozygous Reference",
        "Heterozygous": "Heterozygous",
        "Homozygous Alt": "Homozygous Alternate",
    }

    genotype_hue_labels = [gt_pal_map[label] for label in genotype_hue_order]

    for ax, gti_method_label in zip(axes, gti_panel_order):
        method_summary = genotype_summary.loc[
            genotype_summary["gti_method_label"].eq(gti_method_label)
        ].copy()

        method_summary["genotype_class"] = method_summary["genotype_class"].map(
            gt_pal_map
        )

        sns.barplot(
            data=method_summary,
            x="pgsui_model",
            y="mean",
            hue="genotype_class",
            order=pgsui_order,
            hue_order=genotype_hue_labels,
            palette={
                gt_pal_map[label]: genotype_palette[label] for label in genotype_palette
            },
            errorbar=None,
            edgecolor="black",
            ax=ax,
        )

        method_lookup = method_summary.set_index(["pgsui_model", "genotype_class"])

        for container, genotype_label in zip(ax.containers, genotype_hue_labels):
            for bar, pgsui_model in zip(container, pgsui_order):
                if (pgsui_model, genotype_label) not in method_lookup.index:
                    continue

                row = method_lookup.loc[(pgsui_model, genotype_label)]
                mean = float(row["mean"])
                ci95_min = float(row["ci95_min"])
                ci95_max = float(row["ci95_max"])
                x_position = bar.get_x() + bar.get_width() / 2

                ax.errorbar(
                    x_position,
                    mean,
                    yerr=[
                        [mean - ci95_min],
                        [ci95_max - mean],
                    ],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.5,
                    capsize=5.0,
                    capthick=1.5,
                    zorder=3,
                )

        ax.set_ylim(
            min(genotype_y_limits[0], -0.02),
            max(genotype_y_limits[1], 0.02),
        )

        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xticks(np.arange(len(pgsui_order)))
        ax.set_xticklabels(pgsui_order, rotation=35, ha="right", fontsize="xx-large")

        format_decimal_ticks(ax, axis="y", decimals=2)

        ax.set_xlabel("PG-SUI Model", fontsize="xx-large")
        ax.set_ylabel("Mean Delta F1-score", fontsize="xx-large")

    handles, labels = axes[-1].get_legend_handles_labels()

    for ax in axes:
        legend = ax.get_legend()

        if legend is not None:
            legend.remove()

    fig.legend(
        handles,
        labels,
        title="Genotype",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(genotype_hue_order),
        shadow=True,
        fancybox=True,
        fontsize="xx-large",
        title_fontsize="xx-large",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    save_figure(fig, out_dir / "best_genotype_f1_delta")


def plot_best_delta(combined: pd.DataFrame, out_dir: Path) -> None:
    plot_best_deltas(combined, out_dir)


def plot_runtime_by_model(combined: pd.DataFrame, out_dir: Path) -> None:
    """Plot runtime by model.

    Args:
        combined (pd.DataFrame): Combined metrics DataFrame containing runtime information.
        out_dir (Path): Output directory to save the plot.
    """
    plot_df = combined.dropna(subset=["runtime_seconds"]).copy()
    plot_df = plot_df.loc[plot_df["runtime_seconds"].gt(0)].copy()

    if plot_df.empty:
        return

    apply_house_style()

    panel_labels = runtime_panel_backend_labels(plot_df)

    fig, axes = plt.subplots(
        1,
        len(panel_labels),
        figsize=(6.8 * len(panel_labels), 4.8),
        sharey=True,
    )

    axes = np.atleast_1d(axes)
    sns.despine(fig=fig)

    for ax, panel_label in zip(axes, panel_labels):
        panel_df = runtime_panel_subset(plot_df, panel_label)
        order = runtime_panel_model_order(panel_df)
        palette = runtime_panel_palette(order)

        summary = summarize_mean_ci(panel_df, ["model_label"], "runtime_seconds")

        draw_single_barplot_with_ci(
            summary,
            ax=ax,
            x_col="model_label",
            x_order=order,
            palette=palette,
            xlabel="Model",
            ylabel="Execution Time (seconds)",
            use_step=False,
        )

        ax.set_title(runtime_panel_title(panel_label), fontsize="xx-large")
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), fontsize="xx-large")
        format_decimal_ticks(ax, axis="y", decimals=2)

    save_figure(fig, out_dir / "runtime_by_model")


def plot_parallel_adjusted_runtime_by_model(
    combined: pd.DataFrame, out_dir: Path
) -> None:
    """Plot parallel-adjusted runtime by model.

    Args:
        combined (pd.DataFrame): Combined metrics DataFrame containing runtime information.
        out_dir (Path): Output directory to save the plot.
    """
    plot_df = combined.dropna(subset=["parallel_adjusted_runtime_seconds"]).copy()

    plot_df = plot_df.loc[plot_df["parallel_adjusted_runtime_seconds"].gt(0)].copy()

    if plot_df.empty:
        return

    apply_house_style()

    panel_labels = runtime_panel_backend_labels(plot_df)

    fig, axes = plt.subplots(
        1,
        len(panel_labels),
        figsize=(6.8 * len(panel_labels), 4.8),
        sharey=True,
    )

    axes = np.atleast_1d(axes)
    sns.despine(fig=fig)

    for ax, panel_label in zip(axes, panel_labels):
        panel_df = runtime_panel_subset(plot_df, panel_label)
        order = runtime_panel_model_order(panel_df)
        palette = runtime_panel_palette(order)

        summary = summarize_mean_ci(
            panel_df,
            ["model_label"],
            "parallel_adjusted_runtime_seconds",
        )
        draw_single_barplot_with_ci(
            summary,
            ax=ax,
            x_col="model_label",
            x_order=order,
            palette=palette,
            xlabel="Model",
            ylabel="Parallel-adjusted Execution Time (seconds)",
            use_step=False,
        )
        ax.set_title(runtime_panel_title(panel_label), fontsize="x-large")
        format_decimal_ticks(ax, axis="y", decimals=2)

    save_figure(fig, out_dir / "parallel_adjusted_runtime_by_model")


def plot_class_f1_by_model(combined_class: pd.DataFrame, out_dir: Path) -> None:
    """Plot per-genotype-class F1 as horizontal small multiples.

    One panel per genotype class (REF/HET/ALT), with models on a shared vertical
    axis. Horizontal bars avoid rotated tick labels colliding across panels and
    keep the long model names legible.

    Args:
        combined_class (pd.DataFrame): Combined class report DataFrame.
        out_dir (Path): Output directory for saving the plot.
    """
    plot_df = combined_class.dropna(subset=["f1"]).copy()

    if plot_df.empty:
        return

    apply_house_style()

    model_order = [
        model
        for model in MODEL_ORDER
        if model in set(plot_df["model_label"].astype(str))
    ]

    # Top-to-bottom reads best when the first model sits at the top of the axis.
    y_positions = np.arange(len(model_order))[::-1]

    fig, axes = plt.subplots(
        1, len(GENOTYPE_CLASS_ORDER), figsize=(15.0, 5.4), sharey=True
    )

    axes = np.atleast_1d(axes)
    sns.despine(fig=fig)

    for ax, genotype_class in zip(axes, GENOTYPE_CLASS_ORDER):
        class_df = plot_df.loc[plot_df["genotype_class"].eq(genotype_class)]

        summary = summarize_mean_ci(class_df, ["model_label"], "f1").set_index(
            "model_label"
        )

        means = [
            summary.loc[model, "mean"] if model in summary.index else np.nan
            for model in model_order
        ]

        means = [float(x) for x in means]  # type: ignore[no-untyped-call]

        lower = [
            (
                max(
                    float(summary.loc[model, "mean"] - summary.loc[model, "ci95_min"]),  # type: ignore[no-untyped-call]
                    0.0,
                )
                if model in summary.index
                else 0.0
            )
            for model in model_order
        ]

        upper = [
            (
                max(
                    float(summary.loc[model, "ci95_max"]) - summary.loc[model, "mean"],  # type: ignore[no-untyped-call]
                    0.0,
                )
                if model in summary.index
                else 0.0
            )
            for model in model_order
        ]
        colors = [MODEL_PALETTE.get(model, "#777777") for model in model_order]

        ax.barh(
            y_positions,
            means,
            xerr=[lower, upper],
            color=colors,
            edgecolor="black",
            linewidth=0.9,
            error_kw={
                "ecolor": "black",
                "elinewidth": 1.4,
                "capsize": 4.0,
                "capthick": 1.4,
            },
        )

        for y_pos, mean, up in zip(y_positions, means, upper):
            if np.isfinite(mean):
                ax.text(
                    min(mean + up + 0.03, 1.06),
                    y_pos,
                    f"{mean:.2f}",
                    va="center",
                    ha="left",
                    fontsize="x-large",
                    color="#222222",
                )

        ax.set_xlim(0.0, 1.08)
        ax.set_xticks(np.arange(0.0, 1.01, 0.25))

        ax.set_title(
            GENOTYPE_CLASS_SHORT_LABELS.get(genotype_class, genotype_class),
            fontsize="x-large",
        )

        ax.set_xlabel("F1-score", fontsize="x-large")
        format_decimal_ticks(ax, axis="x", decimals=2)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(model_order, fontsize="x-large")

    fig.suptitle(
        "Per-Genotype-Class F1 by Model", fontsize="x-large", fontweight="bold", y=1.02
    )

    fig.tight_layout()
    save_figure(fig, out_dir / "class_f1_by_model")


def plot_class_f1_by_software(combined_class: pd.DataFrame, out_dir: Path) -> None:
    """Plot F1-score by genotype class and model.

    GTImputation Naive and SOM are kept separate because they have different performance profiles.

    Args:
        combined_class (pd.DataFrame): Combined class report DataFrame.
        out_dir (Path): Output directory for saving the plot.
    """
    plot_df = combined_class.dropna(subset=["f1"]).copy()

    if plot_df.empty:
        return

    plot_df["model_label"] = plot_df["model_label"].astype(str)
    model_order = present_model_order(plot_df)
    model_palette = present_model_palette(model_order)

    apply_house_style()

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    sns.despine(fig=fig)

    summary = summarize_mean_ci(plot_df, ["genotype_class", "model_label"], "f1")

    draw_grouped_barplot_with_ci(
        summary,
        ax=ax,
        x_col="genotype_class",
        hue_col="model_label",
        x_order=GENOTYPE_CLASS_ORDER,
        hue_order=model_order,
        palette=model_palette,
        xlabel="Genotype Class",
        ylabel="F1-score",
        ylim=(-0.02, 1.02),
    )

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(["REF", "HET", "ALT"], fontsize="xx-large")

    format_decimal_ticks(ax, axis="y", decimals=2)
    ax.set_ylim(-0.02, 1.02)

    ax.legend(
        title="Model",
        shadow=True,
        fancybox=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=4,
        fontsize="xx-large",
        title_fontsize="xx-large",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    save_figure(fig, out_dir / "class_f1_by_software_model")


def plot_class_f1_by_strategy(combined_class: pd.DataFrame, out_dir: Path) -> None:
    """Plot genotype-class F1 scores by missingness strategy.

    Args:
        combined_class: Combined class report DataFrame.
        out_dir: Output directory for saving the plot.
    """
    plot_df = combined_class.dropna(subset=["f1"]).copy()
    plot_df = plot_df.loc[plot_df["model_label"].ne("RefAllele")].copy()

    if plot_df.empty:
        print("Skipping class-F1 strategy plot: no non-RefAllele rows.")
        return

    plot_df = add_strategy_abbreviations(plot_df)
    plot_df = plot_df.dropna(subset=["strategy_label_abbr"])
    plot_df["model_label"] = plot_df["model_label"].astype(str)

    model_order = present_model_order(plot_df)

    genotype_palette = {
        "Homozygous Reference": RETRO_90S_PALETTE[0],
        "Heterozygous": RETRO_90S_PALETTE[4],
        "Homozygous Alternate": RETRO_90S_PALETTE[1],
    }

    apply_house_style()

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16.0, 8.6),
        sharey=True,
    )

    axes = np.asarray(axes).ravel()
    sns.despine(fig=fig)

    for axis_index, (ax, model_label) in enumerate(zip(axes, model_order)):
        model_df = plot_df.loc[plot_df["model_label"].eq(model_label)]

        summary = summarize_mean_ci(
            model_df,
            ["strategy_label_abbr", "genotype_class"],
            "f1",
        )

        draw_grouped_barplot_with_ci(
            summary,
            ax=ax,
            x_col="strategy_label_abbr",
            hue_col="genotype_class",
            x_order=STRATEGY_ABBR_ORDER,
            hue_order=GENOTYPE_CLASS_ORDER,
            palette=genotype_palette,
            xlabel="Simulation Strategy",
            ylabel="F1-score",
            ylim=(-0.02, 1.02),
        )

        ax.set_title(model_label, fontsize="x-large")
        ax.set_xticks(np.arange(len(STRATEGY_ABBR_ORDER)))
        ax.set_xticklabels(STRATEGY_ABBR_ORDER, fontsize="xx-large")
        ax.tick_params(axis="y", labelsize="xx-large")

        if axis_index % 4 != 0:
            ax.set_ylabel("")

    for ax in axes[len(model_order) :]:
        ax.set_axis_off()

    handles, labels = axes[0].get_legend_handles_labels()

    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.legend(
        handles,
        labels,
        title="Genotype Class",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(GENOTYPE_CLASS_ORDER),
        frameon=False,
        fontsize="xx-large",
        title_fontsize="xx-large",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    add_strategy_key(fig)
    save_figure(fig, out_dir / "class_f1_by_strategy")


def plot_class_f1_strategy_model_bars(
    combined_class: pd.DataFrame, out_dir: Path
) -> None:
    """Plot genotype-class F1 across simulation strategies for every model."""
    plot_df = combined_class.dropna(subset=["f1"]).copy()
    if plot_df.empty:
        return

    plot_df = add_strategy_abbreviations(plot_df)

    plot_df = plot_df.dropna(
        subset=["strategy_label_abbr", "genotype_class", "model_label", "f1"]
    ).copy()

    if plot_df.empty:
        print("Skipping class-F1 strategy/model barplot: no plottable rows.")
        return

    plot_df["model_label"] = plot_df["model_label"].astype(str)
    plot_df["genotype_class"] = plot_df["genotype_class"].astype(str)
    model_order = present_model_order(plot_df)
    model_palette = present_model_palette(model_order)
    summary = summarize_mean_ci(
        plot_df,
        ["genotype_class", "strategy_label_abbr", "model_label"],
        "f1",
    )
    summary["strategy_label_abbr"] = summary["strategy_label_abbr"].astype(str)
    summary["model_label"] = summary["model_label"].astype(str)

    fig, axes = plt.subplots(
        1,
        len(GENOTYPE_CLASS_ORDER),
        figsize=(6.8 * len(GENOTYPE_CLASS_ORDER), 5.2),
        sharey=True,
    )

    axes = np.atleast_1d(axes)
    sns.despine(fig=fig)

    for ax, genotype_class in zip(axes, GENOTYPE_CLASS_ORDER):
        class_summary = summary.loc[summary["genotype_class"].eq(genotype_class)].copy()

        draw_grouped_barplot_with_ci(
            class_summary,
            ax=ax,
            x_col="strategy_label_abbr",
            hue_col="model_label",
            x_order=STRATEGY_ABBR_ORDER,
            hue_order=model_order,
            palette=model_palette,
            xlabel="Simulation Strategy",
            ylabel="F1-score" if ax is axes[0] else "",
            ylim=(-0.02, 1.02),
        )

        ax.set_title(
            GENOTYPE_CLASS_SHORT_LABELS.get(genotype_class, genotype_class),
            fontsize="x-large",
        )

        ax.set_xticks(np.arange(len(STRATEGY_ABBR_ORDER)))
        ax.set_xticklabels(STRATEGY_ABBR_ORDER, fontsize="x-large")
        ax.tick_params(axis="y", labelsize="x-large")

    handles, labels = axes[-1].get_legend_handles_labels()

    for ax in axes:
        legend = ax.get_legend()

        if legend is not None:
            legend.remove()

    fig.legend(
        handles,
        labels,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        shadow=True,
        fancybox=True,
        fontsize="x-large",
        title_fontsize="x-large",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    add_strategy_key(fig)
    save_figure(fig, out_dir / "class_f1_strategy_model_bars")


def plot_mcc_by_strategy_model(combined: pd.DataFrame, out_dir: Path) -> None:
    """Plot MCC across simulation strategies for every model."""
    plot_df = combined.dropna(subset=["mcc"]).copy()

    if plot_df.empty:
        return

    plot_df = add_strategy_abbreviations(plot_df)

    plot_df = plot_df.dropna(subset=["strategy_label_abbr", "model_label", "mcc"])

    if plot_df.empty:
        print("Skipping MCC strategy/model plot: no plottable rows.")
        return

    plot_df["model_label"] = plot_df["model_label"].astype(str)
    model_order = present_model_order(plot_df)
    model_palette = present_model_palette(model_order)

    summary = summarize_mean_ci(
        plot_df,
        ["strategy_label_abbr", "model_label"],
        "mcc",
    )

    summary["strategy_label_abbr"] = summary["strategy_label_abbr"].astype(str)
    summary["model_label"] = summary["model_label"].astype(str)

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    sns.despine(fig=fig)

    draw_grouped_barplot_with_ci(
        summary,
        ax=ax,
        x_col="strategy_label_abbr",
        hue_col="model_label",
        x_order=STRATEGY_ABBR_ORDER,
        hue_order=model_order,
        palette=model_palette,
        xlabel="Simulation Strategy",
        ylabel="Matthews Correlation Coefficient",
    )

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(np.arange(len(STRATEGY_ABBR_ORDER)))
    ax.set_xticklabels(STRATEGY_ABBR_ORDER, fontsize="x-large")
    observed_values = summary[["ci95_min", "ci95_max"]].to_numpy().ravel()
    observed_values = observed_values[np.isfinite(observed_values)]

    if observed_values.size:
        lower = min(float(observed_values.min()) - 0.08, -0.08)
        upper = min(float(observed_values.max()) + 0.08, 1.02)
        ax.set_ylim(lower, upper)

    ax.tick_params(axis="y", labelsize="x-large")

    ax.legend(
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=4,
        shadow=True,
        fancybox=True,
        fontsize="x-large",
        title_fontsize="x-large",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    add_strategy_key(fig)
    save_figure(fig, out_dir / "mcc_by_strategy_model")


def plot_f1_class_balance(combined: pd.DataFrame, out_dir: Path) -> None:
    """Plot macro F1 against genotype-class F1 balance."""
    summary = f1_class_balance_summary_df(combined)

    if summary.empty:
        print("Skipping F1 class-balance plot: no class-specific F1 rows.")
        return

    summary["model_label"] = summary["model_label"].astype(str)
    summary["software"] = summary["software"].astype(str)
    markers = {"GTImputation": "s", "PG-SUI": "o"}

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    sns.despine(fig=fig)

    mcc_values = (
        pd.to_numeric(summary.get("mean_mcc", pd.Series(dtype=float)), errors="coerce")
        if "mean_mcc" in summary.columns
        else pd.Series(np.zeros(len(summary)), index=summary.index)
    )

    label_points: list[tuple[float, float]] = []
    label_texts: list[str] = []

    for index, row in summary.iterrows():
        model_label = str(row["model_label"])
        software = str(row["software"])
        mcc = as_float(mcc_values.loc[index]) if index in mcc_values.index else 0.0  # type: ignore[no-untyped-call]

        if not np.isfinite(mcc):
            mcc = 0.0

        size = 130.0 + 300.0 * np.clip((mcc + 1.0) / 2.0, 0.0, 1.0)

        x_value = float(row["mean_macro_f1"])
        y_value = float(row["mean_class_f1_spread"])

        ax.scatter(
            x_value,
            y_value,
            s=size,
            marker=markers.get(software, "o"),
            color=MODEL_PALETTE.get(model_label, "#777777"),
            edgecolor="black",
            linewidth=0.9,
            alpha=0.82,
            zorder=3,
        )

        label_points.append((x_value, y_value))
        label_texts.append(model_label)

    ax.axvline(
        float(summary["mean_macro_f1"].median()),
        color="black",
        linestyle=":",
        linewidth=1.0,
    )

    ax.axhline(
        float(summary["mean_class_f1_spread"].median()),
        color="black",
        linestyle=":",
        linewidth=1.0,
    )

    ax.set_xlim(
        max(0.0, float(summary["mean_macro_f1"].min()) - 0.09),
        min(1.04, float(summary["mean_macro_f1"].max()) + 0.14),
    )

    ax.set_ylim(0.0, min(1.0, float(summary["mean_class_f1_spread"].max()) + 0.14))

    ax.set_xlabel("Mean Macro F1-score", fontsize="xx-large")
    ax.set_ylabel("Mean Class F1 Spread (max − min)", fontsize="xx-large")
    ax.set_title("Overall Accuracy vs Genotype-Class Balance", fontsize="xx-large")

    ax.tick_params(axis="both", labelsize="xx-large")

    ax.annotate(
        "Balanced across genotypes ↓",
        xy=(0.02, 0.03),
        xycoords="axes fraction",
        fontsize="xx-large",
        color="#555555",
        ha="left",
        va="bottom",
    )
    repel_point_labels(ax, label_points, label_texts, fontsize="xx-large")

    legend_handles = [
        mlines.Line2D(
            [0],
            [0],
            marker=marker,
            color="white",
            markerfacecolor="#777777",
            markeredgecolor="black",
            markersize=10,
            linestyle="",
            label=software,
        )
        for software, marker in markers.items()
        if software in set(summary["software"])
    ]

    ax.legend(
        handles=legend_handles,
        title="Software",
        loc="upper right",
        shadow=True,
        fancybox=True,
        fontsize="xx-large",
        title_fontsize="xx-large",
    )

    fig.tight_layout()
    save_figure(fig, out_dir / "f1_class_balance_by_model")


def plot_cpu_gpu_runtime_difference(
    runtime_combined: pd.DataFrame, out_dir: Path
) -> None:
    """Plot paired CPU-minus-GPU runtime differences for PG-SUI deep models."""
    paired = cpu_gpu_runtime_delta_df(runtime_combined)

    if paired.empty:
        print("Skipping CPU/GPU runtime-difference plot: no paired deep-model runs.")
        return

    model_order = [
        PGSUI_MODEL_LABELS[model]
        for model in PGSUI_DEEP_MODELS
        if PGSUI_MODEL_LABELS[model] in set(paired["model_label"].astype(str))
    ]

    palette = present_model_palette(model_order)

    delta_summary = summarize_mean_ci(paired, ["model_label"], "cpu_minus_gpu_seconds")

    speedup_summary = summarize_mean_ci(paired, ["model_label"], "cpu_gpu_speedup")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    sns.despine(fig=fig)

    draw_single_barplot_with_ci(
        delta_summary,
        ax=axes[0],
        x_col="model_label",
        x_order=model_order,
        palette=palette,
        xlabel="Deep Model",
        ylabel="CPU - GPU Execution Time (seconds)",
        use_step=False,
    )

    axes[0].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[0].set_title("Absolute Execution Time Difference", fontsize="xx-large")

    draw_single_barplot_with_ci(
        speedup_summary,
        ax=axes[1],
        x_col="model_label",
        x_order=model_order,
        palette=palette,
        xlabel="Deep Model",
        ylabel="CPU / GPU Execution Time Ratio",
        step=0.25,
        use_step=True,
    )

    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_title("CPU/GPU Execution Time Ratio", fontsize="xx-large")

    for ax in axes:
        ax.set_ylim(bottom=0.0, top=max(float(np.ceil(ax.get_ylim()[1])), 1.0))
        ax.tick_params(axis="both", labelsize="xx-large")

    fig.tight_layout()
    save_figure(fig, out_dir / "cpu_gpu_runtime_difference")


def plot_performance_efficiency_frontier(combined: pd.DataFrame, out_dir: Path) -> None:
    """Plot a macro-F1, runtime, and MCC efficiency frontier.

    Args:
        combined (pd.DataFrame): DataFrame containing model performance metrics.
        out_dir (Path): Directory to save the efficiency frontier figure.
    """
    required = {"software", "model_label", "macro_f1", "runtime_seconds"}

    if combined.empty or not required.issubset(combined.columns):
        return

    plot_df = combined.copy()
    plot_df["runtime_seconds"] = pd.to_numeric(
        plot_df["runtime_seconds"], errors="coerce"
    )

    plot_df["macro_f1"] = pd.to_numeric(plot_df["macro_f1"], errors="coerce")

    if "mcc" in plot_df.columns:
        plot_df["mcc"] = pd.to_numeric(plot_df["mcc"], errors="coerce")
    else:
        plot_df["mcc"] = np.nan

    plot_df = plot_df.dropna(subset=["runtime_seconds", "macro_f1"]).loc[
        lambda frame: frame["runtime_seconds"].gt(0)
    ]

    if plot_df.empty:
        print("Skipping efficiency frontier plot: no runtime/F1 rows.")
        return

    summary = (
        plot_df.groupby(["software", "model_label"], observed=True)
        .agg(
            n=("macro_f1", "count"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_mcc=("mcc", "mean"),
        )
        .reset_index()
        .dropna(subset=["mean_macro_f1", "mean_runtime_seconds"])
    )

    if summary.empty:
        return

    frontier_flags: list[bool] = []

    for _, row in summary.iterrows():
        faster_or_equal = summary["mean_runtime_seconds"].le(
            row["mean_runtime_seconds"]
        )

        better_or_equal = summary["mean_macro_f1"].ge(row["mean_macro_f1"])
        strictly_better = summary["mean_runtime_seconds"].lt(
            row["mean_runtime_seconds"]
        ) | summary["mean_macro_f1"].gt(row["mean_macro_f1"])

        frontier_flags.append(
            not bool((faster_or_equal & better_or_equal & strictly_better).any())
        )

    summary["is_frontier"] = frontier_flags

    fig, ax = plt.subplots(figsize=(9.6, 6.4))

    fig.patch.set_facecolor("#FAFAFC")
    ax.set_facecolor("#FAFAFC")

    sns.despine(fig=fig, ax=ax, trim=True)

    markers = {"GTImputation": "s", "PG-SUI": "o"}
    runtime_values = summary["mean_runtime_seconds"].astype(float)
    f1_values = summary["mean_macro_f1"].astype(float)
    x_min = max(float(runtime_values.min()) * 0.55, 1e-3)
    x_max = float(runtime_values.max()) * 1.95
    y_min = max(0.0, float(f1_values.min()) - 0.02)
    y_max = min(1.0, float(f1_values.max()) + 0.02)

    ax.axhspan(
        float(f1_values.quantile(0.75)),
        y_max,
        color=RETRO_90S_PALETTE[2],
        alpha=0.08,
        zorder=0,
    )

    ax.axvspan(
        x_min,
        float(runtime_values.quantile(0.25)),
        color=RETRO_90S_PALETTE[1],
        alpha=0.08,
        zorder=0,
    )

    ax.grid(which="major", axis="both", color="#D7D7DE", linewidth=0.8, alpha=0.7)

    ax.grid(which="minor", axis="x", color="#E8E8EF", linewidth=0.5, alpha=0.55)

    for _, row in summary.iterrows():
        model_label = str(row["model_label"])
        software = str(row["software"])
        mean_mcc = as_float(row.get("mean_mcc"))

        if not np.isfinite(mean_mcc):
            mean_mcc = 0.0

        size = 220.0 + 620.0 * np.clip((mean_mcc + 1.0) / 2.0, 0.0, 1.0)

        ax.scatter(
            float(row["mean_runtime_seconds"]),
            float(row["mean_macro_f1"]),
            s=size,
            marker=markers.get(software, "o"),
            color=MODEL_PALETTE.get(model_label, "#777777"),
            edgecolor="black",
            linewidth=1.15,
            alpha=0.92,
            zorder=3,
        )

        is_frontier = bool(row.get("is_frontier", False))

        label_offsets = {
            "Naive (GTImputation)": (8, 8),
            "SOM (GTImputation)": (8, -10),
            "MostFrequent": (10, 10),
            "Autoencoder": (12, -22),
            "VAE": (10, 14),
            "NLPCA": (10, 22),
            "UBP": (10, 18),
            "RefAllele": (6, 8),
        }

        x_offset, y_offset = label_offsets.get(
            model_label,
            (10 if is_frontier else 8, 12 if is_frontier else 8),
        )

        ax.annotate(
            model_label,
            xy=(
                float(row["mean_runtime_seconds"]),
                float(row["mean_macro_f1"]),
            ),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize="large" if is_frontier else "x-large",
            fontweight="bold" if is_frontier else "normal",
            ha="left",
            va="bottom",
            arrowprops={
                "arrowstyle": "-",
                "color": "#333333",
                "linewidth": 0.8,
                "alpha": 0.7,
                "shrinkA": 0,
                "shrinkB": 6,
            },
            path_effects=[
                path_effects.withStroke(linewidth=3.0, foreground="white", alpha=0.85)
            ],
        )

    frontier = summary.loc[summary["is_frontier"]].sort_values("mean_runtime_seconds")

    if len(frontier) >= 2:
        ax.fill_between(
            frontier["mean_runtime_seconds"].astype(float),
            frontier["mean_macro_f1"].astype(float),
            y_min,
            color="#111111",
            alpha=0.06,
            step=None,
            zorder=1,
        )

        ax.plot(
            frontier["mean_runtime_seconds"],
            frontier["mean_macro_f1"],
            color="black",
            linestyle="--",
            linewidth=2.2,
            label="Efficiency Frontier",
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Mean Execution Time (seconds, log10-scaled)", fontsize="xx-large")
    ax.set_ylabel("Mean Macro F1-score", fontsize="xx-large")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis="both", labelsize="xx-large")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    legend_handles = [
        mlines.Line2D(
            [0],
            [0],
            marker=marker,
            color="white",
            markerfacecolor="#777777",
            markeredgecolor="black",
            markersize=10,
            linestyle="",
            label=software,
        )
        for software, marker in markers.items()
        if software in set(summary["software"])
    ]

    if len(frontier) >= 2:
        legend_handles.append(
            mlines.Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=2.2,
                label="Efficiency Frontier",
            )
        )

    legend_handles.extend(
        [
            mlines.Line2D(
                [0],
                [0],
                marker="o",
                color="white",
                markerfacecolor="#777777",
                markeredgecolor="black",
                markersize=9,
                linestyle="",
                label="Smaller bubble = lower MCC",
            ),
            mlines.Line2D(
                [0],
                [0],
                marker="o",
                color="white",
                markerfacecolor="#777777",
                markeredgecolor="black",
                markersize=15,
                linestyle="",
                label="Larger bubble = higher MCC",
            ),
        ]
    )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        shadow=True,
        fancybox=True,
        fontsize="xx-large",
        title_fontsize="xx-large",
        bbox_to_anchor=(1.0, 1.02),
    )

    ax.text(
        x_min * 1.1,
        y_max - 0.012,
        "High F1 + fast runtime",
        fontsize="xx-large",
        fontweight="bold",
        ha="left",
        va="top",
        color="#333333",
        path_effects=[
            path_effects.withStroke(linewidth=3.0, foreground="white", alpha=0.85)
        ],
    )

    ax.text(
        x_max * 0.55,
        y_min + 0.012,
        "Low F1 + slow runtime",
        fontsize="xx-large",
        fontweight="bold",
        ha="left",
        va="top",
        color="#333333",
        path_effects=[
            path_effects.withStroke(linewidth=3.0, foreground="white", alpha=0.85)
        ],
    )

    ax.text(
        x_max * 0.55,
        y_max - 0.012,
        "High F1 + slow runtime",
        fontsize="xx-large",
        fontweight="bold",
        ha="left",
        va="top",
        color="#333333",
        path_effects=[
            path_effects.withStroke(linewidth=3.0, foreground="white", alpha=0.85)
        ],
    )

    ax.text(
        x_min * 1.1,
        y_min + 0.012,
        "Low F1 + fast runtime",
        fontsize="xx-large",
        fontweight="bold",
        ha="left",
        va="top",
        color="#333333",
        path_effects=[
            path_effects.withStroke(linewidth=3.0, foreground="white", alpha=0.85)
        ],
    )

    fig.tight_layout()
    save_figure(fig, out_dir / "performance_efficiency_frontier")


def with_model_category(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the four-way analysis family to each row via ``model_label``."""
    out = df.copy()
    out["model_category"] = out["model_label"].astype(str).map(MODEL_CATEGORY)
    return out


def _coerce_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def build_pgsui_gtimputation_delta_table(combined: pd.DataFrame) -> pd.DataFrame:
    """Pair PG-SUI and GTImputation macro-F1 rows by dataset and strategy."""
    required = {"dataset_id", "strategy", "software", "model_label", "macro_f1"}

    if combined.empty or not required.issubset(combined.columns):
        return pd.DataFrame()

    pair_cols = ["dataset_id", "strategy"]
    df = _coerce_numeric(combined, ["macro_f1"]).dropna(subset=["macro_f1"])

    pgsui = df.loc[
        df["software"].eq("PG-SUI"), [*pair_cols, "model_label", "macro_f1"]
    ].rename(columns={"model_label": "pgsui_model", "macro_f1": "pgsui_macro_f1"})

    gti = df.loc[
        df["software"].eq("GTImputation"), [*pair_cols, "model_label", "macro_f1"]
    ].rename(columns={"model_label": "gti_method", "macro_f1": "gti_macro_f1"})

    merged = pgsui.merge(gti, on=pair_cols, how="inner")

    if merged.empty:
        return pd.DataFrame()

    merged["delta_macro_f1"] = merged["pgsui_macro_f1"] - merged["gti_macro_f1"]
    merged["delta"] = merged["delta_macro_f1"]

    return merged


def build_category_summary_table(combined: pd.DataFrame) -> pd.DataFrame:
    """Summarize accuracy, class F1, MCC, and runtime by analysis family.

    Collapses the eight models into the four families (GTImputation Naive,
    GTImputation SOM, PG-SUI deterministic, PG-SUI deep learning) so the
    head-to-head comparison can be read at a glance.
    """
    if combined.empty or "model_label" not in combined.columns:
        return pd.DataFrame()

    numeric = ["macro_f1", "mcc", "ref_f1", "het_f1", "alt_f1", "runtime_seconds"]

    df = _coerce_numeric(with_model_category(combined), numeric)
    df = df.dropna(subset=["model_category", "macro_f1"])

    if df.empty:
        return pd.DataFrame()

    agg_spec: dict[str, tuple[str, str]] = {
        "n_runs": ("macro_f1", "count"),
        "mean_macro_f1": ("macro_f1", "mean"),
        "sd_macro_f1": ("macro_f1", "std"),
    }

    for column, label in (
        ("mcc", "mean_mcc"),
        ("ref_f1", "mean_ref_f1"),
        ("het_f1", "mean_het_f1"),
        ("alt_f1", "mean_alt_f1"),
    ):
        if column in df.columns:
            agg_spec[label] = (column, "mean")

    if "runtime_seconds" in df.columns:
        agg_spec["mean_runtime_seconds"] = ("runtime_seconds", "median")

    result = df.groupby("model_category", observed=True).agg(**agg_spec).reset_index()

    result["model_category"] = pd.Categorical(
        result["model_category"], categories=CATEGORY_ORDER, ordered=True
    )

    return result.sort_values("model_category").reset_index(drop=True)


def exact_sign_flip_test(values: Sequence[float]) -> float:
    """Exact two-sided sign-flip p-value for paired dataset-level deltas."""
    deltas = np.asarray(values, dtype=float)
    deltas = deltas[np.isfinite(deltas)]

    if deltas.size == 0:
        return math.nan

    observed = abs(float(np.mean(deltas)))
    if observed == 0.0:
        return 1.0

    n = int(deltas.size)
    sign_count = 2**n
    extreme = 0

    for mask in range(sign_count):
        signs = np.ones(n, dtype=float)
        for bit in range(n):
            if mask & (1 << bit):
                signs[bit] = -1.0
        permuted = abs(float(np.mean(signs * deltas)))
        if permuted >= observed - 1e-12:
            extreme += 1

    return float(extreme / sign_count)


def bca_mean_confidence_interval(
    values: Sequence[float],
    confidence_level: float = 0.95,
    n_resamples: int = 50_000,
    seed: int = 42,
) -> tuple[float, float]:
    """BCa bootstrap confidence interval for the mean of dataset-level deltas."""
    deltas = np.asarray(values, dtype=float)
    deltas = deltas[np.isfinite(deltas)]

    if deltas.size < 2:
        return (math.nan, math.nan)

    try:
        from scipy.stats import bootstrap
    except Exception:  # pragma: no cover - scipy is expected in validation envs
        return (math.nan, math.nan)

    try:
        result = bootstrap(
            (deltas,),
            np.mean,
            paired=False,
            vectorized=False,
            method="BCa",
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=np.random.default_rng(seed),
        )
    except Exception:
        return (math.nan, math.nan)

    return (
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def holm_adjust_p_values(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm-adjusted p-values while preserving input order."""
    raw = np.asarray(p_values, dtype=float)
    adjusted = np.full(raw.shape, np.nan, dtype=float)
    valid = np.isfinite(raw)

    if not np.any(valid):
        return adjusted

    valid_indices = np.flatnonzero(valid)
    ordered_indices = valid_indices[np.argsort(raw[valid])]
    n_tests = int(ordered_indices.size)
    running_max = 0.0

    for rank, original_index in enumerate(ordered_indices):
        holm_value = float((n_tests - rank) * raw[original_index])
        running_max = max(running_max, holm_value)
        adjusted[original_index] = min(running_max, 1.0)

    return adjusted


def build_head_to_head_table(combined: pd.DataFrame) -> pd.DataFrame:
    """Compute dataset-block PG-SUI-minus-GTImputation macro-F1 statistics."""
    merged = build_pgsui_gtimputation_delta_table(combined)

    if merged.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (pgsui_model, gti_method), group in merged.groupby(
        ["pgsui_model", "gti_method"], observed=True
    ):
        pair_deltas = group["delta"].dropna().to_numpy()
        n_pairs = int(pair_deltas.size)

        if n_pairs == 0:
            continue

        dataset_deltas = (
            group.dropna(subset=["delta"])
            .groupby("dataset_id", observed=True)["delta"]
            .mean()
            .sort_index()
            .to_numpy(dtype=float)
        )
        n_datasets = int(dataset_deltas.size)

        if n_datasets == 0:
            continue

        mean = float(np.mean(dataset_deltas))
        ci95_low, ci95_high = bca_mean_confidence_interval(dataset_deltas)
        block_p = exact_sign_flip_test(dataset_deltas)

        rows.append(
            {
                "pgsui_model": pgsui_model,
                "gti_method": gti_method,
                "n_datasets": n_datasets,
                "n_dataset_strategy_pairs": n_pairs,
                "n_pairs": n_pairs,
                "dataset_win_count": int(np.sum(dataset_deltas > 0)),
                "dataset_win_rate": float(np.mean(dataset_deltas > 0)),
                "dataset_tie_rate": float(np.mean(dataset_deltas == 0)),
                "dataset_strategy_win_rate": float(np.mean(pair_deltas > 0)),
                "dataset_strategy_tie_rate": float(np.mean(pair_deltas == 0)),
                "pgsui_win_rate": float(np.mean(pair_deltas > 0)),
                "tie_rate": float(np.mean(pair_deltas == 0)),
                "mean_delta_macro_f1": mean,
                "ci95_low": ci95_low,
                "ci95_high": ci95_high,
                "ci95_min": ci95_low,
                "ci95_max": ci95_high,
                "median_delta_macro_f1": float(np.median(dataset_deltas)),
                "p_exact_block_sign_flip": block_p,
            }
        )

    table = pd.DataFrame(rows)

    if table.empty:
        return table

    table["p_holm"] = holm_adjust_p_values(table["p_exact_block_sign_flip"])
    table["reject_holm_0_05"] = table["p_holm"].le(0.05)
    table["primary_test_family"] = "macro_f1_pgsui_vs_gtimputation"
    table["p_adjust_method"] = "holm"

    pgsui_rank = {PGSUI_MODEL_LABELS[m]: i for i, m in enumerate(PGSUI_MODELS)}
    gti_rank = {GTI_MODEL_LABELS[m]: i for i, m in enumerate(GTI_METHODS)}
    table["_p"] = table["pgsui_model"].map(pgsui_rank).fillna(99)
    table["_g"] = table["gti_method"].map(gti_rank).fillna(99)

    return (
        table.sort_values(["_p", "_g"])
        .drop(columns=["_p", "_g"])
        .reset_index(drop=True)
    )


def build_strategy_sensitivity_table(combined: pd.DataFrame) -> pd.DataFrame:
    """Fit secondary mixed models for strategy dependence of PG-SUI advantage."""
    merged = build_pgsui_gtimputation_delta_table(combined)

    if merged.empty:
        return pd.DataFrame()

    try:
        import statsmodels.formula.api as smf
        from scipy.stats import chi2
    except Exception:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for (pgsui_model, gti_method), group in merged.groupby(
        ["pgsui_model", "gti_method"], observed=True
    ):
        analysis = group[["dataset_id", "strategy", "delta_macro_f1"]].dropna().copy()
        n_datasets = int(analysis["dataset_id"].nunique())
        n_pairs = int(analysis.shape[0])
        n_strategies = int(analysis["strategy"].nunique())

        base_row: dict[str, object] = {
            "pgsui_model": pgsui_model,
            "gti_method": gti_method,
            "n_datasets": n_datasets,
            "n_dataset_strategy_pairs": n_pairs,
            "n_strategies": n_strategies,
            "formula": "delta_macro_f1 ~ C(strategy, Sum) + (1 | dataset_id)",
            "strategy_effect_test": "joint_wald_sum_coding",
        }

        if n_datasets < 2 or n_strategies < 2:
            rows.append(
                {
                    **base_row,
                    "fit_status": "insufficient_dataset_or_strategy_levels",
                    "converged": False,
                    "strategy_wald_chi2": math.nan,
                    "strategy_wald_df": math.nan,
                    "strategy_wald_p": math.nan,
                    "strategy_coefficients": "",
                    "random_intercept_variance": math.nan,
                    "residual_variance": math.nan,
                    "log_likelihood": math.nan,
                    "aic": math.nan,
                }
            )
            continue

        raw_strategies = set(analysis["strategy"].astype(str))
        present_strategies = [
            strategy for strategy in STRATEGY_ORDER if strategy in raw_strategies
        ]
        present_strategies.extend(sorted(raw_strategies.difference(STRATEGY_ORDER)))
        analysis["strategy"] = pd.Categorical(
            analysis["strategy"].astype(str),
            categories=present_strategies,
            ordered=True,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(
                    "delta_macro_f1 ~ C(strategy, Sum)",
                    analysis,
                    groups=analysis["dataset_id"].astype(str),
                ).fit(reml=False, method="lbfgs", disp=False)
        except Exception as exc:
            rows.append(
                {
                    **base_row,
                    "fit_status": f"fit_failed: {type(exc).__name__}",
                    "converged": False,
                    "strategy_wald_chi2": math.nan,
                    "strategy_wald_df": math.nan,
                    "strategy_wald_p": math.nan,
                    "strategy_coefficients": "",
                    "random_intercept_variance": math.nan,
                    "residual_variance": math.nan,
                    "log_likelihood": math.nan,
                    "aic": math.nan,
                }
            )
            continue

        strategy_terms = [
            term
            for term in fit.fe_params.index
            if str(term).startswith("C(strategy, Sum)")
        ]

        if strategy_terms:
            coefs = fit.fe_params.loc[strategy_terms].to_numpy(dtype=float)
            cov = (
                fit.cov_params()
                .loc[strategy_terms, strategy_terms]
                .to_numpy(dtype=float)
            )
            wald_stat = float(coefs.T @ np.linalg.pinv(cov) @ coefs)
            wald_df = int(len(strategy_terms))
            wald_p = float(chi2.sf(wald_stat, wald_df))
        else:
            wald_stat = math.nan
            wald_df = math.nan
            wald_p = math.nan

        if getattr(fit, "cov_re", pd.DataFrame()).empty:
            random_intercept_variance = math.nan
        else:
            random_intercept_variance = float(fit.cov_re.iloc[0, 0])

        rows.append(
            {
                **base_row,
                "fit_status": "ok",
                "converged": bool(getattr(fit, "converged", False)),
                "strategy_wald_chi2": wald_stat,
                "strategy_wald_df": wald_df,
                "strategy_wald_p": wald_p,
                "strategy_coefficients": "; ".join(
                    f"{term}={float(fit.fe_params.loc[term]):.6g}"
                    for term in strategy_terms
                ),
                "random_intercept_variance": random_intercept_variance,
                "residual_variance": float(fit.scale),
                "log_likelihood": float(fit.llf),
                "aic": float(fit.aic),
            }
        )

    table = pd.DataFrame(rows)

    if table.empty:
        return table

    pgsui_rank = {PGSUI_MODEL_LABELS[m]: i for i, m in enumerate(PGSUI_MODELS)}
    gti_rank = {GTI_MODEL_LABELS[m]: i for i, m in enumerate(GTI_METHODS)}
    table["_p"] = table["pgsui_model"].map(pgsui_rank).fillna(99)
    table["_g"] = table["gti_method"].map(gti_rank).fillna(99)

    return (
        table.sort_values(["_p", "_g"])
        .drop(columns=["_p", "_g"])
        .reset_index(drop=True)
    )


def build_dataset_leaderboard_table(combined: pd.DataFrame) -> pd.DataFrame:
    """Rank datasets by the best PG-SUI advantage over GTImputation.

    Each row reports the overall winning model for a dataset (averaged over strategies) alongside the best PG-SUI and best GTImputation model.
    """
    required = {"dataset_id", "software", "model_label", "macro_f1"}

    if combined.empty or not required.issubset(combined.columns):
        return pd.DataFrame()

    df = _coerce_numeric(combined, ["macro_f1", "mcc"]).dropna(subset=["macro_f1"])

    per = (
        df.groupby(["dataset_id", "software", "model_label"], observed=True)
        .agg(macro_f1=("macro_f1", "mean"), mcc=("mcc", "mean"))
        .reset_index()
    )

    rows: list[dict[str, object]] = []

    for dataset_id, group in per.groupby("dataset_id", observed=True):
        overall = group.loc[group["macro_f1"].idxmax()]
        pgsui = group.loc[group["software"].eq("PG-SUI")]
        gti = group.loc[group["software"].eq("GTImputation")]
        if pgsui.empty or gti.empty:
            continue
        best_pgsui = pgsui.loc[pgsui["macro_f1"].idxmax()]
        best_gti = gti.loc[gti["macro_f1"].idxmax()]
        rows.append(
            {
                "dataset_id": dataset_id,
                "overall_best_model": str(overall["model_label"]),
                "overall_best_macro_f1": float(overall["macro_f1"]),  # type: ignore[no-untyped-call]
                "overall_best_mcc": float(overall["mcc"]),  # type: ignore[no-untyped-call]
                "best_pgsui_model": str(best_pgsui["model_label"]),
                "best_pgsui_macro_f1": float(best_pgsui["macro_f1"]),  # type: ignore[no-untyped-call]
                "best_gti_model": str(best_gti["model_label"]),
                "best_gti_macro_f1": float(best_gti["macro_f1"]),  # type: ignore[no-untyped-call]
                "pgsui_minus_gti": float(best_pgsui["macro_f1"] - best_gti["macro_f1"]),  # type: ignore[no-untyped-call]
            }
        )

    table = pd.DataFrame(rows)

    if table.empty:
        return table

    return table.sort_values("pgsui_minus_gti", ascending=False).reset_index(drop=True)


def build_class_f1_wide_table(combined: pd.DataFrame) -> pd.DataFrame:
    """Per-model wide summary of class F1, macro F1, MCC, and class balance."""
    class_cols = ["ref_f1", "het_f1", "alt_f1"]
    required = {"model_label", "macro_f1", *class_cols}

    if combined.empty or not required.issubset(combined.columns):
        return pd.DataFrame()

    df = _coerce_numeric(
        with_model_category(combined), ["macro_f1", "mcc", *class_cols]
    )

    df = df.dropna(subset=["macro_f1", *class_cols])

    if df.empty:
        return pd.DataFrame()

    agg_spec: dict[str, tuple[str, str]] = {
        "n_runs": ("macro_f1", "count"),
        "mean_macro_f1": ("macro_f1", "mean"),
        "mean_ref_f1": ("ref_f1", "mean"),
        "mean_het_f1": ("het_f1", "mean"),
        "mean_alt_f1": ("alt_f1", "mean"),
    }
    if "mcc" in df.columns:
        agg_spec["mean_mcc"] = ("mcc", "mean")

    grouped = (
        df.groupby(["model_category", "model_label"], observed=True)
        .agg(**agg_spec)
        .reset_index()
    )

    grouped["worst_class_f1"] = grouped[
        ["mean_ref_f1", "mean_het_f1", "mean_alt_f1"]
    ].min(axis=1)

    grouped["class_f1_spread"] = (
        grouped[["mean_ref_f1", "mean_het_f1", "mean_alt_f1"]].max(axis=1)
        - grouped["worst_class_f1"]
    )

    model_rank = {model: index for index, model in enumerate(MODEL_ORDER)}
    grouped["_rank"] = grouped["model_label"].astype(str).map(model_rank).fillna(99)

    return grouped.sort_values("_rank").drop(columns=["_rank"]).reset_index(drop=True)


def dataset_size_performance_summary_df(combined: pd.DataFrame) -> pd.DataFrame:
    """Summarize model-family performance by dataset sample and locus counts."""
    required = {"dataset_id", "model_label", "macro_f1", "n_samples", "n_loci"}

    if combined.empty or not required.issubset(combined.columns):
        return pd.DataFrame()

    numeric = ["macro_f1", "mcc", "n_samples", "n_loci", "n_cells"]
    df = _coerce_numeric(with_model_category(combined), numeric)
    df = df.dropna(
        subset=["dataset_id", "model_category", "macro_f1", "n_samples", "n_loci"]
    )

    if df.empty:
        return pd.DataFrame()

    if "n_cells" not in df.columns:
        df["n_cells"] = df["n_samples"] * df["n_loci"]

    df["n_cells"] = df["n_cells"].fillna(df["n_samples"] * df["n_loci"])

    agg_spec: dict[str, tuple[str, str]] = {
        "n_runs": ("macro_f1", "count"),
        "mean_macro_f1": ("macro_f1", "mean"),
        "sd_macro_f1": ("macro_f1", "std"),
    }

    if "mcc" in df.columns:
        agg_spec["mean_mcc"] = ("mcc", "mean")

    summary = (
        df.groupby(
            ["dataset_id", "model_category", "n_samples", "n_loci", "n_cells"],
            observed=True,
        )
        .agg(**agg_spec)
        .reset_index()
    )

    summary["model_category"] = pd.Categorical(
        summary["model_category"], categories=CATEGORY_ORDER, ordered=True
    )

    return summary.sort_values(["model_category", "n_samples", "n_loci"]).reset_index(
        drop=True
    )


def plot_performance_by_dataset_size(combined: pd.DataFrame, out_dir: Path) -> None:
    """Plot model-family macro-F1 over dataset sample-count and locus-count space."""
    summary = dataset_size_performance_summary_df(combined)

    if summary.empty:
        print(
            "Skipping dataset-size performance plot: missing n_samples/n_loci metadata."
        )
        return

    categories = [
        category
        for category in CATEGORY_ORDER
        if category in set(summary["model_category"].dropna().astype(str))
    ]

    if not categories:
        return

    x_values = pd.to_numeric(summary["n_samples"], errors="coerce")
    y_values = pd.to_numeric(summary["n_loci"], errors="coerce")
    x_min = max(1.0, float(x_values.min()) * 0.75)
    x_max = float(x_values.max()) * 1.75
    y_min = max(1.0, float(y_values.min()) * 0.75)
    y_max = float(y_values.max()) * 1.35
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap = mpl.colormaps["viridis"]

    apply_house_style()
    ncols = 2
    nrows = int(math.ceil(len(categories) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(13.6, 5.7 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    axes_flat = axes.ravel()
    scatter = None

    dataset_label_offsets = {
        "mr-HY": (8, 8),
        "hr-HY20": (8, -14),
        "hr-QI96": (-42, 8),
        "hr-QS07": (-42, -17),
        "mr-QI": (-42, -4),
        "mr-QS": (8, -4),
        "mr-QS98": (8, -13),
        "mr-Mice98": (-64, -13),
        "2SP-QSQI": (16, 0),
        "mr-Mice1940": (-86, 0),
    }

    for axis_index, category in enumerate(categories):
        ax = axes_flat[axis_index]
        category_df = summary.loc[summary["model_category"].astype(str).eq(category)]
        category_df = category_df.sort_values(["n_samples", "n_loci", "dataset_id"])
        ax.set_facecolor(lighten_color(CATEGORY_COLORS.get(category, "#999999"), 0.9))
        scatter = ax.scatter(
            category_df["n_samples"],
            category_df["n_loci"],
            c=category_df["mean_macro_f1"],
            cmap=cmap,
            norm=norm,
            s=230,
            edgecolors="black",
            linewidths=1.0,
            alpha=0.92,
            zorder=4,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
        ax.grid(True, which="minor", linewidth=0.4, alpha=0.16)
        ax.set_title(
            CATEGORY_SHORT_LABELS.get(category, category),
            fontsize="xx-large",
            fontweight="bold",
            color=CATEGORY_COLORS.get(category, "#333333"),
        )

        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
            spine.set_edgecolor(CATEGORY_COLORS.get(category, "#333333"))

        for _, row in category_df.iterrows():
            offset = dataset_label_offsets.get(str(row["dataset_id"]), (8, 8))
            ax.annotate(
                str(row["dataset_id"]),
                xy=(row["n_samples"], row["n_loci"]),
                xytext=offset,
                textcoords="offset points",
                fontsize="large",
                color="#1f1f24",
                path_effects=[
                    path_effects.withStroke(
                        linewidth=3.2, foreground="white", alpha=0.92
                    )
                ],
                annotation_clip=False,
                zorder=6,
            )

    for ax in axes_flat[len(categories) :]:
        ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Samples per dataset (log10-scaled)", fontsize="xx-large")

    for ax in axes[:, 0]:
        ax.set_ylabel("Loci per dataset (log10-scaled)", fontsize="xx-large")

    for ax in axes_flat[: len(categories)]:
        ax.tick_params(axis="both", labelsize="xx-large")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    if scatter is not None:
        colorbar_axis = fig.add_axes((0.91, 0.18, 0.025, 0.64))
        colorbar = fig.colorbar(scatter, cax=colorbar_axis)
        colorbar.set_label("Mean Macro F1-score", fontsize="xx-large")
        colorbar.ax.tick_params(labelsize="xx-large")

    fig.text(
        0.5,
        0.01,
        "Each point is one dataset; values are averaged over retained simulation strategies "
        "and over models within each family.",
        ha="center",
        va="bottom",
        fontsize="xx-large",
        color="#444444",
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.86,
        bottom=0.10,
        top=0.90,
        wspace=0.14,
        hspace=0.24,
    )

    save_figure(fig, out_dir / "performance_by_dataset_size")


def f1_size_regression_summary_df(combined: pd.DataFrame) -> pd.DataFrame:
    """Fit simple linear F1 regressions against sample and locus counts."""
    summary = dataset_size_performance_summary_df(combined)

    if summary.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    predictors = {
        "n_samples": "Sample count",
        "n_loci": "Locus count",
    }

    for category, category_df in summary.groupby("model_category", observed=True):
        category_text = str(category)
        group_label = CATEGORY_SHORT_LABELS.get(category_text, category_text)

        for predictor, predictor_label in predictors.items():
            fit_df = _coerce_numeric(category_df, [predictor, "mean_macro_f1"]).dropna(
                subset=[predictor, "mean_macro_f1"]
            )

            n_datasets = int(fit_df["dataset_id"].nunique())

            row: dict[str, object] = {
                "predictor": predictor,
                "predictor_label": predictor_label,
                "model_category": category_text,
                "comparison_group": group_label,
                "n_datasets": n_datasets,
                "x_min": float(fit_df[predictor].min()) if not fit_df.empty else np.nan,
                "x_max": float(fit_df[predictor].max()) if not fit_df.empty else np.nan,
                "mean_macro_f1_min": (
                    float(fit_df["mean_macro_f1"].min()) if not fit_df.empty else np.nan
                ),
                "mean_macro_f1_max": (
                    float(fit_df["mean_macro_f1"].max()) if not fit_df.empty else np.nan
                ),
            }

            if n_datasets < 2 or fit_df[predictor].nunique() < 2:
                row.update(
                    {
                        "slope": np.nan,
                        "intercept": np.nan,
                        "r_squared": np.nan,
                        "pearson_r": np.nan,
                    }
                )

            else:
                x_values = fit_df[predictor].to_numpy(dtype=float)
                y_values = fit_df["mean_macro_f1"].to_numpy(dtype=float)
                slope, intercept = np.polyfit(x_values, y_values, deg=1)
                y_hat = slope * x_values + intercept
                residual_ss = float(np.sum((y_values - y_hat) ** 2))
                total_ss = float(np.sum((y_values - np.mean(y_values)) ** 2))
                pearson_r = float(np.corrcoef(x_values, y_values)[0, 1])

                row.update(
                    {
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "r_squared": (
                            1.0 - residual_ss / total_ss if total_ss > 0 else np.nan
                        ),
                        "pearson_r": pearson_r,
                    }
                )

            rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    predictor_rank = {"n_samples": 0, "n_loci": 1}
    category_rank = {category: index for index, category in enumerate(CATEGORY_ORDER)}
    out["_predictor_rank"] = out["predictor"].map(predictor_rank).fillna(99)
    out["_category_rank"] = out["model_category"].map(category_rank).fillna(99)

    return (
        out.sort_values(["_predictor_rank", "_category_rank"], kind="mergesort")
        .drop(columns=["_predictor_rank", "_category_rank"])
        .reset_index(drop=True)
    )


def plot_f1_size_regressions(combined: pd.DataFrame, out_dir: Path) -> None:
    """Plot linear F1 regressions against sample count and locus count."""
    summary = dataset_size_performance_summary_df(combined)
    regression = f1_size_regression_summary_df(combined)

    if summary.empty or regression.empty:
        print("Skipping F1 size regression plot: missing size/performance summary.")
        return

    categories = [
        category
        for category in CATEGORY_ORDER
        if category in set(summary["model_category"].dropna().astype(str))
    ]

    if not categories:
        return

    predictors = [
        ("n_samples", "Sample Count", "Samples per Dataset"),
        ("n_loci", "Locus Count", "Loci per Dataset"),
    ]

    apply_house_style()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15.2, 7.1), sharey=True)
    handles: list[mlines.Line2D] = []

    for ax, (predictor, predictor_label, xlabel) in zip(axes, predictors):
        predictor_regression = regression.loc[regression["predictor"].eq(predictor)]

        for category in categories:
            category_df = summary.loc[
                summary["model_category"].astype(str).eq(category)
            ].copy()

            category_df = _coerce_numeric(category_df, [predictor, "mean_macro_f1"])

            category_df = category_df.dropna(subset=[predictor, "mean_macro_f1"])

            if category_df.empty:
                continue

            color = CATEGORY_COLORS.get(category, "#777777")
            short_label = CATEGORY_SHORT_LABELS.get(category, category)

            ax.scatter(
                category_df[predictor],
                category_df["mean_macro_f1"],
                s=110,
                color=color,
                edgecolor="black",
                linewidth=0.9,
                alpha=0.86,
                zorder=4,
            )

            reg_rows = predictor_regression.loc[
                predictor_regression["model_category"].eq(category)
            ]

            if not reg_rows.empty:
                reg_row = reg_rows.iloc[0]

                if pd.notna(reg_row["slope"]) and pd.notna(reg_row["intercept"]):
                    x_line = np.linspace(
                        float(category_df[predictor].min()),
                        float(category_df[predictor].max()),
                        100,
                    )

                    y_line = np.clip(
                        float(reg_row["slope"]) * x_line + float(reg_row["intercept"]),
                        0.0,
                        1.0,
                    )

                    r_squared = reg_row["r_squared"]

                    line_label = (
                        f"{short_label} (R²={float(r_squared):.2f})"
                        if pd.notna(r_squared)
                        else short_label
                    )

                    line = ax.plot(
                        x_line,
                        y_line,
                        color=color,
                        linewidth=2.8,
                        label=line_label,
                        zorder=3,
                    )[0]

                    if predictor == "n_samples":
                        handles.append(line)

        ax.set_xlabel(xlabel, fontsize="xx-large")
        ax.set_ylim(0.0, 1.02)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.tick_params(axis="both", labelsize="x-large")

    axes[0].set_ylabel("Mean Macro F1-score", fontsize="xx-large")
    format_decimal_ticks(axes[0], axis="y", decimals=2)

    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            fontsize="xx-large",
            title="Linear regression by model family",
            title_fontsize="xx-large",
            fancybox=True,
            shadow=True,
        )

    fig.text(
        0.5,
        0.01,
        "Each point is one dataset-family mean across retained simulation strategies; regression fits use untransformed sample and locus counts.",
        ha="center",
        va="bottom",
        fontsize="xx-large",
        color="#444444",
    )

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.82))
    save_figure(fig, out_dir / "f1_size_regression")


def f1_genotype_count_regression_summary_df(combined: pd.DataFrame) -> pd.DataFrame:
    """Fit per-model F1 regressions against total genotype count."""
    required = {
        "dataset_id",
        "software",
        "model_label",
        "macro_f1",
        "n_samples",
        "n_loci",
    }

    if combined.empty or not required.issubset(combined.columns):
        return pd.DataFrame()

    df = _coerce_numeric(
        combined, ["macro_f1", "n_samples", "n_loci", "n_cells"]
    ).dropna(
        subset=[
            "dataset_id",
            "software",
            "model_label",
            "macro_f1",
            "n_samples",
            "n_loci",
        ]
    )

    if df.empty:
        return pd.DataFrame()

    if "model" not in df.columns:
        df["model"] = df["model_label"].astype(str)

    if "n_cells" not in df.columns:
        df["n_cells"] = df["n_samples"] * df["n_loci"]

    df["n_genotypes"] = df["n_cells"].fillna(df["n_samples"] * df["n_loci"])
    df = df.dropna(subset=["n_genotypes"]).loc[lambda frame: frame["n_genotypes"].gt(0)]

    if df.empty:
        return pd.DataFrame()

    per_dataset_model = (
        df.groupby(
            [
                "dataset_id",
                "software",
                "model",
                "model_label",
                "n_samples",
                "n_loci",
                "n_genotypes",
            ],
            observed=True,
        )["macro_f1"]
        .agg(n_runs="count", mean_macro_f1="mean", sd_macro_f1="std")
        .reset_index()
    )

    rows: list[dict[str, object]] = []

    for (software, model, model_label), group in per_dataset_model.groupby(
        ["software", "model", "model_label"], observed=True
    ):
        fit_df = group.dropna(subset=["n_genotypes", "mean_macro_f1"])
        n_datasets = int(fit_df["dataset_id"].nunique())
        n_runs = int(fit_df["n_runs"].sum())

        row: dict[str, object] = {
            "software": software,
            "model": model,
            "model_label": str(model_label),
            "n_datasets": n_datasets,
            "n_runs": n_runs,
            "n_genotypes_min": (
                float(fit_df["n_genotypes"].min()) if not fit_df.empty else np.nan
            ),
            "n_genotypes_max": (
                float(fit_df["n_genotypes"].max()) if not fit_df.empty else np.nan
            ),
            "mean_macro_f1_min": (
                float(fit_df["mean_macro_f1"].min()) if not fit_df.empty else np.nan
            ),
            "mean_macro_f1_max": (
                float(fit_df["mean_macro_f1"].max()) if not fit_df.empty else np.nan
            ),
        }

        if n_datasets < 2 or fit_df["n_genotypes"].nunique() < 2:
            row.update(
                {
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r_squared": np.nan,
                    "pearson_r": np.nan,
                }
            )
        else:
            x_values = fit_df["n_genotypes"].to_numpy(dtype=float)
            y_values = fit_df["mean_macro_f1"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x_values, y_values, deg=1)
            y_hat = slope * x_values + intercept
            residual_ss = float(np.sum((y_values - y_hat) ** 2))
            total_ss = float(np.sum((y_values - np.mean(y_values)) ** 2))

            row.update(
                {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": (
                        1.0 - residual_ss / total_ss if total_ss > 0 else np.nan
                    ),
                    "pearson_r": float(np.corrcoef(x_values, y_values)[0, 1]),
                }
            )

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    model_rank = {model: index for index, model in enumerate(MODEL_ORDER)}

    out["_model_rank"] = out["model_label"].astype(str).map(model_rank).fillna(99)

    return (
        out.sort_values("_model_rank")
        .drop(columns="_model_rank")
        .reset_index(drop=True)
    )


def plot_f1_genotype_count_regressions(combined: pd.DataFrame, out_dir: Path) -> None:
    """Plot per-model F1 regressions against total genotype count."""
    required = {
        "dataset_id",
        "software",
        "model_label",
        "macro_f1",
        "n_samples",
        "n_loci",
    }

    if combined.empty or not required.issubset(combined.columns):
        print("Skipping genotype-count regression plot: missing size/F1 columns.")
        return

    df = _coerce_numeric(
        combined, ["macro_f1", "n_samples", "n_loci", "n_cells"]
    ).dropna(
        subset=[
            "dataset_id",
            "software",
            "model_label",
            "macro_f1",
            "n_samples",
            "n_loci",
        ]
    )

    if df.empty:
        return

    if "model" not in df.columns:
        df["model"] = df["model_label"].astype(str)

    if "n_cells" not in df.columns:
        df["n_cells"] = df["n_samples"] * df["n_loci"]

    df["n_genotypes"] = df["n_cells"].fillna(df["n_samples"] * df["n_loci"])
    df = df.dropna(subset=["n_genotypes"]).loc[lambda frame: frame["n_genotypes"].gt(0)]

    if df.empty:
        return

    plot_df = (
        df.groupby(
            [
                "dataset_id",
                "software",
                "model",
                "model_label",
                "n_genotypes",
            ],
            observed=True,
        )["macro_f1"]
        .agg(mean_macro_f1="mean")
        .reset_index()
    )

    regression = f1_genotype_count_regression_summary_df(combined)

    if plot_df.empty or regression.empty:
        return

    software_order = [
        software
        for software in ("GTImputation", "PG-SUI")
        if software in set(plot_df["software"].astype(str))
    ]

    if not software_order:
        return

    apply_house_style()

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(software_order),
        figsize=(7.8 * len(software_order), 6.8),
        sharey=True,
        squeeze=False,
    )

    axes_flat = axes.ravel()
    legend_handles: list[mlines.Line2D] = []
    legend_labels: list[str] = []

    for ax, software in zip(axes_flat, software_order):
        software_df = plot_df.loc[plot_df["software"].eq(software)].copy()
        model_order = present_model_order(software_df)
        for model_label in model_order:
            model_df = software_df.loc[
                software_df["model_label"].astype(str).eq(model_label)
            ].copy()

            if model_df.empty:
                continue

            color = MODEL_PALETTE.get(model_label, "#777777")

            ax.scatter(
                model_df["n_genotypes"],
                model_df["mean_macro_f1"],
                s=95,
                color=color,
                edgecolor="black",
                linewidth=0.85,
                alpha=0.82,
                zorder=4,
            )

            fit_row = regression.loc[
                regression["model_label"].astype(str).eq(model_label)
            ]

            if fit_row.empty:
                continue

            fit = fit_row.iloc[0]

            if pd.isna(fit["slope"]) or pd.isna(fit["intercept"]):
                continue

            x_line = np.linspace(
                float(model_df["n_genotypes"].min()),
                float(model_df["n_genotypes"].max()),
                100,
            )

            y_line = np.clip(
                float(fit["slope"]) * x_line + float(fit["intercept"]),
                0.0,
                1.0,
            )

            r_squared = fit["r_squared"]

            label = (
                f"{model_label} (R²={float(r_squared):.2f})"
                if pd.notna(r_squared)
                else model_label
            )

            (line,) = ax.plot(
                x_line,
                y_line,
                color=color,
                linewidth=2.4,
                label=label,
                zorder=3,
            )

            legend_handles.append(line)
            legend_labels.append(label)

        ax.set_title(software, fontsize="xx-large", fontweight="bold")

        ax.set_xlabel("Number of Genotypes (samples × loci)", fontsize="xx-large")

        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.tick_params(axis="both", labelsize="x-large")
        ax.grid(True, axis="both", alpha=0.25, linewidth=0.8)

    axes_flat[0].set_ylabel("Mean Macro F1-score", fontsize="xx-large")
    axes_flat[0].set_ylim(0.0, 1.02)
    format_decimal_ticks(axes_flat[0], axis="y", decimals=2)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(4, len(legend_handles)),
            fontsize="xx-large",
            title="Model Regression",
            title_fontsize="xx-large",
            fancybox=True,
            shadow=True,
        )

    fig.text(
        0.5,
        0.01,
        "Each point is a dataset-model mean across retained simulation strategies; "
        "genotype count is sample count multiplied by locus count.",
        ha="center",
        va="bottom",
        fontsize="xx-large",
        color="#444444",
    )

    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.78))
    save_figure(fig, out_dir / "f1_genotype_count_regression")


def ensure_runtime_display_labels(runtime_df: pd.DataFrame) -> pd.DataFrame:
    """Attach runtime backend/model display labels when older tables lack them."""
    df = runtime_df.copy()

    if "runtime_backend" not in df.columns:
        df["runtime_backend"] = "not_applicable"

    if "runtime_backend_label" not in df.columns:
        df["runtime_backend_label"] = df["runtime_backend"].map(backend_display_label)

        if "software" in df.columns:
            df.loc[
                df["software"].astype(str).eq("GTImputation"),
                "runtime_backend_label",
            ] = "GTImputation"

    if "runtime_model_label" not in df.columns:
        deep_mask = df["software"].astype(str).eq("PG-SUI") & df[
            "runtime_backend"
        ].astype(str).isin(RUNTIME_PGSUI_BACKENDS)

        df["runtime_model_label"] = df["model_label"].astype(str)

        df.loc[deep_mask, "runtime_model_label"] = (
            df.loc[deep_mask, "model_label"].astype(str)
            + " ("
            + df.loc[deep_mask, "runtime_backend"].map(backend_display_label)
            + ")"
        )

    return df


def runtime_genotype_count_regression_summary_df(
    runtime_combined: pd.DataFrame,
) -> pd.DataFrame:
    """Fit per-runtime-model execution-time regressions against genotype count."""
    required = {
        "dataset_id",
        "software",
        "model_label",
        "runtime_seconds",
        "n_samples",
        "n_loci",
    }

    if runtime_combined.empty or not required.issubset(runtime_combined.columns):
        return pd.DataFrame()

    df = ensure_runtime_display_labels(runtime_combined)
    df = _coerce_numeric(
        df, ["runtime_seconds", "n_samples", "n_loci", "n_cells"]
    ).dropna(
        subset=[
            "dataset_id",
            "software",
            "model_label",
            "runtime_seconds",
            "n_samples",
            "n_loci",
        ]
    )

    if df.empty:
        return pd.DataFrame()

    if "model" not in df.columns:
        df["model"] = df["model_label"].astype(str)

    if "n_cells" not in df.columns:
        df["n_cells"] = df["n_samples"] * df["n_loci"]

    df["n_genotypes"] = df["n_cells"].fillna(df["n_samples"] * df["n_loci"])

    df = df.dropna(subset=["n_genotypes"]).loc[
        lambda frame: frame["n_genotypes"].gt(0) & frame["runtime_seconds"].gt(0)
    ]

    if df.empty:
        return pd.DataFrame()

    per_dataset_runtime = (
        df.groupby(
            [
                "dataset_id",
                "software",
                "model",
                "model_label",
                "runtime_backend",
                "runtime_backend_label",
                "runtime_model_label",
                "n_samples",
                "n_loci",
                "n_genotypes",
            ],
            observed=True,
        )["runtime_seconds"]
        .agg(n_runs="count", mean_runtime_seconds="mean")
        .reset_index()
    )

    rows: list[dict[str, object]] = []

    group_cols = [
        "software",
        "model",
        "model_label",
        "runtime_backend",
        "runtime_backend_label",
        "runtime_model_label",
    ]

    for group_key, group in per_dataset_runtime.groupby(group_cols, observed=True):
        (
            software,
            model,
            model_label,
            runtime_backend,
            runtime_backend_label,
            runtime_model_label,
        ) = group_key

        fit_df = group.dropna(subset=["n_genotypes", "mean_runtime_seconds"])
        n_datasets = int(fit_df["dataset_id"].nunique())
        n_runs = int(fit_df["n_runs"].sum())

        row: dict[str, object] = {
            "software": software,
            "model": model,
            "model_label": str(model_label),
            "runtime_backend": runtime_backend,
            "runtime_backend_label": str(runtime_backend_label),
            "runtime_model_label": str(runtime_model_label),
            "n_datasets": n_datasets,
            "n_runs": n_runs,
            "n_genotypes_min": (
                float(fit_df["n_genotypes"].min()) if not fit_df.empty else np.nan
            ),
            "n_genotypes_max": (
                float(fit_df["n_genotypes"].max()) if not fit_df.empty else np.nan
            ),
            "mean_runtime_seconds_min": (
                float(fit_df["mean_runtime_seconds"].min())
                if not fit_df.empty
                else np.nan
            ),
            "mean_runtime_seconds_max": (
                float(fit_df["mean_runtime_seconds"].max())
                if not fit_df.empty
                else np.nan
            ),
        }

        if n_datasets < 2 or fit_df["n_genotypes"].nunique() < 2:
            row.update(
                {
                    "slope_seconds_per_genotype": np.nan,
                    "intercept_seconds": np.nan,
                    "r_squared": np.nan,
                    "pearson_r": np.nan,
                }
            )

        else:
            x_values = fit_df["n_genotypes"].to_numpy(dtype=float)
            y_values = fit_df["mean_runtime_seconds"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x_values, y_values, deg=1)
            y_hat = slope * x_values + intercept
            residual_ss = float(np.sum((y_values - y_hat) ** 2))
            total_ss = float(np.sum((y_values - np.mean(y_values)) ** 2))
            row.update(
                {
                    "slope_seconds_per_genotype": float(slope),
                    "intercept_seconds": float(intercept),
                    "r_squared": (
                        1.0 - residual_ss / total_ss if total_ss > 0 else np.nan
                    ),
                    "pearson_r": float(np.corrcoef(x_values, y_values)[0, 1]),
                }
            )

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    label_rank = {label: index for index, label in enumerate(runtime_model_order(out))}

    backend_rank = {
        "GTImputation": 0,
        "Deterministic": 1,
        "CPU": 2,
        "GPU": 3,
    }

    out["_runtime_model_rank"] = (
        out["runtime_model_label"].astype(str).map(label_rank).fillna(99)
    )

    out["_backend_rank"] = (
        out["runtime_backend_label"].astype(str).map(backend_rank).fillna(99)
    )

    return (
        out.sort_values(["_backend_rank", "_runtime_model_rank"])
        .drop(columns=["_backend_rank", "_runtime_model_rank"])
        .reset_index(drop=True)
    )


def plot_runtime_genotype_count_regressions(
    runtime_combined: pd.DataFrame, out_dir: Path
) -> None:
    """Plot execution-time regressions against total genotype count."""
    required = {
        "dataset_id",
        "software",
        "model_label",
        "runtime_seconds",
        "n_samples",
        "n_loci",
    }

    if runtime_combined.empty or not required.issubset(runtime_combined.columns):
        print("Skipping runtime genotype-count regression plot: missing columns.")
        return

    df = ensure_runtime_display_labels(runtime_combined)

    df = _coerce_numeric(
        df, ["runtime_seconds", "n_samples", "n_loci", "n_cells"]
    ).dropna(
        subset=[
            "dataset_id",
            "software",
            "model_label",
            "runtime_seconds",
            "n_samples",
            "n_loci",
        ]
    )

    if df.empty:
        return

    if "model" not in df.columns:
        df["model"] = df["model_label"].astype(str)

    if "n_cells" not in df.columns:
        df["n_cells"] = df["n_samples"] * df["n_loci"]

    df["n_genotypes"] = df["n_cells"].fillna(df["n_samples"] * df["n_loci"])

    df = df.dropna(subset=["n_genotypes"]).loc[
        lambda frame: frame["n_genotypes"].gt(0) & frame["runtime_seconds"].gt(0)
    ]

    if df.empty:
        return

    plot_df = (
        df.groupby(
            [
                "dataset_id",
                "software",
                "model",
                "model_label",
                "runtime_backend",
                "runtime_backend_label",
                "runtime_model_label",
                "n_genotypes",
            ],
            observed=True,
        )["runtime_seconds"]
        .agg(mean_runtime_seconds="mean")
        .reset_index()
    )

    regression = runtime_genotype_count_regression_summary_df(runtime_combined)

    if plot_df.empty or regression.empty:
        return

    panel_labels = runtime_panel_backend_labels(plot_df)

    if not panel_labels:
        return

    apply_house_style()

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(panel_labels),
        figsize=(7.8 * len(panel_labels), 7.2),
        sharey=False,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    legend_by_label: dict[str, mlines.Line2D] = {}

    for ax, panel_label in zip(axes_flat, panel_labels):
        panel_df = runtime_panel_subset(plot_df, panel_label)

        if panel_df.empty:
            continue

        model_order = runtime_model_order(panel_df)
        palette = runtime_model_palette(model_order)

        for runtime_model_label in model_order:
            model_df = panel_df.loc[
                panel_df["runtime_model_label"].astype(str).eq(runtime_model_label)
            ].copy()

            if model_df.empty:
                continue

            color = palette.get(runtime_model_label, "#777777")

            ax.scatter(
                model_df["n_genotypes"],
                model_df["mean_runtime_seconds"],
                s=95,
                color=color,
                edgecolor="black",
                linewidth=0.85,
                alpha=0.82,
                zorder=4,
            )

            fit_row = regression.loc[
                regression["runtime_model_label"].astype(str).eq(runtime_model_label)
            ]

            if fit_row.empty:
                continue

            fit = fit_row.iloc[0]

            if pd.isna(fit["slope_seconds_per_genotype"]) or pd.isna(
                fit["intercept_seconds"]
            ):
                continue

            x_line = np.linspace(
                float(model_df["n_genotypes"].min()),
                float(model_df["n_genotypes"].max()),
                100,
            )

            y_line = np.maximum(
                float(fit["slope_seconds_per_genotype"]) * x_line
                + float(fit["intercept_seconds"]),
                0.0,
            )

            r_squared = fit["r_squared"]

            label = (
                f"{runtime_model_label} (R²={float(r_squared):.2f})"
                if pd.notna(r_squared)
                else runtime_model_label
            )

            (line,) = ax.plot(
                x_line,
                y_line,
                color=color,
                linewidth=2.4,
                label=label,
                zorder=3,
            )
            legend_by_label.setdefault(label, line)

        ax.set_title(runtime_panel_title(panel_label), fontsize="xx-large")

        ax.set_xlabel("Number of Genotypes (Samples × Loci)", fontsize="xx-large")

        # Apply the EngFormatter to the x-axis
        ax.xaxis.set_major_formatter(mticker.EngFormatter(unit=""))
        ax.tick_params(axis="both", labelsize="xx-large")

    axes_flat[0].set_ylabel("Mean Execution Time (seconds)", fontsize="xx-large")

    axes_flat[0].set_ylim(bottom=0.0)

    axes_flat[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    if legend_by_label:
        fig.legend(
            list(legend_by_label.values()),
            list(legend_by_label.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            fontsize="xx-large",
            title="Execution Time (seconds)",
            title_fontsize="xx-large",
            fancybox=True,
            shadow=True,
        )

    fig.text(
        0.5,
        0.01,
        "Each point is a dataset-model-backend mean across retained simulation strategies; execution times are untransformed seconds.",
        ha="center",
        va="bottom",
        fontsize="xx-large",
        color="#444444",
    )

    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.75))
    save_figure(fig, out_dir / "runtime_genotype_count_regression")


def dataset_size_label(df: pd.DataFrame) -> pd.Series:
    """Return compact dataset labels with sample and locus counts."""
    samples = pd.to_numeric(df["n_samples"], errors="coerce").round().astype("Int64")

    loci = pd.to_numeric(df["n_loci"], errors="coerce").round().astype("Int64")

    return (
        df["dataset_id"].astype(str)
        + "\n"
        + samples.map(lambda value: "" if pd.isna(value) else f"{int(value):,}")
        + " samples × "
        + loci.map(lambda value: "" if pd.isna(value) else f"{int(value):,}")
        + " loci"
    )


def dataset_size_order(df: pd.DataFrame) -> list[str]:
    """Order dataset-size labels from smallest to largest genotype matrix."""
    required = {"dataset_size_label", "n_cells", "n_samples", "n_loci", "dataset_id"}

    if df.empty or not required.issubset(df.columns):
        return []

    order_df = df[
        ["dataset_size_label", "n_cells", "n_samples", "n_loci", "dataset_id"]
    ].drop_duplicates()

    for column in ("n_cells", "n_samples", "n_loci"):
        order_df[column] = pd.to_numeric(order_df[column], errors="coerce")

    order_df = order_df.sort_values(
        ["n_cells", "n_samples", "n_loci", "dataset_id"],
        kind="mergesort",
    )

    return order_df["dataset_size_label"].astype(str).tolist()


def runtime_size_comparison_group(row: pd.Series) -> str | None:
    """Map runtime rows to direct-comparison groups."""
    software = str(row.get("software", ""))
    model = str(row.get("model", ""))
    model_label = str(row.get("model_label", model))

    if software == "PG-SUI" and model in PGSUI_DEEP_MODELS:
        backend_label = str(row.get("runtime_backend_label", "")).strip()

        if not backend_label or backend_label.lower() == "nan":
            backend_label = backend_display_label(row.get("runtime_backend", ""))

        if backend_label in {"CPU", "GPU", "MPS"}:
            return f"PG-SUI DL ({backend_label})"

        return "PG-SUI DL"

    if software == "PG-SUI" and model in PGSUI_DETERMINISTIC_MODELS:
        return CATEGORY_SHORT_LABELS["PG-SUI: Deterministic"]

    return CATEGORY_SHORT_LABELS.get(MODEL_CATEGORY.get(model_label))  # type: ignore


def dataset_size_f1_runtime_comparison_df(
    combined: pd.DataFrame,
    runtime_combined: pd.DataFrame,
) -> pd.DataFrame:
    """Build a direct dataset-size comparison table for F1 and runtime."""
    summary_frames: list[pd.DataFrame] = []
    key_cols = [
        "dataset_id",
        "n_samples",
        "n_loci",
        "n_cells",
        "dataset_size_label",
        "comparison_group",
    ]

    f1_required = {"dataset_id", "model_label", "macro_f1", "n_samples", "n_loci"}

    if not combined.empty and f1_required.issubset(combined.columns):
        f1_df = _coerce_numeric(
            with_model_category(combined),
            ["macro_f1", "n_samples", "n_loci", "n_cells"],
        )

        f1_df = f1_df.dropna(
            subset=["dataset_id", "model_category", "macro_f1", "n_samples", "n_loci"]
        )

        if not f1_df.empty:
            if "n_cells" not in f1_df.columns:
                f1_df["n_cells"] = f1_df["n_samples"] * f1_df["n_loci"]

            f1_df["n_cells"] = f1_df["n_cells"].fillna(
                f1_df["n_samples"] * f1_df["n_loci"]
            )

            f1_df["comparison_group"] = f1_df["model_category"].map(
                CATEGORY_SHORT_LABELS
            )

            f1_df["dataset_size_label"] = dataset_size_label(f1_df)

            f1_summary = (
                f1_df.dropna(subset=["comparison_group"])
                .groupby(key_cols, observed=True)["macro_f1"]
                .agg(
                    n_runs="count",
                    mean="mean",
                    median="median",
                    sd="std",
                    min="min",
                    max="max",
                )
                .reset_index()
            )

            f1_summary["metric"] = "Macro F1"
            f1_summary["plot_value"] = f1_summary["mean"]
            f1_summary["plot_statistic"] = "mean"
            summary_frames.append(f1_summary)

    runtime_required = {
        "dataset_id",
        "software",
        "model",
        "model_label",
        "runtime_seconds",
        "n_samples",
        "n_loci",
    }

    if not runtime_combined.empty and runtime_required.issubset(
        runtime_combined.columns
    ):
        runtime_df = _coerce_numeric(
            runtime_combined,
            ["runtime_seconds", "n_samples", "n_loci", "n_cells"],
        )

        runtime_df = runtime_df.dropna(
            subset=["dataset_id", "runtime_seconds", "n_samples", "n_loci"]
        ).loc[lambda frame: frame["runtime_seconds"].gt(0)]

        if not runtime_df.empty:
            if "n_cells" not in runtime_df.columns:
                runtime_df["n_cells"] = runtime_df["n_samples"] * runtime_df["n_loci"]

            runtime_df["n_cells"] = runtime_df["n_cells"].fillna(
                runtime_df["n_samples"] * runtime_df["n_loci"]
            )

            runtime_df["comparison_group"] = runtime_df.apply(
                runtime_size_comparison_group, axis=1
            )

            runtime_df["dataset_size_label"] = dataset_size_label(runtime_df)  # type: ignore

            runtime_summary = (
                runtime_df.dropna(subset=["comparison_group"])  # type: ignore
                .groupby(key_cols, observed=True)["runtime_seconds"]
                .agg(
                    n_runs="count",
                    mean="mean",
                    median="median",
                    sd="std",
                    min="min",
                    max="max",
                )
                .reset_index()
            )

            runtime_summary["metric"] = "Runtime seconds"
            runtime_summary["plot_value"] = runtime_summary["median"]
            runtime_summary["plot_statistic"] = "median"
            summary_frames.append(runtime_summary)

    if not summary_frames:
        return pd.DataFrame()

    out = pd.concat(summary_frames, ignore_index=True)

    out["metric"] = pd.Categorical(
        out["metric"], categories=["Macro F1", "Runtime seconds"], ordered=True
    )

    metric_rank = {"Macro F1": 0, "Runtime seconds": 1}

    group_rank = {
        "Naive": 0,
        "SOM": 1,
        "PG-SUI Det.": 2,
        "PG-SUI DL": 3,
        "PG-SUI DL (CPU)": 3,
        "PG-SUI DL (GPU)": 4,
        "PG-SUI DL (MPS)": 5,
    }

    out["_metric_rank"] = out["metric"].astype(str).map(metric_rank).fillna(99)

    out["_group_rank"] = out["comparison_group"].astype(str).map(group_rank).fillna(99)

    out = out.sort_values(
        ["_metric_rank", "n_cells", "dataset_id", "_group_rank"],
        kind="mergesort",
    )

    return out.drop(columns=["_metric_rank", "_group_rank"]).reset_index(drop=True)


def plot_dataset_size_f1_runtime_comparison(
    combined: pd.DataFrame,
    runtime_combined: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Directly compare F1 and runtime across ordered dataset sizes."""
    summary = dataset_size_f1_runtime_comparison_df(combined, runtime_combined)

    if summary.empty:
        print("Skipping direct dataset-size comparison plot: no F1/runtime summary.")
        return

    dataset_order = dataset_size_order(summary)

    if not dataset_order:
        return

    f1_order = ["Naive", "SOM", "PG-SUI Det.", "PG-SUI DL"]

    runtime_order = [
        "Naive",
        "SOM",
        "PG-SUI Det.",
        "PG-SUI DL (CPU)",
        "PG-SUI DL (GPU)",
    ]

    palette = {
        "Naive": CATEGORY_COLORS["GTImputation: Naive"],
        "SOM": CATEGORY_COLORS["GTImputation: SOM (deep learning)"],
        "PG-SUI Det.": CATEGORY_COLORS["PG-SUI: Deterministic"],
        "PG-SUI DL": CATEGORY_COLORS["PG-SUI: Deep learning"],
        "PG-SUI DL (CPU)": lighten_color(
            CATEGORY_COLORS["PG-SUI: Deep learning"], 0.45
        ),
        "PG-SUI DL (GPU)": CATEGORY_COLORS["PG-SUI: Deep learning"],
    }

    f1_df = summary.loc[summary["metric"].astype(str).eq("Macro F1")].copy()

    runtime_df = summary.loc[summary["metric"].astype(str).eq("Runtime seconds")].copy()

    apply_house_style()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(max(14.8, len(dataset_order) * 1.42), 11.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )
    sns.despine(fig=fig)

    if not f1_df.empty:
        present_f1_order = [
            group for group in f1_order if group in set(f1_df["comparison_group"])
        ]

        sns.barplot(
            data=f1_df,
            x="dataset_size_label",
            y="plot_value",
            hue="comparison_group",
            order=dataset_order,
            hue_order=present_f1_order,
            palette=palette,
            errorbar=None,
            edgecolor="black",
            linewidth=0.7,
            ax=axes[0],
        )

        axes[0].set_ylim(0.0, 1.02)
        axes[0].set_ylabel("Mean Macro F1-score", fontsize="xx-large")
        axes[0].set_xlabel("Dataset", fontsize="xx-large")

        format_decimal_ticks(axes[0], axis="y", decimals=2)

        axes[0].legend(
            title="F1-score",
            ncol=len(present_f1_order),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            fontsize="xx-large",
            title_fontsize="xx-large",
            fancybox=True,
            shadow=True,
        )

    if not runtime_df.empty:
        present_runtime_order = [
            group
            for group in runtime_order
            if group in set(runtime_df["comparison_group"])
        ]

        sns.barplot(
            data=runtime_df,
            x="dataset_size_label",
            y="plot_value",
            hue="comparison_group",
            order=dataset_order,
            hue_order=present_runtime_order,
            palette=palette,
            errorbar=None,
            edgecolor="black",
            linewidth=0.7,
            ax=axes[1],
        )

        axes[1].set_ylabel("Median Execution Time (seconds)", fontsize="xx-large")

        axes[1].set_xlabel("Dataset", fontsize="xx-large")

        axes[1].legend(
            title="Execution Time",
            ncol=len(present_runtime_order),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            fontsize="xx-large",
            title_fontsize="xx-large",
            fancybox=True,
            shadow=True,
        )

    for ax in axes:
        ax.tick_params(axis="y", labelsize="xx-large")

    axes[1].set_xticks(range(len(dataset_order)))

    axes[1].set_xticklabels(dataset_order, rotation=34, ha="right", fontsize="xx-large")

    axes[1].set_yticks(axes[1].get_yticks())

    axes[1].set_yticklabels(
        [f"{int(y):,}" for y in axes[1].get_yticks()], fontsize="xx-large"
    )

    fig.text(
        0.5,
        0.01,
        "F1 bars are means over retained strategies and models within each family; runtime bars are median execution times, with PG-SUI deep learning split by device.",
        ha="center",
        va="bottom",
        fontsize="xx-large",
        color="#444444",
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.22,
        top=0.88,
        hspace=0.46,
    )

    save_figure(fig, out_dir / "dataset_size_f1_runtime_comparison")


def plot_macro_f1_model_category(combined: pd.DataFrame, out_dir: Path) -> None:
    """Grouped mean macro-F1 bars organized into the four analysis families.

    Coloured background bands and family headers make the PG-SUI (deterministic
    and deep learning) versus GTImputation (naive and SOM) contrast explicit.
    """
    df = _coerce_numeric(combined, ["macro_f1"]).dropna(subset=["macro_f1"]).copy()

    if df.empty:
        return

    df["model_label"] = df["model_label"].astype(str)

    apply_house_style()

    summary = summarize_mean_ci(df, ["model_label"], "macro_f1").set_index(
        "model_label"
    )

    model_x: dict[str, float] = {}

    category_spans: dict[str, tuple[float, float]] = {}
    cursor = 0.0

    for category in CATEGORY_ORDER:
        members = [
            model
            for model in MODEL_ORDER
            if MODEL_CATEGORY.get(model) == category and model in summary.index
        ]

        if not members:
            continue

        start = cursor

        for model in members:
            model_x[model] = cursor
            cursor += 1.0
        category_spans[category] = (start - 0.5, cursor - 0.5)
        cursor += 0.8  # gap between families

    if not model_x:
        return

    fig, ax = plt.subplots(figsize=(max(9.5, cursor * 0.95), 5.8))
    sns.despine(fig=fig)

    for category, (left, right) in category_spans.items():
        ax.axvspan(
            left,
            right,
            color=lighten_color(CATEGORY_COLORS[category], 0.82),
            alpha=0.55,
            zorder=0,
        )

        ax.text(
            (left + right) / 2.0,
            1.02,
            CATEGORY_SHORT_LABELS.get(category, category),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize="x-large",
            fontweight="bold",
            color="#333333",
        )

    for model, x_pos in model_x.items():
        row = summary.loc[model]
        mean = float(row["mean"])  # type: ignore
        lower = max(mean - float(row["ci95_min"]), 0.0)  # type: ignore
        upper = max(float(row["ci95_max"]) - mean, 0.0)  # type: ignore

        ax.bar(
            x_pos,
            mean,
            width=0.8,
            color=MODEL_PALETTE.get(model, "#777777"),
            edgecolor="black",
            linewidth=0.9,
            yerr=[[lower], [upper]],
            error_kw={
                "ecolor": "black",
                "elinewidth": 1.4,
                "capsize": 4.5,
                "capthick": 1.4,
            },
            zorder=3,
        )

        ax.text(
            x_pos,
            mean + upper + 0.02,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize="x-large",
            color="#222222",
        )

    ax.set_xticks(list(model_x.values()))

    ax.set_xticklabels(
        list(model_x.keys()), rotation=30, ha="right", fontsize="x-large"
    )

    ax.set_xlim(-0.9, cursor - 1.1)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_ylabel("Mean Macro F1-score", fontsize="x-large")
    ax.set_xlabel("")

    format_decimal_ticks(ax, axis="y", decimals=2)

    fig.tight_layout()
    save_figure(fig, out_dir / "macro_f1_by_model_category")


def plot_macro_f1_heatmap(combined: pd.DataFrame, out_dir: Path) -> None:
    """Annotated model x dataset macro-F1 heatmap with family separators."""
    df = _coerce_numeric(combined, ["macro_f1"]).dropna(subset=["macro_f1"]).copy()

    if df.empty:
        return

    df["model_label"] = df["model_label"].astype(str)

    model_order = [model for model in MODEL_ORDER if model in set(df["model_label"])]

    pivot = df.pivot_table(
        index="model_label",
        columns="dataset_id",
        values="macro_f1",
        aggfunc="mean",
        observed=True,
    ).reindex(model_order)

    datasets = sorted(pivot.columns.astype(str))
    pivot = pivot[datasets]

    if pivot.empty:
        return

    apply_house_style()

    fig, ax = plt.subplots(
        figsize=(max(10.0, 0.95 * len(datasets) + 3.0), 0.62 * len(model_order) + 2.6)
    )

    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]

            if not np.isfinite(value):
                continue

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize="x-large",
                color="white" if value < 0.62 else "#111111",
            )

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize="x-large")
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels(model_order, fontsize="x-large")

    # Draw a separator between adjacent models from different families.
    for i in range(1, len(model_order)):
        if MODEL_CATEGORY.get(model_order[i]) != MODEL_CATEGORY.get(model_order[i - 1]):
            ax.axhline(i - 0.5, color="white", linewidth=2.4)

    ax.set_xlabel("Dataset", fontsize="x-large")
    ax.set_ylabel("Model", fontsize="x-large")

    ax.set_title(
        "Mean Macro F1 by Model and Dataset", fontsize="x-large", fontweight="bold"
    )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Macro F1-score", fontsize="x-large")
    colorbar.ax.tick_params(labelsize="x-large")

    fig.tight_layout()
    save_figure(fig, out_dir / "macro_f1_heatmap_model_dataset")


def plot_metric_radar(combined: pd.DataFrame, out_dir: Path) -> None:
    """Radar chart of the best model in each family across six metrics.

    Args:
        combined (pd.DataFrame): DataFrame containing model performance metrics.
        out_dir (Path): Directory to save the radar chart figure.
    """
    numeric = ["macro_f1", "mcc", "ref_f1", "het_f1", "alt_f1", "runtime_seconds"]

    df = _coerce_numeric(with_model_category(combined), numeric)
    df = df.dropna(subset=["model_category", "macro_f1"])

    if df.empty:
        return

    stats = (
        df.groupby(["model_category", "model_label"], observed=True)
        .agg(
            macro_f1=("macro_f1", "mean"),
            mcc=("mcc", "mean"),
            ref_f1=("ref_f1", "mean"),
            het_f1=("het_f1", "mean"),
            alt_f1=("alt_f1", "mean"),
            runtime=("runtime_seconds", "mean"),
        )
        .reset_index()
    )

    champions: list[pd.Series] = []

    for category in CATEGORY_ORDER:
        subset = stats.loc[stats["model_category"].eq(category)]

        if subset.empty:
            continue

        champions.append(subset.loc[subset["macro_f1"].idxmax()])  # type: ignore

    if len(champions) < 2:
        return

    champ_df = pd.DataFrame(champions).reset_index(drop=True)

    # Convert mean runtime to a 0-1 "speed" score (fastest champion = 1).
    log_runtime = np.log10(
        np.clip(champ_df["runtime"].to_numpy(dtype=float), 1e-3, None)
    )

    if np.ptp(log_runtime) > 0:
        speed = (log_runtime.max() - log_runtime) / (
            log_runtime.max() - log_runtime.min()
        )
    else:
        speed = np.ones_like(log_runtime)

    metric_labels = ["Macro F1", "MCC", "REF F1", "HET F1", "ALT F1", "Speed"]

    metric_values = np.column_stack(
        [
            champ_df["macro_f1"].to_numpy(dtype=float),
            np.clip((champ_df["mcc"].to_numpy(dtype=float) + 1.0) / 2.0, 0.0, 1.0),
            champ_df["ref_f1"].to_numpy(dtype=float),
            champ_df["het_f1"].to_numpy(dtype=float),
            champ_df["alt_f1"].to_numpy(dtype=float),
            speed,
        ]
    )

    apply_house_style()

    angles = np.linspace(0.0, 2.0 * np.pi, len(metric_labels), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(8.6, 8.6), subplot_kw={"polar": True})

    ax.set_theta_offset(np.pi / 2.0)  # type: ignore
    ax.set_theta_direction(-1)  # type: ignore

    for index, row in champ_df.iterrows():
        category = str(row["model_category"])

        values = np.concatenate([metric_values[index], metric_values[index][:1]])  # type: ignore

        color = CATEGORY_COLORS.get(category, "#777777")
        label = f"{row['model_label']}"

        ax.plot(closed_angles, values, color=color, linewidth=4.0, label=label)
        ax.fill(closed_angles, values, color=color, alpha=0.12)

        yscatter = metric_values[index]  # type: ignore

        ax.scatter(angles, yscatter, color=color, edgecolor="black", s=45, zorder=5)

    ax.set_xticks(angles)
    ax.set_xticklabels(metric_labels, fontsize="xx-large")
    ax.set_ylim(0.0, 1.0)

    yticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(x) for x in yticks], fontsize="xx-large", color="#555555")

    ax.tick_params(axis="x", labelsize="xx-large", pad=24)
    ax.tick_params(axis="y", labelsize="xx-large", pad=16)

    ax.set_rlabel_position(180.0 / len(metric_labels))  # type: ignore

    for label in ax.get_yticklabels():
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 2.0})

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        title_fontsize="xx-large",
        fontsize="xx-large",
        shadow=True,
        fancybox=True,
        title="Speed = Mean Runtime Rescaled (slowest=0, fastest=1)",
    )

    save_figure(fig, out_dir / "metric_radar_family_champions")


def plot_macro_f1_distribution(combined: pd.DataFrame, out_dir: Path) -> None:
    """Per-model macro-F1 distribution as horizontal violins with points."""
    df = _coerce_numeric(combined, ["macro_f1"]).dropna(subset=["macro_f1"]).copy()

    if df.empty:
        return

    df["model_label"] = df["model_label"].astype(str)
    model_order = [model for model in MODEL_ORDER if model in set(df["model_label"])]

    if not model_order:
        return

    apply_house_style()

    fig, ax = plt.subplots(figsize=(10.0, 7.4))
    sns.despine(fig=fig)

    sns.violinplot(
        data=df,
        x="macro_f1",
        y="model_label",
        order=model_order,
        hue="model_label",
        hue_order=model_order,
        palette=MODEL_PALETTE,
        legend=False,
        orient="h",
        inner="box",
        cut=0,
        linewidth=1.0,
        density_norm="width",
        ax=ax,
    )

    sns.stripplot(
        data=df,
        x="macro_f1",
        y="model_label",
        order=model_order,
        color="black",
        size=2.6,
        alpha=0.3,
        jitter=0.18,
        ax=ax,
    )

    means = df.groupby("model_label", observed=True)["macro_f1"].mean()
    for index, model in enumerate(model_order):
        if model in means.index:
            ax.scatter(
                float(means[model]),
                index,
                marker="D",
                s=65,
                color="white",
                edgecolor="black",
                linewidth=1.1,
                zorder=6,
            )

    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Macro F1-score", fontsize="xx-large")
    ax.set_ylabel("")

    format_decimal_ticks(ax, axis="x", decimals=2)
    ax.tick_params(axis="y", labelsize="xx-large")

    ax.set_title(
        "Macro F1 Distribution Across Dataset-Strategy Runs",
        fontsize="xx-large",
        fontweight="bold",
    )

    mean_handle = mlines.Line2D(
        [0],
        [0],
        marker="D",
        color="white",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=9,
        linestyle="",
        label="Mean",
    )

    ax.legend(
        handles=[mean_handle],
        loc="lower right",
        fontsize="xx-large",
        title_fontsize="xx-large",
    )

    fig.tight_layout()
    save_figure(fig, out_dir / "macro_f1_distribution_by_model")


def plot_head_to_head_dumbbell(combined: pd.DataFrame, out_dir: Path) -> None:
    """Dumbbell chart of best PG-SUI vs best GTImputation macro F1 per dataset.

    Args:
        combined (pd.DataFrame): DataFrame containing model performance metrics.
        out_dir (Path): Directory to save the dumbbell chart figure.
    """
    required = {"dataset_id", "software", "model_label", "macro_f1"}

    if combined.empty or not required.issubset(combined.columns):
        return

    df = _coerce_numeric(combined, ["macro_f1"]).dropna(subset=["macro_f1"])

    per = (
        df.groupby(["dataset_id", "software", "model_label"], observed=True)["macro_f1"]
        .mean()
        .reset_index()
    )

    rows: list[dict[str, object]] = []

    for dataset_id, group in per.groupby("dataset_id", observed=True):
        pgsui = group.loc[group["software"].eq("PG-SUI")]
        gti = group.loc[group["software"].eq("GTImputation")]

        if pgsui.empty or gti.empty:
            continue

        best_pgsui = pgsui.loc[pgsui["macro_f1"].idxmax()]
        best_gti = gti.loc[gti["macro_f1"].idxmax()]

        rows.append(
            {
                "dataset_id": str(dataset_id),
                "pgsui": float(best_pgsui["macro_f1"]),  # type: ignore
                "pgsui_model": str(best_pgsui["model_label"]),
                "gti": float(best_gti["macro_f1"]),  # type: ignore
                "gti_model": str(best_gti["model_label"]),
                "delta": float(best_pgsui["macro_f1"] - best_gti["macro_f1"]),  # type: ignore
            }
        )

    plot_df = pd.DataFrame(rows)

    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("delta", ascending=True).reset_index(drop=True)

    y_positions = np.arange(len(plot_df))

    apply_house_style()

    fig, ax = plt.subplots(figsize=(8.6, max(4.5, 0.5 * len(plot_df) + 2.0)))

    sns.despine(fig=fig)

    pgsui_color = CATEGORY_COLORS["PG-SUI: Deep learning"]
    gti_color = "#4D4D4D"

    for y_pos, row in zip(y_positions, plot_df.itertuples(index=False)):
        ax.plot(
            [row.gti, row.pgsui],  # type: ignore
            [y_pos, y_pos],
            color="#B8B8C0",
            linewidth=2.4,
            zorder=1,
        )

    ax.scatter(
        plot_df["gti"],
        y_positions,
        s=160,
        marker="s",
        color=gti_color,
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
        label="Best GTImputation",
    )

    ax.scatter(
        plot_df["pgsui"],
        y_positions,
        s=160,
        marker="o",
        color=pgsui_color,
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
        label="Best PG-SUI",
    )

    for y_pos, row in zip(y_positions, plot_df.itertuples(index=False)):
        ytext = row.delta >= 0  # type: ignore

        ax.text(
            max(row.pgsui, row.gti) + 0.025,  # type: ignore
            y_pos,  # type: ignore
            f"+{row.delta:.2f}" if ytext else f"{row.delta:.2f}",
            va="center",
            ha="left",
            fontsize="xx-large",
            color="#555555" if ytext else "#a11",
        )

    format_decimal_ticks(ax, axis="x", decimals=2)
    ax.set_xlim(0.0, 1.02)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["dataset_id"], fontsize="xx-large")

    ax.set_xlabel("Macro F1-score", fontsize="xx-large")
    ax.set_ylabel("Dataset", fontsize="xx-large")

    ax.legend(
        loc="lower left",
        title_fontsize="xx-large",
        fontsize="xx-large",
        shadow=True,
        fancybox=True,
    )

    fig.tight_layout()
    save_figure(fig, out_dir / "head_to_head_dumbbell_by_dataset")


def plot_win_rate_matrix(combined: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of dataset-level PG-SUI win rate over each GTImputation method."""
    table = build_head_to_head_table(combined)

    if table.empty:
        return

    pgsui_order = [
        PGSUI_MODEL_LABELS[model]
        for model in PGSUI_MODELS
        if PGSUI_MODEL_LABELS[model] in set(table["pgsui_model"])
    ]

    gti_order = [
        GTI_MODEL_LABELS[method]
        for method in ("naive", "som")
        if GTI_MODEL_LABELS[method] in set(table["gti_method"])
    ]

    if not pgsui_order or not gti_order:
        return

    win = table.pivot(
        index="pgsui_model", columns="gti_method", values="dataset_win_rate"
    )

    delta = table.pivot(
        index="pgsui_model", columns="gti_method", values="mean_delta_macro_f1"
    )

    win = win.reindex(index=pgsui_order, columns=gti_order)
    delta = delta.reindex(index=pgsui_order, columns=gti_order)

    apply_house_style()

    fig, ax = plt.subplots(
        figsize=(1.9 * len(gti_order) + 3.2, 0.7 * len(pgsui_order) + 2.4)
    )

    values = win.to_numpy(dtype=float) * 100.0

    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            win_value = values[i, j]
            delta_value = delta.to_numpy(dtype=float)[i, j]

            if not np.isfinite(win_value):
                continue

            text_color = _get_contrasting_text_color(image.cmap(win_value / 100.0))

            ax.text(
                j,
                i,
                f"{win_value:.0f}%\nΔ {delta_value:+.2f}",
                ha="center",
                va="center",
                fontsize="xx-large",
                color=text_color,
                fontweight="bold",
            )

    ax.set_xticks(np.arange(len(gti_order)))

    ax.set_xticklabels(
        [x.replace(" (GTImputation)", "") for x in gti_order], fontsize="xx-large"
    )

    ax.set_yticks(np.arange(len(pgsui_order)))
    ax.set_yticklabels(pgsui_order, fontsize="xx-large")
    ax.set_xlabel("GTImputation Model", fontsize="xx-large")
    ax.set_ylabel("PG-SUI Model", fontsize="xx-large")

    ax.set_title(
        "Dataset-Level PG-SUI Win Rate over GTImputation",
        fontsize="xx-large",
        fontweight="bold",
    )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label("Dataset-level PG-SUI win rate (%)", fontsize="xx-large")
    colorbar.ax.tick_params(labelsize="xx-large")

    fig.text(
        0.5,
        0.005,
        "Win rate = share of datasets whose mean difference across strategies favors PG-SUI; Δ = mean dataset-level macro-F1 difference.",
        ha="center",
        va="bottom",
        fontsize="xx-large",
        color="#555555",
    )

    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    save_figure(fig, out_dir / "win_rate_matrix")


def plot_all(
    combined: pd.DataFrame,
    combined_class: pd.DataFrame,
    runtime_combined: pd.DataFrame,
    plots_dir: Path,
    dpi: int,
) -> None:
    configure_plot_style(dpi)
    plot_macro_f1_by_model(combined.copy(), plots_dir)
    plot_macro_f1_model_category(combined.copy(), plots_dir)
    plot_macro_f1_by_strategy(combined.copy(), plots_dir)
    plot_macro_f1_heatmap(combined.copy(), plots_dir)
    plot_macro_f1_distribution(combined.copy(), plots_dir)
    plot_performance_by_dataset_size(combined.copy(), plots_dir)
    plot_f1_size_regressions(combined.copy(), plots_dir)
    plot_f1_genotype_count_regressions(combined.copy(), plots_dir)
    plot_runtime_genotype_count_regressions(runtime_combined.copy(), plots_dir)

    plot_dataset_size_f1_runtime_comparison(
        combined.copy(), runtime_combined.copy(), plots_dir
    )

    plot_runtime_by_model(runtime_combined.copy(), plots_dir)
    plot_class_f1_by_model(combined_class.copy(), plots_dir)
    plot_class_f1_by_software(combined_class.copy(), plots_dir)
    plot_class_f1_by_strategy(combined_class.copy(), plots_dir)
    plot_class_f1_strategy_model_bars(combined_class.copy(), plots_dir)
    plot_mcc_by_strategy_model(combined.copy(), plots_dir)
    plot_f1_class_balance(combined.copy(), plots_dir)
    plot_metric_radar(combined.copy(), plots_dir)
    plot_cpu_gpu_runtime_difference(runtime_combined.copy(), plots_dir)
    plot_performance_efficiency_frontier(combined.copy(), plots_dir)
    plot_best_deltas(combined.copy(), plots_dir)
    plot_head_to_head_dumbbell(combined.copy(), plots_dir)
    plot_win_rate_matrix(combined.copy(), plots_dir)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    formatted = df.copy()

    for column in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )
        else:
            formatted[column] = (
                formatted[column].astype("object").where(formatted[column].notna(), "")
            )

    headers = list(formatted.columns)
    rows = formatted.astype(str).values.tolist()

    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def render(values: Sequence[str]) -> str:
        return (
            "| "
            + " | ".join(
                value.ljust(widths[index]) for index, value in enumerate(values)
            )
            + " |"
        )

    separator = "| " + " | ".join("-" * width for width in widths) + " |"

    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def write_readme(
    output_dir: Path,
    *,
    pgsui_dir: Path,
    gti_dir: Path,
    sim_manifest: Path,
    pgsui_status: pd.DataFrame,
    selected_pgsui: pd.DataFrame,
    selected_pgsui_runtime: pd.DataFrame,
    retained_keys: pd.DataFrame,
    combined: pd.DataFrame,
    runtime_combined: pd.DataFrame,
    best_summary_df: pd.DataFrame,
    gti_errors: pd.DataFrame,
    allow_partial_grid: bool,
) -> None:
    """Write a README.md summary of the current-results directory.

    Args:
        output_dir (Path): Directory to write the README.md file.
        pgsui_dir (Path): Path to the PG-SUI results directory.
        gti_dir (Path): Path to the GTImputation results directory.
        sim_manifest (Path): Path to the simulation manifest file.
        pgsui_status (pd.DataFrame): DataFrame containing PG-SUI status information.
        selected_pgsui (pd.DataFrame): DataFrame containing selected PG-SUI results.
        selected_pgsui_runtime (pd.DataFrame): DataFrame containing selected PG-SUI runtime information.
        retained_keys (pd.DataFrame): DataFrame containing retained dataset-strategy keys.
        combined (pd.DataFrame): DataFrame containing combined results.
        runtime_combined (pd.DataFrame): DataFrame containing combined runtime results.
        best_summary_df (pd.DataFrame): DataFrame containing best model summary information.
        gti_errors (pd.DataFrame): DataFrame containing GTImputation error information.
        allow_partial_grid (bool): Flag indicating whether partial grid results are allowed.
    """

    complete_dataset_count = (
        retained_keys["dataset_id"].nunique() if not retained_keys.empty else 0
    )

    complete_pair_count = (
        retained_keys[["dataset_id", "strategy"]].drop_duplicates().shape[0]
    )

    incomplete_pgsui = pgsui_status.groupby(
        ["dataset_id", "strategy", "backend", "output_dir"], as_index=False
    ).agg(complete_models=("model_complete", "sum"), models=("model", "nunique"))

    incomplete_pgsui = incomplete_pgsui.loc[
        incomplete_pgsui["complete_models"].lt(incomplete_pgsui["models"])
    ]

    scored_site_counts = pd.DataFrame()

    if not combined.empty and {
        "mask_rows",
        "scored_masked_sites",
        "dropped_unmasked_mask_rows",
    }.issubset(combined.columns):
        scored_site_counts = combined[
            [
                "dataset_id",
                "strategy",
                "mask_rows",
                "scored_masked_sites",
                "dropped_unmasked_mask_rows",
            ]
        ].drop_duplicates()

    deep_runtime = (
        runtime_combined.loc[
            runtime_combined["model"].isin(PGSUI_DEEP_MODELS)
            & runtime_combined["software"].eq("PG-SUI")
        ]
        if not runtime_combined.empty and "run_mode" in runtime_combined.columns
        else pd.DataFrame()
    )

    deep_parallel_jobs = ""

    if not deep_runtime.empty and "parallel_jobs" in deep_runtime.columns:
        jobs = sorted(
            {
                int(value)
                for value in deep_runtime["parallel_jobs"].dropna().unique()
                if np.isfinite(float(value))
            }
        )
        deep_parallel_jobs = ", ".join(str(value) for value in jobs)

    f1_backends = (
        ", ".join(sorted(selected_pgsui["backend"].dropna().astype(str).unique()))
        if not selected_pgsui.empty and "backend" in selected_pgsui.columns
        else "not available"
    )

    runtime_backends = (
        ", ".join(
            sorted(selected_pgsui_runtime["backend"].dropna().astype(str).unique())
        )
        if not selected_pgsui_runtime.empty
        and "backend" in selected_pgsui_runtime.columns
        else "not available"
    )

    best_readme_columns = [
        "dataset_id",
        "strategy",
        "pgsui_model",
        "gti_model",
        "pgsui_macro_f1",
        "gti_macro_f1",
        "delta_macro_f1",
    ]

    missing_best_columns = set(best_readme_columns).difference(best_summary_df.columns)

    if missing_best_columns:
        raise ValueError(
            f"best_summary_df is missing README columns: {sorted(missing_best_columns)}"
        )

    best_readme_df = (
        best_summary_df.loc[:, best_readme_columns]
        .drop_duplicates(subset=["dataset_id", "strategy"])
        .sort_values(
            ["dataset_id", "strategy"],
            kind="mergesort",
        )
        .head(30)
    )

    lines = [
        "# PG-SUI vs GTImputation current-results summary",
        "",
        f"- PG-SUI directory: `{pgsui_dir}`",
        f"- GTImputation directory: `{gti_dir}`",
        f"- Simulation manifest: `{sim_manifest}`",
        f"- Grid mode: `{'partial complete pairs' if allow_partial_grid else 'complete all-strategy datasets'}`",
        "- PG-SUI F1 scoring: PG-SUI classification reports generated from held-out simulated missing genotypes",
        "- GTImputation F1 scoring: imputed VCFs versus original VCFs at actual missing genotypes in the masked simulation VCF",
        f"- PG-SUI F1 backend filter: `{f1_backends}` only; CPU/GPU duplicate PG-SUI runs are not duplicated in F1 tables",
        "- PG-SUI VCF note: saved `imputed/*.vcf.gz` files appear truth-restored at masked sites and are not used for F1 scoring",
        "- GTImputation SOM timing: `gtdb_build_real_seconds + imputation_phase_real_seconds`",
        f"- Runtime backend comparison: PG-SUI deep-learning models are separated by `{runtime_backends}`; deterministic PG-SUI, GTImputation Naive, and GTImputation SOM are shown once as baselines",
        "- Runtime plots: PG-SUI deep-learning values use the latest logged rolling Optuna `Avg trial (s)`; deterministic PG-SUI and GTImputation values are single-run runtimes",
        "- Runtime accounting: `runtime_wall_seconds` retains full Optuna tuning wall-clock time; `parallel_adjusted_runtime_seconds` is retained only in tables",
        "- Formatted table views: `tables/formatted/index.html` links CSV, HTML, LaTeX, and compact Word-readable RTF table views",
        "- Dataset-size plot: `plots/performance_by_dataset_size.png` summarizes mean macro F1 against per-dataset sample and locus counts",
        "- F1-size regression plot: `plots/f1_size_regression.png` fits linear regressions of macro F1 against sample and locus counts",
        "- Genotype-count regression plot: `plots/f1_genotype_count_regression.png` fits per-model macro-F1 regressions against `n_samples * n_loci`",
        "- Runtime genotype-count regression plot: `plots/runtime_genotype_count_regression.png` fits per-runtime-model execution-time regressions against `n_samples * n_loci`",
        "- Direct size comparison: `plots/dataset_size_f1_runtime_comparison.png` compares F1 and runtime across datasets ordered by `n_samples * n_loci`",
        "- Strategy sensitivity table: `tables/pgsui_vs_gtimputation_strategy_sensitivity_mixed_model.csv` fits `delta_macro_f1 ~ C(strategy, Sum) + (1 | dataset_id)` for each PG-SUI x GTImputation pair",
        f"- PG-SUI deep-learning Optuna jobs parsed from logs: {deep_parallel_jobs or 'not available'} parallel jobs",
        f"- Selected PG-SUI complete dataset-strategy pairs: {selected_pgsui[['dataset_id', 'strategy']].drop_duplicates().shape[0]}",
        f"- Retained dataset-strategy pairs after VCF scoring: {complete_pair_count}",
        f"- Retained datasets: {complete_dataset_count}",
        f"- Combined metric rows: {len(combined)}",
        f"- GTImputation scoring errors: {len(gti_errors)}",
        f"- Mask TSV rows dropped because they were not missing in the masked VCF: {int(scored_site_counts['dropped_unmasked_mask_rows'].sum()) if not scored_site_counts.empty else 0}",
        "",
        "## Best Macro F1 by Software",
        "",
        markdown_table(best_readme_df),
    ]

    family_summary = build_category_summary_table(combined)

    if not family_summary.empty:
        family_columns = [
            column
            for column in (
                "model_category",
                "n_runs",
                "mean_macro_f1",
                "mean_mcc",
                "mean_het_f1",
                "mean_runtime_seconds",
            )
            if column in family_summary.columns
        ]

        lines.extend(
            [
                "",
                "## Accuracy by Model Family",
                "",
                markdown_table(family_summary[family_columns]),
            ]
        )

    head_to_head = build_head_to_head_table(combined)

    if not head_to_head.empty:
        head_columns = [
            column
            for column in (
                "pgsui_model",
                "gti_method",
                "n_datasets",
                "n_dataset_strategy_pairs",
                "dataset_win_rate",
                "dataset_strategy_win_rate",
                "mean_delta_macro_f1",
                "median_delta_macro_f1",
                "ci95_low",
                "ci95_high",
                "p_exact_block_sign_flip",
                "p_holm",
                "reject_holm_0_05",
            )
            if column in head_to_head.columns
        ]

        lines.extend(
            [
                "",
                "## PG-SUI vs GTImputation Head-to-Head",
                "",
                "Primary inference averages PG-SUI-minus-GTImputation macro-F1 "
                "deltas across strategies within each dataset, then tests the "
                "dataset-level means with an exact two-sided sign-flip test. "
                "`p_holm` is Holm-adjusted across the macro-F1 PG-SUI x "
                "GTImputation primary family. `dataset_strategy_win_rate` retains "
                "the descriptive percentage of matched dataset-by-strategy "
                "evaluations favoring PG-SUI.",
                "",
                markdown_table(head_to_head[head_columns]),
            ]
        )

    lines.extend(
        [
            "",
            "## Incomplete PG-SUI Runs",
            "",
            markdown_table(
                incomplete_pgsui[
                    ["dataset_id", "strategy", "backend", "complete_models", "models"]
                ]
                .sort_values(["dataset_id", "strategy", "backend"])
                .head(50)
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## QC Note",
            "",
            "- PG-SUI saved imputed VCFs were sanity-checked and found to match original "
            "truth genotypes at masked sites for trivial models such as ImputeRefAllele. "
            "Using those VCFs would produce impossible F1=1.0 values, so PG-SUI F1 values "
            "are taken from the model classification reports instead.",
        ]
    )

    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    pgsui_dir = args.pgsui_dir.expanduser().resolve()
    gti_dir = resolve_gti_dir(args.gti_dir)
    sim_manifest_path = args.sim_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for stale_table in (
        "pgsui_metrics_vcf_scored.csv",
        "pgsui_class_report_vcf_long.csv",
        "pgsui_vcf_scoring_errors.csv",
        "macro_f1_summary_by_software.csv",
        "runtime_summary_by_software.csv",
        "parallel_adjusted_runtime_summary_by_software.csv",
        "class_f1_summary_by_software.csv",
    ):
        stale_path = tables_dir / stale_table

        if stale_path.exists():
            stale_path.unlink()

    for stale_plot in (
        "parallel_adjusted_runtime_by_model.png",
        "parallel_adjusted_runtime_by_model.pdf",
        "class_f1_by_software.png",
        "class_f1_by_software.pdf",
        "class_f1_strategy_model_lines.png",
        "class_f1_strategy_model_lines.pdf",
    ):
        stale_path = plots_dir / stale_plot

        if stale_path.exists():
            stale_path.unlink()

    pgsui_status, pgsui_report_metrics = collect_pgsui_outputs(
        pgsui_dir,
        models=args.pgsui_models,
        strategies=args.strategies,
        report_type=args.report_type,
    )

    selected_pgsui = select_complete_pgsui_runs(
        pgsui_status,
        models=args.pgsui_models,
        strategies=args.strategies,
        backend=args.pgsui_backend,
        backend_priority=args.backend_priority,
        allow_partial_grid=args.allow_partial_grid,
    )

    selected_pgsui_runtime = select_complete_pgsui_backend_runs(
        pgsui_status,
        models=args.pgsui_models,
        strategies=args.strategies,
        backends=args.runtime_pgsui_backends,
        allow_partial_grid=args.allow_partial_grid,
    )

    gti_manifest = load_gti_manifest(
        gti_dir, methods=args.gti_methods, strategies=args.strategies
    )

    sim_manifest = load_sim_manifest(sim_manifest_path, strategies=args.strategies)

    retained_keys = select_retained_keys(
        selected_pgsui,
        gti_manifest,
        strategies=args.strategies,
        methods=args.gti_methods,
        allow_partial_grid=args.allow_partial_grid,
    )

    write_table(pgsui_status, tables_dir / "pgsui_model_completion_status.csv")

    write_table(
        pgsui_report_metrics,
        tables_dir / "pgsui_report_metrics_all_complete_reports.csv",
    )

    write_table(selected_pgsui, tables_dir / "pgsui_complete_run_selection.csv")

    write_table(
        selected_pgsui_runtime,
        tables_dir / "pgsui_complete_runtime_backend_selection.csv",
    )

    write_table(gti_manifest, tables_dir / "gtimputation_manifest_filtered.csv")

    write_table(
        retained_keys, tables_dir / "retained_complete_dataset_strategy_pairs.csv"
    )

    write_table_view_index(tables_dir)

    if args.skip_gti_scoring:
        print(f"Wrote completion tables to: {output_dir}")
        return

    gti_metrics, gti_class_report, gti_errors = score_gti_metrics(
        gti_manifest,
        sim_manifest,
        retained_keys,
        gti_dir=gti_dir,
        sim_manifest_path=sim_manifest_path,
    )

    retained_keys = filter_after_gti_scoring(
        retained_keys,
        gti_metrics,
        methods=args.gti_methods,
        strategies=args.strategies,
        allow_partial_grid=args.allow_partial_grid,
    )

    pgsui_metrics = select_pgsui_metrics_for_runs(pgsui_report_metrics, retained_keys)

    pgsui_class_report = build_pgsui_class_report_from_reports(pgsui_metrics)
    combined = build_combined_metrics(pgsui_metrics, gti_metrics, retained_keys)
    combined = attach_simulation_size_metadata(combined, sim_manifest)

    combined_class_report = build_combined_class_report(
        pgsui_class_report,
        gti_class_report,
        retained_keys,
    )

    runtime_combined = build_runtime_comparison(
        pgsui_report_metrics,
        gti_metrics,
        retained_keys,
        selected_pgsui,
        selected_pgsui_runtime,
    )

    runtime_combined = attach_simulation_size_metadata(runtime_combined, sim_manifest)

    best_summary_df = best_model_summary(
        combined,
        combined_class_report,
    )

    f1_summary = summarize_numeric(combined, ["software", "model_label"], "macro_f1")

    runtime_summary = summarize_numeric(
        runtime_combined,
        ["software", "runtime_backend_label", "runtime_model_label"],
        "runtime_seconds",
    )

    adjusted_runtime_summary = summarize_numeric(
        runtime_combined,
        ["software", "runtime_backend_label", "runtime_model_label", "run_mode"],
        "parallel_adjusted_runtime_seconds",
    )

    software_model_f1_summary = summarize_numeric(
        combined, ["software", "model_label"], "macro_f1"
    )

    software_model_runtime_summary = summarize_numeric(
        runtime_combined,
        ["software", "runtime_backend_label", "runtime_model_label"],
        "runtime_seconds",
    )

    software_model_adjusted_runtime_summary = summarize_numeric(
        runtime_combined,
        ["software", "runtime_backend_label", "runtime_model_label", "run_mode"],
        "parallel_adjusted_runtime_seconds",
    )

    class_f1_summary = summarize_numeric(
        combined_class_report,
        ["software", "model_label", "genotype_class"],
        "f1",
    )

    class_f1_software_model_summary = summarize_numeric(
        combined_class_report,
        ["software", "model_label", "genotype_class"],
        "f1",
    )

    class_f1_strategy_model_summary = summarize_numeric(
        add_strategy_abbreviations(combined_class_report),
        ["software", "model_label", "genotype_class", "strategy_label_abbr"],
        "f1",
    )

    mcc_strategy_model_summary = summarize_numeric(
        add_strategy_abbreviations(combined),
        ["software", "model_label", "strategy_label_abbr"],
        "mcc",
    )

    cpu_gpu_runtime_delta = cpu_gpu_runtime_delta_df(runtime_combined)
    f1_balance_summary = f1_class_balance_summary_df(combined)
    category_summary = build_category_summary_table(combined)
    head_to_head_summary = build_head_to_head_table(combined)
    strategy_sensitivity = build_strategy_sensitivity_table(combined)
    dataset_leaderboard = build_dataset_leaderboard_table(combined)
    class_f1_wide = build_class_f1_wide_table(combined)
    dataset_size_performance = dataset_size_performance_summary_df(combined)
    f1_size_regression = f1_size_regression_summary_df(combined)

    f1_genotype_count_regression = f1_genotype_count_regression_summary_df(combined)

    runtime_genotype_count_regression = runtime_genotype_count_regression_summary_df(
        runtime_combined
    )

    dataset_size_f1_runtime = dataset_size_f1_runtime_comparison_df(
        combined, runtime_combined
    )

    write_table(pgsui_metrics, tables_dir / "pgsui_metrics_from_reports.csv")
    write_table(
        pgsui_class_report, tables_dir / "pgsui_class_report_from_reports_long.csv"
    )
    write_table(gti_metrics, tables_dir / "gtimputation_metrics_scored.csv")
    write_table(gti_class_report, tables_dir / "gtimputation_class_report_long.csv")
    write_table(gti_errors, tables_dir / "gtimputation_scoring_errors.csv")
    write_table(
        retained_keys,
        tables_dir / "retained_complete_dataset_strategy_pairs_after_scoring.csv",
    )
    write_table(combined, tables_dir / "combined_metrics_long.csv")
    write_table(combined_class_report, tables_dir / "combined_class_report_long.csv")
    write_table(runtime_combined, tables_dir / "runtime_comparison_long.csv")
    write_table(best_summary_df, tables_dir / "best_model_summary.csv")
    write_table(f1_summary, tables_dir / "macro_f1_summary_by_model.csv")
    write_table(runtime_summary, tables_dir / "runtime_summary_by_model.csv")
    write_table(
        adjusted_runtime_summary,
        tables_dir / "parallel_adjusted_runtime_summary_by_model.csv",
    )
    write_table(
        software_model_f1_summary,
        tables_dir / "macro_f1_summary_by_software_model.csv",
    )
    write_table(
        software_model_runtime_summary,
        tables_dir / "runtime_summary_by_software_model.csv",
    )
    write_table(
        software_model_adjusted_runtime_summary,
        tables_dir / "parallel_adjusted_runtime_summary_by_software_model.csv",
    )
    write_table(class_f1_summary, tables_dir / "class_f1_summary_by_model.csv")
    write_table(
        class_f1_software_model_summary,
        tables_dir / "class_f1_summary_by_software_model.csv",
    )
    write_table(
        class_f1_strategy_model_summary,
        tables_dir / "class_f1_summary_by_strategy_model.csv",
    )
    write_table(
        mcc_strategy_model_summary,
        tables_dir / "mcc_summary_by_strategy_model.csv",
    )
    write_table(
        cpu_gpu_runtime_delta,
        tables_dir / "cpu_gpu_runtime_delta_by_deep_model.csv",
    )
    write_table(
        f1_balance_summary,
        tables_dir / "f1_class_balance_summary_by_model.csv",
    )
    write_table(
        category_summary,
        tables_dir / "model_family_summary.csv",
    )
    write_table(
        head_to_head_summary,
        tables_dir / "pgsui_vs_gtimputation_head_to_head.csv",
    )
    write_table(
        strategy_sensitivity,
        tables_dir / "pgsui_vs_gtimputation_strategy_sensitivity_mixed_model.csv",
    )
    write_table(
        dataset_leaderboard,
        tables_dir / "dataset_leaderboard.csv",
    )
    write_table(
        class_f1_wide,
        tables_dir / "class_f1_wide_by_model.csv",
    )
    write_table(
        dataset_size_performance,
        tables_dir / "dataset_size_performance_summary.csv",
    )
    write_table(
        f1_size_regression,
        tables_dir / "f1_size_regression_summary.csv",
    )
    write_table(
        f1_genotype_count_regression,
        tables_dir / "f1_genotype_count_regression_summary.csv",
    )
    write_table(
        runtime_genotype_count_regression,
        tables_dir / "runtime_genotype_count_regression_summary.csv",
    )
    write_table(
        dataset_size_f1_runtime,
        tables_dir / "dataset_size_f1_runtime_comparison.csv",
    )
    write_table_view_index(tables_dir)

    if not args.no_plots and not combined.empty:
        plot_all(combined, combined_class_report, runtime_combined, plots_dir, args.dpi)

    write_readme(
        output_dir,
        pgsui_dir=pgsui_dir,
        gti_dir=gti_dir,
        sim_manifest=sim_manifest_path,
        pgsui_status=pgsui_status,
        selected_pgsui=selected_pgsui,
        selected_pgsui_runtime=selected_pgsui_runtime,
        retained_keys=retained_keys,
        combined=combined,
        runtime_combined=runtime_combined,
        best_summary_df=best_summary_df,
        gti_errors=gti_errors,
        allow_partial_grid=args.allow_partial_grid,
    )
    print(f"Wrote comparison tables and plots to: {output_dir}")


if __name__ == "__main__":
    main()
