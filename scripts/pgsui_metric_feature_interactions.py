#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PG-SUI metric ↔ population-genetic feature interaction analysis.

Inputs
------
1) --stats   : per-dataset summary stats CSV (e.g., *_all_datasets_summary_stats.csv). Typically contains columns like: dataset, Pi_mean, TajimaD_mean, Missingness_mean, etc.
2) --metrics : long-form zygosity metrics CSV from collect_zygosity_metrics.py. Must contain at least: dataset, model, sim_strategy, metric, value

Outputs
-------
- spearman_long.csv: feature ↔ performance Spearman correlations per (model, strategy)
- adjusted_permutation_ols_long.csv (optional): adjusted regression results with permutation p-values and FDR correction
- Heatmaps (feature x performance-metric) per model x strategy
- Driver ranking plot: mean absolute standardized adjusted coefficient per feature aggregated across conditions
- Metric↔Metric correlation heatmaps per model x strategy.
- If --make-adjusted-permutation is set, the script runs covariate-adjusted OLS models:
        target ~ feature + controls
    and evaluates feature significance using Freedman-Lane permutation tests.
  Default controls are Missingness, Sample_Size, and Locus_Count.
- If --make-no-control-permutation is set with the no-control supplement, the
  script runs a separate no-control permutation analysis:
        target ~ feature
  using an intercept-only reduced model. These results are cached separately
  from the adjusted permutation results and can be reloaded with
  --no-control-load-permutations.

Notes
-----
- Joins stats↔metrics using a "join_key" extracted from dataset strings (results\\d+).
- Performance targets include: REF/HET/ALT x {F1score, Precision, Recall, Jaccard, AveragePrecision} + MCC (Overall).
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import transforms
from scipy.stats import spearmanr
from tqdm import tqdm

if TYPE_CHECKING:
    import matplotlib.axes as mpl_ax
    from pandas.api.typing import NAType

# Non-interactive backend for HPC/server environments without display
# Set before any plotting commands.
plt.switch_backend("Agg")

FONTSIZE = 9
TITLE_FONTSIZE = 9
AXIS_LABEL_FONTSIZE = 9
TICK_FONTSIZE = 9
ANNOT_FONTSIZE = 9
CBAR_FONTSIZE = 11
MODEL_GROUP_FONTSIZE = 10
STRATEGY_TICK_FONTSIZE = 11
VERTICAL_FONTSIZE = 48

MANUSCRIPT_HEATMAP_WIDTH = 6.5
MANUSCRIPT_HEATMAP_HEIGHT = 3.5
DIFFERENCE_AXIS_TICKS = [
    -1.0,
    -0.8,
    -0.6,
    -0.4,
    -0.2,
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
]
DIFFERENCE_AXIS_TICKLABELS = [
    "-1.0",
    "-0.8",
    "-0.6",
    "-0.4",
    "-0.2",
    "0",
    "0.2",
    "0.4",
    "0.6",
    "0.8",
    "1.0",
]

param_dict = {
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "axes.titlesize": TITLE_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": FONTSIZE,
    "legend.title_fontsize": FONTSIZE,
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FONTSIZE,
    "lines.linewidth": 0.8,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

mpl.rcParams.update(param_dict)


_RESULTS_KEY_RE = re.compile(r"(results\d+)", flags=re.IGNORECASE)

# Keep aligned with naming scheme.
FEATURE_CANONICAL = [
    "F_inbreeding",
    "ThetaWatt",
    "Pi",
    "TajimaD",
    "Missingness",
    "Locus_Count",
    "Ho",
    "He",
    "SegSites",
    "Singletons",
    "Hap",
    "Hd",
    "Sample_Size",
    "MAF",
]

FEATURE_DISPLAY_LABELS: dict[Any | NAType, Any] = {
    "F_inbreeding": r"$\it{F}_{\it{is}}$",
    "ThetaWatt": r"$\it{\theta}_{\it{w}}$",
    "Pi": r"$\it{\pi}$",
    "TajimaD": r"Tajima's $\it{D}$",
    "Missingness": "Missingness",
    "Locus_Count": "Locus Count",
    "Ho": r"$\it{H}_{\it{o}}$",
    "He": r"$\it{H}_{\it{e}}$",
    "SegSites": r"$\it{S}$",
    "Singletons": "Singletons",
    "Hap": r"$\it{N}_{\it{hap}}$",
    "Hd": r"$\it{H}_{\it{d}}$",
    "Sample_Size": "Sample Size",
    "MAF": "MAF",
}

GENOTYPES = ("REF", "HET", "ALT")
GENOTYPE_DISPLAY_LABELS = {
    "REF": "Homozygous Reference",
    "HET": "Heterozygous",
    "ALT": "Homozygous Alternate",
    "Overall": "Overall",
}

# Normalize metric names coming from your JSON leaf paths
METRIC_NAME_MAP = {
    "f1-score": "F1score",
    "f1_score": "F1score",
    "f1score": "F1score",
    "precision": "Precision",
    "recall": "Recall",
    "support": "Support",
    "jaccard": "Jaccard",
    "average-precision": "AveragePrecision",
    "average_precision": "AveragePrecision",
    "averageprecision": "AveragePrecision",
    "mcc": "MCC",
}

# Performance metrics we will analyze (class metrics become {GENO}_{METRIC})
CLASS_TARGETS = ("F1score", "Precision", "Recall", "Jaccard", "AveragePrecision")
GLOBAL_TARGETS = ("MCC",)

LOCUS_COUNT_SOURCE_COLUMNS = (
    "Locus_Count",
    "locus_count",
    "Sequence_Length",
    "sequence_length",
    "n_loci",
    "num_loci",
    "Sample_Size_count",
    "SegSites_count",
    "MAF_count",
)

DEFAULT_CONTROLS = ("Missingness", "Sample_Size", "Locus_Count")

ORDER_MODELS = [
    "ImputeRefAllele",
    "ImputeMostFrequent",
    "ImputeAutoencoder",
    "ImputeVAE",
    "ImputeNLPCA",
    "ImputeUBP",
]

DETERMINISTIC_MODELS = ("ImputeMostFrequent",)
DEEP_LEARNING_MODELS = (
    "ImputeAutoencoder",
    "ImputeVAE",
    "ImputeNLPCA",
    "ImputeUBP",
)

MODEL_DISPLAY_LABELS = {
    "ImputeRefAllele": "RefAllele",
    "ImputeMostFrequent": "MostFrequent",
    "ImputeAutoencoder": "Autoencoder",
    "ImputeVAE": "VAE",
    "ImputeNLPCA": "NLPCA",
    "ImputeUBP": "UBP",
}

MODEL_GROUP_MULTILINE_LABELS = {
    "MostFrequent": "Most\nFrequent",
    "Autoencoder": "Auto\nencoder",
}


ORDER_STRATEGIES = [
    "Random",
    "Random Weighted",
    "Random Weighted Inv",
    "Nonrandom",
    "Nonrandom Weighted",
]

STRATEGY_DISPLAY_LABELS = {
    "Random": "R",
    "Random Weighted": "RW",
    "Random Weighted Inv": "RWI",
    "Nonrandom": "N",
    "Nonrandom Weighted": "NW",
}

METRIC_LABELS = {
    "F1score": "F1-score",
    "MCC": "MCC",
    "REF_F1score": "Homozygous Reference F1-score",
    "HET_F1score": "Heterozygous F1-score",
    "ALT_F1score": "Homozygous Alternate F1-score",
}

FEATURE_DISPLAY_MAP = FEATURE_DISPLAY_LABELS.copy()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(
        description="Compute and plot interactions between PG-SUI performance metrics and popgen summary stats."
    )
    p.add_argument(
        "--stats", type=Path, required=True, help="Per-dataset summary stats CSV."
    )
    p.add_argument(
        "--metrics", type=Path, required=True, help="Long-form zygosity metrics CSV."
    )
    p.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    p.add_argument(
        "--controls",
        nargs="*",
        default=list(DEFAULT_CONTROLS),
        help=(
            "Control variables for adjusted analyses. Defaults to "
            "Missingness Sample_Size Locus_Count; use --controls with no values to disable."
        ),
    )
    p.add_argument(
        "--make-partial",
        action="store_true",
        help=(
            "Deprecated no-op retained for command compatibility. The adjusted "
            "OLS workflow always reports covariate-adjusted partial R2; it does "
            "not compute partial correlations."
        ),
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=8,
        help="Minimum number of datasets required to compute a correlation in a condition (default: 8).",
    )

    p.add_argument(
        "--make-adjusted-permutation",
        action="store_true",
        help="If set, run adjusted OLS with Freedman-Lane permutation tests.",
    )
    p.add_argument(
        "--n-permutations",
        type=int,
        default=9999,
        help="Number of permutations for adjusted regression tests (default: 9999).",
    )
    p.add_argument(
        "--fdr-scope",
        choices=("global", "by_target", "by_condition"),
        default="global",
        help="Scope for BH-FDR correction of permutation p-values.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for permutation testing.",
    )
    p.add_argument(
        "--load-permutations",
        action="store_true",
        help="If set, skip permutation tests and load existing results from CSV.",
    )
    p.add_argument(
        "--driver-bootstrap-reps",
        type=int,
        default=500,
        help=(
            "Dataset-level bootstrap replicates for driver-summary uncertainty. "
            "Use a smaller value while testing and a larger value, e.g. 10000, "
            "for archived manuscript outputs."
        ),
    )
    p.add_argument(
        "--driver-bootstrap-mode",
        choices=("dataset_refit", "effect_row", "none"),
        default="dataset_refit",
        help=(
            "Uncertainty for driver-summary plots. dataset_refit resamples "
            "join_key datasets jointly and refits OLS models; effect_row keeps "
            "the legacy descriptive row-resampling intervals; none suppresses "
            "driver-summary intervals."
        ),
    )
    p.add_argument(
        "--driver-bootstrap-n-jobs",
        type=int,
        default=1,
        help=(
            "Parallel workers for dataset-refit driver bootstrapping. Use 1 for "
            "serial execution, a positive integer for that many workers, or -1 "
            "for all available CPU cores."
        ),
    )
    p.add_argument(
        "--driver-bootstrap-backend",
        choices=("process", "thread", "serial"),
        default="process",
        help=(
            "Parallel backend for dataset-refit driver bootstrapping. Process "
            "uses separate Python workers; thread avoids process startup and "
            "data-copy overhead; serial forces one worker."
        ),
    )
    p.add_argument(
        "--no-driver-bootstrap-progress",
        action="store_true",
        help="Disable the tqdm progress bar for dataset-refit driver bootstrapping.",
    )
    p.add_argument(
        "--make-no-control-supplement",
        action="store_true",
        help=(
            "Generate supplementary no-control OLS driver-ranking plots with "
            "unique filenames."
        ),
    )
    p.add_argument(
        "--only-no-control-supplement",
        action="store_true",
        help=(
            "Only generate supplementary no-control OLS driver-ranking plots; "
            "skip standard outputs so existing figures are not overwritten."
        ),
    )
    p.add_argument(
        "--make-no-control-permutation",
        action="store_true",
        help=(
            "For no-control supplemental plots, compute/load separate "
            "Freedman-Lane permutation p-values for target ~ feature. "
            "Uses --n-permutations, --random-seed, --fdr-scope, and "
            "--no-control-load-permutations; writes "
            "no_control_permutation_ols_long.csv."
        ),
    )
    p.add_argument(
        "--no-control-load-permutations",
        action="store_true",
        help=(
            "When generating no-control permutation outputs, skip no-control "
            "permutation tests and reload no_control_permutation_ols_long.csv. "
            "This is separate from the adjusted-analysis --load-permutations flag."
        ),
    )
    p.add_argument(
        "--make-deep-baseline-difference",
        action="store_true",
        help=(
            "Generate adjusted |Bstd| difference plots comparing deep learning "
            "models against the MostFrequent baseline."
        ),
    )
    p.add_argument(
        "--only-deep-baseline-difference",
        action="store_true",
        help=(
            "Only generate adjusted |Bstd| difference plots from the cached "
            "adjusted_permutation_ols_long.csv file."
        ),
    )
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300).")
    return p.parse_args()


def save_figure_dual(fig, out_path: Path, *, dpi: int = 300) -> None:
    """Save a figure to both PNG and PDF using out_path as the base.

    Args:
        fig: Matplotlib Figure.
        out_path: Output path (may be .png or any suffix). We will write:
            - <stem>.png
            - <stem>.pdf
        dpi: Raster dpi for PNG.
    """
    out_path = Path(out_path)
    png_path = out_path.with_suffix(".png")
    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")


def _split_condition_label(label: str) -> Tuple[str, str]:
    """Split a combined 'model | strategy' condition label.

    Args:
        label (str): Combined condition label.

    Returns:
        Tuple[str, str]: (model, strategy)
    """
    parts = re.split(r"\s*\|\s*", str(label), maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return str(label), ""


def _condition_sort_key(label: str) -> Tuple[int, int]:
    """Sort key for combined condition labels.

    Args:
        label (str): Combined 'model | strategy' label.

    Returns:
        Tuple[int, int]: Sort key by model then strategy.
    """
    model, strategy = _split_condition_label(label)
    m_idx = ORDER_MODELS.index(model) if model in ORDER_MODELS else 999
    s_idx = ORDER_STRATEGIES.index(strategy) if strategy in ORDER_STRATEGIES else 999
    return (m_idx, s_idx)


def _drop_control_feature_rows(df: pd.DataFrame, controls: List[str]) -> pd.DataFrame:
    """Remove control covariates from feature-effect result rows."""
    if df.empty or "feature" not in df.columns or not controls:
        return df

    return df.loc[~df["feature"].isin(controls)].copy()


def _add_model_grouping_labels(
    ax: "mpl_ax.Axes",
    sorted_cols: List[str],
    *,
    label_y_pos: float = -0.22,
    model_text_y_offset: float = 0.06,
    strategy_fontsize: str | int = STRATEGY_TICK_FONTSIZE,
    model_fontsize: str | int = MODEL_GROUP_FONTSIZE,
    line_width: float = 0.8,
    wrap_model_labels: bool = False,
) -> None:
    """Add grouped model labels under a heatmap with model|strategy columns.

    Args:
        ax (mpl_ax.Axes): Heatmap axes.
        sorted_cols (List[str]): Ordered condition labels.
        label_y_pos (float): Y position in axes coordinates for group bars.
            More negative values move the grouping band farther below the axis.
        model_text_y_offset (float): Extra downward offset for model names
            relative to the grouping line.
        strategy_fontsize (str | int): Font size for strategy tick labels.
        model_fontsize (str | int): Font size for grouped model labels.
        line_width (float): Line width for grouping bars and separators.
        wrap_model_labels (bool): Whether to wrap long model labels onto two
            lines.
    """
    ax.set_xlabel("")
    ax.set_ylabel("Summary Statistic")

    new_xticklabels = []
    for lbl in ax.get_xticklabels():
        raw = lbl.get_text()
        strategy = raw.split(" | ")[1] if " | " in raw else raw
        new_xticklabels.append(STRATEGY_DISPLAY_LABELS.get(strategy, strategy))

    ax.set_xticklabels(
        new_xticklabels,
        rotation=90,
        ha="center",
        va="top",
        fontsize=strategy_fontsize,
    )
    ax.tick_params(axis="x", pad=2)

    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    current_model = None
    group_start_idx = 0
    line_y = label_y_pos
    text_y = line_y - model_text_y_offset
    tick_len = 0.018

    for i, col_name in enumerate(sorted_cols):
        model = col_name.split(" | ")[0]

        if current_model is None:
            current_model = model
            continue

        if model != current_model:
            # separator through heatmap
            ax.axvline(
                i, color="k", lw=line_width, ls="-", ymin=0.0, ymax=1.0, clip_on=False
            )

            # dashed separator through lower label band
            ax.plot(
                [i, i],
                [0.0, line_y + 0.05],
                transform=trans,
                color="k",
                lw=line_width,
                ls="--",
                clip_on=False,
            )

            center = (group_start_idx + (i - 1)) / 2.0
            model_label = MODEL_DISPLAY_LABELS.get(
                current_model, current_model.replace("Impute", "")
            )
            if wrap_model_labels:
                model_label = MODEL_GROUP_MULTILINE_LABELS.get(model_label, model_label)
            ax.text(
                center + 0.5,
                text_y,
                model_label,
                transform=trans,
                ha="center",
                va="top",
                fontsize=model_fontsize,
                fontweight="bold",
                linespacing=0.9,
            )

            # horizontal grouping bar
            ax.plot(
                [group_start_idx + 0.2, i - 0.2],
                [line_y, line_y],
                transform=trans,
                color="k",
                lw=line_width,
                clip_on=False,
            )
            # left end cap
            ax.plot(
                [group_start_idx + 0.2, group_start_idx + 0.2],
                [line_y - tick_len, line_y + tick_len],
                transform=trans,
                color="k",
                lw=line_width,
                clip_on=False,
            )
            # right end cap
            ax.plot(
                [i - 0.2, i - 0.2],
                [line_y - tick_len, line_y + tick_len],
                transform=trans,
                color="k",
                lw=line_width,
                clip_on=False,
            )

            group_start_idx = i
            current_model = model

    if current_model is None:
        return

    n = len(sorted_cols)
    center = (group_start_idx + (n - 1)) / 2.0
    model_label = MODEL_DISPLAY_LABELS.get(
        current_model, current_model.replace("Impute", "")
    )
    if wrap_model_labels:
        model_label = MODEL_GROUP_MULTILINE_LABELS.get(model_label, model_label)
    ax.text(
        center + 0.5,
        text_y,
        model_label,
        transform=trans,
        ha="center",
        va="top",
        fontsize=model_fontsize,
        fontweight="bold",
        linespacing=0.9,
    )
    ax.plot(
        [group_start_idx + 0.2, n - 0.2],
        [line_y, line_y],
        transform=trans,
        color="k",
        lw=line_width,
        clip_on=False,
    )
    ax.plot(
        [group_start_idx + 0.2, group_start_idx + 0.2],
        [line_y - tick_len, line_y + tick_len],
        transform=trans,
        color="k",
        lw=line_width,
        clip_on=False,
    )
    ax.plot(
        [n - 0.2, n - 0.2],
        [line_y - tick_len, line_y + tick_len],
        transform=trans,
        color="k",
        lw=line_width,
        clip_on=False,
    )


def format_target_label(target: str) -> str:
    """Format a target family label for plotting.

    Args:
        target (str): Target family label.

    Returns:
        str: Nicely formatted target label.
    """
    if "_" in target:
        genotype, metric = target.split("_", 1)
        genotype_label = GENOTYPE_DISPLAY_LABELS.get(genotype, genotype)
        metric_label = METRIC_LABELS.get(metric, metric)
        return f"{genotype_label} {metric_label}"
    return METRIC_LABELS.get(target, target)


def _standardize_series(x: np.ndarray) -> np.ndarray:
    """Z-standardize a numeric vector.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        np.ndarray: Standardized vector.

    Raises:
        ValueError: If x is constant or invalid.
    """
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sd = np.std(x, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        raise ValueError("Cannot standardize a constant or invalid vector.")
    return (x - mu) / sd


def _compute_partial_r2(
    y: np.ndarray,
    x_feat: np.ndarray,
    X_ctrl: np.ndarray,
) -> float:
    """Compute partial R^2 for feature given controls.

    Args:
        y (np.ndarray): Response vector, shape (n,).
        x_feat (np.ndarray): Feature vector, shape (n,).
        X_ctrl (np.ndarray): Controls matrix, shape (n, p).

    Returns:
        float: Partial R^2.
    """
    y = np.asarray(y, dtype=float)
    x_feat = np.asarray(x_feat, dtype=float)
    X_ctrl = np.asarray(X_ctrl, dtype=float)
    if X_ctrl.ndim == 1:
        X_ctrl = X_ctrl.reshape(-1, 1)
    if X_ctrl.size == 0:
        X_ctrl = np.empty((y.shape[0], 0), dtype=float)

    X_red = (
        sm.add_constant(X_ctrl, has_constant="add")
        if X_ctrl.shape[1]
        else np.ones((y.shape[0], 1), dtype=float)
    )
    X_full_raw = (
        np.column_stack([x_feat, X_ctrl]) if X_ctrl.shape[1] else x_feat.reshape(-1, 1)
    )
    X_full = sm.add_constant(X_full_raw, has_constant="add")

    red_fit = sm.OLS(y, X_red).fit()
    full_fit = sm.OLS(y, X_full).fit()

    sse_red = float(np.sum(red_fit.resid**2))
    sse_full = float(np.sum(full_fit.resid**2))

    if not np.isfinite(sse_red) or sse_red <= 0:
        return np.nan

    return float((sse_red - sse_full) / sse_red)


def _freedman_lane_perm_test(
    y: np.ndarray,
    x_feat: np.ndarray,
    X_ctrl: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Run a Freedman-Lane permutation test for the feature effect.

    Tests the null that the feature has no effect after conditioning on controls.

    Args:
        y (np.ndarray): Response vector, shape (n,).
        x_feat (np.ndarray): Feature vector, shape (n,).
        X_ctrl (np.ndarray): Controls matrix, shape (n, p).
        n_permutations (int): Number of permutations.
        rng (np.random.Generator): NumPy random generator.

    Returns:
        Dict[str, float]: Observed statistics and permutation p-value.
    """
    y = np.asarray(y, dtype=float)
    x_feat = np.asarray(x_feat, dtype=float)
    X_ctrl = np.asarray(X_ctrl, dtype=float)
    if X_ctrl.ndim == 1:
        X_ctrl = X_ctrl.reshape(-1, 1)
    if X_ctrl.size == 0:
        X_ctrl = np.empty((y.shape[0], 0), dtype=float)

    X_red = (
        sm.add_constant(X_ctrl, has_constant="add")
        if X_ctrl.shape[1]
        else np.ones((y.shape[0], 1), dtype=float)
    )
    X_full_raw = (
        np.column_stack([x_feat, X_ctrl]) if X_ctrl.shape[1] else x_feat.reshape(-1, 1)
    )
    X_full = sm.add_constant(X_full_raw, has_constant="add")

    red_fit = sm.OLS(y, X_red).fit()
    full_fit = sm.OLS(y, X_full).fit()

    obs_coef = float(full_fit.params[1])
    obs_t = float(full_fit.tvalues[1])

    yhat_red = red_fit.fittedvalues
    resid_red = red_fit.resid

    perm_abs_t = np.empty(n_permutations, dtype=float)

    for i in range(n_permutations):
        perm_idx = rng.permutation(resid_red.shape[0])
        y_star = yhat_red + resid_red[perm_idx]
        fit_star = sm.OLS(y_star, X_full).fit()
        perm_abs_t[i] = abs(float(fit_star.tvalues[1]))

    perm_p = (1.0 + np.sum(perm_abs_t >= abs(obs_t))) / (n_permutations + 1.0)

    return {
        "obs_coef": obs_coef,
        "obs_t": obs_t,
        "perm_pvalue": float(perm_p),
    }


def _add_fdr(
    df: pd.DataFrame,
    *,
    p_col: str,
    scope: str,
) -> pd.DataFrame:
    """Add BH-FDR q-values to a results table.

    Args:
        df (pd.DataFrame): Results table.
        p_col (str): Column containing raw p-values.
        scope (str): One of {"global", "by_target", "by_condition"}.

    Returns:
        pd.DataFrame: Copy of df with qvalue_bh column added.
    """
    out = df.copy()
    out["qvalue_bh"] = np.nan

    if out.empty:
        return out

    if scope == "global":
        mask = out[p_col].notna()
        if mask.any():
            _, qvals, _, _ = sm.stats.multipletests(
                out.loc[mask, p_col], method="fdr_bh"
            )
            out.loc[mask, "qvalue_bh"] = qvals
        return out

    if scope == "by_target":
        group_cols = ["target"]
    elif scope == "by_condition":
        group_cols = ["model", "sim_strategy"]
    else:
        raise ValueError(f"Unsupported FDR scope: {scope}")

    for _, idx in out.groupby(group_cols).groups.items():
        idx = list(idx)
        sub = out.loc[idx, p_col]
        mask = sub.notna()
        if mask.any():
            _, qvals, _, _ = sm.stats.multipletests(sub.loc[mask], method="fdr_bh")
            out.loc[sub.loc[mask].index, "qvalue_bh"] = qvals

    return out


def _sig_label_from_q(q: float) -> str:
    """Return significance stars from q-value.

    Args:
        q (float): FDR-adjusted q-value.

    Returns:
        str: Significance label.
    """
    q = float(q)

    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def _extract_join_key(val: str) -> Optional[str]:
    """Extract results\\d+ join key (lowercased) from an identifier.

    Args:
        val (str): Dataset identifier.

    Returns:
        Optional[str]: Join key like "results52" if found else None.
    """
    if not isinstance(val, str):
        return None
    m = _RESULTS_KEY_RE.search(val)
    return m.group(1).lower() if m else None


def _canonicalize_strategy(val: str) -> str:
    """Canonicalize simulation strategy labels to your display names.

    Args:
        val (str): Strategy string.

    Returns:
        str: Canonical strategy label.
    """
    s = re.sub(r"[^a-z0-9]+", "", str(val).lower())
    if "nonrandom" in s and "weighted" in s:
        return "Nonrandom Weighted"
    if "nonrandom" in s:
        return "Nonrandom"
    if "random" in s and "weighted" in s and ("inv" in s or "inverse" in s):
        return "Random Weighted Inv"
    if "random" in s and "weighted" in s:
        return "Random Weighted"
    if "random" in s:
        return "Random"
    return "Unknown"


def normalize_metric_name(raw: str) -> str:
    """Normalize a metric token.

    Args:
        raw (str): Raw metric name token.

    Returns:
        str: Canonical metric name.
    """
    norm = re.sub(r"\s+", "", str(raw).strip().lower()).replace("_", "-")
    return METRIC_NAME_MAP.get(norm, raw)


def load_stats(stats_path: Path) -> pd.DataFrame:
    """Load dataset stats and standardize feature columns.

    Accepts either *_mean columns or raw columns.

    Args:
        stats_path (Path): Stats CSV.

    Returns:
        pd.DataFrame: DataFrame with columns: dataset, join_key, <FEATURES>.
    """
    df = pd.read_csv(stats_path)
    if "dataset" not in df.columns:
        raise ValueError("Stats CSV must contain a 'dataset' column.")

    out = pd.DataFrame({"dataset": df["dataset"].astype(str)})
    out["join_key"] = out["dataset"].apply(_extract_join_key)

    for feat in FEATURE_CANONICAL:
        if feat == "Locus_Count":
            source_col = next(
                (col for col in LOCUS_COUNT_SOURCE_COLUMNS if col in df.columns),
                None,
            )
            out[feat] = (
                pd.to_numeric(df[source_col], errors="coerce")
                if source_col is not None
                else np.nan
            )
            continue

        mean_col = f"{feat}_mean"
        if mean_col in df.columns:
            out[feat] = pd.to_numeric(df[mean_col], errors="coerce")
        elif feat in df.columns:
            out[feat] = pd.to_numeric(df[feat], errors="coerce")
        else:
            out[feat] = np.nan

    return out


def load_metrics_long(metrics_path: Path) -> pd.DataFrame:
    """Load your long-form zygosity metric table and build performance columns.

    Expected columns include:
        - dataset, model, sim_strategy, metric, value
    where metric is a JSON leaf path like "REF.precision" or "mcc".

    Args:
        metrics_path (Path): Metrics CSV from collect_zygosity_metrics.py.

    Returns:
        pd.DataFrame: Tidy table: join_key, model, sim_strategy, <targets...>
    """
    df = pd.read_csv(metrics_path)

    required = {"dataset", "model", "sim_strategy", "metric", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metrics CSV missing required columns: {missing}")

    df = df.copy()
    df["join_key"] = df["dataset"].astype(str).apply(_extract_join_key)
    df["sim_strategy"] = df["sim_strategy"].apply(_canonicalize_strategy)
    df["model"] = df["model"].astype(str)

    # Parse metric into genotype + measure when possible
    # Example: "REF.precision" -> geno=REF, measure=Precision
    geno = []
    meas = []
    for m in df["metric"].astype(str).tolist():
        if "." in m:
            g, rest = m.split(".", 1)
            geno.append(g)
            meas.append(normalize_metric_name(rest))
        else:
            geno.append("Overall")
            meas.append(normalize_metric_name(m))
    df["geno"] = geno
    df["measure"] = meas
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Keep only targets we care about
    keep_rows = (df["geno"].isin(GENOTYPES) & df["measure"].isin(CLASS_TARGETS)) | (
        (df["geno"] == "Overall") & (df["measure"].isin(GLOBAL_TARGETS))
    )
    df = df.loc[keep_rows].copy()

    # Build wide columns like REF_F1score, HET_AveragePrecision, etc., plus MCC
    df["target"] = np.where(
        df["geno"].isin(GENOTYPES),
        df["geno"].astype(str) + "_" + df["measure"].astype(str),
        df["measure"].astype(str),
    )

    # Collapse duplicates robustly: take first non-null (your pipeline should already be unique)
    wide = (
        df.pivot_table(
            index=["join_key", "model", "sim_strategy"],
            columns="target",
            values="value",
            aggfunc=lambda x: x.dropna().iloc[0] if x.dropna().shape[0] else np.nan,
            observed=False,
        )
        .reset_index()
        .copy()
    )

    return wide


def spearman_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute Spearman correlation with NaN safety.

    Args:
        x (np.ndarray): Vector.
        y (np.ndarray): Vector.

    Returns:
        Tuple[float, float]: (rho, pvalue)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return (np.nan, np.nan)

    # If either vector is constant on the valid rows, Spearman is undefined.
    # Use a neutral effect for plotting/table completeness. This keeps
    # structurally constant baselines such as ImputeRefAllele HET/ALT F1score
    # visible instead of dropping or masking their cells.
    if np.nanmin(x[mask]) == np.nanmax(x[mask]) or np.nanmin(y[mask]) == np.nanmax(
        y[mask]
    ):
        return (0.0, 1.0)

    rho, p = spearmanr(x[mask], y[mask])
    if rho is None or p is None:
        return (np.nan, np.nan)

    # Be defensive: some edge cases can yield nan
    if not np.isfinite(rho) or not np.isfinite(p):  # type: ignore
        return (np.nan, np.nan)

    return (float(rho), float(p))  # type: ignore


def compute_feature_target_correlations(
    merged: pd.DataFrame,
    features: List[str],
    targets: List[str],
    *,
    min_n: int,
) -> pd.DataFrame:
    """Compute Spearman feature↔target correlations per (model, strategy).

    Args:
        merged (pd.DataFrame): Contains columns: model, sim_strategy, join_key, <features>, <targets>.
        features (List[str]): Feature column names.
        targets (List[str]): Target column names.
        min_n (int): Minimum n required in condition.

    Returns:
        pd.DataFrame: Long table with rho/p per row.
    """
    rows: List[Dict[str, object]] = []
    for (model, strat), sub in merged.groupby(["model", "sim_strategy"], dropna=False):
        # Count unique datasets contributing
        n_ds = sub["join_key"].nunique()
        if n_ds < min_n:
            continue

        for feat in features:
            x = pd.to_numeric(sub[feat], errors="coerce").to_numpy(dtype=float)
            for targ in targets:
                y = pd.to_numeric(sub[targ], errors="coerce").to_numpy(dtype=float)
                rho, p = spearman_corr(x, y)
                rows.append(
                    {
                        "model": str(model),
                        "sim_strategy": str(strat),
                        "feature": feat,
                        "target": targ,
                        "rho": rho,
                        "pvalue": p,
                        "n_datasets": int(n_ds),
                    }
                )
    return pd.DataFrame(rows)


def compute_adjusted_permutation_ols(
    merged: pd.DataFrame,
    features: List[str],
    targets: List[str],
    controls: List[str],
    *,
    min_n: int,
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    """Compute adjusted OLS effects with Freedman-Lane permutation tests.

    For each (model, sim_strategy, feature, target), fit:
        target ~ feature + controls

    Uses:
        - adjusted coefficient
        - standardized adjusted coefficient
        - partial R^2
        - permutation p-value for feature effect

    Args:
        merged (pd.DataFrame): Merged performance + stats table.
        features (List[str]): Candidate feature columns.
        targets (List[str]): Performance target columns.
        controls (List[str]): Control variables.
        min_n (int): Minimum number of datasets.
        n_permutations (int): Number of permutations.
        random_seed (int): Base random seed.

    Returns:
        pd.DataFrame: Long-form adjusted regression results.
    """
    rows: List[Dict[str, object]] = []

    feature_set = [f for f in features if f not in controls]
    base_rng = np.random.default_rng(random_seed)

    for (model, strat), sub in merged.groupby(["model", "sim_strategy"], dropna=False):
        n_ds_total = sub["join_key"].nunique()
        if n_ds_total < min_n:
            continue

        ctrl_df = sub[controls].apply(pd.to_numeric, errors="coerce")
        keep_ctrl = ctrl_df.columns[ctrl_df.notna().any(axis=0)].tolist()

        if not keep_ctrl:
            continue

        Xc_all = ctrl_df[keep_ctrl]

        for feat in feature_set:
            x_raw = pd.to_numeric(sub[feat], errors="coerce")

            for targ in targets:
                y_raw = pd.to_numeric(sub[targ], errors="coerce")

                dat = pd.DataFrame(
                    {
                        "join_key": sub["join_key"].astype(str),
                        "feature": x_raw,
                        "target": y_raw,
                    }
                )
                for c in keep_ctrl:
                    dat[c] = Xc_all[c]

                dat = dat.dropna(axis=0).copy()
                n_used = dat["join_key"].nunique()
                if n_used < min_n:
                    continue

                if dat["feature"].nunique(dropna=True) < 2:
                    continue
                if dat["target"].nunique(dropna=True) < 2:
                    continue

                y = dat["target"].to_numpy(dtype=float)
                x = dat["feature"].to_numpy(dtype=float)
                Xc = dat[keep_ctrl].to_numpy(dtype=float)

                X_full = sm.add_constant(np.column_stack([x, Xc]), has_constant="add")
                fit = sm.OLS(y, X_full).fit(cov_type="HC3")

                coef = float(fit.params[1])
                se = float(fit.bse[1])
                tval = float(fit.tvalues[1])
                pval_hc3 = float(fit.pvalues[1])

                try:
                    y_std = _standardize_series(y)
                    x_std = _standardize_series(x)
                    Xc_std = np.column_stack(
                        [_standardize_series(Xc[:, j]) for j in range(Xc.shape[1])]
                    )
                    X_std_full = sm.add_constant(
                        np.column_stack([x_std, Xc_std]),
                        has_constant="add",
                    )
                    fit_std = sm.OLS(y_std, X_std_full).fit(cov_type="HC3")
                    std_coef = float(fit_std.params[1])
                except ValueError:
                    std_coef = np.nan

                partial_r2 = _compute_partial_r2(y=y, x_feat=x, X_ctrl=Xc)

                test_rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
                perm_res = _freedman_lane_perm_test(
                    y=y,
                    x_feat=x,
                    X_ctrl=Xc,
                    n_permutations=n_permutations,
                    rng=test_rng,
                )

                rows.append(
                    {
                        "model": str(model),
                        "sim_strategy": str(strat),
                        "feature": feat,
                        "target": targ,
                        "coef": coef,
                        "std_coef": std_coef,
                        "se_hc3": se,
                        "tvalue_hc3": tval,
                        "pvalue_hc3": pval_hc3,
                        "partial_r2": partial_r2,
                        "perm_tvalue_obs": perm_res["obs_t"],
                        "perm_pvalue": perm_res["perm_pvalue"],
                        "n_datasets": int(n_used),
                        "n_permutations": int(n_permutations),
                        "controls": ",".join(keep_ctrl),
                    }
                )

    return pd.DataFrame(rows)


def compute_refit_ols_effects(
    merged: pd.DataFrame,
    features: List[str],
    targets: List[str],
    controls: List[str],
    *,
    min_n: int,
    count_unique_datasets: bool = True,
    ols_mode: str = "adjusted_refit",
) -> pd.DataFrame:
    """Refit OLS effects without permutation tests.

    This is used for dataset-level bootstrap replicates. It follows the same
    fitted-model surface as the adjusted/no-control OLS workflows, but skips
    Freedman-Lane permutations because permutation inference is not nested
    inside the dataset bootstrap.
    """
    rows: List[Dict[str, object]] = []
    features = [feature for feature in features if feature in merged.columns]
    controls = [control for control in controls if control in merged.columns]
    feature_set = [feature for feature in features if feature not in controls]

    for (model, strat), sub in merged.groupby(["model", "sim_strategy"], dropna=False):
        n_group = (
            sub["join_key"].nunique() if count_unique_datasets else int(sub.shape[0])
        )
        if n_group < min_n:
            continue

        for feat in feature_set:
            x_raw = pd.to_numeric(sub[feat], errors="coerce")

            for targ in targets:
                if targ not in sub.columns:
                    continue

                y_raw = pd.to_numeric(sub[targ], errors="coerce")
                dat = pd.DataFrame(
                    {
                        "join_key": sub["join_key"].astype(str),
                        "feature": x_raw,
                        "target": y_raw,
                    }
                )
                for control in controls:
                    if control != feat:
                        dat[control] = pd.to_numeric(sub[control], errors="coerce")

                dat = dat.dropna(axis=0).copy()
                n_used = (
                    int(dat["join_key"].nunique())
                    if count_unique_datasets
                    else int(dat.shape[0])
                )
                if n_used < min_n:
                    continue
                if dat["feature"].nunique(dropna=True) < 2:
                    continue
                if dat["target"].nunique(dropna=True) < 2:
                    continue

                keep_ctrl = [
                    control
                    for control in controls
                    if control in dat.columns
                    and control != feat
                    and dat[control].nunique(dropna=True) >= 2
                ]

                y = dat["target"].to_numpy(dtype=float)
                x = dat["feature"].to_numpy(dtype=float)
                Xc = (
                    dat[keep_ctrl].to_numpy(dtype=float)
                    if keep_ctrl
                    else np.empty((dat.shape[0], 0), dtype=float)
                )
                X_full_raw = (
                    np.column_stack([x, Xc]) if Xc.shape[1] else x.reshape(-1, 1)
                )
                X_full = sm.add_constant(X_full_raw, has_constant="add")

                try:
                    fit = sm.OLS(y, X_full).fit(cov_type="HC3")
                except (ValueError, np.linalg.LinAlgError):
                    continue

                coef = float(fit.params[1])
                se = float(fit.bse[1])
                tval = float(fit.tvalues[1])
                pval_hc3 = float(fit.pvalues[1])

                try:
                    y_std = _standardize_series(y)
                    x_std = _standardize_series(x)
                    if Xc.shape[1]:
                        Xc_std = np.column_stack(
                            [_standardize_series(Xc[:, j]) for j in range(Xc.shape[1])]
                        )
                        X_std_raw = np.column_stack([x_std, Xc_std])
                    else:
                        X_std_raw = x_std.reshape(-1, 1)
                    X_std_full = sm.add_constant(X_std_raw, has_constant="add")
                    fit_std = sm.OLS(y_std, X_std_full).fit(cov_type="HC3")
                    std_coef = float(fit_std.params[1])
                except (ValueError, np.linalg.LinAlgError):
                    std_coef = np.nan

                partial_r2 = _compute_partial_r2(y=y, x_feat=x, X_ctrl=Xc)

                rows.append(
                    {
                        "model": str(model),
                        "sim_strategy": str(strat),
                        "feature": feat,
                        "target": targ,
                        "coef": coef,
                        "std_coef": std_coef,
                        "se_hc3": se,
                        "tvalue_hc3": tval,
                        "pvalue_hc3": pval_hc3,
                        "partial_r2": partial_r2,
                        "n_datasets": n_used,
                        "n_unique_join_keys": int(dat["join_key"].nunique()),
                        "controls": ",".join(keep_ctrl),
                        "ols_mode": ols_mode,
                    }
                )

    return pd.DataFrame(rows)


def _resample_merged_by_join_key(
    merged: pd.DataFrame,
    sampled_keys: np.ndarray,
) -> pd.DataFrame:
    """Return a bootstrap-resampled merged table, preserving paired methods."""
    by_key = {
        str(join_key): group
        for join_key, group in merged.groupby("join_key", dropna=False, sort=False)
    }
    pieces: List[pd.DataFrame] = []
    for draw_idx, join_key in enumerate(sampled_keys):
        key = str(join_key)
        if key not in by_key:
            continue
        piece = by_key[key].copy()
        piece["bootstrap_draw"] = int(draw_idx)
        piece["bootstrap_source_join_key"] = piece["join_key"].astype(str)
        piece["bootstrap_join_key"] = f"{key}__draw{draw_idx}"
        pieces.append(piece)

    if not pieces:
        return merged.iloc[0:0].copy()

    return pd.concat(pieces, ignore_index=True, sort=False)


def _resolve_bootstrap_n_jobs(n_jobs: int, n_tasks: int) -> int:
    """Resolve user-facing bootstrap worker counts."""
    if n_tasks <= 1:
        return 1
    if n_jobs == 0:
        return 1

    cpu_count = os.cpu_count() or 1
    if n_jobs < 0:
        # Match joblib-style semantics: -1 means all CPUs, -2 all but one, etc.
        resolved = max(cpu_count + 1 + n_jobs, 1)
    else:
        resolved = n_jobs

    return max(1, min(int(resolved), int(n_tasks)))


def _iter_with_tqdm(
    iterable: Iterable[Any],
    *,
    total: int,
    enabled: bool,
    desc: str,
) -> Iterable[Any]:
    """Wrap an iterator in tqdm when requested."""
    if not enabled:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="rep", dynamic_ncols=True)


_BOOTSTRAP_WORKER_STATE: Dict[str, Any] = {}


def _init_dataset_bootstrap_worker(
    merged: pd.DataFrame,
    features: List[str],
    targets: List[str],
    controls: List[str],
    min_n: int,
    ols_mode: str,
) -> None:
    """Initialize process-local state for dataset bootstrap workers."""
    global _BOOTSTRAP_WORKER_STATE
    _BOOTSTRAP_WORKER_STATE = {
        "merged": merged,
        "features": features,
        "targets": targets,
        "controls": controls,
        "min_n": min_n,
        "ols_mode": ols_mode,
    }


def _compute_dataset_bootstrap_replicate(
    merged: pd.DataFrame,
    features: List[str],
    targets: List[str],
    controls: List[str],
    *,
    min_n: int,
    ols_mode: str,
    replicate: int,
    sampled_keys: np.ndarray,
) -> pd.DataFrame:
    """Compute one dataset-refit bootstrap replicate."""
    boot_merged = _resample_merged_by_join_key(merged, sampled_keys)
    boot_effects = compute_refit_ols_effects(
        boot_merged,
        features,
        targets,
        controls,
        min_n=min_n,
        count_unique_datasets=False,
        ols_mode=ols_mode,
    )
    if boot_effects.empty:
        return boot_effects

    boot_effects["bootstrap_replicate"] = int(replicate)
    boot_effects["bootstrap_n_draws"] = int(len(sampled_keys))
    return boot_effects


def _compute_dataset_bootstrap_replicate_from_worker_state(
    task: Tuple[int, np.ndarray],
) -> pd.DataFrame:
    """Compute one bootstrap replicate from process-local worker state."""
    if not _BOOTSTRAP_WORKER_STATE:
        raise RuntimeError("Dataset bootstrap worker state has not been initialized.")

    replicate, sampled_keys = task
    return _compute_dataset_bootstrap_replicate(
        _BOOTSTRAP_WORKER_STATE["merged"],
        _BOOTSTRAP_WORKER_STATE["features"],
        _BOOTSTRAP_WORKER_STATE["targets"],
        _BOOTSTRAP_WORKER_STATE["controls"],
        min_n=_BOOTSTRAP_WORKER_STATE["min_n"],
        ols_mode=_BOOTSTRAP_WORKER_STATE["ols_mode"],
        replicate=replicate,
        sampled_keys=sampled_keys,
    )


def compute_dataset_refit_bootstrap_effects(
    merged: pd.DataFrame,
    features: List[str],
    targets: List[str],
    controls: List[str],
    *,
    min_n: int,
    n_boot: int,
    random_seed: int,
    ols_mode: str,
    n_jobs: int = 1,
    backend: str = "process",
    show_progress: bool = False,
    progress_label: Optional[str] = None,
) -> pd.DataFrame:
    """Bootstrap `join_key` datasets jointly and refit OLS effects."""
    if n_boot <= 0 or merged.empty:
        return pd.DataFrame()
    if "join_key" not in merged.columns:
        raise ValueError("merged must contain a 'join_key' column.")

    keys = merged["join_key"].dropna().astype(str).unique()
    if keys.size < min_n:
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    tasks = [
        (replicate, rng.choice(keys, size=keys.size, replace=True))
        for replicate in range(n_boot)
    ]
    resolved_n_jobs = (
        1 if backend == "serial" else _resolve_bootstrap_n_jobs(n_jobs, len(tasks))
    )
    progress_desc = progress_label or f"{ols_mode} bootstrap"
    rows: List[pd.DataFrame] = []

    if resolved_n_jobs == 1:
        for replicate, sampled_keys in _iter_with_tqdm(
            tasks,
            total=len(tasks),
            enabled=show_progress,
            desc=progress_desc,
        ):
            boot_effects = _compute_dataset_bootstrap_replicate(
                merged,
                features,
                targets,
                controls,
                min_n=min_n,
                ols_mode=ols_mode,
                replicate=replicate,
                sampled_keys=sampled_keys,
            )
            if not boot_effects.empty:
                rows.append(boot_effects)
    elif backend == "thread":
        _init_dataset_bootstrap_worker(
            merged, features, targets, controls, min_n, ols_mode
        )
        with ThreadPoolExecutor(max_workers=resolved_n_jobs) as executor:
            futures = [
                executor.submit(
                    _compute_dataset_bootstrap_replicate_from_worker_state,
                    task,
                )
                for task in tasks
            ]
            for future in _iter_with_tqdm(
                as_completed(futures),
                total=len(futures),
                enabled=show_progress,
                desc=progress_desc,
            ):
                boot_effects = future.result()
                if not boot_effects.empty:
                    rows.append(boot_effects)
    else:
        try:
            with ProcessPoolExecutor(
                max_workers=resolved_n_jobs,
                initializer=_init_dataset_bootstrap_worker,
                initargs=(merged, features, targets, controls, min_n, ols_mode),
            ) as executor:
                futures = [
                    executor.submit(
                        _compute_dataset_bootstrap_replicate_from_worker_state,
                        task,
                    )
                    for task in tasks
                ]
                for future in _iter_with_tqdm(
                    as_completed(futures),
                    total=len(futures),
                    enabled=show_progress,
                    desc=progress_desc,
                ):
                    boot_effects = future.result()
                    if not boot_effects.empty:
                        rows.append(boot_effects)
        except (OSError, NotImplementedError, BrokenProcessPool) as exc:
            print(
                "[WARN] Process-based bootstrap parallelism unavailable "
                f"({exc}); falling back to thread workers."
            )
            _init_dataset_bootstrap_worker(
                merged, features, targets, controls, min_n, ols_mode
            )
            with ThreadPoolExecutor(max_workers=resolved_n_jobs) as executor:
                futures = [
                    executor.submit(
                        _compute_dataset_bootstrap_replicate_from_worker_state,
                        task,
                    )
                    for task in tasks
                ]
                for future in _iter_with_tqdm(
                    as_completed(futures),
                    total=len(futures),
                    enabled=show_progress,
                    desc=progress_desc,
                ):
                    boot_effects = future.result()
                    if not boot_effects.empty:
                        rows.append(boot_effects)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True, sort=False)
    sort_cols = [
        col
        for col in (
            "bootstrap_replicate",
            "model",
            "sim_strategy",
            "feature",
            "target",
        )
        if col in out.columns
    ]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def compute_covariate_diagnostic_ols(
    merged: pd.DataFrame,
    covariates: List[str],
    targets: List[str],
    *,
    min_n: int,
) -> pd.DataFrame:
    """Compute diagnostic adjusted OLS effects for each covariate.

    Each covariate is tested as the focal predictor while the remaining
    covariates are retained as controls:
        target ~ focal_covariate + remaining_covariates

    Args:
        merged (pd.DataFrame): Merged performance + stats table.
        covariates (List[str]): Covariates to test as focal predictors.
        targets (List[str]): Performance target columns.
        min_n (int): Minimum number of datasets.

    Returns:
        pd.DataFrame: Long-form diagnostic regression results.
    """
    rows: List[Dict[str, object]] = []
    covariates = [c for c in covariates if c in merged.columns]

    for (model, strat), sub in merged.groupby(["model", "sim_strategy"], dropna=False):
        if sub["join_key"].nunique() < min_n:
            continue

        for focal in covariates:
            focal_raw = pd.to_numeric(sub[focal], errors="coerce")
            focal_controls = [
                c
                for c in covariates
                if c != focal
                and c in sub.columns
                and pd.to_numeric(sub[c], errors="coerce").nunique(dropna=True) > 1
            ]

            for targ in targets:
                y_raw = pd.to_numeric(sub[targ], errors="coerce")
                dat = pd.DataFrame(
                    {
                        "join_key": sub["join_key"].astype(str),
                        "feature": focal_raw,
                        "target": y_raw,
                    }
                )
                for c in focal_controls:
                    dat[c] = pd.to_numeric(sub[c], errors="coerce")

                dat = dat.dropna(axis=0).copy()
                n_used = dat["join_key"].nunique()
                if n_used < min_n:
                    continue
                if dat["feature"].nunique(dropna=True) < 2:
                    continue
                if dat["target"].nunique(dropna=True) < 2:
                    continue

                y = dat["target"].to_numpy(dtype=float)
                x = dat["feature"].to_numpy(dtype=float)
                Xc = (
                    dat[focal_controls].to_numpy(dtype=float)
                    if focal_controls
                    else np.empty((dat.shape[0], 0), dtype=float)
                )

                X_full_raw = (
                    np.column_stack([x, Xc]) if Xc.shape[1] else x.reshape(-1, 1)
                )
                X_full = sm.add_constant(X_full_raw, has_constant="add")
                fit = sm.OLS(y, X_full).fit(cov_type="HC3")

                coef = float(fit.params[1])
                se = float(fit.bse[1])
                tval = float(fit.tvalues[1])
                pval_hc3 = float(fit.pvalues[1])

                try:
                    y_std = _standardize_series(y)
                    x_std = _standardize_series(x)
                    if Xc.shape[1]:
                        Xc_std = np.column_stack(
                            [_standardize_series(Xc[:, j]) for j in range(Xc.shape[1])]
                        )
                        X_std_raw = np.column_stack([x_std, Xc_std])
                    else:
                        X_std_raw = x_std.reshape(-1, 1)
                    X_std_full = sm.add_constant(X_std_raw, has_constant="add")
                    fit_std = sm.OLS(y_std, X_std_full).fit(cov_type="HC3")
                    std_coef = float(fit_std.params[1])
                except ValueError:
                    std_coef = np.nan

                partial_r2 = _compute_partial_r2(y=y, x_feat=x, X_ctrl=Xc)

                rows.append(
                    {
                        "model": str(model),
                        "sim_strategy": str(strat),
                        "feature": focal,
                        "target": targ,
                        "coef": coef,
                        "std_coef": std_coef,
                        "se_hc3": se,
                        "tvalue_hc3": tval,
                        "pvalue_hc3": pval_hc3,
                        "partial_r2": partial_r2,
                        "n_datasets": int(n_used),
                        "controls": ",".join(focal_controls),
                        "ols_mode": "covariate_diagnostic",
                    }
                )

    return pd.DataFrame(rows)


def compute_no_control_ols(
    merged: pd.DataFrame,
    features: List[str],
    targets: List[str],
    *,
    min_n: int,
    n_permutations: int = 0,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Compute unadjusted per-feature OLS effects.

    For each (model, sim_strategy, feature, target), fit:
        target ~ feature

    Args:
        merged (pd.DataFrame): Merged performance + stats table.
        features (List[str]): Focal feature columns.
        targets (List[str]): Performance target columns.
        min_n (int): Minimum number of datasets.
        n_permutations (int): If > 0, compute no-control permutation p-values.
        random_seed (Optional[int]): Base random seed for permutation testing.

    Returns:
        pd.DataFrame: Long-form no-control regression results.
    """
    rows: List[Dict[str, object]] = []
    features = [f for f in features if f in merged.columns]
    do_permutation = n_permutations > 0
    base_rng = np.random.default_rng(random_seed) if do_permutation else None

    for (model, strat), sub in merged.groupby(["model", "sim_strategy"], dropna=False):
        if sub["join_key"].nunique() < min_n:
            continue

        for feat in features:
            x_raw = pd.to_numeric(sub[feat], errors="coerce")
            for targ in targets:
                y_raw = pd.to_numeric(sub[targ], errors="coerce")
                dat = pd.DataFrame(
                    {
                        "join_key": sub["join_key"].astype(str),
                        "feature": x_raw,
                        "target": y_raw,
                    }
                ).dropna(axis=0)

                n_used = dat["join_key"].nunique()
                if n_used < min_n:
                    continue
                if dat["feature"].nunique(dropna=True) < 2:
                    continue
                if dat["target"].nunique(dropna=True) < 2:
                    continue

                y = dat["target"].to_numpy(dtype=float)
                x = dat["feature"].to_numpy(dtype=float)
                X_full = sm.add_constant(x.reshape(-1, 1), has_constant="add")
                fit = sm.OLS(y, X_full).fit(cov_type="HC3")

                coef = float(fit.params[1])
                se = float(fit.bse[1])
                tval = float(fit.tvalues[1])
                pval_hc3 = float(fit.pvalues[1])
                r_squared = float(fit.rsquared)

                try:
                    y_std = _standardize_series(y)
                    x_std = _standardize_series(x)
                    X_std_full = sm.add_constant(
                        x_std.reshape(-1, 1), has_constant="add"
                    )
                    fit_std = sm.OLS(y_std, X_std_full).fit(cov_type="HC3")
                    std_coef = float(fit_std.params[1])
                except ValueError:
                    std_coef = np.nan

                row = {
                    "model": str(model),
                    "sim_strategy": str(strat),
                    "feature": feat,
                    "target": targ,
                    "coef": coef,
                    "std_coef": std_coef,
                    "se_hc3": se,
                    "tvalue_hc3": tval,
                    "pvalue_hc3": pval_hc3,
                    "partial_r2": r_squared,
                    "r_squared": r_squared,
                    "n_datasets": int(n_used),
                    "controls": "",
                    "ols_mode": "no_control",
                }

                if do_permutation and base_rng is not None:
                    test_rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
                    perm_res = _freedman_lane_perm_test(
                        y=y,
                        x_feat=x,
                        X_ctrl=np.empty((y.shape[0], 0), dtype=float),
                        n_permutations=n_permutations,
                        rng=test_rng,
                    )
                    row.update(
                        {
                            "perm_tvalue_obs": perm_res["obs_t"],
                            "perm_pvalue": perm_res["perm_pvalue"],
                            "n_permutations": int(n_permutations),
                            "ols_mode": "no_control_permutation",
                        }
                    )

                rows.append(row)

    return pd.DataFrame(rows)


def add_neutral_rows_for_constant_targets(
    res_long: pd.DataFrame,
    merged: pd.DataFrame,
    features: List[str],
    targets: List[str],
    controls: List[str],
    *,
    min_n: int,
    n_permutations: int,
) -> pd.DataFrame:
    """Add neutral adjusted-effect rows for structurally constant targets.

    Constant targets cannot be fit by OLS/permutation models, so the adjusted
    analysis naturally skips them. For plotting, a constant performance metric
    has no feature-explainable variation; represent it as a neutral zero effect
    with non-significant p/q values so expected model sections remain visible.
    """
    feature_set = [f for f in features if f not in controls]
    existing = set()

    if not res_long.empty:
        existing = set(
            zip(
                res_long["model"].astype(str),
                res_long["sim_strategy"].astype(str),
                res_long["feature"].astype(str),
                res_long["target"].astype(str),
            )
        )

    rows: List[Dict[str, object]] = []
    for (model, strat), sub in merged.groupby(["model", "sim_strategy"], dropna=False):
        if sub["join_key"].nunique() < min_n:
            continue

        keep_ctrl = [c for c in controls if c in sub.columns and sub[c].notna().any()]
        for targ in targets:
            if targ not in sub.columns:
                continue

            y = pd.to_numeric(sub[targ], errors="coerce")
            if y.dropna().nunique() >= 2:
                continue

            n_used = int(sub.loc[y.notna(), "join_key"].nunique())
            if n_used < min_n:
                continue

            for feat in feature_set:
                key = (str(model), str(strat), str(feat), str(targ))
                if key in existing:
                    continue

                rows.append(
                    {
                        "model": str(model),
                        "sim_strategy": str(strat),
                        "feature": feat,
                        "target": targ,
                        "coef": 0.0,
                        "std_coef": 0.0,
                        "se_hc3": 0.0,
                        "tvalue_hc3": 0.0,
                        "pvalue_hc3": 1.0,
                        "partial_r2": 0.0,
                        "perm_tvalue_obs": 0.0,
                        "perm_pvalue": 1.0,
                        "qvalue_bh": 1.0,
                        "n_datasets": n_used,
                        "n_permutations": int(n_permutations),
                        "controls": ",".join(keep_ctrl),
                        "note": "constant_target_neutral_effect",
                    }
                )

    if not rows:
        return res_long

    neutral = pd.DataFrame(rows)
    if res_long.empty:
        return neutral

    return pd.concat([res_long, neutral], ignore_index=True, sort=False)


def _driver_ranking_summary(
    df: pd.DataFrame,
    value_col: str,
    *,
    n_boot: int,
    ci: float,
    random_seed: int,
    fixed_feature_order: Optional[List[str]] = None,
    bootstrap_effects: Optional[pd.DataFrame] = None,
    interval_mode: str = "effect_row",
) -> pd.DataFrame:
    """Summarize driver effects with optional dataset-refit intervals."""
    rng = np.random.default_rng(random_seed)
    alpha = 100.0 - ci
    lower_q = alpha / 2.0
    upper_q = 100.0 - (alpha / 2.0)
    boot_df = (
        bootstrap_effects.dropna(subset=["feature", value_col, "bootstrap_replicate"])
        if bootstrap_effects is not None
        and not bootstrap_effects.empty
        and "bootstrap_replicate" in bootstrap_effects.columns
        and "feature" in bootstrap_effects.columns
        and value_col in bootstrap_effects.columns
        else pd.DataFrame()
    )

    rows = []
    for feature, feat_df in df.groupby("feature", dropna=False):
        vals = np.abs(feat_df[value_col].to_numpy(dtype=float))
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        obs = float(np.mean(vals))
        ci_low = obs
        ci_high = obs
        ci_source = "none"
        n_bootstrap_replicates = 0

        if interval_mode == "dataset_refit" and not boot_df.empty:
            boot_feature = boot_df.loc[boot_df["feature"].eq(feature)].copy()
            boot_stats = []
            for _, rep_df in boot_feature.groupby("bootstrap_replicate", sort=False):
                rep_vals = np.abs(rep_df[value_col].to_numpy(dtype=float))
                rep_vals = rep_vals[np.isfinite(rep_vals)]
                if rep_vals.size:
                    boot_stats.append(float(np.mean(rep_vals)))
            if boot_stats:
                boot_stats_arr = np.asarray(boot_stats, dtype=float)
                ci_low = float(np.percentile(boot_stats_arr, lower_q))
                ci_high = float(np.percentile(boot_stats_arr, upper_q))
                ci_source = "dataset_refit_join_key"
                n_bootstrap_replicates = int(boot_stats_arr.size)
        elif interval_mode == "effect_row" and vals.size > 1 and n_boot > 0:
            boot_stats = np.empty(n_boot, dtype=float)
            n = vals.size
            for i in range(n_boot):
                sample = rng.choice(vals, size=n, replace=True)
                boot_stats[i] = np.mean(sample)

            ci_low = float(np.percentile(boot_stats, lower_q))
            ci_high = float(np.percentile(boot_stats, upper_q))
            ci_source = "effect_row_resampling_descriptive"
            n_bootstrap_replicates = int(n_boot)
        elif interval_mode == "none" or vals.size == 1:
            ci_low = obs
            ci_high = obs
        else:
            ci_source = "none"

        rows.append(
            {
                "feature": feature,
                "mean_abs_effect": obs,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_source": ci_source,
                "n_bootstrap_replicates": n_bootstrap_replicates,
            }
        )

    agg = pd.DataFrame(rows)
    if agg.empty:
        return agg

    if fixed_feature_order is None:
        agg = agg.sort_values("mean_abs_effect", ascending=False).copy()
    else:
        ordered = [f for f in fixed_feature_order if f in set(agg["feature"])]
        extras = [f for f in agg["feature"].tolist() if f not in set(ordered)]
        agg = agg.set_index("feature").loc[ordered + extras].reset_index().copy()

    agg["feature_display"] = agg["feature"].map(
        lambda x: FEATURE_DISPLAY_LABELS[x] if x in FEATURE_DISPLAY_LABELS else x
    )
    return agg


def plot_adjusted_driver_ranking(
    ols_long: pd.DataFrame,
    outdir: Path,
    *,
    dpi: int,
    prefix: str = "adjusted_permutation",
    value_col: str = "std_coef",
    n_boot: int = 10000,
    ci: float = 95.0,
    random_seed: int = 42,
    target_filter: Optional[List[str]] = None,
    target_label: Optional[str] = None,
    model_filter: Optional[List[str]] = None,
    model_group_label: Optional[str] = None,
    fixed_feature_order: Optional[List[str]] = None,
    rank_subdir: str = "adjusted_driver_ranking",
    analysis_label: str = "Adjusted",
    effect_label: str = "adjusted",
    bootstrap_effects: Optional[pd.DataFrame] = None,
    interval_mode: str = "effect_row",
) -> None:
    """Plot global adjusted driver ranking with bootstrap confidence intervals.

    The ranking is based on the mean absolute value of `value_col` across the
    selected targets for each feature. Preferred uncertainty uses paired
    dataset-level bootstrap refits supplied in `bootstrap_effects`.

    Args:
        ols_long (pd.DataFrame): Long-form adjusted regression results table.
        outdir (Path): Output directory.
        dpi (int): Figure DPI.
        prefix (str, optional): Filename prefix.
        value_col (str, optional): Column used for ranking. Typically one of
            {"std_coef", "partial_r2"}.
        n_boot (int, optional): Number of bootstrap resamples per feature.
        ci (float, optional): Confidence interval width in percent.
        random_seed (int, optional): Random seed for reproducibility.
        target_filter (Optional[List[str]], optional): Restrict plot to these
            targets only.
        target_label (Optional[str], optional): Label to display in the title
            and output filename.
        model_filter (Optional[List[str]], optional): Restrict plot to these
            imputation models only.
        model_group_label (Optional[str], optional): Label to display in the title
            for a model-filtered ranking.
        fixed_feature_order (Optional[List[str]], optional): If provided, plot
            features in this order instead of ranking by mean absolute effect.
        rank_subdir (str, optional): Output subdirectory for ranking plots.
        analysis_label (str, optional): Prefix used in the plot title.
        effect_label (str, optional): Descriptor used in x-axis labels.
        bootstrap_effects (Optional[pd.DataFrame], optional): Refit bootstrap
            coefficients generated by `compute_dataset_refit_bootstrap_effects`.
        interval_mode (str, optional): One of {"dataset_refit", "effect_row",
            "none"}.

    Returns:
        None
    """
    if ols_long.empty:
        return

    if "feature" not in ols_long.columns:
        raise ValueError("ols_long must contain a 'feature' column.")
    if value_col not in ols_long.columns:
        raise ValueError(f"ols_long must contain '{value_col}'.")
    if "target" not in ols_long.columns:
        raise ValueError("ols_long must contain a 'target' column.")

    sub = ols_long.dropna(subset=["feature", value_col, "target"]).copy()
    if sub.empty:
        return

    if target_filter is not None:
        sub = sub.loc[sub["target"].isin(target_filter)].copy()
        if sub.empty:
            return

    if model_filter is not None:
        if "model" not in sub.columns:
            raise ValueError("ols_long must contain a 'model' column.")
        sub = sub.loc[sub["model"].isin(model_filter)].copy()
        if sub.empty:
            return

    rank_dir = outdir / rank_subdir
    rank_dir.mkdir(parents=True, exist_ok=True)

    boot_sub = bootstrap_effects
    if boot_sub is not None and not boot_sub.empty:
        boot_sub = boot_sub.copy()
        if target_filter is not None and "target" in boot_sub.columns:
            boot_sub = boot_sub.loc[boot_sub["target"].isin(target_filter)].copy()
        if model_filter is not None and "model" in boot_sub.columns:
            boot_sub = boot_sub.loc[boot_sub["model"].isin(model_filter)].copy()

    agg = _driver_ranking_summary(
        sub,
        value_col,
        n_boot=n_boot,
        ci=ci,
        random_seed=random_seed,
        fixed_feature_order=fixed_feature_order,
        bootstrap_effects=boot_sub,
        interval_mode=interval_mode,
    )
    if agg.empty:
        return

    ordered_features = agg["feature_display"].tolist()

    panel_maps = {
        "REF F1-score": "Homozygous Reference F1-score",
        "HET F1-score": "Heterozygous F1-score",
        "ALT F1-score": "Homozygous Alternate F1-score",
        "MCC": "Matthews Correlation Coefficient",
    }

    panel_label = panel_maps.get(target_label, target_label) if target_label else ""
    panel_suffix = f": {panel_label}" if panel_label else ""
    suffix = (
        "_" + re.sub(r"[^a-zA-Z0-9]+", "_", target_label).strip("_").lower()
        if target_label
        else ""
    )

    effect_prefix = f"{effect_label.strip()} " if effect_label.strip() else ""
    has_intervals = "ci_source" in agg.columns and agg["ci_source"].ne("none").any()
    if has_intervals and interval_mode == "dataset_refit":
        interval_suffix = f" ± {int(ci)}% dataset-bootstrap CI"
    elif has_intervals and interval_mode == "effect_row":
        interval_suffix = f" ± {int(ci)}% descriptive interval"
    elif has_intervals:
        interval_suffix = f" ± {int(ci)}% interval"
    else:
        interval_suffix = ""

    if value_col == "partial_r2":
        xlabel = rf"Mean {effect_prefix}Partial R2{interval_suffix}"
    elif value_col == "std_coef":
        xlabel = rf"Mean {effect_prefix}|Bstd|{interval_suffix}"
    else:
        pretty_name = value_col.replace("_", " ")
        xlabel = f"Mean |{pretty_name}|{interval_suffix}"

    title_parts = []
    if model_group_label:
        title_parts.append(model_group_label)
    if fixed_feature_order is not None:
        title_parts.append("Fixed Feature Order")
    group_suffix = f" ({'; '.join(title_parts)})" if title_parts else ""
    title_prefix = f"{analysis_label} " if analysis_label else ""
    title = f"{title_prefix}Global Driver Ranking{group_suffix}{panel_suffix}"

    fig_h = max(5, 0.45 * agg.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    sns.barplot(
        data=agg,
        x="mean_abs_effect",
        y="feature_display",
        order=ordered_features,
        orient="h",
        errorbar=None,
        ax=ax,
        color="#811BF3",
    )

    ax.set_xlim(left=0.0, right=1.0)

    y_positions = np.arange(len(agg), dtype=float)
    x = agg["mean_abs_effect"].to_numpy(dtype=float)
    xerr_low = x - agg["ci_low"].to_numpy(dtype=float)
    xerr_high = agg["ci_high"].to_numpy(dtype=float) - x

    finite = np.isfinite(x) & np.isfinite(xerr_low) & np.isfinite(xerr_high)
    if "ci_source" in agg.columns:
        finite &= agg["ci_source"].ne("none").to_numpy(dtype=bool)
    if finite.any():
        ax.errorbar(
            x=x[finite],
            y=y_positions[finite],
            xerr=np.vstack(
                [
                    np.maximum(xerr_low[finite], 0.0),
                    np.maximum(xerr_high[finite], 0.0),
                ]
            ),
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=5,
            capthick=1.5,
            zorder=3,
        )

    ax.set_title(title, fontsize="large")
    ax.set_xlabel(xlabel, fontsize="large")
    ax.set_ylabel("Population Genomic Summary Statistic", fontsize="large")
    ax.tick_params(axis="both", which="major", labelsize="large")
    fig.tight_layout()

    of = rank_dir / f"{prefix}_driver_ranking_{value_col}{suffix}"
    [fig.savefig(of.with_suffix(f".{ext}"), dpi=dpi) for ext in ("png", "pdf")]
    plt.close(fig)


def plot_combined_driver_ranking_grid(
    ols_long: pd.DataFrame,
    outdir: Path,
    *,
    dpi: int,
    prefix: str,
    value_col: str,
    fixed_feature_order: List[str],
    n_boot: int = 10000,
    ci: float = 95.0,
    random_seed: int = 42,
    rank_subdir: str = "adjusted_driver_ranking",
    effect_label: str = "adjusted",
    bootstrap_effects: Optional[pd.DataFrame] = None,
    interval_mode: str = "effect_row",
) -> None:
    """Plot a 3x2 fixed-order driver ranking grid for baseline vs deep learning."""
    if ols_long.empty or not fixed_feature_order:
        return
    if value_col not in ols_long.columns:
        raise ValueError(f"ols_long must contain '{value_col}'.")

    target_panels = (
        ("REF_F1score", "Homozygous Reference"),
        ("HET_F1score", "Heterozygous"),
        ("ALT_F1score", "Homozygous Alternate"),
    )
    model_panels = (
        ("MostFrequent baseline", list(DETERMINISTIC_MODELS)),
        ("Deep learning", list(DEEP_LEARNING_MODELS)),
    )

    panel_aggs: dict[tuple[int, int], pd.DataFrame] = {}
    x_max = 0.0
    for row_idx, (target, _) in enumerate(target_panels):
        for col_idx, (_, model_filter) in enumerate(model_panels):
            sub = ols_long.dropna(subset=["feature", value_col, "target"]).copy()
            sub = sub.loc[
                sub["target"].eq(target) & sub["model"].isin(model_filter)
            ].copy()
            if sub.empty:
                panel_aggs[(row_idx, col_idx)] = pd.DataFrame()
                continue

            boot_sub = bootstrap_effects
            if boot_sub is not None and not boot_sub.empty:
                boot_sub = boot_sub.loc[
                    boot_sub["target"].eq(target) & boot_sub["model"].isin(model_filter)
                ].copy()

            agg = _driver_ranking_summary(
                sub,
                value_col,
                n_boot=n_boot,
                ci=ci,
                random_seed=random_seed + row_idx * 10 + col_idx,
                fixed_feature_order=fixed_feature_order,
                bootstrap_effects=boot_sub,
                interval_mode=interval_mode,
            )
            panel_aggs[(row_idx, col_idx)] = agg
            if not agg.empty:
                finite_high = agg["ci_high"].to_numpy(dtype=float)
                finite_high = finite_high[np.isfinite(finite_high)]
                if finite_high.size:
                    x_max = max(x_max, float(np.nanmax(finite_high)))

    if all(agg.empty for agg in panel_aggs.values()):
        return

    x_max = max(0.05, x_max * 1.12)
    feature_labels = [
        FEATURE_DISPLAY_LABELS.get(feature, feature) for feature in fixed_feature_order
    ]
    y_positions = np.arange(len(fixed_feature_order), dtype=float)

    effect_prefix = f"{effect_label.strip()} " if effect_label.strip() else ""
    has_intervals = any(
        not agg.empty
        and "ci_source" in agg.columns
        and agg["ci_source"].ne("none").any()
        for agg in panel_aggs.values()
    )
    if has_intervals and interval_mode == "dataset_refit":
        interval_suffix = f" ± {int(ci)}% dataset-bootstrap CI"
    elif has_intervals and interval_mode == "effect_row":
        interval_suffix = f" ± {int(ci)}% descriptive interval"
    elif has_intervals:
        interval_suffix = f" ± {int(ci)}% interval"
    else:
        interval_suffix = ""

    if value_col == "partial_r2":
        xlabel = rf"Mean {effect_prefix}partial $R^2${interval_suffix}"
    elif value_col == "std_coef":
        xlabel = rf"Mean {effect_prefix}|$\beta_\mathrm{{std}}$|{interval_suffix}"
    else:
        pretty_name = value_col.replace("_", " ")
        xlabel = f"Mean |{pretty_name}|{interval_suffix}"

    rc_params = {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 24,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with mpl.rc_context(rc_params):
        fig, axes = plt.subplots(
            nrows=3,
            ncols=2,
            figsize=(28, 24),
            sharex=True,
            sharey=False,
        )

        for row_idx, (_, row_label) in enumerate(target_panels):
            for col_idx, (col_label, _) in enumerate(model_panels):
                ax = axes[row_idx, col_idx]
                agg = panel_aggs[(row_idx, col_idx)]
                if agg.empty:
                    widths = np.zeros(len(fixed_feature_order), dtype=float)
                    ci_low = widths.copy()
                    ci_high = widths.copy()
                else:
                    panel = (
                        agg.set_index("feature")
                        .reindex(fixed_feature_order)
                        .reset_index()
                    )
                    widths = panel["mean_abs_effect"].to_numpy(dtype=float)
                    ci_low = panel["ci_low"].to_numpy(dtype=float)
                    ci_high = panel["ci_high"].to_numpy(dtype=float)

                ax.barh(
                    y_positions,
                    widths,
                    color="#811BF3",
                    edgecolor="black",
                    linewidth=0.8,
                )

                finite = (
                    np.isfinite(widths) & np.isfinite(ci_low) & np.isfinite(ci_high)
                )
                if not agg.empty and "ci_source" in panel.columns:
                    finite &= panel["ci_source"].ne("none").to_numpy(dtype=bool)
                if finite.any():
                    xerr_low = np.maximum(widths[finite] - ci_low[finite], 0.0)
                    xerr_high = np.maximum(ci_high[finite] - widths[finite], 0.0)
                    ax.errorbar(
                        x=widths[finite],
                        y=y_positions[finite],
                        xerr=np.vstack([xerr_low, xerr_high]),
                        fmt="none",
                        ecolor="black",
                        elinewidth=2.0,
                        capsize=5,
                        capthick=2.0,
                        zorder=3,
                    )

                ax.set_xlim(left=0.0, right=x_max)
                ax.set_yticks(y_positions)
                ax.set_yticklabels(feature_labels)
                ax.invert_yaxis()
                ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.55)
                ax.set_axisbelow(True)

                title = f"{col_label}\n{row_label}"
                ax.set_title(title, fontweight="bold", pad=10)
                if row_idx == 2:
                    ax.set_xlabel(xlabel)
                if col_idx == 0:
                    ax.set_ylabel("Summary statistic")
                else:
                    ax.set_ylabel("")

        fig.tight_layout(w_pad=2.0, h_pad=2.0)
        rank_dir = outdir / rank_subdir
        rank_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            rank_dir
            / f"{prefix}_combined_baseline_deep_learning_driver_ranking_{value_col}_f1_genotypes"
        )
        save_figure_dual(fig, out_path, dpi=dpi)
        plt.close(fig)


def _deep_minus_baseline_summary(
    ols_long: pd.DataFrame,
    *,
    target_filter: Optional[List[str]],
    n_boot: int,
    ci: float,
    random_seed: int,
    fixed_feature_order: Optional[List[str]] = None,
    bootstrap_effects: Optional[pd.DataFrame] = None,
    interval_mode: str = "effect_row",
) -> pd.DataFrame:
    """Summarize Δ adjusted |Bstd|: deep learning minus MostFrequent."""
    sub = ols_long.dropna(subset=["feature", "std_coef", "target", "model"]).copy()
    if target_filter is not None:
        sub = sub.loc[sub["target"].isin(target_filter)].copy()
    if sub.empty:
        return pd.DataFrame()

    baseline = sub.loc[sub["model"].isin(DETERMINISTIC_MODELS)].copy()
    deep = sub.loc[sub["model"].isin(DEEP_LEARNING_MODELS)].copy()
    feature_set = sorted(set(baseline["feature"]).intersection(set(deep["feature"])))
    if not feature_set:
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    alpha = 100.0 - ci
    lower_q = alpha / 2.0
    upper_q = 100.0 - (alpha / 2.0)
    boot_df = (
        bootstrap_effects.dropna(
            subset=["feature", "std_coef", "target", "model", "bootstrap_replicate"]
        ).copy()
        if bootstrap_effects is not None
        and not bootstrap_effects.empty
        and {
            "feature",
            "std_coef",
            "target",
            "model",
            "bootstrap_replicate",
        }.issubset(bootstrap_effects.columns)
        else pd.DataFrame()
    )
    if target_filter is not None and not boot_df.empty:
        boot_df = boot_df.loc[boot_df["target"].isin(target_filter)].copy()

    rows = []

    for feature in feature_set:
        baseline_vals = np.abs(
            baseline.loc[baseline["feature"].eq(feature), "std_coef"].to_numpy(
                dtype=float
            )
        )
        deep_vals = np.abs(
            deep.loc[deep["feature"].eq(feature), "std_coef"].to_numpy(dtype=float)
        )
        baseline_vals = baseline_vals[np.isfinite(baseline_vals)]
        deep_vals = deep_vals[np.isfinite(deep_vals)]

        if baseline_vals.size == 0 or deep_vals.size == 0:
            continue

        baseline_mean = float(np.mean(baseline_vals))
        deep_mean = float(np.mean(deep_vals))
        delta = deep_mean - baseline_mean

        ci_low = delta
        ci_high = delta
        abs_ci_low = abs(delta)
        abs_ci_high = abs(delta)
        ci_source = "none"
        n_bootstrap_replicates = 0

        if interval_mode == "dataset_refit" and not boot_df.empty:
            boot_stats = []
            boot_feature = boot_df.loc[boot_df["feature"].eq(feature)].copy()
            for _, rep_df in boot_feature.groupby("bootstrap_replicate", sort=False):
                rep_baseline_vals = np.abs(
                    rep_df.loc[
                        rep_df["model"].isin(DETERMINISTIC_MODELS),
                        "std_coef",
                    ].to_numpy(dtype=float)
                )
                rep_deep_vals = np.abs(
                    rep_df.loc[
                        rep_df["model"].isin(DEEP_LEARNING_MODELS),
                        "std_coef",
                    ].to_numpy(dtype=float)
                )
                rep_baseline_vals = rep_baseline_vals[np.isfinite(rep_baseline_vals)]
                rep_deep_vals = rep_deep_vals[np.isfinite(rep_deep_vals)]
                if rep_baseline_vals.size and rep_deep_vals.size:
                    boot_stats.append(
                        float(np.mean(rep_deep_vals) - np.mean(rep_baseline_vals))
                    )

            if boot_stats:
                boot_stats_arr = np.asarray(boot_stats, dtype=float)
                ci_low = float(np.percentile(boot_stats_arr, lower_q))
                ci_high = float(np.percentile(boot_stats_arr, upper_q))
                abs_boot_stats = np.abs(boot_stats_arr)
                abs_ci_low = float(np.percentile(abs_boot_stats, lower_q))
                abs_ci_high = float(np.percentile(abs_boot_stats, upper_q))
                ci_source = "dataset_refit_join_key"
                n_bootstrap_replicates = int(boot_stats_arr.size)
        elif (
            interval_mode == "effect_row"
            and n_boot > 0
            and not (baseline_vals.size == 1 and deep_vals.size == 1)
        ):
            boot_stats = np.empty(n_boot, dtype=float)
            for i in range(n_boot):
                baseline_sample = rng.choice(
                    baseline_vals, size=baseline_vals.size, replace=True
                )
                deep_sample = rng.choice(deep_vals, size=deep_vals.size, replace=True)
                boot_stats[i] = float(np.mean(deep_sample) - np.mean(baseline_sample))

            ci_low = float(np.percentile(boot_stats, lower_q))
            ci_high = float(np.percentile(boot_stats, upper_q))
            abs_boot_stats = np.abs(boot_stats)
            abs_ci_low = float(np.percentile(abs_boot_stats, lower_q))
            abs_ci_high = float(np.percentile(abs_boot_stats, upper_q))
            ci_source = "effect_row_resampling_descriptive"
            n_bootstrap_replicates = int(n_boot)

        rows.append(
            {
                "feature": feature,
                "baseline_mean_abs_bstd": baseline_mean,
                "deep_learning_mean_abs_bstd": deep_mean,
                "delta_deep_minus_mostfrequent_abs_bstd": delta,
                "abs_delta_deep_minus_mostfrequent_abs_bstd": abs(delta),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "abs_ci_low": abs_ci_low,
                "abs_ci_high": abs_ci_high,
                "n_baseline": int(baseline_vals.size),
                "n_deep_learning": int(deep_vals.size),
                "ci_source": ci_source,
                "n_bootstrap_replicates": n_bootstrap_replicates,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    if fixed_feature_order is None:
        summary = summary.sort_values(
            "abs_delta_deep_minus_mostfrequent_abs_bstd", ascending=False
        ).copy()
    else:
        ordered = [f for f in fixed_feature_order if f in set(summary["feature"])]
        extras = [f for f in summary["feature"].tolist() if f not in set(ordered)]
        summary = summary.set_index("feature").loc[ordered + extras].reset_index()

    summary["feature_display"] = summary["feature"].map(
        lambda x: FEATURE_DISPLAY_LABELS[x] if x in FEATURE_DISPLAY_LABELS else x
    )
    return summary


def plot_deep_minus_baseline_driver_difference(
    ols_long: pd.DataFrame,
    outdir: Path,
    *,
    dpi: int,
    prefix: str,
    target_filter: Optional[List[str]] = None,
    target_label: Optional[str] = None,
    fixed_feature_order: Optional[List[str]] = None,
    n_boot: int = 10000,
    ci: float = 95.0,
    random_seed: int = 42,
    rank_subdir: str = "adjusted_deep_minus_mostfrequent_driver_difference",
    title_effect_label: str = "Adjusted",
    xlabel_effect_label: str = "adjusted",
    bootstrap_effects: Optional[pd.DataFrame] = None,
    interval_mode: str = "effect_row",
) -> pd.DataFrame:
    """Plot Δ adjusted |Bstd| for deep learning minus MostFrequent."""
    summary = _deep_minus_baseline_summary(
        ols_long,
        target_filter=target_filter,
        n_boot=n_boot,
        ci=ci,
        random_seed=random_seed,
        fixed_feature_order=fixed_feature_order,
        bootstrap_effects=bootstrap_effects,
        interval_mode=interval_mode,
    )
    if summary.empty:
        return summary

    panel_maps = {
        "REF F1-score": "Homozygous Reference F1-score",
        "HET F1-score": "Heterozygous F1-score",
        "ALT F1-score": "Homozygous Alternate F1-score",
        "MCC": "Matthews Correlation Coefficient",
    }
    panel_label = panel_maps.get(target_label, target_label) if target_label else ""
    panel_suffix = f": {panel_label}" if panel_label else ""
    suffix = (
        "_" + re.sub(r"[^a-zA-Z0-9]+", "_", target_label).strip("_").lower()
        if target_label
        else ""
    )

    rank_dir = outdir / rank_subdir
    rank_dir.mkdir(parents=True, exist_ok=True)

    y_positions = np.arange(summary.shape[0], dtype=float)
    delta = summary["delta_deep_minus_mostfrequent_abs_bstd"].to_numpy(dtype=float)
    ci_low = summary["ci_low"].to_numpy(dtype=float)
    ci_high = summary["ci_high"].to_numpy(dtype=float)
    colors = np.where(delta >= 0, "#2E86AB", "#D95F02")

    fig_h = max(5, 0.45 * summary.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(
        y_positions,
        delta,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )

    finite = np.isfinite(delta) & np.isfinite(ci_low) & np.isfinite(ci_high)
    if "ci_source" in summary.columns:
        finite &= summary["ci_source"].ne("none").to_numpy(dtype=bool)
    if finite.any():
        xerr_low = np.maximum(delta[finite] - ci_low[finite], 0.0)
        xerr_high = np.maximum(ci_high[finite] - delta[finite], 0.0)
        ax.errorbar(
            x=delta[finite],
            y=y_positions[finite],
            xerr=np.vstack([xerr_low, xerr_high]),
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=5,
            capthick=1.5,
            zorder=3,
        )

    ax.axvline(0.0, color="black", linewidth=1.2)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks(DIFFERENCE_AXIS_TICKS)
    ax.set_xticklabels(DIFFERENCE_AXIS_TICKLABELS)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary["feature_display"].tolist())
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    title_effect = (
        f"{title_effect_label.strip()} " if title_effect_label.strip() else ""
    )
    xlabel_effect = (
        f"{xlabel_effect_label.strip()} " if xlabel_effect_label.strip() else ""
    )
    has_intervals = (
        "ci_source" in summary.columns and summary["ci_source"].ne("none").any()
    )
    if has_intervals and interval_mode == "dataset_refit":
        interval_suffix = f" ± {int(ci)}% dataset-bootstrap CI"
    elif has_intervals and interval_mode == "effect_row":
        interval_suffix = f" ± {int(ci)}% descriptive interval"
    elif has_intervals:
        interval_suffix = f" ± {int(ci)}% interval"
    else:
        interval_suffix = ""
    ax.set_title(
        f"Deep Learning − MostFrequent {title_effect}|Bstd|{panel_suffix}",
        fontsize=16,
    )
    ax.set_xlabel(
        rf"Δ mean {xlabel_effect}|Bstd|{interval_suffix} "
        "(Deep learning − MostFrequent)",
        fontsize=16,
    )
    ax.set_ylabel("Population Genomic Summary Statistic", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    fig.tight_layout()

    out_path = rank_dir / f"{prefix}_driver_difference_abs_bstd{suffix}"
    save_figure_dual(fig, out_path, dpi=dpi)
    plt.close(fig)
    return summary


def plot_combined_deep_minus_baseline_driver_difference(
    ols_long: pd.DataFrame,
    outdir: Path,
    *,
    dpi: int,
    prefix: str,
    fixed_feature_order: List[str],
    n_boot: int = 10000,
    ci: float = 95.0,
    random_seed: int = 42,
    rank_subdir: str = "adjusted_deep_minus_mostfrequent_driver_difference",
    xlabel_effect_label: str = "Adjusted",
    bootstrap_effects: Optional[pd.DataFrame] = None,
    interval_mode: str = "effect_row",
) -> None:
    """Plot REF/HET/ALT Δ adjusted |Bstd| panels in fixed feature order."""
    if ols_long.empty or not fixed_feature_order:
        return

    target_panels = (
        ("REF_F1score", "Homozygous Reference"),
        ("HET_F1score", "Heterozygous"),
        ("ALT_F1score", "Homozygous Alternate"),
    )
    summaries: dict[int, pd.DataFrame] = {}
    for idx, (target, _) in enumerate(target_panels):
        summary = _deep_minus_baseline_summary(
            ols_long,
            target_filter=[target],
            n_boot=n_boot,
            ci=ci,
            random_seed=random_seed + idx,
            fixed_feature_order=fixed_feature_order,
            bootstrap_effects=bootstrap_effects,
            interval_mode=interval_mode,
        )
        summaries[idx] = summary

    if all(summary.empty for summary in summaries.values()):
        return

    feature_labels = [
        FEATURE_DISPLAY_LABELS.get(feature, feature) for feature in fixed_feature_order
    ]
    y_positions = np.arange(len(fixed_feature_order), dtype=float)

    rc_params = {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 36,
        "axes.titlesize": 36,
        "axes.labelsize": 36,
        "xtick.labelsize": 36,
        "ytick.labelsize": 36,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with mpl.rc_context(rc_params):
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(18, 24),
            sharex=True,
            sharey=False,
        )

        for idx, (_, row_label) in enumerate(target_panels):
            ax = axes[idx]
            summary = summaries[idx]
            if summary.empty:
                delta = np.zeros(len(fixed_feature_order), dtype=float)
                ci_low = delta.copy()
                ci_high = delta.copy()
            else:
                panel = (
                    summary.set_index("feature")
                    .reindex(fixed_feature_order)
                    .reset_index()
                )
                delta = panel["delta_deep_minus_mostfrequent_abs_bstd"].to_numpy(
                    dtype=float
                )
                ci_low = panel["ci_low"].to_numpy(dtype=float)
                ci_high = panel["ci_high"].to_numpy(dtype=float)

            colors = np.where(delta >= 0, "#2E86AB", "#D95F02")
            ax.barh(y_positions, delta, color=colors, edgecolor="black", linewidth=0.8)

            finite = np.isfinite(delta) & np.isfinite(ci_low) & np.isfinite(ci_high)
            if not summary.empty and "ci_source" in panel.columns:
                finite &= panel["ci_source"].ne("none").to_numpy(dtype=bool)
            if finite.any():
                xerr_low = np.maximum(delta[finite] - ci_low[finite], 0.0)
                xerr_high = np.maximum(ci_high[finite] - delta[finite], 0.0)
                ax.errorbar(
                    x=delta[finite],
                    y=y_positions[finite],
                    xerr=np.vstack([xerr_low, xerr_high]),
                    fmt="none",
                    ecolor="black",
                    elinewidth=2.0,
                    capsize=5,
                    capthick=2.0,
                    zorder=3,
                )

            ax.axvline(0.0, color="black", linewidth=1.2)
            ax.set_xlim(-1.0, 1.0)
            ax.set_xticks(DIFFERENCE_AXIS_TICKS)
            ax.set_xticklabels(DIFFERENCE_AXIS_TICKLABELS)
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(feature_labels)
            ax.invert_yaxis()
            ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.55)
            ax.set_axisbelow(True)
            ax.set_title(row_label, fontweight="bold", pad=10)
            xlabel_effect = (
                f"{xlabel_effect_label.strip()} " if xlabel_effect_label.strip() else ""
            )
            has_intervals = (
                not summary.empty
                and "ci_source" in summary.columns
                and summary["ci_source"].ne("none").any()
            )
            if has_intervals and interval_mode == "dataset_refit":
                interval_suffix = f" ± {int(ci)}% dataset-bootstrap CI"
            elif has_intervals and interval_mode == "effect_row":
                interval_suffix = f" ± {int(ci)}% descriptive interval"
            elif has_intervals:
                interval_suffix = f" ± {int(ci)}% interval"
            else:
                interval_suffix = ""
            ax.set_xlabel(f"Δ Mean {xlabel_effect}|Bstd|{interval_suffix}")
            ax.set_ylabel("Summary Statistic")

        fig.tight_layout(h_pad=2.0)
        rank_dir = outdir / rank_subdir
        rank_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            rank_dir / f"{prefix}_combined_driver_difference_abs_bstd_f1_genotypes"
        )
        save_figure_dual(fig, out_path, dpi=dpi)
        vertical_out_path = (
            rank_dir
            / f"{prefix}_vertical_panels_driver_difference_abs_bstd_f1_genotypes"
        )
        save_figure_dual(fig, vertical_out_path, dpi=dpi)
        plt.close(fig)


def plot_adjusted_permutation_unified_heatmaps(
    res_long: pd.DataFrame,
    outdir: Path,
    *,
    dpi: int,
    value_col: str = "std_coef",
    q_col: str = "qvalue_bh",
    prefix: str = "adjusted_permutation",
    heatmap_subdir: str = "adjusted_permutation_unified_heatmaps",
    std_effect_label: str = "Adjusted βstd",
    partial_effect_label: str = "Adjusted partial R2",
) -> None:
    """Plot unified adjusted-effect heatmaps across all model × strategy conditions.

    Produces separate plots for:
        - REF F1-score
        - HET F1-score
        - ALT F1-score
        - MCC

    Columns are arranged as:
        model | strategy

    Significance is annotated with stars derived from q-values.

    Args:
        res_long (pd.DataFrame): Long adjusted regression results table.
        outdir (Path): Output directory.
        dpi (int): Figure DPI.
        value_col (str): Numeric value to visualize. Typically "std_coef" or
            "partial_r2".
        q_col (str): Q-value column for significance annotation.
        prefix (str): Filename prefix.
        heatmap_subdir (str): Output subdirectory.
        std_effect_label (str): Label for standardized coefficient heatmaps.
        partial_effect_label (str): Label for partial R2/R2 heatmaps.
    """
    if res_long.empty:
        return

    heat_dir = outdir / heatmap_subdir
    heat_dir.mkdir(parents=True, exist_ok=True)

    # Manuscript-oriented styling. These panels are designed to be placed at
    # roughly 6.5 inches wide on an 8.5 x 11 page with 1 inch margins.
    title_fs = TITLE_FONTSIZE
    axis_label_fs = AXIS_LABEL_FONTSIZE
    tick_fs = TICK_FONTSIZE
    cbar_tick_fs = CBAR_FONTSIZE
    cbar_label_fs = CBAR_FONTSIZE

    target_groups = {
        "ref_f1score": ["REF_F1score"],
        "het_f1score": ["HET_F1score"],
        "alt_f1score": ["ALT_F1score"],
        "mcc": ["MCC"],
    }

    panel_titles = {
        "ref_f1score": "Homozygous Reference F1-score",
        "het_f1score": "Heterozygous F1-score",
        "alt_f1score": "Homozygous Alternate F1-score",
        "mcc": "Matthews Correlation Coefficient",
    }

    for group_name, keep_targets in target_groups.items():
        sub = res_long.loc[res_long["target"].isin(keep_targets)].copy()
        if sub.empty:
            continue

        sub["condition"] = (
            sub["model"].astype(str) + " | " + sub["sim_strategy"].astype(str)
        )

        mat = sub.pivot(index="feature", columns="condition", values=value_col)
        qmat = sub.pivot(index="feature", columns="condition", values=q_col)

        if mat.empty:
            continue

        present_order = [f for f in FEATURE_CANONICAL if f in mat.index]
        mat = mat.reindex(index=present_order)
        qmat = qmat.reindex(index=present_order)

        sorted_cols = sorted(mat.columns, key=_condition_sort_key)
        mat = mat[sorted_cols]
        qmat = qmat[sorted_cols]

        mat.index = [FEATURE_DISPLAY_MAP.get(idx, idx) for idx in mat.index]
        qmat.index = mat.index

        annot = qmat.copy().astype(object)
        for i in range(qmat.shape[0]):
            for j in range(qmat.shape[1]):
                annot.iloc[i, j] = (
                    _sig_label_from_q(float(pd.to_numeric(qmat.iloc[i, j])))
                    if pd.notna(qmat.iloc[i, j])
                    else ""
                )

        fig_w = MANUSCRIPT_HEATMAP_WIDTH
        fig_h = max(MANUSCRIPT_HEATMAP_HEIGHT, len(mat.index) * 0.20 + 0.72)

        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(111)

        if value_col == "partial_r2":
            cmap = sns.color_palette("coolwarm", as_cmap=True)
            arr = mat.to_numpy(dtype=float)
            finite = arr[np.isfinite(arr)]
            vmax = float(np.nanmax(finite)) if finite.size else 1.0

            if vmax <= 0:
                vmax = 1.0

            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

            sns.heatmap(
                mat,
                cmap=cmap,
                norm=norm,
                linewidths=0.45,
                linecolor="white",
                annot=annot,
                fmt="",
                cbar_kws={
                    "label": partial_effect_label,
                    "aspect": 16,
                    "fraction": 0.032,
                    "pad": 0.025,
                },
                annot_kws={"fontsize": ANNOT_FONTSIZE - 2, "color": "black"},
                ax=ax,
            )
            title_metric = partial_effect_label
        else:
            cmap = sns.color_palette("coolwarm", as_cmap=True)
            arr = mat.to_numpy(dtype=float)
            finite = arr[np.isfinite(arr)]

            sns.heatmap(
                mat,
                cmap=cmap,
                center=0,
                vmin=-1.0,
                vmax=1.0,
                linewidths=0.45,
                linecolor="white",
                annot=annot,
                fmt="",
                cbar_kws={
                    "label": std_effect_label,
                    "aspect": 16,
                    "fraction": 0.032,
                    "pad": 0.025,
                    "ticks": [-1.0, -0.5, 0.0, 0.5, 1.0],
                },
                annot_kws={"fontsize": ANNOT_FONTSIZE - 2, "color": "black"},
                ax=ax,
            )
            title_metric = std_effect_label

        cbar = ax.collections[0].colorbar

        if cbar is not None:
            cbar.ax.tick_params(which="both", labelsize=cbar_tick_fs, pad=2)
            cbar.ax.set_yticks(cbar.ax.get_yticks())
            cbar.set_label(cbar.ax.get_ylabel(), fontsize=cbar_label_fs, labelpad=4)

        ax.set_title(
            f"{panel_titles[group_name]}: {title_metric}", pad=6, fontsize=title_fs
        )
        ax.set_ylabel("Summary Statistic", fontsize=axis_label_fs)
        ax.set_xlabel("")
        ax.tick_params(axis="both", which="major", labelsize=tick_fs, pad=2)
        ax.tick_params(axis="x", pad=2)
        ax.tick_params(axis="y", pad=2)

        _add_model_grouping_labels(
            ax,
            sorted_cols,
            label_y_pos=-0.22,
            model_text_y_offset=0.06,
            model_fontsize=MODEL_GROUP_FONTSIZE,
            strategy_fontsize=STRATEGY_TICK_FONTSIZE,
            line_width=0.65,
        )

        fig.subplots_adjust(left=0.18, right=0.91, top=0.88, bottom=0.36)
        out_path = heat_dir / f"{prefix}_{value_col}_{group_name}"
        save_figure_dual(fig, out_path, dpi=dpi)
        plt.close(fig)


def plot_vertical_unified_heatmap_panels(
    res_long: pd.DataFrame,
    outdir: Path,
    *,
    dpi: int,
    value_col: str,
    q_col: str,
    prefix: str,
    heatmap_subdir: str,
    std_effect_label: str,
    partial_effect_label: str,
) -> None:
    """Plot REF/HET/ALT unified heatmaps stacked vertically."""
    if res_long.empty:
        return

    target_groups = (
        ("ref_f1score", ["REF_F1score"], "Homozygous Reference"),
        ("het_f1score", ["HET_F1score"], "Heterozygous"),
        ("alt_f1score", ["ALT_F1score"], "Homozygous Alternate"),
    )
    panels: list[tuple[str, pd.DataFrame, pd.DataFrame, list[str]]] = []
    all_cols: set[str] = set()

    for _, keep_targets, title in target_groups:
        sub = res_long.loc[res_long["target"].isin(keep_targets)].copy()
        if sub.empty:
            panels.append((title, pd.DataFrame(), pd.DataFrame(), []))
            continue

        sub["condition"] = (
            sub["model"].astype(str) + " | " + sub["sim_strategy"].astype(str)
        )
        mat = sub.pivot(index="feature", columns="condition", values=value_col)
        qmat = sub.pivot(index="feature", columns="condition", values=q_col)
        present_order = [f for f in FEATURE_CANONICAL if f in mat.index]
        mat = mat.reindex(index=present_order)
        qmat = qmat.reindex(index=present_order)
        all_cols.update(str(col) for col in mat.columns)
        panels.append((title, mat, qmat, present_order))

    sorted_cols = sorted(all_cols, key=_condition_sort_key)
    if not sorted_cols:
        return

    file_value_col = value_col
    if file_value_col == "partial_r2":
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        effect_label = partial_effect_label
        finite_values = []
        for _, mat, _, _ in panels:
            if not mat.empty:
                arr = mat.reindex(columns=sorted_cols).to_numpy(dtype=float)
                finite_values.append(arr[np.isfinite(arr)])

        finite = np.concatenate(finite_values) if finite_values else np.array([])
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
        if vmax <= 0:
            vmax = 1.0
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        colorbar_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        effect_label = std_effect_label
        norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        colorbar_ticks = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

    heat_dir = outdir / heatmap_subdir
    heat_dir.mkdir(parents=True, exist_ok=True)

    rc_params = {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": VERTICAL_FONTSIZE,
        "axes.titlesize": VERTICAL_FONTSIZE,
        "axes.labelsize": VERTICAL_FONTSIZE,
        "xtick.labelsize": VERTICAL_FONTSIZE,
        "ytick.labelsize": VERTICAL_FONTSIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with mpl.rc_context(rc_params):
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(28, 36),
            sharex=False,
            sharey=False,
        )

        for panel_idx, (ax, (_, mat, qmat, _)) in enumerate(zip(axes, panels)):
            if mat.empty:
                ax.axis("off")
                continue

            mat = mat.reindex(columns=sorted_cols)
            qmat = qmat.reindex(columns=sorted_cols)
            display_mat = mat.copy()
            display_qmat = qmat.copy()
            display_mat.index = [
                FEATURE_DISPLAY_MAP.get(idx, idx) for idx in display_mat.index
            ]
            display_qmat.index = display_mat.index

            annot = display_qmat.copy().astype(object)
            for i in range(display_qmat.shape[0]):
                for j in range(display_qmat.shape[1]):
                    annot.iloc[i, j] = (
                        _sig_label_from_q(float(pd.to_numeric(display_qmat.iloc[i, j])))
                        if pd.notna(display_qmat.iloc[i, j])
                        else ""
                    )

            sns.heatmap(
                display_mat,
                cmap=cmap,
                norm=norm,
                linewidths=0.45,
                linecolor="white",
                annot=annot,
                fmt="",
                cbar=False,
                xticklabels=True,
                annot_kws={"color": "black"},
                ax=ax,
            )

            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(axis="y", pad=2, length=0)
            _add_model_grouping_labels(
                ax,
                sorted_cols,
                label_y_pos=-0.22,
                model_text_y_offset=0.06,
                model_fontsize=VERTICAL_FONTSIZE,
                strategy_fontsize=VERTICAL_FONTSIZE,
                line_width=1.1,
                wrap_model_labels=True,
            )
            ax.set_ylabel("Summary Statistic", fontsize=VERTICAL_FONTSIZE)
            ax.tick_params(axis="both", pad=2, length=4)
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=VERTICAL_FONTSIZE)

        fig.subplots_adjust(left=0.11, right=0.89, top=0.99, bottom=0.135, hspace=0.5)

        scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)  # type: ignore
        scalar_mappable.set_array([])
        for ax in axes:
            if not ax.collections:
                continue
            pos = ax.get_position()
            cbar_ax = fig.add_axes((0.915, pos.y0, 0.018, pos.height))
            cbar = fig.colorbar(scalar_mappable, cax=cbar_ax, ticks=colorbar_ticks)
            cbar.ax.tick_params(labelsize=VERTICAL_FONTSIZE, pad=3, length=7)
            cbar.set_label(effect_label, fontsize=VERTICAL_FONTSIZE, labelpad=8)

        out_path = heat_dir / f"{prefix}_{file_value_col}_vertical_f1_genotype_panels"
        save_figure_dual(fig, out_path, dpi=dpi)
        plt.close(fig)


def plot_condition_heatmaps(
    corr_long: pd.DataFrame,
    outdir: Path,
    *,
    dpi: int,
    prefix: str,
) -> None:
    """Plot feature x target heatmaps per (model, strategy).

    Args:
        corr_long (pd.DataFrame): Long correlation table with columns: model, sim_strategy, feature, target, rho.
        outdir (Path): Output directory.
        dpi (int): Figure DPI.
        prefix (str): Filename prefix.
    """
    if corr_long.empty:
        return

    heat_dir = outdir / "heatmaps_by_condition"
    heat_dir.mkdir(parents=True, exist_ok=True)

    for (model, strat), sub in corr_long.groupby(
        ["model", "sim_strategy"], dropna=False
    ):
        mat = sub.pivot(index="feature", columns="target", values="rho")
        present_order = [f for f in FEATURE_CANONICAL if f in mat.index]
        mat = mat.reindex(index=present_order)
        mat.index = [FEATURE_DISPLAY_LABELS.get(idx, idx) for idx in mat.index]

        # dynamic size
        fig_w = max(10, 0.6 * mat.shape[1] + 4)
        fig_h = max(6, 0.45 * mat.shape[0] + 2)

        cmap = sns.color_palette("coolwarm", n_colors=4, as_cmap=True)

        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(
            mat,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
            annot=True if (mat.shape[0] <= 14 and mat.shape[1] <= 18) else False,
            fmt=".2f",
            cbar_kws={"label": "Spearman ρ"},
        )
        plt.title(
            f"{prefix}: Feature ↔ Performance (ρ)\nModel={model} | Strategy={strat}"
        )
        plt.xlabel("Performance Target")
        plt.ylabel("Popgen Feature")
        plt.tight_layout()

        safe_model = re.sub(r"[^a-zA-Z0-9]+", "_", str(model)).strip("_")
        safe_strat = re.sub(r"[^a-zA-Z0-9]+", "_", str(strat)).strip("_")
        out_png = heat_dir / f"{prefix}_heatmap_{safe_model}__{safe_strat}.png"
        out_pdf = heat_dir / f"{prefix}_heatmap_{safe_model}__{safe_strat}.pdf"
        plt.savefig(out_png, dpi=dpi)
        plt.savefig(out_pdf, dpi=dpi)
        plt.close()


def plot_metric_metric_correlations(
    merged: pd.DataFrame,
    targets: List[str],
    outdir: Path,
    *,
    dpi: int,
    prefix: str,
) -> None:
    """Plot metric↔metric correlation matrices per (model, strategy).

    Args:
        merged (pd.DataFrame): Merged table including targets.
        targets (List[str]): Target columns.
        outdir (Path): Output directory.
        dpi (int): Figure DPI.
        prefix (str): Filename prefix.
    """
    mm_dir = outdir / "metric_metric_correlations"
    mm_dir.mkdir(parents=True, exist_ok=True)

    for (model, strat), sub in merged.groupby(["model", "sim_strategy"], dropna=False):
        # Use dataset-level rows; compute Spearman among targets
        M = sub[targets].apply(pd.to_numeric, errors="coerce")
        if M.dropna(how="all").shape[0] < 5:
            continue

        corr = M.corr(method="spearman", min_periods=5)

        cmap = sns.color_palette("coolwarm", n_colors=4, as_cmap=True)

        plt.figure(
            figsize=(max(10, 0.6 * corr.shape[0] + 4), max(8, 0.6 * corr.shape[1] + 3))
        )
        sns.heatmap(
            corr,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
            annot=True if corr.shape[0] <= 18 else False,
            fmt=".2f",
            cbar_kws={"label": "Spearman ρ"},
        )
        plt.title(
            f"{prefix}: Metric↔Metric (Spearman)\nModel={model} | Strategy={strat}"
        )
        plt.tight_layout()

        safe_model = re.sub(r"[^a-zA-Z0-9]+", "_", str(model)).strip("_")
        safe_strat = re.sub(r"[^a-zA-Z0-9]+", "_", str(strat)).strip("_")
        out_png = mm_dir / f"{prefix}_metric_metric_{safe_model}__{safe_strat}.png"
        out_pdf = mm_dir / f"{prefix}_metric_metric_{safe_model}__{safe_strat}.pdf"
        plt.savefig(out_png, dpi=dpi)
        plt.savefig(out_pdf, dpi=dpi)
        plt.close()


def generate_no_control_supplement(
    merged: pd.DataFrame,
    features: List[str],
    controls: List[str],
    targets: List[str],
    outdir: Path,
    *,
    min_n: int,
    dpi: int,
    fdr_scope: str,
    use_permutation: bool = False,
    load_permutations: bool = False,
    n_permutations: int = 0,
    random_seed: Optional[int] = None,
    driver_bootstrap_mode: str = "effect_row",
    driver_bootstrap_reps: int = 10000,
    driver_bootstrap_n_jobs: int = 1,
    driver_bootstrap_backend: str = "process",
    driver_bootstrap_show_progress: bool = False,
) -> pd.DataFrame:
    """Generate supplementary no-control OLS driver-ranking outputs."""
    if use_permutation and n_permutations <= 0:
        raise ValueError("--make-no-control-permutation requires --n-permutations > 0.")

    no_control_features = [feature for feature in features if feature not in controls]
    output_name = (
        "no_control_permutation_ols_long.csv"
        if use_permutation
        else "no_control_ols_long.csv"
    )
    output_path = outdir / output_name

    if use_permutation and load_permutations and output_path.exists():
        no_control_df = pd.read_csv(output_path)
    else:
        no_control_df = compute_no_control_ols(
            merged,
            no_control_features,
            targets,
            min_n=min_n,
            n_permutations=n_permutations if use_permutation else 0,
            random_seed=random_seed,
        )
        no_control_df = _add_fdr(
            no_control_df,
            p_col="perm_pvalue" if use_permutation else "pvalue_hc3",
            scope=fdr_scope,
        )

    no_control_df = add_neutral_rows_for_constant_targets(
        no_control_df,
        merged,
        no_control_features,
        targets,
        [],
        min_n=min_n,
        n_permutations=n_permutations if use_permutation else 0,
    )
    no_control_df.to_csv(output_path, index=False)

    if use_permutation:
        sig_path = outdir / "no_control_permutation_ols_significant_q_lt_0_05.csv"
        sig_df = no_control_df.loc[
            no_control_df["qvalue_bh"].notna() & (no_control_df["qvalue_bh"] < 0.05)
        ].copy()
        sig_df.to_csv(sig_path, index=False)

    if no_control_df.empty:
        return no_control_df

    fixed_feature_order = [
        feature
        for feature in FEATURE_CANONICAL
        if feature in set(no_control_df["feature"])
    ]
    no_control_bootstrap_effects = pd.DataFrame()
    if driver_bootstrap_mode == "dataset_refit" and driver_bootstrap_reps > 0:
        no_control_bootstrap_effects = compute_dataset_refit_bootstrap_effects(
            merged,
            no_control_features,
            targets,
            [],
            min_n=min_n,
            n_boot=driver_bootstrap_reps,
            random_seed=(random_seed if random_seed is not None else 42) + 200_000,
            ols_mode="no_control_bootstrap_refit",
            n_jobs=driver_bootstrap_n_jobs,
            backend=driver_bootstrap_backend,
            show_progress=driver_bootstrap_show_progress,
            progress_label="No-control dataset bootstrap",
        )
        if not no_control_bootstrap_effects.empty:
            no_control_bootstrap_effects.to_csv(
                outdir / "no_control_dataset_refit_bootstrap_ols_long.csv",
                index=False,
            )

    plot_adjusted_permutation_unified_heatmaps(
        no_control_df,
        outdir,
        dpi=dpi,
        value_col="std_coef",
        q_col="qvalue_bh",
        prefix="no_controls",
        heatmap_subdir="no_control_unified_heatmaps",
        std_effect_label="No-control Bstd",
        partial_effect_label="No-control R2",
    )
    plot_vertical_unified_heatmap_panels(
        no_control_df,
        outdir,
        dpi=dpi,
        value_col="std_coef",
        q_col="qvalue_bh",
        prefix="no_controls",
        heatmap_subdir="no_control_unified_heatmaps",
        std_effect_label="No-control Bstd",
        partial_effect_label="No-control R2",
    )

    plot_adjusted_permutation_unified_heatmaps(
        no_control_df,
        outdir,
        dpi=dpi,
        value_col="partial_r2",
        q_col="qvalue_bh",
        prefix="no_controls",
        heatmap_subdir="no_control_unified_heatmaps",
        std_effect_label="No-control Bstd",
        partial_effect_label="No-control R2",
    )
    plot_vertical_unified_heatmap_panels(
        no_control_df,
        outdir,
        dpi=dpi,
        value_col="partial_r2",
        q_col="qvalue_bh",
        prefix="no_controls",
        heatmap_subdir="no_control_unified_heatmaps",
        std_effect_label="No-control Bstd",
        partial_effect_label="No-control R2",
    )

    driver_ranking_targets = (
        (None, None),
        (["REF_F1score"], "Homozygous Reference F1-score"),
        (["HET_F1score"], "Heterozygous F1-score"),
        (["ALT_F1score"], "Homozygous Alternate F1-score"),
        (["MCC"], "MCC"),
    )
    no_control_model_groups = (
        ("no_controls", None, None),
        (
            "no_controls_deterministic",
            list(DETERMINISTIC_MODELS),
            "MostFrequent Deterministic Baseline",
        ),
        (
            "no_controls_deep_learning",
            list(DEEP_LEARNING_MODELS),
            "Deep Learning Models",
        ),
    )
    no_control_fixed_order_model_groups = (
        ("no_controls_fixed_order", None, None),
        (
            "no_controls_deterministic_fixed_order",
            list(DETERMINISTIC_MODELS),
            "MostFrequent Deterministic Baseline",
        ),
        (
            "no_controls_deep_learning_fixed_order",
            list(DEEP_LEARNING_MODELS),
            "Deep Learning Models",
        ),
    )

    for prefix, model_filter, model_group_label in no_control_model_groups:
        for value_col in ("std_coef", "partial_r2"):
            for target_filter, target_label in driver_ranking_targets:
                plot_adjusted_driver_ranking(
                    no_control_df,
                    outdir,
                    dpi=dpi,
                    prefix=prefix,
                    value_col=value_col,
                    target_filter=target_filter,
                    target_label=target_label,
                    model_filter=model_filter,
                    model_group_label=model_group_label,
                    rank_subdir="no_control_driver_ranking",
                    analysis_label="No-Control",
                    effect_label="",
                    bootstrap_effects=no_control_bootstrap_effects,
                    interval_mode=driver_bootstrap_mode,
                    n_boot=driver_bootstrap_reps,
                )

    for prefix, model_filter, model_group_label in no_control_fixed_order_model_groups:
        for value_col in ("std_coef", "partial_r2"):
            for target_filter, target_label in driver_ranking_targets:
                plot_adjusted_driver_ranking(
                    no_control_df,
                    outdir,
                    dpi=dpi,
                    prefix=prefix,
                    value_col=value_col,
                    target_filter=target_filter,
                    target_label=target_label,
                    model_filter=model_filter,
                    model_group_label=model_group_label,
                    fixed_feature_order=fixed_feature_order,
                    rank_subdir="no_control_driver_ranking",
                    analysis_label="No-Control",
                    effect_label="",
                    bootstrap_effects=no_control_bootstrap_effects,
                    interval_mode=driver_bootstrap_mode,
                    n_boot=driver_bootstrap_reps,
                )

    for value_col in ("std_coef", "partial_r2"):
        plot_combined_driver_ranking_grid(
            no_control_df,
            outdir,
            dpi=dpi,
            prefix="no_controls_fixed_order",
            value_col=value_col,
            fixed_feature_order=fixed_feature_order,
            rank_subdir="no_control_driver_ranking",
            effect_label="",
            bootstrap_effects=no_control_bootstrap_effects,
            interval_mode=driver_bootstrap_mode,
            n_boot=driver_bootstrap_reps,
        )

    generate_no_control_deep_baseline_difference_plots(
        no_control_df,
        outdir,
        dpi=dpi,
        bootstrap_effects=no_control_bootstrap_effects,
        interval_mode=driver_bootstrap_mode,
        n_boot=driver_bootstrap_reps,
        random_seed=random_seed if random_seed is not None else 42,
    )

    return no_control_df


def generate_adjusted_deep_baseline_difference_plots(
    adjusted_perm_df: pd.DataFrame,
    outdir: Path,
    *,
    dpi: int,
    bootstrap_effects: Optional[pd.DataFrame] = None,
    interval_mode: str = "effect_row",
    n_boot: int = 10000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate adjusted |Bstd| difference plots for deep learning vs MostFrequent."""
    if adjusted_perm_df.empty:
        return pd.DataFrame()

    fixed_feature_order = [
        feature
        for feature in FEATURE_CANONICAL
        if feature in set(adjusted_perm_df["feature"])
    ]
    driver_ranking_targets = (
        (None, None),
        (["REF_F1score"], "Homozygous Reference F1-score"),
        (["HET_F1score"], "Heterozygous F1-score"),
        (["ALT_F1score"], "Homozygous Alternate F1-score"),
        (["MCC"], "MCC"),
    )

    summaries = []
    for target_filter, target_label in driver_ranking_targets:
        summary = plot_deep_minus_baseline_driver_difference(
            adjusted_perm_df,
            outdir,
            dpi=dpi,
            prefix="adjusted_deep_minus_mostfrequent",
            target_filter=target_filter,
            target_label=target_label,
            bootstrap_effects=bootstrap_effects,
            interval_mode=interval_mode,
            n_boot=n_boot,
            random_seed=random_seed,
        )
        if not summary.empty:
            summary["target_group"] = target_label or "All targets"
            summary["feature_order"] = "ranked_abs_delta"
            summaries.append(summary)

    for target_filter, target_label in driver_ranking_targets:
        summary = plot_deep_minus_baseline_driver_difference(
            adjusted_perm_df,
            outdir,
            dpi=dpi,
            prefix="adjusted_deep_minus_mostfrequent_fixed_order",
            target_filter=target_filter,
            target_label=target_label,
            fixed_feature_order=fixed_feature_order,
            bootstrap_effects=bootstrap_effects,
            interval_mode=interval_mode,
            n_boot=n_boot,
            random_seed=random_seed,
        )
        if not summary.empty:
            summary["target_group"] = target_label or "All targets"
            summary["feature_order"] = "fixed_feature_order"
            summaries.append(summary)

    plot_combined_deep_minus_baseline_driver_difference(
        adjusted_perm_df,
        outdir,
        dpi=dpi,
        prefix="adjusted_deep_minus_mostfrequent_fixed_order",
        fixed_feature_order=fixed_feature_order,
        bootstrap_effects=bootstrap_effects,
        interval_mode=interval_mode,
        n_boot=n_boot,
        random_seed=random_seed,
    )

    if not summaries:
        return pd.DataFrame()

    out = pd.concat(summaries, ignore_index=True, sort=False)
    out.to_csv(
        outdir / "adjusted_deep_minus_mostfrequent_driver_difference_summary.csv",
        index=False,
    )
    return out


def generate_no_control_deep_baseline_difference_plots(
    no_control_df: pd.DataFrame,
    outdir: Path,
    *,
    dpi: int,
    bootstrap_effects: Optional[pd.DataFrame] = None,
    interval_mode: str = "effect_row",
    n_boot: int = 10000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate no-control |Bstd| difference plots for deep learning vs MostFrequent."""
    if no_control_df.empty:
        return pd.DataFrame()

    fixed_feature_order = [
        feature
        for feature in FEATURE_CANONICAL
        if feature in set(no_control_df["feature"])
    ]
    driver_ranking_targets = (
        (None, None),
        (["REF_F1score"], "Homozygous Reference F1-score"),
        (["HET_F1score"], "Heterozygous F1-score"),
        (["ALT_F1score"], "Homozygous Alternate F1-score"),
        (["MCC"], "MCC"),
    )
    rank_subdir = "no_control_deep_minus_mostfrequent_driver_difference"

    summaries = []
    for target_filter, target_label in driver_ranking_targets:
        summary = plot_deep_minus_baseline_driver_difference(
            no_control_df,
            outdir,
            dpi=dpi,
            prefix="no_controls_deep_minus_mostfrequent",
            target_filter=target_filter,
            target_label=target_label,
            rank_subdir=rank_subdir,
            title_effect_label="No-Control",
            xlabel_effect_label="no-control",
            bootstrap_effects=bootstrap_effects,
            interval_mode=interval_mode,
            n_boot=n_boot,
            random_seed=random_seed,
        )
        if not summary.empty:
            summary["target_group"] = target_label or "All targets"
            summary["feature_order"] = "ranked_abs_delta"
            summaries.append(summary)

    for target_filter, target_label in driver_ranking_targets:
        summary = plot_deep_minus_baseline_driver_difference(
            no_control_df,
            outdir,
            dpi=dpi,
            prefix="no_controls_deep_minus_mostfrequent_fixed_order",
            target_filter=target_filter,
            target_label=target_label,
            fixed_feature_order=fixed_feature_order,
            rank_subdir=rank_subdir,
            title_effect_label="No-Control",
            xlabel_effect_label="no-control",
            bootstrap_effects=bootstrap_effects,
            interval_mode=interval_mode,
            n_boot=n_boot,
            random_seed=random_seed,
        )
        if not summary.empty:
            summary["target_group"] = target_label or "All targets"
            summary["feature_order"] = "fixed_feature_order"
            summaries.append(summary)

    plot_combined_deep_minus_baseline_driver_difference(
        no_control_df,
        outdir,
        dpi=dpi,
        prefix="no_controls_deep_minus_mostfrequent_fixed_order",
        fixed_feature_order=fixed_feature_order,
        rank_subdir=rank_subdir,
        xlabel_effect_label="No-Control",
        bootstrap_effects=bootstrap_effects,
        interval_mode=interval_mode,
        n_boot=n_boot,
        random_seed=random_seed,
    )

    if not summaries:
        return pd.DataFrame()

    out = pd.concat(summaries, ignore_index=True, sort=False)
    out.to_csv(
        outdir / "no_control_deep_minus_mostfrequent_driver_difference_summary.csv",
        index=False,
    )
    return out


def main() -> None:
    """Entry point."""
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.make_partial:
        print(
            "[WARN] --make-partial is deprecated and ignored; adjusted OLS "
            "partial R2 is always reported, and partial correlations are not "
            "computed."
        )

    stats = load_stats(args.stats)
    perf = load_metrics_long(args.metrics)

    # Merge on join_key (inner: only datasets present in both)
    merged = perf.merge(stats, on="join_key", how="inner", suffixes=("", "_stats"))

    # Targets present
    targets = [
        c
        for c in merged.columns
        if c in list(GLOBAL_TARGETS) or any(c.startswith(f"{g}_") for g in GENOTYPES)
    ]
    # Features present
    features = [f for f in FEATURE_CANONICAL if f in merged.columns]

    # Controls: allow names like Missingness or Missingness_mean; we standardize to canonical in load_stats
    controls = [c for c in args.controls if c in merged.columns]
    focal_features = [f for f in features if f not in controls]
    no_control_output_name = (
        "no_control_permutation_ols_long.csv"
        if args.make_no_control_permutation
        else "no_control_ols_long.csv"
    )
    no_control_load_perms = args.no_control_load_permutations or args.load_permutations

    if args.only_no_control_supplement:
        no_control_df = generate_no_control_supplement(
            merged,
            features,
            controls,
            targets,
            outdir,
            min_n=args.min_n,
            dpi=args.dpi,
            fdr_scope=args.fdr_scope,
            use_permutation=args.make_no_control_permutation,
            load_permutations=no_control_load_perms,
            n_permutations=args.n_permutations,
            random_seed=args.random_seed,
            driver_bootstrap_mode=args.driver_bootstrap_mode,
            driver_bootstrap_reps=args.driver_bootstrap_reps,
            driver_bootstrap_n_jobs=args.driver_bootstrap_n_jobs,
            driver_bootstrap_backend=args.driver_bootstrap_backend,
            driver_bootstrap_show_progress=not args.no_driver_bootstrap_progress,
        )
        print(f"[OK] wrote: {outdir / no_control_output_name}")
        print(
            "[OK] no-control supplemental plots in: "
            f"{outdir / 'no_control_driver_ranking'}"
        )
        print(f"[OK] no-control rows: {len(no_control_df)}")
        return

    if args.only_deep_baseline_difference:
        adjusted_path = outdir / "adjusted_permutation_ols_long.csv"
        if not adjusted_path.exists():
            raise FileNotFoundError(
                f"Cannot generate difference plots; missing {adjusted_path}"
            )
        adjusted_perm_df = pd.read_csv(adjusted_path)
        adjusted_perm_df = _drop_control_feature_rows(adjusted_perm_df, controls)
        adjusted_bootstrap_effects = pd.DataFrame()
        if (
            args.driver_bootstrap_mode == "dataset_refit"
            and args.driver_bootstrap_reps > 0
        ):
            adjusted_bootstrap_effects = compute_dataset_refit_bootstrap_effects(
                merged,
                focal_features,
                targets,
                controls,
                min_n=args.min_n,
                n_boot=args.driver_bootstrap_reps,
                random_seed=args.random_seed + 100_000,
                ols_mode="adjusted_bootstrap_refit",
                n_jobs=args.driver_bootstrap_n_jobs,
                backend=args.driver_bootstrap_backend,
                show_progress=not args.no_driver_bootstrap_progress,
                progress_label="Adjusted dataset bootstrap",
            )
            if not adjusted_bootstrap_effects.empty:
                adjusted_bootstrap_effects.to_csv(
                    outdir / "adjusted_dataset_refit_bootstrap_ols_long.csv",
                    index=False,
                )
        difference_df = generate_adjusted_deep_baseline_difference_plots(
            adjusted_perm_df,
            outdir,
            dpi=args.dpi,
            bootstrap_effects=adjusted_bootstrap_effects,
            interval_mode=args.driver_bootstrap_mode,
            n_boot=args.driver_bootstrap_reps,
            random_seed=args.random_seed,
        )
        print(
            "[OK] adjusted deep-vs-MostFrequent difference plots in: "
            f"{outdir / 'adjusted_deep_minus_mostfrequent_driver_difference'}"
        )
        print(f"[OK] difference rows: {len(difference_df)}")
        return

    # Save merged snapshot
    merged.to_csv(outdir / "merged_perf_stats.csv", index=False)

    # Spearman
    spearman_df = compute_feature_target_correlations(
        merged, focal_features, targets, min_n=args.min_n
    )
    spearman_df.to_csv(outdir / "spearman_long.csv", index=False)

    plot_condition_heatmaps(spearman_df, outdir, dpi=args.dpi, prefix="spearman")
    plot_metric_metric_correlations(
        merged, targets, outdir, dpi=args.dpi, prefix="spearman"
    )

    fn = "adjusted_permutation_ols_long.csv"
    load_perms = args.load_permutations and Path(outdir / fn).exists()

    # Adjusted regression with permutation inference
    if args.make_adjusted_permutation and controls and not load_perms:
        adjusted_perm_df = compute_adjusted_permutation_ols(
            merged,
            focal_features,
            targets,
            controls,
            min_n=args.min_n,
            n_permutations=args.n_permutations,
            random_seed=args.random_seed,
        )

        adjusted_perm_df = _add_fdr(
            adjusted_perm_df,
            p_col="perm_pvalue",
            scope=args.fdr_scope,
        )

        adjusted_perm_df.to_csv(outdir / fn, index=False)

        fn_sig = "adjusted_permutation_ols_significant_q_lt_0_05.csv"

        sig_df = adjusted_perm_df.loc[
            adjusted_perm_df["qvalue_bh"].notna()
            & (adjusted_perm_df["qvalue_bh"] < 0.05)
        ].copy()

        sig_df.to_csv(outdir / fn_sig, index=False)

    elif args.make_adjusted_permutation and controls and load_perms:
        adjusted_perm_df = pd.read_csv(outdir / fn)
        adjusted_perm_df = _drop_control_feature_rows(adjusted_perm_df, controls)

    if args.make_adjusted_permutation and controls:
        adjusted_perm_df = add_neutral_rows_for_constant_targets(
            adjusted_perm_df,
            merged,
            focal_features,
            targets,
            controls,
            min_n=args.min_n,
            n_permutations=args.n_permutations,
        )
        adjusted_perm_df = _drop_control_feature_rows(adjusted_perm_df, controls)
        adjusted_perm_df.to_csv(outdir / fn, index=False)

        plot_adjusted_permutation_unified_heatmaps(
            adjusted_perm_df,
            outdir,
            dpi=args.dpi,
            value_col="std_coef",
            q_col="qvalue_bh",
            prefix="adjusted_permutation",
        )

        plot_adjusted_permutation_unified_heatmaps(
            adjusted_perm_df,
            outdir,
            dpi=args.dpi,
            value_col="partial_r2",
            q_col="qvalue_bh",
            prefix="adjusted_permutation",
        )
        plot_vertical_unified_heatmap_panels(
            adjusted_perm_df,
            outdir,
            dpi=args.dpi,
            value_col="partial_r2",
            q_col="qvalue_bh",
            prefix="adjusted_permutation",
            heatmap_subdir="adjusted_permutation_unified_heatmaps",
            std_effect_label="Adjusted Bstd",
            partial_effect_label="Adjusted Partial R²",
        )

        adjusted_bootstrap_effects = pd.DataFrame()
        if (
            args.driver_bootstrap_mode == "dataset_refit"
            and args.driver_bootstrap_reps > 0
        ):
            adjusted_bootstrap_effects = compute_dataset_refit_bootstrap_effects(
                merged,
                focal_features,
                targets,
                controls,
                min_n=args.min_n,
                n_boot=args.driver_bootstrap_reps,
                random_seed=args.random_seed + 100_000,
                ols_mode="adjusted_bootstrap_refit",
                n_jobs=args.driver_bootstrap_n_jobs,
                backend=args.driver_bootstrap_backend,
                show_progress=not args.no_driver_bootstrap_progress,
                progress_label="Adjusted dataset bootstrap",
            )
            if not adjusted_bootstrap_effects.empty:
                adjusted_bootstrap_effects.to_csv(
                    outdir / "adjusted_dataset_refit_bootstrap_ols_long.csv",
                    index=False,
                )

        driver_ranking_model_groups = (
            ("adjusted_permutation", None, None),
            (
                "adjusted_permutation_deterministic",
                list(DETERMINISTIC_MODELS),
                "MostFrequent Deterministic Baseline",
            ),
            (
                "adjusted_permutation_deep_learning",
                list(DEEP_LEARNING_MODELS),
                "Deep Learning Models",
            ),
        )
        driver_ranking_targets = (
            (None, None),
            (["REF_F1score"], "Homozygous Reference F1-score"),
            (["HET_F1score"], "Heterozygous F1-score"),
            (["ALT_F1score"], "Homozygous Alternate F1-score"),
            (["MCC"], "MCC"),
        )

        for prefix, model_filter, model_group_label in driver_ranking_model_groups:
            for value_col in ("std_coef", "partial_r2"):
                for target_filter, target_label in driver_ranking_targets:
                    plot_adjusted_driver_ranking(
                        adjusted_perm_df,
                        outdir,
                        dpi=args.dpi,
                        prefix=prefix,
                        value_col=value_col,
                        target_filter=target_filter,
                        target_label=target_label,
                        model_filter=model_filter,
                        model_group_label=model_group_label,
                        bootstrap_effects=adjusted_bootstrap_effects,
                        interval_mode=args.driver_bootstrap_mode,
                        n_boot=args.driver_bootstrap_reps,
                    )

        fixed_driver_feature_order = [
            feature
            for feature in FEATURE_CANONICAL
            if feature in set(adjusted_perm_df["feature"])
        ]
        fixed_order_driver_ranking_model_groups = (
            ("adjusted_permutation_fixed_order", None, None),
            (
                "adjusted_permutation_deterministic_fixed_order",
                list(DETERMINISTIC_MODELS),
                "MostFrequent Deterministic Baseline",
            ),
            (
                "adjusted_permutation_deep_learning_fixed_order",
                list(DEEP_LEARNING_MODELS),
                "Deep Learning Models",
            ),
        )

        for (
            prefix,
            model_filter,
            model_group_label,
        ) in fixed_order_driver_ranking_model_groups:
            for value_col in ("std_coef", "partial_r2"):
                for target_filter, target_label in driver_ranking_targets:
                    plot_adjusted_driver_ranking(
                        adjusted_perm_df,
                        outdir,
                        dpi=args.dpi,
                        prefix=prefix,
                        value_col=value_col,
                        target_filter=target_filter,
                        target_label=target_label,
                        model_filter=model_filter,
                        model_group_label=model_group_label,
                        fixed_feature_order=fixed_driver_feature_order,
                        bootstrap_effects=adjusted_bootstrap_effects,
                        interval_mode=args.driver_bootstrap_mode,
                        n_boot=args.driver_bootstrap_reps,
                    )

        for value_col in ("std_coef", "partial_r2"):
            plot_combined_driver_ranking_grid(
                adjusted_perm_df,
                outdir,
                dpi=args.dpi,
                prefix="adjusted_permutation_fixed_order",
                value_col=value_col,
                fixed_feature_order=fixed_driver_feature_order,
                bootstrap_effects=adjusted_bootstrap_effects,
                interval_mode=args.driver_bootstrap_mode,
                n_boot=args.driver_bootstrap_reps,
            )

        generate_adjusted_deep_baseline_difference_plots(
            adjusted_perm_df,
            outdir,
            dpi=args.dpi,
            bootstrap_effects=adjusted_bootstrap_effects,
            interval_mode=args.driver_bootstrap_mode,
            n_boot=args.driver_bootstrap_reps,
            random_seed=args.random_seed,
        )

        covariate_diag_df = compute_covariate_diagnostic_ols(
            merged,
            controls,
            targets,
            min_n=args.min_n,
        )
        covariate_diag_df = _add_fdr(
            covariate_diag_df,
            p_col="pvalue_hc3",
            scope=args.fdr_scope,
        )
        covariate_diag_df.to_csv(
            outdir / "covariate_diagnostic_ols_long.csv", index=False
        )

        covariate_diagnostic_model_groups = (
            ("covariate_diagnostic", None, "Covariate Diagnostics"),
            (
                "covariate_diagnostic_deterministic",
                list(DETERMINISTIC_MODELS),
                "Covariate Diagnostics: MostFrequent Deterministic Baseline",
            ),
            (
                "covariate_diagnostic_deep_learning",
                list(DEEP_LEARNING_MODELS),
                "Covariate Diagnostics: Deep Learning Models",
            ),
        )

        for (
            prefix,
            model_filter,
            model_group_label,
        ) in covariate_diagnostic_model_groups:
            for value_col in ("std_coef", "partial_r2"):
                for target_filter, target_label in driver_ranking_targets:
                    plot_adjusted_driver_ranking(
                        covariate_diag_df,
                        outdir,
                        dpi=args.dpi,
                        prefix=prefix,
                        value_col=value_col,
                        target_filter=target_filter,
                        target_label=target_label,
                        model_filter=model_filter,
                        model_group_label=model_group_label,
                    )

    if args.make_no_control_supplement or args.make_no_control_permutation:
        no_control_df = generate_no_control_supplement(
            merged,
            features,
            controls,
            targets,
            outdir,
            min_n=args.min_n,
            dpi=args.dpi,
            fdr_scope=args.fdr_scope,
            use_permutation=args.make_no_control_permutation,
            load_permutations=no_control_load_perms,
            n_permutations=args.n_permutations,
            random_seed=args.random_seed,
            driver_bootstrap_mode=args.driver_bootstrap_mode,
            driver_bootstrap_reps=args.driver_bootstrap_reps,
            driver_bootstrap_n_jobs=args.driver_bootstrap_n_jobs,
            driver_bootstrap_backend=args.driver_bootstrap_backend,
            driver_bootstrap_show_progress=not args.no_driver_bootstrap_progress,
        )
        print(f"[OK] wrote: {outdir / no_control_output_name}")
        print(
            "[OK] no-control supplemental plots in: "
            f"{outdir / 'no_control_driver_ranking'}"
        )
        print(f"[OK] no-control rows: {len(no_control_df)}")

    # Quick console summary
    print(f"[OK] wrote: {outdir / 'spearman_long.csv'}")

    if args.make_adjusted_permutation and controls and not load_perms:
        print(f"[OK] wrote: {outdir / fn} and generated plots")

    elif args.make_adjusted_permutation and controls and load_perms:
        print(f"[OK] loaded: {outdir / fn} and generated plots")

    else:
        print("[SKIP] adjusted permutation OLS (missing controls or not toggled)")

    print(f"[OK] plots in: {outdir}")


if __name__ == "__main__":
    main()
