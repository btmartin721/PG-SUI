#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recursively aggregate per-locus summary stats across many datasets and test which metrics differ by dataset.
Includes diagnostic debugging for dataset completion gating.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

# Use non-interactive backend
plt.switch_backend("Agg")

# Updated regex to capture multiple naming conventions
_RESULTS_KEY_RE = re.compile(
    r"((?:results|dataset|sim|run|rep)[_]?\d+)", flags=re.IGNORECASE
)
_PYARROW_FALLBACK_WARNED = False

EXPECTED_MODELS = {
    "MostFrequent",
    "RefAllele",
    "Autoencoder",
    "VAE",
    "NLPCA",
    "UBP",
}
# --- STRICT REQUIREMENTS ---
ALL_STRATEGIES = {
    "Nonrandom",
    "Nonrandom Weighted",
    "Random",
    "Random Weighted",
    "Random Weighted Inv",
}

PLOT_FONTSIZE = 14
PLOT_TITLESIZE = 16
mpl_params = {
    "xtick.labelsize": PLOT_FONTSIZE,
    "ytick.labelsize": PLOT_FONTSIZE,
    "legend.fontsize": PLOT_FONTSIZE,
    "figure.titlesize": PLOT_TITLESIZE,
    "figure.facecolor": "white",
    "figure.dpi": 300,
    "font.size": PLOT_FONTSIZE,
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": PLOT_FONTSIZE,
    "axes.labelsize": PLOT_FONTSIZE,
    "axes.grid": False,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

mpl.rcParams.update(mpl_params)
plt.rcParams.update(mpl_params)

METRIC_MAP = {
    "F_inbreeding": r"$\it{F}_{\it{is}}$",
    "ThetaWatt": r"$\it{\theta}_{\it{w}}$",
    "Pi": r"$\it{\pi}$",
    "TajimaD": r"Tajima's $\it{D}$",
    "Missingness": "Missingness",
    "Ho": r"$\it{H}_{\it{o}}$",
    "He": r"$\it{H}_{\it{e}}$",
    "HetzPositions": r"Heterozygous Sites",
    "SegSites": r"$\it{S}$",
    "Singletons": r"Singletons",
    "Hap": r"$\it{N}_{\it{hap}}$",
    "Hd": r"$\it{H}_{\it{d}}$",
    "Sample_Size": r"Sample Size",
    "MAF": r"MAF",
}


@dataclass
class DatasetFiles:
    dataset_id: str
    locus_csv: Path
    summary_json: Optional[Path]


def analyze_deviations(
    desc: pd.DataFrame, metrics: Sequence[str], out_dir: Path, out_prefix: str
) -> pd.DataFrame:
    """Identifies outlier datasets that deviate from the global mean for each metric.

    Transforms summary statistic columns into a long format to calculate the global mean and standard deviation per metric vectorially. Computes Z-scores to flag significant 2-sigma and 3-sigma deviations.

    Args:
        desc: A DataFrame containing per-dataset descriptive statistics, including columns formatted as '{metric}_mean'.
        metrics: A sequence of target metric names to analyze.
        out_dir: The directory where the resulting CSV will be saved.
        out_prefix: A string prefix for the output filename.

    Returns:
        A DataFrame containing outlier rankings sorted by absolute Z-score, or an empty DataFrame if no required columns are found.
    """
    mean_cols = [f"{m}_mean" for m in metrics if f"{m}_mean" in desc.columns]

    if not mean_cols:
        return pd.DataFrame()

    # Subset and reshape into a long format DataFrame
    sub_desc = desc[["dataset"] + mean_cols].copy()
    long_df = sub_desc.melt(
        id_vars=["dataset"],
        value_vars=mean_cols,
        var_name="metric_mean",
        value_name="dataset_mean",
    )

    # Strip the '_mean' suffix to isolate the core metric name
    long_df["metric"] = long_df["metric_mean"].str.replace("_mean", "", regex=False)

    # Calculate grouped statistics and broadcast them back to the original rows
    grouped = long_df.groupby("metric")["dataset_mean"]
    long_df["global_mean_of_means"] = grouped.transform("mean")
    long_df["std_dev"] = grouped.transform(lambda x: x.std(ddof=1))

    # Filter out metrics with zero variance to prevent division by zero errors
    long_df = long_df[long_df["std_dev"] > 0].copy()

    if long_df.empty:
        return pd.DataFrame()

    # Vectorized computation of Z-scores and significance masks
    long_df["z_score"] = (
        long_df["dataset_mean"] - long_df["global_mean_of_means"]
    ) / long_df["std_dev"]

    long_df["abs_z_score"] = long_df["z_score"].abs()
    long_df["is_outlier_2sigma"] = long_df["abs_z_score"] > 2.0
    long_df["is_outlier_3sigma"] = long_df["abs_z_score"] > 3.0

    # Define final output structure and sort
    out_cols = [
        "dataset",
        "metric",
        "dataset_mean",
        "global_mean_of_means",
        "z_score",
        "abs_z_score",
        "is_outlier_2sigma",
        "is_outlier_3sigma",
    ]

    dev_df = long_df[out_cols].sort_values("abs_z_score", ascending=False)

    out_path = out_dir / f"{out_prefix}_outlier_rankings.csv"
    dev_df.to_csv(out_path, index=False)
    print(
        f"Wrote outlier analysis (datasets contributing most to variance): {out_path}"
    )

    return dev_df


def _normalize_model_name(val: str) -> str:
    """Normalize model name variants to canonical names."""
    s = re.sub(r"[^a-z0-9]+", "", str(val).strip().lower())
    aliases = {
        "mostfrequent": "MostFrequent",
        "imputemostfrequent": "MostFrequent",
        "mostfreq": "MostFrequent",
        "mostfrequentbaseline": "MostFrequent",  # Synced with plot_dnasp_pgsui.py
        "refallele": "RefAllele",
        "imputerefallele": "RefAllele",
        "autoencoder": "Autoencoder",
        "imputeautoencoder": "Autoencoder",
        "vae": "VAE",
        "imputevae": "VAE",
        "nlpca": "NLPCA",
        "imputenlpca": "NLPCA",
        "ubp": "UBP",
        "imputeubp": "UBP",
    }

    return aliases.get(s, "UnknownModel")


def plot_effectsize_summary(anova_res: pd.DataFrame, outpath: Path) -> None:
    """Plots a bar chart of eta-squared effect sizes for each metric.

    Sorts the summary statistics by differentiation power and maps the values to a continuous colormap to visualize which metrics best distinguish the datasets.

    Args:
        anova_res: DataFrame containing Welch ANOVA results, including 'metric' and 'eta_sq' columns.
        outpath: The file path where the resulting plot will be saved.
    """
    sub = anova_res.dropna(subset=["eta_sq"]).copy()
    if sub.empty:
        return

    sub = sub.sort_values("eta_sq", ascending=False)
    sub["metric"] = sub["metric"].map(METRIC_MAP).fillna(sub["metric"])

    # Extract NumPy arrays for plotting
    metrics = sub["metric"].to_numpy()
    eta_vals = sub["eta_sq"].to_numpy()

    # Define color normalization bounded between 0 and 1 for eta-squared
    norm = mcolors.Normalize(vmin=0, vmax=max(eta_vals.max(), 1.0))

    try:
        cmap = plt.colormaps.get_cmap("viridis")
    except AttributeError:
        cmap = cm.get_cmap("viridis")

    colors = cmap(norm(eta_vals))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use native Matplotlib bar for exact color mapping and lower overhead
    bars = ax.bar(metrics, eta_vals, color=colors, edgecolor="k", linewidth=0.7)

    ax.set_ylabel(r"$\eta^2$ (Effect Size)")
    ax.set_xlabel("Summary Statistic")
    ax.set_title("Metric Differentiation Power (from Welch's ANOVA)", pad=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Construct the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"$\eta^2$ Value")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_metric_histograms(
    combined: pd.DataFrame,
    desc: pd.DataFrame,
    metric: str,
    outpath: Path,
    *,
    verbose: bool = True,
) -> None:
    """Plot histograms comparing dataset means to global raw locus values.

    The left panel shows one value per retained dataset: the per-dataset mean
    for the selected metric. The right panel shows all pooled per-locus values
    for the same retained datasets.

    Args:
        combined: DataFrame containing raw, per-locus statistics for retained datasets.
        desc: DataFrame containing aggregated per-dataset descriptive statistics.
        metric: The specific summary statistic to plot.
        outpath: Output file path.
        verbose: Whether to print diagnostic counts.
    """
    mean_col = f"{metric}_mean"
    if mean_col not in desc.columns:
        if verbose:
            print(
                f"[Histogram Debug] Skipping {metric}: missing column '{mean_col}' in desc."
            )
        return

    if metric not in combined.columns:
        if verbose:
            print(
                f"[Histogram Debug] Skipping {metric}: missing column '{metric}' in combined."
            )
        return

    # --- Left panel input: one mean per retained dataset ---
    desc_sub = desc.loc[:, ["dataset", mean_col]].copy()
    desc_sub[mean_col] = pd.to_numeric(desc_sub[mean_col], errors="coerce")
    desc_sub = desc_sub.dropna(subset=[mean_col]).copy()

    dataset_means = desc_sub[mean_col].to_numpy(dtype=float)
    n_dataset_means = dataset_means.size
    n_unique_desc_datasets = desc_sub["dataset"].astype(str).nunique()

    # --- Right panel input: all raw locus values from retained datasets ---
    raw_series = pd.to_numeric(combined[metric], errors="coerce")
    raw_values = raw_series.to_numpy(dtype=float)
    raw_values = raw_values[~np.isnan(raw_values)]

    n_raw_loci = raw_values.size
    n_unique_combined_datasets = (
        combined["dataset_key"].astype(str).nunique()
        if "dataset_key" in combined.columns
        else combined["dataset"].astype(str).nunique()
    )

    # Extra consistency checks
    if verbose:
        print(
            f"[Histogram Debug] Metric={metric} | "
            f"dataset_means={n_dataset_means} | "
            f"unique_desc_datasets={n_unique_desc_datasets} | "
            f"unique_combined_datasets={n_unique_combined_datasets} | "
            f"raw_loci={n_raw_loci}"
        )

    if n_dataset_means == 0 or n_raw_loci == 0:
        if verbose:
            print(f"[Histogram Debug] Skipping {metric}: no data after filtering.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # ---------------- Left panel ----------------
    ax0 = axes[0]
    ax0.hist(dataset_means, bins=30, color="#2c7bb6", edgecolor="k", alpha=0.8)
    ax0.set_title(
        f"Dataset Mean Distribution: {metric}\n" f"(n datasets = {n_dataset_means})"
    )
    ax0.set_xlabel(f"Mean {metric}")
    ax0.set_ylabel("Count of Datasets")

    grand_mean = float(np.mean(dataset_means))
    ax0.axvline(
        grand_mean,
        color="r",
        linestyle="--",
        linewidth=1.5,
        label=f"Grand Mean = {grand_mean:.4g}",
    )
    ax0.legend()

    # ---------------- Right panel ----------------
    ax1 = axes[1]
    ax1.hist(raw_values, bins=50, color="#d7191c", edgecolor="none", alpha=0.6)
    ax1.set_title(
        f"Pooled Raw Locus Distribution: {metric}\n"
        f"(n loci = {n_raw_loci:,}; n datasets = {n_unique_combined_datasets})"
    )
    ax1.set_xlabel(f"Raw {metric}")
    ax1.set_ylabel("Count of Loci")

    p05, p95 = np.percentile(raw_values, [5, 95])
    ax1.axvline(p05, color="k", linestyle=":", alpha=0.6)
    ax1.axvline(p95, color="k", linestyle=":", alpha=0.6, label="5th/95th %ile")
    ax1.legend()

    # ---------------- Figure footer check ----------------
    fig.suptitle(
        f"Histogram diagnostics for {metric}",
        y=1.02,
        fontsize=14,
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_dataset_metric_heatmap(
    desc: pd.DataFrame,
    metrics: Sequence[str],
    outpath: Path,
    zscore: bool = True,
) -> None:
    """Generates a heatmap of per-dataset metric profiles.

    Extracts the specified metric means, optionally applies Z-score standardization to normalize scales across different summary statistics and plots the resulting matrix as a heatmap. Handles missing or invalid data by masking them in the visualization.

    Args:
        desc: A DataFrame containing per-dataset descriptive statistics.
        metrics: A sequence of target metric names to plot.
        outpath: The file path where the resulting plot will be saved.
        zscore: Whether to independently standardize each metric column to have a mean of 0 and standard deviation of 1. Defaults to True.
    """
    mean_cols = [f"{m}_mean" for m in metrics if f"{m}_mean" in desc.columns]
    if not mean_cols:
        return

    # Extract matrix and vectorize column name formatting
    mat = desc.set_index("dataset")[mean_cols].astype(float)
    mat.columns = mat.columns.str.replace("_mean", "", regex=False)

    if zscore:
        # Calculate standard deviation and replace 0 with 1.0 to prevent division by zero
        stds = mat.std(axis=0, ddof=1).replace(0, 1.0)
        mat = (mat - mat.mean(axis=0)) / stds

    n_datasets, n_metrics = mat.shape

    fig_height = max(8, n_datasets * 0.25)
    fig_width = max(10, n_metrics * 0.8 + 4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    masked_mat = np.ma.masked_invalid(mat.to_numpy())

    # Use modern colormap registry access with a fallback
    try:
        cmap = plt.colormaps.get_cmap("viridis").copy()
    except AttributeError:
        cmap = cm.get_cmap("viridis").copy()

    cmap.set_bad(color="lightgrey")

    im = ax.imshow(masked_mat, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_title(
        f"Per-dataset Mean Profiles ({'Z-scored' if zscore else 'Raw'})", pad=20
    )

    ax.set_xticks(np.arange(n_metrics))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")

    ax.set_yticks(np.arange(n_datasets))

    # Apply the dynamically calculated font size to the y-axis labels
    y_fontsize = 8 if n_datasets < 60 else 6 if n_datasets < 120 else 4
    ax.set_yticklabels(mat.index, fontsize=y_fontsize)

    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("Z-score" if zscore else "Raw Mean")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def find_dataset_files(root: Path) -> List[DatasetFiles]:
    """Scans the root directory recursively for complete datasets.

    Evaluates the directory tree lazily to locate 'Total_LocusStats.csv' files,
    pairing them with their corresponding summary JSON if present.

    Args:
        root: The base path to search within.

    Returns:
        A list of DatasetFiles dataclass instances containing the extracted
        paths and dataset identifiers.
    """
    datasets: List[DatasetFiles] = []

    # Process the generator directly to avoid loading the full path tree into memory at once
    for csv_path in root.rglob("Total_LocusStats.csv"):
        parent = csv_path.parent
        summary_path = parent / "Total_Summary.json"

        datasets.append(
            DatasetFiles(
                dataset_id=parent.relative_to(root).as_posix(),
                locus_csv=csv_path,
                summary_json=summary_path if summary_path.exists() else None,
            )
        )

    return datasets


def read_locus_stats(dset: DatasetFiles) -> pd.DataFrame:
    """Reads locus summary statistics into a DataFrame and extracts the join key.

    Utilizes the PyArrow engine for high-performance CSV reading. Identifies the canonical dataset key from the 'Filename' column if available, otherwise falls back to the raw directory structure identifier.

    Args:
        dset: A DatasetFiles instance containing the paths and identifier for a single dataset.

    Returns:
        A DataFrame containing the locus statistics with standardized dataset identification columns attached.
    """
    global _PYARROW_FALLBACK_WARNED

    # PyArrow engine handles type inference and memory
    # allocation more efficiently
    try:
        df = pd.read_csv(dset.locus_csv, engine="pyarrow")
    except ImportError:
        if not _PYARROW_FALLBACK_WARNED:
            print(
                "[Warning] pyarrow is not installed. Falling back to default pandas CSV parser."
            )
            _PYARROW_FALLBACK_WARNED = True
        df = pd.read_csv(dset.locus_csv)

    dataset_key = dset.dataset_id

    if "Filename" in df.columns:
        first_fn = df["Filename"].dropna().astype(str)
        if not first_fn.empty:
            extracted = extract_dataset_key(first_fn.iloc[0])
            if extracted:
                dataset_key = extracted

    dataset_key = str(dataset_key).strip().lower()

    # Assign metadata columns en masse
    df = df.assign(
        dataset_raw=dset.dataset_id,
        dataset_key=dataset_key,
        dataset=dataset_key,
        sim_strategy="Unknown",
    )

    return df


def deduplicate_datasets_by_key(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset_key" not in df.columns:
        return df

    counts = (
        df.groupby(["dataset_key", "dataset"], observed=False)
        .size()
        .reset_index(name="n_loci")
        .sort_values(["dataset_key", "n_loci"], ascending=[True, False])
    )
    keep = counts.drop_duplicates(["dataset_key"])
    return df.merge(
        keep[["dataset_key", "dataset"]],
        on=["dataset_key", "dataset"],
        how="inner",
    )


def extract_dataset_key(text: str) -> Optional[str]:
    """Extracts and normalizes the dataset identifier from a string.

    Searches for dataset naming conventions defined by _RESULTS_KEY_RE, converts the match to lowercase, and strips underscores to create a standardized join key.

    Args:
        text: The raw string containing the dataset identifier (e.g., a filename, path string, or raw dataset ID).

    Returns:
        The normalized dataset key if a match is found, otherwise None.
    """
    if not isinstance(text, str):
        return None

    match = _RESULTS_KEY_RE.search(text)
    return match.group(1).lower().replace("_", "") if match else None


def infer_metrics(df: pd.DataFrame) -> List[str]:
    exclude = {
        "Locus",
        "Filename",
        "dataset",
        "dataset_key",
        "TotalPos",
        "NetSegSites",
        "Chrom",
        "Position",
    }
    numeric_cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric_cols


def yeo_johnson_transform(values: np.ndarray) -> np.ndarray:
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    if len(values) == 0:
        return values
    try:
        return pt.fit_transform(values.reshape(-1, 1)).ravel()
    except Exception:
        return values


def welch_anova_by_metric(
    df_long: pd.DataFrame,
    metrics: Sequence[str],
    transform: bool = True,
    min_loci_per_group: int = 3,
) -> pd.DataFrame:
    """Computes Welch's ANOVA and effect sizes for summary statistics across datasets.

    Applies an optional Yeo-Johnson transformation and calculates the Welch ANOVA statistic, p-value, and eta-squared effect size for each specified metric. Filters out groups with fewer loci than the specified minimum threshold.

    Args:
        df_long: A DataFrame containing the long-format metrics, with at least a 'dataset' column and columns for each metric in `metrics`.
        metrics: A sequence of column names representing the metrics to test.
        transform: Whether to apply the Yeo-Johnson transformation to the data prior to testing. Defaults to True.
        min_loci_per_group: The minimum number of non-null observations required for a dataset to be included in the analysis for a given metric. Defaults to 3.

    Returns:
        A DataFrame containing the test results (F-statistic, p-value, eta-squared,
        dataset count, locus count, and transform status) for each metric.
    """
    results = []

    # Pre-filter datasets that do not meet the global minimum threshold to reduce downstream grouping overhead
    counts = df_long.groupby("dataset", observed=False).size()
    valid_datasets = counts[counts >= min_loci_per_group].index
    df_filtered = df_long[df_long["dataset"].isin(valid_datasets)]

    for metric in metrics:
        # Isolate metric data and drop NaNs specific to this metric
        sub = df_filtered[["dataset", metric]].dropna()
        if sub.empty:
            continue

        y = sub[metric].to_numpy(dtype=float).copy()
        datasets = sub["dataset"].to_numpy()

        # Apply jitter to prevent zero-variance errors in transformation/ANOVA
        y += np.random.normal(0, 1e-9, size=y.shape)

        if transform:
            y = yeo_johnson_transform(y)

        # Group the transformed array efficiently using a Series wrapper
        # avoiding full DataFrame reallocation
        grouped = pd.Series(y).groupby(datasets)
        groups = [g.to_numpy() for _, g in grouped if len(g) >= min_loci_per_group]

        if len(groups) < 2:
            continue

        # Compute Welch's ANOVA
        with np.errstate(invalid="ignore", divide="ignore"):
            res_obj = anova_oneway(groups, use_var="unequal", welch_correction=True)

        F_val = float(getattr(res_obj, "statistic", np.nan))
        p_val = float(getattr(res_obj, "pvalue", np.nan))

        # Vectorized Effect Size (Eta-squared) Calculation
        group_sizes = np.array([g.size for g in groups])
        group_means = np.array([g.mean() for g in groups])
        all_vals = np.concatenate(groups)

        grand_mean = all_vals.mean()

        # Calculate sums of squares purely in NumPy
        ss_between = np.sum(group_sizes * (group_means - grand_mean) ** 2)
        ss_total = np.sum((all_vals - grand_mean) ** 2)

        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

        results.append(
            {
                "metric": metric,
                "welch_F": F_val,
                "p_raw": p_val,
                "eta_sq": float(eta_sq),
                "n_datasets": len(groups),
                "n_loci_total": all_vals.size,
                "transform": transform,
            }
        )

    return pd.DataFrame(results)


def bh_fdr_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    finite_mask = np.isfinite(pvals)
    if finite_mask.sum() > 0:
        _, q_finite, _, _ = multipletests(
            pvals[finite_mask], alpha=0.05, method="fdr_bh"
        )
        qvals[finite_mask] = q_finite
    return qvals


def per_dataset_descriptives(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Calculates summary statistics for specified metrics across datasets.

    Aggregates per-locus data to compute the mean, variance, standard deviation,
    median, and count for each metric. Converts the grouping key to a categorical
    type prior to aggregation to optimize memory and computation speed.

    Args:
        df: A DataFrame containing a 'dataset' column and the numeric metric columns.
        metrics: A sequence of metric column names to aggregate.

    Returns:
        A DataFrame containing aggregated statistics per dataset, with flattened
        column names (e.g., 'Pi_mean', 'Pi_var').
    """
    # Cast grouping key to categorical for memory reduction and faster integer hashing
    group_key = df["dataset"].astype("category")

    # Construct aggregation dictionary directly via comprehension
    agg_funcs = {m: ["mean", "var", "std", "median", "count"] for m in metrics}

    # Execute grouping
    desc = df.groupby(group_key, observed=True).agg(agg_funcs)

    # Iterate over the underlying tuple array to bypass Pylance
    # Index[str] strictness
    desc.columns = [f"{c[0]}_{c[1]}" for c in desc.columns.values]

    return desc.reset_index()


def load_completed_dataset_labels(
    metrics_long_csv: Path,
    *,
    expected_models: set[str],
    require_all_strategies: bool = True,
    expected_strategies: Optional[set[str]] = None,
    debug_missing: bool = False,
) -> set[str]:
    """Load completed dataset keys from the metrics CSV file.

    Evaluates whether each dataset has rows for the Cartesian product of
    the specified expected models and expected simulation strategies.

    Args:
        metrics_long_csv: Path to the long-format metrics CSV.
        expected_models: A set of canonical model names that must be present.
        require_all_strategies: If True, strictly enforces the presence of all
            expected strategies and models. If False, requires only at least one
            strategy per expected model.
        expected_strategies: A set of expected simulation strategy names.
        debug_missing: If True, prints verbose diagnostic output.

    Returns:
        A set of valid dataset keys.
    """
    print(f"\n[Completion Gate] Loading metrics from: {metrics_long_csv}")

    try:
        df = pd.read_csv(metrics_long_csv)
    except Exception as e:
        print(f"[Completion Gate] Error reading CSV: {e}")
        return set()

    required_cols = {"dataset", "sim_strategy", "model"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Completion gating requires columns {sorted(required_cols)}. Missing: {sorted(missing)}"
        )

    sub = df.loc[:, list(required_cols)].copy()
    sub["dataset"] = sub["dataset"].astype(str)

    # Vectorized extraction and normalization
    sub["join_key"] = (
        sub["dataset"]
        .astype(str)
        .str.extract(_RESULTS_KEY_RE, expand=False)
        .str.lower()
        .str.replace("_", "", regex=False)
    )
    failed_keys = sub[sub["join_key"].isna()]["dataset"].unique()
    if len(failed_keys) > 0:
        print(
            f"[Completion Gate] Warning: Could not extract join key from {len(failed_keys)} datasets (e.g., {failed_keys[:3]}). "
            "Check your _RESULTS_KEY_RE regex."
        )

    sub = sub.dropna(subset=["join_key"])

    # Define your mapping dictionary once
    strategy_map = {
        "random weighted inv": "Random Weighted Inv",
        "nonrandom weighted": "Nonrandom Weighted",
        "random weighted": "Random Weighted",
        "nonrandom": "Nonrandom",
        "random": "Random",
    }

    # Vectorized mapping, filling unmapped values with "Unknown"
    sub["sim_strategy"] = (
        sub["sim_strategy"]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(strategy_map)
        .fillna("Unknown")
    )
    sub["model"] = sub["model"].astype(str).map(_normalize_model_name)

    unknown_model_count = int((sub["model"] == "UnknownModel").sum())
    if unknown_model_count > 0:
        print(
            f"[Completion Gate] Warning: Ignoring {unknown_model_count} rows with unrecognized model labels."
        )
        sub = sub[sub["model"] != "UnknownModel"].copy()

    if expected_strategies is None:
        expected_strategies = set(sub["sim_strategy"].dropna().unique()) - {"Unknown"}

    required_pairs = {(m, s) for m in expected_models for s in expected_strategies}

    # Pre-allocate a defaultdict to avoid key-error checks during iteration
    observed_pairs_dict = defaultdict(set)

    # Zip iterates over the underlying column arrays directly, which is highly efficient
    for key, mod, strat in zip(sub["join_key"], sub["model"], sub["sim_strategy"]):
        observed_pairs_dict[key].add((mod, strat))

    observed_pairs_by_ds = dict(observed_pairs_dict)

    complete_keys = set()
    miss_records = []

    for ds_key, observed in observed_pairs_by_ds.items():
        if require_all_strategies:
            if required_pairs.issubset(observed):
                complete_keys.add(ds_key)
            else:
                missing_pairs = required_pairs - observed
                miss_records.append((ds_key, missing_pairs))
        else:
            observed_models = {m for m, s in observed}
            if expected_models.issubset(observed_models):
                complete_keys.add(ds_key)

    print(
        f"[Completion Gate] Complete datasets (missing nothing): {len(complete_keys)}/{len(observed_pairs_by_ds)}"
    )

    if debug_missing and require_all_strategies and miss_records:
        print(
            f"[Completion Gate][Debug] Dropped {len(miss_records)} datasets due to missing model/strategy pairs."
        )
        print("[Completion Gate][Debug] Sample Rejected Datasets:")
        for ds_key, missing_pairs in sorted(miss_records)[:5]:
            print(f"  ** Dataset: {ds_key}")
            print(f"     MISSING PAIRS: {sorted(missing_pairs)}")

    return complete_keys


def main() -> None:
    """Executes the pipeline to aggregate, test, and plot summary statistics."""
    args = parse_args()
    root: Path = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else Path.cwd() / "dataset_stats_output"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching for datasets in {root}...")
    datasets = find_dataset_files(root)
    if not datasets:
        raise FileNotFoundError(f"No Total_LocusStats.csv files found under {root}")

    frames = [read_locus_stats(d) for d in datasets]
    combined = pd.concat(frames, ignore_index=True)
    combined = deduplicate_datasets_by_key(combined)

    if args.completed_metrics_long is not None:
        metrics_path = args.completed_metrics_long.resolve()
        strict = args.strict_strategies

        print(
            f"[Completion gate] Strategy Strictness: {'STRICT (All 30 pairs)' if strict else 'PERMISSIVE (Baseline Models Only)'}"
        )

        allowed_keys = load_completed_dataset_labels(
            metrics_path,
            expected_models=EXPECTED_MODELS if strict else {"RefAllele"},
            require_all_strategies=strict,
            expected_strategies=ALL_STRATEGIES if strict else None,
            debug_missing=True,
        )

        before_keys = set(combined["dataset_key"].astype(str).unique())
        inter = before_keys & allowed_keys

        print(
            f"[Completion gate] Allowed dataset_keys (from {metrics_path.name}): {len(allowed_keys)}\n"
            f"[Completion gate] dataset_keys in locus stats before:            {len(before_keys)}\n"
            f"[Completion gate] Intersection size:                             {len(inter)}"
        )

        combined = combined[combined["dataset_key"].isin(allowed_keys)].copy()

        if combined.empty:
            raise ValueError(
                "After completion gating, no datasets remained. This indicates a join-key mismatch "
                "or missing baseline rows in the completed metrics file."
            )

        print(
            f"[Completion gate] Datasets retained after: {combined['dataset_key'].nunique()}"
        )

    # Cast dataset to category early to optimize all downstream memory
    # and grouping operations
    combined["dataset"] = combined["dataset"].astype("category")

    metrics = args.metrics if args.metrics else infer_metrics(combined)
    if not metrics:
        raise ValueError("No numeric metrics found to test.")

    print(f"Analyzing metrics: {metrics}")

    retained_n = combined["dataset_key"].astype(str).nunique()
    print(f"[Sanity Check] Retained unique dataset_key count: {retained_n}")

    desc = per_dataset_descriptives(combined, metrics)
    desc.to_csv(
        out_dir / f"{args.out_prefix}_all_datasets_summary_stats.csv", index=False
    )

    print("Running Welch ANOVA...")
    transform = not args.no_transform
    anova_res = welch_anova_by_metric(
        combined[["dataset"] + list(metrics)], metrics, transform, args.min_loci
    )

    if not anova_res.empty:
        pvals = anova_res["p_raw"].to_numpy(dtype=float).copy()
        pvals[pvals == 0.0] = np.nextafter(0, 1)
        anova_res["p_bh"] = bh_fdr_adjust(pvals)
        anova_res["significant_bh_0.05"] = anova_res["p_bh"] < 0.05
        anova_res.sort_values("p_bh").to_csv(
            out_dir / f"{args.out_prefix}_metrics_welch_anova.csv", index=False
        )
        anova_res = anova_res.drop(columns=["welch_F"])

    print("Running Post-Hoc Outlier Analysis...")
    outliers_df = analyze_deviations(desc, metrics, out_dir, args.out_prefix)

    if args.plots and not anova_res.empty:
        plot_dir = out_dir / f"{args.out_prefix}_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        print("Generating plots...")

        plot_effectsize_summary(anova_res, plot_dir / "metrics_effectsize_summary.png")
        plot_dataset_metric_heatmap(
            desc, metrics, plot_dir / "dataset_metric_heatmap.png", zscore=True
        )

        sorted_metrics = anova_res.sort_values("eta_sq", ascending=False)[
            "metric"
        ].tolist()
        for m in sorted_metrics[:10]:
            plot_metric_histograms(
                combined,
                desc,
                m,
                plot_dir / f"{m}_distributions.png",
                verbose=True,
            )

        print(f"Plots written to: {plot_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare per-locus summary statistics across datasets."
    )
    p.add_argument(
        "--root", type=Path, required=True, help="Root directory containing results."
    )
    p.add_argument(
        "--output-dir", type=Path, default=None, help="Directory to save all outputs."
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default="dataset_stats",
        help="Output filename prefix.",
    )
    p.add_argument(
        "--no-transform",
        action="store_true",
        help="Disable Yeo-Johnson transform for ANOVA.",
    )
    p.add_argument(
        "--metrics", nargs="*", default=None, help="Specific metrics to test."
    )
    p.add_argument(
        "--min-loci",
        type=int,
        default=3,
        help="Min loci per dataset required for testing.",
    )
    p.add_argument(
        "--plots", action="store_true", help="Generate static matplotlib plots."
    )

    p.add_argument(
        "--completed-metrics-long",
        type=Path,
        default=None,
        help=(
            "Optional: path to zygosity_metrics_long.csv. If provided, popgen stats "
            "will be computed ONLY for datasets (and strategies) that have produced "
            "baseline PG-SUI metrics."
        ),
    )

    p.add_argument(
        "--strict-strategies",
        action="store_true",
        help=(
            "If set, datasets are only included if the baseline model has results for ALL 5 simulation strategies. "
            "Defaults to False (include datasets even if some strategies are missing)."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
