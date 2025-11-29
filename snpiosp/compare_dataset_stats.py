#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recursively aggregate per-locus summary stats across many datasets and test which metrics differ by dataset.

Updates:
- Added --output-dir support.
- Refactored SNPioMultiQC integration to use static queuing methods.
- Reversed effect size plot colors (High Effect = Dark/Bold).
- Added Histogram Barplots to MultiQC.

Usage:
------
python compare_dataset_stats.py \
    --root /path/to/results_root \
    --output-dir /path/to/output \
    --out-prefix pg_sui_validation \
    --transform \
    --plots
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

# Use non-interactive backend
plt.switch_backend("Agg")

_RESULTS_KEY_RE = re.compile(r"(results\d+)", flags=re.IGNORECASE)


@dataclass
class DatasetFiles:
    dataset_id: str
    locus_csv: Path
    summary_json: Optional[Path]


def analyze_deviations(
    desc: pd.DataFrame, metrics: Sequence[str], out_dir: Path, out_prefix: str
) -> pd.DataFrame:
    """
    Post-hoc "Analysis of Means" to identify which datasets contribute most to variance.
    Calculates Z-scores of dataset means.
    """
    deviation_rows = []

    for m in metrics:
        mean_col = f"{m}_mean"
        if mean_col not in desc.columns:
            continue

        # Extract the means for this metric across all datasets
        series = desc.set_index("dataset")[mean_col].astype(float)

        # Calculate Grand Mean of Means and Std Dev of Means
        grand_mean = series.mean()
        std_dev = series.std(ddof=1)

        if std_dev == 0:
            continue

        # Calculate Z-score for each dataset
        z_scores = (series - grand_mean) / std_dev

        for dataset, z in z_scores.items():
            deviation_rows.append(
                {
                    "dataset": dataset,
                    "metric": m,
                    "dataset_mean": series[dataset],
                    "global_mean_of_means": grand_mean,
                    "z_score": z,
                    "abs_z_score": abs(z),
                    "is_outlier_2sigma": abs(z) > 2.0,
                    "is_outlier_3sigma": abs(z) > 3.0,
                }
            )

    if not deviation_rows:
        return pd.DataFrame()

    dev_df = pd.DataFrame(deviation_rows)

    # Sort by 'Impact' (Absolute Z-score)
    dev_df = dev_df.sort_values("abs_z_score", ascending=False)

    # Write to CSV
    out_path = out_dir / f"{out_prefix}_outlier_rankings.csv"
    dev_df.to_csv(out_path, index=False)
    print(
        f"Wrote outlier analysis (datasets contributing most to variance): {out_path}"
    )
    return dev_df


def plot_metric_histograms(
    combined: pd.DataFrame,
    desc: pd.DataFrame,
    metric: str,
    outpath: Path,
) -> None:
    """Plot histograms: Dataset Means (Panel A) and Global Raw Values (Panel B)."""
    mean_col = f"{metric}_mean"
    if mean_col not in desc.columns:
        return

    dataset_means = desc[mean_col].dropna()
    raw_values = combined[metric].dropna()

    if dataset_means.empty or raw_values.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Panel A: Dataset Means
    ax0 = axes[0]
    ax0.hist(dataset_means, bins=30, color="#2c7bb6", edgecolor="k", alpha=0.8)
    ax0.set_title(f"Distribution of Dataset Means: {metric}")
    ax0.set_xlabel(f"Mean {metric}")
    ax0.set_ylabel("Count of Datasets")

    grand_mean = dataset_means.mean()
    ax0.axvline(
        grand_mean, color="r", linestyle="--", linewidth=1.5, label="Grand Mean"
    )
    ax0.legend()

    # Panel B: Global Raw Values
    ax1 = axes[1]
    ax1.hist(raw_values, bins=50, color="#d7191c", edgecolor="none", alpha=0.6)
    ax1.set_title(f"Global Distribution of Raw Locus Values: {metric}")
    ax1.set_xlabel(f"Raw {metric}")
    ax1.set_ylabel("Count of Loci")

    try:
        p05 = raw_values.quantile(0.05)
        p95 = raw_values.quantile(0.95)
        ax1.axvline(p05, color="k", linestyle=":", alpha=0.5)
        ax1.axvline(p95, color="k", linestyle=":", alpha=0.5, label="5th/95th %ile")
        ax1.legend()
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_effectsize_summary(anova_res: pd.DataFrame, outpath: Path) -> None:
    """Plot Eta-squared as a bar plot, sorted by magnitude."""
    # Filter for valid eta_sq values and sort
    sub = anova_res.dropna(subset=["eta_sq"]).copy()
    if sub.empty:
        return
    sub = sub.sort_values("eta_sq", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- COLORS: Viridis Reversed (Dark = High) ---
    cmap = sns.color_palette("viridis_r", as_cmap=True)

    norm = mcolors.Normalize(
        vmin=min(sub["eta_sq"].min(), 0), vmax=max(sub["eta_sq"].max(), 1)
    )
    colors = [cmap(norm(val)) for val in sub["eta_sq"]]

    # Create the bar plot
    ax = sns.barplot(
        data=sub,
        x="metric",
        y="eta_sq",
        hue="eta_sq",
        palette=colors,
        edgecolor="k",
        linewidth=0.7,
        ax=ax,
        errorbar=None,
        legend=False,
    )

    # Customize axes and title
    ax.set_ylabel("η² (Effect Size)", fontsize=12)
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_title(
        "Metric Differentiation Power (η²-squared from Welch's ANOVA)",
        fontsize=14,
        pad=15,
    )

    # Add y-axis grid for easier reading
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Rotate x-axis labels to prevent overlap
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=10,
    )

    # Add a colorbar to act as a legend for the bar colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for ScalarMappable
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("η² (Eta-squared) Value", fontsize=10)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_dataset_metric_heatmap(
    desc: pd.DataFrame,
    metrics: Sequence[str],
    outpath: Path,
    zscore: bool = True,
) -> None:
    """Scalable heatmap of per-dataset means."""
    mean_cols = [f"{m}_mean" for m in metrics if f"{m}_mean" in desc.columns]
    if not mean_cols:
        return

    mat = desc.set_index("dataset")[mean_cols].astype(float)
    mat.columns = [c.replace("_mean", "") for c in mat.columns]

    if zscore:
        means = mat.mean(axis=0)
        stds = mat.std(axis=0, ddof=1)
        stds[stds == 0] = 1.0
        mat = (mat - means) / stds

    n_datasets = mat.shape[0]
    n_metrics = mat.shape[1]

    fig_height = max(8, n_datasets * 0.25)
    fig_width = max(10, n_metrics * 0.8 + 4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    masked_mat = np.ma.masked_invalid(mat.values)
    try:
        cmap = copy.copy(plt.get_cmap("viridis"))
    except:
        cmap = copy.copy(cm.viridis)
    cmap.set_bad(color="lightgrey")

    im = ax.imshow(masked_mat, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_title(
        f"Per-dataset Mean Profiles ({'Z-scored' if zscore else 'Raw'})",
        fontsize=14,
        pad=20,
    )
    ax.set_xticks(np.arange(n_metrics))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(n_datasets))

    y_fontsize = 8 if n_datasets < 60 else 6 if n_datasets < 120 else 4
    ax.set_yticklabels(mat.index, fontsize=y_fontsize)

    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("Z-score" if zscore else "Raw Mean")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def find_dataset_files(root: Path) -> List[DatasetFiles]:
    csv_paths = sorted(root.rglob("Total_LocusStats.csv"))
    datasets: List[DatasetFiles] = []
    for csv_path in csv_paths:
        parent = csv_path.parent
        dataset_id = parent.relative_to(root).as_posix()
        summary_path = parent / "Total_Summary.json"
        datasets.append(
            DatasetFiles(
                dataset_id=dataset_id,
                locus_csv=csv_path,
                summary_json=summary_path if summary_path.exists() else None,
            )
        )
    return datasets


def read_locus_stats(dset: DatasetFiles) -> pd.DataFrame:
    df = pd.read_csv(dset.locus_csv)
    df["dataset"] = dset.dataset_id
    dataset_key = dset.dataset_id
    if "Filename" in df.columns:
        first_fn = df["Filename"].dropna().astype(str)
        if not first_fn.empty:
            extracted = extract_dataset_key_from_filename(first_fn.iloc[0])
            if extracted:
                dataset_key = extracted
    df["dataset_key"] = dataset_key
    return df


def deduplicate_datasets_by_key(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset_key" not in df.columns:
        return df
    counts = df.groupby(["dataset_key", "dataset"]).size().reset_index(name="n_loci")
    keep = counts.sort_values(
        ["dataset_key", "n_loci"], ascending=[True, False]
    ).drop_duplicates("dataset_key")
    kept_pairs = keep[["dataset_key", "dataset"]]
    df_keep = df.merge(kept_pairs, on=["dataset_key", "dataset"], how="inner")
    df_keep["dataset"] = df_keep["dataset_key"]
    return df_keep


def extract_dataset_key_from_filename(filename: str) -> Optional[str]:
    if not isinstance(filename, str):
        return None
    m = _RESULTS_KEY_RE.search(filename)
    return m.group(1).lower() if m else None


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
    results = []
    counts = df_long.groupby("dataset").size()
    valid_datasets = counts[counts >= min_loci_per_group].index
    df_long = df_long[df_long["dataset"].isin(valid_datasets)].copy()

    for metric in metrics:
        sub = df_long[["dataset", metric]].dropna()
        if sub.empty:
            continue

        y = sub[metric].to_numpy(dtype=float)
        # Jitter to fix zero-variance constant metrics
        jitter = np.random.normal(0, 1e-9, size=y.shape)
        y = y + jitter

        if transform:
            y = yeo_johnson_transform(y)

        groups = []
        for ds, vals in sub.assign(y=y).groupby("dataset")["y"]:
            v = vals.to_numpy()
            if v.size >= min_loci_per_group:
                groups.append(v)

        if len(groups) < 2:
            continue

        with np.errstate(invalid="ignore", divide="ignore"):
            welch_res = anova_oneway(groups, use_var="unequal", welch_correction=True)

        all_vals = np.concatenate(groups)
        grand_mean = all_vals.mean()
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        ss_between = sum([v.size * (v.mean() - grand_mean) ** 2 for v in groups])
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

        results.append(
            {
                "metric": metric,
                "welch_F": float(welch_res.statistic),
                "p_raw": float(welch_res.pvalue),
                "eta_sq": float(eta_sq),
                "n_datasets": int(welch_res.n_groups),
                "n_loci_total": int(welch_res.nobs_t),
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
    agg = {}
    for m in metrics:
        agg[m] = ["mean", "var", "std", "median", "count"]
    desc = df.groupby("dataset").agg(agg)
    desc.columns = ["_".join(c).strip() for c in desc.columns.to_flat_index()]
    desc = desc.reset_index()
    return desc


# ---------------------------------------------------------------------
# MultiQC Integration
# ---------------------------------------------------------------------


def run_multiqc_integration(
    output_dir: Path,
    prefix: str,
    anova_df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    desc: pd.DataFrame,
) -> None:
    """
    Summarize results into a MultiQC report using SNPioMultiQC static methods.
    Added logic to generate histograms (as BarPlots) for top metrics.
    """
    try:
        from snpio import SNPioMultiQC
    except ImportError:
        print(
            "[Warning] SNPio is not installed. Skipping MultiQC generation. Run 'pip install snpio' to enable this feature."
        )
        return

    print(f"Queuing results for MultiQC report...")

    # 1. Queue ANOVA Table (Table)
    if not anova_df.empty:
        # Sort for display and clean up columns
        disp_df = anova_df.sort_values("eta_sq", ascending=False).copy()

        # Preserve internal metric ID for column lookups and panel IDs
        disp_df["metric_id"] = disp_df["metric"].astype(str)

        # Human-readable display label
        disp_df["metric"] = disp_df["metric"].astype(str)
        disp_df.loc[disp_df["metric"] == "F_inbreeding", "metric"] = (
            "F<sub>IS</sub> (Inbreeding Coefficient)"
        )
        disp_df.loc[disp_df["metric"] == "Sample_Size", "metric"] = "Sample Size"
        disp_df.loc[disp_df["metric"] == "ThetaWatt", "metric"] = "Watterson's θ"
        disp_df.loc[disp_df["metric"] == "Pi", "metric"] = "Nucleotide Diversity (π)"
        disp_df.loc[disp_df["metric"] == "TajimasD", "metric"] = "Tajima's D"
        disp_df.loc[disp_df["metric"] == "Missingness", "metric"] = "Missingness Prop."
        disp_df.loc[disp_df["metric"] == "Ho", "metric"] = "Observed Heterozygosity"
        disp_df.loc[disp_df["metric"] == "He", "metric"] = "Expected Heterozygosity"
        disp_df.loc[disp_df["metric"] == "HetzPositions", "metric"] = (
            "Heterozygous Sites"
        )
        disp_df.loc[disp_df["metric"] == "SegSites", "metric"] = "Segregating Sites"
        disp_df.loc[disp_df["metric"] == "Singletons", "metric"] = "Singleton Prop."
        disp_df.loc[disp_df["metric"] == "Hap", "metric"] = "Number of Haplotypes"
        disp_df.loc[disp_df["metric"] == "Hd", "metric"] = "Haplotype Diversity"

        # Rename columns for the ANOVA table
        disp_df = disp_df.rename(
            columns={
                "p_raw": "P-value (raw)",
                "p_bh": "P-value (B-H adjusted)",
                "eta_sq": "Eta-Squared (η²)",
                "n_datasets": "Number of Datasets",
                "n_loci_total": "Total Loci",
                "transform": "Yeo-Johnson Transform",
                "significant_bh_0.05": "Significant (P-adj < 0.05)",
            }
        )

        # Add table
        SNPioMultiQC.queue_table(
            df=disp_df,
            panel_id="anova_results",
            section="ANOVA Results (Differentiation Power)",
            title="SNPioSumStats: Welch's ANOVA Results",
            index_label="metric",
            description=(
                "Welch ANOVA results showing which summary statistics best differentiate the "
                "datasets (ranked by η²). Lower P-values and higher η² indicate stronger "
                "differentiation power. The Yeo-Johnson transform was applied to metrics prior "
                "to testing to improve normality. Summary Statistics include: "
                "F<sub>IS</sub> (Inbreeding Coefficient), Watterson's θ (genetic diversity), "
                "Nucleotide Diversity (π), Tajima's D (neutrality test), Missingness Proportion, "
                "Observed and Expected Heterozygosity, Number of Heterozygous Sites, Number of "
                "Segregating Sites, Singleton Proportion, Number of Haplotypes, and Haplotype "
                "Diversity. The results help identify which summary statistics are most effective "
                "at distinguishing between different datasets based on their genetic variation profiles."
            ),
            pconfig={
                "id": "anova_results",
                "title": "SNPioSumStats: Welch's ANOVA Results",
                "sort_rows": False,
            },
        )

        # Effect size barplot: use the *display* labels on the x-axis
        effect_size_data = disp_df.set_index("metric")["Eta-Squared (η²)"]

        SNPioMultiQC.queue_barplot(
            df=effect_size_data,
            panel_id="effect_size_plot",
            section="Metric Effect Sizes",
            title="SNPioSumStats: Metric Differentiation Power (η² from Welch's ANOVA)",
            index_label="metric",
            description=(
                "Bar plot of η² (Eta-Squared) values indicating the strength of differentiation "
                "for each metric. Higher values denote greater ability to distinguish between datasets. "
                "Metrics are ordered by effect size for clarity. The Yeo-Johnson transform was applied "
                "prior to testing to enhance normality. This visualization aids in quickly identifying "
                "which summary statistics are most informative for dataset differentiation based on genetic "
                "variation profiles."
            ),
            pconfig={
                "id": "effect_size_plot",
                "title": "SNPioSumStats: Metric Differentiation Power (η² from Welch's ANOVA)",
                "xlab": "Metric",
                "ylab": "η² (Eta-Squared)",
                "sort_samples": False,
            },
        )

        # 4. Queue Histograms (Distributions of Means) for metrics
        # Use metric_id (internal) for column lookups and panel ids; use 'metric' (display) for titles
        metric_ids = disp_df["metric_id"].tolist()

        for metric_id in metric_ids:
            mean_col = f"{metric_id}_mean"
            if mean_col not in desc.columns:
                continue

            data = desc[mean_col].dropna()
            if data.empty:
                continue

            # Get pretty display label for this metric
            row = disp_df.loc[disp_df["metric_id"] == metric_id].iloc[0]
            metric_display = str(row["metric"])

            # Calculate Histogram counts
            counts, edges = np.histogram(data, bins=15)

            # Create generic bin labels (ranges)
            bin_labels = [
                (
                    f"{edges[i]:.2G}-{edges[i+1]:.2G}"
                    if edges[i] <= 1
                    else f"{edges[i]:.1f}-{edges[i+1]:.1f}"
                )
                for i in range(len(edges) - 1)
            ]

            # MultiQC Barplot format: Rows=Samples, Cols=Categories.
            # Here: one "sample" row per metric_id, columns are histogram bins.
            hist = pd.Series(counts, index=bin_labels, name=metric_id)

            panel_id = f"hist_{metric_id}"

            SNPioMultiQC.queue_barplot(
                df=hist,
                panel_id=panel_id,
                section="Metric Distributions (Histograms)",
                title=f"Distribution of Dataset Means: {metric_display}",
                index_label="metric",
                description=(
                    f"Histogram showing the frequency distribution of mean {metric_display} values "
                    f"across all datasets. This visualization helps assess how datasets vary in their "
                    f"average {metric_display} values, indicating potential differences in genetic "
                    f"variation profiles. The histogram bins represent ranges of mean values, with the "
                    f"height of each bar indicating the number of datasets falling within that range."
                ),
                pconfig={
                    "id": panel_id,
                    "title": f"Distribution: {metric_display}",
                    "xlab": f"{metric_display}: Mean Ranges",
                    "ylab": "Dataset Counts",
                    "sort_samples": False,
                },
            )

    # 2. Queue Outlier/Deviation Table
    if not outliers_df.empty:
        outliers_df = outliers_df.copy()
        outliers_df.loc[outliers_df["metric"] == "F_inbreeding", "metric"] = (
            "F<sub>IS</sub>"
        )
        outliers_df.loc[outliers_df["metric"] == "ThetaWatt", "metric"] = (
            "Watterson's θ"
        )
        outliers_df.loc[outliers_df["metric"] == "Pi", "metric"] = (
            "Nucleotide Diversity (π)"
        )
        outliers_df.loc[outliers_df["metric"] == "TajimasD", "metric"] = "Tajima's D"
        outliers_df.loc[outliers_df["metric"] == "Missingness", "metric"] = (
            "Missingness Prop."
        )
        outliers_df.loc[outliers_df["metric"] == "Ho", "metric"] = (
            "Observed Heterozygosity"
        )
        outliers_df.loc[outliers_df["metric"] == "He", "metric"] = (
            "Expected Heterozygosity"
        )
        outliers_df.loc[outliers_df["metric"] == "HetzPositions", "metric"] = (
            "Heterozygous Sites"
        )
        outliers_df.loc[outliers_df["metric"] == "SegSites", "metric"] = (
            "Segregating Sites"
        )
        outliers_df.loc[outliers_df["metric"] == "Singletons", "metric"] = (
            "Singleton Prop."
        )
        outliers_df.loc[outliers_df["metric"] == "Hap", "metric"] = (
            "Number of Haplotypes"
        )
        outliers_df.loc[outliers_df["metric"] == "Hd", "metric"] = "Haplotype Diversity"

        # Limit to top 50 deviations to prevent massive tables
        top_deviations = outliers_df[outliers_df["abs_z_score"] > 1.96].copy()
        top_deviations["Index"] = (
            top_deviations["dataset"].astype(str)
            + " | "
            + top_deviations["metric"].astype(str)
        )

        top_deviations = top_deviations.set_index("Index")
        top_deviations = top_deviations.drop(columns=["dataset", "metric"])
        top_deviations = top_deviations.sort_values("abs_z_score", ascending=False)

        SNPioMultiQC.queue_table(
            df=top_deviations,
            panel_id="outlier_analysis",
            section="Outlier Datasets",
            title="SNPioSumStats: Significantly Different Datasets per Metric",
            index_label="dataset",
            description="Post-hoc analysis identifying specific datasets that deviate significantly from the global mean (Z-scores > 1.96). This table highlights datasets that contribute most to variance for each metric, indicating potential outliers in genetic variation profiles. Columns include the dataset mean, global mean of means, Z-score, and flags for significance at 2σ and 3σ thresholds. This analysis helps pinpoint datasets with unusual summary statistic values that may warrant further investigation. Summary statistics include: F<sub>IS</sub> (Inbreeding Coefficient), Watterson's θ (genetic diversity), Nucleotide Diversity (π), Tajima's D (neutrality test), Missingness Proportion, Observed and Expected Heterozygosity, Number of Heterozygous Sites, Number of Segregating Sites, Singleton Proportion, Number of Haplotypes, and Haplotype Diversity.",
            pconfig={
                "id": "outlier_analysis",
                "title": "SNPioSumStats: Significantly Different Datasets per Metric",
                "sort_rows": False,
            },
        )

    # 3. Build Report
    print("Building MultiQC report...")
    SNPioMultiQC.build(
        prefix=prefix,
        output_dir=str(output_dir),
        title="SNPioSumStats Validation Dataset Report",
        overwrite=True,
    )
    print("MultiQC Report generation queued/built successfully.")


def main() -> None:
    args = parse_args()
    root: Path = args.root.resolve()

    # Setup Output Directory
    if args.output_dir is not None:
        out_dir = args.output_dir.resolve()
    else:
        out_dir = Path.cwd() / "dataset_stats_output"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching for datasets in {root}...")
    datasets = find_dataset_files(root)
    if not datasets:
        raise FileNotFoundError(f"No Total_LocusStats.csv files found under {root}")

    frames = [read_locus_stats(d) for d in datasets]
    combined = pd.concat(frames, ignore_index=True)
    combined = deduplicate_datasets_by_key(combined)

    metrics = args.metrics if args.metrics else infer_metrics(combined)
    if not metrics:
        raise ValueError("No numeric metrics found to test.")
    print(f"Analyzing metrics: {metrics}")

    # 1. Summary Stats
    desc = per_dataset_descriptives(combined, metrics)
    summary_csv_path = out_dir / f"{args.out_prefix}_all_datasets_summary_stats.csv"
    desc.to_csv(summary_csv_path, index=False)

    # 2. ANOVA
    print("Running Welch ANOVA...")
    transform = not args.no_transform
    anova_res = welch_anova_by_metric(
        combined[["dataset"] + list(metrics)], metrics, transform, args.min_loci
    )

    if not anova_res.empty:
        pvals = anova_res["p_raw"].to_numpy(dtype=float)
        pvals[pvals == 0.0] = np.nextafter(0, 1)
        anova_res["p_bh"] = bh_fdr_adjust(pvals)
        anova_res["significant_bh_0.05"] = anova_res["p_bh"] < 0.05
        anova_res.sort_values("p_bh").to_csv(
            out_dir / f"{args.out_prefix}_metrics_welch_anova.csv", index=False
        )

        # Ensure correct dtypes
        anova_res = anova_res.drop(columns=["welch_F"])
        anova_res["eta_sq"] = anova_res["eta_sq"].astype(float)

    # 3. Post-Hoc: Outlier Analysis
    print("Running Post-Hoc Outlier Analysis...")
    outliers_df = analyze_deviations(desc, metrics, out_dir, args.out_prefix)

    # 4. Plots (Static PNGs)
    plot_dir = out_dir / f"{args.out_prefix}_plots"
    if args.plots and not anova_res.empty:
        plot_dir.mkdir(parents=True, exist_ok=True)
        print("Generating plots...")

        plot_effectsize_summary(anova_res, plot_dir / "metrics_effectsize_summary.png")
        plot_dataset_metric_heatmap(
            desc, metrics, plot_dir / "dataset_metric_heatmap.png", zscore=True
        )

        sorted_metrics = anova_res.sort_values("eta_sq", ascending=False)[
            "metric"
        ].tolist()

        # Limit histogram generation to top 10 metrics to save time/space
        for m in sorted_metrics[:10]:
            plot_metric_histograms(
                combined, desc, m, plot_dir / f"{m}_distributions.png"
            )

        print(f"Plots written to: {plot_dir}")

    # 5. MultiQC Report Integration
    # Note: MultiQC report file will be generated in `output_dir`
    run_multiqc_integration(
        output_dir=out_dir,
        prefix=args.out_prefix,
        anova_df=anova_res,
        outliers_df=outliers_df,
        desc=desc,  # Passed for histograms
    )


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
    return p.parse_args()


if __name__ == "__main__":
    main()
