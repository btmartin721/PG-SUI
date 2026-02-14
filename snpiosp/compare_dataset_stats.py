#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recursively aggregate per-locus summary stats across many datasets and test which metrics differ by dataset.
Includes diagnostic debugging for dataset completion gating.
"""

from __future__ import annotations

import argparse
import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

# Use non-interactive backend
plt.switch_backend("Agg")

# Updated regex to allow optional underscore (e.g. results01 OR results_01)
_RESULTS_KEY_RE = re.compile(r"(results[_]?\d+)", flags=re.IGNORECASE)

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
    "F_inbreeding": r"$F_{IS}$",
    "ThetaWatt": r"Watterson's $\theta$",
    "Pi": r"Nucleotide Diversity ($\pi$)",
    "TajimaD": r"Tajima's $D$",
    "Missingness": "Missingness Prop.",
    "Ho": r"$H_o$",
    "He": r"$H_e$",
    "HetzPositions": r"Heterozygous Sites",
    "SegSites": r"$S$ (Seg. Sites)",
    "Singletons": r"Singleton Prop.",
    "Hap": r"Haplotype Count",
    "Hd": r"Haplotype Diversity ($H_d$)",
    "Sample_Size": r"Sample Size",
    "MAF": r"Minor Allele Frequency",
}


@dataclass
class DatasetFiles:
    dataset_id: str
    locus_csv: Path
    summary_json: Optional[Path]


def analyze_deviations(
    desc: pd.DataFrame, metrics: Sequence[str], out_dir: Path, out_prefix: str
) -> pd.DataFrame:
    """Post-hoc "Analysis of Means" to identify which datasets contribute most to variance."""
    deviation_rows = []

    for m in metrics:
        mean_col = f"{m}_mean"
        if mean_col not in desc.columns:
            continue

        series = desc.set_index("dataset")[mean_col].astype(float)
        grand_mean = series.mean()
        std_dev = series.std(ddof=1)

        if std_dev == 0:
            continue

        z_scores = (series - grand_mean) / std_dev

        for dataset, z in z_scores.items():
            deviation_rows.append(
                {
                    "dataset": dataset,
                    "metric": m,
                    "dataset_mean": series.loc[series.index == dataset].iloc[0],
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
    dev_df = dev_df.sort_values("abs_z_score", ascending=False)
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

    if s not in aliases:
        raise ValueError(
            f"Unrecognized model name variant: '{val}' (normalized to '{s}')."
        )

    return aliases[s]


def plot_metric_histograms(
    combined: pd.DataFrame,
    desc: pd.DataFrame,
    metric: str,
    outpath: Path,
) -> None:
    mean_col = f"{metric}_mean"
    if mean_col not in desc.columns:
        return

    dataset_means = desc[mean_col].dropna()
    raw_values = combined[metric].dropna()

    if dataset_means.empty or raw_values.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

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
    sub = anova_res.dropna(subset=["eta_sq"]).copy()
    if sub.empty:
        return
    sub = sub.sort_values("eta_sq", ascending=False)
    sub["metric"] = sub["metric"].map(METRIC_MAP).fillna(sub["metric"])

    cmap = sns.color_palette("viridis", as_cmap=True, n_colors=len(sub))
    norm = mcolors.Normalize(
        vmin=min(sub["eta_sq"].min(), 0), vmax=max(sub["eta_sq"].max(), 1)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.barplot(
        data=sub,
        x="metric",
        y="eta_sq",
        hue="eta_sq",
        hue_norm=norm,
        palette=cmap,  # type: ignore
        edgecolor="k",
        linewidth=0.7,
        ax=ax,
        errorbar=None,
        legend=False,
    )
    ax.set_ylabel("η² (Effect Size)")
    ax.set_xlabel("Summary Statistic")
    ax.set_title("Metric Differentiation Power (from Welch's ANOVA)", pad=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("η² (Eta-squared) Value")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_dataset_metric_heatmap(
    desc: pd.DataFrame,
    metrics: Sequence[str],
    outpath: Path,
    zscore: bool = True,
) -> None:
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
        cmap = copy.copy(cm.get_cmap("viridis"))
    cmap.set_bad(color="lightgrey")

    im = ax.imshow(masked_mat, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_title(
        f"Per-dataset Mean Profiles ({'Z-scored' if zscore else 'Raw'})", pad=20
    )
    ax.set_xticks(np.arange(n_metrics))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_datasets))

    y_fontsize = 8 if n_datasets < 60 else 6 if n_datasets < 120 else 4
    ax.set_yticklabels(mat.index)

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
    df["dataset_raw"] = dset.dataset_id

    dataset_key = dset.dataset_id
    if "Filename" in df.columns:
        first_fn = df["Filename"].dropna().astype(str)
        if not first_fn.empty:
            extracted = extract_dataset_key_from_filename(first_fn.iloc[0])
            if extracted:
                dataset_key = extracted

    dataset_key = str(dataset_key).strip().lower()
    df["dataset_key"] = dataset_key
    df["dataset"] = dataset_key
    df["sim_strategy"] = "Unknown"
    return df


def deduplicate_datasets_by_key(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset_key" not in df.columns:
        return df

    counts = (
        df.groupby(["dataset_key", "dataset"])
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


def extract_dataset_key_from_filename(filename: str) -> Optional[str]:
    if not isinstance(filename, str):
        return None
    m = _RESULTS_KEY_RE.search(filename)
    return m.group(1).lower().replace("_", "") if m else None


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
            res_obj = anova_oneway(groups, use_var="unequal", welch_correction=True)

        F_val = float(getattr(res_obj, "statistic", np.nan))
        p_val = float(getattr(res_obj, "pvalue", np.nan))
        n_groups = int(getattr(res_obj, "n_groups", len(groups)))
        nobs_t = int(getattr(res_obj, "nobs_t", sum(len(g) for g in groups)))

        all_vals = np.concatenate(groups)
        grand_mean = all_vals.mean()
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        ss_between = sum([v.size * (v.mean() - grand_mean) ** 2 for v in groups])
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

        results.append(
            {
                "metric": metric,
                "welch_F": F_val,
                "p_raw": p_val,
                "eta_sq": float(eta_sq),
                "n_datasets": n_groups,
                "n_loci_total": nobs_t,
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
    try:
        from snpio import SNPioMultiQC
    except ImportError:
        print(
            "[Warning] SNPio is not installed. Skipping MultiQC generation. Run 'pip install snpio' to enable this feature."
        )
        return

    print(f"Queuing results for MultiQC report...")

    if not anova_df.empty:
        disp_df = anova_df.sort_values("eta_sq", ascending=False).copy()
        disp_df["metric_id"] = disp_df["metric"].astype(str)

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

        metric_ids = disp_df["metric_id"].tolist()

        for metric_id in metric_ids:
            mean_col = f"{metric_id}_mean"
            if mean_col not in desc.columns:
                continue

            data = desc[mean_col].dropna()
            if data.empty:
                continue

            row = disp_df.loc[disp_df["metric_id"] == metric_id].iloc[0]
            metric_display = str(row["metric"])

            counts, edges = np.histogram(data, bins=15)

            bin_labels = [
                (
                    f"{edges[i]:.2G}-{edges[i+1]:.2G}"
                    if edges[i] <= 1
                    else f"{edges[i]:.1f}-{edges[i+1]:.1f}"
                )
                for i in range(len(edges) - 1)
            ]

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

    print("Building MultiQC report...")
    SNPioMultiQC.build(
        prefix=prefix,
        output_dir=str(output_dir),
        title="SNPioSumStats Validation Dataset Report",
        overwrite=True,
    )
    print("MultiQC Report generation queued/built successfully.")


def load_completed_dataset_labels(
    metrics_long_csv: Path,
    *,
    require_baseline_model: str = "RefAllele",
    require_all_strategies: bool = True,
    expected_strategies: Optional[set[str]] = None,
    debug_missing: bool = False,
) -> set[str]:
    """Load completed dataset KEYS (resultsNNN) from zygosity_metrics_long.csv."""
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
            f"Completion gating requires columns {sorted(required_cols)} in metrics CSV. Missing: {sorted(missing)}"
        )

    # 1. Initial Load
    sub = df.loc[:, ["dataset", "sim_strategy", "model"]].copy()
    sub["dataset"] = sub["dataset"].astype(str)
    raw_unique_datasets = sub["dataset"].nunique()
    print(
        f"[Completion Gate] Step 1: Found {raw_unique_datasets} unique datasets in CSV."
    )

    # 2. Key Extraction Filter
    sub["join_key"] = sub["dataset"].map(_extract_join_key)
    failed_keys = sub[sub["join_key"].isna()]["dataset"].unique()
    if len(failed_keys) > 0:
        print(
            f"[Completion Gate] Warning: Could not extract 'resultsNN' key from {len(failed_keys)} datasets (e.g., {failed_keys[:3]})."
        )

    sub = sub.dropna(subset=["join_key"])
    unique_with_keys = sub["join_key"].nunique()
    print(
        f"[Completion Gate] Step 2: {unique_with_keys} datasets have valid join keys (e.g., 'results01')."
    )

    # 3. Model Filter
    sub["sim_strategy_raw"] = sub["sim_strategy"].astype(str)

    sub["sim_strategy"] = sub["sim_strategy"].map(_canonicalize_strategy)
    sub["model"] = sub["model"].astype(str).map(_normalize_model_name)

    baseline = _normalize_model_name(require_baseline_model)

    model_counts = sub.groupby("join_key")["model"].apply(
        lambda x: baseline in x.values
    )
    datasets_with_model = model_counts[model_counts].index.tolist()

    if len(datasets_with_model) < unique_with_keys:
        print(
            f"[Completion Gate] CRITICAL: Only {len(datasets_with_model)} datasets contain the model '{baseline}'."
        )
        print(
            f"                  (Dropped {unique_with_keys - len(datasets_with_model)} datasets because they lack '{baseline}' rows)"
        )

    sub = sub[sub["model"].eq(baseline)]
    if sub.empty:
        print(f"[Warning] No rows found for baseline model '{baseline}' (normalized).")
        return set()

    observed_strategies = set(sub["sim_strategy"].dropna().unique().tolist())
    observed_strategies.discard("Unknown")

    if expected_strategies is None:
        expected_strategies = observed_strategies

    # Group by join key
    strat_by_ds: dict[str, set[str]] = (
        sub.groupby("join_key")["sim_strategy"].apply(lambda x: set(x)).to_dict()
    )

    raw_strat_by_ds: dict[str, list[str]] = (
        sub.groupby("join_key")["sim_strategy_raw"]
        .apply(lambda x: sorted(list(set(x))))
        .to_dict()
    )

    # Always summarize strategy USED + MISSING counts (strict ON or OFF) Counts are per-dataset (a dataset contributes at most 1 to a strategy).
    expected = set(expected_strategies)  # alias

    # Explode canonical strategies present per dataset
    used_records: list[tuple[str, str]] = []
    for ds_key, sset in strat_by_ds.items():
        # Optionally ignore "Unknown" even if it somehow got in
        for strat in set(sset) - {"Unknown"}:
            used_records.append((ds_key, str(strat)))

    used_df = pd.DataFrame(used_records, columns=["dataset_key", "strategy_used"])

    used_counts = (
        used_df["strategy_used"]
        .value_counts()
        .reindex(sorted(expected), fill_value=0)  # ensure all expected shown
    )

    # MISSING: explode missing strategies per dataset relative to expected set
    miss_records: list[tuple[str, str]] = []
    for ds_key, sset in strat_by_ds.items():
        missing_set = expected - sset
        if not missing_set:
            miss_records.append((ds_key, "<NONE>"))
        else:
            for strat in missing_set:
                miss_records.append((ds_key, str(strat)))

    miss_df = pd.DataFrame(miss_records, columns=["dataset_key", "strategy_missing"])

    missing_counts = (
        miss_df.loc[miss_df["strategy_missing"].ne("<NONE>"), "strategy_missing"]
        .value_counts()
        .reindex(sorted(expected), fill_value=0)  # ensure all expected shown
    )

    n_complete = int((miss_df["strategy_missing"] == "<NONE>").sum())
    n_total = len(strat_by_ds)

    # Combine into one table for a clean print
    summary = (
        pd.DataFrame(
            {
                "strategy": sorted(expected),
                "n_datasets_used": [
                    int(used_counts.get(s, 0)) for s in sorted(expected)
                ],
                "n_datasets_missing": [
                    int(missing_counts.get(s, 0)) for s in sorted(expected)
                ],
            }
        )
        .sort_values(["n_datasets_missing", "n_datasets_used"], ascending=[False, True])
        .reset_index(drop=True)
    )

    print("\n[Completion Gate] Strategy coverage (per-dataset counts):")
    print(summary.to_string(index=False))
    print(
        f"\n[Completion Gate] Complete datasets (missing nothing): {n_complete}/{n_total}"
    )
    # ------------------------------------------------------------------

    if require_all_strategies:
        complete = {
            ds_key
            for ds_key, sset in strat_by_ds.items()
            if expected_strategies.issubset(sset)
        }

        if debug_missing:
            dropped_keys = set(strat_by_ds.keys()) - complete
            print(
                f"[Completion Gate] Step 3: {len(complete)} datasets passed strict strategy check (Required: {len(expected_strategies)} strategies)."
            )

            if dropped_keys:
                print(
                    f"[Completion Gate][Debug] Dropping {len(dropped_keys)} datasets due to missing strategies."
                )
                miss_rows_dbg = []
                sorted_dropped = sorted(list(dropped_keys))

                print(f"[Completion Gate][Debug] Sample Rejected Datasets:")
                for ds_key in sorted_dropped[:5]:
                    sset = strat_by_ds[ds_key]
                    raw_s = raw_strat_by_ds.get(ds_key, [])
                    missing = sorted(expected_strategies - sset)

                    print(f"  -- Dataset: {ds_key}")
                    print(f"     Found Canonical: {sorted(sset)}")
                    print(f"     Raw Strategies: {raw_s}")
                    print(f"     MISSING: {missing}")
                    miss_rows_dbg.append(
                        (ds_key, ",".join(missing) if missing else "<NONE>")
                    )

                if len(sorted_dropped) > 5:
                    for ds_key in sorted_dropped[5:]:
                        sset = strat_by_ds[ds_key]
                        missing = sorted(expected_strategies - sset)
                        miss_rows_dbg.append(
                            (ds_key, ",".join(missing) if missing else "<NONE>")
                        )

                if miss_rows_dbg:
                    miss_df_dbg = pd.DataFrame(
                        miss_rows_dbg, columns=["dataset_key", "missing_strategies"]
                    )
                    top = (
                        miss_df_dbg["missing_strategies"]
                        .value_counts()
                        .reset_index()
                        .rename(
                            columns={
                                "index": "missing_strategies",
                                "missing_strategies": "n_datasets",
                            }
                        )
                    )
                    print(
                        "\n[Completion Gate][Debug] Top missing-strategy patterns summary:"
                    )
                    print(top.head(15).to_string(index=False))

        return complete

    return set(strat_by_ds.keys())


def _extract_join_key(val: str) -> Optional[str]:
    if not isinstance(val, str):
        return None
    m = _RESULTS_KEY_RE.search(val)
    if m:
        return m.group(1).lower().replace("_", "")
    return None


def _canonicalize_strategy(val: str) -> str:
    """Canonicalize simulation strategy labels to your display names.

    Strict, mutually exclusive matching order.
    """
    s = str(val).lower().strip()

    # 1. Random Weighted Inv
    if "random" in s and "weighted" in s and ("inv" in s or "inverse" in s):
        return "Random Weighted Inv"

    # 2. Nonrandom Weighted (Check before Nonrandom)
    if "nonrandom" in s and "weighted" in s:
        return "Nonrandom Weighted"

    # 3. Random Weighted (Check before Random)
    if "random" in s and "weighted" in s:
        return "Random Weighted"

    # 4. Nonrandom
    if "nonrandom" in s:
        return "Nonrandom"

    # 5. Random
    if "random" in s:
        return "Random"

    return "Unknown"


def main() -> None:
    args = parse_args()
    root: Path = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

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

    if args.completed_metrics_long is not None:
        metrics_path = args.completed_metrics_long.resolve()
        strict = args.strict_strategies

        print(
            f"[Completion gate] Strategy Strictness: {'STRICT (All 5)' if strict else 'PERMISSIVE (Any 1)'}"
        )

        allowed_keys = load_completed_dataset_labels(
            metrics_path,
            require_baseline_model="RefAllele",
            require_all_strategies=strict,
            expected_strategies=ALL_STRATEGIES if strict else None,
            debug_missing=True,
        )

        before_keys = set(combined["dataset_key"].astype(str).unique().tolist())
        inter = before_keys & allowed_keys

        print(
            f"[Completion gate] Allowed dataset_keys (from {metrics_path.name}): {len(allowed_keys)}\n"
            f"[Completion gate] dataset_keys in locus stats before:            {len(before_keys)}\n"
            f"[Completion gate] Intersection size:                             {len(inter)}"
        )

        combined = combined[combined["dataset_key"].isin(allowed_keys)].copy()

        if combined.empty:
            ex_allowed = sorted(list(allowed_keys))[:10]
            ex_have = sorted(list(before_keys))[:10]
            raise ValueError(
                "After completion gating, no datasets remained.\n"
                f"- Example allowed dataset_keys: {ex_allowed}\n"
                f"- Example locus dataset_keys:   {ex_have}\n"
                "This indicates a join-key mismatch (extract/join_key parsing) or missing baseline rows."
            )

        n_after = combined["dataset_key"].nunique()
        print(f"[Completion gate] Datasets retained after: {n_after}")

    metrics = args.metrics if args.metrics else infer_metrics(combined)
    if not metrics:
        raise ValueError("No numeric metrics found to test.")
    print(f"Analyzing metrics: {metrics}")

    desc = per_dataset_descriptives(combined, metrics)
    summary_csv_path = out_dir / f"{args.out_prefix}_all_datasets_summary_stats.csv"
    desc.to_csv(summary_csv_path, index=False)

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

        anova_res = anova_res.drop(columns=["welch_F"])
        anova_res["eta_sq"] = anova_res["eta_sq"].astype(float)

    print("Running Post-Hoc Outlier Analysis...")
    outliers_df = analyze_deviations(desc, metrics, out_dir, args.out_prefix)

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

        for m in sorted_metrics[:10]:
            plot_metric_histograms(
                combined, desc, m, plot_dir / f"{m}_distributions.png"
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
