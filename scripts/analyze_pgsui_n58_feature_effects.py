#!/usr/bin/env python3
"""Relate audited N=58 F1 scores to corrected dataset-level covariates."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

MODEL_ORDER: tuple[str, ...] = (
    "ImputeAutoencoder",
    "ImputeVAE",
    "ImputeNLPCA",
    "ImputeUBP",
    "ImputeMostFrequent",
    "ImputeRefAllele",
)
CLASS_ORDER: tuple[str, ...] = ("REF", "HET", "ALT", "macro avg")
FEATURE_LABELS: dict[str, str] = {
    "F_inbreeding_mean": "Mean inbreeding coefficient",
    "ThetaWatt_Per_Site_mean": "Mean Watterson theta per site",
    "Pi_mean": "Mean nucleotide diversity",
    "TajimaD_CompleteSites": "Tajima's D (complete-call sites)",
    "Missingness_mean": "Mean natural missingness",
    "Ho_mean": "Mean observed heterozygosity",
    "He_mean": "Mean expected heterozygosity",
    "SegSites_mean": "Proportion segregating sites",
    "Singletons_mean": "Proportion singleton sites",
    "Genotype_Diversity_mean": "Mean genotype-category diversity",
    "Sample_Size_mean": "Mean called sample size",
    "MAF_mean": "Mean minor-allele frequency",
    "SNPio_LD_r2D": "Unbiased unphased LD r2D",
    "N_Loci": "Locus count",
}
DEFAULT_CONTROLS: tuple[str, ...] = (
    "Missingness_mean",
    "Sample_Size_mean",
    "N_Loci",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path(
            "analysis/popgen_stats/comparison/n58_popgen_all_datasets_summary_stats.csv"
        ),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("analysis/tables/n58_metrics_long.tsv"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    """Resolve one path relative to the portable run root."""
    return path.expanduser().resolve() if path.is_absolute() else root / path


def numeric_feature_columns(stats: pd.DataFrame) -> list[str]:
    """Return recognized, nonconstant scientific feature columns."""
    return [
        feature
        for feature in FEATURE_LABELS
        if feature in stats.columns
        and pd.to_numeric(stats[feature], errors="coerce").nunique(dropna=True) > 1
    ]


def independent_design_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Keep ordered nonconstant columns that increase numeric matrix rank."""
    selected: list[str] = []
    current_rank = 0
    for column in columns:
        candidate = [*selected, column]
        values = frame[candidate].to_numpy(float)
        rank = int(np.linalg.matrix_rank(values))
        if rank > current_rank:
            selected.append(column)
            current_rank = rank
    return selected


def one_feature_model(
    frame: pd.DataFrame,
    feature: str,
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
) -> dict[str, Any]:
    """Fit one standardized focal-feature OLS model with HC3 covariance."""
    available_controls = [
        control
        for control in controls
        if control != feature and control in frame.columns
    ]
    columns = [feature, *available_controls, "f1"]
    data = frame[columns].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    base = {
        "feature": feature,
        "feature_label": FEATURE_LABELS[feature],
        "n_datasets": len(data),
    }
    if len(data) < 10 or data[feature].nunique() < 2 or data["f1"].nunique() < 2:
        return {**base, "status": "insufficient_variation"}

    design_candidates = [feature, *available_controls]
    standardized = data[design_candidates].copy()
    standard_deviation = standardized.std(ddof=0)
    nonconstant = standard_deviation.gt(0)
    standardized = standardized.loc[:, nonconstant]
    standardized = (standardized - standardized.mean()) / standardized.std(ddof=0)
    selected = independent_design_columns(
        standardized,
        [column for column in design_candidates if column in standardized.columns],
    )
    if feature not in selected:
        return {**base, "status": "focal_feature_collinear"}
    if len(data) <= len(selected) + 2:
        return {**base, "status": "insufficient_degrees_of_freedom"}

    design = sm.add_constant(standardized[selected], has_constant="add")
    fitted = sm.OLS(data["f1"], design).fit(cov_type="HC3")
    rho, spearman_p = spearmanr(data[feature], data["f1"])
    return {
        **base,
        "status": "ok",
        "coefficient_per_sd": float(fitted.params[feature]),
        "standard_error_hc3": float(fitted.bse[feature]),
        "p_value_hc3": float(fitted.pvalues[feature]),
        "ci95_low_hc3": float(fitted.conf_int().loc[feature, 0]),
        "ci95_high_hc3": float(fitted.conf_int().loc[feature, 1]),
        "r_squared": float(fitted.rsquared),
        "spearman_rho": float(rho),
        "spearman_p_value": float(spearman_p),
        "controls_retained": " ".join(
            column for column in selected if column != feature
        ),
    }


def adjust_p_values(results: pd.DataFrame) -> pd.DataFrame:
    """Add global Holm and Benjamini-Hochberg adjusted p-values."""
    output = results.copy()
    for source, prefix in (
        ("p_value_hc3", "ols"),
        ("spearman_p_value", "spearman"),
    ):
        output[f"{prefix}_p_holm_global"] = np.nan
        output[f"{prefix}_q_bh_global"] = np.nan
        valid = output[source].notna()
        if valid.any():
            values = output.loc[valid, source].to_numpy(float)
            output.loc[valid, f"{prefix}_p_holm_global"] = multipletests(
                values, method="holm"
            )[1]
            output.loc[valid, f"{prefix}_q_bh_global"] = multipletests(
                values, method="fdr_bh"
            )[1]
    return output


def analyze_features(
    stats: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Fit one feature model per genotype class and imputation model."""
    if "dataset" not in stats.columns:
        raise ValueError("Population-statistics table requires a dataset column")
    stats = stats.copy()
    stats["dataset_id"] = stats["dataset"].astype(str).str.lower()
    if stats["dataset_id"].duplicated().any():
        raise ValueError("Population-statistics table contains duplicate datasets")
    features = numeric_feature_columns(stats)
    if not features:
        raise ValueError("No recognized nonconstant population-genetic features")

    primary = (
        metrics.loc[metrics["class"].isin(CLASS_ORDER)]
        .groupby(["dataset_id", "class", "model"], observed=True)["f1"]
        .mean()
        .reset_index()
    )
    merged = primary.merge(
        stats[["dataset_id", *features]],
        on="dataset_id",
        how="inner",
        validate="many_to_one",
    )
    if merged["dataset_id"].nunique() != 58:
        raise ValueError(
            "Feature analysis requires population statistics for all 58 datasets"
        )

    rows: list[dict[str, Any]] = []
    for genotype_class in CLASS_ORDER:
        for model in MODEL_ORDER:
            subset = merged.loc[
                merged["class"].eq(genotype_class) & merged["model"].eq(model)
            ]
            for feature in features:
                rows.append(
                    {
                        "class": genotype_class,
                        "model": model,
                        "outcome": "F1 averaged over five strategies",
                        **one_feature_model(subset, feature),
                    }
                )
    return adjust_p_values(pd.DataFrame(rows)), features


def plot_coefficients(results: pd.DataFrame, output: Path, dpi: int) -> None:
    """Plot adjusted standardized feature coefficients by model and class."""
    valid = results.loc[results["status"].eq("ok")].copy()
    maximum = float(valid["coefficient_per_sd"].abs().max())
    if not np.isfinite(maximum) or maximum == 0:
        maximum = 1.0
    fig, axes = plt.subplots(2, 2, figsize=(25, 20))
    for axis, genotype_class in zip(axes.flat, CLASS_ORDER):
        subset = valid.loc[valid["class"].eq(genotype_class)]
        if subset.empty:
            axis.text(
                0.5,
                0.5,
                "No estimable coefficients",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_title(genotype_class.replace("macro avg", "Macro average"))
            axis.set_axis_off()
            continue
        table = subset.pivot(
            index="feature_label", columns="model", values="coefficient_per_sd"
        )
        table = table.reindex(columns=MODEL_ORDER)
        table.columns = table.columns.str.removeprefix("Impute")
        sns.heatmap(
            table,
            cmap="vlag",
            center=0,
            vmin=-maximum,
            vmax=maximum,
            cbar_kws={"label": "F1 change per feature SD"},
            ax=axis,
        )
        axis.set_title(genotype_class.replace("macro avg", "Macro average"))
        axis.set_xlabel("Model")
        axis.set_ylabel("Dataset feature")
        axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Run the corrected, dataset-blocked feature analysis."""
    args = parse_args()
    root = args.bundle_root.expanduser().resolve()
    with (root / "provenance" / "n58_run_manifest.json").open(
        encoding="utf-8"
    ) as handle:
        provenance = json.load(handle)
    stats_path = resolve(root, args.stats)
    metrics_path = resolve(root, args.metrics)
    stats = pd.read_csv(stats_path)
    metrics = pd.read_csv(metrics_path, sep="\t")
    results, features = analyze_features(stats, metrics)
    output = root / "analysis" / "feature_effects"
    output.mkdir(parents=True, exist_ok=True)
    table_path = output / "n58_dataset_feature_effects.tsv"
    results.to_csv(table_path, sep="\t", index=False, na_rep="NA")

    plots: list[str] = []
    if not args.no_plots:
        sns.set_theme(context="talk", style="white", font_scale=1.05)
        plot_stem = output / "n58_adjusted_feature_coefficients"
        plot_coefficients(results, plot_stem, args.dpi)
        plots = [
            str(plot_stem.with_suffix(suffix).relative_to(root))
            for suffix in (".pdf", ".png")
        ]
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pgsui_version": provenance["pgsui_version"],
        "snpio_version": provenance["snpio_version"],
        "pgsui_git_revision": provenance["pgsui_git_revision"],
        "pgsui_git_dirty": provenance["pgsui_git_dirty"],
        "pgsui_source_sha256": provenance["pgsui_source_sha256"],
        "statistics_source": str(stats_path.relative_to(root)),
        "metrics_source": str(metrics_path.relative_to(root)),
        "features": features,
        "controls_requested": list(DEFAULT_CONTROLS),
        "model": (
            "F1 ~ standardized focal feature + noncollinear standardized "
            "controls, OLS with HC3 covariance"
        ),
        "independent_unit": "dataset",
        "dataset_count": int(metrics["dataset_id"].nunique()),
        "table": str(table_path.relative_to(root)),
        "plots": plots,
    }
    (output / "n58_feature_effects_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
