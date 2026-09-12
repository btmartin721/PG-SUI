#!/usr/bin/env python3
"""Generate manuscript tables and plots from audited N=58 PG-SUI metrics."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

MODEL_ORDER: tuple[str, ...] = (
    "ImputeAutoencoder",
    "ImputeVAE",
    "ImputeNLPCA",
    "ImputeUBP",
    "ImputeMostFrequent",
    "ImputeRefAllele",
)
STRATEGY_ORDER: tuple[str, ...] = (
    "random",
    "random_weighted",
    "random_weighted_inv",
    "nonrandom",
    "nonrandom_weighted",
)
CLASS_ORDER: tuple[str, ...] = ("REF", "HET", "ALT", "macro avg")
EXPECTED_DATASETS = 58
EXPECTED_TASKS = 290


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("analysis/tables/n58_metrics_long.tsv"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def resolve_path(root: Path, path: Path) -> Path:
    """Resolve a CLI path relative to the portable run root."""
    return path.expanduser().resolve() if path.is_absolute() else root / path


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Return Holm family-wise-error adjusted p-values."""
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return adjusted
    order = finite_indices[np.argsort(values[finite_indices], kind="mergesort")]
    running = 0.0
    family_size = order.size
    for rank, index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def validate_metrics(metrics: pd.DataFrame) -> None:
    """Validate the complete diploid 58 x 5 x 6 metric grid."""
    required = {
        "task_id",
        "dataset_id",
        "strategy",
        "model",
        "ploidy",
        "class",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "jaccard",
        "support",
        "mcc",
        "accuracy",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Metrics table is missing columns: {sorted(missing)}")
    if set(metrics["model"]) != set(MODEL_ORDER):
        raise ValueError("Metrics table model set differs from the N=58 protocol")
    if set(metrics["strategy"]) != set(STRATEGY_ORDER):
        raise ValueError("Metrics table strategy set differs from the N=58 protocol")
    expected_classes = {*CLASS_ORDER, "weighted avg"}
    if set(metrics["class"]) != expected_classes:
        raise ValueError("Metrics table class set differs from the expected reports")
    pairs = metrics[["dataset_id", "strategy", "task_id"]].drop_duplicates()
    if len(pairs) != EXPECTED_TASKS:
        raise ValueError("Metrics table does not contain 290 unique tasks")
    if pairs["dataset_id"].nunique() != EXPECTED_DATASETS:
        raise ValueError("Metrics table does not contain 58 unique datasets")
    duplicates = metrics.duplicated(["dataset_id", "strategy", "model", "class"])
    if duplicates.any():
        raise ValueError("Metrics table contains duplicate model/class task rows")
    ploidies = metrics[["task_id", "ploidy"]].drop_duplicates()
    if len(ploidies) != EXPECTED_TASKS or set(ploidies["ploidy"]) != {2}:
        raise ValueError("N=58 metrics must contain only diploid tasks")
    expected_metric_rows = EXPECTED_TASKS * len(MODEL_ORDER) * 5
    if len(metrics) != expected_metric_rows:
        raise ValueError(
            f"Expected {expected_metric_rows} ploidy-aware metric rows, "
            f"found {len(metrics)}"
        )
    expected_classes = {*CLASS_ORDER, "weighted avg"}
    for (task_id, model), frame in metrics.groupby(["task_id", "model"]):
        if set(frame["class"]) != expected_classes:
            raise ValueError(
                f"Task {task_id}/{model} class set is invalid for diploid scoring"
            )


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic tab-delimited analysis table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="NA")


def metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance distributions by class, model, and strategy."""
    values = metrics.loc[metrics["class"].isin(CLASS_ORDER)].copy()
    numeric = [
        "precision",
        "recall",
        "f1",
        "average_precision",
        "jaccard",
        "mcc",
        "accuracy",
    ]
    summary = (
        values.groupby(["class", "model", "strategy"], observed=True)[numeric]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return summary


def paired_wilcoxon(
    wide: pd.DataFrame,
    levels: tuple[str, ...],
    *,
    family: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run all paired Wilcoxon comparisons for one aligned block table."""
    rows: list[dict[str, Any]] = []
    for first_index, first in enumerate(levels):
        for second in levels[first_index + 1 :]:
            first_values = wide[first].to_numpy(float)
            second_values = wide[second].to_numpy(float)
            difference = first_values - second_values
            if np.allclose(difference, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(
                    first_values,
                    second_values,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic, p_value = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    **family,
                    "level_1": first,
                    "level_2": second,
                    "n_blocks": len(wide),
                    "wilcoxon_statistic": statistic,
                    "p_raw": p_value,
                    "mean_delta_level_1_minus_2": float(difference.mean()),
                    "median_delta_level_1_minus_2": float(np.median(difference)),
                    "level_1_win_rate": float(np.mean(difference > 0)),
                    "tie_rate": float(np.mean(np.isclose(difference, 0.0))),
                }
            )
    adjusted = holm_adjust(row["p_raw"] for row in rows)
    for row, p_holm in zip(rows, adjusted):
        row["p_holm"] = p_holm
        row["reject_holm_0_05"] = bool(p_holm < 0.05)
    return rows


def friedman_result(
    wide: pd.DataFrame,
    levels: tuple[str, ...],
    *,
    family: dict[str, Any],
) -> dict[str, Any]:
    """Run a complete-block Friedman test with an all-equal guard."""
    arrays = [wide[level].to_numpy(float) for level in levels]
    stacked = np.column_stack(arrays)
    if np.allclose(stacked, stacked[:, [0]]):
        statistic, p_value, status = 0.0, 1.0, "all_levels_equal"
    else:
        result = friedmanchisquare(*arrays)
        statistic, p_value, status = (
            float(result.statistic),
            float(result.pvalue),
            "ok",
        )
    return {
        **family,
        "n_blocks": len(wide),
        "n_levels": len(levels),
        "friedman_chi_square": statistic,
        "p_value": p_value,
        "status": status,
    }


def model_posthoc(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare models using 58 independent dataset-level blocks."""
    f1 = metrics.loc[metrics["class"].isin(CLASS_ORDER)].copy()
    dataset_means = (
        f1.groupby(["dataset_id", "class", "model"], observed=True)["f1"]
        .mean()
        .reset_index()
    )
    omnibus: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    ranks: list[dict[str, Any]] = []
    for genotype_class in CLASS_ORDER:
        subset = dataset_means.loc[dataset_means["class"].eq(genotype_class)]
        wide = subset.pivot(index="dataset_id", columns="model", values="f1")
        wide = wide.loc[:, MODEL_ORDER].dropna()
        expected_blocks = metrics.loc[
            metrics["class"].eq(genotype_class), "dataset_id"
        ].nunique()
        if len(wide) != expected_blocks:
            raise ValueError(
                f"Model comparison for {genotype_class} has {len(wide)} blocks; "
                f"expected {expected_blocks}"
            )
        family = {"class": genotype_class, "averaged_over": "five_strategies"}
        omnibus.append(friedman_result(wide, MODEL_ORDER, family=family))
        pairwise.extend(paired_wilcoxon(wide, MODEL_ORDER, family=family))
        per_dataset_ranks = wide.apply(
            lambda row: rankdata(-row.to_numpy(float), method="average"), axis=1
        )
        rank_matrix = np.vstack(per_dataset_ranks.to_numpy())
        for model, mean_rank in zip(MODEL_ORDER, rank_matrix.mean(axis=0)):
            ranks.append(
                {
                    "class": genotype_class,
                    "model": model,
                    "mean_rank": float(mean_rank),
                    "n_datasets": len(wide),
                }
            )
    return (
        pd.DataFrame(dataset_means),
        pd.DataFrame(omnibus),
        pd.DataFrame(pairwise),
        pd.DataFrame(ranks),
    )


def strategy_posthoc(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare strategies within every model using datasets as blocks."""
    f1 = metrics.loc[metrics["class"].isin(CLASS_ORDER)]
    omnibus: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    for genotype_class in CLASS_ORDER:
        expected_blocks = metrics.loc[
            metrics["class"].eq(genotype_class), "dataset_id"
        ].nunique()
        for model in MODEL_ORDER:
            subset = f1.loc[f1["class"].eq(genotype_class) & f1["model"].eq(model)]
            wide = subset.pivot(index="dataset_id", columns="strategy", values="f1")
            wide = wide.loc[:, STRATEGY_ORDER].dropna()
            if len(wide) != expected_blocks:
                raise ValueError(
                    f"Strategy comparison for {model}/{genotype_class} has "
                    f"{len(wide)} blocks; expected {expected_blocks}"
                )
            family = {"class": genotype_class, "model": model}
            omnibus.append(friedman_result(wide, STRATEGY_ORDER, family=family))
            pairwise.extend(paired_wilcoxon(wide, STRATEGY_ORDER, family=family))
    return pd.DataFrame(omnibus), pd.DataFrame(pairwise)


def winner_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return tied best-F1 models for every dataset, strategy, and class."""
    values = metrics.loc[metrics["class"].isin(CLASS_ORDER)].copy()
    group = ["dataset_id", "strategy", "class"]
    values["best_f1"] = values.groupby(group, observed=True)["f1"].transform("max")
    winners = values.loc[np.isclose(values["f1"], values["best_f1"])].copy()
    winners["n_tied_winners"] = winners.groupby(group, observed=True)[
        "model"
    ].transform("size")
    return winners[[*group, "model", "f1", "n_tied_winners"]].sort_values(
        [*group, "model"]
    )


def display_labels(values: pd.Series) -> pd.Series:
    """Remove the redundant Impute prefix for plot labels."""
    return values.str.removeprefix("Impute")


def plot_f1_boxplots(metrics: pd.DataFrame, output: Path, dpi: int) -> None:
    """Plot model F1 distributions for every class and simulation strategy."""
    values = metrics.loc[metrics["class"].isin(CLASS_ORDER)].copy()
    values["model_label"] = display_labels(values["model"])
    palette = dict(zip(STRATEGY_ORDER, sns.color_palette("colorblind", 5)))
    fig, axes = plt.subplots(2, 2, figsize=(26, 18), sharey=True)
    for axis, genotype_class in zip(axes.flat, CLASS_ORDER):
        subset = values.loc[values["class"].eq(genotype_class)]
        sns.boxplot(
            data=subset,
            x="model_label",
            y="f1",
            hue="strategy",
            hue_order=STRATEGY_ORDER,
            palette=palette,
            showfliers=False,
            ax=axis,
        )
        axis.set_title(genotype_class.replace("macro avg", "Macro average"))
        axis.set_xlabel("Model")
        axis.set_ylabel("F1 score")
        axis.tick_params(axis="x", rotation=30)
        if axis is not axes.flat[0] and axis.get_legend() is not None:
            axis.get_legend().remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[0].get_legend().remove()
    fig.legend(handles, labels, loc="upper center", ncol=5, title="Strategy")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_f1_heatmaps(metrics: pd.DataFrame, output: Path, dpi: int) -> None:
    """Plot mean F1 across the fixed model-by-strategy grid."""
    values = metrics.loc[metrics["class"].isin(CLASS_ORDER)]
    fig, axes = plt.subplots(2, 2, figsize=(22, 17))
    for axis, genotype_class in zip(axes.flat, CLASS_ORDER):
        subset = values.loc[values["class"].eq(genotype_class)]
        table = subset.pivot_table(
            index="model", columns="strategy", values="f1", aggfunc="mean"
        ).loc[MODEL_ORDER, STRATEGY_ORDER]
        table.index = table.index.str.removeprefix("Impute")
        sns.heatmap(
            table,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Mean F1"},
            ax=axis,
        )
        axis.set_title(genotype_class.replace("macro avg", "Macro average"))
        axis.set_xlabel("Simulation strategy")
        axis.set_ylabel("Model")
        axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_model_ranks(ranks: pd.DataFrame, output: Path, dpi: int) -> None:
    """Plot dataset-block mean ranks, where lower is better."""
    values = ranks.copy()
    values["model_label"] = display_labels(values["model"])
    fig, axes = plt.subplots(2, 2, figsize=(22, 16), sharex=True)
    for axis, genotype_class in zip(axes.flat, CLASS_ORDER):
        subset = values.loc[values["class"].eq(genotype_class)].sort_values("mean_rank")
        sns.barplot(
            data=subset,
            x="mean_rank",
            y="model_label",
            color=sns.color_palette("colorblind")[0],
            ax=axis,
        )
        axis.set_title(genotype_class.replace("macro avg", "Macro average"))
        axis.set_xlabel("Mean model rank (lower is better)")
        axis.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Validate audited inputs and regenerate all core post-hoc artifacts."""
    args = parse_args()
    root = args.bundle_root.expanduser().resolve()
    audit = read_json(root / "analysis" / "n58_audit_summary.json")
    provenance = read_json(root / "provenance" / "n58_run_manifest.json")
    if not audit.get("complete"):
        raise ValueError("Run the N=58 auditor successfully before post-hoc analysis")
    metrics_path = resolve_path(root, args.metrics)
    metrics = pd.read_csv(metrics_path, sep="\t")
    validate_metrics(metrics)

    tables = root / "analysis" / "tables"
    plots = root / "analysis" / "plots"
    summary = metric_summary(metrics)
    dataset_means, model_omnibus, model_pairwise, ranks = model_posthoc(metrics)
    strategy_omnibus, strategy_pairwise = strategy_posthoc(metrics)
    winners = winner_table(metrics)
    outputs = {
        "metric_summary": tables / "n58_metric_summary.tsv",
        "dataset_model_means": tables / "n58_f1_dataset_model_means.tsv",
        "model_friedman": tables / "n58_model_friedman.tsv",
        "model_pairwise": tables / "n58_model_pairwise_wilcoxon_holm.tsv",
        "model_ranks": tables / "n58_model_mean_ranks.tsv",
        "strategy_friedman": tables / "n58_strategy_friedman.tsv",
        "strategy_pairwise": tables / "n58_strategy_pairwise_wilcoxon_holm.tsv",
        "winners": tables / "n58_f1_winners.tsv",
    }
    frames = (
        summary,
        dataset_means,
        model_omnibus,
        model_pairwise,
        ranks,
        strategy_omnibus,
        strategy_pairwise,
        winners,
    )
    for frame, path in zip(frames, outputs.values()):
        write_tsv(frame, path)

    plot_paths: list[str] = []
    if not args.no_plots:
        sns.set_theme(context="talk", style="whitegrid", font_scale=1.15)
        plots.mkdir(parents=True, exist_ok=True)
        plot_stems = (
            plots / "n58_f1_by_model_strategy_class",
            plots / "n58_mean_f1_heatmaps",
            plots / "n58_model_mean_ranks",
        )
        plot_f1_boxplots(metrics, plot_stems[0], args.dpi)
        plot_f1_heatmaps(metrics, plot_stems[1], args.dpi)
        plot_model_ranks(ranks, plot_stems[2], args.dpi)
        plot_paths = [
            str(path.relative_to(root))
            for stem in plot_stems
            for path in (stem.with_suffix(".pdf"), stem.with_suffix(".png"))
        ]

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pgsui_version": provenance["pgsui_version"],
        "snpio_version": provenance["snpio_version"],
        "pgsui_git_revision": provenance["pgsui_git_revision"],
        "pgsui_git_dirty": provenance["pgsui_git_dirty"],
        "pgsui_source_sha256": provenance["pgsui_source_sha256"],
        "benchmark_profile": provenance["profile"],
        "source_metrics": str(metrics_path.relative_to(root)),
        "dataset_count": metrics["dataset_id"].nunique(),
        "task_count": metrics["task_id"].nunique(),
        "models": list(MODEL_ORDER),
        "strategies": list(STRATEGY_ORDER),
        "primary_model_inference": (
            "Friedman and paired Wilcoxon tests across 58 dataset blocks after "
            "averaging F1 over the five strategies; Holm correction within class"
        ),
        "strategy_inference": (
            "Friedman and paired Wilcoxon tests across 58 dataset blocks within "
            "each model and class; Holm correction within model-class family"
        ),
        "tables": {name: str(path.relative_to(root)) for name, path in outputs.items()},
        "plots": plot_paths,
    }
    manifest_path = root / "analysis" / "n58_posthoc_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
