"""Tests for the strict N=58 post-hoc analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_pgsui_n58_feature_effects import (
    analyze_features,
    plot_coefficients,
)
from scripts.analyze_pgsui_n58_results import (
    CLASS_ORDER,
    MODEL_ORDER,
    STRATEGY_ORDER,
    holm_adjust,
    model_posthoc,
    strategy_posthoc,
    validate_metrics,
    winner_table,
)


def complete_metrics() -> pd.DataFrame:
    """Create a complete grid with 58 diploid datasets."""
    rows = []
    classes = (*CLASS_ORDER, "weighted avg")
    for dataset_index in range(58):
        ploidy = 2
        for strategy_index, strategy in enumerate(STRATEGY_ORDER):
            task_id = dataset_index * len(STRATEGY_ORDER) + strategy_index
            for model_index, model in enumerate(MODEL_ORDER):
                for class_index, genotype_class in enumerate(classes):
                    f1 = (
                        0.40
                        + model_index * 0.05
                        + strategy_index * 0.01
                        + class_index * 0.005
                        + dataset_index * 0.0001
                    )
                    rows.append(
                        {
                            "task_id": task_id,
                            "dataset_id": f"results{dataset_index + 1}",
                            "strategy": strategy,
                            "model": model,
                            "ploidy": ploidy,
                            "class": genotype_class,
                            "precision": f1,
                            "recall": f1,
                            "f1": f1,
                            "average_precision": f1,
                            "jaccard": f1,
                            "support": 10,
                            "mcc": f1,
                            "accuracy": f1,
                        }
                    )
    return pd.DataFrame(rows)


def test_holm_adjust_is_monotonic_in_sorted_p_values() -> None:
    observed = holm_adjust([0.01, 0.04, 0.03, np.nan])
    np.testing.assert_allclose(observed[:3], [0.03, 0.06, 0.06])
    assert np.isnan(observed[3])


def test_validate_metrics_rejects_duplicate_grid_rows() -> None:
    metrics = complete_metrics()
    validate_metrics(metrics)
    duplicated = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_metrics(duplicated)


def test_posthoc_uses_fifty_eight_dataset_blocks() -> None:
    metrics = complete_metrics()
    means, model_omnibus, model_pairs, ranks = model_posthoc(metrics)
    strategy_omnibus, strategy_pairs = strategy_posthoc(metrics)
    winners = winner_table(metrics)

    assert len(means) == 58 * len(CLASS_ORDER) * len(MODEL_ORDER)
    assert model_omnibus.set_index("class")["n_blocks"].to_dict() == {
        "REF": 58,
        "HET": 58,
        "ALT": 58,
        "macro avg": 58,
    }
    assert model_pairs["n_blocks"].eq(58).all()
    assert ranks["n_datasets"].eq(58).all()
    assert strategy_omnibus["n_blocks"].eq(58).all()
    assert strategy_pairs["n_blocks"].eq(58).all()
    assert winners["model"].eq(MODEL_ORDER[-1]).all()


def test_feature_analysis_uses_dataset_level_covariates() -> None:
    metrics = complete_metrics()
    index = np.arange(58, dtype=float)
    stats = pd.DataFrame(
        {
            "dataset": [f"results{value + 1}" for value in range(58)],
            "Pi_mean": 0.1 + index / 1_000,
            "Missingness_mean": 0.01 + (index % 7) / 100,
            "Sample_Size_mean": 20 + index,
            "N_Loci": 100 + index**2,
        }
    )
    results, features = analyze_features(stats, metrics)
    assert set(features) == {
        "Pi_mean",
        "Missingness_mean",
        "Sample_Size_mean",
        "N_Loci",
    }
    assert len(results) == len(CLASS_ORDER) * len(MODEL_ORDER) * len(features)
    assert results["n_datasets"].eq(58).all()
    assert {
        "ols_p_holm_global",
        "ols_q_bh_global",
        "spearman_p_holm_global",
        "spearman_q_bh_global",
    }.issubset(results.columns)


def test_feature_coefficient_plot_handles_no_estimable_models(tmp_path) -> None:
    results = pd.DataFrame(
        {
            "class": ["REF"],
            "status": ["insufficient_variation"],
            "coefficient_per_sd": [np.nan],
        }
    )
    output = tmp_path / "coefficients"

    plot_coefficients(results, output, dpi=72)

    assert output.with_suffix(".pdf").is_file()
    assert output.with_suffix(".png").is_file()
