from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "pgsui_metric_feature_interactions.py"
)
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SPEC = importlib.util.spec_from_file_location(
    "pgsui_metric_feature_interactions", SCRIPT_PATH
)
assert SPEC is not None
metric = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = metric
assert SPEC.loader is not None
SPEC.loader.exec_module(metric)


def test_dataset_refit_bootstrap_resamples_join_keys_and_refits_ols() -> None:
    n_datasets = 20
    dataset_index = pd.Series(range(1, n_datasets + 1), dtype=float)
    base = pd.DataFrame(
        {
            "join_key": [f"results{i}" for i in range(1, n_datasets + 1)],
            "Pi": (0.005 + 0.002 * dataset_index + 0.0003 * ((dataset_index * 7) % 5)),
            "Missingness": 0.08 + 0.01 * ((dataset_index * 7) % 13),
            "Sample_Size": 10 + ((dataset_index * 11) % 23),
            "Locus_Count": 100 + ((dataset_index * 37) % 151),
        }
    )
    rows = []
    for model, offset in (("ImputeMostFrequent", 0.0), ("ImputeVAE", 0.08)):
        sub = base.copy()
        sub["model"] = model
        sub["sim_strategy"] = "Random"
        sub["REF_F1score"] = (
            0.50
            + offset
            + 2.0 * sub["Pi"]
            - 0.20 * sub["Missingness"]
            + 0.001 * sub["Sample_Size"]
            + 0.0002 * ((dataset_index * 13) % 7)
        )
        rows.append(sub)
    merged = pd.concat(rows, ignore_index=True)

    boot = metric.compute_dataset_refit_bootstrap_effects(
        merged,
        ["Pi", "Missingness", "Sample_Size", "Locus_Count"],
        ["REF_F1score"],
        ["Missingness", "Sample_Size", "Locus_Count"],
        min_n=5,
        n_boot=3,
        random_seed=123,
        ols_mode="adjusted_bootstrap_refit",
        n_jobs=1,
        show_progress=False,
    )

    assert not boot.empty
    assert set(boot["bootstrap_replicate"]).issubset({0, 1, 2})
    assert set(boot["feature"]) == {"Pi"}
    assert set(boot["model"]) == {"ImputeMostFrequent", "ImputeVAE"}
    assert set(boot["ols_mode"]) == {"adjusted_bootstrap_refit"}
    assert boot["n_datasets"].min() >= 5
    assert {"std_coef", "partial_r2", "n_unique_join_keys"}.issubset(boot.columns)

    parallel = metric.compute_dataset_refit_bootstrap_effects(
        merged,
        ["Pi", "Missingness", "Sample_Size", "Locus_Count"],
        ["REF_F1score"],
        ["Missingness", "Sample_Size", "Locus_Count"],
        min_n=5,
        n_boot=3,
        random_seed=123,
        ols_mode="adjusted_bootstrap_refit",
        n_jobs=2,
        backend="thread",
        show_progress=False,
    )
    pd.testing.assert_frame_equal(
        boot.reset_index(drop=True),
        parallel.reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_resolve_bootstrap_n_jobs_uses_joblib_style_negative_counts() -> None:
    assert metric._resolve_bootstrap_n_jobs(0, 10) == 1
    assert metric._resolve_bootstrap_n_jobs(4, 2) == 2
    assert metric._resolve_bootstrap_n_jobs(-1, 2) == 2


def test_deep_minus_baseline_summary_uses_dataset_refit_replicates() -> None:
    observed = pd.DataFrame(
        {
            "feature": ["Pi", "Pi", "Pi"],
            "target": ["REF_F1score"] * 3,
            "model": ["ImputeMostFrequent", "ImputeAutoencoder", "ImputeVAE"],
            "std_coef": [0.2, 0.5, 0.7],
        }
    )
    bootstrap = pd.DataFrame(
        {
            "bootstrap_replicate": [0, 0, 0, 1, 1, 1],
            "feature": ["Pi"] * 6,
            "target": ["REF_F1score"] * 6,
            "model": [
                "ImputeMostFrequent",
                "ImputeAutoencoder",
                "ImputeVAE",
                "ImputeMostFrequent",
                "ImputeAutoencoder",
                "ImputeVAE",
            ],
            "std_coef": [0.1, 0.3, 0.5, 0.2, 0.6, 0.8],
        }
    )

    summary = metric._deep_minus_baseline_summary(
        observed,
        target_filter=["REF_F1score"],
        n_boot=999,
        ci=100.0,
        random_seed=42,
        bootstrap_effects=bootstrap,
        interval_mode="dataset_refit",
    )

    row = summary.loc[summary["feature"].eq("Pi")].iloc[0]
    assert row["delta_deep_minus_mostfrequent_abs_bstd"] == pytest.approx(0.4)
    assert row["ci_low"] == pytest.approx(0.3)
    assert row["ci_high"] == pytest.approx(0.5)
    assert row["ci_source"] == "dataset_refit_join_key"
    assert int(row["n_bootstrap_replicates"]) == 2


def test_driver_ranking_summary_uses_dataset_refit_replicates() -> None:
    observed = pd.DataFrame(
        {
            "feature": ["Pi", "Pi"],
            "target": ["REF_F1score", "HET_F1score"],
            "std_coef": [0.2, 0.6],
        }
    )
    bootstrap = pd.DataFrame(
        {
            "bootstrap_replicate": [0, 0, 1, 1],
            "feature": ["Pi", "Pi", "Pi", "Pi"],
            "target": ["REF_F1score", "HET_F1score"] * 2,
            "std_coef": [0.1, 0.3, 0.5, 0.7],
        }
    )

    summary = metric._driver_ranking_summary(
        observed,
        "std_coef",
        n_boot=999,
        ci=100.0,
        random_seed=42,
        bootstrap_effects=bootstrap,
        interval_mode="dataset_refit",
    )

    row = summary.loc[summary["feature"].eq("Pi")].iloc[0]
    assert row["mean_abs_effect"] == pytest.approx(0.4)
    assert row["ci_low"] == pytest.approx(0.2)
    assert row["ci_high"] == pytest.approx(0.6)
    assert row["ci_source"] == "dataset_refit_join_key"
    assert int(row["n_bootstrap_replicates"]) == 2
