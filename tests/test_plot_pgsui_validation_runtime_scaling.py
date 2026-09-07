from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_pgsui_validation_runtime_scaling.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plot_pgsui_validation_runtime_scaling", SCRIPT_PATH
)
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
assert SPEC.loader is not None
SPEC.loader.exec_module(runtime)


def test_parse_phylip_header_reads_sample_and_locus_counts(tmp_path: Path) -> None:
    phylip = tmp_path / "dataset.phy"
    phylip.write_text("12 345\nsample_1 ACGT\n")

    assert runtime.parse_phylip_header(phylip) == (12, 345)


def test_parse_samples_loci_text_handles_commas() -> None:
    assert runtime.parse_samples_loci_text("1,234 samples; 56,789 SNPs/loci") == (
        1234,
        56789,
    )


def test_parse_duration_text_handles_day_prefix() -> None:
    assert runtime.parse_duration_text("1 day, 5:09:55.319500") == pytest.approx(
        104995.3195
    )
    assert runtime.parse_duration_text("12.75") == pytest.approx(12.75)


def test_parse_deep_runtime_segments_selects_longest_complete_segment(
    tmp_path: Path,
) -> None:
    log = tmp_path / "pgsui.impute.unsupervised.base.log"
    log.write_text(
        "\n".join(
            [
                "2026-02-01 00:00:00 - INFO - x - fit - Fitting ImputeVAE model...",
                "2026-02-01 00:05:00 - INFO - x - tune_hyperparameters - Tuning completed in: 0:05:00 (HH:MM:SS) for 100 trials.",
                "2026-02-01 00:05:00 - INFO - x - tune_hyperparameters - Average trial duration: 0:00:03",
                "2026-02-01 00:10:00 - INFO - x - transform - ImputeVAE Imputation complete!",
                "2026-02-02 00:00:00 - INFO - x - fit - Fitting ImputeVAE model...",
                "2026-02-02 00:01:00 - INFO - x - transform - ImputeVAE Imputation complete!",
            ]
        )
        + "\n"
    )

    selected = runtime.select_deep_runtime(log, "ImputeVAE")

    assert selected["runtime_seconds"] == pytest.approx(600.0)
    assert selected["runtime_segment_count"] == 2
    assert selected["tuning_wall_seconds"] == pytest.approx(300.0)
    assert selected["mean_trial_seconds"] == pytest.approx(3.0)
    assert selected["per_trial_runtime_seconds"] == pytest.approx(3.0)
    assert selected["trial_count"] == pytest.approx(100.0)


def test_parse_deep_runtime_segments_computes_per_trial_from_tuning_wall(
    tmp_path: Path,
) -> None:
    log = tmp_path / "pgsui.impute.unsupervised.base.log"
    log.write_text(
        "\n".join(
            [
                "2026-02-01 00:00:00 - INFO - x - fit - Fitting ImputeVAE model...",
                "2026-02-01 00:05:00 - INFO - x - tune_hyperparameters - Tuning completed in: 0:05:00 (HH:MM:SS) for 100 trials.",
                "2026-02-01 00:10:00 - INFO - x - transform - ImputeVAE Imputation complete!",
            ]
        )
        + "\n"
    )

    selected = runtime.select_deep_runtime(log, "ImputeVAE")

    assert selected["runtime_seconds"] == pytest.approx(600.0)
    assert selected["tuning_wall_seconds"] == pytest.approx(300.0)
    assert pd.isna(selected["mean_trial_seconds"])
    assert selected["per_trial_runtime_seconds"] == pytest.approx(3.0)
    assert selected["trial_count"] == pytest.approx(100.0)


def test_parse_deep_runtime_segments_falls_back_to_fit_complete(
    tmp_path: Path,
) -> None:
    log = tmp_path / "pgsui.impute.unsupervised.base.log"
    log.write_text(
        "\n".join(
            [
                "2026-02-01 00:00:00 - INFO - x - fit - Fitting ImputeUBP model...",
                "2026-02-01 00:30:00 - INFO - x - fit - ImputeUBP fitting complete!",
                "2026-02-01 00:31:00 - INFO - x - fit - Fitting ImputeVAE model...",
            ]
        )
        + "\n"
    )

    selected = runtime.select_deep_runtime(log, "ImputeUBP")

    assert selected["runtime_seconds"] == pytest.approx(1800.0)
    assert (
        selected["runtime_source"]
        == "base_log_longest_fit_complete_segment_missing_transform_marker"
    )


def test_regression_summary_reports_effect_per_million_genotypes() -> None:
    summary = pd.DataFrame(
        {
            "dataset_id": ["ds1", "ds2"],
            "dataset_folder": ["ds1", "ds2"],
            "model": ["ImputeUBP", "ImputeUBP"],
            "model_label": ["UBP", "UBP"],
            "sample_size": [10, 20],
            "locus_count": [1000, 1000],
            "total_genotypes": [10_000, 20_000],
            "n_strategy_runs": [5, 5],
            "runtime_seconds_for_regression": [10.0, 30.0],
        }
    )

    regressions = runtime.regression_summary(summary)
    ubp = regressions.loc[regressions["model"].eq("ImputeUBP")].iloc[0]

    assert int(ubp["n_datasets"]) == 2
    assert ubp["slope_seconds_per_genotype"] == pytest.approx(0.002)
    assert ubp["slope_seconds_per_million_genotypes"] == pytest.approx(2000.0)
    assert ubp["slope_hours_per_million_genotypes"] == pytest.approx(2000.0 / 3600.0)
    assert ubp["r_squared_linear"] == pytest.approx(1.0)


def test_summarize_dataset_model_runtimes_uses_deep_per_trial_metric() -> None:
    runtime_df = pd.DataFrame(
        {
            "dataset_id": ["ds1", "ds1", "ds1", "ds1"],
            "dataset_folder": ["ds1", "ds1", "ds1", "ds1"],
            "strategy": ["random", "nonrandom", "random", "nonrandom"],
            "model": [
                "ImputeVAE",
                "ImputeVAE",
                "ImputeRefAllele",
                "ImputeRefAllele",
            ],
            "model_label": ["VAE", "VAE", "RefAllele", "RefAllele"],
            "sample_size": [10, 10, 10, 10],
            "locus_count": [1000, 1000, 1000, 1000],
            "total_genotypes": [10_000, 10_000, 10_000, 10_000],
            "report_complete": [True, True, True, True],
            "runtime_seconds": [300.0, 500.0, 2.0, 4.0],
            "tuning_wall_seconds": [300.0, 500.0, float("nan"), float("nan")],
            "mean_trial_seconds": [3.0, 5.0, float("nan"), float("nan")],
            "per_trial_runtime_seconds": [3.0, 5.0, float("nan"), float("nan")],
            "trial_count": [100.0, 100.0, float("nan"), float("nan")],
        }
    )

    summary = runtime.summarize_dataset_model_runtimes(
        runtime_df,
        strategy_aggregation="mean",
    )

    vae = summary.loc[summary["model"].eq("ImputeVAE")].iloc[0]
    ref = summary.loc[summary["model"].eq("ImputeRefAllele")].iloc[0]

    assert vae["mean_runtime_seconds"] == pytest.approx(400.0)
    assert vae["mean_per_trial_runtime_seconds"] == pytest.approx(4.0)
    assert vae["runtime_seconds_for_regression"] == pytest.approx(4.0)
    assert vae["runtime_metric_sources"] == "deep_log_per_trial_seconds"
    assert ref["runtime_seconds_for_regression"] == pytest.approx(3.0)
    assert ref["runtime_metric_sources"] == "deterministic_log_runtime_seconds"
