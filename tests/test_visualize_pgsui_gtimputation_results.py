from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "visualize_pgsui_gtimputation_results.py"
)
SPEC = importlib.util.spec_from_file_location(
    "visualize_pgsui_gtimputation_results", SCRIPT_PATH
)
viz = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = viz
assert SPEC.loader is not None
SPEC.loader.exec_module(viz)


def write_vcf(path: Path, rows: list[tuple[str, str, str, str, str, str]]) -> None:
    lines = [
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2",
    ]
    for chrom, pos, ref, alt, sample_1, sample_2 in rows:
        lines.append(
            f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\tGT\t{sample_1}\t{sample_2}"
        )
    path.write_text("\n".join(lines) + "\n")


def test_runtime_panel_subset_keeps_gtimputation_cpu_only() -> None:
    runtime_df = pd.DataFrame(
        {
            "runtime_backend_label": ["CPU", "GPU", "Deterministic", "GTImputation"],
            "model_label": ["VAE", "VAE", "RefAllele", "SOM (GTImputation)"],
            "runtime_seconds": [10.0, 8.0, 1.0, 2.0],
        }
    )

    cpu_panel = viz.runtime_panel_subset(runtime_df, "CPU")
    gpu_panel = viz.runtime_panel_subset(runtime_df, "GPU")

    assert set(cpu_panel["runtime_backend_label"]) == {
        "CPU",
        "Deterministic",
        "GTImputation",
    }
    assert set(gpu_panel["runtime_backend_label"]) == {"GPU", "Deterministic"}


def test_cpu_gpu_runtime_delta_pairs_deep_models_only() -> None:
    runtime_df = pd.DataFrame(
        {
            "dataset_id": ["ds1", "ds1", "ds1", "ds1", "ds1"],
            "strategy": ["random", "random", "random", "random", "random"],
            "software": ["PG-SUI", "PG-SUI", "PG-SUI", "PG-SUI", "GTImputation"],
            "model": [
                "ImputeAutoencoder",
                "ImputeAutoencoder",
                "ImputeRefAllele",
                "ImputeRefAllele",
                "som",
            ],
            "model_label": [
                "Autoencoder",
                "Autoencoder",
                "RefAllele",
                "RefAllele",
                "SOM (GTImputation)",
            ],
            "runtime_backend": [
                "cpu",
                "cuda",
                "deterministic",
                "deterministic",
                "not_applicable",
            ],
            "runtime_seconds": [90.0, 30.0, 5.0, 5.0, 12.0],
        }
    )

    paired = viz.cpu_gpu_runtime_delta_df(runtime_df)

    assert paired[["model_label", "cpu", "cuda"]].to_dict("records") == [
        {"model_label": "Autoencoder", "cpu": 90.0, "cuda": 30.0}
    ]
    assert paired["cpu_minus_gpu_seconds"].iloc[0] == pytest.approx(60.0)
    assert paired["cpu_gpu_speedup"].iloc[0] == pytest.approx(3.0)


def test_score_gti_run_scores_imputed_vcf_against_nonmissing_truth(
    tmp_path: Path,
) -> None:
    truth_vcf = tmp_path / "truth.vcf"
    masked_vcf = tmp_path / "masked.vcf"
    imputed_vcf = tmp_path / "imputed.vcf"
    mask_tsv = tmp_path / "mask.tsv"

    write_vcf(
        truth_vcf,
        [
            ("chr1", "1", "A", "C", "0/0", "0/1"),
            ("chr1", "2", "A", "C", "1/1", "0/0"),
        ],
    )
    write_vcf(
        masked_vcf,
        [
            ("chr1", "1", "A", "C", "./.", "./."),
            ("chr1", "2", "A", "C", "./.", "0/0"),
        ],
    )
    write_vcf(
        imputed_vcf,
        [
            ("chr1", "1", "A", "C", "0/0", "0/0"),
            ("chr1", "2", "A", "C", "1/1", "0/0"),
        ],
    )
    mask_tsv.write_text(
        "\n".join(
            [
                "sample_id\tlocus_index\tchrom\tpos\tref\talt",
                "S1\t0\tchr1\t1\tA\tC",
                "S2\t0\tchr1\t1\tA\tC",
                "S1\t1\tchr1\t2\tA\tC",
                "S2\t1\tchr1\t2\tA\tC",
            ]
        )
        + "\n"
    )

    gti_row = pd.Series(
        {
            "dataset_id": "ds1",
            "strategy": "random",
            "method": "naive",
            "run_id": "run-1",
            "copied_vcf": str(imputed_vcf),
            "candidate_vcf_filename": imputed_vcf.name,
            "imputation_phase_real_seconds": 3.0,
        }
    )
    sim_row = pd.Series(
        {
            "dataset_id": "ds1",
            "strategy": "random",
            "input_vcf": str(truth_vcf),
            "output_vcf": str(masked_vcf),
            "mask_tsv": str(mask_tsv),
        }
    )

    metrics, class_rows = viz.score_gti_run(
        gti_row,
        sim_row,
        gti_dir=tmp_path,
        sim_manifest_path=tmp_path / "manifest.csv",
        truth_cache={},
        masked_cache={},
    )

    assert metrics["support"] == pytest.approx(3.0)
    assert metrics["mask_rows"] == 4
    assert metrics["scored_masked_sites"] == 3
    assert metrics["dropped_unmasked_mask_rows"] == 1
    assert metrics["accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["macro_f1"] == pytest.approx((2.0 / 3.0 + 0.0 + 1.0) / 3.0)

    class_by_label = {row["class_label"]: row for row in class_rows}
    assert class_by_label["REF"]["f1"] == pytest.approx(2.0 / 3.0)
    assert class_by_label["HET"]["f1"] == pytest.approx(0.0)
    assert class_by_label["ALT"]["f1"] == pytest.approx(1.0)


def test_validate_nonmissing_truth_rejects_missing_truth() -> None:
    with pytest.raises(ValueError, match="truth VCF has 1 missing genotypes"):
        viz.validate_nonmissing_truth(np.array([0, -1, 2]), "test context")


def test_canonical_mask_rejects_unmasked_evaluation_coordinate(
    tmp_path: Path,
) -> None:
    masked_vcf_path = tmp_path / "masked.vcf"
    write_vcf(
        masked_vcf_path,
        [("chr1", "1", "A", "C", "./.", "0/1")],
    )
    mask = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "locus_index": [0, 0],
            "chrom": ["chr1", "chr1"],
            "pos": ["1", "1"],
            "ref": ["A", "A"],
            "alt": ["C", "C"],
        }
    )
    sim_row = pd.Series(
        {
            "dataset_id": "ds1",
            "strategy": "random",
            "masked_vcf": str(masked_vcf_path),
            "evaluation_mask_tsv": str(tmp_path / "evaluation.tsv"),
        }
    )

    with pytest.raises(ValueError, match="1 unmasked evaluation coordinates"):
        viz.mask_actual_missing_sites(
            mask,
            sim_row,
            sim_manifest_path=tmp_path / "manifest.csv",
            masked_cache={},
        )


def test_scoring_vectors_use_pgsui_class_mapping(tmp_path: Path) -> None:
    truth_path = tmp_path / "truth.vcf"
    prediction_path = tmp_path / "prediction.vcf"
    write_vcf(
        truth_path,
        [
            ("chr1", "1", "A", "C", "0/0", "0/1"),
            ("chr1", "2", "A", "C", "1/1", "0/0"),
        ],
    )
    write_vcf(
        prediction_path,
        [
            ("chr1", "1", "A", "C", "1/1", "0/1"),
            ("chr1", "2", "A", "C", "0/0", "1/1"),
        ],
    )
    mask = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S1", "S2"],
            "locus_index": [0, 0, 1, 1],
            "chrom": ["chr1"] * 4,
            "pos": ["1", "1", "2", "2"],
            "ref": ["A"] * 4,
            "alt": ["C"] * 4,
            "pgsui_truth_012": [2, 1, 0, 2],
            "pgsui_class_for_vcf_ref": [2] * 4,
            "pgsui_class_for_vcf_het": [1] * 4,
            "pgsui_class_for_vcf_alt": [0] * 4,
        }
    )

    y_true, y_pred, encoding = viz.scoring_vectors_from_mask(
        mask,
        viz.load_vcf_matrix(truth_path),
        viz.load_vcf_matrix(prediction_path),
    )

    assert encoding == "pgsui_012"
    assert y_true.tolist() == [2, 1, 0, 2]
    assert y_pred.tolist() == [0, 1, 2, 0]


def test_canonical_support_audit_requires_class_match() -> None:
    pgsui = pd.DataFrame(
        {
            "dataset_id": ["ds1"],
            "strategy": ["random"],
            "model": ["ImputeUBP"],
            "model_label": ["UBP"],
            "support": [10],
            "ref_support": [5],
            "het_support": [3],
            "alt_support": [2],
        }
    )
    gti = pd.DataFrame(
        {
            "dataset_id": ["ds1"],
            "strategy": ["random"],
            "model_label": ["SOM (GTImputation)"],
            "class_encoding": ["pgsui_012"],
            "support": [10],
            "ref_support": [5],
            "het_support": [3],
            "alt_support": [2],
        }
    )

    audit = viz.build_evaluation_support_audit(pgsui, gti)
    assert bool(audit["support_match"].all())

    gti.loc[0, "alt_support"] = 3
    with pytest.raises(ValueError, match="class supports differ"):
        viz.build_evaluation_support_audit(pgsui, gti)


def _toy_combined() -> pd.DataFrame:
    """Two datasets x one strategy, covering all four model families."""
    rows = [
        # dataset, software, model_label, macro_f1, mcc, ref, het, alt, runtime
        (
            "ds1",
            "GTImputation",
            "Naive (GTImputation)",
            0.50,
            0.30,
            0.70,
            0.40,
            0.40,
            0.2,
        ),
        (
            "ds1",
            "GTImputation",
            "SOM (GTImputation)",
            0.55,
            0.35,
            0.72,
            0.45,
            0.48,
            100.0,
        ),
        ("ds1", "PG-SUI", "RefAllele", 0.20, 0.00, 0.60, 0.00, 0.00, 1.0),
        ("ds1", "PG-SUI", "MostFrequent", 0.58, 0.40, 0.74, 0.50, 0.50, 1.0),
        ("ds1", "PG-SUI", "UBP", 0.70, 0.55, 0.78, 0.63, 0.69, 15.0),
        (
            "ds2",
            "GTImputation",
            "Naive (GTImputation)",
            0.60,
            0.40,
            0.72,
            0.54,
            0.54,
            0.3,
        ),
        (
            "ds2",
            "GTImputation",
            "SOM (GTImputation)",
            0.62,
            0.42,
            0.73,
            0.56,
            0.57,
            110.0,
        ),
        ("ds2", "PG-SUI", "RefAllele", 0.22, 0.00, 0.62, 0.00, 0.00, 1.0),
        ("ds2", "PG-SUI", "MostFrequent", 0.64, 0.45, 0.76, 0.55, 0.61, 1.0),
        ("ds2", "PG-SUI", "UBP", 0.68, 0.53, 0.77, 0.62, 0.65, 14.0),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "dataset_id",
            "software",
            "model_label",
            "macro_f1",
            "mcc",
            "ref_f1",
            "het_f1",
            "alt_f1",
            "runtime_seconds",
        ],
    ).assign(strategy="random")


def _toy_sim_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": ["ds1", "ds2"],
            "strategy": ["random", "random"],
            "n_samples": [25, 100],
            "n_loci": [500, 2500],
            "n_cells": [12500, 250000],
            "simulated_missing_rate": [0.3, 0.3],
        }
    )


def _toy_runtime_combined() -> pd.DataFrame:
    rows = [
        (
            "ds1",
            "GTImputation",
            "naive",
            "Naive (GTImputation)",
            "not_applicable",
            "GTImputation",
            0.2,
        ),
        (
            "ds1",
            "GTImputation",
            "som",
            "SOM (GTImputation)",
            "not_applicable",
            "GTImputation",
            100.0,
        ),
        (
            "ds1",
            "PG-SUI",
            "ImputeMostFrequent",
            "MostFrequent",
            "deterministic",
            "Deterministic",
            1.0,
        ),
        ("ds1", "PG-SUI", "ImputeUBP", "UBP", "cpu", "CPU", 20.0),
        ("ds1", "PG-SUI", "ImputeUBP", "UBP", "cuda", "GPU", 5.0),
        (
            "ds2",
            "GTImputation",
            "naive",
            "Naive (GTImputation)",
            "not_applicable",
            "GTImputation",
            0.3,
        ),
        (
            "ds2",
            "GTImputation",
            "som",
            "SOM (GTImputation)",
            "not_applicable",
            "GTImputation",
            110.0,
        ),
        (
            "ds2",
            "PG-SUI",
            "ImputeMostFrequent",
            "MostFrequent",
            "deterministic",
            "Deterministic",
            1.1,
        ),
        ("ds2", "PG-SUI", "ImputeUBP", "UBP", "cpu", "CPU", 21.0),
        ("ds2", "PG-SUI", "ImputeUBP", "UBP", "cuda", "GPU", 5.5),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "dataset_id",
            "software",
            "model",
            "model_label",
            "runtime_backend",
            "runtime_backend_label",
            "runtime_seconds",
        ],
    ).assign(strategy="random")


def test_format_decimal_ticks_preserves_ylim() -> None:
    # Regression: the old set_yticks(get_yticks()) idiom expanded (-0.02, 1.02)
    # out to (-0.2, 1.2). The formatter-based helper must leave limits intact.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [0.2, 0.6, 0.7])
    ax.set_ylim(-0.02, 1.02)
    viz.format_decimal_ticks(ax, axis="y", decimals=2)
    lower, upper = ax.get_ylim()
    plt.close(fig)
    assert lower == pytest.approx(-0.02)
    assert upper == pytest.approx(1.02)


def test_build_category_summary_table_collapses_four_families() -> None:
    summary = viz.build_category_summary_table(_toy_combined())
    assert list(summary["model_category"]) == list(viz.CATEGORY_ORDER)
    det = summary.loc[summary["model_category"].eq("PG-SUI: Deterministic")].iloc[0]
    # Deterministic family = RefAllele + MostFrequent across two datasets.
    assert int(det["n_runs"]) == 4
    assert det["mean_macro_f1"] == pytest.approx((0.20 + 0.58 + 0.22 + 0.64) / 4.0)


def test_build_head_to_head_table_reports_win_rate_and_delta() -> None:
    table = viz.build_head_to_head_table(_toy_combined())
    ubp_naive = table.loc[
        table["pgsui_model"].eq("UBP") & table["gti_method"].eq("Naive (GTImputation)")
    ].iloc[0]
    assert int(ubp_naive["n_pairs"]) == 2
    assert int(ubp_naive["n_datasets"]) == 2
    assert int(ubp_naive["n_dataset_strategy_pairs"]) == 2
    assert ubp_naive["pgsui_win_rate"] == pytest.approx(1.0)
    assert ubp_naive["dataset_win_rate"] == pytest.approx(1.0)
    assert ubp_naive["dataset_strategy_win_rate"] == pytest.approx(1.0)
    assert ubp_naive["mean_delta_macro_f1"] == pytest.approx(
        ((0.70 - 0.50) + (0.68 - 0.60)) / 2.0
    )
    assert ubp_naive["p_exact_block_sign_flip"] == pytest.approx(0.5)
    assert "wilcoxon_p" not in table.columns

    ref_naive = table.loc[
        table["pgsui_model"].eq("RefAllele")
        & table["gti_method"].eq("Naive (GTImputation)")
    ].iloc[0]
    assert ref_naive["dataset_win_rate"] == pytest.approx(0.0)


def test_build_head_to_head_table_averages_strategy_deltas_by_dataset() -> None:
    combined = pd.DataFrame(
        [
            ("ds1", "random", "PG-SUI", "UBP", 0.70),
            ("ds1", "random", "GTImputation", "Naive (GTImputation)", 0.50),
            ("ds1", "nonrandom", "PG-SUI", "UBP", 0.40),
            ("ds1", "nonrandom", "GTImputation", "Naive (GTImputation)", 0.50),
            ("ds2", "random", "PG-SUI", "UBP", 0.70),
            ("ds2", "random", "GTImputation", "Naive (GTImputation)", 0.60),
            ("ds2", "nonrandom", "PG-SUI", "UBP", 0.70),
            ("ds2", "nonrandom", "GTImputation", "Naive (GTImputation)", 0.60),
        ],
        columns=["dataset_id", "strategy", "software", "model_label", "macro_f1"],
    )

    row = viz.build_head_to_head_table(combined).iloc[0]

    assert int(row["n_datasets"]) == 2
    assert int(row["n_dataset_strategy_pairs"]) == 4
    assert row["dataset_strategy_win_rate"] == pytest.approx(3.0 / 4.0)
    assert row["dataset_win_rate"] == pytest.approx(1.0)
    assert row["dataset_win_count"] == 2
    assert row["mean_delta_macro_f1"] == pytest.approx(
        (((0.70 - 0.50) + (0.40 - 0.50)) / 2.0 + (0.70 - 0.60)) / 2.0
    )
    assert row["p_exact_block_sign_flip"] == pytest.approx(0.5)
    assert row["p_holm"] == pytest.approx(0.5)


def test_exact_sign_flip_test_enumerates_all_signs() -> None:
    assert viz.exact_sign_flip_test([1.0, 2.0]) == pytest.approx(0.5)


def test_build_strategy_sensitivity_table_fits_mixed_model() -> None:
    rows = []
    for dataset_id, dataset_shift in (
        ("ds1", 0.00),
        ("ds2", 0.02),
        ("ds3", -0.01),
        ("ds4", 0.01),
    ):
        for strategy, strategy_effect in (
            ("random", 0.05),
            ("random_weighted", 0.02),
            ("nonrandom", 0.10),
        ):
            gti_f1 = 0.60 + dataset_shift
            pgsui_f1 = gti_f1 + strategy_effect + dataset_shift * 0.1
            rows.append((dataset_id, strategy, "PG-SUI", "UBP", pgsui_f1))
            rows.append(
                (
                    dataset_id,
                    strategy,
                    "GTImputation",
                    "Naive (GTImputation)",
                    gti_f1,
                )
            )

    combined = pd.DataFrame(
        rows,
        columns=["dataset_id", "strategy", "software", "model_label", "macro_f1"],
    )

    table = viz.build_strategy_sensitivity_table(combined)
    row = table.iloc[0]

    assert row["fit_status"] == "ok"
    assert int(row["n_datasets"]) == 4
    assert int(row["n_strategies"]) == 3
    assert int(row["n_dataset_strategy_pairs"]) == 12
    assert row["formula"] == "delta_macro_f1 ~ C(strategy, Sum) + (1 | dataset_id)"
    assert 0.0 <= row["strategy_wald_p"] <= 1.0
    assert "C(strategy, Sum)" in row["strategy_coefficients"]


def test_build_dataset_leaderboard_table_sorts_by_pgsui_advantage() -> None:
    table = viz.build_dataset_leaderboard_table(_toy_combined())
    assert list(table["dataset_id"]) == ["ds1", "ds2"]
    ds1 = table.loc[table["dataset_id"].eq("ds1")].iloc[0]
    assert ds1["best_pgsui_model"] == "UBP"
    assert ds1["best_gti_model"] == "SOM (GTImputation)"
    assert ds1["pgsui_minus_gti"] == pytest.approx(0.70 - 0.55)


def test_build_class_f1_wide_table_computes_spread() -> None:
    table = viz.build_class_f1_wide_table(_toy_combined())
    ref_allele = table.loc[table["model_label"].eq("RefAllele")].iloc[0]
    assert ref_allele["mean_het_f1"] == pytest.approx(0.0)
    assert ref_allele["worst_class_f1"] == pytest.approx(0.0)
    # REF mean = (0.60 + 0.62)/2 = 0.61; HET/ALT means = 0 -> spread == REF mean.
    assert ref_allele["class_f1_spread"] == pytest.approx(0.61)


def test_dataset_size_performance_summary_uses_manifest_metadata() -> None:
    combined = viz.attach_simulation_size_metadata(_toy_combined(), _toy_sim_manifest())
    assert {"n_samples", "n_loci", "n_cells"}.issubset(combined.columns)

    summary = viz.dataset_size_performance_summary_df(combined)

    det = summary.loc[
        summary["dataset_id"].eq("ds1")
        & summary["model_category"].eq("PG-SUI: Deterministic")
    ].iloc[0]
    assert int(det["n_runs"]) == 2
    assert int(det["n_samples"]) == 25
    assert int(det["n_loci"]) == 500
    assert det["mean_macro_f1"] == pytest.approx((0.20 + 0.58) / 2.0)


def test_plot_performance_by_dataset_size_writes_outputs(tmp_path: Path) -> None:
    combined = viz.attach_simulation_size_metadata(_toy_combined(), _toy_sim_manifest())

    viz.configure_plot_style(100)
    viz.plot_performance_by_dataset_size(combined, tmp_path)

    assert (tmp_path / "performance_by_dataset_size.png").exists()
    assert (tmp_path / "performance_by_dataset_size.pdf").exists()


def test_f1_size_regression_summary_fits_sample_and_locus_counts() -> None:
    combined = viz.attach_simulation_size_metadata(_toy_combined(), _toy_sim_manifest())

    regression = viz.f1_size_regression_summary_df(combined)

    assert set(regression["predictor"]) == {"n_samples", "n_loci"}
    det_samples = regression.loc[
        regression["predictor"].eq("n_samples")
        & regression["model_category"].eq("PG-SUI: Deterministic")
    ].iloc[0]
    assert int(det_samples["n_datasets"]) == 2
    assert det_samples["slope"] > 0
    assert det_samples["r_squared"] == pytest.approx(1.0)


def test_plot_f1_size_regressions_writes_outputs(tmp_path: Path) -> None:
    combined = viz.attach_simulation_size_metadata(_toy_combined(), _toy_sim_manifest())

    viz.configure_plot_style(100)
    viz.plot_f1_size_regressions(combined, tmp_path)

    assert (tmp_path / "f1_size_regression.png").exists()
    assert (tmp_path / "f1_size_regression.pdf").exists()


def test_f1_genotype_count_regression_summary_fits_all_models() -> None:
    combined = viz.attach_simulation_size_metadata(_toy_combined(), _toy_sim_manifest())

    regression = viz.f1_genotype_count_regression_summary_df(combined)

    assert set(regression["model_label"]) == set(combined["model_label"])
    naive = regression.loc[regression["model_label"].eq("Naive (GTImputation)")].iloc[0]
    assert int(naive["n_datasets"]) == 2
    assert int(naive["n_runs"]) == 2
    assert naive["n_genotypes_min"] == pytest.approx(25 * 500)
    assert naive["n_genotypes_max"] == pytest.approx(100 * 2500)
    assert naive["slope"] > 0
    assert naive["r_squared"] == pytest.approx(1.0)


def test_plot_f1_genotype_count_regressions_writes_outputs(tmp_path: Path) -> None:
    combined = viz.attach_simulation_size_metadata(_toy_combined(), _toy_sim_manifest())

    viz.configure_plot_style(100)
    viz.plot_f1_genotype_count_regressions(combined, tmp_path)

    assert (tmp_path / "f1_genotype_count_regression.png").exists()
    assert (tmp_path / "f1_genotype_count_regression.pdf").exists()


def test_runtime_genotype_count_regression_summary_separates_backends() -> None:
    runtime = viz.attach_simulation_size_metadata(
        _toy_runtime_combined(), _toy_sim_manifest()
    )

    regression = viz.runtime_genotype_count_regression_summary_df(runtime)

    assert {"UBP (CPU)", "UBP (GPU)"}.issubset(set(regression["runtime_model_label"]))
    som = regression.loc[
        regression["runtime_model_label"].eq("SOM (GTImputation)")
    ].iloc[0]
    ubp_cpu = regression.loc[regression["runtime_model_label"].eq("UBP (CPU)")].iloc[0]
    assert int(som["n_datasets"]) == 2
    assert int(som["n_runs"]) == 2
    assert som["n_genotypes_min"] == pytest.approx(25 * 500)
    assert som["n_genotypes_max"] == pytest.approx(100 * 2500)
    assert som["slope_seconds_per_genotype"] > 0
    assert ubp_cpu["runtime_backend_label"] == "CPU"
    assert ubp_cpu["slope_seconds_per_genotype"] > 0


def test_plot_runtime_genotype_count_regressions_writes_outputs(tmp_path: Path) -> None:
    runtime = viz.attach_simulation_size_metadata(
        _toy_runtime_combined(), _toy_sim_manifest()
    )

    viz.configure_plot_style(100)
    viz.plot_runtime_genotype_count_regressions(runtime, tmp_path)

    assert (tmp_path / "runtime_genotype_count_regression.png").exists()
    assert (tmp_path / "runtime_genotype_count_regression.pdf").exists()


def test_dataset_size_f1_runtime_comparison_summarizes_both_metrics() -> None:
    combined = viz.attach_simulation_size_metadata(_toy_combined(), _toy_sim_manifest())
    runtime = viz.attach_simulation_size_metadata(
        _toy_runtime_combined(), _toy_sim_manifest()
    )

    summary = viz.dataset_size_f1_runtime_comparison_df(combined, runtime)

    f1_det = summary.loc[
        summary["metric"].astype(str).eq("Macro F1")
        & summary["dataset_id"].eq("ds1")
        & summary["comparison_group"].eq("PG-SUI Det.")
    ].iloc[0]
    assert f1_det["plot_statistic"] == "mean"
    assert f1_det["plot_value"] == pytest.approx((0.20 + 0.58) / 2.0)

    runtime_gpu = summary.loc[
        summary["metric"].astype(str).eq("Runtime seconds")
        & summary["dataset_id"].eq("ds1")
        & summary["comparison_group"].eq("PG-SUI DL (GPU)")
    ].iloc[0]
    assert runtime_gpu["plot_statistic"] == "median"
    assert runtime_gpu["plot_value"] == pytest.approx(5.0)


def test_plot_dataset_size_f1_runtime_comparison_writes_outputs(tmp_path: Path) -> None:
    combined = viz.attach_simulation_size_metadata(_toy_combined(), _toy_sim_manifest())
    runtime = viz.attach_simulation_size_metadata(
        _toy_runtime_combined(), _toy_sim_manifest()
    )

    viz.configure_plot_style(100)
    viz.plot_dataset_size_f1_runtime_comparison(combined, runtime, tmp_path)

    assert (tmp_path / "dataset_size_f1_runtime_comparison.png").exists()
    assert (tmp_path / "dataset_size_f1_runtime_comparison.pdf").exists()


def test_write_table_keeps_csv_and_writes_formatted_views(tmp_path: Path) -> None:
    table_df = pd.DataFrame(
        {
            "model_label": ["Naive (GTImputation)", "SOM (GTImputation)"],
            "macro_f1": [0.123456, 0.987654],
            "support": [10, 12],
        }
    )
    csv_path = tmp_path / "tables" / "best_model_summary.csv"

    viz.write_table(table_df, csv_path)
    viz.write_table_view_index(csv_path.parent)

    html_path = csv_path.parent / "formatted" / "html" / "best_model_summary.html"
    latex_path = csv_path.parent / "formatted" / "latex" / "best_model_summary.tex"
    rtf_path = csv_path.parent / "formatted" / "rtf" / "best_model_summary.rtf"
    index_path = csv_path.parent / "formatted" / "index.html"

    assert csv_path.exists()
    assert html_path.exists()
    assert latex_path.exists()
    assert rtf_path.exists()
    assert index_path.exists()
    assert "Naive (GTImputation)" in html_path.read_text()
    assert "best_model_summary.csv" in index_path.read_text()
