"""Scientific and edge-case tests for the SNPio-backed summary workflow."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from snpiosp.compare_dataset_stats import (
    ALL_STRATEGIES,
    EXPECTED_MODELS,
    DatasetFiles,
    load_completed_dataset_labels,
    read_dataset_summaries,
)
from snpiosp.run_snpiosp import infer_format, json_ready
from snpiosp.snpiosp import (
    DnaSPSingleLocusAnalyzer,
    clean_012_matrix,
    infer_vcf_ploidy,
    per_locus_statistics,
    polyploid_locus_statistics,
    read_biallelic_vcf_alt_dosage,
    snpio_ld_overall,
    tajimas_d_complete_sites,
)


def test_clean_012_matrix_handles_only_snpio_missing_sentinel() -> None:
    observed = clean_012_matrix(np.array([[0, 1, 2, -9]]))
    np.testing.assert_array_equal(observed[:, :3], np.array([[0.0, 1.0, 2.0]]))
    assert np.isnan(observed[0, 3])
    with pytest.raises(ValueError, match="Unexpected SNPio 012 values"):
        clean_012_matrix(np.array([[0, 3]]))


def test_per_locus_statistics_handles_all_missing_and_monomorphic_loci() -> None:
    matrix = np.array(
        [
            [0.0, 0.0, np.nan, 1.0],
            [1.0, 0.0, np.nan, 1.0],
            [2.0, 0.0, np.nan, 1.0],
        ]
    )
    snpio_statistics = pd.DataFrame(
        {
            "Ho": [1 / 3, 0.0, np.nan, 1.0],
            "He": [0.5, 0.0, np.nan, 0.5],
            "Pi": [0.75, 0.0, np.nan, 0.75],
        }
    )
    observed = per_locus_statistics(matrix, snpio_statistics)

    assert observed["Sample_Size"].tolist() == [3, 3, 0, 3]
    assert observed["SegSites"].tolist() == [1, 0, 0, 1]
    assert observed["Singletons"].tolist() == [0, 0, 0, 0]
    assert observed.loc[2, "Missingness"] == 1.0
    assert np.isnan(observed.loc[2, "MAF"])
    assert np.isnan(observed.loc[2, "Genotype_Diversity"])
    assert observed.loc[0, "Genotype_Category_Count"] == 3
    assert observed.loc[0, "Genotype_Diversity"] == pytest.approx(1.0)
    assert observed.loc[3, "Genotype_Diversity"] == pytest.approx(0.0)
    assert observed["TajimaD_Per_Site"].isna().all()


def test_haploid_per_locus_statistics_use_one_allele_copy_per_sample() -> None:
    matrix = np.array([[0.0], [2.0], [2.0]])
    snpio_statistics = pd.DataFrame({"Ho": [0.0], "He": [4 / 9], "Pi": [2 / 3]})

    observed = per_locus_statistics(matrix, snpio_statistics, ploidy=1)

    assert observed.loc[0, "MAF"] == pytest.approx(1 / 3)
    assert observed.loc[0, "Singletons"] == 1
    assert np.isnan(observed.loc[0, "F_inbreeding"])


def test_tajimas_d_is_multisite_and_excludes_missing_loci() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 1.0, np.nan],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 2.0, 2.0, 2.0],
        ]
    )
    observed = tajimas_d_complete_sites(matrix)
    assert observed.complete_loci == 3
    assert observed.segregating_sites == 3
    assert observed.reason == "complete_call_loci"
    assert observed.value == pytest.approx(-0.30440834979308423)


@pytest.mark.parametrize(
    ("matrix", "reason"),
    [
        (np.array([[0.0, 1.0]]), "fewer_than_two_samples"),
        (
            np.array([[np.nan, np.nan], [0.0, np.nan]]),
            "no_complete_call_loci",
        ),
        (np.array([[0.0, 2.0], [0.0, 2.0]]), "no_segregating_complete_call_loci"),
    ],
)
def test_tajimas_d_reports_undefined_edge_cases(
    matrix: np.ndarray,
    reason: str,
) -> None:
    observed = tajimas_d_complete_sites(matrix)
    assert np.isnan(observed.value)
    assert observed.reason == reason


def test_snpio_ld_uses_bounded_reproducible_unlinked_estimator() -> None:
    class FakePopGenStatistics:
        def calculate_linkage_disequilibrium(self, **kwargs):
            assert kwargs == {
                "assume_unlinked": True,
                "n_bootstraps": 0,
                "n_jobs": 1,
                "max_pairs": 100_000,
                "pairwise_sample_size": 0,
                "seed": 42,
                "save_pairwise": False,
                "save_plots": False,
            }
            return SimpleNamespace(
                summary=pd.DataFrame(
                    {
                        "Population": ["Overall"],
                        "r2D": [0.2],
                        "rDz": [0.1],
                        "Ne": [5 / 3],
                        "Loci": [20],
                        "Pairs": [100],
                    }
                )
            )

    observed = snpio_ld_overall(FakePopGenStatistics(), np.zeros((4, 20)))
    assert observed.r2d == pytest.approx(0.2)
    assert observed.pairs == 100
    assert observed.reason == "snpio_ragsdale_gravel_unbiased"


def test_snpio_ld_handles_too_few_samples_without_calling_snpio() -> None:
    class UnexpectedCall:
        def calculate_linkage_disequilibrium(self, **kwargs):
            raise AssertionError("SNPio LD should not be called")

    observed = snpio_ld_overall(UnexpectedCall(), np.zeros((3, 2)))
    assert np.isnan(observed.r2d)
    assert observed.reason == "fewer_than_four_samples"


def test_snpio_ld_marks_haploid_data_not_estimable_without_calling_snpio() -> None:
    class UnexpectedCall:
        def calculate_linkage_disequilibrium(self, **kwargs):
            raise AssertionError("SNPio diploid LD should not be called")

    observed = snpio_ld_overall(UnexpectedCall(), np.zeros((10, 20)), ploidy=1)
    assert np.isnan(observed.r2d)
    assert observed.reason == "not_estimable_for_haploid_data"


def test_infer_format_handles_compressed_vcf_and_json_nan(tmp_path) -> None:
    assert infer_format(tmp_path / "input.vcf.gz") == "vcf"
    assert infer_format(tmp_path / "input.phy") == "phy"
    assert json_ready({"value": np.float64(np.nan)}) == {"value": None}


def test_infer_vcf_ploidy_ignores_fully_missing_gt_tokens(tmp_path) -> None:
    vcf = tmp_path / "haploid.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
        "1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0\t1\n"
        "1\t2\t.\tC\tT\t.\tPASS\t.\tGT\t.\t./.\n",
        encoding="utf-8",
    )
    assert infer_vcf_ploidy(vcf) == 1


def test_tetraploid_vcf_uses_exact_allele_copy_counts(tmp_path) -> None:
    vcf = tmp_path / "tetraploid.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "##contig=<ID=1,length=1>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n"
        "1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t"
        "0/0/0/0\t0/0/1/1\t1/1/1/1\n",
        encoding="utf-8",
    )

    dosage = read_biallelic_vcf_alt_dosage(vcf, ploidy=4)
    statistics = polyploid_locus_statistics(dosage, ploidy=4)
    assert infer_vcf_ploidy(vcf) == 4
    np.testing.assert_array_equal(dosage[:, 0], np.array([0.0, 2.0, 4.0]))
    assert statistics.loc[0, "Ho"] == pytest.approx(1 / 3)
    assert statistics.loc[0, "He"] == pytest.approx(0.5)
    assert statistics.loc[0, "Pi"] == pytest.approx(6 / 11)

    with DnaSPSingleLocusAnalyzer(vcf, file_format="vcf") as analyzer:
        locus = analyzer.analyze_population_per_locus("Total")
        summary = analyzer.compute_overall_summary("Total", locus)
    assert locus.loc[0, "MAF"] == pytest.approx(0.5)
    assert np.isnan(locus.loc[0, "F_inbreeding"])
    assert summary["Ploidy"] == 4
    assert summary["Ho_He_Pi_Source"] == "Exact raw VCF allele-copy counts"
    assert summary["SNPio_LD_Status"] == "not_estimable_for_polyploid_data"


def test_analyzer_uses_snpio_without_mutating_source_vcf(tmp_path) -> None:
    vcf = tmp_path / "tiny.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "##contig=<ID=1,length=2>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n"
        "1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\n"
        "1\t2\t.\tC\tT\t.\tPASS\t.\tGT\t0/0\t./.\t0/1\n",
        encoding="utf-8",
    )
    original = vcf.read_bytes()
    with DnaSPSingleLocusAnalyzer(vcf, file_format="vcf") as analyzer:
        locus = analyzer.analyze_population_per_locus("Total")
        summary = analyzer.compute_overall_summary("Total", locus)

    assert locus.shape[0] == 2
    assert summary["Ho_He_Pi_Source"].startswith("SNPio")
    assert summary["Ploidy"] == 2
    assert summary["SNPio_LD_Status"] == "fewer_than_four_samples"
    assert summary["Genotype_Encoding_Source"].startswith("SNPio")
    assert vcf.read_bytes() == original
    assert not (tmp_path / "tiny.vcf.gz").exists()


def test_read_dataset_summaries_keeps_multisite_statistics_at_dataset_level(
    tmp_path,
) -> None:
    locus = tmp_path / "results2" / "Total_LocusStats.csv"
    locus.parent.mkdir()
    locus.write_text("Locus,Pi\nsite1,0.1\n", encoding="utf-8")
    summary = locus.with_name("Total_Summary.json")
    summary.write_text(
        '{"TajimaD_CompleteSites": -0.5, "TajimaD_Status": "complete_call_loci"}\n',
        encoding="utf-8",
    )
    observed = read_dataset_summaries([DatasetFiles("results2", locus, summary)])
    assert observed.to_dict("records") == [
        {
            "dataset": "results2",
            "TajimaD_CompleteSites": -0.5,
            "TajimaD_Status": "complete_call_loci",
        }
    ]


def test_completion_gate_accepts_audited_tsv_schema(tmp_path) -> None:
    metrics = tmp_path / "n58_metrics_long.tsv"
    rows = [
        {
            "dataset_id": "results2",
            "strategy": strategy.lower().replace(" ", "_"),
            "model": model,
        }
        for strategy in ALL_STRATEGIES
        for model in EXPECTED_MODELS
    ]
    pd.DataFrame(rows).to_csv(metrics, sep="\t", index=False)
    observed = load_completed_dataset_labels(
        metrics,
        expected_models=EXPECTED_MODELS,
        require_all_strategies=True,
        expected_strategies=ALL_STRATEGIES,
    )
    assert observed == {"results2"}
