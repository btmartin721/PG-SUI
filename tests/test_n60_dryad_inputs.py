"""Tests for authoritative N=60 Dryad VCF recovery."""

from __future__ import annotations

import csv
import gzip
import urllib.error
from pathlib import Path

import pytest

import pgsui.validation.dryad_inputs as dryad_inputs
from pgsui.validation.dryad_inputs import (
    DryadClient,
    DryadFile,
    DryadInventory,
    SourceDataset,
    all_pair_checks_pass,
    canonicalize_vcf_for_snpio,
    inspect_vcf,
    read_source_manifest,
    select_vcf_candidate,
    stage_iqtree_companions,
    validate_snpio_genotype_equivalence,
    validate_source_pair,
)


def dataset(source: str = "subset/results2_target.recode.phy") -> SourceDataset:
    return SourceDataset(
        dataset_id="results2",
        source_phylip=source,
        expected_samples=2,
        expected_loci=2,
        doi="10.5061/dryad.example",
        title="Example",
    )


def dryad_file(file_id: int, name: str) -> DryadFile:
    return DryadFile(
        file_id=file_id,
        path=name,
        size=10,
        digest="",
        digest_type="",
        download_url=f"https://example.test/{file_id}",
    )


def inventory(*files: DryadFile) -> DryadInventory:
    return DryadInventory(
        dataset_id="results2",
        doi="10.5061/dryad.example",
        version_id=12,
        version_number=1,
        publication_date="2026-01-01",
        download_url="https://example.test/version.zip",
        files=files,
    )


def test_manifest_requires_unique_datasets(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    fields = (
        "DatasetID",
        "Source_PHYLIP",
        "PHYLIP_N_Samples",
        "PHYLIP_N_Loci",
        "DOI",
        "Title",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for _ in range(2):
            writer.writerow(
                {
                    "DatasetID": "results2",
                    "Source_PHYLIP": "results2_example.phy",
                    "PHYLIP_N_Samples": 2,
                    "PHYLIP_N_Loci": 2,
                    "DOI": "10.5061/dryad.example",
                    "Title": "Example",
                }
            )
    with pytest.raises(ValueError, match="duplicate DatasetID"):
        read_source_manifest(path)


def test_single_vcf_is_selected() -> None:
    selected = select_vcf_candidate(
        dataset(),
        (dryad_file(1, "unrelated.vcf"), dryad_file(2, "README.txt")),
    )
    assert selected.status == "selected"
    assert selected.reason == "only_vcf_in_deposit"
    assert selected.selected is not None
    assert selected.selected.path == "unrelated.vcf"


def test_exact_phylip_stem_selects_one_of_multiple_vcfs() -> None:
    selected = select_vcf_candidate(
        dataset(),
        (
            dryad_file(1, "alternative.vcf"),
            dryad_file(2, "target.vcf.gz"),
        ),
    )
    assert selected.status == "selected"
    assert selected.reason == "exact_normalized_filename"
    assert selected.selected is not None
    assert selected.selected.path == "target.vcf.gz"


def test_ambiguous_vcfs_are_not_guessed() -> None:
    selected = select_vcf_candidate(
        dataset("results2_target.phy"),
        (dryad_file(1, "target_a.vcf"), dryad_file(2, "target_b.vcf")),
    )
    assert selected.status == "ambiguous"
    assert selected.selected is None


@pytest.mark.parametrize("compressed", (False, True))
def test_source_pair_validation_checks_dimensions_and_sample_order(
    tmp_path: Path, compressed: bool
) -> None:
    phylip = tmp_path / "results2_target.phy"
    phylip.write_text("2 2\nsample1  AY\nsample2  RT\n", encoding="utf-8")
    suffix = ".vcf.gz" if compressed else ".vcf"
    vcf = tmp_path / f"results2{suffix}"
    content = (
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\tsample2\n"
        "1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\n"
        "1\t2\t.\tC\tT\t.\tPASS\t.\tGT\t0/1\t1/1\n"
    )
    if compressed:
        with gzip.open(vcf, "wt", encoding="utf-8") as handle:
            handle.write(content)
    else:
        vcf.write_text(content, encoding="utf-8")

    inspection = inspect_vcf(vcf)
    checks = validate_source_pair(dataset(), phylip, vcf)

    assert inspection.n_samples == 2
    assert inspection.n_loci == 2
    assert inspection.observed_ploidies == (2,)
    assert all_pair_checks_pass(checks)


def test_snpio_source_pair_encoding_allows_per_locus_ref_alt_inversion(
    tmp_path: Path,
) -> None:
    phylip = tmp_path / "results2_target.phy"
    phylip.write_text("3 2\nsample1  AT\nsample2  RY\nsample3  GC\n", encoding="utf-8")
    vcf = tmp_path / "results2.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "##contig=<ID=1,length=2>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        "sample1\tsample2\tsample3\n"
        "1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\n"
        "1\t2\t.\tC\tT\t.\tPASS\t.\tGT\t1/1\t0/1\t0/0\n",
        encoding="utf-8",
    )

    checks = validate_snpio_genotype_equivalence(phylip, vcf)

    assert checks["snpio_genotype_equivalent"] is True
    assert checks["snpio_symbol_genotype_match"] is True
    assert checks["snpio_biallelic_012_equivalent"] is True
    assert checks["snpio_biallelic_orientation_equivalent_loci"] == 2
    assert checks["snpio_mismatched_loci"] == 0


def test_snpio_source_pair_preserves_haploid_multiallelic_symbols(
    tmp_path: Path,
) -> None:
    phylip = tmp_path / "results2_target.phy"
    phylip.write_text("3 1\nsample1  C\nsample2  A\nsample3  T\n", encoding="utf-8")
    vcf = tmp_path / "results2.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "##contig=<ID=1,length=1>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        "sample1\tsample2\tsample3\n"
        "1\t1\t.\tC\tA,T\t.\tPASS\t.\tGT\t0\t1\t2\n",
        encoding="utf-8",
    )

    checks = validate_snpio_genotype_equivalence(phylip, vcf)

    assert checks["snpio_symbol_genotype_match"] is True
    assert checks["snpio_biallelic_loci"] == 0
    assert checks["snpio_biallelic_012_equivalent"] is True
    assert checks["snpio_genotype_equivalent"] is True


def test_snpio_canonicalization_preserves_valid_historical_gap_calls(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.vcf"
    source.write_text(
        "##fileformat=VCFv4.1\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=Genotype>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
        "1\t0\t.\t-\tC\t.\tPASS\t.\tGT\t0\t1\n"
        "1\t2\t.\tA\tG\t.\tPASS\t.\tGT\t0\t1\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    phylip = tmp_path / "source.phy"
    phylip.write_text("2 2\ns1  NA\ns2  CG\n", encoding="utf-8")
    canonical = tmp_path / "canonical.vcf"

    normalized = canonicalize_vcf_for_snpio(source, canonical)
    checks = validate_snpio_genotype_equivalence(phylip, canonical)

    assert normalized["unsupported_allele_loci"] == 1
    assert normalized["invalid_position_loci"] == 1
    assert normalized["added_contig_headers"] == 1
    assert source.read_bytes() == original
    assert "\tC\tA\t" in canonical.read_text(encoding="utf-8")
    assert checks["snpio_genotype_equivalent"] is True


def test_snpio_canonicalization_collapses_tetraploid_dosage_to_categories(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tetraploid.vcf"
    source.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n"
        "1\t1\t.\tA\tG\t.\tPASS\t.\tGT:DP\t"
        "0/0/0/0:8\t0/0/1/1:9\t1/1/1/1:10\n",
        encoding="utf-8",
    )
    phylip = tmp_path / "tetraploid.phy"
    phylip.write_text("3 1\ns1 A\ns2 R\ns3 G\n", encoding="utf-8")
    canonical = tmp_path / "canonical.vcf"

    normalized = canonicalize_vcf_for_snpio(source, canonical)
    checks = validate_snpio_genotype_equivalence(phylip, canonical)

    assert inspect_vcf(source).observed_ploidies == (4,)
    assert inspect_vcf(canonical).observed_ploidies == (2,)
    assert normalized["collapsed_polyploid_loci"] == 1
    assert normalized["collapsed_polyploid_calls"] == 3
    assert "\tGT\t0/0\t0/1\t1/1\n" in canonical.read_text(encoding="utf-8")
    assert checks["snpio_genotype_equivalent"] is True

    canonical.write_text("conflicting\n", encoding="utf-8")
    canonicalize_vcf_for_snpio(source, canonical, force=True)
    assert inspect_vcf(canonical).observed_ploidies == (2,)


def test_source_pair_validation_accepts_haploid_and_rejects_mixed_or_unknown_calls(
    tmp_path: Path,
) -> None:
    phylip = tmp_path / "results2_target.phy"
    phylip.write_text("2 2\nsample1  AC\nsample2  GT\n", encoding="utf-8")
    for name, genotypes, expected_ploidies, expected_supported in (
        ("haploid.vcf", (("0", "1"), ("1", "0")), (1,), True),
        ("mixed.vcf", (("0", "0/1"), ("1", "1/1")), (1, 2), False),
        ("unknown.vcf", ((".", "."), ("./.", "./.")), (), False),
    ):
        vcf = tmp_path / name
        rows = "".join(
            f"1\t{index}\t.\tA\tG\t.\tPASS\t.\tGT\t{first}\t{second}\n"
            for index, (first, second) in enumerate(genotypes, start=1)
        )
        vcf.write_text(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            f"sample1\tsample2\n{rows}",
            encoding="utf-8",
        )
        inspection = inspect_vcf(vcf)
        checks = validate_source_pair(dataset(), phylip, vcf)
        assert inspection.observed_ploidies == expected_ploidies
        assert checks["supported_consistent_gt_ploidy"] is expected_supported
        assert all_pair_checks_pass(checks) is expected_supported


def test_stage_iqtree_companions_uses_one_rate_run(tmp_path: Path) -> None:
    phylip = tmp_path / "results2_target.phy"
    phylip.write_text("2 2\nsample1 AC\nsample2 GT\n", encoding="utf-8")
    iqtree_root = tmp_path / "source-trees"
    iqtree_root.mkdir()
    for suffix, content in (
        (".treefile", "(sample1,sample2);\n"),
        (".iqtree", "Rate matrix Q\n"),
        (".rate", "Site Rate Cat C_rate\n1 0.5 1 0.5\n2 1.5 2 1.5\n"),
    ):
        (iqtree_root / f"results2_target_rates{suffix}").write_text(
            content, encoding="utf-8"
        )

    paths = stage_iqtree_companions(dataset(), phylip, iqtree_root, tmp_path / "staged")

    assert Path(paths["treefile"]).name == "results2.treefile"
    assert Path(paths["qmatrix"]).name == "results2.iqtree"
    assert Path(paths["siterates"]).name == "results2.rate"
    assert paths["siterates_status"] == "original_complete"
    assert all(paths[f"{key}_sha256"] for key in ("treefile", "qmatrix", "siterates"))


def test_stage_iqtree_companions_expands_uniform_header_only_rates(
    tmp_path: Path,
) -> None:
    phylip = tmp_path / "results2_target.phy"
    phylip.write_text("2 2\nsample1 AC\nsample2 GT\n", encoding="utf-8")
    iqtree_root = tmp_path / "source-trees"
    iqtree_root.mkdir()
    for suffix, content in (
        (".treefile", "(sample1,sample2);\n"),
        (".iqtree", "Model of rate heterogeneity: Uniform\nRate matrix Q\n"),
        (".rate", "# IQ-TREE output\nSite Rate Cat C_rate\n"),
    ):
        (iqtree_root / f"results2_target_rates{suffix}").write_text(
            content, encoding="utf-8"
        )

    paths = stage_iqtree_companions(dataset(), phylip, iqtree_root, tmp_path / "staged")

    rates = Path(paths["siterates"]).read_text(encoding="utf-8")
    assert paths["siterates_status"] == "derived_uniform_from_iqtree"
    assert (
        Path(paths["siterates_original"])
        .read_text(encoding="utf-8")
        .startswith("# IQ-TREE output")
    )
    assert "1\t1.00000\t0\t1.00000" in rates
    assert "2\t1.00000\t0\t1.00000" in rates


def test_archive_fallback_refuses_oversized_deposit_without_downloading(
    tmp_path: Path,
) -> None:
    source = dryad_file(1, "target.vcf")
    oversized = DryadFile(
        **{**source.__dict__, "size": 2_000_000_000},
    )
    client = DryadClient()
    with pytest.raises(PermissionError, match="DRYAD_API_TOKEN"):
        client.download_from_version_archive(
            oversized,
            inventory(oversized),
            tmp_path / "target.vcf",
            max_archive_bytes=1_000_000_000,
        )


def test_client_caps_rate_limit_sleep_and_retries(monkeypatch) -> None:
    response = object()
    responses = iter(
        (
            urllib.error.HTTPError(
                "https://example.test",
                429,
                "Too Many Requests",
                {"RateLimit-Reset": "600"},
                None,
            ),
            response,
        )
    )

    def fake_urlopen(request, timeout):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    sleeps: list[float] = []
    monkeypatch.setattr(dryad_inputs.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(dryad_inputs.time, "time", lambda: 0.0)
    monkeypatch.setattr(dryad_inputs.time, "sleep", sleeps.append)
    client = DryadClient(request_interval=0.0, max_retries=2)
    assert client._request("https://example.test") is response
    assert sleeps == [60.0]
