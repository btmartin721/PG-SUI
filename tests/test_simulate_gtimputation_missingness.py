from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "simulate_gtimputation_missingness.py"
)
SPEC = importlib.util.spec_from_file_location(
    "simulate_gtimputation_missingness", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
SIMULATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SIMULATOR
SPEC.loader.exec_module(SIMULATOR)


def write_tiny_vcf(path: Path, n_samples: int = 10) -> tuple[str, ...]:
    samples = tuple(f"sample_{index}" for index in range(n_samples))
    rows = [
        ("1", "10", "A", "G", ["0/0", "0/1", "1/1"]),
        ("1", "20", "C", "T", ["1/1", "0/0", "0/1"]),
        ("2", "30", "G", "A", ["0/1", "1/1", "0/0"]),
    ]
    lines = [
        "##fileformat=VCFv4.2",
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=Genotype>",
        "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=Depth>",
        "\t".join(
            ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
            + list(samples)
        ),
    ]
    for chrom, pos, ref, alt, genotypes in rows:
        sample_fields = [
            f"{genotypes[index % len(genotypes)]}:{10 + index}"
            for index in range(n_samples)
        ]
        lines.append(
            "\t".join(
                [chrom, pos, f"v{pos}", ref, alt, ".", "PASS", ".", "GT:DP"]
                + sample_fields
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return samples


def test_reconstruct_split_is_deterministic_and_complete() -> None:
    first = SIMULATOR.reconstruct_split(25, validation_split=0.30, seed=42)
    second = SIMULATOR.reconstruct_split(25, validation_split=0.30, seed=42)

    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.validation, second.validation)
    assert np.array_equal(first.test, second.test)

    combined = np.concatenate([first.train, first.validation, first.test])
    assert sorted(combined.tolist()) == list(range(25))
    assert len(np.unique(combined)) == 25


def test_write_masked_vcf_changes_only_selected_gt_fields(tmp_path: Path) -> None:
    input_vcf = tmp_path / "input.vcf"
    samples = write_tiny_vcf(input_vcf)
    layout = SIMULATOR.read_vcf_layout(input_vcf)
    mask = np.zeros(layout.original_missing.shape, dtype=bool)
    mask[0, 0] = True
    mask[4, 1] = True
    mask[9, 2] = True
    output_vcf = tmp_path / "masked.vcf"

    SIMULATOR.write_masked_vcf(
        input_vcf=input_vcf,
        output_vcf=output_vcf,
        sim_mask=mask,
        expected_samples=samples,
        ploidy=2,
        force=False,
    )
    n_missing, n_dropped = SIMULATOR.validate_written_vcf(
        output_vcf=output_vcf,
        layout=layout,
        sim_mask=mask,
    )

    assert n_missing == 3
    assert n_dropped == 0
    text = output_vcf.read_text(encoding="utf-8")
    assert "./.:10" in text
    assert "./.:14" in text
    assert "./.:19" in text
    assert "0/1:11" in text


def test_runtime_dataset_exposes_tree_parser_metadata(tmp_path: Path) -> None:
    input_vcf = tmp_path / "input.vcf"
    samples = write_tiny_vcf(input_vcf)
    layout = SIMULATOR.read_vcf_layout(input_vcf)

    genotype_data, truth = SIMULATOR.build_runtime_dataset(
        layout, input_vcf=input_vcf, verbose=False
    )

    assert genotype_data.filename == str(input_vcf)
    assert genotype_data.samples == list(samples)
    assert np.array_equal(
        genotype_data.sample_indices, np.ones(len(samples), dtype=bool)
    )
    assert np.array_equal(
        genotype_data.loci_indices, np.ones(len(layout.variants), dtype=bool)
    )
    assert np.array_equal(truth, layout.truth_zygosity)


def test_tree_container_loads_without_snpio_reader_state(tmp_path: Path) -> None:
    treefile = tmp_path / "dataset.treefile"
    treefile.write_text("(sample_0:0.1,sample_1:0.2);\n", encoding="utf-8")
    dependencies = SIMULATOR.load_runtime_dependencies()

    tree_parser, resolved_treefile = SIMULATOR.build_tree_parser(
        dataset_id="dataset",
        tree_dir=tmp_path,
        tree_suffix=".treefile",
        iqtree_suffix=".iqtree",
        dependencies=dependencies,
    )

    assert resolved_treefile == treefile
    assert tree_parser.treefile == str(treefile)
    assert set(tree_parser.tree.get_tip_labels()) == {"sample_0", "sample_1"}
    assert tree_parser.qmatrix is None
    assert tree_parser.siterates is None


def test_module_origin_reports_missing_dependency() -> None:
    assert (
        SIMULATOR.module_origin("module_that_does_not_exist_for_pgsui_test")
        == "unavailable"
    )


def test_reuse_mode_exports_exact_pgsui_test_coordinates(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    input_vcf = input_dir / "dataset.vcf"
    samples = write_tiny_vcf(input_vcf)
    layout = SIMULATOR.read_vcf_layout(input_vcf)
    split = SIMULATOR.reconstruct_split(len(samples), validation_split=0.40, seed=7)

    reference_root = tmp_path / "reference"
    reference_dir = reference_root / "dataset" / "random" / "masks"
    reference_dir.mkdir(parents=True)
    reference_path = reference_dir / "dataset__sim30__random__seed7.mask.npz"
    sim_mask = np.zeros(layout.original_missing.shape, dtype=bool)
    sim_mask[split.train[0], 0] = True
    sim_mask[split.validation[0], 1] = True
    sim_mask[split.test[0], 0] = True
    sim_mask[split.test[-1], 2] = True
    np.savez_compressed(
        reference_path,
        sim_missing_mask=sim_mask,
        original_missing_mask=layout.original_missing,
        samples=np.asarray(samples, dtype=object),
        dataset_id=np.asarray("dataset"),
        strategy=np.asarray("random"),
    )

    output_dir = tmp_path / "canonical"
    split_tsv, _ = SIMULATOR.write_split_files(
        output_dir=output_dir,
        dataset_id="dataset",
        samples=samples,
        split=split,
        seed=7,
        validation_split=0.40,
        force=False,
    )
    support_rows: list[dict[str, object]] = []
    pgsui_truth = np.array(layout.truth_zygosity, copy=True)
    pgsui_mapping = np.tile(
        np.asarray([0, 1, 2], dtype=np.int8), (len(layout.variants), 1)
    )
    result = SIMULATOR.prepare_strategy(
        input_vcf=input_vcf,
        output_dir=output_dir,
        reference_mask_dir=reference_root,
        pgsui_results_dir=None,
        dataset_id="dataset",
        strategy="random",
        layout=layout,
        split=split,
        split_tsv=split_tsv,
        genotype_data=None,
        pgsui_truth=pgsui_truth,
        pgsui_class_mapping=pgsui_mapping,
        dependencies=None,
        tree_dir=tmp_path / "trees",
        tree_suffix=".treefile",
        iqtree_suffix=".iqtree",
        mask_mode="reuse",
        sim_prop=0.30,
        validation_split=0.40,
        seed=7,
        ploidy=2,
        force=False,
        verbose=False,
        support_rows=support_rows,
    )

    assert result.n_simulated_missing == 4
    assert result.n_test_evaluation == 2
    assert result.n_dropped_mask_positions == 0
    evaluation_path = (output_dir / "manifests" / result.evaluation_mask_tsv).resolve()
    lines = evaluation_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert all("\ttest\t" in line for line in lines[1:])
    header = lines[0].split("\t")
    assert {
        "pgsui_truth_012",
        "pgsui_class_for_vcf_ref",
        "pgsui_class_for_vcf_het",
        "pgsui_class_for_vcf_alt",
    }.issubset(header)

    mask_npz = (output_dir / "manifests" / result.mask_npz).resolve()
    with np.load(mask_npz, allow_pickle=True) as archive:
        assert np.array_equal(archive["test_idx"], split.test)
        assert int(np.asarray(archive["evaluation_mask"], dtype=bool).sum()) == 2


def test_reviewer_run_sheets_record_inputs_and_rerun_settings(
    tmp_path: Path,
) -> None:
    result = SIMULATOR.SimulationResult(
        dataset_id="dataset",
        strategy="nonrandom",
        seed=42,
        sim_prop=0.30,
        validation_split=0.30,
        mask_mode="regenerate",
        reference_mask_checked=True,
        reference_mask_match=True,
        n_reference_simulated_mask_differences=0,
        n_reference_original_mask_differences=0,
        n_samples=10,
        n_loci=3,
        n_cells=30,
        n_original_missing=0,
        n_simulated_missing=9,
        simulated_missing_rate=0.30,
        n_train_samples=7,
        n_validation_samples=1,
        n_test_samples=2,
        n_test_evaluation=2,
        n_written_missing=9,
        n_dropped_mask_positions=0,
        input_vcf="../../inputs/test-vcf-files/dataset.vcf",
        reference_mask="../../legacy/dataset.mask.npz",
        masked_vcf="../masked_vcfs/dataset/nonrandom/input.vcf",
        mask_npz="../masks/dataset/nonrandom/mask.npz",
        full_mask_tsv="../masks/dataset/nonrandom/mask.tsv",
        evaluation_mask_tsv="../masks/dataset/nonrandom/evaluation.tsv",
        split_tsv="../splits/dataset/split.tsv",
        treefile="../../legacy/iqtree/dataset.treefile",
        input_sha256="input-hash",
        masked_vcf_sha256="masked-hash",
        mask_sha256="mask-hash",
    )

    SIMULATOR.write_reviewer_run_sheets(tmp_path, [result])

    with (tmp_path / "manifests" / "pgsui_run_sheet.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        pgsui_rows = list(csv.DictReader(handle))
    assert len(pgsui_rows) == 2
    assert {row["backend"] for row in pgsui_rows} == {"cpu", "cuda"}
    assert pgsui_rows[0]["qmatrix"] == "../../legacy/iqtree/dataset.iqtree"
    assert pgsui_rows[0]["siterates"] == "../../legacy/iqtree/dataset.iqtree"
    assert pgsui_rows[0]["tune_enabled"] == "True"
    assert pgsui_rows[0]["tune_metrics"] == "f1 mcc average_precision"
    assert pgsui_rows[0]["tune_n_trials"] == "100"
    assert pgsui_rows[0]["train_max_epochs"] == "2000"

    with (tmp_path / "manifests" / "gtimputation_run_sheet.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        gti_rows = list(csv.DictReader(handle))
    assert len(gti_rows) == 2
    assert {row["method"] for row in gti_rows} == {"naive", "som"}
    assert gti_rows[0]["source_masked_vcf"] == "masked_vcfs/dataset/nonrandom/input.vcf"
