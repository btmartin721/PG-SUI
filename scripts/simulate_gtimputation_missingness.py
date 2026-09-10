#!/usr/bin/env python3
"""Create verified PG-SUI missingness inputs for GTImputation.

The script generates the five missingness strategies used by the PG-SUI
manuscript, writes the simulated calls directly into each VCF ``GT`` field,
and verifies the written VCF against the intended boolean mask.  It also
reconstructs the PG-SUI neural-model train/validation/test split and exports
the exact test coordinates used for head-to-head scoring.

Existing masks can be reused, regenerated, or regenerated and compared
coordinate-for-coordinate with a reference mask set.  The verification mode
is intended for manuscript and reviewer-package regeneration.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pgsui.data_processing.splitting import train_validation_test_indices  # noqa: E402

DEFAULT_STRATEGIES = (
    "random",
    "random_weighted",
    "random_weighted_inv",
    "nonrandom",
    "nonrandom_weighted",
)
NONRANDOM_STRATEGIES = frozenset({"nonrandom", "nonrandom_weighted"})
CLASS_LABELS = {0: "REF", 1: "HET", 2: "ALT", -1: "MISSING"}


@dataclass(frozen=True)
class VariantRecord:
    """VCF locus metadata in input order."""

    locus_index: int
    chrom: str
    pos: str
    variant_id: str
    ref: str
    alt: str


@dataclass(frozen=True)
class VCFLayout:
    """Sample/locus layout and truth calls read directly from a VCF."""

    samples: tuple[str, ...]
    variants: tuple[VariantRecord, ...]
    original_missing: np.ndarray
    truth_zygosity: np.ndarray


@dataclass(frozen=True)
class SplitIndices:
    """Deterministic PG-SUI sample split."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class RuntimeDependencies:
    """Lazily imported PG-SUI simulation and tree-reading callables."""

    vcf_reader_class: type
    genotype_encoder_class: type
    sim_missing_transformer_class: type
    tree_reader: Callable[[str], Any]


@dataclass(frozen=True)
class BenchmarkTreeParser:
    """Tree container exposing the interface used by SimMissingTransformer."""

    tree: Any
    treefile: str
    qmatrix: str | None
    siterates: str | None


@dataclass(frozen=True)
class SimulationResult:
    """One dataset-by-strategy preparation result."""

    dataset_id: str
    strategy: str
    seed: int
    sim_prop: float
    validation_split: float
    mask_mode: str
    reference_mask_checked: bool
    reference_mask_match: bool | None
    n_reference_simulated_mask_differences: int | None
    n_reference_original_mask_differences: int | None
    n_samples: int
    n_loci: int
    n_cells: int
    n_original_missing: int
    n_simulated_missing: int
    simulated_missing_rate: float
    n_train_samples: int
    n_validation_samples: int
    n_test_samples: int
    n_test_evaluation: int
    n_written_missing: int
    n_dropped_mask_positions: int
    input_vcf: str
    reference_mask: str
    masked_vcf: str
    mask_npz: str
    full_mask_tsv: str
    evaluation_mask_tsv: str
    split_tsv: str
    treefile: str
    input_sha256: str
    masked_vcf_sha256: str
    mask_sha256: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate and verify masked VCF inputs for the PG-SUI versus "
            "GTImputation benchmark."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the original unmasked VCF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination for canonical masked VCFs, masks, and manifests.",
    )
    parser.add_argument(
        "--tree-dir",
        type=Path,
        required=True,
        help="Directory containing <dataset>.treefile and optional .iqtree files.",
    )
    parser.add_argument(
        "--reference-mask-dir",
        type=Path,
        help=(
            "Existing mask root used by --mask-mode reuse or verify. Expected "
            "layout: <dataset>/<strategy>/masks/*.mask.npz."
        ),
    )
    parser.add_argument(
        "--pgsui-results-dir",
        type=Path,
        help="Optional PG-SUI results root used to validate saved report support.",
    )
    parser.add_argument(
        "--datasets-file",
        type=Path,
        help="Text file containing one dataset identifier per line.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset identifier to include. Repeat to include multiple datasets.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=DEFAULT_STRATEGIES,
        default=list(DEFAULT_STRATEGIES),
        help="Missingness strategies to prepare.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("reuse", "regenerate", "verify"),
        default="verify",
        help=(
            "reuse loads saved masks; regenerate creates new masks; verify "
            "regenerates and requires exact agreement with saved masks."
        ),
    )
    parser.add_argument("--sim-prop", type=float, default=0.30)
    parser.add_argument("--validation-split", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ploidy", type=int, choices=(1, 2), default=2)
    parser.add_argument("--tree-suffix", default=".treefile")
    parser.add_argument("--iqtree-suffix", default=".iqtree")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace files in the canonical output directory.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def open_text(path: Path, mode: str = "rt"):
    """Open plain-text or gzip-compressed input by magic bytes."""
    if "r" not in mode:
        raise ValueError("open_text is restricted to read modes")
    with path.open("rb") as handle:
        is_gzip = handle.read(2) == b"\x1f\x8b"
    if is_gzip:
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def vcf_stem(path: Path) -> str:
    """Return the dataset identifier for .vcf or .vcf.gz files."""
    name = path.name
    for suffix in (".vcf.gz", ".vcf"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def genotype_is_missing(gt: str) -> bool:
    """Return whether any allele in a VCF GT token is missing."""
    normalized = gt.strip()
    if not normalized:
        return True
    alleles = normalized.replace("|", "/").split("/")
    return any(allele in {"", "."} for allele in alleles)


def gt_to_zygosity(gt: str) -> int:
    """Encode a VCF GT token as REF=0, HET=1, ALT=2, or missing=-1."""
    normalized = gt.strip().replace("|", "/")
    alleles = normalized.split("/")
    if not alleles or any(allele in {"", "."} for allele in alleles):
        return -1
    try:
        allele_ids = [int(allele) for allele in alleles]
    except ValueError:
        return -1
    if len(set(allele_ids)) > 1:
        return 1
    return 0 if allele_ids[0] == 0 else 2


def extract_gt(format_field: str, sample_field: str) -> tuple[list[str], int, str]:
    """Extract GT and return mutable sample fields plus the GT index."""
    format_keys = format_field.split(":")
    if "GT" not in format_keys:
        raise ValueError(f"VCF FORMAT does not contain GT: {format_field!r}")
    gt_index = format_keys.index("GT")
    sample_values = sample_field.split(":")
    if gt_index >= len(sample_values):
        raise ValueError(f"Sample field lacks GT position {gt_index}: {sample_field!r}")
    return sample_values, gt_index, sample_values[gt_index]


def missing_gt_like(gt: str, ploidy: int) -> str:
    """Return a missing GT token with the source separator and allele count."""
    separator = "|" if "|" in gt else "/"
    if separator in gt:
        allele_count = max(1, len(gt.split(separator)))
    else:
        allele_count = ploidy
    if allele_count == 1:
        return "."
    return separator.join("." for _ in range(allele_count))


def read_vcf_layout(path: Path) -> VCFLayout:
    """Read VCF sample/locus order and truth zygosity."""
    samples: tuple[str, ...] | None = None
    variants: list[VariantRecord] = []
    missing_columns: list[np.ndarray] = []
    zygosity_columns: list[np.ndarray] = []

    with open_text(path) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) < 10:
                    raise ValueError(f"VCF has no sample columns: {path}")
                samples = tuple(fields[9:])
                continue
            if line.startswith("#") or not line.strip():
                continue
            if samples is None:
                raise ValueError(f"VCF data encountered before #CHROM header: {path}")
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 9 + len(samples):
                raise ValueError(
                    f"Malformed VCF record {len(variants)} in {path}: "
                    f"expected {9 + len(samples)} fields, found {len(fields)}"
                )
            variants.append(
                VariantRecord(
                    locus_index=len(variants),
                    chrom=fields[0],
                    pos=fields[1],
                    variant_id=fields[2],
                    ref=fields[3],
                    alt=fields[4],
                )
            )
            missing_column = np.zeros(len(samples), dtype=bool)
            zygosity_column = np.full(len(samples), -1, dtype=np.int8)
            for sample_index, sample_field in enumerate(fields[9:]):
                _, _, gt = extract_gt(fields[8], sample_field)
                missing_column[sample_index] = genotype_is_missing(gt)
                zygosity_column[sample_index] = gt_to_zygosity(gt)
            missing_columns.append(missing_column)
            zygosity_columns.append(zygosity_column)

    if samples is None:
        raise ValueError(f"VCF is missing a #CHROM header: {path}")
    if not variants:
        raise ValueError(f"VCF contains no variant records: {path}")
    return VCFLayout(
        samples=samples,
        variants=tuple(variants),
        original_missing=np.stack(missing_columns, axis=1),
        truth_zygosity=np.stack(zygosity_columns, axis=1),
    )


def canonical_class_mapping(n_loci: int) -> np.ndarray:
    """Return SNPio's canonical REF/HET/ALT 012 mapping per locus."""
    return np.tile(np.asarray([0, 1, 2], dtype=np.int8), (n_loci, 1))


def reconstruct_split(
    n_samples: int, validation_split: float, seed: int
) -> SplitIndices:
    """Reconstruct PG-SUI's deterministic train/validation/test sample split."""
    train_idx, validation_idx, test_idx = train_validation_test_indices(
        n_samples,
        validation_split=validation_split,
        seed=seed,
    )
    return SplitIndices(
        train=np.asarray(train_idx, dtype=np.int64),
        validation=np.asarray(validation_idx, dtype=np.int64),
        test=np.asarray(test_idx, dtype=np.int64),
    )


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_mask(mask: np.ndarray) -> str:
    """Return a stable SHA-256 digest for a boolean mask and its shape."""
    arr = np.ascontiguousarray(mask, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def load_runtime_dependencies() -> RuntimeDependencies:
    """Load SNPio, PG-SUI, and ToyTree only when the workflow executes."""
    import toytree
    from snpio import GenotypeEncoder, VCFReader

    from pgsui.data_processing.transformers import SimMissingTransformer

    return RuntimeDependencies(
        vcf_reader_class=VCFReader,
        genotype_encoder_class=GenotypeEncoder,
        sim_missing_transformer_class=SimMissingTransformer,
        tree_reader=toytree.tree,
    )


def build_runtime_dataset(
    *,
    input_vcf: Path,
    snpio_prefix: Path,
    dependencies: RuntimeDependencies,
    verbose: bool,
) -> tuple[Any, np.ndarray, VCFLayout, Path]:
    """Load and encode a VCF with SNPio's canonical REF/HET/ALT logic."""
    snpio_prefix.parent.mkdir(parents=True, exist_ok=True)
    staged_vcf = snpio_prefix.parent / input_vcf.name
    shutil.copy2(input_vcf, staged_vcf)
    source_index = Path(f"{input_vcf}.tbi")
    if source_index.is_file():
        shutil.copy2(source_index, Path(f"{staged_vcf}.tbi"))
    genotype_data = dependencies.vcf_reader_class(
        filename=str(staged_vcf),
        prefix=str(snpio_prefix),
        disable_progress_bar=not verbose,
        verbose=verbose,
    )
    canonical_vcf = Path(genotype_data.filename).resolve()
    layout = read_vcf_layout(canonical_vcf)
    samples = tuple(str(sample) for sample in genotype_data.samples)
    if samples != layout.samples:
        raise ValueError(f"SNPio sample order is internally inconsistent: {input_vcf}")

    encoder = dependencies.genotype_encoder_class(genotype_data)
    truth = np.asarray(encoder.genotypes_012, dtype=np.int8)
    expected_shape = (len(layout.samples), len(layout.variants))
    if truth.shape != expected_shape:
        raise ValueError(
            f"SNPio encoded {input_vcf} as {truth.shape}; expected "
            f"{expected_shape}. The benchmark does not permit sample or locus "
            "filtering."
        )

    decoded = np.asarray(encoder.decode_012(truth, is_nuc=False))
    roundtrip = np.asarray(encoder.convert_012(decoded.tolist()), dtype=np.int8)
    if not np.array_equal(roundtrip, truth):
        differences = int(np.count_nonzero(roundtrip != truth))
        raise ValueError(
            f"SNPio 012 encode/decode round trip differs at {differences} cells "
            f"for {input_vcf}"
        )
    return genotype_data, truth, layout, canonical_vcf


def persist_canonical_vcf(
    source: Path,
    destination: Path,
    *,
    force: bool,
) -> Path:
    """Persist SNPio's sorted/indexed VCF as the benchmark truth source."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        raise FileExistsError(
            f"Canonical VCF exists: {destination}. Use --force to replace it."
        )
    shutil.copy2(source, destination)
    source_index = Path(f"{source}.tbi")
    if source_index.is_file():
        shutil.copy2(source_index, Path(f"{destination}.tbi"))
    return destination


def build_tree_parser(
    *,
    dataset_id: str,
    tree_dir: Path,
    tree_suffix: str,
    iqtree_suffix: str,
    dependencies: RuntimeDependencies,
) -> tuple[BenchmarkTreeParser, Path]:
    """Build the tree container needed by nonrandom missingness strategies."""
    treefile = tree_dir / f"{dataset_id}{tree_suffix}"
    if not treefile.is_file():
        raise FileNotFoundError(f"Missing tree for {dataset_id}: {treefile}")
    iqtree_file = tree_dir / f"{dataset_id}{iqtree_suffix}"
    auxiliary = str(iqtree_file) if iqtree_file.is_file() else None
    parser = BenchmarkTreeParser(
        tree=dependencies.tree_reader(str(treefile)),
        treefile=str(treefile),
        qmatrix=auxiliary,
        siterates=auxiliary,
    )
    return parser, treefile


def regenerate_mask(
    *,
    genotype_data: Any,
    ground_truth: np.ndarray,
    strategy: str,
    sim_prop: float,
    seed: int,
    tree_parser: Any | None,
    dependencies: RuntimeDependencies,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Regenerate one PG-SUI simulated mask."""
    transformer = dependencies.sim_missing_transformer_class(
        genotype_data,
        tree_parser=tree_parser,
        prop_missing=sim_prop,
        strategy=strategy,
        missing_val=-1,
        mask_missing=True,
        verbose=int(verbose),
        seed=seed,
    )
    transformer.fit(ground_truth.copy())
    transformer.transform(ground_truth.copy())
    return (
        np.asarray(transformer.sim_missing_mask_, dtype=bool),
        np.asarray(transformer.original_missing_mask_, dtype=bool),
    )


def find_reference_mask(
    reference_root: Path,
    dataset_id: str,
    strategy: str,
    sim_prop: float,
    seed: int,
) -> Path:
    """Find the unique saved mask for a dataset and strategy."""
    prefix = (
        f"{dataset_id}__sim{int(round(sim_prop * 100)):02d}__{strategy}__seed{seed}"
    )
    expected_paths = (
        reference_root / dataset_id / strategy / "masks" / f"{prefix}.mask.npz",
        reference_root / "masks" / dataset_id / strategy / f"{prefix}.mask.npz",
    )
    for expected in expected_paths:
        if expected.is_file():
            return expected
    matches = sorted(reference_root.glob(f"**/{prefix}*.mask.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one reference mask for {dataset_id}/{strategy}; "
            f"found {len(matches)} under {reference_root}"
        )
    return matches[0]


def load_reference_mask(
    path: Path,
    *,
    dataset_id: str,
    strategy: str,
    samples: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate a saved mask archive."""
    with np.load(path, allow_pickle=True) as archive:
        sim_mask = np.asarray(archive["sim_missing_mask"], dtype=bool)
        orig_mask = np.asarray(archive["original_missing_mask"], dtype=bool)
        if "samples" in archive:
            saved_samples = tuple(str(value) for value in archive["samples"])
            if saved_samples != tuple(samples):
                raise ValueError(
                    f"Reference-mask sample order differs for {dataset_id}/{strategy}"
                )
        if "dataset_id" in archive and str(archive["dataset_id"]) != dataset_id:
            raise ValueError(f"Reference-mask dataset metadata mismatch: {path}")
        if "strategy" in archive and str(archive["strategy"]) != strategy:
            raise ValueError(f"Reference-mask strategy metadata mismatch: {path}")
    return sim_mask, orig_mask


def validate_mask(
    *,
    sim_mask: np.ndarray,
    orig_mask: np.ndarray,
    layout: VCFLayout,
    dataset_id: str,
    strategy: str,
) -> None:
    """Validate mask shape, original missingness, and non-overlap."""
    expected_shape = (len(layout.samples), len(layout.variants))
    if sim_mask.shape != expected_shape or orig_mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape mismatch for {dataset_id}/{strategy}: sim={sim_mask.shape}, "
            f"orig={orig_mask.shape}, expected={expected_shape}"
        )
    if not np.array_equal(orig_mask, layout.original_missing):
        differing = int(np.count_nonzero(orig_mask ^ layout.original_missing))
        raise ValueError(
            f"Original-missing mask differs from the VCF at {differing} coordinates "
            f"for {dataset_id}/{strategy}"
        )
    overlap = sim_mask & orig_mask
    if bool(overlap.any()):
        raise ValueError(
            f"Simulated and original masks overlap at {int(overlap.sum())} "
            f"coordinates for {dataset_id}/{strategy}"
        )


def write_masked_vcf(
    *,
    input_vcf: Path,
    output_vcf: Path,
    sim_mask: np.ndarray,
    expected_samples: Sequence[str],
    ploidy: int,
    force: bool,
) -> None:
    """Write simulated missing calls directly into VCF GT subfields."""
    output_vcf.parent.mkdir(parents=True, exist_ok=True)
    if output_vcf.exists() and not force:
        raise FileExistsError(
            f"Output exists: {output_vcf}. Use --force to replace it."
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_vcf.name}.", suffix=".tmp", dir=output_vcf.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    locus_index = 0
    try:
        with (
            open_text(input_vcf) as source,
            temporary_path.open("w", encoding="utf-8", newline="") as destination,
        ):
            for line in source:
                if line.startswith("#CHROM"):
                    fields = line.rstrip("\r\n").split("\t")
                    if tuple(fields[9:]) != tuple(expected_samples):
                        raise ValueError(
                            f"VCF sample order changed while writing {input_vcf}"
                        )
                    destination.write(line)
                    continue
                if line.startswith("#") or not line.strip():
                    destination.write(line)
                    continue
                fields = line.rstrip("\r\n").split("\t")
                if locus_index >= sim_mask.shape[1]:
                    raise ValueError(f"VCF has more loci than mask: {input_vcf}")
                if len(fields) != 9 + sim_mask.shape[0]:
                    raise ValueError(
                        f"Sample count mismatch at locus {locus_index} in {input_vcf}"
                    )
                for sample_index in np.flatnonzero(sim_mask[:, locus_index]):
                    column = 9 + int(sample_index)
                    values, gt_index, gt = extract_gt(fields[8], fields[column])
                    if genotype_is_missing(gt):
                        raise ValueError(
                            f"Simulated mask targets an already missing genotype at "
                            f"sample {sample_index}, locus {locus_index}"
                        )
                    values[gt_index] = missing_gt_like(gt, ploidy)
                    fields[column] = ":".join(values)
                destination.write("\t".join(fields) + "\n")
                locus_index += 1
        if locus_index != sim_mask.shape[1]:
            raise ValueError(
                f"VCF has {locus_index} loci but mask has {sim_mask.shape[1]}"
            )
        os.replace(temporary_path, output_vcf)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def validate_written_vcf(
    *,
    output_vcf: Path,
    layout: VCFLayout,
    sim_mask: np.ndarray,
) -> tuple[int, int]:
    """Require exact equality between written and expected missing coordinates."""
    written = read_vcf_layout(output_vcf)
    if written.samples != layout.samples:
        raise ValueError(f"Written VCF sample order differs: {output_vcf}")
    if written.variants != layout.variants:
        raise ValueError(f"Written VCF locus order differs: {output_vcf}")
    expected_missing = layout.original_missing | sim_mask
    differing = written.original_missing ^ expected_missing
    dropped = int(np.count_nonzero(sim_mask & ~written.original_missing))
    if bool(differing.any()):
        unexpected = int(np.count_nonzero(differing & written.original_missing))
        raise ValueError(
            f"Written VCF missing-mask mismatch for {output_vcf}: "
            f"dropped={dropped}, unexpected={unexpected}"
        )
    return int(written.original_missing.sum()), dropped


def write_split_files(
    *,
    output_dir: Path,
    dataset_id: str,
    samples: Sequence[str],
    split: SplitIndices,
    seed: int,
    validation_split: float,
    force: bool,
) -> tuple[Path, Path]:
    """Write portable TSV and NPZ representations of sample splits."""
    split_dir = output_dir / "splits" / dataset_id
    split_dir.mkdir(parents=True, exist_ok=True)
    split_tsv = split_dir / f"{dataset_id}__seed{seed}.split.tsv"
    split_npz = split_dir / f"{dataset_id}__seed{seed}.split.npz"
    if (split_tsv.exists() or split_npz.exists()) and not force:
        raise FileExistsError(
            f"Split output already exists for {dataset_id}; use --force to replace it"
        )
    membership: dict[int, str] = {}
    for label, values in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        membership.update({int(index): label for index in values})
    with split_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample_index", "sample_id", "split"])
        for sample_index, sample_id in enumerate(samples):
            writer.writerow([sample_index, sample_id, membership[sample_index]])
    np.savez_compressed(
        split_npz,
        train_idx=split.train,
        validation_idx=split.validation,
        test_idx=split.test,
        samples=np.asarray(samples, dtype=object),
        seed=np.asarray(seed),
        validation_split=np.asarray(validation_split),
    )
    return split_tsv, split_npz


def mask_rows(
    mask: np.ndarray,
    *,
    layout: VCFLayout,
    split_lookup: dict[int, str],
    pgsui_truth: np.ndarray | None,
    pgsui_class_mapping: np.ndarray | None,
) -> Iterator[list[object]]:
    """Yield portable coordinate rows for a boolean mask."""
    sample_indices, locus_indices = np.where(mask)
    for sample_index, locus_index in zip(sample_indices, locus_indices, strict=True):
        variant = layout.variants[int(locus_index)]
        truth = int(
            pgsui_truth[sample_index, locus_index]
            if pgsui_truth is not None
            else layout.truth_zygosity[sample_index, locus_index]
        )
        pgsui_value: object = truth if pgsui_truth is not None else ""
        class_mapping: list[object] = ["", "", ""]
        if pgsui_class_mapping is not None:
            class_mapping = [
                int(pgsui_class_mapping[locus_index, raw_class])
                for raw_class in range(3)
            ]
        yield [
            int(sample_index),
            layout.samples[int(sample_index)],
            int(locus_index),
            variant.chrom,
            variant.pos,
            variant.variant_id,
            variant.ref,
            variant.alt,
            split_lookup[int(sample_index)],
            truth,
            CLASS_LABELS.get(truth, "UNKNOWN"),
            pgsui_value,
            *class_mapping,
        ]


def write_mask_tsv(
    path: Path,
    *,
    mask: np.ndarray,
    layout: VCFLayout,
    split: SplitIndices,
    pgsui_truth: np.ndarray | None,
    pgsui_class_mapping: np.ndarray | None,
) -> None:
    """Write a coordinate-explicit simulated or evaluation mask."""
    split_lookup: dict[int, str] = {}
    for label, values in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        split_lookup.update({int(index): label for index in values})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "sample_index",
                "sample_id",
                "locus_index",
                "chrom",
                "pos",
                "variant_id",
                "ref",
                "alt",
                "split",
                "truth_zygosity",
                "truth_class",
                "pgsui_truth_012",
                "pgsui_class_for_vcf_ref",
                "pgsui_class_for_vcf_het",
                "pgsui_class_for_vcf_alt",
            ]
        )
        writer.writerows(
            mask_rows(
                mask,
                layout=layout,
                split_lookup=split_lookup,
                pgsui_truth=pgsui_truth,
                pgsui_class_mapping=pgsui_class_mapping,
            )
        )


def prepare_strategy(
    *,
    input_vcf: Path,
    output_dir: Path,
    reference_mask_dir: Path | None,
    pgsui_results_dir: Path | None,
    dataset_id: str,
    strategy: str,
    layout: VCFLayout,
    split: SplitIndices,
    split_tsv: Path,
    genotype_data: Any | None,
    pgsui_truth: np.ndarray | None,
    pgsui_class_mapping: np.ndarray | None,
    dependencies: RuntimeDependencies | None,
    tree_dir: Path,
    tree_suffix: str,
    iqtree_suffix: str,
    mask_mode: str,
    sim_prop: float,
    validation_split: float,
    seed: int,
    ploidy: int,
    force: bool,
    verbose: bool,
    support_rows: list[dict[str, object]],
) -> SimulationResult:
    """Prepare and verify one dataset-by-strategy benchmark input."""
    prefix = (
        f"{dataset_id}__sim{int(round(sim_prop * 100)):02d}__{strategy}__seed{seed}"
    )
    masked_vcf = output_dir / "masked_vcfs" / dataset_id / strategy / f"{prefix}.vcf"
    mask_dir = output_dir / "masks" / dataset_id / strategy
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_npz = mask_dir / f"{prefix}.mask.npz"
    full_mask_tsv = mask_dir / f"{prefix}.full_mask.tsv"
    evaluation_mask_tsv = mask_dir / f"{prefix}.evaluation_mask.tsv"

    reference_mask: np.ndarray | None = None
    reference_orig: np.ndarray | None = None
    reference_path: Path | None = None
    if mask_mode in {"reuse", "verify"} and reference_mask_dir is None:
        raise ValueError(f"--mask-mode {mask_mode} requires --reference-mask-dir")
    if reference_mask_dir is not None:
        reference_path = find_reference_mask(
            reference_mask_dir, dataset_id, strategy, sim_prop, seed
        )
        reference_mask, reference_orig = load_reference_mask(
            reference_path,
            dataset_id=dataset_id,
            strategy=strategy,
            samples=layout.samples,
        )

    tree_parser = None
    treefile: Path | None = None
    regenerated_mask: np.ndarray | None = None
    regenerated_orig: np.ndarray | None = None
    if mask_mode in {"regenerate", "verify"}:
        if dependencies is None or genotype_data is None or pgsui_truth is None:
            raise RuntimeError("PG-SUI runtime data were not initialized")
        if strategy in NONRANDOM_STRATEGIES:
            tree_parser, treefile = build_tree_parser(
                dataset_id=dataset_id,
                tree_dir=tree_dir,
                tree_suffix=tree_suffix,
                iqtree_suffix=iqtree_suffix,
                dependencies=dependencies,
            )
        regenerated_mask, regenerated_orig = regenerate_mask(
            genotype_data=genotype_data,
            ground_truth=pgsui_truth,
            strategy=strategy,
            sim_prop=sim_prop,
            seed=seed,
            tree_parser=tree_parser,
            dependencies=dependencies,
            verbose=verbose,
        )

    reference_match: bool | None = None
    reference_sim_difference: int | None = None
    reference_orig_difference: int | None = None
    if mask_mode == "reuse":
        if reference_mask is None or reference_orig is None:
            raise RuntimeError("Reference mask was not loaded")
        sim_mask, orig_mask = reference_mask, reference_orig
    elif mask_mode == "regenerate":
        if regenerated_mask is None or regenerated_orig is None:
            raise RuntimeError("Mask regeneration did not return masks")
        sim_mask, orig_mask = regenerated_mask, regenerated_orig
        if reference_mask is not None and reference_orig is not None:
            reference_sim_difference = int(
                np.count_nonzero(reference_mask ^ regenerated_mask)
            )
            reference_orig_difference = int(
                np.count_nonzero(reference_orig ^ regenerated_orig)
            )
            reference_match = (
                reference_sim_difference == 0 and reference_orig_difference == 0
            )
    else:
        if any(
            value is None
            for value in (
                reference_mask,
                reference_orig,
                regenerated_mask,
                regenerated_orig,
            )
        ):
            raise RuntimeError(
                "Mask verification requires reference and regenerated masks"
            )
        reference_match = bool(
            np.array_equal(reference_mask, regenerated_mask)
            and np.array_equal(reference_orig, regenerated_orig)
        )
        reference_sim_difference = int(
            np.count_nonzero(reference_mask ^ regenerated_mask)
        )
        reference_orig_difference = int(
            np.count_nonzero(reference_orig ^ regenerated_orig)
        )
        if not reference_match:
            raise ValueError(
                f"Regenerated mask differs from the saved reference for "
                f"{dataset_id}/{strategy}: simulated={reference_sim_difference}, "
                f"original={reference_orig_difference}. Use --mask-mode reuse to retain the "
                "mask associated with the saved PG-SUI reports."
            )
        sim_mask, orig_mask = regenerated_mask, regenerated_orig

    validate_mask(
        sim_mask=sim_mask,
        orig_mask=orig_mask,
        layout=layout,
        dataset_id=dataset_id,
        strategy=strategy,
    )
    evaluation_mask = np.zeros_like(sim_mask, dtype=bool)
    evaluation_mask[split.test] = sim_mask[split.test] & ~orig_mask[split.test]
    if not bool(evaluation_mask.any()):
        raise ValueError(f"No test evaluation coordinates for {dataset_id}/{strategy}")

    write_masked_vcf(
        input_vcf=input_vcf,
        output_vcf=masked_vcf,
        sim_mask=sim_mask,
        expected_samples=layout.samples,
        ploidy=ploidy,
        force=force,
    )
    n_written_missing, n_dropped = validate_written_vcf(
        output_vcf=masked_vcf,
        layout=layout,
        sim_mask=sim_mask,
    )
    if n_dropped:
        raise ValueError(
            f"Masked VCF dropped {n_dropped} simulated coordinates for "
            f"{dataset_id}/{strategy}"
        )

    if mask_npz.exists() and not force:
        raise FileExistsError(f"Mask output exists: {mask_npz}")
    np.savez_compressed(
        mask_npz,
        sim_missing_mask=sim_mask,
        original_missing_mask=orig_mask,
        all_missing_mask=sim_mask | orig_mask,
        evaluation_mask=evaluation_mask,
        train_idx=split.train,
        validation_idx=split.validation,
        test_idx=split.test,
        samples=np.asarray(layout.samples, dtype=object),
        dataset_id=np.asarray(dataset_id),
        strategy=np.asarray(strategy),
        sim_prop=np.asarray(sim_prop),
        validation_split=np.asarray(validation_split),
        seed=np.asarray(seed),
        source_mask=(
            np.asarray(str(reference_path)) if reference_path else np.asarray("")
        ),
    )
    write_mask_tsv(
        full_mask_tsv,
        mask=sim_mask,
        layout=layout,
        split=split,
        pgsui_truth=pgsui_truth,
        pgsui_class_mapping=pgsui_class_mapping,
    )
    write_mask_tsv(
        evaluation_mask_tsv,
        mask=evaluation_mask,
        layout=layout,
        split=split,
        pgsui_truth=pgsui_truth,
        pgsui_class_mapping=pgsui_class_mapping,
    )

    evaluation_truth = (
        pgsui_truth[evaluation_mask]
        if pgsui_truth is not None
        else layout.truth_zygosity[evaluation_mask]
    )
    validate_pgsui_support(
        dataset_id=dataset_id,
        strategy=strategy,
        n_evaluation=int(evaluation_mask.sum()),
        evaluation_truth=evaluation_truth,
        pgsui_results_dir=pgsui_results_dir,
        support_rows=support_rows,
    )

    manifest_dir = (output_dir / "manifests").resolve()

    def portable_path(value: Path) -> str:
        return os.path.relpath(value.resolve(), start=manifest_dir)

    return SimulationResult(
        dataset_id=dataset_id,
        strategy=strategy,
        seed=seed,
        sim_prop=sim_prop,
        validation_split=validation_split,
        mask_mode=mask_mode,
        reference_mask_checked=reference_path is not None,
        reference_mask_match=reference_match,
        n_reference_simulated_mask_differences=reference_sim_difference,
        n_reference_original_mask_differences=reference_orig_difference,
        n_samples=len(layout.samples),
        n_loci=len(layout.variants),
        n_cells=sim_mask.size,
        n_original_missing=int(orig_mask.sum()),
        n_simulated_missing=int(sim_mask.sum()),
        simulated_missing_rate=float(sim_mask.sum() / sim_mask.size),
        n_train_samples=len(split.train),
        n_validation_samples=len(split.validation),
        n_test_samples=len(split.test),
        n_test_evaluation=int(evaluation_mask.sum()),
        n_written_missing=n_written_missing,
        n_dropped_mask_positions=n_dropped,
        input_vcf=portable_path(input_vcf),
        reference_mask=(
            "" if reference_path is None else portable_path(reference_path)
        ),
        masked_vcf=portable_path(masked_vcf),
        mask_npz=portable_path(mask_npz),
        full_mask_tsv=portable_path(full_mask_tsv),
        evaluation_mask_tsv=portable_path(evaluation_mask_tsv),
        split_tsv=portable_path(split_tsv),
        treefile="" if treefile is None else portable_path(treefile),
        input_sha256=sha256_file(input_vcf),
        masked_vcf_sha256=sha256_file(masked_vcf),
        mask_sha256=sha256_mask(sim_mask),
    )


def validate_pgsui_support(
    *,
    dataset_id: str,
    strategy: str,
    n_evaluation: int,
    evaluation_truth: np.ndarray,
    pgsui_results_dir: Path | None,
    support_rows: list[dict[str, object]],
) -> None:
    """Compare reconstructed evaluation support with saved neural reports."""
    if pgsui_results_dir is None:
        return
    raw_counts = {
        label: int(np.count_nonzero(evaluation_truth == class_id))
        for class_id, label in ((0, "REF"), (1, "HET"), (2, "ALT"))
    }
    prefix = f"{dataset_id}_{strategy}_"
    candidates = sorted(
        path
        for path in pgsui_results_dir.glob(f"{prefix}*_output")
        if path.name.removesuffix("_output").removeprefix(prefix)
        in {"cpu", "cuda", "mps", "none"}
    )
    if not candidates:
        raise FileNotFoundError(
            f"No PG-SUI output directories found for {dataset_id}/{strategy}"
        )
    for output_dir in candidates:
        backend = output_dir.name.removesuffix("_output").removeprefix(prefix)
        report_paths = sorted(
            (output_dir / "Unsupervised" / "metrics").glob(
                "Impute*/zygosity_report.json"
            )
        )
        if not report_paths:
            raise FileNotFoundError(f"No neural zygosity reports in {output_dir}")
        for report_path in report_paths:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            reported_total = int(round(float(report["macro avg"]["support"])))
            total_match = reported_total == n_evaluation
            reported_counts = {
                label: int(round(float(report[label]["support"])))
                for label in ("REF", "HET", "ALT")
            }
            class_match = reported_counts == raw_counts
            support_rows.append(
                {
                    "dataset_id": dataset_id,
                    "strategy": strategy,
                    "backend": backend,
                    "model": report_path.parent.name,
                    "evaluation_support": n_evaluation,
                    "reported_support": reported_total,
                    "total_support_match": total_match,
                    "evaluation_ref_support": raw_counts["REF"],
                    "reported_ref_support": reported_counts["REF"],
                    "evaluation_het_support": raw_counts["HET"],
                    "reported_het_support": reported_counts["HET"],
                    "evaluation_alt_support": raw_counts["ALT"],
                    "reported_alt_support": reported_counts["ALT"],
                    "class_support_match": class_match,
                    "report_path": str(report_path.resolve()),
                }
            )
            if not total_match:
                raise ValueError(
                    f"Reconstructed support {n_evaluation} differs from saved PG-SUI "
                    f"support {reported_total}: {report_path}"
                )


def read_dataset_ids(
    *, datasets_file: Path | None, command_line_ids: Iterable[str] | None
) -> list[str] | None:
    """Read and combine explicit dataset filters."""
    values: list[str] = []
    if datasets_file is not None:
        for raw_line in datasets_file.read_text(encoding="utf-8").splitlines():
            value = raw_line.strip()
            if value and not value.startswith("#"):
                values.append(value)
    values.extend(command_line_ids or [])
    if not values:
        return None
    return list(dict.fromkeys(values))


def discover_vcfs(input_dir: Path, dataset_ids: Sequence[str] | None) -> list[Path]:
    """Discover input VCFs, retaining explicit dataset order when supplied."""
    available: dict[str, Path] = {}
    for path in sorted(input_dir.glob("*.vcf*")):
        if not path.is_file() or path.name.endswith((".tbi", ".csi")):
            continue
        dataset_id = vcf_stem(path)
        if dataset_id in available:
            raise ValueError(f"Multiple input VCFs found for dataset {dataset_id}")
        available[dataset_id] = path
    if dataset_ids is None:
        selected = sorted(available)
    else:
        missing = [
            dataset_id for dataset_id in dataset_ids if dataset_id not in available
        ]
        if missing:
            raise FileNotFoundError(
                f"Input VCFs not found for datasets: {', '.join(missing)}"
            )
        selected = list(dataset_ids)
    if not selected:
        raise FileNotFoundError(f"No VCF files found in {input_dir}")
    return [available[dataset_id] for dataset_id in selected]


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write a list of dictionaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_reviewer_run_sheets(
    output_dir: Path, results: Sequence[SimulationResult]
) -> None:
    """Write predictable destinations for manual GTImputation and PG-SUI runs."""
    manifest_dir = output_dir / "manifests"
    gti_root = output_dir / "results-gti"
    gti_metadata_dir = gti_root / "metadata"
    gti_rows: list[dict[str, object]] = []
    pgsui_rows: list[dict[str, object]] = []

    for method in ("naive", "som"):
        (gti_root / "vcfs" / method).mkdir(parents=True, exist_ok=True)
    (output_dir / "results-pgsui").mkdir(parents=True, exist_ok=True)

    for result in results:
        prefix = (
            f"{result.dataset_id}__sim{int(round(result.sim_prop * 100)):02d}__"
            f"{result.strategy}__seed{result.seed}"
        )
        source_masked_vcf = str(Path(result.masked_vcf).relative_to(".."))
        iqtree_file = (
            str(Path(result.treefile).with_suffix(".iqtree")) if result.treefile else ""
        )
        for method in ("naive", "som"):
            filename = f"{prefix}__{method}.vcf"
            copied_vcf = Path("vcfs") / method / filename
            gti_rows.append(
                {
                    "run_id": f"{prefix}__{method}",
                    "method": method,
                    "dataset_name": result.dataset_id,
                    "simulation_strategy": result.strategy,
                    "seed": result.seed,
                    "source_masked_vcf": source_masked_vcf,
                    "copied_vcf": str(copied_vcf),
                    "candidate_vcf_filename": filename,
                    "runtime_seconds": "",
                }
            )

        for backend in ("cpu", "cuda"):
            pgsui_rows.append(
                {
                    "dataset_id": result.dataset_id,
                    "strategy": result.strategy,
                    "backend": backend,
                    "seed": result.seed,
                    "sim_prop": result.sim_prop,
                    "validation_split": result.validation_split,
                    "input_vcf": result.input_vcf,
                    "treefile": result.treefile,
                    "qmatrix": iqtree_file,
                    "siterates": iqtree_file,
                    "output_prefix": (
                        f"../results-pgsui/{result.dataset_id}_{result.strategy}_"
                        f"{backend}"
                    ),
                    "models": (
                        "ImputeAutoencoder ImputeVAE ImputeNLPCA ImputeUBP "
                        "ImputeMostFrequent ImputeRefAllele"
                    ),
                    "scoring_averaging": "macro",
                    "n_jobs": 4,
                    "tune_enabled": True,
                    "tune_metrics": "f1 mcc average_precision",
                    "tune_n_trials": 100,
                    "tune_epochs": 500,
                    "tune_batch_size": 64,
                    "train_batch_size": 128,
                    "train_learning_rate": 0.001,
                    "train_early_stop_gen": 25,
                    "train_min_epochs": 100,
                    "train_max_epochs": 2000,
                }
            )

    write_csv(gti_metadata_dir / "manifest.csv", gti_rows)
    write_csv(manifest_dir / "gtimputation_run_sheet.csv", gti_rows)
    write_csv(manifest_dir / "pgsui_run_sheet.csv", pgsui_rows)


def package_version(name: str) -> str:
    """Return an installed package version or an explicit unavailable marker."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def module_origin(name: str) -> str:
    """Return the resolved source file for an importable module."""
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        return "unavailable"
    return str(Path(spec.origin).resolve())


def git_value(arguments: Sequence[str], *, module_name: str = "pgsui") -> str:
    """Return Git metadata for the source supplying an imported module."""
    origin = module_origin(module_name)
    if origin == "unavailable":
        return "unavailable"
    completed = subprocess.run(
        ["git", *arguments],
        cwd=Path(origin).parent,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def write_provenance(output_dir: Path, arguments: argparse.Namespace) -> None:
    """Write software versions and effective arguments."""
    provenance_dir = output_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    revision = git_value(("rev-parse", "HEAD"))
    description = git_value(("describe", "--tags", "--always", "--dirty"))
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pg_sui": package_version("pg-sui"),
        "pg_sui_module_origin": module_origin("pgsui"),
        "snpio": package_version("snpio"),
        "snpio_module_origin": module_origin("snpio"),
        "numpy": package_version("numpy"),
        "scikit_learn": package_version("scikit-learn"),
        "git_revision": revision,
        "pg_sui_git_revision": revision,
        "pg_sui_git_describe": description,
    }
    (provenance_dir / "software_versions.json").write_text(
        json.dumps(versions, indent=2) + "\n", encoding="utf-8"
    )
    serializable_arguments = {
        key: (
            [str(item) for item in value]
            if isinstance(value, list)
            else str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value in vars(arguments).items()
    }
    (provenance_dir / "effective_arguments.json").write_text(
        json.dumps(serializable_arguments, indent=2) + "\n", encoding="utf-8"
    )


def write_checksums(output_dir: Path) -> None:
    """Write SHA-256 checksums for portable benchmark artifacts."""
    checksum_path = output_dir / "provenance" / "SHA256SUMS"
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path != checksum_path and ".snpio_cache" not in path.parts
    )
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.relative_to(output_dir)}\n")


def write_package_readme(output_dir: Path) -> None:
    """Copy the reviewer instructions into the generated benchmark root."""
    source_dir = Path(__file__).parent
    candidates = (
        source_dir / "PGSUI_GTImputation_REPRODUCIBILITY.md",
        source_dir / "README.md",
    )
    source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None:
        raise FileNotFoundError(
            "Reviewer instructions not found; expected one of: "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    (output_dir / "README.md").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )


def main() -> None:
    """Prepare the complete manuscript comparison grid."""
    args = parse_args()
    if not 0.0 < args.sim_prop < 1.0:
        raise ValueError(f"--sim-prop must be in (0, 1); found {args.sim_prop}")
    if not 0.0 < args.validation_split < 1.0:
        raise ValueError(
            f"--validation-split must be in (0, 1); found {args.validation_split}"
        )

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tree_dir = args.tree_dir.expanduser().resolve()
    reference_mask_dir = (
        args.reference_mask_dir.expanduser().resolve()
        if args.reference_mask_dir
        else None
    )
    pgsui_results_dir = (
        args.pgsui_results_dir.expanduser().resolve()
        if args.pgsui_results_dir
        else None
    )
    datasets_file = (
        args.datasets_file.expanduser().resolve() if args.datasets_file else None
    )
    dataset_ids = read_dataset_ids(
        datasets_file=datasets_file,
        command_line_ids=args.dataset,
    )
    input_vcfs = discover_vcfs(input_dir, dataset_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    dependencies = load_runtime_dependencies()

    results: list[SimulationResult] = []
    support_rows: list[dict[str, object]] = []
    for input_vcf in input_vcfs:
        dataset_id = vcf_stem(input_vcf)
        genotype_data, pgsui_truth, layout, snpio_vcf = build_runtime_dataset(
            input_vcf=input_vcf.resolve(),
            snpio_prefix=output_dir / ".snpio_cache" / dataset_id / "source",
            dependencies=dependencies,
            verbose=args.verbose,
        )
        canonical_input_vcf = persist_canonical_vcf(
            snpio_vcf,
            output_dir / "canonical_vcfs" / dataset_id / f"{dataset_id}.vcf.gz",
            force=args.force,
        )
        split = reconstruct_split(len(layout.samples), args.validation_split, args.seed)
        split_tsv, _ = write_split_files(
            output_dir=output_dir,
            dataset_id=dataset_id,
            samples=layout.samples,
            split=split,
            seed=args.seed,
            validation_split=args.validation_split,
            force=args.force,
        )
        pgsui_class_mapping = canonical_class_mapping(len(layout.variants))

        for strategy in args.strategies:
            result = prepare_strategy(
                input_vcf=canonical_input_vcf,
                output_dir=output_dir,
                reference_mask_dir=reference_mask_dir,
                pgsui_results_dir=pgsui_results_dir,
                dataset_id=dataset_id,
                strategy=strategy,
                layout=layout,
                split=split,
                split_tsv=split_tsv,
                genotype_data=genotype_data,
                pgsui_truth=pgsui_truth,
                pgsui_class_mapping=pgsui_class_mapping,
                dependencies=dependencies,
                tree_dir=tree_dir,
                tree_suffix=args.tree_suffix,
                iqtree_suffix=args.iqtree_suffix,
                mask_mode=args.mask_mode,
                sim_prop=args.sim_prop,
                validation_split=args.validation_split,
                seed=args.seed,
                ploidy=args.ploidy,
                force=args.force,
                verbose=args.verbose,
                support_rows=support_rows,
            )
            results.append(result)
            print(
                f"{dataset_id}\t{strategy}\tsim={result.n_simulated_missing}\t"
                f"test={result.n_test_evaluation}\tdropped=0"
            )

    manifest_rows = [asdict(result) for result in results]
    write_csv(output_dir / "manifests" / "simulation_manifest.csv", manifest_rows)
    write_csv(output_dir / "manifests" / "pgsui_support_validation.csv", support_rows)
    write_reviewer_run_sheets(output_dir, results)
    write_package_readme(output_dir)
    write_provenance(output_dir, args)
    write_checksums(output_dir)
    print(
        f"Prepared {len(results)} dataset-strategy inputs in {output_dir}. "
        "All written VCF masks passed exact coordinate validation."
    )


if __name__ == "__main__":
    main()
