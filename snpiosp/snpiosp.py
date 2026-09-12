"""SNPio-backed population-genetic covariates for PG-SUI validation data."""

from __future__ import annotations

import gzip
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import pandas as pd

try:
    from snpio import (
        GenePopReader,
        GenotypeEncoder,
        PhylipReader,
        PopGenStatistics,
        StructureReader,
        VCFReader,
    )
except ImportError as exc:
    raise ImportError(
        "SNPio is required. Install PG-SUI with its dependencies."
    ) from exc

logger = logging.getLogger(__name__)
SUPPORTED_FORMATS = frozenset({"vcf", "phylip", "phy", "genepop", "structure", "str"})
SNPIO_LD_MAX_PAIRS = 100_000
SNPIO_LD_SEED = 42


@dataclass(frozen=True)
class TajimasDResult:
    """Classical multi-site Tajima's D result for complete-call loci."""

    value: float
    complete_loci: int
    segregating_sites: int
    reason: str


@dataclass(frozen=True)
class SNPioLDResult:
    """SNPio Ragsdale-Gravel LD estimates and computation status."""

    r2d: float
    rdz: float
    effective_population_size: float
    loci: int
    pairs: int
    reason: str


def snpio_version() -> str:
    """Return the installed SNPio distribution version."""
    try:
        return version("snpio")
    except PackageNotFoundError:
        return "unknown"


def clean_012_matrix(matrix: np.ndarray) -> np.ndarray:
    """Validate SNPio's diploid 0/1/2 encoding and convert missing calls to NaN.

    Args:
        matrix: SNPio ``GenotypeEncoder.genotypes_012`` matrix with shape
            ``(n_samples, n_loci)``.

    Returns:
        Owned float matrix containing only 0, 1, 2, and NaN.

    Raises:
        ValueError: If the matrix is not two-dimensional, is empty, or contains
            a nonmissing value outside SNPio's 0/1/2 encoding.
    """
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D genotype matrix, found shape {values.shape}")
    if 0 in values.shape:
        raise ValueError(
            f"Genotype matrix must be nonempty, found shape {values.shape}"
        )
    cleaned = np.array(values, dtype=float, copy=True)
    cleaned[cleaned == -9] = np.nan
    valid = np.isnan(cleaned) | np.isin(cleaned, (0.0, 1.0, 2.0))
    if not np.all(valid):
        unexpected = np.unique(cleaned[~valid]).tolist()
        raise ValueError(f"Unexpected SNPio 012 values: {unexpected}")
    return cleaned


def infer_vcf_ploidy(path: Path) -> int:
    """Infer one consistent positive ploidy from called VCF GTs."""
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    observed: set[int] = set()
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 10:
                continue
            format_keys = fields[8].split(":")
            if "GT" not in format_keys:
                continue
            genotype_index = format_keys.index("GT")
            for sample in fields[9:]:
                values = sample.split(":")
                if genotype_index >= len(values):
                    continue
                alleles = re.split(r"[/|]", values[genotype_index])
                if not alleles or any(allele in {"", "."} for allele in alleles):
                    continue
                observed.add(len(alleles))
    ploidies = tuple(sorted(observed))
    if len(ploidies) != 1 or ploidies[0] < 1:
        raise ValueError(
            f"VCF must contain one consistent positive GT ploidy: {path} has {ploidies}"
        )
    return ploidies[0]


def read_biallelic_vcf_alt_dosage(path: Path, *, ploidy: int) -> np.ndarray:
    """Read exact non-reference allele-copy counts from a biallelic VCF."""
    if ploidy < 1:
        raise ValueError("ploidy must be positive")
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    n_samples: int | None = None
    loci: list[list[float]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                n_samples = len(line.rstrip("\r\n").split("\t")) - 9
                continue
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if n_samples is None or len(fields) != 9 + n_samples:
                raise ValueError(f"Malformed VCF sample columns: {path}")
            if len(fields[4].split(",")) != 1:
                raise ValueError(
                    "Exact polyploid dosage currently requires biallelic VCF loci: "
                    f"{path} has {fields[0]}:{fields[1]}"
                )
            format_keys = fields[8].split(":")
            if "GT" not in format_keys:
                raise ValueError(f"VCF record lacks GT: {path}")
            genotype_index = format_keys.index("GT")
            locus: list[float] = []
            for sample in fields[9:]:
                values = sample.split(":")
                genotype = values[genotype_index]
                alleles = re.split(r"[/|]", genotype)
                if any(allele in {"", "."} for allele in alleles):
                    locus.append(np.nan)
                    continue
                if len(alleles) != ploidy:
                    raise ValueError(
                        f"VCF GT ploidy differs from {ploidy}: {genotype!r} in {path}"
                    )
                try:
                    indices = [int(allele) for allele in alleles]
                except ValueError as exc:
                    raise ValueError(f"Invalid GT in {path}: {genotype!r}") from exc
                if not set(indices).issubset({0, 1}):
                    raise ValueError(f"Invalid biallelic GT in {path}: {genotype!r}")
                locus.append(float(sum(index > 0 for index in indices)))
            loci.append(locus)
    if n_samples is None or not loci:
        raise ValueError(f"VCF contains no genotype matrix: {path}")
    return np.asarray(loci, dtype=float).T


def polyploid_locus_statistics(
    allele_copies: np.ndarray,
    *,
    ploidy: int,
) -> pd.DataFrame:
    """Calculate Ho, He, and unbiased pi from exact polyploid allele copies."""
    values = np.asarray(allele_copies, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError(f"Expected a nonempty dosage matrix, found {values.shape}")
    valid = np.isnan(values) | ((values >= 0) & (values <= ploidy))
    if not np.all(valid):
        raise ValueError("Polyploid dosage is outside the allowed allele-copy range")
    observed = np.sum(~np.isnan(values), axis=0)
    chromosomes = ploidy * observed
    alt_counts = np.nansum(values, axis=0, dtype=float)
    allele_frequency = _safe_divide(alt_counts, chromosomes)
    heterozygotes = np.sum(
        (~np.isnan(values)) & (values > 0) & (values < ploidy), axis=0
    )
    ho = _safe_divide(heterozygotes, observed)
    he = 2.0 * allele_frequency * (1.0 - allele_frequency)
    he[observed == 0] = np.nan
    correction = _safe_divide(chromosomes, chromosomes - 1)
    pi = he * correction
    pi[chromosomes <= 1] = np.nan
    return pd.DataFrame({"Ho": ho, "He": he, "Pi": pi})


def _allele_copy_matrix(matrix: np.ndarray, ploidy: int) -> np.ndarray:
    """Return allele-copy counts consistent with haploid or diploid ploidy."""
    if ploidy not in {1, 2}:
        raise ValueError("ploidy must be 1 or 2")
    values = clean_012_matrix(matrix)
    if ploidy == 2:
        return values
    return np.where(np.isnan(values), np.nan, (values > 0).astype(float))


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    np.divide(numerator, denominator, out=output, where=denominator != 0)
    return output


def _harmonic_number(order: int, power: int = 1) -> float:
    if order < 1:
        return 0.0
    values = np.arange(1, order + 1, dtype=float)
    return float(np.sum(1.0 / values**power))


def tajimas_d_complete_sites(
    matrix: np.ndarray,
    *,
    ploidy: int = 2,
    allele_copies: np.ndarray | None = None,
) -> TajimasDResult:
    """Calculate classical multi-site Tajima's D on complete-call SNP loci.

    The standard statistic assumes a common chromosome sample size across
    sites. To avoid silently applying an invalid variable-sample-size formula,
    loci containing missing calls are excluded and their count is reported.

    Args:
        matrix: Clean diploid dosage matrix with shape ``(samples, loci)``.

    Returns:
        Statistic, number of complete loci, number of segregating complete
        sites, and an explicit status reason.
    """
    values = (
        _allele_copy_matrix(matrix, ploidy)
        if allele_copies is None
        else np.asarray(allele_copies, dtype=float)
    )
    if values.shape != np.asarray(matrix).shape:
        raise ValueError("Allele-copy matrix shape differs from genotype categories")
    n_samples = values.shape[0]
    complete = ~np.isnan(values).any(axis=0)
    complete_values = values[:, complete]
    complete_loci = int(complete.sum())
    if n_samples < 2:
        return TajimasDResult(np.nan, complete_loci, 0, "fewer_than_two_samples")
    if complete_loci == 0:
        return TajimasDResult(np.nan, 0, 0, "no_complete_call_loci")

    n_chromosomes = ploidy * n_samples
    alt_counts = complete_values.sum(axis=0, dtype=float)
    segregating = (alt_counts > 0) & (alt_counts < n_chromosomes)
    segregating_sites = int(segregating.sum())
    if segregating_sites == 0:
        return TajimasDResult(
            np.nan, complete_loci, 0, "no_segregating_complete_call_loci"
        )

    allele_frequency = alt_counts[segregating] / n_chromosomes
    pi_total = float(
        np.sum(
            2.0
            * allele_frequency
            * (1.0 - allele_frequency)
            * n_chromosomes
            / (n_chromosomes - 1)
        )
    )
    a1 = _harmonic_number(n_chromosomes - 1)
    a2 = _harmonic_number(n_chromosomes - 1, power=2)
    b1 = (n_chromosomes + 1) / (3.0 * (n_chromosomes - 1))
    b2 = (2.0 * (n_chromosomes**2 + n_chromosomes + 3)) / (
        9.0 * n_chromosomes * (n_chromosomes - 1)
    )
    c1 = b1 - 1.0 / a1
    c2 = b2 - (n_chromosomes + 2) / (a1 * n_chromosomes) + a2 / a1**2
    e1 = c1 / a1
    e2 = c2 / (a1**2 + a2)
    variance = e1 * segregating_sites + e2 * segregating_sites * (segregating_sites - 1)
    if not np.isfinite(variance) or variance <= 0:
        return TajimasDResult(
            np.nan, complete_loci, segregating_sites, "nonpositive_variance"
        )
    theta_watterson = segregating_sites / a1
    value = (pi_total - theta_watterson) / np.sqrt(variance)
    return TajimasDResult(
        float(value), complete_loci, segregating_sites, "complete_call_loci"
    )


def snpio_ld_overall(
    popgen: PopGenStatistics,
    matrix: np.ndarray,
    *,
    n_jobs: int = 1,
    ploidy: int = 2,
) -> SNPioLDResult:
    """Calculate SNPio's unbiased overall LD estimate without bootstrapping.

    The validation SNPs are explicitly treated as unlinked. SNPio evaluates at
    most 100,000 deterministically sampled pairs and reports the aggregate
    Ragsdale-Gravel ``r2D`` ratio rather than averaging biased pairwise ratios.
    """
    values = clean_012_matrix(matrix)
    if ploidy == 1:
        return SNPioLDResult(
            np.nan,
            np.nan,
            np.nan,
            values.shape[1],
            0,
            "not_estimable_for_haploid_data",
        )
    if ploidy != 2:
        return SNPioLDResult(
            np.nan,
            np.nan,
            np.nan,
            values.shape[1],
            0,
            "not_estimable_for_polyploid_data",
        )
    if values.shape[0] < 4:
        return SNPioLDResult(
            np.nan, np.nan, np.nan, values.shape[1], 0, "fewer_than_four_samples"
        )
    try:
        result = popgen.calculate_linkage_disequilibrium(
            assume_unlinked=True,
            n_bootstraps=0,
            n_jobs=n_jobs,
            max_pairs=SNPIO_LD_MAX_PAIRS,
            pairwise_sample_size=0,
            seed=SNPIO_LD_SEED,
            save_pairwise=False,
            save_plots=False,
        )
    except ValueError as exc:
        return SNPioLDResult(
            np.nan,
            np.nan,
            np.nan,
            values.shape[1],
            0,
            f"not_estimable: {exc}",
        )
    summary = result.summary
    if not isinstance(summary, pd.DataFrame) or len(summary) != 1:
        raise ValueError("SNPio overall LD returned an unexpected summary table")
    row = summary.iloc[0]
    if str(row["Population"]) != "Overall":
        raise ValueError("SNPio overall LD did not label the pooled population")
    return SNPioLDResult(
        r2d=float(row["r2D"]),
        rdz=float(row["rDz"]),
        effective_population_size=float(row["Ne"]),
        loci=int(row["Loci"]),
        pairs=int(row["Pairs"]),
        reason="snpio_ragsdale_gravel_unbiased",
    )


def per_locus_statistics(
    matrix: np.ndarray,
    snpio_statistics: pd.DataFrame,
    *,
    locus_names: np.ndarray | None = None,
    filename: str = "",
    ploidy: int = 2,
    allele_copies: np.ndarray | None = None,
) -> pd.DataFrame:
    """Combine SNPio Ho/He/Pi with explicit dosage-derived statistics."""
    values = clean_012_matrix(matrix)
    allele_copies = (
        _allele_copy_matrix(values, ploidy)
        if allele_copies is None
        else np.asarray(allele_copies, dtype=float)
    )
    if allele_copies.shape != values.shape:
        raise ValueError("Allele-copy matrix shape differs from genotype categories")
    n_samples, n_loci = values.shape
    required = {"Ho", "He", "Pi"}
    if not required.issubset(snpio_statistics.columns):
        missing = sorted(required.difference(snpio_statistics.columns))
        raise ValueError(f"SNPio summary statistics are missing columns: {missing}")
    if len(snpio_statistics) != n_loci:
        raise ValueError(
            "SNPio summary-statistic length differs from the genotype matrix: "
            f"{len(snpio_statistics)} != {n_loci}"
        )
    if locus_names is None:
        locus_names = np.array([f"Locus_{index + 1}" for index in range(n_loci)])
    if len(locus_names) != n_loci:
        raise ValueError("Locus-name count differs from the genotype matrix")

    observed = np.sum(~np.isnan(values), axis=0)
    chromosomes = ploidy * observed
    alt_counts = np.nansum(allele_copies, axis=0, dtype=float)
    allele_frequency = _safe_divide(alt_counts, chromosomes)
    minor_frequency = np.minimum(allele_frequency, 1.0 - allele_frequency)
    segregating = (observed > 0) & (alt_counts > 0) & (alt_counts < chromosomes)
    minor_counts = np.minimum(alt_counts, chromosomes - alt_counts)
    singletons = segregating & np.isclose(minor_counts, 1.0)
    genotype_counts = np.column_stack(
        [np.sum(values == genotype, axis=0) for genotype in (0.0, 1.0, 2.0)]
    )
    genotype_category_count = np.sum(genotype_counts > 0, axis=1)
    frequencies = _safe_divide(genotype_counts, observed[:, np.newaxis])
    raw_diversity = 1.0 - np.nansum(frequencies**2, axis=1)
    diversity_correction = _safe_divide(observed, observed - 1)
    genotype_diversity = raw_diversity * diversity_correction
    genotype_diversity[observed <= 1] = np.nan

    theta_watterson = np.full(n_loci, np.nan)
    theta_watterson[observed > 0] = 0.0
    for sample_size in np.unique(observed[segregating]):
        n_chromosomes = ploidy * int(sample_size)
        a1 = _harmonic_number(n_chromosomes - 1)
        if a1 > 0:
            theta_watterson[segregating & (observed == sample_size)] = 1.0 / a1

    snpio_values = snpio_statistics.reset_index(drop=True)
    ho = pd.to_numeric(snpio_values["Ho"], errors="coerce").to_numpy(float)
    he = pd.to_numeric(snpio_values["He"], errors="coerce").to_numpy(float)
    pi = pd.to_numeric(snpio_values["Pi"], errors="coerce").to_numpy(float)
    fixation = np.full(n_loci, np.nan)
    valid_he = np.isfinite(he) & (he > 0) & (ploidy == 2)
    fixation[valid_he] = 1.0 - ho[valid_he] / he[valid_he]

    return pd.DataFrame(
        {
            "Locus": np.asarray(locus_names),
            "Sample_Size": observed,
            "Missingness": 1.0 - observed / n_samples,
            "SegSites": segregating.astype(int),
            "Singletons": singletons.astype(int),
            "MAF": minor_frequency,
            "Ho": ho,
            "He": he,
            "F_inbreeding": fixation,
            "Heterozygote_Count": genotype_counts[:, 1],
            "Genotype_Category_Count": genotype_category_count,
            "Genotype_Diversity": genotype_diversity,
            "Pi": pi,
            "ThetaWatt_Per_Site": theta_watterson,
            "TajimaD_Per_Site": np.full(n_loci, np.nan),
            "Filename": filename,
            "TotalPos": 1,
            "SNPio_Version": snpio_version(),
        }
    )


def _finite_statistic(series: pd.Series, operation: str) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    if operation == "mean":
        return float(values.mean())
    if operation == "variance":
        return float(values.var(ddof=1)) if values.size > 1 else np.nan
    if operation == "stddev":
        return float(values.std(ddof=1)) if values.size > 1 else np.nan
    raise ValueError(f"Unknown summary operation: {operation}")


class DnaSPSingleLocusAnalyzer:
    """Calculate transparent per-SNP covariates using SNPio's 0/1/2 data model."""

    def __init__(
        self,
        filename: str | Path,
        popmap_file: str | Path | None = None,
        file_format: str = "vcf",
        *,
        n_jobs: int = 1,
        ploidy: int | None = None,
    ) -> None:
        self.filename = Path(filename).expanduser().resolve()
        self.popmap_file = (
            None if popmap_file is None else Path(popmap_file).expanduser().resolve()
        )
        self.file_format = file_format.lower()
        self.n_jobs = n_jobs
        self.requested_ploidy = ploidy
        if self.file_format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format {file_format!r}; choose {sorted(SUPPORTED_FORMATS)}"
            )
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")
        if self.requested_ploidy is not None and self.requested_ploidy < 1:
            raise ValueError("ploidy must be positive or None for VCF inference")
        if not self.filename.is_file():
            raise FileNotFoundError(self.filename)
        if self.popmap_file is not None and not self.popmap_file.is_file():
            raise FileNotFoundError(self.popmap_file)
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="snpiosp_")
        self._load_data()

    def _stage_input(self) -> Path:
        root = Path(self._temporary_directory.name)
        staged = root / self.filename.name
        shutil.copy2(self.filename, staged)
        for suffix in (".tbi", ".csi"):
            index = Path(f"{self.filename}{suffix}")
            if index.is_file():
                shutil.copy2(index, Path(f"{staged}{suffix}"))
        return staged

    @staticmethod
    def _read_phylip_sample_names(path: Path) -> list[str]:
        with path.open(encoding="utf-8") as handle:
            header = handle.readline().split()
            if not header:
                return []
            n_samples = int(header[0])
            samples = [
                line.split()[0] for line in handle if line.strip() and line.split()
            ]
        return samples[:n_samples]

    def _single_pool_popmap(self, staged: Path) -> Path:
        samples = self._read_phylip_sample_names(staged)
        if not samples:
            raise ValueError(f"Could not read PHYLIP samples from {self.filename}")
        path = Path(self._temporary_directory.name) / "single_pool.popmap"
        path.write_text(
            "".join(f"{sample}\tPop1\n" for sample in samples), encoding="utf-8"
        )
        return path

    def _load_data(self) -> None:
        staged = self._stage_input()
        inferred_ploidy = (
            infer_vcf_ploidy(staged)
            if self.file_format == "vcf"
            else self.requested_ploidy or 2
        )
        if (
            self.requested_ploidy is not None
            and self.requested_ploidy != inferred_ploidy
        ):
            raise ValueError(
                f"Configured ploidy {self.requested_ploidy} differs from VCF GT "
                f"ploidy {inferred_ploidy}: {self.filename}"
            )
        self.ploidy = inferred_ploidy
        reader_class = {
            "vcf": VCFReader,
            "phylip": PhylipReader,
            "phy": PhylipReader,
            "genepop": GenePopReader,
            "structure": StructureReader,
            "str": StructureReader,
        }[self.file_format]
        effective_popmap = self.popmap_file
        if effective_popmap is None and self.file_format in {"phylip", "phy"}:
            effective_popmap = self._single_pool_popmap(staged)
        prefix = Path(self._temporary_directory.name) / "snpio"
        kwargs: dict[str, Any] = {
            "filename": str(staged),
            "popmapfile": None if effective_popmap is None else str(effective_popmap),
            "prefix": str(prefix),
            "show_plots": False,
        }
        if self.file_format == "vcf":
            kwargs["disable_progress_bar"] = True
        self.gd = reader_class(**kwargs)
        self.genotype_matrix = clean_012_matrix(GenotypeEncoder(self.gd).genotypes_012)
        self.allele_copy_matrix = (
            read_biallelic_vcf_alt_dosage(staged, ploidy=self.ploidy)
            if self.file_format == "vcf" and self.ploidy > 2
            else _allele_copy_matrix(self.genotype_matrix, self.ploidy)
        )
        if self.allele_copy_matrix.shape != self.genotype_matrix.shape:
            raise ValueError("VCF dosage and SNPio genotype matrices differ in shape")
        if not np.array_equal(
            np.isnan(self.allele_copy_matrix),
            np.isnan(self.genotype_matrix),
        ):
            raise ValueError("VCF dosage and SNPio genotype missingness differ")
        self.samples = np.asarray(self.gd.samples, dtype=str)
        loci = getattr(self.gd, "locus_names", None)
        self.locus_names = (
            np.asarray(loci, dtype=str)
            if loci is not None and len(loci) == self.genotype_matrix.shape[1]
            else np.array(
                [f"Locus_{index + 1}" for index in range(self.genotype_matrix.shape[1])]
            )
        )
        if self.popmap_file is None:
            self.populations = pd.Series("Pop1", index=self.samples, dtype=str)
            self._analyze_populations = False
        else:
            self.populations = pd.Series(
                np.asarray(self.gd.populations, dtype=str), index=self.samples
            )
            self._analyze_populations = True
        popgen = PopGenStatistics(self.gd, verbose=False, debug=False)
        if self.ploidy > 2:
            per_population = {
                population: polyploid_locus_statistics(
                    self.allele_copy_matrix[self.populations.eq(population).to_numpy()],
                    ploidy=self.ploidy,
                )
                for population in self.populations_to_analyze
            }
            self.snpio_summary = {
                "overall": polyploid_locus_statistics(
                    self.allele_copy_matrix, ploidy=self.ploidy
                ),
                "per_population": per_population,
            }
            self.snpio_allele_summary = {}
            self.ho_he_pi_source = "Exact raw VCF allele-copy counts"
        else:
            self.snpio_summary, self.snpio_allele_summary = popgen.summary_statistics(
                method="observed",
                n_jobs=self.n_jobs,
                save_plots=False,
                include_nei=False,
            )
            self.ho_he_pi_source = "SNPio PopGenStatistics.summary_statistics"
        self.snpio_ld = (
            SNPioLDResult(
                np.nan,
                np.nan,
                np.nan,
                self.genotype_matrix.shape[1],
                0,
                "not_run_for_population_mapped_input",
            )
            if self._analyze_populations
            else snpio_ld_overall(
                popgen,
                self.genotype_matrix,
                n_jobs=self.n_jobs,
                ploidy=self.ploidy,
            )
        )

    @property
    def populations_to_analyze(self) -> list[str]:
        """Return explicit populations only; inferred populations are never used."""
        if not self._analyze_populations:
            return []
        return sorted(self.populations.unique().tolist())

    def _get_pop_matrix(self, pop_id: str) -> np.ndarray:
        if pop_id == "Total":
            return self.genotype_matrix
        mask = self.populations.eq(pop_id).to_numpy()
        if not np.any(mask):
            raise KeyError(f"Unknown population: {pop_id}")
        return self.genotype_matrix[mask]

    def _get_pop_allele_copies(self, pop_id: str) -> np.ndarray:
        """Return exact allele-copy counts for one population or the total."""
        if pop_id == "Total":
            return self.allele_copy_matrix
        mask = self.populations.eq(pop_id).to_numpy()
        if not np.any(mask):
            raise KeyError(f"Unknown population: {pop_id}")
        return self.allele_copy_matrix[mask]

    def _get_snpio_statistics(self, pop_id: str) -> pd.DataFrame:
        if pop_id == "Total":
            frame = self.snpio_summary["overall"]
        else:
            frame = self.snpio_summary["per_population"].get(pop_id)
            if frame is None:
                raise KeyError(f"SNPio did not return statistics for {pop_id}")
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"Unexpected SNPio summary type for {pop_id}: {type(frame)}"
            )
        return frame

    def analyze_population_per_locus(self, pop_id: str) -> pd.DataFrame:
        """Return per-locus covariates for the total data or one explicit population."""
        return per_locus_statistics(
            self._get_pop_matrix(pop_id),
            self._get_snpio_statistics(pop_id),
            locus_names=self.locus_names,
            filename=self.filename.name,
            ploidy=self.ploidy,
            allele_copies=self._get_pop_allele_copies(pop_id),
        )

    def compute_overall_summary(
        self,
        pop_id: str,
        df_locus: pd.DataFrame,
    ) -> dict[str, Any]:
        """Summarize one population with explicit statistic definitions."""
        if df_locus.empty:
            raise ValueError(f"Cannot summarize empty per-locus data for {pop_id}")
        matrix = self._get_pop_matrix(pop_id)
        tajima = tajimas_d_complete_sites(
            matrix,
            ploidy=self.ploidy,
            allele_copies=self._get_pop_allele_copies(pop_id),
        )
        snpio_ld = self.snpio_ld
        summary: dict[str, Any] = {
            "Population": pop_id,
            "N_Loci": len(df_locus),
            "N_Samples": matrix.shape[0],
            "Ploidy": self.ploidy,
            "Mean_Sample_Size": _finite_statistic(df_locus["Sample_Size"], "mean"),
            "Var_Sample_Size": _finite_statistic(df_locus["Sample_Size"], "variance"),
            "Total_Segregating_Sites": int(df_locus["SegSites"].sum()),
            "Total_Singletons": int(df_locus["Singletons"].sum()),
            "Mean_Pi": _finite_statistic(df_locus["Pi"], "mean"),
            "Mean_ThetaWatt_Per_Site": _finite_statistic(
                df_locus["ThetaWatt_Per_Site"], "mean"
            ),
            "TajimaD_CompleteSites": tajima.value,
            "TajimaD_CompleteSite_Count": tajima.complete_loci,
            "TajimaD_Segregating_Site_Count": tajima.segregating_sites,
            "TajimaD_Status": tajima.reason,
            "Mean_Genotype_Diversity": _finite_statistic(
                df_locus["Genotype_Diversity"], "mean"
            ),
            "Mean_Ho": _finite_statistic(df_locus["Ho"], "mean"),
            "Mean_He": _finite_statistic(df_locus["He"], "mean"),
            "Mean_F": _finite_statistic(df_locus["F_inbreeding"], "mean"),
            "Mean_MAF": _finite_statistic(df_locus["MAF"], "mean"),
            "Mean_Missingness": _finite_statistic(df_locus["Missingness"], "mean"),
            "Var_Pi": _finite_statistic(df_locus["Pi"], "variance"),
            "StdDev_Pi": _finite_statistic(df_locus["Pi"], "stddev"),
            "StdDev_F": _finite_statistic(df_locus["F_inbreeding"], "stddev"),
            "SNPio_LD_r2D": snpio_ld.r2d,
            "SNPio_LD_rDz": snpio_ld.rdz,
            "SNPio_LD_Ne": snpio_ld.effective_population_size,
            "SNPio_LD_Loci": snpio_ld.loci,
            "SNPio_LD_Pairs": snpio_ld.pairs,
            "SNPio_LD_Status": snpio_ld.reason,
            "SNPio_LD_Max_Pairs": SNPIO_LD_MAX_PAIRS,
            "SNPio_LD_Seed": SNPIO_LD_SEED,
            "SNPio_LD_Assume_Unlinked": not self._analyze_populations,
            "SNPio_Version": snpio_version(),
            "Ho_He_Pi_Source": self.ho_he_pi_source,
            "LD_Source": "SNPio PopGenStatistics.calculate_linkage_disequilibrium",
            "Genotype_Encoding_Source": "SNPio GenotypeEncoder.genotypes_012",
        }
        return summary

    def close(self) -> None:
        """Remove task-private staged inputs and SNPio outputs."""
        self._temporary_directory.cleanup()

    def __enter__(self) -> DnaSPSingleLocusAnalyzer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
