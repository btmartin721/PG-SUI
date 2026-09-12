"""Recover and verify source VCFs for the N=60 validation datasets."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.response
import zipfile
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

DRYAD_API_ROOT = "https://datadryad.org"
VCF_SUFFIXES: tuple[str, ...] = (".vcf.gz", ".vcf")


@dataclass(frozen=True)
class SourceDataset:
    """One manuscript dataset and its curated Dryad metadata."""

    dataset_id: str
    source_phylip: str
    expected_samples: int
    expected_loci: int
    doi: str
    title: str


@dataclass(frozen=True)
class DryadFile:
    """One file exposed by a Dryad dataset-version record."""

    file_id: int
    path: str
    size: int
    digest: str
    digest_type: str
    download_url: str

    @property
    def is_vcf(self) -> bool:
        """Return whether the file has a supported VCF/BCF suffix."""
        return self.path.lower().endswith(VCF_SUFFIXES)


@dataclass(frozen=True)
class DryadInventory:
    """Latest public Dryad version and its files."""

    dataset_id: str
    doi: str
    version_id: int
    version_number: int
    publication_date: str
    download_url: str
    files: tuple[DryadFile, ...]

    @property
    def total_size(self) -> int:
        """Return the sum of uncompressed file sizes reported by Dryad."""
        return sum(item.size for item in self.files)


@dataclass(frozen=True)
class CandidateSelection:
    """Conservative mapping from a manuscript dataset to one Dryad VCF."""

    status: str
    reason: str
    selected: DryadFile | None
    score: float | None
    candidates: tuple[tuple[DryadFile, float], ...]


@dataclass(frozen=True)
class VCFInspection:
    """Basic VCF dimensions, sample order, and observed GT ploidies."""

    n_samples: int
    n_loci: int
    samples: tuple[str, ...]
    observed_ploidies: tuple[int, ...]


def read_source_manifest(path: Path) -> list[SourceDataset]:
    """Read and validate the curated final-dataset source manifest.

    Args:
        path: CSV containing the N=60 dataset-to-DOI mapping.

    Returns:
        Ordered source datasets.

    Raises:
        ValueError: If required fields, identifiers, or dimensions are invalid.
    """
    required = {
        "DatasetID",
        "Source_PHYLIP",
        "PHYLIP_N_Samples",
        "PHYLIP_N_Loci",
        "DOI",
        "Title",
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Source manifest is missing fields: {sorted(missing)}")
        datasets = [
            SourceDataset(
                dataset_id=row["DatasetID"].strip(),
                source_phylip=row["Source_PHYLIP"].strip(),
                expected_samples=int(row["PHYLIP_N_Samples"]),
                expected_loci=int(row["PHYLIP_N_Loci"]),
                doi=row["DOI"].strip(),
                title=row["Title"].strip(),
            )
            for row in reader
        ]
    if not datasets:
        raise ValueError(f"Source manifest is empty: {path}")
    ids = [dataset.dataset_id for dataset in datasets]
    if len(ids) != len(set(ids)):
        raise ValueError("Source manifest contains duplicate DatasetID values")
    dois = [dataset.doi for dataset in datasets]
    if len(dois) != len(set(dois)):
        raise ValueError("Source manifest contains duplicate DOI values")
    for dataset in datasets:
        if not re.fullmatch(r"results\d+", dataset.dataset_id):
            raise ValueError(f"Invalid DatasetID: {dataset.dataset_id!r}")
        if not dataset.doi.startswith("10.5061/dryad."):
            raise ValueError(f"Expected a Dryad DOI, found {dataset.doi!r}")
        if dataset.expected_samples < 1 or dataset.expected_loci < 1:
            raise ValueError(f"Invalid dimensions for {dataset.dataset_id}")
    return datasets


def _strip_known_suffixes(name: str) -> str:
    lowered = name.lower()
    for suffix in (".phy", ".phylip", ".vcf.gz", ".vcf", ".bcf"):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name


def normalized_source_name(dataset: SourceDataset) -> str:
    """Return a normalized PHYLIP-derived stem for candidate matching."""
    name = _strip_known_suffixes(Path(dataset.source_phylip).name)
    prefix = f"{dataset.dataset_id}_"
    if name.lower().startswith(prefix.lower()):
        name = name[len(prefix) :]
    return normalize_genotype_name(name)


def normalize_genotype_name(name: str) -> str:
    """Normalize a genotype filename without erasing biological identifiers."""
    stem = _strip_known_suffixes(Path(name).name)
    tokens = re.split(r"[^a-z0-9]+", stem.lower())
    disposable = {"vcf", "recode"}
    return "".join(token for token in tokens if token and token not in disposable)


def score_vcf_candidate(dataset: SourceDataset, candidate: DryadFile) -> float:
    """Score filename agreement between a PHYLIP source and Dryad VCF."""
    target = normalized_source_name(dataset)
    observed = normalize_genotype_name(candidate.path)
    if not target or not observed:
        return 0.0
    if target == observed:
        return 1.0
    ratio = SequenceMatcher(None, target, observed).ratio()
    containment = min(len(target), len(observed)) / max(len(target), len(observed))
    if target in observed or observed in target:
        ratio = max(ratio, 0.75 + 0.25 * containment)
    return round(ratio, 6)


def select_vcf_candidate(
    dataset: SourceDataset,
    files: Sequence[DryadFile],
    *,
    minimum_score: float = 0.72,
    minimum_margin: float = 0.08,
) -> CandidateSelection:
    """Select one VCF only when the mapping is unambiguous.

    Single-VCF deposits are selected directly. Multi-VCF deposits require an
    exact normalized filename match or a strong, uniquely best fuzzy match.
    """
    candidates = tuple(
        sorted(
            (
                (item, score_vcf_candidate(dataset, item))
                for item in files
                if item.is_vcf
            ),
            key=lambda pair: (-pair[1], pair[0].path.lower()),
        )
    )
    if not candidates:
        return CandidateSelection("unresolved", "no_vcf_files", None, None, ())
    if len(candidates) == 1:
        selected, score = candidates[0]
        return CandidateSelection(
            "selected", "only_vcf_in_deposit", selected, score, candidates
        )
    exact = [pair for pair in candidates if pair[1] == 1.0]
    if len(exact) == 1:
        selected, score = exact[0]
        return CandidateSelection(
            "selected", "exact_normalized_filename", selected, score, candidates
        )
    best_file, best_score = candidates[0]
    runner_up_score = candidates[1][1]
    if best_score >= minimum_score and best_score - runner_up_score >= minimum_margin:
        return CandidateSelection(
            "selected", "unique_filename_match", best_file, best_score, candidates
        )
    return CandidateSelection(
        "ambiguous",
        "multiple_vcfs_without_unique_filename_match",
        None,
        best_score,
        candidates,
    )


class DryadClient:
    """Small rate-limited client for the public Dryad v2 API."""

    def __init__(
        self,
        *,
        api_root: str = DRYAD_API_ROOT,
        user_agent: str = "PG-SUI-N60-validation/1.8.6",
        request_interval: float = 2.1,
        timeout: float = 60.0,
        max_retries: int = 130,
        bearer_token: str | None = None,
    ) -> None:
        self.api_root = api_root.rstrip("/")
        self.user_agent = user_agent
        self.request_interval = request_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.bearer_token = bearer_token
        self._last_request_at = 0.0

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.request_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _request(
        self,
        url: str,
        *,
        accept: str = "application/json",
    ) -> urllib.response.addinfourl:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._wait_for_rate_limit()
            headers = {"User-Agent": self.user_agent, "Accept": accept}
            if self.bearer_token:
                headers["Authorization"] = f"Bearer {self.bearer_token}"
            request = urllib.request.Request(url, headers=headers)
            try:
                response = urllib.request.urlopen(request, timeout=self.timeout)
                self._last_request_at = time.monotonic()
                return response
            except urllib.error.HTTPError as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                if exc.code != 429 or attempt == self.max_retries - 1:
                    raise
                retry_after = exc.headers.get("Retry-After")
                reset_at = exc.headers.get("RateLimit-Reset")
                if retry_after:
                    delay = float(retry_after)
                elif reset_at:
                    delay = max(1.0, float(reset_at) - time.time())
                else:
                    delay = min(60.0, 2.0**attempt)
                time.sleep(min(60.0, max(delay, self.request_interval)))
            except (TimeoutError, urllib.error.URLError) as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(min(30.0, 2.0**attempt))
        raise RuntimeError(f"Dryad request failed: {url}") from last_error

    def get_json(self, url: str) -> dict[str, Any]:
        """Fetch one JSON resource."""
        with self._request(url) as response:
            return json.load(response)

    def latest_inventory(
        self,
        dataset: SourceDataset,
        *,
        cache_path: Path | None = None,
        refresh: bool = False,
    ) -> DryadInventory:
        """Return the latest file inventory, optionally using a JSON cache."""
        if cache_path is not None and cache_path.is_file() and not refresh:
            return inventory_from_mapping(
                json.loads(cache_path.read_text(encoding="utf-8"))
            )

        identifier = urllib.parse.quote(f"doi:{dataset.doi}", safe="")
        dataset_url = f"{self.api_root}/api/v2/datasets/{identifier}"
        dataset_data = self.get_json(dataset_url)
        version_href = dataset_data["_links"]["stash:version"]["href"]
        version_data = self.get_json(f"{self.api_root}{version_href}")
        version_id = int(version_href.rstrip("/").split("/")[-1])
        files_href = version_data["_links"]["stash:files"]["href"]
        raw_files: list[Mapping[str, Any]] = []
        while files_href:
            page = self.get_json(f"{self.api_root}{files_href}")
            raw_files.extend(page.get("_embedded", {}).get("stash:files", ()))
            files_href = page.get("_links", {}).get("next", {}).get("href", "")
        files = tuple(
            dryad_file_from_mapping(item, self.api_root) for item in raw_files
        )
        inventory = DryadInventory(
            dataset_id=dataset.dataset_id,
            doi=dataset.doi,
            version_id=version_id,
            version_number=int(version_data.get("versionNumber", 0)),
            publication_date=str(version_data.get("publicationDate", "")),
            download_url=(
                f"{self.api_root}{version_data['_links']['stash:download']['href']}"
            ),
            files=files,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(inventory_to_mapping(inventory), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        return inventory

    def download(self, source: DryadFile, destination: Path) -> None:
        """Download one Dryad file atomically without replacing an existing file."""
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f".{destination.name}.part")
        try:
            with self._request(
                source.download_url,
                accept="application/octet-stream",
            ) as response:
                with partial.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    def download_from_version_archive(
        self,
        source: DryadFile,
        inventory: DryadInventory,
        destination: Path,
        *,
        max_archive_bytes: int,
    ) -> None:
        """Download one file through a bounded public whole-version ZIP."""
        if inventory.total_size > max_archive_bytes:
            raise PermissionError(
                "Dryad requires a bearer token for individual-file downloads, and "
                f"the public version archive is {inventory.total_size} bytes, above "
                f"the configured {max_archive_bytes}-byte safety limit. Set "
                "DRYAD_API_TOKEN or raise --max-archive-download-gb deliberately."
            )
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        archive_path = destination.parent / f".{inventory.dataset_id}.dryad.zip"
        archive_partial = archive_path.with_suffix(".zip.part")
        destination_partial = destination.with_name(f".{destination.name}.part")
        try:
            with self._request(
                inventory.download_url,
                accept="application/zip",
            ) as response:
                with archive_partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            archive_partial.replace(archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                matches = [name for name in names if name == source.path]
                if not matches:
                    matches = [
                        name
                        for name in names
                        if Path(name).name == Path(source.path).name
                    ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one {source.path!r} member in {archive_path}; "
                        f"found {matches}"
                    )
                with archive.open(matches[0]) as source_handle:
                    with destination_partial.open("wb") as destination_handle:
                        shutil.copyfileobj(
                            source_handle, destination_handle, length=1024 * 1024
                        )
            destination_partial.replace(destination)
        except Exception:
            destination_partial.unlink(missing_ok=True)
            raise
        finally:
            archive_partial.unlink(missing_ok=True)
            archive_path.unlink(missing_ok=True)

    def download_selected(
        self,
        source: DryadFile,
        inventory: DryadInventory,
        destination: Path,
        *,
        max_archive_bytes: int,
    ) -> str:
        """Download through the token API or a size-bounded public ZIP fallback."""
        if self.bearer_token:
            self.download(source, destination)
            return "individual_file_api"
        self.download_from_version_archive(
            source,
            inventory,
            destination,
            max_archive_bytes=max_archive_bytes,
        )
        return "public_version_archive"


def dryad_file_from_mapping(data: Mapping[str, Any], api_root: str) -> DryadFile:
    """Construct one :class:`DryadFile` from API or cache data."""
    if "file_id" in data:
        return DryadFile(
            file_id=int(data["file_id"]),
            path=str(data["path"]),
            size=int(data["size"]),
            digest=str(data.get("digest", "")),
            digest_type=str(data.get("digest_type", "")),
            download_url=str(data["download_url"]),
        )
    self_href = data["_links"]["self"]["href"]
    download_href = data["_links"]["stash:download"]["href"]
    return DryadFile(
        file_id=int(self_href.rstrip("/").split("/")[-1]),
        path=str(data["path"]),
        size=int(data["size"]),
        digest=str(data.get("digest", "")),
        digest_type=str(data.get("digestType", "")),
        download_url=f"{api_root}{download_href}",
    )


def inventory_to_mapping(inventory: DryadInventory) -> dict[str, Any]:
    """Return a stable JSON-serializable inventory mapping."""
    return {
        "dataset_id": inventory.dataset_id,
        "doi": inventory.doi,
        "version_id": inventory.version_id,
        "version_number": inventory.version_number,
        "publication_date": inventory.publication_date,
        "download_url": inventory.download_url,
        "files": [asdict(item) for item in inventory.files],
    }


def inventory_from_mapping(data: Mapping[str, Any]) -> DryadInventory:
    """Reconstruct an inventory from the stable JSON cache schema."""
    return DryadInventory(
        dataset_id=str(data["dataset_id"]),
        doi=str(data["doi"]),
        version_id=int(data["version_id"]),
        version_number=int(data["version_number"]),
        publication_date=str(data.get("publication_date", "")),
        download_url=str(
            data.get(
                "download_url",
                f"{DRYAD_API_ROOT}/api/v2/versions/{data['version_id']}/download",
            )
        ),
        files=tuple(
            dryad_file_from_mapping(item, DRYAD_API_ROOT) for item in data["files"]
        ),
    )


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Calculate a hexadecimal file digest without loading the file into memory."""
    hasher = hashlib.new(algorithm.replace("-", ""))
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_dryad_digest(path: Path, source: DryadFile) -> None:
    """Raise when a downloaded file disagrees with its Dryad digest or size."""
    if path.stat().st_size != source.size:
        raise ValueError(
            f"Downloaded size mismatch for {path}: {path.stat().st_size} != {source.size}"
        )
    if source.digest and source.digest_type:
        observed = file_digest(path, source.digest_type)
        if observed.lower() != source.digest.lower():
            raise ValueError(
                f"Downloaded {source.digest_type} mismatch for {path}: "
                f"{observed} != {source.digest}"
            )


def canonicalize_vcf_for_snpio(
    source: Path,
    destination: Path,
    *,
    force: bool = False,
) -> dict[str, int]:
    """Write a SNPio-readable VCF while preserving every locus and valid call.

    Some historical VCFs encode alignment gaps as ``-`` alleles. At those
    loci, valid nucleotide alleles and genotypes are remapped, while any call
    containing a non-ACGT allele becomes missing. The locus remains in place
    so IQ-TREE site-rate coordinates do not shift. All standard SNP records
    are copied unchanged.

    Returns:
        Counts of normalized allele records, normalized positions, and added
        contig declarations.
    """
    source_inspection = inspect_vcf(source)
    if len(source_inspection.observed_ploidies) != 1:
        raise ValueError(
            "VCF canonicalization requires one consistent called GT ploidy: "
            f"{source} has {source_inspection.observed_ploidies}"
        )
    source_ploidy = source_inspection.observed_ploidies[0]
    collapse_polyploid = source_ploidy > 2
    opener = gzip.open if source.name.lower().endswith(".gz") else open
    observed_contigs: list[str] = []
    observed_contig_set: set[str] = set()
    declared_contigs: set[str] = set()
    with opener(source, "rt", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.startswith("##contig=<"):
                match = re.search(r"<ID=([^,>]+)", line)
                if match:
                    declared_contigs.add(match.group(1))
            elif line and not line.startswith("#"):
                chrom = line.split("\t", maxsplit=1)[0]
                if chrom not in observed_contig_set:
                    observed_contigs.append(chrom)
                    observed_contig_set.add(chrom)
    missing_contigs = [
        contig for contig in observed_contigs if contig not in declared_contigs
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    normalized_loci = 0
    normalized_positions = 0
    collapsed_polyploid_loci = 0
    collapsed_polyploid_calls = 0
    try:
        with (
            opener(source, "rt", encoding="utf-8", errors="strict") as input_handle,
            temporary.open("w", encoding="utf-8", newline="") as output_handle,
        ):
            for line in input_handle:
                if line.startswith("#CHROM"):
                    if collapse_polyploid:
                        output_handle.write(
                            "##pgsui_snpio_canonicalization=<Version=2,"
                            "InvalidAlleleCalls=missing,"
                            "PolyploidCategoricalCollapse=REF_HET_ALT,"
                            f"SourcePloidy={source_ploidy},AnalysisPloidy=2>\n"
                        )
                    else:
                        output_handle.write(
                            "##pgsui_snpio_canonicalization="
                            "<Version=1,InvalidAlleleCalls=missing>\n"
                        )
                    for contig in missing_contigs:
                        output_handle.write(f"##contig=<ID={contig}>\n")
                    output_handle.write(line)
                    continue
                if line.startswith("#") or not line.strip():
                    output_handle.write(line)
                    continue
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) < 10:
                    raise ValueError(f"Malformed VCF record in {source}")
                try:
                    position = int(fields[1])
                except ValueError:
                    position = 0
                if position < 1:
                    fields[1] = "1"
                    normalized_positions += 1
                alleles = [fields[3], *fields[4].split(",")]
                standard_snp = all(
                    len(allele) == 1 and allele in "ACGT" for allele in alleles
                )
                if standard_snp and not collapse_polyploid:
                    output_handle.write("\t".join(fields) + "\n")
                    continue

                if collapse_polyploid:
                    if not standard_snp or len(alleles) != 2:
                        raise ValueError(
                            "Polyploid categorical collapse requires biallelic ACGT "
                            f"SNPs; unsupported record in {source}: "
                            f"{fields[0]}:{fields[1]}"
                        )
                    format_keys = fields[8].split(":")
                    if "GT" not in format_keys:
                        raise ValueError(f"Polyploid VCF record lacks GT in {source}")
                    genotype_index = format_keys.index("GT")
                    collapsed: list[str] = []
                    for sample in fields[9:]:
                        values = sample.split(":")
                        genotype = values[genotype_index]
                        genotype_alleles = re.split(r"[/|]", genotype)
                        if any(allele in {"", "."} for allele in genotype_alleles):
                            collapsed.append("./.")
                            continue
                        if len(genotype_alleles) != source_ploidy:
                            raise ValueError(
                                "Called GT ploidy changed within polyploid VCF "
                                f"{source}: {genotype!r}"
                            )
                        allele_indices = {int(allele) for allele in genotype_alleles}
                        if not allele_indices.issubset({0, 1}):
                            raise ValueError(
                                f"Invalid biallelic GT in {source}: {genotype!r}"
                            )
                        if allele_indices == {0}:
                            collapsed.append("0/0")
                        elif allele_indices == {1}:
                            collapsed.append("1/1")
                        else:
                            collapsed.append("0/1")
                        collapsed_polyploid_calls += 1
                    fields[7] = "."
                    fields[8] = "GT"
                    fields[9:] = collapsed
                    output_handle.write("\t".join(fields) + "\n")
                    collapsed_polyploid_loci += 1
                    continue

                normalized_loci += 1
                valid_alleles = list(
                    dict.fromkeys(
                        allele
                        for allele in alleles
                        if len(allele) == 1 and allele in "ACGT"
                    )
                )
                if not valid_alleles:
                    valid_alleles = ["A"]
                new_ref = valid_alleles[0]
                new_alt = valid_alleles[1:]
                if not new_alt:
                    new_alt = [next(base for base in "ACGT" if base != new_ref)]
                allele_mapping = {
                    index: valid_alleles.index(allele)
                    for index, allele in enumerate(alleles)
                    if allele in valid_alleles
                }
                format_keys = fields[8].split(":")
                if "GT" not in format_keys:
                    raise ValueError(
                        f"Unsupported-allele VCF record lacks GT in {source}"
                    )
                genotype_index = format_keys.index("GT")
                remapped: list[str] = []
                for sample in fields[9:]:
                    values = sample.split(":")
                    genotype = values[genotype_index]
                    separator = "|" if "|" in genotype else "/"
                    genotype_alleles = genotype.replace("|", "/").split("/")
                    mapped: list[str] = []
                    for allele in genotype_alleles:
                        if allele in {"", "."}:
                            mapped = []
                            break
                        try:
                            mapped.append(str(allele_mapping[int(allele)]))
                        except (KeyError, ValueError):
                            mapped = []
                            break
                    if not mapped:
                        missing = ["."] * max(1, len(genotype_alleles))
                        remapped.append(separator.join(missing))
                    else:
                        remapped.append(separator.join(mapped))
                fields[3] = new_ref
                fields[4] = ",".join(new_alt)
                fields[7] = "."
                fields[8] = "GT"
                fields[9:] = remapped
                output_handle.write("\t".join(fields) + "\n")

        if destination.exists():
            if file_digest(destination) == file_digest(temporary):
                temporary.unlink()
            elif force:
                temporary.replace(destination)
            else:
                raise FileExistsError(
                    f"Canonical VCF exists with different content: {destination}"
                )
        else:
            temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "unsupported_allele_loci": normalized_loci,
        "invalid_position_loci": normalized_positions,
        "added_contig_headers": len(missing_contigs),
        "collapsed_polyploid_loci": collapsed_polyploid_loci,
        "collapsed_polyploid_calls": collapsed_polyploid_calls,
    }


def inspect_vcf(path: Path) -> VCFInspection:
    """Read VCF dimensions, ordered samples, and observed GT ploidies.

    BCF input requires conversion to VCF before this validation step.
    """
    if path.suffix.lower() == ".bcf":
        raise ValueError("BCF inspection requires conversion to VCF")
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    samples: tuple[str, ...] | None = None
    n_loci = 0
    observed_ploidies: set[int] = set()
    with opener(path, "rt", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n\r").split("\t")
                samples = tuple(fields[9:])
            elif line and not line.startswith("#"):
                fields = line.rstrip("\n\r").split("\t")
                if len(fields) < 9:
                    raise ValueError(f"VCF record has fewer than 9 fields: {path}")
                n_loci += 1
                format_keys = fields[8].split(":")
                if "GT" not in format_keys:
                    continue
                genotype_index = format_keys.index("GT")
                for sample_field in fields[9:]:
                    values = sample_field.split(":")
                    if genotype_index >= len(values):
                        continue
                    genotype = values[genotype_index]
                    alleles = re.split(r"[/|]", genotype)
                    if not alleles or any(allele in {"", "."} for allele in alleles):
                        continue
                    observed_ploidies.add(len(alleles))
    if samples is None:
        raise ValueError(f"VCF is missing the #CHROM header: {path}")
    return VCFInspection(
        len(samples), n_loci, samples, tuple(sorted(observed_ploidies))
    )


def read_phylip_samples(path: Path) -> tuple[int, int, tuple[str, ...]]:
    """Read PHYLIP dimensions and sample identifiers from a sequential matrix."""
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().split()
        if len(header) < 2:
            raise ValueError(f"Invalid PHYLIP header: {path}")
        n_samples, n_loci = int(header[0]), int(header[1])
        sample_ids = tuple(line.split(maxsplit=1)[0] for line in handle if line.strip())
    if len(sample_ids) != n_samples:
        raise ValueError(
            f"PHYLIP sample-row count mismatch for {path}: "
            f"{len(sample_ids)} != {n_samples}"
        )
    return n_samples, n_loci, sample_ids


def validate_source_pair(
    dataset: SourceDataset,
    phylip_path: Path,
    vcf_path: Path,
) -> dict[str, Any]:
    """Validate source dimensions and sample identity before benchmark use."""
    phy_samples, phy_loci, phy_ids = read_phylip_samples(phylip_path)
    vcf = inspect_vcf(vcf_path)
    return {
        "phylip_samples": phy_samples,
        "phylip_loci": phy_loci,
        "vcf_samples": vcf.n_samples,
        "vcf_loci": vcf.n_loci,
        "manifest_samples_match": phy_samples == dataset.expected_samples,
        "manifest_loci_match": phy_loci == dataset.expected_loci,
        "sample_count_match": phy_samples == vcf.n_samples,
        "locus_count_match": phy_loci == vcf.n_loci,
        "sample_order_match": phy_ids == vcf.samples,
        "sample_set_match": set(phy_ids) == set(vcf.samples),
        "observed_gt_ploidies": ",".join(map(str, vcf.observed_ploidies)),
        "gt_ploidy": (
            vcf.observed_ploidies[0] if len(vcf.observed_ploidies) == 1 else ""
        ),
        "supported_consistent_gt_ploidy": vcf.observed_ploidies in {(1,), (2,)},
    }


def validate_snpio_genotype_equivalence(
    phylip_path: Path,
    vcf_path: Path,
) -> dict[str, Any]:
    """Compare SNPio 0/1/2 encodings, allowing per-locus allele inversion.

    SNPio's nucleotide matrix provides the lossless cross-format content
    check, including haploid and multiallelic loci. Its 0/1/2 matrices provide
    a second check at biallelic loci, where PHYLIP lacks VCF REF/ALT orientation
    and therefore may be identical or exactly inverted.
    """
    from snpio import GenotypeEncoder, PhylipReader, VCFReader

    _, _, phylip_samples = read_phylip_samples(phylip_path)
    with tempfile.TemporaryDirectory(prefix="pgsui_n60_snpio_") as temporary:
        root = Path(temporary)
        staged_phylip = root / phylip_path.name
        staged_vcf = root / vcf_path.name
        shutil.copy2(phylip_path, staged_phylip)
        shutil.copy2(vcf_path, staged_vcf)
        popmap = root / "single_pool.popmap"
        popmap.write_text(
            "".join(f"{sample}\tPop1\n" for sample in phylip_samples),
            encoding="utf-8",
        )
        vcf_data = VCFReader(
            filename=str(staged_vcf),
            popmapfile=None,
            prefix=str(root / "vcf"),
            show_plots=False,
            disable_progress_bar=True,
        )
        phylip_data = PhylipReader(
            filename=str(staged_phylip),
            popmapfile=str(popmap),
            prefix=str(root / "phylip"),
            verbose=False,
        )
        vcf_012 = np.asarray(GenotypeEncoder(vcf_data).genotypes_012)
        phylip_012 = np.asarray(GenotypeEncoder(phylip_data).genotypes_012)
        vcf_symbols = np.asarray(vcf_data.snp_data, dtype=str)
        phylip_symbols = np.asarray(phylip_data.snp_data, dtype=str)
        biallelic = np.asarray(
            [
                len(alt) == 1
                if isinstance(alt, (list, tuple, np.ndarray))
                else len(str(alt).split(",")) == 1
                for alt in vcf_data.alt
            ],
            dtype=bool,
        )

    shape_match = vcf_012.shape == phylip_012.shape
    sample_order_match = tuple(vcf_data.samples) == tuple(phylip_data.samples)
    if not shape_match:
        return {
            "snpio_encoding": "GenotypeEncoder.genotypes_012",
            "snpio_matrix_shape_match": False,
            "snpio_sample_order_match": sample_order_match,
            "snpio_symbol_genotype_match": False,
            "snpio_biallelic_012_equivalent": False,
            "snpio_genotype_equivalent": False,
            "snpio_biallelic_loci": 0,
            "snpio_biallelic_orientation_equivalent_loci": 0,
            "snpio_mismatched_loci": max(vcf_012.shape[1], phylip_012.shape[1]),
        }

    symbol_match = bool(
        vcf_symbols.shape == phylip_symbols.shape
        and np.array_equal(vcf_symbols, phylip_symbols)
    )
    direct = np.all(vcf_012 == phylip_012, axis=0)
    inverted_vcf = np.where(vcf_012 >= 0, 2 - vcf_012, vcf_012)
    inverted = ~direct & np.all(inverted_vcf == phylip_012, axis=0)
    equivalent = direct | inverted
    biallelic_equivalent = bool(np.all(equivalent[biallelic]))
    return {
        "snpio_encoding": "GenotypeEncoder.genotypes_012",
        "snpio_matrix_shape_match": True,
        "snpio_sample_order_match": sample_order_match,
        "snpio_symbol_genotype_match": symbol_match,
        "snpio_biallelic_012_equivalent": biallelic_equivalent,
        "snpio_genotype_equivalent": symbol_match and biallelic_equivalent,
        "snpio_biallelic_loci": int(biallelic.sum()),
        "snpio_biallelic_orientation_equivalent_loci": int(equivalent[biallelic].sum()),
        "snpio_mismatched_loci": int(np.count_nonzero(vcf_symbols != phylip_symbols)),
    }


def all_pair_checks_pass(checks: Mapping[str, Any]) -> bool:
    """Return whether every boolean source-pair invariant passed."""
    values = [value for value in checks.values() if isinstance(value, bool)]
    return bool(values) and all(values)


def stage_iqtree_companions(
    dataset: SourceDataset,
    phylip_path: Path,
    iqtree_root: Path,
    destination_root: Path,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Stage a coherent IQ-TREE rate-run triplet under the dataset identifier.

    The downloaded N=60 archive stores rate-run companions as
    ``<PHYLIP stem>_rates.{treefile,iqtree,rate}``. Keeping the three artifacts
    from the same IQ-TREE run avoids mixing topology, rate matrix, and site
    rates from separate analyses.
    """
    stem = phylip_path.name
    for suffix in (".phy", ".phylip"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    source_paths = {
        "treefile": iqtree_root / f"{stem}_rates.treefile",
        "qmatrix": iqtree_root / f"{stem}_rates.iqtree",
        "siterates": iqtree_root / f"{stem}_rates.rate",
    }
    missing = [path for path in source_paths.values() if not path.is_file()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing IQ-TREE rate-run companions for {dataset.dataset_id}: {formatted}"
        )
    destination_root.mkdir(parents=True, exist_ok=True)
    destinations: dict[str, str] = {}
    suffixes = {
        "treefile": ".treefile",
        "qmatrix": ".iqtree",
        "siterates": ".rate",
    }
    for kind, source in source_paths.items():
        destination = destination_root / f"{dataset.dataset_id}{suffixes[kind]}"
        if kind == "siterates":
            data_rows = [
                line.split()
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
                and not line.lstrip().startswith("#")
                and not line.lower().lstrip().startswith("site")
            ]
            sites: list[int] = []
            try:
                sites = [int(row[0]) for row in data_rows]
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid IQ-TREE site-rate table: {source}") from exc
            expected_sites = list(range(1, dataset.expected_loci + 1))
            if sites == expected_sites:
                destinations["siterates_status"] = "original_complete"
            elif not sites and "Model of rate heterogeneity: Uniform" in source_paths[
                "qmatrix"
            ].read_text(encoding="utf-8"):
                original = (
                    destination_root
                    / "original"
                    / f"{dataset.dataset_id}{suffixes[kind]}"
                )
                original.parent.mkdir(parents=True, exist_ok=True)
                if original.exists() and file_digest(original) != file_digest(source):
                    if not force:
                        raise FileExistsError(
                            "Original IQ-TREE rate file exists with different "
                            f"content: {original}"
                        )
                    original.unlink()
                if not original.exists():
                    shutil.copy2(source, original)
                destinations["siterates_original"] = str(original)
                destinations["siterates_original_sha256"] = file_digest(original)
                temporary = destination.with_name(f".{destination.name}.tmp")
                with temporary.open("w", encoding="utf-8") as handle:
                    handle.write(
                        "# PG-SUI explicit rates derived from IQ-TREE's "
                        "uniform-rate model\n"
                    )
                    handle.write("Site\tRate\tCat\tC_Rate\n")
                    for site in expected_sites:
                        handle.write(f"{site}\t1.00000\t0\t1.00000\n")
                if destination.exists() and file_digest(destination) != file_digest(
                    temporary
                ):
                    if not force:
                        temporary.unlink()
                        raise FileExistsError(
                            "IQ-TREE destination exists with different content: "
                            f"{destination}"
                        )
                    destination.unlink()
                if destination.exists():
                    temporary.unlink()
                else:
                    temporary.replace(destination)
                destinations["siterates_status"] = "derived_uniform_from_iqtree"
                destinations[kind] = str(destination)
                destinations[f"{kind}_sha256"] = file_digest(destination)
                continue
            else:
                raise ValueError(
                    f"IQ-TREE site-rate coordinates are incomplete for "
                    f"{dataset.dataset_id}: {len(sites)} != "
                    f"{dataset.expected_loci}"
                )
        if destination.exists():
            if file_digest(destination) != file_digest(source):
                if not force:
                    raise FileExistsError(
                        "IQ-TREE destination exists with different content: "
                        f"{destination}"
                    )
                destination.unlink()
                shutil.copy2(source, destination)
        else:
            shutil.copy2(source, destination)
        destinations[kind] = str(destination)
        destinations[f"{kind}_sha256"] = file_digest(destination)
    return destinations


def write_tsv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write dictionaries to a tab-delimited manifest."""
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        raise ValueError("Cannot write an empty manifest")
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(materialized)
