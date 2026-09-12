#!/usr/bin/env python3
"""Discover, download, and verify original Dryad VCFs for the N=60 study."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pgsui.validation.dryad_inputs import (  # noqa: E402
    CandidateSelection,
    DryadClient,
    SourceDataset,
    all_pair_checks_pass,
    canonicalize_vcf_for_snpio,
    file_digest,
    inspect_vcf,
    read_source_manifest,
    select_vcf_candidate,
    stage_iqtree_companions,
    validate_snpio_genotype_equivalence,
    validate_source_pair,
    verify_dryad_digest,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--phylip-dir", type=Path, required=True)
    parser.add_argument(
        "--iqtree-dir",
        type=Path,
        help="Directory containing <PHYLIP stem>_rates IQ-TREE outputs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download conservatively selected VCFs after inventory discovery.",
    )
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--request-interval", type=float, default=2.1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=130,
        help="Maximum retries, allowing bounded 60-second waits across rate resets.",
    )
    parser.add_argument(
        "--token-env",
        default="DRYAD_API_TOKEN",
        help="Environment variable containing a Dryad API bearer token.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--force-canonicalize",
        action="store_true",
        help="Replace derived SNPio-ready VCFs while preserving raw downloads.",
    )
    parser.add_argument(
        "--max-archive-download-gb",
        type=float,
        default=1.0,
        help=(
            "Without a Dryad token, allow a public whole-version ZIP only when "
            "its reported uncompressed size is at most this many GiB."
        ),
    )
    return parser.parse_args()


def phylip_path(dataset: SourceDataset, root: Path) -> Path:
    path = root / Path(dataset.source_phylip).name
    if not path.is_file():
        raise FileNotFoundError(
            f"PHYLIP input not found for {dataset.dataset_id}: {path}"
        )
    return path


def destination_suffix(source_name: str) -> str:
    lowered = source_name.lower()
    if lowered.endswith(".vcf.gz"):
        return ".vcf.gz"
    if lowered.endswith(".bcf"):
        return ".bcf"
    return ".vcf"


def selected_file_fields(selection: CandidateSelection) -> dict[str, Any]:
    selected = selection.selected
    return {
        "selection_status": selection.status,
        "selection_reason": selection.reason,
        "selection_score": "" if selection.score is None else selection.score,
        "dryad_file_id": "" if selected is None else selected.file_id,
        "dryad_path": "" if selected is None else selected.path,
        "dryad_size": "" if selected is None else selected.size,
        "dryad_digest_type": "" if selected is None else selected.digest_type,
        "dryad_digest": "" if selected is None else selected.digest,
        "dryad_download_url": "" if selected is None else selected.download_url,
        "candidate_count": len(selection.candidates),
        "candidate_scores": json.dumps(
            [
                {"path": candidate.path, "score": score}
                for candidate, score in selection.candidates
            ],
            sort_keys=True,
        ),
    }


def installed_version(distribution: str) -> str:
    """Return an installed package version or an explicit marker."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def main() -> int:
    args = parse_args()
    source_manifest = args.source_manifest.expanduser().resolve()
    phylip_root = args.phylip_dir.expanduser().resolve()
    iqtree_root = args.iqtree_dir.expanduser().resolve() if args.iqtree_dir else None
    output_dir = args.output_dir.expanduser().resolve()
    if args.max_archive_download_gb <= 0:
        raise ValueError("--max-archive-download-gb must be positive")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be positive")
    if args.force_download and not args.download:
        raise ValueError("--force-download requires --download")
    max_archive_bytes = int(args.max_archive_download_gb * 1024**3)
    cache_dir = output_dir / "dryad_api_cache"
    vcf_dir = output_dir / "vcfs"
    benchmark_vcf_dir = output_dir / "benchmark_vcfs"
    selected_ids = set(args.dataset)
    datasets = read_source_manifest(source_manifest)
    if selected_ids:
        unknown = selected_ids.difference(dataset.dataset_id for dataset in datasets)
        if unknown:
            raise ValueError(f"Unknown dataset IDs: {sorted(unknown)}")
        datasets = [
            dataset for dataset in datasets if dataset.dataset_id in selected_ids
        ]

    client = DryadClient(
        request_interval=args.request_interval,
        timeout=args.timeout,
        max_retries=args.max_retries,
        bearer_token=os.environ.get(args.token_env),
    )
    pgsui_version = installed_version("pg-sui")
    snpio_version = installed_version("snpio")
    rows: list[dict[str, Any]] = []
    failures = 0
    for index, dataset in enumerate(datasets, start=1):
        print(
            f"[{index}/{len(datasets)}] {dataset.dataset_id} {dataset.doi}", flush=True
        )
        row: dict[str, Any] = {
            "dataset_id": dataset.dataset_id,
            "pgsui_version": pgsui_version,
            "snpio_version": snpio_version,
            "doi": dataset.doi,
            "title": dataset.title,
            "source_phylip": str(phylip_path(dataset, phylip_root)),
            "expected_samples": dataset.expected_samples,
            "expected_loci": dataset.expected_loci,
        }
        try:
            if iqtree_root is not None:
                row.update(
                    stage_iqtree_companions(
                        dataset,
                        Path(row["source_phylip"]),
                        iqtree_root,
                        output_dir / "iqtree",
                        force=args.force_canonicalize,
                    )
                )
            inventory = client.latest_inventory(
                dataset,
                cache_path=cache_dir / f"{dataset.dataset_id}.json",
                refresh=args.refresh,
            )
            selection = select_vcf_candidate(dataset, inventory.files)
            row.update(
                {
                    "dryad_version_id": inventory.version_id,
                    "dryad_version_number": inventory.version_number,
                    "dryad_publication_date": inventory.publication_date,
                    "dryad_deposit_size": inventory.total_size,
                    **selected_file_fields(selection),
                }
            )
            if selection.selected is None:
                failures += 1
                row["validation_status"] = "not_downloaded"
                rows.append(row)
                continue

            suffix = destination_suffix(selection.selected.path)
            destination = vcf_dir / f"{dataset.dataset_id}{suffix}"
            row["local_vcf"] = str(destination)
            if destination.exists() or args.download:
                if destination.exists() and args.force_download:
                    destination.unlink()
                if not destination.exists():
                    row["download_method"] = client.download_selected(
                        selection.selected,
                        inventory,
                        destination,
                        max_archive_bytes=max_archive_bytes,
                    )
                else:
                    row["download_method"] = "existing_verified_file"
                verify_dryad_digest(destination, selection.selected)
                source_inspection = inspect_vcf(destination)
                row["source_observed_gt_ploidies"] = ",".join(
                    map(str, source_inspection.observed_ploidies)
                )
                row["source_gt_ploidy"] = (
                    source_inspection.observed_ploidies[0]
                    if len(source_inspection.observed_ploidies) == 1
                    else ""
                )
                benchmark_vcf = benchmark_vcf_dir / f"{dataset.dataset_id}.vcf"
                canonicalization = canonicalize_vcf_for_snpio(
                    destination,
                    benchmark_vcf,
                    force=args.force_canonicalize,
                )
                row["benchmark_vcf"] = str(benchmark_vcf)
                row["benchmark_vcf_sha256"] = file_digest(benchmark_vcf)
                row.update(
                    {
                        f"canonicalized_{name}": value
                        for name, value in canonicalization.items()
                    }
                )
                checks = validate_source_pair(
                    dataset,
                    Path(row["source_phylip"]),
                    benchmark_vcf,
                )
                checks.update(
                    validate_snpio_genotype_equivalence(
                        Path(row["source_phylip"]),
                        benchmark_vcf,
                    )
                )
                row.update(checks)
                row["ploidy_treatment"] = (
                    "polyploid_ref_het_alt_category_collapse"
                    if row["source_gt_ploidy"] > 2
                    else "preserved"
                )
                row["local_sha256"] = file_digest(destination)
                row["validation_status"] = (
                    "verified" if all_pair_checks_pass(checks) else "mismatch"
                )
                if row["validation_status"] != "verified":
                    failures += 1
            else:
                failures += 1
                row["validation_status"] = "not_downloaded"
        except PermissionError as exc:
            failures += 1
            row["validation_status"] = "requires_token"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"BLOCKED {dataset.dataset_id}: {row['error']}", file=sys.stderr)
        except urllib.error.HTTPError as exc:
            failures += 1
            row["validation_status"] = (
                "rate_limited" if exc.code == 429 else "download_error"
            )
            row["http_status"] = exc.code
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"ERROR {dataset.dataset_id}: {row['error']}", file=sys.stderr)
        except Exception as exc:
            failures += 1
            row["validation_status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"ERROR {dataset.dataset_id}: {row['error']}", file=sys.stderr)
        rows.append(row)

    write_tsv(output_dir / "n60_vcf_source_manifest.tsv", rows)
    summary = {
        "source_manifest": str(source_manifest),
        "phylip_dir": str(phylip_root),
        "iqtree_dir": "" if iqtree_root is None else str(iqtree_root),
        "output_dir": str(output_dir),
        "download_enabled": args.download,
        "pgsui_version": pgsui_version,
        "snpio_version": snpio_version,
        "datasets_requested": len(datasets),
        "datasets_failed_or_unresolved": failures,
        "status_counts": {
            status: sum(row.get("validation_status") == status for row in rows)
            for status in sorted({str(row.get("validation_status")) for row in rows})
        },
    }
    (output_dir / "n60_vcf_source_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
