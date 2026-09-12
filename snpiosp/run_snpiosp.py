#!/usr/bin/env python3
"""Run SNPio-backed population-genetic summaries for one genotype file."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

try:
    from .snpiosp import SUPPORTED_FORMATS, DnaSPSingleLocusAnalyzer
except ImportError:
    from snpiosp import SUPPORTED_FORMATS, DnaSPSingleLocusAnalyzer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--popmap", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "-f",
        "--format",
        choices=("infer", *sorted(SUPPORTED_FORMATS)),
        default="infer",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--ploidy",
        choices=("auto", "1", "2", "4"),
        default="auto",
        help="Infer VCF GT ploidy or require haploid, diploid, or tetraploid GTs.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def infer_format(path: Path) -> str:
    """Infer a supported input format from a compound filename suffix."""
    name = path.name.lower()
    suffixes = {
        ".vcf.gz": "vcf",
        ".vcf": "vcf",
        ".phy": "phy",
        ".phylip": "phylip",
        ".genepop": "genepop",
        ".structure": "structure",
        ".str": "str",
    }
    for suffix, file_format in suffixes.items():
        if name.endswith(suffix):
            return file_format
    raise ValueError(f"Cannot infer genotype format from {path.name!r}")


def json_ready(value: Any) -> Any:
    """Recursively convert NumPy and non-finite values to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def safe_population_filename(population: str) -> str:
    """Return a conservative output filename component."""
    safe = "".join(
        character if character.isalnum() else "_" for character in population
    )
    safe = safe.strip("_")
    return safe or "population"


def write_population_outputs(
    analyzer: DnaSPSingleLocusAnalyzer,
    population: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Write per-locus and summary outputs for one population."""
    locus = analyzer.analyze_population_per_locus(population)
    summary = analyzer.compute_overall_summary(population, locus)
    stem = "Total" if population == "Total" else safe_population_filename(population)
    locus.to_csv(output_dir / f"{stem}_LocusStats.csv", index=False)
    (output_dir / f"{stem}_Summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    """Run SNPio-backed statistics without overwriting an existing result tree."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    input_path = args.input.expanduser().resolve()
    output_dir = args.outdir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new path: {output_dir}"
        )
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1")
    file_format = infer_format(input_path) if args.format == "infer" else args.format
    output_dir.mkdir(parents=True)

    with DnaSPSingleLocusAnalyzer(
        input_path,
        popmap_file=args.popmap,
        file_format=file_format,
        n_jobs=args.n_jobs,
        ploidy=None if args.ploidy == "auto" else int(args.ploidy),
    ) as analyzer:
        summaries: dict[str, Any] = {}
        for population in analyzer.populations_to_analyze:
            summaries[population] = write_population_outputs(
                analyzer, population, output_dir
            )
        summaries["Total"] = write_population_outputs(analyzer, "Total", output_dir)
    (output_dir / "all_summaries.json").write_text(
        json.dumps(json_ready(summaries), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Analysis complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
