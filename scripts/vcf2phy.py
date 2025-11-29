import argparse
import gc
import shutil
import sys
from pathlib import Path

from snpio import VCFReader


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Convert VCF file to PHYLIP format.")
    parser.add_argument(
        "-v", "--vcf", type=Path, required=True, help="Input VCF file path."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for PHYLIP files.",
    )
    return parser.parse_args()


def get_output_path(vcf_path: Path, output_dir: Path) -> Path:
    """Constructs output path, handling .vcf and .vcf.gz extensions correctly.

    Args:
        vcf_path: Path to input VCF.
        output_dir: Path to output directory.

    Returns:
        Path object for the destination file with .phy extension.
    """
    # Handle .vcf.gz or .vcf.bgz by stripping all suffixes
    stem = vcf_path.name
    if stem.endswith(".gz"):
        stem = Path(stem).stem
    if stem.endswith(".vcf"):
        stem = Path(stem).stem

    return output_dir / f"{stem}.phy"


def main():
    args = parse_args()

    # Validation
    if not args.vcf.exists():
        raise FileNotFoundError(f"Input VCF file {args.vcf} does not exist.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = get_output_path(args.vcf, args.output_dir)

    print(f"Converting {args.vcf} to {output_file}...")

    try:
        # Cast path to string for safety with C-ext or strict string APIs
        reader = VCFReader(
            filename=str(args.vcf), chunk_size=5000, prefix="phylip_files"
        )

        if reader.num_snps > 500_000:
            print(
                f"Warning: VCF file contains {reader.num_snps} SNPs. Skipping conversion to avoid excessive memory usage.",
                file=sys.stderr,
            )
            return

        reader.write_phylip(output_file)

        print("Conversion complete.")

        print("Cleaning up files...")
        snpio_dir = Path(f"{reader.prefix}_output")
        shutil.rmtree(snpio_dir)

    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        gc.collect()


if __name__ == "__main__":
    main()
