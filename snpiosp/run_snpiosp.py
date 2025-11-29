from __future__ import annotations

import logging
import warnings
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union
import itertools

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Configure logger (can be controlled by user application)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

try:
    from snpio import StructureReader  # Added StructureReader
    from snpio import GenePopReader, GenotypeEncoder, PhylipReader, VCFReader
except ImportError:
    # Fallback for testing/running without snpio installed if user is just checking logic
    # In production this raises the error as requested.
    pass


class DnaSPSingleLocusAnalyzer:
    """
    Replicates DnaSP v6 output treating EACH COLUMN (SNP) as a separate Locus.
    Includes Global/Non-Population based statistics (Ho, He, F, MAF, Missingness).
    """

    def __init__(
        self,
        filename: Union[str, Path],
        popmap_file: Optional[Union[str, Path]] = None,
        file_format: str = "vcf",
    ):
        """Initialize the analyzer.

        Args:
            filename: Path to genotype file.
            popmap_file: Optional path to population map file.
            file_format: Format of the genotype file.
        """
        self.filename = Path(filename)
        self.popmap_file = Path(popmap_file) if popmap_file else None
        self.file_format = file_format.lower()

        # Safe initialization without global side effects
        self._load_data(self.file_format)

    def _load_data(self, ft: str) -> None:
        """Load genotype data using SNPio readers."""
        logger.info(f"Loading data from {self.filename} using SNPio...")

        reader_mapper = {
            "vcf": VCFReader,
            "phylip": PhylipReader,
            "phy": PhylipReader,
            "genepop": GenePopReader,
            "structure": StructureReader,
            "str": StructureReader,
        }

        if ft not in reader_mapper:
            keys = list(reader_mapper.keys())
            msg = f"Format '{ft}' not supported. Use: {keys}"
            logger.error(msg)
            raise ValueError(msg)

        # SNPio readers generally accept filename and popmapfile as base kwargs
        kwargs = {"filename": self.filename, "popmapfile": self.popmap_file}

        try:
            # Suppress specific warnings just for the reader if necessary
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gd = reader_mapper[ft](**kwargs)
        except NameError:
            raise ImportError("SNPio is required. Run 'pip install snpio'.")
        except Exception as e:
            msg = f"Failed to load data: {e}"
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info("Encoding genotypes...")
        encoder = GenotypeEncoder(self.gd)
        self.genotype_matrix = self._clean_012_matrix(encoder.genotypes_012)

        self.samples = np.asarray(self.gd.samples)
        self.locus_names = self._get_locus_names()

        # Population Inference (Only if popmap missing, but stats work without it)
        if self.popmap_file is None:
            logger.info(
                "No popmap provided. Treating dataset as single pool for Total stats."
            )
            # We assign a default "Total" pop to everyone to ensure 'Total' logic works
            # The DBSCAN logic is available if specifically requested, but default is single pool
            # to respect the "not based on populations" preference.
            self.populations = pd.Series(
                ["Pop1"] * len(self.samples), index=self.samples
            )
        else:
            self.populations = pd.Series(self.gd.populations, index=self.samples)

        logger.info(
            f"Loaded: {len(self.samples)} samples, {self.genotype_matrix.shape[1]} loci."
        )

    def _get_locus_names(self) -> np.ndarray:
        """Generate locus names based on SNPio data or default naming."""
        n_snps = self.genotype_matrix.shape[1]
        gd_loci = getattr(self.gd, "locus_names", None)

        if gd_loci is not None and len(gd_loci) == n_snps:
            return np.asarray(gd_loci)

        return np.array([f"Locus_{i+1}" for i in range(n_snps)])

    @staticmethod
    def _clean_012_matrix(X: np.ndarray) -> np.ndarray:
        """Convert to float, set invalid to NaN."""
        Xc = X.astype(float, copy=True)
        # Optimized: Set anything not 0, 1, 2 to NaN in one go
        mask_invalid = (Xc != 0.0) & (Xc != 1.0) & (Xc != 2.0)
        Xc[mask_invalid] = np.nan
        return Xc

    # ---------------------------------------------------------------------
    # Analysis Logic
    # ---------------------------------------------------------------------
    def _get_pop_matrix(self, pop_id: str) -> np.ndarray:
        """Get genotype matrix for a specific population."""
        if pop_id == "Total":
            return self.genotype_matrix

        mask = (self.populations == pop_id).to_numpy()
        if not np.any(mask):
            return np.empty((0, self.genotype_matrix.shape[1]))
        return self.genotype_matrix[mask, :]

    def analyze_population_per_locus(self, pop_id: str) -> pd.DataFrame:
        matrix = self._get_pop_matrix(pop_id)
        if matrix.shape[0] == 0:
            return pd.DataFrame()

        n_total_samples = matrix.shape[0]

        # --- Vectorized Calculations ---
        is_nan = np.isnan(matrix)
        n_samples_per_locus = np.sum(~is_nan, axis=0)

        # 1. Missingness
        missing_rate = 1.0 - (n_samples_per_locus / n_total_samples)

        # Guard against division by zero for empty columns
        with np.errstate(divide="ignore", invalid="ignore"):
            n_chroms = 2.0 * n_samples_per_locus

            # Allele counts
            c0 = np.sum(matrix == 0.0, axis=0)
            c1 = np.sum(matrix == 1.0, axis=0)  # Heterozygotes
            c2 = np.sum(matrix == 2.0, axis=0)

            alt_counts = np.nansum(matrix, axis=0)  # Sum of (0*c0 + 1*c1 + 2*c2)
            p = alt_counts / n_chroms
            q = 1.0 - p

            # 2. MAF (Minor Allele Frequency)
            maf = np.minimum(p, q)

            # 3. Ho (Observed Heterozygosity)
            # Heterozygotes are coded as 1.0
            Ho = c1 / n_samples_per_locus

            # 4. He (Expected Heterozygosity) & Unbiased He
            He = 2.0 * p * q
            # Nei's correction: (2n / 2n-1) * He
            correction_he = n_chroms / (n_chroms - 1.0)
            correction_he[n_chroms <= 1] = 0.0
            He_unbiased = He * correction_he

            # 5. Global Fixation Index (F)
            # 1 - (Ho/He). If He is 0, F is undefined (NaN).
            F_stat = np.full_like(Ho, np.nan)
            valid_he = He_unbiased > 1e-9
            F_stat[valid_he] = 1.0 - (Ho[valid_he] / He_unbiased[valid_he])

            # 6. Singletons (SFS Tail)
            # Count how often the minor allele appears.
            # Unfolded: check alt_counts. Folded: check min(alt, ref).
            # Assuming folded is safer if ancestral state unknown.
            minor_counts = np.minimum(alt_counts, n_chroms - alt_counts)
            singletons = (minor_counts == 1).astype(int)

            # Pi (Nucleotide Diversity)
            Pi = (2.0 * p * q) * correction_he

        is_segregating = (p > 1e-9) & (p < (1.0 - 1e-9))
        S = is_segregating.astype(int)

        # --- Optimization: Vectorized Harmonic Numbers ---
        max_n = int(np.max(n_chroms)) if len(n_chroms) > 0 else 0

        # Precompute harmonic series up to max_n
        reciprocals = 1.0 / np.arange(1, max_n + 1)
        a1_lookup = np.concatenate(([0], np.cumsum(reciprocals)))

        n_chroms_int = n_chroms.astype(int)
        a1_vals = a1_lookup[np.clip(n_chroms_int - 1, 0, max_n)]

        ThetaWatt = np.zeros_like(Pi)
        valid_theta = (S == 1) & (a1_vals > 0)
        ThetaWatt[valid_theta] = 1.0 / a1_vals[valid_theta]

        # Tajima's D
        TajimaD = np.full(len(S), np.nan)
        calc_d = (S == 1) & (n_chroms_int >= 2)

        if np.any(calc_d):
            n_sub = n_chroms[calc_d].astype(int)
            a1_sub = a1_vals[calc_d]

            # a2: sum(1/i^2)
            reciprocals_sq = 1.0 / (np.arange(1, max_n + 1) ** 2)
            a2_lookup = np.concatenate(([0], np.cumsum(reciprocals_sq)))
            a2_sub = a2_lookup[np.clip(n_sub - 1, 0, max_n)]

            b1 = (n_sub + 1) / (3 * (n_sub - 1))
            b2 = (2 * (n_sub**2 + n_sub + 3)) / (9 * n_sub * (n_sub - 1))

            c1 = b1 - (1.0 / a1_sub)
            c2 = b2 - ((n_sub + 2) / (a1_sub * n_sub)) + (a2_sub / (a1_sub**2))

            e1 = c1 / a1_sub
            e2 = c2 / (a1_sub**2 + a2_sub)

            S_vec = S[calc_d]
            var_D = (e1 * S_vec) + (e2 * S_vec * (S_vec - 1))

            diff = Pi[calc_d] - ThetaWatt[calc_d]
            with np.errstate(invalid="ignore", divide="ignore"):
                d_vals = diff / np.sqrt(var_D)

            TajimaD[calc_d] = d_vals

        # Haplotype Diversity (Hd)
        Hap = (c0 > 0).astype(int) + (c1 > 0).astype(int) + (c2 > 0).astype(int)
        HetzPositions = c1

        with np.errstate(invalid="ignore", divide="ignore"):
            sum_sq_g = (
                (c0 / n_samples_per_locus) ** 2
                + (c1 / n_samples_per_locus) ** 2
                + (c2 / n_samples_per_locus) ** 2
            )
            Hd = (1.0 - sum_sq_g) * correction_he

        # Create DataFrame
        df = pd.DataFrame(
            {
                "Locus": self.locus_names,
                "Sample_Size": n_samples_per_locus,
                "Missingness": missing_rate,
                "SegSites": S,
                "Singletons": singletons,
                "MAF": maf,
                "Ho": Ho,
                "He": He_unbiased,
                "F_inbreeding": F_stat,
                "HetzPositions": HetzPositions,
                "Hap": Hap,
                "Hd": Hd,
                "Pi": Pi,
                "ThetaWatt": ThetaWatt,
                "TajimaD": TajimaD,
                "Filename": self.filename.name,
            }
        )

        df["TotalPos"] = 1
        return df

    def compute_overall_summary(
        self, pop_id: str, df_locus: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute overall summary statistics."""

        if df_locus.empty:
            return {}

        # Ensure numeric
        numeric_cols = [
            "Sample_Size",
            "SegSites",
            "Pi",
            "ThetaWatt",
            "TajimaD",
            "Hd",
            "Ho",
            "He",
            "F_inbreeding",
            "MAF",
            "Missingness",
        ]
        for col in numeric_cols:
            if col in df_locus.columns:
                df_locus[col] = pd.to_numeric(df_locus[col], errors="coerce")

        summary = {
            "Population": pop_id,
            "N_Loci": len(df_locus),
            # Basics
            "Mean_Sample_Size": float(df_locus["Sample_Size"].mean()),
            "Total_Segregating_Sites": int(df_locus["SegSites"].sum()),
            "Total_Singletons": int(df_locus["Singletons"].sum()),
            # Means of Stats
            "Mean_Pi": float(df_locus["Pi"].mean()),
            "Mean_ThetaWatt": float(df_locus["ThetaWatt"].mean()),
            "Mean_TajimaD": float(df_locus["TajimaD"].mean()),
            "Mean_Hd": float(df_locus["Hd"].mean()),
            "Mean_Ho": float(df_locus["Ho"].mean()),
            "Mean_He": float(df_locus["He"].mean()),
            "Mean_F": float(df_locus["F_inbreeding"].mean()),
            "Mean_MAF": float(df_locus["MAF"].mean()),
            "Mean_Missingness": float(df_locus["Missingness"].mean()),
            # Variances/Std
            "StdDev_Pi": float(df_locus["Pi"].std()),
            "StdDev_TajimaD": float(df_locus["TajimaD"].std()),
            "StdDev_F": float(df_locus["F_inbreeding"].std()),
        }

        # Kelly's ZnS (LD)
        MAX_LOCI_FOR_CORR = 5000  # Reduced for safety

        seg_mask = (df_locus["SegSites"] == 1).to_numpy()
        n_seg = np.sum(seg_mask)
        ZnS = np.nan

        if n_seg > 1:
            if n_seg > MAX_LOCI_FOR_CORR:
                logger.warning(
                    f"Skipping ZnS: Too many segregating sites ({n_seg}) for memory safety."
                )
            else:
                try:
                    matrix = self._get_pop_matrix(pop_id)
                    mat_seg = matrix[:, seg_mask]

                    col_means = np.nanmean(mat_seg, axis=0)
                    inds = np.where(np.isnan(mat_seg))
                    mat_seg[inds] = np.take(col_means, inds[1])

                    with np.errstate(invalid="ignore"):
                        corr_mat = np.corrcoef(mat_seg, rowvar=False)
                        r2_mat = corr_mat**2

                    indices = np.triu_indices_from(r2_mat, k=1)
                    r2_values = r2_mat[indices]

                    valid_r2 = r2_values[~np.isnan(r2_values)]
                    if len(valid_r2) > 0:
                        ZnS = float(np.mean(valid_r2))
                except Exception as e:
                    logger.error(f"ZnS Calculation failed: {e}")

        summary["ZnS_Kelly"] = ZnS
        return summary


def parse_args():
    parser = argparse.ArgumentParser(description="DnaSP Single-Locus (SNP) Analyzer")
    parser.add_argument(
        "--input", type=str, required=True, help="Input genotype file (VCF/Phylip)"
    )
    parser.add_argument(
        "--popmap", type=str, default=None, help="Popmap file (Optional)"
    )
    parser.add_argument(
        "--outdir", type=str, default="dnasp_locus_results", help="Output directory"
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("infer", "vcf", "phylip", "phy", "genepop", "structure", "str"),
        default="infer",
        help="Input file format. Default: infer from file extension",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir)

    if args.format == "infer":
        input_pth = Path(args.input)
        ext = input_pth.suffix.lower()

        if ext == ".gz":
            ext = "".join(input_pth.name.split(".")[-2:]).lower()

        format_map = {
            ".vcf": "vcf",
            ".vcf.gz": "vcf",
            ".phy": "phy",
            ".phylip": "phylip",
            ".genepop": "genepop",
            ".structure": "structure",
            ".str": "str",
        }

        if ext in format_map:
            args.format = format_map[ext]
        else:
            raise ValueError(
                f"Cannot infer format from extension '{ext}'. Please specify with --format."
            )

    print("Initializing Analyzer...")
    analyzer = DnaSPSingleLocusAnalyzer(
        args.input, popmap_file=args.popmap, file_format=args.format
    )

    all_summaries = {}

    # If popmap exists, analyze sub-populations
    unique_pops = [p for p in sorted(analyzer.populations.unique())]
    if len(unique_pops) > 1:  # Only do split if we actually have distinct groups
        print(f"\nProcessing {len(unique_pops)} populations...")
        for pop in unique_pops:
            print(f"  > Analyzing Population: {pop}")
            df = analyzer.analyze_population_per_locus(pop)
            outfile = os.path.join(args.outdir, f"{pop}_LocusStats.csv")
            df.to_csv(outfile, index=False)

            summary = analyzer.compute_overall_summary(pop, df)
            all_summaries[pop] = summary

            with open(os.path.join(args.outdir, f"{pop}_Summary.json"), "w") as f:
                json.dump(summary, f, indent=4)

    # Always analyze Total Dataset (this is the key for non-pop differentiation)
    print("\n  > Analyzing Total Dataset (Global Stats)...")
    df_total = analyzer.analyze_population_per_locus("Total")
    outfile_total = os.path.join(args.outdir, "Total_LocusStats.csv")
    df_total.to_csv(outfile_total, index=False)

    summary_total = analyzer.compute_overall_summary("Total", df_total)
    all_summaries["Total"] = summary_total

    # Save Total Summary JSON
    with open(os.path.join(args.outdir, "Total_Summary.json"), "w") as f:
        json.dump(summary_total, f, indent=4)

    # Master JSON
    with open(os.path.join(args.outdir, "all_summaries.json"), "w") as f:
        json.dump(all_summaries, f, indent=4)

    print(f"\nAnalysis Complete. Results in {args.outdir}")


if __name__ == "__main__":
    main()
