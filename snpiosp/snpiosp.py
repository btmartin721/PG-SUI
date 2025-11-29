from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

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

try:
    from snpio import StructureReader  # Added StructureReader
    from snpio import GenePopReader, GenotypeEncoder, PhylipReader, VCFReader
except ImportError:
    raise ImportError("SNPio is required. Run 'pip install snpio'.")


class DnaSPSingleLocusAnalyzer:
    """Replicates DnaSP v6 output treating EACH COLUMN (SNP) as a separate Locus Now supports VCF, Phylip, GenePop, and Structure formats."""

    def __init__(
        self,
        filename: Union[str, Path],
        popmap_file: Optional[Union[str, Path]] = None,
        file_format: str = "vcf",
    ):
        """Initialize the analyzer.

        Args:
            filename: Path to genotype file (VCF, Phylip, GenePop, Structure).
            popmap_file: Optional path to population map file.
            file_format: Format of the genotype file ('vcf', 'phylip', 'genepop', 'structure').
        """
        self.filename = Path(filename)
        self.popmap_file = Path(popmap_file) if popmap_file else None
        self.file_format = file_format.lower()

        # Safe initialization without global side effects
        self._load_data(self.file_format)

    def _load_data(self, ft: str) -> None:
        """Load genotype data using SNPio readers.

        Args:
            ft: File format string.
        """
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
        except Exception as e:
            msg = f"Failed to load data: {e}"
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info("Encoding genotypes...")
        encoder = GenotypeEncoder(self.gd)
        self.genotype_matrix = self._clean_012_matrix(encoder.genotypes_012)

        self.samples = np.asarray(self.gd.samples)
        self.locus_names = self._get_locus_names()

        # Population Inference
        if self.popmap_file is None:
            logger.info("No popmap. Inferring populations via PCA+DBSCAN...")
            self.populations = self._infer_populations_dbscan()
        else:
            self.populations = pd.Series(self.gd.populations, index=self.samples)

        logger.info(
            f"Loaded: {len(self.samples)} samples, {self.genotype_matrix.shape[1]} loci."
        )
        logger.debug(f"Populations: {sorted(self.populations.unique())}")

    def _get_locus_names(self) -> np.ndarray:
        """Generate locus names based on SNPio data or default naming.

        Returns:
            Numpy array of locus names.
        """
        n_snps = self.genotype_matrix.shape[1]

        # specific check for numpy array or list presence
        gd_loci = getattr(self.gd, "locus_names", None)

        if gd_loci is not None and len(gd_loci) == n_snps:
            return np.asarray(gd_loci)

        return np.array([f"Locus_{i+1}" for i in range(n_snps)])

    @staticmethod
    def _clean_012_matrix(X: np.ndarray) -> np.ndarray:
        """Convert to float, set invalid to NaN.

        Args:
            X: Input genotype matrix (numpy array).

        Returns:
            Cleaned genotype matrix with invalid entries as NaN.
        """
        Xc = X.astype(float, copy=True)

        # Optimized: Set anything not 0, 1, 2 to NaN in one go
        mask_invalid = (Xc != 0.0) & (Xc != 1.0) & (Xc != 2.0)
        Xc[mask_invalid] = np.nan
        return Xc

    # ---------------------------------------------------------------------
    # Robust DBSCAN Logic
    # ---------------------------------------------------------------------
    @staticmethod
    def _score_dbscan_solution(
        X: np.ndarray,
        labels: np.ndarray,
        noise_cap: float = 0.7,
        w_clusters: float = 1.0,
        w_noise: float = 1.5,
        w_sil: float = 0.5,
    ) -> float:
        """Score DBSCAN clustering solution.

        Args:
            X: Input data matrix.
            labels: Cluster labels from DBSCAN.
            noise_cap: Maximum noise fraction before heavy penalty.
            w_clusters: Weight for number of clusters.
            w_noise: Weight for noise penalty.
            w_sil: Weight for silhouette score bonus.
        """
        # Optimized: np.unique returns counts directly
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Calculate noise fraction
        if -1 in unique_labels:
            noise_count = counts[unique_labels == -1][0]
            noise_frac = float(noise_count / len(labels))
            n_clusters = len(unique_labels) - 1
        else:
            noise_frac = 0.0
            n_clusters = len(unique_labels)

        if n_clusters == 0:
            return -np.inf

        sil_bonus = 0.0
        if n_clusters >= 2:
            mask = labels != -1
            # Safety check for minimum samples for silhouette
            if np.sum(mask) > n_clusters:
                try:
                    sil_bonus = float(silhouette_score(X[mask], labels[mask]))
                except ValueError:
                    pass  # Keep bonus at 0.0

        noise_pen = (
            noise_frac
            if noise_frac <= noise_cap
            else noise_cap + 3.0 * (noise_frac - noise_cap)
        )
        return (w_clusters * n_clusters) - (w_noise * noise_pen) + (w_sil * sil_bonus)

    def _estimate_dbscan_eps(
        self, X: np.ndarray, min_samples: int = 5, quantile_fallback: float = 0.9
    ) -> float:
        """Estimate initial DBSCAN eps using k-NN distances.

        Args:
            X: Input data matrix.
            min_samples: Minimum samples for DBSCAN.
            quantile_fallback: Fallback quantile for eps if knee detection fails.

        Returns:
            Estimated eps value.
        """
        n = X.shape[0]
        k = max(2, min(min_samples, n - 1))

        nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X)
        distances, _ = nn.kneighbors(X)
        kth_dist = np.sort(distances[:, -1])

        if np.allclose(kth_dist, kth_dist[0]):
            return float(np.quantile(kth_dist, quantile_fallback) + 1e-6)

        # Knee detection (Vectorized)
        y = (kth_dist - kth_dist.min()) / (kth_dist.max() - kth_dist.min() + 1e-9)
        x = np.linspace(0.0, 1.0, len(y))
        # Find point with max distance from diagonal
        knee_idx = np.argmax(y - x)
        eps = float(kth_dist[knee_idx])

        return (
            eps
            if (np.isfinite(eps) and eps > 0)
            else float(np.quantile(kth_dist, quantile_fallback))
        )

    def _tune_dbscan_params(
        self,
        X,
        eps0,
        min_samples_grid,
        eps_factors=(0.03, 2.5),
        n_eps=25,
        noise_cap=0.7,
    ) -> tuple[float, int]:
        """Tune DBSCAN eps and min_samples using grid search.

        Args:
            X: Input data matrix.
            eps0: Initial eps estimate.
            min_samples_grid: Grid of min_samples to try.
            eps_factors: Multiplicative factors for eps grid.
            n_eps: Number of eps values to try.
            noise_cap: Maximum noise fraction before heavy penalty.

        Returns:
            Tuple of best (eps, min_samples).
        """

        low = max(eps0 * eps_factors[0], 1e-6)
        high = max(eps0 * eps_factors[1], 1e-6 * 1.01)

        # Logspace ensures we sample smaller epsilons more densely
        eps_grid = np.geomspace(low, high, n_eps)

        best_score = -np.inf
        best_eps = eps0
        best_ms = min_samples_grid[0]

        for ms in min_samples_grid:
            for eps in eps_grid:
                labels = DBSCAN(eps=eps, min_samples=int(ms), n_jobs=-1).fit_predict(X)
                score = self._score_dbscan_solution(X, labels, noise_cap=noise_cap)
                if score > best_score:
                    best_score, best_eps, best_ms = score, eps, int(ms)

        return float(best_eps), int(best_ms)

    def _infer_populations_dbscan(self, n_pcs: int = 30) -> pd.Series:
        """Infer populations using PCA + DBSCAN.

        Args:
            n_pcs: Number of principal components to use.

        Returns:
            Pandas Series of inferred population labels indexed by sample names.
        """
        # Impute missing data with most frequent genotype
        imputer = SimpleImputer(strategy="most_frequent")
        X_imp = imputer.fit_transform(self.genotype_matrix)

        n_components = min(n_pcs, X_imp.shape[0] - 1, X_imp.shape[1])
        if n_components < 2:
            return pd.Series(["Inferred_Pop_1"] * len(self.samples), index=self.samples)

        # Use Pipeline for cleanliness
        pca_pipe = make_pipeline(PCA(n_components=n_components), StandardScaler())
        X_pca = pca_pipe.fit_transform(X_imp)

        min_samples_guess = int(np.clip(np.sqrt(X_pca.shape[0]) / 3, 2, 10))
        eps0 = self._estimate_dbscan_eps(X_pca, min_samples=min_samples_guess)
        eps, min_samples = self._tune_dbscan_params(X_pca, eps0, (2, 3, 4, 5, 6, 8, 10))

        logger.info(f"Tuned DBSCAN: eps={eps:.3f}, min_samples={min_samples}")

        labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X_pca)

        pop_labels = [
            "Inferred_Noise" if l == -1 else f"Inferred_Pop_{l + 1}" for l in labels
        ]
        return pd.Series(pop_labels, index=self.samples)

    # ---------------------------------------------------------------------
    # Analysis Logic
    # ---------------------------------------------------------------------
    def _get_pop_matrix(self, pop_id: str) -> np.ndarray:
        """Get genotype matrix for a specific population.

        Args:
            pop_id: Population identifier.

        Returns:
            Numpy array of genotype matrix for the population.
        """
        if pop_id == "Total":
            return self.genotype_matrix

        # boolean indexing is safer and faster than string comparison on index usually
        mask = (self.populations == pop_id).to_numpy()
        if not np.any(mask):
            return np.empty((0, self.genotype_matrix.shape[1]))
        return self.genotype_matrix[mask, :]

    def analyze_population_per_locus(self, pop_id: str) -> pd.DataFrame:
        matrix = self._get_pop_matrix(pop_id)
        if matrix.shape[0] == 0:
            return pd.DataFrame()

        # --- Vectorized Calculations ---
        is_nan = np.isnan(matrix)
        n_samples_per_locus = np.sum(~is_nan, axis=0)

        # Guard against division by zero for empty columns
        with np.errstate(divide="ignore", invalid="ignore"):
            n_chroms = 2.0 * n_samples_per_locus
            alt_counts = np.nansum(matrix, axis=0)
            p = alt_counts / n_chroms
            q = 1.0 - p

            # Pi
            correction = n_chroms / (n_chroms - 1.0)
            correction[n_chroms <= 1] = 0.0
            Pi = (2.0 * p * q) * correction

        is_segregating = (p > 1e-9) & (p < (1.0 - 1e-9))
        S = is_segregating.astype(int)
        Eta = S.copy()

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

            # 1. Calculate b1 and b2 (Tajima 1989, Eqs 35, 36)
            b1 = (n_sub + 1) / (3 * (n_sub - 1))
            b2 = (2 * (n_sub**2 + n_sub + 3)) / (9 * n_sub * (n_sub - 1))

            # 2. Calculate c1 and c2 (Tajima 1989, Eqs 33, 34)
            c1 = b1 - (1.0 / a1_sub)
            c2 = b2 - ((n_sub + 2) / (a1_sub * n_sub)) + (a2_sub / (a1_sub**2))

            # 3. Calculate e1 and e2 (Tajima 1989, Eqs 31, 32)
            e1 = c1 / a1_sub
            e2 = c2 / (a1_sub**2 + a2_sub)

            # 4. Calculate Variance (Tajima 1989, Eq 30)
            # NOTE: DnaSP Logic: Var(d) = e1*S + e2*S*(S-1)
            S_vec = S[calc_d]
            var_D = (e1 * S_vec) + (e2 * S_vec * (S_vec - 1))

            diff = Pi[calc_d] - ThetaWatt[calc_d]
            with np.errstate(invalid="ignore", divide="ignore"):
                d_vals = diff / np.sqrt(var_D)

            TajimaD[calc_d] = d_vals

        # Haplotype Diversity (Hd)
        c0 = np.sum(matrix == 0.0, axis=0)
        c1 = np.sum(matrix == 1.0, axis=0)
        c2 = np.sum(matrix == 2.0, axis=0)

        Hap = (c0 > 0).astype(int) + (c1 > 0).astype(int) + (c2 > 0).astype(int)
        HetzPositions = c1

        with np.errstate(invalid="ignore", divide="ignore"):
            sum_sq_g = (
                (c0 / n_samples_per_locus) ** 2
                + (c1 / n_samples_per_locus) ** 2
                + (c2 / n_samples_per_locus) ** 2
            )

            Hd_corr = n_samples_per_locus / (n_samples_per_locus - 1.0)
            Hd_corr[n_samples_per_locus <= 1] = 0.0
            Hd = (1.0 - sum_sq_g) * Hd_corr

        # Create DataFrame
        df = pd.DataFrame(
            {
                "Locus": self.locus_names,
                "Sample_Size": n_samples_per_locus,
                "SegSites": S,
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
        df["NetSegSites"] = 0
        return df

    def compute_overall_summary(
        self, pop_id: str, df_locus: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute overall summary statistics for a population.

        Args:
            pop_id: Population identifier.
            df_locus: DataFrame of per-locus statistics.

        Returns:
            Dictionary of overall summary statistics.
        """

        if df_locus.empty:
            return {}

        # Ensure numeric
        numeric_cols = ["Sample_Size", "SegSites", "Pi", "ThetaWatt", "TajimaD", "Hd"]
        for col in numeric_cols:
            if col in df_locus.columns:
                df_locus[col] = pd.to_numeric(df_locus[col], errors="coerce")

        summary = {
            "Population": pop_id,
            "Mean_Sample_Size": float(df_locus["Sample_Size"].mean()),
            "Var_Sample_Size": float(df_locus["Sample_Size"].var()),
            "Total_Segregating_Sites": int(df_locus["SegSites"].sum()),
            "Mean_Pi": float(df_locus["Pi"].mean()),
            "Mean_ThetaWatt": float(df_locus["ThetaWatt"].mean()),
            "Mean_TajimaD": float(df_locus["TajimaD"].mean()),
            "Mean_Hd": float(df_locus["Hd"].mean()),
            "Var_Pi": float(df_locus["Pi"].var()),
            "Var_ThetaWatt": float(df_locus["ThetaWatt"].var()),
            "Var_Hd": float(df_locus["Hd"].var()),
            "Var_TajimaD": float(df_locus["TajimaD"].var()),
            "StdDev_Pi": float(df_locus["Pi"].std()),
            "StdDev_ThetaWatt": float(df_locus["ThetaWatt"].std()),
            "StdDev_Hd": float(df_locus["Hd"].std()),
            "StdDev_TajimaD": float(df_locus["TajimaD"].std()),
        }

        # Kelly's ZnS
        MAX_LOCI_FOR_CORR = 10000

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
