# Standard library imports
import logging
from typing import Literal

# Third-party imports
import numpy as np


class SimGenotypeDataTransformer:
    """Simulates missing genotypes at the locus level on a 2D integer matrix.

    This transformer masks a proportion of known genotypes in the input matrix X, setting them to a specified missing value. The masking can be done randomly or based on inverse genotype frequencies, with an option to boost the likelihood of masking heterozygous genotypes.

    Args:
        prop_missing (float): Proportion of *known* loci to mask (0..1).
        strategy (Literal): Strategy name.
        missing_val (int): Missing code value (default: -9).
        seed (int | None): RNG seed.
        logger (logging.Logger | None): Logger for messages.
        het_boost (float): Multiplier for heterozygotes in inv-genotype mode.
    """

    def __init__(
        self,
        *,
        prop_missing: float = 0.1,
        strategy: Literal["random", "random_inv_genotype"] = "random",
        missing_val: int = -1,
        seed: int | None = None,
        logger: logging.Logger | None = None,
        het_boost: float = 1.0,
    ):
        self.prop_missing = float(prop_missing)
        self.strategy = strategy
        self.missing_val = int(missing_val)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.het_boost = float(het_boost)
        self.logger = logger or logging.getLogger(__name__)

    def fit(self, X, y=None) -> "SimGenotypeDataTransformer":
        """Stateless.

        Args:
            X (np.ndarray): (n_samples, n_features), integer codes {0..9} or <0 as missing.
            y: Ignored.
        """
        return self

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """Apply missing-data simulation on a 2D genotype matrix.

        Args:
            X (np.ndarray): (n_samples, n_features), integer codes {0..9} or <0 as missing.

        Returns:
            tuple[np.ndarray, dict]: (X_masked, masks) where masks has keys: 'original': original missing (boolean 2D). 'simulated': loci masked here (boolean 2D). 'all': union of original + simulated (boolean 2D)
        """
        if X.ndim != 2:
            msg = f"X must be 2D, got shape {X.shape}"
            self.logger.error(msg)
            raise ValueError(msg)

        X = np.asarray(X)
        original_mask = X < 0

        sim_mask = self._simulate_missing_mask(X, original_mask)
        sim_mask = sim_mask & (~original_mask)
        sim_mask = self._validate_mask(sim_mask)

        all_mask = original_mask | sim_mask
        Xt = X.copy()
        Xt[all_mask] = self.missing_val

        masks = {"original": original_mask, "simulated": sim_mask, "all": all_mask}
        return Xt, masks

    # ---- strategies ----
    def _simulate_missing_mask(
        self, X: np.ndarray, original_mask: np.ndarray
    ) -> np.ndarray:
        """Simulate missingness mask based on the chosen strategy.

        Args:
            X (np.ndarray): Input genotype matrix.
            original_mask (np.ndarray): Boolean mask of original missing values.

        Returns:
            np.ndarray: Simulated missing mask.
        """
        if self.strategy == "random":
            return self._simulate_random(original_mask)
        elif self.strategy == "random_inv_genotype":
            return self._simulate_inv_genotype(X, original_mask)

        msg = "strategy must be one of {'random','random_inv_genotype'}"
        self.logger.error(msg)
        raise ValueError(msg)

    def _simulate_random(self, original_mask: np.ndarray) -> np.ndarray:
        rows, cols = np.where(~original_mask)
        n_known = len(rows)
        mask = np.zeros_like(original_mask, dtype=bool)

        if n_known == 0:
            return mask

        n_to_mask = int(np.floor(self.prop_missing * n_known))

        if n_to_mask <= 0:
            return mask

        idx = self.rng.choice(n_known, size=n_to_mask, replace=False)
        mask[rows[idx], cols[idx]] = True
        return mask

    def _simulate_inv_genotype(
        self, X: np.ndarray, original_mask: np.ndarray
    ) -> np.ndarray:
        """Simulate missingness mask inversely proportional to genotype frequencies.

        Args:
            X (np.ndarray): Input genotype matrix.
            original_mask (np.ndarray): Boolean mask of original missing values.

        Returns:
            np.ndarray: Simulated missing mask. 0..3: homozygous (0,1,2,3). 4..9: heterozygous (0/1,0/2,0/3,1/2,1/3,2/3).
        """

        rows, cols = np.where(~original_mask)
        n_known = len(rows)
        mask = np.zeros_like(original_mask, dtype=bool)
        if n_known == 0:
            return mask

        # Global genotype frequencies (0..9) from all known
        vals = X[~original_mask].astype(int)
        vals = vals[(vals >= 0) & (vals < 10)]
        if vals.size == 0:
            return self._simulate_random(original_mask)

        cnt = np.bincount(vals, minlength=10).astype(float)
        freqs = cnt / (cnt.sum() + 1e-12)

        # Candidate weights
        geno_known = X[rows, cols].astype(int)  # (n_known,)
        inv = 1.0 / (freqs[geno_known] + 1e-12)

        # Optional het boost (heterozygous codes are 4..9)
        if self.het_boost != 1.0:
            is_het = (geno_known >= 4) & (geno_known <= 9)
            inv = inv * np.where(is_het, self.het_boost, 1.0)

        n_to_mask = int(np.floor(self.prop_missing * n_known))
        if n_to_mask <= 0:
            return mask

        probs = inv / (inv.sum() + 1e-12)
        idx = self.rng.choice(n_known, size=n_to_mask, replace=False, p=probs)
        mask[rows[idx], cols[idx]] = True
        return mask

    def _validate_mask(self, mask: np.ndarray) -> np.ndarray:
        """Avoid fully-masked rows/columns.

        Args:
            mask (np.ndarray): Input boolean mask.

        Returns:
            np.ndarray: Validated mask.
        """
        rng = self.rng
        # columns
        full_cols = np.where(mask.all(axis=0))[0]
        for c in full_cols:
            r = int(rng.integers(0, mask.shape[0]))
            mask[r, c] = False
        # rows
        full_rows = np.where(mask.all(axis=1))[0]
        for r in full_rows:
            c = int(rng.integers(0, mask.shape[1]))
            mask[r, c] = False
        return mask
