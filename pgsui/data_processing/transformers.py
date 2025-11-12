# Standard library imports
import copy
import logging
from pathlib import Path
from typing import Literal, Tuple

# Third-party imports
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from pgsui.utils.misc import validate_input_type


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


class SimMissingTransformer(BaseEstimator, TransformerMixin):
    """Simulate missing data on genotypes encoded as 0/1/2 integers.

    This transformer is designed to work with genotype data that has been preprocessed into a suitable format. It simulates missing data according to various strategies, allowing for the testing and evaluation of imputation methods. The simulated missing data can be controlled in terms of proportion and distribution across samples and loci.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance.
        prop_missing (float, optional): Proportion of missing data desired in output. Must be in the interval [0, 1]. Defaults to 0.1
        strategy (Literal["nonrandom", "nonrandom_weighted", "random_weighted", "random_weighted_inv", "random"]): Strategy for simulating missing data. May be one of: "nonrandom", "nonrandom_weighted", "random_weighted", "random_weighted_inv", or "random". The "random" setting randomly replaces known genotypes as missing. "random_weighted". When set to "nonrandom", branches from GenotypeData.tree will be randomly sampled to generate missing data on descendant nodes. For "nonrandom_weighted", missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data as might be the case with mutation-disruption of RAD sites). Defaults to "random"
        missing_val (int, optional): Value that represents missing data. Defaults to -9.
        mask_missing (bool, optional): True if you want to skip original missing values when simulating new missing data, False otherwise. Defaults to True.
        verbose (bool, optional): Verbosity level. Defaults to 0.
        tol (float): Tolerance to reach proportion specified in self.prop_missing. Defaults to 1/num_snps*num_inds
        max_tries (int): Maximum number of tries to reach targeted missing data proportion within specified tol. If None, num_inds will be used. Defaults to None.

    Attributes:
        original_missing_mask_ (numpy.ndarray): Array with boolean mask for original missing locations.
        simulated_missing_mask_ (numpy.ndarray): Array with boolean mask for simulated missing locations, excluding the original ones.
        all_missing_mask_ (numpy.ndarray): Array with boolean mask for all missing locations, including both simulated and original.
    """

    def __init__(
        self,
        genotype_data,
        *,
        prop_missing=0.1,
        strategy="random",
        missing_val=-9,
        mask_missing=True,
        verbose=0,
        tol=None,
        max_tries=None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.genotype_data = genotype_data
        self.prop_missing = prop_missing
        self.strategy = strategy
        self.missing_val = missing_val
        self.mask_missing = mask_missing
        self.verbose = verbose
        self.tol = tol
        self.max_tries = max_tries
        self.logger = logger or logging.getLogger(__name__)

    def fit(self, X: np.ndarray, y=None) -> "SimMissingTransformer":
        """Fit to input data X by simulating missing data.

        Missing data will be simulated in varying ways depending on the ``strategy`` setting.

        Args:
            X (np.ndarray): Data with which to simulate missing data. It should have already been imputed with one of the non-machine learning simple imputers, and there should be no missing data present in X.

        Raises:
            TypeError: ``SimGenotypeDataTreeTransformer.tree`` must not be NoneType when using strategy="nonrandom" or "nonrandom_weighted".
            ValueError: Invalid ``strategy`` parameter provided.
        """
        X = validate_input_type(X, return_type="array").astype("float32")

        self.logger.info(
            f"Adding {self.prop_missing} missing data per column using strategy: {self.strategy}"
        )

        if not np.isnan(self.missing_val):
            X = X.copy()
            X[X == self.missing_val] = np.nan

        self.original_missing_mask_ = np.isnan(X)

        if self.strategy == "random":
            present = ~self.original_missing_mask_
            self.mask_ = np.zeros_like(X, dtype=bool)

            # sample only over present sites
            draws = np.random.random(X.shape)
            self.mask_[present] = draws[present] < self.prop_missing

            if self.mask_missing:
                # keep original-missing as not simulated
                pass
            else:
                # optionally also include original-missing as masked (no-op in
                # transform anyway)
                self.mask_[~present] = True

            self._validate_mask(use_non_original_only=True)

        elif self.strategy == "random_weighted":
            self.mask_ = self.random_weighted_missing_data(
                X, inv=False, target_rate=self.prop_missing
            )

        elif self.strategy == "random_weighted_inv":
            self.mask_ = self.random_weighted_missing_data(
                X, inv=True, target_rate=self.prop_missing
            )

        elif self.strategy.startswith("nonrandom"):
            if self.strategy not in {"nonrandom", "nonrandom_weighted"}:
                msg = f"strategy must be one of {{'nonrandom','nonrandom_weighted'}}, got: {self.strategy}"
                self.logger.error(msg)
                raise ValueError(msg)

            if self.genotype_data.tree is None:
                msg = "SimMissingTransformer.tree cannot be NoneType when strategy='nonrandom' or strategy='nonrandom_weighted'"
                self.logger.error(msg)
                raise TypeError(msg)

            rng = np.random.default_rng()
            skip_root = True
            weighted = self.strategy == "nonrandom_weighted"

            # working mask
            mask = np.zeros_like(X, dtype=bool)

            # eligible cells
            present = (
                ~self.original_missing_mask_
                if self.mask_missing
                else np.ones_like(mask, dtype=bool)
            )

            total_eligible = int(present.sum())
            if total_eligible == 0:
                self.mask_ = mask
                self._validate_mask(use_non_original_only=self.mask_missing)
                self.all_missing_mask_ = np.logical_or(
                    self.mask_, self.original_missing_mask_
                )
                self.sim_missing_mask_ = np.logical_and(
                    self.all_missing_mask_, ~self.original_missing_mask_
                )
                return self

            target = int(round(self.prop_missing * total_eligible))
            tol = int(
                max(
                    1,
                    (self.tol if self.tol is not None else 1.0 / mask.size)
                    * total_eligible,
                )
            )

            # map tip labels -> row indices
            name_to_idx = {name: i for i, name in enumerate(self.genotype_data.samples)}

            max_outer = (
                self.max_tries
                if self.max_tries is not None
                else max(10_000, mask.shape[0] * 10)
            )
            placed = int(mask.sum())
            best_delta = abs(placed - target)
            tries = 0

            # simple per-locus quota to distribute hits
            col_quota = np.full(
                mask.shape[1],
                max(1, int(np.ceil(target / max(1, mask.shape[1])))),
                dtype=int,
            )

            while tries < max_outer and abs(placed - target) > tol:
                tries += 1

                # >>> Call _sample_tree here <<<
                try:
                    tips = self._sample_tree(
                        internal_only=False,
                        tips_only=False,
                        skip_root=skip_root,
                        weighted=weighted,
                        rng=rng,
                    )
                except ValueError:
                    # no eligible nodes or no tips intersect samples; try again
                    continue

                # Convert to row indices; skip labels not in matrix
                rows = [name_to_idx[t] for t in tips if t in name_to_idx]
                if not rows:
                    continue

                # choose a column to edit
                cols_left = np.flatnonzero(col_quota > 0)
                if cols_left.size == 0:
                    cols_left = np.arange(mask.shape[1])
                j = int(rng.choice(cols_left))

                # only edit eligible cells in this column
                eligible_rows = np.fromiter(
                    (r for r in rows if present[r, j]), dtype=int
                )
                if eligible_rows.size == 0:
                    continue

                if placed < target:
                    prev_col = mask[:, j].copy()
                    mask[eligible_rows, j] = True

                    # avoid fully missing column among observed
                    col_after = mask[present[:, j], j]
                    if col_after.all():
                        idx_present = np.flatnonzero(present[:, j])
                        k = int(rng.choice(idx_present))
                        mask[k, j] = False

                    new_placed = int(mask.sum())
                    delta = abs(new_placed - target)
                    if delta <= best_delta:
                        best_delta = delta
                        placed = new_placed
                        col_quota[j] = max(0, col_quota[j] - 1)
                    else:
                        mask[:, j] = prev_col
                else:
                    # remove within the same clade and column
                    prev_col = mask[:, j].copy()
                    col_idxs = eligible_rows[mask[eligible_rows, j]]
                    if col_idxs.size == 0:
                        continue
                    need = min(col_idxs.size, max(1, placed - target))
                    to_clear = rng.choice(col_idxs, size=need, replace=False)
                    mask[to_clear, j] = False

                    new_placed = int(mask.sum())
                    delta = abs(new_placed - target)
                    if delta <= best_delta:
                        best_delta = delta
                        placed = new_placed
                    else:
                        mask[:, j] = prev_col

            self.mask_ = mask
            self._validate_mask(use_non_original_only=self.mask_missing)
        else:
            msg = f"Invalid SimMissingTransformer.strategy value: {self.strategy}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Get all missing values.
        self.all_missing_mask_ = np.logical_or(self.mask_, self.original_missing_mask_)

        # Get values where original value was not missing and simulated.
        # data is missing.
        self.sim_missing_mask_ = np.logical_and(
            self.all_missing_mask_, self.original_missing_mask_ == False
        )

        self._validate_mask(use_non_original_only=self.mask_missing)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Function to generate masked sites in a SimGenotypeData object

        Args:
            X (np.ndarray): Data to transform. No missing data should be present in X. It should have already been imputed with one of the non-machine learning simple imputers.

        Returns:
            np.ndarray: Transformed data with missing data added.
        """
        X = validate_input_type(X, return_type="array")

        # mask 012-encoded and one-hot encoded genotypes.
        return self._mask_snps(X)

    def sqrt_transform(self, proportions: np.ndarray) -> np.ndarray:
        """Apply the square root transformation to an array of proportions.

        Args:
            proportions (np.ndarray): An array of proportions.

        Returns:
            np.ndarray: The transformed proportions.
        """
        return np.sqrt(proportions)

    def random_weighted_missing_data(
        self,
        X: np.ndarray,
        transform_fn: Literal["sqrt", "exp"] = "sqrt",
        power: float = 0.5,
        inv: bool = False,
        rng: np.random.Generator | None = None,
        target_rate: float | None = None,  # if None, use realized draw
    ) -> np.ndarray:
        tf = transform_fn.lower()
        if tf not in {"sqrt", "exp"}:
            msg = f"transform_fn must be 'sqrt' or 'exp', got: {transform_fn}"
            self.logger.error(msg)
            raise ValueError(msg)

        rng = np.random.default_rng() if rng is None else rng
        eps = 1e-12

        def _tf(arr: np.ndarray) -> np.ndarray:
            arr = np.clip(arr, eps, None)
            return np.sqrt(arr) if tf == "sqrt" else np.exp(-arr)

        n_samples, n_snps = X.shape
        out_mask = np.zeros((n_samples, n_snps), dtype=bool)

        for j in range(n_snps):
            col = X[:, j]
            present = ~np.isnan(col)
            if not np.any(present):
                continue

            vals = col[present]
            classes, counts = np.unique(vals, return_counts=True)
            if classes.size == 1:  # never wipe entire column
                continue

            p = counts.astype(float) / counts.sum()
            base = 1.0 / np.clip(p, eps, None) if inv else p
            w = _tf(base)
            w = np.clip(w, 0.0, None) ** power
            s = w.sum()
            w = (
                np.full_like(w, 1.0 / w.size, dtype=float)
                if (s <= 0 or ~np.isfinite(s))
                else (w / s)
            )

            probs = np.zeros(n_samples, dtype=float)
            for c, pw in zip(classes, w):
                probs[present & (col == c)] = pw

            if target_rate is not None:
                probs *= float(target_rate)  # scale global intensity

            draws = rng.random(n_samples)
            out_mask[:, j] = draws < probs
            out_mask[~present, j] = False  # never alter already-missing

            # guard against accidentally wiping this column (using only non-original-missing)
            col_after = out_mask[present, j]
            if col_after.sum() == col_after.size:
                # clear a random observed index
                k = rng.integers(0, col_after.size)
                out_mask[np.flatnonzero(present)[k], j] = False

        return out_mask

    def _sample_tree(
        self,
        internal_only: bool = False,
        tips_only: bool = False,
        skip_root: bool = True,
        weighted: bool = False,
        rng: np.random.Generator | None = None,
    ) -> list[str]:
        """Sample a node and return descendant tip labels.

        Args:
            internal_only: Sample only internal nodes.
            tips_only: Sample only tip nodes.
            skip_root: Exclude the root from sampling.
            weighted: Weight node sampling by branch length.
            rng: Optional NumPy Generator for reproducibility.

        Returns:
            List[str]: Tip labels under the sampled node.

        Raises:
            ValueError: If no eligible nodes exist or both tips_only and internal_only are True.
        """
        if tips_only and internal_only:
            msg = "tips_only and internal_only cannot both be True"
            self.logger.error(msg)
            raise ValueError(msg)

        rng = np.random.default_rng() if rng is None else rng

        node_dict: dict[int | object, float] = {}

        # Traverse using the tree backend you have; be tolerant of API differences.
        for node in self.genotype_data.tree.treenode.traverse("preorder"):
            # Robust root detection: prefer is_root(), then fall back to parent None, finally fall back to idx==nnodes-1 only if needed.
            is_root = False
            if hasattr(node, "is_root"):
                is_root = bool(node.is_root())
            elif getattr(node, "up", None) is None:
                is_root = True
            elif hasattr(self.genotype_data.tree, "nnodes") and hasattr(node, "idx"):
                is_root = node.idx == self.genotype_data.tree.nnodes - 1

            if skip_root and is_root:
                continue

            if tips_only and not node.is_leaf():
                continue
            if internal_only and node.is_leaf():
                continue

            # Branch length; coerce invalid to 0
            dist = float(getattr(node, "dist", 0.0) or 0.0)
            if not np.isfinite(dist):
                dist = 0.0

            # Use node.idx if stable, else the node object as key
            key = getattr(node, "idx", node)
            node_dict[key] = dist

        if not node_dict:
            raise ValueError("No eligible nodes found to sample from the tree.")

        # Choose a node
        if weighted:
            w = np.asarray(list(node_dict.values()), dtype=float)
            w[~np.isfinite(w)] = 0.0
            s = w.sum()
            keys = list(node_dict.keys())
            if s <= 0.0:
                chosen_key = rng.choice(keys)
            else:
                p = w / s
                chosen_key = rng.choice(keys, p=p)
        else:
            chosen_key = rng.choice(list(node_dict.keys()))

        # Retrieve descendant tips for the chosen node
        if hasattr(self.genotype_data.tree, "get_tip_labels"):
            # If API expects idx, pass idx; if not, pass the key as needed.
            if isinstance(chosen_key, (int, np.integer)):
                tips = self.genotype_data.tree.get_tip_labels(idx=int(chosen_key))
            else:
                # Fallback if your API can accept a node object
                tips = self.genotype_data.tree.get_tip_labels(node=chosen_key)
        else:
            # Generic fallback: walk leaves from a node handle
            node = chosen_key if not isinstance(chosen_key, (int, np.integer)) else None
            if node is None:
                # If only idx is available, you need a resolver from idx -> node
                # Provide a resolver in your tree wrapper if this path can happen.
                raise ValueError(
                    "Tree API lacks get_tip_labels and node handle resolution by idx."
                )
            tips = [t.name for t in node.iter_leaves()]

        # Filter to sample IDs present in the matrix
        sample_set = set(self.genotype_data.samples)
        tips = [t for t in tips if t in sample_set]
        if not tips:
            # You may want to resample instead of erroring; here we error to avoid silent empties.
            raise ValueError(
                "Sampled clade contains no tips present in genotype_data.samples."
            )

        return tips

    def _validate_mask(self, use_non_original_only: bool = False) -> None:
        """Ensure no column is entirely masked on observed entries.

        Args:
            use_non_original_only (bool): If True, only consider non-original-missing entries when validating. Defaults to False.
        """
        m = self.mask_
        for j in range(m.shape[1]):
            if use_non_original_only:
                obs = ~self.original_missing_mask_[:, j]
            else:
                obs = np.ones(m.shape[0], dtype=bool)
            if not np.any(obs):
                continue
            col = m[obs, j]
            if col.size and col.all():
                # clear one random observed index
                idxs = np.flatnonzero(obs)
                k = np.random.randint(0, idxs.size)
                self.mask_[idxs[k], j] = False

    def _mask_snps(self, X):
        """Mask positions in SimGenotypeData.snps and SimGenotypeData.onehot"""
        if len(X.shape) == 3:
            # One-hot encoded.
            mask_val = [0.0, 0.0, 0.0, 0.0]
        elif len(X.shape) == 2:
            # 012-encoded.
            mask_val = -9
        else:
            raise ValueError(f"Invalid shape of input X: {X.shape}")

        Xt = X.copy()
        mask_boolean = self.mask_ != 0
        Xt[mask_boolean] = mask_val
        return Xt

    def write_mask(self, filename_prefix: str):
        """Write mask to file.

        Args:
            filename_prefix (str): Prefix for the filenames to write to.
        """
        np.save(filename_prefix + "_mask.npy", self.mask_)
        np.save(
            filename_prefix + "_original_missing_mask.npy",
            self.original_missing_mask_,
        )

    def read_mask(
        self, filename_prefix: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read mask from file.

        Args:
            filename_prefix (str): Prefix for the filenames to read from.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The read masks. (mask, original_missing_mask, all_missing_mask).
        """
        # Check if files exist
        if not Path(filename_prefix + "_mask.npy").is_file():
            msg = filename_prefix + "_mask.npy" + " does not exist."
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        if not Path(filename_prefix + "_original_missing_mask.npy").is_file():
            msg = filename_prefix + "_original_missing_mask.npy" + " does not exist."
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        # Load mask from file
        self.mask_ = np.load(filename_prefix + "_mask.npy")
        self.original_missing_mask_ = np.load(
            filename_prefix + "_original_missing_mask.npy"
        )

        # Recalculate all_missing_mask_ from mask_ and original_missing_mask_
        self.all_missing_mask_ = np.logical_or(self.mask_, self.original_missing_mask_)

        return self.mask_, self.original_missing_mask_, self.all_missing_mask_

    @property
    def missing_count(self) -> int:
        """Count of masked genotypes in SimGenotypeData.mask

        Returns:
            int: Integer count of masked alleles.
        """
        return np.sum(self.mask_)

    @property
    def prop_missing_real(self) -> float:
        """Proportion of genotypes masked in SimGenotypeData.mask

        Returns:
            float: Total number of masked alleles divided by SNP matrix size.
        """
        return np.sum(self.mask_) / self.mask_.size
