# Standard library imports
import logging
from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from snpio import GenotypeEncoder

# Local imports
from pgsui.utils.plotting import Plotting

# Type checking imports
if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


class ImputeAlleleFreq:
    """Frequency-based imputer for integer-encoded categorical genotype data with test-only evaluation.

    This implementation imputes missing values by sampling from the empirical frequency distribution of observed integer codes (e.g., 0-9 for nucleotides). It supports a strict train/test protocol: distributions are learned on the train split, missingness is simulated only on the test split, and all metrics are computed exclusively on the test split.

    The algorithm is as follows:
        1. Split the dataset into train and test sets (row-wise).
        2. For each feature (column), compute the empirical frequency distribution of observed values in the train set.
        3. On the test set, simulate additional missing values by randomly masking a specified proportion of observed entries.
        4. Impute missing values in the test set by sampling from the train-learned distributions.
        5. Evaluate imputation accuracy using various metrics on the test set only.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        prefix: str = "pgsui",
        by_populations: bool = False,
        default: int = 0,
        missing: int = -9,
        verbose: bool = True,
        seed: Optional[int] = None,
        sim_prop_missing: float = 0.30,
        debug: bool = False,
        test_size: float = 0.2,
        test_indices: Optional[Sequence[int]] = None,
        stratify_by_populations: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize ImputeAlleleFreq.

        Args:
            genotype_data: Object with `.genotypes_int`, `.ref`, `.alt`, and optional `.populations`.
            prefix: Output prefix.
            by_populations: Learn separate dists per population.
            default: Default genotype for cold-start loci.
            missing: Integer code for missing values in X_.
            verbose: Verbosity switch.
            seed: RNG seed.
            sim_prop_missing: Fraction of OBSERVED test cells to mask for evaluation.
            debug: Debug switch.
            test_size: Fraction of rows held out for test if `test_indices` not provided.
            test_indices: Explicit test row indices. Overrides `test_size` if given.
            stratify_by_populations: If True and populations are available, create a stratified test split per population.
        """
        self.genotype_data: "GenotypeData" = genotype_data
        self.prefix: str = prefix
        self.by_populations: bool = by_populations
        self.default: int = int(default)
        self.missing: int = int(missing)
        self.sim_prop_missing: float = float(sim_prop_missing)
        self.verbose: bool = verbose
        self.debug: bool = debug
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

        if not (0.0 <= self.sim_prop_missing <= 0.95):
            raise ValueError("sim_prop_missing must be in [0, 0.95].")

        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.encoder: GenotypeEncoder = GenotypeEncoder(self.genotype_data)
        self.X_: np.ndarray = np.asarray(self.encoder.genotypes_int, dtype=np.int8)
        self.num_features_: int = self.X_.shape[1]

        # --- split controls ---
        self.test_size: float = float(test_size)
        self.test_indices: np.ndarray | None = (
            None if test_indices is None else np.asarray(test_indices, dtype=int)
        )

        self.stratify_by_populations: bool = bool(stratify_by_populations)
        self.pops: np.ndarray | None = None
        if self.by_populations:
            pops = getattr(self.genotype_data, "populations", None)
            if pops is None:
                raise TypeError(
                    "by_populations=True requires genotype_data.populations."
                )
            self.pops = np.asarray(pops)
            if len(self.pops) != self.X_.shape[0]:
                raise ValueError(
                    f"`populations` length ({len(self.pops)}) != number of samples ({self.X_.shape[0]})."
                )

        self.is_fit_: bool = False
        self.global_dist_: Dict[int | str, Tuple[np.ndarray, np.ndarray]] = {}
        self.group_dist_: Dict[
            str | int, Dict[int | str, Tuple[np.ndarray, np.ndarray]]
        ] = {}
        self.sim_mask_: Optional[np.ndarray] = None
        self.train_idx_: Optional[np.ndarray] = None
        self.test_idx_: Optional[np.ndarray] = None
        self.X_train_: Optional[pd.DataFrame] = None
        self.ground_truths_: Optional[np.ndarray] = None
        self.metrics_: Dict[str, float] = {}
        self.X_imputed_: Optional[np.ndarray] = None

        # VCF ref/alt cache + IUPAC LUT
        self.ref_codes_: Optional[np.ndarray] = None  # (nF,) in {0..3}
        self.alt_mask_: Optional[np.ndarray] = None  # (nF,4) bool
        self._iupac_presence_lut_: Optional[np.ndarray] = None  # (10,4) bool

        plot_fmt: Literal["pdf", "png", "jpg", "jpeg"] = getattr(
            genotype_data, "plot_format", "png"
        )

        self.plotter = Plotting(
            "ImputeAlleleFreq",
            prefix=self.prefix,
            plot_format=plot_fmt,
            plot_fontsize=genotype_data.plot_fontsize,
            plot_dpi=genotype_data.plot_dpi,
            title_fontsize=genotype_data.plot_fontsize,
            despine=genotype_data.plot_despine,
            show_plots=genotype_data.show_plots,
            verbose=self.verbose,
            debug=self.debug,
        )

    # ------------------------------------------
    # Helpers for VCF ref/alt and IUPAC mapping
    # ------------------------------------------
    def _map_base_to_int(self, arr: np.ndarray | None) -> np.ndarray | None:
        """Map bases to A/C/G/T -> 0/1/2/3; pass integers through; others -> -1.

        Args:
            arr (np.ndarray | None): Array-like of bases (str) or integer codes.

        Returns:
            np.ndarray | None: Mapped integer array, or None if input is None or invalid.
        """
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.dtype.kind in ("i", "u"):
            return arr.astype(np.int32, copy=False)
        if arr.dtype.kind in ("U", "S", "O"):
            up = np.char.upper(arr.astype("U1"))
            out = np.full(up.shape, -1, dtype=np.int32)
            out[up == "A"] = 0
            out[up == "C"] = 1
            out[up == "G"] = 2
            out[up == "T"] = 3
            return out
        return None

    def _ref_codes_from_genotype_data(self) -> np.ndarray:
        """Fetch per-locus reference base from genotype_data.ref as 0..3.

        Returns:
            np.ndarray: Array of shape (n_features,) with values in {0,1,2,3}.
        """
        ref_raw = getattr(self.genotype_data, "ref", None)
        ref_codes = self._map_base_to_int(ref_raw)
        if ref_codes is None or ref_codes.shape[0] != self.num_features_:
            msg = (
                "genotype_data.ref missing or wrong length; "
                f"expected ({self.num_features_},) got "
                f"{None if ref_codes is None else ref_codes.shape}"
            )
            raise ValueError(msg)
        return ref_codes

    def _alt_mask_from_genotype_data(self) -> np.ndarray:
        """Build a per-locus mask of which bases are ALT (supports multi-alt).

        Returns:
            np.ndarray: Boolean array of shape (n_features, 4) indicating presence of A,C,G,T as ALT.
        """
        nF = self.num_features_
        alt_raw = getattr(self.genotype_data, "alt", None)
        alt_mask = np.zeros((nF, 4), dtype=bool)
        if alt_raw is None:
            return alt_mask

        alt_arr = np.asarray(alt_raw, dtype=object)
        if alt_arr.shape[0] != nF and self.verbose:
            print(
                f"[warn] genotype_data.alt length {alt_arr.shape[0]} != n_features {nF}; truncating."
            )

        def add_code(mask_row, x):
            if x is None:
                return
            if isinstance(x, (int, np.integer)):
                v = int(x)
                if 0 <= v <= 3:
                    mask_row[v] = True
                return
            if isinstance(x, str):
                s = x.strip().upper()
                if not s:
                    return
                if "," in s:
                    for token in s.split(","):
                        add_code(mask_row, token.strip())
                        return
                if s in ("A", "C", "G", "T"):
                    idx = {"A": 0, "C": 1, "G": 2, "T": 3}[s]
                    mask_row[idx] = True
                return
            if isinstance(x, (list, tuple, np.ndarray)):
                for t in x:
                    add_code(mask_row, t)
                return

        for i in range(min(nF, alt_arr.shape[0])):
            add_code(alt_mask[i], alt_arr[i])
        return alt_mask

    def _build_iupac_presence_lut(self) -> np.ndarray:
        """Create LUT mapping integer codes {0..9} -> allele presence over A,C,G,T.

        Returns:
            np.ndarray: Boolean array of shape (10,4) indicating presence of A,C,G,T for each IUPAC code.
        """
        lut = np.zeros((10, 4), dtype=bool)  # A,C,G,T
        lut[0, 0] = True
        lut[1, 3] = True
        lut[2, 2] = True
        lut[3, 1] = True  # A T G C
        lut[4, [0, 3]] = True
        lut[5, [0, 2]] = True
        lut[6, [0, 1]] = True  # W R M
        lut[7, [2, 3]] = True
        lut[8, [1, 3]] = True
        lut[9, [1, 2]] = True  # K Y S
        return lut

    # -----------------------
    # Fit / Transform
    # -----------------------
    def _make_train_test_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create row-wise train/test split according to init settings.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (train indices, test indices) as integer arrays.
        """
        n = self.X_.shape[0]
        all_idx = np.arange(n, dtype=int)

        if self.test_indices is not None:
            test_idx = np.unique(self.test_indices)
            if np.any((test_idx < 0) | (test_idx >= n)):
                raise IndexError("Some test_indices are out of bounds.")
            train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
            return train_idx, test_idx

        # Random split (optionally stratified by population)
        if (
            self.by_populations
            and self.stratify_by_populations
            and (self.pops is not None)
        ):
            test_buckets = []
            for pop in np.unique(self.pops):
                pop_rows = np.where(self.pops == pop)[0]
                k = int(round(self.test_size * pop_rows.size))
                if k > 0:
                    chosen = self.rng.choice(pop_rows, size=k, replace=False)
                    test_buckets.append(chosen)
            test_idx = (
                np.sort(np.concatenate(test_buckets))
                if test_buckets
                else np.array([], dtype=int)
            )
        else:
            k = int(round(self.test_size * n))
            test_idx = (
                self.rng.choice(n, size=k, replace=False)
                if k > 0
                else np.array([], dtype=int)
            )

        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
        return train_idx, test_idx

    def fit(self) -> "ImputeAlleleFreq":
        """Learn per-locus distributions on TRAIN rows; simulate missingness on TEST rows.

        Notes:
            The general workflow is:
            1) Split rows into train/test.
            2) Build distributions from TRAIN only.
            3) Simulate missingness on TEST only (stores `sim_mask_`).
            4) Cache ref/alt and IUPAC LUT.
        """
        # 0) Row split
        self.train_idx_, self.test_idx_ = self._make_train_test_split()

        self.ground_truths_ = self.X_.copy()
        df_all = pd.DataFrame(self.ground_truths_, dtype=np.float32)
        df_all.replace(self.missing, np.nan, inplace=True)

        # 1) TRAIN-only distributions (no simulated holes here)
        df_train = df_all.iloc[self.train_idx_].copy()

        self.global_dist_ = {
            col: self._series_distribution(df_train[col]) for col in df_train.columns
        }

        self.group_dist_.clear()
        if self.by_populations:
            tmp = df_train.copy()

            if self.pops is not None:
                tmp["_pops_"] = self.pops[self.train_idx_]
            for pop, grp in tmp.groupby("_pops_"):
                pop_key = str(pop)
                gdf = grp.drop(columns=["_pops_"])
                self.group_dist_[pop_key] = {
                    col: self._series_distribution(gdf[col]) for col in gdf.columns
                }

        # 2) TEST-only simulated missingness
        obs_mask = df_all.notna().to_numpy()
        sim_mask = np.zeros_like(obs_mask, dtype=bool)
        if self.test_idx_.size > 0 and self.sim_prop_missing > 0.0:
            # restrict candidate coords to TEST rows that are observed
            coords = np.argwhere(obs_mask)
            test_row_mask = np.zeros(obs_mask.shape[0], dtype=bool)
            test_row_mask[self.test_idx_] = True
            coords_test = coords[test_row_mask[coords[:, 0]]]
            total_obs_test = coords_test.shape[0]
            if total_obs_test > 0:
                n_to_mask = int(round(self.sim_prop_missing * total_obs_test))
                if n_to_mask > 0:
                    choice_idx = self.rng.choice(
                        total_obs_test, size=n_to_mask, replace=False
                    )
                    chosen_coords = coords_test[choice_idx]
                    sim_mask[chosen_coords[:, 0], chosen_coords[:, 1]] = True

        df_sim = df_all.copy()
        df_sim.values[sim_mask] = np.nan

        # Store matrix to be imputed (train rows intact; test rows with simulated NaNs)
        self.sim_mask_ = sim_mask
        self.X_train_ = df_sim

        # 3) Cache VCF ref/alt + IUPAC LUT once
        self.ref_codes_ = self._ref_codes_from_genotype_data()  # (nF,)
        self.alt_mask_ = self._alt_mask_from_genotype_data()  # (nF,4)
        self._iupac_presence_lut_ = self._build_iupac_presence_lut()

        self.is_fit_ = True
        if self.verbose:
            n_masked = int(sim_mask.sum())
            print(
                f"Fit complete. Train rows: {self.train_idx_.size}, Test rows: {self.test_idx_.size}."
            )
            print(
                f"Simulated {n_masked} missing values (TEST rows only) for evaluation."
            )
        return self

    def transform(self) -> np.ndarray:
        """Impute the matrix and evaluate on the TEST-set simulated cells.

        Returns:
            np.ndarray: Imputed genotypes in the original (IUPAC/int) encoding.
        """
        if not self.is_fit_:
            msg = "Model is not fitted. Call `fit()` before `transform()`."
            self.logger.error(msg)
            raise NotFittedError(msg)
        assert (
            self.X_train_ is not None
            and self.sim_mask_ is not None
            and self.ground_truths_ is not None
        )

        # ---- Impute using train-learned distributions ----
        # uses self.global_dist_/group_dist_
        imputed_df = self._impute_df(self.X_train_)
        X_imp = imputed_df.to_numpy(dtype=np.int8)
        self.X_imputed_ = X_imp

        # -------------------------------------------------------------------
        # Test-set evaluation on IUPAC/int codes (mask is test-only by design)
        # -------------------------------------------------------------------
        sim_mask = self.sim_mask_
        y_true = self.ground_truths_[sim_mask]
        y_pred = X_imp[sim_mask]

        if y_true.size > 0:
            labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
            self.metrics_ = {
                "n_masked_test": int(y_true.size),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float(
                    f1_score(
                        y_true, y_pred, average="macro", labels=labels, zero_division=0
                    )
                ),
                "precision": float(
                    precision_score(
                        y_true, y_pred, average="macro", labels=labels, zero_division=0
                    )
                ),
                "recall": float(
                    recall_score(
                        y_true, y_pred, average="macro", labels=labels, zero_division=0
                    )
                ),
            }
            if self.verbose:
                print("\n--- TEST-only Evaluation (IUPAC/int) ---")
                for k, v in self.metrics_.items():
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
                print("\nClassification Report (IUPAC/int, TEST-only):")
                print(
                    classification_report(
                        y_true, y_pred, labels=labels, zero_division=0
                    )
                )
        else:
            self.metrics_.update({"n_masked_test": 0})
            if self.verbose:
                print("No TEST cells were held out for evaluation (n_masked_test=0).")

        # Optional confusion matrix (IUPAC/int)
        labels_map = {
            "A": 0,
            "T": 1,
            "G": 2,
            "C": 3,
            "W": 4,
            "R": 5,
            "M": 6,
            "K": 7,
            "Y": 8,
            "S": 9,
            "N": -9,
        }
        self.plotter.plot_confusion_matrix(y_true, y_pred, label_names=labels_map)

        # ----------------------------------------------------------------
        # TEST-only Zygosity 0/1/2 evaluation using VCF ref/alt (hom-ref/het/hom-alt)
        # ----------------------------------------------------------------
        r_idx, f_idx = np.nonzero(sim_mask)
        if r_idx.size > 0:
            true_codes = self.ground_truths_[r_idx, f_idx].astype(np.int16, copy=False)
            pred_codes = X_imp[r_idx, f_idx].astype(np.int16, copy=False)

            keep_nm = (true_codes != self.missing) & (pred_codes != self.missing)
            if np.any(keep_nm):
                true_codes = true_codes[keep_nm]
                pred_codes = pred_codes[keep_nm]
                f_k = f_idx[keep_nm]

                if self.ref_codes_ is None or self.alt_mask_ is None:
                    raise RuntimeError(
                        "VCF ref/alt codes not cached; cannot compute zygosity metrics."
                    )

                ref_k = self.ref_codes_[f_k]  # (n,)
                alt_rows = self.alt_mask_[f_k, :]  # (n,4)
                ra_mask = alt_rows.copy()
                ra_mask[np.arange(ref_k.size), ref_k] = True

                lut = self._iupac_presence_lut_

                if lut is None:
                    raise RuntimeError(
                        "IUPAC presence LUT not cached; cannot compute zygosity metrics."
                    )
                valid_true = (true_codes >= 0) & (true_codes < lut.shape[0])
                valid_pred = (pred_codes >= 0) & (pred_codes < lut.shape[0])
                keep_valid = valid_true & valid_pred

                if np.any(keep_valid):
                    true_codes = true_codes[keep_valid]
                    pred_codes = pred_codes[keep_valid]
                    ref_k = ref_k[keep_valid]
                    ra_mask = ra_mask[keep_valid, :]

                    A_true = lut[true_codes]  # (n,4)
                    A_pred = lut[pred_codes]  # (n,4)

                    any_true = A_true.any(axis=1)
                    any_pred = A_pred.any(axis=1)
                    out_true = (A_true & ~ra_mask).any(axis=1)
                    out_pred = (A_pred & ~ra_mask).any(axis=1)
                    valid_rows = any_true & any_pred & (~out_true) & (~out_pred)

                    if np.any(valid_rows):
                        A_true = A_true[valid_rows]
                        A_pred = A_pred[valid_rows]
                        ref_kv = ref_k[valid_rows]
                        n = A_true.shape[0]
                        rows = np.arange(n, dtype=int)

                        cnt_true = A_true.sum(axis=1)
                        homref_true = (cnt_true == 1) & A_true[rows, ref_kv]
                        homalt_true = (cnt_true == 1) & (~A_true[rows, ref_kv])
                        y_true_3 = np.empty(n, dtype=np.int8)
                        y_true_3[homref_true] = 0
                        y_true_3[homalt_true] = 2
                        y_true_3[~(homref_true | homalt_true)] = 1

                        cnt_pred = A_pred.sum(axis=1)
                        homref_pred = (cnt_pred == 1) & A_pred[rows, ref_kv]
                        homalt_pred = (cnt_pred == 1) & (~A_pred[rows, ref_kv])
                        y_pred_3 = np.empty(n, dtype=np.int8)
                        y_pred_3[homref_pred] = 0
                        y_pred_3[homalt_pred] = 2
                        y_pred_3[~(homref_pred | homalt_pred)] = 1

                        labels_3 = [0, 1, 2]
                        self.metrics_.update(
                            {
                                "zyg_n_test": int(n),
                                "zyg_accuracy": float(
                                    accuracy_score(y_true_3, y_pred_3)
                                ),
                                "zyg_f1": float(
                                    f1_score(
                                        y_true_3,
                                        y_pred_3,
                                        average="macro",
                                        labels=labels_3,
                                        zero_division=0,
                                    )
                                ),
                                "zyg_precision": float(
                                    precision_score(
                                        y_true_3,
                                        y_pred_3,
                                        average="macro",
                                        labels=labels_3,
                                        zero_division=0,
                                    )
                                ),
                                "zyg_recall": float(
                                    recall_score(
                                        y_true_3,
                                        y_pred_3,
                                        average="macro",
                                        labels=labels_3,
                                        zero_division=0,
                                    )
                                ),
                            }
                        )
                        if self.verbose:
                            print(
                                "\n--- TEST-only Zygosity (0=hom-ref,1=het,2=hom-alt) ---"
                            )
                            for k in (
                                "zyg_n_test",
                                "zyg_accuracy",
                                "zyg_f1",
                                "zyg_precision",
                                "zyg_recall",
                            ):
                                v = self.metrics_[k]
                                print(
                                    f"  {k}: {v:.4f}"
                                    if isinstance(v, float)
                                    else f"  {k}: {v}"
                                )
                            print("\nClassification Report (zyg, TEST-only):")
                            print(
                                classification_report(
                                    y_true_3,
                                    y_pred_3,
                                    labels=labels_3,
                                    target_names=["hom-ref", "het", "hom-alt"],
                                    zero_division=0,
                                )
                            )
                        self.plotter.plot_confusion_matrix(
                            y_true_3,
                            y_pred_3,
                            label_names=["hom-ref", "het", "hom-alt"],
                        )
                    else:
                        if self.verbose:
                            print(
                                "[info] Zygosity TEST-only: no valid rows after RA filtering."
                            )
                else:
                    if self.verbose:
                        print(
                            "[info] Zygosity TEST-only: no valid rows after code filtering."
                        )
            else:
                if self.verbose:
                    print(
                        "[info] Zygosity TEST-only: nothing to score (all masked entries missing)."
                    )
        else:
            if self.verbose:
                print("[info] TEST-only evaluation: no masked coordinates found.")

        return self.encoder.inverse_int_iupac(X_imp)

    def fit_transform(self) -> np.ndarray:
        """Convenience method that calls `fit()` then `transform()`."""
        self.fit()
        return self.transform()

    # -----------------------
    # Core frequency model
    # -----------------------
    def _safe_probs(self, probs: np.ndarray) -> np.ndarray:
        """Ensure probs are non-negative and sum to 1; fallback to uniform if invalid.

        Args:
            probs: Array of non-negative values (not necessarily summing to 1).

        Returns:
            np.ndarray: Valid probability distribution summing to 1.
        """
        probs = np.asarray(probs, dtype=float)
        probs[probs < 0] = 0
        s = probs.sum()

        if not np.isfinite(s) or s <= 0:
            return np.full(probs.size, 1.0 / max(1, probs.size))
        return probs / s

    def _series_distribution(self, s: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Compute empirical (states, probs) for one locus from observed integer codes.

        Args:
            s: One column (locus) as a pandas Series with NaNs for missing.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (states, probs) sorted by state.
        """
        s_valid = s.dropna().astype(int)
        if s_valid.empty:
            return np.array([self.default], dtype=int), np.array([1.0])
        freqs = s_valid.value_counts(normalize=True).sort_index()
        states = freqs.index.to_numpy(dtype=int)
        probs = self._safe_probs(freqs.to_numpy(dtype=float))
        return states, probs

    def _impute_df(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Impute NaNs in df_in using precomputed TRAIN distributions.

        Args:
            df_in: DataFrame with NaNs to impute.

        Returns:
            DataFrame with NaNs imputed.
        """
        return (
            self._impute_global(df_in)
            if not self.by_populations
            else self._impute_by_population(df_in)
        )

    def _impute_global(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Impute dataframe globally, preserving original sample order.

        Args:
            df_in: DataFrame with NaNs to impute.

        Returns:
            DataFrame with NaNs imputed.
        """
        df = df_in.copy()
        for col in df.columns:
            if not df[col].isnull().any():
                continue
            states, probs = self.global_dist_[col]
            n_missing = int(df[col].isnull().sum())
            samples = self.rng.choice(states, size=n_missing, p=probs)
            df.loc[df[col].isnull(), col] = samples
        return df.astype(np.int8)

    def _impute_by_population(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Impute dataframe by population, preserving original sample order.

        Args:
            df_in: DataFrame with NaNs to impute.

        Returns:
            DataFrame with NaNs imputed.
        """
        df = df_in.copy()
        df["_pops_"] = getattr(self, "pops", None)
        for pop, grp in df.groupby("_pops_"):
            pop_key = str(pop)
            grp_imputed = grp.copy()
            per_pop_dist = self.group_dist_.get(pop_key, {})
            for col in grp.columns:
                if col == "_pops_":
                    continue
                if not grp[col].isnull().any():
                    continue
                states, probs = per_pop_dist.get(col, self.global_dist_[col])
                n_missing = int(grp[col].isnull().sum())
                samples = self.rng.choice(states, size=n_missing, p=probs)
                grp_imputed.loc[grp_imputed[col].isnull(), col] = samples
            df.update(grp_imputed)
        return df.drop(columns=["_pops_"]).astype(np.int8)
