import copy
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import scipy.linalg
import toytree as tt

from pgsui.utils.plotting import Plotting
from pgsui.utils.scorers import Scorer

if TYPE_CHECKING:
    from snpio.analysis.tree_parser import TreeParser
    from snpio.read_input.genotype_data import GenotypeData


class _DiploidAggregator:
    """Precompute mappings to lump ordered diploid states (16) into unordered genotypes (10).

    Notes:
        Genotype class order matches the SNPio allele → genotype encodings:
            - 0: AA, 1: AC, 2: AG, 3: AT, 4: CC, 5: CG, 6: CT, 7: GG, 8: GT, 9: T
            - Allele order is A(0), C(1), G(2), T(3).
    """

    def __init__(self) -> None:
        # Ordered pairs (i,j) lexicographic over {0,1,2,3}; 16 states
        self.ordered_pairs = [(i, j) for i in range(4) for j in range(4)]
        # Unordered genotype classes as (min,max) in the specified class order
        self.genotype_classes = (
            [(0, 0)]
            + [(0, 1), (0, 2), (0, 3)]
            + [(1, 1), (1, 2), (1, 3)]
            + [(2, 2), (2, 3)]
            + [(3, 3)]
        )  # 10 states

        # Map: class index -> list of ordered-state indices
        self.class_to_ordered: list[list[int]] = []
        for i, j in self.genotype_classes:
            members = []
            for k, (a, b) in enumerate(self.ordered_pairs):
                if i == j:  # homozygote
                    if a == i and b == j:
                        members.append(k)  # exactly one member
                else:  # heterozygote, both permutations
                    if (a == i and b == j) or (a == j and b == i):
                        members.append(k)
            self.class_to_ordered.append(members)

        # Build R (16x10) and C (10x16)
        self.R = np.zeros((16, 10), dtype=float)
        self.C = np.zeros((10, 16), dtype=float)
        for c, members in enumerate(self.class_to_ordered):
            m = float(len(members))
            for o in members:
                self.R[o, c] = 1.0 / m  # spread class prob equally to ordered members
                self.C[c, o] = 1.0  # sum ordered probs back to class

        # For allele-marginalization from 10-state genotype posterior to 4 alleles
        # p_allele[i] = P(ii) + 0.5 * sum_{j!=i} P(min(i,j), max(i,j))
        self.het_classes_by_allele: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
        self.hom_class_index = {0: 0, 1: 4, 2: 7, 3: 9}  # AA, CC, GG, TT indices
        for idx, (i, j) in enumerate(self.genotype_classes):
            if i != j:
                self.het_classes_by_allele[i].append(idx)
                self.het_classes_by_allele[j].append(idx)


class _QCache:
    """Cache P(t) for haploid (4) and optionally diploid (10) via Kronecker + lumping."""

    def __init__(
        self,
        q_df: pd.DataFrame,
        mode: str = "haploid",
        diploid_agg: "_DiploidAggregator | None" = None,
    ) -> None:
        """Precompute eigendecomposition of haploid Q.

        Args:
            q_df: 4x4 generator in A,C,G,T order.
            mode: "haploid" or "diploid".
            diploid_agg: required if mode="diploid".
        """
        m = np.asarray(q_df, dtype=float)
        evals, V = scipy.linalg.eig(m)
        Vinv = scipy.linalg.inv(V)
        self.evals = evals
        self.V = V
        self.Vinv = Vinv
        self.mode = mode
        self.agg = diploid_agg
        if self.mode == "diploid" and self.agg is None:
            raise ValueError("Diploid mode requires a _DiploidAggregator.")
        self._cache4: dict[float, np.ndarray] = {}
        self._cache10: dict[float, np.ndarray] = {}

    def _P4(self, s: float) -> np.ndarray:
        """Return P(t) for haploid (4 states)."""
        key = round(float(s), 12)
        if key in self._cache4:
            return self._cache4[key]
        expo = np.exp(self.evals * key)
        P4 = (self.V * expo) @ self.Vinv
        P4 = np.real_if_close(P4, tol=1e5)
        P4[P4 < 0.0] = 0.0
        P4 /= P4.sum(axis=1, keepdims=True).clip(min=1.0)
        self._cache4[key] = P4
        return P4

    def P(self, t: float, rate: float = 1.0) -> np.ndarray:
        """Return P(t) in the active mode."""
        s = float(rate) * float(t)
        if self.mode == "haploid":
            return self._P4(s)
        # diploid
        key = round(s, 12)
        if key in self._cache10:
            return self._cache10[key]
        P4 = self._P4(s)
        P16 = np.kron(P4, P4)  # independent alleles
        P10 = self.agg.C @ P16 @ self.agg.R  # lump to unordered genotypes
        P10 = np.maximum(P10, 0.0)
        P10 /= P10.sum(axis=1, keepdims=True).clip(min=1.0)
        self._cache10[key] = P10
        return P10


class ImputePhylo:
    """Imputes missing genotype data using a phylogenetic likelihood model.

    This imputer uses a continuous-time Markov chain (CTMC) model of sequence evolution to impute missing genotype data based on a provided phylogenetic tree. It supports both haploid and diploid data, with options for evaluating imputation accuracy through simulated missingness.

    Notes:
        - Haploid CTMC (4 states: A,C,G,T) [default].
        - Diploid CTMC (10 unordered genotype states: AA,...,TT), derived by independent allele evolution and lumping ordered pairs.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        tree_parser: "TreeParser",
        prefix: str,
        min_branch_length: float = 1e-10,
        *,
        haploid: bool = False,
        eval_missing_rate: float = 0.0,
        column_subset: Optional[List[int]] = None,
        save_plots: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        self.genotype_data = genotype_data
        self.tree_parser = tree_parser
        self.prefix = prefix
        self.min_branch_length = min_branch_length
        self.eval_missing_rate = eval_missing_rate
        self.column_subset = column_subset
        self.save_plots = save_plots
        self.logger = genotype_data.logger
        self.verbose = verbose
        self.debug = debug
        self.char_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        self.nuc_map = {v: k for k, v in self.char_map.items()}
        self.imputer_data_: Optional[Tuple] = None
        self.ground_truth_: Optional[Dict] = None
        self.scorer = Scorer(
            self.prefix, average="weighted", verbose=verbose, debug=debug
        )
        self.plotter = Plotting(
            "ImputePhylo",
            prefix=self.prefix,
            plot_format=self.genotype_data.plot_format,
            plot_fontsize=self.genotype_data.plot_fontsize,
            plot_dpi=self.genotype_data.plot_dpi,
            title_fontsize=self.genotype_data.plot_fontsize,
            verbose=verbose,
            debug=debug,
        )

        self._MISSING_TOKENS = {"N", "n", "-9", ".", "?", "./.", ""}

        self.haploid = haploid
        self._dip_agg: _DiploidAggregator | None = None

        self.imputed_likelihoods_: Dict[Tuple[str, int], np.ndarray] = {}
        self.imputed_genotype_likelihoods_: Dict[Tuple[str, int], np.ndarray] = {}
        self.evaluation_results_: dict | None = None

    def fit(self) -> "ImputePhylo":
        """Prepares the imputer by parsing and validating input data.

        This method does the following:
            - Validates the genotype data and phylogenetic tree.
            - Extracts the genotype matrix, pruned tree, Q-matrix, and site rates.
            - Sets up internal structures for imputation.
        """
        self.imputer_data_ = self._parse_arguments()
        return self

    def transform(self) -> np.ndarray:
        """Transforms the data by imputing missing values.

        This method does the following:
            - Uses the fitted imputer to perform phylogenetic imputation.
            - Returns the imputed genotype DataFrame.
        """
        if self.imputer_data_ is None:
            msg = "The imputer has not been fitted. Call 'fit' first."
            self.logger.error(msg)
            raise RuntimeError(msg)

        original_genotypes, tree, q_matrix_in, site_rates = self.imputer_data_
        q_matrix = self._repair_and_scale_q(q_matrix_in, target_mean_rate=1.0)

        if not self.haploid:
            if self._dip_agg is None:
                self._dip_agg = _DiploidAggregator()
            self._qcache = _QCache(q_matrix, mode="diploid", diploid_agg=self._dip_agg)
        else:
            self._qcache = _QCache(q_matrix, mode="haploid")

        genotypes_to_impute = original_genotypes

        if self.eval_missing_rate > 0:
            genotypes_to_impute = self._simulate_missing_data(original_genotypes)

        imputed_df = self.impute_phylo(genotypes_to_impute, tree, q_matrix, site_rates)

        if self.ground_truth_:
            self._evaluate_imputation(imputed_df)
        return imputed_df  # keep as DataFrame with sample index

    def fit_transform(self) -> pd.DataFrame:
        """Fits the imputer and transforms the data in one step.

        This method does the following:
            - Calls the `fit` method to prepare the imputer.
            - Calls the `transform` method to perform imputation.
            - Returns the imputed genotype DataFrame.
        """
        self.fit()
        return self.transform()

    def _simulate_missing_data(
        self, original_genotypes: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Masks a fraction of known genotypes for evaluation."""
        genotypes_to_impute = copy.deepcopy(original_genotypes)
        known_positions = [
            (sample, site_idx)
            for sample, seq in genotypes_to_impute.items()
            for site_idx, base in enumerate(seq)
            if self._is_known(base)
        ]

        if not known_positions:
            self.logger.warning("No known values to mask for evaluation.")
            return genotypes_to_impute

        num_to_mask = int(len(known_positions) * self.eval_missing_rate)
        if num_to_mask == 0:
            self.logger.warning(f"eval_missing_rate is too low to mask any values.")
            return genotypes_to_impute

        # Sample the (sample, site_idx) tuples to be masked
        positions_to_mask = random.sample(known_positions, num_to_mask)

        # Correctly build the ground_truth dictionary
        self.ground_truth_ = {}
        for sample, site_idx in positions_to_mask:
            # Store the original base before masking it
            self.ground_truth_[(sample, site_idx)] = genotypes_to_impute[sample][
                site_idx
            ]
            # Now, apply the mask
            genotypes_to_impute[sample][site_idx] = "N"

        self.logger.info(f"Masked {len(self.ground_truth_)} values for evaluation.")
        return genotypes_to_impute

    def _evaluate_imputation(self, imputed_df: pd.DataFrame) -> None:
        """Evaluate imputation with 4 classes (haploid) or 10/16 classes (diploid).

        Behavior:
            - If self.haploid is True:
                Evaluate per-allele with 4 classes (A,C,G,T), keeping your existing logic.
            - If self.haploid is False:
                Evaluate per-genotype.
                * Default: 10 unordered classes (AA,AC,AG,AT,CC,CG,CT,GG,GT,TT).
                * If getattr(self, 'diploid_ordered', False) is True:
                    Also compute 16-class ordered metrics (AA,AC,...,TT).
                Requires self._dip_agg (the _DiploidAggregator). If absent, it is created.

        Notes:
            - Truth handling: only unambiguous IUPAC that maps to exactly one genotype class
            is used for 10-class evaluation (e.g., A, C, G, T, M, R, W, S, Y, K).
            Ambiguous codes (N, B, D, H, V, '-', etc.) are skipped for strict genotype
            evaluation.
            - Probabilities: prefer stored 10-class posteriors in
            self.imputed_genotype_likelihoods_; if missing, fall back by converting the
            4-class allele posterior to 10-class using HW independence.
        """
        if not self.ground_truth_:
            return

        # -------------------------
        # Haploid path (4 classes)
        # -------------------------
        if getattr(self, "haploid", True):
            # --- Gather ground truth bases and posterior vectors (only if both exist)
            rows = []
            for (sample, site_idx), true_base in self.ground_truth_.items():
                if (
                    sample in imputed_df.index
                    and (sample, site_idx) in self.imputed_likelihoods_
                ):
                    rows.append(
                        (true_base, self.imputed_likelihoods_[(sample, site_idx)])
                    )
            if not rows:
                self.logger.warning("No matching masked sites found to evaluate.")
                return

            true_bases = np.array([b for (b, _) in rows], dtype=object)
            proba_mat = np.vstack([p for (_, p) in rows])  # (n,4) in A,C,G,T
            nuc_to_idx = self.char_map  # {'A':0,'C':1,'G':2,'T':3}

            y_true_site = np.array(
                [nuc_to_idx.get(str(b).upper(), -1) for b in true_bases], dtype=int
            )
            valid_mask = (y_true_site >= 0) & np.isfinite(proba_mat).all(axis=1)
            if not np.any(valid_mask):
                self.logger.error(
                    "Evaluation arrays empty after filtering invalid entries."
                )
                return

            y_true_site = y_true_site[valid_mask]
            proba_mat = proba_mat[valid_mask]

            # Interleave each site as two alleles (iid/HW assumption) for per-allele metrics
            y_true_int = np.repeat(y_true_site, 2)
            y_pred_proba = np.repeat(proba_mat, 2, axis=0)
            y_pred_int = np.argmax(y_pred_proba, axis=1)

            n_classes = 4
            y_true_ohe = np.eye(n_classes, dtype=float)[y_true_int]
            idx_to_nuc = {v: k for k, v in nuc_to_idx.items()}

            self.logger.info("--- Per-Allele Imputation Performance (4-class) ---")
            self.evaluation_results_ = self.scorer.evaluate(
                y_true=y_true_int,
                y_pred=y_pred_int,
                y_true_ohe=y_true_ohe,
                y_pred_proba=y_pred_proba,
            )
            self.logger.info(f"Evaluation results: {self.evaluation_results_}")

            # Plots
            self.plotter.plot_metrics(
                y_true=y_true_int,
                y_pred_proba=y_pred_proba,
                metrics=self.evaluation_results_,
            )
            self.plotter.plot_confusion_matrix(
                y_true_1d=y_true_int,
                y_pred_1d=y_pred_int,
                label_names=["A", "C", "G", "T"],
            )
            return

        # -------------------------
        # Diploid path (10/16 classes)
        # -------------------------
        # Ensure aggregator
        if getattr(self, "_dip_agg", None) is None:
            self._dip_agg = _DiploidAggregator()

        # Label catalogs
        geno_classes = self._dip_agg.genotype_classes  # list of (i,j) with i<=j
        alleles = ["A", "C", "G", "T"]
        labels10 = [f"{alleles[i]}{alleles[j]}" for (i, j) in geno_classes]  # 10 labels
        ordered_pairs = [(i, j) for i in range(4) for j in range(4)]
        labels16 = [
            f"{alleles[i]}{alleles[j]}" for (i, j) in ordered_pairs
        ]  # 16 labels

        # Helper: strict IUPAC→single 10-class index (skip ambiguous >2-allele codes)
        iupac_to_10_single = {
            "A": "AA",
            "C": "CC",
            "G": "GG",
            "T": "TT",
            "M": "AC",
            "R": "AG",
            "W": "AT",
            "S": "CG",
            "Y": "CT",
            "K": "GT",
        }
        label_to_10_idx = {lab: idx for idx, lab in enumerate(labels10)}

        rows_10: list[tuple[int, np.ndarray]] = []  # (true_idx10, p10)
        rows_16_truth: list[int] = []  # expanded true labels (for 16)
        rows_16_proba: list[np.ndarray] = []  # matching (p16) rows

        # Collect rows
        for (sample, site_idx), true_code in self.ground_truth_.items():
            # Need probabilities for this masked (sample,site)
            p10 = None
            if (sample, site_idx) in self.imputed_genotype_likelihoods_:
                p10 = np.asarray(
                    self.imputed_genotype_likelihoods_[(sample, site_idx)], float
                )
            elif (sample, site_idx) in self.imputed_likelihoods_:
                # Fallback: build 10-class from 4-class posterior (HW independence)
                p4 = np.asarray(self.imputed_likelihoods_[(sample, site_idx)], float)
                p4 = np.clip(p4, 0.0, np.inf)
                p4 = p4 / (p4.sum() or 1.0)
                # Unordered genotype probs: AA=pA^2, AC=2pApC, ...
                p10 = np.zeros(10, dtype=float)
                # class indices in our order:
                # 0:AA, 1:AC, 2:AG, 3:AT, 4:CC, 5:CG, 6:CT, 7:GG, 8:GT, 9:TT
                pA, pC, pG, pT = p4
                p10[0] = pA * pA
                p10[1] = 2 * pA * pC
                p10[2] = 2 * pA * pG
                p10[3] = 2 * pA * pT
                p10[4] = pC * pC
                p10[5] = 2 * pC * pG
                p10[6] = 2 * pC * pT
                p10[7] = pG * pG
                p10[8] = 2 * pG * pT
                p10[9] = pT * pT
                p10 = p10 / (p10.sum() or 1.0)
            else:
                continue  # no probabilities available

            # Truth mapping to a single unordered class (skip ambiguous >2-allele codes)
            c = str(true_code).upper()
            lab10 = iupac_to_10_single.get(c, None)
            if lab10 is None:
                # Skip N, B, D, H, V, '-', etc., because they expand to multiple classes
                continue
            true_idx10 = label_to_10_idx[lab10]
            rows_10.append((true_idx10, p10))

            # Optional ordered-16 expansion (for confusion/metrics if requested)
            if getattr(self, "diploid_ordered", False):
                # Spread 10->16 via R (16x10)
                p16 = self._dip_agg.R @ p10
                p16 = p16 / (p16.sum() or 1.0)
                i, j = geno_classes[true_idx10]
                if i == j:
                    # Homozygote: single ordered index
                    true16 = ordered_pairs.index((i, j))
                    rows_16_truth.append(true16)
                    rows_16_proba.append(p16)
                else:
                    # Heterozygote: duplicate as two ordered permutations
                    true16a = ordered_pairs.index((i, j))
                    true16b = ordered_pairs.index((j, i))
                    rows_16_truth.append(true16a)
                    rows_16_proba.append(p16)
                    rows_16_truth.append(true16b)
                    rows_16_proba.append(p16)

        if not rows_10:
            self.logger.warning("No valid diploid truth rows for evaluation.")
            return

        # ---- 10-class metrics ----
        y_true_10 = np.array([t for (t, _) in rows_10], dtype=int)
        y_pred_proba_10 = np.vstack([p for (_, p) in rows_10])  # (n,10)

        y_pred_10 = np.argmax(y_pred_proba_10, axis=1)
        y_true_ohe_10 = np.eye(10, dtype=float)[y_true_10]

        self.logger.info("--- Per-Genotype Imputation Performance (10-class) ---")
        self.evaluation_results_ = self.scorer.evaluate(
            y_true=y_true_10,
            y_pred=y_pred_10,
            y_true_ohe=y_true_ohe_10,
            y_pred_proba=y_pred_proba_10,
        )
        self.logger.info(f"Evaluation results (10-class): {self.evaluation_results_}")

        # Plots (10-class)
        self.plotter.plot_metrics(
            y_true=y_true_10,
            y_pred_proba=y_pred_proba_10,
            metrics=self.evaluation_results_,
            label_names=labels10,
        )
        self.plotter.plot_confusion_matrix(
            y_true_1d=y_true_10, y_pred_1d=y_pred_10, label_names=labels10
        )

        # ---- Optional 16-class ordered metrics ----
        if getattr(self, "diploid_ordered", False) and rows_16_truth:
            y_true_16 = np.array(rows_16_truth, dtype=int)
            y_pred_proba_16 = np.vstack(rows_16_proba)
            y_pred_16 = np.argmax(y_pred_proba_16, axis=1)
            y_true_ohe_16 = np.eye(16, dtype=float)[y_true_16]

            self.logger.info(
                "--- Per-Genotype Imputation Performance (16-class ordered) ---"
            )
            eval16 = self.scorer.evaluate(
                y_true=y_true_16,
                y_pred=y_pred_16,
                y_true_ohe=y_true_ohe_16,
                y_pred_proba=y_pred_proba_16,
            )
            self.logger.info(f"Evaluation results (16-class): {eval16}")

            self.plotter.plot_metrics(
                y_true=y_true_16,
                y_pred_proba=y_pred_proba_16,
                metrics=eval16,
                label_names=labels16,
            )
            self.plotter.plot_confusion_matrix(
                y_true_1d=y_true_16, y_pred_1d=y_pred_16, label_names=labels16
            )

    def _infer_proba_permutation(
        self, y_ref: np.ndarray, P: np.ndarray, n_classes: int = 4
    ) -> tuple[np.ndarray, list[int]]:
        """Infers a permutation to align probability columns to the label space of y_ref."""
        perm = [-1] * n_classes
        taken = set()
        for k in range(n_classes):
            mask = y_ref == k
            if not np.any(mask):
                means = P.mean(axis=0)
            else:
                means = P[mask].mean(axis=0)

            # Find best unused column
            for col in np.argsort(means)[::-1]:
                if col not in taken:
                    perm[k] = col
                    taken.add(col)
                    break

        if len(taken) != n_classes:  # Fallback if permutation is incomplete
            unassigned_cols = [c for c in range(n_classes) if c not in taken]
            for i in range(n_classes):
                if perm[i] == -1:
                    perm[i] = unassigned_cols.pop(0)

        self.logger.info(f"Inferred probability permutation (label->col): {perm}")
        return P[:, perm], perm

    def _stationary_pi(self, Q: np.ndarray) -> np.ndarray:
        """Robustly calculates the stationary distribution of Q."""
        w, v = scipy.linalg.eig(Q.T)
        k = int(np.argmin(np.abs(w)))
        pi = np.real(v[:, k])
        pi = np.maximum(pi, 0.0)
        s = pi.sum()
        return (pi / s) if s > 0 else np.ones(Q.shape[0]) / Q.shape[0]

    def impute_phylo(
        self,
        genotypes: Dict[str, List[Union[str, int]]],
        tree: tt.tree,
        q_matrix: pd.DataFrame,
        site_rates: Optional[List[float]],
    ) -> pd.DataFrame:
        """Imputes missing values using a phylogenetic guide tree."""
        self.imputed_likelihoods_.clear()
        self.imputed_genotype_likelihoods_.clear()

        common_samples = set(tree.get_tip_labels()) & set(genotypes.keys())
        if not common_samples:
            raise ValueError("No samples in common between tree and genotypes.")
        filt_genotypes = copy.deepcopy({s: genotypes[s] for s in common_samples})
        num_snps = len(next(iter(filt_genotypes.values())))

        if site_rates is not None and len(site_rates) != num_snps:
            raise ValueError(
                f"len(site_rates)={len(site_rates)} != num_snps={num_snps}"
            )

        # Stationary prior at root
        if not self.haploid:
            # pi4 from Q; then lump pi4⊗pi4 to 10 classes
            pi4 = self._stationary_pi(q_matrix.to_numpy())
            pi_ord = np.kron(pi4, pi4)  # 16
            pi10 = self._dip_agg.C @ pi_ord
            pi10 = pi10 / (pi10.sum() or 1.0)
            root_prior = pi10
            n_states = 10
        else:
            root_prior = self._stationary_pi(q_matrix.to_numpy())  # 4
            n_states = 4

        for snp_index in range(num_snps):
            rate = site_rates[snp_index] if site_rates is not None else 1.0
            tips_with_missing = [
                s
                for s, seq in filt_genotypes.items()
                if self._is_missing(seq[snp_index])
            ]
            if not tips_with_missing:
                continue

            down_liks: Dict[int, np.ndarray] = {}
            for node in tree.treenode.traverse("postorder"):
                lik = np.zeros(n_states, dtype=float)
                if node.is_leaf():
                    if node.name not in filt_genotypes or self._is_missing(
                        filt_genotypes[node.name][snp_index]
                    ):
                        lik[:] = 1.0  # missing: uniform emission
                    else:
                        obs = filt_genotypes[node.name][snp_index]
                        if not self.haploid:
                            cls = self._iupac_to_genotype_classes(obs)
                            lik[cls] = 1.0
                        else:
                            for state in self._get_iupac_full(obs):
                                lik[self.char_map[state]] = 1.0
                    down_liks[node.idx] = lik / lik.sum()
                else:
                    msg = np.ones(n_states, dtype=float)
                    for child in node.children:
                        P = self._qcache.P(
                            max(child.dist, self.min_branch_length), rate
                        )
                        msg *= P @ down_liks[child.idx]
                    down_liks[node.idx] = self._norm(msg)

            up_liks: Dict[int, np.ndarray] = {tree.treenode.idx: root_prior.copy()}
            for node in tree.treenode.traverse("preorder"):
                if node.is_root():
                    continue
                parent = node.up
                sib_prod = np.ones(n_states, dtype=float)
                for sib in parent.children:
                    if sib.idx == node.idx:
                        continue
                    P_sib = self._qcache.P(max(sib.dist, self.min_branch_length), rate)
                    sib_prod *= P_sib @ down_liks[sib.idx]
                parent_msg = up_liks[parent.idx] * sib_prod
                P = self._qcache.P(max(node.dist, self.min_branch_length), rate)
                up = parent_msg @ P
                up_liks[node.idx] = up / (up.sum() or 1.0)

            for samp in tips_with_missing:
                node = tree.get_nodes(samp)[0]
                leaf_emission = down_liks[node.idx]  # uniform if missing
                tip_post = self._norm(up_liks[node.idx] * leaf_emission)

                if self.ground_truth_ and (samp, snp_index) in self.ground_truth_:
                    if not self.haploid:
                        # store genotype posterior (10) and allele-marginal (4)
                        self.imputed_genotype_likelihoods_[(samp, snp_index)] = (
                            tip_post.copy()
                        )
                        self.imputed_likelihoods_[(samp, snp_index)] = (
                            self._marginalize_genotype_to_allele(tip_post)
                        )
                    else:
                        self.imputed_likelihoods_[(samp, snp_index)] = tip_post.copy()

                if not self.haploid:
                    call = self._genotype_posterior_to_iupac(tip_post, mode="MAP")
                else:
                    call = self._allele_posterior_to_iupac_genotype(
                        tip_post, mode="MAP"
                    )
                filt_genotypes[samp][snp_index] = call

        df = pd.DataFrame.from_dict(filt_genotypes, orient="index")

        if df.applymap(self._is_missing).any().any():
            raise AssertionError("Imputation failed. Missing values remain.")

        return df

    def _iupac_to_genotype_classes(self, char: str) -> list[int]:
        """Map IUPAC code to allowed unordered genotype class indices (10-state)."""
        c = str(char).upper()
        # Allele indices: A=0, C=1, G=2, T=3
        single = {"A": 0, "C": 1, "G": 2, "T": 3}
        het_map = {  # two-allele ambiguity
            "M": (0, 1),
            "R": (0, 2),
            "W": (0, 3),
            "S": (2, 1),
            "Y": (1, 3),
            "K": (2, 3),
        }
        if c in single:
            # homozygote only
            return [[0, 4, 7, 9][single[c]]]  # AA,CC,GG,TT class indices
        if c in het_map:
            i, j = het_map[c]
            # find index in our class order (i<=j by construction)
            i, j = (i, j) if i <= j else (j, i)
            class_order = self._dip_agg.genotype_classes
            return [class_order.index((i, j))]
        # Ambiguity codes with >2 alleles or missing: allow any compatible genotype
        amb = {
            "N": {0, 1, 2, 3},
            "-": {0, 1, 2, 3},
            "B": {1, 2, 3},
            "D": {0, 2, 3},
            "H": {0, 1, 3},
            "V": {0, 1, 2},
        }
        if c in amb:
            allowed = amb[c]
            classes = []
            for idx, (i, j) in enumerate(self._dip_agg.genotype_classes):
                if i in allowed and j in allowed:
                    classes.append(idx)
            return classes
        # Fallback: allow all
        return list(range(10))

    def _genotype_posterior_to_iupac(self, p10: np.ndarray, mode: str = "MAP") -> str:
        """Convert genotype posterior over 10 classes to an IUPAC code."""
        p = np.asarray(p10, dtype=float)
        p = p / (p.sum() or 1.0)
        if mode.upper() == "SAMPLE":
            k = int(np.random.choice(len(p), p=p))
        else:
            k = int(np.argmax(p))
        i, j = self._dip_agg.genotype_classes[k]
        alleles = ["A", "C", "G", "T"]
        gt = alleles[i] + alleles[j]
        return self._genotype_to_iupac(gt)

    def _marginalize_genotype_to_allele(self, p10: np.ndarray) -> np.ndarray:
        """Allele posterior from genotype posterior for eval/diagnostics (length 4)."""
        p10 = np.asarray(p10, dtype=float)
        p10 = p10 / (p10.sum() or 1.0)
        agg = self._dip_agg
        out = np.zeros(4, dtype=float)
        for a in range(4):
            out[a] = p10[agg.hom_class_index[a]] + 0.5 * np.sum(
                p10[agg.het_classes_by_allele[a]]
            )
        s = out.sum()
        return out / (s or 1.0)

    def _parse_arguments(
        self,
    ) -> Tuple[
        Dict[str, List[Union[str, int]]], tt.tree, pd.DataFrame, Optional[List[float]]
    ]:
        if (
            not hasattr(self.genotype_data, "snpsdict")
            or self.genotype_data.snpsdict is None
        ):
            raise TypeError("`GenotypeData.snpsdict` must be defined.")
        if not hasattr(self.tree_parser, "tree") or self.tree_parser.tree is None:
            raise TypeError("`TreeParser.tree` must be defined.")
        if not hasattr(self.tree_parser, "qmat") or self.tree_parser.qmat is None:
            raise TypeError("`TreeParser.qmat` must be defined.")
        site_rates = getattr(self.tree_parser, "site_rates", None)
        return (
            self.genotype_data.snpsdict,
            self.tree_parser.tree,
            self.tree_parser.qmat,
            site_rates,
        )

    def _get_iupac_full(self, char: str) -> List[str]:
        iupac_map = {
            "A": ["A"],
            "G": ["G"],
            "C": ["C"],
            "T": ["T"],
            "N": ["A", "C", "T", "G"],
            "-": ["A", "C", "T", "G"],
            "R": ["A", "G"],
            "Y": ["C", "T"],
            "S": ["G", "C"],
            "W": ["A", "T"],
            "K": ["G", "T"],
            "M": ["A", "C"],
            "B": ["C", "G", "T"],
            "D": ["A", "G", "T"],
            "H": ["A", "C", "T"],
            "V": ["A", "C", "G"],
        }
        return iupac_map.get(char.upper(), ["A", "C", "T", "G"])

    def _allele_posterior_to_genotype_probs(self, p: np.ndarray) -> Dict[str, float]:
        p = np.maximum(p, 0.0)
        s = p.sum()
        p = (p / s) if s > 0 else np.ones_like(p) / len(p)
        alleles = ["A", "C", "G", "T"]
        probs = {}
        for i, a in enumerate(alleles):
            for j, b in enumerate(alleles[i:], start=i):
                probs[a + b] = float(p[i] * p[j] * (2.0 if i != j else 1.0))
        z = sum(probs.values()) or 1.0
        return {k: v / z for k, v in probs.items()}

    def _genotype_to_iupac(self, gt: str) -> str:
        if gt[0] == gt[1]:
            return gt[0]
        pair = "".join(sorted(gt))
        het_map = {"AC": "M", "AG": "R", "AT": "W", "CG": "S", "CT": "Y", "GT": "K"}
        return het_map.get(pair, "N")

    def _allele_posterior_to_iupac_genotype(
        self, p: np.ndarray, mode: str = "MAP"
    ) -> str:
        gprobs = self._allele_posterior_to_genotype_probs(p)
        if mode.upper() == "SAMPLE":
            gts, vals = zip(*gprobs.items())
            choice = np.random.choice(len(gts), p=np.array(vals, dtype=float))
            return self._genotype_to_iupac(gts[choice])
        best_gt = max(gprobs, key=gprobs.get)
        return self._genotype_to_iupac(best_gt)

    def _align_and_check_q(self, q_df: pd.DataFrame) -> pd.DataFrame:
        """Return Q reindexed to ['A','C','G','T'] and assert CTMC sanity.

        Args:
            q_df: Square rate matrix with nucleotide index/columns.

        Returns:
            Reindexed Q as a DataFrame in A,C,G,T order.

        Raises:
            ValueError: If Q is malformed or not alignable.
        """
        required = ["A", "C", "G", "T"]
        if not set(required).issubset(set(q_df.index)) or not set(required).issubset(
            set(q_df.columns)
        ):
            raise ValueError("Q must have index/columns including exactly A,C,G,T.")

        q_df = q_df.loc[required, required].astype(float)

        q = q_df.to_numpy()

        # Off-diagonals >= 0, diagonals <= 0, rows sum ~ 0
        if np.any(q[~np.eye(4, dtype=bool)] < -1e-12):
            raise ValueError("Q off-diagonal entries must be non-negative.")

        if np.any(np.diag(q) > 1e-12):
            raise ValueError("Q diagonal entries must be non-positive.")

        if not np.allclose(q.sum(axis=1), 0.0, atol=1e-8):
            self.logger.error(q.sum(axis=1))
            raise ValueError("Q rows must sum to 0.")

        return pd.DataFrame(q, index=required, columns=required)

    def _stationary_pi_from_q(self, Q: np.ndarray) -> np.ndarray:
        """Compute stationary distribution pi for generator Q.

        Uses eigenvector of Q^T at eigenvalue 0; clips tiny negatives; renormalizes.

        Args:
            Q: (4,4) CTMC generator.

        Returns:
            (4,) stationary distribution pi with sum=1.
        """
        w, v = scipy.linalg.eig(Q.T)
        k = int(np.argmin(np.abs(w)))
        pi = np.real(v[:, k])
        pi = np.maximum(pi, 0.0)
        s = float(pi.sum())
        if not np.isfinite(s) or s <= 0:
            # Fallback uniform if eigen failed
            return np.ones(Q.shape[0]) / Q.shape[0]
        return pi / s

    def _repair_and_scale_q(
        self,
        q_df: pd.DataFrame,
        target_mean_rate: float = 1.0,
        state_order: tuple[str, ...] = ("A", "C", "G", "T"),
        neg_offdiag_tol: float = 1e-8,
    ) -> pd.DataFrame:
        """Repair Q to a valid CTMC generator and scale its mean rate.

        Steps:
        1) Reindex to `state_order` and cast to float.
        2) Set negative off-diagonals to 0 (warn if any < -neg_offdiag_tol).
        3) Set diagonal q_ii = -sum_{j!=i} q_ij so rows sum to 0 exactly.
        4) If a row has zero off-diagonal sum, inject a tiny uniform exit rate.
        5) Compute stationary pi and scale Q so that -sum_i pi_i q_ii = target_mean_rate.

        Args:
            q_df: DataFrame with index=columns=states.
            target_mean_rate: Desired average rate under pi (commonly 1.0).
            state_order: Desired nucleotide order.
            neg_offdiag_tol: Tolerance to log warnings for negative off-diags.

        Returns:
            Repaired, scaled Q as a DataFrame in `state_order`.
        """
        # 1) Align
        missing = set(state_order) - set(q_df.index) | set(state_order) - set(
            q_df.columns
        )
        if missing:
            raise ValueError(
                f"Q must have states {state_order}, missing: {sorted(missing)}"
            )
        Q = q_df.loc[state_order, state_order].to_numpy(dtype=float, copy=True)

        # 2) Clip negative off-diagonals
        off_mask = ~np.eye(Q.shape[0], dtype=bool)
        neg_off = Q[off_mask] < 0
        if np.any(Q[off_mask] < -neg_offdiag_tol):
            self.logger.warning(
                f"Q has negative off-diagonals; clipping {int(np.sum(neg_off))} entries."
            )
        Q[off_mask] = np.maximum(Q[off_mask], 0.0)

        # 3) Set diagonal so rows sum to 0 exactly
        row_off_sum = Q.sum(axis=1) - np.diag(Q)  # includes diag; fix next
        np.fill_diagonal(Q, -row_off_sum)

        # 4) Ensure no absorbing rows
        zero_rows = row_off_sum <= 0
        if np.any(zero_rows):
            eps = 1e-8
            Q[zero_rows, :] = 0.0
            for i in np.where(zero_rows)[0]:
                Q[i, :] = eps / (Q.shape[0] - 1)
                Q[i, i] = -eps
            self.logger.warning(
                f"Injected tiny exit rates in {int(np.sum(zero_rows))} rows."
            )

        # Sanity: rows now sum to 0 exactly
        if not np.allclose(Q.sum(axis=1), 0.0, atol=1e-12):
            raise RuntimeError("Internal error: row sums not zero after repair.")

        # 5) Scale to target mean rate under stationary pi
        pi = self._stationary_pi_from_q(Q)
        mean_rate = float(-np.dot(pi, np.diag(Q)))
        if not (np.isfinite(mean_rate) and mean_rate > 0):
            self.logger.warning("Mean rate non-positive; skipping scaling.")
            scale = 1.0
        else:
            scale = mean_rate / float(target_mean_rate)
        Q /= scale

        return pd.DataFrame(Q, index=state_order, columns=state_order)

    def _is_missing(self, x: object) -> bool:
        """Return True if x represents a missing genotype token."""
        s = str(x).strip()
        return s in self._MISSING_TOKENS

    def _is_known(self, x: object) -> bool:
        return not self._is_missing(x)

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        s = float(np.sum(v))
        if s <= 0 or not np.isfinite(s):
            return np.ones_like(v) / v.size
        return v / s
