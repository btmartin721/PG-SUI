# Standard library imports
import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from plotly.graph_objs import Figure as PlotlyFigure
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    jaccard_score,
    matthews_corrcoef,
)
from snpio import GenotypeEncoder
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import validate_input_type

from pgsui.data_processing.config import apply_dot_overrides, load_yaml_to_dataclass
from pgsui.data_processing.containers import MostFrequentConfig
from pgsui.data_processing.transformers import SimMissingTransformer
from pgsui.utils.classification_viz import ClassificationReportVisualizer
from pgsui.utils.logging_utils import configure_logger

# Local imports
from pgsui.utils.plotting import Plotting
from pgsui.utils.pretty_metrics import PrettyMetrics

# Type checking imports
if TYPE_CHECKING:
    from snpio import TreeParser
    from snpio.read_input.genotype_data import GenotypeData


def ensure_mostfrequent_config(
    config: Union[MostFrequentConfig, dict, str, None],
) -> MostFrequentConfig:
    """Return a concrete MostFrequentConfig (dataclass, dict, YAML path, or None).

    Args:
        config (Union[MostFrequentConfig, dict, str, None]): The configuration to ensure is a MostFrequentConfig.

    Returns:
        MostFrequentConfig: The ensured MostFrequentConfig.
    """
    if config is None:
        return MostFrequentConfig()
    if isinstance(config, MostFrequentConfig):
        return config
    if isinstance(config, str):
        return load_yaml_to_dataclass(config, MostFrequentConfig)
    if isinstance(config, dict):
        config = copy.deepcopy(config)  # copy
        base = MostFrequentConfig()
        # honor optional top-level 'preset'
        preset = config.pop("preset", None)

        if preset:
            base = MostFrequentConfig.from_preset(preset)

        def _flatten(prefix: str, d: dict, out: dict) -> dict:
            for k, v in d.items():
                kk = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(kk, v, out)
                else:
                    out[kk] = v
            return out

        flat = _flatten("", config, {})
        return apply_dot_overrides(base, flat)

    raise TypeError("config must be a MostFrequentConfig, dict, YAML path, or None.")


class ImputeMostFrequent:
    """Most-frequent (mode) deterministic imputer for 0/1/2 genotypes.

    Computes the per-locus mode (globally or per population) from the training set and uses it to fill missing values. The evaluation protocol mirrors the DL imputers: train/test split with evaluation on either all observed test cells or a simulated-missing subset (depending on config), plus classification reports and plots. It handles both diploid and haploid data. Input genotypes are expected in 0/1/2 encoding with missing values represented by any negative integer. Output is returned as IUPAC strings via ``decode_012``.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        tree_parser: Optional["TreeParser"] = None,
        config: Optional[Union[MostFrequentConfig, dict, str]] = None,
        overrides: Optional[dict] = None,
        simulate_missing: bool = True,
        sim_strategy: Literal[
            "random",
            "random_weighted",
            "random_weighted_inv",
            "nonrandom",
            "nonrandom_weighted",
        ] = "random",
        sim_prop: float = 0.2,
        sim_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize the Most-Frequent (mode) imputer from a unified config.

        This constructor ensures that the provided configuration is valid and initializes the imputer's internal state. It sets up logging, random number generation, genotype encoding, and various parameters based on the configuration. The imputer is prepared to handle population-specific modes if specified in the configuration.

        Args:
            genotype_data (GenotypeData): Backing genotype data.
            tree_parser (TreeParser | None): Optional SNPio phylogenetic tree parser for nonrandom sim_strategy modes.
            config (MostFrequentConfig | dict | str | None): Configuration as a dataclass,
                nested dict, or YAML path. If None, defaults are used.
            overrides (Optional[dict]): Flat dot-key overrides applied last with highest precedence, e.g. {'algo.by_populations': True, 'split.test_size': 0.3}.
            simulate_missing (bool): Whether to simulate missing data if enabled in config. Defaults to True.
            sim_strategy (Literal["random", "random_weighted", "random_weighted_inv", "nonrandom", "nonrandom_weighted"]): Strategy for simulating missing data if enabled in config.
            sim_prop (float): Proportion of data to simulate as missing if enabled in config. Default is 0.2.
            sim_kwargs (Optional[dict]): Additional keyword arguments for the simulated missing data transformer.

        Notes:
            - This mirrors other config-driven models (AE/VAE).
            - Evaluation split behavior uses cfg.split; plotting uses cfg.plot.
            - I/O/logging seeds and verbosity use cfg.io.
        """
        # Normalize config then apply highest-precedence overrides
        cfg = ensure_mostfrequent_config(config)
        if overrides:
            cfg = apply_dot_overrides(cfg, overrides)
        self.cfg = cfg

        # Basic fields
        self.genotype_data = genotype_data
        self.tree_parser = tree_parser
        self.prefix = cfg.io.prefix
        self.verbose = cfg.io.verbose
        self.debug = cfg.io.debug

        self.parameters_dir: Path
        self.metrics_dir: Path
        self.plots_dir: Path
        self.models_dir: Path
        self.optimize_dir: Path

        # Logger
        logman = LoggerManager(
            __name__, prefix=self.prefix, verbose=self.verbose, debug=self.debug
        )
        self.logger = configure_logger(
            logman.get_logger(), verbose=self.verbose, debug=self.debug
        )

        # RNG / encoder
        self.rng = np.random.default_rng(cfg.io.seed)
        self.encoder = GenotypeEncoder(self.genotype_data)

        self.missing_internal = -1

        # include common missing value aliases
        self.missing_aliases = {int(cfg.algo.missing), -9, -1}

        X = np.asarray(self.encoder.genotypes_012)
        Xf = X.astype(np.float32, copy=True)
        Xf = np.where(np.isnan(Xf), -1.0, Xf)
        Xf[Xf < 0] = -1.0
        self.X012_ = Xf.astype(np.float32, copy=True)
        self.num_features_ = self.X012_.shape[1]

        # Simulated-missing controls (mirror VAE/AE semantics where possible)
        sim_cfg = getattr(self.cfg, "sim", None)
        sim_cfg_kwargs = dict(getattr(sim_cfg, "sim_kwargs", {}) or {})

        self.simulate_missing: bool
        self.sim_strategy: str
        self.sim_prop: float
        self.sim_kwargs: dict

        # Missing simulation config
        if sim_cfg is None:
            # Fallback defaults if MostFrequentConfig has no .sim block
            self.simulate_missing = simulate_missing
            self.sim_strategy = sim_strategy
            self.sim_prop = sim_prop
        else:
            self.simulate_missing = bool(
                getattr(sim_cfg, "simulate_missing", simulate_missing)
            )
            self.sim_strategy = getattr(sim_cfg, "sim_strategy", sim_strategy)
            self.sim_prop = float(getattr(sim_cfg, "sim_prop", sim_prop))
            if getattr(sim_cfg, "sim_kwargs", sim_kwargs):
                sim_cfg_kwargs.update(sim_cfg.sim_kwargs)

        self.sim_kwargs = sim_cfg_kwargs

        # Simulated-missing masks (global + test-only)
        self.sim_mask_global_: Optional[np.ndarray] = None  # shape (N, L), bool
        self.sim_mask_test_only_: Optional[np.ndarray] = None

        # Split & algo knobs
        self.test_size = float(cfg.split.test_size)
        self.test_indices = (
            None
            if cfg.split.test_indices is None
            else np.asarray(cfg.split.test_indices, dtype=int)
        )
        self.by_populations = bool(cfg.algo.by_populations)
        self.default = int(cfg.algo.default)
        self.missing = int(cfg.algo.missing)

        # Populations (if requested)
        self.pops = None
        if self.by_populations:
            pops = getattr(self.genotype_data, "populations", None)
            if pops is None:
                msg = "by_populations=True requires genotype_data.populations."
                self.logger.error(msg)
                raise TypeError(msg)
            self.pops = np.asarray(pops)
            if len(self.pops) != self.X012_.shape[0]:
                msg = f"`populations` length ({len(self.pops)}) != number of samples ({self.X012_.shape[0]})."
                self.logger.error(msg)
                raise ValueError(msg)

        # State
        self.is_fit_: bool = False
        self.global_modes_: Dict[str, int] = {}
        self.group_modes_: dict = {}
        self.sim_mask_: Optional[np.ndarray] = None
        self.train_idx_: Optional[np.ndarray] = None
        self.test_idx_: Optional[np.ndarray] = None
        self.X_train_df_: Optional[pd.DataFrame] = None
        self.ground_truth012_: Optional[np.ndarray] = None
        self.X_imputed012_: Optional[np.ndarray] = None

        # Ploidy heuristic for 0/1/2 scoring parity
        self.ploidy = self.cfg.io.ploidy
        self.is_haploid_ = self.ploidy == 1

        # Plotting (use config, not genotype_data fields)
        self.plot_format = cfg.plot.fmt
        self.plot_fontsize = cfg.plot.fontsize
        self.plot_despine = cfg.plot.despine
        self.plot_dpi = cfg.plot.dpi
        self.show_plots = cfg.plot.show
        self.use_multiqc = bool(cfg.plot.multiqc)

        self.model_name = (
            "ImputeMostFrequentPerPop" if self.by_populations else "ImputeMostFrequent"
        )

        # Output dirs
        dirs = ["models", "plots", "metrics", "optimize", "parameters"]
        self._create_model_directories(self.prefix, dirs)

        self.plotter_ = Plotting(
            self.model_name,
            prefix=self.prefix,
            plot_format=self.plot_format,
            plot_fontsize=self.plot_fontsize,
            plot_dpi=self.plot_dpi,
            title_fontsize=self.plot_fontsize,
            despine=self.plot_despine,
            show_plots=self.show_plots,
            verbose=self.verbose,
            debug=self.debug,
            multiqc=True,
            multiqc_section=f"PG-SUI: {self.model_name} Model Imputation",
        )

        if self.tree_parser is None and self.sim_strategy.startswith("nonrandom"):
            msg = "tree_parser is required for nonrandom and nonrandom_weighted simulated missing strategies."
            self.logger.error(msg)
            raise ValueError(msg)

    def fit(self) -> "ImputeMostFrequent":
        """Learn per-locus modes on TRAIN rows; mask simulated cells on TEST rows.

        This method computes the most frequent genotype (mode) for each locus based on the training set and prepares the evaluation masks for the test set. It supports both global modes and population-specific modes if population data is provided. The method sets up the internal state required for imputation and evaluation.

        Returns:
            ImputeMostFrequent: The fitted imputer instance.
        """
        self.train_idx_, self.test_idx_ = self._make_train_test_split()
        self.ground_truth012_ = self.X012_.copy()

        # Work in DataFrame with NaN as missing for mode computation
        df_all = pd.DataFrame(self.ground_truth012_).astype("float32").copy()
        df_all[df_all < 0] = np.nan

        # Modes from TRAIN rows only (per-locus)
        df_train = df_all.iloc[self.train_idx_].copy()

        modes = {}
        for col in df_train.columns:
            s = df_train[col].dropna()
            if s.empty:
                modes[col] = int(self.default)
            else:
                vc = s.value_counts()

                # deterministic tie-break: smallest genotype among ties
                m = int(vc.index[vc.to_numpy() == vc.to_numpy().max()].min())
                modes[col] = m

        self.global_modes_ = modes

        self.group_modes_.clear()
        if self.by_populations:
            tmp = df_train.copy()
            if self.pops is not None:
                tmp["_pops_"] = self.pops[self.train_idx_]
                for pop, grp in tmp.groupby("_pops_"):
                    gdf = grp.drop(columns=["_pops_"])
                    self.group_modes_[pop] = {
                        col: self._series_mode(gdf[col]) for col in gdf.columns
                    }
            else:
                msg = "Population data is required when by_populations=True."
                self.logger.error(msg)
                raise ValueError(msg)

        # Simulated-missing mask (global → test-only)
        obs_mask = df_all.notna().to_numpy()  # observed = not NaN
        n_samples = obs_mask.shape[0]

        if self.simulate_missing:
            X_for_sim = self.ground_truth012_.astype(np.float32, copy=True)
            X_for_sim[X_for_sim < 0] = -9.0

            # Use the same transformer as VAE
            tr = SimMissingTransformer(
                genotype_data=self.genotype_data,
                tree_parser=self.tree_parser,
                prop_missing=self.sim_prop,
                strategy=self.sim_strategy,
                missing_val=-9,
                mask_missing=True,
                verbose=self.verbose,
                **self.sim_kwargs,
            )
            tr.fit(X_for_sim)

            sim_mask_global = tr.sim_missing_mask_.astype(bool)
            if sim_mask_global.shape != obs_mask.shape:
                msg = f"sim_missing_mask_ shape {sim_mask_global.shape} != obs_mask shape {obs_mask.shape}"
                self.logger.error(msg)
                raise ValueError(msg)

            sim_mask_global &= obs_mask

            # Restrict evaluation to TEST rows only
            test_rows_mask = np.zeros(n_samples, dtype=bool)
            if self.test_idx_.size > 0:
                test_rows_mask[self.test_idx_] = True

            sim_mask = sim_mask_global & test_rows_mask[:, None]

            self.sim_mask_global_ = sim_mask_global
            self.sim_mask_test_only_ = sim_mask
        else:
            # Fallback: current behavior – mask all observed cells on TEST rows
            test_rows_mask = np.zeros(n_samples, dtype=bool)
            if self.test_idx_.size > 0:
                test_rows_mask[self.test_idx_] = True
            sim_mask = obs_mask & test_rows_mask[:, None]

            self.sim_mask_global_ = None
            self.sim_mask_test_only_ = sim_mask

        # Apply the mask to create the evaluation DataFrame
        df_sim = df_all.copy()
        df_sim = df_sim.mask(sim_mask, other=np.nan)

        self.sim_mask_ = self.sim_mask_test_only_
        self.X_train_df_ = df_sim
        self.is_fit_ = True

        # Save parameters
        best_params = self.cfg.to_dict()
        params_fp = self.parameters_dir / "best_parameters.json"
        with open(params_fp, "w") as f:
            json.dump(best_params, f, indent=4)

        n_masked = int(self.sim_mask_test_only_.sum())
        self.logger.info(
            f"Fit complete. Train rows: {self.train_idx_.size}, "
            f"Test rows: {self.test_idx_.size}. "
            f"Masked {n_masked} test cells for evaluation "
            f"({'simulated' if self.simulate_missing else 'all observed'})."
        )
        return self

    def transform(self) -> np.ndarray:
        """Impute missing cells in the FULL dataset; evaluate on masked test cells.

        This method first imputes the evaluation-masked training DataFrame to compute metrics, then imputes the full dataset (only true missings) for final output. It produces the same evaluation reports and plots as the DL models, including both 0/1/2 zygosity and 10-class IUPAC reports.

        Returns:
            np.ndarray: Imputed genotypes as IUPAC strings, shape (n_samples, n_variants).

        Raises:
            NotFittedError: If fit() has not been called prior to transform().
        """
        if not self.is_fit_:
            msg = "Model is not fitted. Call fit() before transform()."
            self.logger.error(msg)
            raise NotFittedError(msg)

        assert (
            self.X_train_df_ is not None
        ), f"[{self.model_name}] X_train_df_ is not set after fit()."

        # 1) Impute the evaluation-masked copy (to compute metrics)
        imputed_eval_df = self._impute_df(self.X_train_df_)
        X_imputed_eval = imputed_eval_df.to_numpy(dtype=np.float32)
        self.X_imputed012_ = X_imputed_eval

        # Evaluate like DL models (0/1/2, then 10-class from decoded strings)
        self._evaluate_and_report()

        # 2) Impute the FULL dataset (only true missings)
        df_missingonly = pd.DataFrame(self.ground_truth012_, dtype=np.float32)
        df_missingonly[df_missingonly < 0] = np.nan

        imputed_full_df = self._impute_df(df_missingonly)
        X_imputed_full_012 = imputed_full_df.to_numpy(dtype=np.float32)

        if np.isnan(X_imputed_full_012).any():
            msg = "NaN entries remain after imputation; cannot decode safely."
            self.logger.error(msg)
            raise RuntimeError(msg)

        neg = int(np.count_nonzero(X_imputed_full_012 < 0))
        if neg:
            msg = f"{neg} negative entries remain after REF imputation. Unique: {np.unique(X_imputed_full_012[X_imputed_full_012 < 0])[:10]}"
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Plot distributions (parity with DL transform())
        if self.ground_truth012_ is None:
            msg = "ground_truth012_ is not set; cannot plot distributions."
            self.logger.error(msg)
            raise NotFittedError(msg)

        decode_input = (
            self._canonicalize_haploid_decode_input(X_imputed_full_012)
            if self.is_haploid_
            else X_imputed_full_012
        )
        imp_decoded = self.decode_012(decode_input)

        if self.show_plots:
            orig_input = (
                self._canonicalize_haploid_decode_input(self.ground_truth012_)
                if self.is_haploid_
                else self.ground_truth012_
            )
            orig_dec = self.decode_012(orig_input)
            self.plotter_.plot_gt_distribution(imp_decoded, orig_dec, True)

        # Return IUPAC strings
        return imp_decoded

    def _canonicalize_haploid_decode_input(self, X: np.ndarray) -> np.ndarray:
        """Map haploid ALT calls to diploid-style ALT-hom code before decode_012.

        decode_012 interprets code 1 as heterozygous (REF/ALT). For haploid data we
        treat any present ALT state as ALT-homozygous semantics for decoding.
        """
        arr = np.asarray(X).copy()
        miss = arr < 0
        arr = np.where(arr > 0, 2, arr)
        arr[miss] = -1
        return arr

    def _impute_df(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Impute missing cells in df_in using global or population-specific modes.

        This method imputes missing values in the provided DataFrame using either global modes or population-specific modes, depending on the configuration of the imputer. It fills in missing values (NaNs) with the most frequent genotype for each locus.

        Args:
            df_in (pd.DataFrame): Input DataFrame with missing values as NaN.

        Returns:
            pd.DataFrame: DataFrame with missing values imputed.
        """
        return (
            self._impute_global_mode(df_in)
            if not self.by_populations
            else self._impute_by_population_mode(df_in)
        )

    def _impute_global_mode(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Impute missing cells in df_in using global modes.

        This method imputes missing values in the provided DataFrame using global modes. It fills in missing values (NaNs) with the most frequent genotype for each locus across all samples.

        Args:
            df_in (pd.DataFrame): Input DataFrame with missing values as NaN.

        Returns:
            pd.DataFrame: DataFrame with missing values imputed.
        """
        if df_in.isnull().to_numpy().any():
            modes = pd.Series(self.global_modes_, index=df_in.columns, dtype="float32")
            df = df_in.fillna(modes)
        else:
            df = df_in.copy()
        return df.astype(np.float32)

    def _impute_by_population_mode(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Impute missing cells in df_in using population-specific modes with safe fallbacks.

        Notes:
            - No NaNs remain after imputation.
            - Falls back to global mode if a population has no learned mode (e.g., pop absent from train).
            - Falls back to `self.default` if everything else fails (should not happen).
        """
        # Fast path
        if not df_in.isnull().to_numpy().any():
            return df_in.astype(np.float32)

        df = df_in.copy()

        # Map population labels to df rows robustly by original sample index
        pops = pd.Series(self.pops, index=np.arange(len(self.pops))).reindex(df.index)
        if pops.isna().any():
            # If any rows cannot be mapped,
            # we still proceed with global fallback.
            self.logger.warning(
                "Some df rows could not be mapped to populations; using global fallback for those rows."
            )

        # Global modes: enforce exact column alignment + float32 dtype
        global_modes = pd.Series(self.global_modes_, index=df.columns, dtype="float32")

        # Population modes table: rows=pop, cols=loci; enforce columns + float32
        pop_modes = pd.DataFrame.from_dict(self.group_modes_, orient="index")
        if pop_modes.empty:
            pop_modes = pd.DataFrame(
                index=pd.Index([], name="population"), columns=df.columns
            )

        pop_modes = pop_modes.reindex(columns=df.columns)
        # Ensure numeric and fill any missing entries with global modes
        pop_modes = pop_modes.apply(pd.to_numeric, errors="coerce").astype("float32")
        pop_modes = pop_modes.fillna(global_modes)

        # Align per-row modes: index by population label for each row
        aligned_modes = pop_modes.reindex(pops.to_numpy())
        # Rows with unknown populations or missing pop modes -> global
        aligned_modes = aligned_modes.fillna(global_modes)

        # Impute: use aligned_modes only where df is NaN
        out = df.where(df.notna(), aligned_modes)

        # Guarantee: no NaNs remain (otherwise they'll decode to 'N')
        if out.isna().to_numpy().any():
            # Fill remaining NaNs deterministically
            out = out.fillna(global_modes).fillna(float(self.default))

            # If still any NaN, that's a serious structural issue
            if out.isna().to_numpy().any():
                msg = "NaNs remain after population+global+default fallback; cannot safely decode."
                self.logger.error(msg)
                raise RuntimeError(msg)

        return out.astype(np.float32)

    def _series_mode(self, s: pd.Series) -> int:
        """Compute the mode of a pandas Series with deterministic tie-breaking,
        excluding invalid (<0) codes.

        Rules:
            - Ignore NaNs.
            - Ignore any values < 0 (e.g., -1, -9), which represent missing ('N').
            - Choose the most frequent remaining value.
            - If multiple values are tied for max frequency, choose the smallest value.
            - If no valid values remain, return self.default.

        Args:
            s (pd.Series): Input pandas Series.

        Returns:
            int: Deterministic mode among valid genotype/base codes.
        """
        s_valid = s.dropna()

        if s_valid.empty:
            return int(self.default)

        # Coerce to numeric; drop anything non-numeric
        vals = pd.to_numeric(s_valid, errors="coerce").dropna()

        if vals.empty:
            return int(self.default)

        # Exclude missing/invalid sentinel codes (<0)
        vals = vals[vals >= 0]

        if vals.empty:
            return int(self.default)

        vc = vals.value_counts()
        max_count = vc.max()

        # Deterministic tie-break: smallest value among ties
        tied = vc.index[vc.to_numpy() == max_count]

        # value_counts index can be float; safe-cast after choosing
        m = float(tied.min())
        return int(m)

    def _evaluate_and_report(self) -> None:
        """Evaluate imputed vs. ground truth on masked test cells; produce reports and plots.

        Requires that fit() and transform() have been called. This method evaluates the imputed genotypes against the ground truth for the masked test cells, generating classification reports and confusion matrices for both 0/1/2 zygosity and 10-class IUPAC codes. It logs the results and saves the reports and plots to the designated output directories.

        Raises:
            NotFittedError: If fit() and transform() have not been called.
        """
        assert (
            self.sim_mask_ is not None
            and self.ground_truth012_ is not None
            and self.X_imputed012_ is not None
        )
        # Cells we masked for eval
        y_true_012 = self.ground_truth012_[self.sim_mask_]
        y_pred_012 = self.X_imputed012_[self.sim_mask_]
        if y_true_012.size == 0:
            self.logger.info("No masked test cells; skipping evaluation.")
            return

        # 0/1/2 report (REF/HET/ALT), with haploid folding 2->1
        self._evaluate_012_and_plot(y_true_012.copy(), y_pred_012.copy())

        # 10-class report from decoded IUPAC strings
        # Rebuild per-row/pcol predictions to decode:
        X_pred_eval = self.ground_truth012_.copy()
        X_pred_eval[self.sim_mask_] = self.X_imputed012_[self.sim_mask_]

        y_true_eval_input = (
            self._canonicalize_haploid_decode_input(self.ground_truth012_)
            if self.is_haploid_
            else self.ground_truth012_
        )
        y_pred_eval_input = (
            self._canonicalize_haploid_decode_input(X_pred_eval)
            if self.is_haploid_
            else X_pred_eval
        )

        y_true_dec = self.decode_012(y_true_eval_input)
        y_pred_dec = self.decode_012(y_pred_eval_input)

        encodings_dict = (
            {"A": 0, "C": 1, "G": 2, "T": 3, "N": -1}
            if self.is_haploid_
            else {
                "A": 0,
                "C": 1,
                "G": 2,
                "T": 3,
                "W": 4,
                "R": 5,
                "M": 6,
                "K": 7,
                "Y": 8,
                "S": 9,
                "N": -1,
            }
        )
        y_true_int = self.encoder.convert_int_iupac(
            y_true_dec, encodings_dict=encodings_dict
        )
        y_pred_int = self.encoder.convert_int_iupac(
            y_pred_dec, encodings_dict=encodings_dict
        )

        y_true_iupac = y_true_int[self.sim_mask_]
        y_pred_iupac = y_pred_int[self.sim_mask_]

        m = (y_true_iupac >= 0) & (y_pred_iupac >= 0)
        y_true_iupac, y_pred_iupac = y_true_iupac[m], y_pred_iupac[m]
        if y_true_iupac.size == 0:
            self.logger.warning("No valid IUPAC test cells; skipping IUPAC evaluation.")
            return

        self._evaluate_iupac10_and_plot(y_true_iupac, y_pred_iupac)

    def _evaluate_012_and_plot(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """0/1/2 zygosity report & confusion matrix.

        This method generates a classification report and confusion matrix for genotypes encoded as 0 (REF), 1 (HET), and 2 (ALT). If the data is haploid (only 0 and 2 present), it folds ALT (2) into the binary ALT/PRESENT class (1) for evaluation. The method computes metrics, logs the report, and creates visualizations of the results.

        Args:
            y_true (np.ndarray): True genotypes (0/1/2) for masked
            y_pred (np.ndarray): Predicted genotypes (0/1/2) for masked
        """
        # Ensures haploid folding and sklearn metrics operate on integers.
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        labels: list[int] = [0, 1, 2]
        report_names: list[str] = ["REF", "HET", "ALT"]

        # Haploid parity: fold any non-REF ALT state into ALT/Present (1)
        if self.is_haploid_:
            y_true = np.where(y_true > 0, 1, y_true)
            y_pred = np.where(y_pred > 0, 1, y_pred)
            labels = [0, 1]
            report_names = ["REF", "ALT"]

        report: dict | str = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=report_names,
            zero_division=0,
            output_dict=True,
        )

        if not isinstance(report, dict):
            msg = "classification_report did not return a dict as expected."
            self.logger.error(msg)
            raise TypeError(msg)

        if self.show_plots:
            viz = ClassificationReportVisualizer(reset_kwargs=self.plotter_.param_dict)

            plots = viz.plot_all(
                report,
                title_prefix=f"{self.model_name} Zygosity Report",
                show=self.show_plots,
                heatmap_classes_only=True,
            )

            for name, fig in plots.items():
                fout = self.plots_dir / f"zygosity_report_{name}.{self.plot_format}"
                if hasattr(fig, "savefig") and isinstance(fig, Figure):
                    fig.savefig(fout, dpi=300, facecolor="#111122")
                    plt.close(fig)
                elif isinstance(fig, PlotlyFigure):
                    fig.write_html(file=fout.with_suffix(".html"))

            viz._reset_mpl_style()

            # Confusion matrix
            self.plotter_.plot_confusion_matrix(
                y_true, y_pred, label_names=report_names, prefix="zygosity"
            )

        # ------ Additional metrics ------
        report_full = self._additional_metrics(
            y_true, y_pred, labels, report_names, report
        )

        if self.verbose or self.debug:
            pm = PrettyMetrics(
                report_full,
                precision=2,
                title=f"{self.model_name} Zygosity Report",
            )
            pm.render()

        # Save JSON
        self._save_report(report_full, suffix="zygosity")

    def _evaluate_iupac10_and_plot(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """IUPAC report & confusion matrix (ploidy-aware).

        Diploid: evaluates 10 IUPAC classes (A,C,G,T,W,R,M,K,Y,S).
        Haploid: evaluates 4 base classes (A,C,G,T).

        Args:
            y_true (np.ndarray): True encoded IUPAC labels for masked cells.
            y_pred (np.ndarray): Predicted encoded IUPAC labels for masked cells.
        """
        # --- FIX: Cast to int immediately ---
        # Guards against float inputs causing IndexError in np.eye indexing below
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        if self.is_haploid_:
            labels_idx = [0, 1, 2, 3]
            report_names = ["A", "C", "G", "T"]
        else:
            labels_idx = list(range(10))
            report_names = ["A", "C", "G", "T", "W", "R", "M", "K", "Y", "S"]

        max_label = int(max(labels_idx))
        m = (
            (y_true >= 0)
            & (y_true <= max_label)
            & (y_pred >= 0)
            & (y_pred <= max_label)
        )
        y_true, y_pred = y_true[m], y_pred[m]

        if y_true.size == 0:
            self.logger.warning("No valid IUPAC labels in expected range; skipping.")
            return

        report: dict | str = classification_report(
            y_true,
            y_pred,
            labels=labels_idx,
            target_names=report_names,
            zero_division=0,
            output_dict=True,
        )

        if not isinstance(report, dict):
            msg = "classification_report did not return a dict as expected."
            self.logger.error(msg)
            raise TypeError(msg)

        if self.show_plots:
            viz = ClassificationReportVisualizer(reset_kwargs=self.plotter_.param_dict)

            plots = viz.plot_all(
                report,
                title_prefix=f"{self.model_name} IUPAC Report",
                show=self.show_plots,
                heatmap_classes_only=True,
            )

            # Reset the style from Optuna's plotting.
            plt.rcParams.update(self.plotter_.param_dict)

            for name, fig in plots.items():
                fout = self.plots_dir / f"iupac_report_{name}.{self.plot_format}"
                if hasattr(fig, "savefig") and isinstance(fig, Figure):
                    fig.savefig(fout, dpi=300, facecolor="#111122")
                    plt.close(fig)
                elif isinstance(fig, PlotlyFigure):
                    fig.write_html(file=fout.with_suffix(".html"))

            # Reset the style
            viz._reset_mpl_style()

            # Confusion matrix
            self.plotter_.plot_confusion_matrix(
                y_true, y_pred, label_names=report_names, prefix="iupac"
            )

        # ------ Additional metrics ------
        report_full = self._additional_metrics(
            y_true, y_pred, labels_idx, report_names, report
        )

        if self.verbose or self.debug:
            pm = PrettyMetrics(
                report_full,
                precision=2,
                title=f"{self.model_name} IUPAC {len(labels_idx)}-Class Report",
            )
            pm.render()

        # Save JSON
        self._save_report(report_full, suffix="iupac")

    def _make_train_test_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create train/test split indices.

        This method creates training and testing indices based on the specified test size or provided test indices. If population-based splitting is enabled, it ensures that the test set includes samples from each population according to the specified test size.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of train and test indices.

        Raises:
            IndexError: If provided test_indices are out of bounds.
        """
        n = self.X012_.shape[0]
        all_idx = np.arange(n, dtype=int)
        if self.test_indices is not None:
            test_idx = np.unique(self.test_indices)
            if np.any((test_idx < 0) | (test_idx >= n)):
                raise IndexError("Some test_indices are out of bounds.")
            train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
            return train_idx, test_idx

        if self.by_populations and self.pops is not None:
            buckets = []
            for pop in np.unique(self.pops):
                rows = np.where(self.pops == pop)[0]
                k = max(1, int(round(self.test_size * rows.size)))
                if k > 0:
                    buckets.append(self.rng.choice(rows, size=k, replace=False))
            test_idx = (
                np.sort(np.concatenate(buckets)) if buckets else np.array([], dtype=int)
            )
        else:
            k = max(1, int(round(self.test_size * n)))
            test_idx = (
                self.rng.choice(n, size=k, replace=False)
                if k > 0
                else np.array([], dtype=int)
            )

        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
        return train_idx, test_idx

    def _save_report(self, report_dict: Dict[str, Any], suffix: str) -> None:
        """Save classification report dictionary as a JSON file.

        This method saves the provided classification report dictionary to a JSON file in the metrics directory, appending the specified suffix to the filename.

        Args:
            report_dict (Dict[str, Any]): The classification report dictionary to save.
            suffix (str): Suffix to append to the filename (e.g., 'zygosity' or 'iupac').

        Raises:
            NotFittedError: If fit() and transform() have not been called.
        """
        if not self.is_fit_ or self.X_imputed012_ is None:
            msg = "No report to save. Ensure fit() and transform() have been called."
            raise NotFittedError(msg)

        out_fp = self.metrics_dir / f"classification_report_{suffix}.json"
        with open(out_fp, "w") as f:
            json.dump(report_dict, f, indent=4)
        self.logger.info(f"{self.model_name} {suffix} report saved to {out_fp}.")

    def _create_model_directories(self, prefix: str, outdirs: List[str]) -> None:
        """Creates the directory structure for storing model outputs.

        This method sets up a standardized folder hierarchy for saving models, plots, metrics, and optimization results, organized under a main directory named after the provided prefix.

        Args:
            prefix (str): The prefix for the main output directory.
            outdirs (List[str]): A list of subdirectory names to create within the main directory.

        Raises:
            Exception: If any of the directories cannot be created.
        """
        formatted_output_dir = Path(f"{prefix}_output")
        base_dir = formatted_output_dir / "Deterministic"

        for d in outdirs:
            subdir = base_dir / d / self.model_name
            setattr(self, f"{d}_dir", subdir)
            try:
                getattr(self, f"{d}_dir").mkdir(parents=True, exist_ok=True)
            except Exception as e:
                msg = f"Failed to create directory {getattr(self, f'{d}_dir')}: {e}"
                self.logger.error(msg)
                raise Exception(msg)

    def decode_012(
        self, X: np.ndarray | pd.DataFrame | list[list[int]], is_nuc: bool = False
    ) -> np.ndarray:
        """Decode 012-encodings to IUPAC chars with metadata repair.

        Supports:
        - is_nuc=True: direct 0..9 -> IUPAC mapping
        - is_nuc=False: ref/alt-based decoding with metadata repair

        Additional behavior:
        - Multiallelic ALT is allowed. The ALT used for decoding is chosen as the
            most common alternate base (A/C/G/T) observed in the source SNP column.
        - If REF/ALT are missing or ambiguous, they are inferred from observed
            base counts in the source SNP column (if available).

        Returns:
            np.ndarray: IUPAC strings as a 2D array of shape (n_samples, n_snps).
        """
        df = validate_input_type(X, return_type="df")
        if not isinstance(df, pd.DataFrame):
            msg = f"Expected a pandas.DataFrame in 'decode_012', but got: {type(df)}."
            self.logger.error(msg)
            raise ValueError(msg)

        # IUPAC Definitions
        iupac_to_bases: dict[str, set[str]] = {
            "A": {"A"},
            "C": {"C"},
            "G": {"G"},
            "T": {"T"},
            "R": {"A", "G"},
            "Y": {"C", "T"},
            "S": {"G", "C"},
            "W": {"A", "T"},
            "K": {"G", "T"},
            "M": {"A", "C"},
            "B": {"C", "G", "T"},
            "D": {"A", "G", "T"},
            "H": {"A", "C", "T"},
            "V": {"A", "C", "G"},
            "N": set(),
        }
        bases_to_iupac = {
            frozenset(v): k for k, v in iupac_to_bases.items() if k != "N"
        }
        missing_codes = {"", ".", "N", "NONE", "-", "?", "./.", ".|.", "NAN", "nan"}

        def _normalize_iupac(value: object) -> str | None:
            """Normalize an input into a single IUPAC code token or None."""
            if value is None:
                return None
            if isinstance(value, (bytes, np.bytes_)):
                value = bytes(value).decode("utf-8", errors="ignore")

            if isinstance(value, (list, tuple, pd.Series, np.ndarray)):
                if isinstance(value, pd.Series):
                    arr = value.to_numpy()
                else:
                    arr = value
                if isinstance(arr, np.ndarray) and arr.ndim == 0:
                    return _normalize_iupac(arr.item())
                if len(arr) == 0:
                    return None
                for item in arr:
                    code = _normalize_iupac(item)
                    if code is not None:
                        return code
                return None

            s = str(value).upper().strip()
            if not s or s in missing_codes:
                return None

            if "," in s:
                for tok in (t.strip() for t in s.split(",")):
                    if tok and tok not in missing_codes and tok in iupac_to_bases:
                        return tok
                return None

            return s if s in iupac_to_bases else None

        def _extract_candidates(value: object) -> list[str]:
            """Extract all candidate IUPAC tokens from multiallelic/list-like metadata."""
            if value is None:
                return []

            if isinstance(value, (bytes, np.bytes_)):
                value = bytes(value).decode("utf-8", errors="ignore")

            # list-like: flatten
            if isinstance(value, (list, tuple, pd.Series, np.ndarray)):
                if isinstance(value, pd.Series):
                    seq = value.to_numpy()
                else:
                    seq = value
                out: list[str] = []
                for item in seq:
                    out.extend(_extract_candidates(item))
                return out

            s = str(value).upper().strip()
            if not s or s in missing_codes:
                return []

            toks = [t.strip() for t in s.split(",")] if "," in s else [s]
            out: list[str] = []
            for tok in toks:
                if not tok or tok in missing_codes:
                    continue
                if tok in iupac_to_bases:
                    out.append(tok)
            return out

        def _base_counts_from_column(
            col: np.ndarray, *, max_scan: int = 5000
        ) -> dict[str, int]:
            """Count A/C/G/T from a source SNP column of IUPAC codes.

            Counting rule:
            - Homozygote (single-base) contributes +2 to that base
            - Heterozygote/ambiguity contributes +1 to each base in the set
            """
            counts = {"A": 0, "C": 0, "G": 0, "T": 0}
            seen = 0
            for val in col:
                code = _normalize_iupac(val)
                if code is None or code == "N":
                    continue
                bases = iupac_to_bases.get(code, set())
                if not bases:
                    continue
                if len(bases) == 1:
                    b = next(iter(bases))
                    if b in counts:
                        counts[b] += 2
                else:
                    for b in bases:
                        if b in counts:
                            counts[b] += 1
                seen += 1
                if seen >= max_scan:
                    break
            return counts

        def _choose_single_base(
            token: str | None, counts: dict[str, int]
        ) -> str | None:
            """If token is ambiguous, pick the most frequent constituent base; else return token."""
            if token is None:
                return None
            bases = iupac_to_bases.get(token, set())
            if not bases:
                return None
            if len(bases) == 1:
                b = next(iter(bases))
                return b if b in {"A", "C", "G", "T"} else token
            # Ambiguous: choose most common base in observed counts
            best = None
            best_ct = -1
            for b in bases:
                ct = counts.get(b, 0)
                if ct > best_ct:
                    best_ct = ct
                    best = b
            return best if best in {"A", "C", "G", "T"} else None

        def _choose_alt_from_candidates(
            ref_base: str | None,
            alt_candidates: list[str],
            counts: dict[str, int],
        ) -> str | None:
            """Pick ALT as the most common base among candidates, excluding REF."""
            # Reduce candidates to base set
            base_cands: set[str] = set()
            for tok in alt_candidates:
                bases = iupac_to_bases.get(tok, set())
                for b in bases:
                    if b in {"A", "C", "G", "T"}:
                        base_cands.add(b)

            if ref_base in base_cands:
                base_cands.remove(ref_base)

            if not base_cands:
                return None

            # Most common by counts; deterministic tie-breaker by base order
            order = {"A": 0, "C": 1, "G": 2, "T": 3}
            best = max(base_cands, key=lambda b: (counts.get(b, 0), -order[b]))
            return best

        # numeric codes
        codes_df = df.apply(pd.to_numeric, errors="coerce")
        codes = codes_df.fillna(-1).astype(np.int8).to_numpy()
        n_rows, n_cols = codes.shape

        if is_nuc:
            iupac_list = np.array(
                ["A", "C", "G", "T", "W", "R", "M", "K", "Y", "S"], dtype="<U1"
            )
            out = np.full((n_rows, n_cols), "N", dtype="<U1")
            mask = (codes >= 0) & (codes <= 9)
            out[mask] = iupac_list[codes[mask]]
            return out

        # Metadata fetch
        ref_alleles = getattr(self.genotype_data, "ref", [])
        alt_alleles = getattr(self.genotype_data, "alt", [])

        if len(ref_alleles) != n_cols:
            ref_alleles = getattr(self, "_ref", [None] * n_cols)
        if len(alt_alleles) != n_cols:
            alt_alleles = getattr(self, "_alt", [None] * n_cols)

        if len(ref_alleles) != n_cols:
            ref_alleles = [None] * n_cols
        if len(alt_alleles) != n_cols:
            alt_alleles = [None] * n_cols

        out = np.full((n_rows, n_cols), "N", dtype="<U1")

        # Lazy-load source SNP data once
        source_snp_data = None
        if getattr(self.genotype_data, "snp_data", None) is not None:
            try:
                source_snp_data = np.asarray(self.genotype_data.snp_data)
            except Exception:
                source_snp_data = None

        for j in range(n_cols):
            ref_tok = _normalize_iupac(ref_alleles[j])
            alt_toks = _extract_candidates(alt_alleles[j])  # multiallelic-safe

            # Column base counts (if we have source data)
            counts = {"A": 0, "C": 0, "G": 0, "T": 0}
            if (
                source_snp_data is not None
                and source_snp_data.ndim == 2
                and source_snp_data.shape[1] > j
            ):
                try:
                    counts = _base_counts_from_column(source_snp_data[:, j])
                except Exception:
                    counts = {"A": 0, "C": 0, "G": 0, "T": 0}

            # Canonicalize REF to a single base if possible
            ref_base = _choose_single_base(ref_tok, counts)

            # Choose ALT:
            #  - if multiallelic candidates exist, pick most common base among them
            #  - else if single ALT token exists, canonicalize it
            alt_base = None
            if alt_toks:
                alt_base = _choose_alt_from_candidates(ref_base, alt_toks, counts)
                if alt_base is None and len(alt_toks) == 1:
                    alt_base = _choose_single_base(alt_toks[0], counts)
            else:
                # no ALT candidates in metadata
                alt_base = None

            # --- REPAIR LOGIC (frequency-aware) ---
            # If still missing, infer from observed counts in source column:
            if (ref_base is None or alt_base is None) and any(
                v > 0 for v in counts.values()
            ):
                # Sort bases by count desc, then A/C/G/T deterministic
                order = {"A": 0, "C": 1, "G": 2, "T": 3}
                ranked = sorted(counts.keys(), key=lambda b: (-counts[b], order[b]))
                if ref_base is None:
                    ref_base = ranked[0]
                if alt_base is None:
                    alt_base = next(
                        (b for b in ranked if b != ref_base and counts[b] > 0), None
                    )

            # --- DEFAULTS FOR MISSING ---
            if ref_base is None and alt_base is None:
                ref_base = "N"
                alt_base = "N"
            elif ref_base is None:
                ref_base = alt_base if alt_base is not None else "N"
            elif alt_base is None:
                # Monomorphic or truly no alt info -> treat as ref
                alt_base = ref_base

            ref = ref_base
            alt = alt_base

            # --- COMPUTE HET CODE ---
            if ref == alt:
                het_code = ref
            else:
                union_set = frozenset({ref, alt})
                het_code = bases_to_iupac.get(union_set, "N")

            col_codes = codes[:, j]

            # Case 0: REF
            if ref != "N":
                out[col_codes == 0, j] = ref

            # Case 1: HET
            if het_code != "N":
                out[col_codes == 1, j] = het_code
            else:
                # fallback to REF if het is not representable
                if ref != "N":
                    out[col_codes == 1, j] = ref

            # Case 2: ALT
            if alt != "N":
                out[col_codes == 2, j] = alt
            else:
                if ref != "N":
                    out[col_codes == 2, j] = ref

        return out

    def _additional_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[int],
        report_names: list[str],
        report: dict[str, dict[str, float] | float],
    ) -> dict[str, dict[str, float] | float]:
        """Compute additional metrics and augment the report dictionary.

        Notes:
            - Safely computes Average Precision (AP) even when some classes are absent
            in y_true (common in haploid eval slices after 2->1 folding).
            - AP is computed per-class as a one-vs-rest binary AP **only if that class
            has at least one positive example** in y_true; otherwise AP is set to NaN.
            - Macro/weighted AP are computed over classes with support > 0.

        Args:
            y_true (np.ndarray): True genotypes.
            y_pred (np.ndarray): Predicted genotypes.
            labels (list[int]): List of label indices.
            report_names (list[str]): List of report names corresponding to labels.
            report (dict[str, dict[str, float] | float]): Classification report dictionary to augment.

        Returns:
            dict[str, dict[str, float] | float]: Augmented report dictionary with additional metrics.
        """
        y_true = np.asarray(y_true).astype(int, copy=False)
        y_pred = np.asarray(y_pred).astype(int, copy=False)

        K = len(report_names)
        # Keep only valid label indices (protects np.eye indexing)
        m = (y_true >= 0) & (y_true < K) & (y_pred >= 0) & (y_pred < K)
        y_true = y_true[m]
        y_pred = y_pred[m]

        if y_true.size == 0:
            self.logger.warning("No valid labels for AP/Jaccard computation; skipping.")
            return report

        # Hard prediction "scores" (deterministic imputer has no probabilities).
        # Shape: (N, K)
        y_score_ohe = np.eye(K, dtype=float)[y_pred]

        # --- Per-class AP (safe) ---
        # Compute one-vs-rest AP only when the class exists in y_true.
        ap_pc = np.full(K, np.nan, dtype=float)
        support = np.zeros(K, dtype=int)
        for k in range(K):
            yk = (y_true == k).astype(int)
            support[k] = int(yk.sum())
            if support[k] == 0:
                continue  # no positives -> AP undefined; leave NaN
            # Use scores for class k (0/1 here)
            ap_pc[k] = float(average_precision_score(yk, y_score_ohe[:, k]))

        # Macro/weighted AP over supported classes only
        supported = support > 0
        if supported.any():
            ap_macro = float(np.nanmean(ap_pc[supported]))
            ap_weighted = float(
                np.nansum(ap_pc[supported] * support[supported])
                / support[supported].sum()
            )
        else:
            ap_macro = float("nan")
            ap_weighted = float("nan")

        # --- Jaccard (safe with zero_division=0) ---
        jaccard_pc = jaccard_score(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        )
        jaccard_macro = float(
            jaccard_score(y_true, y_pred, average="macro", zero_division=0)
        )
        jaccard_weighted = float(
            jaccard_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        # --- MCC ---
        mcc = float(matthews_corrcoef(y_true, y_pred))

        if not isinstance(jaccard_pc, np.ndarray):
            msg = "jaccard_score did not return np.ndarray as expected."
            self.logger.error(msg)
            raise TypeError(msg)

        # Build augmented report
        report_full: dict[str, dict[str, float] | float] = {}

        dd_subset = {
            k: v for k, v in report.items() if k in report_names and isinstance(v, dict)
        }
        for i, class_name in enumerate(report_names):
            class_report = dd_subset.get(class_name, {})
            if not class_report:
                continue
            report_full[class_name] = dict(class_report)
            # AP may be NaN if class absent in y_true (that’s correct)
            report_full[class_name]["average-precision"] = (
                float(ap_pc[i]) if np.isfinite(ap_pc[i]) else float("nan")
            )
            report_full[class_name]["jaccard"] = float(jaccard_pc[i])

        macro_avg = report.get("macro avg")
        if isinstance(macro_avg, dict):
            report_full["macro avg"] = dict(macro_avg)
            report_full["macro avg"]["average-precision"] = ap_macro
            report_full["macro avg"]["jaccard"] = jaccard_macro

        weighted_avg = report.get("weighted avg")
        if isinstance(weighted_avg, dict):
            report_full["weighted avg"] = dict(weighted_avg)
            report_full["weighted avg"]["average-precision"] = ap_weighted
            report_full["weighted avg"]["jaccard"] = jaccard_weighted

        report_full["mcc"] = mcc
        accuracy_val = report.get("accuracy")
        if isinstance(accuracy_val, (int, float)):
            report_full["accuracy"] = float(accuracy_val)

        # Optional: log once if AP had undefined classes (helps debugging haploid slices)
        if np.any((support == 0)):
            missing_classes = [report_names[i] for i in range(K) if support[i] == 0]
            self.logger.debug(
                f"AP undefined for classes absent in y_true (support=0): {missing_classes}"
            )

        return report_full
