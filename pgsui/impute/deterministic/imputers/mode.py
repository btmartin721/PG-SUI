# Standard library imports
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

# Third-party imports
import matplotlib.pyplot as plt
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
from snpio.utils.logging import LoggerManager

from pgsui.data_processing.config import apply_dot_overrides, load_yaml_to_dataclass
from pgsui.data_processing.containers import MostFrequentConfig
from pgsui.utils.classification_viz import ClassificationReportVisualizer

# Local imports
from pgsui.utils.plotting import Plotting

# Type checking imports
if TYPE_CHECKING:
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
        return load_yaml_to_dataclass(
            config, MostFrequentConfig, preset_builder=MostFrequentConfig.from_preset
        )
    if isinstance(config, dict):
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
    """Most-frequent (mode) imputer that mirrors DL evaluation on 0/1/2.

    This imputer computes the most frequent genotype (mode) for each locus based on the training set and uses it to fill in missing values. It supports both global modes and population-specific modes if population data is provided. The imputer follows an evaluation protocol similar to deep learning models, including splitting the data into training and testing sets, masking observed cells in the test set for evaluation, and producing detailed classification reports and plots. It handles both diploid and haploid data, with special considerations for haploid scenarios. The imputer is designed to work seamlessly with genotype data encoded in 0/1/2 format, where -1 indicates missing values.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        config: Optional[Union[MostFrequentConfig, dict, str]] = None,
        overrides: Optional[dict] = None,
    ) -> None:
        """Initialize the Most-Frequent (mode) imputer from a unified config.

        Args:
            genotype_data (GenotypeData): Backing genotype data.
            config (MostFrequentConfig | dict | str | None): Configuration as a dataclass,
                nested dict, or YAML path. If None, defaults are used.
            overrides (dict | None): Flat dot-key overrides applied last with highest precedence, e.g. {'algo.by_populations': True, 'split.test_size': 0.3}.

        Notes:
            - This mirrors other config-driven models (AE/VAE/NLPCA/UBP).
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
        self.prefix = cfg.io.prefix
        self.verbose = cfg.io.verbose
        self.debug = cfg.io.debug

        # Logger
        logman = LoggerManager(
            __name__, prefix=self.prefix, verbose=self.verbose, debug=self.debug
        )
        self.logger = logman.get_logger()

        # RNG / encoder
        self.rng = np.random.default_rng(cfg.io.seed)
        self.encoder = GenotypeEncoder(self.genotype_data)

        # Work in 0/1/2 with -1 for missing (parity with DL modules)
        X012 = self.encoder.genotypes_012.astype(np.int16, copy=True)
        X012[X012 < 0] = -1
        self.X012_ = X012
        self.num_features_ = X012.shape[1]

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

        # Populations (only if requested)
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
        self.global_modes_: Dict[int, int] = {}
        self.group_modes_: Dict[str | int, Dict[int, int]] = {}
        self.sim_mask_: Optional[np.ndarray] = None
        self.train_idx_: Optional[np.ndarray] = None
        self.test_idx_: Optional[np.ndarray] = None
        self.X_train_df_: Optional[pd.DataFrame] = None
        self.ground_truth012_: Optional[np.ndarray] = None
        self.metrics_: Dict[str, int | float] = {}
        self.X_imputed012_: Optional[np.ndarray] = None

        # Ploidy heuristic for 0/1/2 scoring parity
        uniq = np.unique(self.X012_[self.X012_ != -1])
        self.is_haploid_ = np.array_equal(np.sort(uniq), np.array([0, 2]))

        # Plotting (use config, not genotype_data fields)
        self.plot_format = cfg.plot.fmt
        self.plot_fontsize = cfg.plot.fontsize
        self.plot_despine = cfg.plot.despine
        self.plot_dpi = cfg.plot.dpi
        self.show_plots = cfg.plot.show

        self.model_name = (
            "ImputeMostFrequentPerPop" if self.by_populations else "ImputeMostFrequent"
        )
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
        )

        # Output dirs
        dirs = ["models", "plots", "metrics", "optimize"]
        self._create_model_directories(self.prefix, dirs)

    def fit(self) -> "ImputeMostFrequent":
        """Learn per-locus modes on TRAIN rows; mask all observed cells on TEST rows.

        This method prepares the data for imputation by splitting it into training and testing sets, computing the most frequent genotype (mode) for each locus based on the training set, and creating a mask to simulate missing data in the test set for evaluation purposes.

        Returns:
            ImputeMostFrequent: The fitted imputer instance.s
        """
        self.train_idx_, self.test_idx_ = self._make_train_test_split()
        self.ground_truth012_ = self.X012_.copy()

        # Work in DataFrame with NaN as missing for mode computation
        df_all = pd.DataFrame(self.ground_truth012_, dtype=np.float32)
        df_all = df_all.replace(self.missing, np.nan)
        df_all = df_all.replace(-9, np.nan)  # Just in case

        # Modes from TRAIN rows only (per-locus)
        df_train = df_all.iloc[self.train_idx_].copy()

        self.global_modes_ = {
            col: self._series_mode(df_train[col]) for col in df_train.columns
        }

        self.group_modes_.clear()
        if self.by_populations:
            tmp = df_train.copy()
            tmp["_pops_"] = self.pops[self.train_idx_]
            for pop, grp in tmp.groupby("_pops_"):
                gdf = grp.drop(columns=["_pops_"])
                self.group_modes_[pop] = {
                    col: self._series_mode(gdf[col]) for col in gdf.columns
                }

        # Mask ALL observed cells on TEST rows (evaluation protocol parity)
        obs_mask = df_all.notna().to_numpy()  # observed = not NaN
        test_rows_mask = np.zeros(obs_mask.shape[0], dtype=bool)

        if self.test_idx_.size > 0:
            test_rows_mask[self.test_idx_] = True
        sim_mask = obs_mask & test_rows_mask[:, None]  # cells to mask for eval

        df_sim = df_all.copy()
        df_sim.values[sim_mask] = np.nan

        self.sim_mask_ = sim_mask
        self.X_train_df_ = df_sim
        self.is_fit_ = True

        self.logger.info(
            f"Fit complete. Train rows: {self.train_idx_.size}, Test rows: {self.test_idx_.size}. "
            f"Masked {int(sim_mask.sum())} observed test cells for evaluation."
        )
        return self

    def transform(self) -> np.ndarray:
        """Impute missing cells in the FULL dataset; evaluate on masked test cells.

        This method first imputes the evaluation-masked training DataFrame to compute metrics, then imputes the full dataset (only true missings) for final output. It produces the same evaluation reports and plots as the DL models, including both 0/1/2 zygosity and 10-class IUPAC reports.

        Returns:
            np.ndarray: Imputed genotypes as IUPAC strings, shape (n_samples, n_variants).
        """
        if not self.is_fit_:
            raise RuntimeError("Model is not fitted. Call fit() before transform().")
        assert self.X_train_df_ is not None

        # 1) Impute the evaluation-masked copy (to compute metrics)
        imputed_eval_df = self._impute_df(self.X_train_df_)
        X_imputed_eval = imputed_eval_df.to_numpy(dtype=np.int16)
        self.X_imputed012_ = X_imputed_eval

        # Evaluate like the DL models (first 0/1/2, then 10-class from decoded strings)
        self._evaluate_and_report()

        # 2) Impute the FULL dataset (only true missings)
        df_missingonly = pd.DataFrame(self.ground_truth012_, dtype=np.float32)
        df_missingonly.replace(self.missing, np.nan, inplace=True)
        imputed_full_df = self._impute_df(df_missingonly)
        X_imputed_full_012 = imputed_full_df.to_numpy(dtype=np.int16)

        # Plot distributions (parity with DL transform())
        gt_decoded = self.encoder.decode_012(self.ground_truth012_)
        imp_decoded = self.encoder.decode_012(X_imputed_full_012)
        self.plotter_.plot_gt_distribution(gt_decoded, is_imputed=False)
        self.plotter_.plot_gt_distribution(imp_decoded, is_imputed=True)

        # Return IUPAC strings (same as DL .transform())
        return imp_decoded

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
        df = df_in.copy()
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(self.global_modes_[col])
        return df.astype(np.int16)

    def _impute_by_population_mode(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Impute missing cells in df_in using population-specific modes.

        This method imputes missing values in the provided DataFrame using population-specific modes. It fills in missing values (NaNs) with the most frequent genotype for each locus within the corresponding population. If a population-specific mode is not available for a locus, it falls back to the global mode.

        Args:
            df_in (pd.DataFrame): Input DataFrame with missing values as NaN.

        Returns:
            pd.DataFrame: DataFrame with missing values imputed.
        """
        df = df_in.copy()
        df["_pops_"] = self.pops
        out_cols = []
        for col in df.columns:
            if col == "_pops_" or not df[col].isnull().any():
                out_cols.append(df[col])
                continue
            pop_to_mode = {
                pop: self.group_modes_.get(pop, {}).get(col, self.global_modes_[col])
                for pop in df["_pops_"].unique()
            }
            filled_col = df[col].fillna(df["_pops_"].map(pop_to_mode))
            out_cols.append(filled_col)
        imputed_df = pd.concat(out_cols, axis=1)
        return imputed_df.drop(columns=["_pops_"]).astype(np.int16)

    def _series_mode(self, s: pd.Series) -> int:
        """Compute the mode of a pandas Series, ignoring NaNs.

        This method computes the mode of a pandas Series, ignoring NaN values. If the Series is empty after removing NaNs, it returns a default value. The method ensures that the mode is one of the valid genotype values (0, 1, or 2), clamping to the default if necessary.

        Args:
            s (pd.Series): Input pandas Series.

        Returns:
            int: The mode of the series, or the default value if no valid entries exist.
        """
        s_valid = s.dropna().astype(int)
        if s_valid.empty:
            return self.default
        # Mode among {0,1,2}; if ties, pandas picks the smallest (okay)
        mode_val = int(s_valid.mode().iloc[0])
        if mode_val not in (0, 1, 2):
            # Safety: clamp to valid zygosity in case of odd inputs
            mode_val = self.default if self.default in (0, 1, 2) else 0
        return mode_val

    def _evaluate_and_report(self) -> None:
        """Evaluate imputed vs. ground truth on masked test cells; produce reports and plots.

        Requires that fit() and transform() have been called.

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

        y_true_dec = self.encoder.decode_012(self.ground_truth012_)
        y_pred_dec = self.encoder.decode_012(X_pred_eval)

        encodings_dict = {
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
        y_true_int = self.encoder.convert_int_iupac(
            y_true_dec, encodings_dict=encodings_dict
        )
        y_pred_int = self.encoder.convert_int_iupac(
            y_pred_dec, encodings_dict=encodings_dict
        )

        y_true_10 = y_true_int[self.sim_mask_]
        y_pred_10 = y_pred_int[self.sim_mask_]
        self._evaluate_iupac10_and_plot(y_true_10, y_pred_10)

    def _evaluate_012_and_plot(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """0/1/2 zygosity report & confusion matrix.

        This method generates a classification report and confusion matrix for genotypes encoded as 0 (REF), 1 (HET), and 2 (ALT). If the data is determined to be haploid (only 0 and 2 present), it folds the ALT genotype (2) into HET (1) for evaluation purposes. The method computes various performance metrics, logs the classification report, and creates visualizations of the results.

        Args:
            y_true (np.ndarray): True genotypes (0/1/2) for masked
            y_pred (np.ndarray): Predicted genotypes (0/1/2) for masked

        Raises:
            NotFittedError: If fit() and transform() have not been called.
        """
        labels = [0, 1, 2]
        # Haploid parity: fold ALT (2) into ALT/Present (1)
        if self.is_haploid_:
            y_true[y_true == 2] = 1
            y_pred[y_pred == 2] = 1
            labels = [0, 1]

        metrics = {
            "n_masked_test": int(y_true.size),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(
                y_true, y_pred, average="macro", labels=labels, zero_division=0
            ),
            "precision": precision_score(
                y_true, y_pred, average="macro", labels=labels, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, average="macro", labels=labels, zero_division=0
            ),
        }
        self.metrics_.update({f"zygosity_{k}": v for k, v in metrics.items()})

        report_names = ["REF", "HET"] if self.is_haploid_ else ["REF", "HET", "ALT"]

        self.logger.info(
            f"\n{classification_report(y_true, y_pred, labels=labels, target_names=report_names, zero_division=0)}"
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=report_names,
            zero_division=0,
            output_dict=True,
        )

        viz = ClassificationReportVisualizer(reset_kwargs=self.plotter_.param_dict)

        plots = viz.plot_all(
            report,
            title_prefix=f"{self.model_name} Zygosity Report",
            show=getattr(self, "show_plots", False),
            heatmap_classes_only=True,
        )

        for name, fig in plots.items():
            fout = self.plots_dir / f"zygosity_report_{name}.{self.plot_format}"
            if hasattr(fig, "savefig"):
                fig.savefig(fout, dpi=300, facecolor="#111122")
                plt.close(fig)
            else:
                fig.write_html(file=fout.with_suffix(".html"))

        viz._reset_mpl_style()

        # Save JSON
        self._save_report(report, suffix="zygosity")

        # Confusion matrix
        self.plotter_.plot_confusion_matrix(
            y_true, y_pred, label_names=report_names, prefix="zygosity"
        )

    def _evaluate_iupac10_and_plot(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """10-class IUPAC report & confusion matrix.

        This method generates a classification report and confusion matrix for genotypes encoded as 10-class IUPAC codes (0-9). It computes various performance metrics, logs the classification report, and creates visualizations of the results.

        Args:
            y_true (np.ndarray): True genotypes (0-9) for masked
            y_pred (np.ndarray): Predicted genotypes (0-9) for masked

        Raises:
            NotFittedError: If fit() and transform() have not been called.
        """
        labels_idx = list(range(10))
        labels_names = ["A", "C", "G", "T", "W", "R", "M", "K", "Y", "S"]

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(
                y_true, y_pred, average="macro", labels=labels_idx, zero_division=0
            ),
            "precision": precision_score(
                y_true, y_pred, average="macro", labels=labels_idx, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, average="macro", labels=labels_idx, zero_division=0
            ),
        }
        self.metrics_.update({f"iupac_{k}": v for k, v in metrics.items()})

        self.logger.info(
            f"\n{classification_report(y_true, y_pred, labels=labels_idx, target_names=labels_names, zero_division=0)}"
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=labels_idx,
            target_names=labels_names,
            zero_division=0,
            output_dict=True,
        )

        viz = ClassificationReportVisualizer(reset_kwargs=self.plotter_.param_dict)

        plots = viz.plot_all(
            report,
            title_prefix=f"{self.model_name} IUPAC Report",
            show=getattr(self, "show_plots", False),
            heatmap_classes_only=True,
        )

        # Reset the style from Optuna's plotting.
        plt.rcParams.update(self.plotter_.param_dict)

        for name, fig in plots.items():
            fout = self.plots_dir / f"iupac_report_{name}.{self.plot_format}"
            if hasattr(fig, "savefig"):
                fig.savefig(fout, dpi=300, facecolor="#111122")
                plt.close(fig)
            else:
                fig.write_html(file=fout.with_suffix(".html"))

        # Reset the style
        viz._reset_mpl_style()

        # Save JSON
        self._save_report(report, suffix="iupac")

        # Confusion matrix
        self.plotter_.plot_confusion_matrix(
            y_true, y_pred, label_names=labels_names, prefix="iupac"
        )

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
                k = int(round(self.test_size * rows.size))
                if k > 0:
                    buckets.append(self.rng.choice(rows, size=k, replace=False))
            test_idx = (
                np.sort(np.concatenate(buckets)) if buckets else np.array([], dtype=int)
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

    def _save_report(self, report_dict: Dict[str, float], suffix: str) -> None:
        """Save classification report dictionary as a JSON file.

        This method saves the provided classification report dictionary to a JSON file in the metrics directory, appending the specified suffix to the filename.

        Args:
            report_dict (Dict[str, float]): The classification report dictionary to save.
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
