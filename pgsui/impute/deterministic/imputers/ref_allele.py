# Standard library
import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from plotly.graph_objs import Figure as PlotlyFigure
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Project
from snpio import GenotypeEncoder
from snpio.utils.logging import LoggerManager

from pgsui.data_processing.config import apply_dot_overrides, load_yaml_to_dataclass
from pgsui.data_processing.containers import RefAlleleConfig
from pgsui.data_processing.transformers import SimMissingTransformer
from pgsui.utils.classification_viz import ClassificationReportVisualizer
from pgsui.utils.logging_utils import configure_logger
from pgsui.utils.plotting import Plotting
from pgsui.utils.pretty_metrics import PrettyMetrics

if TYPE_CHECKING:
    from snpio import TreeParser
    from snpio.read_input.genotype_data import GenotypeData


def ensure_refallele_config(
    config: Union[RefAlleleConfig, dict, str, None],
) -> RefAlleleConfig:
    """Return a concrete RefAlleleConfig (dataclass, dict, YAML path, or None).

    This function normalizes the input configuration for the RefAllele imputer. It accepts a RefAlleleConfig instance, a dictionary of parameters, a path to a YAML file, or None. If None is provided, it returns a default RefAlleleConfig instance. If a dictionary is provided, it flattens any nested structures and applies the parameters to a base configuration, honoring any top-level 'preset' key. If a string path is provided, it loads the configuration from the specified YAML file.

    Args:
        config (Union[RefAlleleConfig, dict, str, None]): Configuration input which can be a RefAlleleConfig instance, a dictionary of parameters, a path to a YAML file, or None.

    Returns:
        RefAlleleConfig: A concrete RefAlleleConfig instance.

    Raises:
        TypeError: If the input type is not supported.
    """
    if config is None:
        return RefAlleleConfig()
    if isinstance(config, RefAlleleConfig):
        return config
    if isinstance(config, str):
        return load_yaml_to_dataclass(config, RefAlleleConfig)
    if isinstance(config, dict):
        config = copy.deepcopy(config)  # copy
        base = RefAlleleConfig()
        # honor optional top-level 'preset'
        preset = config.pop("preset", None)
        if preset:
            base = RefAlleleConfig.from_preset(preset)

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

    raise TypeError(
        f"config must be RefAlleleConfig, dict, YAML path, or None, but got: {type(config)}."
    )


class ImputeRefAllele:
    """Deterministic imputer that replaces all missing 0/1/2 genotype values with the REF genotype (0).

    The imputer works on 0/1/2 with -1 as missing. Evaluation splits samples into TRAIN/TEST once. Masks ALL originally observed cells on TEST rows for eval. Produces: 0/1/2 (zygosity) classification report + confusion matrix 10-class IUPAC classification report (via decode_012) + confusion matrix. Plots genotype distribution before/after imputation.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        tree_parser: Optional["TreeParser"] = None,
        config: Optional[Union[RefAlleleConfig, dict, str]] = None,
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
        """Initialize the Ref-Allele imputer from a unified config.

        This constructor ensures that the provided configuration is valid and initializes the imputer's internal state. It sets up logging, random number generation, genotype encoding, and various parameters based on the configuration. The imputer is prepared to handle population-specific modes if specified in the configuration.

        Args:
            genotype_data (GenotypeData): Backing genotype data.
            tree_parser (Optional[TreeParser]): Optional SNPio phylogenetic tree parser for population-specific modes.
            config (RefAlleleConfig | dict | str | None): Configuration as a dataclass, nested dict, or YAML path. If None, defaults are used.
            overrides (dict | None): Flat dot-key overrides applied last with highest precedence, e.g. {'split.test_size': 0.25, 'algo.missing': -1}.
            simulate_missing (bool): Whether to simulate missing data during evaluation. Default is True.
            sim_strategy (Literal): Strategy for simulating missing data if enabled in config.
            sim_prop (float): Proportion of data to simulate as missing if enabled in config.
            sim_kwargs (Optional[dict]): Additional keyword arguments for the simulated missing data transformer.
        """
        # Normalize config then apply highest-precedence overrides
        cfg = ensure_refallele_config(config)
        if overrides:
            cfg = apply_dot_overrides(cfg, overrides)
        self.cfg = cfg

        # Basic fields
        self.genotype_data = genotype_data
        self.tree_parser = tree_parser
        self.prefix = cfg.io.prefix
        self.verbose = cfg.io.verbose
        self.debug = cfg.io.debug

        # Simulation knobs (shared with other deterministic imputers)
        if cfg.sim is None:
            self.simulate_missing = simulate_missing
            self.sim_strategy = sim_strategy
            self.sim_prop = sim_prop
            self.sim_kwargs = sim_kwargs or {}
        else:
            sim_cfg = cfg.sim
            self.simulate_missing = getattr(
                sim_cfg, "simulate_missing", simulate_missing
            )
            self.sim_strategy = getattr(sim_cfg, "sim_strategy", sim_strategy)
            self.sim_prop = float(getattr(sim_cfg, "sim_prop", sim_prop))
            self.sim_kwargs: Dict[str, Any] = dict(
                getattr(sim_cfg, "sim_kwargs", sim_kwargs) or {}
            )

        # Output dirs
        self.plots_dir: Path
        self.metrics_dir: Path
        self.parameters_dir: Path
        self.models_dir: Path
        self.optimize_dir: Path

        # Logger
        logman = LoggerManager(
            __name__, prefix=self.prefix, verbose=self.verbose, debug=self.debug
        )
        self.logger = configure_logger(
            logman.get_logger(), verbose=self.verbose, debug=self.debug
        )

        if self.tree_parser is None and self.sim_strategy.startswith("nonrandom"):
            msg = "tree_parser is required for nonrandom and nonrandom_weighted simulated missing strategies."
            self.logger.error(msg)
            raise ValueError(msg)

        # RNG / encoder
        self.rng = np.random.default_rng(cfg.io.seed)
        self.encoder = GenotypeEncoder(self.genotype_data)

        # Work in 0/1/2 with -1 for missing
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
        self.missing = int(cfg.algo.missing)

        # State
        self.is_fit_: bool = False
        self.sim_mask_: np.ndarray | None = None
        self.train_idx_: np.ndarray | None = None
        self.test_idx_: np.ndarray | None = None
        self.X_train_df_: pd.DataFrame | None = None
        self.ground_truth012_: np.ndarray | None = None
        self.X_imputed012_: np.ndarray | None = None
        self.metrics_: Dict[str, int | float] = {}

        # Ploidy heuristic for 0/1/2 scoring parity
        self.ploidy = self.cfg.io.ploidy
        self.is_haploid_ = self.ploidy == 1

        # Plotting (use config)
        self.plot_format = cfg.plot.fmt
        self.plot_fontsize = cfg.plot.fontsize
        self.plot_despine = cfg.plot.despine
        self.plot_dpi = cfg.plot.dpi
        self.show_plots = cfg.plot.show

        self.model_name = "ImputeRefAllele"
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

        # Output dirs
        dirs = ["models", "plots", "metrics", "optimize", "parameters"]
        self._create_model_directories(self.prefix, dirs)

    def fit(self) -> "ImputeRefAllele":
        """Create TRAIN/TEST split and build eval mask, with optional sim-missing.

        This method prepares the imputer by splitting the data into training and testing sets and constructing an evaluation mask. If `cfg.sim.simulate_missing` is False (default), it masks all originally observed genotype entries on TEST rows. If `cfg.sim.simulate_missing` is True, it uses SimMissingTransformer to select a subset of observed cells as simulated-missing, then restricts that mask to TEST rows only. Evaluation is then performed only on these simulated-missing cells, mirroring the deep learning models.

        Returns:
            ImputeRefAllele: The fitted imputer instance.
        """
        # Train/test split indices
        self.train_idx_, self.test_idx_ = self._make_train_test_split()
        self.ground_truth012_ = self.X012_.copy()

        # Use NaN for missing inside a DataFrame to leverage fillna
        df_all = pd.DataFrame(self.ground_truth012_, dtype=np.float32)
        df_all[df_all < 0] = np.nan

        # Observed mask in the ORIGINAL data (before any simulated-missing)
        obs_mask = df_all.notna().to_numpy()  # shape (n_samples, n_loci)

        # TEST row selector
        test_rows_mask = np.zeros(obs_mask.shape[0], dtype=bool)
        if self.test_idx_ is not None and self.test_idx_.size > 0:
            test_rows_mask[self.test_idx_] = True

        # Decide how to build the sim mask: legacy vs simulated-missing
        if getattr(self, "simulate_missing", False):
            X_for_sim = self.ground_truth012_.astype(np.float32, copy=True)
            X_for_sim[X_for_sim < 0] = -9.0

            # Simulate missing on the full matrix; we only use the mask.
            tr = SimMissingTransformer(
                genotype_data=self.genotype_data,
                tree_parser=self.tree_parser,
                prop_missing=self.sim_prop,
                strategy=self.sim_strategy,
                missing_val=-9,
                mask_missing=True,
                verbose=self.verbose,
                **(self.sim_kwargs or {}),
            )
            tr.fit(X_for_sim)
            sim_mask_global = tr.sim_missing_mask_.astype(bool)

            # Only consider cells that were originally observed
            sim_mask_global = sim_mask_global & obs_mask

            # Restrict evaluation to TEST rows only
            sim_mask = sim_mask_global & test_rows_mask[:, None]
            mode_desc = "simulated missing on TEST rows"
        else:
            # Legacy behavior: mask ALL originally observed TEST cells
            sim_mask = obs_mask & test_rows_mask[:, None]
            mode_desc = "all originally observed cells on TEST rows"

        # Apply eval mask: set these cells to NaN in the eval DataFrame
        df_sim = df_all.copy()
        df_sim.values[sim_mask] = np.nan

        # Store state
        self.sim_mask_ = sim_mask
        self.X_train_df_ = df_sim
        self.is_fit_ = True

        n_masked = int(sim_mask.sum())
        self.logger.info(
            f"Fit complete. Train rows: {self.train_idx_.size}, "
            f"Test rows: {self.test_idx_.size}. "
            f"Masked {n_masked} cells for evaluation ({mode_desc})."
        )

        # Persist config for reproducibility
        params_fp = self.parameters_dir / "best_parameters.json"
        best_params = self.cfg.to_dict()
        with open(params_fp, "w") as f:
            json.dump(best_params, f, indent=4)

        return self

    def transform(self) -> np.ndarray:
        """Impute missing values with REF genotype (0) and evaluate on masked test cells.

        This method performs the imputation by replacing all missing genotype values with the REF genotype (0). It evaluates the imputation performance on the masked test cells, producing classification reports and plots that mirror those generated by deep learning models. The final output is the fully imputed genotype matrix in IUPAC string format.

        Returns:
            np.ndarray: The fully imputed genotype matrix in IUPAC string format.

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        if not self.is_fit_:
            msg = "ImputeRefAllele instance is not fitted yet. Call 'fit()' before 'transform()'."
            self.logger.error(msg)
            raise NotFittedError(msg)

        assert (
            self.X_train_df_ is not None
        ), f"[{self.model_name}] X_train_df_ is not set after fit()."

        # 1) Impute the evaluation-masked copy (compute metrics)
        imputed_eval_df = self._impute_ref(df_in=self.X_train_df_)
        X_imputed_eval = imputed_eval_df.to_numpy(dtype=np.int16)
        self.X_imputed012_ = X_imputed_eval

        # Evaluate parity with DL models
        self._evaluate_and_report()

        # 2) Impute the FULL dataset (only true missings)
        df_missingonly = pd.DataFrame(self.ground_truth012_, dtype=np.float32)
        df_missingonly[df_missingonly < 0] = np.nan

        imputed_full_df = self._impute_ref(df_in=df_missingonly)
        X_imputed_full_012 = imputed_full_df.to_numpy(dtype=np.int16)

        # Plot distributions (like DL .transform())

        if self.ground_truth012_ is None:
            msg = "ground_truth012_ is NoneType; cannot plot distributions."
            self.logger.error(msg, exc_info=True)
            raise NotFittedError(msg)

        imp_decoded = self.encoder.decode_012(X_imputed_full_012)

        if self.show_plots:
            gt_decoded = self.encoder.decode_012(self.ground_truth012_)
            self.plotter_.plot_gt_distribution(gt_decoded, is_imputed=False)
            self.plotter_.plot_gt_distribution(imp_decoded, is_imputed=True)

        # Return IUPAC strings
        return imp_decoded

    def _impute_ref(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Replace every NaN with the REF genotype code (0) across all loci.

        This is the deterministic REF-allele imputation in 0/1/2 encoding. The method fills all NaN values in the input DataFrame with 0, representing the REF genotype. The operation is performed column-wise, and since the fill value is constant, it is efficient to apply it in a vectorized manner.

        Args:
            df_in (pd.DataFrame): Input DataFrame with NaNs representing missing genotypes.

        Returns:
            pd.DataFrame: DataFrame with NaNs replaced by 0 (REF genotype).
        """
        df = df_in.copy()
        # Fill all NaNs with 0 (homozygous REF) column-wise; constant so vectorized is fine
        df = df.fillna(0)
        return df.astype(np.int16)

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
        y_true_012 = self.ground_truth012_[self.sim_mask_]
        y_pred_012 = self.X_imputed012_[self.sim_mask_]

        if y_true_012.size == 0:
            self.logger.info("No masked test cells; skipping evaluation.")
            return

        # 0/1/2 report (REF/HET/ALT), with haploid folding 2->1 if needed
        self._evaluate_012_and_plot(y_true_012.copy(), y_pred_012.copy())

        # 10-class IUPAC report from decoded strings (parity with DL)
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

        m = (y_true_10 >= 0) & (y_pred_10 >= 0)
        y_true_10, y_pred_10 = y_true_10[m], y_pred_10[m]
        if y_true_10.size == 0:
            self.logger.warning(
                "No valid IUPAC test cells; skipping 10-class evaluation."
            )
            return

        self._evaluate_iupac10_and_plot(y_true_10, y_pred_10)

    def _evaluate_012_and_plot(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """0/1/2 zygosity report & confusion matrix.

        This method generates a classification report and confusion matrix for genotypes encoded as 0 (REF), 1 (HET), and 2 (ALT). If the data is determined to be haploid (only 0 and 2 present), it folds the ALT genotype (2) into HET (1) for evaluation purposes. The method computes various performance metrics, logs the classification report, and creates visualizations of the results.

        Args:
            y_true (np.ndarray): True genotypes (0/1/2) for masked
            y_pred (np.ndarray): Predicted genotypes (0/1/2) for
        """
        labels = [0, 1, 2]
        report_names = ["REF", "HET", "ALT"]

        # Haploid parity: fold 2 -> 1
        if self.is_haploid_:
            y_true[y_true == 2] = 1
            y_pred[y_pred == 2] = 1
            labels = [0, 1]
            report_names = ["REF", "ALT"]

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

        report: str | dict = classification_report(
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

        report_subset = {}
        for k, v in report.items():
            tmp = {}
            if isinstance(v, dict) and "support" in v:
                for k2, v2 in v.items():
                    if k2 != "support":
                        tmp[k2] = v2
                if tmp:
                    report_subset[k] = tmp

        if report_subset and (self.verbose or self.debug):
            pm = PrettyMetrics(
                report_subset,
                precision=3,
                title=f"{self.model_name} Zygosity Report",
            )
            pm.render()

        if self.show_plots:
            viz = ClassificationReportVisualizer(reset_kwargs=self.plotter_.param_dict)

            if not isinstance(report, dict):
                msg = "classification_report did not return a dict as expected."
                self.logger.error(msg)
                raise TypeError(msg)

            plots = viz.plot_all(
                report,
                title_prefix=f"{self.model_name} Zygosity Report",
                show=self.show_plots,
                heatmap_classes_only=True,
            )

            # Reset the style from Optuna's plotting.
            plt.rcParams.update(self.plotter_.param_dict)

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

        self._save_report(report, suffix="zygosity")

    def _evaluate_iupac10_and_plot(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """10-class IUPAC report & confusion matrix.

        This method generates a classification report and confusion matrix for genotypes encoded using the 10 IUPAC codes (0-9). The IUPAC codes represent various nucleotide combinations, including ambiguous bases.

        Args:
            y_true (np.ndarray): True genotypes (0-9) for masked test cells.
            y_pred (np.ndarray): Predicted genotypes (0-9) for masked test cells.
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

        report = classification_report(
            y_true,
            y_pred,
            labels=labels_idx,
            target_names=labels_names,
            zero_division=0,
            output_dict=True,
        )

        if not isinstance(report, dict):
            msg = "classification_report did not return a dict as expected."
            self.logger.error(msg)
            raise TypeError(msg)

        report_subset = {}
        for k, v in report.items():
            tmp = {}
            if isinstance(v, dict) and "support" in v:
                for k2, v2 in v.items():
                    if k2 != "support":
                        tmp[k2] = v2
                if tmp:
                    report_subset[k] = tmp

        if report_subset and (self.verbose or self.debug):
            pm = PrettyMetrics(
                report_subset,
                precision=3,
                title=f"{self.model_name} IUPAC 10-Class Report",
            )
            pm.render()

        self._save_report(report, suffix="iupac")

        if self.show_plots:
            # Confusion matrix
            self.plotter_.plot_confusion_matrix(
                y_true, y_pred, label_names=labels_names, prefix="iupac"
            )

    def _make_train_test_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create train/test split indices.

        This method generates training and testing indices for the dataset. If specific test indices are provided, it uses those; otherwise, it randomly selects a proportion of samples as the test set based on the specified test size. The method ensures that the selected test indices are within valid bounds and that there is no overlap between training and testing sets.

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
                msg = "Some test_indices are out of bounds."
                self.logger.error(msg)
                raise IndexError(msg)

            train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
            return train_idx, test_idx

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

        This method saves the provided classification report dictionary to a JSON file in the metrics directory. The filename includes a suffix to distinguish between different types of reports (e.g., 'zygosity' or 'iupac').

        Args:
            report_dict (Dict[str, float]): The classification report dictionary to save.
            suffix (str): Suffix to append to the filename (e.g., 'zygosity' or 'iupac').

        Raises:
            NotFittedError: If fit() and transform() have not been called.
        """
        if not self.is_fit_ or self.X_imputed012_ is None:
            raise NotFittedError("No report to save. Ensure fit() and transform() ran.")

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
