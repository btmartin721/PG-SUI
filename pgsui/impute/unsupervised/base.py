import copy
import gc
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    jaccard_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from snpio import SNPioMultiQC
from snpio.utils.logging import LoggerManager
from snpio.utils.misc import validate_input_type

from pgsui.data_processing.transformers import SimMissingTransformer
from pgsui.impute.unsupervised.nn_scorers import Scorer
from pgsui.utils.classification_viz import ClassificationReportVisualizer
from pgsui.utils.logging_utils import configure_logger
from pgsui.utils.plotting import Plotting
from pgsui.utils.pretty_metrics import PrettyMetrics

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


class _MaskedNumpyDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray):
        self.X = X
        self.y = y
        self.mask = mask.astype(np.bool_, copy=False)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.mask[idx]


class BaseNNImputer:
    """An abstract base class for neural network-based imputers.

    This class provides a shared framework and common functionality for all neural network imputers. It is not meant to be instantiated directly. Instead, child classes should inherit from it and implement the abstract methods. Provided functionality: Directory setup and logging initialization; A hyperparameter tuning pipeline using Optuna; Utility methods for building models (`build_model`), initializing weights (`initialize_weights`), and checking for fitted attributes (`ensure_attribute`); Helper methods for calculating class weights for imbalanced data; Setup for standardized plotting and model scoring classes.
    """

    def __init__(
        self,
        model_name: str,
        genotype_data: "GenotypeData",
        prefix: str,
        *,
        device: Literal["gpu", "cpu", "mps"] = "cpu",
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initializes the base class for neural network imputers.

        This constructor sets up the device (CPU, GPU, or MPS), creates the necessary output directories for models and results, and a logger. It also initializes a genotype encoder for handling genotype data.

        Args:
            prefix (str): A prefix used to name the output directory (e.g., 'pgsui_output').
            device (Literal["gpu", "cpu", "mps"]): The device to use for PyTorch operations. If 'gpu' or 'mps' is chosen, it will fall back to 'cpu' if the required hardware is not available. Defaults to "cpu".
            verbose (bool): If True, enables detailed logging output. Defaults to False.
            debug (bool): If True, enables debug mode. Defaults to False.
        """
        self.model_name = model_name
        self.genotype_data = genotype_data

        if not hasattr(self, "tree_parser"):
            self.tree_parser = None
        if not hasattr(self, "sim_kwargs"):
            self.sim_kwargs = {}

        self.prefix = prefix
        self.verbose = verbose
        self.debug = debug

        # Quiet Matplotlib/fontTools INFO logging when saving PDF/SVG
        for name in (
            "fontTools",
            "fontTools.subset",
            "fontTools.ttLib",
            "matplotlib.font_manager",
        ):
            lg = logging.getLogger(name)
            lg.setLevel(logging.WARNING)
            lg.propagate = False

        self.device = self._select_device(device)

        # Prepare directory structure
        outdirs = ["models", "plots", "metrics", "optimize", "parameters"]
        self._create_model_directories(prefix, outdirs)

        # Initialize loggers
        kwargs = {"prefix": prefix, "verbose": verbose, "debug": debug}
        logman = LoggerManager(__name__, **kwargs)
        self.logger = configure_logger(
            logman.get_logger(), verbose=self.verbose, debug=self.debug
        )

        self.logger.info(f"Using PyTorch device: {self.device.type}.")

        # To be initialized by child classes or fit method
        self.tune_save_db: bool = False
        self.tune_resume: bool = False
        self.n_trials: int = 100
        self.model_params: Dict[str, Any] = {}
        self.tune_metric: str = "f1"
        self.learning_rate: float = 1e-3
        self.plotter_: "Plotting"
        self.num_features_: int = 0
        self.num_classes_: int = 3
        self.plot_format: Literal["pdf", "png", "jpg", "jpeg", "svg"] = "pdf"
        self.plot_fontsize: int = 10
        self.plot_dpi: int = 300
        self.title_fontsize: int = 12
        self.despine: bool = True
        self.show_plots: bool = False
        self.scoring_averaging: Literal["macro", "micro", "weighted"] = "macro"
        self.pgenc: Any = None
        self.is_haploid_: bool = False
        self.ploidy: int = 2
        self.beta: float = 0.9999
        self.max_ratio: Optional[float] = None
        self.sim_strategy: str = "random"
        self.sim_prop: float = 0.2
        self.seed: Optional[int] = None
        self.rng: np.random.Generator = np.random.default_rng(self.seed)
        self.ground_truth_: np.ndarray
        self.validation_split: float = 0.2
        self.batch_size: int = 64
        self.best_params_: Dict[str, Any] = {}

        self.optimize_dir: Path
        self.models_dir: Path
        self.plots_dir: Path
        self.metrics_dir: Path
        self.parameters_dir: Path
        self.study_db: Optional[Path] = None
        self.X_model_input_: Optional[np.ndarray] = None
        self.class_weights_: Optional[torch.Tensor] = None

    def tune_hyperparameters(self) -> Dict[str, Any]:
        """Tunes model hyperparameters using an Optuna study.

        This method orchestrates the hyperparameter search process. It creates an Optuna study that aims to maximize the metric defined in `self.tune_metric`. The search is driven by the `_objective` method, which must be implemented by the child class. After the search, the best parameters are logged, saved to a JSON file, and visualizations of the study are generated.

        Raises:
            NotImplementedError: If the `_objective` or `_set_best_params` methods are not implemented in the inheriting child class.
        """
        self.logger.info("Tuning hyperparameters. This might take a while...")

        if self.verbose or self.debug:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study_db = None
        load_if_exists = False
        if self.tune_save_db:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            study_db = (
                self.optimize_dir / "study_database" / f"optuna_study_{timestamp}.db"
            )
            study_db.parent.mkdir(parents=True, exist_ok=True)

            if self.tune_resume and study_db.exists():
                load_if_exists = True

            if not self.tune_resume and study_db.exists():
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                study_db = study_db.with_name(f"optuna_study_{timestamp}.db")

        self.study_db = study_db
        study_name = f"{self.prefix} {self.model_name} Model Optimization"
        storage = f"sqlite:///{study_db}" if self.tune_save_db else None

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
            pruner=optuna.pruners.MedianPruner(
                # Guard against small `n_trials` values (e.g., 1)
                # that can otherwise produce 0 startup/warmup/min trials.
                n_startup_trials=max(
                    1, min(int(self.n_trials * 0.1), 10, int(self.n_trials))
                ),
                n_warmup_steps=150,
                n_min_trials=max(
                    1, min(int(0.5 * self.n_trials), 25, int(self.n_trials))
                ),
            ),
        )

        if not hasattr(self, "_objective"):
            msg = "`_objective()` must be implemented in the child class."
            self.logger.error(msg)
            raise NotImplementedError(msg)

        self.n_jobs = getattr(self, "n_jobs", 1)
        if self.n_jobs < -1 or self.n_jobs == 0:
            self.logger.warning(f"Invalid n_jobs={self.n_jobs}. Setting n_jobs=1.")
            self.n_jobs = 1

        show_progress_bar = not self.verbose and not self.debug and self.n_jobs == 1

        # Set the best parameters.
        # NOTE: _set_best_params() must be implemented in the child class.
        if not hasattr(self, "_set_best_params"):
            msg = "Method `_set_best_params()` must be implemented in the child class."
            self.logger.error(msg)
            raise NotImplementedError(msg)

        study.optimize(
            lambda trial: self._objective(trial),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            gc_after_trial=True,
            show_progress_bar=show_progress_bar,
        )

        try:
            best_metric = study.best_value
            best_params = study.best_params
        except Exception:
            msg = "Tuning failed: No successful trials completed."
            self.logger.error(msg)
            raise RuntimeError(msg)

        self.best_params_ = self._set_best_params(best_params)
        self.model_params.update(self.best_params_)
        self.logger.info(f"Best {self.tune_metric} metric: {best_metric}")
        self.logger.info("Best parameters:")
        best_params_tmp = copy.deepcopy(best_params)
        best_params_tmp["learning_rate"] = self.learning_rate

        tn = f"{self.tune_metric} Value"

        if self.show_plots:
            self.plotter_.plot_tuning(
                study, self.model_name, self.optimize_dir / "plots", target_name=tn
            )

        return best_params_tmp

    @staticmethod
    def initialize_weights(module: torch.nn.Module) -> None:
        """Initializes model weights using Xavier/Glorot Uniform distribution.

        Switching from Kaiming to Xavier is safer for deep VAEs to prevent
        exploding gradients or dead neurons in the early epochs.
        """
        if isinstance(
            module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.ConvTranspose1d)
        ):
            # Xavier is generally more stable for VAEs than Kaiming
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def build_model(
        self,
        Model: Type[torch.nn.Module],
        model_params: Dict[str, Any],
    ) -> torch.nn.Module:
        """Builds and initializes a neural network model instance.

        This method instantiates a model by combining fixed, data-dependent parameters (like `n_features`) with variable hyperparameters (like `latent_dim`). The resulting model is then moved to the appropriate compute device.

        Args:
            Model (torch.nn.Module): The model class to be instantiated.
            model_params (Dict[str, Any]): A dictionary of variable model hyperparameters, typically sampled during a hyperparameter search.

        Returns:
            torch.nn.Module: The constructed model instance, ready for training.

        Raises:
            TypeError: If `model_params` is not a dictionary.
            AttributeError: If a required data-dependent attribute like `num_features_` has not been set, typically by calling `fit` first.
        """
        if not isinstance(model_params, dict):
            msg = f"'model_params' must be a dictionary, but got {type(model_params)}."
            self.logger.error(msg)
            raise TypeError(msg)

        if not hasattr(self, "num_features_"):
            msg = (
                "Attribute 'num_features_' is not set. Call fit() before build_model()."
            )
            self.logger.error(msg)
            raise AttributeError(msg)

        all_params = {
            "n_features": self.num_features_,
            "prefix": self.prefix,
            "num_classes": self.num_classes_,
            "verbose": self.verbose,
            "debug": self.debug,
            "device": self.device,
        }

        # Update with the variable hyperparameters
        all_params.update(model_params)

        return Model(**all_params).to(self.device)

    def initialize_plotting_and_scorers(self) -> Tuple[Plotting, Scorer]:
        """Initializes and returns the plotting and scoring utility classes.

        This method should be called within a `fit` method to set up the standardized utilities for generating plots and calculating performance metrics.

        Returns:
            Tuple[Plotting, Scorer]: A tuple containing the initialized Plotting and Scorer objects.
        """
        fmt = self.plot_format

        # Initialize plotter.
        plotter = Plotting(
            model_name=self.model_name,
            prefix=self.prefix,
            plot_format=fmt,
            plot_fontsize=self.plot_fontsize,
            plot_dpi=self.plot_dpi,
            title_fontsize=self.title_fontsize,
            despine=self.despine,
            show_plots=self.show_plots,
            verbose=self.verbose,
            debug=self.debug,
            multiqc=True,
            multiqc_section=f"PG-SUI: {self.model_name} Model Imputation",
        )

        # Metrics
        scorers = Scorer(
            prefix=self.prefix,
            average=self.scoring_averaging,
            verbose=self.verbose,
            debug=self.debug,
        )

        return plotter, scorers

    def _objective(self, trial: optuna.Trial) -> float:
        """Defines the objective function for Optuna hyperparameter tuning.

        This abstract method must be implemented by the child class. It should define a single hyperparameter tuning trial, which typically involves building, training, and evaluating a model with a set of sampled hyperparameters.

        Args:
            trial (optuna.Trial): The Optuna trial object, used to sample hyperparameters.

        Returns:
            float: The value of the metric to be optimized (e.g., validation accuracy, F1-score).
        """
        msg = "Method `_objective()` must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def fit(self, X: np.ndarray | pd.DataFrame | list | None = None) -> "BaseNNImputer":
        """Fits the imputer model to the data.

        This abstract method must be implemented by the child class. It should contain the logic for training the neural network model on the provided input data `X`.

        Args:
            X (np.ndarray | pd.DataFrame | list | None): The input data, which may contain missing values.

        Returns:
            BaseNNImputer: The fitted imputer instance.
        """
        msg = "Method ``fit()`` must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def transform(
        self, X: np.ndarray | pd.DataFrame | list | None = None
    ) -> np.ndarray:
        """Imputes missing values in the data using the trained model.

        This abstract method must be implemented by the child class. It should use the fitted model to fill in missing values in the provided data `X`.

        Args:
            X (np.ndarray | pd.DataFrame | list | None): The input data with missing values.

        Returns:
            np.ndarray: IUPAC strings with missing values imputed.
        """
        msg = "Method ``transform()`` must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def _select_device(self, device: Literal["gpu", "cpu", "mps"]) -> torch.device:
        """Selects the appropriate PyTorch device based on user preference and availability.

        This method checks the user's device preference ('gpu', 'cpu', or 'mps') and verifies if the requested hardware is available. If the preferred device is not available, it falls back to CPU and logs a warning.

        Args:
            device (Literal["gpu", "cpu", "mps"]): The preferred device type for PyTorch operations.

        Returns:
            torch.device: The selected PyTorch device.
        """
        dvc = device.lower().strip()
        if dvc == "cpu":
            return torch.device("cpu")
        if dvc == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _create_model_directories(
        self, prefix: str, outdirs: List[str], *, outdir: Path | str | None = None
    ) -> None:
        """Creates the directory structure for storing model outputs.

        This method sets up a standardized folder hierarchy for saving models, plots, metrics, and optimization results, organized under a main directory named after the provided prefix.

        Args:
            prefix (str): The prefix for the main output directory.
            outdirs (List[str]): A list of subdirectory names to create within the main directory.
            outdir (Path | str | None): The base output directory. If None, uses the current working directory. Defaults to None.

        Raises:
            Exception: If any of the directories cannot be created.
        """
        base_root = Path(outdir) if outdir is not None else Path.cwd()
        formatted_output_dir = base_root / f"{prefix}_output"
        formatted_output_dir = Path(f"{prefix}_output")
        base_dir = formatted_output_dir / "Unsupervised"

        for d in outdirs:
            subdir = base_dir / d / self.model_name
            setattr(self, f"{d}_dir", subdir)
            try:
                getattr(self, f"{d}_dir").mkdir(parents=True, exist_ok=True)
            except Exception as e:
                msg = f"Failed to create directory {getattr(self, f'{d}_dir')}: {e}"
                self.logger.error(msg)
                raise Exception(msg)

    def _clear_resources(self, model: torch.nn.Module) -> None:
        """Releases GPU and CPU memory after an Optuna trial.

        This is a crucial step during hyperparameter tuning to prevent memory leaks between trials, ensuring that each trial runs in a clean environment.

        Args:
            model (torch.nn.Module): The model from the completed trial.
        """
        try:
            del model
        except NameError:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def _make_eval_visualizations(
        self,
        labels: List[str],
        y_pred_proba: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Dict[str, float],
        msg: str,
    ):
        """Generate and save evaluation visualizations.

        3-class (zygosity) or 10-class (IUPAC) depending on `labels` length.

        Args:
            labels (List[str]): Class label names.
            y_pred_proba (np.ndarray): Predicted probabilities (2D array).
            y_true (np.ndarray): True labels (1D array).
            y_pred (np.ndarray): Predicted labels (1D array).
            metrics (Dict[str, float]): Computed metrics.
            msg (str): Message to log before generating plots.
        """
        self.logger.info(msg)

        prefix = "zygosity" if len(labels) == 3 else "iupac"
        n_labels = len(labels)

        if self.show_plots:
            self.plotter_.plot_metrics(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                metrics=metrics,
                label_names=labels,
                prefix=f"geno{n_labels}_{prefix}",
            )
            self.plotter_.plot_confusion_matrix(
                y_true_1d=y_true,
                y_pred_1d=y_pred,
                label_names=labels,
                prefix=f"geno{n_labels}_{prefix}",
            )

    def _additional_metrics(self, y_true, y_pred, labels, report_names, report):
        """Compute additional metrics and augment the report dictionary.

        Args:
            y_true (np.ndarray): True genotypes.
            y_pred (np.ndarray): Predicted genotypes.
            labels (list[int]): List of label indices.
            report_names (list[str]): List of report names corresponding to labels.
            report (dict): Classification report dictionary to augment.

        Returns:
            dict[str, dict[str, float] | float]: Augmented report dictionary with additional metrics.
        """
        # Create an identity matrix and use the targets array as indices
        y_score = np.eye(len(report_names))[y_pred]

        # Per-class metrics
        ap_pc = average_precision_score(y_true, y_score, average=None)
        jaccard_pc = jaccard_score(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        )

        # Macro/weighted metrics
        ap_macro = average_precision_score(y_true, y_score, average="macro")
        ap_weighted = average_precision_score(y_true, y_score, average="weighted")
        jaccard_macro = jaccard_score(y_true, y_pred, average="macro", zero_division=0)
        jaccard_weighted = jaccard_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Matthews correlation coefficient (MCC)
        mcc = matthews_corrcoef(y_true, y_pred)

        if not isinstance(ap_pc, np.ndarray):
            msg = "average_precision_score or f1_score did not return np.ndarray as expected."
            self.logger.error(msg)
            raise TypeError(msg)

        if not isinstance(jaccard_pc, np.ndarray):
            msg = "jaccard_score did not return np.ndarray as expected."
            self.logger.error(msg)
            raise TypeError(msg)

        # Add per-class metrics
        report_full = {}
        for i, class_name in enumerate(report_names):
            class_report = report.get(class_name)

            if isinstance(class_report, float):
                continue

            report_full[class_name] = dict(class_report)
            report_full[class_name]["average-precision"] = float(ap_pc[i])
            report_full[class_name]["jaccard"] = float(jaccard_pc[i])

        macro_avg = report.get("macro avg")
        if isinstance(macro_avg, dict):
            report_full["macro avg"] = dict(macro_avg)
            report_full["macro avg"]["average-precision"] = float(ap_macro)
            report_full["macro avg"]["jaccard"] = float(jaccard_macro)

        weighted_avg = report.get("weighted avg")
        if isinstance(weighted_avg, dict):
            report_full["weighted avg"] = dict(weighted_avg)
            report_full["weighted avg"]["average-precision"] = float(ap_weighted)
            report_full["weighted avg"]["jaccard"] = float(jaccard_weighted)

        # Add scalar summary metrics
        report_full["mcc"] = float(mcc)
        accuracy_val = report.get("accuracy")

        if isinstance(accuracy_val, (int, float)):
            report_full["accuracy"] = float(accuracy_val)

        return report_full

    def _make_class_reports(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Dict[str, float],
        y_pred_proba: np.ndarray | None = None,
        labels: List[str] = ["REF", "HET", "ALT"],
    ) -> None:
        """Generate and save detailed classification reports and visualizations.

        3-class (zygosity) or 10-class (IUPAC) depending on `labels` length.

        Args:
            y_true (np.ndarray): True labels (1D array).
            y_pred (np.ndarray): Predicted labels (1D array).
            metrics (Dict[str, float]): Computed metrics.
            y_pred_proba (np.ndarray | None): Predicted probabilities (2D array). Defaults to None.
            labels (List[str]): Class label names (default: ["REF", "HET", "ALT"] for 3-class).
        """
        report_name = "zygosity" if len(labels) <= 3 else "iupac"
        middle = "IUPAC" if report_name == "iupac" else "Zygosity"

        msg = f"{middle} Report (on {y_pred.size} total genotypes)"
        self.logger.info(msg)

        if y_pred_proba is not None:
            if self.show_plots:
                self.plotter_.plot_metrics(
                    y_true,
                    y_pred_proba,
                    metrics,
                    label_names=labels,
                    prefix=report_name,
                )

            if self.show_plots:
                self.plotter_.plot_confusion_matrix(
                    y_true, y_pred, label_names=labels, prefix=report_name
                )

        report: str | dict = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
            output_dict=True,
        )

        if not isinstance(report, dict):
            msg = "Expected classification_report to return a dict."
            self.logger.error(msg, exc_info=True)
            raise ValueError(msg)

        if self.show_plots:
            viz = ClassificationReportVisualizer(reset_kwargs=self.plotter_.param_dict)
            try:
                plots = viz.plot_all(
                    report,  # type: ignore
                    title_prefix=f"{self.model_name} {middle} Report",
                    show=self.show_plots,
                    heatmap_classes_only=True,
                )
            finally:
                viz._reset_mpl_style()

            for name, fig in plots.items():
                fout = (
                    self.plots_dir / f"{report_name}_report_{name}.{self.plot_format}"
                )
                if hasattr(fig, "savefig") and isinstance(fig, Figure):
                    fig.savefig(fout, dpi=300, facecolor="#111122")
                    plt.close(fig)
                elif hasattr(fig, "write_html") and isinstance(fig, go.Figure):
                    fout_html = fout.with_suffix(".html")
                    fig.write_html(file=fout_html)

                    SNPioMultiQC.queue_html(
                        fout_html,
                        panel_id=f"pgsui_{self.model_name.lower()}_{report_name}_radar",
                        section=f"PG-SUI: {self.model_name} Model Imputation",
                        title=f"{self.model_name} {middle} Radar Plot",
                        index_label=name,
                        description=f"{self.model_name} {middle} {len(labels)}-base Radar Plot. This radar plot visualizes model performance for three metrics per-class: precision, recall, and F1-score. Each axis represents one of these metrics, allowing for a quick visual assessment of the model's strengths and weaknesses. Higher values towards the outer edge indicate better performance.",
                    )

            if not self.is_haploid_:
                msg = f"Ploidy: {self.ploidy}. Evaluating per genotype (REF, HET, ALT)."
                self.logger.info(msg)

        report_full = self._additional_metrics(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            report_names=labels,
            report=report,
        )

        if self.verbose or self.debug:
            pm = PrettyMetrics(
                report_full,
                precision=2,
                title=f"{self.model_name} {middle} Report",
            )
            pm.render()

        with open(self.metrics_dir / f"{report_name}_report.json", "w") as f:
            json.dump(report, f, indent=4)

    def _compute_hidden_layer_sizes(
        self,
        n_inputs: int,
        n_outputs: int,
        n_samples: int,
        n_hidden: int,
        latent_dim: int,
        *,
        alpha: float = 4.0,
        schedule: str = "pyramid",
        min_size: int = 16,
        max_size: int | None = None,
        multiple_of: int = 8,
        decay: float | None = None,
        cap_by_inputs: bool = True,
    ) -> list[int]:
        """Compute hidden layer sizes given problem scale and a layer count.

        Notes:
            - Returns sizes for *hidden layers only* (length = n_hidden).
            - Does NOT include the input layer (n_inputs) or the latent layer (latent_dim).
            - Enforces a latent-aware minimum: one discrete level above latent_dim, where a level is `multiple_of`.
            - Enforces *strictly decreasing* hidden sizes (no repeats). This may require bumping `base` upward.

        Args:
            n_inputs: Number of input features (e.g., flattened one-hot: num_features * num_classes).
            n_outputs: Number of output classes (often equals num_classes).
            n_samples: Number of training samples.
            n_hidden: Number of hidden layers (excluding input and latent layers).
            latent_dim: Latent dimensionality (not returned, used only to set a floor).
            alpha: Scaling factor for base layer size.
            schedule: Size schedule ("pyramid" or "linear").
            min_size: Minimum layer size floor before latent-aware adjustment.
            max_size: Maximum layer size cap. If None, a heuristic cap is used.
            multiple_of: Hidden sizes are multiples of this value.
            decay: Pyramid decay factor. If None, computed to land near the target.
            cap_by_inputs: If True, cap layer sizes to n_inputs.

        Returns:
            list[int]: Hidden layer sizes (len = n_hidden).

        Raises:
            ValueError: On invalid arguments or conflicting constraints.
        """
        # ----------------------------
        # Basic validation
        # ----------------------------
        if n_hidden < 0:
            msg = f"n_hidden must be >= 0, got {n_hidden}."
            self.logger.error(msg)
            raise ValueError(msg)

        if n_hidden == 0:
            return []

        if n_inputs <= 0:
            msg = f"n_inputs must be > 0, got {n_inputs}."
            self.logger.error(msg)
            raise ValueError(msg)

        if n_outputs <= 0:
            msg = f"n_outputs must be > 0, got {n_outputs}."
            self.logger.error(msg)
            raise ValueError(msg)

        if n_samples <= 0:
            msg = f"n_samples must be > 0, got {n_samples}."
            self.logger.error(msg)
            raise ValueError(msg)

        if latent_dim <= 0:
            msg = f"latent_dim must be > 0, got {latent_dim}."
            self.logger.error(msg)
            raise ValueError(msg)

        if multiple_of <= 0:
            msg = f"multiple_of must be > 0, got {multiple_of}."
            self.logger.error(msg)
            raise ValueError(msg)

        if alpha <= 0:
            msg = f"alpha must be > 0, got {alpha}."
            self.logger.error(msg)
            raise ValueError(msg)

        schedule = str(schedule).lower().strip()
        if schedule not in {"pyramid", "linear"}:
            msg = f"Invalid schedule '{schedule}'. Must be 'pyramid' or 'linear'."
            self.logger.error(msg)
            raise ValueError(msg)

        # ----------------------------
        # Latent-aware minimum floor
        # ----------------------------
        # Smallest multiple_of strictly greater than latent_dim
        min_hidden_floor = int(np.ceil((latent_dim + 1) / multiple_of) * multiple_of)
        effective_min = max(int(min_size), min_hidden_floor)

        if cap_by_inputs and n_inputs < effective_min:
            msg = (
                "Cannot satisfy latent-aware minimum hidden size with cap_by_inputs=True. "
                f"Required hidden size >= {effective_min} (one level above latent_dim={latent_dim}), "
                f"but n_inputs={n_inputs}. Set cap_by_inputs=False or reduce latent_dim/multiple_of."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # ----------------------------
        # Infer num_features (if using flattened one-hot: n_inputs = num_features * num_classes)
        # ----------------------------
        if n_inputs % n_outputs == 0:
            num_features = n_inputs // n_outputs
        else:
            num_features = n_inputs
            self.logger.warning(
                "n_inputs is not divisible by n_outputs; falling back to num_features=n_inputs "
                f"(n_inputs={n_inputs}, n_outputs={n_outputs}). If using one-hot flattening, "
                "pass n_outputs=num_classes so num_features can be inferred correctly."
            )

        # ----------------------------
        # Base size heuristic (feature-matrix aware; avoids collapse for huge n_inputs)
        # ----------------------------
        obs_scale = (float(n_samples) * float(num_features)) / float(
            num_features + n_outputs
        )
        base = int(np.ceil(float(alpha) * np.sqrt(obs_scale)))

        # ----------------------------
        # Determine max_size
        # ----------------------------
        if max_size is None:
            max_size = max(int(n_inputs), int(base), int(effective_min))

        if cap_by_inputs:
            max_size = min(int(max_size), int(n_inputs))
        else:
            max_size = int(max_size)

        if max_size < effective_min:
            msg = (
                f"max_size ({max_size}) must be >= effective_min ({effective_min}), where effective_min "
                f"is max(min_size={min_size}, one-level-above latent_dim={latent_dim})."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Round base up to a multiple and clip to bounds
        base = int(np.clip(base, effective_min, max_size))
        base = int(np.ceil(base / multiple_of) * multiple_of)
        base = int(np.clip(base, effective_min, max_size))

        # ----------------------------
        # Enforce "no repeats" feasibility in discrete levels
        # Need n_hidden distinct multiples between base and effective_min:
        # base >= effective_min + (n_hidden - 1) * multiple_of
        # ----------------------------
        required_min_base = effective_min + (n_hidden - 1) * multiple_of

        if required_min_base > max_size:
            msg = (
                "Cannot build strictly-decreasing (no-repeat) hidden sizes under current constraints. "
                f"Need base >= {required_min_base} to fit n_hidden={n_hidden} distinct layers "
                f"with multiple_of={multiple_of} down to effective_min={effective_min}, "
                f"but max_size={max_size}. Reduce n_hidden, reduce multiple_of, lower latent_dim/min_size, "
                "or increase max_size / set cap_by_inputs=False."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        if base < required_min_base:
            # Bump base upward so a strict staircase is possible
            base = required_min_base
            base = int(np.ceil(base / multiple_of) * multiple_of)
            base = int(np.clip(base, effective_min, max_size))

        # Work in "levels" of multiple_of for guaranteed uniqueness
        start_level = base // multiple_of
        end_level = effective_min // multiple_of

        # Sanity: distinct levels available
        if (start_level - end_level) < (n_hidden - 1):
            # This should not happen due to required_min_base logic, but keep a hard guard.
            msg = (
                "Internal constraint failure: insufficient discrete levels to enforce no repeats. "
                f"start_level={start_level}, end_level={end_level}, n_hidden={n_hidden}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # ----------------------------
        # Build schedule in level space (integers), then convert to sizes
        # ----------------------------
        if n_hidden == 1:
            levels = np.array([start_level], dtype=int)

        elif schedule == "linear":
            # Linear interpolation in level space, then strictify
            levels = np.round(np.linspace(start_level, end_level, num=n_hidden)).astype(
                int
            )

            # Enforce bounds then strict decrease
            levels = np.clip(levels, end_level, start_level)

            for i in range(1, n_hidden):
                if levels[i] >= levels[i - 1]:
                    levels[i] = levels[i - 1] - 1

            if levels[-1] < end_level:
                msg = (
                    "Failed to enforce strictly-decreasing linear schedule without violating the floor. "
                    f"(levels[-1]={levels[-1]} < end_level={end_level}). "
                    "Reduce n_hidden or multiple_of, or increase max_size."
                )
                self.logger.error(msg)
                raise ValueError(msg)

            # Force exact floor at the end (still strict because we have enough room by construction)
            levels[-1] = end_level
            for i in range(n_hidden - 2, -1, -1):
                if levels[i] <= levels[i + 1]:
                    levels[i] = levels[i + 1] + 1

            if levels[0] > start_level:
                # If this happens, we would need an even larger base; handle by raising base once.
                needed_base = int(levels[0] * multiple_of)
                if needed_base > max_size:
                    msg = (
                        "Cannot enforce strictly-decreasing linear schedule after floor anchoring; "
                        f"would require base={needed_base} > max_size={max_size}."
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)
                # Rebuild with bumped base
                start_level = needed_base // multiple_of
                levels = np.arange(start_level, start_level - n_hidden, -1, dtype=int)
                levels[-1] = end_level  # keep floor
                # Ensure strict with backward adjust
                for i in range(n_hidden - 2, -1, -1):
                    if levels[i] <= levels[i + 1]:
                        levels[i] = levels[i + 1] + 1

        elif schedule == "pyramid":
            # Geometric decay in level space (more aggressive early taper than linear)
            if decay is not None:
                dcy = float(decay)
            else:
                # Choose decay to land exactly at end_level (in float space)
                dcy = (float(end_level) / float(start_level)) ** (
                    1.0 / float(n_hidden - 1)
                )

            # Keep it in a sensible range
            dcy = float(np.clip(dcy, 0.05, 0.99))

            exponents = np.arange(n_hidden, dtype=float)
            levels_float = float(start_level) * (dcy**exponents)

            levels = np.round(levels_float).astype(int)
            levels = np.clip(levels, end_level, start_level)

            # Anchor the last layer at the floor, then strictify backward
            levels[-1] = end_level
            for i in range(n_hidden - 2, -1, -1):
                if levels[i] <= levels[i + 1]:
                    levels[i] = levels[i + 1] + 1

            # If we overshot the start, bump base (once) if possible, then rebuild
            if levels[0] > start_level:
                needed_base = int(levels[0] * multiple_of)
                if needed_base > max_size:
                    msg = (
                        "Cannot enforce strictly-decreasing pyramid schedule; "
                        f"would require base={needed_base} > max_size={max_size}."
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)

                start_level = needed_base // multiple_of
                # Recompute with new start_level and same decay (or recompute decay if decay is None)
                if decay is None:
                    dcy = (float(end_level) / float(start_level)) ** (
                        1.0 / float(n_hidden - 1)
                    )
                    dcy = float(np.clip(dcy, 0.05, 0.99))

                levels_float = float(start_level) * (dcy**exponents)
                levels = np.round(levels_float).astype(int)
                levels = np.clip(levels, end_level, start_level)
                levels[-1] = end_level
                for i in range(n_hidden - 2, -1, -1):
                    if levels[i] <= levels[i + 1]:
                        levels[i] = levels[i + 1] + 1

        else:
            msg = f"Unknown schedule '{schedule}'. Use 'pyramid' or 'linear' (constant disallowed with no repeats)."
            self.logger.error(msg)
            raise ValueError(msg)

        # Convert levels -> sizes
        sizes = (levels * multiple_of).astype(int)

        # Final clip (should be redundant, but safe)
        sizes = np.clip(sizes, effective_min, max_size).astype(int)

        # Final strict no-repeat assertion
        if np.any(np.diff(sizes) >= 0):
            msg = (
                "Internal error: produced non-decreasing or repeated hidden sizes after strict enforcement. "
                f"sizes={sizes.tolist()}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        return sizes.tolist()

    def _class_weights_from_zygosity(
        self,
        X: np.ndarray,
        train_mask: Optional[np.ndarray] = None,
        *,
        inverse: bool = False,
        normalize: bool = False,
        power: float = 1.0,
        max_ratio: float | None = None,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Compute class weights for zygosity labels.

        If inverse=False (default):
            w_c = N / (K * n_c)   ("balanced")

        If inverse=True:
            w_c = N / n_c         (same ratios, scaled by K)

        If power != 1.0:
            w_c <- w_c ** power   (amplifies or softens imbalance handling)

        If normalize=True:
            rescales nonzero weights so mean(nonzero_weights) == 1.

        Returns:
            torch.Tensor: Class weights of shape (num_classes,) on self.device.
        """
        y = np.asarray(X).ravel().astype(np.int8)

        m = y >= 0
        if train_mask is not None:
            tm = np.asarray(train_mask, dtype=bool).ravel()
            if tm.shape != y.shape:
                msg = "train_mask must have the same shape as X."
                self.logger.error(msg)
                raise ValueError(msg)
            m &= tm

        is_hap = bool(getattr(self, "is_haploid_", False))
        num_classes = 2 if is_hap else int(self.num_classes_)

        if not np.any(m):
            return torch.ones(num_classes, dtype=torch.long, device=self.device)

        if is_hap:
            y = y.copy()
            y[(y == 2) & m] = 1

        y_m = y[m]
        if y_m.size:
            ymin = int(y_m.min())
            ymax = int(y_m.max())
            if ymin < 0 or ymax >= num_classes:
                msg = (
                    f"Found out-of-range labels under mask: min={ymin}, max={ymax}, "
                    f"expected in [0, {num_classes - 1}]."
                )
                self.logger.error(msg)
                raise ValueError(msg)

        counts = np.bincount(y_m, minlength=num_classes).astype(np.float32)
        N = float(counts.sum())
        K = float(num_classes)

        w = np.zeros(num_classes, dtype=np.float32)
        nz = counts > 0

        if np.any(nz):
            if inverse:
                w[nz] = N / (counts[nz] + eps)
            else:
                w[nz] = N / (K * (counts[nz] + eps))

            # Amplify / soften class contrast
            if power <= 0.0:
                msg = "power must be > 0."
                self.logger.error(msg)
                raise ValueError(msg)
            if power != 1.0:
                w[nz] = np.power(w[nz], power)

        if np.any(~nz):
            self.logger.warning(
                "Some classes have zero count under the provided mask: "
                f"{np.where(~nz)[0].tolist()}. Setting their weights to 0."
            )

        # Cap ratio among observed classes
        if max_ratio is not None and np.any(nz):
            cap = float(max_ratio)
            if cap <= 1.0:
                msg = "max_ratio must be > 1.0 or None."
                self.logger.error(msg)
                raise ValueError(msg)

            wmin = max(float(w[nz].min()), eps)
            wmax = wmin * cap
            w[nz] = np.clip(w[nz], wmin, wmax)

        # Optional normalization: mean(nonzero) -> 1.0
        if normalize and np.any(nz):
            mean_nz = float(w[nz].mean())
            if mean_nz > 0.0:
                w[nz] /= mean_nz
            else:
                self.logger.warning(
                    "normalize=True requested, but mean of nonzero weights is not positive; skipping normalization."
                )

        self.logger.debug(f"Class counts: {counts.astype(np.int8)}")
        self.logger.debug(
            f"Class weights (inverse={inverse}, power={power}, normalize={normalize}): {w}"
        )

        return torch.as_tensor(w, dtype=torch.long, device=self.device)

    def _one_hot_encode_012(
        self, X: np.ndarray | torch.Tensor, num_classes: int | None
    ) -> torch.Tensor:
        """One-hot encode genotype calls. Missing inputs (<0) result in a vector of -1s.

        Args:
            X (np.ndarray | torch.Tensor): Input genotype calls as integers (0,1, 2, etc.).
            num_classes (int | None): Number of classes (K). If None, uses self.num_classes_.

        Returns:
            torch.Tensor: One-hot encoded tensor of shape (B, L, K) with float32 dtype. Valid calls are 0/1, missing calls are all -1.

        Notes:
            - Valid classes must be integers in [0, K-1]
            - Missing is any value < 0; these positions become [-1, -1, ..., -1]
            - If K==2 and values are in {0,2} (no 1s), map 2->1.
        """
        Xt = (
            torch.from_numpy(X).to(self.device)
            if isinstance(X, np.ndarray)
            else X.to(self.device)
        )

        # Make sure we have integer class labels
        if Xt.dtype.is_floating_point:
            # Convert NaN -> -1 and cast to long
            Xt = torch.nan_to_num(Xt, nan=-1.0).long()
        else:
            Xt = Xt.long()

        B, L = Xt.shape
        K = int(num_classes) if num_classes is not None else int(self.num_classes_)

        # Missing is anything < 0 (covers -1, -9, etc.)
        valid = Xt >= 0

        # If binary mode but data is {0,2}
        # (haploid-like or "ref vs non-ref"), map 2->1
        if K == 2:
            has_het = torch.any(valid & (Xt == 1))
            has_alt2 = torch.any(valid & (Xt == 2))
            if has_alt2 and not has_het:
                Xt = Xt.clone()
                Xt[valid & (Xt == 2)] = 1

        # Now enforce the one-hot precondition
        if torch.any(valid & (Xt >= K)):
            bad_vals = torch.unique(Xt[valid & (Xt >= K)]).detach().cpu().tolist()
            all_vals = torch.unique(Xt[valid]).detach().cpu().tolist()
            msg = f"_one_hot_encode_012 received class values outside [0, {K-1}]. num_classes={K}, offending_values={bad_vals}, observed_values={all_vals}. Upstream encoding mismatch (e.g., passing 0/1/2 with num_classes=2)."
            self.logger.error(msg)
            raise ValueError(msg)

        # CHANGE: Initialize with -1.0 to ensure missing values are represented as [-1, -1, ... -1]
        X_ohe = torch.full((B, L, K), -1.0, dtype=torch.long, device=self.device)

        idx = Xt[valid]

        if idx.numel() > 0:
            # Overwrite valid positions (which were -1) with the correct one-hot vectors
            X_ohe[valid] = F.one_hot(idx, num_classes=K).long()

        return X_ohe

    def decode_012(
        self, X: np.ndarray | pd.DataFrame | list[list[int]], is_nuc: bool = False
    ) -> np.ndarray:
        """Decode 012-encodings to IUPAC chars with metadata repair.

        This method converts genotype calls encoded as integers (0, 1, 2, etc.) into their corresponding IUPAC nucleotide codes. It supports two modes of decoding:
        1. Nucleotide mode (`is_nuc=True`): Decodes integer codes (0-9) directly to IUPAC nucleotide codes.
        2. Metadata mode (`is_nuc=False`): Uses reference and alternate allele metadata to determine the appropriate IUPAC codes.

        Args:
            X (np.ndarray | pd.DataFrame | list[list[int]]): Input genotype calls as integers.
            is_nuc (bool): If True, decode 0-9 nucleotide codes; else use ref/alt metadata. Defaults to False.

        Returns:
            np.ndarray: IUPAC strings as a 2D array of shape (n_samples, n_snps).

        Raises:
            ValueError: If input is not a DataFrame.
        """
        df = validate_input_type(X, return_type="df")
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Expected DataFrame.")

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

            # Bytes -> str (make type narrowing explicit)
            if isinstance(value, (bytes, np.bytes_)):
                value = bytes(value).decode("utf-8", errors="ignore")

            # Handle list/tuple/array/Series: take first valid
            if isinstance(value, (list, tuple, pd.Series, np.ndarray)):
                # Convert Series to numpy array for consistent behavior
                if isinstance(value, pd.Series):
                    arr = value.to_numpy()
                else:
                    arr = value

                # Scalar numpy array fast path
                if isinstance(arr, np.ndarray) and arr.ndim == 0:
                    return _normalize_iupac(arr.item())

                # Empty sequence/array
                if len(arr) == 0:
                    return None

                # First valid element wins
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

        # Ensure list length matches
        if len(ref_alleles) != n_cols:
            ref_alleles = [None] * n_cols
        if len(alt_alleles) != n_cols:
            alt_alleles = [None] * n_cols

        out = np.full((n_rows, n_cols), "N", dtype="<U1")
        source_snp_data = None

        for j in range(n_cols):
            ref = _normalize_iupac(ref_alleles[j])
            alt = _normalize_iupac(alt_alleles[j])

            # --- REPAIR LOGIC ---
            # If metadata is missing, scan the source column.
            if ref is None or alt is None:
                if source_snp_data is None and self.genotype_data.snp_data is not None:
                    try:
                        source_snp_data = np.asarray(self.genotype_data.snp_data)
                    except Exception:
                        pass  # if lazy loading fails

                if source_snp_data is not None:
                    try:
                        col_data = source_snp_data[:, j]
                        uniques = set()
                        # Optimization: check up to 200 non-empty values
                        count = 0
                        for val in col_data:
                            norm = _normalize_iupac(val)
                            if norm:
                                uniques.add(norm)
                                count += 1
                            if len(uniques) >= 2 or count > 200:
                                break

                        sorted_u = sorted(list(uniques))
                        if len(sorted_u) >= 1 and ref is None:
                            ref = sorted_u[0]
                        if len(sorted_u) >= 2 and alt is None:
                            alt = sorted_u[1]
                    except Exception:
                        pass

            # --- DEFAULTS FOR MISSING ---
            # If still missing, we cannot decode.
            if ref is None and alt is None:
                ref = "N"
                alt = "N"
            elif ref is None:
                ref = alt
            elif alt is None:
                alt = ref  # Monomorphic site: ALT becomes REF

            # --- COMPUTE HET CODE ---
            if ref == alt:
                het_code = ref
            else:
                ref_set = iupac_to_bases.get(ref, set()) if ref is not None else set()
                alt_set = iupac_to_bases.get(alt, set()) if alt is not None else set()
                union_set = frozenset(ref_set | alt_set)
                het_code = bases_to_iupac.get(union_set, "N")

            # --- ASSIGNMENT WITH SAFETY FALLBACKS ---
            col_codes = codes[:, j]

            # Case 0: REF
            if ref != "N":
                out[col_codes == 0, j] = ref

            # Case 1: HET
            if het_code != "N":
                out[col_codes == 1, j] = het_code
            else:
                # If HET code is invalid (e.g. ref='A', alt='N'),
                # fallback to REF
                # Fix for an issue where a HET prediction at a monomorphic site
                # produced 'N'
                if ref != "N":
                    out[col_codes == 1, j] = ref

            # Case 2: ALT
            if alt != "N":
                out[col_codes == 2, j] = alt
            else:
                # If ALT is invalid (e.g. ref='A', alt='N'), fallback to REF
                # Fix for an issue where an ALT prediction on a monomorphic site
                # produced 'N'
                if ref != "N":
                    out[col_codes == 2, j] = ref

        return out

    def _save_best_params(
        self, best_params: Dict[str, Any], objective_mode: bool = False
    ) -> None:
        """Save the best hyperparameters to a JSON file.

        This method saves the best hyperparameters found during hyperparameter tuning to a JSON file in the optimization directory. The filename includes the model name for easy identification.

        Args:
            best_params (Dict[str, Any]): A dictionary of the best hyperparameters to save.
        """
        if not hasattr(self, "parameters_dir"):
            msg = "Attribute 'parameters_dir' not found. Ensure _create_model_directories() has been called."
            self.logger.error(msg)
            raise AttributeError(msg)

        if objective_mode:
            fout = self.optimize_dir / "parameters" / "best_tuned_parameters.json"
        else:
            fout = self.parameters_dir / "best_parameters.json"

        fout.parent.mkdir(parents=True, exist_ok=True)

        with open(fout, "w") as f:
            json.dump(best_params, f, indent=4)

    def _set_best_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """An abstract method for setting best parameters."""
        raise NotImplementedError

    def sim_missing_transform(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate missing data according to the specified strategy.

        Args:
            X (np.ndarray): Genotype matrix to simulate missing data on.

        Returns:
            X_for_model (np.ndarray): Genotype matrix with simulated missing data.
            sim_mask (np.ndarray): Boolean mask of simulated missing entries.
            orig_mask (np.ndarray): Boolean mask of original missing entries.
        """
        if (
            not hasattr(self, "sim_prop")
            or self.sim_prop <= 0.0
            or self.sim_prop >= 1.0
        ):
            msg = "sim_prop must be set and between 0.0 and 1.0."
            self.logger.error(msg)
            raise AttributeError(msg)

        if not hasattr(self, "tree_parser") and "nonrandom" in self.sim_strategy:
            msg = "tree_parser must be set for 'nonrandom' or 'nonrandom_weighted' sim_strategy."
            self.logger.error(msg)
            raise AttributeError(msg)

        # --- Simulate missing data ---
        X_for_sim = X.astype(np.float32, copy=True)
        tr = SimMissingTransformer(
            genotype_data=self.genotype_data,
            tree_parser=self.tree_parser,
            prop_missing=self.sim_prop,
            strategy=self.sim_strategy,
            missing_val=-1,
            mask_missing=True,
            verbose=self.verbose,
            seed=self.seed,
            **self.sim_kwargs,
        )
        tr.fit(X_for_sim.copy())
        X_for_model = tr.transform(X_for_sim.copy())
        sim_mask = tr.sim_missing_mask_.astype(bool)
        orig_mask = tr.original_missing_mask_.astype(bool)

        return X_for_model, sim_mask, orig_mask

    def _train_val_test_split(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train, validation, and test sets.

        Args:
            X (np.ndarray): Genotype matrix to split.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Indices for train, validation, and test sets.

        Raises:
            ValueError: If there are not enough samples for splitting.
            AssertionError: If validation_split is not in (0.0, 1.0).
        """
        n_samples = X.shape[0]

        if n_samples < 3:
            msg = f"Not enough samples ({n_samples}) for train/val/test split."
            self.logger.error(msg)
            raise ValueError(msg)

        assert (
            self.validation_split > 0.0 and self.validation_split < 1.0
        ), f"validation_split must be in (0.0, 1.0), but got {self.validation_split}."

        # Train/Val split
        indices = np.arange(n_samples)
        train_idx, val_test_idx = train_test_split(
            indices,
            test_size=self.validation_split,
            random_state=self.seed,
        )

        if not val_test_idx.size >= 4:
            msg = f"Not enough samples ({val_test_idx.size}) for validation/test split."
            self.logger.error(msg)
            raise ValueError(msg)

        # Split val and test equally
        val_idx, test_idx = train_test_split(
            val_test_idx, test_size=0.5, random_state=self.seed
        )

        return train_idx, val_idx, test_idx

    def _get_data_loaders(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        batch_size: int,
        *,
        shuffle: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Create DataLoader for training and validation.

        Args:
            X (np.ndarray): 0/1/2-encoded input matrix.
            y (np.ndarray): 0/1/2-encoded matrix with -1 for missing.
            mask (np.ndarray): Boolean mask of entries to score in the loss.
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle batches. Defaults to True.

        Returns:
            The DataLoader.
        """
        dataset = _MaskedNumpyDataset(X, y, mask)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=(str(self.device).startswith("cuda")),
        )

    def _update_anneal_schedule(
        self,
        final: float,
        warm: int,
        ramp: int,
        epoch: int,
        *,
        init_val: float = 0.0,
    ) -> torch.Tensor:
        """Update annealed hyperparameter value based on epoch.

        Args:
            final (float): Final value after annealing.
            warm (int): Number of warm-up epochs.
            ramp (int): Number of ramp-up epochs.
            epoch (int): Current epoch number.
            init_val (float): Initial value before annealing starts.

        Returns:
            torch.Tensor: Current value of the hyperparameter.
        """
        if epoch < warm:
            val = torch.tensor(init_val)
        elif epoch < warm + ramp:
            val = torch.tensor(final * ((epoch - warm) / ramp))
        else:
            val = torch.tensor(final)

        return val.to(self.device)

    def _anneal_config(
        self,
        params: Optional[dict],
        key: str,
        default: float,
        max_epochs: int,
        *,
        warm_alt: int = 50,
        ramp_alt: int = 100,
    ) -> Tuple[float, int, int]:
        """Configure annealing schedule for a hyperparameter.

        Args:
            params (Optional[dict]): Dictionary of parameters to extract from.
            key (str): Key to look for in params.
            default (float): Default final value if not specified in params.
            max_epochs (int): Total number of training epochs.
            warm_alt (int): Alternative warm-up period if 10% of epochs is too long
            ramp_alt (int): Alternative ramp-up period if 20% of epochs is too long

        Returns:
            Tuple[float, int, int]: Final value, warm-up epochs, ramp-up epochs.
        """
        val = None
        if params is not None and params:
            if not hasattr(self, key):
                msg = f"Attribute '{key}' not found for anneal_config."
                self.logger.error(msg)
                raise AttributeError(msg)

            val = params.get(key, getattr(self, key))

        if val is not None and isinstance(val, (float, int)):
            final = float(val)
        else:
            final = default

        warm, ramp = min(int(0.1 * max_epochs), warm_alt), min(
            int(0.2 * max_epochs), ramp_alt
        )
        return final, warm, ramp

    def _repair_ref_alt_from_iupac(self, loci: np.ndarray) -> None:
        """Repair REF/ALT for specific loci using observed IUPAC genotypes.

        Args:
            loci (np.ndarray): Array of locus indices to repair.

        Notes:
            - Modifies self.genotype_data.ref and self.genotype_data.alt in place.
        """
        iupac_to_bases = {
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
        }
        missing_codes = {"", ".", "N", "NONE", "-", "?", "./.", ".|."}

        def norm(v: object) -> str | None:
            if v is None:
                return None
            s = str(v).upper().strip()
            if not s or s in missing_codes:
                return None
            return s if s in iupac_to_bases else None

        snp = np.asarray(self.genotype_data.snp_data, dtype=object)  # (N,L) IUPAC-ish
        refs = list(getattr(self.genotype_data, "ref", [None] * snp.shape[1]))
        alts = list(getattr(self.genotype_data, "alt", [None] * snp.shape[1]))

        for j in loci:
            cnt = Counter()
            col = snp[:, int(j)]
            for g in col:
                code = norm(g)
                if code is None:
                    continue
                for b in iupac_to_bases[code]:
                    cnt[b] += 1

            if not cnt:
                continue

            common = [b for b, _ in cnt.most_common()]
            ref = common[0]
            alt = common[1] if len(common) > 1 else None

            refs[int(j)] = ref
            alts[int(j)] = alt if alt is not None else "."

        self.genotype_data.ref = np.asarray(refs, dtype=object)

        if not isinstance(alts, np.ndarray):
            alts = np.array(alts, dtype=object).tolist()

        self.genotype_data.alt = alts

    def _aligned_ref_alt(self, L: int) -> tuple[list[object], list[object]]:
        """Return REF/ALT aligned to the genotype matrix columns.

        Args:
            L (int): Number of loci (columns in genotype matrix).

        Returns:
            tuple[list[object], list[object]]: Aligned REF and ALT lists.
        """
        refs = getattr(self.genotype_data, "ref", None)
        alts = getattr(self.genotype_data, "alt", None)

        if refs is None or alts is None:
            msg = "genotype_data.ref/alt are required but missing."
            self.logger.error(msg)
            raise ValueError(msg)

        refs_arr = np.asarray(refs, dtype=object)
        alts_arr = np.asarray(alts, dtype=object)

        if refs_arr.shape[0] != L or alts_arr.shape[0] != L:
            msg = f"REF/ALT length mismatch vs matrix columns: L={L}, len(ref)={refs_arr.shape[0]}, len(alt)={alts_arr.shape[0]}. You are using REF/ALT metadata that is not aligned to pgenc.genotypes_012 columns. Fix by subsetting/refiltering ref/alt with the same locus mask used for the genotype matrix."
            self.logger.error(msg)
            raise ValueError(msg)

        # Unwrap singleton ALT arrays like array(['T'], dtype=object)
        def unwrap(x: object) -> object:
            if isinstance(x, np.ndarray):
                if x.size == 0:
                    return None
                if x.size == 1:
                    return x.item()
            return x

        refs_list = [unwrap(x) for x in refs_arr.tolist()]
        alts_list = [unwrap(x) for x in alts_arr.tolist()]
        return refs_list, alts_list

    def _build_valid_class_mask(self) -> torch.Tensor:
        L = self.num_features_
        K = self.num_classes_
        mask = np.ones((L, K), dtype=bool)

        # --- IUPAC helpers (single-character only) ---
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
        }
        missing_codes = {"", ".", "N", "NONE", "-", "?", "./.", ".|."}

        # get aligned ref/alt (should be exactly length L)
        refs, alts = self._aligned_ref_alt(L)

        def _normalize_iupac(value: object) -> str | None:
            """Return a single-letter IUPAC code or None if missing/invalid."""
            if value is None:
                return None
            if isinstance(value, (bytes, np.bytes_)):
                value = value.decode("utf-8", errors="ignore")

            # allow list/tuple/array containers (take first valid)
            if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                for item in value:
                    code = _normalize_iupac(item)
                    if code is not None:
                        return code
                return None

            s = str(value).upper().strip()
            if not s or s in missing_codes:
                return None

            # handle comma-separated values
            if "," in s:
                for tok in (t.strip() for t in s.split(",")):
                    if tok and tok not in missing_codes and tok in iupac_to_bases:
                        return tok
                return None

            return s if s in iupac_to_bases else None

        # 1) metadata restriction
        for j in range(L):
            ref = _normalize_iupac(refs[j])
            alt = _normalize_iupac(alts[j])

            if alt is None or (ref is not None and alt == ref):
                mask[j, :] = False
                mask[j, 0] = True

        # 2) data-driven override
        y_train = getattr(self, "y_train_", None)
        if y_train is not None:
            y = np.asarray(y_train)
            if y.ndim == 2 and y.shape[1] == L:
                if K == 2:
                    y = y.copy()
                    y[y == 2] = 1
                valid = y >= 0
                if valid.any():
                    observed = np.zeros((L, K), dtype=bool)
                    for c in range(K):
                        observed[:, c] = np.any(valid & (y == c), axis=0)

                    conflict = observed & (~mask)
                    if conflict.any():
                        loci = np.where(conflict.any(axis=1))[0]
                        self.valid_class_mask_conflict_loci_ = loci
                        self.logger.warning(
                            f"valid_class_mask_ metadata forbids observed classes at {loci.size} loci. "
                            "Expanding mask to include observed classes."
                        )
                        mask |= observed

        bad = np.where(~mask.any(axis=1))[0]
        if bad.size:
            mask[bad, :] = False
            mask[bad, 0] = True

        return torch.as_tensor(mask, dtype=torch.bool, device=self.device)
