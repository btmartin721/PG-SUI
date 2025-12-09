import copy
import gc
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from snpio import SNPioMultiQC
from snpio.utils.logging import LoggerManager

from pgsui.impute.unsupervised.nn_scorers import Scorer
from pgsui.utils.classification_viz import ClassificationReportVisualizer
from pgsui.utils.logging_utils import configure_logger
from pgsui.utils.plotting import Plotting
from pgsui.utils.pretty_metrics import PrettyMetrics

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData

    from pgsui.impute.unsupervised.models.autoencoder_model import AutoencoderModel
    from pgsui.impute.unsupervised.models.nlpca_model import NLPCAModel
    from pgsui.impute.unsupervised.models.ubp_model import UBPModel
    from pgsui.impute.unsupervised.models.vae_model import VAEModel


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
        self._float_genotype_cache: np.ndarray | None = None
        self._sim_mask_cache: dict[tuple, np.ndarray] = {}
        self._tune_ready: bool = False

        # To be initialized by child classes or fit method
        self.tune_save_db: bool = False
        self.tune_resume: bool = False
        self.n_trials: int = 100
        self.model_params: Dict[str, Any] = {}
        self.tune_metric: str = "val_f1_macro"
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
        self.is_haploid: bool = False
        self.ploidy: int = 2
        self.beta: float = 0.9999
        self.max_ratio: float = 5.0
        self.sim_strategy: str = "mcar"
        self.sim_prop: float = 0.1
        self.seed: int | None = 42
        self.rng: np.random.Generator = np.random.default_rng(self.seed)
        self.ground_truth_: np.ndarray
        self.tune_fast: bool = False
        self.tune_max_samples: int = 1000
        self.tune_max_loci: int = 500
        self.validation_split: float = 0.2
        self.tune_batch_size: int = 64
        self.tune_proxy_metric_batch: int = 512
        self.batch_size: int = 64
        self.best_params_: Dict[str, Any] = {}

        self.optimize_dir: Path
        self.models_dir: Path
        self.plots_dir: Path
        self.metrics_dir: Path
        self.parameters_dir: Path
        self.study_db: Path

    def tune_hyperparameters(self) -> None:
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
            study_db = self.optimize_dir / "study_database" / "optuna_study.db"
            study_db.parent.mkdir(parents=True, exist_ok=True)

            if self.tune_resume and study_db.exists():
                load_if_exists = True

            if not self.tune_resume and study_db.exists():
                study_db.unlink()

        study_name = f"{self.prefix} {self.model_name} Model Optimization"
        storage = f"sqlite:///{study_db}" if self.tune_save_db else None

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
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

        study.optimize(
            lambda trial: self._objective(trial),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            gc_after_trial=True,
            show_progress_bar=show_progress_bar,
        )

        best_metric = study.best_value
        best_params = study.best_params

        # Set the best parameters.
        # NOTE: `_set_best_params()` must be implemented in the child class.
        if not hasattr(self, "_set_best_params"):
            msg = "Method `_set_best_params()` must be implemented in the child class."
            self.logger.error(msg)
            raise NotImplementedError(msg)

        self.best_params_ = self._set_best_params(best_params)
        self.model_params.update(self.best_params_)
        self.logger.info(f"Best {self.tune_metric} metric: {best_metric}")
        self.logger.info("Best parameters:")
        best_params_tmp = copy.deepcopy(best_params)
        best_params_tmp["learning_rate"] = self.learning_rate

        title = f"{self.model_name} Optimized Parameters"
        pm = PrettyMetrics(best_params_tmp, precision=6, title=title)
        pm.render()

        # Save best parameters to a JSON file.
        self._save_best_params(best_params)

        tn = f"{self.tune_metric} Value"
        self.plotter_.plot_tuning(
            study, self.model_name, self.optimize_dir / "plots", target_name=tn
        )

    @staticmethod
    def initialize_weights(module: torch.nn.Module) -> None:
        """Initializes model weights using the Kaiming Uniform distribution.

        This static method is intended to be applied to a PyTorch model to initialize the weights of its linear and convolutional layers. This initialization scheme is particularly effective for networks that use ReLU-family activation functions, as it helps maintain stable activation variances during training.

        Args:
            module (torch.nn.Module): The PyTorch module (e.g., a layer) to initialize.
        """
        if isinstance(
            module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.ConvTranspose1d)
        ):
            # Use Kaiming Uniform initialization for Linear and Conv layers
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def build_model(
        self,
        Model: (
            torch.nn.Module
            | type["AutoencoderModel"]
            | type["NLPCAModel"]
            | type["UBPModel"]
            | type["VAEModel"]
        ),
        model_params: Dict[str, int | float | str | bool],
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

        # Start with a base set of fixed (non-tuned) parameters.
        base_num_classes = getattr(self, "output_classes_", None)
        if base_num_classes is None:
            base_num_classes = self.num_classes_
        all_params = {
            "n_features": self.num_features_,
            "prefix": self.prefix,
            "num_classes": base_num_classes,
            "verbose": self.verbose,
            "debug": self.debug,
            "device": self.device,
        }

        # Update with the variable hyperparameters from the provided dictionary
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
            np.ndarray: The data with missing values imputed.
        """
        msg = "Method ``transform()`` must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def _class_balanced_weights_from_mask(
        self,
        y: np.ndarray,
        train_mask: np.ndarray,
        num_classes: int,
        beta: float = 0.9999,
        max_ratio: float = 5.0,
        mode: Literal["allele", "genotype10"] = "allele",
    ) -> torch.Tensor:
        """Class-balanced weights (Cui et al. 2019) with overflow-safe effective number.

        mode="allele": y is 1D alleles in {0..3}, train_mask same shape. mode="genotype10": y is (nS,nF,2) alleles; train_mask is (nS,nF) loci where both alleles known.

        Args:
            y (np.ndarray): Ground truth labels.
            train_mask (np.ndarray): Boolean mask of training examples (same shape as y or y without last dim for genotype10).
            num_classes (int): Number of classes.
            beta (float): Hyperparameter for effective number calculation. Clamped to (0,1). Default is 0.9999.
            max_ratio (float): Maximum allowed ratio between largest and smallest non-zero weight. Default is 5.0.
            mode (Literal["allele", "genotype10"]): Whether y contains allele labels or 10-class genotypes. Default is "allele".

        Returns:
            torch.Tensor: Class weights of shape (num_classes,). Mean weight is 1.0, zero-weight classes remain zero.
        """
        if mode == "allele":
            valid = (y >= 0) & train_mask
            cls, cnt = np.unique(y[valid].astype(np.int64), return_counts=True)
            counts = np.zeros(num_classes, dtype=np.float64)
            counts[cls] = cnt

        elif mode == "genotype10":
            if y.ndim != 3 or y.shape[-1] != 2:
                msg = "For genotype10, y must be (nS,nF,2)."
                self.logger.error(msg)
                raise ValueError(msg)

            if train_mask.shape != y.shape[:2]:
                msg = "train_mask must be (nS,nF) for genotype10."
                self.logger.error(msg)
                raise ValueError(msg)

            # only loci where both alleles known and in training
            m = train_mask & np.all(y >= 0, axis=-1)
            if not np.any(m):
                counts = np.zeros(num_classes, dtype=np.float64)

            else:
                a1 = y[:, :, 0][m].astype(int)
                a2 = y[:, :, 1][m].astype(int)
                lo, hi = np.minimum(a1, a2), np.maximum(a1, a2)
                # map to 10-class index
                map10 = self.pgenc.map10
                idx10 = map10[lo, hi]
                idx10 = idx10[(idx10 >= 0) & (idx10 < num_classes)]
                counts = np.bincount(idx10, minlength=num_classes).astype(np.float64)
        else:
            msg = f"Unknown mode supplied to _class_balanced_weights_from_mask: {mode}"
            self.logger.error(msg)
            raise ValueError(msg)

        # ---- Effective number ----
        beta = float(beta)

        # clamp beta ∈ (0,1)
        if not np.isfinite(beta):
            beta = 0.9999

        beta = min(max(beta, 1e-8), 1.0 - 1e-8)

        logb = np.log(beta)  # < 0
        t = counts * logb  # ≤ 0

        # 1 - beta^n = 1 - exp(n*log(beta)) = -(exp(n*log(beta)) - 1)
        # use expm1 for accuracy near 0; for very negative t, eff≈1.0
        eff = np.where(t > -50.0, -np.expm1(t), 1.0)

        # class-balanced weights
        w = (1.0 - beta) / (eff + 1e-12)

        # Give unseen classes the largest non-zero weight (keeps it learnable)
        if np.any(counts == 0) and np.any(counts > 0):
            w[counts == 0] = w[counts > 0].max()

        # normalize by mean of non-zero
        nz = w > 0
        w[nz] /= w[nz].mean() + 1e-12

        # cap spread consistently with a single 'cap'
        cap = float(max_ratio) if max_ratio is not None else 10.0
        cap = max(cap, 5.0)  # ensure we allow some differentiation
        if np.any(nz):
            spread = w[nz].max() / max(w[nz].min(), 1e-12)
            if spread > cap:
                scale = cap / spread
                w[nz] = 1.0 + (w[nz] - 1.0) * scale

        return torch.tensor(w.astype(np.float32), device=self.device)

    def _select_device(self, device: Literal["gpu", "cpu", "mps"]) -> torch.device:
        """Selects the appropriate PyTorch device based on user preference and availability.

        This method checks the user's device preference ('gpu', 'cpu', or 'mps') and verifies if the requested hardware is available. If the preferred device is not available, it falls back to CPU and logs a warning.

        Args:
            device (Literal["gpu", "cpu", "mps"]): The preferred device type for PyTorch operations.

        Returns:
            torch.device: The selected PyTorch device.
        """
        dvc: str = device
        dvc = dvc.lower().strip()
        if dvc == "cpu":
            self.logger.info("Using PyTorch device: CPU.")
            return torch.device("cpu")
        if dvc == "mps":
            if torch.backends.mps.is_available():
                self.logger.info("Using PyTorch device: mps.")
                return torch.device("mps")
            self.logger.warning("MPS unavailable; falling back to CPU.")
            return torch.device("cpu")
        # gpu
        if torch.cuda.is_available():
            self.logger.info("Using PyTorch device: cuda.")
            return torch.device("cuda")
        self.logger.warning("CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")

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

    def _clear_resources(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        latent_vectors: torch.nn.Parameter | None = None,
    ) -> None:
        """Releases GPU and CPU memory after an Optuna trial.

        This is a crucial step during hyperparameter tuning to prevent memory leaks between trials, ensuring that each trial runs in a clean environment.

        Args:
            model (torch.nn.Module): The model from the completed trial.
            train_loader (torch.utils.data.DataLoader): The data loader from the trial.
            latent_vectors (torch.nn.Parameter | None): The latent vectors from the trial.
        """
        try:
            del model, train_loader

            if latent_vectors is not None:
                del latent_vectors

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
            labels (List[str]): Class label names
                (default: ["REF", "HET", "ALT"] for 3-class).
        """
        report_name = "zygosity" if len(labels) == 3 else "iupac"
        middle = "IUPAC" if report_name == "iupac" else "Zygosity"

        msg = f"{middle} Report (on {y_true.size} total genotypes)"
        self.logger.info(msg)

        if y_pred_proba is not None:
            self.plotter_.plot_metrics(
                y_true,
                y_pred_proba,
                metrics,
                label_names=labels,
                prefix=report_name,
            )

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
            self.logger.error(msg)
            raise ValueError(msg)

        report_subset = {}
        for k, v in report.items():
            tmp = {}
            if isinstance(v, dict) and "support" in v:
                for k2, v2 in v.items():
                    if k2 != "support":
                        tmp[k2] = v2
                if tmp:
                    report_subset[k] = tmp

        if report_subset:
            pm = PrettyMetrics(
                report_subset,
                precision=3,
                title=f"{self.model_name} {middle} Report",
            )
            pm.render()

        with open(self.metrics_dir / f"{report_name}_report.json", "w") as f:
            json.dump(report, f, indent=4)

        viz = ClassificationReportVisualizer(reset_kwargs=self.plotter_.param_dict)

        plots = viz.plot_all(
            report,  # type: ignore
            title_prefix=f"{self.model_name} {middle} Report",
            show=getattr(self, "show_plots", False),
            heatmap_classes_only=True,
        )

        for name, fig in plots.items():
            fout = self.plots_dir / f"{report_name}_report_{name}.{self.plot_format}"
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

            if not self.is_haploid:
                msg = f"Ploidy: {self.ploidy}. Evaluating per allele."
                self.logger.info(msg)

        viz._reset_mpl_style()

    def _compute_hidden_layer_sizes(
        self,
        n_inputs: int,
        n_outputs: int,
        n_samples: int,
        n_hidden: int,
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

        This method computes a list of hidden layer sizes based on the number of input features, output classes, training samples, and desired hidden layers. The sizes are determined using a specified schedule (pyramid, constant, or linear) and are constrained by minimum and maximum sizes, as well as rounding to multiples of a specified value.

        Args:
            n_inputs (int): Number of input features.
            n_outputs (int): Number of output classes.
            n_samples (int): Number of training samples.
            n_hidden (int): Number of hidden layers.
            alpha (float): Scaling factor for base layer size. Default is 4.0.
            schedule (Literal["pyramid", "constant", "linear"]): Size schedule. Default is "pyramid".
            min_size (int): Minimum layer size. Default is 16.
            max_size (int | None): Maximum layer size. Default is None (no limit).
            multiple_of (int): Round layer sizes to be multiples of this. Default is 8.
            decay (float | None): Decay factor for "pyramid" schedule. If None, it is computed automatically. Default is None.
            cap_by_inputs (bool): If True, cap layer sizes to n_inputs. Default is True.

        Returns:
            list[int]: List of hidden layer sizes.

        Raises:
            ValueError: If n_hidden < 0 or if alpha * (n_inputs + n_outputs) <= 0 or if schedule is unknown.
            TypeError: If any argument is not of the expected type.

        Notes:
            - If n_hidden is 0, returns an empty list.
            - The base layer size is computed as ceil(n_samples / (alpha * (n_inputs + n_outputs))).
            - The sizes are adjusted according to the specified schedule and constraints.
        """
        if n_hidden < 0:
            msg = f"n_hidden must be >= 0, got {n_hidden}."
            self.logger.error(msg)
            raise ValueError(msg)

        if schedule not in {"pyramid", "constant", "linear"}:
            msg = f"Unknown schedule '{schedule}'. Use 'pyramid', 'constant', or 'linear'."
            self.logger.error(msg)
            raise ValueError(msg)

        if n_hidden == 0:
            return []

        denom = float(alpha) * float(n_inputs + n_outputs)

        if denom <= 0:
            msg = f"alpha * (n_inputs + n_outputs) must be > 0, got {denom}."
            self.logger.error(msg)
            raise ValueError(msg)

        base = int(np.ceil(float(n_samples) / denom))

        if max_size is None:
            max_size = max(n_inputs, base)

        base = int(np.clip(base, min_size, max_size))

        if schedule == "constant":
            sizes = np.full(shape=(n_hidden,), fill_value=base, dtype=float)

        elif schedule == "linear":
            target = max(min_size, min(base, base // 4))
            sizes = (
                np.array([base], dtype=float)
                if n_hidden == 1
                else np.linspace(base, target, num=n_hidden, dtype=float)
            )

        elif schedule == "pyramid":
            if n_hidden == 1:
                sizes = np.array([base], dtype=float)
            else:
                if decay is None:
                    target = max(min_size, base // 4)
                    if base <= 0 or target <= 0:
                        dcy = 1.0
                    else:
                        dcy = (target / float(base)) ** (1.0 / (n_hidden - 1))
                        dcy = float(np.clip(dcy, 0.25, 0.99))
                exponents = np.arange(n_hidden, dtype=float)
                sizes = base * (dcy**exponents)

        else:
            msg = f"Unknown schedule '{schedule}'. Use 'pyramid', 'constant', or 'linear'."
            self.logger.error(msg)
            raise ValueError(msg)

        sizes = np.clip(sizes, min_size, max_size)

        if cap_by_inputs:
            sizes = np.minimum(sizes, float(n_inputs))

        sizes = (np.ceil(sizes / multiple_of) * multiple_of).astype(int)
        sizes = np.minimum.accumulate(sizes)
        return np.clip(sizes, min_size, max_size).astype(int).tolist()

    def _class_weights_from_zygosity(self, X: np.ndarray) -> torch.Tensor:
        """Class-balanced weights for 0/1/2 (handles haploid collapse if needed).

        This method computes class-balanced weights for the genotype classes (0/1/2) based on the provided genotype matrix. It handles cases where the data is haploid by collapsing the ALT class to 1, effectively treating the problem as binary classification (REF vs ALT). The weights are calculated using a class-balanced weighting scheme that considers the frequency of each class in the training data, with parameters for beta and maximum ratio to control the weighting behavior. The resulting weights are returned as a PyTorch tensor on the current device.

        Args:
            X (np.ndarray): 0/1/2 with -1 for missing.

        Returns:
            torch.Tensor: Weights on current device.
        """
        y = X[X != -1].ravel().astype(np.int64)
        if y.size == 0:
            return torch.ones(
                self.num_classes_, dtype=torch.float32, device=self.device
            )

        return self._class_balanced_weights_from_mask(
            y=y,
            train_mask=np.ones_like(y, dtype=bool),
            num_classes=self.num_classes_,
            beta=self.beta,
            max_ratio=self.max_ratio,
            mode="allele",  # 1D int vector
        ).to(self.device)

    @staticmethod
    def _normalize_class_weights(
        weights: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Normalize class weights once to keep loss scale stable.

        Args:
            weights (torch.Tensor | None): Class weights to normalize.

        Returns:
            torch.Tensor | None: Normalized class weights or None if input is None.
        """
        if weights is None:
            return None
        return weights / weights.mean().clamp_min(1e-8)

    def _get_float_genotypes(self, *, copy: bool = True) -> np.ndarray:
        """Float32 0/1/2 matrix with NaNs for missing, cached per dataset.

        Args:
            copy (bool): If True, return a copy of the cached array. Default is True.

        Returns:
            np.ndarray: Float32 genotype matrix with NaNs for missing values.
        """
        cache = self._float_genotype_cache
        current = self.pgenc.genotypes_012
        if cache is None or cache.shape != current.shape or cache.dtype != np.float32:
            arr = np.asarray(current, dtype=np.float32)
            arr = np.where(arr < 0, np.nan, arr)
            self._float_genotype_cache = arr
            cache = arr
        return cache.copy() if copy else cache

    def _sim_mask_cache_key(self) -> tuple | None:
        """Key for caching simulated-missing masks."""
        if not getattr(self, "simulate_missing", False):
            return None
        shape = tuple(self.pgenc.genotypes_012.shape)
        return (
            id(self.genotype_data),
            self.sim_strategy,
            round(float(self.sim_prop), 6),
            self.seed,
            shape,
        )

    def _one_hot_encode_012(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """One-hot 0/1/2; -1 rows are all-zeros (B, L, K).

        This method performs one-hot encoding of the input genotype data (0, 1, 2) while handling missing values represented by -1. The output is a tensor of shape (B, L, K), where B is the batch size, L is the number of features, and K is the number of classes.

        Args:
            X (np.ndarray | torch.Tensor): The input data to be one-hot encoded, either as a NumPy array or a PyTorch tensor.

        Returns:
            torch.Tensor: A one-hot encoded tensor of shape (B, L, K), where B is the batch size, L is the number of features, and K is the number of classes.
        """
        Xt = (
            torch.from_numpy(X).to(self.device)
            if isinstance(X, np.ndarray)
            else X.to(self.device)
        )

        # B=batch, L=features, K=classes
        B, L = Xt.shape
        K = self.num_classes_
        X_ohe = torch.zeros(B, L, K, dtype=torch.float32, device=self.device)
        valid = Xt != -1
        idx = Xt[valid].long()

        if idx.numel() > 0:
            X_ohe[valid] = F.one_hot(idx, num_classes=K).float()

        return X_ohe

    def _eval_for_pruning(
        self,
        *,
        model: torch.nn.Module,
        X_val: np.ndarray,
        params: dict,
        metric: str,
        objective_mode: bool = True,
        do_latent_infer: bool = False,
        latent_steps: int = 50,
        latent_lr: float = 1e-2,
        latent_weight_decay: float = 0.0,
        latent_seed: int = 123,
        _latent_cache: dict | None = None,
        _latent_cache_key: str | None = None,
        eval_mask_override: np.ndarray | None = None,
    ) -> float:
        """Compute a scalar metric (to MAXIMIZE) on a fixed validation set.

        This method evaluates the model on a validation dataset and computes a specified metric, which is used for pruning decisions during hyperparameter tuning. It supports optional latent inference to optimize latent representations before evaluation. The method handles potential issues with non-finite metric values by returning negative infinity, making it easier to prune poorly performing trials.

        Args:
            model (torch.nn.Module): The model to evaluate.
            X_val (np.ndarray): Validation data.
            params (dict): Model parameters.
            metric (str): Metric name to return.
            objective_mode (bool): If True, use objective-mode evaluation. Default is True.
            do_latent_infer (bool): If True, perform latent inference before evaluation. Default
            latent_steps (int): Number of steps for latent inference. Default is 50.
            latent_lr (float): Learning rate for latent inference. Default is 1e-2
            latent_weight_decay (float): Weight decay for latent inference. Default is 0.0.
            latent_seed (int): Random seed for latent inference. Default is 123.
            _latent_cache (dict | None): Optional cache for storing/retrieving optimized latents
            _latent_cache_key (str | None): Key for storing/retrieving in _latent_cache.
            eval_mask_override (np.ndarray | None): Optional mask to override default evaluation mask.

        Returns:
            float: The computed metric value to maximize. Returns -inf on failure.
        """
        optimized_val_latents = None

        # Optional latent inference path for models that need it.
        if do_latent_infer and hasattr(self, "_latent_infer_for_eval"):
            optimized_val_latents = self._latent_infer_for_eval(  # type: ignore
                model=model,
                X_val=X_val,
                steps=latent_steps,
                lr=latent_lr,
                weight_decay=latent_weight_decay,
                seed=latent_seed,
                cache=_latent_cache,
                cache_key=_latent_cache_key,
            )
            # Retrieve the optimized latents from the cache
            if _latent_cache is not None and _latent_cache_key in _latent_cache:
                optimized_val_latents = _latent_cache[_latent_cache_key]

        if getattr(self, "_tune_eval_slice", None) is not None:
            X_val = X_val[self._tune_eval_slice]
            if eval_mask_override is not None:
                eval_mask_override = eval_mask_override[self._tune_eval_slice]

        # Child's evaluator now accepts the pre-computed latents
        metrics = self._evaluate_model(  # type: ignore
            X_val=X_val,
            model=model,
            params=params,
            objective_mode=objective_mode,
            latent_vectors_val=optimized_val_latents,
            eval_mask_override=eval_mask_override,
        )

        # Prefer the requested metric; fall back to self.tune_metric if needed.
        val = metrics.get(metric, metrics.get(getattr(self, "tune_metric", ""), None))

        if val is None or not np.isfinite(val):
            return -np.inf  # make pruning decisions easy/robust on bad reads

        return float(val)

    def _first_linear_in_features(self, model: torch.nn.Module) -> int:
        """Return in_features of the model's first Linear layer.

        This method iterates through the modules of the provided PyTorch model to find the first instance of a Linear layer. It then retrieves and returns the `in_features` attribute of that layer, which indicates the number of input features expected by the layer.

        Args:
            model (torch.nn.Module): The model to inspect.

        Returns:
            int: The in_features of the first Linear layer.
        """
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                return int(m.in_features)
        raise RuntimeError("No Linear layers found in model.")

    def _assert_model_latent_compat(
        self, model: torch.nn.Module, latent_vectors: torch.nn.Parameter
    ) -> None:
        """Raise if model's first Linear doesn't match latent_vectors width.

        This method checks that the dimensionality of the provided latent vectors matches the expected input feature size of the model's first linear layer. If there is a mismatch, it raises a ValueError with a descriptive message.

        Args:
            model (torch.nn.Module): The model to check.
            latent_vectors (torch.nn.Parameter): The latent vectors to check.

        Raises:
            ValueError: If the latent dimension does not match the model's expected input features.
        """
        zdim = int(latent_vectors.shape[1])
        first_in = self._first_linear_in_features(model)
        if first_in != zdim:
            raise ValueError(
                f"Latent mismatch: zdim={zdim}, model first Linear expects in_features={first_in}"
            )

    def _prepare_tuning_artifacts(self) -> None:
        """Prepare data and artifacts needed for hyperparameter tuning.

        This method sets up the necessary data splits, data loaders, and class weights required for hyperparameter tuning. It creates training and validation sets from the ground truth data, initializes data loaders with a specified batch size, and computes class-balanced weights based on the training data. The method also handles optional subsampling of the dataset for faster tuning and prepares slices for evaluation if needed.

        Raises:
            AttributeError: If the ground truth data (`ground_truth_`) is not set.
        """
        if getattr(self, "_tune_ready", False):
            return

        X = self.ground_truth_
        n_samp, n_loci = X.shape
        rng = self.rng

        if self.tune_fast:
            s = min(n_samp, self.tune_max_samples)
            l = n_loci if self.tune_max_loci == 0 else min(n_loci, self.tune_max_loci)

            samp_idx = np.sort(rng.choice(n_samp, size=s, replace=False))
            loci_idx = np.sort(rng.choice(n_loci, size=l, replace=False))
            X_small = X[samp_idx][:, loci_idx]
        else:
            X_small = X

        idx = np.arange(X_small.shape[0])
        tr, te = train_test_split(
            idx, test_size=self.validation_split, random_state=self.seed
        )
        self._tune_train_idx = tr
        self._tune_test_idx = te
        self._tune_X_train = X_small[tr]
        self._tune_X_test = X_small[te]

        self._tune_class_weights = self._normalize_class_weights(
            self._class_weights_from_zygosity(self._tune_X_train)
        )

        # Temporarily bump batch size only for tuning loader
        orig_bs = self.batch_size
        self.batch_size = self.tune_batch_size
        self._tune_loader = self._get_data_loaders(self._tune_X_train)  # type: ignore
        self.batch_size = orig_bs

        self._tune_num_features = self._tune_X_train.shape[1]
        self._tune_val_latents_source = None
        self._tune_train_latents_source = None

        # Optional: for huge val sets, thin them for proxy metric
        if (
            self.tune_proxy_metric_batch
            and self._tune_X_test.shape[0] > self.tune_proxy_metric_batch
        ):
            self._tune_eval_slice = np.arange(self.tune_proxy_metric_batch)
        else:
            self._tune_eval_slice = None

        self._tune_ready = True

    def _save_best_params(self, best_params: Dict[str, Any]) -> None:
        """Save the best hyperparameters to a JSON file.

        This method saves the best hyperparameters found during hyperparameter tuning to a JSON file in the optimization directory. The filename includes the model name for easy identification.

        Args:
            best_params (Dict[str, Any]): A dictionary of the best hyperparameters to save.
        """
        if not hasattr(self, "parameters_dir"):
            msg = "Attribute 'parameters_dir' not found. Ensure _create_model_directories() has been called."
            self.logger.error(msg)
            raise AttributeError(msg)

        fout = self.parameters_dir / "best_parameters.json"

        with open(fout, "w") as f:
            json.dump(best_params, f, indent=4)

    def _set_best_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """An abstract method for setting best parameters."""
        raise NotImplementedError
