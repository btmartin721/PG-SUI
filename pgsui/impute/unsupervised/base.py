import copy
import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from snpio.utils.logging import LoggerManager

from pgsui.impute.unsupervised.nn_scorers import Scorer
from pgsui.utils.classification_viz import ClassificationReportVisualizer
from pgsui.utils.plotting import Plotting


class BaseNNImputer:
    """An abstract base class for neural network-based imputers.

    This class provides a shared framework and common functionality for all neural network imputers. It is not meant to be instantiated directly. Instead, child classes should inherit from it and implement the abstract methods. Provided functionality: Directory setup and logging initialization; A hyperparameter tuning pipeline using Optuna; Utility methods for building models (`build_model`), initializing weights (`initialize_weights`), and checking for fitted attributes (`ensure_attribute`); Helper methods for calculating class weights for imbalanced data; Setup for standardized plotting and model scoring classes.
    """

    def __init__(
        self,
        prefix: str,
        *,
        device: Literal["gpu", "cpu", "mps"] = "cpu",
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initializes the base class for neural network imputers.

        This constructor sets up the device (CPU, GPU, or MPS), creates the necessary output directories for models and results, and configures a logger. It also initializes a genotype encoder for handling genotype data.

        Args:
            prefix (str): A prefix used to name the output directory (e.g., 'pgsui_output').
            device (Literal["gpu", "cpu", "mps"]): The device to use for PyTorch operations. If 'gpu' or 'mps' is chosen, it will fall back to 'cpu' if the required hardware is not available. Defaults to "cpu".
            verbose (bool): If True, enables detailed logging output. Defaults to False.
            debug (bool): If True, enables debug mode. Defaults to False.
        """
        self.device = self._select_device(device)

        # Prepare directory structure
        outdirs = ["models", "plots", "metrics", "optimize"]
        self._create_model_directories(prefix, outdirs)
        self.debug = debug

        # Initialize logger
        kwargs = {"prefix": prefix, "verbose": verbose, "debug": debug}
        logman = LoggerManager(__name__, **kwargs)
        self.logger = logman.get_logger()

    def tune_hyperparameters(self) -> None:
        """Tunes model hyperparameters using an Optuna study.

        This method orchestrates the hyperparameter search process. It creates an Optuna study that aims to maximize the metric defined in `self.tune_metric`. The search is driven by the `_objective` method, which must be implemented by the child class. After the search, the best parameters are logged, saved to a JSON file, and visualizations of the study are generated.

        Raises:
            NotImplementedError: If the `_objective` or `_set_best_params` methods are not implemented in the inheriting child class.
        """
        self.logger.info("Tuning hyperparameters...")

        study_db = None
        load_if_exists = False
        if self.tune_save_db:
            study_db = self.optimize_dir / "study_database" / "optuna_study.db"
            study_db.parent.mkdir(parents=True, exist_ok=True)

            if self.tune_resume and study_db.exists():
                load_if_exists = True

            if not self.tune_resume and study_db.exists():
                study_db.unlink()

        study_name = f"{self.prefix}_{self.model_name} Model Optimization"
        storage = f"sqlite:///{study_db}" if self.tune_save_db else None

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
        )

        if not hasattr(self, "_objective"):
            msg = "`_objective()` must be implemented in the child class."
            self.logger.error(msg)
            raise NotImplementedError(msg)

        self.n_jobs = getattr(self, "n_jobs", 1)
        if self.n_jobs < -1 or self.n_jobs == 0:
            self.n_jobs = max(1, (os.cpu_count() or 2) - 1)

        study.optimize(
            lambda trial: self._objective(trial),
            n_trials=self.n_trials,
            n_jobs=getattr(self, "n_jobs", 1),
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

        self.logger.info(best_params_tmp)

        # Save best parameters to a JSON file.
        fn = self.optimize_dir / "parameters" / "best_params.json"

        if not fn.parent.exists():
            fn.parent.mkdir(parents=True, exist_ok=True)

        with open(fn, "w") as fp:
            json.dump(best_params, fp, indent=4)

        tn = f"{self.tune_metric} Value"
        self.plotter_.plot_tuning(study, self.model_name, target_name=tn)

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
        self, Model: torch.nn.Module, model_params: Dict[str, Any]
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
        all_params = {
            "n_features": self.num_features_,
            "prefix": self.prefix,
            "num_classes": self.num_classes_,
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
        # Initialize plotter.
        plotter = Plotting(
            model_name=self.model_name,
            prefix=self.prefix,
            plot_format=self.plot_format,
            plot_fontsize=self.plot_fontsize,
            plot_dpi=self.plot_dpi,
            title_fontsize=self.title_fontsize,
            despine=self.despine,
            show_plots=self.show_plots,
            verbose=self.verbose,
            debug=self.debug,
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

        mode="allele": y is 1D alleles in {0..3}, train_mask same shape.

        mode="genotype10": y is (nS,nF,2) alleles; train_mask is (nS,nF) loci where both alleles known.

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
        device = device.lower().strip()
        if device == "cpu":
            self.logger.info("Using PyTorch device: CPU.")
            return torch.device("cpu")
        if device == "mps":
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
        latent_vectors: torch.nn.Parameter,
    ) -> None:
        """Releases GPU and CPU memory after an Optuna trial.

        This is a crucial step during hyperparameter tuning to prevent memory leaks between trials, ensuring that each trial runs in a clean environment.

        Args:
            model (torch.nn.Module): The model from the completed trial.
            train_loader (torch.utils.data.DataLoader): The data loader from the trial.
            latent_vectors (torch.nn.Parameter): The latent vectors from the trial.
        """
        try:
            del model, train_loader, latent_vectors
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

        Raises:
            ValueError: If `labels` length is not 3 or 10.
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
            labels (List[str], optional): Class label names
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

        self.logger.info(
            "\n"
            + classification_report(
                y_true,
                y_pred,
                labels=list(range(len(labels))),
                target_names=labels,
                zero_division=0,
            )
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
            output_dict=True,
        )

        with open(self.metrics_dir / f"{report_name}_report.json", "w") as f:
            json.dump(report, f, indent=4)

        viz = ClassificationReportVisualizer(reset_kwargs=self.plotter_.param_dict)

        plots = viz.plot_all(
            report,
            title_prefix=f"{self.model_name} {middle} Report",
            show=getattr(self, "show_plots", False),
            heatmap_classes_only=True,
        )

        for name, fig in plots.items():
            fout = self.plots_dir / f"{report_name}_report_{name}.{self.plot_format}"
            if hasattr(fig, "savefig"):
                fig.savefig(fout, dpi=300, facecolor="#111122")
                plt.close(fig)
            else:
                fig.write_html(file=fout.with_suffix(".html"))

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
        schedule: Literal["pyramid", "constant", "linear"] = "pyramid",
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
            raise ValueError("n_hidden must be >= 0.")
        if n_hidden == 0:
            return []
        denom = float(alpha) * float(n_inputs + n_outputs)
        if denom <= 0:
            raise ValueError("alpha * (n_inputs + n_outputs) must be > 0.")
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
                        decay = 1.0
                    else:
                        decay = (target / float(base)) ** (1.0 / (n_hidden - 1))
                        decay = float(np.clip(decay, 0.25, 0.99))
                exponents = np.arange(n_hidden, dtype=float)
                sizes = base * (decay**exponents)
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
