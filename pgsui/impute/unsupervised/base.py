import copy
import json
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from snpio.utils.logging import LoggerManager

from pgsui.data_processing.transformers import AutoEncoderFeatureTransformer
from pgsui.impute.unsupervised.nn_scorers import Scorer
from pgsui.utils.misc import validate_input_type
from pgsui.utils.plotting import Plotting


class BaseNNImputer(BaseEstimator, TransformerMixin):
    """Base class for neural network imputers.

    This class provides a common interface and functionality for all neural network-based imputers. The main responsibilities of this class include:

    - Defining the architecture of the neural network.
    - Implementing the training loop.
    - Providing methods for imputing missing values.
    """

    def __init__(
        self,
        *,
        prefix: str = "pgsui",
        output_dir: str | Path = "output",
        device: Literal["gpu", "cpu"] = "cpu",
        verbose: int = 0,
        debug: bool = False,
    ):
        """Base (parent) class for neural network imputers.

        This class is the parent class for all neural network imputers. It provides common functionality for initializing the model, training the model, tuning hyperparameters, and imputing missing values. The class also provides methods for saving and loading models, plotting optimization results, and logging messages.

        Args:
            prefix (str, optional): Prefix for the output directory. Defaults to "pgsui".
            output_dir (str | Path, optional): Output directory name. Defaults to "output".
            device (Literal["gpu", "cpu"], optional): PyTorch Device. Will use GPU if "gpu" is specified and if a valid GPU device can be found. Defaults to "gpu".
            verbose (int, optional): Verbosity level. Defaults to 0.
            debug (bool, optional): Debug mode. Defaults to False.
        """
        self.device = self._select_device(device)

        # Prepare directory structure
        outdirs = ["models", "plots", "metrics", "optimize"]
        self._create_model_directories(prefix, output_dir, outdirs)
        self.debug = debug if verbose < 2 else True

        # Initialize logger
        kwargs = {"prefix": prefix, "verbose": verbose >= 1, "debug": debug}
        logman = LoggerManager(__name__, **kwargs)
        self.logger = logman.get_logger()

    def tune_hyperparameters(self) -> None:
        """Tune hyperparameters using Optuna study.

        This method tunes the hyperparameters of the model using Optuna. It creates an Optuna study and optimizes the model hyperparameters using the `_objective()` method. The method saves the best hyperparameters to a JSON file and plots the optimization results.

        Raises:
            ValueError: If the model is not fitted yet.
            NotImplementedError: If the `_objective()` method is not implemented in the child class.
            NotImplementedError: If the `set_best_params()` method is not implemented in the child class.
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
            msg = "Method `_objective()` must be implemented in the child class."
            self.logger.error(msg)
            raise NotImplementedError(msg)

        study.optimize(
            lambda trial: self._objective(trial, self.Model),
            n_trials=self.n_trials,
            n_jobs=1,
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
        best_params_fmt = pformat(best_params_tmp, indent=4).split("\n")
        [self.logger.info(p) for p in best_params_fmt]

        # Save best parameters to a JSON file.
        fn = self.optimize_dir / "parameters" / "best_params.json"
        if not fn.parent.exists():
            fn.parent.mkdir(parents=True, exist_ok=True)

        with open(fn, "w") as fp:
            json.dump(best_params, fp, indent=4)

        tn = f"{self.tune_metric} Value"
        self.plotter_.plot_tuning(study, self.model_name, target_name=tn)

    def reset_weights(self, model: torch.nn.Module):
        """Reset the parameters of all layers that have `reset_parameters` defined.

        This method resets the parameters of all layers in the model that have a `reset_parameters` method defined. It is useful for reinitializing the model weights before training. The method iterates over all modules in the model and resets the parameters of any module that has a `reset_parameters` method.

        Args:
            model (torch.nn.Module): The model whose parameters to reset.
        """
        for layer in model.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def impute(
        self, X: np.ndarray | pd.DataFrame | list | torch.Tensor, model: torch.nn.Module
    ) -> np.ndarray:
        """Impute the real missing values in X using the trained model.

        This method uses the trained model to predict and fill in the missing values in the input data.

        Args:
            X (np.ndarray | pd.DataFrame | list | torch.Tensor): Input data with missing values.
            model (torch.nn.Module): The trained model to use for imputation.

        Returns:
            np.ndarray: The imputed data.
        """
        self.ensure_attribute("original_missing_mask_")

        if model is None:
            msg = "Model is not fitted yet. Call `fit()` before imputation."
            self.logger.error(msg)
            raise TypeError(msg)

        # Convert X to array, unify missing indicator.
        X = np.where(np.logical_or(X < 0, np.isnan(X)), -1, X)

        # This block can be simplified as we only need to define X_imputed once.
        if X.ndim == 2:
            X_imputed = X.copy()
        elif X.ndim == 3:
            # Assuming X is one-hot encoded if 3D
            X_imputed = self.tt_.inverse_transform(X)
        else:
            msg = f"Invalid input shape: {X.shape}. Must be 2D or 3D."
            self.logger.error(msg)
            raise ValueError(msg)

        if self.is_backprop:
            # Get the final trained latent vectors
            Xtensor = self.latent_vectors_.to(self.device)
        else:
            # For other models like VAE, transform the input data
            Xtensor = validate_input_type(self.tt_.transform(X), "tensor")

        model.eval()
        with torch.no_grad():
            if self.is_backprop:
                outputs = model.phase23_decoder(Xtensor)
            else:
                outputs = model(Xtensor.to(self.device))

        recon_logits = outputs[0] if isinstance(outputs, tuple) else outputs

        # The ConvDecoder should already output a 3D tensor.
        # A blind reshape is risky; better to check dimensions.
        if recon_logits.dim() == 2:
            recon_logits = recon_logits.reshape(
                -1, self.num_features_, self.num_classes_
            )

        y_pred_proba = torch.softmax(recon_logits, dim=-1)
        y_pred_proba = validate_input_type(y_pred_proba)
        y_pred_labels = self.tt_.inverse_transform(y_pred_proba)

        real_missing_mask = self.original_missing_mask_
        if real_missing_mask.shape != y_pred_labels.shape:
            msg = (
                f"Shape mismatch between real_missing_mask "
                f"({real_missing_mask.shape}) and predictions "
                f"({y_pred_labels.shape})."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        X_imputed[real_missing_mask] = y_pred_labels[real_missing_mask]
        return X_imputed

    @staticmethod
    def initialize_weights(module: torch.nn.Module) -> None:
        """Initialize model weights using Kaiming Uniform for layers followed by a ReLU-family activation.

        This initialization method is designed to help with the convergence of deep networks by maintaining a stable variance in the activations.

        Args:
            module (torch.nn.Module): The PyTorch module to initialize.
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
        """Build the neural network model.

        This method builds the neural network model using the provided model class and parameters. It instantiates the model and initializes the weights. The method returns the built model.

        Args:
            Model (torch.nn.Module): Model class to instantiate.
            model_params (Dict[str, Any]): Model parameters.

        Returns:
            torch.nn.Module: Built model.

        Raises:
            TypeError: If model_params is not provided or is empty.
        """
        if not model_params or not isinstance(model_params, dict):
            msg = "'model_params' must be provided and must not be empty."
            self.logger.error(msg)
            raise TypeError(msg)

        all_prms = {
            "prefix": self.prefix,
            "logger": self.logger,
            "verbose": self.verbose,
            "debug": self.debug,
            "device": self.device,
            "use_convolution": self.use_convolution,
        }
        all_prms.update(model_params)
        return Model(**all_prms).to(self.device)

    def ensure_attribute(self, attribute: str) -> None:
        """Ensure that the attribute exists in the class.

        This method checks if the attribute exists in the class. If the attribute does not exist, it raises an AttributeError. If the attribute is not a string, it raises a TypeError. The method is used to ensure that the required attributes are initialized before using them.

        Args:
            attribute (str): Attribute to check. Must be a string. The attribute must exist in the class.

        Raises:
            TypeError: If the attribute is not a string.
            AttributeError: If the attribute does not exist.
        """
        if not isinstance(attribute, str):
            msg = "Argument 'attribute' must be a string."
            self.logger.error(msg)
            raise TypeError(msg)

        if not hasattr(self, attribute):
            msg = f"{self.model_name} has no attribute '{attribute}'."
            self.logger.error(msg)
            raise AttributeError(msg)

    def compute_class_weights(
        self,
        y: np.ndarray,
        train_mask: np.ndarray | None = None,
        use_log_scale: bool = False,
        alpha: float = 1.0,
        normalize: bool = False,
        temperature: float = 1.0,
        max_weight: float = 20.0,
        min_weight: float = 0.01,
    ) -> torch.Tensor:
        """Compute inverse-frequency class weights from the targets actually used for training.

        Args:
            y: Integer-encoded targets (shape: [n_samples, n_features]).
            train_mask: Boolean mask True where a token participates in the loss.
            ...
        Returns:
            torch.Tensor: shape (num_classes,)
        """
        yy = y.copy()
        yy[(yy < 0) | np.isnan(yy)] = -1
        if train_mask is not None:
            valid = (yy >= 0) & train_mask
        else:
            valid = yy >= 0

        valid_flat = yy[valid].astype(np.int64)
        if valid_flat.size == 0:
            raise ValueError("No valid tokens for weight computation.")

        unique, counts = np.unique(valid_flat, return_counts=True)
        total = counts.sum()

        raw = {c: (total / cnt) for c, cnt in zip(unique, counts)}

        if use_log_scale:
            raw = {c: np.log(alpha + w) for c, w in raw.items()}

        mean_w = np.mean(list(raw.values()))
        raw = {c: raw[c] / mean_w for c in raw}

        bounded = {c: float(np.clip(raw[c], min_weight, max_weight)) for c in raw}

        weights = np.array(
            [bounded.get(i, 1.0) for i in range(self.num_classes_)], dtype=np.float32
        )

        wt = torch.tensor(weights)
        wt = torch.pow(wt, 1.0 / temperature)

        if normalize:
            wt = wt / wt.sum()
        return wt.to(self.device)

    def init_transformers(
        self,
    ) -> Tuple[AutoEncoderFeatureTransformer, Plotting, Scorer]:
        """Initialize the transformers for encoding.

        This method should be called in a `fit` method to initialize the transformers. It returns the transformers and utilities.

        Returns:
            Tuple[AutoEncoderFeatureTransformer, Plotting, Scorer]: Transformers and utilities.
        """
        # Transformers for encoding
        feature_transformer = AutoEncoderFeatureTransformer(
            num_classes=self.num_classes_,
            return_int=False,
            activate=self.activate_,
            logger=self.logger,
            verbose=self.verbose,
            debug=self.debug,
        )

        # Initialize plotter.
        plotter = Plotting(
            model_name=self.model_name,
            prefix=self.prefix,
            output_dir=self.output_dir,
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
            average=self.scoring_averaging,
            logger=self.logger,
            verbose=self.verbose,
            debug=self.debug,
        )

        return feature_transformer, plotter, scorers

    def _objective(self, trial: optuna.Trial, model: torch.nn.Module) -> float:
        """Objective function for Optuna hyperparameter tuning.

        This method is the objective function for hyperparameter tuning using Optuna. It trains the model using the given hyperparameters and returns the validation loss.

        Args:
            trial (optuna.Trial): Optuna trial object.
            model (torch.nn.Module): The model to train.

        Returns:
            float: Validation loss.
        """
        msg = "Method `_objective()` must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def fit(
        self,
        X: np.ndarray | pd.DataFrame | list,
        y: np.ndarray | pd.DataFrame | list | None = None,
    ):
        """Fit the model using the input data.

        Args:
            X (np.ndarray | pd.DataFrame | list): Input data.
            y (np.ndarray | pd.DataFrame | list): Target labels.

        """
        msg = "Method ``fit()`` must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def transform(self, X: np.ndarray | pd.DataFrame | list) -> np.ndarray:
        """Transform the input data using the trained model.

        Args:
            X (np.ndarray | pd.DataFrame | list): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        msg = "Method ``transform()`` must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def _enforce_min_observed_per_class_prop(
        self,
        y: np.ndarray,
        sim_mask: np.ndarray,
        original_missing_mask: np.ndarray,
        *,
        min_prop_per_class: float = 0.20,
        per_class_min: int = 0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Relax the simulated-missing mask so each class retains enough observed tokens.

        This function adjusts `sim_mask` (True = simulated-missing) to ensure that, for each genotype class c ∈ {0,1,2}, the number of **observed** training tokens (i.e., not simulated-missing, not originally-missing) is at least a target: target_c = max(ceil(min_prop_per_class * total_c), per_class_min). It never alters `original_missing_mask` (real missing), only simulated tokens.  Selection is **random within class** to avoid bias.

        Args:
            y (np.ndarray): Integer-encoded targets with shape (N, L). Values in {0,1,2} or -1 for missing.
            sim_mask (np.ndarray): Boolean mask (N, L); True where value was *simulated* as missing.
            original_missing_mask (np.ndarray): Boolean mask (N, L); True where value was originally missing.
            min_prop_per_class (float): Minimum *proportion* of each class's total tokens to keep observed in training.
            per_class_min (int): Optional absolute floor per class (applies in addition to proportion).
            rng (np.random.Generator | None): Optional RNG for reproducibility.

        Returns:
            np.ndarray: A possibly *relaxed* sim_mask meeting the per-class floors.

        Raises:
            ValueError: If shapes mismatch or if no valid tokens exist.
        """
        if rng is None:
            rng = np.random.default_rng()

        if y.shape != sim_mask.shape or y.shape != original_missing_mask.shape:
            msg = "y, sim_mask, and original_missing_mask must share the same shape."
            self.logger.error(msg)
            raise ValueError(msg)

        # Valid tokens (non-missing labels)
        valid = (y >= 0) & (~np.isnan(y))  # (N, L) boolean
        if not np.any(valid):
            return sim_mask

        # Observed training tokens = valid & not originally-missing & not simulated
        observed_train = valid & (~original_missing_mask) & (~sim_mask)

        # Totals per class in the dataset (among valid tokens only)
        num_classes = int(np.nanmax(y[valid]) + 1)
        y_flat = y[valid].astype(np.int64).ravel()
        totals = np.bincount(y_flat, minlength=num_classes)  # total tokens per class

        # Current observed counts per class (among tokens that will contribute to loss)
        y_obs_flat = y[observed_train].astype(np.int64).ravel()
        observed_counts = np.bincount(y_obs_flat, minlength=num_classes)

        # Targets per class: proportion-based with optional absolute floor
        targets = np.maximum(
            np.ceil(min_prop_per_class * totals).astype(int),
            int(per_class_min),
        )

        # Compute deficits
        deficits = np.clip(targets - observed_counts, 0, None)

        # For each class with a deficit, randomly un-simulate that many tokens from the class
        for c, deficit in enumerate(deficits):
            if deficit <= 0:
                continue

            # Candidates: currently simulated, not originally missing, class == c
            pool = sim_mask & (~original_missing_mask) & (y == c)
            if not np.any(pool):
                continue  # nothing to relax for this class

            # Choose up to 'deficit' positions uniformly at random
            idx = np.argwhere(pool)
            k = min(deficit, idx.shape[0])
            choose = idx[rng.choice(idx.shape[0], size=k, replace=False)]

            # Flip those from simulated to observed
            sim_mask[tuple(choose.T)] = False

        return sim_mask

    def _class_balanced_weights_from_mask(
        self,
        y: np.ndarray,
        train_mask: np.ndarray,
        num_classes: int,
        beta: float = 0.9999,
        max_ratio: float = 5.0,
    ) -> torch.Tensor:
        """Class-balanced weights (Cui et al. 2019) computed on the tokens that actually train.

        Args:
            y (np.ndarray): The target labels.
            train_mask (np.ndarray): The mask indicating which samples are in the training set.
            num_classes (int): The number of classes.
            beta (float, optional): The beta parameter for computing effective number. Defaults to 0.9999.
            max_ratio (float, optional): The maximum ratio for class weights. Defaults to 5.0.

        Returns:
            torch.Tensor: The class-balanced weights.
        """
        valid = (y >= 0) & train_mask
        model_cls, cnt = np.unique(y[valid].astype(np.int64), return_counts=True)
        counts = np.zeros(num_classes, dtype=np.float64)
        counts[model_cls] = cnt

        # Effective number
        eff = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / (eff + 1e-12)
        w[counts == 0] = 0.0  # no samples => no weight

        # Normalize & cap spread
        w = w / (w.mean() + 1e-12)
        w = np.clip(w, w.min(), w.max())
        if w.max() / (w[w > 0].min() if np.any(w > 0) else 1.0) > max_ratio:
            # compress dynamically
            scale = max_ratio / (w.max() / (w[w > 0].min() + 1e-12))
            w = 1.0 + (w - 1.0) * scale

        return torch.tensor(w.astype(np.float32))

    def _select_device(self, device: Literal["gpu", "cpu", "mps"]) -> torch.device:
        """Selects the appropriate device for PyTorch.

        This method selects the appropriate device for PyTorch based on the input device string. It checks if a GPU device is available and selects it if the input device is "gpu". If no GPU device is available, it falls back to the CPU device. If the input device is "cpu", it selects the CPU device.

        Args:
            device (Literal["gpu", "cpu", "mps"]): Device to use.

        Returns:
            torch.device: The selected PyTorch device.
        """
        device = device.lower().strip()

        # Validate device selection.
        if device not in {"gpu", "cpu", "mps"}:
            msg = f"Invalid device: {device}. Must be one of 'gpu', 'cpu', or 'mps'."
            self.logger.error(msg)
            raise ValueError(msg)

        if device == "cpu":
            self.logger.info("Using PyTorch device: CPU.")
            return torch.device("cpu")

        if torch.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            self.logger.warning("No GPU device could be found. Using CPU.")
            device = torch.device("cpu")

        self.logger.info(f"Using PyTorch device: {device}.")
        return device

    def _create_model_directories(
        self, prefix: str, output_dir: str | Path, outdirs: List[str]
    ) -> None:
        """Create the necessary directories for the model outputs.

        This method creates the necessary directories for the model outputs, including directories for models, plots, metrics, and optimization results.

        Args:
            prefix (str): Prefix for the output directory.
            output_dir (str | Path): Output directory name.
            outdirs (List[str]): List of subdirectories to create.

        Raises:
            Exception: If any of the directories cannot be created.
        """
        self.logger.debug(
            f"Creating model directories in {output_dir} with prefix {prefix}."
        )
        self.formatted_output_dir = Path(f"{prefix}_{output_dir}")
        self.base_dir = self.formatted_output_dir / "Unsupervised"

        for d in outdirs:
            subdir = self.base_dir / d / self.model_name
            setattr(self, f"{d}_dir", subdir)
            try:
                getattr(self, f"{d}_dir").mkdir(parents=True, exist_ok=True)
            except Exception as e:
                msg = f"Failed to create directory {getattr(self, f'{d}_dir')}: {e}"
                self.logger.error(msg)
                raise Exception(msg)

    def _count_num_classes(self, X: np.ndarray, mask: np.ndarray) -> int:
        """Count distinct non-missing classes (0,1,2) safely.

        This method counts the number of distinct classes present in the observed (non-missing) values of the input genotype matrix. It ignores missing values indicated by the mask.

        Args:
            X (np.ndarray): Raw genotype matrix with -1/NaN as missing.
            mask (np.ndarray): Boolean mask True for missing/invalid.

        Returns:
            int: Number of classes present among observed values.
        """
        observed = (~mask) & (X >= 0)
        return int(np.unique(X[observed]).size)
