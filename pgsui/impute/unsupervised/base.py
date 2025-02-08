import copy
import json
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as optim
from sklearn.base import BaseEstimator, TransformerMixin
from snpio.utils.logging import LoggerManager
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from pgsui.data_processing.transformers import (
    AutoEncoderFeatureTransformer,
    SimGenotypeDataTransformer,
)
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.dataset import CustomTensorDataset
from pgsui.impute.unsupervised.nn_scorers import Scorer
from pgsui.utils.misc import validate_input_type
from pgsui.utils.plotting import Plotting


class BaseNNImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        prefix: str = "pgsui",
        output_dir: str | Path = "output",
        device: Literal["gpu", "cpu"] = "gpu",
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

    def _select_device(self, device: Literal["gpu", "cpu"]) -> torch.device:
        """Selects the appropriate device for PyTorch.

        This method selects the appropriate device for PyTorch based on the input device string. It checks if a GPU device is available and selects it if the input device is "gpu". If no GPU device is available, it falls back to the CPU device. If the input device is "cpu", it selects the CPU device.

        Args:
            device (Literal["gpu", "cpu"]): Device to use.

        Returns:
            torch.device: The selected PyTorch device.
        """
        device = device.lower().strip()

        # Validate device selection.
        if device not in {"gpu", "cpu"}:
            msg = f"Invalid device: {device}. Must be one of 'gpu' or 'cpu'."
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
                self.logger.error(
                    f"Failed to create directory {getattr(self, f'{d}_dir')}: {e}"
                )
                raise e

    def tune_hyperparameters(self) -> None:
        """Tune hyperparameters using Optuna study.

        This method tunes the hyperparameters of the model using Optuna. It creates an Optuna study and optimizes the model hyperparameters using the `objective()` method. The method saves the best hyperparameters to a JSON file and plots the optimization results.

        Raises:
            ValueError: If the model is not fitted yet.
            NotImplementedError: If the `objective()` method is not implemented in the child class.
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

        if not hasattr(self, "objective"):
            msg = "Method `objective()` must be implemented in the child class."
            self.logger.error(msg)
            raise NotImplementedError(msg)

        study.optimize(
            lambda trial: self.objective(trial, self.Model),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
        )

        best_metric = study.best_value
        best_params = study.best_params

        # Set the best parameters.
        # NOTE: `set_best_params()` must be implemented in the child class.
        if not hasattr(self, "set_best_params"):
            msg = "Method `set_best_params()` must be implemented in the child class."
            self.logger.error(msg)
            raise NotImplementedError(msg)

        self.best_params_ = self.set_best_params(best_params)
        self.model_params.update(self.best_params_)
        self.logger.info(f"Best {self.tune_metric} metric: {best_metric}")
        self.logger.info("Best parameters:")
        best_params_tmp = copy.deepcopy(best_params)
        best_params_tmp["learning_rate"] = self.lr_
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
            model (nn.Module): The model whose parameters to reset.
        """
        for layer in model.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def train_and_validate_model(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        l1_penalty: float,
        lr: float,
        trial=None,
        return_history: bool = False,
    ):
        """Train the model, returning the best loss and final model.

        This method trains the model using the training DataLoader and evaluates it using the validation DataLoader. It returns the best validation loss, the best model, and optionally the training and validation histories if `return_history` is True. The method initializes the model, optimizer, and scheduler, and executes the training loop. It also handles early stopping and Optuna pruning. The method returns the best validation loss, the best model, and optionally the training and validation histories if `return_history` is True. The method also returns the best model from the training loop.

        Args:
            model (nn.Module): The initialized model.
            loader (DataLoader): Training DataLoader.
            l1_penalty (float): L1 coefficient for regularization.
            lr (float): Learning rate for input refinement.
            trial: Optuna trial object (if tuning).
            return_history (bool): Whether to return per-epoch history.

        Returns:
            (float, nn.Module, dict) or (float, nn.Module):
                The best validation loss, trained model, and optionally the training histories if `return_history` is True.

        Raises:
            TypeError: If model or loader are not initialized.
        """
        if model is None:
            msg = "Model is not initialized."
            self.logger.error(msg)
            raise TypeError(msg)

        if loader is None:
            msg = "DataLoader is not initialized."
            self.logger.error(msg)
            raise TypeError(msg)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        (best_loss, best_model, train_hist) = self.execute_training_loop(
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            l1_penalty=l1_penalty,
            trial=trial,
            return_history=return_history,
        )

        # Return best of Phase 3
        if return_history:
            return (best_loss, best_model, {"Train": train_hist})
        else:
            return best_loss, best_model

    def execute_training_loop(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        model: torch.nn.Module,
        l1_penalty: float,
        trial,
        return_history: bool,
    ):
        """Execute the training loop.

        This method executes the training loop for the model. It trains the model using the training DataLoader and evaluates it using the validation DataLoader. It returns the best validation loss, the best model, and optionally the training and validation histories if `return_history` is True.

        Args:
            loader (torch.utils.data.DataLoader): Training DataLoader.
            optimizer (Optimizer): Optimizer for this phase.
            scheduler (_LRScheduler): Scheduler for this phase.
            model (nn.Module): The UBP model.
            l1_penalty (float): L1 penalty coefficient.
            trial: Optuna trial object if tuning.
            return_history (bool): Whether to return history for each epoch.

        Returns:
            tuple: (best_loss, best_model, train_history) if `return_history` is True, else (best_loss, best_model).
        """
        best_loss = float("inf")
        best_model = None
        train_history = []

        # Early stopping or other controls
        early_stopping = EarlyStopping(
            patience=self.early_stop_gen,
            verbose=self.verbose,
            prefix=self.prefix,
            min_epochs=self.min_epochs,
            debug=self.debug,
        )

        for epoch in range(self.epochs):
            # ---- TRAIN STEP ----
            train_loss = self.train_step(loader, optimizer, model, l1_penalty)

            scheduler.step()

            if return_history:
                train_history.append(train_loss)

            # ---- OPTUNA PRUNING ----
            if trial is not None and self.tune:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    trial.report(train_loss, epoch)
                    if trial.should_prune():
                        self.logger.warning(
                            f"Error encountered in trial {trial.number}. Pruning."
                        )
                        self.logger.warning(f"Current train loss: {train_loss}")
                        raise optuna.exceptions.TrialPruned()

            # ---- EARLY STOPPING ----
            early_stopping(train_loss, model)
            if early_stopping.early_stop:
                best_loss = early_stopping.best_score
                best_model = copy.deepcopy(early_stopping.best_model)
                break

        # If we never triggered early stopping, get the best from the end
        if best_model is None:
            best_loss = early_stopping.best_score
            best_model = copy.deepcopy(early_stopping.best_model)

        return best_loss, best_model, train_history

    def train_step(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        l1_penalty: float,
    ):
        """Perform one epoch of training.

        This method performs one epoch of training using the training DataLoader. It computes the loss for each batch and updates the model weights. If an L1 penalty is specified, it is applied to the loss. The method returns the average training loss over the epoch. The model is set to training mode during training. The method also clips the gradients to prevent exploding gradients. The average training loss is computed as the total loss divided by the number of batches. The method returns the average training loss over the epoch.

        Args:
            loader (torch.utils.data.DataLoader): Training DataLoader.
            optimizer (Optimizer): Current phase optimizer.
            model (nn.Module): The model to train.
            l1_penalty (float): L1 penalty coefficient.

        Returns:
            float: The average training loss over the epoch.

        Raises:
            ValueError: If the output shape is invalid.
        """
        model.train()
        running_loss = 0.0
        num_batches = 0

        for X_batch, y_batch, mask_batch, _ in loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            if logits.dim() == 2:
                logits = logits.view(
                    X_batch.size(0), self.num_features_, self.num_classes_
                )
            elif logits.dim() != 3:
                msg = f"Invalid output shape: {outputs.shape}. Must be 2D or 3D."
                self.logger.error(msg)
                raise ValueError(msg)

            if isinstance(outputs, tuple):
                z_mean, z_log_var = outputs[1], outputs[2]
                outputs = (logits, z_mean, z_log_var)
            else:
                outputs = logits

            # Compute loss
            loss = model.compute_loss(
                y_batch, outputs, mask_batch, class_weights=self.class_weights_
            )

            # Add L1 penalty
            if l1_penalty > 0:
                l1_reg = 0.0
                for param in model.parameters():
                    l1_reg += param.abs().sum()
                loss = loss + l1_penalty * l1_reg

            loss.backward()

            # First, update model weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / num_batches
        return epoch_loss

    def predict(
        self,
        Xenc: np.ndarray | torch.Tensor | pd.DataFrame | list,
        model: torch.nn.Module,
        return_proba: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Predict using the trained model.

        This method predicts the labels for the input data using the trained model. It returns the predicted labels. If ``return_proba`` is True, it also returns the predicted probabilities.

        Args:
            Xenc (np.ndarray): Encoded input data.
            model (torch.nn.Module): Trained model.
            return_proba (bool, optional): Return probabilities. Defaults to False.

        Returns:
            np.ndarray: Predicted labels. If ``return_proba`` is True, instead returns a Tuple of predicted labels and probabilities.

        Raises:
            AttributeError: If the model is not fitted yet.
            TypeError: If the transformer is not initialized.
            ValueError: If the input shape is invalid.
        """
        if model is None:
            msg = "Model is not fitted yet. Call `fit()` before prediction."
            self.logger.error(msg)
            raise AttributeError(msg)

        self.ensure_attribute("tt_")

        if self.tt_ is None:
            msg = "Transformer (`tt_`) has not been initialized."
            self.logger.error(msg)
            raise TypeError(msg)

        Xtensor = validate_input_type(Xenc, "tensor")

        model.eval()
        with torch.no_grad():
            if self.is_backprop:
                outputs = model.phase23_decoder(Xtensor)
            else:
                outputs = model(Xtensor.to(self.device))

        # If the model returns multiple outputs, assume first is recon logits
        recon_logits = outputs[0] if isinstance(outputs, tuple) else outputs

        if recon_logits.dim() < 3:
            recon_logits = recon_logits.view(-1, self.num_features_, self.num_classes_)

        y_pred_proba = torch.softmax(recon_logits, dim=-1)

        # Reshape only if necessary
        if y_pred_proba.dim() < 3:
            y_pred_proba = y_pred_proba.view(-1, self.num_features_, self.num_classes_)
        y_pred_proba = validate_input_type(y_pred_proba)
        y_pred_labels = self.tt_.inverse_transform(y_pred_proba)

        # Safety check
        if np.isnan(y_pred_labels).any() or np.any(y_pred_labels < 0):
            msg = "Imputation failed: Missing values remain after prediction."
            self.logger.error(msg)
            raise ValueError(msg)

        return (y_pred_labels, y_pred_proba) if return_proba else y_pred_labels

    def impute(
        self, X: np.ndarray | pd.DataFrame | list | torch.Tensor, model: torch.nn.Module
    ) -> np.ndarray:
        """Impute the real missing values in X using the trained model.

        This method imputes the real missing values in the input data using the trained model. It returns the imputed data. The input data must be encoded. The method overwrites only the real missing values in the input data.

        Args:
            X (numpy.ndarray | pd.DataFrame | list | torch.Tensor): Data to impute.
            model (torch.nn.Module): Trained model.

        Returns:
            numpy.ndarray: Imputed data.

        Raises:
            TypeError: If the model is not fitted yet.
            ValueError: If the input shape is invalid.
            ValueError: If the shape of the real missing mask and predictions do not match.
        """
        self.ensure_attribute("original_missing_mask_")

        if model is None:
            msg = "Model is not fitted yet. Call `fit()` before imputation."
            self.logger.error(msg)
            raise TypeError(msg)

        # Convert X to array, unify missing indicator.
        X = np.where(np.logical_or(X < 0, np.isnan(X)), -1, X)

        if not self.is_backprop:
            if X.ndim == 2:
                Xtensor = validate_input_type(self.tt_.transform(X), "tensor")
                X_imputed = X.copy()
            elif X.ndim == 3:
                Xtensor = validate_input_type(X, "tensor")
                X_imputed = self.tt_.inverse_transform(X)
            else:
                msg = f"Invalid input shape: {X_imputed.shape}. Must be 2D or 3D."
                self.logger.error(msg)
                raise ValueError(msg)
        else:
            Xtensor = self.loader_.dataset.data
            if X.ndim == 2:
                X_imputed = X.copy()
            elif X.ndim == 3:
                X_imputed = self.tt_.inverse_transform(X)
            else:
                msg = f"Invalid input shape: {X_imputed.shape}. Must be 2D or 3D."
                self.logger.error(msg)
                raise ValueError(msg)

        model.eval()
        with torch.no_grad():
            if self.model_name == "ImputeUBP":
                outputs = model.phase23_decoder(Xtensor)
            else:
                outputs = model(Xtensor.to(self.device))

        recon_logits = outputs[0] if isinstance(outputs, tuple) else outputs
        recon_logits = recon_logits.view(-1, self.num_features_, self.num_classes_)
        y_pred_proba = torch.softmax(recon_logits, dim=-1)
        y_pred_proba = validate_input_type(y_pred_proba)
        y_pred_labels = self.tt_.inverse_transform(y_pred_proba)

        # Overwrite only real missing values
        real_missing_mask = self.original_missing_mask_

        if real_missing_mask.shape != y_pred_labels.shape:
            msg = (
                f"Shape mismatch between real_missing_mask "
                f"({real_missing_mask.shape}) and predictions "
                f"({y_pred_labels.shape})."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Only overwrite the real missing values.
        X_imputed = np.where(real_missing_mask, X_imputed, y_pred_labels)
        return X_imputed

    def get_data_loaders(
        self,
        Xenc: np.ndarray | torch.Tensor | list,
        y: np.ndarray | torch.Tensor | list,
        latent_dim: int | None = None,
    ) -> Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:
        """Create DataLoader objects for the input data.

        This method creates DataLoader objects for the input data. It splits the data into training, validation, and test sets. It also creates a missing value mask for each set. The method returns the DataLoader objects for the training, validation, and test sets. If the model is a backpropagation model, the method initializes the latent space with random values. Otherwise, it splits the data into training, validation, and test sets and re-encodes the input data.

        Args:
            Xenc (np.ndarray | torch.Tensor | list): Encoded input data.
            y (np.ndarray | torch.Tensor | list): Target labels.
            latent_dim (int, optional): Latent dimension for backpropagation models. Defaults to None.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Training, validation, and test DataLoader objects.

        Raises:
            ValueError: If the latent dimension is not specified for backpropagation models.
        """
        # Decode input data
        Xenc = validate_input_type(Xenc)
        y_decoded = validate_input_type(y)

        if self.is_backprop:
            if latent_dim is None:
                msg = "Latent dimension must be specified for backpropagation models."
                self.logger.error(msg)
                raise ValueError(msg)
            Xlatent = torch.zeros(y.shape[0], latent_dim)
            X_train = torch.nn.init.xavier_uniform_(Xlatent)
        else:
            X_train = Xenc.copy()

        y_train = y.copy()
        sim_mask_train = self.sim_missing_mask_

        # Transform and re-encode datasets
        train_data = self._process_data_subset(X_train, y_train)
        sim_mask_train = torch.tensor(sim_mask_train, dtype=torch.bool)

        # Create DataLoader objects
        train_loader = self._create_dataloader(
            *train_data, mask=sim_mask_train, shuffle=True
        )

        return train_loader

    def _process_data_subset(self, X, y):
        """Process a subset of the data.

        This method processes a subset of the data, including missing values and creating a missing value mask.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target labels.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - X_final: Processed input features.
            - y_final: Processed target labels.
            - mask: Missing value
        """
        try:
            X_final = np.where(np.logical_or(X < 0, np.isnan(X)), -1, X)
            y_final = np.where(np.logical_or(y < 0, np.isnan(y)), -1, y)
            observed_mask = np.logical_and(y_final >= 0, ~np.isnan(y_final))
            X_final = np.where(observed_mask[:, :, np.newaxis], X_final, -1)
            y_final = np.where(observed_mask, y_final, -1)
            return X_final, y_final
        except ValueError:
            y_final = np.where(np.logical_or(y < 0, np.isnan(y)), -1, y)
            observed_mask = np.logical_and(y_final >= 0, ~np.isnan(y_final))
            y_final = np.where(observed_mask, y_final, -1)
            return X, y_final

    def _create_dataloader(self, X, y, mask, shuffle: bool):
        """Helper method to create a PyTorch DataLoader.

        Args:
            X (torch.Tensor): Input features.
            y (torch.Tensor): Target labels.
            mask (torch.Tensor): Missing value mask.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: PyTorch DataLoader object.
        """
        X_tensor = validate_input_type(X, "tensor")
        y_tensor = validate_input_type(y, "tensor")
        mask_tensor = validate_input_type(mask, "tensor")
        dataset = CustomTensorDataset(
            X_tensor, y_tensor, mask_tensor, logger=self.logger
        )

        kwargs = {"batch_size": self.batch_size, "shuffle": shuffle}
        return DataLoader(dataset, **kwargs)

    @staticmethod
    def initialize_weights(module):
        """Initialize the weights of the neural network model.

        This method initializes the weights of the neural network model using Xavier initialization for the weights and zero initialization for the biases.

        Args:
            module (torch.nn.Module): Module to initialize.
        """
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
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
        }
        all_prms.update(model_params)
        return Model(**all_prms).to(self.device)

    def train_final_model(
        self, loader: DataLoader
    ) -> Tuple[float, torch.nn.Module, dict]:
        """Train the final model.

        This method trains the final model using the training DataLoader. It returns the loss, trained model, and training history. If the validation DataLoader is provided, it also returns the validation history. The method saves the final model to a file. The method also plots the training and validation histories.

        Args:
            loader (torch.utils.data.DataLoader): Training DataLoader.

        Returns:
            Tuple[float, torch.nn.Module, dict]: Loss, model, and training history. If validation DataLoader is provided, also returns validation history.

        Raises:
            AttributeError: If the DataLoader objects do not exist.
            AttributeError: If model parameters are not provided.
            AttributeError: If learning rate is not provided.
            AttributeError: If L1 penalty is not provided.
        """
        which_model = "tuned model" if self.tune else "model"
        self.logger.info(f"Training the {which_model}...")

        attributes = ["Model", "model_params"]
        attributes += ["learning_rate", "l1_penalty", "lr_patience", "gamma"]

        if self.model_name == "ImputeVAE":
            attributes += ["beta"]

        [self.ensure_attribute(attr) for attr in attributes]

        self.lr_ = self.learning_rate
        self.lr_patience_ = self.lr_patience
        self.l1_penalty_ = self.l1_penalty

        # Build model
        model = self.build_model(self.Model, self.model_params)
        model.apply(self.initialize_weights)
        self.optimizer_ = optim.Adam(model.parameters(), lr=self.lr_)

        args = [model, loader, self.l1_penalty_, self.lr_]
        res = self.train_and_validate_model(*args, return_history=True)

        if len(res) == 4:
            loss, trained_model, history, self.loader_ = res
        else:
            loss, trained_model, history = res

        if trained_model is None:
            msg = "Model was not properly trained. Check the training process."
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Save final model
        fn = self.models_dir / "final_model.pt"
        torch.save(trained_model.state_dict(), fn)

        return loss, trained_model, history

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
            msg = "Attribute must be a string."
            self.logger.error(msg)
            raise TypeError(msg)

        if not hasattr(self, attribute):
            msg = f"{self.model_name} has no attribute '{attribute}'."
            self.logger.error(msg)
            raise AttributeError(msg)

    def evaluate_model(
        self,
        objective_mode: bool = False,
        trial: optuna.Trial | None = None,
        model: torch.nn.Module | None = None,
        loader: torch.utils.data.DataLoader | None = None,
    ):
        """Peform evaluation of the model.

        This method evaluates the model using the test set. It returns the evaluation metrics and plots the results. If in objective mode, it returns the metrics for Optuna to optimize. The model must be provided as a keyword argument. If the DataLoader is not provided, it must be an attribute of the class. If in objective mode, the trial object must be provided. The method also saves the evaluation metrics to a JSON file. The method also plots the evaluation metrics and confusion matrix.

        Args:
            objective_mode (bool, optional): Whether to run in objective mode. Defaults to False.
            trial (optuna.Trial, optional): Optuna trial object. Defaults to None.
            model (torch.nn.Module): Model to evaluate. Required keyword argument.
            loader (torch.utils.data.DataLoader, optional): DataLoader for evaluation. Defaults to None.

        Returns:
            dict: Evaluation metrics. Only returned in objective mode. The dictionary contains the evaluation metrics, including: accuracy, precision, recall, average precision, f1-score, ROC-AUC, and precision-recall macro ('pr_macro') averaged scores.

        Raises:
            TypeError: If objective_mode is True but trial is not provided.
            TypeError: If model is not provided.
        """
        if objective_mode and trial is None:
            msg = "Trial object must be provided for objective mode."
            self.logger.error(msg)
            raise TypeError(msg)

        if model is None:
            msg = "Model must be provided for evaluation, but got None."
            self.logger.error(msg)
            raise TypeError(msg)

        if loader is None:
            msg = "Data loader must be provided for evaluation, but got None."
            self.logger.error(msg)
            raise TypeError(msg)

        # Data preparation.
        X = validate_input_type(loader.dataset.data)
        y_true_labels = validate_input_type(loader.dataset.target)
        mask_test = validate_input_type(loader.dataset.mask)

        if np.all(np.logical_or(y_true_labels < 0, np.isnan(y_true_labels))):
            msg = "No valid classes found in the target data."
            self.logger.error(msg)
            raise ValueError(msg)

        if y_true_labels.ndim == 2:
            y_true_enc = self.tt_.transform(y_true_labels)
        elif y_true_labels.ndim == 3:
            y_true_enc = y_true_labels.copy()
            y_true_labels = np.argmax(y_true_labels, axis=-1)
        else:
            msg = f"Invalid target shape: {y_true_labels.shape}. Must be 2D or 3D."
            self.logger.error(msg)
            raise ValueError(msg)

        # Predict and filter missing data
        pred_labels, pred_proba = self.predict(X, model, return_proba=True)

        y_true_labels, pred_labels, pred_proba, y_true_enc = (
            y_true_labels[mask_test],
            pred_labels[mask_test],
            pred_proba[mask_test],
            y_true_enc[mask_test],
        )

        # Efficient metric computation
        metrics = self.scorers_.evaluate(
            y_true_labels,
            pred_labels,
            y_true_enc,
            pred_proba,
            objective_mode,
            self.tune_metric,
        )

        if objective_mode:
            return metrics

        # Save and plot metrics
        id_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = self.metrics_dir / f"metrics_{id_str}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as fp:
            json.dump(metrics, fp, indent=4)

        self.plotter_.plot_metrics(y_true_labels, pred_proba, metrics)
        self.plotter_.plot_confusion_matrix(y_true_labels, pred_labels)
        return metrics

    def compute_class_weights(
        self,
        X: np.ndarray,
        use_log_scale: bool = False,
        alpha: float = 1.0,
        normalize: bool = False,
        temperature: float = 1.0,
        max_weight: float = 10.0,
        min_weight: float = 0.01,
    ) -> torch.Tensor:
        """Compute inverse class weights based on class distribution in the data.

        This method computes the inverse class weights based on the class distribution in the data. It optionally applies a log-scale transform and normalizes the weights. The weights are bounded within a range, and a temperature parameter can be used for smoothing. The alpha parameter is used for the log-scale transform to adjust the weights.

        Args:
            X (numpy.ndarray): Input data.
            use_log_scale (bool, optional): Use log-scale transform. Defaults to True.
            alpha (float, optional): Alpha parameter for log-scale transform. Defaults to 1.0.
            normalize (bool, optional): Normalize the weights. Defaults to False.
            temperature (float, optional): Temperature parameter for smoothing. Defaults to 1.0.
            max_weight (float, optional): Maximum weight. Defaults to 20.0.
            min_weight (float, optional): Minimum weight. Defaults to 0.5.

        Returns:
            torch.Tensor: Class weights.
        """
        self.logger.info("Computing class weights...")

        # Flatten the array to calculate class frequencies
        X = np.where(np.logical_or(X < 0, self.original_missing_mask_), -1, X)
        flat_X = X.flatten()
        valid_classes = flat_X[flat_X >= 0]  # Exclude missing values (< 0)

        if valid_classes.size == 0:
            msg = "No valid classes found in the data."
            self.logger.error(msg)
            raise ValueError(msg)

        # Count the occurrences of each class
        unique, counts = np.unique(valid_classes, return_counts=True)
        total_counts = np.sum(counts)

        # Compute raw inverse class weights
        raw_weights = {
            enc: (total_counts / count if count > 0 else 0.0)
            for enc, count in zip(unique, counts)
        }

        # Optionally apply a log-scale transform
        if use_log_scale:
            raw_weights = {enc: np.log(alpha + raw_weights[enc]) for enc in raw_weights}

        mean_val = np.mean(list(raw_weights.values()))
        raw_weights = {enc: raw_weights[enc] / mean_val for enc in raw_weights}

        # Bound the weights within [min_weight, max_weight]
        bounded_weights = {}
        for enc in raw_weights:
            w = raw_weights[enc]
            w = max(min_weight, min(max_weight, w))
            bounded_weights[enc] = w

        # Convert final weights to a tensor
        final_weight_list = [
            bounded_weights.get(class_index, 1.0)
            for class_index in range(self.num_classes_)
        ]
        final_weight_list = np.array(final_weight_list, dtype=np.float32)

        # Smooth class weights
        weight_tensor = validate_input_type(final_weight_list, "tensor")
        weight_tensor = torch.pow(weight_tensor, 1 / temperature)

        if normalize:
            weight_tensor = weight_tensor / weight_tensor.sum()

        cw = validate_input_type(weight_tensor)
        self.logger.info("Class weights:")
        [self.logger.info(f"Class {i}: {x:.3f}") for i, x in enumerate(cw)]

        return (
            weight_tensor
            if self.model_name == "ImputeUBP"
            else weight_tensor.to(self.device)
        )

    def init_transformers(
        self,
    ) -> Tuple[
        AutoEncoderFeatureTransformer, SimGenotypeDataTransformer, Plotting, Scorer
    ]:
        """Initialize the transformers for encoding.

        This method should be called in a `fit` method to initialize the transformers. It returns the transformers and utilities.

        Returns:
            Tuple[AutoEncoderFeatureTransformer, SimGenotypeDataTransformer, Plotting, Scorer]: Transformers and utilities.
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

        sim_transformer = SimGenotypeDataTransformer(
            self.genotype_data,
            prop_missing=self.sim_prop_missing,
            strategy=self.sim_strategy,
            seed=self.seed,
            logger=self.logger,
            verbose=self.verbose,
            debug=self.debug,
            class_weights=self.class_weights_,
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

        return feature_transformer, sim_transformer, plotter, scorers

    def objective(self, trial, model):
        """Objective function for Optuna hyperparameter tuning.

        This method is the objective function for hyperparameter tuning using Optuna. It trains the model using the given hyperparameters and returns the validation loss.

        Args:
            trial (optuna.Trial): Optuna trial object.
            model (torch.nn.Module): The model to train.

        Returns:
            float: Validation loss.
        """
        raise NotImplementedError(
            "Method `objective()` must be implemented in the child class."
        )

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
        raise NotImplementedError(
            "Method `fit()` must be implemented in the child class."
        )

    def transform(self, X: np.ndarray | pd.DataFrame | list) -> np.ndarray:
        """Transform the input data using the trained model.

        Args:
            X (np.ndarray | pd.DataFrame | list): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        raise NotImplementedError(
            "Method `transform()` must be implemented in the child class."
        )
