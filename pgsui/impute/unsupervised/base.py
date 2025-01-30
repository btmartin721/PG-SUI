import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as optim
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
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
        prefix: str = "pgsui",
        output_dir: str | Path = "output",
        device: Literal["gpu"] | Literal["cpu"] = "gpu",
        verbose: int = 0,
        debug: bool = False,
    ):
        """Base class for neural network imputers.

        This class is a base class for neural network imputers. It includes methods for training and evaluating the model, as well as for hyperparameter tuning.

        Args:
            prefix (str, optional): Prefix for the output directory. Defaults to "pgsui".
            output_dir (str | Path, optional): Output directory name. Defaults to "output".
            device (Literal["gpu"] | Literal["cpu"], optional): PyTorch Device. Will use GPU if "gpu" is specified and if a valid GPU device can be found. Defaults to "gpu".
            verbose (int, optional): Verbosity level. Defaults to 0.
            debug (bool, optional): Debug mode. Defaults to False.

        Raises:
            ValueError: If the optimizer is not supported.
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

    def _select_device(self, device: Literal["gpu"] | Literal["cpu"]) -> torch.device:
        """Selects the appropriate device for PyTorch.

        MPS is preferred if available, otherwise CUDA, and finally CPU. MPS is only available on recent MacOS versions with the M chips. CUDA is used for Linux and Windows systems with compatible GPUs. Otherwise, CPU is used.

        Args:
            device (Literal["gpu"] | Literal["cpu"]): Device to use.

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

        Args:
            prefix (str): Prefix for the output directory.
            output_dir (str | Path): Output directory name.
            outdirs (List[str]): List of subdirectories to create.
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
                raise

    def tune_hyperparameters(self) -> None:
        """Tune hyperparameters using Optuna.

        This method tunes the hyperparameters of the model using Optuna. It uses the objective function to optimize the hyperparameters and saves the best parameters to a JSON file.

        Raises:
            ValueError: If the model is not fitted yet.
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

        # Save best parameters to a JSON file.
        fn = self.optimize_dir / "parameters" / "best_params.json"
        if not fn.parent.exists():
            fn.parent.mkdir(parents=True, exist_ok=True)

        with open(fn, "w") as fp:
            json.dump(best_params, fp, indent=4)

        tn = f"{self.tune_metric} Value"
        self.plotter_.plot_tuning(study, self.model_name, target_name=tn)

    def train_and_validate_model(
        self,
        loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        l1_penalty: float,
        trial: optuna.Trial | None = None,
        return_history: bool = False,
    ) -> Tuple[float, torch.nn.Module, dict] | Tuple[float, torch.nn.Module]:
        """
        Train and validate the model, with conditional handling for UBP phases.
        """
        if model is None:
            msg = "Model is not fitted yet. Call `fit()` before training."
            self.logger.error(msg)
            raise TypeError(msg)

        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        best_loss = float("inf")
        early_stopping = EarlyStopping(
            patience=self.early_stop_gen,
            verbose=self.verbose,
            prefix=self.prefix,
            min_epochs=self.min_epochs,
            debug=self.debug,
        )

        if return_history:
            train_history = []
            val_history = []

        best_model = None
        phase = 1  # Start with Phase 1 for UBP

        for epoch in range(self.epochs):
            train_loss = self.train_step(loader, optimizer, model, l1_penalty, phase)
            validation_loss = self.val_step(val_loader, model, l1_penalty)

            if self.model_name != "ImputeUBP":
                scheduler.step()

            if return_history:
                train_history.append(train_loss)
                val_history.append(validation_loss)

            if trial is not None and self.tune:
                trial.report(validation_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            early_stopping(validation_loss, model)

            if early_stopping.early_stop:
                best_loss = early_stopping.best_score
                best_model = copy.deepcopy(early_stopping.best_model)
                if best_model is None:
                    raise RuntimeError("No valid model found during training.")
                break

            # Transition between UBP phases
            if (
                self.model_name == "ImputeUBP"
                and epoch + 1 == (self.epochs // 3) * phase
            ):
                phase = min(3, phase + 1)

        if return_history:
            histories = {"Train": train_history, "Validation": val_history}
            return best_loss, best_model, histories

        return best_loss, best_model

    def train_step(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        l1_penalty: float,
        phase: int = 1,
    ) -> float:
        """
        Train the model using the training set, with specific handling for UBP phases.
        """
        model.train()
        running_loss = 0.0
        num_batches = 0

        for X_batch, y_batch, mask_batch in loader:
            X_batch, y_batch, mask_batch = (
                X_batch.to(self.device),
                y_batch.to(self.device),
                mask_batch.to(self.device),
            )

            if self.model_name == "ImputeUBP":
                X_batch = X_batch.view(X_batch.size(0), -1)
                y_batch = y_batch.view(y_batch.size(0), -1)

                # Handle input refinement in Phases 1 and 3
                if phase in {1, 3}:
                    X_batch.requires_grad = True

            num_batches += 1
            optimizer.zero_grad()

            outputs = model(X_batch)

            output_shape = (X_batch.size(0), self.num_features_, self.num_classes_)
            if isinstance(outputs, tuple):
                recon_logits = outputs[0].view(output_shape)
                z_mean, z_log_var = outputs[1], outputs[2]
                logits = (recon_logits, z_mean, z_log_var)
            else:
                logits = outputs.view(output_shape)

            loss = model.compute_loss(
                y_batch, logits, mask=mask_batch, class_weights=self.class_weights_
            )

            if l1_penalty > 0:
                l1_reg = sum(param.abs().sum() for param in model.parameters())
                loss += l1_penalty * l1_reg

            # Backward pass for UBP phases
            if self.model_name == "ImputeUBP":
                if phase in {1, 3}:  # Refine inputs
                    loss.backward(inputs=[X_batch])
                if phase in {2, 3}:  # Refine weights
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                if phase == 1:
                    optimizer.step()
            else:  # Standard model training
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()

        return running_loss / num_batches

    def val_step(
        self,
        val_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        l1_penalty: float,
    ) -> Tuple[float, torch.nn.Module]:
        """
        Validate the model using the validation set. For UBP, inputs are fixed and not refined.

        Args:
            val_loader (DataLoader): Validation DataLoader.
            model (torch.nn.Module): Model to validate.
            l1_penalty (float): L1 regularization penalty to apply.

        Returns:
            float: Validation loss.
        """
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        val_batches = 0

        with torch.no_grad():  # Disable gradient tracking
            for X_batch, y_batch, mask_batch in val_loader:
                # Move data to device
                X_batch, y_batch, mask_batch = (
                    X_batch.to(self.device),
                    y_batch.to(self.device),
                    mask_batch.to(self.device),
                )

                if self.model_name == "ImputeUBP":
                    X_batch.requires_grad = False
                    X_batch = X_batch.view(X_batch.size(0), -1)
                    y_batch = y_batch.view(y_batch.size(0), -1)

                # Forward pass
                outputs = model(X_batch)

                output_shape = (X_batch.size(0), self.num_features_, self.num_classes_)
                if isinstance(outputs, tuple):
                    recon_logits = outputs[0].view(output_shape)
                    z_mean, z_log_var = outputs[1], outputs[2]
                    logits = (recon_logits, z_mean, z_log_var)
                else:
                    logits = outputs.view(output_shape)

                # Compute loss
                val_loss = model.compute_loss(
                    y_batch, logits, mask=mask_batch, class_weights=None
                )

                # Apply L1 penalty
                if l1_penalty > 0:
                    l1_reg = sum(param.abs().sum() for param in model.parameters())
                    val_loss += l1_penalty * l1_reg

                # Accumulate loss
                running_val_loss += val_loss.item()
                val_batches += 1

        # Compute average validation loss
        validation_loss = running_val_loss / val_batches
        return validation_loss

    def predict(
        self,
        Xenc: np.ndarray | torch.Tensor | pd.DataFrame | list,
        model: torch.nn.Module,
        return_proba: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Predict using the trained model.

        This method predicts the labels using the trained model. It returns the predicted labels and probabilities (if ``return_proba`` is True). The input data must be encoded. It decodes the predicted labels and probabilities back to the original integer encoding.

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

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(Xtensor.to(self.device))

        # If the model returns multiple outputs, assume first is recon logits
        recon_logits = outputs[0] if isinstance(outputs, tuple) else outputs
        y_pred_proba = torch.softmax(recon_logits, dim=-1)
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

        Overwrites only the original missing entries (where X had -9 or -1).

        Args:
            X (numpy.ndarray | pd.DataFrame | list | torch.Tensor): Data to impute.
            model (torch.nn.Module): Trained model.

        Returns:
            numpy.ndarray: Imputed data.

        Raises:
            AttributeError: If the original missing mask is not initialized.
            TypeError: If the model is not fitted yet.
            ValueError: If the input shape is invalid.
        """
        self.ensure_attribute("original_missing_mask_")

        if model is None:
            msg = "Model is not fitted yet. Call `fit()` before imputation."
            self.logger.error(msg)
            raise TypeError(msg)

        # Convert X to array, unify missing indicator.
        X = np.where(np.logical_or(X < 0, np.isnan(X)), -1, X)

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

        # Forward pass.
        model.eval()
        with torch.no_grad():
            outputs = model(Xtensor.to(self.device))

        recon_logits = outputs[0] if isinstance(outputs, tuple) else outputs
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
        self, X: np.ndarray | torch.Tensor | list, y: np.ndarray | torch.Tensor | list
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Create DataLoader objects for the input data."""
        self.logger.info("Creating DataLoader objects for inputs.")

        # Decode input data
        Xenc = validate_input_type(X)
        y_decoded = validate_input_type(y)

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            sim_mask_train,
            sim_mask_val,
            sim_mask_test,
        ) = self._split_data(
            Xenc, y_decoded, self.sim_missing_mask_, self.validation_split
        )

        # Transform and re-encode datasets
        train_data = self._process_data_subset(X_train, y_train)
        val_data = self._process_data_subset(X_val, y_val)
        test_data = self._process_data_subset(X_test, y_test)

        sim_mask_train = torch.tensor(sim_mask_train, dtype=torch.bool)
        sim_mask_val = torch.tensor(sim_mask_val, dtype=torch.bool)
        sim_mask_test = torch.tensor(sim_mask_test, dtype=torch.bool)

        # Create DataLoader objects
        self.loader_ = self._create_dataloader(
            *train_data[:2], mask=sim_mask_train, shuffle=True
        )
        self.val_loader_ = self._create_dataloader(
            *val_data[:2], mask=sim_mask_val, shuffle=False
        )
        self.test_loader_ = self._create_dataloader(
            *test_data[:2], mask=sim_mask_test, shuffle=False
        )

        self.logger.info("DataLoaders created successfully.")

    def _split_data(self, X, y, mask, validation_split: float):
        """
        Splits the dataset into train, validation, and test sets based on the validation_split.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target labels.
            mask (np.ndarray): Missing value mask.
            validation_split (float): Combined size of validation and test sets, e.g., 0.2 means 80% train, 10% validation, and 10% test.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - X_train: Training features.
            - X_val: Validation features.
            - X_test: Testing features.
            - y_train: Training labels.
            - y_val: Validation labels.
            - y_test: Testing labels.
            - mask_train: Training mask.
            - mask_val: Validation mask.
            - mask_test: Testing mask.
        """
        # Ensure the validation_split is valid
        if not (0 < validation_split < 1):
            msg = "Validation_split must be between 0 and 1."
            self.logger.error(msg)
            raise ValueError(msg)

        # First split: Train vs (Validation + Test)
        train_size = 1 - validation_split
        X_train, X_temp, y_train, y_temp, mask_train, mask_temp = train_test_split(
            X, y, mask, train_size=train_size, random_state=self.seed, shuffle=True
        )

        # Second split: Validation vs Test (equal sizes)
        val_test_split = 0.5  # Validation and Test are equally sized
        X_val, X_test, y_val, y_test, mask_val, mask_test = train_test_split(
            X_temp,
            y_temp,
            mask_temp,
            test_size=val_test_split,
            random_state=self.seed,
            shuffle=True,
        )

        self.logger.info(
            f"Data split into: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}"
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            mask_train,
            mask_val,
            mask_test,
        )

    def _process_data_subset(self, X, y):
        """Process a subset of the data."""
        X_final = np.where(np.logical_or(X < 0, np.isnan(X)), -1, X)
        y_final = np.where(np.logical_or(y < 0, np.isnan(y)), -1, y)
        observed_mask = np.logical_and(y_final >= 0, ~np.isnan(y_final))
        X_final = np.where(observed_mask[:, :, np.newaxis], X_final, -1)
        y_final = np.where(observed_mask, y_final, -1)
        return X_final, y_final, ~observed_mask

    def _create_dataloader(self, X, y, mask, shuffle: bool):
        """Helper method to create a PyTorch DataLoader."""
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

        Args:
            module (torch.nn.Module): Module to initialize.

        Raises:
            AttributeError: If the module is not an instance of torch.nn.Linear.
        """
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def build_model(
        self, Model: torch.nn.Module, model_params: Dict[str, Any]
    ) -> torch.nn.Module:
        """Build the neural network model.

        Args:
            Model (torch.nn.Module): Model class to instantiate.
            model_params (Dict[str, Any]): Model parameters.

        Returns:
            torch.nn.Module: Built model.

        Raises:
            ValueError: If model_params is empty.
        """
        if not model_params:
            msg = "'model_params' must not be empty."
            self.logger.error(msg)
            raise ValueError(msg)

        # Pass the weight_tensor into the model
        return Model(**model_params).to(self.device)

    def train_final_model(self) -> Tuple[float, torch.nn.Module, dict]:
        """Train the final model.

        This method trains the final model using the training and validation sets. It returns the loss, model, and training history.

        Returns:
            Tuple[float, torch.nn.Module, dict]: Loss, model, and training history.

        Raises:
            AttributeError: If the DataLoader objects do not exist.
            AttributeError: If model parameters are not provided.
            AttributeError: If learning rate is not provided.
            AttributeError: If L1 penalty is not provided.
        """
        which_model = "tuned model" if self.tune else "model"
        self.logger.info(f"Training the {which_model}...")

        attributes = ["loader_", "val_loader_", "Model", "model_params"]
        attributes += ["learning_rate", "l1_penalty", "lr_patience", "gamma"]
        attributes += ["beta"]
        [self.ensure_attribute(attr) for attr in attributes]

        self.lr_ = self.learning_rate
        self.lr_patience_ = self.lr_patience
        self.l1_penalty_ = self.l1_penalty

        # Build model
        model = self.build_model(self.Model, self.model_params)
        model.apply(self.initialize_weights)
        self.optimizer_ = optim.Adam(model.parameters(), lr=self.lr_)

        args = [self.loader_, self.val_loader_, self.optimizer_]
        args += [model, self.l1_penalty_]

        res = self.train_and_validate_model(*args, return_history=True)
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

        Args:
            attribute (str): Attribute to check.

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
    ):
        """Efficient evaluation of the model.

        This method evaluates the model using the test set. It computes the evaluation metrics, saves them to a JSON file, and plots the metrics and confusion matrix.

        Args:
            objective_mode (bool, optional): Whether to run in objective mode. Defaults to False.
            trial (optuna.Trial, optional): Optuna trial object. Defaults to None.
            model (torch.nn.Module): Model to evaluate. Required keyword argument.

        Returns:
            dict: Evaluation metrics. Only returned in objective mode.

        Raises:
            TypeError: If objective_mode is True but trial is not provided.
            AttributeError: If test_loader_ does not exist.
            TypeError: If model is not provided.
        """
        if objective_mode and trial is None:
            msg = "Trial object must be provided for objective mode."
            self.logger.error(msg)
            raise TypeError(msg)

        self.ensure_attribute("test_loader_")

        if model is None:
            msg = "Model must be provided for evaluation, but got None."
            self.logger.error(msg)
            raise TypeError(msg)

        # Efficient data preparation
        X = validate_input_type(self.test_loader_.dataset.data)
        y_true_labels = validate_input_type(self.test_loader_.dataset.target)
        mask_test = validate_input_type(self.test_loader_.dataset.mask)
        y_true_enc = self.tt_.transform(y_true_labels)

        # Predict and filter missing data
        pred_labels, pred_proba = self.predict(X, model, return_proba=True)

        # Focus only on the simulated missing values.
        valid_mask = mask_test

        y_true_labels, pred_labels, pred_proba, y_true_enc = (
            y_true_labels[valid_mask],
            pred_labels[valid_mask],
            pred_proba[valid_mask],
            y_true_enc[valid_mask],
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
        metrics_path = self.metrics_dir / f"test_metrics_{id_str}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as fp:
            json.dump(metrics, fp, indent=4)

        self.plotter_.plot_metrics(y_true_labels, pred_proba, metrics)
        self.plotter_.plot_confusion_matrix(y_true_labels, pred_labels)

        if not objective_mode:
            self.metrics_ = metrics

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

        This method computes the class weights based on the class distribution in the data. It calculates the raw inverse class weights, optionally applies a log-scale transform, and bounds the weights within a specified range.

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
        return weight_tensor.to(self.device)

    def init_transformers(
        self,
    ) -> Tuple[
        AutoEncoderFeatureTransformer, SimGenotypeDataTransformer, Plotting, Scorer
    ]:
        """Initialize the transformers for encoding.

        This method should be called in a `fit` method to initialize the transformers.

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
