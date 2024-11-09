import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import torch
import torch.optim as optim
from sklearn.base import BaseEstimator, TransformerMixin
from snpio.utils.logging import LoggerManager
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from pgsui.data_processing.transformers import SimGenotypeDataTransformer
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.dataset import CustomTensorDataset
from pgsui.utils.misc import validate_input_type


class BaseNNImputer(BaseEstimator, TransformerMixin):
    """Base class for PyTorch-based neural network imputers with Optuna hyperparameter optimization.

    This class serves as the base class for PyTorch-based neural network imputers with Optuna hyperparameter optimization. It provides methods for training and validating the model, predicting the missing values, and optimizing hyperparameters using Optuna. It also uses scikit-learn's ``BaseEstimator`` and ``TransformerMixin`` classes for compatibility with the ``scikit-learn`` API.

    Example:
        >>> class MyImputer(BaseNNImputer):
        >>>     def __init__(self, **kwargs):
        >>>        super().__init__(**kwargs)
        >>>     def fit(self, X, y=None):
        >>>         # Implement the fit method
        >>>         return self
        >>>     def transform(self, X, y=None):
        >>>         # Implement the transform method
        >>>         return X

    Attributes:
        genotype_data (Any): Genotype data object.
        num_classes (int): Number of classes.
        model_name (str): Name of the model.
        prefix (str): Prefix for the output directory.
        output_dir (str): Output directory for saving model output.
        latent_dim (int): Latent dimension of the VAE model.
        dropout_rate (float): Dropout rate for the model.
        num_hidden_layers (int): Number of hidden layers in the model.
        hidden_layer_sizes (List[int]): List of hidden layer sizes.
        hidden_activation (str): Activation function for hidden layers.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        early_stop_gen (int): Number of generations for early stopping.
        lr_patience (int): Patience for learning rate scheduler.
        epochs (int): Number of epochs for training.
        l1_penalty (float): L1 regularization penalty.
        l2_penalty (float): L2 regularization penalty.
        optimizer (str): Optimizer for training.
        scoring_averaging (str): Averaging method for scoring.
        sim_strategy (str): Strategy for simulating missing data.
        sim_prop_missing (float): Proportion of missing data to simulate.
        validation_split (float): Validation split for training.
        tune (bool): Whether to tune hyperparameters.
        n_trials (int): Number of trials for hyperparameter optimization.
        n_jobs (int): Number of parallel jobs for hyperparameter optimization.
        seed (int): Random seed for reproducibility.
        verbose (int): Verbosity level.
        debug (bool): Whether to enable debug mode.
        device (torch.device): Device (GPU or CPU) for training.
        logger (LoggerManager): Logger instance.
        sim_ (SimGenotypeDataTransformer): Genotype data transformer for simulating missing data.
        tt_ (AutoEncoderFeatureTransformer): Feature transformer for encoding features.
        model_ (nn.Module): Model instance.
        output_dir (Path): Output directory for saving model output.
        best_params_ (Dict[str, Any]): Best hyperparameters found during optimization.
        best_tuned_loss_ (float): Best loss after hyperparameter optimization.
        train_loader_ (DataLoader): DataLoader for training data.
        val_loader_ (DataLoader): DataLoader for validation data.
        optimizer_ (optim.Optimizer): Optimizer instance.
        history_ (Dict[str, List[float]]): Training and validation history.
    """

    def __init__(
        self,
        genotype_data: Any,
        num_classes: int,
        model_name: str,
        prefix: str = "pgsui",
        output_dir: Union[str, Path] = "output",
        latent_dim: int = 2,
        dropout_rate: float = 0.2,
        num_hidden_layers: int = 2,
        hidden_layer_sizes: List[int] = [128, 64],
        activation: str = "elu",
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stop_gen: int = 25,
        lr_patience: int = 1,
        epochs: int = 100,
        optimizer: str = "adam",
        l1_penalty: float = 0.0001,
        l2_penalty: float = 0.0001,
        scoring_averaging: str = "weighted",
        sim_strategy: str = "random",
        sim_prop_missing: float = 0.2,
        validation_split: float = 0.2,
        tune: bool = False,
        n_trials: int = 100,
        n_jobs: int = 1,
        seed: Optional[int] = None,
        verbose: int = 0,
        debug: bool = False,
    ) -> None:
        """Initialize the base neural network imputer.

        This method initializes the base neural network imputer with the specified hyperparameters and settings. It sets up the logger and the device (GPU or CPU) for training the model.

        Args:
            genotype_data (Any): Genotype data object.
            num_classes (int): Number of classes.
            model_name (str): Name of the model.
            prefix (str): Prefix for the output directory. Defaults to "pgsui".
            output_dir (Union[str, Path]): Output directory for saving model output. Defaults to "output".
            latent_dim (int): Latent dimension of the VAE model. Defaults to 2.
            dropout_rate (float): Dropout rate for the model. Defaults to 0.2.
            num_hidden_layers (int): Number of hidden layers in the model. Defaults to 2.
            hidden_layer_sizes (List[int]): List of hidden layer sizes. Defaults to [128, 64].
            activation (str): Activation function for hidden layers. Defaults to "elu".
            batch_size (int): Batch size for training. Defaults to 32.
            learning_rate (float): Learning rate for the optimizer. Defaults to 0.001.
            early_stop_gen (int): Number of generations for early stopping. Defaults to 25.
            lr_patience (int): Patience for learning rate scheduler. Defaults to 1.
            epochs (int): Number of epochs for training. Defaults to 100.
            optimizer (str): Optimizer for training. Defaults to "adam".
            l1_penalty (float): L1 regularization penalty. Defaults to 0.0001.
            l2_penalty (float): L2 regularization penalty. Defaults to 0.0001.
            scoring_averaging (str): Averaging method for scoring. Valid options are "micro", "macro", or "weighted". Defaults to "weighted".
            sim_strategy (str): Strategy for simulating missing data. Defaults to "random".
            sim_prop_missing (float): Proportion of missing data to simulate. Defaults to 0.2.
            validation_split (float): Validation split for training. Defaults to 0.2.
            tune (bool): Whether to tune hyperparameters. Defaults to False.
            n_trials (int): Number of trials for hyperparameter optimization. Defaults to 100.
            n_jobs (int): Number of parallel jobs for hyperparameter optimization. Defaults to 1.
            seed (int, optional): Random seed for reproducibility. If seed is None, the random number generator is not seeded and the split will be random and non-deterministic. Defaults to None.
            verbose (int): Verbosity level. Defaults to 0.
            debug (bool): Whether to enable debug mode. Defaults to False.

        """
        self.genotype_data = genotype_data
        self.num_classes = num_classes
        self.model_name = model_name
        self.prefix = prefix
        self.output_dir = output_dir
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_activation = activation
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_gen = early_stop_gen
        self.lr_patience = lr_patience
        self.epochs = epochs
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.optimizer = optimizer
        self.scoring_averaging = scoring_averaging
        self.sim_strategy = sim_strategy
        self.sim_prop_missing = sim_prop_missing
        self.validation_split = validation_split
        self.tune = tune
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose
        self.debug = debug

        has_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if has_cuda else "cpu")

        logman = LoggerManager(
            name=__name__,
            prefix=self.prefix,
            level=logging.DEBUG if debug else logging.INFO,
            verbose=verbose >= 1,
            debug=debug,
        )

        self.logger = logman.get_logger()
        self.model_ = None

        outdirs = ["models", "logs", "plots", "metrics", "optimize"]
        self._create_model_directories(prefix, output_dir, outdirs)

    def _create_model_directories(
        self, prefix: str, output_dir: Union[str, Path], outdirs: List[str]
    ) -> None:
        """Create the output directory for saving model output.

        This method creates the output directory for saving model output. It creates a subdirectory for the model type and the model name.

        Args:
            prefix (str): Prefix for the output directory.
            output_dir (Union[str, Path]): Output directory for saving model output.
            outdirs (List[str]): List of subdirectories to create inside the output directory.
        """

        self.output_dir = Path(f"{prefix}_{output_dir}")

        for d in outdirs:
            subdir = self.output_dir / d / "Unsupervised" / self.model_name
            subdir.mkdir(parents=True, exist_ok=True)

    def objective(self, trial: optuna.Trial, Model: Any) -> float:
        """Objective function for Optuna optimization.

        This method defines the objective function for Optuna optimization. It sets up the model with the hyperparameters suggested by Optuna and trains the model using the training and validation data loaders. It returns the best validation loss found during training.

        Args:
            trial (optuna.Trial): Optuna trial object.
            Model (Any): Model class to use for training.

        Returns:
            float: The best validation loss found during training.
        """
        # Define hyperparameters using Optuna's trial object
        latent_dim = trial.suggest_int("latent_dim", 2, 10)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)

        num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 10)

        hidden_layer_sizes = [
            trial.suggest_int(f"n_units_l{i}", 16, 128)
            for i in range(num_hidden_layers)
        ]
        activation = trial.suggest_categorical(
            "activation", ["relu", "elu", "selu", "leaky_relu"]
        )

        lr_patience = trial.suggest_int("lr_patience", 1, 10)
        l1_penalty = trial.suggest_float("l1_penalty", 1e-5, 1e-2)
        l2_penalty = trial.suggest_float("l2_penalty", 1e-5, 1e-2)

        model_params = {
            "n_features": self.num_features,
            "num_classes": self.num_classes,
            "latent_dim": latent_dim,
            "dropout_rate": dropout_rate,
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
        }

        # Initialize the model with hyperparameters
        model = self.build_model(Model, model_params)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training and validation steps
        best_loss, _ = self.train_and_validate_model(
            self.train_loader_,
            self.val_loader_,
            optimizer,
            model=model,
            lr_patience=lr_patience,
            l1_penalty=l1_penalty,
            l2_penalty=l2_penalty,
        )
        return best_loss

    def train_and_validate_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        model: Optional[nn.Module] = None,
        lr_patience: int = 1,
        l1_penalty: float = 0.0,
        l2_penalty: float = 0.0,
    ) -> Tuple[float, Dict[str, List[float]]]:
        """Train and validate the model.

        This method trains and validates the model using the training and validation data loaders. It computes the training and validation losses and returns the best validation loss and the training and validation history.

        Args:
            train_loader (DataLoader): DataLoader for training data. The DataLoader should return a tuple of input and target tensors.
            val_loader (DataLoader): DataLoader for validation data. The DataLoader should return a tuple of input and target tensors.
            optimizer (torch.optim.Optimizer): Optimizer for training. The optimizer should be initialized with the model parameters and learning rate.
            model (nn.Module): The model instance to train. If None, the model is set to the instance attribute `model_`. Defaults to None.
            lr_patience (int): Patience for learning rate scheduler. Defaults to 1.
            l1_penalty (float): L1 regularization penalty. Defaults to 0.0.
            l2_penalty (float): L2 regularization penalty. Defaults to 0.0.

        Returns:
            Tuple[float, Dict[str, List[float]]]: The best validation loss and the training and validation history.
        """
        if model is None:
            model = self.model_

        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=lr_patience, factor=0.5
        )

        best_loss = float("inf")
        early_stopping = EarlyStopping(
            patience=self.early_stop_gen,
            verbose=self.verbose,
            prefix=self.prefix,
            output_dir=self.output_dir,
            model_name=self.model_name,
        )

        train_history = []
        val_history = []

        for epoch in range(self.epochs):
            if self.verbose >= 2:
                self.logger.info(f"Epoch {epoch + 1} / {self.epochs}")

            # Training phase
            model.train()
            running_loss = 0.0
            total_samples = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                # Mask missing values in the input (train on known values only).
                mask = X_batch != -1
                X_known = X_batch * mask

                # Forward pass
                outputs = model(X_known)

                # Compute loss with regularization (L1 and L2).
                loss = model.compute_loss(y_batch * mask, outputs)

                # Compute L1 and L2 regularization.
                l2_reg = sum(torch.norm(param, 2) ** 2 for param in model.parameters())
                l1_reg = sum(torch.norm(param, 1) for param in model.parameters())

                # Add regularization to the loss.
                loss += l1_penalty * l1_reg + l2_penalty * l2_reg

                known_values = mask.sum().item()
                running_loss += loss.item() * known_values
                total_samples += known_values

                # Backpropagation
                loss.backward()

                # Gradient clipping to prevent exploding gradients.
                # NOTE: Call after loss.backward() but before optimizer.step().
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                # Optimizer step
                optimizer.step()

            train_loss = running_loss / total_samples
            train_history.append(train_loss)

            # Validation phase (no regularization or gradient updates)
            val_loss = self.validate_model(val_loader, model=model)
            val_history.append(val_loss)

            # Scheduler step based on validation loss
            scheduler.step(val_loss)

            if self.verbose >= 2:
                self.logger.info(f"Final validation loss: {val_loss}")

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                best_loss = val_loss

                if self.verbose >= 2:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")

                break

        return best_loss, {"Train": train_history, "Val": val_history}

    def validate_model(
        self, val_loader: DataLoader, model: Optional[nn.Module] = None
    ) -> float:
        """Validate the model on the simulated missing values.

        This method validates the model on the simulated missing values using the validation data loader. It computes the validation loss and returns the average loss across all validation samples.

        Args:
            val_loader (DataLoader): Validation data loader.
            model (nn.Module): The model instance to validate. If None, the model is set to the instance attribute `model_`.

        Returns:
            float: The validation loss.
        """
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:

                mask = X_batch == -1
                X_known = X_batch * mask

                # Forward pass
                outputs = model(X_known)

                # Compute loss only on the simulated missing values
                loss = model.compute_loss(y_batch * mask, outputs)

                known_values = mask.sum().item()

                # Accumulate loss
                val_loss += loss.item() * known_values
                total_samples += known_values

        # Average loss across all validation samples
        return val_loss / total_samples

    def predict(
        self, X: Union[np.ndarray, torch.Tensor], return_proba: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Predict the missing values using the model.

        This method predicts the missing values in the input data using the trained model. It returns the input data with the imputed values. The input data should be in the same format as the training data.

        Args:
            X (Union[torch.Tensor, numpy.ndarray]): The input tensor containing missing values. The input data should be in the same format as the training data.

            return_proba (bool): Whether to return the per-class prediction probabilities in addition to the imputed values. Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: If `return_proba` is False, the method returns an array with imputed values, where only the original missing values have been replaced. The imputed values are in the same format as the input data. If `return_proba` is True, the method returns a tuple of the imputed values and the per-class prediction probabilities.

        Raises:
            Exception: If an error occurs during conversion of the input data to a PyTorch tensor.
            ValueError: If missing values remain in the input data after prediction (should not be any).
        """
        X = validate_input_type(X, return_type="array")
        X[X == -9] = -1  # Replace -9 with -1 for missing values

        Xt = X if X.ndim == 3 else self.tt_.fit_transform(X)

        # Ensure that Xt is converted to a PyTorch tensor.
        Xt_tensor = torch.tensor(Xt, dtype=torch.float32).to(self.device)

        self.model_.eval()  # Set the model to evaluation mode

        # Predict the original missing values using the VAE model
        with torch.no_grad():
            outputs = self.model_(Xt_tensor)

        # VAE model returns a tuple of outputs and KL divergence; we use the first element (reconstructed data)
        reconstruction = outputs[0] if isinstance(outputs, tuple) else outputs

        # Inverse transform the reconstruction to match the original format
        reconstruction_arr = self.tt_.inverse_transform(reconstruction)

        if return_proba:
            reconstruction_proba = self.tt_.inverse_transform(
                reconstruction, return_proba=True
            )

        # Get the imputed values from the reconstruction.
        # If the number of classes is 3, we use the argmax of the output.
        # If the number of classes is 4, we use the one-hot encoding.
        # If the number of classes is greater than 4, use argmax of output.
        if Xt.shape[-1] == 3:
            X_imp = np.argmax(Xt, axis=-1)
        elif Xt.shape[-1] == 4:  # One-hot encoding
            X_imp = np.eye(4)[np.argmax(Xt, axis=-1)]
        elif Xt.shape[-1] > 4:  # General integer encoding for >4 classes
            X_imp = np.argmax(Xt, axis=-1)
        else:
            msg = "Invalid number of classes for imputation."
            self.logger.error(msg)
            raise ValueError(msg)

        if len(X_imp) != len(self.original_missing_mask_):
            missing_mask = self.original_missing_mask_[self.test_dataset_.indices]
        else:
            missing_mask = self.original_missing_mask_

        # Get the original missing values' mask and ensure it's a boolean mask
        # Impute only the original missing values into the Xt dataset
        X_imp[missing_mask] = reconstruction_arr[missing_mask]

        # Check if any missing values remain in Xt after imputation (should not
        # be any)
        if np.isnan(X_imp).any():
            msg = "Imputation failed: Missing values remain after prediction."
            self.logger.error(msg)
            raise ValueError(msg)

        # Debug logs for reconstruction and imputation
        self.logger.debug(f"Reconstruction array: {reconstruction_arr}")
        self.logger.debug(f"Reconstruction shape: {reconstruction_arr.shape}")
        self.logger.debug(f"Unique Imputed values: {np.unique(X_imp[missing_mask])}")
        self.logger.debug(f"Imputed values shape: {X_imp[missing_mask].shape}")

        if return_proba:
            return X_imp, reconstruction_proba

        return X_imp

    def get_data_loaders(
        self, X: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare DataLoader objects for training, validation, and hold-out test sets.

        This method applies the data transformation pipeline using `SimGenotypeDataTransformer` to simulate missing data and the `AutoEncoderFeatureTransformer` to encode the features.

        It splits the transformed data into training, validation, and test sets and returns DataLoader objects for each.

        Args:
            X (numpy.ndarray): The input data with missing values to predict. Should be one-hot encoded.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: DataLoader objects for the training, validation, and test sets. The DataLoader objects return a tuple of input and target tensors.
        """
        X = validate_input_type(X, return_type="array")

        self.logger.debug(f"get_data_loaders input X: {X}")

        # Split the dataset into train, validation, and test sets first
        Xtensor = torch.tensor(X, dtype=torch.float32)
        dataset = CustomTensorDataset(Xtensor, Xtensor, logger=self.logger)

        train_split = 1 - self.validation_split
        val_split = self.validation_split

        train_dataset, val_dataset, test_dataset = dataset.split_dataset(
            train_ratio=train_split, val_ratio=val_split, seed=self.seed
        )

        # Set the dataset attributes.
        self.dataset_ = dataset
        self.train_dataset_ = train_dataset
        self.val_dataset_ = val_dataset
        self.test_dataset_ = test_dataset

        # Set batch size for DataLoader objects.
        bs = self.batch_size
        bs = len(val_dataset) if bs > len(val_dataset) else bs

        # Step 5: Create DataLoader objects for training, validation, test sets.
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

        return train_loader, val_loader, test_loader

    def build_model(self, Model: Callable, model_params: Dict[str, Any]) -> nn.Module:
        """Build the model.

        This method instantiates the model class with the specified hyperparameters and moves the model to the device (GPU or CPU).

        Args:
            Model (callable): The model class callable.
            model_params (Dict[str, Any]): Dictionary of model hyperparameters.

        Returns:
            nn.Module: The VAE model instance built with the specified hyperparameters and moved to the device.

        Raises:
            ValueError: If the model_params dictionary is empty.
        """
        if not model_params:
            msg = "'model_params' must be provided to build the model, but found an empty dictionary."
            self.logger.error(msg)
            raise ValueError(msg)

        return Model(**model_params).to(self.device)
