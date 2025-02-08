from pathlib import Path
from typing import Any, Literal

import numpy as np
import optuna
import pandas as pd
from snpio.utils.logging import LoggerManager
from torch import Tensor, optim

from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.models.in_development.lstm_model import LSTMModel
from pgsui.utils.misc import validate_input_type


class ImputeLSTM(BaseNNImputer):

    def __init__(
        self,
        genotype_data: Any,
        *,
        n_jobs: int = 1,
        seed: int | None = None,
        prefix: str = "pgsui",
        output_dir: str = "output",
        verbose: int = 0,
        weights: bool = True,
        weights_log_scale: bool = False,
        weights_alpha: float = 2.0,
        weights_normalize: bool = False,
        weights_temperature: float = 1.0,
        tune: bool = False,
        tune_metric: str = "pr_macro",
        tune_save_db: bool = False,
        tune_resume: bool = False,
        tune_n_trials: int = 100,
        model_lstm_hidden_size: int = 128,
        model_num_lstm_layers: int = 2,
        model_dropout_rate: float = 0.2,
        model_bidirectional: bool = False,
        model_batch_size: int = 32,
        model_learning_rate: float = 0.001,
        model_early_stop_gen: int = 25,
        model_min_epochs: int = 100,
        model_optimizer: str = "adam",
        model_hidden_activation: str = "elu",
        model_lr_patience: int = 10,
        model_epochs: int = 5000,
        model_validation_split: float = 0.2,
        model_l1_penalty: float = 0.0,
        model_gamma: float = 2.0,
        model_device: Literal["gpu", "cpu"] = "gpu",
        scoring_averaging: str = "weighted",
        plot_format: str = "pdf",
        plot_fontsize: int | float = 18,
        plot_dpi: int = 300,
        plot_title_fontsize: int = 20,
        plot_despine: bool = True,
        plot_show_plots: bool = False,
        debug: bool = False,
    ):
        """Initialize the ImputeLSTM class.

        This class is used to impute missing values in genotype data using a Long-term Short-term Time-series Network (LSTM). The LSTM is trained on the genotype data and used to impute the missing values. The class inherits from the BaseNNImputer class.

        Args:
            genotype_data (Any): Genotype data.
            n_jobs (int, optional): Number of jobs. Defaults to 1.
            seed (int | None, optional): Random seed. Defaults to None.
            prefix (str, optional): Prefix for logging. Defaults to "pgsui".
            output_dir (str, optional): Output directory. Defaults to "output".
            verbose (int, optional): Verbosity level. Defaults to 0.
            use_weights (bool, optional): Whether to use class weights. Defaults to True.
            weights_log_scale (bool, optional): Whether to use log scale for class weights. Defaults to False.
            weights_alpha (float, optional): Alpha parameter for class weights. Defaults to 2.0.
            weights_normalize (bool, optional): Whether to normalize class weights. Defaults to False.
            weights_temperature (float, optional): Temperature parameter for class weights. Defaults to 1.0.
            tune (bool, optional): Whether to tune hyperparameters. Defaults to False.
            tune_metric (str, optional): Metric to use for hyperparameter tuning. Defaults to "f1".
            tune_save_db (bool, optional): Whether to save the hyperparameter tuning database. Defaults to False.
            tune_resume (bool, optional): Whether to resume hyperparameter tuning. Defaults to False.
            tune_n_trials (int, optional): Number of hyperparameter tuning trials. Defaults to 100.
            model_lstm_hidden_size (int, optional): Hidden size of the LSTM layer. Defaults to 128.
            model_num_lstm_layers (int, optional): Number of stacked LSTM layers. Defaults to 2.
            model_dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
            model_bidirectional (bool, optional): Whether to use bidirectional LSTM. Defaults to False.
            model_batch_size (int, optional): Batch size. Defaults to 32.
            model_learning_rate (float, optional): Learning rate. Defaults to 0.
            model_early_stop_gen (int, optional): Number of generations to early stop. Defaults to 25.
            model_min_epochs (int, optional): Minimum number of generations to train. Defaults to 100.
            model_optimizer (str, optional): Optimizer to use. Defaults to "adam".
            model_hidden_activation (str, optional): Activation function for hidden layers. Defaults to "elu".
            model_lr_patience (int, optional): Patience for learning rate scheduler. Defaults to 10.
            model_epochs (int, optional): Number of epochs. Defaults to 5000.
            model_validation_split (float, optional): Validation split. Defaults to 0.2.
            model_l1_penalty (float, optional): L1 penalty. Defaults to 0.0001.
            model_gamma (float, optional): Gamma parameter. Defaults to 2.0.
            model_device (Literal["gpu", "cpu"], optional): Device to use. Will use GPU if available, otherwise defaults to CPU. Defaults to "gpu".
            scoring_averaging (str, optional): Averaging strategy for scoring. Defaults to "weighted".
            plot_format (str, optional): Plot format. Defaults to "pdf".
            plot_fontsize (int | float, optional): Plot font size. Defaults to 18.
            plot_dpi (int, optional): Plot DPI. Defaults to 300.
            plot_title_fontsize (int, optional): Plot title font size. Defaults to 20.
            plot_despine (bool, optional): Whether to despine plots. Defaults to True.
            plot_show_plots (bool, optional): Whether to show plots. Defaults to False.
            debug (bool, optional): Whether to enable debug mode. Defaults to False.
        """
        self.model_name = "ImputeLSTM"

        kwargs = {"prefix": prefix, "debug": debug, "verbose": verbose >= 1}
        logman = LoggerManager(__name__, **kwargs)
        self.logger = logman.get_logger()

        super().__init__(
            genotype_data,
            model_name=self.model_name,
            lstm_hidden_size=model_lstm_hidden_size,
            num_lstm_layers=model_num_lstm_layers,
            dropout_rate=model_dropout_rate,
            bidirectional=model_bidirectional,
            activation=model_hidden_activation,
            batch_size=model_batch_size,
            learning_rate=model_learning_rate,
            tune=tune,
            tune_metric=tune_metric,
            tune_save_db=tune_save_db,
            tune_resume=tune_resume,
            n_trials=tune_n_trials,
            early_stop_gen=model_early_stop_gen,
            min_epochs=model_min_epochs,
            optimizer=model_optimizer,
            lr_patience=model_lr_patience,
            epochs=model_epochs,
            l1_penalty=model_l1_penalty,
            gamma=model_gamma,
            device=model_device,
            scoring_averaging=scoring_averaging,
            n_jobs=n_jobs,
            seed=seed,
            validation_split=model_validation_split,
            prefix=prefix,
            output_dir=output_dir,
            verbose=verbose,
            debug=debug,
        )
        self.Model = LSTMModel

        self.genotype_data = genotype_data
        self.num_lstm_layers = model_num_lstm_layers
        self.lstm_hidden_size = model_lstm_hidden_size
        self.dropout_rate = model_dropout_rate
        self.bidirectional = model_bidirectional
        self.activation = model_hidden_activation
        self.hidden_activation = model_hidden_activation
        self.batch_size = model_batch_size
        self.learning_rate = model_learning_rate
        self.tune = tune
        self.tune_metric = tune_metric
        self.tune_resume = tune_resume
        self.n_trials = tune_n_trials
        self.early_stop_gen = model_early_stop_gen
        self.min_epochs = model_min_epochs
        self.optimizer = model_optimizer
        self.lr_patience = model_lr_patience
        self.epochs = model_epochs
        self.l1_penalty = model_l1_penalty
        self.gamma = model_gamma
        self.scoring_averaging = scoring_averaging
        self.n_jobs = n_jobs
        self.validation_split = model_validation_split
        self.prefix = prefix
        self.output_dir = output_dir
        self.plot_format = plot_format
        self.plot_fontsize = plot_fontsize
        self.plot_dpi = plot_dpi
        self.title_fontsize = plot_title_fontsize
        self.despine = plot_despine
        self.show_plots = plot_show_plots
        self.verbose = verbose
        self.weights = weights
        self.weights_log_scale = weights_log_scale
        self.weights_alpha = weights_alpha
        self.weights_normalize = weights_normalize
        self.weights_temperature = weights_temperature

        _ = self.genotype_data.snp_data  # Ensure SNP data is loaded

        self.model_params = {
            "num_lstm_layers": self.num_lstm_layers,
            "lstm_hidden_size": self.lstm_hidden_size,
            "dropout_rate": self.dropout_rate,
            "bidirectional": self.bidirectional,
            "activation": self.activation,
            "gamma": self.gamma,
        }

        # Convert output_dir to Path if not already
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

    def fit(self, X: np.ndarray | pd.DataFrame | list | Tensor, y: None = None):
        """Fit the model using the input data.

        Args:
            X (numpy.ndarray): Input data to fit the model.
            y (None): Ignored. Only for compatibility with the scikit-learn API.

        Returns:
            self: Returns an instance of the class.
        """
        self.logger.info(f"Fitting the {self.model_name} model...")

        # Activation for final layer
        self.activate_ = "softmax"

        # Validate input and unify missing indicators
        # Ensure NaNs are replaced by -1
        X = validate_input_type(X)
        mask = np.logical_or(X < 0, np.isnan(X))
        self.original_missing_mask_ = mask
        X = X.astype(float)

        # Count number of classes for activation.
        # If 4 classes, use sigmoid, else use softmax.
        # Ignore missing values (-9) in counting of classes.
        # 1. Compute the number of distinct classes
        self.num_classes_ = len(np.unique(X[X >= 0 & ~mask]))
        self.model_params["num_classes"] = self.num_classes_

        # 2. Compute class weights
        self.class_weights_ = self.compute_class_weights(
            X,
            use_log_scale=self.weights_log_scale,
            alpha=self.weights_alpha,
            normalize=self.weights_normalize,
            temperature=self.weights_temperature,
            max_weight=20.0,
            min_weight=0.01,
        )

        # For final dictionary of hyperparameters
        self.best_params_ = self.model_params

        self.tt_, self.plotter_, self.scorers_ = self.init_transformers()

        # Encode the data.
        Xenc = self.tt_.fit_transform(X)

        self.num_features_ = Xenc.shape[1]
        self.model_params["n_features"] = self.num_features_

        train_mask, val_mask, test_mask = self.get_data_loaders(Xenc, X)
        self.orig_mask_train_ = train_mask
        self.orig_mask_val_ = val_mask
        self.orig_mask_test_ = test_mask

        if self.tune:
            self.tune_hyperparameters()

        self.best_loss_, self.model_, self.history_ = self.train_final_model()
        self.evaluate_model(objective_mode=False, trial=None, model=self.model_)
        self.plotter_.plot_history(self.history_)

        return self

    def transform(self, X: np.ndarray | pd.DataFrame | list | Tensor) -> np.ndarray:
        """Transform and impute the data using the trained model.

        This method transforms the input data and imputes the missing values using the trained model.

        Args:
            X (numpy.ndarray): Data to transform and impute.

        Returns:
            numpy.ndarray: Transformed and imputed data.
        """
        Xenc = self.tt_.transform(validate_input_type(X))
        X_imputed = self.impute(Xenc, self.model_)

        self.plotter_.plot_gt_distribution(X, is_imputed=False)
        self.plotter_.plot_gt_distribution(X_imputed, is_imputed=True)

        return X_imputed

    def objective(self, trial, Model) -> float:
        """Optimized Objective function for Optuna.

        Args:
            trial (optuna.Trial): Optuna trial object.
            Model (torch.nn.Module): Model class to instantiate.

        Returns:
            float: Best metric value.
        """
        # Efficient hyperparameter sampling
        num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 3)
        lstm_hidden_size = trial.suggest_int("lstm_hidden_size", 64, 256, step=32)
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.05)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.025, 5.0, step=0.025)
        activation = trial.suggest_categorical(
            "activation", ["relu", "elu", "selu", "leaky_relu"]
        )

        # Model parameters
        model_params = {
            "n_features": self.num_features_,
            "num_lstm_layers": num_lstm_layers,
            "lstm_hidden_size": lstm_hidden_size,
            "dropout_rate": dropout_rate,
            "bidirectional": bidirectional,
            "activation": activation,
            "gamma": gamma,
        }

        # Build and initialize model
        model = self.build_model(Model, model_params)
        model.apply(self.initialize_weights)

        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        try:
            # Train and validate the model
            _, model = self.train_and_validate_model(
                self.loader_,
                self.val_loader_,
                optimizer,
                model=model,
                l1_penalty=self.l1_penalty,
                trial=trial,
            )

            if model is None:
                msg = f"Trial {trial.number} pruned due to failed model training. Model was NoneType."
                self.logger.warning(msg)
                raise optuna.exceptions.TrialPruned()

            # Efficient evaluation
            metrics = self.evaluate_model(objective_mode=True, trial=trial, model=model)

            if self.tune_metric not in metrics:
                f"Invalid tuning metric: {self.tune_metric}"
                self.logger.error(msg)
                raise KeyError(msg)

            return metrics[self.tune_metric]

        except Exception as e:
            msg = f"Trial {trial.number} pruned due to exception: {e}"
            self.logger.warning(msg)
            raise optuna.exceptions.TrialPruned()
