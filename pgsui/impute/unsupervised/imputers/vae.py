from pathlib import Path
from typing import Any, List, Literal

import numpy as np
import optuna
import pandas as pd
from snpio.utils.logging import LoggerManager
from torch import Tensor

from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.models.vae_model import VAEModel
from pgsui.utils.misc import validate_input_type


class ImputeVAE(BaseNNImputer):
    """ImputeVAE class for imputing missing values in genotype data using a Variational Autoencoder (VAE) model.

    This class is used to impute missing values in genotype data using a Variational Autoencoder (VAE) model. The class can be used to impute missing values in genotype data and evaluate the performance of the model using various metrics. The class can also be used to tune hyperparameters using Optuna. The class can be used to impute missing values in genotype data and evaluate the performance of the model using various metrics. The class can also be used to tune hyperparameters using Optuna.
    """

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
        sim_prop_missing: float = 0.1,
        sim_strategy: str = "random_inv_multinom",
        tune: bool = False,
        tune_metric: str = "pr_macro",
        tune_save_db: bool = False,
        tune_resume: bool = False,
        tune_n_trials: int = 100,
        model_latent_dim: int = 2,
        model_dropout_rate: float = 0.2,
        model_num_hidden_layers: int = 2,
        model_hidden_layer_sizes: List[int] = [128, 64],
        model_batch_size: int = 32,
        model_learning_rate: float = 0.001,
        model_early_stop_gen: int = 25,
        model_min_epochs: int = 100,
        model_optimizer: str = "adam",
        model_hidden_activation: str = "elu",
        model_lr_patience: int = 10,
        model_epochs: int = 5000,
        model_validation_split: float = 0.2,
        model_l1_penalty: float = 0.0001,
        model_gamma: float = 2.0,
        model_beta: float = 1.0,
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
        """Initialize the ImputeVAE class.

        This class is used to impute missing values in genotype data using a Variational Autoencoder (VAE) model. The class can be used to impute missing values in genotype data and evaluate the performance of the model using various metrics. The class can also be used to tune hyperparameters using Optuna. The class can be used to impute missing values in genotype data and evaluate the performance of the model using various metrics. The class can also be used to tune hyperparameters using Optuna.

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
            sim_prop_missing (float, optional): Proportion of missing values to simulate. Defaults to 0.1.
            sim_strategy (str, optional): Strategy to simulate missing values. Defaults to "random_inv_multinom".
            tune (bool, optional): Whether to tune hyperparameters. Defaults to False.
            tune_metric (str, optional): Metric to use for hyperparameter tuning. Defaults to "f1".
            tune_save_db (bool, optional): Whether to save the hyperparameter tuning database. Defaults to False.
            tune_resume (bool, optional): Whether to resume hyperparameter tuning. Defaults to False.
            tune_n_trials (int, optional): Number of hyperparameter tuning trials. Defaults to 100.
            model_latent_dim (int, optional): Latent dimension of the VAE. Defaults to 2.
            model_dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
            model_num_hidden_layers (int, optional): Number of hidden layers. Defaults to 2.
            model_hidden_layer_sizes (List[int], optional): Sizes of hidden layers. Defaults to [128, 64].
            model_batch_size (int, optional): Batch size. Defaults to 32.
            model_learning_rate (float, optional): Learning rate. Defaults to 0.001.
            model_early_stop_gen (int, optional): Number of generations to early stop. Defaults to 25.
            model_min_epochs (int, optional): Minimum number of generations to train. Defaults to 100.
            model_optimizer (str, optional): Optimizer to use. Defaults to "adam".
            model_hidden_activation (str, optional): Activation function for hidden layers. Defaults to "elu".
            model_lr_patience (int, optional): Patience for learning rate scheduler. Defaults to 10.
            model_epochs (int, optional): Number of epochs. Defaults to 5000.
            model_validation_split (float, optional): Validation split. Defaults to 0.2.
            model_l1_penalty (float, optional): L1 penalty. Defaults to 0.0001.
            model_gamma (float, optional): Gamma parameter. Defaults to 2.0.
            model_beta (float, optional): Beta parameter. Defaults to 1.0.
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
        self.model_name = "ImputeVAE"
        self.is_backprop = self.model_name in {"ImputeUBP", "ImputeNLPCA"}

        kwargs = {"prefix": prefix, "debug": debug, "verbose": verbose >= 1}
        logman = LoggerManager(__name__, **kwargs)
        self.logger = logman.get_logger()

        super().__init__(
            prefix=prefix,
            output_dir=output_dir,
            device=model_device,
            verbose=verbose,
            debug=debug,
        )
        self.Model = VAEModel

        self.genotype_data = genotype_data
        self.latent_dim = model_latent_dim
        self.dropout_rate = model_dropout_rate
        self.num_hidden_layers = model_num_hidden_layers
        self.hidden_layer_sizes = model_hidden_layer_sizes
        self.activation = model_hidden_activation
        self.hidden_activation = model_hidden_activation
        self.batch_size = model_batch_size
        self.learning_rate = model_learning_rate
        self.sim_prop_missing = sim_prop_missing
        self.sim_strategy = sim_strategy
        self.tune = tune
        self.tune_metric = tune_metric
        self.tune_resume = tune_resume
        self.tune_save_db = tune_save_db
        self.n_trials = tune_n_trials
        self.early_stop_gen = model_early_stop_gen
        self.min_epochs = model_min_epochs
        self.optimizer = model_optimizer
        self.lr_patience = model_lr_patience
        self.epochs = model_epochs
        self.l1_penalty = model_l1_penalty
        self.gamma = model_gamma
        self.beta = model_beta
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
        self.seed = seed

        _ = self.genotype_data.snp_data  # Ensure SNP data is loaded

        self.model_params = {
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "gamma": self.gamma,
            "beta": self.beta,
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

        self.tt_, self.sim_, self.plotter_, self.scorers_ = self.init_transformers()

        Xsim, missing_masks = self.sim_.fit_transform(X)
        self.original_missing_mask_ = missing_masks["original"]
        self.sim_missing_mask_ = missing_masks["simulated"]
        self.all_missing_mask = missing_masks["all"]

        # Encode the data.
        Xsim_enc = self.tt_.fit_transform(Xsim)

        self.num_features_ = Xsim_enc.shape[1]
        self.model_params["n_features"] = self.num_features_

        self.loader_ = self.get_data_loaders(Xsim_enc, X, self.latent_dim)

        if self.tune:
            self.tune_hyperparameters()

        res = self.train_final_model(self.loader_)
        self.best_loss_, self.model_, self.history_ = res

        self.metrics_ = self.evaluate_model(
            objective_mode=False, trial=None, model=self.model_, loader=self.loader_
        )

        self.plotter_.plot_history(self.history_)

        return self

    def transform(self, X: np.ndarray | pd.DataFrame | list | Tensor) -> np.ndarray:
        """Transform and impute the data using the trained VAE.

        This method transforms the input data and imputes the missing values using the trained VAE model.

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
        latent_dim = trial.suggest_int("latent_dim", 2, 4)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.05)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 10)
        hidden_layer_sizes = [
            int(x) for x in np.linspace(16, 256, num_hidden_layers)[::-1]
        ]
        gamma = trial.suggest_float("gamma", 0.025, 5.0, step=0.025)
        beta = trial.suggest_float("beta", 0.025, 5.0, step=0.025)
        activation = trial.suggest_categorical(
            "activation", ["relu", "elu", "selu", "leaky_relu"]
        )

        # Model parameters
        model_params = {
            "n_features": self.num_features_,
            "num_classes": self.num_classes_,
            "latent_dim": latent_dim,
            "dropout_rate": dropout_rate,
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
            "gamma": gamma,
            "beta": beta,
        }

        # Build and initialize model
        model = self.build_model(Model, model_params)
        model.apply(self.initialize_weights)

        try:
            # Train and validate the model
            _, model = self.train_and_validate_model(
                model,
                self.loader_,
                l1_penalty=self.l1_penalty,
                lr=learning_rate,
                trial=trial,
            )

            if model is None:
                self.logger.warning(
                    f"Trial {trial.number} pruned due to failed model training. Model was NoneType."
                )
                raise optuna.exceptions.TrialPruned()

            # Efficient evaluation
            metrics = self.evaluate_model(
                objective_mode=True, trial=trial, model=model, loader=self.loader_
            )

            if self.tune_metric not in metrics:
                msg = f"Invalid tuning metric: {self.tune_metric}"
                self.logger.error(msg)
                raise KeyError(msg)

            return metrics[self.tune_metric]

        except Exception as e:
            self.logger.warning(f"Trial {trial.number} pruned due to exception: {e}")
            raise optuna.exceptions.TrialPruned()

    def set_best_params(self, best_params: dict) -> dict:
        """Set the best hyperparameters.

        Args:
            best_params (dict): Dictionary of best hyperparameters.

        Returns:
            dict: Dictionary of best hyperparameters.
        """
        # Load best hyperparameters
        self.latent_dim = best_params["latent_dim"]
        self.hidden_layer_sizes = np.linspace(
            16, 256, best_params["num_hidden_layers"]
        ).astype(int)[::-1]
        self.dropout_rate = best_params["dropout_rate"]
        self.lr_ = best_params["learning_rate"]
        self.gamma = best_params["gamma"]
        self.beta = best_params["beta"]

        best_params_ = {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "gamma": self.gamma,
            "beta": self.beta,
        }

        return best_params_
