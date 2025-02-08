import copy
import warnings
from pathlib import Path
from typing import Any, List, Literal

import numpy as np
import optuna
import pandas as pd
import torch
from snpio.utils.logging import LoggerManager
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR

from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.models.ubp_model import UBPModel
from pgsui.utils.misc import validate_input_type
from pgsui.impute.unsupervised.callbacks import EarlyStopping


class ImputeUBP(BaseNNImputer):
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
        """Impute missing genotypes using Unsupervised Backpropagation (UBP).

        This class is used to impute missing values in genotype data using Unsupervised Backpropagation (UBP). The model is trained on the genotype data and used to impute the missing values. The class inherits from BaseNNImputer. The model is trained in three phases: Phase 1 uses a linear decoder that refines both inputs and weights, intended to get the refined inputs to a good starting point for Phase 2. Phases 2 and 3 use a flexible deeper network, but Phase 2 only refines weights, whereas Phase 3 refines the inputs and weights. The model uses a MaskedFocalLoss for training to handle class imbalance. The model is trained using PyTorch. The class uses the UBPModel class for the model architecture, the Scorer class for evaluating the performance of the model, and the SNPio LoggerManager class for logging messages.

        The parameters prefixed with `model_` are used to set the hyperparameters for the model. The parameters prefixed with `weights_` are used to set the class weights for the model. The parameters prefixed with `sim_` are used to set the parameters for simulating missing values (for evaluation). The parameters prefixed with `tune_` are used to set the hyperparameter tuning parameters. The parameters prefixed with `plot_` are used to set the parameters for plotting the results. All other parameters are used to set the general parameters for the class.

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
            model_latent_dim (int, optional): Latent dimension of the model. Defaults to 2.
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
        self.model_name = "ImputeUBP"
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
        self.Model = UBPModel

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
        }

        # Convert output_dir to Path if not already
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

    def fit(self, X: np.ndarray | pd.DataFrame | list | Tensor, y: Any | None = None):
        """Fit the model using the input data.

        This method fits the model using the input data. The ``transform`` method then transforms the input data and imputes the missing values using the trained model.

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

        self.Xsim_enc_ = Xsim_enc
        self.X_ = X

        if self.tune:
            self.tune_hyperparameters()

        self.loader_ = self.get_data_loaders(Xsim_enc, X, self.latent_dim)

        self.best_loss_, self.model_, self.history_ = self.train_final_model(
            self.loader_
        )
        self.metrics_ = self.evaluate_model(
            objective_mode=False, trial=None, model=self.model_, loader=self.loader_
        )
        self.plotter_.plot_history(self.history_)

        return self

    def transform(self, X: np.ndarray | pd.DataFrame | list | Tensor) -> np.ndarray:
        """Transform and impute the data using the trained model.

        This method transforms the input data and imputes the missing values using the trained model. The input data is transformed using the transformers and then imputed using the trained model. The ``fit()`` method must be called before calling this method.

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

    def objective(self, trial: optuna.Trial, Model: torch.nn.Module) -> float:
        """Optimized Objective function for Optuna.

        This method is used as the objective function for hyperparameter tuning using Optuna. It is used to optimize the hyperparameters of the model.

        Args:
            trial (optuna.Trial): Optuna trial object.
            Model (torch.nn.Module): Model class to instantiate.

        Returns:
            float: The metric value to optimize. Which metric to use is based on the `tune_metric` attribute. Defaults to 'pr_macro', which works well with imbalanced classes.
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
        }

        # Build and initialize model
        model = self.build_model(Model, model_params)
        model.apply(self.initialize_weights)

        train_loader = self.get_data_loaders(self.Xsim_enc_, self.X_, latent_dim)

        try:
            # Train and validate the model
            _, model = self.train_and_validate_model(
                model=model,
                loader=train_loader,
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
                objective_mode=True, trial=trial, model=model, loader=train_loader
            )

            if self.tune_metric not in metrics:
                msg = f"Invalid tuning metric: {self.tune_metric}"
                self.logger.error(msg)
                raise KeyError(msg)

            return metrics[self.tune_metric]

        except Exception as e:
            self.logger.warning(f"Trial {trial.number} pruned due to exception: {e}")
            raise optuna.exceptions.TrialPruned()

        finally:
            self.reset_weights(model.phase1_decoder)
            self.reset_weights(model.phase23_decoder)
            self.reset_weights(model)

    def set_best_params(self, best_params: dict) -> dict:
        """Set the best hyperparameters.

        This method sets the best hyperparameters for the model after tuning.

        Args:
            best_params (dict): Dictionary of best hyperparameters.

        Returns:
            dict: Dictionary of best hyperparameters. The best hyperparameters are set as attributes of the class.
        """
        # Load best hyperparameters
        self.latent_dim = best_params["latent_dim"]
        self.hidden_layer_sizes = np.linspace(
            16, 256, best_params["num_hidden_layers"]
        ).astype(int)[::-1]
        self.dropout_rate = best_params["dropout_rate"]
        self.lr_ = best_params["learning_rate"]
        self.gamma = best_params["gamma"]

        best_params_ = {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "gamma": self.gamma,
        }

        return best_params_

    def train_step(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        l1_penalty: float,
        phase: int,
        lr_input: float,
    ):
        """Perform one epoch of training.

        This method performs one epoch of training on the model using the input DataLoader. The model is trained in three phases: Phase 1 uses a linear decoder that refines both inputs and weights, intended to get the refined inputs to a good starting point for Phase 2. Phases 2 and 3 use a flexible deeper network, but Phase 2 only refines weights, whereas Phase 3 refines the inputs and weights. The model uses a MaskedFocalLoss for training to handle class imbalance. The model is trained using PyTorch.

        Args:
            loader (torch.utils.data.DataLoader): Training DataLoader.
            optimizer (Optimizer): Current phase optimizer.
            model (nn.Module): The model to train.
            l1_penalty (float): L1 penalty coefficient.
            phase (int): Current training phase (1, 2, or 3).
            lr_input (float): Learning rate for manual input updates.

        Returns:
            float: The mean training loss across batches over the epoch.
        """
        model.train()
        running_loss = 0.0
        num_batches = 0

        for X_batch, y_batch, mask_batch, batch_indices in loader:
            # Decide whether to track gradients on X_batch
            # Phase 1 and 3 => refine inputs
            # Phase 2 => do NOT refine inputs
            if phase in {1, 3}:
                X_batch.requires_grad = True
            else:
                X_batch.requires_grad = False

            optimizer.zero_grad()

            # Forward pass
            if phase == 1:
                # Phase 1 uses the linear decoder
                outputs = model.phase1_decoder(X_batch)
            else:
                # Phase 2 or 3 uses the deeper network
                outputs = model.phase23_decoder(X_batch)

            if outputs.dim() == 2:
                logits = outputs.view(
                    X_batch.size(0), self.num_features_, self.num_classes_
                )
            elif outputs.dim() == 3:
                logits = outputs
            else:
                msg = f"Invalid output shape: {outputs.shape}. Must be 2D or 3D."
                self.logger.error(msg)
                raise ValueError(msg)

            # Compute loss
            loss = model.compute_loss(
                y_batch, logits, mask_batch, class_weights=self.class_weights_
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

            # If in Phase 1 or Phase 3, manually update input
            if phase in {1, 3}:
                with torch.no_grad():
                    # Gradient w.r.t. X_batch
                    X_batch -= lr_input * X_batch.grad
                    X_batch.grad.zero_()

                # Update the dataset's underlying array with the new refined inputs
                loader.dataset.data[batch_indices] = X_batch.detach().cpu()

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / num_batches
        return epoch_loss, loader

    def train_and_validate_model(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        l1_penalty: float,
        lr: float,
        trial=None,
        return_history: bool = False,
    ):
        """Train the model in three phases, returning the best loss and final model.

        Phase 1:
            - Linear model (phase1_decoder).
            - Refine both inputs and weights.
            - End with updated inputs, discard phase1_decoder weights.

        Phase 2:
            - Reset deeper model weights (phase23_decoder).
            - Use refined inputs from Phase 1 but do not refine them.
            - Only refine weights of phase23_decoder.

        Phase 3:
            - Same deeper architecture from Phase 2.
            - Refine both inputs and the weights from Phase 2.

        Args:
            model (nn.Module): The initialized model.
            loader (DataLoader): Training DataLoader.
            l1_penalty (float): L1 coefficient for regularization.
            lr (float): Learning rate for input refinement.
            trial: Optuna trial object (if tuning).
            return_history (bool): Whether to return per-epoch history.

        Returns:
            (float, nn.Module, dict) or (float, nn.Module):
                The best validation loss, trained model, and optionally the training histories.
        """
        if model is None:
            msg = "Model is not initialized."
            self.logger.error(msg)
            raise ValueError(msg)

        if loader is None:
            msg = "DataLoader is not initialized."
            self.logger.error(msg)
            raise ValueError(msg)

        # -------------
        # PHASE 1
        # -------------
        if trial is None:
            self.logger.info("PHASE 1: Linear model, refine inputs + weights")

        phase1_optimizer = torch.optim.Adam(model.phase1_decoder.parameters(), lr=lr)
        phase1_scheduler = CosineAnnealingLR(phase1_optimizer, T_max=self.epochs)

        (best_loss_phase1, best_model_phase1, train_hist_p1) = (
            self.execute_training_loop(
                loader=loader,
                optimizer=phase1_optimizer,
                scheduler=phase1_scheduler,
                model=model,
                l1_penalty=l1_penalty,
                trial=trial,
                return_history=return_history,
                phase=1,
                lr_input=lr,  # used to refine inputs manually
            )
        )

        # -------------
        # PHASE 2
        # -------------
        if trial is None:
            self.logger.info("PHASE 2: Deeper architecture, refine weights only")

        # For Phase 2, do not refine inputs => we do not set X_batch.
        # requires_grad_(True).
        # We'll re-init from the base_optimizer for the deeper network weights.
        phase2_optimizer = torch.optim.Adam(model.phase23_decoder.parameters(), lr=lr)
        phase2_scheduler = CosineAnnealingLR(phase2_optimizer, T_max=self.epochs)

        (best_loss_phase2, best_model_phase2, train_hist_p2) = (
            self.execute_training_loop(
                loader=loader,
                optimizer=phase2_optimizer,
                scheduler=phase2_scheduler,
                model=model,
                l1_penalty=l1_penalty,
                trial=trial,
                return_history=return_history,
                phase=2,
                lr_input=lr,
            )
        )

        # -------------
        # PHASE 3
        # -------------
        if trial is None:
            self.logger.info("PHASE 3: Same architecture, refine inputs + weights")

        # Do not reset weights. Just keep whatever was found in Phase 2.
        # Now refine both the deeper model’s weights and the inputs.
        phase3_optimizer = torch.optim.Adam(model.phase23_decoder.parameters(), lr=lr)
        phase3_scheduler = CosineAnnealingLR(phase3_optimizer, T_max=self.epochs)

        (best_loss_phase3, best_model_phase3, train_hist_p3) = (
            self.execute_training_loop(
                loader=loader,
                optimizer=phase3_optimizer,
                scheduler=phase3_scheduler,
                model=model,
                l1_penalty=l1_penalty,
                trial=trial,
                return_history=return_history,
                phase=3,
                lr_input=lr,
            )
        )

        # Return best of Phase 3.rtfgft
        if return_history:
            return (
                best_loss_phase3,
                best_model_phase3,
                {
                    "Train": {
                        "Phase 1": train_hist_p1,
                        "Phase 2": train_hist_p2,
                        "Phase 3": train_hist_p3,
                    },
                },
            )
        else:
            return best_loss_phase3, best_model_phase3

    def execute_training_loop(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        model: torch.nn.Module,
        l1_penalty: float,
        trial,
        return_history: bool,
        phase: int,
        lr_input: float,
    ):
        """Execute one phase of training (which may be Phase 1, 2, or 3).

        The training loop for each phase is executed here. The model is trained for the specified number of epochs. The training loop includes the training step, optimizer step, scheduler step, and early stopping. The training loop also includes Optuna pruning if tuning is enabled.

        Args:
            loader (torch.utils.data.DataLoader): Training DataLoader.
            optimizer (Optimizer): Optimizer for this phase.
            scheduler (_LRScheduler): Scheduler for this phase.
            model (nn.Module): The UBP model.
            l1_penalty (float): L1 penalty coefficient.
            trial: Optuna trial object if tuning.
            return_history (bool): Whether to return history for each epoch.
            phase (int): Current phase (1,2,3).
            lr_input (float): Learning rate used to refine inputs (Phase 1 & 3).

        Returns:
            tuple: (best_loss, best_model, train_history)
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
            train_loss, loader = self.train_step(
                loader, optimizer, model, l1_penalty, phase, lr_input
            )

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
                        self.logger.warning(f"Current train_loss: {train_loss}")
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
