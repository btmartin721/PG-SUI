import copy
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Literal, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from snpio.utils.logging import LoggerManager
from torch.optim.lr_scheduler import CosineAnnealingLR

from pgsui.data_processing.transformers import SimGenotypeDataTransformer
from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.models.nlpca_model import NLPCAModel
from pgsui.utils.misc import validate_input_type


class ImputeNLPCA(BaseNNImputer):

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
        weights_beta: float = 0.9999,
        weights_normalize: bool = False,
        weights_temperature: float = 1.0,
        weights_max_weight: float = 20.0,
        weights_min_weight: float = 0.001,
        weights_max_ratio: float = 1.0,
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
        model_lr_input_factor: float = 1.0,
        model_early_stop_gen: int = 25,
        model_min_epochs: int = 100,
        model_optimizer: str = "adam",
        model_hidden_activation: str = "elu",
        model_lr_patience: int = 10,
        model_epochs: int = 5000,
        model_validation_split: float = 0.2,
        model_l1_penalty: float = 0.0,
        model_gamma: float = 2.0,
        model_device: Literal["gpu", "cpu"] = "cpu",
        scoring_averaging: str = "weighted",
        plot_format: str = "pdf",
        plot_fontsize: int | float = 18,
        plot_dpi: int = 300,
        plot_title_fontsize: int = 20,
        plot_despine: bool = True,
        plot_show_plots: bool = False,
        debug: bool = False,
    ):
        """Impute missing genotypes using Non-linear Principal Component Analysis (NLPCA).

        This class is used to impute missing values in genotype data using Non-linear Principal Component Analysis (NLPCA). The model is trained on the genotype data and used to impute the missing values. The class inherits from BaseNNImputer. The model refines the inputs and weights to fit the real data (targets) using backpropagation. The model also uses a MaskedFocalLoss for training to handle class imbalance and is trained using PyTorch. It uses the NLPCAModel class for the model architecture, the Scorer class for evaluating the performance of the model, and the SNPio LoggerManager class for logging messages.

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
            weights_beta (float, optional): Beta parameter for class weights. Defaults to 0.9999.
            weights_normalize (bool, optional): Whether to normalize class weights. Defaults to False.
            weights_temperature (float, optional): Temperature parameter for class weights. Defaults to 1.0.
            weights_max_weight (float, optional): Maximum weight for class weights. Defaults to 20.0.
            weights_min_weight (float, optional): Minimum weight for class weights. Defaults to 0.001.
            weights_max_ratio (float, optional): Maximum ratio for class weights. Defaults to 1.0.
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
            model_lr_input_factor (float, optional): Learning rate input factor. Defaults to 1.0.
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
        self.model_name = "ImputeNLPCA"
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
        self.Model = NLPCAModel

        self.genotype_data = genotype_data
        self.latent_dim = model_latent_dim
        self.dropout_rate = model_dropout_rate
        self.num_hidden_layers = model_num_hidden_layers
        self.hidden_layer_sizes = model_hidden_layer_sizes
        self.activation = model_hidden_activation
        self.hidden_activation = model_hidden_activation
        self.batch_size = model_batch_size
        self.learning_rate = model_learning_rate
        self.lr_input_factor = model_lr_input_factor
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
        self.max_weight = weights_max_weight
        self.min_weight = weights_min_weight
        self.max_ratio = weights_max_ratio
        self.beta = weights_beta
        self.seed = seed

        self.use_convolution = False

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

    def fit(self, X: np.ndarray | pd.DataFrame | list | torch.Tensor, y: None = None):
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
        n_samples, n_features_raw = X.shape
        mask = np.logical_or(X < 0, np.isnan(X))
        self.original_missing_mask_ = mask

        X = X.astype(float)
        X[mask] = -1  # Replace missing values with -1
        self.X_ = X

        # Count number of classes for activation.
        # If 4 classes, use sigmoid, else use softmax.
        # Ignore missing values (-1) in counting of classes.
        # 1. Compute the number of distinct classes
        self.num_classes_ = self._count_num_classes(X, mask)
        self.num_features_ = n_features_raw  # Use the raw feature count
        self.model_params.update(
            {"num_classes": self.num_classes_, "n_features": self.num_features_}
        )

        self.tt_, self.plotter_, self.scorers_ = self.init_transformers()
        self.tt_.fit(self.X_)

        # --- HYPERPARAMETER TUNING ---
        if self.tune:
            self.tune_hyperparameters()  # This will call the objective function

        self.best_params_ = getattr(self, "best_params_", self.model_params.copy())
        self.best_params_["latent_dim"] = self.latent_dim

        # Create a tuneable latent space, initialized with random values
        self.latent_vectors_ = torch.nn.Parameter(
            torch.randn(
                n_samples,
                self.best_params_["latent_dim"],
                device=self.device,
                requires_grad=True,
            ),
            requires_grad=True,
        )

        original_missing_mask = self.X_ < 0
        class_weights = self.compute_class_weights(
            self.X_,
            ~original_missing_mask,
            use_log_scale=self.weights_log_scale,
            alpha=self.weights_alpha,
            normalize=self.weights_normalize,
            temperature=self.weights_temperature,
            max_weight=self.max_weight,
            min_weight=self.min_weight,
        )

        self.sim_ = SimGenotypeDataTransformer(
            self.genotype_data,
            prop_missing=self.sim_prop_missing,
            strategy=self.sim_strategy,
            seed=self.seed,
            class_weights=class_weights,
        )

        _, missing_masks = self.sim_.fit_transform(self.X_)
        self.original_missing_mask_ = missing_masks["original"]
        self.sim_missing_mask_ = missing_masks["simulated"]
        self.all_missing_mask_ = missing_masks["all"]

        self.class_weights_ = self._class_balanced_weights_from_mask(
            self.X_,
            train_mask=~self.all_missing_mask_,
            num_classes=self.num_classes_,
            beta=self.beta,
            max_ratio=self.max_ratio,
        )

        # --- FINAL MODEL TRAINING SETUP ---
        # After tuning, self.latent_dim, self.learning_rate, etc., are updated
        # with the best parameters. Now we create the final optimizers.
        final_model = self.build_model(self.Model, self.best_params_)
        final_model.apply(self.initialize_weights)

        # Create the index-based loader
        final_loader = self._get_data_loaders(
            y=self.X_, n_samples=n_samples, train_mask=self.all_missing_mask_
        )

        # --- TRAIN THE FINAL MODEL ---
        self.best_loss_, self.model_, self.history_, self.latent_vectors_ = (
            self._train_final_model(
                loader=final_loader, latent_dim=self.best_params_["latent_dim"]
            )
        )

        # --- EVALUATE THE FINAL MODEL ---
        self.metrics_ = self._evaluate_model(
            objective_mode=False,
            trial=None,
            model=self.model_,
            loader=final_loader,
            latent_vectors=self.latent_vectors_,
            eval_mask=self.sim_missing_mask_,
        )

        # --- PLOT TRAINING HISTORY ---
        self.plotter_.plot_history(self.history_)

        return self

    def transform(
        self, X: np.ndarray | pd.DataFrame | list | torch.Tensor
    ) -> np.ndarray:
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

    def _objective(self, trial: optuna.Trial, Model: torch.nn.Module) -> float:
        """Optimized Objective function for Optuna.

        This method is used as the objective function for hyperparameter tuning using Optuna. It is used to optimize the hyperparameters of the model.

        Args:
            trial (optuna.Trial): Optuna trial object.
            Model (torch.nn.Module): Model class to instantiate.

        Returns:
            float: The metric value to optimize. Which metric to use is based on the `tune_metric` attribute. Defaults to 'pr_macro', which works well with imbalanced classes.
        """
        # Sample hyperparameters
        latent_dim = trial.suggest_int("latent_dim", 2, 32)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.05)
        num_hidden_layers = trial.suggest_int("num_hidden_layers", 2, 16)
        hidden_layer_sizes = [int(x) for x in np.linspace(16, 256, num_hidden_layers)][
            ::-1
        ]
        activation = trial.suggest_categorical(
            "activation", ["relu", "elu", "selu", "leaky_relu"]
        )
        gamma = trial.suggest_float("gamma", 0.25, 5.0, step=0.25)
        lr_input_factor = trial.suggest_float("lr_input_factor", 0.05, 5.0, step=0.05)
        beta = trial.suggest_categorical("beta", [0.9, 0.99, 0.999, 0.9999])
        max_ratio = trial.suggest_float("max_ratio", 0.5, 10.0, step=0.5)
        prop_missing = trial.suggest_float("prop_missing", 0.1, 0.5, step=0.05)
        sim_strategy = trial.suggest_categorical(
            "sim_strategy",
            [
                "random",
                "random_balanced",
                "random_inv",
                "random_balanced_multinom",
                "random_inv_multinom",
                "nonrandom",
                "nonrandom_distance",
            ],
        )
        use_log_scale = trial.suggest_categorical("use_log_scale", [True, False])

        alpha = trial.suggest_float("alpha", 0.1, 4.0, step=0.1)
        temperature = trial.suggest_float("temperature", 0.1, 4.0, step=0.1)
        max_weight = trial.suggest_float("max_weight", 5.0, 50.0, step=5.0)
        min_weight = trial.suggest_float("min_weight", 0.001, 1.0, log=True)
        normalize = trial.suggest_categorical("normalize", [True, False])

        original_missing_mask = self.X_ < 0

        class_weights = self.compute_class_weights(
            self.X_,
            train_mask=~original_missing_mask,
            use_log_scale=use_log_scale,
            normalize=normalize,
            alpha=alpha,
            temperature=temperature,
            max_weight=max_weight,
            min_weight=min_weight,
        )

        class_weights = class_weights.float().to(self.device)

        sim = SimGenotypeDataTransformer(
            self.genotype_data,
            prop_missing=prop_missing,
            strategy=sim_strategy,
            seed=self.seed,
            class_weights=class_weights,
        )

        _, missing_masks = sim.fit_transform(self.X_)
        original_missing_mask = missing_masks["original"]
        sim_missing_mask = missing_masks["simulated"]
        all_missing_mask = missing_masks["all"]

        # --- Recompute anything that depends on the loader / masks here ---
        # 2. Compute class weights and move them to the correct device
        class_weights = self._class_balanced_weights_from_mask(
            y=self.X_,
            train_mask=~all_missing_mask,
            num_classes=self.num_classes_,
            beta=beta,
            max_ratio=max_ratio,
        )

        class_weights = class_weights.float().to(self.device)

        model_params = {
            "n_features": self.num_features_,
            "num_classes": self.num_classes_,
            "latent_dim": latent_dim,
            "dropout_rate": dropout_rate,
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
        }

        # Build model + per-trial latents
        model = self.build_model(self.Model, model_params)
        model.apply(self.initialize_weights)

        train_loader = self._get_data_loaders(
            y=self.X_, n_samples=self.X_.shape[0], train_mask=all_missing_mask
        )

        latent_vectors = torch.nn.Parameter(
            torch.randn(
                self.X_.shape[0], latent_dim, device=self.device, requires_grad=True
            ),
            requires_grad=True,
        )

        _, model, latent_vectors = self._train_and_validate_model(
            model=model,
            loader=train_loader,
            lr=learning_rate,
            l1_penalty=self.l1_penalty,
            latent_dim=latent_dim,
            trial=trial,
            latent_vectors=latent_vectors,
            gamma=gamma,
            lr_input_factor=lr_input_factor,
            class_weights=class_weights,
        )

        if model is None:
            msg = f"Model training failed. 'model' object was NoneType after training."
            self.logger.warning(msg)
            raise optuna.exceptions.TrialPruned()

        metrics = self._evaluate_model(
            objective_mode=True,
            trial=trial,
            model=model,
            loader=train_loader,
            latent_vectors=latent_vectors,
            eval_mask=sim_missing_mask,
        )

        if self.tune_metric not in metrics:
            msg = f"Invalid tuning metric: {self.tune_metric}"
            self.logger.error(msg)
            raise KeyError(msg)

        self._clear_resources(model, train_loader, latent_vectors)

        return metrics[self.tune_metric]

    def _clear_resources(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        latent_vectors: torch.nn.Parameter,
    ) -> None:
        """Clear resources during Optuna optimization.

        This method clears the resources used during the Optuna optimization process. This ensures that memory is released and not leaked between trials.

        Args:
            model (torch.nn.Module): The model to clear.
            train_loader (torch.utils.data.DataLoader): The training data loader to clear.
            latent_vectors (torch.nn.Parameter): The latent vectors to clear.
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

    def _set_best_params(self, best_params: dict) -> dict:
        """Set the best hyperparameters.

        This method sets the best hyperparameters for the model after tuning. The best hyperparameters are those that resulted in the highest validation score during the tuning process.

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
        self.learning_rate = best_params["learning_rate"]
        self.gamma = best_params["gamma"]
        self.lr_input_factor = best_params["lr_input_factor"]
        self.activation = best_params["activation"]
        self.beta = best_params["beta"]
        self.max_ratio = best_params["max_ratio"]
        self.sim_prop_missing = best_params["prop_missing"]
        self.sim_strategy = best_params["sim_strategy"]
        self.weights_log_scale = best_params["use_log_scale"]
        self.weights_alpha = best_params["alpha"]
        self.weights_temperature = best_params["temperature"]
        self.max_weight = best_params["max_weight"]
        self.min_weight = best_params["min_weight"]
        self.weights_normalize = best_params["normalize"]

        best_params_ = {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "gamma": self.gamma,
        }

        return best_params_

    def _train_step(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        latent_optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        l1_penalty: float,
        latent_vectors: torch.nn.Parameter,
        gamma: float,
        class_weights: torch.Tensor,
    ) -> Tuple[float, torch.utils.data.DataLoader]:
        """One epoch of training.

        This method performs a single training step over the provided data loader.

        Args:
            loader (torch.utils.data.DataLoader): The data loader for training data.
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            latent_optimizer (torch.optim.Optimizer): The optimizer for the latent vectors.
            model (torch.nn.Module): The model to train.
            l1_penalty (float): The L1 regularization penalty.
            latent_vectors (torch.nn.Parameter): The latent vectors for the model.
            gamma (float): Weighting factor for the loss function.
            class_weights (torch.Tensor | None): Class weights for the loss function.
        """
        model.train()
        running_loss, num_batches = 0.0, 0

        # Unpack the mask along with indices and labels
        for batch_indices, y_batch, mask_batch in loader:
            optimizer.zero_grad()
            latent_optimizer.zero_grad()

            # Slice the latent vectors. This allows the tuning to adjust the
            # latent space representation.
            X_batch_latents = latent_vectors[batch_indices]

            outputs = model.phase23_decoder(X_batch_latents)

            logits = outputs
            if outputs.dim() == 2:
                logits = logits.reshape(
                    len(batch_indices), self.num_features_, self.num_classes_
                )

            # Pass the mask_batch to the loss function
            loss = model.compute_loss(
                y_batch,
                logits,
                mask=mask_batch,
                class_weights=class_weights,
                gamma=gamma,
            )

            if l1_penalty > 0:
                l1_reg = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_penalty * l1_reg

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([latent_vectors], max_norm=1.0)

            optimizer.step()
            latent_optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        return running_loss / max(1, num_batches), loader, latent_vectors

    def _train_and_validate_model(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        lr: float,
        l1_penalty: float,
        latent_dim: int,
        trial: optuna.Trial | None = None,
        return_history: bool = False,
        latent_vectors: torch.nn.Parameter | None = None,
        gamma: float = 2.0,
        lr_input_factor: float = 0.1,
        class_weights: torch.Tensor | None = None,
    ) -> (
        Tuple[float, torch.nn.Module, dict, torch.nn.Parameter]
        | Tuple[float, torch.nn.Module, torch.nn.Parameter]
    ):
        """Train the model; uses per-trial latent_vectors if provided.

        This method orchestrates the training process for NLPCA.

        Args:
            model (torch.nn.Module): The model to train.
            loader (torch.utils.data.DataLoader): The data loader for training data.
            lr (float): Learning rate for the optimizer.
            l1_penalty (float): L1 regularization penalty.
            latent_dim (int): Dimensionality of the latent space.
            trial (optuna.Trial | None): Optuna trial object for hyperparameter optimization.
            return_history (bool): Whether to return training history.
            latent_vectors (torch.nn.Parameter | None): Latent vectors for the model.
            gamma (float): Weighting factor for the loss function.
            lr_input_factor (float): Learning rate factor for the input latent vectors.
            class_weights (torch.Tensor | None): Class weights for the loss function.

        Returns:
            Tuple[float, torch.nn.Module, dict | torch.nn.Parameter]: A tuple containing the best loss, best model, and optionally the training history.
        """
        if latent_vectors is None:
            msg = "Must provide 'latent_vectors' argument, but got NoneType."
            self.logger.error(msg)
            raise TypeError(msg)

        latent_optimizer = torch.optim.Adam([latent_vectors], lr=lr * lr_input_factor)

        # Phase 3 (weights + inputs again)
        optimizer = torch.optim.Adam(model.phase23_decoder.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_loss, best_model, hist, latent_vectors = self._execute_training_loop(
            loader=loader,
            optimizer=optimizer,
            latent_optimizer=latent_optimizer,
            scheduler=scheduler,
            model=model,
            l1_penalty=l1_penalty,
            trial=trial,
            return_history=return_history,
            latent_vectors=latent_vectors,
            gamma=gamma,
            class_weights=class_weights,
        )

        if return_history:
            return (best_loss, best_model, {"Train": hist}, latent_vectors)
        return best_loss, best_model, latent_vectors

    def _train_final_model(
        self, loader: torch.utils.data.DataLoader, latent_dim: int
    ) -> Tuple[float, torch.nn.Module, dict, torch.nn.Parameter]:
        """Train the final model.

        This method trains the final model using the provided data loader and latent dimension. It returns the loss, trained model, and training history.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader for the training data.
            latent_dim (int): Latent dimension for the model.

        Returns:
            Tuple[float, torch.nn.Module, dict, torch.nn.Parameter]: Loss, trained model, and training history.
        """
        which_model = "tuned model" if self.tune else "model"
        self.logger.info(f"Training the {which_model}...")

        # Latent dimension is taken from the shape of the Parameter tensor
        in_dim = self.latent_vectors_.shape[1]
        if not in_dim == latent_dim:
            msg = f"Loader latent_dim={in_dim} != model latent_dim={latent_dim}"
            self.logger.error(msg)
            raise AssertionError(msg)

        self.lr_ = self.learning_rate
        self.lr_patience_ = self.lr_patience
        self.l1_penalty_ = self.l1_penalty
        self.gamma_ = self.gamma
        self.lr_input_factor_ = self.lr_input_factor

        # Build model
        model = self.build_model(self.Model, self.best_params_)
        model.apply(self.initialize_weights)

        # NOTE: train_and_validate_model function should now handle creating
        # both the model optimizer and the latent vector optimizer internally.
        args = [model, loader, self.lr_, self.l1_penalty_, latent_dim]

        loss, trained_model, history, latent_vectors_ = self._train_and_validate_model(
            *args,
            latent_vectors=self.latent_vectors_,
            return_history=True,
            gamma=self.gamma_,
            lr_input_factor=self.lr_input_factor_,
            class_weights=self.class_weights_,
        )

        if trained_model is None:
            msg = "Model was not properly trained. Check the training process."
            self.logger.error(msg)
            raise RuntimeError(msg)

        fn = self.models_dir / "final_model.pt"
        torch.save(trained_model.state_dict(), fn)

        return loss, trained_model, history, latent_vectors_

    def _execute_training_loop(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        latent_optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        model: torch.nn.Module,
        l1_penalty: float,
        trial,
        return_history: bool,
        latent_vectors: torch.nn.Parameter | None = None,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
    ) -> Tuple[float, torch.nn.Module, dict | None, torch.nn.Parameter]:
        """Execute NLPCA training.

        The training loop is executed here. The model is trained for the specified number of epochs. The training loop includes the training step, optimizer step, scheduler step, and early stopping. The training loop also includes Optuna pruning if tuning is enabled.

        Args:
            loader (torch.utils.data.DataLoader): Training DataLoader.
            optimizer (Optimizer): Optimizer.
            latent_optimizer (Optimizer): Optimizer for the latent space.
            scheduler (_LRScheduler): Scheduler.
            model (nn.Module): The UBP model.
            l1_penalty (float): L1 penalty coefficient.
            trial: Optuna trial object if tuning.
            return_history (bool): Whether to return history for each epoch.
            latent_vectors (torch.nn.Parameter | None): Latent vectors for the trial.
            gamma (float): Focal loss gamma parameter.
            class_weights (torch.Tensor | None): Class weights for the loss function.

        Returns:
            Tuple[float, torch.nn.Module, dict | None, torch.nn.Parameter]: Best_loss, best_model, train_history, latent_vectors.
        """
        if class_weights is None:
            msg = "Must provide 'class_weights' argument, but got NoneType."
            self.logger.error(msg)
            raise TypeError(msg)

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

        warm, ramp, gamma_final = 50, 100, gamma
        for epoch in range(self.epochs):
            # ---- TRAIN STEP ----
            if epoch < warm:
                model.gamma = 0.0
            elif epoch < warm + ramp:
                t = (epoch - warm) / ramp
                model.gamma = gamma_final * t
            else:
                model.gamma = gamma_final

            train_loss, loader, latent_vectors = self._train_step(
                loader,
                optimizer,
                latent_optimizer,
                model,
                l1_penalty,
                latent_vectors=latent_vectors,
                gamma=gamma,
                class_weights=class_weights,
            )

            if trial is not None and (
                torch.isnan(torch.tensor(train_loss))
                or torch.isinf(torch.tensor(train_loss))
            ):
                msg = f"Loss became NaN or Inf at epoch {epoch}. Pruning trial."
                self.logger.error(msg)
                raise optuna.exceptions.TrialPruned()

            scheduler.step()

            if return_history:
                train_history.append(train_loss)

            # ---- EARLY STOPPING ----
            early_stopping(train_loss, model)
            if early_stopping.early_stop:
                best_loss = early_stopping.best_score
                best_model = copy.deepcopy(early_stopping.best_model)
                break

        # If early stopping was not triggered, get the best from the end
        if best_model is None:
            best_loss = early_stopping.best_score
            best_model = copy.deepcopy(early_stopping.best_model)

        return best_loss, best_model, train_history, latent_vectors

    def _evaluate_model(
        self,
        objective_mode: bool = False,
        trial: optuna.Trial | None = None,
        model: torch.nn.Module | None = None,
        loader: torch.utils.data.DataLoader | None = None,
        latent_vectors: torch.nn.Parameter | None = None,
        eval_mask: np.ndarray | None = None,
    ) -> dict:
        """Perform evaluation of the model.

        This method evaluates the model on the provided data loader and computes relevant metrics. It handles both objective mode (for hyperparameter tuning) and standard evaluation mode. In objective mode, it returns the metrics directly for use in optimization. In standard mode, it saves the metrics to a file and generates plots.

        Args:
            objective_mode (bool): Whether the evaluation is for the objective function.
            trial (optuna.Trial | None): The Optuna trial object.
            model (torch.nn.Module | None): The model to evaluate.
            loader (torch.utils.data.DataLoader | None): The data loader for evaluation.
            latent_vectors (torch.nn.Parameter): The latent vectors for evaluation.

        Returns:
            dict: The evaluation metrics as a dictionary.

        Raises:
            TypeError: If any of the required arguments are None.
            ValueError: If the target data is invalid.
        """
        if objective_mode and trial is None:
            msg = "Trial object must be provided for objective mode."
            self.logger.error(msg)
            raise TypeError(msg)

        if any(x is None for x in [model, loader, latent_vectors, eval_mask]):
            if model is None:
                msg = "Model must be provided for evaluation, but got None."
                self.logger.error(msg)
                raise TypeError(msg)

            if loader is None:
                msg = "Data loader must be provided for evaluation, but got None."
                self.logger.error(msg)
                raise TypeError(msg)

            if latent_vectors is None:
                msg = f"Latent vectors must be provided for evaluation with {self.model_name}, but got NoneType."
                self.logger.error(msg)
                raise TypeError(msg)

            if eval_mask is None:
                msg = "Evaluation mask must be provided for evaluation, but got None."
                self.logger.error(msg)
                raise TypeError(msg)

        # For TensorDataset, the tensors are stored in a tuple. We need the
        # second one, which holds the ground-truth labels.
        if not isinstance(loader.dataset, torch.utils.data.TensorDataset):
            msg = "Expected loader to wrap a TensorDataset."
            self.logger.error(msg)
            raise TypeError(msg)

        y_true_labels = loader.dataset.tensors[1].detach().cpu().numpy()
        mask_test = eval_mask

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
            # If the target is not 2D or 3D, raise an error
            ytl_shape = y_true_labels.shape
            msg = f"Invalid target shape: {ytl_shape}. Must be 2D or 3D."
            self.logger.error(msg)
            raise ValueError(msg)

        pred_labels, pred_proba = self._predict(
            model=model, return_proba=True, latent_vectors=latent_vectors
        )

        # Filter to only the simulated missing values for scoring
        y_true_labels, pred_labels, pred_proba, y_true_enc = (
            y_true_labels[mask_test],
            pred_labels[mask_test],
            pred_proba[mask_test],
            y_true_enc[mask_test],
        )

        # sanitize probabilities before scoring
        if not np.isfinite(pred_proba).all():
            self.logger.warning(
                "Non-finite scores in pred_proba; sanitizing for metrics."
            )
            pred_proba = np.nan_to_num(pred_proba, nan=1e-9, posinf=1.0, neginf=0.0)
            denom = pred_proba.sum(axis=-1, keepdims=True)
            denom[denom == 0] = 1.0
            pred_proba = pred_proba / denom

        try:
            # Metric computation
            metrics = self.scorers_.evaluate(
                y_true_labels,
                pred_labels,
                y_true_enc,
                pred_proba,
                objective_mode,
                self.tune_metric,
            )
        except IndexError as e:
            msg = f"IndexError occurred during metric computation: {e}"
            self.logger.error(msg)
            if trial is not None:
                raise optuna.exceptions.TrialPruned()
            else:
                raise e

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

    def _predict(
        self,
        model: torch.nn.Module,
        return_proba: bool = False,
        latent_vectors: torch.nn.Parameter | None = None,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Predict using the trained model.

        For the NLPCA model, this method ignores Xenc and returns the full reconstruction from the trained latent vectors. For other models, it predicts on Xenc.

        Args:
            model (torch.nn.Module): The model to use for prediction.
            return_proba (bool): Whether to return predicted probabilities.
            latent_vectors (torch.nn.Parameter | None): The latent vectors for prediction.

        Returns:
            np.ndarray | Tuple[np.ndarray, np.ndarray]: The predicted labels and/or probabilities. If `return_proba` is True, returns a tuple of predicted labels and probabilities; otherwise, returns only the predicted labels.

        Raises:
            ValueError: If the target data is invalid.
            TypeError: If any of the required arguments are None.
        """
        self.ensure_attribute("tt_")

        if any(x is None for x in [model, latent_vectors, self.tt_]):
            if model is None:
                msg = "Model is not fitted yet. Call `fit()` before prediction."
                self.logger.error(msg)
                raise AttributeError(msg)

            if latent_vectors is None:
                msg = f"Latent vectors must be provided for prediction with the '{self.model_name}' model."
                self.logger.error(msg)
                raise TypeError(msg)

            if self.tt_ is None:
                msg = "Transformer ('tt_') has not been initialized."
                self.logger.error(msg)
                raise TypeError(msg)

        # For UBP, the input to the decoder is always the trained latent
        # vectors. The Xenc argument is ignored.
        Xtensor = latent_vectors.to(self.device)

        model.eval()
        with torch.no_grad():
            outputs = model.phase23_decoder(Xtensor)

        # If the model returns multiple outputs, assume first is recon logits
        recon_logits = outputs[0] if isinstance(outputs, tuple) else outputs

        if recon_logits.dim() == 2:
            recon_logits = recon_logits.reshape(
                -1, self.num_features_, self.num_classes_
            )

        y_pred_proba = torch.softmax(recon_logits, dim=-1)
        y_pred_proba = validate_input_type(y_pred_proba)  # to numpy
        y_pred_labels = self.tt_.inverse_transform(y_pred_proba)

        # Safety check
        if np.isnan(y_pred_labels).any() or np.any(y_pred_labels < 0):
            # This check is important. If the model outputs all zeros for a SNP,
            # the inverse_transform might produce a NaN or -1.
            # We can fill these with the most probable class (0, homozygous ref).
            self.logger.warning(
                "NaNs or negative values found in prediction. Filling with 0."
            )
            y_pred_labels = np.nan_to_num(y_pred_labels, nan=0, neginf=0, posinf=0)

        return (y_pred_labels, y_pred_proba) if return_proba else y_pred_labels

    def _get_data_loaders(
        self,
        y: np.ndarray | torch.Tensor | list,
        n_samples: int,
        train_mask: np.ndarray,
    ) -> torch.utils.data.DataLoader:
        """Creates a unified, index-based DataLoader.

        This DataLoader will provide batches of data for training, including both the input features and the target labels.

        Args:
            y (np.ndarray | torch.Tensor | list): The target values.
            n_samples (int): The number of samples in the dataset.
            train_mask (np.ndarray): The training mask array.
        """
        y_tensor = validate_input_type(y, "tensor").long().to(self.device)

        # Create the training mask tensor and move it to the correct device
        train_mask_tensor = torch.from_numpy(~train_mask).bool().to(self.device)

        dataset = torch.utils.data.TensorDataset(
            torch.arange(n_samples, device=self.device), y_tensor, train_mask_tensor
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
