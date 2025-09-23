import copy
from typing import TYPE_CHECKING, Dict, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.utils.logging import LoggerManager
from torch.optim.lr_scheduler import CosineAnnealingLR

from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.models.autoencoder_model import AutoencoderModel

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


class ImputeAutoencoder(BaseNNImputer):
    """Impute missing genotypes with a standard Autoencoder on 0/1/2 encodings.

    This class mirrors `ImputeNLPCA` data flow and evaluation, but trains a *conventional* autoencoder (no backprop into inputs / no latent optimization): inputs are the visible 0/1/2 calls (one-hot, zeros where missing), and targets are the same integer 0/1/2 matrix with -1 for missing. The decoder predicts logits for 0/1/2 at each locus, and cross-entropy is computed with masking.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        seed: int | None = None,
        n_jobs: int = 1,
        prefix: str = "pgsui",
        verbose: bool = False,
        weights_beta: float = 0.9999,
        weights_max_ratio: float = 1.0,
        tune: bool = False,
        tune_metric: Literal["f1", "accuracy", "pr_macro"] = "f1",
        tune_n_trials: int = 100,
        model_validation_split: float = 0.2,
        model_latent_dim: int = 16,
        model_dropout_rate: float = 0.2,
        model_num_hidden_layers: int = 3,
        model_batch_size: int = 64,
        model_learning_rate: float = 1e-3,
        model_early_stop_gen: int = 25,
        model_min_epochs: int = 100,
        model_epochs: int = 5000,
        model_l1_penalty: float = 0.0,
        model_layer_scaling_factor: float = 5.0,
        model_layer_schedule: Literal["pyramid", "constant", "linear"] = "pyramid",
        model_gamma: float = 2.0,
        model_hidden_activation: Literal["relu", "elu", "selu", "leaky_relu"] = "relu",
        model_device: Literal["gpu", "cpu", "mps"] = "cpu",
        plot_format: Literal["pdf", "png", "jpg", "jpeg"] = "pdf",
        plot_fontsize: int = 18,
        plot_despine: bool = True,
        plot_dpi: int = 300,
        plot_show_plots: bool = False,
        debug: bool = False,
    ):
        """Initialize the Autoencoder imputer.

        Args:
            genotype_data (GenotypeData): Source genotype data object.
            seed (int | None): Random seed. If None, use random seed.
            n_jobs (int): Number of parallel jobs. If -1, use all available cores.
            prefix (str): Output prefix.
            verbose (bool): Verbose logging.
            weights_beta (float): Beta for class-balanced weights.
            weights_max_ratio (float): Max ratio clamp for class weights.
            tune (bool): Whether to run Optuna tuning.
            tune_metric (str): Metric name optimized during tuning.
            tune_n_trials (int): Number of Optuna trials.
            model_validation_split (float): Validation split fraction.
            model_latent_dim (int): Bottleneck dimension for AE.
            model_dropout_rate (float): Dropout rate.
            model_num_hidden_layers (int): Number of hidden layers (encoder/decoder sized internally).
            model_batch_size (int): Batch size.
            model_learning_rate (float): Learning rate.
            model_early_stop_gen (int): Early stopping patience.
            model_min_epochs (int): Minimum epochs before early stop.
            model_epochs (int): Max epochs.
            model_l1_penalty (float): L1 regularization.
            model_layer_scaling_factor (float): Hidden layer scaling factor.
            model_layer_schedule (str): Hidden size schedule ('pyramid', 'constant', 'linear').
            model_gamma (float): Focal loss gamma.
            model_hidden_activation (str): Activation function.
            model_device (str): Device to run on.
            plot_format (str): Plot format.
            plot_fontsize (int): Plot font size.
            plot_despine (bool): If True, despine plots.
            plot_dpi (int): Plot DPI.
            plot_show_plots (bool): If True, show plots interactively.
            debug (bool): Debug mode.
        """
        self.model_name = "ImputeAutoencoder"
        kwargs = {"prefix": prefix, "debug": debug, "verbose": verbose}
        logman = LoggerManager(__name__, **kwargs)
        self.logger = logman.get_logger()

        super().__init__(
            prefix=prefix, device=model_device, verbose=verbose, debug=debug
        )

        self.genotype_data = genotype_data
        self.pgenc = GenotypeEncoder(genotype_data)
        self.Model = AutoencoderModel
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.n_jobs = n_jobs

        # Model & training params
        self.latent_dim = model_latent_dim
        self.dropout_rate = model_dropout_rate
        self.num_hidden_layers = model_num_hidden_layers
        self.layer_scaling_factor = model_layer_scaling_factor
        self.layer_schedule = model_layer_schedule
        self.batch_size = model_batch_size
        self.learning_rate = model_learning_rate
        self.early_stop_gen = model_early_stop_gen
        self.min_epochs = model_min_epochs
        self.epochs = model_epochs
        self.l1_penalty = model_l1_penalty
        self.gamma = model_gamma
        self.activation = model_hidden_activation
        self.validation_split = model_validation_split
        self.beta = weights_beta
        self.max_ratio = weights_max_ratio

        # Tuning
        self.tune = tune
        self.tune_metric = tune_metric
        self.n_trials = tune_n_trials

        # Plotting & Output
        self.prefix = prefix
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi
        self.show_plots = plot_show_plots
        self.plot_fontsize = plot_fontsize
        self.title_fontsize = plot_fontsize
        self.despine = plot_despine
        self.scoring_averaging = "weighted"

        # Core model config
        self.is_haploid = None
        self.num_classes_ = None  # 2 if haploid else 3
        self.model_params = {}

    def fit(self) -> "ImputeAutoencoder":
        """Fit the Autoencoder on 0/1/2 encoded genotypes (missing = -9 or -1).

        This method prepares the data, splits into training and validation sets, initializes the model, and trains it using focal cross-entropy loss with class weighting. It also handles hyperparameter tuning if enabled.

        Returns:
            ImputeAutoencoder: Fitted instance.

        Raises:
            NotFittedError: If training fails.
        """
        self.logger.info(f"Fitting {self.model_name} (0/1/2 AE) ...")

        # --- DATA PREPARATION (parity with ImputeNLPCA) ---
        X = self.pgenc.genotypes_012.astype(np.float32)
        X[X < 0] = np.nan
        X[np.isnan(X)] = -1
        self.ground_truth_ = X.astype(np.int64)

        # Ploidy & classes
        self.is_haploid = np.all(
            np.isin(
                self.genotype_data.snp_data, ["A", "C", "G", "T", "N", "-", ".", "?"]
            )
        )
        self.ploidy = 1 if self.is_haploid else 2
        self.num_classes_ = 2 if self.is_haploid else 3
        self.logger.info(
            f"Data is {'haploid' if self.is_haploid else 'diploid'}. "
            f"Using {self.num_classes_} classes."
        )

        n_samples, self.num_features_ = X.shape

        # AE model params: decoder outputs L * num_classes
        self.model_params = {
            "n_features": self.num_features_,
            "num_classes": self.num_classes_,
            "latent_dim": self.latent_dim,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            # hidden_layer_sizes added below
        }

        # Train/Val split
        indices = np.arange(n_samples)
        train_idx, val_idx = train_test_split(
            indices, test_size=self.validation_split, random_state=self.seed
        )
        self.train_idx_, self.test_idx_ = train_idx, val_idx
        self.X_train_ = self.ground_truth_[train_idx]
        self.X_val_ = self.ground_truth_[val_idx]

        # Plotters/scorers & tuning
        self.plotter_, self.scorers_ = self.initialize_plotting_and_scorers()
        if self.tune:
            self.tune_hyperparameters()

        self.best_params_ = getattr(self, "best_params_", self._default_best_params())

        # Class weights from train set
        self.class_weights_ = self._class_weights_from_zygosity(self.X_train_).to(
            self.device
        )

        # Loader (indices + targets); inputs are created ad hoc in _train_step
        train_loader = self._get_data_loaders(self.X_train_)

        # Build and train model
        model = self.build_model(self.Model, self.best_params_)
        model.apply(self.initialize_weights)

        loss, trained_model, history = self._train_and_validate_model(
            model=model,
            loader=train_loader,
            lr=self.learning_rate,
            l1_penalty=self.l1_penalty,
            return_history=True,
            class_weights=self.class_weights_,
        )

        if trained_model is None:
            raise RuntimeError("Final model training failed.")

        torch.save(
            trained_model.state_dict(), self.models_dir / "final_model_ae_012.pt"
        )

        self.best_loss_, self.model_, self.history_ = (
            loss,
            trained_model,
            {"Train": history},
        )
        self.is_fit_ = True

        # Evaluate on validation set (same flow as ImputeNLPCA)
        self._evaluate_model(self.X_val_, self.model_, self.best_params_)
        self.plotter_.plot_history(self.history_)
        return self

    def transform(self) -> np.ndarray:
        """Impute missing genotypes and return IUPAC strings.

        This method uses the trained autoencoder to predict missing genotypes in the dataset. It fills in only the missing values and decodes the imputed 0/1/2 matrix back to IUPAC strings. It also generates distribution plots of the original and imputed genotypes.

        Returns:
            np.ndarray: IUPAC strings array of shape (n_samples, L).
        """
        if not getattr(self, "is_fit_", False):
            raise NotFittedError("Model is not fitted. Call fit() before transform().")

        self.logger.info("Imputing entire dataset with AE (0/1/2)...")
        X_to_impute = self.ground_truth_.copy()

        # Predict with masked inputs (no latent optimization)
        pred_labels, _ = self._predict(self.model_, X=X_to_impute, return_proba=True)

        # Fill only missing
        missing_mask = X_to_impute == -1
        imputed_array = X_to_impute.copy()
        imputed_array[missing_mask] = pred_labels[missing_mask]

        # Decode to IUPAC strings & plot
        imputed_genotypes = self.pgenc.decode_012(imputed_array)
        original_genotypes = self.pgenc.decode_012(X_to_impute)

        plt.rcParams.update(self.plotter_.param_dict)  # ensure consistent style
        self.plotter_.plot_gt_distribution(original_genotypes, is_imputed=False)
        self.plotter_.plot_gt_distribution(imputed_genotypes, is_imputed=True)

        return imputed_genotypes

    def _get_data_loaders(self, y: np.ndarray) -> torch.utils.data.DataLoader:
        """Create DataLoader over indices + integer targets (-1 for missing).

        This method creates a PyTorch DataLoader that yields batches of indices and their corresponding genotype targets encoded as integers (0, 1, 2) with -1 indicating missing values. The DataLoader is shuffled to ensure random sampling during training.

        Args:
            y (np.ndarray): 0/1/2 matrix with -1 for missing.

        Returns:
            torch.utils.data.DataLoader: Shuffled DataLoader.
        """
        y_tensor = torch.from_numpy(y).long().to(self.device)
        dataset = torch.utils.data.TensorDataset(
            torch.arange(len(y), device=self.device), y_tensor
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

    def _train_and_validate_model(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        lr: float,
        l1_penalty: float,
        trial: optuna.Trial | None = None,
        return_history: bool = False,
        class_weights: torch.Tensor | None = None,
    ) -> Tuple[float, torch.nn.Module | None, list | None]:
        """Wrap the AE training loop (no latent optimizer).

        Args:
            model (torch.nn.Module): Autoencoder model.
            loader (DataLoader): Yields (indices, y_int) where y_int is 0/1/2, -1 for missing.
            lr (float): Learning rate.
            l1_penalty (float): L1 regularization.
            trial (optuna.Trial | None): Optuna trial for pruning, or None.
            return_history (bool): If True, return training history.
            class_weights (torch.Tensor | None): Class weights for cross-entropy loss.

        Returns:
            Tuple[float, torch.nn.Module | None, list | None]: Best loss, best model, and training history (if requested). If training fails, returns (inf, None, None).
        """
        if class_weights is None:
            raise TypeError("Must provide class_weights.")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_loss, best_model, hist = self._execute_training_loop(
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            l1_penalty=l1_penalty,
            trial=trial,
            return_history=return_history,
            class_weights=class_weights,
        )
        if return_history:
            return best_loss, best_model, hist
        return best_loss, best_model, None

    def _execute_training_loop(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        model: torch.nn.Module,
        l1_penalty: float,
        trial,
        return_history: bool,
        class_weights: torch.Tensor,
    ) -> Tuple[float, torch.nn.Module, list]:
        """Execute training with focal CE (gamma warm/ramp) mirroring NLPCA.

        This method runs the training loop for the autoencoder model, applying focal cross-entropy loss with class weighting and handling early stopping. It includes a learning rate scheduler and supports Optuna pruning if a trial is provided.

        Args:
            loader (DataLoader): Yields (indices, y_int) where y_int is 0/1/2, -1 for missing.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            model (torch.nn.Module): Autoencoder model.
            l1_penalty (float): L1 regularization.
            trial (optuna.Trial | None): Optuna trial for pruning, or None.
            return_history (bool): If True, return training history.
            class_weights (torch.Tensor): Class weights for
                cross-entropy loss.

        Returns:
            Tuple[float, torch.nn.Module, list]: Best loss, best model, and training history (if requested).
        """
        best_loss = float("inf")
        best_model = None
        history: list[float] = []
        early_stopping = EarlyStopping(
            patience=self.early_stop_gen,
            min_epochs=self.min_epochs,
            verbose=self.verbose,
            prefix=self.prefix,
            debug=self.debug,
        )

        warm, ramp, gamma_final = 50, 100, self.gamma
        for epoch in range(self.epochs):
            # Gamma schedule
            if epoch < warm:
                model.gamma = 0.0
            elif epoch < warm + ramp:
                model.gamma = gamma_final * ((epoch - warm) / ramp)
            else:
                model.gamma = gamma_final

            train_loss = self._train_step(
                loader=loader,
                optimizer=optimizer,
                model=model,
                l1_penalty=l1_penalty,
                class_weights=class_weights,
            )

            if trial and (np.isnan(train_loss) or np.isinf(train_loss)):
                raise optuna.exceptions.TrialPruned("Loss is NaN or Inf.")

            scheduler.step()
            if return_history:
                history.append(train_loss)

            early_stopping(train_loss, model)
            if early_stopping.early_stop:
                self.logger.info(f"Early stopping at epoch {epoch + 1}.")
                break

        best_loss = early_stopping.best_score
        best_model = copy.deepcopy(early_stopping.best_model)
        return best_loss, best_model, history

    def _train_step(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        l1_penalty: float,
        class_weights: torch.Tensor,
    ) -> float:
        """One epoch (indices, y_int) → one-hot inputs → logits → masked focal CE.

        This method performs a single training epoch, processing batches of data from the DataLoader. It computes the focal cross-entropy loss while ignoring masked (missing) values and applies L1 regularization if specified.

        Args:
            loader (DataLoader): Yields (indices, y_int) where y_int is 0/1/2, -1 for missing.
            optimizer (torch.optim.Optimizer): Optimizer.
            model (torch.nn.Module): Autoencoder model.
            l1_penalty (float): L1 regularization.
            class_weights (torch.Tensor): Class weights for CE.

        Returns:
            float: Mean training loss for the epoch.
        """
        model.train()
        running = 0.0

        for _, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)

            # Inputs: one-hot with zeros for missing; Targets: ints with -1
            x_ohe = self._one_hot_encode_012(y_batch)  # (B, L, K)
            logits = model(x_ohe).view(-1, self.num_features_, self.num_classes_)

            logits_flat = logits.view(-1, self.num_classes_)
            targets_flat = y_batch.view(-1)

            ce = F.cross_entropy(
                logits_flat,
                targets_flat,
                weight=class_weights,
                reduction="none",
                ignore_index=-1,
            )
            pt = torch.exp(-ce)
            gamma = getattr(model, "gamma", self.gamma)
            focal = ((1 - pt) ** gamma) * ce

            valid_mask = targets_flat != -1
            loss = (
                focal[valid_mask].mean()
                if valid_mask.any()
                else torch.tensor(0.0, device=logits.device)
            )

            if l1_penalty > 0:
                loss = loss + l1_penalty * sum(
                    p.abs().sum() for p in model.parameters()
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += float(loss.item())

        return running / len(loader)

    def _predict(
        self,
        model: torch.nn.Module,
        X: np.ndarray | torch.Tensor,
        return_proba: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict 0/1/2 labels (and probabilities) from masked inputs.

        Args:
            model (torch.nn.Module): Trained model.
            X (np.ndarray | torch.Tensor): 0/1/2 matrix with -1
                for missing.
            return_proba (bool): If True, return probabilities.

        Returns:
            Tuple[np.ndarray, np.ndarray] | np.ndarray: Predicted labels,
                and probabilities if requested.
        """
        if model is None:
            msg = "Model is not trained. Call fit() before predict()."
            self.logger.error(msg)
            raise NotFittedError(msg)

        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
            X_tensor = X_tensor.to(self.device).long()
            x_ohe = self._one_hot_encode_012(X_tensor)
            logits = model(x_ohe).view(-1, self.num_features_, self.num_classes_)
            probas = torch.softmax(logits, dim=-1)
            labels = torch.argmax(probas, dim=-1)

        if return_proba:
            return labels.cpu().numpy(), probas.cpu().numpy()

        return labels.cpu().numpy()

    def _evaluate_model(
        self,
        X_val: np.ndarray,
        model: torch.nn.Module,
        params: dict,
        objective_mode: bool = False,
    ) -> Dict[str, float]:
        """Evaluate on 0/1/2; then IUPAC decoding and 10-base integer reports.

        This method evaluates the trained autoencoder model on a validation set, computing various classification metrics based on the predicted and true genotypes. It handles both haploid and diploid data appropriately and generates detailed classification reports for both genotype and IUPAC/10-base integer encodings.

        Args:
            X_val (np.ndarray): Validation set 0/1/2 matrix with -1
                for missing.
            model (torch.nn.Module): Trained model.
            params (dict): Model parameters.
            objective_mode (bool): If True, suppress logging and reports.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        pred_labels, pred_probas = self._predict(
            model=model, X=X_val, return_proba=True
        )

        eval_mask = X_val != -1
        y_true_flat = X_val[eval_mask]
        y_pred_flat = pred_labels[eval_mask]
        y_proba_flat = pred_probas[eval_mask]

        if y_true_flat.size == 0:
            return {self.tune_metric: 0.0}

        labels_for_scoring = [0, 1] if self.is_haploid else [0, 1, 2]
        target_names = ["REF", "ALT"] if self.is_haploid else ["REF", "HET", "ALT"]

        if self.is_haploid:
            y_true_flat = y_true_flat.copy()
            y_pred_flat = y_pred_flat.copy()
            y_true_flat[y_true_flat == 2] = 1
            y_pred_flat[y_pred_flat == 2] = 1
            # collapse probs to 2-class
            proba_2 = np.zeros((len(y_proba_flat), 2), dtype=y_proba_flat.dtype)
            proba_2[:, 0] = y_proba_flat[:, 0]
            proba_2[:, 1] = y_proba_flat[:, 2]
            y_proba_flat = proba_2

        y_true_ohe = np.eye(len(labels_for_scoring))[y_true_flat]

        metrics = self.scorers_.evaluate(
            y_true_flat,
            y_pred_flat,
            y_true_ohe,
            y_proba_flat,
            objective_mode,
            self.tune_metric,
        )

        if not objective_mode:
            self.logger.info(f"Validation Metrics: {metrics}")

            # Primary report (REF/HET/ALT or REF/ALT)
            self._make_class_reports(
                y_true=y_true_flat,
                y_pred_proba=y_proba_flat,
                y_pred=y_pred_flat,
                metrics=metrics,
                labels=target_names,
            )

            # IUPAC decode & 10-base integer report (parity with ImputeNLPCA)
            y_true_dec = self.pgenc.decode_012(X_val)
            X_pred = X_val.copy()
            X_pred[eval_mask] = y_pred_flat
            y_pred_dec = self.pgenc.decode_012(
                X_pred.reshape(X_val.shape[0], self.num_features_)
            )

            encodings_dict = {
                "A": 0,
                "C": 1,
                "G": 2,
                "T": 3,
                "W": 4,
                "R": 5,
                "M": 6,
                "K": 7,
                "Y": 8,
                "S": 9,
                "N": -1,
            }
            y_true_int = self.pgenc.convert_int_iupac(
                y_true_dec, encodings_dict=encodings_dict
            )
            y_pred_int = self.pgenc.convert_int_iupac(
                y_pred_dec, encodings_dict=encodings_dict
            )

            self._make_class_reports(
                y_true=y_true_int[eval_mask],
                y_pred=y_pred_int[eval_mask],
                metrics=metrics,
                y_pred_proba=None,
                labels=["A", "C", "G", "T", "W", "R", "M", "K", "Y", "S"],
            )

        return metrics

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective for AE on 0/1/2 data.

        This method defines the objective function for hyperparameter tuning using Optuna. It samples hyperparameters, trains the autoencoder model on the training set, evaluates it on the validation set, and returns the value of the metric being optimized.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float: Value of the metric being optimized.
        """
        try:
            params = self._sample_hyperparameters(trial)

            X_train = self.ground_truth_[self.train_idx_]
            X_val = self.ground_truth_[self.test_idx_]

            class_weights = self._class_weights_from_zygosity(X_train).to(self.device)
            train_loader = self._get_data_loaders(X_train)

            model = self.build_model(self.Model, params["model_params"])
            model.apply(self.initialize_weights)

            _, model, _ = self._train_and_validate_model(
                model=model,
                loader=train_loader,
                lr=params["lr"],
                l1_penalty=params["l1_penalty"],
                trial=trial,
                return_history=False,
                class_weights=class_weights,
            )

            metrics = self._evaluate_model(X_val, model, params, objective_mode=True)
            self._clear_resources(model, train_loader)
            return metrics[self.tune_metric]
        except Exception as e:
            raise optuna.exceptions.TrialPruned(f"Trial failed with error: {e}")

    def _sample_hyperparameters(
        self, trial: optuna.Trial
    ) -> Dict[str, int | float | str]:
        """Sample AE hyperparameters and compute hidden sizes for model params.

        This method samples hyperparameters for the autoencoder model using Optuna's trial object. It computes the hidden layer sizes based on the sampled parameters and prepares the model parameters dictionary.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            Dict[str, int | float | str]: Sampled hyperparameters and model_params.
        """
        params = {
            "latent_dim": trial.suggest_int("latent_dim", 2, 64),
            "lr": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.6),
            "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 8),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "elu", "selu"]
            ),
            "l1_penalty": trial.suggest_float("l1_penalty", 1e-7, 1e-2, log=True),
            "layer_scaling_factor": trial.suggest_float(
                "layer_scaling_factor", 2.0, 10.0
            ),
            "layer_schedule": trial.suggest_categorical(
                "layer_schedule", ["pyramid", "constant", "linear"]
            ),
        }

        input_dim = self.num_features_ * self.num_classes_
        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=input_dim,
            n_outputs=input_dim,
            n_samples=len(self.train_idx_),
            n_hidden=params["num_hidden_layers"],
            alpha=params["layer_scaling_factor"],
            schedule=params["layer_schedule"],
        )

        params["model_params"] = {
            "n_features": self.num_features_,
            "num_classes": self.num_classes_,
            "latent_dim": params["latent_dim"],
            "dropout_rate": params["dropout_rate"],
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": params["activation"],
        }
        return params

    def _set_best_params(
        self, best_params: Dict[str, int | float | str | list]
    ) -> Dict[str, int | float | str | list]:
        """Adopt best params (ImputeNLPCA parity) and return model_params.

        This method sets the best hyperparameters found during tuning and computes the hidden layer sizes for the autoencoder model. It prepares the final model parameters dictionary to be used for building the model.

        Args:
            best_params (Dict[str, int | float | str | list]): Best hyperparameters from tuning.

        Returns:
            Dict[str, int | float | str | list]: Model parameters for building the model.
        """
        self.latent_dim = best_params["latent_dim"]
        self.dropout_rate = best_params["dropout_rate"]
        self.learning_rate = best_params["learning_rate"]
        self.l1_penalty = best_params["l1_penalty"]
        self.activation = best_params["activation"]
        self.layer_scaling_factor = best_params["layer_scaling_factor"]
        self.layer_schedule = best_params["layer_schedule"]

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=self.num_features_ * self.num_classes_,
            n_outputs=self.num_features_ * self.num_classes_,
            n_samples=len(self.train_idx_),
            n_hidden=best_params["num_hidden_layers"],
            alpha=best_params["layer_scaling_factor"],
            schedule=best_params["layer_schedule"],
        )

        return {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": hidden_layer_sizes,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "num_classes": self.num_classes_,
        }

    def _default_best_params(self) -> Dict[str, int | float | str | list]:
        """Default model params when tuning is disabled.

        This method computes the default model parameters for the autoencoder when hyperparameter tuning is not performed. It calculates the hidden layer sizes based on the initial configuration.

        Returns:
            Dict[str, int | float | str | list]: Default model parameters.
        """
        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=self.num_features_ * self.num_classes_,
            n_outputs=self.num_features_ * self.num_classes_,
            n_samples=len(self.ground_truth_),
            n_hidden=self.num_hidden_layers,
            alpha=self.layer_scaling_factor,
            schedule=self.layer_schedule,
        )
        return {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": hidden_layer_sizes,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "num_classes": self.num_classes_,
        }
