import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

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

from pgsui.data_processing.config import apply_dot_overrides, load_yaml_to_dataclass
from pgsui.data_processing.containers import AutoencoderConfig
from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.models.autoencoder_model import AutoencoderModel

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


def ensure_autoencoder_config(
    config: AutoencoderConfig | dict | str | None,
) -> AutoencoderConfig:
    """Return a concrete AutoencoderConfig from dataclass, dict, YAML path, or None.

    This method normalizes the configuration input for the Autoencoder imputer. It accepts a structured configuration in various formats, including a dataclass instance, a nested dictionary, a YAML file path, or None. The method processes the input accordingly and returns a concrete instance of AutoencoderConfig with all necessary fields populated.

    Args:
        config (AutoencoderConfig | dict | str | None): Structured configuration as dataclass, nested dict, YAML path, or None.

    Returns:
        AutoencoderConfig: Concrete configuration instance.
    """
    if config is None:
        return AutoencoderConfig()
    if isinstance(config, AutoencoderConfig):
        return config
    if isinstance(config, str):
        # YAML path — top-level `preset` key is supported
        return load_yaml_to_dataclass(
            config, AutoencoderConfig, preset_builder=AutoencoderConfig.from_preset
        )
    if isinstance(config, dict):
        # Flatten dict into dot-keys then overlay onto a fresh instance
        base = AutoencoderConfig()

        def _flatten(prefix: str, d: dict, out: dict) -> dict:
            for k, v in d.items():
                kk = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(kk, v, out)
                else:
                    out[kk] = v
            return out

        # Lift any present preset first
        preset_name = config.pop("preset", None)
        if "io" in config and isinstance(config["io"], dict):
            preset_name = preset_name or config["io"].pop("preset", None)

        if preset_name:
            base = AutoencoderConfig.from_preset(preset_name)

        flat = _flatten("", config, {})
        return apply_dot_overrides(base, flat)

    raise TypeError("config must be an AutoencoderConfig, dict, YAML path, or None.")


class ImputeAutoencoder(BaseNNImputer):
    """Impute missing genotypes with a standard Autoencoder on 0/1/2 encodings.

    This imputer uses a feedforward autoencoder architecture to learn compressed and reconstructive representations of genotype data encoded as 0 (homozygous reference), 1 (heterozygous), and 2 (homozygous alternate). Missing genotypes are represented as -1 during training and imputation.

    The model is trained to minimize a focal cross-entropy loss, which helps to address class imbalance by focusing more on hard-to-classify examples. The architecture includes configurable parameters such as the number of hidden layers, latent dimension size, dropout rate, and activation functions.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        config: Optional[Union["AutoencoderConfig", dict, str]] = None,
        overrides: dict | None = None,
    ) -> None:
        """Initialize the Autoencoder imputer with a unified config interface.

        This initializer sets up the Autoencoder imputer by processing the provided configuration, initializing logging, and preparing the model and data encoder. It supports configuration input as a dataclass, nested dictionary, YAML file path, or None, with optional dot-key overrides for fine-tuning specific parameters.

        Args:
            genotype_data: Backing genotype data object.
            config: Structured configuration as dataclass, nested dict, YAML path, or None.
            overrides: Optional dot-key overrides with highest precedence (e.g., {'model.latent_dim': 32}).
        """
        self.model_name = "ImputeAutoencoder"
        self.genotype_data = genotype_data

        # Normalize config then apply highest-precedence overrides
        cfg = ensure_autoencoder_config(config)
        if overrides:
            cfg = apply_dot_overrides(cfg, overrides)
        self.cfg = cfg

        # Logger consistent with NLPCA
        logman = LoggerManager(
            __name__,
            prefix=self.cfg.io.prefix,
            debug=self.cfg.io.debug,
            verbose=self.cfg.io.verbose,
        )
        self.logger = logman.get_logger()

        # BaseNNImputer bootstrapping (device/dirs/logging handled here)
        super().__init__(
            prefix=self.cfg.io.prefix,
            device=self.cfg.train.device,
            verbose=self.cfg.io.verbose,
            debug=self.cfg.io.debug,
        )

        # Model hook & encoder
        self.Model = AutoencoderModel
        self.pgenc = GenotypeEncoder(genotype_data)

        # IO / global
        self.seed = self.cfg.io.seed
        self.n_jobs = self.cfg.io.n_jobs
        self.prefix = self.cfg.io.prefix
        self.scoring_averaging = self.cfg.io.scoring_averaging
        self.verbose = self.cfg.io.verbose
        self.debug = self.cfg.io.debug
        self.rng = np.random.default_rng(self.seed)

        # Model hyperparams
        self.latent_dim = self.cfg.model.latent_dim
        self.dropout_rate = self.cfg.model.dropout_rate
        self.num_hidden_layers = self.cfg.model.num_hidden_layers
        self.layer_scaling_factor = self.cfg.model.layer_scaling_factor
        self.layer_schedule = self.cfg.model.layer_schedule
        self.activation = self.cfg.model.hidden_activation
        self.gamma = self.cfg.model.gamma

        # Train hyperparams
        self.batch_size = self.cfg.train.batch_size
        self.learning_rate = self.cfg.train.learning_rate
        self.l1_penalty = self.cfg.train.l1_penalty
        self.early_stop_gen = self.cfg.train.early_stop_gen
        self.min_epochs = self.cfg.train.min_epochs
        self.epochs = self.cfg.train.max_epochs
        self.validation_split = self.cfg.train.validation_split
        self.beta = self.cfg.train.weights_beta
        self.max_ratio = self.cfg.train.weights_max_ratio

        # Tuning
        self.tune = self.cfg.tune.enabled
        self.tune_fast = self.cfg.tune.fast
        self.tune_batch_size = self.cfg.tune.batch_size
        self.tune_epochs = self.cfg.tune.epochs
        self.tune_eval_interval = self.cfg.tune.eval_interval
        self.tune_metric = self.cfg.tune.metric
        self.n_trials = self.cfg.tune.n_trials
        self.tune_save_db = self.cfg.tune.save_db
        self.tune_resume = self.cfg.tune.resume
        self.tune_max_samples = self.cfg.tune.max_samples
        self.tune_max_loci = self.cfg.tune.max_loci
        self.tune_infer_epochs = getattr(self.cfg.tune, "infer_epochs", 0)  # AE unused
        self.tune_patience = self.cfg.tune.patience

        # Evaluate
        # AE does not optimize latents, so these are unused / fixed
        self.eval_latent_steps = 0
        self.eval_latent_lr = 0.0
        self.eval_latent_weight_decay = 0.0

        # Plotting (parity with NLPCA PlotConfig)
        self.plot_format = self.cfg.plot.fmt
        self.plot_dpi = self.cfg.plot.dpi
        self.plot_fontsize = self.cfg.plot.fontsize
        self.title_fontsize = self.cfg.plot.fontsize
        self.despine = self.cfg.plot.despine
        self.show_plots = self.cfg.plot.show

        # Core derived at fit-time
        self.is_haploid: bool | None = None
        self.num_classes_: int | None = None
        self.model_params: Dict[str, Any] = {}

    def fit(self) -> "ImputeAutoencoder":
        """Fit the autoencoder on 0/1/2 encoded genotypes (missing → -9).

        This method trains the autoencoder model using the provided genotype data. It prepares the data by encoding genotypes as 0, 1, and 2, with missing values represented as -9. The method splits the data into training and validation sets, initializes the model and training parameters, and performs training with optional hyperparameter tuning. After training, it evaluates the model on the validation set and stores the fitted model and training history.

        Returns:
            ImputeAutoencoder: Fitted instance.

        Raises:
            NotFittedError: If training fails.
        """
        self.logger.info(f"Fitting {self.model_name} (0/1/2 AE) ...")

        # --- Data prep (mirror NLPCA) ---
        X = self.pgenc.genotypes_012.astype(np.float32)
        X[X < 0] = np.nan
        X[np.isnan(X)] = -1
        self.ground_truth_ = X.astype(np.int64)

        # Ploidy & classes
        self.is_haploid = np.all(
            np.isin(
                self.genotype_data.snp_data,
                ["A", "C", "G", "T", "N", "-", ".", "?"],
            )
        )
        self.ploidy = 1 if self.is_haploid else 2
        self.num_classes_ = 2 if self.is_haploid else 3
        self.logger.info(
            f"Data is {'haploid' if self.is_haploid else 'diploid'}; "
            f"using {self.num_classes_} classes."
        )

        n_samples, self.num_features_ = X.shape

        # Model params (decoder outputs L * K logits)
        self.model_params = {
            "n_features": self.num_features_,
            "num_classes": self.num_classes_,
            "latent_dim": self.latent_dim,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
        }

        # Train/Val split
        indices = np.arange(n_samples)
        train_idx, val_idx = train_test_split(
            indices, test_size=self.validation_split, random_state=self.seed
        )
        self.train_idx_, self.test_idx_ = train_idx, val_idx
        self.X_train_ = self.ground_truth_[train_idx]
        self.X_val_ = self.ground_truth_[val_idx]

        # Plotters/scorers (shared utilities)
        self.plotter_, self.scorers_ = self.initialize_plotting_and_scorers()

        # Tuning (optional; AE never needs latent refinement)
        if self.tune:
            self.tune_hyperparameters()

        # Best params (tuned or default)
        self.best_params_ = getattr(self, "best_params_", self._default_best_params())

        # Class weights (device-aware)
        self.class_weights_ = self._class_weights_from_zygosity(self.X_train_).to(
            self.device
        )

        # DataLoader
        train_loader = self._get_data_loaders(self.X_train_)

        # Build & train
        model = self.build_model(self.Model, self.best_params_)
        model.apply(self.initialize_weights)

        loss, trained_model, history = self._train_and_validate_model(
            model=model,
            loader=train_loader,
            lr=self.learning_rate,
            l1_penalty=self.l1_penalty,
            return_history=True,
            class_weights=self.class_weights_,
            X_val=self.X_val_,
            params=self.best_params_,
            prune_metric=self.tune_metric,
            prune_warmup_epochs=5,
            eval_interval=1,
            eval_requires_latents=False,
            eval_latent_steps=0,
            eval_latent_lr=0.0,
            eval_latent_weight_decay=0.0,
        )

        if trained_model is None:
            msg = "Autoencoder training failed; no model was returned."
            self.logger.error(msg)
            raise RuntimeError(msg)

        torch.save(
            trained_model.state_dict(),
            self.models_dir / f"final_model_{self.model_name}.pt",
        )

        self.best_loss_, self.model_, self.history_ = (
            loss,
            trained_model,
            {"Train": history},
        )
        self.is_fit_ = True

        # Evaluate on validation set (parity with NLPCA reporting)
        self._evaluate_model(self.X_val_, self.model_, self.best_params_)
        self.plotter_.plot_history(self.history_)
        return self

    def transform(self) -> np.ndarray:
        """Impute missing genotypes (0/1/2) and return IUPAC strings.

        This method imputes missing genotypes in the dataset using the trained autoencoder model. It predicts the most likely genotype (0, 1, or 2) for each missing entry and fills in these values. The imputed genotypes are then decoded back to IUPAC string format for easier interpretation.

        Returns:
            np.ndarray: IUPAC strings of shape (n_samples, n_loci).

        Raises:
            NotFittedError: If called before fit().
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

        # Decode to IUPAC & plot
        imputed_genotypes = self.pgenc.decode_012(imputed_array)
        original_genotypes = self.pgenc.decode_012(X_to_impute)

        plt.rcParams.update(self.plotter_.param_dict)
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
        *,
        X_val: np.ndarray | None = None,
        params: dict | None = None,
        prune_metric: str | None = None,  # "f1" | "accuracy" | "pr_macro"
        prune_warmup_epochs: int = 3,
        eval_interval: int = 1,
        # Evaluation parameters (AE ignores latent refinement knobs)
        eval_requires_latents: bool = False,  # AE: always False
        eval_latent_steps: int = 0,
        eval_latent_lr: float = 0.0,
        eval_latent_weight_decay: float = 0.0,
    ) -> Tuple[float, torch.nn.Module | None, list | None]:
        """Wrap the AE training loop (no latent optimizer), with Optuna pruning.

        This method orchestrates the training of the autoencoder model using the provided DataLoader. It sets up the optimizer and learning rate scheduler, and executes the training loop with support for early stopping and Optuna pruning based on validation performance. The method returns the best validation loss, the best model state, and optionally the training history.

        Args:
            model (torch.nn.Module): Autoencoder model.
            loader (torch.utils.data.DataLoader): Batches (indices, y_int) where y_int is 0/1/2; -1 for missing.
            lr (float): Learning rate.
            l1_penalty (float): L1 regularization coeff.
            trial (optuna.Trial | None): Optuna trial for pruning (optional).
            return_history (bool): If True, return train loss history.
            class_weights (torch.Tensor | None): Class weights tensor (on device).
            X_val (np.ndarray | None): Validation matrix (0/1/2 with -1 for missing).
            params (dict | None): Model params for evaluation.
            prune_metric (str | None): Metric for pruning reports.
            prune_warmup_epochs (int): Pruning warmup epochs.
            eval_interval (int): Eval frequency (epochs).
            eval_requires_latents (bool): Ignored for AE (no latent inference).
            eval_latent_steps (int): Unused for AE.
            eval_latent_lr (float): Unused for AE.
            eval_latent_weight_decay (float): Unused for AE.

        Returns:
            Tuple[float, torch.nn.Module | None, list | None]: (best_loss, best_model, history or None).
        """
        if class_weights is None:
            msg = "Must provide class_weights."
            self.logger.error(msg)
            raise TypeError(msg)

        # Epoch budget mirrors NLPCA config (tuning vs final)
        max_epochs = (
            self.tune_epochs if (trial is not None and self.tune_fast) else self.epochs
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

        best_loss, best_model, hist = self._execute_training_loop(
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            l1_penalty=l1_penalty,
            trial=trial,
            return_history=return_history,
            class_weights=class_weights,
            X_val=X_val,
            params=params,
            prune_metric=prune_metric,
            prune_warmup_epochs=prune_warmup_epochs,
            eval_interval=eval_interval,
            eval_requires_latents=False,  # AE: no latent inference
            eval_latent_steps=0,
            eval_latent_lr=0.0,
            eval_latent_weight_decay=0.0,
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
        trial: optuna.Trial | None,
        return_history: bool,
        class_weights: torch.Tensor,
        *,
        X_val: np.ndarray | None = None,
        params: dict | None = None,
        prune_metric: str | None = None,
        prune_warmup_epochs: int = 3,
        eval_interval: int = 1,
        # Evaluation parameters (AE ignores latent refinement knobs)
        eval_requires_latents: bool = False,  # AE: False
        eval_latent_steps: int = 0,
        eval_latent_lr: float = 0.0,
        eval_latent_weight_decay: float = 0.0,
    ) -> Tuple[float, torch.nn.Module, list]:
        """Train AE with focal CE (gamma warm/ramp) + early stopping & pruning.

        This method executes the training loop for the autoencoder model, performing one epoch at a time. It computes the focal cross-entropy loss while ignoring masked (missing) values and applies L1 regularization if specified. The method incorporates early stopping based on validation performance and supports Optuna pruning to terminate unpromising trials early. It returns the best validation loss, the best model state, and optionally the training history.

        Args:
            loader (torch.utils.data.DataLoader): Batches (indices, y_int) where y_int is 0/1/2; -1 for missing.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler.
            model (torch.nn.Module): Autoencoder model.
            l1_penalty (float): L1 regularization coeff.
            trial (optuna.Trial | None): Optuna trial for pruning (optional).
            return_history (bool): If True, return train loss history.
            class_weights (torch.Tensor): Class weights tensor (on device).
            X_val (np.ndarray | None): Validation matrix (0/1/2 with -1 for missing).
            params (dict | None): Model params for evaluation.
            prune_metric (str | None): Metric for pruning reports.
            prune_warmup_epochs (int): Pruning warmup epochs.
            eval_interval (int): Eval frequency (epochs).
            eval_requires_latents (bool): Ignored for AE (no latent inference).
            eval_latent_steps (int): Unused for AE.
            eval_latent_lr (float): Unused for AE.
            eval_latent_weight_decay (float): Unused for AE.

        Returns:
            Tuple[float, torch.nn.Module, list]: Best validation loss, best model, and training history.
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

        # Parity with NLPCA (warm/ramp gamma schedule)
        warm, ramp, gamma_final = 50, 100, self.gamma

        # Epoch budget mirrors the caller's scheduler T_max
        # (already set to tune_epochs or epochs).
        for epoch in range(scheduler.T_max):
            # Gamma schedule
            if epoch < warm:
                model.gamma = 0.0
            elif epoch < warm + ramp:
                model.gamma = gamma_final * ((epoch - warm) / ramp)
            else:
                model.gamma = gamma_final

            # ---- one epoch ----
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

            # Optuna report/prune on validation metric
            if (
                trial is not None
                and X_val is not None
                and ((epoch + 1) % eval_interval == 0)
            ):
                metric_key = prune_metric or getattr(self, "tune_metric", "f1")
                metric_val = self._eval_for_pruning(
                    model=model,
                    X_val=X_val,
                    params=params or getattr(self, "best_params_", {}),
                    metric=metric_key,
                    objective_mode=True,
                    do_latent_infer=False,  # AE: False
                    latent_steps=0,
                    latent_lr=0.0,
                    latent_weight_decay=0.0,
                    latent_seed=(self.seed if self.seed is not None else 123),
                    _latent_cache=None,  # AE: not used
                    _latent_cache_key=None,
                )
                trial.report(metric_val, step=epoch + 1)
                if (epoch + 1) >= prune_warmup_epochs and trial.should_prune():
                    raise optuna.exceptions.TrialPruned(
                        f"Pruned at epoch {epoch + 1}: {metric_key}={metric_val:.5f}"
                    )

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

        This method generates predictions from the trained autoencoder model for the provided input data. It processes the input data, performs a forward pass through the model, and computes the predicted genotype labels (0, 1, or 2) along with their associated probabilities if requested.

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
        latent_vectors_val: Optional[np.ndarray] = None,
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

        # mask out true missing AND any non-finite prob rows
        finite_mask = np.all(np.isfinite(pred_probas), axis=-1)  # (N,L)
        eval_mask = (X_val != -1) & finite_mask

        y_true_flat = X_val[eval_mask].astype(np.int64, copy=False)
        y_pred_flat = pred_labels[eval_mask].astype(np.int64, copy=False)
        y_proba_flat = pred_probas[eval_mask].astype(np.float64, copy=False)

        if y_true_flat.size == 0:
            return {self.tune_metric: 0.0}

        # ensure valid probability simplex after masking (no NaNs/Infs, sums=1)
        y_proba_flat = np.clip(y_proba_flat, 0.0, 1.0)
        row_sums = y_proba_flat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        y_proba_flat = y_proba_flat / row_sums

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
        """Optuna objective for AE; mirrors NLPCA study driver without latents.

        This method defines the objective function for hyperparameter tuning using Optuna. It samples hyperparameters, prepares the training and validation data, builds and trains the autoencoder model, and evaluates its performance on the validation set. The method returns the value of the tuning metric to be maximized.

        Args:
            trial (optuna.Trial): Optuna trial.

        Returns:
            float: Value of the tuning metric (maximize).
        """
        try:
            # Sample hyperparameters (existing helper; unchanged signature)
            params = self._sample_hyperparameters(trial)

            # Optionally sub-sample for fast tuning (same keys used by NLPCA if you adopt them)
            X_train = self.ground_truth_[self.train_idx_]
            X_val = self.ground_truth_[self.test_idx_]

            class_weights = self._class_weights_from_zygosity(X_train).to(self.device)
            train_loader = self._get_data_loaders(X_train)

            model = self.build_model(self.Model, params["model_params"])
            model.apply(self.initialize_weights)

            # Train + prune on metric
            _, model, _ = self._train_and_validate_model(
                model=model,
                loader=train_loader,
                lr=params["lr"],
                l1_penalty=params["l1_penalty"],
                trial=trial,
                return_history=False,
                class_weights=class_weights,
                X_val=X_val,
                params=params,
                prune_metric=self.tune_metric,
                prune_warmup_epochs=5,
                eval_interval=self.tune_eval_interval,
                eval_requires_latents=False,
                eval_latent_steps=0,
                eval_latent_lr=0.0,
                eval_latent_weight_decay=0.0,
            )

            metrics = self._evaluate_model(X_val, model, params, objective_mode=True)
            self._clear_resources(model, train_loader)
            return metrics[self.tune_metric]

        except Exception as e:
            # Keep sweeps moving if a trial fails
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

        # Keep the latent_dim as the first element,
        # then the interior hidden widths.
        # If there are no interior widths (very small nets),
        # this still leaves [latent_dim].
        hidden_only = [hidden_layer_sizes[0]] + hidden_layer_sizes[1:-1]

        params["model_params"] = {
            "n_features": self.num_features_,
            "num_classes": self.num_classes_,
            "latent_dim": params["latent_dim"],
            "dropout_rate": params["dropout_rate"],
            "hidden_layer_sizes": hidden_only,
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

        # Keep the latent_dim as the first element,
        # then the interior hidden widths.
        # If there are no interior widths (very small nets),
        # this still leaves [latent_dim].
        hidden_only = [hidden_layer_sizes[0]] + hidden_layer_sizes[1:-1]

        return {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": hidden_only,
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
