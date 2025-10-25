from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.utils.logging import LoggerManager
from torch.optim.lr_scheduler import CosineAnnealingLR

from pgsui.data_processing.config import apply_dot_overrides, load_yaml_to_dataclass
from pgsui.data_processing.containers import VAEConfig
from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.models.vae_model import VAEModel

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


def ensure_vae_config(config: Union[VAEConfig, dict, str, None]) -> VAEConfig:
    """Normalize VAEConfig input from various sources.

    Args:
        config (Union[VAEConfig, dict, str, None]): VAEConfig, nested dict, YAML path, or None (defaults).

    Returns:
        VAEConfig: Normalized configuration dataclass.
    """
    if config is None:
        return VAEConfig()
    if isinstance(config, VAEConfig):
        return config
    if isinstance(config, str):
        return load_yaml_to_dataclass(
            config, VAEConfig, preset_builder=VAEConfig.from_preset
        )
    if isinstance(config, dict):
        base = VAEConfig()
        # Respect top-level preset
        preset = config.pop("preset", None)
        if preset:
            base = VAEConfig.from_preset(preset)
        # Flatten + apply
        flat: Dict[str, object] = {}

        def _flatten(prefix: str, d: dict, out: dict) -> dict:
            for k, v in d.items():
                kk = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(kk, v, out)
                else:
                    out[kk] = v
            return out

        flat = _flatten("", config, {})
        return apply_dot_overrides(base, flat)
    raise TypeError("config must be a VAEConfig, dict, YAML path, or None.")


class ImputeVAE(BaseNNImputer):
    """Variational Autoencoder imputer on 0/1/2 encodings (missing=-1).

    This imputer implements a VAE with a multinomial (categorical) latent space. It is designed to handle missing data by inferring the latent distribution and generating plausible predictions. The model is trained using a combination of reconstruction loss (cross-entropy) and a KL divergence term, with the KL weight (beta) annealed over time. The imputer supports both haploid and diploid genotype data.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        config: Optional[Union["VAEConfig", dict, str]] = None,
        overrides: dict | None = None,
    ):
        """Initialize the VAE imputer with a unified config interface.

        This initializer sets up the VAE imputer by processing the provided configuration, initializing logging, and preparing the model and data encoder. It supports configuration input as a dataclass, nested dictionary, YAML file path, or None, with optional dot-key overrides for fine-tuning specific parameters.

        Args:
            genotype_data (GenotypeData): Backing genotype data object.
            config (Union[VAEConfig, dict, str, None]): VAEConfig, nested dict, YAML path, or None (defaults).
            overrides (dict | None): Optional dot-key overrides with highest precedence.
        """
        self.model_name = "ImputeVAE"
        self.genotype_data = genotype_data

        # Normalize configuration and apply top-precedence overrides
        cfg = ensure_vae_config(config)
        if overrides:
            cfg = apply_dot_overrides(cfg, overrides)
        self.cfg = cfg

        # Logger (align with AE/NLPCA)
        logman = LoggerManager(
            __name__,
            prefix=self.cfg.io.prefix,
            debug=self.cfg.io.debug,
            verbose=self.cfg.io.verbose,
        )
        self.logger = logman.get_logger()

        # BaseNNImputer bootstraps device/dirs/log formatting
        super().__init__(
            prefix=self.cfg.io.prefix,
            device=self.cfg.train.device,
            verbose=self.cfg.io.verbose,
            debug=self.cfg.io.debug,
        )

        # Model hook & encoder
        self.Model = VAEModel
        self.pgenc = GenotypeEncoder(genotype_data)

        # IO/global
        self.seed = self.cfg.io.seed
        self.n_jobs = self.cfg.io.n_jobs
        self.prefix = self.cfg.io.prefix
        self.scoring_averaging = self.cfg.io.scoring_averaging
        self.verbose = self.cfg.io.verbose
        self.debug = self.cfg.io.debug
        self.rng = np.random.default_rng(self.seed)

        # Model hyperparams (AE-parity)
        self.latent_dim = self.cfg.model.latent_dim
        self.dropout_rate = self.cfg.model.dropout_rate
        self.num_hidden_layers = self.cfg.model.num_hidden_layers
        self.layer_scaling_factor = self.cfg.model.layer_scaling_factor
        self.layer_schedule = self.cfg.model.layer_schedule
        self.activation = self.cfg.model.hidden_activation
        self.gamma = self.cfg.model.gamma  # focal loss focusing (for recon CE)

        # VAE-only KL controls
        self.kl_beta_final = self.cfg.vae.kl_beta
        self.kl_warmup = self.cfg.vae.kl_warmup
        self.kl_ramp = self.cfg.vae.kl_ramp

        # Train hyperparams (AE-parity)
        self.batch_size = self.cfg.train.batch_size
        self.learning_rate = self.cfg.train.learning_rate
        self.l1_penalty = self.cfg.train.l1_penalty
        self.early_stop_gen = self.cfg.train.early_stop_gen
        self.min_epochs = self.cfg.train.min_epochs
        self.epochs = self.cfg.train.max_epochs
        self.validation_split = self.cfg.train.validation_split
        self.beta = self.cfg.train.weights_beta
        self.max_ratio = self.cfg.train.weights_max_ratio

        # Tuning (AE-parity surface; VAE ignores latent refinement during eval)
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
        self.tune_patience = self.cfg.tune.patience

        # Plotting (AE-parity)
        self.plot_format = self.cfg.plot.fmt
        self.plot_dpi = self.cfg.plot.dpi
        self.plot_fontsize = self.cfg.plot.fontsize
        self.title_fontsize = self.cfg.plot.fontsize
        self.despine = self.cfg.plot.despine
        self.show_plots = self.cfg.plot.show

        # Derived at fit-time
        self.is_haploid: bool | None = None
        self.num_classes_: int | None = None
        self.model_params: Dict[str, Any] = {}

    # -------------------- Fit -------------------- #
    def fit(self) -> "ImputeVAE":
        """Fit the VAE on 0/1/2 encoded genotypes (missing → -9).

        This method prepares the genotype data, initializes model parameters, splits the data into training and validation sets, and trains the VAE model. It handles both haploid and diploid data, applies class weighting, and supports optional hyperparameter tuning. After training, it evaluates the model on the validation set and saves the trained model.

        Returns:
            ImputeVAE: Fitted instance.

        Raises:
            RuntimeError: If training fails to produce a model.
        """
        self.logger.info(f"Fitting {self.model_name} (0/1/2 VAE) ...")

        # Data prep aligns with AE/NLPCA
        X = self.pgenc.genotypes_012.astype(np.float32)
        X[X < 0] = np.nan
        X[np.isnan(X)] = -1
        self.ground_truth_ = X.astype(np.int64)

        # Ploidy/classes
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

        # Model params (decoder outputs L*K logits)
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

        # Optional tuning
        if self.tune:
            self.tune_hyperparameters()

        # Best params (tuned or default)
        self.best_params_ = getattr(self, "best_params_", self._default_best_params())

        # Class weights (device-aware)
        self.class_weights_ = self._class_weights_from_zygosity(self.X_train_).to(
            self.device
        )

        # DataLoader
        train_loader = self._get_data_loader(self.X_train_)

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
            eval_requires_latents=False,  # no latent refinement for eval
            eval_latent_steps=0,
            eval_latent_lr=0.0,
            eval_latent_weight_decay=0.0,
        )

        if trained_model is None:
            msg = "VAE training failed; no model was returned."
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

        # Evaluate (AE-parity reporting)
        self._evaluate_model(self.X_val_, self.model_, self.best_params_)
        self.plotter_.plot_history(self.history_)
        self._save_best_params(self.best_params_)
        return self

    def transform(self) -> np.ndarray:
        """Impute missing genotypes and return IUPAC strings.

        This method uses the trained VAE model to impute missing genotypes in the dataset. It predicts the most likely genotype for each missing entry based on the learned latent representations and fills in these values. The imputed genotypes are then decoded back to IUPAC string format for easy interpretation.

        Returns:
            np.ndarray: IUPAC strings of shape (n_samples, n_loci).

        Raises:
            NotFittedError: If called before fit().
        """
        if not getattr(self, "is_fit_", False):
            raise NotFittedError("Model is not fitted. Call fit() before transform().")

        self.logger.info("Imputing entire dataset with VAE (0/1/2)...")
        X_to_impute = self.ground_truth_.copy()

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

    # ---------- plumbing identical to AE, naming aligned ---------- #

    def _get_data_loader(self, y: np.ndarray) -> torch.utils.data.DataLoader:
        """Create DataLoader over indices + integer targets (-1 for missing).

        This method creates a PyTorch DataLoader for the training data. It converts the input genotype matrix into a tensor and constructs a dataset that includes both the indices and the genotype values. The DataLoader is configured to shuffle the data and use the specified batch size for training.

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
        eval_requires_latents: bool = False,  # VAE: no latent eval refinement
        eval_latent_steps: int = 0,
        eval_latent_lr: float = 0.0,
        eval_latent_weight_decay: float = 0.0,
    ) -> Tuple[float, torch.nn.Module | None, list | None]:
        """Wrap the VAE training loop with β-anneal & Optuna pruning.

        This method orchestrates the training of the VAE model, including setting up the optimizer and learning rate scheduler, and executing the training loop with support for early stopping and Optuna pruning. It manages the training process, monitors performance on a validation set if provided, and returns the best model and training history.

        Args:
            model (torch.nn.Module): VAE model.
            loader (torch.utils.data.DataLoader): Training data loader.
            lr (float): Learning rate.
            l1_penalty (float): L1 regularization coefficient.
            trial (optuna.Trial | None): Optuna trial for pruning.
            return_history (bool): If True, return training history.
            class_weights (torch.Tensor | None): CE class weights on device.
            X_val (np.ndarray | None): Validation data for pruning eval.
            params (dict | None): Current hyperparameters (for logging).
            prune_metric (str | None): Metric for pruning decisions.
            prune_warmup_epochs (int): Epochs to skip before pruning.
            eval_interval (int): Epochs between validation evaluations.
            eval_requires_latents (bool): If True, refine latents during eval.
            eval_latent_steps (int): Latent refinement steps if needed.
            eval_latent_lr (float): Latent refinement learning rate.
            eval_latent_weight_decay (float): Latent refinement L2 penalty.

        Returns:
            Tuple[float, torch.nn.Module | None, list | None]: Best loss, best model, and training history (if requested).
        """
        if class_weights is None:
            msg = "Must provide class_weights."
            self.logger.error(msg)
            raise TypeError(msg)

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
            eval_requires_latents=eval_requires_latents,
            eval_latent_steps=eval_latent_steps,
            eval_latent_lr=eval_latent_lr,
            eval_latent_weight_decay=eval_latent_weight_decay,
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
        eval_requires_latents: bool = False,
        eval_latent_steps: int = 0,
        eval_latent_lr: float = 0.0,
        eval_latent_weight_decay: float = 0.0,
    ) -> Tuple[float, torch.nn.Module, list]:
        """Train VAE with focal CE + KL(β) anneal, early stopping & pruning.

        This method implements the core training loop for the VAE model, incorporating focal cross-entropy loss for reconstruction and KL divergence with an annealed beta weight. It includes mechanisms for early stopping based on validation performance and supports pruning of unpromising trials when used with Optuna. The training process is monitored, and the best model is retained.

        Args:
            loader (torch.utils.data.DataLoader): Training data loader.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler.
            model (torch.nn.Module): VAE model.
            l1_penalty (float): L1 regularization coefficient.
            trial (optuna.Trial | None): Optuna trial for pruning.
            return_history (bool): If True, return training history.
            class_weights (torch.Tensor): CE class weights on device.
            X_val (np.ndarray | None): Validation data for pruning eval.
            params (dict | None): Current hyperparameters (for logging).
            prune_metric (str | None): Metric for pruning decisions.
            prune_warmup_epochs (int): Epochs to skip before pruning.
            eval_interval (int): Epochs between validation evaluations.
            eval_requires_latents (bool): If True, refine latents during eval.
            eval_latent_steps (int): Latent refinement steps if needed.
            eval_latent_lr (float): Latent refinement learning rate.
            eval_latent_weight_decay (float): Latent refinement L2 penalty.

        Returns:
            Tuple[float, torch.nn.Module, list[float]]: Best loss, best model, history.
        """
        best_model = None
        history: list[float] = []

        early_stopping = EarlyStopping(
            patience=self.early_stop_gen,
            min_epochs=self.min_epochs,
            verbose=self.verbose,
            prefix=self.prefix,
            debug=self.debug,
        )

        # AE-parity gamma schedule for focal CE (reconstruction)
        gamma_warm, gamma_ramp, gamma_final = 50, 100, self.gamma
        # VAE β schedule for KL term
        beta_warm, beta_ramp, beta_final = (
            self.kl_warmup,
            self.kl_ramp,
            self.kl_beta_final,
        )

        for epoch in range(scheduler.T_max):
            # schedules
            # focal γ schedule (if your VAEModel uses it for recon CE)
            if epoch < gamma_warm:
                model.gamma = 0.0
            elif epoch < gamma_warm + gamma_ramp:
                model.gamma = gamma_final * ((epoch - gamma_warm) / gamma_ramp)
            else:
                model.gamma = gamma_final

            # KL β schedule
            if epoch < beta_warm:
                model.beta = 0.0
            elif epoch < beta_warm + beta_ramp:
                model.beta = beta_final * ((epoch - beta_warm) / beta_ramp)
            else:
                model.beta = beta_final

            # one epoch
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
                    do_latent_infer=False,  # VAE: no latent refinement needed
                    latent_steps=0,
                    latent_lr=0.0,
                    latent_weight_decay=0.0,
                    latent_seed=(self.seed if self.seed is not None else 123),
                    _latent_cache=None,
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
        """One epoch: one-hot inputs → VAE forward → recon (focal) + KL.

        The VAEModel is expected to return (recon_logits, mu, logvar, ...) and expose a `compute_loss(outputs, y, mask, class_weights)` method that reads scheduled `model.beta` (and optionally `model.gamma`) attributes.

        Args:
            loader (torch.utils.data.DataLoader): Yields (indices, y_int) where y_int is 0/1/2; -1 for missing.
            optimizer (torch.optim.Optimizer): Optimizer.
            model (torch.nn.Module): VAE model.
            l1_penalty (float): L1 regularization coefficient.
            class_weights (torch.Tensor): CE class weights on device.

        Returns:
            float: Mean training loss for the epoch.
        """
        model.train()
        running = 0.0

        for _, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)

            x_ohe = self._one_hot_encode_012(y_batch)  # (B, L, K), zeros for -1
            outputs = model(x_ohe)  # (recon_logits, mu, logvar, ...)

            # Targets for masked focal CE, same shapes as AE path
            y_ohe = self._one_hot_encode_012(y_batch)
            valid_mask = y_batch != -1

            loss = model.compute_loss(
                outputs=outputs,
                y=y_ohe,  # (B, L, K)
                mask=valid_mask,  # (B, L)
                class_weights=class_weights,
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

        This method uses the trained VAE model to predict genotype labels for the provided input data. It processes the input data, performs a forward pass through the model, and computes the predicted labels and probabilities. The method can return either just the predicted labels or both labels and probabilities based on the `return_proba` flag.

        Args:
            model (torch.nn.Module): Trained model.
            X (np.ndarray | torch.Tensor): 0/1/2 matrix with -1 for missing.
            return_proba (bool): If True, also return probabilities.

        Returns:
            Tuple[np.ndarray, np.ndarray] | np.ndarray: Predicted labels, and probabilities if requested.
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
            outputs = model(x_ohe)  # first element must be recon logits
            logits = outputs[0].view(-1, self.num_features_, self.num_classes_)
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
        latent_vectors_val: np.ndarray | None = None,
    ) -> Dict[str, float]:
        """Evaluate on 0/1/2; then IUPAC decoding and 10-base integer reports.

        This method evaluates the trained VAE model on a validation dataset, computing various performance metrics. It handles missing data appropriately and generates detailed classification reports for both the original 0/1/2 encoding and the decoded IUPAC and integer formats. The evaluation metrics are logged for review.

        Args:
            X_val (np.ndarray): Validation 0/1/2 matrix with -1 for missing.
            model (torch.nn.Module): Trained model.
            params (dict): Current hyperparameters (for logging).
            objective_mode (bool): If True, minimize logging for Optuna.
            latent_vectors_val (np.ndarray | None): Not used by VAE.

        Returns:
            Dict[str, float]: Computed metrics.

        Raises:
            NotFittedError: If called before fit().
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

        # ensure valid probability simplex after masking
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

            # Primary report
            self._make_class_reports(
                y_true=y_true_flat,
                y_pred_proba=y_proba_flat,
                y_pred=y_pred_flat,
                metrics=metrics,
                labels=target_names,
            )

            # IUPAC decode & 10-base integer report
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
        """Optuna objective for VAE (no latent refinement during eval).

        This method defines the objective function for hyperparameter tuning using Optuna. It samples hyperparameters, trains the VAE model with these parameters, and evaluates its performance on a validation set. The evaluation metric specified by `self.tune_metric` is returned for optimization. If training fails, the trial is pruned to keep the tuning process efficient.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float: Value of the tuning metric to be optimized.
        """
        try:
            params = self._sample_hyperparameters(trial)

            X_train = self.ground_truth_[self.train_idx_]
            X_val = self.ground_truth_[self.test_idx_]

            class_weights = self._class_weights_from_zygosity(X_train).to(self.device)
            train_loader = self._get_data_loader(X_train)

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
            # Keep sweeps moving
            self.logger.debug(f"Trial failed with error: {e}")
            raise optuna.exceptions.TrialPruned(
                f"Trial failed with error. Enable debug logging for details."
            )

    def _sample_hyperparameters(
        self, trial: optuna.Trial
    ) -> Dict[str, int | float | str]:
        """Sample VAE hyperparameters; hidden sizes mirror AE/NLPCA helper.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            Dict[str, int | float | str]: Sampled hyperparameters.
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
            # VAE-specific β (final value after anneal)
            "beta": trial.suggest_float("beta", 0.25, 4.0),
            # focal gamma (if used in VAE recon CE)
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
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

        # [latent_dim] + interior widths (exclude output width)
        hidden_only = [hidden_layer_sizes[0]] + hidden_layer_sizes[1:-1]

        params["model_params"] = {
            "n_features": self.num_features_,
            "num_classes": self.num_classes_,
            "latent_dim": params["latent_dim"],
            "dropout_rate": params["dropout_rate"],
            "hidden_layer_sizes": hidden_only,
            "activation": params["activation"],
            # Pass through VAE recon/regularization coefficients
            "beta": params["beta"],
            "gamma": params["gamma"],
        }
        return params

    def _set_best_params(
        self, best_params: Dict[str, int | float | str | list]
    ) -> Dict[str, int | float | str | list]:
        """Adopt best params and return VAE model_params.

        Args:
            best_params (Dict[str, int | float | str | list]): Best hyperparameters from tuning.

        Returns:
            Dict[str, int | float | str | list]: VAE model parameters.
        """
        self.latent_dim = best_params["latent_dim"]
        self.dropout_rate = best_params["dropout_rate"]
        self.learning_rate = best_params["learning_rate"]
        self.l1_penalty = best_params["l1_penalty"]
        self.activation = best_params["activation"]
        self.layer_scaling_factor = best_params["layer_scaling_factor"]
        self.layer_schedule = best_params["layer_schedule"]
        self.kl_beta_final = best_params.get("beta", self.kl_beta_final)
        self.gamma = best_params.get("gamma", self.gamma)

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=self.num_features_ * self.num_classes_,
            n_outputs=self.num_features_ * self.num_classes_,
            n_samples=len(self.train_idx_),
            n_hidden=best_params["num_hidden_layers"],
            alpha=best_params["layer_scaling_factor"],
            schedule=best_params["layer_schedule"],
        )
        hidden_only = [hidden_layer_sizes[0]] + hidden_layer_sizes[1:-1]

        return {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": hidden_only,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "num_classes": self.num_classes_,
            "beta": self.kl_beta_final,
            "gamma": self.gamma,
        }

    def _default_best_params(self) -> Dict[str, int | float | str | list]:
        """Default VAE model params when tuning is disabled.

        Returns:
            Dict[str, int | float | str | list]: VAE model parameters.
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
            "beta": self.kl_beta_final,
            "gamma": self.gamma,
        }
