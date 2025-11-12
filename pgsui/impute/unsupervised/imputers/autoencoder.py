import copy
from pdb import pm
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, Union

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
from pgsui.data_processing.containers import AutoencoderConfig
from pgsui.data_processing.transformers import SimMissingTransformer
from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.loss_functions import SafeFocalCELoss
from pgsui.impute.unsupervised.models.autoencoder_model import AutoencoderModel
from pgsui.utils.pretty_metrics import PrettyMetrics

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
        simulate_missing: bool | None = None,
        sim_strategy: (
            Literal[
                "random",
                "random_weighted",
                "random_weighted_inv",
                "nonrandom",
                "nonrandom_weighted",
            ]
            | None
        ) = None,
        sim_prop: float | None = None,
        sim_kwargs: dict | None = None,
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

        # Simulated-missing controls (config defaults with ctor overrides)
        sim_cfg = getattr(self.cfg, "sim", None)
        sim_cfg_kwargs = copy.deepcopy(getattr(sim_cfg, "sim_kwargs", None) or {})
        if sim_kwargs:
            sim_cfg_kwargs.update(sim_kwargs)
        self.simulate_missing = (
            (
                sim_cfg.simulate_missing
                if simulate_missing is None
                else bool(simulate_missing)
            )
            if sim_cfg is not None
            else bool(simulate_missing)
        )
        if sim_cfg is None:
            default_strategy = "random"
            default_prop = 0.10
        else:
            default_strategy = sim_cfg.sim_strategy
            default_prop = sim_cfg.sim_prop
        self.sim_strategy = sim_strategy or default_strategy
        self.sim_prop = float(sim_prop if sim_prop is not None else default_prop)
        self.sim_kwargs = sim_cfg_kwargs

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
        self.sim_mask_global_: np.ndarray | None = None
        self.sim_mask_train_: np.ndarray | None = None
        self.sim_mask_test_: np.ndarray | None = None

    def fit(self) -> "ImputeAutoencoder":
        """Fit the autoencoder on 0/1/2 encoded genotypes (missing -> -1).

        This method trains the autoencoder model using the provided genotype data. It prepares the data by encoding genotypes as 0, 1, and 2, with missing values represented internally as -1. (When simulated-missing loci are generated via ``SimMissingTransformer`` they are first marked with -9 but are immediately re-encoded as -1 prior to training.) The method splits the data into training and validation sets, initializes the model and training parameters, and performs training with optional hyperparameter tuning. After training, it evaluates the model on the validation set and stores the fitted model and training history.

        Returns:
            ImputeAutoencoder: Fitted instance.

        Raises:
            NotFittedError: If training fails.
        """
        self.logger.info(f"Fitting {self.model_name} model...")

        # --- Data prep (mirror NLPCA) ---
        X012 = self._get_float_genotypes(copy=True)
        GT_full = np.nan_to_num(X012, nan=-1.0, copy=True)
        self.ground_truth_ = GT_full.astype(np.int64, copy=False)

        self.sim_mask_global_ = None
        cache_key = self._sim_mask_cache_key()
        if self.simulate_missing:
            cached_mask = (
                None if cache_key is None else self._sim_mask_cache.get(cache_key)
            )
            if cached_mask is not None:
                self.sim_mask_global_ = cached_mask.copy()
            else:
                tr = SimMissingTransformer(
                    genotype_data=self.genotype_data,
                    prop_missing=self.sim_prop,
                    strategy=self.sim_strategy,
                    missing_val=-9,
                    mask_missing=True,
                    verbose=self.verbose,
                    **self.sim_kwargs,
                )
                tr.fit(X012.copy())
                self.sim_mask_global_ = tr.sim_missing_mask_.astype(bool)
                if cache_key is not None:
                    self._sim_mask_cache[cache_key] = self.sim_mask_global_.copy()

            X_for_model = self.ground_truth_.copy()
            X_for_model[self.sim_mask_global_] = -1
        else:
            X_for_model = self.ground_truth_.copy()

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

        if self.is_haploid:
            self.ground_truth_[self.ground_truth_ == 2] = 1
            X_for_model[X_for_model == 2] = 1

        n_samples, self.num_features_ = X_for_model.shape

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
        self.X_train_ = X_for_model[train_idx]
        self.X_val_ = X_for_model[val_idx]
        self.GT_train_full_ = self.ground_truth_[train_idx]
        self.GT_test_full_ = self.ground_truth_[val_idx]

        if self.sim_mask_global_ is not None:
            self.sim_mask_train_ = self.sim_mask_global_[train_idx]
            self.sim_mask_test_ = self.sim_mask_global_[val_idx]
        else:
            self.sim_mask_train_ = None
            self.sim_mask_test_ = None

        # Plotters/scorers (shared utilities)
        self.plotter_, self.scorers_ = self.initialize_plotting_and_scorers()

        # Tuning (optional; AE never needs latent refinement)
        if self.tune:
            self.tune_hyperparameters()

        # Best params (tuned or default)
        self.best_params_ = getattr(self, "best_params_", self._default_best_params())

        # Class weights (device-aware)
        self.class_weights_ = self._normalize_class_weights(
            self._class_weights_from_zygosity(self.X_train_)
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
        eval_mask = (
            self.sim_mask_test_
            if (self.simulate_missing and self.sim_mask_test_ is not None)
            else None
        )
        self._evaluate_model(
            self.X_val_, self.model_, self.best_params_, eval_mask_override=eval_mask
        )
        self.plotter_.plot_history(self.history_)
        self._save_best_params(self.best_params_)

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

        self.logger.info(f"Imputing entire dataset with {self.model_name}...")
        X_to_impute = self.ground_truth_.copy()

        # Predict with masked inputs (no latent optimization)
        pred_labels, _ = self._predict(self.model_, X=X_to_impute, return_proba=True)

        # Fill only missing
        missing_mask = X_to_impute == -1
        imputed_array = X_to_impute.copy()
        imputed_array[missing_mask] = pred_labels[missing_mask]

        # Decode to IUPAC & optionally plot
        imputed_genotypes = self.pgenc.decode_012(imputed_array)
        if self.show_plots:
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
        y_tensor = torch.from_numpy(y).long()
        indices = torch.arange(len(y), dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(indices, y_tensor)
        pin_memory = self.device.type == "cuda"
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=pin_memory,
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

        warm, ramp, gamma_final = 50, 100, self.gamma

        # Optional: linear LR warmup for first N epochs
        lr0 = optimizer.param_groups[0]["lr"]
        warmup_epochs = getattr(self, "lr_warmup_epochs", 5)
        min_lr = lr0 * 0.1  # start at 10% of target LR

        for epoch in range(scheduler.T_max):
            # Gamma warm/ramp
            if epoch < warm:
                model.gamma = 0.0
            elif epoch < warm + ramp:
                model.gamma = gamma_final * ((epoch - warm) / ramp)
            else:
                model.gamma = gamma_final

            # LR warmup before stepping the main scheduler
            if epoch < warmup_epochs:
                new_lr = min_lr + (lr0 - min_lr) * (epoch + 1) / warmup_epochs
                for g in optimizer.param_groups:
                    g["lr"] = new_lr

            train_loss = self._train_step(
                loader=loader,
                optimizer=optimizer,
                model=model,
                l1_penalty=l1_penalty,
                class_weights=class_weights,
            )

            # Abort or prune on non-finite epoch loss
            if not np.isfinite(train_loss):
                if trial is not None:
                    raise optuna.exceptions.TrialPruned("Epoch loss non-finite.")
                # Soft reset suggestion: reduce LR and continue, or break
                self.logger.warning(
                    "Non-finite epoch loss. Reducing LR by 10 percent and continuing."
                )
                for g in optimizer.param_groups:
                    g["lr"] *= 0.9
                continue

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
                mask_override = None
                if (
                    self.simulate_missing
                    and getattr(self, "sim_mask_test_", None) is not None
                    and getattr(self, "X_val_", None) is not None
                    and X_val.shape == self.X_val_.shape
                ):
                    mask_override = self.sim_mask_test_
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
                    eval_mask_override=mask_override,
                )
                trial.report(metric_val, step=epoch + 1)
                if (epoch + 1) >= prune_warmup_epochs and trial.should_prune():
                    raise optuna.exceptions.TrialPruned(
                        f"Pruned at epoch {epoch + 1}: {metric_key}={metric_val:.5f}"
                    )

        best_loss = early_stopping.best_score
        best_model = copy.deepcopy(early_stopping.best_model)
        if best_model is None:
            best_model = copy.deepcopy(model)
        return best_loss, best_model, history

    def _train_step(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        l1_penalty: float,
        class_weights: torch.Tensor,
    ) -> float:
        """One epoch with stable focal CE and NaN/Inf guards."""
        model.train()
        running = 0.0
        num_batches = 0
        l1_params = tuple(p for p in model.parameters() if p.requires_grad)
        if class_weights is not None and class_weights.device != self.device:
            class_weights = class_weights.to(self.device)

        # Use model.gamma if present, else self.gamma
        gamma = float(getattr(model, "gamma", getattr(self, "gamma", 0.0)))
        gamma = float(torch.tensor(gamma).clamp(min=0.0, max=10.0))  # sane bound
        criterion = SafeFocalCELoss(gamma=gamma, weight=class_weights, ignore_index=-1)

        for _, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            # Inputs: one-hot with zeros for missing; Targets: long ints with -1 for missing
            x_ohe = self._one_hot_encode_012(y_batch)  # (B, L, K)
            logits = model(x_ohe).view(-1, self.num_features_, self.num_classes_)
            logits_flat = logits.view(-1, self.num_classes_)
            targets_flat = y_batch.view(-1).long()

            # Upfront guards on inputs
            if not torch.isfinite(logits_flat).all():
                # Skip this batch if model already produced non-finite
                continue

            loss = criterion(logits_flat, targets_flat)

            if l1_penalty > 0:
                l1 = torch.zeros((), device=self.device)
                for p in l1_params:
                    l1 = l1 + p.abs().sum()
                loss = loss + l1_penalty * l1

            # Final guard
            if not torch.isfinite(loss):
                continue

            loss.backward()

            # Clip to prevent exploding grads
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # If grads blew up to non-finite, skip update
            if any(
                (not torch.isfinite(p.grad).all())
                for p in model.parameters()
                if p.grad is not None
            ):
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

            running += float(loss.detach().item())
            num_batches += 1

        if num_batches == 0:
            return float("inf")  # signal upstream that epoch had no usable batches
        return running / num_batches

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
        *,
        eval_mask_override: np.ndarray | None = None,
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

        finite_mask = np.all(np.isfinite(pred_probas), axis=-1)  # (N, L)

        if (
            hasattr(self, "X_val_")
            and getattr(self, "X_val_", None) is not None
            and X_val.shape == self.X_val_.shape
        ):
            GT_ref = getattr(self, "GT_test_full_", self.ground_truth_)
        elif (
            hasattr(self, "X_train_")
            and getattr(self, "X_train_", None) is not None
            and X_val.shape == self.X_train_.shape
        ):
            GT_ref = getattr(self, "GT_train_full_", self.ground_truth_)
        else:
            GT_ref = self.ground_truth_

        if GT_ref.shape != X_val.shape:
            GT_ref = X_val

        if eval_mask_override is not None:
            if eval_mask_override.shape != X_val.shape:
                msg = (
                    f"eval_mask_override shape {eval_mask_override.shape} "
                    f"does not match X_val shape {X_val.shape}"
                )
                self.logger.error(msg)
                raise ValueError(msg)
            eval_mask = eval_mask_override.astype(bool)
        else:
            eval_mask = X_val != -1

        eval_mask = eval_mask & finite_mask & (GT_ref != -1)

        y_true_flat = GT_ref[eval_mask].astype(np.int64, copy=False)
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
            if self.verbose or self.debug:
                pm = PrettyMetrics(metrics, precision=3, title="Validation Metrics")
                pm.render()  # prints a command-line table

            # Primary report (REF/HET/ALT or REF/ALT)
            self._make_class_reports(
                y_true=y_true_flat,
                y_pred_proba=y_proba_flat,
                y_pred=y_pred_flat,
                metrics=metrics,
                labels=target_names,
            )

            # IUPAC decode & 10-base integer reports
            y_true_dec = self.pgenc.decode_012(
                GT_ref.reshape(X_val.shape[0], X_val.shape[1])
            )
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

            valid_iupac_mask = y_true_int[eval_mask] >= 0
            if valid_iupac_mask.any():
                self._make_class_reports(
                    y_true=y_true_int[eval_mask][valid_iupac_mask],
                    y_pred=y_pred_int[eval_mask][valid_iupac_mask],
                    metrics=metrics,
                    y_pred_proba=None,
                    labels=["A", "C", "G", "T", "W", "R", "M", "K", "Y", "S"],
                )
            else:
                self.logger.warning(
                    "Skipped IUPAC confusion matrix: No valid ground truths."
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
            X_train = getattr(self, "X_train_", self.ground_truth_[self.train_idx_])
            X_val = getattr(self, "X_val_", self.ground_truth_[self.test_idx_])

            class_weights = self._normalize_class_weights(
                self._class_weights_from_zygosity(X_train)
            )
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

            eval_mask = (
                self.sim_mask_test_
                if (
                    self.simulate_missing
                    and getattr(self, "sim_mask_test_", None) is not None
                )
                else None
            )
            metrics = self._evaluate_model(
                X_val, model, params, objective_mode=True, eval_mask_override=eval_mask
            )
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
