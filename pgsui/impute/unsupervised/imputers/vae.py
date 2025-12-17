from __future__ import annotations

import copy
import traceback
from collections import defaultdict
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
from pgsui.data_processing.containers import VAEConfig
from pgsui.data_processing.transformers import SimMissingTransformer
from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.loss_functions import SafeFocalCELoss, compute_vae_loss
from pgsui.impute.unsupervised.models.vae_model import VAEModel
from pgsui.utils.logging_utils import configure_logger
from pgsui.utils.pretty_metrics import PrettyMetrics

if TYPE_CHECKING:
    from snpio import TreeParser
    from snpio.read_input.genotype_data import GenotypeData


def ensure_vae_config(config: VAEConfig | dict | str | None) -> VAEConfig:
    if config is None:
        return VAEConfig()
    if isinstance(config, VAEConfig):
        return config
    if isinstance(config, str):
        return load_yaml_to_dataclass(config, VAEConfig)
    if isinstance(config, dict):
        cfg_in = copy.deepcopy(config)  # avoid mutating caller
        base = VAEConfig()

        preset = cfg_in.pop("preset", None)
        if "io" in cfg_in and isinstance(cfg_in["io"], dict):
            preset = preset or cfg_in["io"].pop("preset", None)
        if preset:
            base = VAEConfig.from_preset(preset)

        def _flatten(prefix: str, d: dict, out: dict) -> dict:
            for k, v in d.items():
                kk = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(kk, v, out)
                else:
                    out[kk] = v
            return out

        flat = _flatten("", cfg_in, {})
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
        tree_parser: Optional["TreeParser"] = None,
        config: Optional[Union["VAEConfig", dict, str]] = None,
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
    ):
        """Initialize the VAE imputer with a unified config interface.

        This initializer sets up the VAE imputer by processing the provided configuration, initializing logging, and preparing the model and data encoder. It supports configuration input as a dataclass, nested dictionary, YAML file path, or None, with optional dot-key overrides for fine-tuning specific parameters.

        Args:
            genotype_data (GenotypeData): Backing genotype data object.
            tree_parser (TreeParser | None): Optional SNPio phylogenetic tree parser for nonrandom sim_strategy modes.
            config (Union[VAEConfig, dict, str, None]): VAEConfig, nested dict, YAML path, or None (defaults).
            overrides (dict | None): Optional dot-key overrides with highest precedence.
            simulate_missing (bool | None): Whether to simulate missing data during training.
            sim_strategy (Literal[...] | None): Simulated missing strategy if simulating.
            sim_prop (float | None): Proportion of data to simulate as missing if simulating.
            sim_kwargs (dict | None): Additional kwargs for SimMissingTransformer.
        """
        self.model_name = "ImputeVAE"
        self.genotype_data = genotype_data
        self.tree_parser = tree_parser

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
        self.logger = configure_logger(
            logman.get_logger(),
            verbose=self.cfg.io.verbose,
            debug=self.cfg.io.debug,
        )

        # BaseNNImputer bootstraps device/dirs/log formatting
        super().__init__(
            model_name=self.model_name,
            genotype_data=self.genotype_data,
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

        # Simulated-missing controls (config defaults + ctor overrides)
        sim_cfg = getattr(self.cfg, "sim", None)
        sim_cfg_kwargs = copy.deepcopy(getattr(sim_cfg, "sim_kwargs", None) or {})
        if sim_kwargs:
            sim_cfg_kwargs.update(sim_kwargs)
        if sim_cfg is None:
            default_sim_flag = bool(simulate_missing)
            default_strategy = "random"
            default_prop = 0.10
        else:
            default_sim_flag = sim_cfg.simulate_missing
            default_strategy = sim_cfg.sim_strategy
            default_prop = sim_cfg.sim_prop
        self.simulate_missing = (
            default_sim_flag if simulate_missing is None else bool(simulate_missing)
        )
        self.sim_strategy = sim_strategy or default_strategy
        self.sim_prop = float(sim_prop if sim_prop is not None else default_prop)
        self.sim_kwargs = sim_cfg_kwargs

        # Model hyperparams (AE-parity)
        self.latent_dim = self.cfg.model.latent_dim
        self.dropout_rate = self.cfg.model.dropout_rate
        self.num_hidden_layers = self.cfg.model.num_hidden_layers
        self.layer_scaling_factor = self.cfg.model.layer_scaling_factor
        self.layer_schedule = self.cfg.model.layer_schedule
        self.activation = self.cfg.model.activation
        self.gamma = self.cfg.model.gamma  # focal loss focusing (for recon CE)

        # VAE-only KL controls
        self.kl_beta_final = self.cfg.vae.kl_beta

        # Train hyperparams (AE-parity)
        self.batch_size = self.cfg.train.batch_size
        self.learning_rate = self.cfg.train.learning_rate
        self.l1_penalty: float = self.cfg.train.l1_penalty
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
        self.tune_eval_interval = 1
        self.tune_metric: Literal[
            "pr_macro",
            "f1",
            "accuracy",
            "average_precision",
            "precision",
            "recall",
            "roc_auc",
        ] = self.cfg.tune.metric
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
        self.is_haploid: bool = False
        self.num_classes_: int = 3  # diploid default
        self.model_params: Dict[str, Any] = {}
        self.sim_mask_global_: np.ndarray | None = None
        self.sim_mask_train_: np.ndarray | None = None
        self.sim_mask_test_: np.ndarray | None = None

        if self.tree_parser is None and self.sim_strategy.startswith("nonrandom"):
            msg = "tree_parser is required for nonrandom and nonrandom_weighted simulated missing strategies."
            self.logger.error(msg)
            raise ValueError(msg)

    # -------------------- Fit -------------------- #
    def fit(self) -> "ImputeVAE":
        """Fit the VAE on 0/1/2 encoded genotypes (missing -> -1).

        This method prepares the genotype data, initializes model parameters, splits the data into training and validation sets, and trains the VAE model. Missing positions are encoded as -1 for loss masking (any simulated-missing loci are temporarily tagged with -9 by ``SimMissingTransformer`` before being re-encoded as -1). It handles both haploid and diploid data, applies class weighting, and supports optional hyperparameter tuning. After training, it evaluates the model on the validation set and saves the trained model.

        Returns:
            ImputeVAE: Fitted instance.

        Raises:
            RuntimeError: If training fails to produce a model.
        """
        self.logger.info(f"Fitting {self.model_name} model...")

        # Data prep aligns with AE/NLPCA
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
                    tree_parser=self.tree_parser,
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

        # Ploidy/classes
        self.ploidy = self.cfg.io.ploidy
        self.is_haploid = self.ploidy == 1

        # Scoring + model head are now the same: categorical K classes
        self.num_classes_ = 2 if self.is_haploid else 3
        self.output_classes_ = self.num_classes_

        self.logger.info(
            f"Data is {'haploid' if self.is_haploid else 'diploid'}; "
            f"using {self.num_classes_} classes for scoring and model output channels."
        )

        # Haploid collapse: keep only {0,1}; treat 2 as ALT(=1)
        if self.is_haploid:
            self.ground_truth_[self.ground_truth_ == 2] = 1
            X_for_model[X_for_model == 2] = 1

        n_samples, self.num_features_ = X_for_model.shape

        self.X_model_input_ = X_for_model

        # Model params (decoder outputs L*K logits)
        self.model_params = {
            "n_features": self.num_features_,
            "num_classes": self.output_classes_,
            "latent_dim": self.latent_dim,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
        }

        if n_samples < 3:
            msg = f"Not enough samples ({n_samples}) for train/val split. Increase tune_max_samples or disable tune_fast."
            self.logger.error(msg)
            raise ValueError(msg)

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

        # Class weights (device-aware)
        self.class_weights_ = self._normalize_class_weights(
            self._class_weights_from_zygosity(self.X_train_)
        )

        # Optional tuning
        if self.tune:
            self.tuned_params_ = self.tune_hyperparameters()

        # Best params (tuned or default)
        self.best_params_ = getattr(self, "best_params_", self._default_best_params())

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
            class_weights=self.class_weights_,
            X_val=self.X_val_,
            params=self.best_params_,
            prune_metric=self.tune_metric,
            eval_interval=1,
            eval_requires_latents=False,  # no latent refinement for eval
            eval_latent_steps=0,
            eval_latent_lr=0.0,
            eval_latent_weight_decay=0.0,
            eval_mask_override=getattr(self, "sim_mask_test_", None),
        )

        if trained_model is None:
            msg = f"{self.model_name} training failed; no model was returned."
            self.logger.error(msg)
            raise RuntimeError(msg)

        torch.save(
            trained_model.state_dict(),
            self.models_dir / f"final_model_{self.model_name}.pt",
        )

        if history is None:
            hist: dict[str, list[float]] = {"Train": []}
        elif isinstance(history, dict):
            # {"Train":[...], "Val":[...]}
            hist = dict(history)
        else:
            # backwards compatibility if history is a list
            hist = {"Train": list(history)}

        self.best_loss_, self.model_, self.history_ = (loss, trained_model, hist)

        self.is_fit_ = True

        # Evaluate (AE-parity reporting)
        eval_mask = (
            self.sim_mask_test_
            if (self.simulate_missing and self.sim_mask_test_ is not None)
            else None
        )
        self._evaluate_model(
            self.X_val_,
            self.model_,
            self.best_params_,
            y_true_matrix=self.GT_test_full_,
            eval_mask_override=eval_mask,
        )

        if self.show_plots:
            self.plotter_.plot_history(self.history_)

        if self.tune:
            self._save_best_params(self.tuned_params_)
        else:
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
            msg = "Model is not fitted. Call fit() before transform()."
            self.logger.error(msg)
            raise NotFittedError(msg)

        self.logger.info(f"Imputing entire dataset with {self.model_name} model...")
        X_to_impute = self.ground_truth_.copy()

        pred_labels, _ = self._predict(self.model_, X=X_to_impute, return_proba=True)

        # Fill only missing
        missing_mask = X_to_impute < 0
        imputed_array = X_to_impute.copy()
        imputed_array[missing_mask] = pred_labels[missing_mask]

        neg_ct = int(np.count_nonzero(imputed_array < 0))
        self.logger.info(
            f"[transform] negative entries remaining in imputed_array: {neg_ct}"
        )
        if neg_ct:
            self.logger.info(
                f"[transform] unique negatives: {np.unique(imputed_array[imputed_array < 0])[:10]}"
            )

        # Decode to IUPAC & optionally plot
        imputed_genotypes = self.pgenc.decode_012(imputed_array)

        if self.show_plots:
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

    # Alias to satisfy BaseNNImputer tuning helper
    def _get_data_loaders(self, y: np.ndarray) -> torch.utils.data.DataLoader:  # type: ignore[override]
        return self._get_data_loader(y)

    def _train_and_validate_model(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        lr: float,
        l1_penalty: float,
        trial: optuna.Trial | None = None,
        class_weights: torch.Tensor | None = None,
        *,
        X_val: np.ndarray | None = None,
        params: dict | None = None,
        prune_metric: str | None = None,  # "f1" | "accuracy" | "pr_macro"
        eval_interval: int = 1,
        eval_requires_latents: bool = False,  # VAE: no latent eval refinement
        eval_latent_steps: int = 0,
        eval_latent_lr: float = 0.0,
        eval_latent_weight_decay: float = 0.0,
        eval_mask_override: np.ndarray | None = None,
    ) -> Tuple[float, torch.nn.Module | None, dict[str, list[float]]]:
        """Wrap the VAE training loop with β-anneal & Optuna pruning.

        This method orchestrates the training of the VAE model, including setting up the optimizer and learning rate scheduler, and executing the training loop with support for early stopping and Optuna pruning. It manages the training process, monitors performance on a validation set if provided, and returns the best model and training history.

        Args:
            model (torch.nn.Module): VAE model.
            loader (torch.utils.data.DataLoader): Training data loader.
            lr (float): Learning rate.
            l1_penalty (float): L1 regularization coefficient.
            trial (optuna.Trial | None): Optuna trial for pruning.
            class_weights (torch.Tensor | None): CE class weights on device.
            X_val (np.ndarray | None): Validation data for pruning eval.
            params (dict | None): Current hyperparameters (for logging).
            prune_metric (str | None): Metric for pruning decisions.
            eval_interval (int): Epochs between validation evaluations.
            eval_requires_latents (bool): If True, refine latents during eval.
            eval_latent_steps (int): Latent refinement steps if needed.
            eval_latent_lr (float): Latent refinement learning rate.
            eval_latent_weight_decay (float): Latent refinement L2 penalty.
            eval_mask_override (np.ndarray | None): Optional mask override for eval.

        Returns:
            Tuple[float, torch.nn.Module | None, dict[str, list[float]]]: Best loss, best model, and training history (if requested).
        """
        if class_weights is None:
            msg = "Must provide class_weights."
            self.logger.error(msg)
            raise TypeError(msg)

        if trial is not None and self.tune_fast:
            max_epochs = self.tune_epochs
        else:
            max_epochs = self.epochs

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

        best_loss, best_model, hist = self._execute_training_loop(
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            l1_penalty=l1_penalty,
            trial=trial,
            class_weights=class_weights,
            X_val=X_val,
            params=params,
            prune_metric=prune_metric,
            eval_interval=eval_interval,
            eval_requires_latents=eval_requires_latents,
            eval_latent_steps=eval_latent_steps,
            eval_latent_lr=eval_latent_lr,
            eval_latent_weight_decay=eval_latent_weight_decay,
            eval_mask_override=eval_mask_override,
        )
        return best_loss, best_model, hist

    def _execute_training_loop(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineAnnealingLR,
        model: torch.nn.Module,
        l1_penalty: float,
        trial: optuna.Trial | None,
        class_weights: torch.Tensor,
        *,
        X_val: np.ndarray | None = None,
        params: dict | None = None,
        prune_metric: str | None = None,
        eval_interval: int = 1,
        eval_requires_latents: bool = False,
        eval_latent_steps: int = 0,
        eval_latent_lr: float = 0.0,
        eval_latent_weight_decay: float = 0.0,
        eval_mask_override: np.ndarray | None = None,
        val_batch_size: int | None = None,
    ) -> Tuple[float, torch.nn.Module, dict[str, list[float]]]:
        """Train VAE with stable focal CE + KL(β) anneal and numeric guards.

        This method implements the core training loop for the VAE model, incorporating a focal cross-entropy loss for reconstruction and a KL divergence term with an annealed weight (beta). It includes mechanisms for early stopping based on validation performance, learning rate scheduling, and optional pruning of unpromising trials when using Optuna for hyperparameter optimization. The method ensures numerical stability throughout the training process.

        Args:
            loader (torch.utils.data.DataLoader): Training data loader.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            model (torch.nn.Module): VAE model.
            l1_penalty (float): L1 regularization coefficient.
            trial (optuna.Trial | None): Optuna trial for pruning.
            return_history (bool): If True, return training history.
            class_weights (torch.Tensor): CE class weights on device.
            X_val (np.ndarray | None): Validation data for pruning eval.
            params (dict | None): Current hyperparameters (for logging).
            prune_metric (str | None): Metric for pruning decisions.
            eval_interval (int): Epochs between validation evaluations.
            eval_requires_latents (bool): If True, refine latents during eval.
            eval_latent_steps (int): Latent refinement steps if needed.
            eval_latent_lr (float): Latent refinement learning rate.
            eval_latent_weight_decay (float): Latent refinement L2 penalty.
            eval_mask_override (np.ndarray | None): Optional mask override for eval.

        Returns:
            Tuple[float, torch.nn.Module, list]: Best loss, best model, and training history.
        """
        best_model = None
        history: dict[str, list[float]] = defaultdict(list)

        early_stopping = EarlyStopping(
            patience=self.early_stop_gen,
            min_epochs=self.min_epochs,
            verbose=self.verbose,
            prefix=self.prefix,
            debug=self.debug,
        )

        # ---- scalarize schedule endpoints up front ----
        beta_src = getattr(model, "beta", self.kl_beta_final)
        gamma_src = getattr(model, "gamma", self.gamma)

        # scalarize if lists
        if isinstance(beta_src, (list, tuple)):
            if not beta_src:
                msg = "beta list is empty."
                self.logger.error(msg)
                raise ValueError(msg)
            beta_src = beta_src[0]
        if isinstance(gamma_src, (list, tuple)):
            if not gamma_src:
                msg = "gamma list is empty."
                self.logger.error(msg)
                raise ValueError(msg)
            gamma_src = gamma_src[0]

        max_epochs = int(getattr(scheduler, "T_max", getattr(self, "epochs", 100)))

        for epoch in range(max_epochs):
            train_loss = self._train_step(
                loader=loader,
                optimizer=optimizer,
                model=model,
                l1_penalty=l1_penalty,
                class_weights=class_weights,
            )

            if not np.isfinite(train_loss):
                if trial is not None:
                    self.logger.debug(
                        "Non-finite train loss encountered; pruning trial.",
                        exc_info=True,
                    )
                    raise optuna.exceptions.TrialPruned(
                        f"[{self.model_name}] Epoch loss non-finite. Trial pruned. Enable debug logging for full traceback."
                    )
                else:
                    self.logger.error(
                        f"[{self.model_name}] Non-finite train loss encountered at epoch {epoch + 1}. Terminating training."
                    )
                    raise RuntimeError(
                        f"[{self.model_name}] Non-finite train loss encountered at epoch {epoch + 1}. Terminating training."
                    )

            val_loss = float("inf")
            if X_val is not None:
                y_true_val = getattr(self, "GT_test_full_", None)

                if y_true_val is None:
                    msg = "GT_test_full_ is missing; cannot compute val_loss."
                    self.logger.error(msg)
                    raise ValueError(msg)

                val_loss = (
                    self._val_step(
                        X_val=X_val,
                        y_true_val=y_true_val,
                        model=model,
                        l1_penalty=l1_penalty,
                        class_weights=class_weights,
                        eval_mask_override=eval_mask_override,
                        batch_size=val_batch_size,
                    )
                    if X_val is not None
                    else None
                )

                if isinstance(val_loss, float) and np.isfinite(val_loss):
                    history["Val"].append(val_loss)

            if scheduler is not None:
                scheduler.step()

            history["Train"].append(train_loss)

            if "Val" in history:
                score_for_es = val_loss  # minimize by default
            else:
                score_for_es = train_loss

            metric_val = None
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
                    do_latent_infer=False,
                    latent_steps=0,
                    latent_lr=0.0,
                    latent_weight_decay=0.0,
                    latent_seed=self.seed,
                    _latent_cache=None,
                    _latent_cache_key=None,
                    y_true_matrix=getattr(self, "_tune_GT_test", None),
                    eval_mask_override=eval_mask_override,
                )

                trial.report(metric_val, step=epoch + 1)
                if trial.should_prune():
                    msg = f"[{self.model_name}] Trial Median-Pruned at epoch {epoch + 1}: {metric_key}={metric_val:.3f}. This is not an error, but indicates the trial was unpromising."
                    raise optuna.exceptions.TrialPruned(msg)

            early_stopping(score_for_es, model)
            if early_stopping.early_stop:
                self.logger.debug(f"Early stopping at epoch {epoch + 1}.")
                break

        best_loss = early_stopping.best_score

        if early_stopping.best_state_dict is not None:
            model.load_state_dict(early_stopping.best_state_dict)
        best_model = model

        return best_loss, best_model, dict(history)

    def _train_step(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        l1_penalty: float,
        class_weights: torch.Tensor,
    ) -> float:
        """One epoch: categorical recon (focal CE) + β·KL with guards."""
        model.train()
        running, used = 0.0, 0
        l1_params = tuple(p for p in model.parameters() if p.requires_grad)

        if class_weights is not None and class_weights.device != self.device:
            class_weights = class_weights.to(self.device)

        nF_model = int(getattr(model, "n_features", self.num_features_))
        K = int(self.output_classes_)  # 2 or 3

        for _, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            y_int = y_batch.to(self.device, non_blocking=True).long()

            if y_int.dim() != 2:
                msg = f"Training batch expected 2D targets, got shape {tuple(y_int.shape)}."
                self.logger.error(msg)
                raise ValueError(msg)

            if y_int.shape[1] != nF_model:
                msg = (
                    f"Model expects {nF_model} loci but batch has {y_int.shape[1]}. "
                    "Ensure tuning subsets and masks use matching loci columns."
                )
                self.logger.error(msg)
                raise ValueError(msg)

            # Inputs: categorical one-hot (missing -> all-zeros via your helper)
            x_in = self._one_hot_encode_012(y_int, num_classes=K)  # (B, L, K)

            out = model(x_in)
            if isinstance(out, (list, tuple)):
                recon_logits, mu, logvar = out[0], out[1], out[2]
            else:
                recon_logits, mu, logvar = out["recon_logits"], out["mu"], out["logvar"]

            # Normalize recon logits to (B, L, K)
            if recon_logits.dim() == 2 and recon_logits.shape[1] == nF_model * K:
                recon_logits = recon_logits.view(-1, nF_model, K)
            elif recon_logits.dim() == 3:
                if recon_logits.shape[1] != nF_model or recon_logits.shape[2] != K:
                    raise ValueError(
                        f"Unexpected recon_logits shape {tuple(recon_logits.shape)}; expected (B,{nF_model},{K})."
                    )
            else:
                raise ValueError(
                    f"Unexpected recon_logits dim={recon_logits.dim()} shape={tuple(recon_logits.shape)}"
                )

            # Numeric guards
            if (
                not torch.isfinite(recon_logits).all()
                or not torch.isfinite(mu).all()
                or not torch.isfinite(logvar).all()
            ):
                continue

            gamma = float(getattr(model, "gamma", getattr(self, "gamma", 0.0)))
            beta = float(getattr(model, "beta", getattr(self, "kl_beta_final", 0.0)))
            gamma = max(0.0, min(gamma, 10.0))

            # Single categorical loss path for both haploid/diploid
            loss = compute_vae_loss(
                recon_logits=recon_logits,
                targets=y_int,
                mu=mu,
                logvar=logvar,
                class_weights=class_weights,
                gamma=gamma,
                beta=beta,
            )

            if l1_penalty > 0:
                l1 = torch.zeros((), device=self.device)
                for p in l1_params:
                    l1 = l1 + p.abs().sum()
                loss = loss + l1_penalty * l1

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            bad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if bad:
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()
            running += float(loss.detach().item())
            used += 1

        return (running / used) if used > 0 else float("inf")

    def _val_step(
        self,
        *,
        X_val: np.ndarray,
        y_true_val: np.ndarray,
        model: torch.nn.Module,
        l1_penalty: float,
        class_weights: torch.Tensor,
        eval_mask_override: np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> float:
        """Compute validation focal CE loss without gradient updates.

        Notes:
            - This does NOT call backward() or optimizer.step().
            - Positions not selected for scoring are set to ignore_index (-1).
            - If a batch has no valid targets (all -1), it is skipped.

        Args:
            X_val: Validation inputs (0/1/2 with -1 for missing / masked).
            y_true_val: Ground truth matrix aligned to X_val (0/1/2; -1 for truly missing).
            model: Trained/active model.
            l1_penalty: Optional L1 coeff. (If you want "pure" val loss, pass 0.0.)
            class_weights: Class weights (C,).
            eval_mask_override: Optional boolean mask (N,L) selecting positions to score.
            batch_size: Optional batch size override.

        Returns:
            Mean validation loss (float). Returns +inf if no valid targets exist.
        """
        if X_val.shape != y_true_val.shape:
            raise ValueError(
                f"X_val and y_true_val must have identical shape; got {X_val.shape} vs {y_true_val.shape}."
            )

        model.eval()
        running = 0.0
        num_batches = 0

        if class_weights.device != self.device:
            class_weights = class_weights.to(self.device)

        nF_x = int(X_val.shape[1])
        nF_model = int(getattr(model, "n_features", nF_x))
        if nF_model != nF_x:
            raise ValueError(f"Model expects {nF_model} loci but X_val has {nF_x}.")

        nC_model = int(getattr(model, "num_classes", self.output_classes_))

        # Build scoring mask
        if eval_mask_override is not None:
            if eval_mask_override.shape != X_val.shape:
                raise ValueError(
                    f"eval_mask_override shape {eval_mask_override.shape} does not match X_val shape {X_val.shape}."
                )
            mask = eval_mask_override.astype(bool, copy=False) & (y_true_val != -1)
        else:
            mask = (X_val != -1) & (y_true_val != -1)

        # Targets: ignore everything not in mask
        y_targets = y_true_val.copy()
        if self.is_haploid:
            y_targets[y_targets == 2] = 1
        y_targets = np.where(mask, y_targets, -1).astype(np.int64, copy=False)

        # Local val loader (do NOT reuse training loader which shuffles)
        N = int(X_val.shape[0])
        idx_tensor = torch.arange(N, dtype=torch.long)
        x_tensor = torch.from_numpy(X_val).long()
        dataset = torch.utils.data.TensorDataset(idx_tensor, x_tensor)
        pin_memory = self.device.type == "cuda"
        bs = int(batch_size or self.batch_size)
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            pin_memory=pin_memory,
        )

        # Keep y_targets as a CPU tensor; gather by idx_batch then move to device
        y_targets_cpu = torch.from_numpy(y_targets).long()

        gamma = float(getattr(model, "gamma", getattr(self, "gamma", 0.0)))
        gamma = float(np.clip(gamma, 0.0, 10.0))

        ce_criterion = SafeFocalCELoss(
            gamma=gamma,
            weight=class_weights,
            ignore_index=-1,
        )

        l1_params = tuple(p for p in model.parameters() if p.requires_grad)

        with torch.inference_mode():
            for idx_batch, x_batch in val_loader:
                x_batch = x_batch.to(self.device, non_blocking=True).long()
                y_batch = y_targets_cpu.index_select(0, idx_batch).to(
                    self.device, non_blocking=True
                )

                if x_batch.dim() != 2 or y_batch.dim() != 2:
                    raise ValueError(
                        f"Expected 2D (B,L) tensors; got x={tuple(x_batch.shape)}, y={tuple(y_batch.shape)}."
                    )

                x_in = self._one_hot_encode_012(
                    x_batch, num_classes=nC_model
                )  # (B,L,C)
                raw = model(x_in)

                logits_flat = raw if isinstance(raw, torch.Tensor) else raw[0]
                expected = (x_batch.shape[0], nF_model * nC_model)

                if logits_flat.dim() != 2 or tuple(logits_flat.shape) != expected:
                    try:
                        logits_flat = logits_flat.view(-1, nF_model * nC_model)
                    except Exception as e:
                        msg = f"Model output logits expected shape {expected}, got {tuple(logits_flat.shape)}. Ensure tuning subsets and masks use matching loci columns."
                        self.logger.error(msg)
                        raise ValueError(msg) from e

                logits = logits_flat.view(-1, nF_model, nC_model).reshape(-1, nC_model)
                targets_flat = y_batch.view(-1)

                if not torch.isfinite(logits).all():
                    continue

                # Skip if there are no valid targets in this batch
                valid = targets_flat != -1
                if not bool(valid.any()):
                    continue

                logits_v = logits[valid]
                targets_v = targets_flat[valid]

                loss = ce_criterion(logits_v, targets_v)

                # Optional: include L1 in the monitored objective
                if l1_penalty > 0:
                    l1 = torch.zeros((), device=self.device)
                    for p in l1_params:
                        l1 = l1 + p.abs().sum()
                    loss = loss + l1_penalty * l1

                if not torch.isfinite(loss):
                    continue

                running += float(loss.item())
                num_batches += 1

        return float("inf") if num_batches == 0 else running / num_batches

    def _predict(
        self,
        model: torch.nn.Module,
        X: np.ndarray,
        return_proba: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict categorical genotype labels from the VAE decoder logits.

        Diploid: K=3 -> {0=REF, 1=HET, 2=ALT}
        Haploid: K=2 -> {0=REF, 1=ALT}

        Args:
            model: Trained VAE model.
            X: 0/1/2 matrix with -1 for missing.
            return_proba: If True, also return probabilities.

        Returns:
            labels (N,L) and optionally probas (N,L,K).
        """
        if model is None:
            msg = "Model is not trained. Call fit() before predict()."
            self.logger.error(msg)
            raise NotFittedError(msg)

        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).to(self.device).long()

            if X_tensor.dim() != 2:
                raise ValueError(f"X must be 2D (N,L); got {tuple(X_tensor.shape)}")

            nF = int(X_tensor.shape[1])
            K = int(self.output_classes_)  # 2 or 3

            # (N, L, K)
            x_in = self._one_hot_encode_012(X_tensor, num_classes=K)

            out = model(x_in)
            if isinstance(out, (list, tuple)):
                recon_logits = out[0]
            else:
                recon_logits = out["recon_logits"]

            # Normalize recon_logits to (N, L, K)
            if recon_logits.dim() == 2 and recon_logits.shape[1] == nF * K:
                logits = recon_logits.view(-1, nF, K)
            elif recon_logits.dim() == 3:
                logits = recon_logits
                if logits.shape[1] != nF or logits.shape[2] != K:
                    raise ValueError(
                        f"Unexpected recon_logits shape {tuple(logits.shape)}; expected (N,{nF},{K})."
                    )
            else:
                raise ValueError(
                    f"Unexpected recon_logits dim={recon_logits.dim()} shape={tuple(recon_logits.shape)}"
                )

            probas = torch.softmax(logits, dim=-1)  # (N,L,K)
            labels = torch.argmax(probas, dim=-1)  # (N,L)

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
        *,
        y_true_matrix: np.ndarray | None = None,
        eval_mask_override: np.ndarray | None = None,
    ) -> Dict[str, float]:
        """Evaluate on 0/1/2; then IUPAC decoding and 10-base integer reports.

        This method evaluates the trained VAE model on a validation dataset, computing various performance metrics. It handles missing data appropriately and generates detailed classification reports for both the original 0/1/2 encoding and the decoded IUPAC and integer formats. The evaluation metrics are logged for review.

        Args:
            X_val (np.ndarray): Validation 0/1/2 matrix with -1 for missing.
            model (torch.nn.Module): Trained model.
            params (dict): Current hyperparameters (for logging).
            objective_mode (bool): If True, minimize logging for Optuna.
            latent_vectors_val (np.ndarray | None): Not used by VAE.
            y_true_matrix (np.ndarray | None): Optional GT 0/1/2 matrix for eval.
            eval_mask_override (np.ndarray | None): Optional mask to override default eval mask.

        Returns:
            Dict[str, float]: Computed metrics.

        Raises:
            NotFittedError: If called before fit().
            ValueError: If GT and X_val shapes do not match.
        """
        pred_labels, pred_probas = self._predict(
            model=model, X=X_val, return_proba=True
        )

        finite_mask = np.all(np.isfinite(pred_probas), axis=-1)  # (N, L)

        GT_ref = y_true_matrix

        # If not provided,
        # select from known aligned stores by exact shape match.
        if GT_ref is None:
            if (
                getattr(self, "_tune_ready", False)
                and getattr(self, "_tune_GT_test", None) is not None
            ):
                if X_val.shape == self._tune_GT_test.shape:
                    GT_ref = self._tune_GT_test
            if (
                GT_ref is None
                and getattr(self, "GT_test_full_", None) is not None
                and X_val.shape == self.GT_test_full_.shape
            ):
                GT_ref = self.GT_test_full_
            if (
                GT_ref is None
                and getattr(self, "GT_train_full_", None) is not None
                and X_val.shape == self.GT_train_full_.shape
            ):
                GT_ref = self.GT_train_full_

        # Hard fail if still unknown; do not “slice columns” or set GT_ref = X_val.
        if GT_ref is None or GT_ref.shape != X_val.shape:
            raise ValueError(
                f"Cannot align evaluation ground truth: X_val shape={X_val.shape}, "
                f"GT shape={(None if GT_ref is None else GT_ref.shape)}."
            )

        if eval_mask_override is not None:
            if eval_mask_override.shape != X_val.shape:
                msg = (
                    f"eval_mask_override shape {eval_mask_override.shape} "
                    f"does not match X_val shape {X_val.shape}."
                )
                self.logger.error(msg)
                raise ValueError(msg)

            eval_mask = eval_mask_override.astype(bool, copy=False)
            eval_mask = eval_mask & finite_mask & (GT_ref != -1)
        else:
            eval_mask = (X_val != -1) & finite_mask & (GT_ref != -1)

        y_true_flat = GT_ref[eval_mask].astype(np.int64, copy=False)
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
            proba_2[:, 1] = y_proba_flat[:, 1]
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
                pm = PrettyMetrics(
                    metrics, precision=3, title=f"{self.model_name} Validation Metrics"
                )
                pm.render()

            # Primary report
            self._make_class_reports(
                y_true=y_true_flat,
                y_pred_proba=y_proba_flat,
                y_pred=y_pred_flat,
                metrics=metrics,
                labels=target_names,
            )

            # IUPAC decode & 10-base integer report
            # FIX 4: Use current shape (X_val.shape) not self.num_features_
            y_true_dec = self.pgenc.decode_012(
                GT_ref.reshape(X_val.shape[0], X_val.shape[1])
            )
            X_pred = X_val.copy()
            X_pred[eval_mask] = y_pred_flat
            y_pred_dec = self.pgenc.decode_012(
                X_pred.reshape(X_val.shape[0], X_val.shape[1])
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
        """Optuna objective for VAE (no latent refinement during eval).

        This method defines the objective function for hyperparameter tuning using Optuna. It samples hyperparameters, trains the VAE model with these parameters, and evaluates its performance on a validation set. The evaluation metric specified by `self.tune_metric` is returned for optimization. If training fails, the trial is pruned to keep the tuning process efficient.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float: Value of the tuning metric to be optimized.

        Raises:
            optuna.exceptions.TrialPruned: If training fails unexpectedly or is unpromising.
            RuntimeError: If model training returns None.
        """
        try:
            self._prepare_tuning_artifacts()
            params = self._sample_hyperparameters(trial)

            # Use tune subsets when available (tune_fast)
            X_train = getattr(self, "_tune_X_train", None)
            X_val = getattr(self, "_tune_X_test", None)
            if X_train is None or X_val is None:
                X_train = getattr(self, "X_train_", self.ground_truth_[self.train_idx_])
                X_val = getattr(self, "X_val_", self.ground_truth_[self.test_idx_])

            if self.tune and self.tune_fast and getattr(self, "_tune_ready", False):
                train_loader = self._tune_loader
                class_weights = self._tune_class_weights
                X_train = self._tune_X_train
                X_val = self._tune_X_test
            else:
                class_weights = self._normalize_class_weights(
                    self._class_weights_from_zygosity(X_train)
                )
                train_loader = self._get_data_loader(X_train)

            # Always align model width to the data used in this trial
            params["model_params"]["n_features"] = int(X_train.shape[1])

            model = self.build_model(self.Model, params["model_params"])
            model.apply(self.initialize_weights)

            lr: float = params["lr"]
            l1_penalty: float = params["l1_penalty"]

            # Train + prune on metric
            res = self._train_and_validate_model(
                model=model,
                loader=train_loader,
                lr=lr,
                l1_penalty=l1_penalty,
                trial=trial,
                class_weights=class_weights,
                X_val=X_val,
                params=params,
                prune_metric=self.tune_metric,
                eval_interval=self.tune_eval_interval,
                eval_requires_latents=False,
                eval_latent_steps=0,
                eval_latent_lr=0.0,
                eval_latent_weight_decay=0.0,
                eval_mask_override=(
                    getattr(self, "_tune_sim_mask_test", None)
                    if self.simulate_missing
                    else None
                ),
            )
            model = res[1]

            # Prefer tune-aligned mask whenever tune artifacts exist
            eval_mask = (
                getattr(self, "_tune_sim_mask_test", None)
                if self.simulate_missing
                else None
            )

            # non-tune path fallback (full fit/val)
            if eval_mask is None:
                eval_mask = getattr(self, "sim_mask_test_", None)

            if model is None:
                msg = "Model training returned None."
                self.logger.error(msg)
                raise RuntimeError(msg)

            metrics = self._evaluate_model(
                X_val,
                model,
                params,
                objective_mode=True,
                y_true_matrix=getattr(self, "_tune_GT_test", None),
                eval_mask_override=eval_mask,
            )
            self._clear_resources(model, train_loader)
            return metrics[self.tune_metric]

        except Exception as e:
            # Unexpected failure: surface full details in logs while still
            # pruning the trial to keep sweeps moving.
            err_type = type(e).__name__
            self.logger.error(
                f"Trial {trial.number} failed due to exception {err_type}: {e}"
            )
            self.logger.debug(traceback.format_exc())
            raise optuna.exceptions.TrialPruned(
                f"Trial {trial.number} failed due to an exception. {err_type}: {e}. Enable debug logging for full traceback."
            ) from e

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """Sample VAE hyperparameters; hidden sizes mirror AE/NLPCA helper.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            Dict[str, int | float | str]: Sampled hyperparameters.
        """
        params = {
            "latent_dim": trial.suggest_int("latent_dim", 4, 16, step=2),
            "lr": trial.suggest_float("learning_rate", 3e-4, 1e-3, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.30, step=0.05),
            "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 6),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "elu", "selu", "leaky_relu"]
            ),
            "l1_penalty": trial.suggest_float("l1_penalty", 1e-6, 1e-3, log=True),
            "layer_scaling_factor": trial.suggest_float(
                "layer_scaling_factor", 2.0, 4.0, step=0.5
            ),
            "layer_schedule": trial.suggest_categorical(
                "layer_schedule", ["pyramid", "linear"]
            ),
            # VAE-specific β (final value after anneal)
            "beta": trial.suggest_float("beta", 0.5, 2.0, step=0.5),
            # focal gamma (if used in VAE recon CE)
            "gamma": trial.suggest_float("gamma", 0.5, 3.0, step=0.5),
        }

        use_n_features = (
            self._tune_num_features
            if (self.tune and self.tune_fast and hasattr(self, "_tune_num_features"))
            else self.num_features_
        )

        # will be set correctly after fit() prep
        K = int(getattr(self, "num_classes_", 3))
        input_dim = use_n_features * K

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=input_dim,
            n_outputs=input_dim,
            n_samples=(
                len(self._tune_train_idx)
                if (self.tune and self.tune_fast and hasattr(self, "_tune_train_idx"))
                else len(self.train_idx_)
            ),
            n_hidden=params["num_hidden_layers"],
            alpha=params["layer_scaling_factor"],
            schedule=params["layer_schedule"],
        )

        # [latent_dim] + interior widths (exclude output width)
        hidden_only = hidden_layer_sizes[1:-1]

        params["model_params"] = {
            "n_features": use_n_features,
            "num_classes": K,  # categorical head: 2 or 3
            "dropout_rate": params["dropout_rate"],
            "hidden_layer_sizes": hidden_only,
            "activation": params["activation"],
            # Pass through VAE recon/regularization coefficients
            "beta": params["beta"],
            "gamma": params["gamma"],
        }
        return params

    def _set_best_params(self, best_params: dict) -> dict:
        """Adopt best params and return VAE model_params.

        Args:
            best_params (dict): Best hyperparameters from tuning.

        Returns:
            dict: VAE model parameters.
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

        K = int(getattr(self, "num_classes_", 3))
        n_inputs = self.num_features_ * K
        n_outputs = self.num_features_ * K

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_samples=len(self.train_idx_),
            n_hidden=best_params["num_hidden_layers"],
            alpha=best_params["layer_scaling_factor"],
            schedule=best_params["layer_schedule"],
        )
        hidden_only = hidden_layer_sizes[1:-1]

        return {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": hidden_only,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "num_classes": K,
            "gamma": self.gamma,
        }

    def _default_best_params(self) -> Dict[str, int | float | str | list]:
        """Default VAE model params when tuning is disabled.

        Returns:
            Dict[str, int | float | str | list]: VAE model parameters.
        """
        K = int(getattr(self, "num_classes_", 3))
        n_inputs = self.num_features_ * K
        n_outputs = self.num_features_ * K

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_samples=len(self.ground_truth_),
            n_hidden=self.num_hidden_layers,
            alpha=self.layer_scaling_factor,
            schedule=self.layer_schedule,
        )
        hidden_only = hidden_layer_sizes[1:-1]

        return {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": hidden_only,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "num_classes": K,
            "beta": self.kl_beta_final,
            "gamma": self.gamma,
        }
