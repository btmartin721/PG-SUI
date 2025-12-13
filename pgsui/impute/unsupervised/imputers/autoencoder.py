import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

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
from pgsui.utils.logging_utils import configure_logger
from pgsui.utils.pretty_metrics import PrettyMetrics

if TYPE_CHECKING:
    from snpio import TreeParser
    from snpio.read_input.genotype_data import GenotypeData


def _make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    max_epochs: int,
    warmup_epochs: int,
    start_factor: float = 0.1,
) -> Any:
    """Create a warmup->cosine LR scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        max_epochs (int): Total number of epochs.
        warmup_epochs (int): Number of warmup epochs.
        start_factor (float): Starting LR factor for warmup.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: LR scheduler.
    """
    warmup_epochs = int(max(0, warmup_epochs))

    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=max_epochs)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=float(start_factor), total_iters=warmup_epochs
    )
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup_epochs))

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


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
        return load_yaml_to_dataclass(config, AutoencoderConfig)
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
        tree_parser: Optional["TreeParser"] = None,
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
            genotype_data ("GenotypeData"): Backing genotype data object.
            tree_parser (Optional["TreeParser"]): Optional SNPio phylogenetic tree parser for population-specific modes.
            config (Union["AutoencoderConfig", dict, str] | None): Structured configuration as dataclass, nested dict, YAML path, or None.
            overrides (dict | None): Optional dot-key overrides with highest precedence (e.g., {'model.latent_dim': 32}).
            simulate_missing (bool | None): Whether to simulate missing data during evaluation. If None, uses config default.
            sim_strategy (Literal["random", "random_weighted", "random_weighted_inv", "nonrandom", "nonrandom_weighted"] | None): Strategy for simulating missing data. If None, uses config default.
            sim_prop (float | None): Proportion of data to simulate as missing. If None, uses config default.
            sim_kwargs (dict | None): Additional keyword arguments for simulating missing data. If None, uses config default.
        """
        self.model_name = "ImputeAutoencoder"
        self.genotype_data = genotype_data
        self.tree_parser = tree_parser

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
        self.logger = configure_logger(
            logman.get_logger(),
            verbose=self.cfg.io.verbose,
            debug=self.cfg.io.debug,
        )

        # BaseNNImputer bootstrapping (device/dirs/logging handled here)
        super().__init__(
            model_name=self.model_name,
            genotype_data=self.genotype_data,
            prefix=self.cfg.io.prefix,
            device=self.cfg.train.device,
            verbose=self.cfg.io.verbose,
            debug=self.cfg.io.debug,
        )

        self.Model = AutoencoderModel

        # Model hook & encoder
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

        if self.tree_parser is None and self.sim_strategy.startswith("nonrandom"):
            msg = "tree_parser is required for nonrandom and nonrandom_weighted simulated missing strategies."
            self.logger.error(msg)
            raise ValueError(msg)

        # Model hyperparams
        self.latent_dim = int(self.cfg.model.latent_dim)
        self.dropout_rate = float(self.cfg.model.dropout_rate)
        self.num_hidden_layers = int(self.cfg.model.num_hidden_layers)
        self.layer_scaling_factor = float(self.cfg.model.layer_scaling_factor)
        self.layer_schedule: str = str(self.cfg.model.layer_schedule)
        self.activation = str(self.cfg.model.hidden_activation)
        gamma_raw = self.cfg.model.gamma
        if isinstance(gamma_raw, (list, tuple)):
            self.gamma = list(gamma_raw)
        else:
            self.gamma = float(gamma_raw)

        # Train hyperparams
        self.batch_size = int(self.cfg.train.batch_size)
        self.learning_rate = float(self.cfg.train.learning_rate)
        self.l1_penalty: float = float(self.cfg.train.l1_penalty)
        self.early_stop_gen = int(self.cfg.train.early_stop_gen)
        self.min_epochs = int(self.cfg.train.min_epochs)
        self.epochs = int(self.cfg.train.max_epochs)
        self.validation_split = float(self.cfg.train.validation_split)
        self.beta = float(self.cfg.train.weights_beta)
        self.max_ratio = float(self.cfg.train.weights_max_ratio)

        # Tuning
        self.tune = bool(self.cfg.tune.enabled)
        self.tune_fast = bool(self.cfg.tune.fast)
        self.tune_batch_size = int(self.cfg.tune.batch_size)
        self.tune_epochs = int(self.cfg.tune.epochs)
        self.tune_eval_interval = 1
        self.tune_metric = self.cfg.tune.metric
        self.tune_metric_: Literal[
            "pr_macro",
            "f1",
            "accuracy",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
        ] = (
            self.cfg.tune.metric or "f1"
        )

        self.n_trials = int(self.cfg.tune.n_trials)
        self.tune_save_db = bool(self.cfg.tune.save_db)
        self.tune_resume = bool(self.cfg.tune.resume)
        self.tune_max_samples = int(self.cfg.tune.max_samples)
        self.tune_max_loci = int(self.cfg.tune.max_loci)
        self.tune_infer_epochs = int(
            getattr(self.cfg.tune, "infer_epochs", 0)
        )  # AE unused
        self.tune_patience = int(self.cfg.tune.patience)

        # Evaluate
        # NOTE: AE does not optimize latents, so these are unused / fixed
        self.eval_latent_steps: int = 0
        self.eval_latent_lr: float = 0.0
        self.eval_latent_weight_decay: float = 0.0

        # Plotting (parity with NLPCA PlotConfig)
        self.plot_format: Literal["pdf", "png", "jpg", "jpeg", "svg"] = (
            self.cfg.plot.fmt
        )
        self.plot_dpi = int(self.cfg.plot.dpi)
        self.plot_fontsize = int(self.cfg.plot.fontsize)
        self.title_fontsize = int(self.cfg.plot.fontsize)
        self.despine = bool(self.cfg.plot.despine)
        self.show_plots = bool(self.cfg.plot.show)

        # Core derived at fit-time
        self.is_haploid: bool = False
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

        if self.genotype_data.snp_data is None:
            msg = "SNP data is required for Autoencoder imputer."
            self.logger.error(msg)
            raise TypeError(msg)

        # Ploidy & classes
        self.ploidy = self.cfg.io.ploidy
        self.is_haploid = self.ploidy == 1

        # Scoring labels == model output channels now
        self.num_classes_ = 2 if self.is_haploid else 3
        self.output_classes_ = self.num_classes_

        self.logger.info(
            f"Data is {'haploid' if self.is_haploid else 'diploid'}; "
            f"using {self.num_classes_} classes for scoring and {self.output_classes_} output channels."
        )

        # Collapse diploid ALT(2)->ALT(1) for haploids
        if self.is_haploid:
            self.ground_truth_[self.ground_truth_ == 2] = 1
            X_for_model[X_for_model == 2] = 1

        # After X_for_model is fully prepared (and after haploid collapsing)
        self.X_model_input_ = X_for_model
        n_samples, self.num_features_ = X_for_model.shape

        # Model params (decoder outputs L * K logits)
        self.model_params = {
            "n_features": self.num_features_,
            "num_classes": self.output_classes_,
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
            max_epochs=self.epochs,
            class_weights=self.class_weights_,
            X_val=self.X_val_,
            y_true_val=self.GT_test_full_,
            params=self.best_params_,
            prune_metric=self.tune_metric,
            prune_warmup_epochs=25,
            eval_interval=1,
            eval_requires_latents=False,
            eval_latent_steps=0,
            eval_latent_lr=0.0,
            eval_latent_weight_decay=0.0,
            eval_mask_override=getattr(self, "sim_mask_test_", None),
        )

        if trained_model is None:
            msg = "Autoencoder training failed; no model was returned."
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

        # Evaluate on validation set
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
            msg = "Model is not fitted. Call fit() before transform()."
            self.logger.error(msg)
            raise NotFittedError(msg)

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

    def _get_data_loaders(
        self,
        y: np.ndarray,
        *,
        shuffle: bool = True,
        batch_size: int | None = None,
    ) -> torch.utils.data.DataLoader:
        """Create DataLoader over indices + integer targets (-1 for missing).

        Args:
            y: 0/1/2 matrix with -1 for missing.
            shuffle: Whether to shuffle batches.
            batch_size: Optional override for batch size.

        Returns:
            Shuffled (or not) DataLoader.
        """
        y_tensor = torch.from_numpy(y).long()
        indices = torch.arange(len(y), dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(indices, y_tensor)
        pin_memory = self.device.type == "cuda"
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=int(batch_size or self.batch_size),
            shuffle=bool(shuffle),
            pin_memory=pin_memory,
        )

    def _train_and_validate_model(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        lr: float,
        l1_penalty: float,
        max_epochs: int,
        trial: optuna.Trial | None = None,
        class_weights: torch.Tensor | None = None,
        *,
        X_val: np.ndarray | None = None,
        y_true_val: np.ndarray | None = None,
        params: dict | None = None,
        prune_metric: str = "f1",  # "f1" | "accuracy" | "pr_macro"
        prune_warmup_epochs: int = 10,
        eval_interval: int = 1,
        # Evaluation parameters (AE ignores latent refinement knobs)
        eval_requires_latents: bool = False,  # AE: always False
        eval_latent_steps: int = 0,
        eval_latent_lr: float = 0.0,
        eval_latent_weight_decay: float = 0.0,
        eval_mask_override: np.ndarray | None = None,
    ) -> Tuple[float, torch.nn.Module | None, dict[str, list[float]]]:
        """Wrap the AE training loop (no latent optimizer), with Optuna pruning.

        This method orchestrates the training of the autoencoder model using the provided DataLoader. It sets up the optimizer and learning rate scheduler, and executes the training loop with support for early stopping and Optuna pruning based on validation performance. The method returns the best validation loss, the best model state, and optionally the training history.

        Args:
            model (torch.nn.Module): Autoencoder model.
            loader (torch.utils.data.DataLoader): Batches (indices, y_int) where y_int is 0/1/2; -1 for missing.
            lr (float): Learning rate.
            l1_penalty (float): L1 regularization coeff.
            max_epochs (int): Maximum training epochs.
            trial (optuna.Trial | None): Optuna trial for pruning (optional).
            class_weights (torch.Tensor | None): Class weights tensor (on device).
            X_val (np.ndarray | None): Validation matrix (0/1/2 with -1 for missing).
            params (dict | None): Model params for evaluation.
            prune_metric (str): Metric for pruning reports.
            prune_warmup_epochs (int): Pruning warmup epochs.
            eval_interval (int): Eval frequency (epochs).
            eval_requires_latents (bool): Ignored for AE (no latent inference).
            eval_latent_steps (int): Unused for AE.
            eval_latent_lr (float): Unused for AE.
            eval_latent_weight_decay (float): Unused for AE.
            eval_mask_override (np.ndarray | None): Optional eval mask to override default.

        Returns:
            Tuple[float, torch.nn.Module | None, dict[str, list[float]]]: (best_loss, best_model, history).
        """
        if class_weights is None:
            msg = "Must provide class_weights."
            self.logger.error(msg)
            raise TypeError(msg)

        if trial is not None and self.tune_fast:
            max_epochs = self.tune_epochs

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = _make_warmup_cosine_scheduler(
            optimizer,
            max_epochs=max_epochs,
            warmup_epochs=int(getattr(self, "lr_warmup_epochs", 5)),
        )

        best_loss, best_model, hist = self._execute_training_loop(
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            l1_penalty=l1_penalty,
            trial=trial,
            class_weights=class_weights,
            max_epochs=max_epochs,
            X_val=X_val,
            y_true_val=y_true_val,
            params=params,
            prune_metric=prune_metric,
            prune_warmup_epochs=prune_warmup_epochs,
            eval_interval=eval_interval,
            eval_requires_latents=False,  # AE: no latent inference
            eval_latent_steps=0,
            eval_latent_lr=0.0,
            eval_latent_weight_decay=0.0,
            eval_mask_override=eval_mask_override,
        )
        return best_loss, best_model, hist

    def _execute_training_loop(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        model: torch.nn.Module,
        l1_penalty: float,
        trial: optuna.Trial | None,
        class_weights: torch.Tensor,
        max_epochs: int,
        *,
        X_val: np.ndarray | None = None,
        y_true_val: np.ndarray | None = None,
        params: dict | None = None,
        prune_metric: str = "f1",
        prune_warmup_epochs: int = 10,
        eval_interval: int = 1,
        eval_requires_latents: bool = False,
        eval_latent_steps: int = 0,
        eval_latent_lr: float = 0.0,
        eval_latent_weight_decay: float = 0.0,
        eval_mask_override: np.ndarray | None = None,
        val_batch_size: int | None = None,
    ) -> tuple[float, torch.nn.Module, dict[str, list[float]]]:
        """Train AE with focal CE and validation-loss EarlyStopping + Optuna pruning.

        Args:
            loader (torch.utils.data.DataLoader): Batches (indices, y_int) where y_int is 0/1/2; -1 for missing.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler.
            model (torch.nn.Module): Autoencoder model.
            l1_penalty (float): L1 regularization coeff.
            trial (optuna.Trial | None): Optuna trial for pruning (optional).
            class_weights (torch.Tensor): Class weights tensor (on device).
            max_epochs (int): Maximum training epochs.
            X_val (np.ndarray | None): Validation matrix (0/1/2 with -1 for missing).
            y_true_val (np.ndarray | None): Ground-truth validation genotypes (0/1/2).
            params (dict | None): Model params for evaluation.
            prune_metric (str): Metric for pruning reports.
            prune_warmup_epochs (int): Pruning warmup epochs.
            eval_interval (int): Eval frequency (epochs).
            eval_requires_latents (bool): Ignored for AE (no latent inference).
            eval_latent_steps (int): Unused for AE.
            eval_latent_lr (float): Unused for AE.
            eval_latent_weight_decay (float): Unused for AE.
            eval_mask_override (np.ndarray | None): Optional eval mask to override default.
            val_batch_size (int | None): Optional validation batch size override.
        """
        history: dict[str, list[float]] = defaultdict(list)

        early_stopping = EarlyStopping(
            patience=self.early_stop_gen,
            min_epochs=self.min_epochs,
            verbose=self.verbose,
            prefix=self.prefix,
            debug=self.debug,
        )

        # Resolve gamma (prefer tuned)
        gamma_val = None
        if isinstance(params, dict):
            gamma_val = params.get("gamma", None)
            if gamma_val is None and isinstance(params.get("model_params", None), dict):
                gamma_val = params["model_params"].get("gamma", None)
        if gamma_val is None:
            gamma_val = getattr(self, "gamma", 0.0)

        if isinstance(gamma_val, (list, tuple)):
            if len(gamma_val) == 0:
                raise ValueError("gamma list is empty.")
            gamma_val = gamma_val[0]

        gamma_final = float(np.clip(float(gamma_val), 0.0, 10.0))
        gamma_warm, gamma_ramp = 50, 100  # keep as-is unless you want it proportional

        # If validation is provided, enforce y_true_val presence + shape agreement
        if X_val is not None:
            if y_true_val is None:
                raise ValueError(
                    "X_val was provided but y_true_val is None. Pass aligned ground-truth explicitly."
                )
            if y_true_val.shape != X_val.shape:
                raise ValueError(
                    f"Shape mismatch: X_val={X_val.shape}, y_true_val={y_true_val.shape}."
                )
            if (
                eval_mask_override is not None
                and eval_mask_override.shape != X_val.shape
            ):
                msg = f"eval_mask_override shape {eval_mask_override.shape} does not match X_val shape {X_val.shape}."
                self.logger.error(msg)
                raise ValueError(msg)

        for epoch in range(int(max_epochs)):
            # Focal gamma schedule
            if epoch < gamma_warm:
                model.gamma = 0.0  # type: ignore[attr-defined]
            elif epoch < gamma_warm + gamma_ramp:
                model.gamma = gamma_final * ((epoch - gamma_warm) / gamma_ramp)  # type: ignore[attr-defined]
            else:
                model.gamma = gamma_final  # type: ignore[attr-defined]

            # ---- TRAIN ----
            train_loss = self._train_step(
                loader=loader,
                optimizer=optimizer,
                model=model,
                l1_penalty=l1_penalty,
                class_weights=class_weights,
            )

            if not np.isfinite(train_loss):
                if trial is not None:
                    raise optuna.exceptions.TrialPruned("Training loss non-finite.")
                self.logger.debug(
                    "Non-finite training loss. Reducing LR by 10 percent and continuing."
                )
                for g in optimizer.param_groups:
                    g["lr"] *= 0.9
                continue

            # ---- VALIDATION ----
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

            scheduler.step()

            history["Train"].append(float(train_loss))

            if X_val is not None and val_loss is not None and np.isfinite(val_loss):
                history["Val"].append(float(val_loss))

            # Early stop on validation loss when available, else training loss
            es_score = val_loss if (X_val is not None) else train_loss
            early_stopping(es_score, model)

            if early_stopping.early_stop:
                self.logger.debug(f"Early stopping at epoch {epoch + 1}.")
                break

            # ---- OPTUNA PRUNING ----
            if (
                trial is not None
                and X_val is not None
                and ((epoch + 1) % int(max(1, eval_interval)) == 0)
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
                    y_true_matrix=y_true_val,
                    eval_mask_override=eval_mask_override,
                )
                trial.report(metric_val, step=epoch + 1)
                if (epoch + 1) >= prune_warmup_epochs and trial.should_prune():
                    raise optuna.exceptions.TrialPruned(
                        f"Pruned at epoch {epoch + 1}: {metric_key}={metric_val:.3f}"
                    )

        # Materialize the best model properly
        if early_stopping.best_model is not None:
            best_state = {
                k: v.cpu() for k, v in early_stopping.best_model.state_dict().items()
            }
            best_model = early_stopping.best_model
            best_model.load_state_dict(best_state)
            best_loss = float(early_stopping.best_score)
        else:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_model = model
            best_model.load_state_dict(best_state)
            best_loss = float(
                history["Val"][-1] if history["Val"] else history["Train"][-1]
            )

        return best_loss, best_model, dict(history)

    def _train_step(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        l1_penalty: float,
        class_weights: torch.Tensor,
    ) -> float:
        """One epoch of categorical focal CE with NaN/Inf guards."""
        model.train()
        running = 0.0
        num_batches = 0

        if class_weights is not None and class_weights.device != self.device:
            class_weights = class_weights.to(self.device)

        nF_model = int(getattr(model, "n_features", self.num_features_))
        nC_model = int(getattr(model, "num_classes", self.output_classes_))

        # Use model.gamma if present, else self.gamma
        gamma = float(getattr(model, "gamma", getattr(self, "gamma", 0.0)))
        gamma = float(torch.tensor(gamma).clamp(min=0.0, max=10.0))

        ce_criterion = SafeFocalCELoss(
            gamma=gamma,
            weight=class_weights,  # expects shape (C,)
            ignore_index=-1,
        )

        l1_params = tuple(p for p in model.parameters() if p.requires_grad)

        for _, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            y_batch = y_batch.to(self.device, non_blocking=True).long()

            if y_batch.dim() != 2:
                msg = f"Training batch expected 2D targets, got shape {tuple(y_batch.shape)}."
                self.logger.error(msg)
                raise ValueError(msg)

            if y_batch.shape[1] != nF_model:
                msg = (
                    f"Model expects {nF_model} loci but batch has {y_batch.shape[1]}. "
                    "Ensure tuning subsets and masks use matching loci columns."
                )
                self.logger.error(msg)
                raise ValueError(msg)

            # (B, L, C) where C is 2 (haploid) or 3 (diploid)
            x_in = self._one_hot_encode_012(y_batch, num_classes=nC_model)

            raw = model(x_in)

            logits_flat = raw if isinstance(raw, torch.Tensor) else raw[0]
            expected = (y_batch.shape[0], nF_model * nC_model)

            if logits_flat.dim() != 2 or tuple(logits_flat.shape) != expected:
                try:
                    logits_flat = logits_flat.view(-1, nF_model * nC_model)
                except Exception as e:
                    msg = f"Model output logits expected shape {expected}, got {tuple(logits_flat.shape)}. Ensure tuning subsets and masks use matching loci columns."
                    self.logger.error(msg)
                    raise ValueError(msg) from e

            logits = logits_flat.view(-1, nF_model, nC_model)
            logits_flat = logits.view(-1, nC_model)
            targets_flat = y_batch.view(-1)

            if not torch.isfinite(logits_flat).all():
                continue

            loss = ce_criterion(logits_flat, targets_flat)

            if l1_penalty > 0:
                l1 = torch.zeros((), device=self.device)
                for p in l1_params:
                    l1 = l1 + p.abs().sum()
                loss = loss + l1_penalty * l1

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
            return float("inf")
        return running / num_batches

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
        X: np.ndarray | torch.Tensor,
        return_proba: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict labels (and probabilities) from categorical logits.

        Diploid: returns {0,1,2} (REF,HET,ALT)
        Haploid: returns {0,1}   (REF,ALT)
        Missing in X (-1) is allowed; outputs are still produced but you typically mask later.
        """
        if model is None:
            msg = "Model is not trained. Call fit() before predict()."
            self.logger.error(msg)
            raise NotFittedError(msg)

        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
            X_tensor = X_tensor.to(self.device).long()

            if X_tensor.dim() != 2:
                raise ValueError(
                    f"X must be 2D (N,L); got shape {tuple(X_tensor.shape)}"
                )

            nF_x = int(X_tensor.shape[1])
            nF_model = int(getattr(model, "n_features", nF_x))
            if nF_model != nF_x:
                msg = (
                    f"Feature mismatch: model expects {nF_model} loci but X has {nF_x}."
                )
                self.logger.error(msg)
                raise ValueError(msg)

            nC_model = int(getattr(model, "num_classes", self.output_classes_))

            x_ohe = self._one_hot_encode_012(X_tensor, num_classes=nC_model)  # (N,L,C)
            raw = model(x_ohe)

            logits_flat = raw if isinstance(raw, torch.Tensor) else raw[0]
            expected = (x_ohe.shape[0], nF_model * nC_model)

            if logits_flat.dim() != 2 or tuple(logits_flat.shape) != expected:
                try:
                    logits_flat = logits_flat.view(-1, nF_model * nC_model)
                except Exception as e:
                    msg = f"Model output logits expected shape {expected}, got {tuple(logits_flat.shape)}. Ensure tuning subsets and masks use matching loci columns."
                    self.logger.error(msg)
                    raise ValueError(msg) from e

            logits = logits_flat.view(-1, nF_model, nC_model)

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
            y_true_matrix (np.ndarray | None): Optional ground truth matrix for eval.
            eval_mask_override (np.ndarray | None): Optional mask to override default eval mask.

        Returns:
            Dict[str, float]: Computed metrics.

        Raises:
            NotFittedError: If called before fit().
        """
        pred_labels, pred_probas = self._predict(
            model=model, X=X_val, return_proba=True
        )

        finite_mask = np.all(np.isfinite(pred_probas), axis=-1)  # (N, L)

        GT_ref = y_true_matrix

        # If not provided, select from known aligned stores by exact shape match.
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

        # Hard fail if still unknown;
        # do not “slice columns” or set GT_ref = X_val.
        if GT_ref is None or GT_ref.shape != X_val.shape:
            msg = f"Evaluation ground truth not found or shape mismatch: X_val shape={X_val.shape}, GT shape={(None if GT_ref is None else GT_ref.shape)}."
            self.logger.error(msg)
            raise ValueError(msg)

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
            self.tune_metric,  # type: ignore
        )

        if not objective_mode:
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
        """Optuna objective for AE; mirrors NLPCA study driver without latents.

        This method defines the objective function for hyperparameter tuning using Optuna. It samples hyperparameters, prepares the training and validation data, builds and trains the autoencoder model, and evaluates its performance on the validation set. The method returns the value of the tuning metric to be maximized.

        Args:
            trial (optuna.Trial): Optuna trial.

        Returns:
            float: Value of the tuning metric (maximize).
        """
        try:
            # Prepare tuning subsets (for tune_fast) once
            self._prepare_tuning_artifacts()

            # Sample hyperparameters (existing helper; unchanged signature)
            params = self._sample_hyperparameters(trial)

            # Optionally sub-sample for fast tuning (same keys used by NLPCA if you adopt them)
            if self.tune and self.tune_fast and getattr(self, "_tune_ready", False):
                X_train = self._tune_X_train
                X_val = self._tune_X_test
                train_loader = self._tune_loader
                class_weights = self._tune_class_weights
                # Ensure model aligns with subset width
                params["model_params"]["n_features"] = self._tune_num_features
            else:
                X_train = getattr(self, "X_train_", self.ground_truth_[self.train_idx_])
                X_val = getattr(self, "X_val_", self.ground_truth_[self.test_idx_])

                class_weights = self._normalize_class_weights(
                    self._class_weights_from_zygosity(X_train)
                )
                train_loader = self._get_data_loaders(X_train)

            # Always align model width to the data used in this trial
            params["model_params"]["n_features"] = int(X_train.shape[1])

            model = self.build_model(self.Model, params["model_params"])
            model.apply(self.initialize_weights)

            lr: float = float(params["lr"])
            l1_penalty: float = float(params["l1_penalty"])

            # Train + prune on metric
            _, model, __ = self._train_and_validate_model(
                model=model,
                loader=train_loader,
                lr=lr,
                l1_penalty=l1_penalty,
                max_epochs=self.tune_epochs,
                trial=trial,
                class_weights=class_weights,
                X_val=X_val,
                y_true_val=getattr(self, "_tune_GT_test", None),
                params=params,
                prune_metric=self.tune_metric,
                prune_warmup_epochs=25,
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
            # Keep sweeps moving if a trial fails
            raise optuna.exceptions.TrialPruned(f"Trial failed with error: {e}")

    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample AE hyperparameters and compute hidden sizes for model params.

        This method samples hyperparameters for the autoencoder model using Optuna's trial object. It computes the hidden layer sizes based on the sampled parameters and prepares the model parameters dictionary.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            Dict[str, int | float | str | bool]: Sampled hyperparameters and model_params.
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
            "gamma": trial.suggest_float("gamma", 0.0, 5.0, step=0.5),
        }

        nF = (
            self._tune_num_features
            if (self.tune and self.tune_fast and getattr(self, "_tune_ready", False))
            else self.num_features_
        )
        nC: int = int(getattr(self, "output_classes_", self.num_classes_ or 3))
        input_dim = nF * nC
        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=input_dim,
            n_outputs=input_dim,
            n_samples=(
                len(self._tune_train_idx)
                if (
                    self.tune and self.tune_fast and getattr(self, "_tune_ready", False)
                )
                else len(self.train_idx_)
            ),
            n_hidden=params["num_hidden_layers"],
            alpha=params["layer_scaling_factor"],
            schedule=params["layer_schedule"],
        )

        # [latent_dim] + interior widths (exclude output width)
        hidden_only = hidden_layer_sizes[1:-1]

        # Keep the latent_dim as the first element,
        # then the interior hidden widths.
        # If there are no interior widths (very small nets),
        # this still leaves [latent_dim].

        params["model_params"] = {
            "n_features": int(nF),
            "num_classes": int(
                getattr(self, "output_classes_", self.num_classes_ or 3)
            ),
            "latent_dim": int(params["latent_dim"]),
            "dropout_rate": float(params["dropout_rate"]),
            "hidden_layer_sizes": hidden_only,
            "activation": str(params["activation"]),
        }
        return params

    def _set_best_params(
        self, best_params: Dict[str, int | float | str | List[int]]
    ) -> Dict[str, int | float | str | List[int]]:
        """Adopt best params (ImputeNLPCA parity) and return model_params.

        This method sets the best hyperparameters found during tuning and computes the hidden layer sizes for the autoencoder model. It prepares the final model parameters dictionary to be used for building the model.

        Args:
            best_params (Dict[str, int | float | str | List[int]]): Best hyperparameters from tuning.

        Returns:
            Dict[str, int | float | str | List[int]]: Model parameters for building the model.
        """
        bp = {}
        for k, v in best_params.items():
            if not isinstance(v, list):
                if k in {"latent_dim", "num_hidden_layers"}:
                    bp[k] = int(v)
                elif k in {
                    "dropout_rate",
                    "learning_rate",
                    "l1_penalty",
                    "layer_scaling_factor",
                    "gamma",
                }:
                    bp[k] = float(v)
                elif k in {"activation", "layer_schedule"}:
                    if k == "layer_schedule":
                        if v not in {"pyramid", "constant", "linear"}:
                            raise ValueError(f"Invalid layer_schedule: {v}")
                        bp[k] = v
                    else:
                        bp[k] = str(v)
            else:
                bp[k] = v  # keep lists as-is

        if "gamma" in bp:
            self.gamma = float(bp["gamma"])

        self.latent_dim: int = bp["latent_dim"]
        self.dropout_rate: float = bp["dropout_rate"]
        self.learning_rate: float = bp["learning_rate"]
        self.l1_penalty: float = bp["l1_penalty"]
        self.activation: str = bp["activation"]
        self.layer_scaling_factor: float = bp["layer_scaling_factor"]
        self.layer_schedule: str = bp["layer_schedule"]

        nF: int = self.num_features_
        nC: int = int(getattr(self, "output_classes_", self.num_classes_ or 3))
        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=nF * nC,
            n_outputs=nF * nC,
            n_samples=len(self.train_idx_),
            n_hidden=bp["num_hidden_layers"],
            alpha=bp["layer_scaling_factor"],
            schedule=bp["layer_schedule"],
        )

        # Keep the latent_dim as the first element,
        # then the interior hidden widths.
        # If there are no interior widths (very small nets),
        # this still leaves [latent_dim].
        hidden_only = hidden_layer_sizes[1:-1]

        return {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": hidden_only,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "num_classes": nC,
        }

    def _default_best_params(self) -> Dict[str, int | float | str | list]:
        """Default model params when tuning is disabled.

        This method computes the default model parameters for the autoencoder when hyperparameter tuning is not performed. It calculates the hidden layer sizes based on the initial configuration.

        Returns:
            Dict[str, int | float | str | list]: Default model parameters.
        """
        nF: int = self.num_features_

        # Use the number of output channels passed to the model (3 for diploid)
        nC: int = int(getattr(self, "output_classes_", self.num_classes_ or 3))
        ls = self.layer_schedule

        if ls not in {"pyramid", "constant", "linear"}:
            raise ValueError(f"Invalid layer_schedule: {ls}")

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=nF * nC,
            n_outputs=nF * nC,
            n_samples=len(self.ground_truth_),
            n_hidden=self.num_hidden_layers,
            alpha=self.layer_scaling_factor,
            schedule=ls,
        )
        hidden_only = hidden_layer_sizes[1:-1]

        return {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": hidden_only,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "num_classes": nC,
        }
