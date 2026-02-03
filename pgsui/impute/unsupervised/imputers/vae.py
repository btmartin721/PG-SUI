from __future__ import annotations

import copy
import traceback
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from sklearn.exceptions import NotFittedError
from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.utils.logging import LoggerManager
from torch.optim.lr_scheduler import CosineAnnealingLR

from pgsui.data_processing.config import apply_dot_overrides, load_yaml_to_dataclass
from pgsui.data_processing.containers import VAEConfig
from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.loss_functions import FocalCELoss, compute_vae_loss
from pgsui.impute.unsupervised.models.vae_model import VAEModel
from pgsui.utils.logging_utils import configure_logger
from pgsui.utils.misc import OBJECTIVE_SPEC_VAE
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
) -> torch.optim.lr_scheduler.CosineAnnealingLR | torch.optim.lr_scheduler.SequentialLR:
    """Create a warmup->cosine LR scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        max_epochs (int): Total number of epochs for training.
        warmup_epochs (int): Number of warmup epochs.
        start_factor (float): Initial LR factor for warmup.

    Returns:
        torch.optim.lr_scheduler.CosineAnnealingLR | torch.optim.lr_scheduler.SequentialLR: The learning rate scheduler.
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


def ensure_vae_config(config: VAEConfig | dict | str | None) -> VAEConfig:
    """Ensure a VAEConfig instance from various input types.

    Args:
        config (VAEConfig | dict | str | None): Configuration input.

    Returns:
        VAEConfig: The resulting VAEConfig instance.
    """
    if config is None:
        return VAEConfig()
    if isinstance(config, VAEConfig):
        return config
    if isinstance(config, str):
        return load_yaml_to_dataclass(config, VAEConfig)
    if isinstance(config, dict):
        cfg_in = copy.deepcopy(config)
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
    """Variational Autoencoder (VAE) imputer for 0/1/2 genotypes.

    Trains a feedforward autoencoder on a genotype matrix encoded as 0/1/2 with missing values represented by any negative integer. Missingness is simulated once on the full matrix, then train/val/test splits reuse those masks. It supports haploid and diploid data, focal-CE reconstruction loss (optional scheduling), and Optuna-based hyperparameter tuning. Output is returned as IUPAC strings via ``decode_012``.

    Notes:
        - Simulates missingness once on the full 0/1/2 matrix, then splits indices on clean ground truth.
        - Maintains clean targets and corrupted inputs per train/val/test, plus per-split masks.
        - Haploid harmonization happens after the single simulation (no re-simulation).
        - Training/validation loss is computed only where targets are known (~orig_mask_*).
        - Evaluation is computed only on simulated-missing sites (sim_mask_*).
        - ``transform()`` fills only originally missing sites and hard-errors if decoding yields "N".
    """

    # Helper (small, used for robust pruning on CUDA OOM / runtime blowups)
    def _maybe_prune_or_raise_runtime(
        self,
        exc: Exception,
        *,
        context: str,
        trial: Optional[optuna.Trial],
    ) -> None:
        """Either prune an Optuna trial or raise a RuntimeError with context.

        Args:
            exc (Exception): The caught exception.
            context (str): Short description of where the error occurred.
            trial (Optional[optuna.Trial]): Active Optuna trial, if any.

        Raises:
            optuna.exceptions.TrialPruned: If trial is not None.
            RuntimeError: Otherwise.
        """
        msg = f"[{self.model_name}] {context}: {type(exc).__name__}: {exc}"
        # Common CUDA OOM signature; treat as prune during tuning.
        if trial is not None:
            self.logger.warning(msg)
            raise optuna.exceptions.TrialPruned(msg) from exc
        self.logger.error(msg)
        raise RuntimeError(msg) from exc

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        tree_parser: Optional["TreeParser"] = None,
        config: Optional[Union["VAEConfig", dict, str]] = None,
        overrides: Optional[dict] = None,
        sim_strategy: Literal[
            "random",
            "random_weighted",
            "random_weighted_inv",
            "nonrandom",
            "nonrandom_weighted",
        ] = "random",
        sim_prop: Optional[float] = None,
        sim_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize the ImputeVAE imputer.

        Adds robustness checks for:
            - missing required genotype_data attributes
            - invalid sim_strategy / sim_prop
            - nonrandom strategies requiring tree_parser
            - misconfigured config fields used in logging/training

        Args:
            genotype_data (GenotypeData): Genotype data for imputation.
            tree_parser (Optional[TreeParser]): Tree parser required for nonrandom strategies.
            config (Optional[Union[VAEConfig, dict, str]]): Config dataclass, nested dict, YAML path, or None.
            overrides (Optional[dict]): Dot-key overrides applied last with highest precedence.
            sim_strategy: Missingness simulation strategy (overrides config).
            sim_prop (Optional[float]): Proportion of entries to simulate as missing (overrides config).
            sim_kwargs (Optional[dict]): Extra missingness kwargs merged into config.

        Raises:
            AttributeError: If genotype_data is missing required attributes.
            ValueError: If nonrandom strategy without tree_parser, or invalid sim_prop.
            TypeError: If config type invalid (via ensure_vae_config).
        """
        self.model_name = "ImputeVAE"
        self.genotype_data = genotype_data
        self.tree_parser = tree_parser

        # --- Defensive: ensure genotype_data has minimally required attributes ---
        if genotype_data is None:
            raise TypeError("genotype_data cannot be None.")
        # Many of your downstream calls assume these exist.
        missing_attrs = [a for a in ("snp_data",) if not hasattr(genotype_data, a)]
        if missing_attrs:
            raise AttributeError(
                f"genotype_data is missing required attribute(s): {missing_attrs}"
            )

        cfg = ensure_vae_config(config)
        if overrides:
            if not isinstance(overrides, dict):
                raise TypeError(
                    f"overrides must be a dict or None; got {type(overrides).__name__}"
                )
            cfg = apply_dot_overrides(cfg, overrides)
        self.cfg = cfg

        # --- Validate config fields we immediately depend on ---
        # If these are missing due to a malformed YAML, fail early with clarity.
        try:
            _prefix = self.cfg.io.prefix
            _debug = self.cfg.io.debug
            _verbose = self.cfg.io.verbose
        except Exception as e:
            raise AttributeError(
                "VAEConfig is missing required io fields (prefix/debug/verbose)."
            ) from e

        logman = LoggerManager(
            __name__,
            prefix=self.cfg.io.prefix,
            debug=self.cfg.io.debug,
            verbose=self.cfg.io.verbose,
        )
        self.logger = configure_logger(
            logman.get_logger(), verbose=self.cfg.io.verbose, debug=self.cfg.io.debug
        )
        self.logger.propagate = False

        super().__init__(
            model_name=self.model_name,
            genotype_data=self.genotype_data,
            prefix=self.cfg.io.prefix,
            device=self.cfg.train.device,
            verbose=self.cfg.io.verbose,
            debug=self.cfg.io.debug,
        )

        self.Model = VAEModel
        self.pgenc = GenotypeEncoder(genotype_data)

        # I/O and general parameters
        self.seed = self.cfg.io.seed
        self.n_jobs = self.cfg.io.n_jobs
        self.prefix = self.cfg.io.prefix
        self.scoring_averaging = self.cfg.io.scoring_averaging
        self.verbose = self.cfg.io.verbose
        self.debug = self.cfg.io.debug
        self.rng = np.random.default_rng(self.seed)

        # Simulation parameters
        sim_cfg = getattr(self.cfg, "sim", None)
        sim_cfg_kwargs = copy.deepcopy(getattr(sim_cfg, "sim_kwargs", None) or {})
        if sim_kwargs:
            if not isinstance(sim_kwargs, dict):
                raise TypeError(
                    f"sim_kwargs must be a dict or None; got {type(sim_kwargs).__name__}"
                )
            sim_cfg_kwargs.update(sim_kwargs)

        if sim_cfg is None:
            default_strategy = "random"
            default_prop = 0.2
        else:
            default_strategy = sim_cfg.sim_strategy
            default_prop = sim_cfg.sim_prop

        self.simulate_missing = True
        self.sim_strategy = sim_strategy or default_strategy

        # Validate sim_strategy (keep core functionality; just guard)
        allowed_strats = {
            "random",
            "random_weighted",
            "random_weighted_inv",
            "nonrandom",
            "nonrandom_weighted",
        }
        if self.sim_strategy not in allowed_strats:
            raise ValueError(
                f"Invalid sim_strategy='{self.sim_strategy}'. Must be one of {sorted(allowed_strats)}."
            )

        # Validate sim_prop
        prop = float(sim_prop if sim_prop is not None else default_prop)
        if not np.isfinite(prop):
            raise ValueError(f"sim_prop must be finite; got {prop}.")
        if prop <= 0.0 or prop >= 1.0:
            raise ValueError(f"sim_prop must be in (0, 1); got {prop}.")
        self.sim_prop = prop
        self.sim_kwargs = sim_cfg_kwargs

        # Model architecture parameters
        self.latent_dim = self.cfg.model.latent_dim
        self.dropout_rate = self.cfg.model.dropout_rate
        self.num_hidden_layers = self.cfg.model.num_hidden_layers
        self.layer_scaling_factor = self.cfg.model.layer_scaling_factor
        self.layer_schedule = self.cfg.model.layer_schedule
        self.activation = self.cfg.model.activation

        # VAE-specific parameters
        self.kl_beta = self.cfg.vae.kl_beta
        self.kl_beta_schedule = self.cfg.vae.kl_beta_schedule

        # Training parameters
        self.power: float = self.cfg.train.weights_power
        self.max_ratio: Optional[float] = self.cfg.train.weights_max_ratio
        self.normalize: bool = self.cfg.train.weights_normalize
        self.inverse: bool = self.cfg.train.weights_inverse
        self.batch_size = self.cfg.train.batch_size
        self.learning_rate = self.cfg.train.learning_rate
        self.l1_penalty: float = self.cfg.train.l1_penalty
        self.early_stop_gen = self.cfg.train.early_stop_gen
        self.min_epochs = self.cfg.train.min_epochs
        self.epochs = self.cfg.train.max_epochs
        self.validation_split = self.cfg.train.validation_split
        self.gamma = self.cfg.train.gamma
        self.gamma_schedule = self.cfg.train.gamma_schedule

        # Defensive validation (no behavior changes—just clearer early failures)
        if int(self.epochs) <= 0:
            raise ValueError(f"max_epochs must be > 0; got {self.epochs}.")
        if int(self.min_epochs) < 0:
            raise ValueError(f"min_epochs must be >= 0; got {self.min_epochs}.")
        if int(self.min_epochs) > int(self.epochs):
            self.logger.warning(
                f"min_epochs ({self.min_epochs}) > max_epochs ({self.epochs}); clipping min_epochs to max_epochs."
            )
            self.min_epochs = int(self.epochs)
        if int(self.early_stop_gen) <= 0:
            raise ValueError(f"early_stop_gen must be > 0; got {self.early_stop_gen}.")
        if self.batch_size is None or int(self.batch_size) <= 0:
            raise ValueError(f"batch_size must be > 0; got {self.batch_size}.")
        if (
            not np.isfinite(float(self.learning_rate))
            or float(self.learning_rate) <= 0.0
        ):
            raise ValueError(
                f"learning_rate must be finite and > 0; got {self.learning_rate}."
            )
        if float(self.l1_penalty) < 0.0 or not np.isfinite(float(self.l1_penalty)):
            raise ValueError(
                f"l1_penalty must be finite and >= 0; got {self.l1_penalty}."
            )
        if not (0.0 < float(self.validation_split) < 1.0):
            raise ValueError(
                f"validation_split must be in (0,1); got {self.validation_split}."
            )

        # Tuning parameters
        self.tune = self.cfg.tune.enabled
        self.tune_metric: str | list[str] | tuple[str, ...]
        self.tune_metric = self.cfg.tune.metrics
        self.primary_metric = self.validate_tuning_metric()

        self.n_trials = self.cfg.tune.n_trials
        self.tune_save_db = self.cfg.tune.save_db
        self.tune_resume = self.cfg.tune.resume
        self.tune_patience = self.cfg.tune.patience

        # Plotting parameters
        self.plot_format = self.cfg.plot.fmt
        self.plot_dpi = self.cfg.plot.dpi
        self.plot_fontsize = self.cfg.plot.fontsize
        self.title_fontsize = self.cfg.plot.fontsize
        self.despine = self.cfg.plot.despine
        self.show_plots = self.cfg.plot.show
        self.use_multiqc = bool(self.cfg.plot.multiqc)

        # Internal attributes set during fitting
        self.is_haploid_: bool = False
        self.num_classes_: int = 3
        self.model_params: dict[str, Any] = {}
        self.sim_mask_test_: np.ndarray

        if self.tree_parser is None and str(self.sim_strategy).startswith("nonrandom"):
            msg = "tree_parser is required for nonrandom sim strategies."
            self.logger.error(msg)
            raise ValueError(msg)

        self.num_tuned_params_ = OBJECTIVE_SPEC_VAE.count()

    def fit(self) -> "ImputeVAE":
        """Fit the VAE imputer model to the genotype data.

        Adds robustness checks for:
            - empty / degenerate genotype matrices
            - NaN/Inf in encoded matrices
            - empty train/val/test splits
            - empty evaluation masks (no sites to score)
            - device / tensor conversion issues
            - save-path failures

        Returns:
            ImputeVAE: The fitted ImputeVAE instance.

        Raises:
            AttributeError, ValueError, RuntimeError: On invalid inputs or failed training.
        """
        self.logger.info(f"Fitting {self.model_name} model...")

        if getattr(self.genotype_data, "snp_data", None) is None:
            msg = f"SNP data is required for {self.model_name}."
            self.logger.error(msg)
            raise AttributeError(msg)

        self.ploidy = self.cfg.io.ploidy
        self.is_haploid_ = self.ploidy == 1

        if self.ploidy > 2 or self.ploidy < 1:
            msg = (
                f"{self.model_name} currently supports only haploid (1) or diploid (2) data; "
                f"got ploidy={self.ploidy}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.debug(
            f"Ploidy set to {self.ploidy}, is_haploid: {self.is_haploid_}"
        )
        self.num_classes_ = 2 if self.is_haploid_ else 3

        # --- Encode / validate ground truth ---
        gt_full = self.pgenc.genotypes_012.copy()
        if not isinstance(gt_full, np.ndarray) or gt_full.ndim != 2:
            msg = f"Expected pgenc.genotypes_012 to be a 2D numpy array; got {type(gt_full).__name__} with ndim={getattr(gt_full, 'ndim', None)}."
            self.logger.error(msg)
            raise ValueError(msg)

        if gt_full.shape[0] == 0 or gt_full.shape[1] == 0:
            msg = f"{self.model_name} received an empty genotype matrix with shape {gt_full.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        # Standardize missing -> -1 and remove NaN/Inf safely
        gt_full = np.nan_to_num(gt_full, nan=-1.0, posinf=-1.0, neginf=-1.0)
        gt_full[gt_full < 0] = -1
        self.ground_truth_ = gt_full.astype(np.int8, copy=False)
        self.num_features_ = int(self.ground_truth_.shape[1])

        if self.num_features_ < 2:
            msg = f"{self.model_name} requires at least 2 loci/features; got {self.num_features_}."
            self.logger.error(msg)
            raise ValueError(msg)

        self.model_params = {
            "n_features": self.num_features_,
            "num_classes": self.num_classes_,
            "latent_dim": self.latent_dim,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
        }
        self.logger.debug(f"Model parameters: {self.model_params}")

        # Simulate missingness on the full matrix
        sim_tup = self.sim_missing_transform(self.ground_truth_)
        if not isinstance(sim_tup, (tuple, list)) or len(sim_tup) < 3:
            msg = f"sim_missing_transform() must return (X_corrupted, sim_mask, orig_mask); got {type(sim_tup).__name__}."
            self.logger.error(msg)
            raise RuntimeError(msg)

        X_for_model_full = sim_tup[0]
        self.sim_mask_ = sim_tup[1]
        self.orig_mask_ = sim_tup[2]

        # Basic mask sanity (shape/dtype) before deeper validation
        for nm, m in (("sim_mask_", self.sim_mask_), ("orig_mask_", self.orig_mask_)):
            if not isinstance(m, np.ndarray):
                raise TypeError(f"{nm} must be a numpy array; got {type(m).__name__}.")
            if m.shape != self.ground_truth_.shape:
                raise ValueError(
                    f"{nm} shape mismatch: expected {self.ground_truth_.shape}, got {m.shape}."
                )
            if m.dtype != bool:
                # keep behavior; just coerce defensively
                self.logger.warning(f"{nm} dtype is {m.dtype}; coercing to bool.")
                m = m.astype(bool, copy=False)
                if nm == "sim_mask_":
                    self.sim_mask_ = m
                else:
                    self.orig_mask_ = m

        # Validate sim and orig masks; there should not be any overlap.
        self._validate_sim_and_orig_masks(
            sim_mask=self.sim_mask_, orig_mask=self.orig_mask_, context="full"
        )

        # Split indices based on clean ground truth
        indices = self._train_val_test_split(self.ground_truth_)
        self.train_idx_, self.val_idx_, self.test_idx_ = indices

        # Split robustness: ensure non-empty
        if (
            len(self.train_idx_) == 0
            or len(self.val_idx_) == 0
            or len(self.test_idx_) == 0
        ):
            msg = (
                f"{self.model_name} produced an empty split: "
                f"train={len(self.train_idx_)}, val={len(self.val_idx_)}, test={len(self.test_idx_)}. "
                "Check dataset size and validation_split."
            )
            self.logger.error(msg)
            raise RuntimeError(msg)

        self.logger.info(
            f"Train/val/test sizes: {len(self.train_idx_)}/{len(self.val_idx_)}/{len(self.test_idx_)}"
        )

        # --- Split matrices ---
        X_train_corrupted, X_val_corrupted, X_test_corrupted = (
            self._extract_masks_indices(X_for_model_full, indices)
        )
        X_train_clean, X_val_clean, X_test_clean = self._extract_masks_indices(
            self.ground_truth_, indices
        )

        # --- Masks per split ---
        self.sim_mask_train_, self.sim_mask_val_, self.sim_mask_test_ = (
            self._extract_masks_indices(self.sim_mask_, indices)
        )
        self.orig_mask_train_, self.orig_mask_val_, self.orig_mask_test_ = (
            self._extract_masks_indices(self.orig_mask_, indices)
        )

        self.validate_and_log_masks()

        self.eval_mask_train_ = self.sim_mask_train_ & ~self.orig_mask_train_
        self.eval_mask_val_ = self.sim_mask_val_ & ~self.orig_mask_val_
        self.eval_mask_test_ = self.sim_mask_test_ & ~self.orig_mask_test_

        self.validate_and_log_masks()

        # Ensure we have something to evaluate on at least in val/test
        if not np.any(self.eval_mask_val_):
            self.logger.warning(
                f"[{self.model_name}] eval_mask_val_ has no True entries; validation metrics will be 0.0."
            )
        if not np.any(self.eval_mask_test_):
            self.logger.warning(
                f"[{self.model_name}] eval_mask_test_ has no True entries; test metrics will be 0.0."
            )

        # --- Haploid harmonization (do NOT resimulate; just recode values) ---
        if self.is_haploid_:
            self.logger.debug(
                "Performing haploid harmonization on split inputs/targets..."
            )

            def _haploidize(arr: np.ndarray) -> np.ndarray:
                out = np.asarray(arr).copy()
                miss = out < 0
                out = np.where(out > 0, 1, out).astype(np.int8, copy=False)
                out[miss] = -1
                return out

            X_train_clean = _haploidize(X_train_clean)
            X_val_clean = _haploidize(X_val_clean)
            X_test_clean = _haploidize(X_test_clean)
            X_train_corrupted = _haploidize(X_train_corrupted)
            X_val_corrupted = _haploidize(X_val_corrupted)
            X_test_corrupted = _haploidize(X_test_corrupted)

        # Persist versions
        self.X_train_clean_ = X_train_clean
        self.X_val_clean_ = X_val_clean
        self.X_test_clean_ = X_test_clean
        self.X_train_corrupted_ = X_train_corrupted
        self.X_val_corrupted_ = X_val_corrupted
        self.X_test_corrupted_ = X_test_corrupted

        # Final training tensors/matrices used by the pipeline
        self.X_train_ = self.X_train_corrupted_
        self.y_train_ = self.X_train_clean_
        self.X_val_ = self.X_val_corrupted_
        self.y_val_ = self.X_val_clean_
        self.y_test_ = self.X_test_clean_

        # One-hot encode inputs (corrupted)
        self.X_train_ = self._one_hot_encode_012(
            self.X_train_, num_classes=self.num_classes_
        )
        self.X_val_ = self._one_hot_encode_012(
            self.X_val_, num_classes=self.num_classes_
        )

        for name, tensor in [("X_train_", self.X_train_), ("X_val_", self.X_val_)]:
            if not torch.is_tensor(tensor):
                msg = f"[{self.model_name}] Expected {name} to be a torch.Tensor after one-hot encoding."
                self.logger.error(msg)
                raise RuntimeError(msg)
            if tensor.numel() == 0:
                msg = f"[{self.model_name}] {name} is empty after one-hot encoding."
                self.logger.error(msg)
                raise RuntimeError(msg)
            if (tensor.sum(dim=-1) > 1).any():
                msg = f"[{self.model_name}] Invalid one-hot: >1 active class in {name}."
                self.logger.error(msg)
                raise RuntimeError(msg)

        self.plotter_, self.scorers_ = self.initialize_plotting_and_scorers()

        # Data loaders expect numpy arrays; force CPU materialization safely
        try:
            Xtr_np = self.X_train_.numpy(force=True)
            Xva_np = self.X_val_.numpy(force=True)

            if Xtr_np.ndim == 3:
                Xtr_np = Xtr_np.reshape(
                    Xtr_np.shape[0], Xtr_np.shape[1] * Xtr_np.shape[2]
                )
            if Xva_np.ndim == 3:
                Xva_np = Xva_np.reshape(
                    Xva_np.shape[0], Xva_np.shape[1] * Xva_np.shape[2]
                )

        except Exception as e:
            msg = f"[{self.model_name}] Failed to convert tensors to numpy and reshape for dataloaders: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg) from e

        train_loader = self._get_data_loaders(
            Xtr_np, self.y_train_, self.eval_mask_train_, self.batch_size, shuffle=True
        )
        val_loader = self._get_data_loaders(
            Xva_np, self.y_val_, self.eval_mask_val_, self.batch_size, shuffle=False
        )
        self.train_loader_ = train_loader
        self.val_loader_ = val_loader

        # Ensure loaders are not trivially empty
        if (
            getattr(self.train_loader_, "__len__", None) is not None
            and len(self.train_loader_) == 0
        ):
            msg = f"[{self.model_name}] train_loader_ is empty."
            self.logger.error(msg)
            raise RuntimeError(msg)
        if (
            getattr(self.val_loader_, "__len__", None) is not None
            and len(self.val_loader_) == 0
        ):
            self.logger.warning(
                f"[{self.model_name}] val_loader_ is empty; validation loss will be undefined."
            )

        if self.tune:
            self.tuned_params_ = self.tune_hyperparameters()
            self.model_tuned_ = True
        else:
            self.model_tuned_ = False
            self.class_weights_ = self._class_weights_from_zygosity(
                self.y_train_,
                train_mask=self.eval_mask_train_,
                inverse=self.inverse,
                normalize=self.normalize,
                max_ratio=self.max_ratio,
                power=self.power,
            )
            keys = OBJECTIVE_SPEC_VAE.keys
            self.tuned_params_ = {k: getattr(self, k) for k in keys}
            self.tuned_params_["model_params"] = self.model_params

        self.best_params_ = copy.deepcopy(self.tuned_params_)
        self._log_class_weights()

        model_params_final = {
            "n_features": self.num_features_,
            "num_classes": self.num_classes_,
            "latent_dim": int(self.best_params_["latent_dim"]),
            "dropout_rate": float(self.best_params_["dropout_rate"]),
            "activation": str(self.best_params_["activation"]),
            "kl_beta": float(
                self.best_params_.get("kl_beta", getattr(self, "kl_beta", 1.0))
            ),
        }

        input_dim = self.num_features_ * self.num_classes_
        model_params_final["hidden_layer_sizes"] = self._compute_hidden_layer_sizes(
            n_inputs=input_dim,
            n_outputs=self.num_classes_,
            n_samples=len(self.X_train_),
            n_hidden=int(self.best_params_["num_hidden_layers"]),
            latent_dim=int(self.best_params_["latent_dim"]),
            alpha=float(self.best_params_["layer_scaling_factor"]),
            schedule=str(self.best_params_["layer_schedule"]),
            min_size=max(16, 2 * int(self.best_params_["latent_dim"])),
        )

        self.best_params_["model_params"] = model_params_final

        model = self.build_model(self.Model, self.best_params_["model_params"])
        model.apply(self.initialize_weights)

        if self.verbose or self.debug:
            self.logger.info("Using model hyperparameters:")
            pm = PrettyMetrics(
                self.best_params_, precision=3, title="Model Hyperparameters"
            )
            pm.render()

        lr_final = float(self.best_params_["learning_rate"])
        l1_final = float(self.best_params_["l1_penalty"])

        loss, trained_model, history = self._train_and_validate_model(
            model=model,
            lr=lr_final,
            l1_penalty=l1_final,
            params=self.best_params_,
            trial=None,
            class_weights=self.class_weights_,
            kl_beta_schedule=self.best_params_["kl_beta_schedule"],
            gamma_schedule=self.best_params_["gamma_schedule"],
        )

        if trained_model is None:
            msg = f"{self.model_name} training failed."
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Save model with robust path handling
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                trained_model.state_dict(),
                self.models_dir / f"final_model_{self.model_name}.pt",
            )
        except Exception as e:
            msg = f"[{self.model_name}] Failed to save model state_dict: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg) from e

        if history is None:
            hist = {"Train": []}
        else:
            hist = (
                dict(history)
                if isinstance(history, dict)
                else {"Train": list(history["Train"]), "Val": list(history["Val"])}
            )
        self.history_ = hist

        self.best_loss_ = float(loss)
        self.model_ = trained_model
        self.is_fit_ = True

        self._evaluate_model(
            self.model_,
            X=self.X_test_corrupted_,
            y=self.y_test_,
            eval_mask=self.eval_mask_test_,
            objective_mode=False,
        )

        if self.show_plots:
            # avoid crashes if plotter/history are malformed
            try:
                self.plotter_.plot_history(self.history_)
            except Exception as e:
                self.logger.warning(f"[{self.model_name}] plot_history failed: {e}")

        self._save_display_model_params(is_tuned=self.model_tuned_)
        self.logger.info(f"{self.model_name} fitting complete!")
        return self

    def transform(self) -> np.ndarray:
        """Impute missing genotypes and return IUPAC strings.

        Adds robustness checks for:
            - presence of fitted model + ground_truth_
            - prediction shape alignment
            - remaining missing values after filling
            - decode failures

        Returns:
            np.ndarray: IUPAC genotype matrix of shape (n_samples, n_loci).

        Raises:
            NotFittedError: If called before fit().
            RuntimeError: If imputation or decoding is inconsistent.
        """
        if not getattr(self, "is_fit_", False):
            msg = f"{self.model_name} is not fitted. Must call 'fit()' before 'transform()'."
            self.logger.error(msg)
            raise NotFittedError(msg)

        if getattr(self, "model_", None) is None:
            msg = f"{self.model_name}.model_ is missing; fit() did not complete successfully."
            self.logger.error(msg)
            raise NotFittedError(msg)

        if getattr(self, "ground_truth_", None) is None:
            msg = f"{self.model_name}.ground_truth_ is missing; cannot transform."
            self.logger.error(msg)
            raise RuntimeError(msg)

        self.logger.info(f"Imputing entire dataset with {self.model_name}...")
        X_to_impute = np.asarray(self.ground_truth_).copy()

        if X_to_impute.ndim != 2:
            raise ValueError(
                f"ground_truth_ must be 2D; got shape {X_to_impute.shape}."
            )

        # 1. Predict labels (0/1/2) for the entire matrix
        pred_labels, _ = self._predict(self.model_, X=X_to_impute)

        if pred_labels.shape != X_to_impute.shape:
            msg = (
                f"[{self.model_name}] Prediction shape mismatch: "
                f"pred_labels={pred_labels.shape} vs X={X_to_impute.shape}."
            )
            self.logger.error(msg)
            raise RuntimeError(msg)

        # 2. Fill ONLY originally missing values
        missing_mask = X_to_impute < 0
        imputed_array = X_to_impute.copy()
        imputed_array[missing_mask] = pred_labels[missing_mask]

        if np.any(imputed_array < 0):
            msg = f"[{self.model_name}] Some missing genotypes remain after imputation. This is unexpected."
            self.logger.error(msg)
            raise RuntimeError(msg)

        # 3. Handle Haploid mapping (2->1) before decoding if needed
        decode_input = imputed_array
        if getattr(self, "is_haploid_", False):
            decode_input = imputed_array.copy()
            decode_input[decode_input == 1] = 2

        # 4. Decode integers to IUPAC strings
        imputed_gt = self.decode_012(decode_input)

        if not isinstance(imputed_gt, np.ndarray):
            raise RuntimeError(
                f"decode_012 must return a numpy array; got {type(imputed_gt).__name__}."
            )

        if (imputed_gt == "N").any():
            msg = f"Something went wrong: {self.model_name} imputation still contains {int((imputed_gt == 'N').sum())} missing values ('N')."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if self.show_plots:
            try:
                original_input = X_to_impute
                if getattr(self, "is_haploid_", False):
                    original_input = X_to_impute.copy()
                    original_input[original_input == 1] = 2

                plt.rcParams.update(self.plotter_.param_dict)
                orig_dec = self.decode_012(original_input)
                self.plotter_.plot_gt_distribution(imputed_gt, orig_dec, True)
            except Exception as e:
                self.logger.warning(
                    f"[{self.model_name}] Plotting failed in transform(): {e}"
                )

        self.logger.info(f"{self.model_name} Imputation complete!")
        return imputed_gt

    def _train_and_validate_model(
        self,
        model: torch.nn.Module,
        *,
        lr: float,
        l1_penalty: float,
        trial: Optional[optuna.Trial] = None,
        params: Optional[dict[str, Any]] = None,
        class_weights: Optional[torch.Tensor] = None,
        kl_beta_schedule: bool = False,
        gamma_schedule: bool = False,
    ) -> tuple[float, torch.nn.Module, dict[str, list[float]]]:
        """Train and validate the model.

        Adds robustness checks for:
            - invalid lr/l1/max_epochs
            - model has parameters
            - scheduler warmup bounds
            - common optimizer/scheduler construction failures

        Returns:
            tuple[float, torch.nn.Module, dict[str, list[float]]]:
                Best validation loss, best model, and training history.
        """
        if model is None:
            msg = "model cannot be None."
            self.logger.error(msg)
            raise TypeError(msg)
        if (
            not isinstance(lr, (float, int))
            or not np.isfinite(float(lr))
            or float(lr) <= 0.0
        ):
            msg = f"lr must be finite and > 0; got {lr}."
            self.logger.error(msg)
            raise ValueError(msg)
        if (
            not isinstance(l1_penalty, (float, int))
            or not np.isfinite(float(l1_penalty))
            or float(l1_penalty) < 0.0
        ):
            msg = f"l1_penalty must be finite and >= 0; got {l1_penalty}."
            self.logger.error(msg)
            raise ValueError(msg)
        if int(self.epochs) <= 0:
            msg = f"epochs must be > 0; got {self.epochs}."
            self.logger.error(msg)
            raise ValueError(msg)

        # Ensure model has trainable parameters
        if not any(p.requires_grad for p in model.parameters()):
            msg = f"[{self.model_name}] Model has no trainable parameters."
            self.logger.error(msg)
            raise ValueError(msg)

        max_epochs = int(self.epochs)
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
        except Exception as e:
            self._maybe_prune_or_raise_runtime(
                e, context="Failed to construct optimizer", trial=trial
            )

        # Calculate default warmup
        warmup_epochs = max(int(0.02 * max_epochs), 10)

        # Check if patience is too short for the calculated warmup
        if int(self.early_stop_gen) <= warmup_epochs:
            warmup_epochs = max(0, int(self.early_stop_gen) - 1)
            self.logger.warning(
                f"Early stopping patience ({self.early_stop_gen}) <= default warmup; adjusting warmup to {warmup_epochs}."
            )

        # Clip warmup to sane range
        warmup_epochs = int(max(0, min(warmup_epochs, max_epochs - 1)))

        try:
            scheduler = _make_warmup_cosine_scheduler(
                optimizer, max_epochs=max_epochs, warmup_epochs=warmup_epochs
            )
        except Exception as e:
            self._maybe_prune_or_raise_runtime(
                e, context="Failed to construct scheduler", trial=trial
            )

        best_loss, best_model, hist = self._execute_training_loop(
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            l1_penalty=float(l1_penalty),
            trial=trial,
            params=params,
            class_weights=class_weights,
            kl_beta_schedule=bool(kl_beta_schedule),
            gamma_schedule=bool(gamma_schedule),
        )
        return float(best_loss), best_model, hist

    def _execute_training_loop(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: (
            torch.optim.lr_scheduler.CosineAnnealingLR
            | torch.optim.lr_scheduler.SequentialLR
        ),
        model: torch.nn.Module,
        l1_penalty: float,
        trial: Optional[optuna.Trial] = None,
        params: Optional[dict[str, Any]] = None,
        class_weights: Optional[torch.Tensor] = None,
        kl_beta_schedule: bool = False,
        gamma_schedule: bool = False,
    ) -> tuple[float, torch.nn.Module, dict[str, list[float]]]:
        """Train the model with focal CE reconstruction + KL divergence.

        Notes:
            - validates loaders exist and are iterable
            - validates epochs/min_epochs/early_stop settings
            - catches and prunes (during tuning) common runtime failures (e.g., CUDA OOM)
            - ensures val_loss is finite before early-stopping bookkeeping

        Returns:
            tuple[float, torch.nn.Module, dict[str, list[float]]]:
                Best validation loss, best model, training history.
        """
        if getattr(self, "train_loader_", None) is None:
            msg = f"[{self.model_name}] train_loader_ is not initialized."
            self.logger.error(msg)
            raise RuntimeError(msg)
        if getattr(self, "val_loader_", None) is None:
            msg = f"[{self.model_name}] val_loader_ is not initialized."
            self.logger.error(msg)
            raise RuntimeError(msg)
        if int(self.epochs) <= 0:
            msg = f"[{self.model_name}] epochs must be > 0; got {self.epochs}."
            self.logger.error(msg)
            raise ValueError(msg)
        if int(self.min_epochs) < 0:
            msg = f"[{self.model_name}] min_epochs must be >= 0; got {self.min_epochs}."
            self.logger.error(msg)
            raise ValueError(msg)
        if int(self.early_stop_gen) <= 0:
            msg = f"[{self.model_name}] early_stop_gen must be > 0; got {self.early_stop_gen}."
            self.logger.error(msg)
            raise ValueError(msg)
        history: dict[str, list[float]] = defaultdict(list)

        early_stopping = EarlyStopping(
            patience=self.early_stop_gen,
            min_epochs=self.min_epochs,
            verbose=self.verbose,
            prefix=self.prefix,
            debug=self.debug,
        )

        # KL schedule config
        kl_beta_target, kl_warm, kl_ramp = self._anneal_config(
            params, "kl_beta", default=self.kl_beta, max_epochs=self.epochs
        )
        kl_beta_target = float(kl_beta_target)

        gamma_target, gamma_warm, gamma_ramp = self._anneal_config(
            params, "gamma", default=self.gamma, max_epochs=self.epochs
        )

        cw = class_weights
        if cw is not None:
            if not torch.is_tensor(cw):
                msg = f"class_weights must be a torch.Tensor or None; got {type(cw).__name__}."
                self.logger.error(msg)
                raise TypeError(msg)

            if cw.device != self.device:
                cw = cw.to(self.device)

        ce_criterion = FocalCELoss(
            alpha=cw, gamma=float(gamma_target), reduction="mean", ignore_index=-1
        )

        for epoch in range(int(self.epochs)):
            try:
                if kl_beta_schedule:
                    kl_beta_current = self._update_anneal_schedule(
                        kl_beta_target,
                        warm=kl_warm,
                        ramp=kl_ramp,
                        epoch=epoch,
                        init_val=0.0,
                    )
                else:
                    kl_beta_current = kl_beta_target

                if gamma_schedule:
                    gamma_current = self._update_anneal_schedule(
                        float(gamma_target),
                        warm=gamma_warm,
                        ramp=gamma_ramp,
                        epoch=epoch,
                        init_val=0.0,
                    )
                    ce_criterion.gamma = float(gamma_current)

                train_loss = self._train_step(
                    loader=self.train_loader_,
                    optimizer=optimizer,
                    model=model,
                    ce_criterion=ce_criterion,
                    trial=trial,
                    l1_penalty=float(l1_penalty),
                    kl_beta=kl_beta_current,
                )

                if not np.isfinite(train_loss):
                    if trial is not None:
                        msg = f"[{self.model_name}] Trial {trial.number} training loss non-finite."
                        self.logger.warning(msg)
                        raise optuna.exceptions.TrialPruned(msg)

                    msg = f"[{self.model_name}] Training loss is non-finite at epoch {epoch + 1}."
                    self.logger.error(msg)
                    raise RuntimeError(msg)

                val_loss = self._val_step(
                    loader=self.val_loader_,
                    model=model,
                    ce_criterion=ce_criterion,
                    trial=trial,
                    l1_penalty=float(l1_penalty),
                    kl_beta=kl_beta_current,
                )

                if not np.isfinite(val_loss):
                    if trial is not None:
                        msg = f"[{self.model_name}] Trial {trial.number} validation loss non-finite."
                        self.logger.warning(msg)
                        raise optuna.exceptions.TrialPruned(msg)

                    msg = f"[{self.model_name}] Validation loss is non-finite at epoch {epoch + 1}."
                    self.logger.error(msg)
                    raise RuntimeError(msg)

                if self.debug and epoch % 10 == 0:
                    self.logger.debug(
                        f"[{self.model_name}] Epoch {epoch + 1}/{self.epochs}"
                    )
                    try:
                        self.logger.debug(
                            f"Learning Rate: {float(scheduler.get_last_lr()[0]):.6f}"
                        )
                    except Exception:
                        self.logger.debug("Learning Rate: <unavailable>")
                    self.logger.debug(f"KL Beta: {float(kl_beta_current):.6f}")
                    if gamma_schedule:
                        self.logger.debug(
                            f"Focal CE Gamma: {float(getattr(ce_criterion, 'gamma', 0.0)):.6f}"
                        )
                    self.logger.debug(f"Train Loss: {float(train_loss):.6f}")
                    self.logger.debug(f"Val Loss: {float(val_loss):.6f}")

                # Scheduler step (keep behavior; just guard)
                try:
                    scheduler.step()
                except Exception as e:
                    self._maybe_prune_or_raise_runtime(
                        e, context="scheduler.step() failed", trial=trial
                    )

                history["Train"].append(float(train_loss))
                history["Val"].append(float(val_loss))

                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    self.logger.debug(
                        f"[{self.model_name}] Early stopping at epoch {epoch + 1}."
                    )
                    break

                if trial is not None and isinstance(self.tune_metric, str):
                    trial.report(-float(val_loss), step=epoch)
                    if trial.should_prune():
                        msg = f"[{self.model_name}] Trial {trial.number} pruned at epoch {epoch}. This is a normal part of the tuning process and is not an error."
                        raise optuna.exceptions.TrialPruned(msg)

            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                # During tuning, prune on unexpected runtime failures; during fit, raise.
                self._maybe_prune_or_raise_runtime(
                    e, context=f"training loop failed at epoch {epoch+1}", trial=trial
                )

        best_loss = float(getattr(early_stopping, "best_score", np.inf))
        if not np.isfinite(best_loss):
            # If early_stopping never received a valid score, fail clearly.
            if trial is not None:
                raise optuna.exceptions.TrialPruned(
                    f"[{self.model_name}] No finite best_loss obtained."
                )
            msg = f"[{self.model_name}] No finite best_loss obtained."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if early_stopping.best_state_dict is not None:
            try:
                model.load_state_dict(early_stopping.best_state_dict)
            except Exception as e:
                self._maybe_prune_or_raise_runtime(
                    e, context="Failed to load best_state_dict", trial=trial
                )

        return best_loss, model, dict(history)

    def _train_step(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        ce_criterion: torch.nn.Module,
        trial: Optional[optuna.Trial] = None,
        *,
        l1_penalty: float,
        kl_beta: torch.Tensor | float,
    ) -> float:
        """Single epoch train step across batches (focal CE + KL + optional L1).

        Hardening:
            - validates batch shapes/dtypes
            - guards against empty batches and non-finite tensors
            - catches and prunes on CUDA/runtime errors during tuning

        Returns:
            float: Average training loss.

        Raises:
            RuntimeError / ValueError: On irrecoverable issues (or pruning during Optuna).
        """
        if loader is None:
            raise TypeError("loader cannot be None.")
        if model is None:
            raise TypeError("model cannot be None.")
        if optimizer is None:
            raise TypeError("optimizer cannot be None.")

        model.train()
        running = 0.0
        num_batches = 0

        nF_model = int(self.num_features_)
        nC_model = int(self.num_classes_)
        l1_params = tuple(p for p in model.parameters() if p.requires_grad)

        for X_batch, y_batch, m_batch in loader:
            try:
                optimizer.zero_grad(set_to_none=True)

                if (
                    not torch.is_tensor(X_batch)
                    or not torch.is_tensor(y_batch)
                    or not torch.is_tensor(m_batch)
                ):
                    msg = f"[{self.model_name}] Loader must yield torch tensors: (X_batch, y_batch, m_batch)."
                    self.logger.error(msg)
                    raise TypeError(msg)

                if X_batch.ndim != 2:
                    # Expect flattened (B, nF*nC) from your loader path
                    msg = f"[{self.model_name}] X_batch must be 2D (B, nF*nC); got shape {tuple(X_batch.shape)}."
                    self.logger.error(msg)
                    raise ValueError(msg)
                if y_batch.ndim != 2:
                    msg = f"[{self.model_name}] y_batch must be 2D (B, nF); got shape {tuple(y_batch.shape)}."
                    self.logger.error(msg)
                    raise ValueError(msg)
                if m_batch.shape != y_batch.shape:
                    msg = f"[{self.model_name}] m_batch shape {tuple(m_batch.shape)} must match y_batch shape {tuple(y_batch.shape)}."
                    self.logger.error(msg)
                    raise ValueError(msg)

                X_batch = X_batch.to(self.device, non_blocking=True).float()
                y_batch = y_batch.to(self.device, non_blocking=True).long()
                m_batch = m_batch.to(self.device, non_blocking=True).bool()

                raw = model(X_batch)

                if not isinstance(raw, (tuple, list)) or len(raw) < 3:
                    msg = f"[{self.model_name}] VAE model forward must return (logits, mu, logvar)."
                    self.logger.error(msg)
                    raise RuntimeError(msg)

                logits0 = raw[0]

                expected = X_batch.shape[0] * nF_model * nC_model
                if logits0.numel() != expected:
                    msg = f"[{self.model_name}] VAE logits size mismatch: got {logits0.numel()}, expected {expected}"
                    self.logger.error(msg)
                    raise ValueError(msg)

                logits_masked = logits0.view(-1, nC_model)[m_batch.view(-1)]
                targets_masked = y_batch.view(-1)[m_batch.view(-1)]

                if targets_masked.numel() == 0:
                    continue

                if torch.any(targets_masked < 0):
                    msg = f"[{self.model_name}] Masked targets contain negative labels; mask/targets are inconsistent."
                    self.logger.error(msg)
                    raise ValueError(msg)

                # average number of masked loci per sample (scalar)
                denom = float(max(1, int(X_batch.shape[0])))
                recon_scale = (m_batch.view(-1).sum().float() / denom).detach()

                loss = compute_vae_loss(
                    ce_criterion,
                    logits_masked,
                    targets_masked,
                    mu=raw[1],
                    logvar=raw[2],
                    kl_beta=kl_beta,
                    recon_scale=recon_scale,
                )

                if l1_penalty > 0:
                    l1 = torch.zeros((), device=self.device)
                    for p in l1_params:
                        l1 = l1 + p.abs().sum()
                    loss = loss + float(l1_penalty) * l1

                if not torch.isfinite(loss):
                    msg = f"[{self.model_name}] Training loss non-finite."
                    if trial is not None:
                        raise optuna.exceptions.TrialPruned(msg)
                    raise RuntimeError(msg)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running += float(loss.detach().item())
                num_batches += 1

            except optuna.exceptions.TrialPruned:
                raise
            except RuntimeError as e:
                # Commonly CUDA OOM, illegal memory access, etc.
                self._maybe_prune_or_raise_runtime(
                    e, context="train_step RuntimeError", trial=trial
                )
            except Exception as e:
                self._maybe_prune_or_raise_runtime(
                    e, context="train_step failed", trial=trial
                )

        if num_batches == 0:
            msg = f"[{self.model_name}] Training loss has no valid batches."
            if trial is not None:
                raise optuna.exceptions.TrialPruned(msg)
            self.logger.error(msg)
            raise RuntimeError(msg)

        return running / num_batches

    def _val_step(
        self,
        loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        ce_criterion: torch.nn.Module,
        trial: Optional[optuna.Trial] = None,
        *,
        l1_penalty: float,
        kl_beta: torch.Tensor | float = 1.0,
    ) -> float:
        """Validation step for a single epoch (focal CE + KL + optional L1).

        Hardening:
            - validates batch shapes/dtypes
            - guards against empty batches
            - prunes on non-finite loss during tuning

        Returns:
            float: Average validation loss.
        """
        if loader is None:
            msg = f"[{self.model_name}] loader cannot be None."
            self.logger.error(msg)
            raise TypeError(msg)
        if model is None:
            msg = f"[{self.model_name}] model cannot be None."
            self.logger.error(msg)
            raise TypeError(msg)

        model.eval()
        running = 0.0
        num_batches = 0

        nF_model = int(self.num_features_)
        nC_model = int(self.num_classes_)
        l1_params = tuple(p for p in model.parameters() if p.requires_grad)

        with torch.no_grad():
            for X_batch, y_batch, m_batch in loader:
                try:
                    if (
                        not torch.is_tensor(X_batch)
                        or not torch.is_tensor(y_batch)
                        or not torch.is_tensor(m_batch)
                    ):
                        msg = f"[{self.model_name}] Loader must yield torch tensors: (X_batch, y_batch, m_batch)."
                        self.logger.error(msg)
                        raise TypeError(msg)

                    if X_batch.ndim != 2:
                        msg = f"[{self.model_name}] X_batch must be 2D (B, nF*nC); got shape {tuple(X_batch.shape)}."
                        self.logger.error(msg)
                        raise ValueError(msg)
                    if y_batch.ndim != 2:
                        msg = f"[{self.model_name}] y_batch must be 2D (B, nF); got shape {tuple(y_batch.shape)}."
                        self.logger.error(msg)
                        raise ValueError(msg)
                    if m_batch.shape != y_batch.shape:
                        msg = f"[{self.model_name}] m_batch shape {tuple(m_batch.shape)} must match y_batch shape {tuple(y_batch.shape)}."
                        self.logger.error(msg)
                        raise ValueError(msg)

                    X_batch = X_batch.to(self.device, non_blocking=True).float()
                    y_batch = y_batch.to(self.device, non_blocking=True).long()
                    m_batch = m_batch.to(self.device, non_blocking=True).bool()

                    raw = model(X_batch)

                    if not isinstance(raw, (tuple, list)) or len(raw) < 3:
                        msg = f"[{self.model_name}] VAE model forward must return (logits, mu, logvar)."
                        self.logger.error(msg)
                        raise RuntimeError(msg)

                    logits0 = raw[0]

                    expected = X_batch.shape[0] * nF_model * nC_model
                    if logits0.numel() != expected:
                        msg = f"[{self.model_name}] VAE logits size mismatch: got {logits0.numel()}, expected {expected}"
                        self.logger.error(msg)
                        raise ValueError(msg)

                    logits_masked = logits0.view(-1, nC_model)[m_batch.view(-1)]
                    targets_masked = y_batch.view(-1)[m_batch.view(-1)]

                    if targets_masked.numel() == 0:
                        continue

                    if torch.any(targets_masked < 0):
                        msg = f"[{self.model_name}] Masked targets contain negative labels; mask/targets are inconsistent."
                        self.logger.error(msg)
                        raise ValueError(msg)

                    denom = float(max(1, int(X_batch.shape[0])))
                    recon_scale = (m_batch.view(-1).sum().float() / denom).detach()

                    loss = compute_vae_loss(
                        ce_criterion,
                        logits_masked,
                        targets_masked,
                        mu=raw[1],
                        logvar=raw[2],
                        kl_beta=kl_beta,
                        recon_scale=recon_scale,
                    )

                    if l1_penalty > 0:
                        l1 = torch.zeros((), device=self.device)
                        for p in l1_params:
                            l1 = l1 + p.abs().sum()
                        loss = loss + float(l1_penalty) * l1

                    if not torch.isfinite(loss):
                        msg = f"[{self.model_name}] Validation loss non-finite."
                        if trial is not None:
                            raise optuna.exceptions.TrialPruned(msg)
                        self.logger.error(msg)
                        raise RuntimeError(msg)

                    running += float(loss.item())
                    num_batches += 1

                except optuna.exceptions.TrialPruned:
                    raise
                except RuntimeError as e:
                    self._maybe_prune_or_raise_runtime(
                        e, context="val_step RuntimeError", trial=trial
                    )
                except Exception as e:
                    self._maybe_prune_or_raise_runtime(
                        e, context="val_step failed", trial=trial
                    )

        if num_batches == 0:
            msg = f"[{self.model_name}] Validation loss has no valid batches."
            if trial is not None:
                raise optuna.exceptions.TrialPruned(msg)
            self.logger.error(msg)
            raise RuntimeError(msg)

        return running / num_batches

    def _predict(
        self,
        model: torch.nn.Module,
        X: np.ndarray | torch.Tensor,
        *,
        return_proba: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict categorical genotype labels from logits.

        Hardening:
            - validates X dimensionality and non-empty batch
            - validates model output structure
            - guards against device mismatch and reshape errors

        Args:
            model (torch.nn.Module): Trained model.
            X (np.ndarray | torch.Tensor): 0/1/2 matrix with -1 for missing, or one-hot encoded (B, L, K).
            return_proba (bool): If True, return probabilities.

        Returns:
            tuple[np.ndarray, np.ndarray | None]: (labels, probas|None).

        Raises:
            NotFittedError, ValueError, RuntimeError
        """
        if model is None:
            msg = (
                "Model passed to predict() is not trained. Call fit() before predict()."
            )
            self.logger.error(msg)
            raise NotFittedError(msg)

        if X is None:
            msg = "_predict X cannot be None."
            self.logger.error(msg)
            raise TypeError(msg)

        model.eval()
        nF = int(self.num_features_)
        nC = int(self.num_classes_)

        X_tensor = X if isinstance(X, torch.Tensor) else torch.from_numpy(np.asarray(X))
        if X_tensor.numel() == 0:
            # Preserve core behavior: return empty arrays rather than crash
            empty_labels = np.empty((0, nF), dtype=np.int64)
            empty_proba = (
                np.empty((0, nF, nC), dtype=np.float32) if return_proba else None
            )
            return empty_labels, empty_proba

        X_tensor = X_tensor.float()
        if X_tensor.device != self.device:
            X_tensor = X_tensor.to(self.device)

        if X_tensor.dim() == 2:
            # 0/1/2 matrix -> one-hot for model input
            X_tensor = self._one_hot_encode_012(X_tensor, num_classes=nC).float()
            if X_tensor.device != self.device:
                X_tensor = X_tensor.to(self.device)
        elif X_tensor.dim() != 3:
            msg = f"_predict expects 2D 0/1/2 inputs or 3D one-hot inputs; got shape {tuple(X_tensor.shape)}."
            self.logger.error(msg)
            raise ValueError(msg)

        if X_tensor.shape[1] != nF or X_tensor.shape[2] != nC:
            msg = f"_predict input shape mismatch: expected (B, {nF}, {nC}), got {tuple(X_tensor.shape)}."
            self.logger.error(msg)
            raise ValueError(msg)

        # Flatten to (B, nF*nC) for VAEModel
        X_tensor = X_tensor.reshape(X_tensor.shape[0], nF * nC)

        with torch.no_grad():
            raw = model(X_tensor)

            if not isinstance(raw, (tuple, list)) or len(raw) < 1:
                msg = f"[{self.model_name}] VAE model forward must return a tuple/list with logits at index 0."
                self.logger.error(msg)
                raise RuntimeError(msg)

            logits0 = raw[0]
            # expected flat size: B*nF*nC
            expected = X_tensor.shape[0] * nF * nC
            if logits0.numel() != expected:
                msg = f"[{self.model_name}] VAE logits size mismatch: got {logits0.numel()}, expected {expected}."
                self.logger.error(msg)
                raise RuntimeError(msg)

            logits = logits0.view(-1, nF, nC)
            probas = torch.softmax(logits, dim=-1)
            labels = torch.argmax(probas, dim=-1)

        if return_proba:
            return labels.detach().cpu().numpy(), probas.detach().cpu().numpy()
        return labels.detach().cpu().numpy(), None

    def _evaluate_model(
        self,
        model: torch.nn.Module,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray,
        eval_mask: np.ndarray,
        *,
        objective_mode: bool = False,
    ) -> dict[str, float]:
        """Evaluate model performance on masked genotypes.

        Hardening:
            - validates y/eval_mask alignment
            - validates prediction/proba shapes before masking
            - handles degenerate evaluation (no valid sites) consistently
            - guards against out-of-range labels before one-hot creation
            - ensures probabilities are finite before normalization

        Returns:
            dict[str, float]: Evaluation metrics.
        """
        if model is None:
            msg = "Model passed to _evaluate_model() is not fitted. Call fit() before evaluation."
            self.logger.error(msg)
            raise NotFittedError(msg)

        if y is None or eval_mask is None:
            msg = "y and eval_mask cannot be None."
            self.logger.error(msg)
            raise TypeError(msg)

        y_arr = np.asarray(y)
        m_arr = np.asarray(eval_mask)

        if y_arr.shape != m_arr.shape:
            msg = f"y and eval_mask must have identical shapes; got y={y_arr.shape}, eval_mask={m_arr.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        if m_arr.dtype != bool:
            self.logger.warning("eval_mask is not boolean; coercing to bool.")
            m_arr = m_arr.astype(bool, copy=False)

        pred_labels, pred_probas = self._predict(model=model, X=X, return_proba=True)

        if pred_probas is None:
            msg = "Predicted probabilities are None in _evaluate_model()."
            self.logger.error(msg)
            raise ValueError(msg)

        if pred_labels.shape != y_arr.shape:
            msg = f"pred_labels shape mismatch: {pred_labels.shape} vs y {y_arr.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        if pred_probas.ndim != 3:
            msg = f"pred_probas must be 3D (B, L, K); got shape {pred_probas.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        if (
            pred_probas.shape[0] != y_arr.shape[0]
            or pred_probas.shape[1] != y_arr.shape[1]
        ):
            msg = f"pred_probas first two dims must match y: pred_probas={pred_probas.shape}, y={y_arr.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        # Mask + filter invalid gt (<0)
        y_true_flat = y_arr[m_arr].astype(np.int64, copy=False)
        y_pred_flat = pred_labels[m_arr].astype(np.int64, copy=False)
        y_proba_flat = pred_probas[m_arr].astype(np.float32, copy=False)

        valid = y_true_flat >= 0
        y_true_flat = y_true_flat[valid]
        y_pred_flat = y_pred_flat[valid]
        y_proba_flat = y_proba_flat[valid]

        if y_true_flat.size == 0:
            if isinstance(self.tune_metric, str):
                return {self.tune_metric: 0.0}
            if isinstance(self.tune_metric, (list, tuple)):
                return {m: 0.0 for m in self.tune_metric}
            msg = f"[{self.model_name}] Invalid tune_metric type: {type(self.tune_metric)}"
            self.logger.error(msg)
            raise ValueError(msg)

        if y_proba_flat.ndim != 2:
            msg = f"Expected y_proba_flat to be 2D (n_eval, n_classes); got shape {y_proba_flat.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        if not np.isfinite(y_proba_flat).all():
            self.logger.warning(
                f"[{self.model_name}] Non-finite probabilities detected; replacing with 0 and renormalizing."
            )
            y_proba_flat = np.nan_to_num(y_proba_flat, nan=0.0, posinf=0.0, neginf=0.0)

        K = int(y_proba_flat.shape[1])

        if self.is_haploid_:
            if K not in (2, 3):
                msg = f"Haploid evaluation expects 2 or 3 classes; got {K}."
                self.logger.error(msg)
                raise RuntimeError(msg)
        else:
            if K != 3:
                msg = f"Diploid evaluation expects 3 classes; got {K}."
                self.logger.error(msg)
                raise RuntimeError(msg)

        if not self.is_haploid_:
            if np.any((y_true_flat < 0) | (y_true_flat > 2)):
                msg = (
                    "Diploid y_true_flat contains values outside {0,1,2} after masking."
                )
                self.logger.error(msg)
                raise ValueError(msg)

        # --- Harmonize for haploid vs diploid ---
        if self.is_haploid_:
            y_true_flat = (y_true_flat > 0).astype(np.int8, copy=False)
            y_pred_flat = (y_pred_flat > 0).astype(np.int8, copy=False)

            if K == 3:
                proba_2 = np.empty((y_proba_flat.shape[0], 2), dtype=y_proba_flat.dtype)
                proba_2[:, 0] = y_proba_flat[:, 0]
                proba_2[:, 1] = y_proba_flat[:, 1] + y_proba_flat[:, 2]
                y_proba_flat = proba_2

            labels_for_scoring = [0, 1]
            target_names = ["REF", "ALT"]
        else:
            labels_for_scoring = [0, 1, 2]
            target_names = ["REF", "HET", "ALT"]

        # Ensure probs are valid simplex
        y_proba_flat = np.clip(y_proba_flat, 0.0, 1.0)
        row_sums = y_proba_flat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        y_proba_flat = y_proba_flat / row_sums

        # Guard one-hot indexing: y_true_flat must be in-range
        n_lbl = len(labels_for_scoring)
        if np.any((y_true_flat < 0) | (y_true_flat >= n_lbl)):
            msg = f"[{self.model_name}] y_true_flat contains out-of-range labels for one-hot: min={y_true_flat.min()}, max={y_true_flat.max()}, n_labels={n_lbl}."
            self.logger.error(msg)
            raise ValueError(msg)

        y_true_ohe = np.eye(n_lbl, dtype=np.int8)[y_true_flat]

        tm = cast(
            Literal[
                "pr_macro",
                "roc_auc",
                "accuracy",
                "f1",
                "average_precision",
                "precision",
                "recall",
                "mcc",
                "jaccard",
            ]
            | list[str]
            | tuple[str, ...],
            self.tune_metric,
        )

        metrics = self.scorers_.evaluate(
            y_true_flat.astype(np.int8, copy=False),
            y_pred_flat.astype(np.int8, copy=False),
            y_true_ohe,
            y_proba_flat,
            objective_mode,
            tune_metric=tm,
        )

        if not objective_mode:
            if self.verbose or self.debug:
                pm = PrettyMetrics(
                    metrics, precision=2, title=f"{self.model_name} Validation Metrics"
                )
                pm.render()

            self._make_class_reports(
                y_true=y_true_flat.astype(np.int8, copy=False),
                y_pred_proba=y_proba_flat,
                y_pred=y_pred_flat.astype(np.int8, copy=False),
                metrics=metrics,
                labels=target_names,
            )

            # --- IUPAC decode and 10-base integer report ---
            try:
                y_true_matrix = np.array(y_arr, copy=True)
                y_pred_matrix = np.array(pred_labels, copy=True)

                if self.is_haploid_:
                    y_true_matrix = np.where(y_true_matrix > 0, 2, y_true_matrix)
                    y_pred_matrix = np.where(y_pred_matrix > 0, 2, y_pred_matrix)

                y_true_dec = self.decode_012(y_true_matrix)
                y_pred_dec = self.decode_012(y_pred_matrix)

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

                valid_iupac_mask = y_true_int[m_arr] >= 0
                if valid_iupac_mask.any():
                    self._make_class_reports(
                        y_true=y_true_int[m_arr][valid_iupac_mask],
                        y_pred=y_pred_int[m_arr][valid_iupac_mask],
                        metrics=metrics,
                        y_pred_proba=None,
                        labels=["A", "C", "G", "T", "W", "R", "M", "K", "Y", "S"],
                    )
                else:
                    self.logger.warning(
                        f"[{self.model_name}] Skipped IUPAC confusion matrix: No ground truths."
                    )
            except Exception as e:
                self.logger.warning(f"[{self.model_name}] IUPAC reporting failed: {e}")

        return metrics

    def _objective(self, trial: optuna.Trial) -> float | tuple[float, ...]:
        """Optuna objective for VAE.

        Hardening:
            - validates preconditions needed for tuning (train/val data present)
            - ensures resources are cleared even when exceptions occur

        Returns:
            float | tuple[float, ...]: Value(s) of the tuning metric(s) to be optimized.
        """
        model: Optional[torch.nn.Module] = None
        try:
            if (
                getattr(self, "X_train_", None) is None
                or getattr(self, "X_val_corrupted_", None) is None
            ):
                msg = f"[{self.model_name}] Training/validation data not prepared for tuning."
                self.logger.error(msg)
                raise RuntimeError(msg)

            params = self._sample_hyperparameters(trial)

            model = self.build_model(self.Model, params["model_params"])
            model.apply(self.initialize_weights)

            lr: float = float(params["learning_rate"])
            l1_penalty: float = float(params["l1_penalty"])

            class_weights = self._class_weights_from_zygosity(
                self.y_train_,
                train_mask=self.eval_mask_train_,
                inverse=params["inverse"],
                normalize=params["normalize"],
                max_ratio=self.max_ratio,
                power=params["power"],
            )

            _, trained_model, _ = self._train_and_validate_model(
                model=model,
                lr=lr,
                l1_penalty=l1_penalty,
                params=params,
                trial=trial,
                class_weights=class_weights,
                kl_beta_schedule=params["kl_beta_schedule"],
                gamma_schedule=params["gamma_schedule"],
            )

            if trained_model is None:
                msg = f"[{self.model_name}] Model training returned None in tuning objective."
                self.logger.error(msg)
                raise RuntimeError(msg)

            metrics = self._evaluate_model(
                model=trained_model,
                X=self.X_val_corrupted_,
                y=self.y_val_,
                eval_mask=self.eval_mask_val_,
                objective_mode=True,
            )

            if isinstance(self.tune_metric, (list, tuple)):
                return tuple(metrics[k] for k in self.tune_metric)
            return float(metrics[self.primary_metric])

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            err_type = type(e).__name__
            self.logger.warning(
                f"Trial {trial.number} failed due to exception {err_type}: {e}"
            )
            self.logger.debug(traceback.format_exc())
            raise optuna.exceptions.TrialPruned(
                f"Trial {trial.number} failed due to an exception. {err_type}: {e}."
            ) from e
        finally:
            # Ensure we clear resources even on exception
            try:
                if model is not None:
                    self._clear_resources(model)
            except Exception:
                # Don't let cleanup failures mask real tuning signals
                pass

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """Sample model hyperparameters; hidden sizes use BaseNNImputer helper.

        Hardening:
            - validates num_features_ and X_train_ availability
            - ensures latent_dim bounds are valid for small feature counts

        Returns:
            dict: Sampled hyperparameters (including "model_params").

        Raises:
            optuna.exceptions.TrialPruned: If feature dimensionality is too small to sample safely.
        """
        if getattr(self, "num_features_", None) is None:
            msg = f"[{self.model_name}] num_features_ is not set. Ensure fit() preprocessing ran before tuning."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if int(self.num_features_) < 2:
            raise optuna.exceptions.TrialPruned(
                "Too few features to tune a VAE (num_features_ < 2)."
            )

        if getattr(self, "X_train_", None) is None:
            msg = f"[{self.model_name}] X_train_ is not set. Ensure fit() preprocessing ran before tuning."
            self.logger.error(msg)
            raise RuntimeError(msg)

        lower_bound = 2
        upper_bound = max(lower_bound, min(32, int(self.num_features_) - 1))

        if upper_bound < lower_bound:
            raise optuna.exceptions.TrialPruned(
                f"Invalid latent_dim bounds: lower={lower_bound}, upper={upper_bound} (num_features_={self.num_features_})."
            )

        params = {
            "latent_dim": trial.suggest_int("latent_dim", lower_bound, upper_bound),
            "learning_rate": trial.suggest_float("learning_rate", 3e-6, 1e-3, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.025),
            "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 20),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "elu", "selu", "leaky_relu"]
            ),
            "l1_penalty": trial.suggest_float("l1_penalty", 1e-6, 1e-3, log=True),
            "layer_scaling_factor": trial.suggest_float(
                "layer_scaling_factor", 2.0, 10.0, step=0.025
            ),
            "layer_schedule": trial.suggest_categorical(
                "layer_schedule", ["pyramid", "linear"]
            ),
            "power": trial.suggest_float("power", 0.1, 2.0, step=0.1),
            "normalize": trial.suggest_categorical("normalize", [True, False]),
            "inverse": trial.suggest_categorical("inverse", [True, False]),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0, step=0.1),
            "kl_beta": trial.suggest_float("kl_beta", 0.1, 5.0, step=0.1),
            "kl_beta_schedule": trial.suggest_categorical(
                "kl_beta_schedule", [True, False]
            ),
            "gamma_schedule": trial.suggest_categorical(
                "gamma_schedule", [True, False]
            ),
        }

        OBJECTIVE_SPEC_VAE.validate(params)

        nF: int = int(self.num_features_)
        nC: int = int(self.num_classes_)
        input_dim = nF * nC

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=input_dim,
            n_outputs=nC,
            n_samples=len(self.X_train_),
            n_hidden=int(params["num_hidden_layers"]),
            latent_dim=int(params["latent_dim"]),
            alpha=float(params["layer_scaling_factor"]),
            schedule=str(params["layer_schedule"]),
            min_size=max(16, 2 * int(params["latent_dim"])),
        )

        params["model_params"] = {
            "n_features": nF,
            "num_classes": nC,
            "dropout_rate": float(params["dropout_rate"]),
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": str(params["activation"]),
            "kl_beta": float(params["kl_beta"]),
        }

        return params

    def _set_best_params(self, params: dict) -> dict:
        """Update instance fields from tuned params and return model_params dict.

        Hardening:
            - validates required keys exist
            - type/finite checks for numeric parameters

        Args:
            params (dict): Best hyperparameters from tuning.

        Returns:
            dict: Model parameters for building the VAE.

        Raises:
            KeyError/ValueError: If required keys missing or invalid.
        """
        required = {
            "latent_dim",
            "dropout_rate",
            "learning_rate",
            "l1_penalty",
            "activation",
            "layer_scaling_factor",
            "layer_schedule",
            "power",
            "normalize",
            "inverse",
            "gamma",
            "gamma_schedule",
            "kl_beta",
            "kl_beta_schedule",
            "num_hidden_layers",
        }
        missing = sorted(required.difference(params.keys()))
        if missing:
            msg = f"[{self.model_name}] Missing required tuned params: {missing}"
            self.logger.error(msg)
            raise KeyError(msg)

        # Assign (keeping behavior)
        self.latent_dim = int(params["latent_dim"])
        self.dropout_rate = float(params["dropout_rate"])
        self.learning_rate = float(params["learning_rate"])
        self.l1_penalty = float(params["l1_penalty"])
        self.activation = str(params["activation"])
        self.layer_scaling_factor = float(params["layer_scaling_factor"])
        self.layer_schedule = str(params["layer_schedule"])
        self.power = float(params["power"])
        self.normalize = bool(params["normalize"])
        self.inverse = bool(params["inverse"])
        self.gamma = float(params["gamma"])
        self.gamma_schedule = bool(params["gamma_schedule"])
        self.kl_beta = float(params["kl_beta"])
        self.kl_beta_schedule = bool(params["kl_beta_schedule"])

        # Basic finite checks (defensive)
        for nm, val in (
            ("latent_dim", self.latent_dim),
            ("dropout_rate", self.dropout_rate),
            ("learning_rate", self.learning_rate),
            ("l1_penalty", self.l1_penalty),
            ("layer_scaling_factor", self.layer_scaling_factor),
            ("power", self.power),
            ("gamma", self.gamma),
            ("kl_beta", self.kl_beta),
        ):
            if isinstance(val, (int, float)) and not np.isfinite(float(val)):
                msg = f"[{self.model_name}] {nm} must be finite; got {val}."
                self.logger.error(msg)
                raise ValueError(msg)

        self.class_weights_ = self._class_weights_from_zygosity(
            self.y_train_,
            train_mask=self.eval_mask_train_,
            inverse=self.inverse,
            normalize=self.normalize,
            max_ratio=self.max_ratio,
            power=self.power,
        )

        nF = int(self.num_features_)
        nC = int(self.num_classes_)
        input_dim = nF * nC

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=input_dim,
            n_outputs=nC,
            n_samples=len(self.X_train_),
            n_hidden=int(params["num_hidden_layers"]),
            latent_dim=int(params["latent_dim"]),
            alpha=float(params["layer_scaling_factor"]),
            schedule=str(params["layer_schedule"]),
            min_size=max(16, 2 * int(params["latent_dim"])),
        )

        return {
            "n_features": nF,
            "latent_dim": int(self.latent_dim),
            "hidden_layer_sizes": hidden_layer_sizes,
            "dropout_rate": float(self.dropout_rate),
            "activation": str(self.activation),
            "num_classes": nC,
            "kl_beta": float(params["kl_beta"]),
        }
