# -*- coding: utf-8 -*-
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
from pgsui.data_processing.containers import AutoencoderConfig
from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.loss_functions import FocalCELoss
from pgsui.impute.unsupervised.models.autoencoder_model import AutoencoderModel
from pgsui.utils.logging_utils import configure_logger
from pgsui.utils.misc import OBJECTIVE_SPEC_AE
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
        optimizer: Optimizer to schedule.
        max_epochs: Total number of epochs.
        warmup_epochs: Number of warmup epochs.
        start_factor: Starting LR factor for warmup.

    Returns:
        torch.optim.lr_scheduler.CosineAnnealingLR | torch.optim.lr_scheduler.SequentialLR: LR scheduler (SequentialLR if warmup_epochs > 0 else CosineAnnealingLR).
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

    Notes:
        - Supports top-level preset, or io.preset inside dict/YAML.
        - Does not mutate user-provided dict (deep-copies before processing).
        - Flattens nested dicts into dot-keys and applies them as overrides.

    Args:
        config: AutoencoderConfig instance, dict, YAML path, or None.

    Returns:
        Concrete AutoencoderConfig.
    """
    if config is None:
        return AutoencoderConfig()
    if isinstance(config, AutoencoderConfig):
        return config
    if isinstance(config, str):
        return load_yaml_to_dataclass(config, AutoencoderConfig)
    if isinstance(config, dict):
        cfg_in = copy.deepcopy(config)
        base = AutoencoderConfig()

        preset = cfg_in.pop("preset", None)
        if "io" in cfg_in and isinstance(cfg_in["io"], dict):
            preset = preset or cfg_in["io"].pop("preset", None)

        if preset:
            base = AutoencoderConfig.from_preset(preset)

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

    raise TypeError("config must be an AutoencoderConfig, dict, YAML path, or None.")


class ImputeAutoencoder(BaseNNImputer):
    """Autoencoder imputer for 0/1/2 genotypes.

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
        config: Optional[Union["AutoencoderConfig", dict, str]] = None,
        overrides: Optional[dict] = None,
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
        sim_prop: Optional[float] = None,
        sim_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize the Autoencoder imputer with a unified config interface.

        Args:
            genotype_data (GenotypeData): Backing genotype data object.
            tree_parser (Optional[TreeParser]): Optional SNPio tree parser for nonrandom simulated-missing modes.
            config (Optional[Union[AutoencoderConfig, dict, str]]): AutoencoderConfig, nested dict, YAML path, or None.
            overrides (Optional[dict]): Optional dot-key overrides with highest precedence.
            sim_strategy (Literal["random", "random_weighted" "random_weighted_inv", "nonrandom", "nonrandom_weighted"]): Override sim strategy; if None, uses config default.
            sim_prop (Optional[float]): Override simulated missing proportion; if None, uses config default. Default is None.
            sim_kwargs (Optional[dict]): Override/extend simulated missing kwargs; if None, uses config default.
        """
        self.model_name = "ImputeAutoencoder"
        self.genotype_data = genotype_data
        self.tree_parser = tree_parser

        if self.genotype_data is None:
            msg = f"{self.model_name} requires a non-null genotype_data."
            self.logger.error(msg) if hasattr(self, "logger") else None
            raise ValueError(msg)

        cfg = ensure_autoencoder_config(config)
        if overrides:
            cfg = apply_dot_overrides(cfg, overrides)
        self.cfg = cfg

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
        self.logger.propagate = False

        super().__init__(
            model_name=self.model_name,
            genotype_data=self.genotype_data,
            prefix=self.cfg.io.prefix,
            device=self.cfg.train.device,
            verbose=self.cfg.io.verbose,
            debug=self.cfg.io.debug,
        )

        self.Model = AutoencoderModel
        self.pgenc = GenotypeEncoder(genotype_data)

        # I/O and global
        self.seed = self.cfg.io.seed
        self.n_jobs = self.cfg.io.n_jobs
        self.prefix = self.cfg.io.prefix
        self.scoring_averaging = self.cfg.io.scoring_averaging
        self.verbose = self.cfg.io.verbose
        self.debug = self.cfg.io.debug

        try:
            self.rng = np.random.default_rng(self.seed)
        except Exception as e:
            msg = f"{self.model_name} failed to initialize RNG with seed={self.seed!r}: {e}"
            self.logger.error(msg)
            raise ValueError(msg) from e

        # Simulation controls
        sim_cfg = getattr(self.cfg, "sim", None)
        sim_cfg_kwargs = copy.deepcopy(getattr(sim_cfg, "sim_kwargs", None) or {})
        if sim_kwargs:
            if not isinstance(sim_kwargs, dict):
                msg = f"{self.model_name} sim_kwargs must be a dict; got {type(sim_kwargs).__name__}."
                self.logger.error(msg)
                raise TypeError(msg)
            sim_cfg_kwargs.update(sim_kwargs)

        if sim_cfg is None:
            default_strategy = "random"
            default_prop = 0.2
        else:
            default_strategy = sim_cfg.sim_strategy
            default_prop = sim_cfg.sim_prop

        self.simulate_missing = True
        self.sim_strategy = sim_strategy or default_strategy
        self.sim_prop = float(sim_prop if sim_prop is not None else default_prop)
        self.sim_kwargs = sim_cfg_kwargs

        if not isinstance(self.sim_strategy, str) or not self.sim_strategy:
            msg = f"{self.model_name} sim_strategy must be a non-empty string; got {self.sim_strategy!r}."
            self.logger.error(msg)
            raise ValueError(msg)

        if not np.isfinite(self.sim_prop) or not (0.0 < self.sim_prop < 1.0):
            msg = f"{self.model_name} sim_prop must be in (0, 1); got {self.sim_prop}."
            self.logger.error(msg)
            raise ValueError(msg)

        if self.tree_parser is None and self.sim_strategy.startswith("nonrandom"):
            msg = "tree_parser is required for nonrandom sim strategies."
            self.logger.error(msg)
            raise ValueError(msg)

        # Model architecture
        self.latent_dim = int(self.cfg.model.latent_dim)
        self.dropout_rate = float(self.cfg.model.dropout_rate)
        self.num_hidden_layers = int(self.cfg.model.num_hidden_layers)
        self.layer_scaling_factor = float(self.cfg.model.layer_scaling_factor)
        self.layer_schedule = str(self.cfg.model.layer_schedule)
        self.activation = str(self.cfg.model.activation)

        if self.latent_dim < 1:
            msg = f"{self.model_name} latent_dim must be >= 1; got {self.latent_dim}."
            self.logger.error(msg)
            raise ValueError(msg)
        if not (0.0 <= self.dropout_rate < 1.0):
            msg = f"{self.model_name} dropout_rate must be in [0, 1); got {self.dropout_rate}."
            self.logger.error(msg)
            raise ValueError(msg)
        if self.num_hidden_layers < 1:
            msg = f"{self.model_name} num_hidden_layers must be >= 1; got {self.num_hidden_layers}."
            self.logger.error(msg)
            raise ValueError(msg)

        # Training / loss controls (align with fields where present)
        self.power = float(getattr(self.cfg.train, "weights_power", 1.0))
        self.max_ratio = getattr(self.cfg.train, "weights_max_ratio", None)
        self.normalize = bool(getattr(self.cfg.train, "weights_normalize", True))
        self.inverse = bool(getattr(self.cfg.train, "weights_inverse", False))

        self.batch_size = int(self.cfg.train.batch_size)
        self.learning_rate = float(self.cfg.train.learning_rate)
        self.l1_penalty = float(self.cfg.train.l1_penalty)
        self.early_stop_gen = int(self.cfg.train.early_stop_gen)
        self.min_epochs = int(self.cfg.train.min_epochs)
        self.epochs = int(self.cfg.train.max_epochs)
        self.validation_split = float(self.cfg.train.validation_split)

        if self.batch_size < 1:
            msg = f"{self.model_name} batch_size must be >= 1; got {self.batch_size}."
            self.logger.error(msg)
            raise ValueError(msg)
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            msg = f"{self.model_name} learning_rate must be > 0 and finite; got {self.learning_rate}."
            self.logger.error(msg)
            raise ValueError(msg)
        if not np.isfinite(self.l1_penalty) or self.l1_penalty < 0:
            msg = f"{self.model_name} l1_penalty must be >= 0 and finite; got {self.l1_penalty}."
            self.logger.error(msg)
            raise ValueError(msg)
        if self.epochs < 1:
            msg = f"{self.model_name} max_epochs must be >= 1; got {self.epochs}."
            self.logger.error(msg)
            raise ValueError(msg)
        if self.min_epochs < 0:
            msg = f"{self.model_name} min_epochs must be >= 0; got {self.min_epochs}."
            self.logger.error(msg)
            raise ValueError(msg)
        if self.early_stop_gen < 1:
            msg = f"{self.model_name} early_stop_gen must be >= 1; got {self.early_stop_gen}."
            self.logger.error(msg)
            raise ValueError(msg)
        if not (0.0 < self.validation_split < 1.0):
            msg = f"{self.model_name} validation_split must be in (0, 1); got {self.validation_split}."
            self.logger.error(msg)
            raise ValueError(msg)

        # Gamma can live in cfg.model or cfg.train depending on your dataclasses
        gamma_raw = getattr(
            self.cfg.train, "gamma", getattr(self.cfg.model, "gamma", 0.0)
        )
        if not isinstance(gamma_raw, (float, int)):
            msg = f"Gamma must be float|int; got {type(gamma_raw).__name__}."
            self.logger.error(msg)
            raise TypeError(msg)
        self.gamma = float(gamma_raw)
        self.gamma_schedule = bool(getattr(self.cfg.train, "gamma_schedule", True))
        if not np.isfinite(self.gamma) or self.gamma < 0:
            msg = f"{self.model_name} gamma must be >= 0 and finite; got {self.gamma}."
            self.logger.error(msg)
            raise ValueError(msg)

        # Hyperparameter tuning
        self.tune = bool(self.cfg.tune.enabled)
        self.tune_metric: str | list[str] | tuple[str, ...]
        self.tune_metric = self.cfg.tune.metrics
        self.primary_metric = self.validate_tuning_metric()

        self.n_trials = int(self.cfg.tune.n_trials)
        self.tune_save_db = bool(self.cfg.tune.save_db)
        self.tune_resume = bool(self.cfg.tune.resume)
        self.tune_patience = int(self.cfg.tune.patience)

        if self.n_trials < 1 and self.tune:
            msg = f"{self.model_name} tune enabled but n_trials < 1 (n_trials={self.n_trials})."
            self.logger.error(msg)
            raise ValueError(msg)

        # Plotting
        self.plot_format = self.cfg.plot.fmt
        self.plot_dpi = int(self.cfg.plot.dpi)
        self.plot_fontsize = int(self.cfg.plot.fontsize)
        self.title_fontsize = int(self.cfg.plot.fontsize)
        self.despine = bool(self.cfg.plot.despine)
        self.show_plots = bool(self.cfg.plot.show)
        self.use_multiqc = bool(self.cfg.plot.multiqc)

        # Fit-time attributes
        self.is_haploid_: bool = False
        self.num_classes_: int = 3
        self.model_params: dict[str, Any] = {}

        self.sim_mask_train_: np.ndarray
        self.sim_mask_val_: np.ndarray
        self.sim_mask_test_: np.ndarray

        self.orig_mask_train_: np.ndarray
        self.orig_mask_val_: np.ndarray
        self.orig_mask_test_: np.ndarray

        self.num_tuned_params_ = OBJECTIVE_SPEC_AE.count()

    def fit(self) -> "ImputeAutoencoder":
        """Fit the Autoencoder imputer model to the genotype data.

        This method performs the following steps:
            1. Validates the presence of SNP data in the genotype data.
            2. Determines ploidy and sets up the number of classes accordingly.
            3. Cleans the ground truth genotype matrix and simulates missingness.
            4. Splits the data into training, validation, and test sets.
            5. Prepares one-hot encoded inputs for the model.
            6. Initializes plotting utilities and valid-class masks.
            7. Sets up data loaders for training and validation.
            8. Performs hyperparameter tuning if enabled, otherwise uses fixed hyperparameters.
            9. Builds and trains the Autoencoder model.
            10. Evaluates the trained model on the test set.
            11. Returns the fitted ImputeAutoencoder instance.

        Returns:
            ImputeAutoencoder: The fitted ImputeAutoencoder instance.
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
                f"{self.model_name} currently supports only haploid (1) or diploid (2) "
                f"data; got ploidy={self.ploidy}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.debug(
            f"Ploidy set to {self.ploidy}, is_haploid: {self.is_haploid_}"
        )

        self.num_classes_ = 2 if self.is_haploid_ else 3

        # Clean 0/1/2 ground truth (missing=-1)
        gt_raw = getattr(self.pgenc, "genotypes_012", None)
        if gt_raw is None:
            msg = f"{self.model_name} requires pgenc.genotypes_012 but it is None."
            self.logger.error(msg)
            raise AttributeError(msg)

        gt_full = np.array(gt_raw, copy=True)
        if gt_full.ndim != 2:
            msg = f"{self.model_name} expects a 2D genotype matrix; got shape {gt_full.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        if gt_full.shape[0] < 1 or gt_full.shape[1] < 1:
            msg = f"{self.model_name} genotype matrix must be non-empty; got shape {gt_full.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        gt_full[gt_full < 0] = -1
        gt_full = np.nan_to_num(gt_full, nan=-1.0)
        self.ground_truth_ = gt_full.astype(np.int8, copy=False)
        self.num_features_ = int(self.ground_truth_.shape[1])

        self.model_params = {
            "n_features": self.num_features_,
            "num_classes": self.num_classes_,
            "latent_dim": self.latent_dim,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
        }
        self.logger.debug(f"Model parameters: {self.model_params}")

        # Simulate missingness ONCE on the full matrix
        sim_tup = self.sim_missing_transform(self.ground_truth_)
        if not isinstance(sim_tup, (tuple, list)) or len(sim_tup) != 3:
            msg = (
                f"{self.model_name} sim_missing_transform must return a 3-tuple "
                f"(X_corrupted, sim_mask, orig_mask); got type={type(sim_tup).__name__}, "
                f"len={len(sim_tup) if isinstance(sim_tup, (tuple, list)) else 'NA'}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        X_for_model_full, self.sim_mask_, self.orig_mask_ = sim_tup
        X_for_model_full = np.asarray(X_for_model_full)
        self.sim_mask_ = np.asarray(self.sim_mask_, dtype=bool)
        self.orig_mask_ = np.asarray(self.orig_mask_, dtype=bool)

        if X_for_model_full.shape != self.ground_truth_.shape:
            msg = (
                f"{self.model_name} corrupted matrix shape mismatch: "
                f"X_for_model_full={X_for_model_full.shape} vs ground_truth_={self.ground_truth_.shape}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        if (
            self.sim_mask_.shape != self.ground_truth_.shape
            or self.orig_mask_.shape != self.ground_truth_.shape
        ):
            msg = (
                f"{self.model_name} mask shape mismatch: sim_mask_={self.sim_mask_.shape}, "
                f"orig_mask_={self.orig_mask_.shape}, expected={self.ground_truth_.shape}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Validate sim/orig masks (no overlap; enough eval sites)
        if hasattr(self, "_validate_sim_and_orig_masks"):
            self._validate_sim_and_orig_masks(
                sim_mask=self.sim_mask_, orig_mask=self.orig_mask_, context="full"
            )

        # Split indices based on clean ground truth
        indices = self._train_val_test_split(self.ground_truth_)
        if not isinstance(indices, (tuple, list)) or len(indices) != 3:
            msg = f"{self.model_name} _train_val_test_split must return (train_idx, val_idx, test_idx)."
            self.logger.error(msg)
            raise ValueError(msg)

        self.train_idx_, self.val_idx_, self.test_idx_ = indices
        if any(len(x) == 0 for x in (self.train_idx_, self.val_idx_, self.test_idx_)):
            msg = f"{self.model_name} produced an empty split: train={len(self.train_idx_)}, val={len(self.val_idx_)}, test={len(self.test_idx_)}."
            self.logger.error(msg)
            raise ValueError(msg)

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

        # Ensure eval masks have at least some evaluation sites per split
        for nm, m in [
            ("eval_mask_train_", self.eval_mask_train_),
            ("eval_mask_val_", self.eval_mask_val_),
            ("eval_mask_test_", self.eval_mask_test_),
        ]:
            if (
                m.shape
                != self.ground_truth_[
                    (
                        self.train_idx_
                        if "train" in nm
                        else self.val_idx_ if "val" in nm else self.test_idx_
                    )
                ].shape
            ):
                # Best-effort: avoid overcomplicating; just require 2D with correct second axis.
                if m.ndim != 2 or m.shape[1] != self.num_features_:
                    msg = f"{self.model_name} {nm} has unexpected shape {m.shape}."
                    self.logger.error(msg)
                    raise ValueError(msg)
            if not bool(np.any(m)):
                msg = f"{self.model_name} {nm} has zero True entries; nothing to evaluate."
                self.logger.error(msg)
                raise ValueError(msg)

        # --- Haploid harmonization (transform before persisting) ---
        if self.is_haploid_:
            self.logger.debug(
                "Performing haploid harmonization on split inputs/targets..."
            )

            def _haploidize(arr: np.ndarray) -> np.ndarray:
                out = np.array(arr, copy=True)
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

        # Persist matrices
        self.X_train_clean_ = X_train_clean
        self.X_val_clean_ = X_val_clean
        self.X_test_clean_ = X_test_clean
        self.X_train_corrupted_ = X_train_corrupted
        self.X_val_corrupted_ = X_val_corrupted
        self.X_test_corrupted_ = X_test_corrupted

        # NOTE: Convention is X_* are corrupted inputs and y_* are clean targets
        self.X_train_ = self.X_train_corrupted_
        self.y_train_ = self.X_train_clean_

        self.X_val_ = self.X_val_corrupted_
        self.y_val_ = self.X_val_clean_

        self.y_test_ = self.X_test_clean_

        self.X_train_ = self._one_hot_encode_012(
            self.X_train_, num_classes=self.num_classes_
        )
        self.X_val_ = self._one_hot_encode_012(
            self.X_val_, num_classes=self.num_classes_
        )

        for name, tensor in [("X_train_", self.X_train_), ("X_val_", self.X_val_)]:
            if not torch.is_tensor(tensor) or tensor.ndim != 3:
                msg = f"[{self.model_name}] {name} must be a 3D torch.Tensor after one-hot; got {type(tensor).__name__} with ndim={getattr(tensor, 'ndim', None)}."
                self.logger.error(msg)
                raise TypeError(msg)
            if tensor.shape[2] != self.num_classes_:
                msg = f"[{self.model_name}] {name} last dim must be num_classes={self.num_classes_}; got {tensor.shape}."
                self.logger.error(msg)
                raise ValueError(msg)
            if (tensor.sum(dim=-1) > 1).any():
                msg = f"[{self.model_name}] Invalid one-hot: >1 active class in {name}."
                self.logger.error(msg)
                raise RuntimeError(msg)

        # Plotters/scorers + valid-class mask repairs
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

        # Create loaders
        train_loader = self._get_data_loaders(
            Xtr_np, self.y_train_, self.eval_mask_train_, self.batch_size, shuffle=True
        )
        val_loader = self._get_data_loaders(
            Xva_np, self.y_val_, self.eval_mask_val_, self.batch_size, shuffle=False
        )

        if train_loader is None or val_loader is None:
            msg = f"{self.model_name} failed to create data loaders."
            self.logger.error(msg)
            raise RuntimeError(msg)

        self.train_loader_ = train_loader
        self.val_loader_ = val_loader

        # Hyperparameter tuning or fixed run
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
            keys = OBJECTIVE_SPEC_AE.keys
            self.tuned_params_ = {k: getattr(self, k) for k in keys}
            self.tuned_params_["model_params"] = self.model_params

        # Always start clean
        self.best_params_ = copy.deepcopy(self.tuned_params_)
        self._log_class_weights()

        # Final model params
        input_dim = int(self.num_features_ * self.num_classes_)
        model_params_final = {
            "n_features": int(self.num_features_),
            "num_classes": int(self.num_classes_),
            "latent_dim": int(self.best_params_["latent_dim"]),
            "dropout_rate": float(self.best_params_["dropout_rate"]),
            "activation": str(self.best_params_["activation"]),
        }
        model_params_final["hidden_layer_sizes"] = self._compute_hidden_layer_sizes(
            n_inputs=input_dim,
            n_outputs=int(self.num_classes_),
            n_samples=len(self.train_idx_),
            n_hidden=int(self.best_params_["num_hidden_layers"]),
            latent_dim=int(self.best_params_["latent_dim"]),
            alpha=float(self.best_params_["layer_scaling_factor"]),
            schedule=str(self.best_params_["layer_schedule"]),
            min_size=max(16, 2 * int(self.best_params_["latent_dim"])),
        )
        self.best_params_["model_params"] = model_params_final

        # Build and train
        model = self.build_model(self.Model, self.best_params_["model_params"])
        model.apply(self.initialize_weights)

        lr_final = float(self.best_params_["learning_rate"])
        l1_final = float(self.best_params_["l1_penalty"])
        gamma_schedule = bool(
            self.best_params_.get("gamma_schedule", self.gamma_schedule)
        )

        loss, trained_model, history = self._train_and_validate_model(
            model=model,
            lr=lr_final,
            l1_penalty=l1_final,
            params=self.best_params_,
            trial=None,
            class_weights=getattr(self, "class_weights_", None),
            gamma_schedule=gamma_schedule,
        )

        if trained_model is None:
            msg = f"{self.model_name} training failed."
            self.logger.error(msg)
            raise RuntimeError(msg)

        torch.save(
            trained_model.state_dict(),
            self.models_dir / f"final_model_{self.model_name}.pt",
        )

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

        # Evaluate on simulated-missing sites only
        self._evaluate_model(
            self.model_,
            X=self.X_test_corrupted_,
            y=self.y_test_,
            eval_mask=self.eval_mask_test_,
            objective_mode=False,
        )

        if self.show_plots:
            self.plotter_.plot_history(self.history_)

        self._save_display_model_params(is_tuned=self.model_tuned_)

        self.logger.info(f"{self.model_name} fitting complete!")
        return self

    def transform(self) -> np.ndarray:
        """Impute missing genotypes and return IUPAC strings.

        This method performs the following steps:
            1. Validates that the model has been fitted.
            2. Uses the trained model to predict missing genotypes for the entire dataset.
            3. Fills in the missing genotypes in the original dataset with the predicted values from the model.
            4. Decodes the imputed genotype matrix from 0/1/2 encoding to IUPAC strings.
            5. Checks for any remaining missing values or decoding issues, raising errors if found.
            6. Optionally generates and displays plots comparing the original and imputed genotype distributions.
            7. Returns the imputed IUPAC genotype matrix.

        Returns:
            np.ndarray: IUPAC genotype matrix of shape (n_samples, n_loci).

        Raises:
            NotFittedError: If called before fit().
            RuntimeError: If any missing values remain or decoding yields "N".
            RuntimeError: If loci contain 'N' after imputation due to missing REF/ALT metadata.
        """
        if not getattr(self, "is_fit_", False):
            msg = f"{self.model_name} is not fitted. Must call 'fit()' before 'transform()'."
            self.logger.error(msg)
            raise NotFittedError(msg)

        if getattr(self, "ground_truth_", None) is None:
            msg = f"{self.model_name} ground_truth_ is missing; fit() did not complete correctly."
            self.logger.error(msg)
            raise RuntimeError(msg)

        self.logger.info(f"Imputing entire dataset with {self.model_name}...")
        X_to_impute = np.array(self.ground_truth_, copy=True)

        pred_labels, _ = self._predict(self.model_, X=X_to_impute)

        if pred_labels.shape != X_to_impute.shape:
            msg = (
                f"{self.model_name} prediction shape mismatch: "
                f"pred_labels={pred_labels.shape}, X_to_impute={X_to_impute.shape}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        missing_mask = X_to_impute < 0
        imputed_array = X_to_impute.copy()
        if np.any(missing_mask):
            imputed_array[missing_mask] = pred_labels[missing_mask]

        # Sanity check: all missing values should be gone
        if np.any(imputed_array < 0):
            msg = f"[{self.model_name}] Some missing genotypes remain after imputation. This is unexpected."
            self.logger.error(msg)
            raise RuntimeError(msg)

        decode_input = imputed_array
        if getattr(self, "is_haploid_", False):
            decode_input = imputed_array.copy()
            decode_input[decode_input == 1] = 2

        try:
            imputed_gt = self.decode_012(decode_input)
        except Exception as e:
            msg = f"{self.model_name} decode_012 failed during transform(): {e}"
            self.logger.error(msg)
            raise RuntimeError(msg) from e

        if (imputed_gt == "N").any():
            msg = f"Something went wrong: {self.model_name} imputation still contains {int((imputed_gt == 'N').sum())} missing values ('N')."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if self.show_plots:
            original_input = X_to_impute
            if self.is_haploid_:
                original_input = X_to_impute.copy()
                original_input[original_input == 1] = 2

            plt.rcParams.update(self.plotter_.param_dict)

            try:
                orig_dec = self.decode_012(original_input)
                self.plotter_.plot_gt_distribution(imputed_gt, orig_dec, True)
            except Exception as e:
                # Plotting should never break transform()
                self.logger.warning(
                    f"{self.model_name} plotting failed in transform(): {e}"
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
        gamma_schedule: bool = False,
    ) -> tuple[float, torch.nn.Module, dict[str, list[float]]]:
        """Train and validate the model.

        This method sets up the optimizer and learning rate scheduler, then executes the training loop with early stopping and optional hyperparameter tuning via Optuna. It returns the best validation loss, the best model, and the training history.

        Args:
            model (torch.nn.Module): Autoencoder model.
            lr (float): Learning rate.
            l1_penalty (float): L1 regularization coefficient.
            trial (Optional[optuna.Trial]): Optuna trial (optional).
            params (Optional[dict[str, Any]]): Hyperparams dict (optional).
            class_weights (Optional[torch.Tensor]): Class weights for focal CE (optional).
            gamma_schedule (bool): Whether to schedule gamma.

        Returns:
            tuple[float, torch.nn.Module, dict[str, list[float]]]: Best validation loss, best model, history.
        """
        if model is None:
            msg = (
                f"{self.model_name} received model=None in _train_and_validate_model()."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        if not np.isfinite(lr) or lr <= 0:
            msg = f"{self.model_name} lr must be > 0 and finite; got {lr}."
            self.logger.error(msg)
            raise ValueError(msg)

        if not np.isfinite(l1_penalty) or l1_penalty < 0:
            msg = f"{self.model_name} l1_penalty must be >= 0 and finite; got {l1_penalty}."
            self.logger.error(msg)
            raise ValueError(msg)

        max_epochs = int(self.epochs)
        if max_epochs < 1:
            msg = f"{self.model_name} epochs must be >= 1; got {max_epochs}."
            self.logger.error(msg)
            raise ValueError(msg)

        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
        except Exception as e:
            self._maybe_prune_or_raise_runtime(
                e, context="Failed to construct optimizer", trial=trial
            )

        # Calculate default warmup
        warmup_epochs = max(int(0.02 * max_epochs), 10)

        # Check if patience is too short for the calculated warmup
        if self.early_stop_gen <= warmup_epochs:
            warmup_epochs = max(0, self.early_stop_gen - 1)
            self.logger.warning(
                f"Early stopping patience ({self.early_stop_gen}) <= default warmup; adjusting warmup to {warmup_epochs}."
            )

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
            l1_penalty=l1_penalty,
            trial=trial,
            params=params,
            class_weights=class_weights,
            gamma_schedule=gamma_schedule,
        )
        return best_loss, best_model, hist

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
        gamma_schedule: bool = False,
    ) -> tuple[float, torch.nn.Module, dict[str, list[float]]]:
        """Train AE (masked focal CE) with EarlyStopping + Optuna pruning.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler.CosineAnnealingLR | torch.optim.lr_scheduler.SequentialLR): LR scheduler.
            model (torch.nn.Module): Autoencoder model.
            l1_penalty (float): L1 regularization coefficient.
            trial (Optional[optuna.Trial]): Optuna trial (optional).
            params (Optional[dict[str, Any]]): Hyperparams dict (optional).
            class_weights (Optional[torch.Tensor]): Class weights for focal CE (optional).
            gamma_schedule (bool): Whether to schedule gamma.

        Returns:
            tuple[float, torch.nn.Module, dict[str, list[float]]]: Best loss, best model, and training history.

        Notes:
            - Computes loss only where targets are known (~orig_mask_*).
            - Evaluates metrics only on simulated-missing sites (sim_mask_*).
        """
        if (
            getattr(self, "train_loader_", None) is None
            or getattr(self, "val_loader_", None) is None
        ):
            msg = f"{self.model_name} train/val loaders are not initialized; did fit() prepare them?"
            self.logger.error(msg)
            raise RuntimeError(msg)

        if self.min_epochs > self.epochs:
            msg = f"{self.model_name} min_epochs ({self.min_epochs}) cannot exceed max_epochs ({self.epochs})."
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

        gamma_target, gamma_warm, gamma_ramp = self._anneal_config(
            params, "gamma", default=self.gamma, max_epochs=self.epochs
        )
        gamma_target = float(gamma_target)

        cw = class_weights
        if cw is not None and cw.device != self.device:
            cw = cw.to(self.device)

        for epoch in range(int(self.epochs)):
            try:
                if gamma_schedule:
                    gamma_current = self._update_anneal_schedule(
                        gamma_target,
                        warm=gamma_warm,
                        ramp=gamma_ramp,
                        epoch=epoch,
                        init_val=0.0,
                    )
                    gamma_val = float(gamma_current)
                else:
                    gamma_val = gamma_target

                ce_criterion = FocalCELoss(
                    alpha=cw, gamma=gamma_val, ignore_index=-1, reduction="mean"
                )

                train_loss = self._train_step(
                    loader=self.train_loader_,
                    optimizer=optimizer,
                    model=model,
                    ce_criterion=ce_criterion,
                    trial=trial,
                    l1_penalty=l1_penalty,
                )

                if not np.isfinite(train_loss):
                    if trial is not None:
                        msg = f"[{self.model_name}] Trial {trial.number} training loss non-finite."
                        # Pruning isn't a "hard error"; keep warning (as you already do)
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
                    l1_penalty=l1_penalty,
                )

                if self.debug and epoch % 10 == 0:
                    self.logger.debug(
                        f"[{self.model_name}] Epoch {epoch + 1}/{self.epochs}"
                    )
                    try:
                        lr_now = float(scheduler.get_last_lr()[0])
                        self.logger.debug(f"Learning Rate: {lr_now:.6f}")
                    except Exception:
                        self.logger.debug("Learning Rate: <unavailable>")
                    if gamma_schedule:
                        self.logger.debug(
                            f"Focal CE Gamma: {float(ce_criterion.gamma):.6f}"
                        )
                    self.logger.debug(f"Train Loss: {train_loss:.6f}")
                    self.logger.debug(f"Val Loss: {val_loss:.6f}")

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
                        raise optuna.exceptions.TrialPruned(
                            f"[{self.model_name}] Trial {trial.number} pruned at epoch {epoch}. This is a normal part of the tuning process and is not an error."
                        )

            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                # During tuning,
                # prune on unexpected runtime failures; during fit, raise.
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
    ) -> float:
        """Single epoch train step (masked focal CE + optional L1).

        Args:
            loader (torch.utils.data.DataLoader): Training data loader.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            model (torch.nn.Module): Autoencoder model.
            ce_criterion (torch.nn.Module): Cross-entropy loss function.
            trial (Optional[optuna.Trial]): Optuna trial object for hyperparameter tuning.
            l1_penalty (float): L1 regularization coefficient.

        Returns:
            float: Average training loss over the epoch.

        Notes:
            Expects loader batches as (X_ohe, y_int, mask_bool) where:
                - X_ohe: (B, L, C) float/compatible
                - y_int: (B, L) int, with -1 for unknown targets
                - mask_bool: (B, L) bool selecting which positions contribute to loss
        """
        if loader is None:
            msg = f"[{self.model_name}] received loader=None in _train_step()."
            self.logger.error(msg)
            raise ValueError(msg)

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

                logits_flat = model(X_batch)

                expected = X_batch.shape[0] * nF_model * nC_model
                if logits_flat.numel() != expected:
                    msg = f"[{self.model_name}] Logits size mismatch: got {logits_flat.numel()}, expected {expected}"
                    self.logger.error(msg)
                    raise ValueError(msg)

                logits_masked = logits_flat.view(-1, nC_model)[m_batch.view(-1)]
                targets_masked = y_batch.view(-1)[m_batch.view(-1)]

                if targets_masked.numel() == 0:
                    continue

                if torch.any(targets_masked < 0):
                    msg = f"[{self.model_name}] Masked targets contain negative labels; mask/targets are inconsistent."
                    self.logger.error(msg)
                    raise ValueError(msg)

                # Compute loss only on masked positions
                loss = ce_criterion(logits_masked, targets_masked)

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
    ) -> float:
        """Validation step (masked focal CE + optional L1).

        Args:
            loader (torch.utils.data.DataLoader): Validation data loader.
            model (torch.nn.Module): Autoencoder model.
            ce_criterion (torch.nn.Module): Cross-entropy loss function.
            trial (Optional[optuna.Trial]): Optuna trial object for hyperparameter tuning.
            l1_penalty (float): L1 regularization coefficient.

        Returns:
            float: Average validation loss over the epoch.
        """
        if loader is None:
            msg = f"[{self.model_name}] received loader=None in _val_step()."
            self.logger.error(msg)
            raise ValueError(msg)

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

                    logits_flat = model(X_batch)

                    expected = (int(X_batch.shape[0]), int(nF_model * nC_model))
                    if logits_flat.dim() != 2 or tuple(logits_flat.shape) != expected:
                        try:
                            logits_flat = logits_flat.view(*expected)
                        except Exception as e:
                            msg = f"Model logits expected shape {expected}, got {tuple(logits_flat.shape)}."
                            self.logger.error(msg)
                            raise ValueError(msg) from e

                    logits_masked = logits_flat.view(-1, nC_model)
                    flat_mask = m_batch.view(-1)
                    logits_masked = logits_masked[flat_mask]

                    targets_masked = y_batch.view(-1)[flat_mask]

                    if targets_masked.numel() == 0:
                        continue

                    if torch.any(targets_masked < 0):
                        msg = f"[{self.model_name}] Masked targets contain negative labels; mask/targets are inconsistent."
                        self.logger.error(msg)
                        raise ValueError(msg)

                    loss = ce_criterion(logits_masked, targets_masked)

                    if l1_penalty > 0:
                        l1 = torch.zeros((), device=self.device)
                        for p in l1_params:
                            l1 = l1 + p.abs().sum()
                        loss = loss + l1_penalty * l1

                    if trial is not None:
                        if not torch.isfinite(loss):
                            msg = f"[{self.model_name}] Trial {trial.number} validation loss non-finite. Pruning trial."
                            self.logger.warning(msg)
                            raise optuna.exceptions.TrialPruned(msg)
                    elif not torch.isfinite(loss):
                        msg = f"[{self.model_name}] Validation loss non-finite."
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

        This method uses the trained model to predict genotype labels for the provided input data. It handles both 0/1/2 encoded matrices and one-hot encoded matrices, converting them as necessary for model input. The method returns the predicted labels and, optionally, the predicted probabilities.

        Args:
            model (torch.nn.Module): Trained model.
            X (np.ndarray | torch.Tensor): 0/1/2 matrix with -1 for missing, or one-hot encoded (B, L, K).
            return_proba (bool): If True, return probabilities.

        Returns:
            tuple[np.ndarray, np.ndarray | None]: (labels, probas|None).
        """
        if model is None:
            msg = (
                "Model passed to predict() is not trained. Call fit() before predict()."
            )
            self.logger.error(msg)
            raise NotFittedError(msg)

        if X is None:
            msg = f"{self.model_name} _predict received X=None."
            self.logger.error(msg)
            raise ValueError(msg)

        model.eval()

        nF = int(self.num_features_)
        nC = int(self.num_classes_)

        try:
            X_tensor = (
                X if isinstance(X, torch.Tensor) else torch.from_numpy(np.asarray(X))
            )
        except Exception as e:
            msg = f"{self.model_name} failed to convert X to tensor in _predict(): {e}"
            self.logger.error(msg)
            raise ValueError(msg) from e

        X_tensor = X_tensor.float()
        if X_tensor.device != self.device:
            X_tensor = X_tensor.to(self.device)

        if X_tensor.dim() == 2:
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

        X_tensor = X_tensor.reshape(X_tensor.shape[0], nF * nC)

        with torch.no_grad():
            raw = model(X_tensor)
            logits = raw.view(-1, nF, nC)
            probas = torch.softmax(logits, dim=-1)
            labels = torch.argmax(probas, dim=-1)

        labels_np = labels.detach().cpu().numpy()
        if return_proba:
            probas_np = probas.detach().cpu().numpy()
            return labels_np, probas_np
        return labels_np, None

    def _evaluate_model(
        self,
        model: torch.nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        eval_mask: np.ndarray,
        *,
        objective_mode: bool = False,
    ) -> dict[str, float]:
        """Evaluate on 0/1/2; then IUPAC decoding and 10-base integer reports.

        Args:
            model (torch.nn.Module): Trained model.
            X (np.ndarray): 2D 0/1/2 matrix with -1 for missing.
            y (np.ndarray): 2D 0/1/2 ground truth matrix with -1 for missing.
            eval_mask (np.ndarray): 2D boolean mask selecting sites to evaluate.
            objective_mode (bool): If True, suppress detailed reports and plots.

        Returns:
            dict[str, float]: Dictionary of evaluation metrics.
        """
        if model is None:
            msg = "Model passed to _evaluate_model() is not fitted. Call fit() before evaluation."
            self.logger.error(msg)
            raise NotFittedError(msg)

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        m_arr = np.asarray(eval_mask, dtype=bool)

        if X_arr.shape != y_arr.shape or X_arr.shape != m_arr.shape:
            msg = (
                f"{self.model_name} _evaluate_model shape mismatch: "
                f"X={X_arr.shape}, y={y_arr.shape}, eval_mask={m_arr.shape}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        if not np.any(m_arr):
            self.logger.debug(
                f"{self.model_name} _evaluate_model: eval_mask contains zero True entries; returning zeros."
            )
            if isinstance(self.tune_metric, str):
                return {self.tune_metric: 0.0}
            if isinstance(self.tune_metric, (list, tuple)):
                return {m: 0.0 for m in self.tune_metric}
            msg = f"[{self.model_name}] tune_metric must be str or list/tuple[str]; got {type(self.tune_metric)}."
            self.logger.error(msg)
            raise ValueError(msg)

        pred_labels, pred_probas = self._predict(
            model=model, X=X_arr, return_proba=True
        )
        if pred_probas is None:
            msg = "Predicted probabilities are None in _evaluate_model()."
            self.logger.error(msg)
            raise ValueError(msg)

        if pred_labels.shape != y_arr.shape:
            msg = (
                f"{self.model_name} prediction shape mismatch in _evaluate_model: "
                f"pred_labels={pred_labels.shape}, y={y_arr.shape}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        y_true_flat = y_arr[m_arr].astype(np.int8, copy=False)
        y_pred_flat = pred_labels[m_arr].astype(np.int8, copy=False)
        y_proba_flat = pred_probas[m_arr].astype(np.float32, copy=False)

        valid = y_true_flat >= 0
        y_true_flat = y_true_flat[valid]
        y_pred_flat = y_pred_flat[valid]
        y_proba_flat = y_proba_flat[valid]

        if y_true_flat.size == 0:
            self.logger.debug(
                f"No valid ground truth genotypes found for evaluation in _evaluate_model(). Returning zeroed metrics."
            )
            if isinstance(self.tune_metric, str):
                return {self.tune_metric: 0.0}
            if isinstance(self.tune_metric, (list, tuple)):
                return {m: 0.0 for m in self.tune_metric}
            msg = f"[{self.model_name}] tune_metric must be a string or list/tuple of strings, but got: {type(self.tune_metric)}."
            self.logger.error(msg)
            raise ValueError(msg)

        if y_proba_flat.ndim != 2:
            msg = f"Expected y_proba_flat to be 2D (n_eval, n_classes); got {y_proba_flat.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        K = int(y_proba_flat.shape[1])
        if self.is_haploid_:
            if K not in (2, 3):
                msg = f"Haploid evaluation expects 2 or 3 classes; got {K}."
                self.logger.error(msg)
                raise ValueError(msg)
        else:
            if K != 3:
                msg = f"Diploid evaluation expects 3 classes; got {K}."
                self.logger.error(msg)
                raise ValueError(msg)

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

        y_proba_flat = np.clip(y_proba_flat, 0.0, 1.0)
        row_sums = y_proba_flat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        y_proba_flat = y_proba_flat / row_sums

        y_true_ohe = np.eye(len(labels_for_scoring), dtype=np.int8)[y_true_flat]

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

        try:
            metrics = self.scorers_.evaluate(
                y_true_flat,
                y_pred_flat,
                y_true_ohe,
                y_proba_flat,
                objective_mode,
                tune_metric=tm,
            )
        except Exception as e:
            msg = (
                f"{self.model_name} scorer evaluation failed in _evaluate_model(): {e}"
            )
            self.logger.error(msg)
            raise RuntimeError(msg) from e

        if not objective_mode:
            if self.verbose or self.debug:
                pm = PrettyMetrics(
                    metrics, precision=2, title=f"{self.model_name} Validation Metrics"
                )
                pm.render()

            self._make_class_reports(
                y_true=y_true_flat,
                y_pred_proba=y_proba_flat,
                y_pred=y_pred_flat,
                metrics=metrics,
                labels=target_names,
            )

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
                    "Skipped IUPAC confusion matrix: No valid ground truths."
                )

        return metrics

    def _objective(self, trial: optuna.Trial) -> float | tuple[float, ...]:
        """Optuna objective for model.

        This method defines the objective function for hyperparameter tuning using Optuna. It samples hyperparameters, trains the model with these parameters, and evaluates its performance on a validation set. The evaluation metric specified by ``self.tune_metric`` is returned for optimization. If training fails, the trial is pruned to keep the tuning process efficient.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float | tuple[float, ...]: Value(s) of the tuning metric(s) to be optimized.

        Raises:
            RuntimeError: If model training returns None.
            optuna.exceptions.TrialPruned: If training fails unexpectedly or is unpromising.
        """
        model: Optional[torch.nn.Module] = None
        try:
            params = self._sample_hyperparameters(trial)

            model = self.build_model(self.Model, params["model_params"])
            model.apply(self.initialize_weights)

            lr: float = params["learning_rate"]
            l1_penalty: float = params["l1_penalty"]

            class_weights = self._class_weights_from_zygosity(
                self.y_train_,
                train_mask=self.eval_mask_train_,
                inverse=params["inverse"],
                normalize=params["normalize"],
                max_ratio=self.max_ratio,
                power=params["power"],
            )

            res = self._train_and_validate_model(
                model=model,
                lr=lr,
                l1_penalty=l1_penalty,
                params=params,
                trial=trial,
                class_weights=class_weights,
                gamma_schedule=params["gamma_schedule"],
            )
            model = res[1]

            if model is None:
                msg = "Model training returned None in tuning objective."
                self.logger.error(msg)
                raise RuntimeError(msg)

            metrics = self._evaluate_model(
                model=model,
                X=self.X_val_corrupted_,
                y=self.y_val_,
                eval_mask=self.eval_mask_val_,
                objective_mode=True,
            )

            if isinstance(self.tune_metric, (list, tuple)):
                return tuple([metrics[k] for k in self.tune_metric])
            return metrics[self.primary_metric]

        except Exception as e:
            err_type = type(e).__name__
            self.logger.warning(
                f"Trial {trial.number} failed due to exception {err_type}: {e}"
            )
            self.logger.debug(traceback.format_exc())
            raise optuna.exceptions.TrialPruned(
                f"Trial {trial.number} failed due to an exception. {err_type}: {e}. Enable debug logging for full traceback."
            ) from e
        finally:
            if model is not None:
                try:
                    self._clear_resources(model)
                except Exception as e:
                    self.logger.warning(
                        f"{self.model_name} _clear_resources failed in objective cleanup: {e}"
                    )

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """Sample model hyperparameters; hidden sizes use BaseNNImputer helper.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            dict[str, int | float | str]: Sampled hyperparameters.
        """
        if (
            getattr(self, "num_features_", None) is None
            or getattr(self, "num_classes_", None) is None
        ):
            msg = f"{self.model_name} _sample_hyperparameters called before fit-time attributes were set."
            self.logger.error(msg)
            raise RuntimeError(msg)

        lower_bound = 2
        upper_bound = max(lower_bound, min(32, int(self.num_features_) - 1))
        if upper_bound < lower_bound:
            msg = f"{self.model_name} invalid latent_dim bounds: lower={lower_bound}, upper={upper_bound}."
            self.logger.error(msg)
            raise ValueError(msg)

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
            "gamma_schedule": trial.suggest_categorical(
                "gamma_schedule", [True, False]
            ),
        }

        try:
            OBJECTIVE_SPEC_AE.validate(params)
        except Exception as e:
            msg = f"{self.model_name} OBJECTIVE_SPEC_AE.validate failed: {e}"
            self.logger.error(msg)
            raise ValueError(msg) from e

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
        }

        return params

    def _set_best_params(self, params: dict) -> dict:
        """Update instance fields from tuned params and return model_params dict.

        Args:
            params (dict): Best hyperparameters from tuning.

        Returns:
            dict: Model parameters for building the final model.
        """
        required = [
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
            "num_hidden_layers",
        ]
        missing = [k for k in required if k not in params]
        if missing:
            msg = f"{self.model_name} _set_best_params missing keys: {missing}"
            self.logger.error(msg)
            raise KeyError(msg)

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
            n_samples=len(self.train_idx_),
            n_hidden=int(params["num_hidden_layers"]),
            latent_dim=int(params["latent_dim"]),
            alpha=float(params["layer_scaling_factor"]),
            schedule=str(params["layer_schedule"]),
            min_size=max(16, 2 * int(params["latent_dim"])),
        )

        return {
            "n_features": nF,
            "num_classes": nC,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": hidden_layer_sizes,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
        }
