# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, Union, cast

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.utils.logging import LoggerManager

from pgsui.data_processing.config import apply_dot_overrides, load_yaml_to_dataclass
from pgsui.data_processing.containers import UBPConfig
from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.loss_functions import FocalCELoss
from pgsui.impute.unsupervised.models.ubp_model import UBPModel
from pgsui.utils.logging_utils import configure_logger
from pgsui.utils.misc import OBJECTIVE_SPEC_UBP
from pgsui.utils.pretty_metrics import PrettyMetrics

if TYPE_CHECKING:
    from snpio import TreeParser
    from snpio.read_input.genotype_data import GenotypeData


def ensure_ubp_config(config: UBPConfig | dict | str | None) -> UBPConfig:
    """Return a concrete UBPConfig."""
    if config is None:
        return UBPConfig()
    if isinstance(config, UBPConfig):
        return config
    if isinstance(config, str):
        return load_yaml_to_dataclass(config, UBPConfig)
    if isinstance(config, dict):
        cfg_in = copy.deepcopy(config)
        base = UBPConfig()
        preset = cfg_in.pop("preset", None)
        if "io" in cfg_in and isinstance(cfg_in["io"], dict):
            preset = preset or cfg_in["io"].pop("preset", None)
        if preset:
            base = UBPConfig.from_preset(preset)

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
    raise TypeError("config must be an UBPConfig, dict, YAML path, or None.")


class ImputeUBP(BaseNNImputer):
    """Unsupervised Backpropagation (UBP) Imputer for Genotype Data.

        This model performs missing value imputation by learning a low-dimensional continuous manifold (latent space) that best explains the observed genotype patterns. Unlike standard autoencoders that require an encoder network to map inputs to the latent space, UBP treats the latent embeddings :math:`V` for each sample as learnable parameters that are optimized alongside the decoder weights :math:`W`.

        **Model Description**

        Imagine every individual in your dataset can be described by a small set of abstract coordinates (like "ancestry X", "ancestry Y", etc.). We don't know thes coordinates, so we guess them randomly (or using PCA). We then train a neural networ (the decoder) to take these coordinates and reconstruct the person's DNA. If th reconstruction is wrong, we adjust both the neural network *and* the person's coordinates t make the prediction better. Once trained, we fill in missing DNA markers based o where the individual sits in this abstract space.

        **Mathematical Formulation**

        The objective is to minimize the reconstruction error between the observed genotypes :math:`X` and the model output :math:`\hat{X}`.

        .. math::

            \hat{X} = f(V; W)

        The optimization minimizes the cost function :math:`J`:

        .. math::

            J(V, W) = \mathcal{L}_{Focal}(X_{obs}, \hat{X}_{obs}) + \lambda \|W\|_1

        Where:
            - :math:`V \in \mathbb{R}^{N \times K}` are the latent embeddings.
            - :math:`W` are the network weights.
            - :math:`\mathcal{L}_{Focal}` is the Focal Cross-Entropy loss (handling class imbalance).
            - :math:`\lambda` is the L1 regularization coefficient.

        **Training Procedure**
        This implementation modifies the original Gashler et al. algorithm for genomics:

        1.  **Initialization (Modified Phase 1):** Instead of training random projections, we initialize :math:`V` using Principal Component Analysis (PCA) on the observed data to provide a "warm start" for the manifold.
        2.  **Decoder Refinement (Phase 2):** We freeze :math:`V` and optimize only the network weights :math:`W` to map the PCA embeddings to the genotypes.
        3.  **Joint Optimization (Phase 3):** We unfreeze :math:`V` and optimize both :math:`V` and :math:`W` simultaneously. This allows the embeddings to drift off the linear PCA plane into a non-linear manifold.

    Attributes:
        model_ : torch.nn.Module
                The trained PyTorch model (Decoder).
        is_fit_ : bool
            Whether the model has been successfully fitted.
        model_tuned_ : bool
            Whether the model hyperparameters were tuned via Optuna.
        model_params : dict
            Dictionary defining the model architecture (layers, dimensions, activations).
        best_params_ : dict
            The optimal hyperparameters found during tuning (or loaded from config).
        tuned_params_ : dict
            The full set of parameters selected after the tuning process.
        num_tuned_params_ : int
            Number of hyperparameters that were subject to tuning.
        total_samples_ : int
            Total number of samples (individuals) in the genotype data.
        num_features_ : int
            Number of SNP features (loci columns) in the genotype data.
        num_classes_ : int
            Number of genotype classes (2 for haploid, 3 for diploid).
        is_haploid_ : bool
            True if the data is haploid, False if diploid.
        v_init_ : torch.Tensor
            The initial PCA-derived embeddings used to warm-start the manifold.
        class_weights_ : torch.Tensor
            Calculated weights for the Focal Loss to handle genotype imbalance.
        ground_truth_ : np.ndarray
            The complete, original genotype matrix (0/1/2 encoded) used for training and evaluation.
        sim_mask_ : np.ndarray
            Boolean mask representing simulated missingness for the full dataset.
        orig_mask_ : np.ndarray
            Boolean mask representing original missingness (already missing in input) for the full dataset.
        train_idx_ : np.ndarray
            Indices of samples used for training.
        val_idx_ : np.ndarray
            Indices of samples used for validation.
        test_idx_ : np.ndarray
            Indices of samples used for testing.
        X_train_ : np.ndarray
            Corrupted genotype matrix (inputs) for training. Alias for `X_train_corrupted_`.
        y_train_ : np.ndarray
            Clean genotype matrix (targets) for training. Alias for `X_train_clean_`.
        X_val_ : np.ndarray
            Corrupted genotype matrix (inputs) for validation.
        y_val_ : np.ndarray
            Clean genotype matrix (targets) for validation.
        X_test_ : np.ndarray
            Corrupted genotype matrix (inputs) for testing.
        y_test_ : np.ndarray
            Clean genotype matrix (targets) for testing.
        X_train_clean_ : np.ndarray
            Persisted clean training genotypes.
        X_train_corrupted_ : np.ndarray
            Persisted corrupted training genotypes.
        X_val_clean_ : np.ndarray
            Persisted clean validation genotypes.
        X_val_corrupted_ : np.ndarray
            Persisted corrupted validation genotypes.
        X_test_clean_ : np.ndarray
            Persisted clean test genotypes.
        X_test_corrupted_ : np.ndarray
            Persisted corrupted test genotypes.
        sim_mask_train_ : np.ndarray
            Simulated missingness mask for training data.
        sim_mask_val_ : np.ndarray
            Simulated missingness mask for validation data.
        sim_mask_test_ : np.ndarray
            Simulated missingness mask for test data.
        orig_mask_train_ : np.ndarray
            Original missingness mask for training data.
        orig_mask_val_ : np.ndarray
            Original missingness mask for validation data.
        orig_mask_test_ : np.ndarray
            Original missingness mask for test data.
        eval_mask_train_ : np.ndarray
            Evaluation mask for training data (intersection of simulated mask and observed data).
        eval_mask_val_ : np.ndarray
            Evaluation mask for validation data.
        eval_mask_test_ : np.ndarray
            Evaluation mask for test data.
        train_loader_ : torch.utils.data.DataLoader
            PyTorch DataLoader for iterating over training batches.
        val_loader_ : torch.utils.data.DataLoader
            PyTorch DataLoader for iterating over validation batches.
        plotter_ : PrettyPlotter
            Plotting utilities for visualizing training progress and results.
        scorers_ : Scorer
            Scoring functions for evaluating imputation performance.

    References:
        Gashler, M.S., Smith, M.R., Morris, R., & Martinez, T.R. (2014). Missing Value Imputation with Unsupervised Backpropagation. Computational Intelligence, 32(2), 196-215. https://doi.org/10.1111/coin.12048
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        tree_parser: Optional["TreeParser"] = None,
        config: Optional[Union[UBPConfig, dict, str]] = None,
        overrides: Optional[dict] = None,
        sim_strategy: Optional[str] = None,
        sim_prop: Optional[float] = None,
        sim_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize the ImputeUBP model.

        Args:
            genotype_data (GenotypeData): Genotype data object.
            tree_parser (Optional[TreeParser]): Tree parser for nonrandom missingness simulation.
            config (Optional[Union[UBPConfig, dict, str]]): Configuration for UBP model.
            overrides (Optional[dict]): Dot-notation overrides for config.
            sim_strategy (Optional[str]): Missingness simulation strategy.
            sim_prop (Optional[float]): Proportion of data to simulate as missing.
            sim_kwargs (Optional[dict]): Additional kwargs for simulation.
        """
        self.model_name = "ImputeUBP"
        self.genotype_data = genotype_data
        self.tree_parser = tree_parser

        cfg = ensure_ubp_config(config)
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

        self.Model = UBPModel
        self.pgenc = GenotypeEncoder(genotype_data)

        # I/O and global
        self.seed = self.cfg.io.seed
        self.prefix = self.cfg.io.prefix
        self.rng = np.random.default_rng(self.seed)

        # Simulation controls
        sim_cfg = getattr(self.cfg, "sim", None)
        sim_cfg_kwargs = copy.deepcopy(getattr(sim_cfg, "sim_kwargs", None) or {})
        if sim_kwargs:
            sim_cfg_kwargs.update(sim_kwargs)

        self.simulate_missing = True
        self.sim_strategy = sim_strategy or (
            sim_cfg.sim_strategy if sim_cfg else "random"
        )
        self.sim_prop = float(
            sim_prop if sim_prop is not None else (sim_cfg.sim_prop if sim_cfg else 0.2)
        )
        self.sim_kwargs = sim_cfg_kwargs

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

        # Training Parameters from Paper
        self.batch_size = int(self.cfg.train.batch_size)
        self.learning_rate = float(self.cfg.train.learning_rate)  # eta
        self.l1_penalty = float(self.cfg.train.l1_penalty)  # lambda
        self.gamma_threshold = 1e-4
        self.eta_min = 1e-6
        self.validation_split = float(self.cfg.train.validation_split)

        # Loss / Weighting
        self.gamma = float(getattr(self.cfg.train, "gamma", 2.0))
        self.power = float(getattr(self.cfg.train, "weights_power", 1.0))
        self.normalize = bool(getattr(self.cfg.train, "weights_normalize", True))
        self.inverse = bool(getattr(self.cfg.train, "weights_inverse", False))
        self.max_ratio = getattr(self.cfg.train, "weights_max_ratio", None)
        self.gamma_schedule = bool(getattr(self.cfg.train, "gamma_schedule", False))
        self.gamma_init = float(getattr(self.cfg.train, "gamma_init", 0.0))

        self.epochs = self.cfg.train.max_epochs

        # Tuning parameters
        self.tune = self.cfg.tune.enabled
        self.tune_metric: str | list[str] | tuple[str, ...]
        self.tune_metric = self.cfg.tune.metrics
        self.primary_metric = self.validate_tuning_metric()
        self.n_trials = int(self.cfg.tune.n_trials)

        # Plotting
        self.show_plots = bool(self.cfg.plot.show)
        self.use_multiqc = bool(self.cfg.plot.multiqc)

        # State
        self.is_haploid_: bool = False
        self.num_classes_: int = 3
        self.total_samples_: int = 0
        self.model_params: dict[str, Any] = {}

        # UBP-specific
        self.projection_lr = float(self.cfg.ubp.projection_lr)
        self.projection_epochs = int(self.cfg.ubp.projection_epochs)

        self.num_tuned_params_ = OBJECTIVE_SPEC_UBP.count()

    def _haploidize_012(self, arr: np.ndarray) -> np.ndarray:
        """Convert diploid 012 to haploid 01 encoding, preserve missing (-1).

        Args:
            arr (np.ndarray): np.ndarray of shape (N, M) in 012 encoding.

        Returns:
            np.ndarray: np.ndarray of shape (N, M) in 01 encoding.
        """
        out = arr.astype(np.int8, copy=True)
        miss = out < 0
        out = np.where(out > 0, 1, out).astype(np.int8, copy=False)
        out[miss] = -1
        return out

    def fit(self) -> "ImputeUBP":
        """Fit the UBP model using the 3-phase algorithm.

        This method coordinates the entire training pipeline:
            1.  **Preprocessing:** Validates ploidy, encodes genotypes (0/1/2), and simulates missingness for self-supervised validation.
            2.  **Split:** partitions data into Train, Validation, and Test sets based on the simulated masks.
            3.  **Initialization:** Performs PCA on the training set to initialize the latent embeddings (Phase 1 variant).
            4.  **Training:** Executes Phase 2 (Decoder Refinement) and Phase 3 (Joint Refinement) loops, optionally using Optuna for hyperparameter tuning.

        Returns:
            ImputeUBP: The fitted instance.

        Raises:
            AttributeError: If `genotype_data` does not contain loaded SNPs.
            ValueError: If ploidy is not 1 (haploid) or 2 (diploid).
            RuntimeError: If training fails to converge or produces non-finite loss.
        """
        self.logger.info(f"Fitting {self.model_name} model...")

        self.logger.debug(
            f"[{self.model_name}] Initializing fit. Device: {self.device}, Seed: {self.seed}"
        )

        if self.genotype_data.snp_data is None:
            msg = f"SNP data is required for {self.model_name}."
            self.logger.error(msg)
            raise AttributeError(msg)

        self.ploidy = self.cfg.io.ploidy
        self.is_haploid_ = self.ploidy == 1
        self.num_classes_ = 2 if self.is_haploid_ else 3

        if self.ploidy > 2 or self.ploidy < 1:
            msg = f"{self.model_name} currently supports only haploid (1) or diploid (2) data; got ploidy={self.ploidy}."
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.debug(
            f"Ploidy set to {self.ploidy}, is_haploid: {self.is_haploid_}"
        )

        # Prepare Data
        gt_full = self.pgenc.genotypes_012.copy()
        gt_full[gt_full < 0] = -1
        gt_full = np.nan_to_num(gt_full, nan=-1.0)
        self.ground_truth_ = gt_full.astype(np.int8)
        self.num_features_ = int(self.ground_truth_.shape[1])
        self.total_samples_ = int(self.ground_truth_.shape[0])

        self.model_params = {
            "num_embeddings": self.total_samples_,
            "n_features": self.num_features_,
            "prefix": self.prefix,
            "num_classes": self.num_classes_,
            "latent_dim": self.latent_dim,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "device": self.device,
            "debug": self.debug,
        }

        self.logger.debug(f"Model parameters: {self.model_params}")

        if self.is_haploid_:
            self.logger.debug("Performing haploid harmonization...")
            self.ground_truth_ = self._haploidize_012(self.ground_truth_)

        # Simulate missingness on the full matrix
        sim_tup = self.sim_missing_transform(self.ground_truth_)
        X_for_model_full = sim_tup[0]
        self.sim_mask_ = sim_tup[1]
        self.orig_mask_ = sim_tup[2]

        X_for_model_full = sim_tup[0].astype(np.int8, copy=False)
        if self.is_haploid_:
            self.logger.debug("Performing haploid conversion on full inputs...")
            X_for_model_full = self._haploidize_012(X_for_model_full)

        # Validate sim and orig masks; there should not be any overlap.
        # Also checks if there are enough sites to evaluate.
        self._validate_sim_and_orig_masks(
            sim_mask=self.sim_mask_, orig_mask=self.orig_mask_, context="full"
        )

        # Split indices based on clean ground truth
        indices = self._train_val_test_split(self.ground_truth_)
        self.train_idx_, self.val_idx_, self.test_idx_ = indices

        self.logger.info(
            f"Train/val/test sizes: {len(self.train_idx_)}/{len(self.val_idx_)}/{len(self.test_idx_)}"
        )

        # --- Split matrices ---
        corrupteds = self._extract_masks_indices(X_for_model_full, indices)
        X_train_corrupted, X_val_corrupted, X_test_corrupted = corrupteds

        cleans = self._extract_masks_indices(self.ground_truth_, indices)
        X_train_clean, X_val_clean, X_test_clean = cleans

        # --- Masks per split ---
        sm = self._extract_masks_indices(self.sim_mask_, indices)
        self.sim_mask_train_, self.sim_mask_val_, self.sim_mask_test_ = sm

        om = self._extract_masks_indices(self.orig_mask_, indices)
        self.orig_mask_train_, self.orig_mask_val_, self.orig_mask_test_ = om

        self.validate_and_log_masks()

        # --- Haploid harmonization ---
        if self.is_haploid_:
            self.logger.debug(
                "Performing haploid harmonization on split inputs/targets..."
            )

            def _haploidize(arr):
                out = np.where(arr > 0, 1, arr).astype(np.int8, copy=True)
                out[arr < 0] = -1
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

        # Final training tensors/matrices used by the pipeline
        # Convention: X_* are corrupted inputs; y_* are clean targets
        self.X_train_ = self.X_train_corrupted_
        self.y_train_ = self.X_train_clean_

        self.X_val_ = self.X_val_corrupted_
        self.y_val_ = self.X_val_clean_

        self.X_test_ = self.X_test_corrupted_
        self.y_test_ = self.X_test_clean_

        self.plotter_, self.scorers_ = self.initialize_plotting_and_scorers()

        # Observed positions are those not originally missing
        # and not simulated-missing
        train_obs = X_train_corrupted >= 0
        val_obs = X_val_corrupted >= 0

        # Evaluation mask: entries we intentionally hid
        # (but were not originally missing)
        self.eval_mask_train_ = self.sim_mask_train_ & ~self.orig_mask_train_
        self.eval_mask_val_ = self.sim_mask_val_ & ~self.orig_mask_val_
        self.eval_mask_test_ = self.sim_mask_test_ & ~self.orig_mask_test_

        train_loader = self._get_ubp_loaders(
            self.train_idx_,
            self.y_train_,
            mask=train_obs,
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = self._get_ubp_loaders(
            self.val_idx_,
            self.y_val_,
            mask=val_obs,
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.train_loader_ = train_loader
        self.val_loader_ = val_loader

        if self.debug:
            assert np.array_equal(train_obs, self.X_train_corrupted_ >= 0)
            assert np.array_equal(val_obs, self.X_val_corrupted_ >= 0)
            em = self.eval_mask_test_
            assert np.all(self.X_test_corrupted_[em] < 0)

        # Tuning
        if self.tune:
            self.tuned_params_ = self.tune_hyperparameters()
            self.model_tuned_ = True
        else:
            self.model_tuned_ = False

            train_loss_mask = self.X_train_corrupted_ >= 0
            self.class_weights_ = self._class_weights_from_zygosity(
                self.y_train_,
                train_mask=train_loss_mask,  # NOTE: # not sim_mask
                inverse=self.inverse,
                normalize=self.normalize,
                max_ratio=self.max_ratio,
                power=self.power,
            )

            # --- Sanitize Haploid/Invalid Weights ---
            if self.class_weights_ is not None:
                # 1. Truncate dimension if needed
                # (Haploid returns 3 weights for 2 classes)
                if self.is_haploid_ and self.class_weights_.numel() > self.num_classes_:
                    self.logger.warning(
                        f"Haploid mode: Truncating class weights from {self.class_weights_.shape} to {self.num_classes_}."
                    )
                    self.class_weights_ = self.class_weights_[: self.num_classes_]

                # 2. Check for NaN/Inf (caused by 0 counts in inverse freq)
                if not torch.isfinite(self.class_weights_).all():
                    self.logger.warning(
                        f"Class weights contain NaN/Inf ({self.class_weights_}). "
                        "This usually happens with rare variants in small splits. Resetting to uniform weights."
                    )
                    self.class_weights_ = torch.ones(
                        self.num_classes_, device=self.device
                    )

            keys = OBJECTIVE_SPEC_UBP.keys
            self.tuned_params_ = {k: getattr(self, k) for k in keys}
            self.tuned_params_["model_params"] = self.model_params

        self.logger.debug(f"Tuned Parameters Dictionary: {self.tuned_params_}")

        self.best_params_ = copy.deepcopy(self.tuned_params_)
        self._log_class_weights()

        # Final model params
        # (compute hidden sizes using n_inputs=L*K)
        input_dim = int(self.num_features_ * self.num_classes_)

        self.logger.debug(
            f"[{self.model_name}] Initializing PCA embedding as Phase 1..."
        )

        self.v_init_ = self._get_pca_embedding_init(
            self.ground_truth_, self.train_idx_, int(self.best_params_["latent_dim"])
        )

        self.logger.debug(f"PCA Embedding Initialization: {self.v_init_}")

        model_params_final = {
            "num_embeddings": self.total_samples_,
            "n_features": int(self.num_features_),
            "prefix": str(self.prefix),
            "num_classes": int(self.num_classes_),
            "latent_dim": int(self.best_params_["latent_dim"]),
            "dropout_rate": float(self.best_params_["dropout_rate"]),
            "activation": str(self.best_params_["activation"]),
            "embedding_init": self.v_init_,
            "device": self.device,
            "debug": self.debug,
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

        # Build Model
        model = self.build_model(self.Model, self.best_params_["model_params"])
        # NOTE: No general init here; Model init handles V (PCA)

        # Run 3-Phase Training
        loss, trained_model, history = self._execute_ubp_training(
            model=model,
            lr=float(self.best_params_["learning_rate"]),
            l1_penalty=float(self.best_params_["l1_penalty"]),
            params=self.best_params_,
            trial=None,
            class_weights=self.class_weights_,
            gamma_schedule=self.gamma_schedule,
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

        torch.save(
            trained_model.state_dict(),
            self.models_dir / f"final_model_{self.model_name}.pt",
        )
        self.model_ = trained_model
        self.is_fit_ = True

        # Test Eval
        self._evaluate_model(
            self.model_,
            self.X_test_corrupted_,
            self.y_test_,
            self.eval_mask_test_,
            indices=self.test_idx_,
            gamma=self.gamma,
            project_embedding=True,
            objective_mode=False,
            class_weights=self.class_weights_,
            persist_projection=False,
        )

        if self.show_plots:
            self.plotter_.plot_history(self.history_)

        self._save_display_model_params(is_tuned=self.model_tuned_)

        self.logger.info(f"{self.model_name} fitting complete!")
        return self

    def transform(self) -> np.ndarray:
        """Impute missing values by projecting samples onto the learned manifold.

        Unlike simple prediction, this method iteratively refines the embeddings :math:`V` for all samples to minimize the reconstruction error of the *observed* genotypes, given the fixed weights :math:`W` learned during `fit()`. Once the embeddings settle, the decoder generates the missing values.

        Returns:
            np.ndarray: The fully imputed genotype matrix in 0/1/2 encoding.

        Raises:
            NotFittedError: If the model has not been trained yet.
            RuntimeError: If imputation results in 'N' (invalid) characters.
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

        self.logger.debug(
            f"[{self.model_name}] Starting embedding projection (refining V with W fixed)."
        )

        # Project all samples (Train/Val/Test) to current manifold
        self._refine_all_embeddings(
            self.model_,
            self.ground_truth_,
            self.gamma,
            class_weights=self.class_weights_,
            lr=self.projection_lr,
            iterations=int(self.projection_epochs * 5),  # Final projection
        )

        self.logger.debug(
            f"[{self.model_name}] Projection complete. Generating predictions..."
        )

        if self.ground_truth_.ndim != 2:
            msg = f"{self.model_name}.ground_truth_ must be 2D; got shape {self.ground_truth_.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        full_indices = np.arange(self.total_samples_)
        pred_labels, _ = self._predict(self.model_, indices=full_indices)
        if pred_labels.shape != self.ground_truth_.shape:
            msg = (
                f"{self.model_name} prediction shape mismatch: "
                f"pred_labels={pred_labels.shape}, ground_truth_={self.ground_truth_.shape}."
            )
            self.logger.error(msg)
            raise RuntimeError(msg)

        imputed = self.ground_truth_.copy()
        missing_mask = imputed < 0
        imputed[missing_mask] = pred_labels[missing_mask]

        # 3. Haploid decode uses diploid decoder semantics: map ALT=1 -> 2
        decode_input = imputed
        if getattr(self, "is_haploid_", False):
            decode_input = imputed.copy()
            decode_input[decode_input == 1] = 2

        decoded = self.decode_012(decode_input)

        if getattr(self, "is_haploid_", False):
            decoded = self._sanitize_haploid_decoded_output(decoded)

        if (decoded == "N").any():
            msg = f"Something went wrong: {self.model_name} imputation still contains {(decoded == 'N').sum()} missing values ('N')."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if self.show_plots:
            orig = self.ground_truth_.copy()

            if getattr(self, "is_haploid_", False):
                orig[orig == 1] = 2

            orig_dec = self.decode_012(orig)
            self.plotter_.plot_gt_distribution(decoded, orig_dec, True)

        self.logger.info(f"{self.model_name} Imputation complete!")

        return decoded

    def _execute_ubp_training(
        self,
        model: nn.Module,
        lr: float,
        l1_penalty: float,
        params: dict[str, Any],
        trial: Optional[optuna.Trial],
        class_weights: Optional[torch.Tensor],
        gamma_schedule: bool,
    ) -> Tuple[float, nn.Module, dict[str, list[float]]]:
        """Execute the 3-phase UBP training procedure.

        This method orchestrates the training of the UBP model through its three distinct phases:
        1.  **Phase 2 (Decoder Refinement):** With the latent embeddings :math:`V` frozen, this phase optimizes the decoder weights :math:`W` to accurately reconstruct the genotypes from the PCA-initialized embeddings.
        2.  **Phase 3 (Joint Refinement):** Both the embeddings :math:`V` and the decoder weights :math:`W` are optimized simultaneously, allowing the model to capture non-linear structures in the data.

        Args:
            model (nn.Module): The UBP model to train.
            lr (float): Learning rate for optimizer.
            l1_penalty (float): L1 regularization penalty.
            params (dict[str, Any]): Hyperparameters for training.
            trial (Optional[optuna.Trial]): Optuna trial for hyperparameter tuning.
            class_weights (Optional[torch.Tensor]): Class weights for Focal Loss.
            gamma_schedule (bool): Whether to use gamma scheduling.

        Returns:
            Tuple[float, nn.Module, dict[str, list[float]]]: Best validation score, trained model, and training histories.

        Raises:
            RuntimeError: If training fails to converge or produces non-finite loss.
        """

        gamma_target, gamma_warm, gamma_ramp = self._anneal_config(
            params, "gamma", default=self.gamma, max_epochs=self.epochs
        )

        cw = class_weights
        if cw is not None and cw.device != self.device:
            cw = cw.to(self.device)

        ce_criterion = FocalCELoss(
            alpha=cw, gamma=gamma_target, reduction="mean", ignore_index=-1
        )

        self.logger.debug(
            f"[{self.model_name}] Starting Phase 2: Refine weights W (freeze V)..."
        )

        histories = {}

        _, histories["Phase2"] = self._run_phase_loop(
            model=model,
            temp_layer=None,
            phase=2,
            lr=lr,
            l1=l1_penalty,
            criterion=ce_criterion,
            trial=trial,
            params=params,
            gamma_schedule=gamma_schedule,
        )

        self.logger.debug(
            f"[{self.model_name}] Starting Phase 3: Joint refinement of V and W..."
        )

        best_score, histories["Phase3"] = self._run_phase_loop(
            model=model,
            temp_layer=None,
            phase=3,
            lr=lr,
            l1=0.0,
            criterion=ce_criterion,
            trial=trial,
            params=params,
            gamma_schedule=gamma_schedule,
        )

        return best_score, model, histories

    def _train_epoch(
        self,
        model: nn.Module,
        temp_layer: Optional[nn.Module],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        l1: float,
        phase: int,
        trial: Optional[optuna.Trial] = None,
    ) -> float:
        """Train one epoch for a given UBP phase.

        Args:
            model (nn.Module): UBP model to train.
            temp_layer (Optional[nn.Module]): Temporary layer for phase 1 (None otherwise).
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (nn.Module): Loss criterion to use.
            l1 (float): L1 regularization penalty.
            phase (int): UBP phase (1, 2, or 3).
            trial (Optional[optuna.Trial]): Optuna trial for pruning. Can be None

        Returns:
            float: Average training loss for the epoch.
        """
        model.train()
        if temp_layer is not None:
            temp_layer.train()

        running = 0.0
        n_batches = 0

        # If some outer code put us under no_grad/inference_mode, this fixes it.
        with torch.enable_grad():
            for idx, y, m in self.train_loader_:
                idx = idx.to(self.device, non_blocking=True).long()
                y = y.to(self.device, non_blocking=True).long()
                m = m.to(self.device, non_blocking=True).bool()

                flat_mask = m.view(-1)
                y_flat = y.view(-1)
                valid_targets = (y_flat >= 0) & (y_flat < self.num_classes_)
                flat_mask = flat_mask & valid_targets
                if flat_mask.sum().item() == 0:
                    continue

                optimizer.zero_grad(set_to_none=True)

                if phase == 1:
                    if temp_layer is None:
                        msg = f"[{self.model_name}] phase=1 requires temp_layer, got None."
                        self.logger.error(msg)
                        raise RuntimeError(msg)

                    z = model.embedding(idx)  # type: ignore[attr-defined]

                    logits = temp_layer(z).view(
                        -1, self.num_features_, self.num_classes_
                    )
                else:
                    logits = model(idx)  # (B, L, K)

                logits_2d = logits.view(-1, self.num_classes_)[flat_mask]
                targets_1d = y_flat[flat_mask]

                # --- Clamp Logits for Numerical Stability ---
                # Prevents huge values from causing log(0) or exp(inf) in FocalLoss
                logits_2d = torch.clamp(logits_2d, min=-30.0, max=30.0)

                loss = criterion(logits_2d, targets_1d)

                # --- L1 Regularization ---
                if l1 > 0.0:
                    reg = torch.zeros((), device=self.device)
                    if phase == 1:
                        # Phase 1: Regulate embedding + temp layer
                        reg = reg + model.embedding.weight.abs().sum()  # type: ignore[attr-defined]
                        reg = reg + temp_layer.weight.abs().sum()  # type: ignore[union-attr]
                    else:
                        # Phase 2 & 3: Regulate network weights
                        for p in model.hidden_layers.parameters():  # type: ignore[attr-defined]
                            reg = reg + p.abs().sum()
                        for p in model.dense_output.parameters():  # type: ignore[attr-defined]
                            reg = reg + p.abs().sum()

                        # NEW: In Phase 3, we MUST regulate embeddings too,
                        # otherwise V explodes while W shrinks (Scale Ambiguity)
                        if phase == 3:
                            reg = reg + model.embedding.weight.abs().sum()  # type: ignore[attr-defined]

                    loss = loss + float(l1) * reg

                # ---- Critical diagnostic ----
                if not loss.requires_grad:
                    grad_enabled = torch.is_grad_enabled()
                    opt_reqs = [
                        p.requires_grad
                        for pg in optimizer.param_groups
                        for p in pg["params"]
                    ]
                    msg = (
                        f"[{self.model_name}] loss.requires_grad=False at backward.\n"
                        f"  grad_enabled={grad_enabled}\n"
                        f"  phase={phase}\n"
                        f"  any_optimizer_param_requires_grad={any(opt_reqs)}\n"
                        f"  logits.requires_grad={bool(getattr(logits, 'requires_grad', False))}\n"
                        f"  logits_2d.requires_grad={bool(getattr(logits_2d, 'requires_grad', False))}\n"
                        f"  temp_layer={'present' if temp_layer is not None else 'None'}\n"
                        "This usually means you are inside an outer no_grad/inference_mode context OR "
                        "the model forward path is detaching tensors / all params are frozen."
                    )
                    self.logger.error(msg)
                    raise RuntimeError(msg)

                # Non-finite checks
                if not torch.isfinite(loss):
                    if trial is not None:
                        msg = f"[{self.model_name}] Trial {trial.number} training loss non-finite. Pruning trial."
                        self.logger.warning(msg)
                        raise optuna.exceptions.TrialPruned(msg)
                    msg = f"[{self.model_name}] Training loss non-finite."
                    self.logger.error(msg)
                    raise RuntimeError(msg)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]["params"], max_norm=1.0
                )
                optimizer.step()

                running += float(loss.detach().item())
                n_batches += 1

        if n_batches == 0:
            msg = f"[{self.model_name}] Training loss has no valid batches."
            self.logger.error(msg)
            raise RuntimeError(msg)

        return running / n_batches

    def _get_ubp_loaders(
        self,
        indices: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> torch.utils.data.DataLoader:
        """Create DataLoader yielding (idx, y, mask) batches.

        Args:
            indices (np.ndarray): Sample indices, shape (B_total,).
            y (np.ndarray): Target genotypes (012 or 01), shape (B_total, L).
            mask (np.ndarray): Observed-entry mask, shape (B_total, L). True = observed.
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle.

        Returns:
            torch.utils.data.DataLoader: Yields (idx, y, mask) where:
                - idx is int64 (for nn.Embedding)
                - y is int64 (for CE-style losses)
                - mask is bool

        Raises:
            ValueError: If shapes are incompatible or dimensions are unexpected.
        """
        idx_np = np.asarray(indices, dtype=np.int64)
        y_np = np.asarray(y, dtype=np.int64)
        m_np = np.asarray(mask, dtype=bool)

        if idx_np.ndim != 1:
            idx_np = idx_np.reshape(-1)

        if y_np.ndim != 2:
            msg = f"y must be 2D (n_samples, n_features); got y.ndim={y_np.ndim}, shape={y_np.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        if m_np.ndim != 2:
            msg = f"mask must be 2D (n_samples, n_features); got mask.ndim={m_np.ndim}, shape={m_np.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        if y_np.shape != m_np.shape:
            msg = f"y/mask shape mismatch: y={y_np.shape}, mask={m_np.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        if y_np.shape[0] != idx_np.shape[0]:
            msg = f"Loader alignment mismatch: idx={idx_np.shape}, y={y_np.shape}, mask={m_np.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        # Optional sanity:
        # feature dimension should match expected L when available.
        if getattr(self, "num_features_", None) is not None:
            L_expected = int(self.num_features_)
            if y_np.shape[1] != L_expected:
                msg = f"Feature-dimension mismatch: expected L={L_expected}, got y.shape[1]={y_np.shape[1]}."
                self.logger.error(msg)
                raise ValueError(msg)

        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(idx_np).long(),
            torch.from_numpy(y_np).long(),
            torch.from_numpy(m_np).bool(),
        )

        return torch.utils.data.DataLoader(
            ds,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=0,
            pin_memory=str(self.device).startswith("cuda"),
        )

    def _run_phase_loop(
        self,
        model: nn.Module,
        temp_layer: Optional[nn.Module],
        phase: int,
        lr: float,
        l1: float,
        criterion: nn.Module,
        trial: Optional[optuna.Trial] = None,
        params: Optional[dict[str, Any]] = None,
        gamma_schedule: bool = False,
    ) -> tuple[float, dict[str, list[float]]]:
        """Run a UBP phase with ReduceLROnPlateau LR scheduling and optional gamma scheduling.

        Adds robustness for:
        - non-finite validation scores (NaN/Inf) to avoid scheduler crashes
        - graceful Optuna pruning on invalid metrics
        - consistent stopping at LR floor + plateau

        Args:
            model (nn.Module): UBP model to train.
            temp_layer (Optional[nn.Module]): Temporary layer for phase 1 (None otherwise).
            phase (int): UBP phase (1, 2, or 3).
            lr (float): Initial learning rate.
            l1 (float): L1 penalty.
            criterion (nn.Module): Loss criterion to use.
            trial (Optional[optuna.Trial]): Optuna trial for pruning (phase 3 only).
            params (Optional[dict[str, Any]]): Full parameter dict (for scheduling). Can be None.
            gamma_schedule (bool): Whether to use gamma scheduling.

        Returns:
            tuple[float, dict[str, list[float]]]: Best validation score and histories.

        Raises:
            ValueError: If phase is invalid.
            RuntimeError: If validation consistently produces no valid batches.
            optuna.exceptions.TrialPruned: If trial should be pruned.
        """
        eta0 = float(lr)
        s_best = float("inf")

        patience = 5
        plateau_counter = 0

        # Freeze everything first
        for p in model.parameters():
            p.requires_grad_(False)
        if temp_layer is not None:
            for p in temp_layer.parameters():
                p.requires_grad_(False)

        params_to_opt: list[torch.nn.Parameter] = []

        if phase == 1:
            model.embedding.weight.requires_grad_(True)  # type: ignore[attr-defined]
            params_to_opt.append(model.embedding.weight)  # type: ignore[attr-defined]
            if temp_layer is None:
                msg = f"[{self.model_name}] phase=1 requires temp_layer, got None."
                self.logger.error(msg)
                raise RuntimeError(msg)
            for p in temp_layer.parameters():
                p.requires_grad_(True)
            params_to_opt.extend(list(temp_layer.parameters()))

        elif phase == 2:
            for p in model.hidden_layers.parameters():  # type: ignore[attr-defined]
                p.requires_grad_(True)
            for p in model.dense_output.parameters():  # type: ignore[attr-defined]
                p.requires_grad_(True)
            params_to_opt.extend(list(model.hidden_layers.parameters()))  # type: ignore[attr-defined]
            params_to_opt.extend(list(model.dense_output.parameters()))  # type: ignore[attr-defined]

        elif phase == 3:
            model.embedding.weight.requires_grad_(True)  # type: ignore[attr-defined]
            params_to_opt.append(model.embedding.weight)  # type: ignore[attr-defined]
            for p in model.hidden_layers.parameters():  # type: ignore[attr-defined]
                p.requires_grad_(True)
            for p in model.dense_output.parameters():  # type: ignore[attr-defined]
                p.requires_grad_(True)
            params_to_opt.extend(list(model.hidden_layers.parameters()))  # type: ignore[attr-defined]
            params_to_opt.extend(list(model.dense_output.parameters()))  # type: ignore[attr-defined]

        else:
            msg = f"Invalid phase={phase}. Must be 1, 2, or 3."
            self.logger.error(msg)
            raise ValueError(msg)

        if self.debug:
            total_params = sum(p.numel() for p in model.parameters())
            if temp_layer is not None:
                total_params += sum(p.numel() for p in temp_layer.parameters())
            trainable_params = sum(p.numel() for p in params_to_opt)
            ratio = (
                (float(trainable_params) / float(total_params))
                if total_params > 0
                else 0.0
            )
            self.logger.debug(
                f"[{self.model_name}] Phase {phase} trainable params={trainable_params}/{total_params} ({ratio:.2%})."
            )

        optimizer = torch.optim.AdamW(params_to_opt, lr=eta0)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=int(patience),
            threshold=float(self.gamma_threshold),
            threshold_mode="rel",
            cooldown=0,
            min_lr=float(self.eta_min),
        )

        gamma_target, gamma_warm, gamma_ramp = self._anneal_config(
            params, "gamma", default=self.gamma, max_epochs=self.epochs
        )

        train_history: list[float] = []
        val_history: list[float] = []

        epoch = 0
        while epoch < int(self.epochs):
            if gamma_schedule:
                gamma_current = self._update_anneal_schedule(
                    gamma_target,
                    warm=gamma_warm,
                    ramp=gamma_ramp,
                    epoch=epoch,
                    init_val=float(getattr(self, "gamma_init", 0.0)),
                )
                criterion.gamma = gamma_current  # type: ignore[attr-defined]

            train_loss = self._train_epoch(
                model=model,
                temp_layer=temp_layer,
                optimizer=optimizer,
                criterion=criterion,
                l1=l1,
                phase=phase,
                trial=trial,
            )

            try:
                s = self._val_step_with_projection(
                    model=model,
                    temp_layer=temp_layer,
                    criterion=criterion,
                    steps=max(int(self.projection_epochs) // 5, 20),
                    lr=float(self.projection_lr),
                )
            except Exception as e:
                # If validation cannot be computed,
                # prune trial (if applicable) or raise.
                if trial is not None:
                    raise optuna.exceptions.TrialPruned(
                        f"[{self.model_name}] Trial {trial.number} failed during validation: {str(e)}"
                    ) from e
                raise

            train_history.append(float(train_loss))
            val_history.append(float(s))

            # Non-finite validation score handling
            # (critical for ReduceLROnPlateau stability)
            if not np.isfinite(s):
                if trial is not None:
                    raise optuna.exceptions.TrialPruned(
                        f"[{self.model_name}] Trial {trial.number} produced non-finite validation score (s={s})."
                    )
                # Treat as "worst possible" for scheduler and plateau logic
                s_for_sched = float("inf")
                plateau_counter += 1
            else:
                s_for_sched = float(s)

                # Track best + plateau counter
                if s < s_best:
                    if s_best == float("inf"):
                        s_best = s
                        plateau_counter = 0
                    else:
                        improvement_ratio = (
                            (1.0 - (s / s_best)) if s_best != 0.0 else 0.0
                        )
                        if improvement_ratio > float(self.gamma_threshold):
                            s_best = s
                            plateau_counter = 0
                        else:
                            plateau_counter += 1
                else:
                    plateau_counter += 1

            lr_before = float(optimizer.param_groups[0]["lr"])
            scheduler.step(s_for_sched)
            lr_after = float(optimizer.param_groups[0]["lr"])

            if lr_after < lr_before:
                self.logger.debug(
                    f"Phase {phase}: ReduceLROnPlateau LR {lr_before:.2e} -> {lr_after:.2e} "
                    f"(train={train_loss:.4f}, val={s_for_sched:.4f}, gamma={float(getattr(criterion, 'gamma', 0.0)):.3f})"
                )

            at_floor = lr_after <= (float(self.eta_min) * (1.0 + 1e-12))
            if at_floor and plateau_counter >= int(patience):
                break

            if trial is not None and isinstance(self.tune_metric, str) and phase == 3:
                trial.report(-float(s_for_sched), step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned(
                        f"[{self.model_name}] Trial {trial.number} pruned at epoch {epoch}. This is a normal part of the tuning process and is not an error."
                    )

            epoch += 1

        return float(s_best), {"Train": train_history, "Val": val_history}

    def _val_step_with_projection(
        self,
        model: nn.Module,
        temp_layer: Optional[nn.Module],
        criterion: nn.Module,
        steps: int = 20,
        lr: float = 0.05,
    ) -> float:
        """Validate using per-batch projection of embeddings (V) while holding decoder fixed.

        Robustness additions:
        - explicit guards for empty/degenerate batches
        - explicit non-finite loss checks (pre and post)
        - restores requires_grad state exactly even if exceptions occur

        Args:
            model (nn.Module): UBP model.
            temp_layer (Optional[nn.Module]): Temp layer for phase 1 validation, if used.
            criterion (nn.Module): Loss criterion.
            steps (int): Projection steps per batch.
            lr (float): Projection learning rate.

        Returns:
            float: Mean validation loss across valid batches.

        Raises:
            RuntimeError: If there are no valid batches or loss becomes non-finite.
        """
        model.eval()
        total_loss = 0.0
        total_loss_pre = 0.0
        count = 0

        saved: list[tuple[torch.nn.Parameter, bool]] = []

        def _save_and_disable(module: Optional[nn.Module]) -> None:
            if module is None:
                return
            for p in module.parameters():
                saved.append((p, bool(p.requires_grad)))
                p.requires_grad_(False)

        _save_and_disable(model.hidden_layers)  # type: ignore[attr-defined]
        _save_and_disable(model.dense_output)  # type: ignore[attr-defined]
        _save_and_disable(temp_layer)

        try:
            with torch.enable_grad():
                for idx, y, m in self.val_loader_:
                    idx = idx.to(self.device, non_blocking=True).long()
                    y = y.to(self.device, non_blocking=True).long()
                    m = m.to(self.device, non_blocking=True).bool()

                    flat_mask = m.view(-1)
                    y_flat = y.view(-1)
                    valid_targets = (y_flat >= 0) & (y_flat < self.num_classes_)
                    flat_mask = flat_mask & valid_targets
                    if flat_mask.sum().item() == 0:
                        continue

                    # Optimize only the batch embeddings
                    v_batch = model.embedding(idx).detach().clone()  # type: ignore[attr-defined]

                    with torch.no_grad():
                        if temp_layer is not None:
                            out_pre = temp_layer(v_batch).view(
                                -1, self.num_features_, self.num_classes_
                            )
                        else:
                            out_pre = model(override_embeddings=v_batch)

                        # NOTE: consider moving stability into the loss implementation instead.
                        out_pre = torch.clamp(out_pre, min=-30.0, max=30.0)

                        loss_pre = criterion(
                            out_pre.view(-1, self.num_classes_)[flat_mask],
                            y_flat[flat_mask],
                        )
                        if not torch.isfinite(loss_pre):
                            raise RuntimeError(
                                f"[{self.model_name}] Pre-projection val loss non-finite."
                            )
                        total_loss_pre += float(loss_pre.item())

                    v_batch.requires_grad_(True)
                    proj_opt = torch.optim.AdamW([v_batch], lr=float(lr))

                    for _ in range(int(steps)):
                        proj_opt.zero_grad(set_to_none=True)

                        if temp_layer is not None:
                            out = temp_layer(v_batch).view(
                                -1, self.num_features_, self.num_classes_
                            )
                        else:
                            out = model(override_embeddings=v_batch)

                        loss = criterion(
                            out.view(-1, self.num_classes_)[flat_mask],
                            y_flat[flat_mask],
                        )
                        if not torch.isfinite(loss):
                            raise RuntimeError(
                                f"[{self.model_name}] Projection-step val loss non-finite."
                            )
                        loss.backward()
                        proj_opt.step()

                    with torch.no_grad():
                        if temp_layer is not None:
                            out_final = temp_layer(v_batch).view(
                                -1, self.num_features_, self.num_classes_
                            )
                        else:
                            out_final = model(override_embeddings=v_batch)

                        val_l = criterion(
                            out_final.view(-1, self.num_classes_)[flat_mask],
                            y_flat[flat_mask],
                        )
                        if not torch.isfinite(val_l):
                            raise RuntimeError(
                                f"[{self.model_name}] Post-projection val loss non-finite."
                            )
                        total_loss += float(val_l.item())
                        count += 1

            if count == 0:
                msg = f"[{self.model_name}] Validation loss has no valid batches."
                self.logger.error(msg)
                raise RuntimeError(msg)

            if self.debug:
                delta = (total_loss_pre - total_loss) / float(count)
                self.logger.debug(
                    f"[{self.model_name}] Projection val loss delta (pre-post)={delta:.6f} over {count} batches."
                )

            return float(total_loss / count)

        finally:
            for p, req in saved:
                p.requires_grad_(req)

    def _refine_all_embeddings(
        self,
        model: nn.Module,
        X_target: np.ndarray,
        gamma: float,
        *,
        indices: np.ndarray | None = None,
        lr: float = 0.05,
        class_weights: torch.Tensor | None = None,
        iterations: int = 100,
        trial: Optional[optuna.Trial] = None,
    ) -> None:
        """Refine embeddings V for given indices using observed entries only.

        Preserves your implementation strategy (optimize embedding.weight directly), but adds:
        - alignment/shape checks
        - degenerate-mask checks (no observed entries)
        - non-finite loss detection -> Optuna prune (if trial) or RuntimeError
        - always restores requires_grad state

        Args:
            model (nn.Module): Trained UBP model.
            X_target (np.ndarray): Matrix used as refinement target. Can be:
                - full matrix (N_total, L), with indices selecting rows, OR
                - subset matrix (len(indices), L) aligned to provided indices
            gamma (float): Focal gamma during refinement.
            indices (np.ndarray | None): Rows to refine; if None, refines all rows in X_target.
            lr (float): Refinement learning rate.
            class_weights (torch.Tensor | None): Optional focal alpha.
            iterations (int): Number of refinement iterations.
            trial (Optional[optuna.Trial]): Trial for prune-on-failure behavior.

        Raises:
            ValueError: On incompatible shapes.
            RuntimeError: On non-finite loss or no valid observed entries.
            optuna.exceptions.TrialPruned: If trial is provided and refinement fails.
        """
        model.eval()

        if indices is None:
            idx_np = np.arange(
                int(getattr(self, "total_samples_", len(X_target))), dtype=np.int64
            )
        else:
            idx_np = np.asarray(indices, dtype=np.int64).reshape(-1)

        X_np = np.asarray(X_target, dtype=np.int64)
        if X_np.ndim != 2:
            msg = f"[{self.model_name}] X_target must be 2D (n_samples, n_features); got shape={X_np.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        # Accept either subset-aligned or full-matrix input
        if X_np.shape[0] == idx_np.shape[0]:
            y_np = X_np
        elif int(getattr(self, "total_samples_", X_np.shape[0])) == X_np.shape[0]:
            y_np = X_np[idx_np]
        else:
            msg = (
                f"[{self.model_name}] X_target has incompatible first dimension for indices. "
                f"Got X_target.shape[0]={X_np.shape[0]}, len(indices)={idx_np.shape[0]}, "
                f"total_samples_={getattr(self, 'total_samples_', 'NA')}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Observed-only refinement objective
        m_np = y_np >= 0
        if not bool(m_np.any()):
            msg = f"[{self.model_name}] Embedding refinement has zero observed entries to optimize."
            self.logger.error(msg)
            if trial is not None:
                raise optuna.exceptions.TrialPruned(msg)
            raise RuntimeError(msg)

        loader = self._get_ubp_loaders(
            idx_np,
            y_np,
            m_np,
            batch_size=int(self.batch_size),
            shuffle=False,
        )

        alpha = (
            class_weights
            if class_weights is not None
            else getattr(self, "class_weights_", None)
        )
        if alpha is not None and alpha.device != self.device:
            alpha = alpha.to(self.device)

        criterion = FocalCELoss(alpha=alpha, gamma=float(gamma), ignore_index=-1)

        saved: list[tuple[torch.nn.Parameter, bool]] = []
        for p in model.parameters():
            saved.append((p, bool(p.requires_grad)))
            p.requires_grad_(False)

        model.embedding.weight.requires_grad_(True)  # type: ignore[attr-defined]
        opt = torch.optim.AdamW([model.embedding.weight], lr=float(lr))  # type: ignore[attr-defined]

        try:
            with torch.enable_grad():
                did_any_update = False

                for _ in range(int(iterations)):
                    for idx_b, y_b, m_b in loader:
                        idx_b = idx_b.to(self.device, non_blocking=True).long()
                        y_b = y_b.to(self.device, non_blocking=True).long()
                        m_b = m_b.to(self.device, non_blocking=True).bool()

                        flat_mask = m_b.view(-1)
                        if flat_mask.sum().item() == 0:
                            continue

                        opt.zero_grad(set_to_none=True)

                        out = model(idx_b)
                        loss = criterion(
                            out.view(-1, self.num_classes_)[flat_mask],
                            y_b.view(-1)[flat_mask],
                        )

                        if not torch.isfinite(loss):
                            msg = f"[{self.model_name}] Embedding refinement loss non-finite."
                            self.logger.error(msg)
                            if trial is not None:
                                raise optuna.exceptions.TrialPruned(msg)
                            raise RuntimeError(msg)

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            opt.param_groups[0]["params"], max_norm=1.0
                        )
                        opt.step()
                        did_any_update = True

                if not did_any_update:
                    msg = f"[{self.model_name}] Embedding refinement performed zero updates (all batches empty after masking)."
                    self.logger.error(msg)
                    if trial is not None:
                        raise optuna.exceptions.TrialPruned(msg)
                    raise RuntimeError(msg)

        finally:
            for p, req in saved:
                p.requires_grad_(req)

    def _prepare_eval_arrays(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        labels_for_scoring: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate/clean eval arrays and build one-hot y_true.

        This is the shared guardrail that prevents downstream metric crashes
        (e.g., average_precision_score) due to bad shapes, NaNs, rows that don't
        sum properly, or label-range issues.

        Args:
            y_true (np.ndarray): 1D int labels (N,).
            y_pred (np.ndarray): 1D int labels (N,).
            y_proba (np.ndarray): 2D float probs (N, K).
            labels_for_scoring (list[int]): The expected label set (e.g., [0,1] or [0,1,2]).

        Returns:
            tuple: (y_true_clean, y_pred_clean, y_true_ohe_clean, y_proba_clean)

        Raises:
            ValueError: If shapes are incompatible or labels are out of range.
            RuntimeError: If no valid rows remain after filtering.
        """
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        y_proba = np.asarray(y_proba)

        if y_proba.ndim != 2:
            raise ValueError(f"y_proba must be 2D (N,K); got shape={y_proba.shape}.")

        if y_true.shape[0] != y_pred.shape[0] or y_true.shape[0] != y_proba.shape[0]:
            raise ValueError(
                f"Eval alignment mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}, y_proba={y_proba.shape}."
            )

        K = int(len(labels_for_scoring))
        if y_proba.shape[1] != K:
            raise ValueError(f"Expected y_proba.shape[1]=={K}, got {y_proba.shape[1]}.")

        # Filter invalid labels (-1 etc.) first
        valid = (y_true >= 0) & (y_true < K)
        if not bool(valid.any()):
            raise RuntimeError("No valid evaluation labels remain after filtering.")

        y_true = y_true[valid].astype(np.int64, copy=False)
        y_pred = y_pred[valid].astype(np.int64, copy=False)
        y_proba = y_proba[valid]

        # Numeric cleanup for proba
        y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=0.0, neginf=0.0)
        y_proba = np.clip(y_proba, 0.0, 1.0)

        row_sums = y_proba.sum(axis=1, keepdims=True)
        good_rows = row_sums[:, 0] > 0.0
        if not bool(good_rows.any()):
            raise RuntimeError(
                "No valid probability rows remain (all row sums are zero)."
            )

        y_true = y_true[good_rows]
        y_pred = y_pred[good_rows]
        y_proba = y_proba[good_rows]
        row_sums = row_sums[good_rows]
        y_proba = y_proba / row_sums

        # One-hot ground truth (guaranteed valid by construction)
        y_true_ohe = np.eye(K, dtype=np.int8)[y_true]

        return (
            y_true.astype(np.int8, copy=False),
            y_pred.astype(np.int8, copy=False),
            y_true_ohe,
            y_proba,
        )

    def _evaluate_model(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        eval_mask: np.ndarray,
        indices: np.ndarray,
        gamma: float,
        project_embedding: bool = False,
        objective_mode: bool = False,
        trial: Optional[optuna.Trial] = None,
        class_weights: Optional[torch.Tensor] = None,
        *,
        persist_projection: bool = False,
    ):
        """Evaluate model on a split, optionally projecting embeddings for the provided indices.

        Preserves your semantics, but adds robust edge handling for:
        - empty eval masks / no valid evaluation rows
        - invalid/non-finite probabilities
        - consistent prune behavior for Optuna objective_mode

        Args:
            model (nn.Module): Trained UBP model.
            X (np.ndarray): Corrupted input matrix (N_split, L) or full.
            y (np.ndarray): Clean target matrix aligned to X (N_split, L) or full.
            eval_mask (np.ndarray): Boolean mask of entries to evaluate (simulated-missing & not orig-missing).
            indices (np.ndarray): Sample indices for this split (rows).
            gamma (float): Gamma used for projection refinement objective.
            project_embedding (bool): Whether to refine embeddings before evaluation.
            objective_mode (bool): If True, suppresses verbose outputs and supports pruning.
            trial (Optional[optuna.Trial]): Trial handle when objective_mode=True.
            class_weights (Optional[torch.Tensor]): Optional class weights for projection refinement.
            persist_projection (bool): If False, revert any embedding projection applied during evaluation.

        Returns:
            dict[str, float]: Metrics dict.

        Raises:
            TypeError: If objective_mode=True but trial is None.
            RuntimeError: If evaluation cannot be computed.
            optuna.exceptions.TrialPruned: For objective_mode trial failures.
        """
        if objective_mode and trial is None:
            msg = "objective_mode=True requires a valid Optuna trial for pruning."
            self.logger.error(msg)
            raise TypeError(msg)

        if eval_mask is None or not bool(np.asarray(eval_mask).any()):
            msg = f"[{self.model_name}] Evaluation mask is empty; no entries to score."
            if trial is not None:
                raise optuna.exceptions.TrialPruned(msg)
            self.logger.warning(msg)
            raise RuntimeError(msg)

        saved_rows = None
        idx_t = None

        if project_embedding:
            if not persist_projection:
                idx_np = np.asarray(indices, dtype=np.int64).reshape(-1)
                idx_t = (
                    torch.from_numpy(idx_np).to(self.device, non_blocking=True).long()
                )
                with torch.no_grad():
                    saved_rows = model.embedding.weight.index_select(0, idx_t).detach().clone()  # type: ignore[attr-defined]

            self._refine_all_embeddings(
                model,
                X,
                gamma,
                indices=indices,
                lr=self.projection_lr,
                class_weights=class_weights,
                iterations=self.projection_epochs,
                trial=trial,
            )

        try:
            pred_labels, pred_probas = self._predict(model, indices, return_proba=True)
            if pred_probas is None:
                msg = "Prediction probabilities are required for evaluation."
                self.logger.error(msg)
                raise RuntimeError(msg)

            # Pull evaluated entries
            y_true = np.asarray(y)[eval_mask].astype(np.int64, copy=False)
            y_pred = np.asarray(pred_labels)[eval_mask].astype(np.int64, copy=False)
            y_proba = np.asarray(pred_probas)[eval_mask]

            # Haploid probability folding if needed (keep your behavior)
            if self.is_haploid_ and y_proba.shape[1] == 3:
                p2 = np.zeros((len(y_proba), 2), dtype=y_proba.dtype)
                p2[:, 0], p2[:, 1] = y_proba[:, 0], y_proba[:, 1] + y_proba[:, 2]
                y_proba = p2

            # Harmonize haploid labels
            if self.is_haploid_:
                y_true = (y_true > 0).astype(np.int64, copy=False)
                y_pred = (y_pred > 0).astype(np.int64, copy=False)
                labels_for_scoring = [0, 1]
                target_names = ["REF", "ALT"]
            else:
                labels_for_scoring = [0, 1, 2]
                target_names = ["REF", "HET", "ALT"]

            # Shared validation/cleaning + one-hot construction
            try:
                y_true_flat, y_pred_flat, y_true_ohe, y_proba_flat = (
                    self._prepare_eval_arrays(
                        y_true=y_true,
                        y_pred=y_pred,
                        y_proba=y_proba,
                        labels_for_scoring=labels_for_scoring,
                    )
                )
            except Exception as e:
                msg = f"[{self.model_name}] Evaluation arrays invalid: {str(e)}"
                if trial is not None:
                    raise optuna.exceptions.TrialPruned(msg) from e
                self.logger.error(msg)
                raise RuntimeError(msg) from e

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
                y_true_flat,
                y_pred_flat,
                y_true_ohe,
                y_proba_flat,
                objective_mode,
                tune_metric=tm,
            )

            if not objective_mode:
                if self.verbose or self.debug:
                    pm = PrettyMetrics(
                        metrics,
                        precision=2,
                        title=f"{self.model_name} Validation Metrics",
                    )
                    pm.render()

                self._make_class_reports(
                    y_true=y_true_flat,
                    y_pred_proba=y_proba_flat,
                    y_pred=y_pred_flat,
                    metrics=metrics,
                    labels=target_names,
                )

                # --- IUPAC decode and 10-base integer report ---
                y_true_matrix = np.array(y, copy=True)
                y_pred_matrix = np.array(pred_labels, copy=True)

                if self.is_haploid_:
                    y_true_matrix = np.where(y_true_matrix > 0, 2, y_true_matrix)
                    y_pred_matrix = np.where(y_pred_matrix > 0, 2, y_pred_matrix)

                y_true_dec = self.decode_012(y_true_matrix)
                y_pred_dec = self.decode_012(y_pred_matrix)

                # Diploid mapping dict (kept for ploidy=2 path)
                encodings_dict_diploid = {
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

                # Haploid mapping dict
                encodings_dict_haploid = {"A": 0, "C": 1, "G": 2, "T": 3, "N": -1}

                y_true_int = self._convert_int_iupac_ploidy(
                    y_true_dec,
                    ploidy=self.ploidy,
                    encodings_dict=(
                        encodings_dict_haploid
                        if self.is_haploid_
                        else encodings_dict_diploid
                    ),
                    ref=getattr(self.genotype_data, "ref", None),
                    alt=getattr(self.genotype_data, "alt", None),
                    ambiguity_mode="ref_alt",  # or "first_base" if you don’t trust ref/alt
                )
                y_pred_int = self._convert_int_iupac_ploidy(
                    y_pred_dec,
                    ploidy=self.ploidy,
                    encodings_dict=(
                        encodings_dict_haploid
                        if self.is_haploid_
                        else encodings_dict_diploid
                    ),
                    ref=getattr(self.genotype_data, "ref", None),
                    alt=getattr(self.genotype_data, "alt", None),
                    ambiguity_mode="ref_alt",
                )

                y_true_eval = y_true_int[eval_mask]
                y_pred_eval = y_pred_int[eval_mask]
                n_iupac_classes = 4 if self.num_classes_ == 2 else 10
                valid_iupac_mask = (
                    (y_true_eval >= 0)
                    & (y_true_eval < n_iupac_classes)
                    & (y_pred_eval >= 0)
                    & (y_pred_eval < n_iupac_classes)
                )
                if bool(valid_iupac_mask.any()) and self.num_classes_ > 2:
                    self._make_class_reports(
                        y_true=y_true_eval[valid_iupac_mask],
                        y_pred=y_pred_eval[valid_iupac_mask],
                        metrics=metrics,
                        y_pred_proba=None,
                        labels=["A", "C", "G", "T", "W", "R", "M", "K", "Y", "S"],
                    )
                elif bool(valid_iupac_mask.any()) and self.num_classes_ == 2:
                    self._make_class_reports(
                        y_true=y_true_eval[valid_iupac_mask],
                        y_pred=y_pred_eval[valid_iupac_mask],
                        metrics=metrics,
                        y_pred_proba=None,
                        labels=["A", "C", "G", "T"],
                    )
                else:
                    self.logger.warning(
                        "Skipped IUPAC confusion matrix: No valid ground truths."
                    )

            return metrics

        finally:
            if (
                project_embedding
                and (not persist_projection)
                and (saved_rows is not None)
                and (idx_t is not None)
            ):
                with torch.no_grad():
                    model.embedding.weight.index_copy_(0, idx_t, saved_rows)  # type: ignore[attr-defined]

    def _predict(
        self,
        model: nn.Module,
        indices: np.ndarray | torch.Tensor | list[int],
        return_proba: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Predict labels/probabilities for given sample indices.

        Args:
            model (nn.Module): The trained UBP model.
            indices (np.ndarray | torch.Tensor | list[int]): Sample indices to predict.
            return_proba (bool): If True, also return class probabilities.

        Returns:
            Tuple[np.ndarray, np.ndarray | None]: Predicted labels and optional probabilities.
        """
        if model is None:
            msg = f"[{self.model_name}] Model is not fitted. Must call 'fit()' before calling '_predict()'."
            self.logger.error(msg)
            raise NotFittedError(msg)

        if isinstance(indices, np.ndarray):
            idx = torch.from_numpy(indices.astype(np.int64, copy=False)).long()
        elif torch.is_tensor(indices):
            idx = indices.long()
        else:
            idx = torch.tensor(indices, dtype=torch.long)

        if idx.dim() != 1:
            idx = idx.view(-1)

        if idx.numel() == 0:
            empty_labels = np.empty((0, int(self.num_features_)), dtype=np.int64)
            empty_proba = (
                np.empty(
                    (0, int(self.num_features_), int(self.num_classes_)),
                    dtype=np.float32,
                )
                if return_proba
                else None
            )
            return empty_labels, empty_proba

        n_samples = int(getattr(self, "total_samples_", 0))
        if n_samples <= 0:
            msg = f"[{self.model_name}] total_samples_ must be > 0; got {n_samples}."
            self.logger.error(msg)
            raise RuntimeError(msg)

        bad = (idx < 0) | (idx >= n_samples)
        if bool(bad.any()):
            msg = (
                f"[{self.model_name}] indices out of range for embedding table size "
                f"{n_samples}: min={int(idx.min().item())}, max={int(idx.max().item())}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        idx = idx.to(self.device, non_blocking=True)

        model.eval()
        with torch.no_grad():
            logits = model(idx)  # (B, L, K)
            if (
                logits.dim() != 3
                or logits.shape[1] != self.num_features_
                or logits.shape[2] != self.num_classes_
            ):
                raise ValueError(
                    f"Model logits shape mismatch: expected (B,{self.num_features_},{self.num_classes_}), got {tuple(logits.shape)}."
                )
            probas = torch.softmax(logits, dim=-1)
            labels = torch.argmax(probas, dim=-1)

        if return_proba:
            return labels.detach().cpu().numpy(), probas.detach().cpu().numpy()
        return labels.detach().cpu().numpy(), None

    def _objective(self, trial: optuna.Trial) -> float | tuple[float, ...]:
        """Objective function for hyperparameter tuning.

        This method defines the objective function used during hyperparameter tuning with Optuna. It samples hyperparameters, builds the UBP model, trains it, and evaluates its performance on the validation set. The metric(s) to optimize are returned.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float | tuple[float, ...]: Metric(s) to optimize. Supports multi-objective.
        """
        params = self._sample_hyperparameters(trial)

        # Must re-calculate PCA for the specific latent_dim of this trial
        # (Or slice a larger pre-computed PCA, but re-calc is safer/easier)
        trial_latent_dim = int(params["latent_dim"])

        # Optimization: set embedding_init to PCA of ground truth
        params["model_params"]["embedding_init"] = self._get_pca_embedding_init(
            self.ground_truth_, self.train_idx_, trial_latent_dim
        )

        model = self.build_model(self.Model, params["model_params"])

        lr: float = params["learning_rate"]
        l1_penalty: float = params["l1_penalty"]

        train_loss_mask = self.X_train_corrupted_ >= 0

        class_weights_ = self._class_weights_from_zygosity(
            self.y_train_,
            train_mask=train_loss_mask,  # NOTE: # not sim_mask
            inverse=bool(params["inverse"]),
            normalize=bool(params["normalize"]),
            max_ratio=self.max_ratio,
            power=float(params["power"]),
        )

        # --- Sanitize Haploid/Invalid Weights ---
        if class_weights_ is not None:
            # 1. Truncate dimension if needed
            # (Haploid returns 3 weights for 2 classes)
            if self.is_haploid_ and class_weights_.numel() > self.num_classes_:
                self.logger.warning(
                    f"Haploid mode: Truncating class weights from {class_weights_.shape} to {self.num_classes_}."
                )
                class_weights_ = class_weights_[: self.num_classes_]

            # 2. Check for NaN/Inf (caused by 0 counts in inverse freq)
            if not torch.isfinite(class_weights_).all():
                self.logger.warning(
                    f"Class weights contain NaN/Inf ({class_weights_}). "
                    "This usually happens with rare variants in small splits. Resetting to uniform weights."
                )
                class_weights_ = torch.ones(self.num_classes_, device=self.device)

        res = self._execute_ubp_training(
            model=model,
            lr=lr,
            l1_penalty=l1_penalty,
            params=params,
            trial=trial,
            class_weights=class_weights_,
            gamma_schedule=params["gamma_schedule"],
        )

        # NOTE: project using corrupted data
        metrics = self._evaluate_model(
            model,
            self.X_val_corrupted_,
            self.y_val_,
            self.eval_mask_val_,
            indices=self.val_idx_,
            gamma=float(params["gamma"]),
            project_embedding=True,
            objective_mode=True,
            trial=trial,
            class_weights=class_weights_,
            persist_projection=False,
        )

        if isinstance(self.tune_metric, (list, tuple)):
            # Multi-objective tuning with Optuna
            return tuple(metrics[m] for m in self.tune_metric)
        return metrics[self.tune_metric]

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample hyperparameters for tuning.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            dict[str, Any]: Sampled hyperparameters.
        """
        nC = int(self.num_classes_)
        input_dim = int(self.num_features_ * nC)

        lower_bound = 2
        # 1. Enforce strict bottleneck: max size is features - 1
        # 2. Enforce hard cap: max size is 32
        # 3. Safety net: ensure upper_bound is never lower than lower_bound
        upper_bound = max(lower_bound, min(32, self.num_features_ - 1))

        params = {
            "latent_dim": trial.suggest_int("latent_dim", lower_bound, upper_bound),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.05),
            "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 8),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "elu", "leaky_relu", "selu"]
            ),
            "l1_penalty": trial.suggest_float("l1_penalty", 0.0, 1e-3, log=False),
            "layer_scaling_factor": trial.suggest_float(
                "layer_scaling_factor", 2.0, 10.0
            ),
            "layer_schedule": trial.suggest_categorical(
                "layer_schedule", ["pyramid", "linear"]
            ),
            # Loss controls
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "gamma_schedule": trial.suggest_categorical(
                "gamma_schedule", [True, False]
            ),
            "power": trial.suggest_float("power", 0.1, 2.0),
            "normalize": trial.suggest_categorical("normalize", [True, False]),
            "inverse": trial.suggest_categorical("inverse", [True, False]),
        }

        OBJECTIVE_SPEC_UBP.validate(params)

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

        params["model_params"] = {
            "num_embeddings": self.total_samples_,
            "n_features": self.num_features_,
            "prefix": self.prefix,
            "num_classes": nC,
            "latent_dim": int(params["latent_dim"]),
            "hidden_layer_sizes": hidden_layer_sizes,
            "dropout_rate": float(params["dropout_rate"]),
            "activation": str(params["activation"]),
            "prefix": self.prefix,
            "device": self.device,
            "verbose": self.verbose,
            "debug": self.debug,
        }
        return params

    def _set_best_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Update instance fields from tuned params and return model_params dict.

        Args:
            params (dict): Best hyperparameters from tuning.

        Returns:
            dict: Model parameters for building the final model.
        """
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

        train_loss_mask = self.X_train_corrupted_ >= 0
        self.class_weights_ = self._class_weights_from_zygosity(
            self.y_train_,
            train_mask=train_loss_mask,
            inverse=self.inverse,
            normalize=self.normalize,
            max_ratio=self.max_ratio,
            power=self.power,
        )

        # --- Sanitize Haploid/Invalid Weights ---
        if self.class_weights_ is not None:
            # 1. Truncate dimension if needed
            # (Haploid returns 3 weights for 2 classes)
            if self.is_haploid_ and self.class_weights_.numel() > self.num_classes_:
                self.logger.warning(
                    f"Haploid mode: Truncating class weights from {self.class_weights_.shape} to {self.num_classes_}."
                )
                self.class_weights_ = self.class_weights_[: self.num_classes_]

            # 2. Check for NaN/Inf (caused by 0 counts in inverse freq)
            if not torch.isfinite(self.class_weights_).all():
                self.logger.warning(
                    f"Class weights contain NaN/Inf ({self.class_weights_}). "
                    "This usually happens with rare variants in small splits. Resetting to uniform weights."
                )
                self.class_weights_ = torch.ones(self.num_classes_, device=self.device)

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

    def _compute_hidden_layer_sizes(
        self,
        n_inputs: int,
        n_outputs: int,
        n_samples: int,
        n_hidden: int,
        latent_dim: int,
        *,
        alpha: float = 4.0,
        schedule: str = "pyramid",
        min_size: int = 16,
        max_size: int | None = None,
        multiple_of: int = 8,
        decay: float | None = None,
        cap_by_inputs: bool = True,
    ) -> list[int]:
        """Compute hidden layer sizes given problem scale and a layer count.

        Args:
            n_inputs (int): Number of input features (e.g., flattened one-hot: num_features * num_classes).
            n_outputs (int): Number of output classes (often equals num_classes).
            n_samples (int): Number of training samples.
            n_hidden (int): Number of hidden layers (excluding input and latent layers).
            latent_dim (int): Latent dimensionality (not returned, used only to set a floor).
            alpha (float): Scaling factor for base layer size.
            schedule (str): Size schedule ("pyramid" or "linear").
            min_size (int): Minimum layer size floor before latent-aware adjustment.
            max_size (int | None): Maximum layer size cap. If None, a heuristic cap is used.
            multiple_of (int): Hidden sizes are multiples of this value.
            decay (float | None): Pyramid decay factor. If None, computed to land near the target.
            cap_by_inputs (bool): If True, cap layer sizes to n_inputs.

        Returns:
            list[int]: Hidden layer sizes (len = n_hidden).

        Notes:
            - Returns sizes for *hidden layers only* (length = n_hidden).
            - Does NOT include the input layer (n_inputs) or the latent layer (latent_dim).
            - Enforces a latent-aware minimum: one discrete level above latent_dim, where a level is `multiple_of`.
            - Enforces *strictly decreasing* hidden sizes (no repeats). This may require bumping `base` upward.

        Args:
            n_inputs: Number of input features (e.g., flattened one-hot: num_features * num_classes).
            n_outputs: Number of output classes (often equals num_classes).
            n_samples: Number of training samples.
            n_hidden: Number of hidden layers (excluding input and latent layers).
            latent_dim: Latent dimensionality (not returned, used only to set a floor).
            alpha: Scaling factor for base layer size.
            schedule: Size schedule ("pyramid" or "linear").
            min_size: Minimum layer size floor before latent-aware adjustment.
            max_size: Maximum layer size cap. If None, a heuristic cap is used.
            multiple_of: Hidden sizes are multiples of this value.
            decay: Pyramid decay factor. If None, computed to land near the target.
            cap_by_inputs: If True, cap layer sizes to n_inputs.

        Returns:
            list[int]: Hidden layer sizes (len = n_hidden).

        Raises:
            ValueError: On invalid arguments or conflicting constraints.
        """
        # Basic validation
        if n_hidden < 0:
            msg = f"n_hidden must be >= 0, got {n_hidden}."
            self.logger.error(msg)
            raise ValueError(msg)

        if n_hidden == 0:
            return []

        if n_inputs <= 0:
            msg = f"n_inputs must be > 0, got {n_inputs}."
            self.logger.error(msg)
            raise ValueError(msg)

        if n_outputs <= 0:
            msg = f"n_outputs must be > 0, got {n_outputs}."
            self.logger.error(msg)
            raise ValueError(msg)

        if n_samples <= 0:
            msg = f"n_samples must be > 0, got {n_samples}."
            self.logger.error(msg)
            raise ValueError(msg)

        if latent_dim <= 0:
            msg = f"latent_dim must be > 0, got {latent_dim}."
            self.logger.error(msg)
            raise ValueError(msg)

        if multiple_of <= 0:
            msg = f"multiple_of must be > 0, got {multiple_of}."
            self.logger.error(msg)
            raise ValueError(msg)

        if alpha <= 0:
            msg = f"alpha must be > 0, got {alpha}."
            self.logger.error(msg)
            raise ValueError(msg)

        schedule = str(schedule).lower().strip()
        if schedule not in {"pyramid", "linear"}:
            msg = f"Invalid schedule '{schedule}'. Must be 'pyramid' or 'linear'."
            self.logger.error(msg)
            raise ValueError(msg)

        # Latent-aware minimum floor
        # Smallest multiple_of strictly greater than latent_dim
        min_hidden_floor = int(np.ceil((latent_dim + 1) / multiple_of) * multiple_of)
        effective_min = max(int(min_size), min_hidden_floor)

        if cap_by_inputs and n_inputs < effective_min:
            msg = (
                "Cannot satisfy latent-aware minimum hidden size with cap_by_inputs=True. "
                f"Required hidden size >= {effective_min} (one level above latent_dim={latent_dim}), "
                f"but n_inputs={n_inputs}. Set cap_by_inputs=False or reduce latent_dim/multiple_of."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Infer num_features
        # (if using flattened one-hot: n_inputs = num_features * num_classes)
        num_features = n_inputs // n_outputs

        # Base size heuristic
        # (feature-matrix aware; avoids collapse for huge n_inputs)
        obs_scale = (float(n_samples) * float(num_features)) / float(
            num_features + n_outputs
        )
        base = int(np.ceil(float(alpha) * np.sqrt(obs_scale)))

        # Determine max_size
        if max_size is None:
            max_size = max(int(n_inputs), int(base), int(effective_min))

        if cap_by_inputs:
            max_size = min(int(max_size), int(n_inputs))
        else:
            max_size = int(max_size)

        if max_size < effective_min:
            msg = (
                f"max_size ({max_size}) must be >= effective_min ({effective_min}), where effective_min "
                f"is max(min_size={min_size}, one-level-above latent_dim={latent_dim})."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Round base up to a multiple and clip to bounds
        base = int(np.clip(base, effective_min, max_size))
        base = int(np.ceil(base / multiple_of) * multiple_of)
        base = int(np.clip(base, effective_min, max_size))

        # Enforce "no repeats" feasibility in discrete levels
        # Need n_hidden distinct multiples between base and effective_min:
        # base >= effective_min + (n_hidden - 1) * multiple_of
        required_min_base = effective_min + (n_hidden - 1) * multiple_of

        if required_min_base > max_size:
            msg = (
                "Cannot build strictly-decreasing (no-repeat) hidden sizes under current constraints. "
                f"Need base >= {required_min_base} to fit n_hidden={n_hidden} distinct layers "
                f"with multiple_of={multiple_of} down to effective_min={effective_min}, "
                f"but max_size={max_size}. Reduce n_hidden, reduce multiple_of, lower latent_dim/min_size, "
                "or increase max_size / set cap_by_inputs=False."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        if base < required_min_base:
            # Bump base upward so a strict staircase is possible
            base = required_min_base
            base = int(np.ceil(base / multiple_of) * multiple_of)
            base = int(np.clip(base, effective_min, max_size))

        # Work in "levels" of multiple_of for guaranteed uniqueness
        start_level = base // multiple_of
        end_level = effective_min // multiple_of

        # Sanity: distinct levels available
        if (start_level - end_level) < (n_hidden - 1):
            # This should not happen due to required_min_base logic, but keep a hard guard.
            msg = (
                "Internal constraint failure: insufficient discrete levels to enforce no repeats. "
                f"start_level={start_level}, end_level={end_level}, n_hidden={n_hidden}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Build schedule in level space (integers), then convert to sizes
        if n_hidden == 1:
            levels = np.array([start_level], dtype=int)

        elif schedule == "linear":
            # Linear interpolation in level space, then strictify
            levels = np.round(np.linspace(start_level, end_level, num=n_hidden)).astype(
                int
            )

            # Enforce bounds then strict decrease
            levels = np.clip(levels, end_level, start_level)

            for i in range(1, n_hidden):
                if levels[i] >= levels[i - 1]:
                    levels[i] = levels[i - 1] - 1

            if levels[-1] < end_level:
                msg = (
                    "Failed to enforce strictly-decreasing linear schedule without violating the floor. "
                    f"(levels[-1]={levels[-1]} < end_level={end_level}). "
                    "Reduce n_hidden or multiple_of, or increase max_size."
                )
                self.logger.error(msg)
                raise ValueError(msg)

            # Force exact floor at the end
            # (still strict because we have enough room by construction)
            levels[-1] = end_level
            for i in range(n_hidden - 2, -1, -1):
                if levels[i] <= levels[i + 1]:
                    levels[i] = levels[i + 1] + 1

            if levels[0] > start_level:
                # If this happens,
                # we would need an even larger base;
                # handle by raising base once.
                needed_base = int(levels[0] * multiple_of)
                if needed_base > max_size:
                    msg = (
                        "Cannot enforce strictly-decreasing linear schedule after floor anchoring; "
                        f"would require base={needed_base} > max_size={max_size}."
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)
                # Rebuild with bumped base
                start_level = needed_base // multiple_of
                levels = np.arange(start_level, start_level - n_hidden, -1, dtype=int)
                levels[-1] = end_level  # keep floor
                # Ensure strict with backward adjust
                for i in range(n_hidden - 2, -1, -1):
                    if levels[i] <= levels[i + 1]:
                        levels[i] = levels[i + 1] + 1

        elif schedule == "pyramid":
            # Geometric decay in level space
            # (more aggressive early taper than linear)
            if decay is not None:
                dcy = float(decay)
            else:
                # Choose decay to land exactly at end_level (in float space)
                dcy = (float(end_level) / float(start_level)) ** (
                    1.0 / float(n_hidden - 1)
                )

            # Keep it in a sensible range
            dcy = float(np.clip(dcy, 0.05, 0.99))

            exponents = np.arange(n_hidden, dtype=float)
            levels_float = float(start_level) * (dcy**exponents)

            levels = np.round(levels_float).astype(int)
            levels = np.clip(levels, end_level, start_level)

            # Anchor the last layer at the floor, then strictify backward
            levels[-1] = end_level
            for i in range(n_hidden - 2, -1, -1):
                if levels[i] <= levels[i + 1]:
                    levels[i] = levels[i + 1] + 1

            # If we overshot the start, bump base (once) if possible,
            # then rebuild
            if levels[0] > start_level:
                needed_base = int(levels[0] * multiple_of)
                if needed_base > max_size:
                    msg = (
                        "Cannot enforce strictly-decreasing pyramid schedule; "
                        f"would require base={needed_base} > max_size={max_size}."
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)

                start_level = needed_base // multiple_of
                # Recompute with new start_level and same decay (or recompute decay if decay is None)
                if decay is None:
                    dcy = (float(end_level) / float(start_level)) ** (
                        1.0 / float(n_hidden - 1)
                    )
                    dcy = float(np.clip(dcy, 0.05, 0.99))

                levels_float = float(start_level) * (dcy**exponents)
                levels = np.round(levels_float).astype(int)
                levels = np.clip(levels, end_level, start_level)
                levels[-1] = end_level
                for i in range(n_hidden - 2, -1, -1):
                    if levels[i] <= levels[i + 1]:
                        levels[i] = levels[i + 1] + 1

        else:
            msg = f"Unknown schedule '{schedule}'. Use 'pyramid' or 'linear' (constant disallowed with no repeats)."
            self.logger.error(msg)
            raise ValueError(msg)

        # Convert levels -> sizes
        sizes = (levels * multiple_of).astype(int)

        # Final clip (should be redundant, but safe)
        sizes = np.clip(sizes, effective_min, max_size).astype(int)

        # Final strict no-repeat assertion
        if np.any(np.diff(sizes) >= 0):
            msg = f"Internal error: produced non-decreasing or repeated hidden sizes after strict enforcement. sizes={sizes.tolist()}"
            self.logger.error(msg)
            raise ValueError(msg)

        return sizes.tolist()

    def _get_pca_embedding_init(
        self,
        X_full: np.ndarray,
        train_idx: np.ndarray,
        latent_dim: int,
    ) -> torch.Tensor:
        """Compute PCA-based embedding init for all samples, fitted on training rows only.

        This method is intentionally defensive: if PCA cannot be fit due to small sample sizes, degenerate data, or invalid dimensionality constraints, it falls back to a deterministic random (small-noise) initialization so UBP can still run.

        Args:
            X_full (np.ndarray): Full 012 matrix shape (N, L) with missing encoded as -1.
            train_idx (np.ndarray): Indices of training samples (subset of [0..N-1]).
            latent_dim (int): Requested PCA components (embedding dimension).

        Returns:
            torch.Tensor: Tensor of shape (N, latent_dim) on self.device.

        Raises:
            ValueError: If X_full is not 2D, empty, or latent_dim is invalid (< 1).
        """
        X = np.asarray(X_full, dtype=np.float32)
        if X.ndim != 2:
            msg = f"[{self.model_name}] X_full must be 2D (N,L). Got shape={getattr(X, 'shape', None)}."
            self.logger.error(msg)
            raise ValueError(msg)

        N, L = X.shape
        if N <= 0 or L <= 0:
            msg = f"[{self.model_name}] X_full must be non-empty. Got N={N}, L={L}."
            self.logger.error(msg)
            raise ValueError(msg)

        try:
            latent_dim_req = int(latent_dim)
        except Exception as e:
            msg = f"[{self.model_name}] latent_dim must be an int-like value; got {latent_dim!r}."
            self.logger.error(msg)
            raise ValueError(msg) from e

        if latent_dim_req < 1:
            msg = f"[{self.model_name}] latent_dim must be >= 1; got {latent_dim_req}."
            self.logger.error(msg)
            raise ValueError(msg)

        # Sanitize train indices
        idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            self.logger.warning(
                f"[{self.model_name}] train_idx is empty; falling back to random embedding init."
            )
            rng = np.random.default_rng(self.seed)
            V_rand = rng.normal(loc=0.0, scale=1e-2, size=(N, latent_dim_req)).astype(
                np.float32, copy=False
            )
            return torch.as_tensor(V_rand, dtype=torch.float32, device=self.device)

        # Keep only valid, unique indices (preserve order)
        mask_valid = (idx >= 0) & (idx < N)
        if not np.all(mask_valid):
            n_bad = int((~mask_valid).sum())
            self.logger.warning(
                f"[{self.model_name}] Dropping {n_bad} out-of-bounds train indices for PCA init."
            )
            idx = idx[mask_valid]

        if idx.size == 0:
            self.logger.warning(
                f"[{self.model_name}] No valid train indices remain; falling back to random embedding init."
            )
            rng = np.random.default_rng(self.seed)
            V_rand = rng.normal(loc=0.0, scale=1e-2, size=(N, latent_dim_req)).astype(
                np.float32, copy=False
            )
            return torch.as_tensor(V_rand, dtype=torch.float32, device=self.device)

        # Unique while preserving order
        _, first_pos = np.unique(idx, return_index=True)
        idx = idx[np.sort(first_pos)]

        # PCA requires at least 2 training samples in sklearn (practically).
        n_train = int(idx.size)
        if n_train < 2:
            self.logger.warning(
                f"[{self.model_name}] PCA init requires >=2 training samples; got n_train={n_train}. "
                "Falling back to random embedding init."
            )
            rng = np.random.default_rng(self.seed)
            V_rand = rng.normal(loc=0.0, scale=1e-2, size=(N, latent_dim_req)).astype(
                np.float32, copy=False
            )
            return torch.as_tensor(V_rand, dtype=torch.float32, device=self.device)

        # Compute training column means ignoring -1 (missing sentinel)
        X_train = X[idx]
        col_means = self._compute_col_means_ignore_missing(
            X_train, missing_value=-1.0, fill_value=0.0
        )

        # Fill missing in full matrix using training means
        missing_full = X == -1.0
        if missing_full.any():
            rows, cols = np.where(missing_full)
            X_filled = X.copy()
            X_filled[rows, cols] = col_means[cols]
        else:
            X_filled = X

        if self.debug:
            n_missing = int(missing_full.sum())
            if n_missing > 0:
                missing_pct = (100.0 * n_missing) / float(X.size)
                self.logger.debug(
                    f"[{self.model_name}] PCA init filled {n_missing} missing values ({missing_pct:.2f}%)."
                )

        # Effective PCA dimensionality: must satisfy n_components <= min(n_train, n_features)
        max_components = int(min(n_train, L))
        eff_dim = int(min(latent_dim_req, max_components))

        if eff_dim < latent_dim_req:
            self.logger.warning(
                f"[{self.model_name}] Requested latent_dim={latent_dim_req} exceeds PCA limit "
                f"min(n_train={n_train}, n_features={L})={max_components}. Using eff_dim={eff_dim} "
                "and padding remaining dimensions."
            )

        # If eff_dim collapses (shouldn't with n_train>=2 and L>=1), fallback.
        if eff_dim < 1:
            self.logger.warning(
                f"[{self.model_name}] PCA eff_dim computed as {eff_dim}; falling back to random embedding init."
            )
            rng = np.random.default_rng(self.seed)
            V_rand = rng.normal(loc=0.0, scale=1e-2, size=(N, latent_dim_req)).astype(
                np.float32, copy=False
            )
            return torch.as_tensor(V_rand, dtype=torch.float32, device=self.device)

        # Fit PCA on training rows; robust fallback on failure.
        try:
            pca = PCA(n_components=eff_dim, random_state=self.seed)
            pca.fit(X_filled[idx])
            V_eff = pca.transform(X_filled).astype(
                np.float32, copy=False
            )  # (N, eff_dim)

            if self.debug and hasattr(pca, "explained_variance_ratio_"):
                evr = np.asarray(pca.explained_variance_ratio_, dtype=float)
                head = evr[: min(5, evr.size)].tolist()
                self.logger.debug(
                    f"[{self.model_name}] PCA init EVR head={head}, total={float(evr.sum()):.4f}."
                )

        except Exception as e:
            self.logger.warning(
                f"[{self.model_name}] PCA init failed (n_train={n_train}, n_features={L}, eff_dim={eff_dim}): {e}. "
                "Falling back to random embedding init."
            )
            rng = np.random.default_rng(self.seed)
            V_rand = rng.normal(loc=0.0, scale=1e-2, size=(N, latent_dim_req)).astype(
                np.float32, copy=False
            )
            return torch.as_tensor(V_rand, dtype=torch.float32, device=self.device)

        # If we had to reduce eff_dim, pad to requested latent_dim_req deterministically.
        if eff_dim == latent_dim_req:
            V = V_eff
        else:
            rng = np.random.default_rng(self.seed)
            V = np.zeros((N, latent_dim_req), dtype=np.float32)
            V[:, :eff_dim] = V_eff
            # Small-noise padding helps avoid exact-zero dimensions (often harmless either way)
            pad = latent_dim_req - eff_dim
            if pad > 0:
                V[:, eff_dim:] = rng.normal(loc=0.0, scale=1e-3, size=(N, pad)).astype(
                    np.float32, copy=False
                )

        return torch.as_tensor(V, dtype=torch.float32, device=self.device)

    def _compute_col_means_ignore_missing(
        self,
        X_train: np.ndarray,
        *,
        missing_value: float = -1.0,
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """Compute per-column means ignoring a sentinel missing value.

        Args:
            X_train (np.ndarray): Training matrix of shape (n_train, n_features).
            missing_value (float): Sentinel value indicating missing entries.
            fill_value (float): Value to use when a column has zero observed entries.

        Returns:
            np.ndarray: Column means of shape (n_features,), with empty columns filled.
        """
        Xf = np.asarray(X_train, dtype=np.float32)
        valid = Xf != float(missing_value)

        counts = valid.sum(axis=0).astype(np.float32)
        sums = np.where(valid, Xf, 0.0).sum(axis=0, dtype=np.float32)

        means = np.divide(
            sums, counts, out=np.full_like(sums, fill_value), where=counts > 0
        )
        return means
