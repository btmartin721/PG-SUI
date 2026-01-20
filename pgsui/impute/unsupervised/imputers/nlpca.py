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
from pgsui.data_processing.containers import NLPCAConfig
from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.loss_functions import FocalCELoss
from pgsui.impute.unsupervised.models.nlpca_model import NLPCAModel
from pgsui.utils.logging_utils import configure_logger
from pgsui.utils.misc import OBJECTIVE_SPEC_NLPCA
from pgsui.utils.pretty_metrics import PrettyMetrics

if TYPE_CHECKING:
    from snpio import TreeParser
    from snpio.read_input.genotype_data import GenotypeData


def ensure_nlpca_config(config: NLPCAConfig | dict | str | None) -> NLPCAConfig:
    """Return a concrete config for NLPCA.

    NLPCA reuses NLPCAConfig (latent_dim, hidden sizes, loss controls, projection controls, tuning spec).

    Args:
        config: NLPCAConfig instance, dict, YAML path, or None.

    Returns:
        NLPCAConfig: Concrete configuration.
    """
    if config is None:
        return NLPCAConfig()
    if isinstance(config, NLPCAConfig):
        return config
    if isinstance(config, str):
        return load_yaml_to_dataclass(config, NLPCAConfig)
    if isinstance(config, dict):
        cfg_in = copy.deepcopy(config)
        base = NLPCAConfig()
        preset = cfg_in.pop("preset", None)
        if "io" in cfg_in and isinstance(cfg_in["io"], dict):
            preset = preset or cfg_in["io"].pop("preset", None)
        if preset:
            base = NLPCAConfig.from_preset(preset)

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
    raise TypeError("config must be an NLPCAConfig, dict, YAML path, or None.")


class ImputeNLPCA(BaseNNImputer):
    """Non-linear PCA (NLPCA) Imputer for Genotype Data.

    This is “UBP Phase 3 only” + explicit input refinement.

    Key differences vs ImputeUBP:
        - No Phase 2 (decoder-only refinement).
        - Joint optimization only (V and W updated together).
            - EM-like updates of *originally missing* inputs during training: after each epoch, replace originally-missing values in the working matrix with the model's current reconstructions. Simulated-missing values are NEVER filled during training (prevents leakage).

    Attributes:
        model_ (nn.Module): Trained decoder model with learnable embeddings.
        is_fit_ (bool): Whether fit() has been called successfully.
        X_train_work_ (np.ndarray): Working training matrix used as targets during NLPCA.
        _X_train_work_init_ (np.ndarray): Initial copy for per-trial reset during tuning.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        tree_parser: Optional["TreeParser"] = None,
        config: Optional[Union[NLPCAConfig, dict, str]] = None,
        overrides: Optional[dict] = None,
        sim_strategy: Optional[str] = None,
        sim_prop: Optional[float] = None,
        sim_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize the ImputeNLPCA model.

        Args:
            genotype_data (GenotypeData): Genotype data object.
            tree_parser (TreeParser): Tree parser for nonrandom missingness simulation.
            config (NLPCAConfig | dict | str | None): Configuration (NLPCAConfig or compatible dict/YAML path).
            overrides (dict | None): Dot-notation overrides for config.
            sim_strategy (str | None): Missingness simulation strategy.
            sim_prop (float | None): Proportion to simulate as missing.
            sim_kwargs (dict | None): Additional simulation kwargs.
        """
        self.model_name = "ImputeNLPCA"
        self.genotype_data = genotype_data
        self.tree_parser = tree_parser

        cfg = ensure_nlpca_config(config)
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

        self.Model = NLPCAModel
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

        # Training parameters
        self.batch_size = int(self.cfg.train.batch_size)
        self.learning_rate = float(self.cfg.train.learning_rate)
        self.l1_penalty = float(self.cfg.train.l1_penalty)
        self.gamma_threshold = 1e-4
        self.eta_min = 1e-6
        self.epochs = int(self.cfg.train.max_epochs)
        self.validation_split = float(self.cfg.train.validation_split)

        # Loss / weighting
        self.gamma = float(getattr(self.cfg.train, "gamma", 2.0))
        self.power = float(getattr(self.cfg.train, "weights_power", 1.0))
        self.normalize = bool(getattr(self.cfg.train, "weights_normalize", True))
        self.inverse = bool(getattr(self.cfg.train, "weights_inverse", False))
        self.max_ratio = getattr(self.cfg.train, "weights_max_ratio", None)
        self.gamma_schedule = bool(getattr(self.cfg.train, "gamma_schedule", False))
        self.gamma_init = float(getattr(self.cfg.train, "gamma_init", 0.0))

        # Projection controls (still used for eval/transform)
        self.projection_lr = float(self.cfg.nlpca.projection_lr)
        self.projection_epochs = int(self.cfg.nlpca.projection_epochs)

        # NLPCA input-refinement controls
        self.input_refine_every = int(getattr(self.cfg.nlpca, "input_refine_every", 1))
        self.input_refine_steps = int(getattr(self.cfg.nlpca, "input_refine_steps", 1))
        # Tuning
        self.tune = bool(self.cfg.tune.enabled)
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
        self.num_features_: int = 0
        self.model_params: dict[str, Any] = {}
        self.class_weights_: torch.Tensor | None = None

        self.num_tuned_params_ = OBJECTIVE_SPEC_NLPCA.count()

        # Working matrices (set during fit)
        self.X_train_work_: np.ndarray | None = None
        self.X_val_work_: np.ndarray | None = None
        self.X_test_work_: np.ndarray | None = None
        self._X_train_work_init_: np.ndarray | None = None

    def _haploidize_012(self, arr: np.ndarray) -> np.ndarray:
        """Convert diploid 012 to haploid 01 encoding, preserving missing (-1).

        Args:
            arr (np.ndarray): Input array with diploid 012 encoding.

        Returns:
            np.ndarray: Haploid 01 encoded array with missing as -1.
        """
        out = arr.astype(np.int8, copy=True)
        miss = out < 0
        out = np.where(out > 0, 1, out).astype(np.int8, copy=False)
        out[miss] = -1
        return out

    def _initialize_orig_missing_with_mode(
        self, X_corrupted: np.ndarray, orig_mask: np.ndarray
    ) -> np.ndarray:
        """Initialize originally-missing entries (orig_mask) with per-locus mode.

        Notes:
            This fills ONLY orig_mask entries. Simulated-missing entries remain -1.

        Args:
            X_corrupted (np.ndarray): Matrix (N, L) with missing as -1.
            orig_mask (np.ndarray): Boolean mask (N, L) for originally-missing entries only.

        Returns:
            np.ndarray: Copy of X_corrupted with orig-missing filled.
        """
        Xc = np.asarray(X_corrupted, dtype=np.int16)
        om = np.asarray(orig_mask, dtype=bool)

        out = Xc.copy()
        # Compute mode per column using observed entries in X_corrupted (>=0)
        obs = out >= 0
        if obs.any():
            L = out.shape[1]
            modes = np.zeros(L, dtype=np.int16)
            for j in range(L):
                col = out[:, j]
                v = col[col >= 0]
                if v.size == 0:
                    modes[j] = 0
                else:
                    # small-class (2 or 3) -> bincount is cheap
                    bc = np.bincount(
                        v.astype(np.int64), minlength=int(self.num_classes_)
                    )
                    modes[j] = int(np.argmax(bc))
            # Fill only orig-missing
            r, c = np.where(om)
            if r.size:
                out[r, c] = modes[c]
        else:
            # No observed at all; default fill = 0
            out[om] = 0

        return out.astype(np.int8, copy=False)

    def _update_orig_missing_from_model(
        self,
        model: nn.Module,
        X_work: np.ndarray,
        orig_mask: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        """Update originally-missing entries in X_work from current model predictions.

        Args:
            model: Trained/partially trained model.
            X_work: Working matrix to update in-place (N_split, L).
            orig_mask: Original-missing mask for the same split (N_split, L).
            indices: Global sample indices for rows of X_work.
        """
        # Predict only the rows in this split,
        # then write back into split-local X_work.
        pred_labels, _ = self._predict(model, indices=indices, return_proba=False)
        pred_labels = np.asarray(pred_labels, dtype=np.int8)

        # Map global indices -> split row order:
        # X_work rows already correspond to `indices` order.
        om = np.asarray(orig_mask, dtype=bool)
        if om.shape != X_work.shape:
            msg = f"orig_mask shape mismatch: {om.shape} != {X_work.shape}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Update only orig-missing positions
        X_work[om] = pred_labels[om]

    def fit(self) -> "ImputeNLPCA":
        """Fit NLPCA model (joint refinement + input refinement)."""
        self.logger.info(f"Fitting {self.model_name} model...")

        if self.genotype_data.snp_data is None:
            msg = f"SNP data is required for {self.model_name}."
            self.logger.error(msg)
            raise AttributeError(msg)

        self.ploidy = self.cfg.io.ploidy
        self.is_haploid_ = self.ploidy == 1
        self.num_classes_ = 2 if self.is_haploid_ else 3

        if self.ploidy > 2 or self.ploidy < 1:
            msg = (
                f"{self.model_name} supports only haploid (1) or diploid (2); "
                f"got ploidy={self.ploidy}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Prepare ground truth (unknowns stay -1)
        gt_full = self.pgenc.genotypes_012.copy()
        gt_full[gt_full < 0] = -1
        gt_full = np.nan_to_num(gt_full, nan=-1.0)
        self.ground_truth_ = gt_full.astype(np.int8)
        self.num_features_ = int(self.ground_truth_.shape[1])
        self.total_samples_ = int(self.ground_truth_.shape[0])

        if self.is_haploid_:
            self.ground_truth_ = self._haploidize_012(self.ground_truth_)

        # Simulate missingness
        # (for self-supervised eval); do NOT overwrite orig missing.
        sim_tup = self.sim_missing_transform(self.ground_truth_)
        X_for_model_full = sim_tup[0].astype(np.int8, copy=False)
        self.sim_mask_ = sim_tup[1]
        self.orig_mask_ = sim_tup[2]

        if self.is_haploid_:
            X_for_model_full = self._haploidize_012(X_for_model_full)

        self._validate_sim_and_orig_masks(
            sim_mask=self.sim_mask_, orig_mask=self.orig_mask_, context="full"
        )

        # Split indices
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

        # Evaluation masks
        # (simulated-missing positions that were originally observed)
        self.eval_mask_train_ = self.sim_mask_train_ & ~self.orig_mask_train_
        self.eval_mask_val_ = self.sim_mask_val_ & ~self.orig_mask_val_
        self.eval_mask_test_ = self.sim_mask_test_ & ~self.orig_mask_test_

        # Persist baseline clean/corrupted
        # (useful for plots and test evaluation)
        self.X_train_clean_ = X_train_clean
        self.X_val_clean_ = X_val_clean
        self.X_test_clean_ = X_test_clean

        self.X_train_corrupted_ = X_train_corrupted
        self.X_val_corrupted_ = X_val_corrupted
        self.X_test_corrupted_ = X_test_corrupted

        # NLPCA working matrices
        # Fill only originally-missing entries with per-locus mode.
        # Never fill simulated-missing entries.
        self.X_train_work_ = self._initialize_orig_missing_with_mode(
            X_train_corrupted, self.orig_mask_train_
        )
        self.X_val_work_ = self._initialize_orig_missing_with_mode(
            X_val_corrupted, self.orig_mask_val_
        )
        self.X_test_work_ = self._initialize_orig_missing_with_mode(
            X_test_corrupted, self.orig_mask_test_
        )
        self._X_train_work_init_ = self.X_train_work_.copy()

        # For NLPCA training, targets are the working matrix itself.
        # Missing (-1) are ignored by loss via ignore_index and by mask.
        self.X_train_ = self.X_train_work_
        self.y_train_ = self.X_train_work_

        # Validation targets remain the CLEAN matrix for
        # val-loss (observed-only),
        # to keep LR-halving anchored to real observed reconstruction quality.
        self.X_val_ = self.X_val_corrupted_
        self.y_val_ = self.X_val_clean_

        self.X_test_ = self.X_test_corrupted_
        self.y_test_ = self.X_test_clean_

        self.plotter_, self.scorers_ = self.initialize_plotting_and_scorers()

        # Build loaders:
        # Train mask includes observed entries from working input
        # excludes simulated-missing (-1)
        # Val mask uses observed entries from corrupted input (classic)
        train_mask = self.X_train_work_ >= 0
        val_mask = self.X_val_corrupted_ >= 0

        self.train_loader_ = self._get_nlpca_loaders(
            self.train_idx_,
            self.y_train_,
            mask=train_mask,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_loader_ = self._get_nlpca_loaders(
            self.val_idx_,
            self.y_val_,
            mask=val_mask,
            batch_size=self.batch_size,
            shuffle=False,
        )

        # Class weights
        # (computed on observed TRAIN entries only; excludes simulated-missing)
        train_loss_mask = self.X_train_corrupted_ >= 0
        self.class_weights_ = self._class_weights_from_zygosity(
            self.y_train_,
            train_mask=train_loss_mask,
            inverse=self.inverse,
            normalize=self.normalize,
            max_ratio=self.max_ratio,
            power=self.power,
        )

        # Tuning
        if self.tune:
            self.tuned_params_ = self.tune_hyperparameters()
            self.model_tuned_ = True
        else:
            self.model_tuned_ = False
            keys = OBJECTIVE_SPEC_NLPCA.keys
            self.tuned_params_ = {k: getattr(self, k) for k in keys}

        self.best_params_ = copy.deepcopy(self.tuned_params_)

        self._log_class_weights()

        # PCA init for embeddings V
        self.v_init_ = self._get_pca_embedding_init(
            self.ground_truth_, self.train_idx_, int(self.best_params_["latent_dim"])
        )

        # Final model params
        nC = int(self.num_classes_)
        input_dim = int(self.num_features_ * nC)

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

        model = self.build_model(self.Model, self.best_params_["model_params"])

        # Train (joint only + input refinement)
        best_score, trained_model, history = self._execute_nlpca_training(
            model=model,
            lr=float(self.best_params_["learning_rate"]),
            l1_penalty=float(self.best_params_["l1_penalty"]),
            params=self.best_params_,
            trial=None,
            class_weights=self.class_weights_,
            gamma_schedule=bool(self.best_params_["gamma_schedule"]),
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

        # Test eval
        # Projection uses corrupted X; safe because sim-missing remain -1
        self._evaluate_model(
            self.model_,
            self.X_test_corrupted_,
            self.y_test_,
            self.eval_mask_test_,
            indices=self.test_idx_,
            gamma=float(self.best_params_.get("gamma", self.gamma)),
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
        """Impute missing values via final projection and decoding.

        This method fills in all missing values (original + simulated) in the genotype matrix. It first refines the embeddings for all samples using the trained model, then predicts the missing genotypes, and finally decodes the imputed genotypes back to their original representation.

        Returns:
            np.ndarray: Imputed genotype matrix with missing values filled.
        """
        if not getattr(self, "is_fit_", False):
            msg = f"{self.model_name} is not fitted. Must call 'fit()' before 'transform()'."
            self.logger.error(msg)
            raise NotFittedError(msg)

        self.logger.info(f"Imputing entire dataset with {self.model_name}...")

        # Final projection: refine V for all samples with W fixed
        self._refine_all_embeddings(
            self.model_,
            self.ground_truth_,
            float(self.best_params_.get("gamma", self.gamma)),
            class_weights=self.class_weights_,
            lr=self.projection_lr,
            iterations=self.projection_epochs,
        )

        full_indices = np.arange(self.total_samples_)
        pred_labels, _ = self._predict(self.model_, indices=full_indices)

        imputed = self.ground_truth_.copy()
        missing_mask = imputed < 0
        imputed[missing_mask] = pred_labels[missing_mask]

        decode_input = imputed
        if getattr(self, "is_haploid_", False):
            decode_input = imputed.copy()
            decode_input[decode_input == 1] = 2

        decoded = self.decode_012(decode_input)

        if (decoded == "N").any():
            msg = f"Imputation still contains {(decoded == 'N').sum()} missing values ('N')."
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

    def _get_nlpca_loaders(
        self,
        indices: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> torch.utils.data.DataLoader:
        """Create DataLoader yielding (idx, y, mask) batches.

        This method constructs a PyTorch DataLoader that yields batches of sample indices, target genotypes, and observation masks. It ensures that the data types are appropriate for model training and evaluation.

        Args:
            indices (np.ndarray): Array-like of sample indices.
            y (np.ndarray): Array-like of target genotypes (0/1/2).
            mask (np.ndarray): Array-like boolean mask indicating observed positions.
            batch_size (int): Batch size for DataLoader.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            torch.utils.data.DataLoader: DataLoader yielding batches of (idx, y, mask).

        Raises:
            ValueError: If the shapes of indices, y, and mask do not align.

        Notes:
            - idx is int64 (required for nn.Embedding).
            - y is int64 (targets for CE-style losses).
            - mask is bool.
        """
        idx_np = np.asarray(indices, dtype=np.int64)
        y_np = np.asarray(y, dtype=np.int64)
        m_np = np.asarray(mask, dtype=bool)

        if y_np.shape[0] != idx_np.shape[0] or m_np.shape[0] != idx_np.shape[0]:
            msg = f"Loader alignment mismatch: idx={idx_np.shape}, y={y_np.shape}, mask={m_np.shape}."
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
        trial: Optional[optuna.trial.Trial] = None,
        class_weights: Optional[torch.Tensor] = None,
        *,
        persist_projection: bool = False,
    ):
        """Evaluate model on a split, optionally projecting embeddings for the provided indices.

        Args:
            model (torch.nn.Module): The trained NLPCA model.
            X (np.ndarray): Genotype matrix for evaluation (corrupted inputs).
            y (np.ndarray): True genotype labels for evaluation.
            eval_mask (np.ndarray): Boolean mask indicating which entries to evaluate (i.e., simulated-missing positions only).
            indices (np.ndarray): Sample indices corresponding to rows in X and y.
            gamma (float): Gamma parameter for projection refinement.
            project_embedding (bool): If True, perform embedding refinement for the provided indices before evaluation.
            objective_mode (bool): If True, suppresses verbose output and reports only the objective metric.
            trial (Optional[optuna.trial.Trial]): Optuna trial object for hyperparameter tuning (if applicable).
            class_weights (Optional[torch.Tensor]): Optional class weights for projection refinement.
            persist_projection (bool): If False (recommended), embedding refinement performed for evaluation is reverted after scoring so evaluation does not mutate model state.
        """
        if objective_mode and trial is None:
            msg = "objective_mode requires a valid Optuna trial."
            self.logger.error(msg)
            raise TypeError(msg)

        # --- Optional: Non-persistent projection ---
        saved_rows = None
        idx_t = None

        if project_embedding:
            if not persist_projection:
                # Save only the rows we will change
                idx_np = np.asarray(indices, dtype=np.int64)
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

            y_true = y[eval_mask].astype(np.int8)
            y_pred = pred_labels[eval_mask].astype(np.int8)
            y_proba = pred_probas[eval_mask]
            valid = y_true >= 0

            if self.is_haploid_ and y_proba.shape[1] == 3:
                p2 = np.zeros((len(y_proba), 2))
                p2[:, 0], p2[:, 1] = y_proba[:, 0], y_proba[:, 1] + y_proba[:, 2]
                y_proba = p2

            y_true_flat = y_true[valid]
            y_pred_flat = y_pred[valid]
            y_proba_flat = y_proba[valid]

            # --- Harmonize for haploid vs diploid ---
            if self.is_haploid_:
                y_true_flat = (y_true_flat > 0).astype(np.int8, copy=False)
                y_pred_flat = (y_pred_flat > 0).astype(np.int8, copy=False)

                K = y_proba_flat.shape[1]
                if K == 2:
                    pass
                elif K == 3:
                    proba_2 = np.empty(
                        (y_proba_flat.shape[0], 2), dtype=y_proba_flat.dtype
                    )
                    proba_2[:, 0] = y_proba_flat[:, 0]
                    proba_2[:, 1] = y_proba_flat[:, 1] + y_proba_flat[:, 2]
                    y_proba_flat = proba_2
                else:
                    msg = f"Haploid evaluation expects 2 or 3 prob columns; got {K}"
                    self.logger.error(msg)
                    raise ValueError(msg)

                labels_for_scoring = [0, 1]
                target_names = ["REF", "ALT"]
            else:
                if y_proba_flat.shape[1] != 3:
                    msg = f"Diploid evaluation expects 3 prob columns; got {y_proba_flat.shape[1]}"
                    self.logger.error(msg)
                    raise ValueError(msg)
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
                    # Map any ALT-coded >0 to 2 for decode_012, preserve missing -1
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

        finally:
            # Restore embedding rows if evaluation projection should not persist
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
            model (nn.Module): The trained NLPCA model.
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
            return labels.cpu().numpy(), probas.cpu().numpy()
        return labels.cpu().numpy(), None

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
        """Refine *stored* embeddings (model.embedding.weight) on observed entries only, with decoder frozen.

        This version is:
            - Correct for both `indices=None` and `indices=subset` while `X_target` is full.
            - Much faster than cloning v_batch + per-batch optimizer creation.
            - Equivalent in effect: updates only the embedding rows that appear in each batch.

        Args:
            model (nn.Module): Trained NLPCA model with `embedding`, `hidden_layers`, `dense_output`.
            X_target (np.ndarray): Matrix whose observed entries define the refinement objective.
                May be full (N,L) or already subset to `indices`.
            gamma (float): Focal loss gamma used during refinement.
            indices (np.ndarray | None): If provided and X_target is full, refines those rows.
            lr (float): Learning rate for embedding refinement.
            class_weights (torch.Tensor | None): Optional alpha for focal loss.
            iterations (int): Number of refinement passes over the selected rows.
            trial (Optional[optuna.Trial]): Optuna trial for pruning / diagnostics.
        """
        model.eval()

        X_np = np.asarray(X_target, dtype=np.int64)
        if indices is None:
            idx_np = np.arange(X_np.shape[0], dtype=np.int64)
            y_np = X_np
        else:
            idx_np = np.asarray(indices, dtype=np.int64)

            # Allow either:
            #   (a) X_target is full matrix: shape[0] == total_samples_
            #   (b) X_target is already subset: shape[0] == len(indices)
            if X_np.shape[0] == idx_np.shape[0]:
                y_np = X_np
            elif X_np.shape[0] == int(self.total_samples_):
                y_np = X_np[idx_np]
            else:
                msg = (
                    f"[{self.model_name}] X_target has incompatible first dimension for indices. "
                    f"Got X_target.shape[0]={X_np.shape[0]}, len(indices)={idx_np.shape[0]}, "
                    f"total_samples_={getattr(self, 'total_samples_', 'NA')}."
                )
                self.logger.error(msg)
                raise ValueError(msg)

        m_np = y_np >= 0  # observed-only mask for refinement objective

        loader = self._get_nlpca_loaders(
            idx_np,
            y_np,
            m_np,
            batch_size=int(self.batch_size),
            shuffle=False,
        )

        alpha = class_weights if class_weights is not None else self.class_weights_
        if alpha is not None and alpha.device != self.device:
            alpha = alpha.to(self.device)

        criterion = FocalCELoss(alpha=alpha, gamma=float(gamma), ignore_index=-1)

        # Save/disable grads for decoder; enable only embedding
        saved: list[tuple[torch.nn.Parameter, bool]] = []
        for p in model.parameters():
            saved.append((p, bool(p.requires_grad)))
            p.requires_grad_(False)

        # Only embedding is trainable
        model.embedding.weight.requires_grad_(True)  # type: ignore[attr-defined]

        opt = torch.optim.Adam([model.embedding.weight], lr=float(lr))  # type: ignore[attr-defined]

        try:
            with torch.enable_grad():
                for _ in range(int(iterations)):
                    for idx_b, y_b, m_b in loader:
                        idx_b = idx_b.to(self.device, non_blocking=True).long()
                        y_b = y_b.to(self.device, non_blocking=True).long()
                        m_b = m_b.to(self.device, non_blocking=True).bool()

                        flat_mask = m_b.view(-1)
                        if flat_mask.sum().item() == 0:
                            continue

                        opt.zero_grad(set_to_none=True)

                        out = model(idx_b)  # uses current embedding rows for idx_b
                        loss = criterion(
                            out.view(-1, self.num_classes_)[flat_mask],
                            y_b.view(-1)[flat_mask],
                        )

                        if not torch.isfinite(loss):
                            msg = f"[{self.model_name}] Embedding refinement loss non-finite."
                            self.logger.error(msg)
                            raise RuntimeError(msg)

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            opt.param_groups[0]["params"], max_norm=1.0
                        )
                        opt.step()

        except Exception as e:
            if trial is not None:
                msg = f"[{self.model_name}] Trial pruned during embedding refinement due to exception: {e}"
                raise optuna.exceptions.TrialPruned(msg) from e
            else:
                raise e

        finally:
            # Restore requires_grad flags exactly as they were
            for p, req in saved:
                p.requires_grad_(req)

    def _execute_nlpca_training(
        self,
        model: nn.Module,
        lr: float,
        l1_penalty: float,
        params: dict[str, Any],
        trial: Optional[optuna.Trial],
        class_weights: Optional[torch.Tensor],
        gamma_schedule: bool,
    ) -> tuple[float, nn.Module, dict[str, list[float]]]:
        """Execute NLPCA training: joint refinement only + input refinement.

        Args:
            model (nn.Module): NLPCA model to train.
            lr (float): Learning rate for joint training.
            l1_penalty (float): L1 penalty coefficient.
            params (dict[str, Any]): Hyperparameter dictionary.
            trial (Optional[optuna.Trial]): Optuna trial for pruning / diagnostics.
            class_weights (Optional[torch.Tensor]): Optional class weights for focal loss.
            gamma_schedule (bool): Whether to use gamma annealing during training.

        Returns:
            tuple[float, nn.Module, dict[str, list[float]]]: Best validation score, trained model, and training history.
        """
        gamma_target, gamma_warm, gamma_ramp = self._anneal_config(
            params, "gamma", default=self.gamma, max_epochs=self.epochs
        )

        cw = class_weights
        if cw is not None and cw.device != self.device:
            cw = cw.to(self.device)

        criterion = FocalCELoss(
            alpha=cw, gamma=gamma_target, reduction="mean", ignore_index=-1
        )

        best_score, history = self._run_nlpca_loop(
            model=model,
            lr=float(lr),
            l1=float(l1_penalty),
            criterion=criterion,
            trial=trial,
            params=params,
            gamma_schedule=gamma_schedule,
        )
        return best_score, model, history

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
        """Train one epoch (NLPCA uses phase=3 only).

        Args:
            model (nn.Module): Decoder model with an nn.Embedding named `embedding`.
            temp_layer (Optional[nn.Module]): Only used for phase=1 legacy paths.
            optimizer (torch.optim.Optimizer): Optimizer over the currently-unfrozen params.
            criterion (nn.Module): Loss function (e.g., FocalCELoss).
            l1 (float): L1 penalty coefficient.
            phase (int): Training phase identifier (1, 2, or 3).
            trial (Optional[optuna.Trial]): Optuna trial for pruning / diagnostics.

        Returns:
            float: Average training loss for the epoch.

        Raises:
            RuntimeError: If no valid batches or non-finite loss occurs.
        """
        model.train()
        if temp_layer is not None:
            temp_layer.train()

        running = 0.0
        n_batches = 0

        with torch.enable_grad():
            for idx, y, m in self.train_loader_:
                idx = idx.to(self.device, non_blocking=True).long()
                y = y.to(self.device, non_blocking=True).long()
                m = m.to(self.device, non_blocking=True).bool()

                flat_mask = m.view(-1)
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
                targets_1d = y.view(-1)[flat_mask]

                loss = criterion(logits_2d, targets_1d)

                # L1 regularization (only on trainable blocks for the phase)
                if l1 > 0.0:
                    reg = torch.zeros((), device=self.device)
                    if phase == 1:
                        reg = reg + model.embedding.weight.abs().sum()  # type: ignore[attr-defined]
                        reg = reg + temp_layer.weight.abs().sum()  # type: ignore[union-attr]
                    else:
                        # NLPCA/phase-3:
                        # decoder weights + embedding may be trainable
                        if hasattr(model, "hidden_layers"):
                            for p in model.hidden_layers.parameters():  # type: ignore[attr-defined]
                                if p.requires_grad:
                                    reg = reg + p.abs().sum()
                        if hasattr(model, "dense_output"):
                            for p in model.dense_output.parameters():  # type: ignore[attr-defined]
                                if p.requires_grad:
                                    reg = reg + p.abs().sum()
                        if hasattr(model, "embedding") and model.embedding.weight.requires_grad:  # type: ignore[attr-defined]
                            reg = reg + model.embedding.weight.abs().sum()  # type: ignore[attr-defined]
                    loss = loss + float(l1) * reg

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
                        f"  temp_layer={'present' if temp_layer is not None else 'None'}\n"
                        "Likely causes: outer no_grad/inference_mode OR all params frozen OR forward detaches tensors."
                    )
                    self.logger.error(msg)
                    raise RuntimeError(msg)

                if not torch.isfinite(loss):
                    if trial is not None:
                        msg = f"[{self.model_name}] Trial {trial.number} training loss non-finite. Pruning."
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

    def _run_nlpca_loop(
        self,
        model: nn.Module,
        lr: float,
        l1: float,
        criterion: nn.Module,
        trial: Optional[optuna.Trial] = None,
        params: Optional[dict[str, Any]] = None,
        gamma_schedule: bool = False,
    ) -> tuple[float, dict[str, list[float]]]:
        """Run joint (phase-3-like) training with ReduceLROnPlateau + input refinement.

        This loop jointly optimizes embeddings (V) and decoder weights (W). It replaces the manual LR-halving-on-plateau logic with PyTorch's ReduceLROnPlateau, stepping on a validation loss computed via projection-based evaluation. Training stops when the LR reaches eta_min and the objective has plateaued, or when max epochs is reached.

        Args:
            model (nn.Module): NLPCA model to train.
            lr (float): Initial learning rate.
            l1 (float): L1 penalty.
            criterion (nn.Module): Loss criterion to use.
            trial (Optional[optuna.Trial]): Optuna trial for pruning. Can be None.
            params (Optional[dict[str, Any]]): Full parameter dict (for scheduling). Can be None.
            gamma_schedule (bool): Whether to use gamma scheduling. Defaults to False.

        Returns:
            tuple[float, dict[str, list[float]]]: Best validation score achieved during training and training history.

        Raises:
            optuna.exceptions.TrialPruned: If the trial is pruned due to lack of improvement.
        """
        eta0 = float(lr)
        s_best = float("inf")

        patience = 5
        plateau_counter = 0

        # Joint optimization only (V + W)
        for p in model.parameters():
            p.requires_grad_(False)

        params_to_opt: list[torch.nn.Parameter] = []

        model.embedding.weight.requires_grad_(True)  # type: ignore[attr-defined]
        params_to_opt.append(model.embedding.weight)  # type: ignore[attr-defined]

        for p in model.hidden_layers.parameters():  # type: ignore[attr-defined]
            p.requires_grad_(True)
        for p in model.dense_output.parameters():  # type: ignore[attr-defined]
            p.requires_grad_(True)

        params_to_opt.extend(list(model.hidden_layers.parameters()))  # type: ignore[attr-defined]
        params_to_opt.extend(list(model.dense_output.parameters()))  # type: ignore[attr-defined]

        optimizer = torch.optim.AdamW(params_to_opt, lr=eta0)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
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
                temp_layer=None,
                optimizer=optimizer,
                criterion=criterion,
                l1=l1,
                phase=3,
                trial=trial,
            )

            # Input refinement (EM-like):
            # update ONLY orig-missing entries in the working matrix
            if (
                getattr(self, "input_refine_every", 0) > 0
                and (epoch % int(self.input_refine_every)) == 0
                and getattr(self, "X_train_work_", None) is not None
            ):
                refine_steps = max(1, int(getattr(self, "input_refine_steps", 1)))
                orig_mask = getattr(self, "orig_mask_train_", None)
                if self.debug and orig_mask is not None:
                    before_vals = self.X_train_work_[orig_mask].copy()  # type: ignore[index]
                else:
                    before_vals = None

                for _ in range(refine_steps):
                    self._update_orig_missing_from_model(
                        model=model,
                        X_work=self.X_train_work_,  # type: ignore[arg-type]
                        orig_mask=self.orig_mask_train_,
                        indices=self.train_idx_,
                    )

                # Rebuild train loader mask so filled orig-missing participate as "observed"
                # (sim-missing remain -1 and stay excluded)
                self.train_loader_ = self._get_nlpca_loaders(
                    self.train_idx_,
                    self.X_train_work_,  # type: ignore[arg-type]
                    mask=(self.X_train_work_ >= 0),  # type: ignore[operator]
                    batch_size=self.batch_size,
                    shuffle=True,
                )

                if self.debug:
                    n_obs = int((self.X_train_work_ >= 0).sum())  # type: ignore[operator]
                    n_total = int(self.X_train_work_.size)  # type: ignore[union-attr]
                    updated = None
                    total_targets = None
                    if before_vals is not None and orig_mask is not None:
                        after_vals = self.X_train_work_[orig_mask]  # type: ignore[index]
                        updated = int(np.count_nonzero(before_vals != after_vals))
                        total_targets = int(orig_mask.sum())
                    update_msg = ""
                    if updated is not None and total_targets is not None:
                        update_msg = f", updated={updated}/{total_targets}"
                    self.logger.debug(
                        f"[{self.model_name}] Input refine epoch {epoch}: steps={refine_steps}{update_msg}, observed={n_obs}/{n_total} ({(100.0 * n_obs / float(n_total)):.2f}%)."
                    )

            # Validation: projection-based observed-loss
            s = self._val_step_with_projection(
                model=model,
                temp_layer=None,
                criterion=criterion,
                steps=max(10, min(int(self.projection_epochs) // 5, 20)),
                lr=float(self.projection_lr),
            )

            train_history.append(train_loss)
            val_history.append(float(s))

            # Plateau tracking (mirrors your prior relative-improvement gating)
            if np.isfinite(s) and s < s_best:
                if s_best == float("inf"):
                    s_best = float(s)
                    plateau_counter = 0
                else:
                    improvement_ratio = (
                        1.0 - (float(s) / float(s_best)) if s_best != 0.0 else 0.0
                    )
                    if improvement_ratio > float(self.gamma_threshold):
                        s_best = float(s)
                        plateau_counter = 0
                    else:
                        plateau_counter += 1
            else:
                plateau_counter += 1

            lr_before = float(optimizer.param_groups[0]["lr"])
            scheduler.step(float(s))
            lr_after = float(optimizer.param_groups[0]["lr"])

            if lr_after < lr_before:
                self.logger.debug(
                    f"NLPCA: ReduceLROnPlateau LR {lr_before:.2e} -> {lr_after:.2e} (train={train_loss:.4f}, val={float(s):.4f}, gamma={float(getattr(criterion, 'gamma', 0.0)):.3f})"
                )

            # Stop once we're at the floor and still plateauing
            at_floor = lr_after <= (float(self.eta_min) * (1.0 + 1e-12))
            if at_floor and plateau_counter >= int(patience):
                break

            if trial is not None and isinstance(self.tune_metric, str):
                trial.report(-float(s), step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned(
                        f"[{self.model_name}] Trial {trial.number} pruned at epoch {epoch}."
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
        """Validate with per-batch embedding projection (optimize v_batch; do NOT mutate model state)."""
        model.eval()
        total_loss = 0.0
        count = 0

        saved: list[tuple[torch.nn.Parameter, bool]] = []

        def _save_and_disable(module: Optional[nn.Module]) -> None:
            if module is None:
                return
            for p in module.parameters():
                saved.append((p, bool(p.requires_grad)))
                p.requires_grad_(False)

        # During projection: grads only w.r.t. v_batch
        _save_and_disable(getattr(model, "hidden_layers", None))
        _save_and_disable(getattr(model, "dense_output", None))
        _save_and_disable(temp_layer)

        try:
            with torch.enable_grad():
                for idx, y, m in self.val_loader_:
                    idx = idx.to(self.device, non_blocking=True).long()
                    y = y.to(self.device, non_blocking=True).long()
                    m = m.to(self.device, non_blocking=True).bool()

                    flat_mask = m.view(-1)
                    if flat_mask.sum().item() == 0:
                        continue

                    v_batch = model.embedding(idx).detach().clone()  # type: ignore[attr-defined]
                    v_batch = v_batch.to(self.device)
                    v_batch.requires_grad_(True)

                    proj_opt = torch.optim.Adam([v_batch], lr=float(lr))

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
                            y.view(-1)[flat_mask],
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
                            y.view(-1)[flat_mask],
                        )
                        total_loss += float(val_l.item())
                        count += 1

            if count == 0:
                msg = f"[{self.model_name}] Validation loss has no valid batches."
                self.logger.error(msg)
                raise RuntimeError(msg)

            return total_loss / count

        finally:
            for p, req in saved:
                p.requires_grad_(req)

    def _objective(self, trial: optuna.Trial) -> float | tuple[float, ...]:
        """Optuna objective for NLPCA (joint-only + safe per-trial work reset)."""
        params = self._sample_hyperparameters(trial)

        # PCA init for this latent_dim
        params["model_params"]["embedding_init"] = self._get_pca_embedding_init(
            self.ground_truth_, self.train_idx_, int(params["latent_dim"])
        )

        model = self.build_model(self.Model, params["model_params"])

        # Recompute class weights on observed train entries
        train_loss_mask = self.X_train_corrupted_ >= 0
        class_weights = self._class_weights_from_zygosity(
            self.X_train_clean_,
            train_mask=train_loss_mask,
            inverse=params["inverse"],
            normalize=params["normalize"],
            max_ratio=self.max_ratio,
            power=params["power"],
        )

        # ---- Per-trial reset of the NLPCA working matrix + loader ----
        if self._X_train_work_init_ is None:
            msg = "Internal error: _X_train_work_init_ is not set."
            self.logger.error(msg)
            raise RuntimeError(msg)

        saved_work = self.X_train_work_
        saved_loader = self.train_loader_
        try:
            self.X_train_work_ = self._X_train_work_init_.copy()
            self.train_loader_ = self._get_nlpca_loaders(
                self.train_idx_,
                self.X_train_work_,
                mask=(self.X_train_work_ >= 0),
                batch_size=self.batch_size,
                shuffle=True,
            )

            _ = self._execute_nlpca_training(
                model=model,
                lr=float(params["learning_rate"]),
                l1_penalty=float(params["l1_penalty"]),
                params=params,
                trial=trial,
                class_weights=class_weights,
                gamma_schedule=bool(params["gamma_schedule"]),
            )

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
                class_weights=class_weights,
                persist_projection=False,
            )

        finally:
            self.X_train_work_ = saved_work
            self.train_loader_ = saved_loader

        if isinstance(self.tune_metric, (list, tuple)):
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

        OBJECTIVE_SPEC_NLPCA.validate(params)

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

        Args:
            X_full (np.ndarray): Full 012 matrix shape (N, L) with missing encoded as -1.
            train_idx (np.ndarray): Indices of training samples (subset of [0..N-1]).
            latent_dim (int): Number of PCA components (embedding dimension).

        Returns:
            Tensor of shape (N, latent_dim) on self.device.
        """
        X = np.asarray(X_full, dtype=np.float32)
        N, L = X.shape
        train_idx = np.asarray(train_idx, dtype=np.int64)

        # Compute training column means ignoring -1
        X_train = X[train_idx]
        col_means = self._compute_col_means_ignore_missing(
            X_train, missing_value=-1.0, fill_value=0.0
        )

        # Fill missing in full matrix using training means
        missing_full = X == -1
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

        pca = PCA(n_components=int(latent_dim), random_state=self.seed)
        pca.fit(X_filled[train_idx])
        V = pca.transform(X_filled)  # (N, latent_dim)
        if self.debug and hasattr(pca, "explained_variance_ratio_"):
            evr = np.asarray(pca.explained_variance_ratio_, dtype=float)
            head = evr[: min(5, evr.size)].tolist()
            self.logger.debug(
                f"[{self.model_name}] PCA init EVR head={head}, total={float(evr.sum()):.4f}."
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
