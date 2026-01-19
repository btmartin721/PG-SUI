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

    Trains a VAE on a genotype matrix encoded as 0/1/2 with missing values represented by any negative integer. The workflow simulates missingness once on the full matrix, then creates train/val/test splits. It supports haploid and diploid data, focal-CE reconstruction loss with a KL term (optional scheduling), and Optuna-based hyperparameter tuning. Output is returned as IUPAC strings via ``decode_012``.

    Notes:
        - Training includes early stopping based on validation loss.
        - The imputer can handle both haploid and diploid genotype data.
    """

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

        Args:
            genotype_data (GenotypeData): Genotype data for imputation.
            tree_parser (Optional[TreeParser]): Tree parser required for nonrandom strategies.
            config (Optional[Union[VAEConfig, dict, str]]): Config dataclass, nested dict, YAML path, or None.
            overrides (Optional[dict]): Dot-key overrides applied last with highest precedence.
            sim_strategy (Literal["random", "random_weighted", "random_weighted_inv", "nonrandom", "nonrandom_weighted"]): Missingness simulation strategy (overrides config).
            sim_prop (Optional[float]): Proportion of entries to simulate as missing (overrides config). Default is None.
            sim_kwargs (Optional[dict]): Extra missingness kwargs merged into config.
        """
        self.model_name = "ImputeVAE"
        self.genotype_data = genotype_data
        self.tree_parser = tree_parser

        cfg = ensure_vae_config(config)
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

        if self.tree_parser is None and self.sim_strategy.startswith("nonrandom"):
            msg = "tree_parser is required for nonrandom sim strategies."
            self.logger.error(msg)
            raise ValueError(msg)

        self.num_tuned_params_ = OBJECTIVE_SPEC_VAE.count()

    def fit(self) -> "ImputeVAE":
        """Fit the VAE imputer model to the genotype data.

        This method performs the following steps:
            1. Validates the presence of SNP data in the genotype data.
            2. Determines the ploidy of the genotype data and sets up haploid/diploid handling.
            3. Simulates missingness in the genotype data based on the specified strategy.
            4. Splits the data into training, validation, and test sets.
            5. One-hot encodes the genotype data for model input.
            6. Initializes data loaders for training and validation.
            7. If hyperparameter tuning is enabled, tunes the model hyperparameters.
            8. Builds the VAE model with the best hyperparameters.
            9. Trains the VAE model using the training data and validates on the validation set.
            10. Evaluates the trained model on the test set and computes performance metrics.
            11. Saves the trained model and best hyperparameters.
            12. Generates plots of training history if enabled.
            13. Returns the fitted ImputeVAE instance.

        Returns:
            ImputeVAE: The fitted ImputeVAE instance.
        """
        self.logger.info(f"Fitting {self.model_name} model...")

        if self.genotype_data.snp_data is None:
            msg = f"SNP data is required for {self.model_name}."
            self.logger.error(msg)
            raise AttributeError(msg)

        self.ploidy = self.cfg.io.ploidy
        self.is_haploid_ = self.ploidy == 1

        if self.ploidy > 2 or self.ploidy < 1:
            msg = f"{self.model_name} currently supports only haploid (1) or diploid (2) data; got ploidy={self.ploidy}."
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.debug(
            f"Ploidy set to {self.ploidy}, is_haploid: {self.is_haploid_}"
        )

        self.num_classes_ = 2 if self.is_haploid_ else 3

        gt_full = self.pgenc.genotypes_012.copy()
        gt_full[gt_full < 0] = -1
        gt_full = np.nan_to_num(gt_full, nan=-1.0)
        self.ground_truth_ = gt_full.astype(np.int8)
        self.num_features_ = gt_full.shape[1]

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
        X_for_model_full = sim_tup[0]
        self.sim_mask_ = sim_tup[1]
        self.orig_mask_ = sim_tup[2]

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

        self.eval_mask_train_ = self.sim_mask_train_ & ~self.orig_mask_train_
        self.eval_mask_val_ = self.sim_mask_val_ & ~self.orig_mask_val_
        self.eval_mask_test_ = self.sim_mask_test_ & ~self.orig_mask_test_

        self.validate_and_log_masks()

        # --- Haploid harmonization (do NOT resimulate; just recode values) ---
        if self.is_haploid_:
            self.logger.debug(
                "Performing haploid harmonization on split inputs/targets..."
            )

            def _haploidize(arr: np.ndarray) -> np.ndarray:
                out = arr.copy()
                miss = out < 0
                out = np.where(out > 0, 1, out).astype(np.int8, copy=True)
                out[miss] = -1
                return out

            X_train_clean = _haploidize(X_train_clean)
            X_val_clean = _haploidize(X_val_clean)
            X_test_clean = _haploidize(X_test_clean)
            X_train_corrupted = _haploidize(X_train_corrupted)
            X_val_corrupted = _haploidize(X_val_corrupted)
            X_test_corrupted = _haploidize(X_test_corrupted)

        # Write back the persisted versions too
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

        self.y_test_ = self.X_test_clean_

        self.X_train_ = self._one_hot_encode_012(
            self.X_train_, num_classes=self.num_classes_
        )

        self.X_val_ = self._one_hot_encode_012(
            self.X_val_, num_classes=self.num_classes_
        )

        for name, tensor in [("X_train_", self.X_train_), ("X_val_", self.X_val_)]:
            if torch.is_tensor(tensor) and (tensor.sum(dim=-1) > 1).any():
                msg = f"[{self.model_name}] Invalid one-hot: >1 active class in {name}."
                self.logger.error(msg)
                raise RuntimeError(msg)

        self.plotter_, self.scorers_ = self.initialize_plotting_and_scorers()

        train_loader = self._get_data_loaders(
            self.X_train_.numpy(force=True),
            self.y_train_,
            self.eval_mask_train_,
            self.batch_size,
            shuffle=True,
        )

        val_loader = self._get_data_loaders(
            self.X_val_.numpy(force=True),
            self.y_val_,
            self.eval_mask_val_,
            self.batch_size,
            shuffle=False,
        )

        self.train_loader_ = train_loader
        self.val_loader_ = val_loader

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

        # Always start clean
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

        # Now build the model
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

        self.best_loss_ = loss
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

        Notes:
            - ``transform()`` does not take any arguments; it operates on the data provided during initialization.
            - Ensure that the model has been fitted before calling this method.
            - For haploid data, genotypes encoded as '1' are treated as '2' during decoding.
            - The method checks for decoding failures (i.e., resulting in 'N') and raises an error if any are found.
        """
        if not getattr(self, "is_fit_", False):
            msg = f"{self.model_name} is not fitted. Must call 'fit()' before 'transform()'."
            self.logger.error(msg)
            raise NotFittedError(msg)

        self.logger.info(f"Imputing entire dataset with {self.model_name}...")
        X_to_impute = self.ground_truth_.copy()

        # 1. Predict labels (0/1/2) for the entire matrix
        pred_labels, _ = self._predict(self.model_, X=X_to_impute)

        # 2. Fill ONLY originally missing values
        missing_mask = X_to_impute < 0
        imputed_array = X_to_impute.copy()
        imputed_array[missing_mask] = pred_labels[missing_mask]

        # Sanity check: all missing values should be gone
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

        if (imputed_gt == "N").any():
            msg = f"Something went wrong: {self.model_name} imputation still contains {(imputed_gt == 'N').sum()} missing values ('N')."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if self.show_plots:
            original_input = X_to_impute

            if getattr(self, "is_haploid_", False):
                original_input = X_to_impute.copy()
                original_input[original_input == 1] = 2

            plt.rcParams.update(self.plotter_.param_dict)

            orig_dec = self.decode_012(original_input)
            self.plotter_.plot_gt_distribution(imputed_gt, orig_dec, True)

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

        This method orchestrates training with early stopping and optional Optuna pruning based on validation performance. It returns the best validation loss, the best model (with best weights loaded), and training history.

        Args:
            model (torch.nn.Module): VAE model.
            lr (float): Learning rate.
            l1_penalty (float): L1 regularization coefficient.
            trial (Optional[optuna.Trial]): Optuna trial for pruning (optional).
            params (Optional[dict[str, float | int | str | dict[str, Any]]]): Model params for evaluation.
            class_weights (Optional[torch.Tensor]): Class weights for loss computation.
            kl_beta_schedule (bool): Whether to use KL beta scheduling.
            gamma_schedule (bool): Whether to use gamma scheduling for focal CE loss.

        Returns:
            tuple[float, torch.nn.Module, dict[str, list[float]]]:
                Best validation loss, best model, and training history.
        """
        max_epochs = self.epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # Calculate default warmup
        warmup_epochs = max(int(0.02 * max_epochs), 10)

        # Check if patience is too short for the calculated warmup
        if self.early_stop_gen <= warmup_epochs:
            warmup_epochs = max(0, self.early_stop_gen - 1)

            msg = f"Early stopping patience ({self.early_stop_gen}) <= default warmup; adjusting warmup to {warmup_epochs}."
            self.logger.warning(msg)

        scheduler = _make_warmup_cosine_scheduler(
            optimizer, max_epochs=max_epochs, warmup_epochs=warmup_epochs
        )

        best_loss, best_model, hist = self._execute_training_loop(
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            l1_penalty=l1_penalty,
            trial=trial,
            params=params,
            class_weights=class_weights,
            kl_beta_schedule=kl_beta_schedule,
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
        kl_beta_schedule: bool = False,
        gamma_schedule: bool = False,
    ) -> tuple[float, torch.nn.Module, dict[str, list[float]]]:
        """Train the model with focal CE reconstruction + KL divergence.

        This method performs the training loop for the model using the provided optimizer and learning rate scheduler. It supports early stopping based on validation loss and integrates with Optuna for hyperparameter tuning. The method returns the best validation loss, the best model state, and the training history.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler.CosineAnnealingLR | torch.optim.lr_scheduler.SequentialLR): Learning rate scheduler.
            model (torch.nn.Module): VAE model.
            l1_penalty (float): L1 regularization coefficient.
            trial (Optional[optuna.Trial]): Optuna trial for pruning (optional).
            params (Optional[dict[str, Any]]): Model params for evaluation.
            class_weights (Optional[torch.Tensor]): Class weights for loss computation.
            kl_beta_schedule (bool): Whether to use KL beta scheduling.
            gamma_schedule (bool): Whether to use gamma scheduling for focal CE loss.

        Returns:
            tuple[float, torch.nn.Module, dict[str, list[float]]]: Best validation loss, best model, training history.

        Notes:
            - Use CE with class weights during training/validation.
            - Inference de-bias happens in _predict (separate).
        """
        history: dict[str, list[float]] = defaultdict(list)

        early_stopping = EarlyStopping(
            patience=self.early_stop_gen,
            min_epochs=self.min_epochs,
            verbose=self.verbose,
            prefix=self.prefix,
            debug=self.debug,
        )

        # KL schedule
        kl_beta_target, kl_warm, kl_ramp = self._anneal_config(
            params, "kl_beta", default=self.kl_beta, max_epochs=self.epochs
        )

        kl_beta_target = float(kl_beta_target)

        gamma_target, gamma_warm, gamma_ramp = self._anneal_config(
            params, "gamma", default=self.gamma, max_epochs=self.epochs
        )

        cw = class_weights
        if cw is not None and cw.device != self.device:
            cw = cw.to(self.device)

        ce_criterion = FocalCELoss(
            alpha=cw, gamma=gamma_target, reduction="mean", ignore_index=-1
        )

        for epoch in range(int(self.epochs)):
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
                    gamma_target,
                    warm=gamma_warm,
                    ramp=gamma_ramp,
                    epoch=epoch,
                    init_val=0.0,
                )
                ce_criterion.gamma = gamma_current

            train_loss = self._train_step(
                loader=self.train_loader_,
                optimizer=optimizer,
                model=model,
                ce_criterion=ce_criterion,
                trial=trial,
                l1_penalty=l1_penalty,
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
                l1_penalty=l1_penalty,
                kl_beta=kl_beta_current,
            )

            if self.debug and epoch % 10 == 0:
                self.logger.debug(
                    f"[{self.model_name}] Epoch {epoch + 1}/{self.epochs}"
                )
                msg = f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}"
                self.logger.debug(msg)
                self.logger.debug(f"KL Beta: {kl_beta_current:.6f}")

                if gamma_schedule:
                    msg2 = f"Focal CE Gamma: {ce_criterion.gamma:.6f}"
                    self.logger.debug(msg2)
                self.logger.debug(f"Train Loss: {train_loss:.6f}")
                self.logger.debug(f"Val Loss: {val_loss:.6f}")

            scheduler.step()

            history["Train"].append(float(train_loss))
            history["Val"].append(float(val_loss))

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                self.logger.debug(
                    f"[{self.model_name}] Early stopping at epoch {epoch + 1}."
                )
                break

            if trial is not None and isinstance(self.tune_metric, str):
                trial.report(-val_loss, step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned(
                        f"[{self.model_name}] Trial {trial.number} pruned at epoch {epoch}. This is not an error, but indicates the trial was not promising and has been stopped early for efficiency."
                    )

        best_loss = float(early_stopping.best_score)

        if early_stopping.best_state_dict is not None:
            model.load_state_dict(early_stopping.best_state_dict)

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

        Args:
            loader (torch.utils.data.DataLoader): Training data loader.
            optimizer (torch.optim.Optimizer): Optimizer.
            model (torch.nn.Module): VAE model.
            ce_criterion (torch.nn.Module): Cross-entropy loss function.
            trial (Optional[optuna.Trial]): Optuna trial for pruning (optional).
            l1_penalty (float): L1 regularization coefficient.
            kl_beta (torch.Tensor | float): KL divergence weight.

        Returns:
            float: Average training loss.

        """
        model.train()
        running = 0.0
        num_batches = 0

        nF_model = self.num_features_
        nC_model = self.num_classes_
        l1_params = tuple(p for p in model.parameters() if p.requires_grad)

        for X_batch, y_batch, m_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            X_batch = X_batch.to(self.device, non_blocking=True).float()
            y_batch = y_batch.to(self.device, non_blocking=True).long()
            m_batch = m_batch.to(self.device, non_blocking=True).bool()

            raw = model(X_batch)
            logits0 = raw[0]

            expected = X_batch.shape[0] * nF_model * nC_model
            if logits0.numel() != expected:
                msg = f"{self.model_name} logits size mismatch: got {logits0.numel()}, expected {expected}"
                self.logger.error(msg)
                raise ValueError(msg)

            logits_masked = logits0.view(-1, nC_model)
            logits_masked = logits_masked[m_batch.view(-1)]

            targets_masked = y_batch.view(-1)
            targets_masked = targets_masked[m_batch.view(-1)]

            if targets_masked.numel() == 0:
                continue

            if torch.any(targets_masked < 0):
                msg = "Masked targets contain negative labels; mask/targets are inconsistent."
                self.logger.error(msg)
                raise ValueError(msg)

            # average number of masked loci per sample (scalar)
            recon_scale = (
                m_batch.view(-1).sum().float() / float(X_batch.shape[0])
            ).detach()

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
                loss = loss + l1_penalty * l1

            if trial is not None:
                if not torch.isfinite(loss):
                    msg = f"[{self.model_name}] Trial {trial.number} training loss non-finite. Pruning trial."
                    self.logger.warning(msg)
                    raise optuna.exceptions.TrialPruned(msg)
            elif trial is None and not torch.isfinite(loss):
                msg = f"[{self.model_name}] Training loss non-finite."
                self.logger.error(msg)
                raise RuntimeError(msg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running += float(loss.detach().item())
            num_batches += 1

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
        kl_beta: torch.Tensor | float = 1.0,
    ) -> float:
        """Validation step for a single epoch (focal CE + KL + optional L1).

        Args:
            loader (torch.utils.data.DataLoader): Validation data loader.
            model (torch.nn.Module): VAE model.
            ce_criterion (torch.nn.Module): Cross-entropy loss function.
            trial (Optional[optuna.Trial]): Optuna trial for pruning (optional).
            l1_penalty (float): L1 regularization coefficient.
            kl_beta (torch.Tensor | float): KL divergence weight.

        Returns:
            float: Average validation loss.
        """
        model.eval()
        running = 0.0
        num_batches = 0

        nF_model = self.num_features_
        nC_model = self.num_classes_
        l1_params = tuple(p for p in model.parameters() if p.requires_grad)

        with torch.no_grad():
            for X_batch, y_batch, m_batch in loader:
                X_batch = X_batch.to(self.device, non_blocking=True).float()
                y_batch = y_batch.to(self.device, non_blocking=True).long()
                m_batch = m_batch.to(self.device, non_blocking=True).bool()

                raw = model(X_batch)
                logits0 = raw[0]

                expected = X_batch.shape[0] * nF_model * nC_model
                if logits0.numel() != expected:
                    msg = f"VAE logits size mismatch: got {logits0.numel()}, expected {expected}"
                    self.logger.error(msg)
                    raise ValueError(msg)

                logits_masked = logits0.view(-1, nC_model)
                logits_masked = logits_masked[m_batch.view(-1)]

                targets_masked = y_batch.view(-1).long()
                targets_masked = targets_masked[m_batch.view(-1)]

                if targets_masked.numel() == 0:
                    continue

                if torch.any(targets_masked < 0):
                    msg = "Masked targets contain negative labels; mask/targets are inconsistent."
                    self.logger.error(msg)
                    raise ValueError(msg)

                # average number of masked loci per sample (scalar)
                recon_scale = (
                    m_batch.view(-1).sum().float() / float(X_batch.shape[0])
                ).detach()

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
                    loss = loss + l1_penalty * l1

                if trial is not None:
                    if not torch.isfinite(loss):
                        msg = f"[{self.model_name}] Trial {trial.number} validation loss non-finite. Pruning trial."
                        self.logger.warning(msg)
                        raise optuna.exceptions.TrialPruned(msg)
                elif trial is None and not torch.isfinite(loss):
                    msg = f"[{self.model_name}] Validation loss non-finite."
                    self.logger.error(msg)
                    raise RuntimeError(msg)

                running += float(loss.item())
                num_batches += 1

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
                "Model passed to predict() is not trained. "
                "Call fit() before predict()."
            )
            self.logger.error(msg)
            raise NotFittedError(msg)

        model.eval()

        nF = self.num_features_
        nC = self.num_classes_

        X_tensor = X if isinstance(X, torch.Tensor) else torch.from_numpy(X)
        X_tensor = X_tensor.float()

        if X_tensor.device != self.device:
            X_tensor = X_tensor.to(self.device)

        if X_tensor.dim() == 2:
            # 0/1/2 matrix -> one-hot for model input
            X_tensor = self._one_hot_encode_012(X_tensor, num_classes=nC)
            X_tensor = X_tensor.float()

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
            logits = raw[0].view(-1, nF, nC)
            probas = torch.softmax(logits, dim=-1)
            labels = torch.argmax(probas, dim=-1)

        if return_proba:
            return labels.cpu().numpy(), probas.cpu().numpy()
        return labels.cpu().numpy(), None

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

        This method evaluates the performance of the trained model on a given dataset using a specified evaluation mask. It computes various classification metrics based on the predicted labels and probabilities, comparing them to the ground truth labels. The method returns a dictionary of evaluation metrics.

        Args:
            model (torch.nn.Module): Trained model.
            X (np.ndarray | torch.Tensor): 0/1/2 matrix with -1 for missing, or one-hot encoded (B, L, K).
            y (np.ndarray): Ground truth 0/1/2 matrix with -1 for missing.
            eval_mask (np.ndarray): Boolean mask indicating which genotypes to evaluate.
            objective_mode (bool): If True, suppresses verbose output.

        Returns:
            dict[str, float]: Evaluation metrics.
        """
        if model is None:
            msg = "Model passed to _evaluate_model() is not fitted. Call fit() before evaluation."
            self.logger.error(msg)
            raise NotFittedError(msg)

        pred_labels, pred_probas = self._predict(model=model, X=X, return_proba=True)

        if pred_probas is None:
            msg = "Predicted probabilities are None in _evaluate_model()."
            self.logger.error(msg)
            raise ValueError(msg)

        y_true_flat = y[eval_mask].astype(np.int8, copy=False)
        y_pred_flat = pred_labels[eval_mask].astype(np.int8, copy=False)
        y_proba_flat = pred_probas[eval_mask].astype(np.float32, copy=False)

        valid = y_true_flat >= 0
        y_true_flat = y_true_flat[valid]
        y_pred_flat = y_pred_flat[valid]
        y_proba_flat = y_proba_flat[valid]

        if y_true_flat.size == 0:
            if isinstance(self.tune_metric, str):
                return {self.tune_metric: 0.0}
            elif isinstance(self.tune_metric, (list, tuple)):
                return {m: 0.0 for m in self.tune_metric}
            else:
                msg = f"[{self.model_name}] Invalid tune_metric type: {type(self.tune_metric)}"
                self.logger.error(msg)
                raise ValueError(msg)

        # --- Hard assertions on probability shape ---
        if y_proba_flat.ndim != 2:
            msg = f"Expected y_proba_flat to be 2D (n_eval, n_classes); got shape {y_proba_flat.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        K = int(y_proba_flat.shape[1])

        if self.is_haploid_:
            # Allow either:
            #   - K==2 (already binary)
            #   - K==3 (REF/HET/ALT) and we'll collapse HET+ALT -> ALT
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
            # Binary scoring: REF=0, ALT=1 (treat any non-zero as ALT)
            y_true_flat = (y_true_flat > 0).astype(np.int8, copy=False)
            y_pred_flat = (y_pred_flat > 0).astype(np.int8, copy=False)

            K = y_proba_flat.shape[1]
            if K == 2:
                pass
            elif K == 3:
                proba_2 = np.empty((y_proba_flat.shape[0], 2), dtype=y_proba_flat.dtype)
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

        # Ensure valid probability simplex after masking/collapsing
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

    def _objective(self, trial: optuna.Trial) -> float | tuple[float, ...]:
        """Optuna objective for VAE.

        This method defines the objective function for hyperparameter tuning using Optuna. It samples hyperparameters, trains the VAE model with these parameters, and evaluates its performance on a validation set. The evaluation metric specified by ``self.tune_metric`` is returned for optimization. If training fails, the trial is pruned to keep the tuning process efficient.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float | tuple[float, ...]: Value(s) of the tuning metric(s) to be optimized.

        Raises:
            RuntimeError: If model training returns None.
            optuna.exceptions.TrialPruned: If training fails unexpectedly or is unpromising.
        """
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
                kl_beta_schedule=params["kl_beta_schedule"],
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

            self._clear_resources(model)

            if isinstance(self.tune_metric, (list, tuple)):
                # Multi-metric objective tuning
                return tuple([metrics[k] for k in self.tune_metric])
            return metrics[self.primary_metric]

        except Exception as e:
            # Unexpected failure: surface full details in logs while still
            # pruning the trial to keep sweeps moving.
            err_type = type(e).__name__
            self.logger.warning(
                f"Trial {trial.number} failed due to exception {err_type}: {e}"
            )
            self.logger.debug(traceback.format_exc())
            raise optuna.exceptions.TrialPruned(
                f"Trial {trial.number} failed due to an exception. {err_type}: {e}. Enable debug logging for full traceback."
            ) from e

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """Sample model hyperparameters; hidden sizes use BaseNNImputer helper.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            dict[str, int | float | str]: Sampled hyperparameters.
        """
        lower_bound = 2
        # 1. Enforce strict bottleneck: max size is features - 1
        # 2. Enforce hard cap: max size is 32
        # 3. Safety net: ensure upper_bound is never lower than lower_bound
        upper_bound = max(lower_bound, min(32, self.num_features_ - 1))
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

        nF: int = self.num_features_
        nC: int = self.num_classes_
        input_dim = nF * nC

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=input_dim,
            n_outputs=nC,
            n_samples=len(self.X_train_),
            n_hidden=params["num_hidden_layers"],
            latent_dim=params["latent_dim"],
            alpha=params["layer_scaling_factor"],
            schedule=params["layer_schedule"],
            min_size=max(16, 2 * int(params["latent_dim"])),
        )

        params["model_params"] = {
            "n_features": self.num_features_,
            "num_classes": nC,  # categorical head: 2 or 3
            "dropout_rate": params["dropout_rate"],
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": params["activation"],
            "kl_beta": params["kl_beta"],
        }

        return params

    def _set_best_params(self, params: dict) -> dict:
        """Update instance fields from tuned params and return model_params dict.

        Args:
            params (dict): Best hyperparameters from tuning.

        Returns:
            dict: Model parameters for building the VAE.
        """
        self.latent_dim = params["latent_dim"]
        self.dropout_rate = params["dropout_rate"]
        self.learning_rate = params["learning_rate"]
        self.l1_penalty = params["l1_penalty"]
        self.activation = params["activation"]
        self.layer_scaling_factor = params["layer_scaling_factor"]
        self.layer_schedule = params["layer_schedule"]
        self.power = params["power"]
        self.normalize = params["normalize"]
        self.inverse = params["inverse"]
        self.gamma = params["gamma"]
        self.gamma_schedule = params["gamma_schedule"]
        self.kl_beta = params["kl_beta"]
        self.kl_beta_schedule = params["kl_beta_schedule"]
        self.class_weights_ = self._class_weights_from_zygosity(
            self.y_train_,
            train_mask=self.eval_mask_train_,
            inverse=self.inverse,
            normalize=self.normalize,
            max_ratio=self.max_ratio,
            power=self.power,
        )
        nF = self.num_features_
        nC = self.num_classes_
        input_dim = nF * nC

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=input_dim,
            n_outputs=nC,
            n_samples=len(self.X_train_),
            n_hidden=params["num_hidden_layers"],
            latent_dim=params["latent_dim"],
            alpha=params["layer_scaling_factor"],
            schedule=params["layer_schedule"],
            min_size=max(16, 2 * int(params["latent_dim"])),
        )

        return {
            "n_features": nF,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": hidden_layer_sizes,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "num_classes": nC,
            "kl_beta": params["kl_beta"],
        }
