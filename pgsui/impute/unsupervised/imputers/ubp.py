import copy
import traceback
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Literal, Mapping, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.utils.logging import LoggerManager

from pgsui.data_processing.config import apply_dot_overrides, load_yaml_to_dataclass
from pgsui.data_processing.containers import UBPConfig
from pgsui.data_processing.transformers import SimMissingTransformer
from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.loss_functions import SafeFocalCELoss
from pgsui.impute.unsupervised.models.ubp_model import UBPModel
from pgsui.utils.logging_utils import configure_logger
from pgsui.utils.pretty_metrics import PrettyMetrics

if TYPE_CHECKING:
    from snpio import TreeParser
    from snpio.read_input.genotype_data import GenotypeData


def ensure_ubp_config(config: UBPConfig | Mapping[str, Any] | str | None) -> UBPConfig:
    """Return a concrete UBPConfig from dataclass, dict, YAML path, or None.

    This method normalizes the input configuration for the UBP imputer. It accepts a UBPConfig instance, a dictionary, a YAML file path, or None. If None is provided, it returns a default UBPConfig instance. If a YAML path is given, it loads the configuration from the file, supporting top-level presets. If a dictionary is provided, it flattens any nested structures and applies dot-key overrides to a base configuration, which can also be influenced by a preset if specified. The method ensures that the final output is a fully populated UBPConfig instance.

    Args:
        config: UBPConfig | dict | YAML path | None.

    Returns:
        UBPConfig: Normalized configuration instance.
    """
    if config is None:
        return UBPConfig()
    if isinstance(config, UBPConfig):
        return config
    if isinstance(config, str):
        # YAML path — support top-level `preset`
        return load_yaml_to_dataclass(config, UBPConfig)
    if isinstance(config, dict):
        config = copy.deepcopy(config)
        base = UBPConfig()

        def _flatten(prefix: str, d: dict, out: dict) -> dict:
            for k, v in d.items():
                kk = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(kk, v, out)
                else:
                    out[kk] = v
            return out

        preset_name = config.pop("preset", None)
        if "io" in config and isinstance(config["io"], dict):
            preset_name = preset_name or config["io"].pop("preset", None)
        if preset_name:
            base = UBPConfig.from_preset(preset_name)

        flat = _flatten("", config, {})
        return apply_dot_overrides(base, flat)

    raise TypeError("config must be a UBPConfig, dict, YAML path, or None.")


class ImputeUBP(BaseNNImputer):
    """UBP imputer for 0/1/2 genotypes with a three-phase decoder schedule.

    This imputer follows the training recipe from Unsupervised Backpropagation:

    1. Phase 1 (joint warm start): Learn latent codes and the shallow linear decoder together.
    2. Phase 2 (deep decoder reset): Reinitialize the deeper decoder, freeze the latent codes, and train only the decoder parameters.
    3. Phase 3 (joint fine-tune): Unfreeze everything and jointly refine latent codes plus the deep decoder before evaluation/reporting.

    References:
        - Gashler, Michael S., Smith, Michael R., Morris, R., and Martinez, T. (2016) Missing Value Imputation with Unsupervised Backpropagation. Computational Intelligence, 32: 196-215. doi: 10.1111/coin.12048.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        tree_parser: Optional["TreeParser"] = None,
        config: UBPConfig | dict | str | None = None,
        overrides: dict[str, Any] | None = None,
        simulate_missing: bool = False,
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
        sim_kwargs: Mapping[str, Any] | None = None,
    ):
        """Initialize the UBP imputer via dataclass/dict/YAML config with overrides.

        This constructor allows for flexible initialization of the UBP imputer by accepting various forms of configuration input. It ensures that the configuration is properly normalized and any specified overrides are applied. The method also sets up logging and initializes various attributes related to the model, training, tuning, and evaluation based on the provided configuration.

        Args:
            genotype_data (GenotypeData): Backing genotype data object.
            tree_parser: "TreeParser" | None = None, Optional SNPio phylogenetic tree parser for nonrandom sim_strategy modes.
            config (UBPConfig | dict | str | None): UBP configuration.
            overrides (dict[str, Any] | None): Flat dot-key overrides applied after `config`.
            simulate_missing (bool): Whether to simulate missing data during training.
            sim_strategy (Literal[...] | None): Simulated missing strategy if simulating.
            sim_prop (float | None): Proportion of data to simulate as missing if simulating.
            sim_kwargs (dict | None): Additional kwargs for SimMissingTransformer.
        """
        self.model_name = "ImputeUBP"
        self.genotype_data = genotype_data
        self.tree_parser = tree_parser

        # ---- normalize config, then apply overrides ----
        cfg = ensure_ubp_config(config)
        if overrides:
            cfg = apply_dot_overrides(cfg, overrides)
        self.cfg = cfg

        # ---- logging ----
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

        # ---- Base init ----
        super().__init__(
            model_name=self.model_name,
            genotype_data=self.genotype_data,
            prefix=self.cfg.io.prefix,
            device=self.cfg.train.device,
            verbose=self.cfg.io.verbose,
            debug=self.cfg.io.debug,
        )

        # ---- model/meta ----
        self.Model = UBPModel
        self.pgenc = GenotypeEncoder(genotype_data)

        self.seed = self.cfg.io.seed
        self.n_jobs = self.cfg.io.n_jobs
        self.prefix = self.cfg.io.prefix
        self.scoring_averaging = self.cfg.io.scoring_averaging
        self.verbose = self.cfg.io.verbose
        self.debug = self.cfg.io.debug
        self.rng = np.random.default_rng(self.seed)

        # Simulated-missing controls (config defaults w/ overrides)
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
        self.simulate_missing = simulate_missing or default_sim_flag
        self.sim_strategy = sim_strategy or default_strategy
        self.sim_prop = float(sim_prop if sim_prop is not None else default_prop)
        self.sim_kwargs = sim_cfg_kwargs

        # ---- model hyperparams ----
        self.latent_dim = self.cfg.model.latent_dim
        self.dropout_rate = self.cfg.model.dropout_rate
        self.num_hidden_layers = self.cfg.model.num_hidden_layers
        self.layer_scaling_factor = self.cfg.model.layer_scaling_factor
        self.layer_schedule = self.cfg.model.layer_schedule
        self.latent_init: Literal["pca", "random"] = self.cfg.model.latent_init
        self.activation = self.cfg.model.activation
        self.gamma = self.cfg.model.gamma

        # ---- training ----
        self.batch_size = self.cfg.train.batch_size
        self.learning_rate = self.cfg.train.learning_rate
        self.lr_input_factor = self.cfg.train.lr_input_factor
        self.l1_penalty = self.cfg.train.l1_penalty
        self.early_stop_gen = self.cfg.train.early_stop_gen
        self.min_epochs = self.cfg.train.min_epochs
        self.epochs = self.cfg.train.max_epochs
        self.validation_split = self.cfg.train.validation_split
        self.beta = self.cfg.train.weights_beta
        self.max_ratio = self.cfg.train.weights_max_ratio

        # ---- tuning ----
        self.tune = self.cfg.tune.enabled
        self.tune_fast = self.cfg.tune.fast
        self.tune_proxy_metric_batch = self.cfg.tune.proxy_metric_batch
        self.tune_batch_size = self.cfg.tune.batch_size
        self.tune_epochs = self.cfg.tune.epochs
        self.tune_eval_interval = self.cfg.tune.eval_interval
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
        self.tune_infer_epochs = getattr(self.cfg.tune, "infer_epochs", 100)
        self.tune_patience = self.cfg.tune.patience

        # ---- evaluation ----
        self.eval_latent_steps = self.cfg.evaluate.eval_latent_steps
        self.eval_latent_lr = self.cfg.evaluate.eval_latent_lr
        self.eval_latent_weight_decay = self.cfg.evaluate.eval_latent_weight_decay

        # ---- plotting ----
        self.plot_format = self.cfg.plot.fmt
        self.plot_dpi = self.cfg.plot.dpi
        self.plot_fontsize = self.cfg.plot.fontsize
        self.title_fontsize = self.cfg.plot.fontsize
        self.despine = self.cfg.plot.despine
        self.show_plots = self.cfg.plot.show

        # ---- core runtime ----
        self.is_haploid = False
        self.num_classes_: int | None = None
        self.model_params: Dict[str, Any] = {}
        self.sim_mask_global_: np.ndarray | None = None
        self.sim_mask_train_: np.ndarray | None = None
        self.sim_mask_test_: np.ndarray | None = None

        if self.tree_parser is None and self.sim_strategy.startswith("nonrandom"):
            msg = "tree_parser is required for nonrandom and nonrandom_weighted simulated missing strategies."
            self.logger.error(msg)
            raise ValueError(msg)

    def fit(self) -> "ImputeUBP":
        """Fit the UBP decoder on 0/1/2 encodings (missing = -1) via three phases.

        1. Phase 1 initializes latent vectors alongside the linear decoder.
        2. Phase 2 resets and trains the deeper decoder while latents remain fixed.
        3. Phase 3 jointly fine-tunes latents plus the deep decoder before evaluation.

        Returns:
            ImputeUBP: Fitted instance.

        Raises:
            NotFittedError: If training fails.
        """
        self.logger.info(f"Fitting {self.model_name} model...")

        # --- Use 0/1/2 with -1 for missing ---
        X012 = self._get_float_genotypes(copy=True)
        GT_full = np.nan_to_num(X012, nan=-1.0, copy=True)
        self.ground_truth_ = GT_full.astype(np.int64, copy=False)

        cache_key = self._sim_mask_cache_key()
        self.sim_mask_global_ = None
        if self.simulate_missing:
            cached_mask = (
                None if cache_key is None else self._sim_mask_cache.get(cache_key)
            )
            if cached_mask is not None:
                self.sim_mask_global_ = cached_mask.copy()
            else:
                X_for_sim = self.ground_truth_.astype(np.float32, copy=True)
                X_for_sim[X_for_sim < 0] = -9.0
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
                tr.fit(X_for_sim.copy())
                self.sim_mask_global_ = tr.sim_missing_mask_.astype(bool)
                orig_missing = self.ground_truth_ == -1
                self.sim_mask_global_ = self.sim_mask_global_ & (~orig_missing)

                if cache_key is not None and self.sim_mask_global_ is not None:
                    self._sim_mask_cache[cache_key] = self.sim_mask_global_.copy()

        X_for_model = self.ground_truth_.copy()
        if self.sim_mask_global_ is not None:
            X_for_model[self.sim_mask_global_] = -1

        # --- Determine ploidy (haploid vs diploid) and classes ---
        self.ploidy = self.cfg.io.ploidy
        self.is_haploid = self.ploidy == 1

        if self.is_haploid:
            self.num_classes_ = 2
            self.ground_truth_[self.ground_truth_ == 2] = 1
            X_for_model[X_for_model == 2] = 1
            self.logger.info("Haploid data detected. Using 2 classes (REF=0, ALT=1).")
        else:
            self.num_classes_ = 3
            self.logger.info(
                "Diploid data detected. Using 3 classes (REF=0, HET=1, ALT=2) for training/scoring."
            )

        self.X_model_input_ = X_for_model

        # IMPORTANT: model head matches scoring classes now
        self.output_classes_ = int(self.num_classes_)

        n_samples, self.num_features_ = X_for_model.shape

        # --- model params (decoder: Z -> L * num_classes) ---
        self.model_params = {
            "n_features": self.num_features_,
            "num_classes": self.output_classes_,
            "latent_dim": self.latent_dim,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            # hidden_layer_sizes injected later
        }

        # --- split ---
        indices = np.arange(n_samples)
        train_idx, test_idx = train_test_split(
            indices, test_size=self.validation_split, random_state=self.seed
        )
        self.train_idx_, self.test_idx_ = train_idx, test_idx
        self.X_train_ = X_for_model[train_idx]
        self.X_test_ = X_for_model[test_idx]
        self.GT_train_full_ = self.ground_truth_[train_idx]
        self.GT_test_full_ = self.ground_truth_[test_idx]

        if self.sim_mask_global_ is not None:
            self.sim_mask_train_ = self.sim_mask_global_[train_idx]
            self.sim_mask_test_ = self.sim_mask_global_[test_idx]
        else:
            self.sim_mask_train_ = None
            self.sim_mask_test_ = None

        # --- plotting/scorers & tuning ---
        self.plotter_, self.scorers_ = self.initialize_plotting_and_scorers()

        self.class_weights_ = self._normalize_class_weights(
            self._class_weights_from_zygosity(self.X_train_)
        )

        if self.tune:
            self.tuned_params_ = self.tune_hyperparameters()

        # Fall back to default model params when none have been selected yet.
        if not getattr(self, "best_params_", None):
            self.best_params_ = self._set_best_params_default()

        # --- latent init & loader ---
        train_latent_vectors = self._create_latent_space(
            self.best_params_, len(self.X_train_), self.X_train_, self.latent_init
        )
        train_loader = self._get_data_loaders(self.X_train_)

        # --- final training (three-phase under the hood) ---
        (self.best_loss_, self.model_, self.history_, self.train_latent_vectors_) = (
            self._train_final_model(
                loader=train_loader,
                best_params=self.best_params_,
                initial_latent_vectors=train_latent_vectors,
            )
        )

        self.is_fit_ = True

        if self.show_plots:
            self.plotter_.plot_history(self.history_)

        eval_mask = (
            self.sim_mask_test_
            if (self.simulate_missing and self.sim_mask_test_ is not None)
            else None
        )
        self._evaluate_model(
            self.X_test_,
            self.model_,
            self.best_params_,
            eval_mask_override=eval_mask,
        )

        if self.tune:
            self._save_best_params(self.best_params_)
        else:
            self._save_best_params(self.best_params_)

        return self

    def transform(self) -> np.ndarray:
        """Impute missing genotypes (0/1/2) and return IUPAC strings.

        This method first checks if the model has been fitted. It then imputes the entire dataset by optimizing latent vectors for the ground truth data and predicting the missing genotypes using the trained UBP model. The imputed genotypes are decoded to IUPAC format, and genotype distributions are plotted only when ``self.show_plots`` is enabled.

        Returns:
            np.ndarray: IUPAC single-character array (n_samples x L).

        Raises:
            NotFittedError: If called before fit().
        """
        if not getattr(self, "is_fit_", False):
            msg = "Model is not fitted. Call fit() before transform()."
            self.logger.error(msg)
            raise NotFittedError(msg)

        self.logger.info(f"Imputing entire dataset with {self.model_name}...")
        X_to_impute = self.ground_truth_.copy()

        optimized_latents = self._optimize_latents_for_inference(
            X_to_impute, self.model_, self.best_params_
        )

        if not isinstance(optimized_latents, torch.nn.Parameter):
            optimized_latents = torch.nn.Parameter(
                optimized_latents, requires_grad=False
            )

        pred_labels, _ = self._predict(self.model_, latent_vectors=optimized_latents)

        missing_mask = X_to_impute == -1
        imputed_array = X_to_impute.copy()
        imputed_array[missing_mask] = pred_labels[missing_mask]

        # Decode to IUPAC for return and optional plots
        imputed_genotypes = self.pgenc.decode_012(imputed_array)

        if self.show_plots:
            original_genotypes = self.pgenc.decode_012(X_to_impute)
            self.plotter_.plot_gt_distribution(original_genotypes, is_imputed=False)
            self.plotter_.plot_gt_distribution(imputed_genotypes, is_imputed=True)

        return imputed_genotypes

    def _train_step(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        latent_optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        l1_penalty: float,
        latent_vectors: torch.nn.Parameter,
        phase: int,
        criterion: torch.nn.Module,
    ) -> Tuple[float, torch.nn.Parameter]:
        """One epoch with stable focal CE, grad clipping, and NaN guards.

        Args:
            loader (torch.utils.data.DataLoader): Training data loader.
            optimizer (torch.optim.Optimizer): Model optimizer.
            latent_optimizer (torch.optim.Optimizer): Latent vectors optimizer.
            model (torch.nn.Module): UBP model.
            l1_penalty (float): L1 regularization penalty.
            latent_vectors (torch.nn.Parameter): Latent vectors for samples.
            phase (int): Training phase (1, 2, or 3).
            criterion (torch.nn.Module): Loss function.

        Returns:
            Tuple[float, torch.nn.Parameter]: Mean loss and updated latents.
        """
        model.train()
        running, used = 0.0, 0

        # Keep target width in sync with model output
        # width to avoid silent shape mismatches
        # that surface later as mask errors.
        nF_model = int(getattr(model, "n_features", self.num_features_))

        decoder: torch.Tensor | torch.nn.Module = (
            model.phase1_decoder if phase == 1 else model.phase23_decoder
        )

        if not isinstance(decoder, torch.nn.Module):
            msg = f"{self.model_name} Decoder is not a torch.nn.Module."
            self.logger.error(msg)
            raise TypeError(msg)

        for batch_indices, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            latent_optimizer.zero_grad(set_to_none=True)

            batch_indices = batch_indices.to(latent_vectors.device, non_blocking=True)
            z = latent_vectors[batch_indices]
            y = y_batch.to(self.device, non_blocking=True).long()

            if y.dim() != 2:
                msg = f"Training batch expected 2D targets, got shape {tuple(y.shape)}."
                self.logger.error(msg)
                raise ValueError(msg)

            if y.shape[1] != nF_model:
                msg = f"Model expects {nF_model} loci but batch has {y.shape[1]}. Ensure tuning subsets and masks use the same loci columns."
                self.logger.error(msg)
                raise ValueError(msg)

            logits = decoder(z).view(len(batch_indices), nF_model, self.output_classes_)

            # Guard upstream explosions
            if not torch.isfinite(logits).all():
                self.logger.debug("Non-finite logits during training step.")
                continue

            # Unified focal CE for both haploid (K=2) and diploid (K=3)
            loss = criterion(
                logits.view(-1, self.output_classes_),
                y.view(-1),
            )

            if l1_penalty > 0.0:
                # Pick what you actually want to regularize:
                # - decoder.parameters() is typical
                # - latent_vectors is also possible, but do that explicitly
                l1 = torch.zeros((), device=self.device)
                for p in decoder.parameters():
                    l1 = l1 + p.abs().sum()
                loss = loss + (float(l1_penalty) * l1)

            if not torch.isfinite(loss):
                self.logger.debug("Non-finite loss during training step.")
                continue

            loss.backward()

            # Clip returns the Total Norm
            model_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            latent_norm = torch.nn.utils.clip_grad_norm_([latent_vectors], 1.0)

            # Skip update on non-finite grads
            # Check norms instead of iterating all parameters
            if torch.isfinite(model_norm) and torch.isfinite(latent_norm):
                optimizer.step()

                if phase != 2:
                    latent_optimizer.step()

            running += float(loss.detach().item())
            used += 1

        return (running / used if used > 0 else float("inf")), latent_vectors

    def _predict(
        self,
        model: torch.nn.Module,
        latent_vectors: Optional[torch.nn.Parameter | torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict 0/1/2 labels & probabilities from latents via phase23 decoder. This method requires a trained model and latent vectors.

        Args:
            model (torch.nn.Module): Trained model.
            latent_vectors (torch.nn.Parameter | None): Latent vectors.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted labels and probabilities.
        """
        if model is None or latent_vectors is None:
            msg = "Model and latent vectors must be provided for prediction. Fit the model first."
            self.logger.error(msg)
            raise NotFittedError(msg)

        model.eval()
        nF = getattr(model, "n_features", self.num_features_)
        with torch.no_grad():
            decoder = model.phase23_decoder

            if not isinstance(decoder, torch.nn.Module):
                msg = f"{self.model_name} decoder is not a valid torch.nn.Module."
                self.logger.error(msg)
                raise TypeError(msg)

            logits = decoder(latent_vectors.to(self.device)).view(
                len(latent_vectors), nF, self.output_classes_
            )
            probas = torch.softmax(logits, dim=-1)
            labels = torch.argmax(probas, dim=-1)

        return labels.cpu().numpy(), probas.cpu().numpy()

    def _evaluate_model(
        self,
        X_val: np.ndarray,
        model: torch.nn.Module,
        params: dict,
        objective_mode: bool = False,
        latent_vectors_val: torch.Tensor | None = None,
        *,
        eval_mask_override: np.ndarray | None = None,
        GT_val: np.ndarray | None = None,
    ) -> Dict[str, float]:
        """Evaluate UBP model on a validation set with explicit ground truth support.

        This method evaluates the trained UBP model on `X_val` by (optionally) optimizing
        latent vectors, predicting genotypes, and computing performance metrics. For
        simulated-missing evaluation, you should pass `eval_mask_override` indicating the
        entries to score, and provide `GT_val` containing the unmasked ground truth.

        Args:
            X_val (np.ndarray): Validation data in 0/1/2 encoding with -1 for missing.
                This may include simulated missingness (set to -1) if applicable.
            model (torch.nn.Module): Trained UBP model.
            params (dict): Model parameters (used for latent inference, if needed).
            objective_mode (bool): If True, suppresses printing/reporting.
            latent_vectors_val (torch.Tensor | None): Pre-optimized latents (optional).
            eval_mask_override (np.ndarray | None): Boolean mask specifying which entries
                to evaluate (e.g., simulated-missing mask). Must match `X_val` rows; columns
                may be >= `X_val` columns (will be sliced).
            GT_val (np.ndarray | None): Ground-truth matrix aligned to `X_val`, containing
                the true genotypes *before* simulated masking. Strongly recommended during
                tuning and simulated-missing evaluation.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        # --- latent vectors ---
        if latent_vectors_val is not None:
            test_latent_vectors = latent_vectors_val
        else:
            test_latent_vectors = self._optimize_latents_for_inference(
                X_val, model, params
            )

        pred_labels, pred_probas = self._predict(
            model=model, latent_vectors=test_latent_vectors
        )

        # --- evaluation mask ---
        if eval_mask_override is not None:
            if eval_mask_override.shape[0] != X_val.shape[0]:
                msg = (
                    f"eval_mask_override rows {eval_mask_override.shape[0]} "
                    f"does not match X_val rows {X_val.shape[0]}"
                )
                self.logger.error(msg)
                raise ValueError(msg)

            if eval_mask_override.shape[1] < X_val.shape[1]:
                msg = (
                    f"eval_mask_override cols {eval_mask_override.shape[1]} "
                    f"is smaller than X_val cols {X_val.shape[1]}"
                )
                self.logger.error(msg)
                raise ValueError(msg)

            eval_mask = eval_mask_override[:, : X_val.shape[1]].astype(bool, copy=False)
        else:
            # Default: score only observed entries
            eval_mask = X_val != -1

        # --- ground truth selection (NO heuristic fallback to X_val when mask override is used) ---
        GT_ref: np.ndarray | None = None

        if GT_val is not None:
            GT_ref = GT_val
        else:
            # Only use stored splits when the shape matches exactly.
            if (
                getattr(self, "GT_test_full_", None) is not None
                and getattr(self, "X_test_", None) is not None
            ):
                if X_val.shape == self.X_test_.shape:
                    GT_ref = self.GT_test_full_
            if (
                GT_ref is None
                and getattr(self, "GT_train_full_", None) is not None
                and getattr(self, "X_train_", None) is not None
            ):
                if X_val.shape == self.X_train_.shape:
                    GT_ref = self.GT_train_full_
            if (
                GT_ref is None
                and getattr(self, "ground_truth_", None) is not None
                and X_val.shape == self.ground_truth_.shape
            ):
                GT_ref = self.ground_truth_

        if GT_ref is None:
            # If scoring simulated missing (mask override), ground truth is required.
            if eval_mask_override is not None:
                msg = (
                    "GT_val must be provided when eval_mask_override is used, "
                    "to avoid scoring against masked (-1) values."
                )
                self.logger.error(msg)
                raise ValueError(msg)

            # Otherwise, we can fall back to using X_val, but only for observed entries scoring.
            GT_ref = X_val

        # Ensure GT_ref aligned to X_val width
        if GT_ref.shape[0] != X_val.shape[0]:
            msg = f"GT_val rows {GT_ref.shape[0]} does not match X_val rows {X_val.shape[0]}."
            self.logger.error(msg)
            raise ValueError(msg)

        if GT_ref.shape[1] < X_val.shape[1]:
            msg = f"GT_val cols {GT_ref.shape[1]} is smaller than X_val cols {X_val.shape[1]}."
            self.logger.error(msg)
            raise ValueError(msg)

        GT_ref = GT_ref[:, : X_val.shape[1]]

        # --- flatten to evaluated entries ---
        y_true_flat = GT_ref[eval_mask]
        pred_labels_flat = pred_labels[eval_mask]
        pred_probas_flat = pred_probas[eval_mask]

        if y_true_flat.size == 0:
            return {self.tune_metric: 0.0}

        # For haploids, scoring uses [0,1]; for diploids [0,1,2]
        labels_for_scoring = [0, 1] if self.is_haploid else [0, 1, 2]
        target_names = ["REF", "ALT"] if self.is_haploid else ["REF", "HET", "ALT"]

        # --- one-hot for scorers ---
        y_true_ohe = np.eye(len(labels_for_scoring), dtype=float)[y_true_flat]

        metrics = self.scorers_.evaluate(
            y_true_flat,
            pred_labels_flat,
            y_true_ohe,
            pred_probas_flat,
            objective_mode,
            self.tune_metric,
        )

        if not objective_mode:
            if self.verbose or self.debug:
                pm = PrettyMetrics(
                    metrics, precision=3, title=f"{self.model_name} Validation Metrics"
                )
                pm.render()

            self._make_class_reports(
                y_true=y_true_flat,
                y_pred_proba=pred_probas_flat,
                y_pred=pred_labels_flat,
                metrics=metrics,
                labels=target_names,
            )

            # Use X_val dimensions for decoding/plotting
            y_true_dec = self.pgenc.decode_012(
                GT_ref.reshape(X_val.shape[0], X_val.shape[1])
            )

            X_pred = X_val.copy()
            X_pred[eval_mask] = pred_labels_flat
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

            valid_true = y_true_int[eval_mask]
            valid_true = valid_true[valid_true >= 0]  # drop N
            iupac_label_set = ["A", "C", "G", "T", "W", "R", "M", "K", "Y", "S"]

            if (
                np.intersect1d(np.unique(y_true_flat), labels_for_scoring).size == 0
                or valid_true.size == 0
            ):
                self.logger.warning(
                    "Skipped numeric confusion matrix: no y_true labels present."
                )
            else:
                self._make_class_reports(
                    y_true=valid_true,
                    y_pred=y_pred_int[eval_mask][y_true_int[eval_mask] >= 0],
                    metrics=metrics,
                    y_pred_proba=None,
                    labels=iupac_label_set,
                )

        return metrics

    def _get_data_loaders(
        self,
        y: np.ndarray,
        *,
        batch_size: int | None = None,
        shuffle: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Create DataLoader over indices + 0/1/2 target matrix.

        Args:
            y (np.ndarray): (n_samples x L) int matrix with -1 missing.
            batch_size (int | None): Optional override for loader batch size.
                If None, uses self.batch_size.
            shuffle (bool): Whether to shuffle batches.

        Returns:
            torch.utils.data.DataLoader: Mini-batches yielding (row_indices, y_batch).
        """
        bs = int(self.batch_size if batch_size is None else batch_size)
        if bs <= 0:
            self.logger.warning(
                f"Invalid batch_size={bs}. Falling back to batch_size=1."
            )
            bs = 1

        y_tensor = torch.from_numpy(y).long()
        indices = torch.arange(len(y), dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(indices, y_tensor)
        pin_memory = self.device.type == "cuda"

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective using the UBP training loop.

        Returns:
            float: Objective metric value for the current trial.

        Raises:
            optuna.exceptions.TrialPruned: If the trial is pruned due to poor performance.
            Exception: Re-raised for unexpected errors (after logging), to avoid hiding bugs.
        """
        try:
            self._prepare_tuning_artifacts()
            trial_params = self._sample_hyperparameters(trial)
            model_params = trial_params["model_params"]

            # --- choose tuning data (aligned artifacts when tune_fast) ---
            if self.tune and self.tune_fast and getattr(self, "_tune_ready", False):
                X_train_trial = self._tune_X_train
                X_test_trial = self._tune_X_test
                GT_test_trial = self._tune_GT_test
                eval_mask = getattr(self, "_tune_sim_mask_test", None)

                # Reuse the prebuilt tune loader (already uses tune_batch_size)
                train_loader = self._tune_loader
            else:
                X_train_trial = getattr(
                    self, "X_train_", self.ground_truth_[self.train_idx_]
                )
                X_test_trial = getattr(
                    self, "X_test_", self.ground_truth_[self.test_idx_]
                )

                # Ground truth for eval should be the pre-mask split if available
                GT_test_trial = getattr(self, "GT_test_full_", None)
                if GT_test_trial is None or GT_test_trial.shape != X_test_trial.shape:
                    # Fall back to using X_test_trial ONLY when not using eval_mask_override
                    GT_test_trial = X_test_trial

                eval_mask = (
                    getattr(self, "sim_mask_test_", None)
                    if self.simulate_missing
                    else None
                )
                train_loader = self._get_data_loaders(
                    X_train_trial, batch_size=self.tune_batch_size
                )

            # Always align model width to the data actually used this trial
            n_features_trial = int(X_train_trial.shape[1])
            model_params["n_features"] = n_features_trial

            hidden_layer_sizes = self._compute_hidden_layer_sizes(
                n_inputs=model_params["latent_dim"],
                n_outputs=n_features_trial * self.output_classes_,
                n_samples=len(X_train_trial),
                n_hidden=trial_params["num_hidden_layers"],
                alpha=trial_params["layer_scaling_factor"],
                schedule=trial_params["layer_schedule"],
            )
            # [latent_dim] + interior widths (exclude output width)
            hidden_only = [hidden_layer_sizes[0]] + hidden_layer_sizes[1:-1]
            model_params["hidden_layer_sizes"] = hidden_only

            # --- weights ---
            class_weights = self._normalize_class_weights(
                self._class_weights_from_zygosity(X_train_trial)
            )
            if (
                self.is_haploid
                and class_weights is not None
                and class_weights.numel() > 2
            ):
                class_weights = class_weights[:2]

            # --- latents ---
            train_latent_vectors = self._create_latent_space(
                model_params,
                len(X_train_trial),
                X_train_trial,
                trial_params["latent_init"],
            )

            # --- model ---
            model = self.build_model(self.Model, model_params)
            model.n_features = model_params["n_features"]
            model.apply(self.initialize_weights)

            # --- train & validate (pruning handled inside) ---
            res = self._train_and_validate_model(
                model=model,
                loader=train_loader,
                lr=float(trial_params["lr"]),
                l1_penalty=trial_params["l1_penalty"],
                trial=trial,
                latent_vectors=train_latent_vectors,
                lr_input_factor=trial_params["lr_input_factor"],
                class_weights=class_weights,
                X_val=X_test_trial,
                params=model_params,
                prune_metric=self.tune_metric,
                eval_interval=self.tune_eval_interval,
                eval_requires_latents=True,
                eval_latent_steps=self.eval_latent_steps,
                eval_latent_lr=self.eval_latent_lr,
                eval_latent_weight_decay=self.eval_latent_weight_decay,
            )
            model = res[1]

            # If proxy eval slice used, slice X, GT, and mask together
            if (
                self.tune
                and self.tune_fast
                and getattr(self, "_tune_ready", False)
                and getattr(self, "_tune_eval_slice", None) is not None
            ):
                sl = self._tune_eval_slice
                X_test_eval = X_test_trial[sl]
                GT_test_eval = GT_test_trial[sl]
                if eval_mask is not None:
                    eval_mask = eval_mask[sl]
            else:
                X_test_eval = X_test_trial
                GT_test_eval = GT_test_trial

            metrics = self._evaluate_model(
                X_test_eval,
                model,
                model_params,
                objective_mode=True,
                eval_mask_override=eval_mask,
                GT_val=GT_test_eval,
            )

            self._clear_resources(
                model, train_loader, latent_vectors=train_latent_vectors
            )
            return float(metrics[self.tune_metric])

        except Exception as e:
            # Unexpected failure: surface full details in logs while still
            # pruning the trial to keep sweeps moving.
            err_type = type(e).__name__
            self.logger.error(
                f"Trial {trial.number} failed due to exception {err_type}: {e}"
            )
            self.logger.debug(traceback.format_exc(), exc_info=True)
            raise optuna.exceptions.TrialPruned(
                f"Trial {trial.number} failed due to an exception. {err_type}: {e}. Enable debug logging for full traceback."
            ) from e

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        lr = trial.suggest_float("learning_rate", 3e-4, 1e-3, log=True)
        params: dict[str, Any] = {
            "latent_dim": trial.suggest_int("latent_dim", 4, 16, step=2),
            "lr": lr,  # <-- internal canonical key
            "learning_rate": lr,  # <-- optional alias if you serialize best_params
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.30, step=0.05),
            "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 6),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "elu", "selu", "leaky_relu"]
            ),
            "gamma": trial.suggest_float("gamma", 0.5, 3.0, step=0.5),
            "lr_input_factor": trial.suggest_float(
                "lr_input_factor", 0.3, 3.0, log=True
            ),
            "l1_penalty": trial.suggest_float("l1_penalty", 1e-6, 1e-3, log=True),
            "layer_scaling_factor": trial.suggest_float(
                "layer_scaling_factor", 2.0, 4.0, step=0.5
            ),
            "layer_schedule": trial.suggest_categorical(
                "layer_schedule", ["pyramid", "linear"]
            ),
            "latent_init": trial.suggest_categorical("latent_init", ["random", "pca"]),
        }
        params["model_params"] = {
            "num_classes": self.output_classes_,
            "latent_dim": params["latent_dim"],
            "dropout_rate": params["dropout_rate"],
            "activation": params["activation"],
            "gamma": params["gamma"],
        }
        return params

    def _set_best_params(self, best_params: dict) -> dict:
        """Set best params onto instance; return model_params payload.

        This method sets the best hyperparameters found during tuning onto the instance attributes of the ImputeUBP class. It extracts the relevant hyperparameters from the provided dictionary and updates the corresponding instance variables. Additionally, it computes the sizes of the hidden layers based on the best hyperparameters and constructs the model parameters dictionary. The method returns a dictionary containing the model parameters that can be used to build the UBP model.

        Args:
            best_params (dict): Best hyperparameters.

        Returns:
            dict: model_params payload.

        Raises:
            ValueError: If best_params is missing required keys.
        """
        self.latent_dim = best_params["latent_dim"]
        self.dropout_rate = best_params["dropout_rate"]
        self.gamma = best_params["gamma"]
        self.lr_input_factor = best_params["lr_input_factor"]
        self.l1_penalty = best_params["l1_penalty"]
        self.activation = best_params["activation"]
        self.latent_init = best_params["latent_init"]
        lr = (
            float(best_params["learning_rate"])
            if "learning_rate" in best_params
            else float(best_params["lr"])
        )
        self.learning_rate = lr

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=self.latent_dim,
            n_outputs=self.num_features_ * self.output_classes_,
            n_samples=len(self.train_idx_),
            n_hidden=best_params["num_hidden_layers"],
            alpha=best_params["layer_scaling_factor"],
            schedule=best_params["layer_schedule"],
        )

        # [latent_dim] + interior widths (exclude output width)
        hidden_only = [hidden_layer_sizes[0]] + hidden_layer_sizes[1:-1]

        return {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": hidden_only,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "gamma": self.gamma,
            "num_classes": self.output_classes_,
        }

    def _set_best_params_default(self) -> dict:
        """Default (no-tuning) model_params aligned with current attributes.

        This method constructs the model parameters dictionary using the current instance attributes of the ImputeUBP class. It computes the sizes of the hidden layers based on the instance's latent dimension, dropout rate, learning rate, and other relevant attributes. The method returns a dictionary containing the model parameters that can be used to build the UBP model when no hyperparameter tuning has been performed.

        Returns:
            dict: model_params payload.
        """
        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=self.latent_dim,
            n_outputs=self.num_features_ * self.output_classes_,
            n_samples=len(self.ground_truth_),
            n_hidden=self.num_hidden_layers,
            alpha=self.layer_scaling_factor,
            schedule=self.layer_schedule,
        )

        # [latent_dim] + interior widths (exclude output width)
        hidden_only = [hidden_layer_sizes[0]] + hidden_layer_sizes[1:-1]

        return {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": hidden_only,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "gamma": self.gamma,
            "num_classes": self.output_classes_,
        }

    def _train_and_validate_model(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        lr: float,
        l1_penalty: float,
        trial: optuna.Trial | None = None,
        latent_vectors: torch.nn.Parameter | torch.Tensor | None = None,
        lr_input_factor: float = 1.0,
        class_weights: torch.Tensor | None = None,
        *,
        X_val: np.ndarray | None = None,
        params: Mapping[str, Any] | None = None,
        prune_metric: str | None = None,  # "f1" | "accuracy" | "pr_macro"
        eval_interval: int = 1,
        eval_requires_latents: bool = True,  # UBP needs latent eval
        eval_latent_steps: int = 50,
        eval_latent_lr: float = 1e-2,
        eval_latent_weight_decay: float = 0.0,
    ) -> tuple:
        """Train & validate UBP model with three-phase loop.

        This method trains and validates the UBP model using a three-phase training loop. It sets up the latent optimizer and invokes the training loop, which includes pre-training, fine-tuning, and joint training phases. The method ensures that the necessary latent vectors and class weights are provided before proceeding with training. It also incorporates new parameters for evaluation and pruning during training. The final best loss, best model, training history, and optimized latent vectors are returned.

        Args:
            model (torch.nn.Module): UBP model with phase1_decoder & phase23_decoder.
            loader (torch.utils.data.DataLoader): DataLoader for training data.
            lr (float): Learning rate for decoder.
            l1_penalty (float): L1 regularization weight.
            trial (optuna.Trial | None): Current trial or None.
            latent_vectors (torch.nn.Parameter | None): Trainable Z.
            lr_input_factor (float): LR factor for latents.
            class_weights (torch.Tensor | None): Class weights for 0/1/2.
            X_val (np.ndarray | None): Validation set for pruning/eval.
            params (Mapping[str, Any] | None): Model params for eval.
            prune_metric (str | None): Metric to monitor for pruning.
            eval_interval (int): Epochs between evaluations.
            eval_requires_latents (bool): If True, optimize latents for eval.
            eval_latent_steps (int): Latent optimization steps for eval.
            eval_latent_lr (float): Latent optimization LR for eval.
            eval_latent_weight_decay (float): Latent optimization weight decay for eval.

        Returns:
            Tuple[float, torch.nn.Module, Mapping[str, Any], torch.nn.Parameter]: Tuple with (best_loss, best_model, history_or_none, latent_vectors).

        Raises:
            TypeError: If latent_vectors or class_weights are
                not provided.
            ValueError: If X_val is not provided for evaluation.
            RuntimeError: If eval_latent_steps is not positive.
        """
        if class_weights is None:
            msg = "Must provide class_weights."
            self.logger.error(msg)
            raise TypeError(msg)

        if latent_vectors is None:
            msg = "Must provide latent_vectors."
            self.logger.error(msg)
            raise TypeError(msg)

        if not isinstance(latent_vectors, torch.nn.Parameter):
            latent_vectors = torch.nn.Parameter(latent_vectors, requires_grad=True)

        # ensure correct device
        if latent_vectors.device != self.device:
            latent_vectors = torch.nn.Parameter(
                latent_vectors.to(self.device), requires_grad=True
            )

        latent_optimizer = torch.optim.Adam([latent_vectors], lr=lr * lr_input_factor)

        result = self._execute_training_loop(
            loader=loader,
            latent_optimizer=latent_optimizer,
            lr=lr,
            model=model,
            l1_penalty=l1_penalty,
            trial=trial,
            latent_vectors=latent_vectors,
            X_val=X_val,
            params=params,
            prune_metric=prune_metric,
            eval_interval=eval_interval,
            eval_requires_latents=eval_requires_latents,
            eval_latent_steps=eval_latent_steps,
            eval_latent_lr=eval_latent_lr,
            eval_latent_weight_decay=eval_latent_weight_decay,
            max_epochs=self.epochs if trial is None else self.tune_epochs,
        )

        return result

    def _train_final_model(
        self,
        loader: torch.utils.data.DataLoader,
        best_params: dict,
        initial_latent_vectors: torch.nn.Parameter,
    ) -> tuple:
        """Train final UBP model with best params; save weights to disk.

        This method trains the final UBP model using the best hyperparameters found during tuning. It builds the model with the specified parameters, initializes the weights, and invokes the training and validation process. The method saves the trained model's state dictionary to disk and returns the final loss, trained model, training history, and optimized latent vectors.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader for training data.
            best_params (Dict[str, int | float | str | list]): Best hyperparameters.
            initial_latent_vectors (torch.nn.Parameter): Initialized latent vectors.

        Returns:
            Tuple[float, torch.nn.Module, dict, torch.nn.Parameter]: (loss, model, {"Train": history["Train"], "Val": history["Val"]}, latents).
        """
        self.logger.info(f"Training the final {self.model_name} model...")

        model = self.build_model(self.Model, best_params)
        model.n_features = best_params["n_features"]
        model.apply(self.initialize_weights)

        loss, trained_model, history, latent_vectors = self._train_and_validate_model(
            model=model,
            loader=loader,
            lr=self.learning_rate,
            l1_penalty=self.l1_penalty,
            latent_vectors=initial_latent_vectors,
            lr_input_factor=self.lr_input_factor,
            class_weights=self.class_weights_,
            X_val=self.X_test_,
            params=best_params,
            prune_metric=self.tune_metric,
            eval_interval=1,
            eval_requires_latents=True,
            eval_latent_steps=self.eval_latent_steps,
            eval_latent_lr=self.eval_latent_lr,
            eval_latent_weight_decay=self.eval_latent_weight_decay,
        )

        if trained_model is None:
            msg = f"{self.model_name} training failed; no model was returned."
            self.logger.error(msg)
            raise RuntimeError(msg)

        fout = self.models_dir / "final_model.pt"
        torch.save(trained_model.state_dict(), fout)
        return loss, trained_model, history, latent_vectors

    def _execute_training_loop(
        self,
        loader: torch.utils.data.DataLoader,
        latent_optimizer: torch.optim.Optimizer,
        lr: float,
        model: torch.nn.Module,
        l1_penalty: float,
        trial: optuna.Trial | None,
        latent_vectors: torch.nn.Parameter,
        *,
        X_val: np.ndarray | None = None,
        params: dict | Mapping[str, Any] | None = None,
        prune_metric: str | None = None,
        eval_interval: int = 1,
        eval_requires_latents: bool = True,
        eval_latent_steps: int = 50,
        eval_latent_lr: float = 1e-2,
        eval_latent_weight_decay: float = 0.0,
        max_epochs: int | None = None,
    ) -> Tuple[float, torch.nn.Module, Mapping[str, Any], torch.nn.Parameter]:

        # NEW: nested history for UBP (matches your plotter’s ImputeUBP expectation)
        history: dict[str, dict[str, list[float]]] = {
            "Train": defaultdict(list),
            "Val": defaultdict(list),
        }

        final_best_loss = float("inf")
        best_state: Mapping[str, torch.Tensor] | None = None

        _latent_cache: dict = {}
        nF = int(getattr(model, "n_features", self.num_features_))
        cache_key_root = f"{self.prefix}_ubp_val_latents_L{nF}_K{self.output_classes_}"

        E = int(self.epochs) if max_epochs is None else int(max_epochs)
        phase_epochs = {
            1: max(1, int(0.15 * E)),
            2: max(1, int(0.35 * E)),
            3: max(1, E - int(0.15 * E) - int(0.35 * E)),
        }

        _eval_every = max(1, int(eval_interval))

        for phase in (1, 2, 3):
            steps_this_phase = int(phase_epochs[phase])

            early_stopping = EarlyStopping(
                patience=self.early_stop_gen,
                min_epochs=self.min_epochs,
                verbose=self.verbose,
                prefix=self.prefix,
                debug=self.debug,
            )

            if phase == 2:
                self._reset_weights(model)
                latent_vectors.requires_grad_(False)
            elif phase == 3:
                latent_vectors.requires_grad_(True)

            decoder: torch.Tensor | torch.nn.Module = (
                model.phase1_decoder if phase == 1 else model.phase23_decoder
            )
            if not isinstance(decoder, torch.nn.Module):
                raise TypeError(f"{self.model_name} Decoder is not a torch.nn.Module.")

            optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, eps=1e-7)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=steps_this_phase
            )

            gamma = self._resolve_gamma(params, model)
            cw = None
            if self.class_weights_ is not None:
                cw = self.class_weights_.to(self.device)
                cw = cw / cw.mean().clamp_min(1e-8)

            criterion = SafeFocalCELoss(gamma=gamma, weight=cw, ignore_index=-1)

            last_val_loss = float("nan")

            for epoch in range(steps_this_phase):
                train_loss, latent_vectors = self._train_step(
                    loader=loader,
                    optimizer=optimizer,
                    latent_optimizer=latent_optimizer,
                    model=model,
                    l1_penalty=l1_penalty,
                    latent_vectors=latent_vectors,
                    phase=phase,
                    criterion=criterion,
                )

                if not np.isfinite(train_loss):
                    msg = f"[{self.model_name}] Non-finite train loss (phase {phase}, epoch {epoch + 1})."
                    if trial is not None:
                        raise optuna.exceptions.TrialPruned(msg)
                    raise RuntimeError(msg)

                scheduler.step()

                # Record train loss
                history["Train"][f"Phase {phase}"].append(float(train_loss))

                # Default val loss (only filled on eval epochs)
                val_loss_epoch = float("nan")

                # --- Validation loss (and pruning metric if trial) ---
                if (X_val is not None) and (((epoch + 1) % _eval_every) == 0):
                    # Schema-aware cache key for val latents
                    zdim = int(self._first_linear_in_features(model))
                    schema_key = f"{cache_key_root}_z{zdim}"

                    # Mask + GT aligned to X_val
                    mask_override, gt_override = self._resolve_prune_eval_mask_and_gt(
                        X_val
                    )

                    # Infer latents ONCE (reused for metric + val loss)
                    latent_vectors_val = None
                    if eval_requires_latents and eval_latent_steps > 0:
                        latent_vectors_val = self._latent_infer_for_eval(
                            model=model,
                            X_val=X_val,
                            steps=int(eval_latent_steps),
                            lr=float(eval_latent_lr),
                            weight_decay=float(eval_latent_weight_decay),
                            seed=int(self.seed or 12345) + epoch,
                            cache=_latent_cache,
                            cache_key=schema_key,
                        )

                    # Compute val loss (even when not tuning)
                    if latent_vectors_val is not None:
                        val_loss_epoch = self._val_loss_from_latents(
                            z_val=latent_vectors_val,
                            X_val=X_val,
                            model=model,
                            params=params or {},
                            eval_mask_override=mask_override,
                            GT_val=gt_override,
                        )
                        last_val_loss = val_loss_epoch

                    # If tuning, compute metric + pruning (phase 3)
                    if trial is not None:
                        metric_key = prune_metric or getattr(self, "tune_metric", "f1")
                        metrics = self._evaluate_model(
                            X_val=X_val,
                            model=model,
                            params=dict(params or {}),
                            objective_mode=True,
                            latent_vectors_val=latent_vectors_val,
                            eval_mask_override=mask_override,
                            GT_val=gt_override,
                        )
                        metric_val = float(metrics.get(metric_key, 0.0))

                        if phase == 3:
                            trial.report(metric_val, step=epoch + 1)
                            if trial.should_prune():
                                msg = f"[{self.model_name}] Trial {trial.number} Median-Pruned at epoch {epoch + 1}: {metric_key}={metric_val:.3f}. This is not an error, but indicates the trial was unpromising."
                                raise optuna.exceptions.TrialPruned(msg)

                # Record val loss (NaN on non-eval epochs)
                history["Val"][f"Phase {phase}"].append(float(val_loss_epoch))

                # Early stopping: ONLY update when we actually computed a real
                # val loss
                if np.isfinite(val_loss_epoch):
                    early_stopping(val_loss_epoch, model, epoch=epoch + 1)
                else:
                    # If no validation set exists at all, do simple
                    # early-stop on train loss.
                    # If X_val exists but we're between eval points, do not
                    # touch early stopping.
                    if X_val is None:
                        early_stopping(train_loss, model, epoch=epoch + 1)

                if early_stopping.early_stop:
                    self.logger.debug(
                        f"Early stopping at epoch {epoch + 1} (phase {phase})."
                    )
                    break

            # Track best across phases
            phase_best = float(early_stopping.best_score)
            if phase_best < final_best_loss:
                final_best_loss = phase_best
                best_state = early_stopping.best_state_dict

        if best_state is None:
            msg = "Training loop failed to produce a best model state."
            self.logger.error(msg)
            raise RuntimeError(msg)

        # IMPORTANT FIX: load the GLOBAL best_state, not the last phase’s early_stopping state
        model.load_state_dict(best_state)
        best_model = model

        return final_best_loss, best_model, history, latent_vectors

    def _optimize_latents_for_inference(
        self,
        X_new: np.ndarray,
        model: torch.nn.Module,
        params: dict,
        inference_epochs: int = 200,
    ) -> torch.Tensor:
        """Optimize latents for new 0/1/2 data with guards.

        This method optimizes the latent vectors for new genotype data using the trained UBP model. It initializes the latent space based on the provided data and iteratively updates the latent vectors to minimize the cross-entropy loss between the model's predictions and the true genotype values. The optimization process includes numeric stability guards to ensure that gradients and losses remain finite. The optimized latent vectors are returned as a PyTorch tensor.

        Args:
            X_new (np.ndarray): New 0/1/2 data with -1 for missing.
            model (torch.nn.Module): Trained UBP model.
            params (dict): Model params.
            inference_epochs (int): Number of optimization epochs.

        Returns:
            torch.Tensor: Optimized latent vectors.
        """
        model.eval()
        nF = getattr(model, "n_features", self.num_features_)

        if self.tune and self.tune_fast:
            inference_epochs = min(
                inference_epochs, getattr(self, "tune_infer_epochs", 20)
            )

        X_new = X_new.astype(np.int64, copy=False)
        X_new[X_new < 0] = -1
        y = torch.from_numpy(X_new).long().to(self.device)

        z = self._create_latent_space(
            params, len(X_new), X_new, self.latent_init
        ).requires_grad_(True)
        opt = torch.optim.AdamW(
            [z],
            lr=self.learning_rate * params.get("lr_input_factor", self.lr_input_factor),
            eps=1e-7,
        )

        gamma = self._resolve_gamma(params, model)

        cw = None
        cw = getattr(self, "class_weights_", None)
        if cw is not None:
            cw = cw.to(self.device)
            cw = cw / cw.mean().clamp_min(1e-8)

        criterion = SafeFocalCELoss(gamma=gamma, weight=cw, ignore_index=-1)

        for _ in range(inference_epochs):
            decoder = model.phase23_decoder

            if not isinstance(decoder, torch.nn.Module):
                msg = f"{self.model_name} Decoder is not a torch.nn.Module."
                self.logger.error(msg)
                raise TypeError(msg)

            opt.zero_grad(set_to_none=True)

            logits = decoder(z).view(len(X_new), nF, self.output_classes_)

            if not torch.isfinite(logits).all():
                self.logger.debug("Non-finite logits during latent optimization.")
                break

            logits_flat = logits.view(-1, self.output_classes_)
            targets_flat = y.view(-1)
            valid = targets_flat != -1

            loss = criterion(logits_flat[valid], targets_flat[valid])

            if not torch.isfinite(loss):
                self.logger.debug("Non-finite loss during latent optimization.")
                break

            loss.backward()

            torch.nn.utils.clip_grad_norm_([z], 1.0)

            if z.grad is None or not torch.isfinite(z.grad).all():
                self.logger.debug("Non-finite latent gradients during optimization.")
                break

            opt.step()

        return z.detach()

    def _create_latent_space(
        self,
        params: dict,
        n_samples: int,
        X: np.ndarray,
        latent_init: Literal["random", "pca"],
    ) -> torch.nn.Parameter:
        """Initialize latent space via random Xavier or PCA on 0/1/2 matrix.

        This method initializes the latent space for the UBP model using either random Xavier initialization or PCA-based initialization. The choice of initialization strategy is determined by the latent_init parameter. If PCA is selected, the method handles missing values by imputing them with column means before performing PCA. The resulting latent vectors are standardized and converted to a PyTorch parameter that can be optimized during training.

        Args:
            params (dict): Contains 'latent_dim'.
            n_samples (int): Number of samples.
            X (np.ndarray): (n_samples x L) 0/1/2 with -1 missing.
            latent_init (Literal["random","pca"]): Init strategy.

        Returns:
            torch.nn.Parameter: Trainable latent matrix.
        """
        latent_dim = int(params["latent_dim"])

        if latent_init == "pca":
            X_pca = X.astype(np.float32, copy=True)

            # mark missing
            X_pca[X_pca < 0] = np.nan

            # ---- SAFE column means without warnings ----
            valid_counts = np.sum(~np.isnan(X_pca), axis=0)
            col_sums = np.nansum(X_pca, axis=0)
            col_means = np.divide(
                col_sums,
                valid_counts,
                out=np.zeros_like(col_sums, dtype=np.float32),
                where=valid_counts > 0,
            )

            # impute NaNs with per-column means
            # (all-NaN cols -> 0.0 by the divide above)
            nan_r, nan_c = np.where(np.isnan(X_pca))

            if nan_r.size:
                X_pca[nan_r, nan_c] = col_means[nan_c]

            # center columns
            X_pca = X_pca - X_pca.mean(axis=0, keepdims=True)

            # guard: degenerate / all-zero after centering ->
            # fall back to random
            if (not np.isfinite(X_pca).all()) or np.allclose(X_pca, 0.0):
                self.logger.debug(
                    "Degenerate or non-finite PCA input; falling back to random initialization."
                )
                latents = torch.empty(n_samples, latent_dim, device=self.device)
                torch.nn.init.xavier_uniform_(latents)
                return torch.nn.Parameter(latents, requires_grad=True)

            # rank-aware component count, at least 1
            try:
                est_rank = np.linalg.matrix_rank(X_pca)
            except Exception:
                est_rank = min(n_samples, X_pca.shape[1])

            # use deterministic SVD to avoid power-iteration warnings
            n_components = min(latent_dim, n_samples - 1, self.num_features_)
            pca = PCA(
                n_components=n_components,
                svd_solver="randomized",
                random_state=self.seed,
            )

            k = min(20000, X_pca.shape[1])
            cols = self.rng.choice(X_pca.shape[1], size=k, replace=False)
            X_pca_subset = X_pca[:, cols]

            # (n_samples, n_components)
            initial = pca.fit_transform(X_pca_subset)

            # pad if latent_dim > n_components
            if n_components < latent_dim:
                pad = self.rng.standard_normal(
                    size=(n_samples, latent_dim - n_components)
                )
                initial = np.hstack([initial, pad])

            # standardize latent dims
            initial = (initial - initial.mean(axis=0)) / (initial.std(axis=0) + 1e-6)

            latents = torch.from_numpy(initial).float().to(self.device)
            return torch.nn.Parameter(latents, requires_grad=True)

        else:
            latents = torch.empty(n_samples, latent_dim, device=self.device)
            torch.nn.init.xavier_uniform_(latents)
            return torch.nn.Parameter(latents, requires_grad=True)

    def _reset_weights(self, model: torch.nn.Module) -> None:
        """Selectively resets only the weights of the phase 2/3 decoder.

        This method targets only the `phase23_decoder` attribute of the UBPModel, leaving the `phase1_decoder` and other potential model components untouched. This allows the model to be re-initialized for the second phase of training without affecting other parts.

        Args:
            model (torch.nn.Module): The PyTorch model whose parameters are to be reset.
        """
        if hasattr(model, "phase23_decoder"):
            decoder = model.phase23_decoder
            if not isinstance(decoder, torch.nn.Module):
                msg = f"{self.model_name} phase23_decoder is not a torch.nn.Module."
                self.logger.error(msg)
                raise TypeError(msg)
            # Iterate through only the modules of the second decoder
            for layer in decoder.modules():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()  # type: ignore[attr-defined]
        else:
            self.logger.warning(
                "Model does not have a 'phase23_decoder' attribute; skipping weight reset."
            )

    def _latent_infer_for_eval(
        self,
        model: torch.nn.Module,
        X_val: np.ndarray,
        *,
        steps: int,
        lr: float,
        weight_decay: float,
        seed: int | None,
        cache: dict | None,
        cache_key: str | None,
    ) -> torch.Tensor:
        """Freeze network; refine validation latents only with guards.

        This method refines latent vectors for the validation dataset using the trained
        UBP model. Model parameters are frozen during this phase; only the latent
        vectors are optimized. If `cache` is provided, optimized latents are stored
        under a schema-aware key and reused on subsequent calls.

        Args:
            model: Trained UBP model.
            X_val: Validation set in 0/1/2 encoding with -1 for missing.
            steps: Number of latent-optimization steps.
            lr: Learning rate for latent optimization.
            weight_decay: Weight decay for latent optimization.
            seed: Random seed for reproducibility. If None, a random seed is used.
            cache: Optional dict cache for latents.
            cache_key: Optional explicit cache key. If None, a schema-aware key is generated.

        Returns:
            torch.Tensor: Optimized latent vectors on `self.device`, detached.
        """
        if seed is None:
            seed = int(self.rng.integers(0, 1_000_000))

        torch.manual_seed(seed)
        np.random.seed(seed)

        model.eval()
        # Freeze model params
        for p in model.parameters():
            p.requires_grad_(False)

        gamma = self._resolve_gamma(None, model)

        cw = getattr(self, "class_weights_", None)
        if cw is not None:
            cw = cw.to(self.device)
            cw = cw / cw.mean().clamp_min(1e-8)

        criterion = SafeFocalCELoss(gamma=gamma, weight=cw, ignore_index=-1)

        try:
            nF = int(getattr(model, "n_features", self.num_features_))

            Xv = X_val.astype(np.int64, copy=False)
            Xv[Xv < 0] = -1
            y = torch.from_numpy(Xv).long().to(self.device)

            zdim = int(self._first_linear_in_features(model))
            schema_key = (
                cache_key
                or f"{self.prefix}_ubp_val_latents_z{zdim}_L{nF}_K{self.output_classes_}"
            )

            # Initialize from cache if present; else create fresh latents
            if cache is not None and schema_key in cache:
                z0 = cache[schema_key]
                z = z0.detach().clone().to(self.device).requires_grad_(True)
            else:
                z = self._create_latent_space(
                    {"latent_dim": zdim}, Xv.shape[0], Xv, self.latent_init
                ).requires_grad_(True)

            opt = torch.optim.AdamW(
                [z], lr=float(lr), weight_decay=float(weight_decay), eps=1e-7
            )

            n_steps = max(int(steps), 0)
            for _ in range(n_steps):
                opt.zero_grad(set_to_none=True)

                decoder = model.phase23_decoder
                if not isinstance(decoder, torch.nn.Module):
                    msg = f"{self.model_name} Decoder is not a torch.nn.Module."
                    self.logger.error(msg)
                    raise TypeError(msg)

                logits = decoder(z).view(Xv.shape[0], nF, self.output_classes_)

                if not torch.isfinite(logits).all():
                    self.logger.debug("Non-finite logits during latent optimization.")
                    break

                logits_flat = logits.view(-1, self.output_classes_)
                targets_flat = y.view(-1)
                valid = targets_flat != -1

                loss = criterion(logits_flat[valid], targets_flat[valid])

                if not torch.isfinite(loss):
                    self.logger.debug("Non-finite loss during latent inference.")
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_([z], 1.0)

                if z.grad is None or not torch.isfinite(z.grad).all():
                    self.logger.debug("Non-finite latent gradients during inference.")
                    break

                opt.step()

            z_out = z.detach()

            # Update cache (store detached copy on device)
            if cache is not None:
                cache[schema_key] = z_out.clone()

            return z_out

        finally:
            # Unfreeze model params
            for p in model.parameters():
                p.requires_grad_(True)

    def _resolve_prune_eval_mask_and_gt(
        self,
        X_val: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Resolve (eval_mask_override, GT_val) aligned to X_val for pruning/eval.

        Returns:
            (mask, gt) where:
            - mask is boolean array with shape (n_rows, >=n_cols) or sliced to X_val width.
            - gt is int array with shape (n_rows, >=n_cols) or sliced to X_val width.
            If alignment cannot be guaranteed, returns (None, None) for safety.
        """
        if not self.simulate_missing:
            return None, None

        # Candidate masks to try, in priority order
        mask_candidates = [
            getattr(self, "sim_mask_test_", None),
            getattr(self, "_tune_sim_mask_test", None),
        ]

        # Candidate GT matrices to try, in priority order
        gt_candidates = [
            getattr(self, "GT_test_full_", None),
            getattr(self, "_tune_GT_test", None),
            getattr(self, "ground_truth_", None),
        ]

        mask = None
        for m in mask_candidates:
            if m is None:
                continue
            if m.shape[0] == X_val.shape[0] and m.shape[1] >= X_val.shape[1]:
                mask = m[:, : X_val.shape[1]].astype(bool, copy=False)
                break

        if mask is None:
            return None, None

        gt = None
        for g in gt_candidates:
            if g is None:
                continue
            if g.shape[0] == X_val.shape[0] and g.shape[1] >= X_val.shape[1]:
                gt = g[:, : X_val.shape[1]]
                break

        # If we have a mask but no GT, we MUST NOT use the mask (your evaluator will error, correctly).
        if gt is None:
            return None, None

        return mask, gt

    def _eval_for_pruning(
        self,
        *,
        model: torch.nn.Module,
        X_val: np.ndarray,
        params: Mapping[str, Any],
        metric: str,
        objective_mode: bool,
        do_latent_infer: bool,
        latent_steps: int,
        latent_lr: float,
        latent_weight_decay: float,
        latent_seed: int | None,
        _latent_cache: dict | None,
        _latent_cache_key: str | None,
        eval_mask_override: np.ndarray | None = None,
        GT_val: np.ndarray | None = None,
    ) -> float:
        """Pruning evaluation that supports eval_mask_override + GT_val.

        This is intentionally a thin wrapper around _latent_infer_for_eval + _evaluate_model.
        """
        latent_vectors_val = None
        if do_latent_infer:
            latent_vectors_val = self._latent_infer_for_eval(
                model=model,
                X_val=X_val,
                steps=int(latent_steps),
                lr=float(latent_lr),
                weight_decay=float(latent_weight_decay),
                seed=latent_seed,
                cache=_latent_cache,
                cache_key=_latent_cache_key,
            )

        metrics = self._evaluate_model(
            X_val=X_val,
            model=model,
            params=dict(params),
            objective_mode=objective_mode,
            latent_vectors_val=latent_vectors_val,
            eval_mask_override=eval_mask_override,
            GT_val=GT_val,
        )
        return float(metrics.get(metric, 0.0))

    def _val_loss_from_latents(
        self,
        *,
        z_val: torch.Tensor,
        X_val: np.ndarray,
        model: torch.nn.Module,
        params: Mapping[str, Any] | None = None,
        eval_mask_override: np.ndarray | None = None,
        GT_val: np.ndarray | None = None,
    ) -> float:
        """Compute focal CE validation loss given pre-optimized validation latents.

        Loss is computed on the evaluation mask. If eval_mask_override is provided, loss is computed ONLY on those entries and requires GT_val to avoid using masked (-1) targets. Otherwise, loss is computed on observed entries (X_val != -1).

        Args:
            z_val: Validation latents (n_val x zdim).
            X_val: Validation genotype matrix (n_val x L) with -1 for missing.
            model: Trained UBP model.
            params: Model params (for gamma/lr_factor resolution).
            eval_mask_override: Optional boolean mask specifying evaluated entries.
            GT_val: Ground-truth matrix aligned to X_val (required if mask override is used).

        Returns:
            float: Validation loss (np.nan if no valid entries).
        """
        model.eval()

        nF = int(getattr(model, "n_features", X_val.shape[1]))
        K = int(self.output_classes_)

        if eval_mask_override is not None and GT_val is None:
            msg = (
                "GT_val must be provided when eval_mask_override is used for val loss."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Resolve mask
        if eval_mask_override is not None:
            if eval_mask_override.shape[0] != X_val.shape[0]:
                raise ValueError("eval_mask_override row count does not match X_val.")
            if eval_mask_override.shape[1] < X_val.shape[1]:
                raise ValueError("eval_mask_override has fewer cols than X_val.")
            mask = eval_mask_override[:, : X_val.shape[1]].astype(bool, copy=False)
        else:
            mask = X_val != -1

        # Resolve targets
        if GT_val is not None:
            if GT_val.shape[0] != X_val.shape[0]:
                raise ValueError("GT_val row count does not match X_val.")
            if GT_val.shape[1] < X_val.shape[1]:
                raise ValueError("GT_val has fewer cols than X_val.")
            GT = GT_val[:, : X_val.shape[1]]
        else:
            GT = X_val

        # Build criterion consistent with training
        gamma = self._resolve_gamma(params, model)

        cw = getattr(self, "class_weights_", None)
        if cw is not None:
            cw = cw.to(self.device)
            cw = cw / cw.mean().clamp_min(1e-8)

        criterion = SafeFocalCELoss(gamma=gamma, weight=cw, ignore_index=-1)

        with torch.no_grad():
            decoder = model.phase23_decoder
            if not isinstance(decoder, torch.nn.Module):
                raise TypeError(f"{self.model_name} Decoder is not a torch.nn.Module.")

            logits = decoder(z_val.to(self.device)).view(X_val.shape[0], nF, K)

            y = torch.from_numpy(GT.astype(np.int64, copy=False)).to(self.device)
            logits_flat = logits.view(-1, K)
            targets_flat = y.view(-1)

            mask_flat = torch.from_numpy(mask.reshape(-1)).to(self.device)
            valid = mask_flat & (targets_flat != -1)

            if valid.sum().item() == 0:
                return float("nan")

            loss = criterion(logits_flat[valid], targets_flat[valid])
            return float(loss.detach().item())
