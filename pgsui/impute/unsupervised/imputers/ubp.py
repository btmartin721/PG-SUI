import copy
import warnings
from typing import TYPE_CHECKING, Any, Dict, Literal, Tuple

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.utils.logging import LoggerManager
from torch.optim.lr_scheduler import CosineAnnealingLR

from pgsui.data_processing.config import apply_dot_overrides, load_yaml_to_dataclass
from pgsui.data_processing.containers import UBPConfig
from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.models.ubp_model import UBPModel

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


def ensure_ubp_config(config: UBPConfig | dict | str | None) -> UBPConfig:
    """Return a concrete UBPConfig from dataclass, dict, YAML path, or None.

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
        return load_yaml_to_dataclass(
            config,
            UBPConfig,
            preset_builder=UBPConfig.from_preset,
        )
    if isinstance(config, dict):
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
    """UBP imputer for 0/1/2 genotypes with three-phase training.

    This imputer uses a three-phase training schedule specific to the UBP model:

    1. Pre-training: Train the model on the full dataset with a small learning rate.
    2. Fine-tuning: Train the model on the full dataset with a larger learning rate.
    3. Evaluation: Evaluate the model on the test set. Optimize latents for test set. Predict 0/1/2. Decode to IUPAC. Plot & report.
    4. Post-processing: Apply any necessary post-processing steps to the imputed genotypes.


    References:
        - Gashler, Michael S., Smith, Michael R., Morris, R., and Martinez, T. (2016) Missing Value Imputation with Unsupervised Backpropagation. Computational Intelligence, 32: 196-215. doi: 10.1111/coin.12048.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        config: UBPConfig | dict | str | None = None,
        overrides: dict | None = None,
    ):
        """Initialize the UBP imputer via dataclass/dict/YAML config with overrides.

        Args:
            genotype_data (GenotypeData): Backing genotype data object.
            config (UBPConfig | dict | str | None): UBP configuration.
            overrides (dict | None): Flat dot-key overrides applied after `config`.
        """
        self.model_name = "ImputeUBP"
        self.genotype_data = genotype_data

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
        self.logger = logman.get_logger()

        # ---- Base init ----
        super().__init__(
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

        # ---- model hyperparams ----
        self.latent_dim = self.cfg.model.latent_dim
        self.dropout_rate = self.cfg.model.dropout_rate
        self.num_hidden_layers = self.cfg.model.num_hidden_layers
        self.layer_scaling_factor = self.cfg.model.layer_scaling_factor
        self.layer_schedule = self.cfg.model.layer_schedule
        self.latent_init = self.cfg.model.latent_init
        self.activation = self.cfg.model.hidden_activation
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
        self.tune_metric = self.cfg.tune.metric
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
        self.is_haploid = None
        self.num_classes_ = None
        self.model_params: Dict[str, Any] = {}

    def fit(self) -> "ImputeUBP":
        """Fit the UBP decoder on 0/1/2 encodings (missing = -1). Three phases.

        1. Pre-training: Train the model on the full dataset with a small learning rate.
        2. Fine-tuning: Train the model on the full dataset with a larger learning rate.
        3. Evaluation: Evaluate the model on the test set. Optimize latents for test set. Predict 0/1/2. Decode to IUPAC. Plot & report.
        4. Post-processing: Apply any necessary post-processing steps to the imputed genotypes.

        Returns:
            ImputeUBP: Fitted instance.

        Raises:
            NotFittedError: If training fails.
        """
        self.logger.info(f"Fitting {self.model_name} model...")

        # --- Use 0/1/2 with -1 for missing ---
        X = self.pgenc.genotypes_012.astype(np.float32)
        X[X < 0] = np.nan
        X[np.isnan(X)] = -1
        self.ground_truth_ = X.astype(np.int64)

        # --- Determine ploidy (haploid vs diploid) and classes ---
        self.is_haploid = np.all(
            np.isin(
                self.genotype_data.snp_data, ["A", "C", "G", "T", "N", "-", ".", "?"]
            )
        )
        self.ploidy = 1 if self.is_haploid else 2

        if self.is_haploid:
            self.num_classes_ = 2
            self.ground_truth_[self.ground_truth_ == 2] = 1
            self.logger.info("Haploid data detected. Using 2 classes (REF=0, ALT=1).")
        else:
            self.num_classes_ = 3
            self.logger.info(
                "Diploid data detected. Using 3 classes (REF=0, HET=1, ALT=2)."
            )

        n_samples, self.num_features_ = X.shape

        # --- model params (decoder: Z -> L * num_classes) ---
        self.model_params = {
            "n_features": self.num_features_,
            "num_classes": self.num_classes_,
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
        self.X_train_ = self.ground_truth_[train_idx]
        self.X_test_ = self.ground_truth_[test_idx]

        # --- plotting/scorers & tuning ---
        self.plotter_, self.scorers_ = self.initialize_plotting_and_scorers()
        if self.tune:
            self.tune_hyperparameters()

        self.best_params_ = getattr(
            self, "best_params_", self._set_best_params_default()
        )

        # --- class weights for 0/1/2 ---
        self.class_weights_ = self._class_weights_from_zygosity(self.X_train_)

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
        self.plotter_.plot_history(self.history_)
        self._evaluate_model(self.X_test_, self.model_, self.best_params_)
        return self

    def transform(self) -> np.ndarray:
        """Impute missing genotypes (0/1/2) and return IUPAC strings.

        This method first checks if the model has been fitted. It then imputes the entire dataset by optimizing latent vectors for the ground truth data and predicting the missing genotypes using the trained UBP model. The imputed genotypes are decoded to IUPAC format, and distributions of original and imputed genotypes are plotted.

        Returns:
            np.ndarray: IUPAC single-character array (n_samples x L).

        Raises:
            NotFittedError: If called before fit().
        """
        if not getattr(self, "is_fit_", False):
            raise NotFittedError("Model is not fitted. Call fit() before transform().")

        self.logger.info("Imputing entire dataset with UBP (0/1/2)...")
        X_to_impute = self.ground_truth_.copy()

        optimized_latents = self._optimize_latents_for_inference(
            X_to_impute, self.model_, self.best_params_
        )
        pred_labels, _ = self._predict(self.model_, latent_vectors=optimized_latents)

        missing_mask = X_to_impute == -1
        imputed_array = X_to_impute.copy()
        imputed_array[missing_mask] = pred_labels[missing_mask]

        # Decode to IUPAC for return & plots
        imputed_genotypes = self.pgenc.decode_012(imputed_array)
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
        class_weights: torch.Tensor,
        phase: int,
    ) -> Tuple[float, torch.nn.Parameter]:
        """Single epoch over batches for UBP with 0/1/2 focal CE.

        This method handles all three UBP phases:
        1. Pre-training: Train the model on the full dataset with a small learning rate.
        2. Fine-tuning: Train the model on the full dataset with a larger learning rate.
        3. Joint training: Train both model and latents.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader (indices, y_batch).
            optimizer (torch.optim.Optimizer): Decoder optimizer.
            latent_optimizer (torch.optim.Optimizer): Latent optimizer.
            model (torch.nn.Module): UBP model with phase1_decoder & phase23_decoder.
            l1_penalty (float): L1 regularization weight.
            latent_vectors (torch.nn.Parameter): Trainable Z.
            class_weights (torch.Tensor): Class weights for 0/1/2.
            phase (int): Phase id (1, 2, 3). Phase 1 = warm-up, phase 2 = decoder-only, phase 3 = joint.

        Returns:
            Tuple[float, torch.nn.Parameter]: Average loss and updated latents.
        """
        model.train()
        running = 0.0

        for batch_indices, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            latent_optimizer.zero_grad(set_to_none=True)

            decoder = model.phase1_decoder if phase == 1 else model.phase23_decoder
            logits = decoder(latent_vectors[batch_indices]).view(
                len(batch_indices), self.num_features_, self.num_classes_
            )

            logits_flat = logits.view(-1, self.num_classes_)
            targets_flat = y_batch.view(-1)

            ce = F.cross_entropy(
                logits_flat,
                targets_flat,
                weight=class_weights,
                reduction="none",
                ignore_index=-1,
            )
            pt = torch.exp(-ce)
            gamma = getattr(model, "gamma", self.gamma)
            focal = ((1 - pt) ** gamma) * ce

            valid_mask = targets_flat != -1
            loss = (
                focal[valid_mask].mean()
                if valid_mask.any()
                else torch.tensor(0.0, device=logits.device)
            )

            if l1_penalty > 0:
                loss = loss + l1_penalty * sum(
                    p.abs().sum() for p in model.parameters()
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_([latent_vectors], 1.0)

            optimizer.step()

            if phase != 2:
                latent_optimizer.step()

            running += float(loss.item())

        return running / len(loader), latent_vectors

    def _predict(
        self, model: torch.nn.Module, latent_vectors: torch.nn.Parameter | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict 0/1/2 labels & probabilities from latents via phase23 decoder.

        Args:
            model: Trained model.
            latent_vectors: Latent vectors.

        Returns:
            (labels, probabilities).
        """
        if model is None or latent_vectors is None:
            msg = "Model and latent vectors must be provided for prediction. Fit the model first."
            self.logger.error(msg)
            raise NotFittedError(msg)

        model.eval()
        nF = getattr(model, "n_features", self.num_features_)
        with torch.no_grad():
            logits = model.phase23_decoder(latent_vectors.to(self.device)).view(
                len(latent_vectors), nF, self.num_classes_
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
    ) -> Dict[str, float]:
        """Evaluate on held-out set with 0/1/2 classes; also IUPAC/10-base reports.

        Args:
            X_val: 0/1/2 with -1 for missing.
            model: Trained model.
            params: Model params.
            objective_mode: If True, return only tuned metric.
            latent_vectors_val: Optional pre-optimized latents.

        Returns:
            Metrics dict.
        """
        if latent_vectors_val is not None:
            test_latent_vectors = latent_vectors_val
        else:
            test_latent_vectors = self._optimize_latents_for_inference(
                X_val, model, params
            )

        pred_labels, pred_probas = self._predict(
            model=model, latent_vectors=test_latent_vectors
        )

        eval_mask = X_val != -1
        y_true_flat = X_val[eval_mask]
        y_pred_flat = pred_labels[eval_mask]
        y_proba_flat = pred_probas[eval_mask]

        if y_true_flat.size == 0:
            return {self.tune_metric: 0.0}

        labels_for_scoring = [0, 1] if self.is_haploid else [0, 1, 2]
        target_names = ["REF", "ALT"] if self.is_haploid else ["REF", "HET", "ALT"]

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
            self.logger.info(f"Validation Metrics (0/1/2): {metrics}")

            self._make_class_reports(
                y_true=y_true_flat,
                y_pred_proba=y_proba_flat,
                y_pred=y_pred_flat,
                metrics=metrics,
                labels=target_names,
            )

            # IUPAC / 10-base auxiliary reports
            y_true_dec = self.pgenc.decode_012(X_val)
            X_pred = X_val.copy()
            X_pred[eval_mask] = y_pred_flat

            nF_eval = X_val.shape[1]
            y_pred_dec = self.pgenc.decode_012(X_pred.reshape(X_val.shape[0], nF_eval))

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

    def _get_data_loaders(self, y: np.ndarray) -> torch.utils.data.DataLoader:
        """Create DataLoader over indices + 0/1/2 target matrix.

        This method creates a PyTorch DataLoader for the given genotype matrix, which contains 0/1/2 encodings with -1 for missing values. The DataLoader is constructed to yield batches of data during training, where each batch consists of indices and the corresponding genotype values. The genotype matrix is converted to a PyTorch tensor and moved to the appropriate device (CPU or GPU) before being wrapped in a TensorDataset. The DataLoader is configured to shuffle the data and use the specified batch size.

        Args:
            y (np.ndarray): (n_samples x L) int matrix with -1 missing.

        Returns:
            torch.utils.data.DataLoader: Shuffled mini-batches.
        """
        y_tensor = torch.from_numpy(y).long().to(self.device)
        dataset = torch.utils.data.TensorDataset(
            torch.arange(len(y), device=self.device), y_tensor.to(self.device)
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective using the UBP training loop."""
        try:
            params = self._sample_hyperparameters(trial)

            X_train_trial = self.ground_truth_[self.train_idx_]
            X_test_trial = self.ground_truth_[self.test_idx_]

            class_weights = self._class_weights_from_zygosity(X_train_trial)
            train_loader = self._get_data_loaders(X_train_trial)

            train_latent_vectors = self._create_latent_space(
                params, len(X_train_trial), X_train_trial, params["latent_init"]
            )

            model = self.build_model(self.Model, params["model_params"])
            model.n_features = params["model_params"]["n_features"]
            model.apply(self.initialize_weights)

            _, model, _ = self._train_and_validate_model(
                model=model,
                loader=train_loader,
                lr=params["lr"],
                l1_penalty=params["l1_penalty"],
                trial=trial,
                return_history=False,
                latent_vectors=train_latent_vectors,
                lr_input_factor=params["lr_input_factor"],
                class_weights=class_weights,
                X_val=X_test_trial,
                params=params,
                prune_metric=self.tune_metric,
                prune_warmup_epochs=5,
                eval_interval=1,
                eval_requires_latents=True,
                eval_latent_steps=50,
                eval_latent_lr=params["lr"] * params["lr_input_factor"],
                eval_latent_weight_decay=0.0,
            )

            metrics = self._evaluate_model(
                X_test_trial, model, params, objective_mode=True
            )
            self._clear_resources(
                model, train_loader, latent_vectors=train_latent_vectors
            )
            return metrics[self.tune_metric]
        except Exception as e:
            raise optuna.exceptions.TrialPruned(f"Trial failed with error: {e}")

    def _sample_hyperparameters(
        self, trial: optuna.Trial
    ) -> Dict[str, int | float | str | list]:
        """Sample UBP hyperparameters; compute hidden sizes for model_params.

        This method samples a set of hyperparameters for the UBP model using the provided Optuna trial object. It defines a search space for various hyperparameters, including latent dimension, learning rate, dropout rate, number of hidden layers, activation function, and others. After sampling the hyperparameters, it computes the sizes of the hidden layers based on the sampled values and constructs the model parameters dictionary. The method returns a dictionary containing all sampled hyperparameters along with the computed model parameters.

        Args:
            trial (optuna.Trial): Current trial.

        Returns:
            Dict[str, int | float | str | list]: Sampled hyperparameters.
        """
        params = {
            "latent_dim": trial.suggest_int("latent_dim", 2, 32),
            "lr": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.6),
            "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 8),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "elu", "selu"]
            ),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "lr_input_factor": trial.suggest_float(
                "lr_input_factor", 0.1, 10.0, log=True
            ),
            "l1_penalty": trial.suggest_float("l1_penalty", 1e-7, 1e-2, log=True),
            "layer_scaling_factor": trial.suggest_float(
                "layer_scaling_factor", 2.0, 10.0
            ),
            "layer_schedule": trial.suggest_categorical(
                "layer_schedule", ["pyramid", "constant", "linear"]
            ),
            "latent_init": trial.suggest_categorical("latent_init", ["random", "pca"]),
        }

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=params["latent_dim"],
            n_outputs=self.num_features_ * self.num_classes_,
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
        """Set best params onto instance; return model_params payload.

        This method sets the best hyperparameters found during tuning onto the instance attributes of the ImputeUBP class. It extracts the relevant hyperparameters from the provided dictionary and updates the corresponding instance variables. Additionally, it computes the sizes of the hidden layers based on the best hyperparameters and constructs the model parameters dictionary. The method returns a dictionary containing the model parameters that can be used to build the UBP model.

        Args:
            best_params (Dict[str, int | float | str | list]): Best hyperparameters.

        Returns:
            Dict[str, int | float | str | list]: model_params payload.

        Raises:
            ValueError: If best_params is missing required keys.
        """
        self.latent_dim = best_params["latent_dim"]
        self.dropout_rate = best_params["dropout_rate"]
        self.learning_rate = best_params["learning_rate"]
        self.gamma = best_params["gamma"]
        self.lr_input_factor = best_params["lr_input_factor"]
        self.l1_penalty = best_params["l1_penalty"]
        self.activation = best_params["activation"]
        self.latent_init = best_params["latent_init"]

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=self.latent_dim,
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
            "gamma": self.gamma,
            "num_classes": self.num_classes_,
        }

    def _set_best_params_default(self) -> Dict[str, int | float | str | list]:
        """Default (no-tuning) model_params aligned with current attributes.

        This method constructs the model parameters dictionary using the current instance attributes of the ImputeUBP class. It computes the sizes of the hidden layers based on the instance's latent dimension, dropout rate, learning rate, and other relevant attributes. The method returns a dictionary containing the model parameters that can be used to build the UBP model when no hyperparameter tuning has been performed.

        Returns:
            Dict[str, int | float | str | list]: model_params payload.
        """
        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=self.latent_dim,
            n_outputs=self.num_features_ * self.num_classes_,
            n_samples=len(self.ground_truth_),
            n_hidden=self.num_hidden_layers,
            alpha=self.layer_scaling_factor,
            schedule=self.layer_schedule,
        )

        hidden_only = [hidden_layer_sizes[0]] + hidden_layer_sizes[1:-1]

        return {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": hidden_only,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "gamma": self.gamma,
            "num_classes": self.num_classes_,
        }

    def _train_and_validate_model(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        lr: float,
        l1_penalty: float,
        trial: optuna.Trial | None = None,
        return_history: bool = False,
        latent_vectors: torch.nn.Parameter | None = None,
        lr_input_factor: float = 1.0,
        class_weights: torch.Tensor | None = None,
        *,
        # NEW ↓↓↓
        X_val: np.ndarray | None = None,
        params: dict | None = None,
        prune_metric: str | None = None,  # "f1" | "accuracy" | "pr_macro"
        prune_warmup_epochs: int = 3,
        eval_interval: int = 1,
        eval_requires_latents: bool = True,  # UBP needs latent eval
        eval_latent_steps: int = 50,
        eval_latent_lr: float = 1e-2,
        eval_latent_weight_decay: float = 0.0,
    ) -> Tuple[float, torch.nn.Module | None, dict, torch.nn.Parameter | None]:
        """Train & validate UBP model with three-phase loop.

        This method trains and validates the UBP model using a three-phase training loop. It sets up the latent optimizer and invokes the training loop, which includes pre-training, fine-tuning, and joint training phases. The method ensures that the necessary latent vectors and class weights are provided before proceeding with training. It also incorporates new parameters for evaluation and pruning during training. The final best loss, best model, training history, and optimized latent vectors are returned.

        Args:
            model (torch.nn.Module): UBP model with phase1_decoder & phase23_decoder.
            loader (torch.utils.data.DataLoader): DataLoader for training data.
            lr (float): Learning rate for decoder.
            l1_penalty (float): L1 regularization weight.
            trial (optuna.Trial | None): Current trial or None.
            return_history (bool): If True, return loss history.
            latent_vectors (torch.nn.Parameter | None): Trainable Z.
            lr_input_factor (float): LR factor for latents.
            class_weights (torch.Tensor | None): Class weights for 0/1/2.
            X_val (np.ndarray | None): Validation set for pruning/eval.
            params (dict | None): Model params for eval.
            prune_metric (str | None): Metric to monitor for pruning.
            prune_warmup_epochs (int): Epochs before pruning starts.
            eval_interval (int): Epochs between evaluations.
            eval_requires_latents (bool): If True, optimize latents for eval.
            eval_latent_steps (int): Latent optimization steps for eval.
            eval_latent_lr (float): Latent optimization LR for eval.
            eval_latent_weight_decay (float): Latent optimization weight decay for eval.

        Returns:
            Tuple[float, torch.nn.Module, dict, torch.nn.Parameter]: (best_loss, best_model, history, latents).

        Raises:
            TypeError: If latent_vectors or class_weights are
                not provided.
            ValueError: If X_val is not provided for evaluation.
            RuntimeError: If eval_latent_steps is not positive.
        """
        if latent_vectors is None or class_weights is None:
            msg = "Must provide latent_vectors and class_weights."
            self.logger.error(msg)
            raise TypeError(msg)

        latent_optimizer = torch.optim.Adam([latent_vectors], lr=lr * lr_input_factor)

        result = self._execute_training_loop(
            loader=loader,
            latent_optimizer=latent_optimizer,
            lr=lr,
            model=model,
            l1_penalty=l1_penalty,
            trial=trial,
            return_history=return_history,
            latent_vectors=latent_vectors,
            class_weights=class_weights,
            # NEW ↓↓↓
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
            return result

        return result[0], result[1], result[3]

    def _train_final_model(
        self,
        loader: torch.utils.data.DataLoader,
        best_params: Dict[str, int | float | str | list],
        initial_latent_vectors: torch.nn.Parameter,
    ) -> Tuple[float, torch.nn.Module, dict, torch.nn.Parameter]:
        """Train final UBP model with best params; save weights to disk.

        Args:
            loader: DataLoader for training data.
            best_params: Best hyperparameters.
            initial_latent_vectors: Initialized latent vectors.

        Returns:
            (loss, model, {"Train": history}, latents).
        """
        self.logger.info("Training the final UBP (0/1/2) model...")

        model = self.build_model(self.Model, best_params)
        model.n_features = best_params["n_features"]
        model.apply(self.initialize_weights)

        loss, trained_model, history, latent_vectors = self._train_and_validate_model(
            model=model,
            loader=loader,
            lr=self.learning_rate,
            l1_penalty=self.l1_penalty,
            return_history=True,
            latent_vectors=initial_latent_vectors,
            lr_input_factor=self.lr_input_factor,
            class_weights=self.class_weights_,
            X_val=self.X_test_,
            params=best_params,
            prune_metric=self.tune_metric,
            prune_warmup_epochs=5,
            eval_interval=1,
            eval_requires_latents=True,
            eval_latent_steps=50,
            eval_latent_lr=self.learning_rate * self.lr_input_factor,
            eval_latent_weight_decay=0.0,
        )

        if trained_model is None:
            msg = "Final model training failed."
            self.logger.error(msg)
            raise RuntimeError(msg)

        fout = self.models_dir / "final_model.pt"
        torch.save(trained_model.state_dict(), fout)
        return loss, trained_model, {"Train": history}, latent_vectors

    def _execute_training_loop(
        self,
        loader: torch.utils.data.DataLoader,
        latent_optimizer: torch.optim.Optimizer,
        lr: float,
        model: torch.nn.Module,
        l1_penalty: float,
        trial,
        return_history: bool,
        latent_vectors: torch.nn.Parameter,
        class_weights: torch.Tensor,
        *,
        X_val: np.ndarray | None = None,
        params: dict | None = None,
        prune_metric: str | None = None,
        prune_warmup_epochs: int = 3,
        eval_interval: int = 1,
        eval_requires_latents: bool = True,
        eval_latent_steps: int = 50,
        eval_latent_lr: float = 1e-2,
        eval_latent_weight_decay: float = 0.0,
    ) -> Tuple[float, torch.nn.Module, dict, torch.nn.Parameter]:
        """Three-phase UBP loop with cosine LR, gamma warmup, and pruning hook."""
        history: dict[str, list[float]] = {}
        final_best_loss = float("inf")
        final_best_model = None

        # Schema-aware latent cache for eval
        _latent_cache: dict = {}
        nF = getattr(model, "n_features", self.num_features_)
        cache_key_root = f"{self.prefix}_ubp_val_latents_L{nF}_K{self.num_classes_}"

        # Epoch budget; if you later add tune_fast behavior to UBP, wire it here
        max_epochs = self.epochs
        warm, ramp, gamma_final = 50, 100, self.gamma

        for phase in (1, 2, 3):
            early_stopping = EarlyStopping(
                patience=self.early_stop_gen,
                min_epochs=self.min_epochs,
                verbose=self.verbose,
                prefix=self.prefix,
                debug=self.debug,
            )

            if phase == 2:
                self._reset_weights(model)

            decoder_params = (
                model.phase1_decoder.parameters()
                if phase == 1
                else model.phase23_decoder.parameters()
            )
            optimizer = torch.optim.Adam(decoder_params, lr=lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

            phase_hist: list[float] = []

            for epoch in range(max_epochs):
                # Focal gamma warmup
                if epoch < warm:
                    model.gamma = 0.0
                elif epoch < warm + ramp:
                    model.gamma = gamma_final * ((epoch - warm) / ramp)
                else:
                    model.gamma = gamma_final

                train_loss, latent_vectors = self._train_step(
                    loader=loader,
                    optimizer=optimizer,
                    latent_optimizer=latent_optimizer,
                    model=model,
                    l1_penalty=l1_penalty,
                    latent_vectors=latent_vectors,
                    class_weights=class_weights,
                    phase=phase,
                )

                if trial and (np.isnan(train_loss) or np.isinf(train_loss)):
                    raise optuna.exceptions.TrialPruned("Loss is NaN or Inf.")

                scheduler.step()
                if return_history:
                    phase_hist.append(train_loss)

                early_stopping(train_loss, model)
                if early_stopping.early_stop:
                    self.logger.info(
                        f"Early stopping at epoch {epoch + 1} (phase {phase})."
                    )
                    break

                # Validation pruning hook
                if (
                    trial is not None
                    and X_val is not None
                    and ((epoch + 1) % eval_interval == 0)
                ):
                    metric_key = prune_metric or getattr(self, "tune_metric", "f1")
                    z = self._first_linear_in_features(model)
                    schema_key = f"{cache_key_root}_z{z}"

                    metric_val = self._eval_for_pruning(
                        model=model,
                        X_val=X_val,
                        params=params or getattr(self, "best_params_", {}),
                        metric=metric_key,
                        objective_mode=True,
                        do_latent_infer=eval_requires_latents,
                        latent_steps=eval_latent_steps,
                        latent_lr=eval_latent_lr,
                        latent_weight_decay=eval_latent_weight_decay,
                        latent_seed=(self.seed if self.seed is not None else 123),
                        _latent_cache=_latent_cache,
                        _latent_cache_key=schema_key,
                    )

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        trial.report(metric_val, step=epoch + 1)

                    if (epoch + 1) >= prune_warmup_epochs and trial.should_prune():
                        raise optuna.exceptions.TrialPruned(
                            f"Pruned at epoch {epoch + 1} (phase {phase}): "
                            f"{metric_key}={metric_val:.5f}"
                        )

            history[f"Phase {phase}"] = phase_hist
            final_best_loss = early_stopping.best_score
            final_best_model = copy.deepcopy(early_stopping.best_model)

        return final_best_loss, final_best_model, history, latent_vectors

    def _optimize_latents_for_inference(
        self,
        X_new: np.ndarray,
        model: torch.nn.Module,
        params: dict,
        inference_epochs: int = 200,
    ) -> torch.Tensor:
        """Optimize latent vectors for new 0/1/2 data by minimizing masked CE.

        Args:
            X_new: 0/1/2 with -1 for missing.
            model: Trained model.
            params: Should include 'latent_dim'.
            inference_epochs: Steps for optimization.

        Returns:
            Optimized latent vectors.
        """
        model.eval()

        nF = getattr(model, "n_features", self.num_features_)

        X_new = X_new.astype(np.int64, copy=False)
        X_new[X_new < 0] = -1

        # Allow shorter inference when tune_fast is enabled, mirroring NLPCA
        if self.tune and self.tune_fast:
            inference_epochs = min(
                inference_epochs, getattr(self, "tune_infer_epochs", 20)
            )

        new_latent_vectors = self._create_latent_space(
            params, len(X_new), X_new, self.latent_init
        )
        opt = torch.optim.Adam(
            [new_latent_vectors], lr=self.learning_rate * self.lr_input_factor
        )
        y_target = torch.from_numpy(X_new).long().to(self.device)

        for _ in range(inference_epochs):
            opt.zero_grad(set_to_none=True)
            logits = model.phase23_decoder(new_latent_vectors).view(
                len(X_new), nF, self.num_classes_
            )
            loss = F.cross_entropy(
                logits.view(-1, self.num_classes_), y_target.view(-1), ignore_index=-1
            )
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(
                    "Inference loss is NaN/Inf; stopping latent refinement."
                )
                break
            loss.backward()
            opt.step()

        return new_latent_vectors.detach()

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

            # impute NaNs with per-column means (all-NaN cols -> 0.0 by the divide above)
            nan_r, nan_c = np.where(np.isnan(X_pca))
            if nan_r.size:
                X_pca[nan_r, nan_c] = col_means[nan_c]

            # center columns
            X_pca = X_pca - X_pca.mean(axis=0, keepdims=True)

            # ---- guard: degenerate / all-zero after centering -> fall back to random ----
            if (not np.isfinite(X_pca).all()) or np.allclose(X_pca, 0.0):
                latents = torch.empty(n_samples, latent_dim, device=self.device)
                torch.nn.init.xavier_uniform_(latents)
                return torch.nn.Parameter(latents, requires_grad=True)

            # rank-aware component count, at least 1
            try:
                est_rank = np.linalg.matrix_rank(X_pca)
            except Exception:
                est_rank = min(n_samples, X_pca.shape[1])
            n_components = max(1, min(latent_dim, est_rank, n_samples, X_pca.shape[1]))

            # use deterministic SVD to avoid power-iteration warnings
            pca = PCA(
                n_components=n_components, svd_solver="full", random_state=self.seed
            )
            initial = pca.fit_transform(X_pca)  # (n_samples, n_components)

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

        This method targets only the `phase23_decoder` attribute of the UBPModel,
        leaving the `phase1_decoder` and other potential model components untouched.
        This allows the model to be re-initialized for the second phase of training
        without affecting other parts.

        Args:
            model (torch.nn.Module): The PyTorch model whose parameters are to be reset.
        """
        if hasattr(model, "phase23_decoder"):
            # Iterate through only the modules of the second decoder
            for layer in model.phase23_decoder.modules():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
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
        seed: int,
        cache: dict | None,
        cache_key: str | None,
    ) -> None:
        """Freeze weights; refine validation latents only (no leakage).

        Args:
            model: Trained UBP model.
            X_val: Validation 0/1/2 with -1 for missing.
            steps: Number of optimization steps.
            lr: Learning rate for latent optimization.
            weight_decay: L2 weight decay on latents.
            seed: RNG seed for determinism across epochs.
            cache: Optional dict to warm-start & persist val latents.
            cache_key: Ignored; we build a schema-aware key internally.
        """
        if seed is None:
            seed = np.random.randint(0, 999999)
        torch.manual_seed(seed)
        np.random.seed(seed)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        nF = getattr(model, "n_features", self.num_features_)

        X_val = X_val.astype(np.int64, copy=False)
        X_val[X_val < 0] = -1
        y_target = torch.from_numpy(X_val).long().to(self.device)

        # Infer current model latent size to avoid shape mismatch
        latent_dim_model = self._first_linear_in_features(model)
        schema_key = f"{self.prefix}_ubp_val_latents_z{latent_dim_model}_L{nF}_K{self.num_classes_}"

        # Warm-start from cache if compatible
        if cache is not None and schema_key in cache:
            val_latents = cache[schema_key].detach().clone().requires_grad_(True)
        else:
            val_latents = self._create_latent_space(
                {"latent_dim": latent_dim_model},
                n_samples=X_val.shape[0],
                X=X_val,
                latent_init=self.latent_init,
            ).requires_grad_(True)

        opt = torch.optim.AdamW([val_latents], lr=lr, weight_decay=weight_decay)

        for _ in range(max(int(steps), 0)):
            opt.zero_grad(set_to_none=True)
            logits = model.phase23_decoder(val_latents).view(
                X_val.shape[0], nF, self.num_classes_
            )
            loss = F.cross_entropy(
                logits.view(-1, self.num_classes_),
                y_target.view(-1),
                ignore_index=-1,
                reduction="mean",
            )
            loss.backward()
            opt.step()

        if cache is not None:
            cache[schema_key] = val_latents.detach().clone()

        for p in model.parameters():
            p.requires_grad_(True)
