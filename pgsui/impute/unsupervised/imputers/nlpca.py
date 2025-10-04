from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Literal, Tuple

import matplotlib.pyplot as plt
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
from pgsui.data_processing.containers import NLPCAConfig
from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.models.nlpca_model import NLPCAModel

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


def ensure_nlpca_config(config: NLPCAConfig | dict | str | None) -> NLPCAConfig:
    """Return a concrete NLPCAConfig from dataclass, dict, YAML path, or None.

    Args:
        config (NLPCAConfig | dict | str | None): Structured configuration as dataclass, nested dict, YAML path, or None.

    Returns:
        NLPCAConfig: Concrete configuration instance.
    """
    if config is None:
        return NLPCAConfig()
    if isinstance(config, NLPCAConfig):
        return config
    if isinstance(config, str):
        # YAML path — top-level `preset` key is supported
        return load_yaml_to_dataclass(
            config,
            NLPCAConfig,
            preset_builder=NLPCAConfig.from_preset,
        )
    if isinstance(config, dict):
        # Flatten dict into dot-keys then overlay onto a fresh instance
        base = NLPCAConfig()

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
            base = NLPCAConfig.from_preset(preset_name)

        flat = _flatten("", config, {})
        return apply_dot_overrides(base, flat)

    raise TypeError("config must be an NLPCAConfig, dict, YAML path, or None.")


class ImputeNLPCA(BaseNNImputer):
    """Imputes missing genotypes using a Non-linear Principal Component Analysis (NLPCA) model.

    This class implements an imputer based on Non-linear Principal Component Analysis (NLPCA) using a neural network architecture. It is designed to handle genotype data encoded in 0/1/2 format, where 0 represents the reference allele, 1 represents the heterozygous genotype, and 2 represents the alternate allele. Missing genotypes should be represented as -9 or -1.

    The NLPCA model consists of an encoder-decoder architecture that learns a low-dimensional latent representation of the genotype data. The model is trained using a focal loss function to address class imbalance, and it can incorporate L1 regularization to promote sparsity in the learned representations.

    Notes:
        - Supports both haploid and diploid genotype data.
        - Configurable model architecture with options for latent dimension, dropout rate, number of hidden layers, and activation functions.
        - Hyperparameter tuning using Optuna for optimal model performance.
        - Evaluation metrics including accuracy, F1-score, precision, recall, and ROC-AUC.
        - Visualization of training history and genotype distributions.
        - Flexible configuration via dataclass, dictionary, or YAML file.

    Example:
        >>> from snpio import VCFReader
        >>> from pgsui import ImputeNLPCA
        >>> gdata = VCFReader("genotypes.vcf.gz")
        >>> imputer = ImputeNLPCA(gdata, config="nlpca_config.yaml")
        >>> imputer.fit()
        >>> imputed_genotypes = imputer.transform()
        >>> print(imputed_genotypes)
        [['A' 'G' 'C' ...],
         ['G' 'G' 'C' ...],
         ...
         ['T' 'C' 'A' ...],
         ['C' 'C' 'C' ...]]
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        config: NLPCAConfig | dict | str | None = None,
        overrides: dict | None = None,
    ):
        """Initializes the ImputeNLPCA imputer with genotype data and configuration.

        This constructor sets up the ImputeNLPCA imputer by accepting genotype data and a configuration that can be provided in various formats. It initializes logging, device settings, and model parameters based on the provided configuration.

        Args:
            genotype_data (GenotypeData): Backing genotype data.
            config (NLPCAConfig | dict | str | None): Structured configuration as dataclass, nested dict, YAML path, or None.
            overrides (dict | None): Dot-key overrides (e.g. {'model.latent_dim': 4}).
        """
        self.model_name = "ImputeNLPCA"
        self.genotype_data = genotype_data

        # Normalize config first, then apply overrides (highest precedence)
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
        self.logger = logman.get_logger()

        # Initialize BaseNNImputer with device/dirs/logging from config
        super().__init__(
            prefix=self.cfg.io.prefix,
            device=self.cfg.train.device,
            verbose=self.cfg.io.verbose,
            debug=self.cfg.io.debug,
        )

        self.Model = NLPCAModel
        self.pgenc = GenotypeEncoder(genotype_data)
        self.seed = self.cfg.io.seed
        self.n_jobs = self.cfg.io.n_jobs
        self.prefix = self.cfg.io.prefix
        self.scoring_averaging = self.cfg.io.scoring_averaging
        self.verbose = self.cfg.io.verbose
        self.debug = self.cfg.io.debug

        self.rng = np.random.default_rng(self.seed)

        # Model/train hyperparams
        self.latent_dim = self.cfg.model.latent_dim
        self.dropout_rate = self.cfg.model.dropout_rate
        self.num_hidden_layers = self.cfg.model.num_hidden_layers
        self.layer_scaling_factor = self.cfg.model.layer_scaling_factor
        self.layer_schedule = self.cfg.model.layer_schedule
        self.latent_init = self.cfg.model.latent_init
        self.activation = self.cfg.model.hidden_activation
        self.gamma = self.cfg.model.gamma

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

        # Tuning
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

        # Eval
        self.eval_latent_steps = self.cfg.evaluate.eval_latent_steps
        self.eval_latent_lr = self.cfg.evaluate.eval_latent_lr
        self.eval_latent_weight_decay = self.cfg.evaluate.eval_latent_weight_decay

        # Plotting (note: PlotConfig has 'show', not 'show_plots')
        self.plot_format = self.cfg.plot.fmt
        self.plot_dpi = self.cfg.plot.dpi
        self.plot_fontsize = self.cfg.plot.fontsize
        self.title_fontsize = self.cfg.plot.fontsize
        self.despine = self.cfg.plot.despine
        self.show_plots = self.cfg.plot.show

        # Core model config
        self.is_haploid = None
        self.num_classes_ = None
        self.model_params: Dict[str, Any] = {}

    def fit(self) -> "ImputeNLPCA":
        """Fits the NLPCA model to the 0/1/2 encoded genotype data.

        This method prepares the data, splits it into training and validation sets, initializes the model, and trains it. If hyperparameter tuning is enabled, it will perform tuning before final training. After training, it evaluates the model on a test set and generates relevant plots.

        Returns:
            ImputeNLPCA: The fitted imputer instance.
        """
        self.logger.info(f"Fitting {self.model_name} model...")

        # --- DATA PREPARATION ---
        X = self.pgenc.genotypes_012.astype(np.float32)
        X[X < 0] = np.nan  # Ensure missing are NaN
        X[np.isnan(X)] = -1  # Use -1 for missing, required by loss function
        self.ground_truth_ = X.astype(np.int64)

        # --- Determine Ploidy and Number of Classes ---
        self.is_haploid = np.all(
            np.isin(
                self.genotype_data.snp_data, ["A", "C", "G", "T", "N", "-", ".", "?"]
            )
        )

        self.ploidy = 1 if self.is_haploid else 2

        if self.is_haploid:
            self.num_classes_ = 2

            # Remap labels from {0, 2} to {0, 1}
            self.ground_truth_[self.ground_truth_ == 2] = 1
            self.logger.info("Haploid data detected. Using 2 classes (REF=0, ALT=1).")
        else:
            self.num_classes_ = 3

            self.logger.info(
                "Diploid data detected. Using 3 classes (REF=0, HET=1, ALT=2)."
            )

        n_samples, self.num_features_ = X.shape

        self.model_params = {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "gamma": self.gamma,
            "num_classes": self.num_classes_,
        }

        # --- Train/Test Split ---
        indices = np.arange(n_samples)
        train_idx, test_idx = train_test_split(
            indices, test_size=self.validation_split, random_state=self.seed
        )
        self.train_idx_, self.test_idx_ = train_idx, test_idx
        self.X_train_ = self.ground_truth_[train_idx]
        self.X_test_ = self.ground_truth_[test_idx]

        # --- Tuning & Model Setup ---
        self.plotter_, self.scorers_ = self.initialize_plotting_and_scorers()

        if self.tune:
            self.tune_hyperparameters()
            self.best_params_ = getattr(self, "best_params_", self.model_params.copy())
        else:
            self.best_params_ = self._set_best_params_default()

        # Class weights from 0/1/2 training data
        self.class_weights_ = self._class_weights_from_zygosity(self.X_train_)

        # Latent vectors for training set
        train_latent_vectors = self._create_latent_space(
            self.best_params_, len(self.X_train_), self.X_train_, self.latent_init
        )

        train_loader = self._get_data_loaders(self.X_train_)

        # Train the final model
        (self.best_loss_, self.model_, self.history_, self.train_latent_vectors_) = (
            self._train_final_model(
                train_loader, self.best_params_, train_latent_vectors
            )
        )

        self.is_fit_ = True
        self.plotter_.plot_history(self.history_)
        self._evaluate_model(self.X_test_, self.model_, self.best_params_)
        return self

    def transform(self) -> np.ndarray:
        """Imputes missing genotypes using the trained model.

        This method uses the trained NLPCA model to impute missing genotypes in the entire dataset. It optimizes latent vectors for all samples, predicts missing values, and fills them in. The imputed genotypes are returned in IUPAC string format.

        Returns:
            np.ndarray: Imputed genotypes in IUPAC string format.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not getattr(self, "is_fit_", False):
            raise NotFittedError("Model is not fitted. Call fit() before transform().")

        self.logger.info("Imputing entire dataset...")
        X_to_impute = self.ground_truth_.copy()

        # Optimize latents for the full dataset
        optimized_latents = self._optimize_latents_for_inference(
            X_to_impute, self.model_, self.best_params_
        )

        # Predict missing values
        pred_labels, _ = self._predict(self.model_, latent_vectors=optimized_latents)

        # Fill in missing values
        missing_mask = X_to_impute == -1
        imputed_array = X_to_impute.copy()
        imputed_array[missing_mask] = pred_labels[missing_mask]

        # Decode back to IUPAC strings
        imputed_genotypes = self.pgenc.decode_012(imputed_array)
        original_genotypes = self.pgenc.decode_012(X_to_impute)

        # Plot distributions
        plt.rcParams.update(self.plotter_.param_dict)  # Ensure consistent style
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
    ) -> Tuple[float, torch.nn.Parameter]:
        """Performs one epoch of training.

        This method executes a single training epoch for the NLPCA model. It processes batches of data, computes the focal loss while handling missing values, applies L1 regularization if specified, and updates both the model parameters and latent vectors using their respective optimizers.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            latent_optimizer (torch.optim.Optimizer): Optimizer for latent vectors.
            model (torch.nn.Module): The NLPCA model.
            l1_penalty (float): L1 regularization penalty.
            latent_vectors (torch.nn.Parameter): Latent vectors for samples.
            class_weights (torch.Tensor): Class weights for handling class imbalance.

        Returns:
            Tuple[float, torch.nn.Parameter]: Average training loss and updated latent vectors.
        """
        model.train()
        running_loss = 0.0

        nF = getattr(model, "n_features", self.num_features_)

        for batch_indices, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            latent_optimizer.zero_grad(set_to_none=True)

            logits = model.phase23_decoder(latent_vectors[batch_indices]).view(
                len(batch_indices), nF, self.num_classes_
            )

            # --- Simplified Focal Loss on 0/1/2 Classes ---
            logits_flat = logits.view(-1, self.num_classes_)
            targets_flat = y_batch.view(-1)

            ce_loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                weight=class_weights,
                reduction="none",
                ignore_index=-1,
            )

            pt = torch.exp(-ce_loss)
            gamma = getattr(model, "gamma", self.gamma)
            focal_loss = ((1 - pt) ** gamma) * ce_loss

            valid_mask = targets_flat != -1
            loss = focal_loss[valid_mask].mean() if valid_mask.any() else 0.0

            if l1_penalty > 0:
                loss += l1_penalty * sum(p.abs().sum() for p in model.parameters())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([latent_vectors], max_norm=1.0)
            optimizer.step()
            latent_optimizer.step()

            running_loss += loss.item()

        return running_loss / len(loader), latent_vectors

    def _predict(
        self, model: torch.nn.Module, latent_vectors: torch.nn.Parameter | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates 0/1/2 predictions from latent vectors.

        This method uses the trained NLPCA model to generate predictions from the latent vectors by passing them through the decoder. It returns both the predicted labels and their associated probabilities.

        Args:
            model (torch.nn.Module): Trained NLPCA model.
            latent_vectors (torch.nn.Parameter | None): Latent vectors for samples.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted labels and probabilities.
        """
        if model is None or latent_vectors is None:
            raise NotFittedError("Model or latent vectors not available.")

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
        """Evaluates the model on a validation set.

        This method evaluates the trained NLPCA model on a validation dataset by optimizing latent vectors for the validation samples, predicting genotypes, and computing various performance metrics. It can operate in an objective mode that suppresses logging for automated evaluations.

        Args:
            X_val (np.ndarray): Validation data in 0/1/2 encoding with -1 for missing.
            model (torch.nn.Module): Trained NLPCA model.
            params (dict): Model parameters.
            objective_mode (bool): If True, suppresses logging and reports only the metric.
            latent_vectors_val (torch.Tensor | None): Pre-optimized latent vectors for validation data.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        if latent_vectors_val is not None:
            test_latent_vectors = latent_vectors_val
        else:
            test_latent_vectors = self._optimize_latents_for_inference(
                X_val, model, params
            )

        # The rest of the function remains the same...
        pred_labels, pred_probas = self._predict(
            model=model, latent_vectors=test_latent_vectors
        )

        eval_mask = X_val != -1
        y_true_flat = X_val[eval_mask]
        pred_labels_flat = pred_labels[eval_mask]
        pred_probas_flat = pred_probas[eval_mask]

        if y_true_flat.size == 0:
            return {self.tune_metric: 0.0}

        # For haploids, remap class 2 to 1 for scoring (e.g., f1-score)
        labels_for_scoring = [0, 1] if self.is_haploid else [0, 1, 2]
        target_names = ["REF", "ALT"] if self.is_haploid else ["REF", "HET", "ALT"]

        y_true_ohe = np.eye(len(labels_for_scoring))[y_true_flat]

        metrics = self.scorers_.evaluate(
            y_true_flat,
            pred_labels_flat,
            y_true_ohe,
            pred_probas_flat,
            objective_mode,
            self.tune_metric,
        )

        if not objective_mode:
            self.logger.info(f"Validation Metrics: {metrics}")

            self._make_class_reports(
                y_true=y_true_flat,
                y_pred_proba=pred_probas_flat,
                y_pred=pred_labels_flat,
                metrics=metrics,
                labels=target_names,
            )

            y_true_dec = self.pgenc.decode_012(X_val)
            X_pred = X_val.copy()
            X_pred[eval_mask] = pred_labels_flat

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
        """Creates a PyTorch DataLoader for the 0/1/2 encoded data.

        This method constructs a DataLoader from the provided genotype data, which is expected to be in 0/1/2 encoding with -1 for missing values. The DataLoader is used for batching and shuffling the data during model training. It converts the numpy array to a PyTorch tensor and creates a TensorDataset. The DataLoader is configured with the specified batch size and shuffling enabled.

        Args:
            y (np.ndarray): 0/1/2 encoded genotype data with -1 for missing.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the dataset.
        """
        y_tensor = torch.from_numpy(y).long().to(self.device)
        dataset = torch.utils.data.TensorDataset(
            torch.arange(len(y), device=self.device), y_tensor.to(self.device)
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

    def _create_latent_space(
        self,
        params: dict,
        n_samples: int,
        X: np.ndarray,
        latent_init: Literal["random", "pca"],
    ) -> torch.nn.Parameter:
        """Initializes the latent space for the NLPCA model.

        This method initializes the latent space for the NLPCA model based on the specified initialization method. It supports two methods: 'random' initialization using Xavier uniform distribution, and 'pca' initialization which uses PCA to derive initial latent vectors from the data. The latent vectors are returned as a PyTorch Parameter, allowing them to be optimized during training.

        Args:
            params (dict): Model parameters including 'latent_dim'.
            n_samples (int): Number of samples in the dataset.
            X (np.ndarray): 0/1/2 encoded genotype data with -1 for missing.
            latent_init (str): Method to initialize latent space ('random' or 'pca').

        Returns:
            torch.nn.Parameter: Initialized latent vectors as a PyTorch Parameter.
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

        # --- Random init path (unchanged) ---
        latents = torch.empty(n_samples, latent_dim, device=self.device)
        torch.nn.init.xavier_uniform_(latents)
        return torch.nn.Parameter(latents, requires_grad=True)

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter tuning with Optuna.

        This method defines the objective function used by Optuna for hyperparameter tuning of the NLPCA model. It samples a set of hyperparameters, prepares the training and validation data, initializes the model and latent vectors, and trains the model. After training, it evaluates the model on a validation set and returns the value of the specified tuning metric.

        Args:
            trial (optuna.Trial): An Optuna trial object for hyperparameter suggestions.

        Returns:
            float: The value of the tuning metric to be minimized or maximized.
        """
        self._prepare_tuning_artifacts()
        trial_params = self._sample_hyperparameters(trial)
        model_params = dict(trial_params["model_params"])

        if self.tune and self.tune_fast:
            model_params["n_features"] = self._tune_num_features

        lr = trial_params["lr"]
        l1_penalty = trial_params["l1_penalty"]
        lr_input_fac = trial_params["lr_input_factor"]

        X_train_trial = self._tune_X_train
        X_test_trial = self._tune_X_test
        class_weights = self._tune_class_weights
        train_loader = self._tune_loader

        train_latents = self._create_latent_space(
            model_params,
            len(X_train_trial),
            X_train_trial,
            trial_params["latent_init"],
        )

        model = self.build_model(self.Model, model_params)
        model.n_features = model_params["n_features"]
        model.apply(self.initialize_weights)

        # train; pass an explicit flag that we are in tuning + whether to fix latents
        _, model, _ = self._train_and_validate_model(
            model=model,
            loader=train_loader,
            lr=lr,
            l1_penalty=l1_penalty,
            trial=trial,
            latent_vectors=train_latents,
            lr_input_factor=lr_input_fac,
            class_weights=class_weights,
            X_val=X_test_trial,
            params=model_params,
            prune_metric=self.tune_metric,
            prune_warmup_epochs=5,
            eval_interval=self.tune_eval_interval,
            eval_latent_steps=0,
            eval_latent_lr=0.0,
            eval_latent_weight_decay=0.0,
        )

        metrics = self._evaluate_model(
            X_test_trial, model, model_params, objective_mode=True
        )

        self._clear_resources(model, train_loader, latent_vectors=train_latents)
        return metrics[self.tune_metric]

    def _sample_hyperparameters(
        self, trial: optuna.Trial
    ) -> Dict[str, int | float | str | list]:
        """Samples hyperparameters for the simplified NLPCA model.

        This method defines the hyperparameter search space for the NLPCA model and samples a set of hyperparameters using the provided Optuna trial object. It computes the hidden layer sizes based on the sampled parameters and prepares the model parameters dictionary.

        Args:
            trial (optuna.Trial): An Optuna trial object for hyperparameter suggestions.

        Returns:
            Dict[str, int | float | str | list]: A dictionary of sampled hyperparameters.
        """
        params = {
            "latent_dim": trial.suggest_int("latent_dim", 2, 32),
            "lr": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.05),
            "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 16),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "elu", "selu", "leaky_relu"]
            ),
            "gamma": trial.suggest_float("gamma", 0.1, 5.0, step=0.1),
            "lr_input_factor": trial.suggest_float(
                "lr_input_factor", 0.1, 10.0, log=True
            ),
            "l1_penalty": trial.suggest_float("l1_penalty", 1e-6, 1e-3, log=True),
            "layer_scaling_factor": trial.suggest_float(
                "layer_scaling_factor", 2.0, 10.0
            ),
            "layer_schedule": trial.suggest_categorical(
                "layer_schedule", ["pyramid", "constant", "linear"]
            ),
            "latent_init": trial.suggest_categorical("latent_init", ["random", "pca"]),
        }

        use_n_features = (
            self._tune_num_features
            if (self.tune and self.tune_fast and hasattr(self, "_tune_num_features"))
            else self.num_features_
        )
        use_n_samples = (
            len(self._tune_train_idx)
            if (self.tune and self.tune_fast and hasattr(self, "_tune_train_idx"))
            else len(self.train_idx_)
        )

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=params["latent_dim"],
            n_outputs=use_n_features * self.num_classes_,
            n_samples=use_n_samples,
            n_hidden=params["num_hidden_layers"],
            alpha=params["layer_scaling_factor"],
            schedule=params["layer_schedule"],
        )

        params["model_params"] = {
            "n_features": use_n_features,
            "num_classes": self.num_classes_,
            "latent_dim": params["latent_dim"],
            "dropout_rate": params["dropout_rate"],
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": params["activation"],
            "gamma": params["gamma"],
        }

        return params

    def _set_best_params(
        self, best_params: Dict[str, int | float | str | list]
    ) -> Dict[str, int | float | str | list]:
        """Sets the best hyperparameters found during tuning.

        This method updates the model's attributes with the best hyperparameters obtained from tuning. It also computes the hidden layer sizes based on these parameters and prepares the final model parameters dictionary.

        Args:
            best_params (dict): Best hyperparameters from tuning.

        Returns:
            Dict[str, int | float | str | list]: Model parameters configured with the best hyperparameters.
        """
        self.latent_dim = best_params["latent_dim"]
        self.dropout_rate = best_params["dropout_rate"]
        self.learning_rate = best_params["learning_rate"]
        self.gamma = best_params["gamma"]
        self.lr_input_factor = best_params["lr_input_factor"]
        self.l1_penalty = best_params["l1_penalty"]
        self.activation = best_params["activation"]

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=self.latent_dim,
            n_outputs=self.num_features_ * self.num_classes_,
            n_samples=len(self.train_idx_),
            n_hidden=best_params["num_hidden_layers"],
            alpha=best_params["layer_scaling_factor"],
            schedule=best_params["layer_schedule"],
        )

        return {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": hidden_layer_sizes,
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

        return {
            "n_features": self.num_features_,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": hidden_layer_sizes,
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
        X_val: np.ndarray | None = None,
        params: dict | None = None,
        prune_metric: str | None = None,  # "f1" | "accuracy" | "pr_macro"
        prune_warmup_epochs: int = 3,
        eval_interval: int = 1,
        eval_latent_steps: int = 50,
        eval_latent_lr: float = 1e-2,
        eval_latent_weight_decay: float = 0.0,
    ) -> Tuple:
        """Trains and validates the NLPCA model.

        This method trains the provided NLPCA model using the specified training data and hyperparameters. It supports optional integration with Optuna for hyperparameter tuning and pruning based on validation performance. The method initializes optimizers for both the model parameters and latent vectors, sets up a learning rate scheduler, and executes the training loop. It can return the training history if requested.

        Args:
            model (torch.nn.Module): The NLPCA model to be trained.
            loader (torch.utils.data.DataLoader): DataLoader for training data.
            lr (float): Learning rate for the model optimizer.
            l1_penalty (float): L1 regularization penalty.
            trial (optuna.Trial | None): Optuna trial for hyperparameter tuning.
            return_history (bool): Whether to return training history.
            latent_vectors (torch.nn.Parameter | None): Latent vectors for samples.
            lr_input_factor (float): Learning rate factor for latent vectors.
            class_weights (torch.Tensor | None): Class weights for handling class imbalance.
            X_val (np.ndarray | None): Validation data for pruning.
            params (dict | None): Model parameters.
            prune_metric (str | None): Metric for pruning decisions.
            prune_warmup_epochs (int): Number of epochs before pruning starts.
            eval_interval (int): Interval (in epochs) for evaluation during training.
            eval_latent_steps (int): Steps for latent optimization during evaluation.
            eval_latent_lr (float): Learning rate for latent optimization during evaluation.
            eval_latent_weight_decay (float): Weight decay for latent optimization during evaluation.

        Returns:
            Tuple[float, torch.nn.Module, Dict[str, float], torch.nn.Parameter] | Tuple[float, torch.nn.Module, torch.nn.Parameter]: Training loss, trained model, training history (if requested), and optimized latent vectors.

        Raises:
            TypeError: If latent_vectors or class_weights are not provided.
        """

        if latent_vectors is None or class_weights is None:
            msg = "latent_vectors and class_weights must be provided."
            self.logger.error(msg)
            raise TypeError("Must provide latent_vectors and class_weights.")

        latent_optimizer = torch.optim.Adam([latent_vectors], lr=lr * lr_input_factor)

        optimizer = torch.optim.Adam(model.phase23_decoder.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        result = self._execute_training_loop(
            loader=loader,
            optimizer=optimizer,
            latent_optimizer=latent_optimizer,
            scheduler=scheduler,
            model=model,
            l1_penalty=l1_penalty,
            return_history=return_history,
            latent_vectors=latent_vectors,
            class_weights=class_weights,
            trial=trial,
            X_val=X_val,
            params=params,
            prune_metric=prune_metric,
            prune_warmup_epochs=prune_warmup_epochs,
            eval_interval=eval_interval,
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
        best_params: dict,
        initial_latent_vectors: torch.nn.Parameter,
    ) -> Tuple[float, torch.nn.Module, dict, torch.nn.Parameter]:
        """Trains the final model using the best hyperparameters.

        This method builds and trains the final NLPCA model using the best hyperparameters obtained from tuning. It initializes the model weights, trains the model on the entire training set, and saves the trained model to disk. It returns the final training loss, trained model, training history, and optimized latent vectors.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader for training data.
            best_params (dict): Best hyperparameters for the model.
            initial_latent_vectors (torch.nn.Parameter): Initial latent vectors for samples.

        Returns:
            Tuple[float, torch.nn.Module, dict, torch.nn.Parameter]: Final training loss, trained model, training history, and optimized latent vectors.
        Raises:
            RuntimeError: If model training fails.
        """
        self.logger.info(f"Training the final model...")

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
            eval_latent_steps=50,
            eval_latent_lr=self.learning_rate * self.lr_input_factor,
            eval_latent_weight_decay=0.0,
        )

        if trained_model is None:
            msg = "Final model training failed."
            self.logger.error(msg)
            raise RuntimeError(msg)

        fn = self.models_dir / "final_model.pt"
        torch.save(trained_model.state_dict(), fn)

        return (loss, trained_model, {"Train": history}, latent_vectors)

    def _execute_training_loop(
        self,
        loader,
        optimizer,
        latent_optimizer,
        scheduler,
        model,
        l1_penalty,
        return_history,
        latent_vectors,
        class_weights,
        *,
        trial: optuna.Trial | None = None,
        X_val: np.ndarray | None = None,
        params: dict | None = None,
        prune_metric: str | None = None,
        prune_warmup_epochs: int = 3,
        eval_interval: int = 1,
        eval_latent_steps: int = 50,
        eval_latent_lr: float = 1e-2,
        eval_latent_weight_decay: float = 0.0,
    ) -> Tuple[float, torch.nn.Module, list, torch.nn.Parameter]:
        """Executes the training loop with optional Optuna pruning.

        This method runs the training loop for the NLPCA model, performing multiple epochs of training. It supports optional integration with Optuna for hyperparameter tuning and pruning based on validation performance. The method tracks training history, applies early stopping, and returns the best model and training metrics.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            latent_optimizer (torch.optim.Optimizer): Optimizer for latent vectors.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            model (torch.nn.Module): The NLPCA model.
            l1_penalty (float): L1 regularization penalty.
            return_history (bool): Whether to return training history.
            latent_vectors (torch.nn.Parameter): Latent vectors for samples.
            class_weights (torch.Tensor): Class weights for
            handling class imbalance.
            trial (optuna.Trial | None): Optuna trial for hyperparameter tuning.
            X_val (np.ndarray | None): Validation data for pruning.
            params (dict | None): Model parameters.
            prune_metric (str | None): Metric to monitor for pruning.
            prune_warmup_epochs (int): Epochs to wait before pruning.
            eval_interval (int): Epoch interval for evaluation.
            eval_latent_steps (int): Steps to refine latents during eval.
            eval_latent_lr (float): Learning rate for latent refinement during eval.
            eval_latent_weight_decay (float): Weight decay for latent refinement during eval.

        Returns:
            Tuple[float, torch.nn.Module, list, torch.nn.Parameter]: Best loss, best model, training history, and optimized latent vectors.

        Raises:
            optuna.exceptions.TrialPruned: If the trial is pruned based on validation performance.
        """
        best_model = None
        train_history = []
        early_stopping = EarlyStopping(
            patience=self.early_stop_gen,
            min_epochs=self.min_epochs,
            verbose=self.verbose,
            prefix=self.prefix,
            debug=self.debug,
        )

        # compute the epoch budget used by the loop
        max_epochs = (
            self.tune_epochs if (trial is not None and self.tune_fast) else self.epochs
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

        # just above the for-epoch loop
        _latent_cache: dict = {}
        _latent_cache_key = f"{self.prefix}_{self.model_name}_val_latents"

        for epoch in range(max_epochs):
            train_loss, latent_vectors = self._train_step(
                loader,
                optimizer,
                latent_optimizer,
                model,
                l1_penalty,
                latent_vectors,
                class_weights,
            )
            scheduler.step()

            if np.isnan(train_loss) or np.isinf(train_loss):
                raise optuna.exceptions.TrialPruned("Loss is NaN or Inf.")

            if return_history:
                train_history.append(train_loss)

            if (
                trial is not None
                and X_val is not None
                and ((epoch + 1) % eval_interval == 0)
            ):
                metric_key = prune_metric or getattr(self, "tune_metric", "f1")

                do_infer = (eval_latent_steps or 0) > 0
                metric_val = self._eval_for_pruning(
                    model=model,
                    X_val=X_val,
                    params=params or getattr(self, "best_params_", {}),
                    metric=metric_key,
                    objective_mode=True,
                    do_latent_infer=do_infer,
                    latent_steps=eval_latent_steps,
                    latent_lr=eval_latent_lr,
                    latent_weight_decay=eval_latent_weight_decay,
                    latent_seed=(self.seed if self.seed is not None else None),
                    _latent_cache=_latent_cache,
                    _latent_cache_key=_latent_cache_key,
                )

                trial.report(metric_val, step=epoch + 1)
                if (epoch + 1) >= prune_warmup_epochs and trial.should_prune():
                    raise optuna.exceptions.TrialPruned(
                        f"Pruned at epoch {epoch + 1}: {metric_key}={metric_val:.3f}"
                    )

            early_stopping(train_loss, model)
            if early_stopping.early_stop:
                self.logger.info(f"Early stopping at epoch {epoch + 1}.")
                break

        best_loss = early_stopping.best_score
        best_model = model  # reuse instance
        best_model.load_state_dict(early_stopping.best_model.state_dict())
        return best_loss, best_model, train_history, latent_vectors

    def _optimize_latents_for_inference(
        self,
        X_new: np.ndarray,
        model: torch.nn.Module,
        params: dict,
        inference_epochs: int = 200,
    ) -> torch.Tensor:
        """Optimizes latent vectors for new, unseen data.

        This method optimizes latent vectors for new data samples that were not part of the training set. It initializes latent vectors and performs gradient-based optimization to minimize the reconstruction loss using the trained NLPCA model. The optimized latent vectors are returned for further predictions.

        Args:
            X_new (np.ndarray): New data in 0/1/2 encoding with -1 for missing values.
            model (torch.nn.Module): Trained NLPCA model.
            params (dict): Model parameters.
            inference_epochs (int): Number of epochs to optimize latent vectors.

        Returns:
            torch.Tensor: Optimized latent vectors for the new data.
        """
        if self.tune and self.tune_fast:
            inference_epochs = min(
                inference_epochs, getattr(self, "tune_infer_epochs", 20)
            )

        model.eval()

        nF = getattr(model, "n_features", self.num_features_)

        new_latent_vectors = self._create_latent_space(
            params, len(X_new), X_new, self.latent_init
        )
        latent_optimizer = torch.optim.Adam(
            [new_latent_vectors], lr=self.learning_rate * self.lr_input_factor
        )
        y_target = torch.from_numpy(X_new).long().to(self.device)

        for _ in range(inference_epochs):
            latent_optimizer.zero_grad()
            logits = model.phase23_decoder(new_latent_vectors).view(
                len(X_new), nF, self.num_classes_
            )
            loss = F.cross_entropy(
                logits.view(-1, self.num_classes_), y_target.view(-1), ignore_index=-1
            )
            if torch.isnan(loss):
                self.logger.warning("Inference loss is NaN; stopping.")
                break
            loss.backward()
            latent_optimizer.step()

        return new_latent_vectors.detach()

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

        This method refines latent vectors for validation data by optimizing them while keeping the model weights frozen. It uses gradient-based optimization to minimize the reconstruction loss on the validation set. The optimized latent vectors can be cached for future use.

        Args:
            model (torch.nn.Module): Trained NLPCA model.
            X_val (np.ndarray): Validation data in 0/1/2 encoding with - 1 for missing.
            steps (int): Number of optimization steps for latent refinement.
            lr (float): Learning rate for latent optimization.
            weight_decay (float): Weight decay for latent optimization.
            seed (int): Random seed for reproducibility.
            cache (dict | None): Cache for storing optimized latents.
            cache_key (str | None): Key for storing/retrieving latents in/from cache

        Returns:
            None. Updates cache in place if provided.
        """
        if seed is None:
            seed = np.random.randint(0, 999999)

        torch.manual_seed(seed)
        np.random.seed(seed)

        model.eval()

        nF = getattr(model, "n_features", self.num_features_)

        for p in model.parameters():
            p.requires_grad_(False)

        X_val = X_val.astype(np.int64, copy=False)
        X_val[X_val < 0] = -1
        y_target = torch.from_numpy(X_val).long().to(self.device)

        # Get latent_dim from the *model actually being evaluated*
        latent_dim_model = self._first_linear_in_features(model)

        # Make a cache key that is specific to this latent size (and feature schema)
        cache_key = (
            f"{self.prefix}_nlpca_val_latents_"
            f"z{latent_dim_model}_L{self.num_features_}_K{self.num_classes_}"
        )

        # Warm-start from cache if available *and* shape-compatible
        if cache is not None and cache_key in cache:
            val_latents = cache[cache_key].detach().clone().requires_grad_(True)
        else:
            val_latents = self._create_latent_space(
                {"latent_dim": latent_dim_model},  # use model's latent size
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
            cache[cache_key] = val_latents.detach().clone()

        for p in model.parameters():
            p.requires_grad_(True)
