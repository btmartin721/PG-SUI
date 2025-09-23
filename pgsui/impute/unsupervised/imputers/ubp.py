import copy
from typing import TYPE_CHECKING, Dict, Literal, Tuple

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

from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.models.ubp_model import UBPModel

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


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
        seed: int | None = None,
        n_jobs: int = 1,
        prefix: str = "pgsui",
        verbose: bool = False,
        weights_beta: float = 0.9999,
        weights_max_ratio: float = 1.0,
        tune: bool = False,
        tune_metric: Literal["f1", "accuracy", "pr_macro"] = "f1",
        tune_n_trials: int = 100,
        model_latent_init: Literal["random", "pca"] = "random",
        model_validation_split: float = 0.2,
        model_latent_dim: int = 2,
        model_dropout_rate: float = 0.2,
        model_num_hidden_layers: int = 2,
        model_batch_size: int = 32,
        model_learning_rate: float = 0.001,
        model_lr_input_factor: float = 1.0,
        model_early_stop_gen: int = 25,
        model_min_epochs: int = 100,
        model_epochs: int = 5000,
        model_l1_penalty: float = 0.0,
        model_layer_scaling_factor: float = 5.0,
        model_layer_schedule: Literal["pyramid", "constant", "linear"] = "pyramid",
        model_gamma: float = 2.0,
        model_hidden_activation: Literal["relu", "elu", "selu", "leaky_relu"] = "relu",
        model_device: Literal["gpu", "cpu", "mps"] = "cpu",
        plot_format: Literal["pdf", "png", "jpg", "jpeg"] = "pdf",
        plot_fontsize: int = 18,
        plot_despine: bool = True,
        plot_dpi: int = 300,
        plot_show_plots: bool = False,
        debug: bool = False,
    ):
        """Initialize the UBP imputer (0/1/2 pipeline; three-phase training).

        This imputer uses a three-phase training schedule specific to the UBP model:

        1. Pre-training: Train the model on the full dataset with a small learning rate.
        2. Fine-tuning: Train the model on the full dataset with a larger learning rate.
        3. Evaluation: Evaluate the model on the test set. Optimize latents for test set. Predict 0/1/2. Decode to IUPAC. Plot & report.
        4. Post-processing: Apply any necessary post-processing steps to the imputed genotypes.

        Args:
            genotype_data (GenotypeData): Backing genotype data object.
            seed (int | None): Random seed. If None, use random seed.
            n_jobs (int): Number of parallel jobs. If -1, use all available cores.
            prefix (str): Output prefix.
            verbose (bool): Verbose logging.
            weights_beta (float): Beta for class-balanced weights.
            weights_max_ratio (float): Clamp ratio for weights.
            tune (bool): Optuna tuning toggle.
            tune_metric (Literal["f1","accuracy","pr_macro"]): Metric to optimize.
            tune_n_trials (int): Number of trials.
            model_latent_init (Literal["random","pca"]): Latent init.
            model_validation_split (float): Validation fraction.
            model_latent_dim (int): Latent dim.
            model_dropout_rate (float): Dropout rate.
            model_num_hidden_layers (int): # hidden layers.
            model_batch_size (int): Batch size.
            model_learning_rate (float): Learning rate.
            model_lr_input_factor (float): LR factor for latents.
            model_early_stop_gen (int): Early stop patience.
            model_min_epochs (int): Minimum epochs before early stop.
            model_epochs (int): Max epochs.
            model_l1_penalty (float): L1 regularization.
            model_layer_scaling_factor (float): Hidden width scaling.
            model_layer_schedule (Literal["pyramid","constant","linear"]): Hidden schedule.
            model_gamma (float): Focal-loss gamma.
            model_hidden_activation (Literal["relu","elu","selu","leaky_relu"]): Activation.
            model_device (Literal["gpu","cpu","mps"]): Device.
            plot_format (Literal["pdf","png","jpg"]): Plot format.
            plot_fontsize (int): Font size.
            plot_despine (bool): Despine matplotlib plots.
            plot_dpi (int): DPI.
            plot_show_plots (bool): Show plots.
            debug (bool): Debug mode.
        """
        self.model_name = "ImputeUBP"
        kwargs = {"prefix": prefix, "debug": debug, "verbose": verbose}
        logman = LoggerManager(__name__, **kwargs)
        self.logger = logman.get_logger()

        super().__init__(
            prefix=prefix, device=model_device, verbose=verbose, debug=debug
        )

        self.genotype_data = genotype_data
        self.pgenc = GenotypeEncoder(genotype_data)
        self.Model = UBPModel
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.n_jobs = n_jobs

        # Model & training params
        self.latent_dim = model_latent_dim
        self.dropout_rate = model_dropout_rate
        self.num_hidden_layers = model_num_hidden_layers
        self.layer_scaling_factor = model_layer_scaling_factor
        self.layer_schedule = model_layer_schedule
        self.latent_init = model_latent_init
        self.batch_size = model_batch_size
        self.learning_rate = model_learning_rate
        self.lr_input_factor = model_lr_input_factor
        self.early_stop_gen = model_early_stop_gen
        self.min_epochs = model_min_epochs
        self.epochs = model_epochs
        self.l1_penalty = model_l1_penalty
        self.gamma = model_gamma
        self.activation = model_hidden_activation
        self.validation_split = model_validation_split
        self.beta = weights_beta
        self.max_ratio = weights_max_ratio

        # Tuning
        self.tune = tune
        self.tune_metric = tune_metric
        self.n_trials = tune_n_trials

        # Plotting & output
        self.prefix = prefix
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi
        self.show_plots = plot_show_plots
        self.plot_fontsize = plot_fontsize
        self.title_fontsize = plot_fontsize
        self.despine = plot_despine
        self.scoring_averaging = "weighted"

        # Core config
        self.is_haploid = None
        self.num_classes_ = None  # 2 if haploid else 3
        self.model_params = {}

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
        self.num_classes_ = 2 if self.is_haploid else 3
        self.logger.info(
            f"Data is {'haploid' if self.is_haploid else 'diploid'}. "
            f"Using {self.num_classes_} classes (0/1/2)."
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

        This method uses the trained UBP model's phase23 decoder to predict genotype labels (0/1/2) and their associated probabilities from the provided latent vectors. It ensures that the model is in evaluation mode and that both the model and latent vectors are available before making predictions.

        Args:
            model (torch.nn.Module): Trained model.
            latent_vectors (torch.nn.Parameter | None): Latent vectors.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (labels, probabilities).
        """
        if model is None or latent_vectors is None:
            msg = "Model and latent vectors must be provided for prediction. Make sure to fit the model beforehand."
            self.logger.error(msg)
            raise NotFittedError(msg)

        model.eval()
        with torch.no_grad():
            logits = model.phase23_decoder(latent_vectors.to(self.device)).view(
                len(latent_vectors), self.num_features_, self.num_classes_
            )
            probas = torch.softmax(logits, dim=-1)
            labels = torch.argmax(probas, dim=-1)

        return labels.cpu().numpy(), probas.cpu().numpy()

    def _evaluate_model(
        self,
        X_test: np.ndarray,
        model: torch.nn.Module,
        params: dict,
        objective_mode: bool = False,
    ) -> Dict[str, float]:
        """Evaluate on held-out set with 0/1/2 classes; also IUPAC/10-base reports.

        This method evaluates the trained UBP model on a held-out test set using 0/1/2 genotype classes. It optimizes latent vectors for the test set, predicts genotype labels and probabilities, and computes various evaluation metrics. If the data is haploid, it collapses the ALT class to 1 for binary classification metrics. The method also generates detailed classification reports for both the primary genotype encoding (0/1/2) and auxiliary encodings (IUPAC/10-base). The results are logged and returned as a dictionary of metrics.

        Args:
            X_test (np.ndarray): 0/1/2 with -1 for missing.
            model (torch.nn.Module): Trained model.
            params (dict): Model params.
            objective_mode (bool): If True, return only tuned metric.

        Returns:
            Dict[str, float]: Metrics dict.
        """
        test_latent_vectors = self._optimize_latents_for_inference(
            X_test, model, params
        )
        pred_labels, pred_probas = self._predict(
            model=model, latent_vectors=test_latent_vectors
        )

        eval_mask = X_test != -1
        y_true_flat = X_test[eval_mask]
        y_pred_flat = pred_labels[eval_mask]
        y_proba_flat = pred_probas[eval_mask]

        if y_true_flat.size == 0:
            return {self.tune_metric: 0.0}

        # Haploid -> collapse ALT class to 1 (REF vs ALT)
        labels_for_scoring = [0, 1] if self.is_haploid else [0, 1, 2]
        target_names = ["REF", "ALT"] if self.is_haploid else ["REF", "HET", "ALT"]

        if self.is_haploid:
            y_true_flat = y_true_flat.copy()
            y_pred_flat = y_pred_flat.copy()
            y_true_flat[y_true_flat == 2] = 1
            y_pred_flat[y_pred_flat == 2] = 1

            # adjust probas to 2-class for metrics display
            proba_2 = np.zeros((len(y_proba_flat), 2), dtype=y_proba_flat.dtype)
            proba_2[:, 0] = y_proba_flat[:, 0]
            proba_2[:, 1] = y_proba_flat[:, 2]
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
            self.logger.info(f"Validation Metrics (0/1/2): {metrics}")

            # Main (REF/HET/ALT) report
            self._make_class_reports(
                y_true=y_true_flat,
                y_pred_proba=y_proba_flat,
                y_pred=y_pred_flat,
                metrics=metrics,
                labels=target_names,
            )

            # IUPAC/10-base auxiliary reports (same as ImputeNLPCA)
            y_true_dec = self.pgenc.decode_012(X_test)
            X_pred = X_test.copy()
            X_pred[eval_mask] = y_pred_flat
            y_pred_dec = self.pgenc.decode_012(
                X_pred.reshape(X_test.shape[0], self.num_features_)
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
            torch.arange(len(y), device=self.device), y_tensor
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective using the UBP training loop.

        This method serves as the objective function for Optuna hyperparameter tuning. It samples a set of hyperparameters for the UBP model using the provided trial object, then trains and evaluates the model on a held-out test set. The training process involves creating latent vectors, building the UBP model, and training it using the sampled hyperparameters. After training, the model is evaluated on the test set to compute various metrics. The method returns the value of the specified tuning metric to be minimized. If any exception occurs during the process, the trial is pruned.

        Args:
            trial (optuna.Trial): Current trial.

        Returns:
            float: Value of tuned metric to maximize.

        Raises:
            optuna.exceptions.TrialPruned: If trial fails.
        """
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
            model.apply(self.initialize_weights)

            _, model, _ = self._train_and_validate_model(
                model=model,
                loader=train_loader,
                lr=params["lr"],
                l1_penalty=params["l1_penalty"],
                trial=trial,
                latent_vectors=train_latent_vectors,
                lr_input_factor=params["lr_input_factor"],
                class_weights=class_weights,
            )

            metrics = self._evaluate_model(
                X_test_trial, model, params, objective_mode=True
            )
            self._clear_resources(model, train_loader, train_latent_vectors)
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

        params["model_params"] = {
            "n_features": self.num_features_,
            "num_classes": self.num_classes_,
            "latent_dim": params["latent_dim"],
            "dropout_rate": params["dropout_rate"],
            "hidden_layer_sizes": hidden_layer_sizes,
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
    ) -> Tuple:
        """Run three-phase training; return best model and history.

        This method orchestrates the training of the UBP model using a three-phase approach. It first checks that the necessary latent vectors and class weights are provided. It then initializes an optimizer for the latent vectors and calls the internal training loop method to perform the actual training. The training loop handles the three phases of UBP training, including pre-training, fine-tuning, and joint training. Depending on the return_history flag, the method returns either just the best loss, best model, and latent vectors, or also includes the training history.

        Returns:
            Tuple: (best_loss, best_model, history, latents) if return_history
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

        This method trains the final UBP model using the best hyperparameters found during tuning. It builds the UBP model with the specified parameters, initializes its weights, and then trains it using the provided DataLoader and initial latent vectors. The training process involves optimizing both the model parameters and the latent vectors. After training, the method saves the trained model's state dictionary to disk and returns the final loss, trained model, training history, and optimized latent vectors.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader for training data.
            best_params (Dict[str, int | float | str | list]): Best hyperparameters.
            initial_latent_vectors (torch.nn.Parameter): Initialized latent vectors.

        Returns:
            Tuple[float, torch.nn.Module, dict, torch.nn.Parameter]: (loss, model, history, latents).
        """
        self.logger.info("Training the final UBP (0/1/2) model...")

        model = self.build_model(self.Model, best_params)
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
    ) -> Tuple[float, torch.nn.Module, dict, torch.nn.Parameter]:
        """Three-phase UBP loop with cosine LR and gamma warmup.

        This method implements the three-phase training loop for the UBP model. It iterates through each phase of training, applying early stopping based on validation loss. In phase 1, the model is pre-trained; in phase 2, the model is fine-tuned; and in phase 3, both the model and latent vectors are jointly trained. The method uses a cosine annealing learning rate scheduler and incorporates a warm-up period for the focal loss gamma parameter. It tracks the best model and loss for each phase and compiles the training history if requested. The final best loss, best model, training history, and optimized latent vectors are returned.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader for training data.
            latent_optimizer (torch.optim.Optimizer): Latent optimizer.
            lr (float): Learning rate for decoder.
            model (torch.nn.Module): UBP model with phase1_decoder & phase23_decoder.
            l1_penalty (float): L1 regularization weight.
            trial (optuna.Trial | None): Current trial or None.
            return_history (bool): If True, return loss history.
            latent_vectors (torch.nn.Parameter): Trainable Z.
            class_weights (torch.Tensor): Class weights for 0/1/2.

        Returns:
            Tuple[float, torch.nn.Module, dict, torch.nn.Parameter]: (best_loss, best_model, history, latents).
        """
        history = {}
        final_best_loss = float("inf")
        final_best_model = None

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
            scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

            phase_hist = []
            warm, ramp, gamma_final = 50, 100, self.gamma

            for epoch in range(self.epochs):
                # focal gamma warmup
                if epoch < warm:
                    model.gamma = 0.0
                elif epoch < warm + ramp:
                    model.gamma = gamma_final * ((epoch - warm) / ramp)
                else:
                    model.gamma = gamma_final

                train_loss, latent_vectors = self._train_step(
                    loader,
                    optimizer,
                    latent_optimizer,
                    model,
                    l1_penalty,
                    latent_vectors,
                    class_weights,
                    phase,
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

        This method optimizes latent vectors for new genotype data (0/1/2 with -1 for missing) using a trained UBP model. It initializes the latent vectors based on the specified initialization strategy and then performs optimization using the Adam optimizer to minimize the cross-entropy loss, focusing only on the observed (non-missing) genotypes. The optimization process runs for a specified number of epochs, and the optimized latent vectors are returned as a PyTorch tensor.

        Args:
            X_new (np.ndarray): 0/1/2 with -1 for missing.
            model (torch.nn.Module): Trained model.
            params (dict): Model params (for latent_dim).
            inference_epochs (int): Steps for optimization.

        Returns:
            torch.Tensor: Optimized latent vectors.
        """
        self.logger.info("Optimizing latent vectors for new data (UBP inference)...")
        model.eval()

        X_new = X_new.astype(np.int64, copy=False)
        X_new[X_new < 0] = -1

        new_latent_vectors = self._create_latent_space(
            params, len(X_new), X_new, self.latent_init
        )
        opt = torch.optim.Adam(
            [new_latent_vectors], lr=self.learning_rate * self.lr_input_factor
        )
        y_target = torch.from_numpy(X_new).long().to(self.device)

        for _ in range(inference_epochs):
            opt.zero_grad()
            logits = model.phase23_decoder(new_latent_vectors).view(
                len(X_new), self.num_features_, self.num_classes_
            )
            loss = F.cross_entropy(
                logits.view(-1, self.num_classes_), y_target.view(-1), ignore_index=-1
            )
            if torch.isnan(loss):
                self.logger.warning("Inference loss is NaN; stopping.")
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
            msg = "PCA initialization of latent space is not yet implemented."
            self.logger.error(msg)
            raise NotImplementedError(msg)
            X_pca = X.copy().astype(np.float32)
            X_pca[X_pca < 0] = np.nan
            col_means = np.nanmean(X_pca, axis=0)
            if np.isnan(col_means).any():
                global_mean = np.nanmean(col_means)
                col_means = np.nan_to_num(
                    col_means, nan=global_mean if not np.isnan(global_mean) else 0.0
                )
            inds = np.where(np.isnan(X_pca))
            X_pca[inds] = np.take(col_means, inds[1])

            n_components = min(latent_dim, n_samples, X_pca.shape[1])
            if n_components < latent_dim:
                self.logger.warning(
                    f"Latent dim reduced from {latent_dim} to {n_components} for PCA."
                )
            pca = PCA(n_components=n_components, random_state=self.seed)
            initial = pca.fit_transform(X_pca)

            if n_components < latent_dim:
                pad = self.rng.standard_normal(
                    size=(n_samples, latent_dim - n_components)
                )
                initial = np.hstack([initial, pad])

            initial = (initial - initial.mean(axis=0)) / (initial.std(axis=0) + 1e-6)
            latents = torch.from_numpy(initial).float().to(self.device)
        else:
            latents = torch.empty(n_samples, latent_dim, device=self.device)
            torch.nn.init.xavier_uniform_(latents)

        return torch.nn.Parameter(latents, requires_grad=True)

    def _reset_weights(self, model: torch.nn.Module) -> None:
        """Resets the parameters of a model's layers.

        This method iterates through all modules of the given model and calls the `reset_parameters` method on any layer that has it defined. This is useful for re-initializing a model before training.

        Args:
            model (torch.nn.Module): The PyTorch model whose parameters are to be reset.
        """
        for layer in model.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
