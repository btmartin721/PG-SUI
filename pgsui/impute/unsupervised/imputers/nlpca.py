import copy
from typing import TYPE_CHECKING, Dict, Literal, Tuple

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

from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.models.nlpca_model import NLPCAModel

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


class ImputeNLPCA(BaseNNImputer):
    """Imputes missing genotypes using a Non-linear PCA model on 0/1/2 encoded data."

    This class implements a Non-linear PCA (NLPCA) imputer, specifically designed for genotype data encoded as 0/1/2. It uses a neural network architecture to learn a low-dimensional representation of the data and reconstruct missing values.
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
        """Initializes the simplified ImputeNLPCA imputer.

        This class extends the BaseNNImputer to implement a Non-linear PCA model for genotype imputation. It supports hyperparameter tuning using Optuna and provides various configuration options for model training and plotting.

        Args:
            genotype_data (GenotypeData): Genotype data object with SNP data.
            seed (int | None): Random seed for reproducibility.
            n_jobs (int): Number of parallel jobs. If -1, use all available cores.
            prefix (str): Prefix for output files.
            verbose (bool): If True, enables verbose logging.
            weights_beta (float): Beta parameter for class-balanced weights.
            weights_max_ratio (float): Max ratio for class weights to prevent extreme values.
            tune (bool): If True, performs hyperparameter tuning using Optuna.
            tune_metric (str): Metric to optimize during tuning ('f1', 'accuracy', 'pr_macro').
            tune_n_trials (int): Number of trials for hyperparameter tuning.
            model_latent_init (str): Method to initialize latent space ('random' or 'pca').
            model_validation_split (float): Fraction of data to use for validation during training.
            model_latent_dim (int): Dimensionality of the latent space.
            model_dropout_rate (float): Dropout rate for the neural network.
            model_num_hidden_layers (int): Number of hidden layers in the neural network.
            model_batch_size (int): Batch size for training.
            model_learning_rate (float): Learning rate for the optimizer.
            model_lr_input_factor (float): Factor to scale learning rate for latent vectors.
            model_early_stop_gen (int): Generations with no improvement before early stopping.
            model_min_epochs (int): Minimum number of epochs to train before considering early stopping.
            model_epochs (int): Maximum number of training epochs.
            model_l1_penalty (float): L1 regularization penalty.
            model_layer_scaling_factor (float): Scaling factor for hidden layer sizes.
            model_layer_schedule (str): Schedule for hidden layer sizes ('pyramid', 'constant', 'linear').
            model_gamma (float): Gamma parameter for focal loss.
            model_hidden_activation (str): Activation function for hidden layers ('relu', 'elu', 'selu', 'leaky_relu').
            model_device (str): Device to run the model on ('gpu', 'cpu', 'mps').
            plot_format (str): Format for saving plots ('pdf', 'png', 'jpg').
            plot_fontsize (int): Font size for plots.
            plot_despine (bool): If True, removes top and right spines from plots.
            plot_dpi (int): DPI resolution for saved plots.
            plot_show_plots (bool): If True, displays plots interactively.
            debug (bool): If True, enables debug mode with more detailed logging.
        """
        self.model_name = "ImputeNLPCA"
        kwargs = {"prefix": prefix, "debug": debug, "verbose": verbose}
        logman = LoggerManager(__name__, **kwargs)
        self.logger = logman.get_logger()

        super().__init__(
            prefix=prefix, device=model_device, verbose=verbose, debug=debug
        )

        self.genotype_data = genotype_data
        self.pgenc = GenotypeEncoder(genotype_data)
        self.Model = NLPCAModel
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.n_jobs = n_jobs

        # Model & Training Params
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

        # Plotting & Output
        self.prefix = prefix
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi
        self.show_plots = plot_show_plots
        self.plot_fontsize = plot_fontsize
        self.title_fontsize = plot_fontsize
        self.despine = plot_despine
        self.scoring_averaging = "weighted"

        # Core model config
        self.is_haploid = None
        self.num_classes_ = None  # Will be 2 or 3
        self.model_params = {}

    def fit(self) -> "ImputeNLPCA":
        """Fits the NLPCA model to the 0/1/2 encoded genotype data.

        This method prepares the data, splits it into training and validation sets, initializes the model, and trains it. If hyperparameter tuning is enabled, it will perform tuning before final training. After training, it evaluates the model on a test set and generates relevant plots.

        Returns:
            ImputeNLPCA: The fitted imputer instance.

        Raises:
            NotFittedError: If the model fails to train.
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

        self.num_classes_ = 2 if self.is_haploid else 3
        self.logger.info(
            f"Data is {'haploid' if self.is_haploid else 'diploid'}. "
            f"Using {self.num_classes_} classes for prediction."
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

        for batch_indices, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            latent_optimizer.zero_grad(set_to_none=True)

            logits = model.phase23_decoder(latent_vectors[batch_indices]).view(
                len(batch_indices), self.num_features_, self.num_classes_
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
        """Evaluates the model on a test set.

        This method evaluates the trained NLPCA model on a test dataset by optimizing latent vectors for the test samples, predicting genotypes, and computing various performance metrics. It can operate in an objective mode that suppresses logging for automated evaluations.

        Args:
            X_test (np.ndarray): Test data in 0/1/2 encoding with -1 for missing.
            model (torch.nn.Module): Trained NLPCA model.
            params (dict): Model parameters.
            objective_mode (bool): If True, suppresses logging and reports only the metric.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        # Optimize latents for the test set
        test_latent_vectors = self._optimize_latents_for_inference(
            X_test, model, params
        )

        pred_labels, pred_probas = self._predict(
            model=model, latent_vectors=test_latent_vectors
        )

        eval_mask = X_test != -1
        y_true_flat = X_test[eval_mask]
        pred_labels_flat = pred_labels[eval_mask]
        pred_probas_flat = pred_probas[eval_mask]

        if y_true_flat.size == 0:
            return {self.tune_metric: 0.0}

        # For haploids, remap class 2 to 1 for scoring (e.g., f1-score)
        labels_for_scoring = [0, 1] if self.is_haploid else [0, 1, 2]
        target_names = ["REF", "ALT"] if self.is_haploid else ["REF", "HET", "ALT"]

        if self.is_haploid:
            y_true_flat[y_true_flat == 2] = 1
            pred_labels_flat[pred_labels_flat == 2] = 1

            # Adjust probabilities for 2 classes
            probas_2_class = np.zeros((len(pred_probas_flat), 2))
            probas_2_class[:, 0] = pred_probas_flat[:, 0]
            probas_2_class[:, 1] = pred_probas_flat[:, 2]
            pred_probas_flat = probas_2_class

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

            y_true_dec = self.pgenc.decode_012(X_test)
            X_pred = X_test.copy()
            X_pred[eval_mask] = pred_labels_flat
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
        """Creates a PyTorch DataLoader for the 0/1/2 encoded data.

        This method constructs a DataLoader from the provided genotype data, which is expected to be in 0/1/2 encoding with -1 for missing values. The DataLoader is used for batching and shuffling the data during model training. It converts the numpy array to a PyTorch tensor and creates a TensorDataset. The DataLoader is configured with the specified batch size and shuffling enabled.

        Args:
            y (np.ndarray): 0/1/2 encoded genotype data with -1 for missing.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the dataset.
        """
        y_tensor = torch.from_numpy(y).long().to(self.device)
        dataset = torch.utils.data.TensorDataset(torch.arange(len(y)), y_tensor)
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
        """Initializes the latent space using random or PCA initialization.

        This method initializes the latent space for the NLPCA model. It supports two initialization methods: random initialization using Xavier uniform distribution and PCA-based initialization. For PCA initialization, it performs mean imputation on missing values and uses sklearn's PCA to derive the initial latent vectors. If the number of PCA components is less than the desired latent dimension, it pads the remaining dimensions with random noise. The resulting latent vectors are returned as a trainable PyTorch parameter.

        Args:
            params (dict): Model parameters including 'latent_dim'.
            n_samples (int): Number of samples in the dataset.
            X (np.ndarray): 0/1/2 encoded genotype data with -1 for missing.
            latent_init (str): Method to initialize latent space ('random' or 'pca').

        Returns:
            torch.nn.Parameter: Initialized latent vectors as a trainable parameter.
        """
        latent_dim = int(params["latent_dim"])
        if latent_init == "pca":
            # Use a copy to avoid modifying the original data
            X_pca = X.copy().astype(np.float32)

            # Simple mean imputation for PCA
            col_means = np.nanmean(np.where(X_pca == -1, np.nan, X_pca), axis=0)

            # Handle columns that are all NaN
            col_means = np.nan_to_num(col_means, nan=np.nanmean(col_means))

            nan_mask = X_pca == -1
            X_pca[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

            # Ensure n_components is valid
            n_components = min(latent_dim, n_samples, self.num_features_)
            if n_components < latent_dim:
                self.logger.warning(
                    f"Latent dim reduced from {latent_dim} to {n_components} for PCA."
                )

            pca = PCA(n_components=n_components, random_state=self.seed)
            initial_latents = pca.fit_transform(X_pca)

            # Pad with random noise if PCA components are fewer than latent_dim
            if n_components < latent_dim:
                padding = self.rng.standard_normal(
                    size=(n_samples, latent_dim - n_components)
                )
                initial_latents = np.hstack([initial_latents, padding])

            initial_latents = (initial_latents - initial_latents.mean(axis=0)) / (
                initial_latents.std(axis=0) + 1e-6
            )
            latents_tensor = torch.from_numpy(initial_latents).float().to(self.device)
        else:  # Random initialization
            latents_tensor = torch.empty(n_samples, latent_dim, device=self.device)
            torch.nn.init.xavier_uniform_(latents_tensor)

        return torch.nn.Parameter(latents_tensor, requires_grad=True)

    def _objective(self, trial: optuna.Trial) -> float:
        """Defines the objective function for Optuna hyperparameter tuning.

        This method serves as the objective function for Optuna during hyperparameter tuning. It samples a set of hyperparameters, initializes the NLPCA model, trains it on the training set, and evaluates its performance on the validation set. The performance metric specified by `self.tune_metric` is returned for optimization. If any error occurs during the trial, it raises a TrialPruned exception to indicate failure.

        Args:
            trial (optuna.Trial): An Optuna trial object for hyperparameter suggestions.

        Returns:
            float: The value of the tuning metric to be optimized.
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
        """Samples hyperparameters for the simplified NLPCA model.

        This method defines the hyperparameter search space for the NLPCA model and samples a set of hyperparameters using the provided Optuna trial object. It computes the hidden layer sizes based on the sampled parameters and prepares the model parameters dictionary.

        Args:
            trial (optuna.Trial): An Optuna trial object for hyperparameter suggestions.

        Returns:
            Dict[str, int | float | str | list]: A dictionary of sampled hyperparameters.
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
        """Orchestrates the training loop for a given model and configuration.

        This method sets up the optimizers and learning rate scheduler, then executes the training loop for the NLPCA model. It handles both model parameter updates and latent vector updates. If specified, it returns the training history along with the best loss and model.

        Args:
            model (torch.nn.Module): The NLPCA model to train.
            loader (torch.utils.data.DataLoader): DataLoader for training data.
            lr (float): Learning rate for the optimizer.
            l1_penalty (float): L1 regularization penalty.
            trial (optuna.Trial | None): Optuna trial object for hyperparameter tuning.
            return_history (bool): If True, returns training history.
            latent_vectors (torch.nn.Parameter | None): Latent vectors for samples.
            lr_input_factor (float): Factor to scale learning rate for latent vectors.
            class_weights (torch.Tensor | None): Class weights for handling class imbalance.

        Returns:
            Tuple: (best_loss, best_model, history, latent_vectors) if return_history is True, else (best_loss, best_model, latent_vectors).
        """
        if latent_vectors is None or class_weights is None:
            raise TypeError("Must provide latent_vectors and class_weights.")

        latent_optimizer = torch.optim.Adam([latent_vectors], lr=lr * lr_input_factor)
        optimizer = torch.optim.Adam(model.phase23_decoder.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        result = self._execute_training_loop(
            loader,
            optimizer,
            latent_optimizer,
            scheduler,
            model,
            l1_penalty,
            return_history,
            latent_vectors,
            class_weights,
        )

        if return_history:
            return result  # (best_loss, best_model, history, latents)

        # (best_loss, best_model, latents)
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

        fn = self.models_dir / "final_model.pt"
        torch.save(trained_model.state_dict(), fn)

        return loss, trained_model, {"Train": history}, latent_vectors

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
    ) -> Tuple[float, torch.nn.Module, list, torch.nn.Parameter]:
        """Executes the core training loop.

        This method runs the training loop for the NLPCA model, performing multiple epochs of training. It utilizes early stopping to prevent overfitting and can return the training history if specified. The method updates both the model parameters and latent vectors during training.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            latent_optimizer (torch.optim.Optimizer): Optimizer for latent vectors.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            model (torch.nn.Module): The NLPCA model.
            l1_penalty (float): L1 regularization penalty.
            return_history (bool): If True, returns training history.
            latent_vectors (torch.nn.Parameter): Latent vectors for samples.
            class_weights (torch.Tensor): Class weights for handling class imbalance.

        Returns:
            Tuple[float, torch.nn.Module, list, torch.nn.Parameter]: Best loss, best model, training history, and optimized latent vectors.

        Raises:
            optuna.exceptions.TrialPruned: If the trial is pruned due to NaN or Inf loss.
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

        for epoch in range(self.epochs):
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

            early_stopping(train_loss, model)
            if early_stopping.early_stop:
                self.logger.info(f"Early stopping at epoch {epoch + 1}.")
                break

        best_loss = early_stopping.best_score
        best_model = copy.deepcopy(early_stopping.best_model)

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
        self.logger.info("Optimizing latent vectors for new data...")
        model.eval()

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
                len(X_new), self.num_features_, self.num_classes_
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
