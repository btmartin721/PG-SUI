import copy
from typing import TYPE_CHECKING, Dict, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from snpio.analysis.genotype_encoder import GenotypeEncoder
from snpio.utils.logging import LoggerManager
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.callbacks import EarlyStopping
from pgsui.impute.unsupervised.models.vae_model import VAEModel

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


class ImputeVAE(BaseNNImputer):
    """VAE imputer on 0/1/2-encoded genotypes (missing=-1), NLPCA/AE-parity.

    Inputs = one-hot(0/1/2) with zeros where missing; Targets = same 0/1/2 ints. Diploid => 3 classes (0/1/2), Haploid => 2 classes (0/2 collapsed to 0/1 for scoring). Preserves VAE specifics: stochastic z, KL term (β), and β-annealing schedule. Produces same reports/plots as ImputeNLPCA/ImputeAutoencoder, including IUPAC→10-int.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        *,
        seed: int | None = None,
        n_jobs: int = 1,
        prefix: str = "pgsui",
        verbose: bool = False,
        # weighting
        weights_beta: float = 0.9999,
        weights_max_ratio: float = 1.0,
        # tuning
        tune: bool = False,
        tune_metric: Literal["f1", "accuracy", "pr_macro"] = "f1",
        tune_n_trials: int = 100,
        # model/training
        model_latent_dim: int = 16,
        model_dropout_rate: float = 0.2,
        model_num_hidden_layers: int = 3,
        model_batch_size: int = 64,
        model_learning_rate: float = 1e-3,
        model_early_stop_gen: int = 25,
        model_min_epochs: int = 100,
        model_epochs: int = 5000,
        model_l1_penalty: float = 0.0,
        model_layer_scaling_factor: float = 5.0,
        model_layer_schedule: Literal["pyramid", "constant", "linear"] = "pyramid",
        model_gamma: float = 2.0,
        model_beta: float = 1.0,
        model_hidden_activation: Literal["relu", "elu", "selu", "leaky_relu"] = "relu",
        model_device: Literal["gpu", "cpu", "mps"] = "cpu",
        model_validation_split: float = 0.2,
        # plotting
        plot_format: Literal["pdf", "png", "jpg", "jpeg"] = "pdf",
        plot_dpi: int = 300,
        plot_show_plots: bool = False,
        plot_fontsize: int = 18,
        plot_despine: bool = True,
        debug: bool = False,
    ):
        """Initializes the ImputeVAE imputer.

        This class extends the BaseNNImputer to implement a Variational Autoencoder (VAE) model for genotype imputation. It supports hyperparameter tuning using Optuna and provides various configuration options for model training and plotting.

        Args:
            genotype_data (GenotypeData): The genotype data object containing SNP data.
            seed (int | None): Random seed for reproducibility. If None, a random seed is used. Defaults to None.
            n_jobs (int): The number of parallel jobs to use for training. Defaults to 1.
            prefix (str): A prefix used for logging and saving models. Defaults to "pgsui".
            verbose (bool): If True, enables detailed logging. Defaults to False.
            weights_beta (float): The beta parameter for weighting in the loss function. Defaults to 0.9999.
            weights_max_ratio (float): The maximum ratio for class weighting. Defaults to 1.0.
            tune (bool): If True, enables hyperparameter tuning using Optuna. Defaults to False.
            tune_metric (Literal["f1", "accuracy", "pr_macro"]): The metric to optimize during hyperparameter tuning. Defaults to "f1".
            tune_n_trials (int): The number of trials for hyperparameter tuning. Defaults to 100.
            model_latent_dim (int): The dimensionality of the latent space in the VAE. Defaults to 16.
            model_dropout_rate (float): The dropout rate for regularization in hidden layers. Defaults to 0.2.
            model_num_hidden_layers (int): The number of hidden layers in the encoder and decoder. Defaults to 3.
            model_batch_size (int): The batch size for training. Defaults to 64.
            model_learning_rate (float): The learning rate for the optimizer. Defaults to 1e-3.
            model_early_stop_gen (int): The number of generations with no improvement before early stopping. Defaults to 25.
            model_min_epochs (int): The minimum number of epochs to train before considering early stopping. Defaults to 100.
            model_epochs (int): The maximum number of epochs for training. Defaults to 5000.
            model_l1_penalty (float): The L1 penalty for regularization in the loss function. Defaults to 0.0.
            model_layer_scaling_factor (float): The scaling factor for determining hidden layer sizes. Defaults to 5.0.
            model_layer_schedule (Literal["pyramid", "constant", "linear"]): The schedule for hidden layer sizes. Defaults to "pyramid".
            model_gamma (float): The focusing parameter for the focal loss function. Defaults to 2.0.
            model_beta (float): The beta parameter for the KL divergence term in the VAE loss. Defaults to 1.0.
            model_hidden_activation (Literal["relu", "elu", "selu", "leaky_relu"]): The activation function for hidden layers. Defaults to "relu".
            model_device (Literal["gpu", "cpu", "mps"]): The PyTorch device to run the model on. Defaults to "cpu".
            model_validation_split (float): The proportion of data to use for validation. Defaults to 0.2.
            plot_format (Literal["pdf", "png", "jpg"]): The file format for saving plots. Defaults to "pdf".
            plot_dpi (int): The resolution (dots per inch) for saved plots. Defaults to 300.
            plot_show_plots (bool): If True, displays plots interactively. Defaults to False.
            plot_fontsize (int): The font size for plot text. Defaults to 18.
            plot_despine (bool): If True, removes the top and right spines from plots. Defaults to True.
            debug (bool): If True, enables debug mode. Defaults to False.
        """
        self.model_name = "ImputeVAE"
        kwargs = {"prefix": prefix, "debug": debug, "verbose": verbose}
        logman = LoggerManager(__name__, **kwargs)
        self.logger = logman.get_logger()

        super().__init__(
            prefix=prefix, device=model_device, verbose=verbose, debug=debug
        )

        self.genotype_data = genotype_data
        self.pgenc = GenotypeEncoder(genotype_data)

        self.Model = VAEModel
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
        self.batch_size = model_batch_size
        self.learning_rate = model_learning_rate
        self.early_stop_gen = model_early_stop_gen
        self.min_epochs = model_min_epochs
        self.epochs = model_epochs
        self.l1_penalty = model_l1_penalty
        self.gamma = model_gamma
        self.beta = model_beta
        self.activation = model_hidden_activation
        self.validation_split = model_validation_split

        # weighting
        self.beta_w = weights_beta
        self.max_ratio = weights_max_ratio

        # tuning
        self.tune = tune
        self.tune_metric = tune_metric
        self.n_trials = tune_n_trials

        # plotting
        self.prefix = prefix
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi
        self.show_plots = plot_show_plots
        self.plot_fontsize = plot_fontsize
        self.title_fontsize = plot_fontsize
        self.despine = plot_despine
        self.scoring_averaging = "weighted"

        # core config (filled in fit)
        self.is_haploid = None
        self.num_classes_ = None  # 2 or 3
        self.model_params = {}

    # -------------------- Fit -------------------- #
    def fit(self) -> "ImputeVAE":
        self.logger.info(f"Fitting {self.model_name} (0/1/2 VAE) ...")

        # 0/1/2 prep (missing -> -1)
        X = self.pgenc.genotypes_012.astype(np.float32)
        X[X < 0] = np.nan
        X[np.isnan(X)] = -1
        self.ground_truth_ = X.astype(np.int64)

        # ploidy/classes
        self.is_haploid = np.all(
            np.isin(
                self.genotype_data.snp_data, ["A", "C", "G", "T", "N", "-", ".", "?"]
            )
        )
        self.ploidy = 1 if self.is_haploid else 2
        self.num_classes_ = 2 if self.is_haploid else 3
        self.logger.info(
            f"Data is {'haploid' if self.is_haploid else 'diploid'}. "
            f"Using {self.num_classes_} classes."
        )

        n_samples, self.num_features_ = X.shape

        # model params (decoder emits L*K logits)
        self.model_params = {
            "n_features": self.num_features_,
            "num_classes": self.num_classes_,
            "latent_dim": self.latent_dim,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "gamma": self.gamma,
            "beta": self.beta,  # VAE loss uses model.beta (we anneal it below)
            # hidden sizes set in _default/_set_best_params
        }

        # train/val split (sample-wise)
        indices = np.arange(n_samples)
        train_idx, val_idx = train_test_split(
            indices, test_size=self.validation_split, random_state=self.seed
        )
        self.train_idx_, self.test_idx_ = train_idx, val_idx
        self.X_train_ = self.ground_truth_[train_idx]
        self.X_val_ = self.ground_truth_[val_idx]

        # plotting/scorers & (optional) tuning
        self.plotter_, self.scorers_ = self.initialize_plotting_and_scorers()
        if self.tune:
            self.tune_hyperparameters()

        self.best_params_ = getattr(self, "best_params_", self._default_best_params())

        # class weights on 0/1/2
        self.class_weights_ = self._class_weights_from_zygosity(self.X_train_).to(
            self.device
        )

        # loader (indices + integer targets; inputs built on the fly)
        train_loader = self._get_data_loader(self.X_train_)

        # build & train VAE
        model = self.build_model(self.Model, self.best_params_)
        model.apply(self.initialize_weights)

        loss, trained_model, history = self._train_and_validate_model(
            model=model,
            loader=train_loader,
            lr=self.learning_rate,
            l1_penalty=self.l1_penalty,
            return_history=True,
            class_weights=self.class_weights_,
        )
        if trained_model is None:
            raise RuntimeError("Final model training failed.")

        torch.save(
            trained_model.state_dict(), self.models_dir / "final_model_vae_012.pt"
        )
        self.best_loss_, self.model_, self.history_ = (
            loss,
            trained_model,
            {"Train": history},
        )
        self.is_fit_ = True

        # validation evaluation (parity with AE/NLPCA)
        self._evaluate_model(self.X_val_, self.model_, self.best_params_)
        self.plotter_.plot_history(self.history_)
        return self

    def transform(self) -> np.ndarray:
        """Impute full dataset -> IUPAC strings; plots distributions.

        This method imputes missing genotype data in the full dataset using the trained VAE model. It generates IUPAC strings for the imputed genotypes and plots the distributions of the original and imputed genotypes.

        Returns:
            np.ndarray: The imputed genotype data in 0/1/2 encoding.

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        if not getattr(self, "is_fit_", False):
            raise NotFittedError("Model is not fitted. Call fit() before transform().")

        self.logger.info("Imputing entire dataset with VAE (0/1/2)...")
        X_to_impute = self.ground_truth_.copy()

        pred_labels, _ = self._predict(self.model_, X=X_to_impute, return_proba=True)

        imputed = X_to_impute.copy()
        imputed[X_to_impute == -1] = pred_labels[X_to_impute == -1]

        imputed_genotypes = self.pgenc.decode_012(imputed)
        original_genotypes = self.pgenc.decode_012(X_to_impute)

        plt.rcParams.update(self.plotter_.param_dict)
        self.plotter_.plot_gt_distribution(original_genotypes, is_imputed=False)
        self.plotter_.plot_gt_distribution(imputed_genotypes, is_imputed=True)
        return imputed_genotypes

    def _get_data_loader(self, y: np.ndarray) -> DataLoader:
        """Yield (indices, y_int) where y_int is 0/1/2 with -1 for missing.

        This method creates a PyTorch DataLoader for the input data, yielding batches of (indices, y_int) where y_int is the integer representation of the genotype data (0, 1, 2) with -1 for missing values.

        Args:
            y (np.ndarray): The target array of shape `(n_samples, n_features)` with integer values (0, 1, 2) and -1 for missing values.

        Returns:
            DataLoader: A PyTorch DataLoader yielding batches of (indices, y_int).

        Raises:
            TypeError: If the input array `y` is not of type `np.ndarray`.
        """
        y_tensor = torch.from_numpy(y).long().to(self.device)
        dataset = TensorDataset(torch.arange(len(y), device=self.device), y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _train_and_validate_model(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        lr: float,
        l1_penalty: float,
        trial: optuna.Trial | None = None,
        return_history: bool = False,
        class_weights: torch.Tensor | None = None,
    ) -> Tuple[float, torch.nn.Module | None, list | None]:
        """Wrap VAE training (no latent-input optimization; β-annealing inside).

        This method handles the training of the VAE model using the provided DataLoader. It sets up the optimizer and learning rate scheduler, then executes the training loop. The method supports returning the training history if specified.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            loader (DataLoader): The DataLoader providing training data.
            lr (float): The learning rate for the optimizer.
            l1_penalty (float): The L1 penalty for regularization in the loss function.
            trial (optuna.Trial | None): An optional Optuna trial object for hyperparameter tuning. Defaults to None.
            return_history (bool): If True, returns the training history. Defaults to False.
            class_weights (torch.Tensor | None): An optional tensor of weights for each class to address imbalance. Defaults to None.

        Returns:
            Tuple[float, torch.nn.Module | None, list | None]: A tuple containing the best loss value, the trained model, and optionally the training history.
        """
        if class_weights is None:
            raise TypeError("Must provide class_weights.")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_loss, best_model, hist = self._execute_training_loop(
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            l1_penalty=l1_penalty,
            trial=trial,
            return_history=return_history,
            class_weights=class_weights,
        )
        if return_history:
            return best_loss, best_model, hist
        return best_loss, best_model, None

    def _execute_training_loop(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        model: torch.nn.Module,
        l1_penalty: float,
        trial,
        return_history: bool,
        class_weights: torch.Tensor,
    ) -> Tuple[float, torch.nn.Module, list]:
        """Train with focal CE recon + KL; β-anneal schedule (warm, ramp, hold).

        This method executes the core training loop for the VAE model. It performs multiple epochs of training, applying β-annealing to the KL divergence term in the loss function. The method utilizes early stopping to prevent overfitting and can return the training history if specified.

        Args:
            loader (DataLoader): The DataLoader providing training data.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
            model (torch.nn.Module): The PyTorch model to be trained.
            l1_penalty (float): The L1 penalty for regularization in the loss function.
            trial (optuna.Trial | None): An optional Optuna trial object for hyperparameter tuning. Defaults to None.
            return_history (bool): If True, returns the training history. Defaults to False.
            class_weights (torch.Tensor): A tensor of weights for each class to address imbalance. Defaults to None.

        Returns:
            Tuple[float, torch.nn.Module, list]: A tuple containing the best loss value, the trained model, and optionally the training history.
        """
        best_model = None
        history: list[float] = []

        early_stopping = EarlyStopping(
            patience=self.early_stop_gen,
            min_epochs=self.min_epochs,
            verbose=self.verbose,
            prefix=self.prefix,
            debug=self.debug,
        )

        warm, ramp, beta_final = 50, 200, self.beta
        for epoch in range(self.epochs):
            # β-annealing (stored on model so compute_loss can read it)
            if epoch < warm:
                model.beta = 0.0
            elif epoch < warm + ramp:
                model.beta = beta_final * ((epoch - warm) / ramp)
            else:
                model.beta = beta_final

            train_loss = self._train_step(
                loader=loader,
                optimizer=optimizer,
                model=model,
                l1_penalty=l1_penalty,
                class_weights=class_weights,
            )
            if trial and (np.isnan(train_loss) or np.isinf(train_loss)):
                raise optuna.exceptions.TrialPruned("Loss is NaN or Inf.")

            scheduler.step()
            if return_history:
                history.append(train_loss)

            early_stopping(train_loss, model)
            if early_stopping.early_stop:
                self.logger.info(f"Early stopping at epoch {epoch + 1}.")
                break

        best_loss = early_stopping.best_score
        best_model = copy.deepcopy(early_stopping.best_model)
        return best_loss, best_model, history

    def _train_step(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        l1_penalty: float,
        class_weights: torch.Tensor,
    ) -> float:
        """One epoch: (indices, y_int) → one-hot → VAE forward → recon+KL.

        This method executes a single training epoch for the VAE model. It processes the input data through the model and computes the loss, which is a combination of reconstruction loss and KL divergence. The method applies backpropagation and updates the model parameters.

        Args:
            loader (DataLoader): The DataLoader providing training data.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            model (torch.nn.Module): The PyTorch model to be trained.
            l1_penalty (float): The L1 penalty for regularization in the loss function.
            class_weights (torch.Tensor): A tensor of weights for each class to address imbalance.

        Returns:
            float: The average training loss for the epoch.
        """
        model.train()
        running = 0.0

        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)

            x_ohe = self._one_hot_encode_012(y_batch)  # (B, L, K), zeros for -1
            outputs = model(x_ohe)  # expect (recon_logits, mu, logvar, ...)

            y_ohe = self._one_hot_encode_012(y_batch)  # reuse for compute_loss
            valid_mask = y_batch != -1

            loss = model.compute_loss(
                outputs=outputs,
                y=y_ohe,  # (B, L, K)
                mask=valid_mask,  # (B, L)
                class_weights=class_weights,
            )

            if l1_penalty > 0:
                loss = loss + l1_penalty * sum(
                    p.abs().sum() for p in model.parameters()
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item())

        return running / len(loader)

    def _predict(
        self,
        model: torch.nn.Module,
        X: np.ndarray | torch.Tensor,
        return_proba: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Forward (encoder→z→decoder) to 0/1/2 labels (+probas).

        This method performs prediction using the trained VAE model. It takes input data and produces predicted labels (0, 1, 2) and optionally class probabilities. The method handles both NumPy arrays and PyTorch tensors as input.

        Args:
            model (torch.nn.Module): The trained PyTorch model for prediction.
            X (np.ndarray | torch.Tensor): The input data for prediction, either as a NumPy array or a PyTorch tensor.
            return_proba (bool): If True, returns class probabilities along with labels. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray] | np.ndarray: If `return_proba` is True, returns a tuple of predicted labels and class probabilities. Otherwise, returns only the predicted labels.

        Raises:
            NotFittedError: If the model is not fitted.
        """
        if model is None:
            msg = "Model is not fitted."
            self.logger.error(msg)
            raise NotFittedError(msg)

        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
            X_tensor = X_tensor.to(self.device).long()
            x_ohe = self._one_hot_encode_012(X_tensor)
            outputs = model(x_ohe)
            logits = outputs[0].view(-1, self.num_features_, self.num_classes_)
            probas = torch.softmax(logits, dim=-1)
            labels = torch.argmax(probas, dim=-1)

        if return_proba:
            return labels.cpu().numpy(), probas.cpu().numpy()

        return labels.cpu().numpy()

    def _evaluate_model(
        self,
        X_val: np.ndarray,
        model: torch.nn.Module,
        params: dict,
        objective_mode: bool = False,
    ) -> Dict[str, float]:
        """Same eval/reporting as AE/NLPCA, including IUPAC→10-int report.

        This method evaluates the trained VAE model on validation data and generates various evaluation metrics and reports. It supports both detailed logging and a simplified mode for hyperparameter tuning.

        Args:
            X_val (np.ndarray): The validation data for evaluation.
            model (torch.nn.Module): The trained PyTorch model for evaluation.
            params (dict): The model parameters used for evaluation.
            objective_mode (bool): If True, suppresses detailed logging and reports. Defaults to False.

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics.
        """
        pred_labels, pred_probas = self._predict(
            model=model, X=X_val, return_proba=True
        )

        eval_mask = X_val != -1
        y_true_flat = X_val[eval_mask]
        y_pred_flat = pred_labels[eval_mask]
        y_proba_flat = pred_probas[eval_mask]

        if y_true_flat.size == 0:
            return {self.tune_metric: 0.0}

        labels_for_scoring = [0, 1] if self.is_haploid else [0, 1, 2]
        target_names = ["REF", "ALT"] if self.is_haploid else ["REF", "HET", "ALT"]

        if self.is_haploid:
            y_true_flat = y_true_flat.copy()
            y_pred_flat = y_pred_flat.copy()
            y_true_flat[y_true_flat == 2] = 1
            y_pred_flat[y_pred_flat == 2] = 1
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
            self.logger.info(f"Validation Metrics: {metrics}")

            # Main report (0/1/2 or 0/1 collapsed)
            self._make_class_reports(
                y_true=y_true_flat,
                y_pred_proba=y_proba_flat,
                y_pred=y_pred_flat,
                metrics=metrics,
                labels=target_names,
            )

            # IUPAC decode & 10-base integer report
            y_true_dec = self.pgenc.decode_012(X_val)
            X_pred = X_val.copy()
            X_pred[eval_mask] = y_pred_flat
            y_pred_dec = self.pgenc.decode_012(
                X_pred.reshape(X_val.shape[0], self.num_features_)
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

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective for 0/1/2 VAE.

        This method defines the objective function for hyperparameter tuning using Optuna. It samples hyperparameters, trains the VAE model, and evaluates it on validation data, returning the value of the tuning metric to be optimized.

        Args:
            trial (optuna.Trial): An Optuna trial object for hyperparameter tuning.

        Returns:
            float: The value of the tuning metric to be optimized.
        """
        try:
            params = self._sample_hyperparameters(trial)

            X_train = self.ground_truth_[self.train_idx_]
            X_val = self.ground_truth_[self.test_idx_]

            class_weights = self._class_weights_from_zygosity(X_train).to(self.device)
            train_loader = self._get_data_loader(X_train)

            model = self.build_model(self.Model, params["model_params"])
            model.apply(self.initialize_weights)

            _, model, _ = self._train_and_validate_model(
                model=model,
                loader=train_loader,
                lr=params["lr"],
                l1_penalty=params["l1_penalty"],
                trial=trial,
                return_history=False,
                class_weights=class_weights,
            )

            metrics = self._evaluate_model(X_val, model, params, objective_mode=True)
            self._clear_resources(model, train_loader)
            return metrics[self.tune_metric]
        except Exception as e:
            raise optuna.exceptions.TrialPruned(f"Trial failed with error: {e}")

    def _sample_hyperparameters(
        self, trial: optuna.Trial
    ) -> Dict[str, int | float | str | list]:
        """Sample VAE hyperparams; hidden sizes follow AE/NLPCA sizing helper.

        This method samples hyperparameters for the VAE model using the provided Optuna trial object. It defines the search space for various hyperparameters and computes the hidden layer sizes based on the sampled parameters.

        Args:
            trial (optuna.Trial): An Optuna trial object for hyperparameter tuning.

        Returns:
            Dict[str, int | float | str | list]: A dictionary of sampled hyperparameters.
        """
        params = {
            "latent_dim": trial.suggest_int("latent_dim", 2, 64),
            "lr": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.6),
            "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 8),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "elu", "selu"]
            ),
            "l1_penalty": trial.suggest_float("l1_penalty", 1e-7, 1e-2, log=True),
            "layer_scaling_factor": trial.suggest_float(
                "layer_scaling_factor", 2.0, 10.0
            ),
            "layer_schedule": trial.suggest_categorical(
                "layer_schedule", ["pyramid", "constant", "linear"]
            ),
            "beta": trial.suggest_float("beta", 0.25, 4.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }

        input_dim = self.num_features_ * self.num_classes_
        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=input_dim,
            n_outputs=input_dim,
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
            "beta": params["beta"],
            "gamma": params["gamma"],
        }
        return params

    def _set_best_params(
        self, best_params: Dict[str, int | float | str | list]
    ) -> Dict[str, int | float | str | list]:
        """Adopt best params and return model_params.

        This method updates the model's hyperparameters with the best ones found during tuning and prepares the model parameters for building the VAE. It computes the hidden layer sizes based on the best hyperparameters.

        Args:
            best_params (Dict[str, int | float | str | list]): The best hyperparameters found during tuning.

        Returns:
            Dict[str, int | float | str | list]: A dictionary of model parameters to be used for building the model.
        """
        self.latent_dim = best_params["latent_dim"]
        self.dropout_rate = best_params["dropout_rate"]
        self.learning_rate = best_params["learning_rate"]
        self.l1_penalty = best_params["l1_penalty"]
        self.activation = best_params["activation"]
        self.layer_scaling_factor = best_params["layer_scaling_factor"]
        self.layer_schedule = best_params["layer_schedule"]
        self.beta = best_params.get("beta", self.beta)
        self.gamma = best_params.get("gamma", self.gamma)

        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=self.num_features_ * self.num_classes_,
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
            "num_classes": self.num_classes_,
            "beta": self.beta,
            "gamma": self.gamma,
        }

    def _default_best_params(self) -> Dict[str, int | float | str | list]:
        """Default model params when tuning is disabled.

        This method provides a set of default hyperparameters for the VAE model when hyperparameter tuning is not performed. It computes the hidden layer sizes based on the default parameters.

        Returns:
            Dict[str, int | float | str | list]: A dictionary of default model parameters.
        """
        hidden_layer_sizes = self._compute_hidden_layer_sizes(
            n_inputs=self.num_features_ * self.num_classes_,
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
            "num_classes": self.num_classes_,
            "beta": self.beta,
            "gamma": self.gamma,
        }
