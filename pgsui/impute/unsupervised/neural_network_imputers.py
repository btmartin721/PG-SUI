import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import optuna
import torch
import torch.optim as optim
from snpio.utils.logging import LoggerManager
from torch import optim

from pgsui.data_processing.transformers import (
    AutoEncoderFeatureTransformer,
    SimGenotypeDataTransformer,
)
from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.impute.unsupervised.models.vae_model import VAEModel
from pgsui.impute.unsupervised.nn_scorers import Scorer
from pgsui.utils.misc import validate_input_type
from pgsui.utils.plotting import Plotting


class ImputeVAE(BaseNNImputer):
    """VAE imputer class.

    This class implements a variational autoencoder (VAE) imputer for imputing missing values in genotype data. It uses a PyTorch-based VAE model to predict the missing values in the input data. The class provides methods for training the model, predicting the missing values, and optimizing hyperparameters using Optuna.

    Example:
        >>> from snpio import VCFReader, GenotypeEncoder
        >>> from pgsui.impute.unsupervised.neural_network_imputers import ImputeVAE
        >>> genotype_data = VCFReader(filename="example.vcf.gz", popmapfile="example.popmap", verbose=1)
        >>> ge = GenotypeEncoder(genotype_data)
        >>> vae_imputer = ImputeVAE(genotype_data, num_classes=3, tune=True, n_trials=10, n_jobs=2, verbose=1)
        >>> vae_imputer.fit(ge.genotypes_012)
        >>> X_imputed = vae_imputer.transform(ge.genoypes_012)

    Attributes:
        genotype_data (Any): Genotype data object.
        num_classes (int): Number of classes.
        model_name (str): Name of the model.
        prefix (str): Prefix for the output directory.
        output_dir (str): Output directory for saving model output.
        latent_dim (int): Latent dimension of the VAE model.
        dropout_rate (float): Dropout rate for the model.
        num_hidden_layers (int): Number of hidden layers in the model.
        hidden_layer_sizes (List[int]): List of hidden layer sizes.
        hidden_activation (str): Activation function for hidden layers.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        early_stop_gen (int): Number of generations for early stopping.
        lr_patience (int): Patience for learning rate scheduler.
        epochs (int): Number of epochs for training.
        l1_penalty (float): L1 regularization penalty.
        l2_penalty (float): L2 regularization penalty.
        optimizer (str): Optimizer for training.
        scoring_averaging (str): Averaging method for scoring metrics.
        sim_strategy (str): Strategy for simulating missing data.
        sim_prop_missing (float): Proportion of missing data to simulate.
        validation_split (float): Validation split for training.
        tune (bool): Whether to tune hyperparameters.
        n_trials (int): Number of trials for hyperparameter optimization.
        n_jobs (int): Number of parallel jobs for hyperparameter optimization.
        seed (int): Random seed for reproducibility.
        verbose (int): Verbosity level.
        debug (bool): Whether to enable debug mode.
        device (torch.device): Device (GPU or CPU) for training.
        logger (LoggerManager): Logger instance.
        sim (SimGenotypeDataTransformer): Genotype data transformer for simulating missing data.
        tt_ (AutoEncoderFeatureTransformer): Feature transformer for encoding features.
        model_ (nn.Module): Model instance.
        output_dir (Path): Output directory for saving model output.
        best_params_ (Dict[str, Any]): Best hyperparameters found during optimization.
        best_tuned_loss_ (float): Best loss after hyperparameter optimization.
        train_loader_ (DataLoader): DataLoader for training data.
        val_loader_ (DataLoader): DataLoader for validation data.
        optimizer_ (optim.Optimizer): Optimizer instance.
        history_ (Dict[str, List[float]]): Training and validation history.
        best_loss_ (float): Best loss after training.
        scorer_ (Scorer): Scorer instance for evaluating the model.
        plotter_ (Plotting): Plotting instance for plotting metrics.
        dataset_ (GenotypeDataset): Genotype dataset instance.
        test_dataset_ (GenotypeDataset): Test dataset instance.
        test_loader_ (DataLoader): DataLoader for test data.
        lr_ (float): Learning rate.
        l1_penalty_ (float): L1 regularization penalty.
        l2_penalty_ (float): L2 regularization penalty.
        lr_patience_ (int): Patience for learning rate scheduler.
        model_params (Dict[str, Any]): Model hyperparameters.
        sim_missing_mask_ (np.ndarray): Missing mask for simulated data.
        original_missing_mask_ (np.ndarray): Missing mask for original data.
        all_missing_mask_ (np.ndarray): Missing mask for all data.
    """

    def __init__(
        self,
        genotype_data: Any,
        num_classes: int = 3,
        latent_dim: int = 2,
        dropout_rate: float = 0.2,
        num_hidden_layers: int = 2,
        hidden_layer_sizes: List[int] = [128, 64],
        batch_size: int = 32,
        learning_rate: float = 0.001,
        tune: bool = False,
        n_trials: int = 100,
        early_stop_gen: int = 25,
        optimizer: str = "adam",
        hidden_activation: str = "elu",
        lr_patience: int = 10,
        epochs: int = 100,
        l1_penalty: float = 0.0001,
        l2_penalty: float = 0.0001,
        scoring_averaging: str = "weighted",
        sim_strategy: str = "random",
        n_jobs: int = 1,
        seed: Optional[int] = None,
        sim_prop_missing: float = 0.2,
        validation_split: float = 0.2,
        prefix: Union[str, Path] = "pgsui",
        output_dir: Union[str, Path] = "output",
        plot_format: str = "pdf",
        plot_fontsize: int = 18,
        plot_dpi: int = 300,
        title_fontsize: int = 20,
        despine: bool = True,
        show_plots: bool = False,
        verbose: int = 0,
        debug: bool = False,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the VAE imputer.

        This method initializes the VAE imputer with the specified hyperparameters and settings. It sets up the logger and the device (GPU or CPU) for training the model. It also initializes the genotype data transformer for simulating missing data and the feature transformer for encoding features.

        Args:
            genotype_data (Any): Genotype data object.
            num_classes (int): Number of classes. Defaults to 3.
            latent_dim (int): Latent dimension of the VAE model. Defaults to 2.
            dropout_rate (float): Dropout rate for the model. Defaults to 0.2.
            num_hidden_layers (int): Number of hidden layers in the model. Defaults to 2.
            hidden_layer_sizes (List[int]): List of hidden layer sizes. Defaults to [128, 64].
            batch_size (int): Batch size for training. Defaults to 32.
            learning_rate (float): Learning rate for the optimizer. Defaults to 0.001.
            tune (bool): Whether to tune hyperparameters. Defaults to False.
            n_trials (int): Number of trials for hyperparameter optimization. Defaults to 100.
            early_stop_gen (int): Number of generations for early stopping. Defaults to 25.
            optimizer (str): Optimizer for training. Defaults to "adam".
            hidden_activation (str): Activation function for hidden layers. Defaults to "elu".
            lr_patience (int): Patience for learning rate scheduler. Defaults to 10.
            epochs (int): Number of epochs for training. Defaults to 100.
            l1_penalty (float): L1 regularization penalty. Defaults to 0.0001.
            l2_penalty (float): L2 regularization penalty. Defaults to 0.0001.
            scoring_averaging (str): Averaging method for scoring metrics. Valid options are 'micro', 'macro', and 'weighted'. Defaults to 'weighted'.
            sim_strategy (str): Strategy for simulating missing data. Defaults to "random".
            n_jobs (int): Number of parallel jobs for hyperparameter optimization. Defaults to 1.
            seed (int, optional): Random seed for reproducibility. If left as None, the seed will be set randomly and be non-deterministic. Defaults to None.
            sim_prop_missing (float): Proportion of missing data to simulate. Defaults to 0.2.
            validation_split (float): Validation split for training. Defaults to 0.2.
            prefix (Union[str, Path]): Prefix for the output directory. Defaults to "pgsui".
            output_dir (Union[str, Path]): Output directory for saving model output. Defaults to "output".
            plot_format (str): Plot format for saving plots. Defaults to "pdf".
            plot_fontsize (int): Font size for plots. Defaults to 18.
            plot_dpi (int): DPI for plots. Defaults to 300.
            title_fontsize (int): Font size for plot titles. Defaults to 20.
            despine (bool): Whether to remove spines from plots. Defaults to True.
            show_plots (bool): Whether to show plots. Defaults to False.
            verbose (int): Verbosity level. Defaults to 0.
            debug (bool): Whether to enable debug mode. Defaults to False.
            **kwargs (Dict[str, Any]): Additional keyword arguments.
        """

        self.model_name = "ImputeVAE"

        super().__init__(
            genotype_data,
            num_classes=num_classes,
            model_name=self.model_name,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            num_hidden_layers=num_hidden_layers,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=hidden_activation,
            batch_size=batch_size,
            learning_rate=learning_rate,
            tune=tune,
            n_trials=n_trials,
            early_stop_gen=early_stop_gen,
            optimizer=optimizer,
            lr_patience=lr_patience,
            epochs=epochs,
            l1_penalty=l1_penalty,
            l2_penalty=l2_penalty,
            scoring_averaging=scoring_averaging,
            sim_strategy=sim_strategy,
            n_jobs=n_jobs,
            seed=seed,
            sim_prop_missing=sim_prop_missing,
            validation_split=validation_split,
            prefix=prefix,
            output_dir=output_dir,
            verbose=verbose,
            debug=debug,
            **kwargs,
        )

        logman = LoggerManager(
            name=__name__,
            prefix=self.prefix,
            level=logging.DEBUG if debug else logging.INFO,
            verbose=verbose >= 1,
            debug=debug,
        )

        self.logger = logman.get_logger()

        self.Model = VAEModel

        if not hasattr(self, "genotype_data"):
            self.genotype_data = genotype_data

        self.genotype_data = genotype_data
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = hidden_activation
        self.hidden_activation = hidden_activation
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tune = tune
        self.n_trials = n_trials
        self.early_stop_gen = early_stop_gen
        self.optimizer = optimizer
        self.lr_patience = lr_patience
        self.epochs = epochs
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.scoring_averaging = scoring_averaging
        self.sim_strategy = sim_strategy
        self.n_jobs = n_jobs
        self.sim_prop_missing = sim_prop_missing
        self.validation_split = validation_split
        self.prefix = prefix
        self.output_dir = output_dir
        self.plot_format = plot_format
        self.plot_fontsize = plot_fontsize
        self.plot_dpi = plot_dpi
        self.title_fontsize = title_fontsize
        self.despine = despine
        self.show_plots = show_plots
        self.verbose = verbose
        self.debug = debug

        self.genotype_data.snp_data
        self.num_features = self.genotype_data.num_snps

        self.model_params = {
            "n_features": self.num_features,
            "num_classes": self.num_classes,
            "latent_dim": self.latent_dim,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
        }

        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, List[List[int]], torch.Tensor],
        y: Optional[np.ndarray] = None,
    ) -> Any:
        """Run Optuna to optimize hyperparameters and train the model.

        This method runs Optuna to optimize the hyperparameters of the VAE model and trains the model using the best hyperparameters found during the optimization. It saves the best hyperparameters and the training history in the output directory. The method also plots the training and validation loss curves.

        Args:
            X (numpy.ndarray): The input data with missing values. The input data should be in the same format as the training data.
            y (None): Ignored. For compatibility with the scikit-learn API. Defaults to None.

        Returns:
            self: The fitted ImputeVAE instance. The instance contains the trained model and the best hyperparameters found during optimization if `tune=True`.
        """
        # Ensure that the input data is in the correct format.
        X = validate_input_type(X, return_type="array")
        X[X == -9] = -1
        X = X.astype(float)

        self.best_params_ = self.model_params

        activate = "sigmoid" if self.num_classes == 4 else "softmax"

        # Feature transformation for autoencoder.
        # One-hot encodes the data.
        self.tt_ = AutoEncoderFeatureTransformer(
            num_classes=self.num_classes,
            return_int=False if self.num_classes in {3, 4} else True,
            activate=activate,
            logger=self.logger,
            verbose=self.verbose,
            debug=self.debug,
        )

        # Simulate missing data.
        self.sim_ = SimGenotypeDataTransformer(
            genotype_data=self.genotype_data,
            prop_missing=self.sim_prop_missing,
            strategy=self.sim_strategy,
            missing_val=-1,
            mask_missing=True,
            seed=self.seed,
            verbose=self.verbose,
            logger=self.logger,
            debug=self.debug,
        )

        Xsim = self.sim_.fit_transform(X)
        self.sim_missing_mask_ = self.sim_.sim_missing_mask_
        self.original_missing_mask_ = self.sim_.original_missing_mask_
        self.all_missing_mask_ = self.sim_.all_missing_mask_

        Xenc = self.tt_.fit_transform(Xsim)

        # X undergoes missing data simulation. Y does not, and has ground truth.
        loaders = self.get_data_loaders(Xenc)
        self.train_loader_, self.val_loader_, self.test_loader_ = loaders

        if self.tune:
            self._tune_hyperparameters()
        else:
            self.lr_ = self.learning_rate
            self.lr_patience_ = self.lr_patience
            self.l1_penalty_ = self.l1_penalty
            self.l2_penalty_ = self.l2_penalty

        self._train_final_model()

        loss_type = "Best" if self.tune else "Final"
        self.logger.info(f"{loss_type} loss: {self.best_loss_}")

        # Get the hold-out dataset for evvaluation.
        self._evaluate_model()

        self.plotter_.plot_history(self.history_)

        return self

    def _evaluate_model(self) -> None:
        """Evaluate the model on the hold-out dataset.

        This method evaluates the model on the hold-out dataset using the ground truth labels. It calculates various metrics, such as accuracy, F1 score, precision, recall, average precision, and ROC AUC. It also plots the confusion matrix and the metrics.
        """
        self.dataset_.indices = self.test_dataset_.indices
        X_test, y_test_ohe = self.dataset_.get_data_from_subset(self.test_dataset_)
        y_test_ohe = y_test_ohe.cpu().detach().numpy()

        y_pred, y_pred_proba = self.predict(X_test, return_proba=True)
        y_test = self.tt_.inverse_transform(y_test_ohe)
        X_test = X_test.cpu().detach().numpy()

        missing_mask = self.sim_missing_mask_[self.test_dataset_.indices]

        # Evaluate the model.
        self.scorer_ = Scorer(
            self.model_,
            X_test,
            y_test,
            y_pred,
            y_test_ohe,
            y_pred_proba,
            mask=missing_mask,
            average=self.scoring_averaging,
            logger=self.logger,
            verbose=self.verbose,
            debug=self.debug,
        )

        self.plotter_ = Plotting(
            self.model_name,
            prefix=self.prefix,
            plot_format=self.plot_format,
            plot_fontsize=self.plot_fontsize,
            plot_dpi=self.plot_dpi,
            title_fontsize=self.title_fontsize,
            despine=self.despine,
            show_plots=self.show_plots,
            verbose=self.verbose,
            debug=self.debug,
        )

        if self.debug:
            self._debug_evaluation_results(
                X_test, y_test_ohe, y_pred, y_pred_proba, y_test, missing_mask
            )

        self.metrics_ = self.scorer_.evaluate()
        self.logger.debug(self.metrics_)

        self.plotter_.plot_confusion_matrix(y_test[missing_mask], y_pred[missing_mask])

        self.plotter_.plot_metrics(
            y_test[missing_mask], y_pred_proba[missing_mask], self.metrics_
        )

        self._export_metrics()

    def _export_metrics(self) -> None:
        """Export the evaluation metrics to a JSON file.

        This method exports the evaluation metrics to a JSON file in the output directory. The metrics include accuracy, F1 score, precision, recall, average precision, and ROC AUC.
        """
        od = Path(f"{self.prefix}_{self.output_dir}")
        od = od / "metrics" / "Unsupervised" / self.model_name
        fn = od / "test_metrics.json"

        with open(fn, "w") as fp:
            json.dump(self.metrics_, fp, indent=4)

    def _debug_evaluation_results(
        self, X_test, y_test_ohe, y_pred, y_pred_proba, y_test, missing_mask
    ):
        self.logger.debug(f"{y_test=}")
        self.logger.debug(f"{y_pred=}")
        self.logger.debug(f"{y_test_ohe=}")
        self.logger.debug(f"{y_pred_proba=}")
        self.logger.debug(f"{y_test.shape=}")
        self.logger.debug(f"{y_pred.shape=}")
        self.logger.debug(f"{missing_mask=}")
        self.logger.debug(f"{y_test_ohe.shape=}")
        self.logger.debug(f"{y_pred_proba.shape=}")
        self.logger.debug(f"{missing_mask.shape=}")
        self.logger.debug(f"{y_test_ohe[missing_mask].shape=}")
        self.logger.debug(f"{y_pred_proba[missing_mask].shape=}")
        self.logger.debug(f"{y_pred[missing_mask].shape=}")
        self.logger.debug(f"{y_test[missing_mask].shape=}")
        self.logger.debug(f"{y_test[missing_mask]=}")
        self.logger.debug(f"{X_test[missing_mask]=}")

    def _train_final_model(self) -> None:
        """Train the final model using the best hyperparameters.

        This method trains the final model using the best hyperparameters found during hyperparameter optimization. It uses the training and validation data to train the model and calculates the loss during training. If hyperparameter optimization was not performed, the model is trained using the default or user-provided hyperparameters specified in the constructor.
        """
        self.model_ = self.build_model(self.Model, self.model_params)
        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=self.lr_)

        self.logger.info(
            f"Training the model with the following parameters: {self.best_params_}"
        )

        self.best_loss_, self.history_ = self.train_and_validate_model(
            self.train_loader_,
            self.val_loader_,
            self.optimizer_,
            lr_patience=self.lr_patience_,
            l1_penalty=self.l1_penalty_,
            l2_penalty=self.l2_penalty_,
        )

        od = Path(f"{self.prefix}_{self.output_dir}")
        od = od / "models" / "Unsupervised" / self.model_name / "final_model.pt"
        torch.save(self.model_, od)

    def _tune_hyperparameters(self) -> None:
        """Tune hyperparameters using Optuna.

        This method tunes the hyperparameters of the model using Optuna. It optimizes the latent dimension, dropout rate, number of hidden layers, hidden layer sizes, activation function, L1 and L2 regularization penalties, learning rate patience, and learning rate. It uses the training and validation data to optimize the hyperparameters and minimize the loss function.
        """
        study = optuna.create_study(
            direction="minimize", study_name=f"{self.prefix}_VAE"
        )
        study.optimize(
            lambda trial: self.objective(trial, self.Model),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
        )

        self.latent_dim_ = study.best_params["latent_dim"]
        self.dropout_rate_ = study.best_params["dropout_rate"]
        self.num_hidden_layers_ = study.best_params["num_hidden_layers"]

        self.hidden_layer_sizes_ = [
            study.best_params[f"n_units_l{i}"] for i in range(self.num_hidden_layers_)
        ]

        self.activation_ = study.best_params["activation"]
        self.l1_penalty_ = study.best_params["l1_penalty"]
        self.l2_penalty_ = study.best_params["l2_penalty"]
        self.lr_patience_ = study.best_params["lr_patience"]
        self.lr_ = study.best_params["learning_rate"]

        self.best_params_ = {
            "n_features": self.num_features,
            "num_classes": self.num_classes,
            "latent_dim": self.latent_dim_,
            "dropout_rate": self.dropout_rate_,
            "hidden_layer_sizes": self.hidden_layer_sizes_,
            "activation": self.activation_,
        }

        self.model_params.update(self.best_params_)
        self.best_tuned_loss_ = study.best_value

        self.all_tuned_params_ = study.best_params

        od = Path(f"{self.prefix}_{self.output_dir}")
        fn = od / "optimize" / "Unsupervised" / self.model_name
        fn = fn / "best_params.json"

        with open(fn, "w") as fp:
            json.dump(self.all_tuned_params_, fp, indent=4)

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, List[List[int]], torch.Tensor],
        y: Any = None,
    ) -> np.ndarray:
        """Predict the missing values using the trained VAE model.

        This method predicts the missing values in the input data using the trained VAE model. It returns the input data with the imputed values. The imputed values replace the missing values in the input data. The input data should be in the same format as the training data.

        Args:
            X (numpy.ndarray): The input data with missing values. The input data should be in the same format as the training data.
            y (None): Ignored. For compatibility with the scikit-learn API. Defaults to None.

        Returns:
            numpy.ndarray: The input data with imputed values replacing the missing values. The imputed values are in the same format as the input data.
        """
        # Ensure that the input data is in the correct format.
        X = validate_input_type(X, return_type="array")

        self.plotter_.plot_gt_distribution(X)

        X_imputed = self.predict(X)

        self.plotter_.plot_gt_distribution(X_imputed, is_imputed=True)

        return X_imputed


class SAE(BaseNNImputer):
    def __init__(
        self,
        **kwargs,
    ):
        self.num_classes = 4
        self.is_multiclass_ = True if self.num_classes != 4 else False
        self.activate = None
        self.nn_method_ = "SAE"
        self.act_func_ = None
        self.testing = kwargs.get("testing", False)

        super().__init__(
            self.activate,
            self.nn_method_,
            self.num_classes,
            self.act_func_,
            **kwargs,
        )

    def run_sae(
        self,
        y_true,
        y_train,
        model_params,
        compile_params,
        fit_params,
    ):
        """Run standard autoencoder using custom subclassed model.

        Args:
            y_true (numpy.ndarray): Original genotypes (training dataset) with known and missing values of shape (n_samples, n_features).

            y_train (numpy.ndarray): Onehot-encoded genotypes (training dataset) with known and missing values of shape (n_samples, n_features, num_classes.)

            model_params (Dict[str, Any]): Dictionary with parameters to pass to the classifier model.

            compile_params (Dict[str, Any]): Dictionary with parameters to pass to the tensorflow compile function.

            fit_params (Dict[str, Any]): Dictionary with parameters to pass to the fit() function.

        Returns:
            List[tf.keras.Model]: List of keras model objects. One for each phase (len=1 if NLPCA, len=3 if UBP).

            List[Dict[str, float]]: List of dictionaries with best neural network model history.

            Dict[str, Any] or None: Best parameters found during a grid search, or None if a grid search was not run.

            float: Best score obtained during grid search.

            tf.keras.Model: Best model found during grid search.

            sklearn.model_selection object (GridSearchCV, RandomizedSearchCV) or GASearchCV object.

            Dict[str, Any]: Per-class, micro, and macro-averaged metrics including accuracy, ROC-AUC, and Precision-Recall with Average Precision scores.
        """
        scorers = Scorers()
        scoring = None

        histories = list()
        models = list()

        (
            model,
            best_history,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        ) = self.run_clf(
            y_train,
            y_true,
            model_params,
            compile_params,
            fit_params,
            testing=False,
        )

        histories.append(best_history)
        models.append(model)
        del model

        return (
            models,
            histories,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        )


class UBP(BaseNNImputer):
    def __init__(
        self,
        *,
        nlpca=False,
        **kwargs,
    ):
        # TODO: Make estimators compatible with variable number of classes.
        # E.g., with morphological data.
        self.nlpca = nlpca
        self.nn_method_ = "NLPCA" if self.nlpca else "UBP"
        self.num_classes = 4
        self.is_multiclass_ = True if self.num_classes != 4 else False
        self.testing = kwargs.get("testing", False)
        self.activate = None
        self.act_func_ = None

        super().__init__(
            self.activate,
            self.nn_method_,
            self.num_classes,
            self.act_func_,
            **kwargs,
            nlpca=self.nlpca,
        )

    def run_nlpca(
        self,
        y_true,
        y_train,
        model_params,
        compile_params,
        fit_params,
    ):
        """Run NLPCA using custom subclassed model.

        Args:
            y_true (numpy.ndarray): Original genotypes with known and missing values.

            y_train (numpy.ndarray): For compatibility with VAE and SAE. Not used.

            model_params (Dict[str, Any]): Dictionary with parameters to pass to the classifier model.

            compile_params (Dict[str, Any]): Dictionary with parameters to pass to the tensorflow compile function.

            fit_params (Dict[str, Any]): Dictionary with parameters to pass to the fit() function.

        Returns:
            List[tf.keras.Model]: List of keras model objects. One for each phase (len=1 if NLPCA, len=3 if UBP).

            List[Dict[str, float]]: List of dictionaries with best neural network model history.

            Dict[str, Any] or None: Best parameters found during a grid search, or None if a grid search was not run.

            float: Best score obtained during grid search.

            tf.keras.Model: Best model found during grid search.

            sklearn.model_selection object (GridSearchCV, RandomizedSearchCV) or GASearchCV object.

            Dict[str, Any]: Per-class, micro, and macro-averaged metrics including accuracy, ROC-AUC, and Precision-Recall with Average Precision scores.
        """
        scorers = Scorers()

        histories = list()
        models = list()
        y_train = model_params.pop("y_train")
        ubp_weights = None
        phase = None

        (
            V,
            model,
            best_history,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        ) = self.run_clf(
            y_train,
            y_true,
            model_params,
            compile_params,
            fit_params,
            ubp_weights=ubp_weights,
            phase=phase,
            testing=False,
        )

        histories.append(best_history)
        models.append(model)
        del model

        return (
            models,
            histories,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        )

    def run_ubp(
        self,
        y_true,
        y_train,
        model_params,
        compile_params,
        fit_params,
    ):
        """Run UBP using custom subclassed model.

        Args:
            y_true (numpy.ndarray): Original genotypes with known and missing values.

            y_train (numpy.ndarray): For compatibility with VAE and SAE. Not used.

            model_params (Dict[str, Any]): Dictionary with parameters to pass to the classifier model.

            compile_params (Dict[str, Any]): Dictionary with parameters to pass to the tensorflow compile function.

            fit_params (Dict[str, Any]): Dictionary with parameters to pass to the fit() function.

        Returns:
            List[tf.keras.Model]: List of keras model objects. One for each phase (len=1 if NLPCA, len=3 if UBP).

            List[Dict[str, float]]: List of dictionaries with best neural network model history.

            Dict[str, Any] or None: Best parameters found during a grid search, or None if a grid search was not run.

            float: Best score obtained during grid search.

            tf.keras.Model: Best model found during grid search.

            sklearn.model_selection object (GridSearchCV, RandomizedSearchCV) or GASearchCV object.

            Dict[str, Any]: Per-class, micro, and macro-averaged metrics including accuracy, ROC-AUC, and Precision-Recall with Average Precision scores.
        """
        scorers = Scorers()

        histories = list()
        models = list()
        search_n_components = False

        y_train = model_params.pop("y_train")

        if self.run_gridsearch_:
            # Cannot do CV because there is no way to use test splits
            # given that the input gets refined. If using a test split,
            # then it would just be the randomly initialized values and
            # would not accurately represent the model.
            # Thus, we disable cross-validation for the grid searches.
            scoring = scorers.make_multimetric_scorer(
                self.scoring_metrics_,
                self.sim_missing_mask_,
                num_classes=self.num_classes,
            )

            if "n_components" in self.gridparams:
                search_n_components = True
                n_components_searched = self.n_components
        else:
            scoring = None

        for phase in range(1, 4):
            ubp_weights = models[1].get_weights() if phase == 3 else None

            (
                V,
                model,
                best_history,
                best_params,
                best_score,
                best_clf,
                search,
                metrics,
            ) = self.run_clf(
                y_train,
                y_true,
                model_params,
                compile_params,
                fit_params,
                ubp_weights=ubp_weights,
                phase=phase,
                testing=False,
            )

            if phase == 1:
                # Cannot have V input with different n_components
                # in other phases than are in phase 1.
                # So the n_components search has to happen in phase 1.
                if best_params is not None and search_n_components:
                    n_components_searched = best_params["n_components"]
                    model_params["V"] = {n_components_searched: model.V_latent.copy()}
                    model_params["n_components"] = n_components_searched
                    self.n_components = n_components_searched
                    self.gridparams.pop("n_components")

                else:
                    model_params["V"] = V
            elif phase == 2:
                model_params["V"] = V

            elif phase == 3:
                if best_params is not None and search_n_components:
                    best_params["n_components"] = n_components_searched

            histories.append(best_history)
            models.append(model)
            del model

        return (
            models,
            histories,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        )

    def _initV(self, y_train, search_mode):
        """Initialize random input V as dictionary of numpy arrays.

        Args:
            y_train (numpy.ndarray): One-hot encoded training dataset (actual data).

            search_mode (bool): Whether doing grid search.

        Returns:
            Dict[int, numpy.ndarray]: Dictionary with n_components: V as key-value pairs.

        Raises:
            ValueError: Number of components must be >= 2.
        """
        vinput = dict()
        if search_mode:
            if "n_components" in self.gridparams:
                n_components = self.gridparams["n_components"]
            else:
                n_components = self.n_components

            if not isinstance(n_components, int):
                if min(n_components) < 2:
                    raise ValueError(
                        f"n_components must be >= 2, but a value of {n_components} was specified."
                    )

                elif len(n_components) == 1:
                    vinput[n_components[0]] = self.nn_.init_weights(
                        y_train.shape[0], n_components[0]
                    )

                else:
                    for c in n_components:
                        vinput[c] = self.nn_.init_weights(y_train.shape[0], c)
            else:
                vinput[self.n_components] = self.nn_.init_weights(
                    y_train.shape[0], self.n_components
                )

        else:
            vinput[self.n_components] = self.nn_.init_weights(
                y_train.shape[0], self.n_components
            )

        return vinput
