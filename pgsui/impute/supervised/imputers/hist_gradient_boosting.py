import copy
from pathlib import Path
from pprint import pformat
from typing import Any, Literal

import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from snpio.utils.logging import LoggerManager

from pgsui.impute.supervised.base_supervised import BaseSupervisedImputer
from pgsui.utils.misc import validate_input_type


class ImputeHistGradientBoosting(BaseSupervisedImputer):
    """Histogram Gradient Boosting imputer for genotype data.

    This class implements a HistGradientBoosting imputer for genotype data. The class inherits from the BaseSupervisedImputer class and provides methods for fitting and transforming genotype data using a HistGradientBoosting model. The class uses the scikit-learn model for imputation.
    """

    def __init__(
        self,
        genotype_data: Any = None,
        *,
        prefix: str = "pgsui",
        output_dir: str | Path = "output",
        verbose: int = 0,
        n_jobs: int = 1,
        seed: int = 42,
        tune: bool = False,
        tune_metric: str = "pr_macro",
        tune_n_trials: int = 100,
        tune_save_db: bool = False,
        sim_prop_missing: float = 0.1,
        sim_strategy: str = "random_inv_multinom",
        scorer_averaging: Literal["macro", "micro", "weighted"] = "weighted",
        model_n_nearest_features: int = 5,
        model_max_iter: int = 10,
        model_max_model_iter: int = 100,
        model_learning_rate: float = 0.1,
        model_loss: str = "log_loss",
        model_early_stopping: bool = True,
        model_min_samples_leaf: int = 20,
        model_l2_regularization: float = 0.0,
        model_max_depth: int = 10,
        model_max_features: str = "auto",
        model_validation_split: float = 0.2,
        plot_format: str = "pdf",
        plot_fontsize: int | float = 18,
        plot_dpi: int | float = 300,
        plot_title_fontsize=20,
        plot_despine=True,
        plot_show_plots=False,
        debug: bool = False,
    ):
        """Initializes the imputer with a logger.

        This class initializes the imputer with a logger using the LoggerManager class. The logger is used to log messages during the model fitting and transformation processes.

        Args:
            genotype_data (Any): Genotype data object.
            prefix (str): Prefix for the logger.
            output_dir (str | Path): Output directory for the logger.
            verbose (int): Verbosity level for the logger.
            n_jobs (int): Number of jobs to run in parallel.
            seed (int): Random seed for reproducibility.
            tune (bool): Whether to tune hyperparameters.
            tune_metric (str): Metric to optimize during hyperparameter tuning.
            tune_n_trials (int): Number of trials for hyperparameter tuning.
            tune_save_db (bool): Whether to save the optuna database.
            sim_prop_missing (float): Proportion of missing data to simulate.
            sim_strategy (str): Strategy for simulating missing data.
            scorer_averaging (str): Averaging strategy for scoring metrics.
            model_n_nearest_features (int): Number of nearest features for imputation.
            model_max_iter (int): Maximum number of iterations for imputation.
            model_max_model_iter (int): Maximum number of iterations for model training.
            model_learning_rate (float): Learning rate for model training.
            model_loss (str): Loss function for model training.
            model_early_stopping (bool): Whether to use early stopping for model training.
            model_min_samples_leaf (int): Minimum number of samples for leaf nodes.
            model_l2_regularization (float): L2 regularization parameter.
            model_max_depth (int): Maximum depth of the model.
            model_max_features (str): Maximum number of features to consider for splitting.
            model_validation_split (float): Validation split for model evaluation.
            plot_format (str): Format for saving plots.
            plot_fontsize (int | float): Font size for plots.
            plot_dpi (int | float): DPI for saving plots.
            plot_title_fontsize (int): Font size for plot titles.
            plot_despine (bool): Whether to remove spines from plots.
            plot_show_plots (bool): Whether to show plots.
            debug (bool): Whether to run in debug mode.
        """
        self.model_name = "ImputeHistGradientBoosting"
        self.Model = HistGradientBoostingClassifier

        logman = LoggerManager(
            __name__, prefix=prefix, verbose=verbose >= 1, debug=debug
        )
        self.logger = logman.get_logger()

        super().__init__(
            prefix=prefix, output_dir=output_dir, verbose=verbose, debug=debug
        )

        self.genotype_data = genotype_data
        self.n_jobs = n_jobs
        self.seed = seed
        self.n_nearest_features = model_n_nearest_features
        self.max_iter = model_max_iter
        self.tune = tune
        self.tune_metric = tune_metric
        self.tune_n_trials = tune_n_trials
        self.tune_save_db = tune_save_db
        self.sim_prop_missing = sim_prop_missing
        self.sim_strategy = sim_strategy
        self.max_depth = model_max_depth
        self.max_features = model_max_features
        self.loss = model_loss
        self.max_model_iter = model_max_model_iter
        self.model_early_stopping = model_early_stopping
        self.model_min_samples_leaf = model_min_samples_leaf
        self.model_l2_regularization = model_l2_regularization
        self.learning_rate = model_learning_rate
        self.validation_split = model_validation_split
        self.plot_format = plot_format
        self.plot_fontsize = plot_fontsize
        self.plot_dpi = plot_dpi
        self.title_fontsize = plot_title_fontsize
        self.despine = plot_despine
        self.show_plots = plot_show_plots
        self.scorer_averaging = scorer_averaging

        # Model parameters
        # Model parameters
        self.model_params = {
            "loss": model_loss,
            "max_iter": model_max_model_iter,
            "max_depth": model_max_depth,
            "min_samples_leaf": model_min_samples_leaf,
            "l2_regularization": model_l2_regularization,
            "learning_rate": model_learning_rate,
            "max_features": model_max_features,
            "early_stopping": model_early_stopping,
            "random_state": self.seed,
        }

        self.is_fit_ = False

        _ = genotype_data.snp_data

    def fit(self, X: np.ndarray, y: None = None) -> "ImputeHistGradientBoosting":
        """Fits the imputer to the genotype data.

        This method fits the imputer to the genotype data using the scikit-learn model. The method takes the genotype data X and the target labels y as input and fits the model to the data.

        Args:
            X (np.ndarray): Genotype data array.
            y (np.ndarray): Ignored. Only for compatibility with scikit-learn API.

        Returns:
            ImputeHistGradientBoosting: Fitted imputer.
        """
        self.logger.info(f"Fitting {self.model_name} imputer to genotype data.")

        X = validate_input_type(X)
        y = X.copy()

        self.n_features_ = X.shape[1]
        self.num_classes_ = len(np.unique(y[y >= 0 & ~np.isnan(y)]))

        self.class_weights_ = self.compute_class_weights(X)
        self.class_weights_ = validate_input_type(self.class_weights_)

        transformers = self.init_transformers()
        self.sim_, self.plotter_, self.scorers_ = transformers

        Xenc = np.where(np.logical_or(X < 0, np.isnan(X)), -1, X)
        yenc = np.where(np.logical_or(y < 0, np.isnan(y)), np.nan, y)

        Xsim, missing_masks = self.sim_.fit_transform(Xenc)
        self.sim_missing_mask_ = missing_masks["simulated"]
        self.original_missing_mask_ = missing_masks["original"]
        self.all_missing_mask_ = missing_masks["all"]

        Xsim = np.where(Xsim < 0, np.nan, Xsim)
        Xenc = np.where(Xenc < 0, np.nan, Xenc)
        data = self.split_data(Xsim, yenc, self.sim_missing_mask_)

        (
            self.X_train_,
            self.X_test_,
            self.y_train_,
            self.y_test_,
            self.sim_mask_train_,
            self.sim_mask_test_,
        ) = data

        if self.tune:
            best_params = self.tune_hyperparameters()

            # Reinitialize model with best hyperparameters.
            self.best_params_ = self.set_best_params(best_params)
            self.model_params.update(self.best_params_)

            # Log best hyperparameters.
            self.logger.info("Best parameters:")
            best_params_tmp = copy.deepcopy(best_params)
            best_params_tmp["n_nearest_features"] = self.n_nearest_features
            best_params_tmp["max_iter"] = self.max_iter
            best_params_fmt = pformat(best_params_tmp, indent=4).split("\n")
            [self.logger.info(p) for p in best_params_fmt]
        else:
            self.best_params_ = self.model_params

        self.Model_ = self.train_model(self.X_train_, self.y_train_)

        self.logger.info(f"Model fitted.")

        self.metrics_ = self.evaluate_model(
            self.X_test_, self.y_test_, self.sim_mask_test_, Model=self.Model_
        )

        self.logger.info(f"Model evaluation complete.")

        self.is_fit_ = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms the genotype data using the imputer.

        This method transforms the genotype data using the imputer fitted during the fit method. The method takes the genotype data X as input and returns the imputed genotype data.

        Args:
            X (np.ndarray): Genotype data array.

        Returns:
            np.ndarray: Imputed genotype data array.
        """
        # Check if model is fitted.
        self._check_is_fitted()

        X = validate_input_type(X)
        Xenc = np.where(np.logical_or(X < 0, np.isnan(X)), np.nan, X)
        X_imputed = self.impute(Xenc, self.Model_)

        self.plotter_.plot_gt_distribution(X, is_imputed=False)
        self.plotter_.plot_gt_distribution(X_imputed, is_imputed=True)

        return X_imputed

    def objective(self, trial, Model):
        """Objective function for hyperparameter tuning.

        This method defines the objective function for hyperparameter tuning using the Optuna library. The method takes a trial object and a model object as input and returns the loss value for the model.

        Args:
            trial: Optuna trial object.
            Model: Model object.

        Returns:
            float: Loss value for the model.
        """
        # Hyperparameter sampling
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
        loss = trial.suggest_categorical("loss", ["log_loss", "binary_crossentropy"])
        max_model_iter = trial.suggest_int("max_model_iter", 100, 1000, step=100)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        n_nearest_features = trial.suggest_int("n_nearest_features", 3, 20)
        max_iter = trial.suggest_int("max_iter", 5, 30, step=5)
        max_features = trial.suggest_float("max_features", 0.1, 1.0, step=0.1)
        early_stopping = trial.suggest_categorical("early_stopping", [True, False])
        l2_regularization = trial.suggest_float(
            "l2_regularization", 0.0, 0.2, step=0.001
        )
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 50, step=10)

        categorical_features = np.ones((self.n_features_,), dtype=bool)

        # Model parameters
        model_params = {
            "loss": loss,
            "max_iter": max_model_iter,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "max_features": max_features,
            "l2_regularization": l2_regularization,
            "min_samples_leaf": min_samples_leaf,
            "categorical_features": categorical_features,
            "early_stopping": early_stopping,
            "random_state": self.seed,
        }

        try:
            imputer = IterativeImputer(
                Model(
                    validation_fraction=0.0, warm_start=True, n_jobs=1, **model_params
                ),
                max_iter=max_iter,
                tol=1e-2,
                n_nearest_features=n_nearest_features,
                initial_strategy="most_frequent",
                random_state=self.seed,
                verbose=self.verbose,
                keep_empty_features=True,
                skip_complete=True,
            )

            imputer.fit(self.X_train_)

            if imputer is None:
                msg = f"Trial {trial.number} pruned due to failed model training. Model was NoneType."
                self.logger.warning(msg)
                raise optuna.exceptions.TrialPruned()

            # Efficient evaluation
            metrics = self.evaluate_model(
                self.X_test_,
                self.y_test_,
                self.sim_mask_test_,
                objective_mode=True,
                trial=trial,
                Model=imputer,
            )

            if self.tune_metric not in metrics:
                msg = f"Invalid tuning metric: {self.tune_metric}"
                self.logger.error(msg)
                raise KeyError(msg)

            score = metrics[self.tune_metric]

            trial.report(score, step=trial.number)

            if trial.should_prune():
                msg = f"Trial {trial.number} pruned based on optuna report."
                self.logger.warning(msg)
                raise optuna.exceptions.TrialPruned()

            return metrics[self.tune_metric]

        except Exception as e:
            msg = f"Trial {trial.number} pruned due to exception: {e}"
            self.logger.warning(msg)
            raise optuna.exceptions.TrialPruned()

    def set_best_params(self, best_params: dict) -> dict:
        """Sets the best hyperparameters for the model.

        This method sets the best hyperparameters for the model based on the hyperparameter tuning results.

        Args:
            best_params: Best hyperparameters for the model.

        Returns:
            dict: Best hyperparameters for the model.
        """
        best_params_fit = {
            "loss": best_params["loss"],
            "max_iter": best_params["max_model_iter"],
            "learning_rate": best_params["learning_rate"],
            "early_stopping": best_params["early_stopping"],
            "min_samples_leaf": best_params["min_samples_leaf"],
            "l2_regularization": best_params["l2_regularization"],
            "max_depth": best_params["max_depth"],
            "max_features": best_params["max_features"],
            "criterion": best_params["criterion"],
            "random_state": self.seed,
        }

        self.n_nearest_features = best_params_fit.pop("n_nearest_features")
        self.max_iter = best_params_fit.pop("max_iter")

        return best_params_fit
