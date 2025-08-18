import json
from pathlib import Path
from typing import Any, Generator, List, Tuple

import numpy as np
import optuna
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from snpio.utils.logging import LoggerManager

from pgsui.data_processing.transformers import SimGenotypeDataTransformer
from pgsui.utils.plotting import Plotting
from pgsui.utils.scorers import Scorer


class BaseSupervisedImputer(BaseEstimator, TransformerMixin):
    """Base class for supervised imputation models.

    This class defines the base interface for supervised imputation models. The class inherits from the scikit-learn BaseEstimator and TransformerMixin classes and provides methods for fitting and transforming data. The class also provides methods for setting and getting the model parameters. Subclasses must implement the fit and transform methods to define the model behavior.
    """

    def __init__(
        self,
        *,
        prefix: str = "pgsui",
        output_dir: str | Path = "output",
        verbose: int = 0,
        debug: bool = False,
    ):
        """Initializes the BaseSupervisedImputer with a logger.

        This class initializes the BaseSupervisedImputer with a logger using the LoggerManager class. The logger is used to log messages during the model fitting and transformation processes.

        Args:
            prefix (str): Prefix for the logger. Default is 'pgsui'.
            output_dir (str | Path): Output directory for logging. Default is 'output'.
            verbose (int): Verbosity level for logging. Default is 0.
            debug (bool): Debug mode flag. Default is False.
        """
        self.prefix = prefix
        self.output_dir = output_dir
        self.verbose = verbose

        # Prepare directory structure
        outdirs = ["models", "plots", "metrics", "optimize"]
        self._create_model_directories(prefix, output_dir, outdirs)
        self.debug = debug if verbose < 2 else True

        # Initialize logger
        kwargs = {"prefix": prefix, "verbose": verbose >= 1, "debug": debug}
        logman = LoggerManager(__name__, **kwargs)
        self.logger = logman.get_logger()

    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model using the given data.

        This method trains the model using the given data. It takes the input data X and the target data y as input and fits the model to the data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Target data.

        Raises:
            AttributeError: If the model attribute is not defined.
            AttributeError: If the model parameters are not defined.
        """
        if not hasattr(self, "Model"):
            msg = "Model attribute must be defined in the child class."
            self.logger.error(msg)
            raise AttributeError(msg)

        if not hasattr(self, "model_params"):
            msg = "Model parameters must be defined in the child class."
            self.logger.error(msg)
            raise AttributeError(msg)

        self.logger.info("Training model...")

        if self.model_name == "ImputeHistGradientBoosting":
            params = {"warm_start": True}
            params["categorical_features"] = None
            params["validation_fraction"] = 0.0
        else:
            params = {
                "n_estimators": self.n_estimators,
                "n_jobs": self.n_jobs,
                "warm_start": True,
            }

        Model = IterativeImputer(
            self.Model(**params, **self.model_params),
            max_iter=self.max_iter,
            n_nearest_features=self.n_nearest_features,
            tol=1e-2,
            initial_strategy="most_frequent",
            random_state=self.seed,
            keep_empty_features=True,
            skip_complete=True,
            verbose=self.verbose,
        )

        Model.fit(X, y)
        return Model

    def evaluate_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        objective_mode: bool = False,
        trial: optuna.Trial | None = None,
        Model: Any | None = None,
    ) -> None:
        """Evaluate the model using the given data.

        This method evaluates the model using the given data. It takes the input data X and the target data y as input and evaluates the model using the data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Target data.
            mask (np.ndarray): Mask for valid data.
            objective_mode (bool): Flag to indicate if the evaluation is in objective mode. Default is False.
            trial (optuna.Trial): Optuna trial object. Default is None.
            Model (Any): Model object. Default is None.

        Raises:
            ValueError: If the model is not fitted yet.
        """
        if not hasattr(self, "Model"):
            msg = "Model attribute must be defined in the child class."
            self.logger.error(msg)
            raise ValueError(msg)

        if not hasattr(self, "model_params"):
            msg = "Model parameters must be defined in the child class."
            self.logger.error(msg)
            raise ValueError(msg)

        if objective_mode and trial is None:
            msg = "Optuna trial object must be provided in objective mode."
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("Evaluating model...")

        y_pred = Model.transform(X)
        y_pred_proba = np.zeros((y_pred.shape[0], self.n_features_, self.num_classes_))

        for column, idx in self._get_probas(X, Model):
            y_pred_proba[:, idx] = column

        y_pred_proba = y_pred_proba[mask]
        y_true_labels = y[mask]
        y_pred_labels = y_pred[mask]

        ohe = OneHotEncoder(
            categories="auto", handle_unknown="ignore", sparse_output=False
        )

        y_true_ohe = ohe.fit_transform(y[mask].reshape(-1, 1))

        metrics = self.scorers_.evaluate(
            y_true_labels,
            y_pred_labels,
            y_true_ohe,
            y_pred_proba,
            objective_mode,
            self.tune_metric,
        )

        self.plotter_.plot_metrics(y_true_labels, y_pred_proba, metrics)
        self.plotter_.plot_confusion_matrix(y_true_labels, y_pred_labels)
        return metrics

    def _get_probas(
        self, X: np.ndarray, Model: Any
    ) -> Generator[np.ndarray, int, None]:
        """Get the predicted probabilities from the model.

        Args:
            X (np.ndarray): Input data.
            Model (Any): Model object.

        Yields:
            np.ndarray: Predicted probabilities.
            int: Feature index.
        """
        for feat_idx, neighbor_feat_idx, estimator in Model.imputation_sequence_:
            if hasattr(estimator, "predict_proba"):
                Xpred = X[:, neighbor_feat_idx].copy()

                si = SimpleImputer(strategy="most_frequent")
                X_pred = si.fit_transform(X_pred)

                # Shape: (n_samples, num_classes_in_col)
                proba = estimator.predict_proba(Xpred)

                # Get actual predicted class labels
                classes_in_col = estimator.classes_
                full_proba = np.zeros((proba.shape[0], self.num_classes_))

                for i, clsses in enumerate(classes_in_col):
                    cls_int = int(clsses)
                    # Assign probability to correct class index.
                    full_proba[:, cls_int] = proba[:, i]

                yield full_proba, feat_idx
            else:
                msg = f"Estimator {estimator} does not have a `predict_proba` method."
                self.logger.error(msg)
                raise AttributeError(msg)

    def split_data(self, X, y, mask):
        """Split the data into training and validation sets.

        This method splits the input data into training and validation sets using the given validation split ratio.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Target data.
            mask (np.ndarray): Mask for valid data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training, validation, and test sets and masks.
        """
        self.logger.info("Splitting data into training and validation sets...")
        args = (X, y, mask)
        kwargs = {"test_size": self.validation_split, "random_state": self.seed}
        return train_test_split(*args, **kwargs)

    def _create_model_directories(
        self, prefix: str, output_dir: str | Path, outdirs: List[str]
    ) -> None:
        """Create the necessary directories for the model outputs.

        This method creates the necessary directories for the model outputs, including directories for models, plots, metrics, and optimization results.

        Args:
            prefix (str): Prefix for the output directory.
            output_dir (str | Path): Output directory name.
            outdirs (List[str]): List of subdirectories to create.

        Raises:
            Exception: If any of the directories cannot be created.
        """
        self.logger.debug(
            f"Creating model directories in {output_dir} with prefix {prefix}."
        )
        self.formatted_output_dir = Path(f"{prefix}_{output_dir}")
        self.base_dir = self.formatted_output_dir / "Supervised"

        for d in outdirs:
            subdir = self.base_dir / d / self.model_name
            setattr(self, f"{d}_dir", subdir)
            try:
                getattr(self, f"{d}_dir").mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errdir = getattr(self, f"{d}_dir")
                msg = f"Failed to create directory {errdir}: {e}"
                self.logger.error(msg)
                raise e

    def tune_hyperparameters(self) -> None:
        """Tune hyperparameters using Optuna study.

        This method tunes the hyperparameters of the model using Optuna. It creates an Optuna study and optimizes the model hyperparameters using the `objective()` method. The method saves the best hyperparameters to a JSON file and plots the optimization results. The method returns the best hyperparameters found during the optimization. The method also raises an error if the `objective()` or `set_best_params()` methods are not implemented in the child class. In the `set_best_params()` method, the child class should set the best hyperparameters for the model based on the results of hyperparameter tuning.

        Returns:
            dict: Dictionary of best hyperparameters.

        Raises:
            ValueError: If the model is not fitted yet.
            NotImplementedError: If the `objective()` method is not implemented in the child class.
            NotImplementedError: If the `set_best_params()` method is not implemented in the child class.
        """
        if not hasattr(self, "plotter_"):
            msg = "Transformers must be initialized before tuning hyperparameters. Please run `init_transformers()` in `fit()` before calling this method."
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("Tuning hyperparameters...")

        study_db = None
        load_if_exists = False
        if self.tune_save_db:
            study_db = self.optimize_dir / "study_database" / "optuna_study.db"

            if not study_db.parent.exists():
                study_db.parent.mkdir(parents=True, exist_ok=True)

            should_load = self.tune_resume and study_db.exists()
            load_if_exists = True if should_load else False

            if not load_if_exists:
                study_db.unlink()

        stnm = f"{self.prefix}_{self.model_name} Model Optimization"
        strg = f"sqlite:///{study_db}" if self.tune_save_db else None

        kwargs = {"direction": "maximize", "study_name": stnm, "storage": strg}
        study = optuna.create_study(load_if_exists=load_if_exists, **kwargs)

        study.optimize(
            lambda trial: self.objective(trial, self.Model),
            n_trials=self.tune_n_trials,
            n_jobs=self.n_jobs,
        )

        best_metric = study.best_value
        best_params = study.best_params

        self.logger.info(f"Best {self.tune_metric} metric: {best_metric}")

        # Save best parameters to a JSON file.
        fn = self.optimize_dir / "parameters" / "best_params.json"

        # Save best parameters to a JSON file.
        if not fn.parent.exists():
            fn.parent.mkdir(parents=True, exist_ok=True)

        with open(fn, "w") as fp:
            json.dump(best_params, fp, indent=4)

        # Plot optimization results.
        tn = f"{self.tune_metric} Value"
        self.plotter_.plot_tuning(study, self.model_name, target_name=tn)

        return best_params

    def init_transformers(self) -> Tuple[SimGenotypeDataTransformer, Plotting, Scorer]:
        """Initialize the transformers for encoding.

        This method should be called in a `fit` method to initialize the transformers. It returns the transformers and utilities. The method should be called before fitting the model. The method initializes the SimGenotypeDataTransformer, Plotting, and Scorer classes. The SimGenotypeDataTransformer class is used to transform the genotype data, the Plotting class is used to plot the results, and the Scorer class is used to compute the scoring metrics.

        Returns:
            Tuple[SimGenotypeDataTransformer, Plotting, Scorer]: Transformers and utilities.
        """
        if not hasattr(self, "class_weights_"):
            msg = "`class_weights_` must be defined in the child class. Please run `compute_class_weights()` before calling this method."
            self.logger.error(msg)
            raise ValueError(msg)

        sim_transformer = SimGenotypeDataTransformer(
            self.genotype_data,
            prop_missing=self.sim_prop_missing,
            strategy=self.sim_strategy,
            seed=self.seed,
            logger=self.logger,
            verbose=self.verbose,
            debug=self.debug,
            class_weights=self.class_weights_,
        )

        # Initialize plotter.
        plotter = Plotting(
            model_name=self.model_name,
            prefix=self.prefix,
            output_dir=self.output_dir,
            plot_format=self.plot_format,
            plot_fontsize=self.plot_fontsize,
            plot_dpi=self.plot_dpi,
            title_fontsize=self.title_fontsize,
            despine=self.despine,
            show_plots=self.show_plots,
            verbose=self.verbose,
            debug=self.debug,
        )

        # Metrics
        scorers = Scorer(
            average=self.scorer_averaging,
            logger=self.logger,
            verbose=self.verbose,
            debug=self.debug,
        )

        return sim_transformer, plotter, scorers

    def objective(self, trial, Model):
        """Objective function for Optuna hyperparameter tuning.

        This method is the objective function for hyperparameter tuning using Optuna. It trains the model using the given hyperparameters and returns the validation loss.

        Args:
            trial (optuna.Trial): Optuna trial object.
            Model (torch.nn.Module): The model to train.

        Returns:
            float: Validation loss.
        """
        msg = "Method `objective()` must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def set_best_params(self, best_params):
        """Set the best parameters for the model.

        This method sets the best hyperparameters for the model based on the results of hyperparameter tuning.

        Args:
            best_params (dict): Dictionary of best hyperparameters.

        Returns:
            dict: Dictionary of best hyperparameters.
        """
        msg = "Method `set_best_params()` must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseSupervisedImputer":
        """Fits the model to the data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Target data.

        Returns:
            BaseSupervisedImputer: The fitted model.
        """
        msg = "Method `fit()` must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms the input data.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: The transformed data.
        """
        msg = "Method `transform()` must be implemented in the child class."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def _check_is_fitted(self):
        """Check if the model is fitted.

        This method checks if the model is fitted by checking if the necessary attributes are present. If the model is not fitted, the method raises an AttributeError. The method checks if the model attribute, number of features, number of classes, best hyperparameters, model metrics, and original missing mask are defined.

        Raises:
            AttributeError: If the model is not fitted yet.
            AttributeError: If the model attribute is not defined.
            AttributeError: If the number of features is not defined.
            AttributeError: If the number of classes is not defined.
            AttributeError: If the best hyperparameters are not defined.
            AttributeError: If the model metrics are not defined.
            AttributeError: If the original missing mask is not defined.
        """
        if not self.is_fit:
            msg = "Model not fitted. Call the `fit()` method before imputing."
            self.logger.error(msg)
            raise AttributeError(msg)

        if not hasattr(self, "Model_"):
            msg = "Model attribute must be defined in the `fit()` method of the child class."
            self.logger.error(msg)
            raise AttributeError(msg)

        if not hasattr(self, "n_features_"):
            msg = "Number of features not found. Model was not fitted."
            self.logger.error(msg)
            raise AttributeError(msg)

        if not hasattr(self, "num_classes_"):
            msg = "Number of classes not found. Model was not fitted."
            self.logger.error(msg)
            raise AttributeError(msg)

        if self.tune and not hasattr(self, "best_params_"):
            msg = "Best hyperparameters not found. Must set best hyperparameters in `fit()`."
            self.logger.error(msg)
            raise AttributeError(msg)

        if not hasattr(self, "metrics_"):
            msg = "Model metrics not found. Model was not evaluated. Run `evaluate_model()` in `fit()` before imputing."
            self.logger.error(msg)
            raise AttributeError(msg)

        if not hasattr(self, "original_missing_mask_"):
            msg = "Original missing mask not found. This attribute should be set in the `fit()` method."
            self.logger.error(msg)
            raise AttributeError(msg)

    def impute(self, X: np.ndarray, Model: Any) -> np.ndarray:
        """Imputes the genotype data using the random forest imputer.

        This method imputes the genotype data using the random forest imputer fitted during the fit method. The method takes the genotype data X as input and returns the imputed genotype data.

        Args:
            X (np.ndarray): Genotype data array.
            Model (Any): Fitted Model object.

        Returns:
            np.ndarray: Imputed genotype data array.
        """
        imputed = Model.transform(X)
        imputed = np.where(self.original_missing_mask_, imputed, X)
        return imputed

    def compute_class_weights(
        self,
        X: np.ndarray,
        use_log_scale: bool = False,
        alpha: float = 1.0,
        normalize: bool = False,
        temperature: float = 1.0,
        max_weight: float = 10.0,
        min_weight: float = 0.01,
    ) -> np.ndarray:
        """Compute inverse class weights based on class distribution in the data.

        This method computes the inverse class weights based on the class distribution in the data. It optionally applies a log-scale transform and normalizes the weights. The weights are bounded within a range, and a temperature parameter can be used for smoothing. The alpha parameter is used for the log-scale transform to adjust the weights.

        Args:
            X (numpy.ndarray): Input data.
            use_log_scale (bool, optional): Use log-scale transform. Defaults to True.
            alpha (float, optional): Alpha parameter for log-scale transform. Defaults to 1.0.
            normalize (bool, optional): Normalize the weights. Defaults to False.
            temperature (float, optional): Temperature parameter for smoothing. Defaults to 1.0.
            max_weight (float, optional): Maximum weight. Defaults to 20.0.
            min_weight (float, optional): Minimum weight. Defaults to 0.5.

        Returns:
            np.ndarray: Class weights.
        """
        self.logger.info("Computing class weights...")

        # Flatten the array to calculate class frequencies
        X = np.where(np.logical_or(X < 0, np.isnan(X)), -1, X)
        flat_X = X.flatten()
        valid_classes = flat_X[flat_X >= 0]  # Exclude missing values (< 0)

        if valid_classes.size == 0:
            msg = "No valid classes found in the data."
            self.logger.error(msg)
            raise ValueError(msg)

        # Count the occurrences of each class
        unique, counts = np.unique(valid_classes, return_counts=True)
        total_counts = np.sum(counts)

        # Compute raw inverse class weights
        raw_weights = {
            enc: (total_counts / count if count > 0 else 0.0)
            for enc, count in zip(unique, counts)
        }

        # Optionally apply a log-scale transform
        if use_log_scale:
            raw_weights = {enc: np.log(alpha + raw_weights[enc]) for enc in raw_weights}

        mean_val = np.mean(list(raw_weights.values()))
        raw_weights = {enc: raw_weights[enc] / mean_val for enc in raw_weights}

        # Bound the weights within [min_weight, max_weight]
        bounded_weights = {}
        for enc in raw_weights:
            w = raw_weights[enc]
            w = max(min_weight, min(max_weight, w))
            bounded_weights[enc] = w

        # Convert final weights to a tensor
        final_weight_list = [
            bounded_weights.get(class_index, 1.0)
            for class_index in range(self.num_classes_)
        ]
        final_weight_list = np.array(final_weight_list, dtype=np.float32)

        # Smooth class weights
        weight_arr = np.pow(final_weight_list, 1 / temperature)

        if normalize:
            weight_arr = weight_arr / weight_arr.sum()

        cw = weight_arr
        self.logger.info("Class weights:")
        [self.logger.info(f"Class {i}: {x:.3f}") for i, x in enumerate(cw)]

        return weight_arr
