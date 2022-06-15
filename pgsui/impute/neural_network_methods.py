import logging
import math
import os
import sys
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

# Grid search imports
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scikeras.wrappers import KerasClassifier

# Genetic algorithm grid search imports
from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import ConsecutiveStopping, ProgressBar

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").disabled = True
warnings.filterwarnings("ignore", category=UserWarning)

# noinspection PyPackageRequirements
import tensorflow as tf
from tqdm.keras import TqdmCallback

# Disable can't find cuda .dll errors. Also turns of GPU support.
tf.config.set_visible_devices([], "GPU")

from tensorflow.python.util import deprecation

# Disable warnings and info logs.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

# Monkey patching deprecation utils to supress warnings.
# noinspection PyUnusedLocal
def deprecated(
    date, instructions, warn_once=True
):  # pylint: disable=unused-argument
    def deprecated_wrapper(func):
        return func

    return deprecated_wrapper


deprecation.deprecated = deprecated

try:
    from .scorers import Scorers
    from .vae_model import VAEModel
    from .nlpca_model import NLPCAModel
    from .ubp_model import UBPPhase1, UBPPhase2, UBPPhase3
    from ..data_processing.transformers import (
        UBPInputTransformer,
        VAEInputTransformer,
        MLPTargetTransformer,
    )
except (ModuleNotFoundError, ValueError):
    from impute.scorers import Scorers
    from impute.vae_model import VAEModel
    from impute.nlpca_model import NLPCAModel
    from impute.ubp_model import UBPPhase1, UBPPhase2, UBPPhase3
    from data_processing.transformers import (
        UBPInputTransformer,
        VAEInputTransformer,
        MLPTargetTransformer,
    )


class DisabledCV:
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        yield (np.arange(len(X)), np.arange(len(y)))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


class VAEClassifier(KerasClassifier):
    """Estimator to be used with the scikit-learn API.

    Args:
        y_decoded (numpy.ndarray): Original target data, y, that is not one-hot encoded. Should have shape (n_samples, n_features). Should be 012-encoded. Defaults to None.

        y (numpy.ndarray): One-hot encoded target data. Defaults to None.

        batch_size (int): Batch size to train with. Defaults to 32.

        missing_mask (np.ndarray): Missing mask with missing values set to False (0) and observed values as True (1). Defaults to None. Defaults to None.

        output_shape (int): Number of units in model output layer. Defaults to None.

        weights_initializer (str): Kernel initializer to use for model weights. Defaults to "glorot_normal".

        hidden_layer_sizes (List[int]): Output unit size for each hidden layer. Should be list of length num_hidden_layers. Defaults to None.

        num_hidden_layers (int): Number of hidden layers to use. Defaults to 1.

        hidden_activation (str): Hidden activation function to use. Defaults to "elu".

        l1_penalty (float): L1 regularization penalty to use to reduce overfitting. Defautls to 0.01.

        l2_penalty (float): L2 regularization penalty to use to reduce overfitting. Defaults to 0.01.

        dropout_rate (float): Dropout rate for each hidden layer to reduce overfitting. Defaults to 0.2.

        num_classes (int): Number of classes in output predictions. Defaults to 1.

        n_components (int): Number of components to use for input V. Defaults to 3.

        search_mode (bool): Whether in grid search mode (True) or single estimator mode (False). Defaults to True.

        optimizer (str or callable): Optimizer to use during training. Should be either a str or tf.keras.optimizers callable. Defaults to "adam".

        loss (str or callable): Loss function to use. Should be a string or a callable if using a custom loss function. Defaults to "binary_crossentropy".

        metrics (List[Union[str, callable]]): Metrics to use. Should be a sstring or a callable if using a custom metrics function. Defaults to "accuracy".

        epochs (int): Number of epochs to train over. Defaults to 100.

        verbose (int): Verbosity mode ranging from 0 - 2. 0 = silent, 2 = most verbose.

        kwargs (Any): Other keyword arguments to route to fit, compile, callbacks, etc. Should have the routing prefix (e.g., optimizer__learning_rate=0.01).
    """

    def __init__(
        self,
        y_train,
        y_original,
        model_params,
        batch_size=32,
        missing_mask=None,
        output_shape=None,
        weights_initializer="glorot_normal",
        hidden_layer_sizes=None,
        num_hidden_layers=1,
        hidden_activation="elu",
        l1_penalty=0.01,
        l2_penalty=0.01,
        dropout_rate=0.2,
        num_classes=3,
        sample_weight=None,
        n_components=3,
        optimizer="adam",
        loss="categorical_crossentropy",
        epochs=100,
        verbose=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.y_train = y_train
        self.y_original = y_original
        self.batch_size = batch_size
        self.missing_mask = missing_mask
        self.output_shape = output_shape
        self.weights_initializer = weights_initializer
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.sample_weight = sample_weight
        self.n_components = n_components
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.verbose = verbose

    def _keras_build_fn(self, compile_kwargs):
        """Build model with custom parameters.

        Args:
            compile_kwargs (Dict[str, Any]): Dictionary with parameters: values. The parameters should be passed to the class constructor, but should be captured as kwargs. They should also have the routing prefix (e.g., optimizer__learning_rate=0.01). compile_kwargs will automatically be parsed from **kwargs by KerasClassifier and sent here.

        Returns:
            tf.keras.Model: Model instance. The chosen model depends on which phase is passed to the class constructor.
        """
        model = VAEModel(
            y=self.y_train,
            batch_size=self.batch_size,
            missing_mask=self.missing_mask,
            output_shape=self.output_shape,
            n_components=self.n_components,
            weights_initializer=self.weights_initializer,
            hidden_layer_sizes=self.hidden_layer_sizes,
            num_hidden_layers=self.num_hidden_layers,
            hidden_activation=self.hidden_activation,
            l1_penalty=self.l1_penalty,
            l2_penalty=self.l2_penalty,
            dropout_rate=self.dropout_rate,
            num_classes=self.num_classes,
            sample_weight=self.sample_weight,
        )

        model.build((None, self.y_train.shape[1]))

        model.compile(
            optimizer=compile_kwargs["optimizer"],
            loss=compile_kwargs["loss"],
            metrics=compile_kwargs["metrics"],
            run_eagerly=True,
        )

        model.set_model_outputs()
        return model

    @staticmethod
    def scorer(y_true, y_pred, **kwargs):
        """Scorer for grid search that masks missing data.

        To use this, do not specify a scoring metric when initializing the grid search object. By default if the scoring_metric option is left as None, then it uses the estimator's scoring metric (this one).

        Args:
            y_true (numpy.ndarray): True target values input to fit().
            y_pred (numpy.ndarray): Predicted target values from estimator. The predictions are modified by self.target_encoder().inverse_transform() before being sent here.
            kwargs (Any): Other parameters sent to sklearn scoring metric. Supported options include missing_mask, scoring_metric, and testing.

        Returns:
            float: Calculated score.
        """
        # Get missing mask if provided.
        # Otherwise default is all missing values (array all True).
        missing_mask = kwargs.get(
            "missing_mask", np.ones(y_true.shape, dtype=bool)
        )

        testing = kwargs.get("testing", False)
        scoring_metric = kwargs.get("scoring_metric", "accuracy")

        y_true_masked = y_true[missing_mask]
        y_pred_masked = y_pred[missing_mask]

        if scoring_metric.startswith("auc"):
            roc_auc = Scorers.compute_roc_auc_micro_macro(
                y_true_masked, y_pred_masked, missing_mask
            )

            if scoring_metric == "auc_macro":
                return roc_auc["macro"]

            elif scoring_metric == "auc_micro":
                return roc_auc["micro"]

            else:
                raise ValueError(
                    f"Invalid scoring_metric provided: {scoring_metric}"
                )

        elif scoring_metric.startswith("precision"):
            pr_ap = Scorers.compute_pr(y_true_masked, y_pred_masked)

            if scoring_metric == "precision_recall_macro":
                return pr_ap["macro"]

            elif scoring_metric == "precision_recall_micro":
                return pr_ap["micro"]

            else:
                raise ValueError(
                    f"Invalid scoring_metric provided: {scoring_metric}"
                )

        elif scoring_metric == "accuracy":
            y_pred_masked_decoded = NeuralNetworkMethods.decode_masked(
                y_pred_masked
            )
            return accuracy_score(y_true_masked, y_pred_masked_decoded)

        else:
            raise ValueError(
                f"Invalid scoring_metric provided: {scoring_metric}"
            )

        if testing:
            np.set_printoptions(threshold=np.inf)
            print(y_true_masked)

            y_pred_masked_decoded = NeuralNetworkMethods.decode_masked(
                y_pred_masked
            )

            print(y_pred_masked_decoded)

    @property
    def feature_encoder(self):
        """Handles feature input, X, before training.

        Returns:
            VAEInputTransformer: InputTransformer object that includes fit() and transform() methods to transform input before estimator fitting.
        """
        return VAEInputTransformer()

    @property
    def target_encoder(self):
        """Handles target input and output, y_true and y_pred, both before and after training.

        Returns:
            NNOutputTransformer: NNOutputTransformer object that includes fit(), transform(), and inverse_transform() methods.
        """
        return MLPTargetTransformer()

    def predict(self, X, **kwargs):
        """Returns predictions for the given test data.

        Args:
            X (Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)): Training samples where n_samples is the number of samples and n_features is the number of features.
        **kwargs (Dict[str, Any]): Extra arguments to route to ``Model.predict``\.

        Warnings:
            Passing estimator parameters as keyword arguments (aka as ``**kwargs``) to ``predict`` is not supported by the Scikit-Learn API, and will be removed in a future version of SciKeras. These parameters can also be specified by prefixing ``predict__`` to a parameter at initialization (``BaseWrapper(..., fit__batch_size=32, predict__batch_size=1000)``) or by using ``set_params`` (``est.set_params(fit__batch_size=32, predict__batch_size=1000)``\).

        Returns:
            array-like: Predictions, of shape shape (n_samples,) or (n_samples, n_outputs).

        Notes:
            Had to override predict() here in order to do the __call__ with the refined input, V_latent.
        """
        y_pred_proba = self.model_(self.y_train, training=False)
        return self.target_encoder_.inverse_transform(y_pred_proba)


class MLPClassifier(KerasClassifier):
    """Estimator to be used with the scikit-learn API.

    Args:
        V (numpy.ndarray or Dict[str, Any]): Input X values of shape (n_samples, n_components). If a dictionary is passed, each key: value pair should have randomly initialized values for n_components: V. self.feature_encoder() will parse it and select the key: value pair with the current n_components. This allows n_components to be grid searched using GridSearchCV. Otherwise, it throws an error that the dimensions are off. Defaults to None.

        y_decoded (numpy.ndarray): Original target data, y, that is not one-hot encoded. Should have shape (n_samples, n_features). Should be 012-encoded. Defaults to None.

        y (numpy.ndarray): One-hot encoded target data. Defaults to None.

        batch_size (int): Batch size to train with. Defaults to 32.

        missing_mask (np.ndarray): Missing mask with missing values set to False (0) and observed values as True (1). Defaults to None. Defaults to None.

        output_shape (int): Number of units in model output layer. Defaults to None.

        weights_initializer (str): Kernel initializer to use for model weights. Defaults to "glorot_normal".

        hidden_layer_sizes (List[int]): Output unit size for each hidden layer. Should be list of length num_hidden_layers. Defaults to None.

        num_hidden_layers (int): Number of hidden layers to use. Defaults to 1.

        hidden_activation (str): Hidden activation function to use. Defaults to "elu".

        l1_penalty (float): L1 regularization penalty to use to reduce overfitting. Defautls to 0.01.

        l2_penalty (float): L2 regularization penalty to use to reduce overfitting. Defaults to 0.01.

        dropout_rate (float): Dropout rate for each hidden layer to reduce overfitting. Defaults to 0.2.

        num_classes (int): Number of classes in output predictions. Defaults to 1.

        phase (int or None): Current phase (if doing UBP), or None if doing NLPCA. Defults to None.

        n_components (int): Number of components to use for input V. Defaults to 3.

        search_mode (bool): Whether in grid search mode (True) or single estimator mode (False). Defaults to True.

        optimizer (str or callable): Optimizer to use during training. Should be either a str or tf.keras.optimizers callable. Defaults to "adam".

        loss (str or callable): Loss function to use. Should be a string or a callable if using a custom loss function. Defaults to "binary_crossentropy".

        metrics (List[Union[str, callable]]): Metrics to use. Should be a sstring or a callable if using a custom metrics function. Defaults to "accuracy".

        epochs (int): Number of epochs to train over. Defaults to 100.

        verbose (int): Verbosity mode ranging from 0 - 2. 0 = silent, 2 = most verbose.

        ubp_weights (tensorflow.Tensor): Weights from UBP model. Fetched by doing model.get_weights() on phase 2 model. Only used if phase 3. Defaults to None.

        kwargs (Any): Other keyword arguments to route to fit, compile, callbacks, etc. Should have the routing prefix (e.g., optimizer__learning_rate=0.01).
    """

    def __init__(
        self,
        V,
        y_train,
        y_original,
        ubp_weights=None,
        batch_size=32,
        missing_mask=None,
        output_shape=None,
        weights_initializer="glorot_normal",
        hidden_layer_sizes=None,
        num_hidden_layers=1,
        hidden_activation="elu",
        l1_penalty=0.01,
        l2_penalty=0.01,
        dropout_rate=0.2,
        num_classes=3,
        phase=None,
        sample_weight=None,
        n_components=3,
        optimizer="adam",
        loss="binary_crossentropy",
        epochs=100,
        verbose=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.V = V
        self.y_train = y_train
        self.y_original = y_original
        self.ubp_weights = ubp_weights
        self.batch_size = batch_size
        self.missing_mask = missing_mask
        self.output_shape = output_shape
        self.weights_initializer = weights_initializer
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.phase = phase
        self.sample_weight = sample_weight
        self.n_components = n_components
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.verbose = verbose

    def _keras_build_fn(self, compile_kwargs):
        """Build model with custom parameters.

        Args:
            compile_kwargs (Dict[str, Any]): Dictionary with parameters: values. The parameters should be passed to the class constructor, but should be captured as kwargs. They should also have the routing prefix (e.g., optimizer__learning_rate=0.01). compile_kwargs will automatically be parsed from **kwargs by KerasClassifier and sent here.

        Returns:
            tf.keras.Model: Model instance. The chosen model depends on which phase is passed to the class constructor.
        """
        if self.phase is None:
            model = NLPCAModel(
                V=self.V,
                y=self.y_train,
                batch_size=self.batch_size,
                missing_mask=self.missing_mask,
                output_shape=self.output_shape,
                n_components=self.n_components,
                weights_initializer=self.weights_initializer,
                hidden_layer_sizes=self.hidden_layer_sizes,
                num_hidden_layers=self.num_hidden_layers,
                hidden_activation=self.hidden_activation,
                l1_penalty=self.l1_penalty,
                l2_penalty=self.l2_penalty,
                dropout_rate=self.dropout_rate,
                num_classes=self.num_classes,
                phase=self.phase,
                sample_weight=self.sample_weight,
            )

        elif self.phase == 1:
            model = UBPPhase1(
                V=self.V,
                y=self.y_train,
                batch_size=self.batch_size,
                missing_mask=self.missing_mask,
                output_shape=self.output_shape,
                n_components=self.n_components,
                weights_initializer=self.weights_initializer,
                hidden_layer_sizes=self.hidden_layer_sizes,
                num_hidden_layers=self.num_hidden_layers,
                hidden_activation=self.hidden_activation,
                l1_penalty=self.l1_penalty,
                l2_penalty=self.l2_penalty,
                dropout_rate=self.dropout_rate,
                num_classes=self.num_classes,
                phase=self.phase,
            )

        elif self.phase == 2:
            model = UBPPhase2(
                V=self.V,
                y=self.y_train,
                batch_size=self.batch_size,
                missing_mask=self.missing_mask,
                output_shape=self.output_shape,
                n_components=self.n_components,
                weights_initializer=self.weights_initializer,
                hidden_layer_sizes=self.hidden_layer_sizes,
                num_hidden_layers=self.num_hidden_layers,
                hidden_activation=self.hidden_activation,
                l1_penalty=self.l1_penalty,
                l2_penalty=self.l2_penalty,
                dropout_rate=self.dropout_rate,
                num_classes=self.num_classes,
                phase=self.phase,
            )

        elif self.phase == 3:
            model = UBPPhase3(
                V=self.V,
                y=self.y_train,
                batch_size=self.batch_size,
                missing_mask=self.missing_mask,
                output_shape=self.output_shape,
                n_components=self.n_components,
                weights_initializer=self.weights_initializer,
                hidden_layer_sizes=self.hidden_layer_sizes,
                num_hidden_layers=self.num_hidden_layers,
                hidden_activation=self.hidden_activation,
                l1_penalty=self.l1_penalty,
                l2_penalty=self.l2_penalty,
                dropout_rate=self.dropout_rate,
                num_classes=self.num_classes,
                phase=self.phase,
            )

            model.build((None, self.n_components))

        model.compile(
            optimizer=compile_kwargs["optimizer"],
            loss=compile_kwargs["loss"],
            metrics=compile_kwargs["metrics"],
            run_eagerly=True,
        )

        model.set_model_outputs()

        if self.phase == 3:
            model.set_weights(self.ubp_weights)

        return model

    @staticmethod
    def scorer(y_true, y_pred, **kwargs):
        """Scorer for grid search that masks missing data.

        To use this, do not specify a scoring metric when initializing the grid search object. By default if the scoring_metric option is left as None, then it uses the estimator's scoring metric (this one).

        Args:
            y_true (numpy.ndarray): True target values input to fit().
            y_pred (numpy.ndarray): Predicted target values from estimator. The predictions are modified by self.target_encoder().inverse_transform() before being sent here.
            kwargs (Any): Other parameters sent to sklearn scoring metric. Supported options include missing_mask, scoring_metric, and testing.

        Returns:
            float: Calculated score.
        """
        # Get missing mask if provided.
        # Otherwise default is all missing values (array all True).
        missing_mask = kwargs.get(
            "missing_mask", np.ones(y_true.shape, dtype=bool)
        )

        testing = kwargs.get("testing", False)
        scoring_metric = kwargs.get("scoring_metric", "accuracy")

        y_true_masked = y_true[missing_mask]
        y_pred_masked = y_pred[missing_mask]

        if scoring_metric.startswith("auc"):
            roc_auc = Scorers.compute_roc_auc_micro_macro(
                y_true_masked, y_pred_masked, missing_mask
            )

            if scoring_metric == "auc_macro":
                return roc_auc["macro"]

            elif scoring_metric == "auc_micro":
                return roc_auc["micro"]

            else:
                raise ValueError(
                    f"Invalid scoring_metric provided: {scoring_metric}"
                )

        elif scoring_metric.startswith("precision"):
            pr_ap = Scorers.compute_pr(y_true_masked, y_pred_masked)

            if scoring_metric == "precision_recall_macro":
                return pr_ap["macro"]

            elif scoring_metric == "precision_recall_micro":
                return pr_ap["micro"]

            else:
                raise ValueError(
                    f"Invalid scoring_metric provided: {scoring_metric}"
                )

        elif scoring_metric == "accuracy":
            y_pred_masked_decoded = NeuralNetworkMethods.decode_masked(
                y_pred_masked
            )
            return accuracy_score(y_true_masked, y_pred_masked_decoded)

        else:
            raise ValueError(
                f"Invalid scoring_metric provided: {scoring_metric}"
            )

        if testing:
            np.set_printoptions(threshold=np.inf)
            print(y_true_masked)

            y_pred_masked_decoded = NeuralNetworkMethods.decode_masked(
                y_pred_masked
            )

            print(y_pred_masked_decoded)

    @property
    def feature_encoder(self):
        """Handles feature input, X, before training.

        Returns:
            UBPInputTransformer: InputTransformer object that includes fit() and transform() methods to transform input before estimator fitting.
        """
        return UBPInputTransformer(self.n_components, self.V)

    @property
    def target_encoder(self):
        """Handles target input and output, y_true and y_pred, both before and after training.

        Returns:
            NNOutputTransformer: NNOutputTransformer object that includes fit(), transform(), and inverse_transform() methods.
        """
        return MLPTargetTransformer()

    def predict(self, X, **kwargs):
        """Returns predictions for the given test data.

        Args:
            X (Union[array-like, sparse matrix, dataframe] of shape (n_samples, n_features)): Training samples where n_samples is the number of samples and n_features is the number of features.
        **kwargs (Dict[str, Any]): Extra arguments to route to ``Model.predict``\.

        Warnings:
            Passing estimator parameters as keyword arguments (aka as ``**kwargs``) to ``predict`` is not supported by the Scikit-Learn API, and will be removed in a future version of SciKeras. These parameters can also be specified by prefixing ``predict__`` to a parameter at initialization (``BaseWrapper(..., fit__batch_size=32, predict__batch_size=1000)``) or by using ``set_params`` (``est.set_params(fit__batch_size=32, predict__batch_size=1000)``\).

        Returns:
            array-like: Predictions, of shape shape (n_samples,) or (n_samples, n_outputs).

        Notes:
            Had to override predict() here in order to do the __call__ with the refined input, V_latent.
        """
        y_pred_proba = self.model_(self.model_.V_latent, training=False)
        return self.target_encoder_.inverse_transform(y_pred_proba)


class NeuralNetworkMethods:
    """Methods common to all neural network imputer classes and loss functions"""

    def __init__(self):
        self.data = None

    @staticmethod
    def decode_onehot(df_dummies):
        """Decode one-hot format to 012-encoded genotypes.

        Args:
            df_dummies (pandas.DataFrame): One-hot encoded imputed data.

        Returns:
            pandas.DataFrame: 012-encoded imputed data.
        """
        pos = defaultdict(list)
        vals = defaultdict(list)

        for i, c in enumerate(df_dummies.columns):
            if "_" in c:
                k, v = c.split("_", 1)
                pos[k].append(i)
                vals[k].append(v)

            else:
                pos["_"].append(i)

        df = pd.DataFrame(
            {
                k: pd.Categorical.from_codes(
                    np.argmax(df_dummies.iloc[:, pos[k]].values, axis=1),
                    vals[k],
                )
                for k in vals
            }
        )

        df[df_dummies.columns[pos["_"]]] = df_dummies.iloc[:, pos["_"]]

        return df

    def decode_masked(self, y):
        """Evaluate model predictions by decoding to 012-encoded format.

        Gets the index of the highest predicted value to obtain the 012-encodings.

        Calucalates highest predicted value for each row vector and each class, setting the most likely class to 1.0.

        Args:
            y (numpy.ndarray): Model predictions of shape (n_samples * n_features,). Array should be flattened and masked.

        Returns:
            numpy.ndarray: Imputed one-hot encoded values.

            pandas.DataFrame: One-hot encoded pandas DataFrame with no missing values.
        """
        Xprob = y
        Xt = np.apply_along_axis(self.mle, axis=-1, arr=Xprob)
        Xpred = np.argmax(Xt, axis=-1)
        return Xpred

    @staticmethod
    def encode_categorical(X):
        """Encode -9 encoded missing values as np.nan.

        Args:
            X (numpy.ndarray): 012-encoded genotypes with -9 as missing values.

        Returns:
            pandas.DataFrame: DataFrame with missing values encoded as np.nan.
        """
        np.nan_to_num(X, copy=False, nan=-9.0)
        X = X.astype(str)
        X[(X == "-9.0") | (X == "-9")] = "none"

        df = pd.DataFrame(X)
        df_incomplete = df.copy()

        # Replace 'none' with np.nan
        for row in df.index:
            for col in df.columns:
                if df_incomplete.iat[row, col] == "none":
                    df_incomplete.iat[row, col] = np.nan

        return df_incomplete

    @staticmethod
    def mle(row):
        """Get the Maximum Likelihood Estimation for the best prediction. Basically, it sets the index of the maxiumum value in a vector (row) to 1.0, since it is one-hot encoded.

        Args:
            row (numpy.ndarray(float)): Row vector with predicted values as floating points.

        Returns:
            numpy.ndarray(float): Row vector with the highest prediction set to 1.0 and the others set to 0.0.
        """
        res = np.zeros(row.shape[0])
        res[np.argmax(row)] = 1
        return res

    @classmethod
    def predict(cls, X, complete_encoded):
        """Evaluate VAE predictions by calculating the highest predicted value.

        Calucalates highest predicted value for each row vector and each class, setting the most likely class to 1.0.

        Args:
            X (numpy.ndarray): Input 012-encoded data.

            complete_encoded (numpy.ndarray): Output one-hot encoded data with the maximum predicted values for each class set to 1.0.

        Returns:
            numpy.ndarray: Imputed one-hot encoded values.

            pandas.DataFrame: One-hot encoded pandas DataFrame with no missing values.
        """

        df = cls.encode_categorical(X)

        # Had to add dropna() to count unique classes while ignoring np.nan
        col_classes = [len(df[c].dropna().unique()) for c in df.columns]
        df_dummies = pd.get_dummies(df)
        mle_complete = None
        for i, cnt in enumerate(col_classes):
            start_idx = int(sum(col_classes[0:i]))
            col_completed = complete_encoded[:, start_idx : start_idx + cnt]
            mle_completed = np.apply_along_axis(
                cls.mle, axis=1, arr=col_completed
            )

            if mle_complete is None:
                mle_complete = mle_completed

            else:
                mle_complete = np.hstack([mle_complete, mle_completed])
        return mle_complete, df_dummies

    def validate_hidden_layers(self, hidden_layer_sizes, num_hidden_layers):
        """Validate hidden_layer_sizes and verify that it is in the correct format.

        Args:
            hidden_layer_sizes (str, int, List[str], or List[int]): Output units for all the hidden layers.

            num_hidden_layers (int): Number of hidden layers to use.

        Returns:
            List[int] or List[str]: List of hidden layer sizes.
        """
        if isinstance(hidden_layer_sizes, (str, int)):
            hidden_layer_sizes = [hidden_layer_sizes] * num_hidden_layers

        # If not all integers
        elif isinstance(hidden_layer_sizes, list):
            if not all([isinstance(x, (str, int)) for x in hidden_layer_sizes]):
                ls = list(set([type(item) for item in hidden_layer_sizes]))
                raise TypeError(
                    f"Variable hidden_layer_sizes must either be None, "
                    f"an integer or string, or a list of integers or "
                    f"strings, but got the following type(s): {ls}"
                )

        else:
            raise TypeError(
                f"Variable hidden_layer_sizes must either be, "
                f"an integer, a string, or a list of integers or strings, "
                f"but got the following type: {type(hidden_layer_sizes)}"
            )

        assert (
            num_hidden_layers == len(hidden_layer_sizes)
            and num_hidden_layers > 0
        ), "num_hidden_layers must be the length of hidden_layer_sizes."

        return hidden_layer_sizes

    def get_hidden_layer_sizes(self, n_dims, n_components, hl_func):
        """Get dimensions of hidden layers.

        Args:
            n_dims (int): The number of feature dimensions (columns) (d).

            n_components (int): The number of reduced dimensions (t).

            hl_func (str): The function to use to calculate the hidden layer sizes. Possible options: "midpoint", "sqrt", "log2".

        Returns:
            [int, int, int, ...]: [Number of dimensions in hidden layers].
        """
        layers = list()
        if not isinstance(hl_func, list):
            raise TypeError("hl_func must be of type list.")

        for func in hl_func:
            if func == "midpoint":
                units = round((n_dims + n_components) / 2)
            elif func == "sqrt":
                units = round(math.sqrt(n_dims))
            elif func == "log2":
                units = round(math.log(n_dims, 2))
            elif isinstance(func, int):
                units = func
            else:
                raise ValueError(
                    f"hidden_layer_sizes must be either integers or any of "
                    f"the following strings: 'midpoint', "
                    f"'sqrt', or 'log2', but got {func} of type {type(func)}"
                )

            assert units > 0 and units < n_dims, (
                f"The hidden layer sizes must be > 0 and < the number of "
                f"features (i.e., columns) in the dataset, but size was {units}"
            )

            layers.append(units)
        return layers

    def validate_model_inputs(self, y, missing_mask, output_shape):
        """Validate inputs to Keras subclass model.

        Args:
            V (numpy.ndarray): Input to refine. Shape: (n_samples, n_components).
            y (numpy.ndarray): Target (but actual input data). Shape: (n_samples, n_features).

            y_test (numpy.ndarray): Target test dataset. Should have been imputed with simple imputer and missing data simulated using SimGenotypeData(). Shape: (n_samples, n_features).

            missing_mask (numpy.ndarray): Missing data mask for y.

            missing_mask_test (numpy.ndarray): Missing data mask for y_test.

            output_shape (int): Output shape for hidden layers.

        Raises:
            TypeError: V, y, missing_mask, output_shape must not be NoneType.
        """
        if y is None:
            raise TypeError("y must not be NoneType.")

        if missing_mask is None:
            raise TypeError("missing_mask must not be NoneType.")

        if output_shape is None:
            raise TypeError("output_shape must not be NoneType.")

    def prepare_training_batches(
        self,
        V,
        y,
        batch_size,
        batch_idx,
        trainable,
        n_components,
        sample_weight,
        missing_mask,
        ubp=True,
    ):
        """Prepare training batches in the custom training loop.

        Args:
            V (numpy.ndarray): Input to batch subset and refine, of shape (n_samples, n_components) (if doing UBP/NLPCA) or (n_samples, n_features) (if doing VAE).

            y (numpy.ndarray): Target to use to refine input V. shape (n_samples, n_features).

            batch_size (int): Batch size to subset.

            batch_idx (int): Current batch index.

            trainable (bool): Whether tensor v should be trainable.

            n_components (int): Number of principal components used in V.

            sample_weight (List[float] or None): List of floats of shape (n_samples,) with sample weights. sample_weight argument must be passed to fit().

            missing_mask (numpy.ndarray): Boolean array with True for missing values and False for observed values.

            ubp (bool, optional): Whether model is UBP/NLPCA (if True) or VAE (if False). Defaults to True.

        Returns:
            tf.Variable: Input tensor v with current batch assigned.
            numpy.ndarray: Current batch of arget data (actual input) used to refine v.
            List[float]: Sample weights
            int: Batch starting index.
            int: Batch ending index.
        """
        # on_train_batch_begin() method.
        n_samples = y.shape[0]

        # Get current batch size and range.
        # self._batch_idx is set in the UBPCallbacks() callback
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        if batch_end > n_samples:
            batch_end = n_samples - 1
            batch_size = batch_end - batch_start

        # override batches. This model refines the input to fit the output, so
        # v_batch and y_true have to be overridden.
        y_true = y[batch_start:batch_end, :]

        v_batch = V[batch_start:batch_end, :]
        missing_mask_batch = missing_mask[batch_start:batch_end, :]

        if sample_weight is not None:
            sample_weight_batch = sample_weight[batch_start:batch_end, :]
        else:
            sample_weight_batch = None

        if ubp:
            v = tf.Variable(
                tf.zeros([batch_size, n_components]),
                trainable=trainable,
                dtype=tf.float32,
            )

            # Assign current batch to tf.Variable v.
            v.assign(v_batch)
        else:
            v = v_batch

        return (
            v,
            y_true,
            sample_weight_batch,
            missing_mask_batch,
            batch_start,
            batch_end,
        )

    def validate_batch_size(self, X, batch_size):
        """Validate the batch size, and adjust as necessary.

        If the specified batch_size is greater than the number of samples in the input data, it will divide batch_size by 2 until it is less than n_samples.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).
            batch_size (int): Batch size to use.

        Returns:
            int: Batch size (adjusted if necessary).
        """
        if batch_size > X.shape[0]:
            while batch_size > X.shape[0]:
                print(
                    "Batch size is larger than the number of samples. "
                    "Dividing batch_size by 2."
                )
                batch_size //= 2
        return batch_size

    def set_compile_params(self, optimizer):
        """Set compile parameters to use.

        Returns:
            Dict[str, callable] or Dict[str, Any]: Callables if search_mode is True, otherwise instantiated objects.

        Raises:
            ValueError: Unsupported optimizer specified.
        """
        if optimizer.lower() == "adam":
            opt = tf.keras.optimizers.Adam
        elif optimizer.lower() == "sgd":
            opt = tf.keras.optimizers.SGD
        elif optimizer.lower() == "adagrad":
            opt = tf.keras.optimizers.Adagrad
        elif optimizer.lower() == "adadelta":
            opt = tf.keras.optimizers.Adadelta
        elif optimizer.lower() == "adamax":
            opt = tf.keras.optimizers.Adamax
        elif optimizer.lower() == "ftrl":
            opt = tf.keras.optimizers.Ftrl
        elif optimizer.lower() == "nadam":
            opt = tf.keras.optimizers.Nadam
        elif optimizer.lower() == "rmsprop":
            opt = tf.keras.optimizers.RMSProp

        # Doing grid search. Params are callables.
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.CategoricalAccuracy()]

        return {
            "optimizer": opt,
            "loss": loss,
            "metrics": metrics,
            "run_eagerly": True,
        }

    def init_weights(self, dim1, dim2, w_mean=0, w_stddev=0.01):
        """Initialize random weights to use with the model.

        Args:
            dim1 (int): Size of first dimension.

            dim2 (int): Size of second dimension.

            w_mean (float, optional): Mean of normal distribution. Defaults to 0.

            w_stddev (float, optional): Standard deviation of normal distribution. Defaults to 0.01.
        """
        # Get reduced-dimension dataset.
        return np.random.normal(loc=w_mean, scale=w_stddev, size=(dim1, dim2))

    def reset_seeds(self):
        """Reset random seeds for initializing weights."""
        seed1 = np.random.randint(1, 1e6)
        seed2 = np.random.randint(1, 1e6)
        seed3 = np.random.randint(1, 1e6)
        np.random.seed(seed1)
        random.seed(seed2)
        if tf.__version__[0] == "2":
            tf.random.set_seed(seed3)
        else:
            tf.set_random_seed(seed3)

    @staticmethod
    def masked_mse(self, X_true, X_pred, mask):
        """Calculates mean squared error with missing values ignored.

        Args:
            X_true (numpy.ndarray): One-hot encoded input data.
            X_pred (numpy.ndarray): Predicted values.
            mask (numpy.ndarray): One-hot encoded missing data mask.

        Returns:
            float: Mean squared error calculation.
        """
        return np.square(np.subtract(X_true[mask], X_pred[mask])).mean()

    def make_reconstruction_loss(self):
        """Make loss function for use with a keras model.

        Returns:
            callable: Function that calculates loss.
        """

        def reconstruction_loss(input_and_mask, y_pred):
            """Custom loss function for neural network model with missing mask.

            Ignores missing data in the calculation of the loss function.

            Args:
                input_and_mask (numpy.ndarray): Input one-hot encoded array with missing values also one-hot encoded and h-stacked.

                y_pred (numpy.ndarray): Predicted values.

            Returns:
                float: Mean squared error loss value with missing data masked.
            """
            n_features = y_pred.numpy().shape[1]

            true_indices = range(n_features)
            missing_indices = range(n_features, n_features * 2)

            # Split features and missing mask.
            y_true = tf.gather(input_and_mask, true_indices, axis=1)
            missing_mask = tf.gather(input_and_mask, missing_indices, axis=1)

            observed_mask = tf.subtract(1.0, missing_mask)
            y_true_observed = tf.multiply(y_true, observed_mask)
            pred_observed = tf.multiply(y_pred, observed_mask)

            # loss_fn = tf.keras.losses.CategoricalCrossentropy()
            # return loss_fn(y_true_observed, pred_observed)

            return tf.keras.metrics.mean_squared_error(
                y_true=y_true_observed, y_pred=pred_observed
            )

        return reconstruction_loss

    @staticmethod
    def normalize_data(data):
        """Normalize data between 0 and 1."""
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    @staticmethod
    def get_class_weights(y_true, original_missing_mask, user_weights=None):
        """Get class weights for each column in a 2D matrix.

        Args:
            y_true (numpy.ndarray): True target values.

            original_missing_mask (numpy.ndarray): Boolean mask with missing values set to True and non-missing to False.

            user_weights (Dict[int, float], optional): Class weights if user-provided.

        Returns:
            numpy.ndarray: Class weights per column of shape (n_samples, n_features).
        """
        # Get list of class_weights (per-column).
        class_weights = list()
        sample_weight = np.zeros(y_true.shape)
        if user_weights is not None:
            # Set user-defined sample_weights
            for k in user_weights.keys():
                sample_weight[y_true == k] = user_weights[k]
        else:
            # Automatically get class weights to set sample_weight.
            for i in np.arange(y_true.shape[1]):
                mm = ~original_missing_mask_[:, i]
                classes = np.unique(y_true[mm, i])
                cw = compute_class_weight(
                    "balanced",
                    classes=classes,
                    y=y_true[mm, i],
                )

                class_weights.append({k: v for k, v in zip(classes, cw)})

            # Make sample_weight_matrix from automatic per-column class_weights.
            for i, w in enumerate(class_weights):
                for j in range(3):
                    if j in w:
                        sample_weight[y_true[:, i] == j, i] = w[j]

        return sample_weight

    @staticmethod
    def run_vae_clf(
        y_train,
        y_test,
        y_true,
        model_params,
        compile_params,
        fit_params,
        disable_progressbar,
        run_gridsearch,
        epochs,
        sim_missing_mask,
        scoring_metric,
        ga,
        early_stop_gen,
        verbosity,
        grid_iter,
        gridparams,
        n_jobs,
        ga_kwargs,
        n_components,
        gridsearch_method,
        prefix,
        tt,
        ubp_weights=None,
        phase=None,
        scoring=None,
        testing=False,
    ):
        # This reduces memory usage.
        # tensorflow builds graphs that
        # will stack if not cleared before
        # building a new model.
        tf.keras.backend.set_learning_phase(1)
        tf.keras.backend.clear_session()
        self.reset_seeds()

        model = None

        if not disable_progressbar and not run_gridsearch:
            fit_params["callbacks"][-1] = TqdmCallback(
                epochs=epochs, verbose=0, desc=desc
            )

        clf = VAEClassifier(
            y_train,
            y_true,
            **model_params,
            optimizer=compile_params["optimizer"],
            optimizer__learning_rate=compile_params["learning_rate"],
            loss=compile_params["loss"],
            metrics=compile_params["metrics"],
            epochs=fit_params["epochs"],
            callbacks=fit_params["callbacks"],
            validation_split=fit_params["validation_split"],
            verbose=0,
            score__missing_mask=sim_missing_mask_,
            score__scoring_metric=scoring_metric,
        )

        if run_gridsearch:
            # Cannot do CV because there is no way to use test splits
            # given that the input gets refined. If using a test split,
            # then it would just be the randomly initialized values and
            # would not accurately represent the model.
            # Thus, we disable cross-validation for the grid searches.
            cross_val = DisabledCV()

            if ga:
                # Stop searching if GA sees no improvement.
                callback = [
                    ConsecutiveStopping(
                        generations=early_stop_gen, metric="fitness"
                    )
                ]

                if not disable_progressbar:
                    callback.append(ProgressBar())

                verbose = False if verbosity == 0 else True

                # Do genetic algorithm
                # with HiddenPrints():
                search = GASearchCV(
                    estimator=clf,
                    cv=cross_val,
                    scoring=scoring,
                    generations=grid_iter,
                    param_grid=gridparams,
                    n_jobs=n_jobs,
                    refit=scoring_metric,
                    verbose=verbose,
                    error_score="raise",
                    **ga_kwargs,
                )

                search.fit(V[n_components], y_true, callbacks=callback)

            else:
                if gridsearch_method.lower() == "gridsearch":
                    # Do GridSearchCV
                    search = GridSearchCV(
                        clf,
                        param_grid=gridparams,
                        n_jobs=n_jobs,
                        cv=cross_val,
                        scoring=scoring,
                        refit=scoring_metric,
                        verbose=verbose * 4,
                        error_score="raise",
                    )

                elif gridsearch_method.lower() == "randomized_gridsearch":
                    search = RandomizedSearchCV(
                        clf,
                        param_distributions=gridparams,
                        n_iter=grid_iter,
                        n_jobs=n_jobs,
                        cv=cross_val,
                        scoring=scoring,
                        refit=scoring_metric,
                        verbose=verbose * 4,
                        error_score="raise",
                    )

                else:
                    raise ValueError(
                        f"Invalid gridsearch_method specified: "
                        f"{gridsearch_method}"
                    )

                search.fit(V[n_components], y=y_true)

            best_params = search.best_params_
            best_score = search.best_score_
            best_clf = search.best_estimator_

            if phase is not None:
                fp = f"{prefix}_ubp_cvresults_phase{phase}.csv"
            else:
                fp = f"{prefix}_nlpca_cvresults.csv"

            cv_results = pd.DataFrame(search.cv_results_)
            cv_results.to_csv(fp, index=False)

        else:
            clf.fit(V[n_components], y=y_true)
            best_params = None
            best_score = None
            search = None
            best_clf = clf

        model = best_clf.model_
        best_history = best_clf.history_
        y_pred_proba = model(model.V_latent, training=False)
        y_pred = tt.inverse_transform(y_pred_proba)

        # Get metric scores.
        metrics = Scorers.scorer(
            y_true,
            y_pred,
            missing_mask=sim_missing_mask_,
            testing=True,
        )

        return (
            V,
            model,
            best_history,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        )

    @staticmethod
    def run_mlp_clf(
        y_train,
        y_true,
        model_params,
        compile_params,
        fit_params,
        disable_progressbar,
        run_gridsearch,
        epochs,
        sim_missing_mask,
        scoring_metric,
        ga,
        early_stop_gen,
        verbosity,
        grid_iter,
        gridparams,
        n_jobs,
        ga_kwargs,
        n_components,
        gridsearch_method,
        prefix,
        tt,
        ubp_weights=None,
        phase=None,
        scoring=None,
        testing=False,
    ):
        # This reduces memory usage.
        # tensorflow builds graphs that
        # will stack if not cleared before
        # building a new model.
        tf.keras.backend.set_learning_phase(1)
        tf.keras.backend.clear_session()
        self.reset_seeds()

        model = None
        V = model_params.pop("V")

        if phase is not None:
            desc = f"Epoch (Phase {phase}): "
        else:
            desc = "Epoch: "

        if not disable_progressbar and not run_gridsearch:
            fit_params["callbacks"][-1] = TqdmCallback(
                epochs=epochs, verbose=0, desc=desc
            )

        clf = MLPClassifier(
            V,
            y_train,
            y_true,
            **model_params,
            ubp_weights=ubp_weights,
            optimizer=compile_params["optimizer"],
            optimizer__learning_rate=compile_params["learning_rate"],
            loss=compile_params["loss"],
            metrics=compile_params["metrics"],
            epochs=fit_params["epochs"],
            phase=phase,
            callbacks=fit_params["callbacks"],
            validation_split=fit_params["validation_split"],
            verbose=0,
            score__missing_mask=sim_missing_mask_,
            score__scoring_metric=scoring_metric,
        )

        if run_gridsearch:
            # Cannot do CV because there is no way to use test splits
            # given that the input gets refined. If using a test split,
            # then it would just be the randomly initialized values and
            # would not accurately represent the model.
            # Thus, we disable cross-validation for the grid searches.
            cross_val = DisabledCV()

            if ga:
                # Stop searching if GA sees no improvement.
                callback = [
                    ConsecutiveStopping(
                        generations=early_stop_gen, metric="fitness"
                    )
                ]

                if not disable_progressbar:
                    callback.append(ProgressBar())

                verbose = False if verbosity == 0 else True

                # Do genetic algorithm
                # with HiddenPrints():
                search = GASearchCV(
                    estimator=clf,
                    cv=cross_val,
                    scoring=scoring,
                    generations=grid_iter,
                    param_grid=gridparams,
                    n_jobs=n_jobs,
                    refit=scoring_metric,
                    verbose=verbose,
                    error_score="raise",
                    **ga_kwargs,
                )

                search.fit(V[n_components], y_true, callbacks=callback)

            else:
                if gridsearch_method.lower() == "gridsearch":
                    # Do GridSearchCV
                    search = GridSearchCV(
                        clf,
                        param_grid=gridparams,
                        n_jobs=n_jobs,
                        cv=cross_val,
                        scoring=scoring,
                        refit=scoring_metric,
                        verbose=verbose * 4,
                        error_score="raise",
                    )

                elif gridsearch_method.lower() == "randomized_gridsearch":
                    search = RandomizedSearchCV(
                        clf,
                        param_distributions=gridparams,
                        n_iter=grid_iter,
                        n_jobs=n_jobs,
                        cv=cross_val,
                        scoring=scoring,
                        refit=scoring_metric,
                        verbose=verbose * 4,
                        error_score="raise",
                    )

                else:
                    raise ValueError(
                        f"Invalid gridsearch_method specified: "
                        f"{gridsearch_method}"
                    )

                search.fit(V[n_components], y=y_true)

            best_params = search.best_params_
            best_score = search.best_score_
            best_clf = search.best_estimator_

            if phase is not None:
                fp = f"{prefix}_ubp_cvresults_phase{phase}.csv"
            else:
                fp = f"{prefix}_nlpca_cvresults.csv"

            cv_results = pd.DataFrame(search.cv_results_)
            cv_results.to_csv(fp, index=False)

        else:
            clf.fit(V[n_components], y=y_true)
            best_params = None
            best_score = None
            search = None
            best_clf = clf

        model = best_clf.model_
        best_history = best_clf.history_
        y_pred_proba = model(model.V_latent, training=False)
        y_pred = tt.inverse_transform(y_pred_proba)

        # Get metric scores.
        metrics = Scorers.scorer(
            y_true,
            y_pred,
            missing_mask=sim_missing_mask_,
            testing=True,
        )

        return (
            V,
            model,
            best_history,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        )
