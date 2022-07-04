# Standard Library Imports
from faulthandler import disable
import logging
import os
import pprint
import sys
import warnings
from collections import defaultdict

# Third-party Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Grid search imports
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Scikit-learn imports
from sklearn.metrics import accuracy_score, jaccard_score
from sklearn.base import BaseEstimator, TransformerMixin

# Genetic algorithm grid search imports
from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import ConsecutiveStopping, ProgressBar
from sklearn_genetic.plots import plot_fitness_evolution

# Import tensorflow with reduced warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").disabled = True
warnings.filterwarnings("ignore", category=UserWarning)

# noinspection PyPackageRequirements
import tensorflow as tf

from scikeras.wrappers import KerasClassifier

# Disable can't find cuda .dll errors. Also turns of GPU support.
tf.config.set_visible_devices([], "GPU")

from tensorflow.python.util import deprecation

# Disable warnings and info logs.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

# Monkey patching deprecation utils to supress warnings.
# noinspection PyUnusedLocal
def deprecated(date, instructions, warn_once=True):  # pylint: disable=unused-argument
    def deprecated_wrapper(func):
        return func

    return deprecated_wrapper


deprecation.deprecated = deprecated

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ProgbarLogger,
    ReduceLROnPlateau,
    CSVLogger,
)

# For development purposes
# from memory_profiler import memory_usage

# Custom module imports
try:
    from ..read_input.read_input import GenotypeData
    from ..utils.misc import timer
    from ..utils.misc import isnotebook
    from ..utils.misc import HiddenPrints
    from ..utils.misc import validate_input_type
    from .neural_network_methods import NeuralNetworkMethods, DisabledCV
    from .scorers import Scorers
    from .plotting import Plotting
    from .vae_model import VAEModel
    from .nlpca_model import NLPCAModel
    from .ubp_model import UBPPhase1, UBPPhase2, UBPPhase3
    from ..data_processing.transformers import (
        SimGenotypeDataTransformer,
        TargetTransformer,
        MLPTargetTransformer,
        UBPInputTransformer,
    )
except (ModuleNotFoundError, ValueError):
    from read_input.read_input import GenotypeData
    from utils.misc import timer
    from utils.misc import isnotebook
    from utils.misc import HiddenPrints
    from utils.misc import validate_input_type
    from impute.neural_network_methods import NeuralNetworkMethods, DisabledCV
    from impute.scorers import Scorers
    from impute.plotting import Plotting
    from impute.vae_model import VAEModel
    from impute.nlpca_model import NLPCAModel
    from impute.ubp_model import UBPPhase1, UBPPhase2, UBPPhase3
    from data_processing.transformers import (
        SimGenotypeDataTransformer,
        TargetTransformer,
        MLPTargetTransformer,
        UBPInputTransformer,
    )

is_notebook = isnotebook()

if is_notebook:
    from tqdm.notebook import tqdm as progressbar
else:
    from tqdm import tqdm as progressbar

from tqdm.keras import TqdmCallback


class VAEClassifier(KerasClassifier):
    """Estimator to be used with the scikit-learn API.

    Args:
        y_true (numpy.ndarray): 012-encoded target data.

        y_train (numpy.ndarray): One-hot encoded target data.

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
        missing_mask=None,
        output_shape=None,
        weights_initializer="glorot_normal",
        hidden_layer_sizes=None,
        num_hidden_layers=1,
        hidden_activation="elu",
        l1_penalty=0.01,
        l2_penalty=0.01,
        dropout_rate=0.2,
        validation_split=0.2,
        num_classes=3,
        sample_weight=None,
        n_components=3,
        optimizer="adam",
        loss="categorical_crossentropy",
        kl_weight=0.1,
        epochs=100,
        verbose=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.missing_mask = missing_mask
        self.output_shape = output_shape
        self.weights_initializer = weights_initializer
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate
        self.validation_split = validation_split
        self.num_classes = num_classes
        self.sample_weight = sample_weight
        self.n_components = n_components
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.kl_weight = kl_weight
        self.verbose = verbose

    def _keras_build_fn(self, compile_kwargs):
        """Build model with custom parameters.

        Args:
            compile_kwargs (Dict[str, Any]): Dictionary with parameters: values. The parameters should be passed to the class constructor, but should be captured as kwargs. They should also have the routing prefix (e.g., optimizer__learning_rate=0.01). compile_kwargs will automatically be parsed from **kwargs by KerasClassifier and sent here.

        Returns:
            tf.keras.Model: Model instance. The chosen model depends on which phase is passed to the class constructor.
        """
        model = VAEModel(
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
            kl_weight=self.kl_weight,
        )

        # model = VAEModel(vae.encoder, vae.decoder)

        model.compile(
            optimizer=compile_kwargs["optimizer"],
            loss=compile_kwargs["loss"],
            metrics=compile_kwargs["metrics"],
            run_eagerly=False,
        )

        # print(model.model().summary())

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
        missing_mask = kwargs.get("missing_mask", np.ones(y_true.shape, dtype=bool))

        testing = kwargs.get("testing", False)
        scoring_metric = kwargs.get("scoring_metric", "accuracy")

        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]

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
                raise ValueError(f"Invalid scoring_metric provided: {scoring_metric}")

        elif scoring_metric.startswith("precision"):
            pr_ap = Scorers.compute_pr(y_true_masked, y_pred_masked)

            if scoring_metric == "precision_recall_macro":
                return pr_ap["macro"]

            elif scoring_metric == "precision_recall_micro":
                return pr_ap["micro"]

            else:
                raise ValueError(f"Invalid scoring_metric provided: {scoring_metric}")

        elif scoring_metric == "accuracy":
            y_pred_masked_decoded = NeuralNetworkMethods.decode_masked(y_pred_masked)
            return accuracy_score(y_true_masked, y_pred_masked_decoded)

        else:
            raise ValueError(f"Invalid scoring_metric provided: {scoring_metric}")

        if testing:
            np.set_printoptions(threshold=np.inf)
            print(y_true_masked)

            y_pred_masked_decoded = NeuralNetworkMethods.decode_masked(y_pred_masked)

            print(y_pred_masked_decoded)

    @property
    def feature_encoder(self):
        """Handles feature input, X, before training.

        Returns:
            MLPTargetTransformer: InputTransformer object that includes fit() and transform() methods to transform input before estimator fitting.
        """
        return MLPTargetTransformer()

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
        X_train = self.feature_encoder_.transform(X)
        y_pred_proba, z_mean, z_log_var, z = self.model_(X_train, training=False)
        return tf.nn.softmax(y_pred_proba).numpy(), z_mean, z_log_var, z


class MLPClassifier(KerasClassifier):
    """Estimator to be used with the scikit-learn API.

    Args:
        V (numpy.ndarray or Dict[str, Any]): Input X values of shape (n_samples, n_components). If a dictionary is passed, each key: value pair should have randomly initialized values for n_components: V. self.feature_encoder() will parse it and select the key: value pair with the current n_components. This allows n_components to be grid searched using GridSearchCV. Otherwise, it throws an error that the dimensions are off. Defaults to None.

        y_train (numpy.ndarray): One-hot encoded target data. Defaults to None.

        y_original (numpy.ndarray): Original target data, y, that is not one-hot encoded. Should have shape (n_samples, n_features). Should be 012-encoded. Defaults to None.

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
        missing_mask = kwargs.get("missing_mask", np.ones(y_true.shape, dtype=bool))

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
                raise ValueError(f"Invalid scoring_metric provided: {scoring_metric}")

        elif scoring_metric.startswith("precision"):
            pr_ap = Scorers.compute_pr(y_true_masked, y_pred_masked)

            if scoring_metric == "precision_recall_macro":
                return pr_ap["macro"]

            elif scoring_metric == "precision_recall_micro":
                return pr_ap["micro"]

            else:
                raise ValueError(f"Invalid scoring_metric provided: {scoring_metric}")

        elif scoring_metric == "accuracy":
            y_pred_masked_decoded = NeuralNetworkMethods.decode_masked(y_pred_masked)
            return accuracy_score(y_true_masked, y_pred_masked_decoded)

        else:
            raise ValueError(f"Invalid scoring_metric provided: {scoring_metric}")

        if testing:
            np.set_printoptions(threshold=np.inf)
            print(y_true_masked)

            y_pred_masked_decoded = NeuralNetworkMethods.decode_masked(y_pred_masked)

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


class VAECallbacks(tf.keras.callbacks.Callback):
    """Custom callbacks to use with subclassed VAE Keras model.

    See tf.keras.callbacks.Callback documentation.
    """

    def __init__(self):
        self.indices = None

    def on_epoch_begin(self, epoch, logs=None):
        """Shuffle input and target at start of epoch."""
        x = self.model.x.copy()
        y = self.model.y.copy()
        y_train = self.model.y_train.copy()
        missing_mask = self.model.missing_mask
        sample_weight = self.model.sample_weight

        n_samples = len(y)
        self.indices = np.arange(n_samples)
        np.random.shuffle(self.indices)

        self.model.x = x[self.indices]
        self.model.y = y[self.indices]
        self.model.y_train = y_train[self.indices]
        self.model.missing_mask = missing_mask[self.indices]

        if sample_weight is not None:
            self.model.sample_weight = sample_weight[self.indices]

    def on_train_batch_begin(self, batch, logs=None):
        """Get batch index."""
        self.model.batch_idx = batch

    def on_epoch_end(self, epoch, logs=None):
        """Unsort the row indices."""
        unshuffled = np.argsort(self.indices)

        self.model.x = self.model.x[unshuffled]
        self.model.y = self.model.y[unshuffled]
        self.model.y_train = self.model.y_train[unshuffled]
        self.model.missing_mask = self.model.missing_mask[unshuffled]

        if self.model.sample_weight is not None:
            self.model.sample_weight = self.model.sample_weight[unshuffled]


class UBPCallbacks(tf.keras.callbacks.Callback):
    """Custom callbacks to use with subclassed NLPCA/ UBP Keras models.

    See tf.keras.callbacks.Callback documentation.
    """

    def __init__(self):
        self.indices = None

    def on_epoch_begin(self, epoch, logs=None):
        """Shuffle input and target at start of epoch."""
        y = self.model.y.copy()
        missing_mask = self.model.missing_mask
        sample_weight = self.model.sample_weight

        n_samples = len(y)
        self.indices = np.arange(n_samples)
        np.random.shuffle(self.indices)

        self.model.y = y[self.indices]
        self.model.V_latent = self.model.V_latent[self.indices]
        self.model.missing_mask = missing_mask[self.indices]

        if sample_weight is not None:
            self.model.sample_weight = sample_weight[self.indices]

    def on_train_batch_begin(self, batch, logs=None):
        """Get batch index."""
        self.model.batch_idx = batch

    def on_epoch_end(self, epoch, logs=None):
        """Unsort the row indices."""
        unshuffled = np.argsort(self.indices)

        self.model.y = self.model.y[unshuffled]
        self.model.V_latent = self.model.V_latent[unshuffled]
        self.model.missing_mask = self.model.missing_mask[unshuffled]

        if self.model.sample_weight is not None:
            self.model.sample_weight = self.model.sample_weight[unshuffled]


class UBPEarlyStopping(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience=0, phase=3):
        super(UBPEarlyStopping, self).__init__()
        self.patience = patience
        self.phase = phase

        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

        # In UBP, the input gets refined during training.
        # So we have to revert it too.
        self.best_input = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()

            if self.phase != 2:
                # Only refine input in phase 2.
                self.best_input = self.model.V_latent
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

                if self.phase != 2:
                    self.model.V_latent = self.best_input


class VAE(BaseEstimator, TransformerMixin):
    """Class to impute missing data using a Variational Autoencoder neural network.

    genotype_data (GenotypeData): Input GenotypeData instance.

    prefix (str, optional): Prefix for output files. Defaults to "output".

    gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If using GridSearchCV, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ``gridparams`` will be used for a randomized grid search with cross-validation. If using the genetic algorithm grid search (GASearchCV) by setting ``ga=True``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. If ``gridparams=None``\, a grid search is not performed. Defaults to None.

    disable_progressbar (bool, optional): Whether to disable the tqdm progress bar. Useful if you are doing the imputation on e.g. a high-performance computing cluster, where sometimes tqdm does not work correctly. If False, uses tqdm progress bar. If True, does not use tqdm. Defaults to False.

    batch_size (int, optional): Batch size per epoch to train the model with.

    n_components (int, optional): Number of components to use as the input data. Defaults to 3.

    early_stop_gen (int, optional): Early stopping criterion for epochs. Training will stop if the loss (error) does not decrease past the tolerance level for ``early_stop_gen`` epochs. Will save the optimal model and reload it once ``early_stop_gen`` has been reached. Defaults to 25.

    num_hidden_layers (int, optional): Number of hidden layers to use in the model. Adjust if overfitting occurs. Defaults to 3.

    hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of neurons to use in hidden layers. If string or a list of strings is supplied, the strings must be either "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units. If "log2" is supplied, the units will be calculated as ``log2(n_features)``. hidden_layer_sizes will calculate and set the number of output units for each hidden layer. If one string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list, which can differ. The list length must be equal to the ``num_hidden_layers``. Defaults to "midpoint".

    optimizer (str, optional): The optimizer to use with gradient descent. Possible value include: "adam", "sgd", "adagrad", "adadelta", "adamax", "ftrl", "nadam", and "rmsprop" are supported. See tf.keras.optimizers for more info. Defaults to "adam".

    hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Commonly used activation functions include "elu", "relu", and "sigmoid". Defaults to "elu".

    learning_rate (float, optional): The learning rate for the optimizers. Adjust if the loss is learning too slowly. Defaults to 0.1.

    lr_patience (int, optional): Number of epochs with no loss improvement to wait before reducing the learning rate.

    epochs (int, optional): Maximum number of epochs to run if the ``early_stop_gen`` criterion is not met.

    weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

    l1_penalty (float, optional): L1 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

    l2_penalty (float, optional): L2 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

    dropout_rate (float, optional): Dropout rate during training to reduce overfitting. Must be a float between 0 and 1. Defaults to 0.2.

    recurrent_weight (float, optional): Recurrent weight to calculate predictions. Defaults to 0.5.

    sample_weights (str or Dict[int, float], optional): Whether to weight each genotype by its class frequency. If ``sample_weights='auto'`` then it automatically calculates sample weights based on genotype class frequencies per locus; for example, if there are a lot more 0s and fewer 2s, then it will balance out the classes by weighting each genotype accordingly. ``sample_weights`` can also be a dictionary with the genotypes (0, 1, and 2) as the keys and the weights as the keys. If ``sample_weights`` is anything else, then they are not calculated. Defaults to False.

    grid_iter (int, optional): Number of iterations for grid search. Defaults to 50.

    gridsearch_method (str, optional): Grid search method to use. Possible options include: 'gridsearch', 'randomized_gridsearch', and 'genetic_algorithm'. 'gridsearch' runs all possible permutations of parameters, 'randomized_gridsearch' runs a random subset of parameters, and 'genetic_algorithm' uses a genetic algorithm gridsearch (via GASearchCV). Defaults to 'gridsearch'.

    ga_kwargs (Dict[str, Any] or None): Keyword arguments to be passed to a Genetic Algorithm grid search. Only used if ``ga==True``\.

    scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

    sim_strategy (str, optional): Strategy to use for simulating missing data. Only used to validate the accuracy of the imputation. The final model will be trained with the non-simulated dataset. Supported options include: "random", "nonrandom", and "nonrandom_weighted". "random" randomly simulates missing data. When set to "nonrandom", branches from ``GenotypeData.guidetree`` will be randomly sampled to generate missing data on descendant nodes. For "nonrandom_weighted", missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data distributed as might be the case with mutation-disruption of RAD sites). Defaults to "random".

    sim_prop_missing (float, optional): Proportion of missing data to simulate with the SimGenotypeDataTransformer. Defaults to 0.1.

    n_jobs (int, optional): Number of parallel jobs to use in the grid search if ``gridparams`` is not None. -1 means use all available processors. Defaults to 1.

    verbose (int, optional): Verbosity setting. Can be 0, 1, or 2. 0 is the least and 2 is the most verbose. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data=None,
        *,
        prefix="output",
        gridparams=None,
        disable_progressbar=False,
        batch_size=32,
        n_components=3,
        early_stop_gen=25,
        num_hidden_layers=3,
        hidden_layer_sizes="midpoint",
        optimizer="adam",
        hidden_activation="elu",
        learning_rate=0.01,
        lr_patience=1,
        epochs=100,
        weights_initializer="glorot_normal",
        l1_penalty=0.0001,
        l2_penalty=0.0001,
        dropout_rate=0.2,
        validation_split=0.2,
        sample_weights=False,
        grid_iter=80,
        gridsearch_method="gridsearch",
        ga_kwargs=None,
        scoring_metric="auc_macro",
        sim_strategy="random",
        sim_prop_missing=0.1,
        n_jobs=1,
        verbose=0,
    ):
        super().__init__()

        self.genotype_data = genotype_data
        self.prefix = prefix
        self.gridparams = gridparams
        self.disable_progressbar = disable_progressbar
        self.batch_size = batch_size
        self.n_components = n_components

        self.early_stop_gen = early_stop_gen
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.hidden_activation = hidden_activation
        self.learning_rate = learning_rate
        self.lr_patience = lr_patience
        self.epochs = epochs
        self.weights_initializer = weights_initializer
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate
        self.validation_split = validation_split
        self.sample_weights = sample_weights
        self.grid_iter = grid_iter
        self.gridsearch_method = gridsearch_method
        self.ga_kwargs = ga_kwargs
        self.scoring_metric = scoring_metric
        self.sim_strategy = sim_strategy
        self.sim_prop_missing = sim_prop_missing
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.num_classes = 3

    @timer
    def fit(self, X):
        """Train the VAE model on input data X.

        Args:
            X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): Input 012-encoded genotypes.

        Returns:
            self: Current instance; allows method chaining.

        Raises:
            TypeError: Must be either pandas.DataFrame, numpy.ndarray, or List[List[int]].
        """
        self.run_gridsearch_ = False if self.gridparams is None else True

        # Treating y as X here for compatibility with UBP/NLPCA.
        # With VAE, y=X anyways.
        y = X
        y = validate_input_type(y, return_type="array")

        self.nn_ = NeuralNetworkMethods()
        plotting = Plotting()

        if self.gridsearch_method == "genetic_algorithm":
            self.ga_ = True
        else:
            self.ga_ = False

        # Simulate missing data and get missing masks.
        sim = SimGenotypeDataTransformer(
            self.genotype_data,
            prop_missing=self.sim_prop_missing,
            strategy=self.sim_strategy,
            mask_missing=True,
        )

        self.y_original_ = y.copy()
        self.y_simulated_ = sim.fit_transform(self.y_original_)
        self.sim_missing_mask_ = sim.sim_missing_mask_
        self.original_missing_mask_ = sim.original_missing_mask_
        self.all_missing_ = sim.all_missing_mask_

        # One-hot encode y to get y_train.
        self.tt_ = TargetTransformer()
        y_train = self.tt_.fit_transform(self.y_original_)

        if self.gridparams is not None:
            self.scoring_metrics_ = [
                "precision_recall_macro",
                "precision_recall_micro",
                "auc_macro",
                "auc_micro",
                "accuracy",
            ]

        # testresults = pd.read_csv(f"{self.prefix}_nlpca_cvresults.csv")
        # plotting.plot_grid_search(testresults, self.prefix)
        # sys.exit()

        (
            logfile,
            callbacks,
            compile_params,
            model_params,
            fit_params,
        ) = self._initialize_parameters(y_train, self.original_missing_mask_)

        (
            self.models_,
            self.histories_,
            self.best_params_,
            self.best_score_,
            self.best_estimator_,
            self.search_,
            self.metrics_,
        ) = self._run_vae(
            self.y_original_,
            model_params,
            compile_params,
            fit_params,
        )

        if (
            self.best_params_ is not None
            and "optimizer__learning_rate" in self.best_params_
        ):
            self.best_params_["learning_rate"] = self.best_params_.pop(
                "optimizer__learning_rate"
            )

        if self.gridparams is not None:
            if self.verbose > 0:
                print("\nBest found parameters:")
                pprint.pprint(self.best_params_)
                print(f"\nBest score: {self.best_score_}")
            plotting.plot_grid_search(self.search_.cv_results_, self.prefix)

        plotting.plot_history(self.histories_, "VAE")
        plotting.plot_metrics(self.metrics_, self.num_classes, self.prefix)

        if self.ga_:
            plot_fitness_evolution(self.search_)
            plt.savefig(f"{self.prefix}_fitness_evolution.pdf", bbox_inches="tight")
            plt.cla()
            plt.clf()
            plt.close()

            g = plotting.plot_search_space(self.search_)
            plt.savefig(f"{self.prefix}_search_space.pdf", bbox_inches="tight")
            plt.cla()
            plt.clf()
            plt.close()

        return self

        # # VAE needs a numpy array, not a dataframe
        # self.data = self.df.copy().values

        # imputed_enc = self.fit(epochs=self.epochs, batch_size=self.batch_size)

        # imputed_enc, dummy_df = self.predict(X, imputed_enc)

        # imputed_df = self._decode_onehot(
        #     pd.DataFrame(data=imputed_enc, columns=dummy_df.columns)
        # )

        # return imputed_df.to_numpy()

    def transform(self, X):
        """Predict and decode imputations and return transformed array.

        Args:
            X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): Input data to transform.

        Returns:
            numpy.ndarray: Imputed data.
        """
        y = X
        y = validate_input_type(y, return_type="array")

        model = self.models_[0]

        # Apply softmax activation.
        y_true = y.copy()
        y_train = self.tt_.transform(y_true)
        y_true_1d = y_true.ravel()

        y_size = y_true.size
        y_missing_idx = np.flatnonzero(self.original_missing_mask_)

        y_pred_proba, z_mean, z_log_var, z = model(y_train, training=False)
        # y_pred = y_pred.numpy()
        y_pred_proba = tf.nn.softmax(y_pred_proba).numpy()
        y_pred_decoded = self.nn_.decode_masked(y_pred_proba)
        # y_pred_decoded = self.nn_.decode_masked(y_pred)
        y_pred_1d = y_pred_decoded.ravel()

        for i in np.arange(y_size):
            if i in y_missing_idx:
                y_true_1d[i] = y_pred_1d[i]

        return np.reshape(y_true_1d, y_true.shape)

    def _run_vae(
        self,
        y_true,
        model_params,
        compile_params,
        fit_params,
    ):
        """Run NLPCA using custom subclassed model.

        Args:
            y_true (numpy.ndarray): Original genotypes (training dataset) with known and missing values.

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
        scoring = None

        if self.run_gridsearch_:
            scoring = scorers.make_multimetric_scorer(
                self.scoring_metrics_, self.sim_missing_mask_
            )

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
            scoring=scoring,
            testing=True,
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

    def run_clf(
        self,
        y_train,
        y_true,
        model_params,
        compile_params,
        fit_params,
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
        self.nn_.reset_seeds()

        model = None

        desc = "Epoch: "

        if not self.disable_progressbar and not self.run_gridsearch_:
            fit_params["callbacks"][-1] = TqdmCallback(
                epochs=self.epochs, verbose=0, desc=desc
            )

        clf = VAEClassifier(
            **model_params,
            optimizer=compile_params["optimizer"],
            optimizer__learning_rate=compile_params["learning_rate"],
            loss=compile_params["loss"],
            metrics=compile_params["metrics"],
            epochs=fit_params["epochs"],
            callbacks=fit_params["callbacks"],
            verbose=0,
            fit__validation_split=fit_params["validation_split"],
            score__missing_mask=self.sim_missing_mask_,
            score__scoring_metric=self.scoring_metric,
        )

        if self.run_gridsearch_:
            # Cannot do CV because there is no way to use test splits
            # given that the input gets refined. If using a test split,
            # then it would just be the randomly initialized values and
            # would not accurately represent the model.
            # Thus, we disable cross-validation for the grid searches.
            cross_val = DisabledCV()
            verbose = False if self.verbose == 0 else True

            if self.ga_:
                # Stop searching if GA sees no improvement.
                callback = [
                    ConsecutiveStopping(
                        generations=self.early_stop_gen, metric="fitness"
                    )
                ]

                if not self.disable_progressbar:
                    callback.append(ProgressBar())

                # Do genetic algorithm
                # with HiddenPrints():
                search = GASearchCV(
                    estimator=clf,
                    cv=cross_val,
                    scoring=scoring,
                    generations=self.grid_iter,
                    param_grid=self.gridparams,
                    n_jobs=self.n_jobs,
                    refit=self.scoring_metric,
                    verbose=verbose,
                    error_score="raise",
                    **self.ga_kwargs,
                )

                search.fit(y_true, y_true, callbacks=callback)

            else:
                if self.gridsearch_method.lower() == "gridsearch":
                    # Do GridSearchCV
                    search = GridSearchCV(
                        clf,
                        param_grid=self.gridparams,
                        n_jobs=self.n_jobs,
                        cv=cross_val,
                        scoring=scoring,
                        refit=self.scoring_metric,
                        verbose=self.verbose * 4,
                        error_score="raise",
                    )

                elif self.gridsearch_method.lower() == "randomized_gridsearch":
                    search = RandomizedSearchCV(
                        clf,
                        param_distributions=self.gridparams,
                        n_iter=self.grid_iter,
                        n_jobs=self.n_jobs,
                        cv=cross_val,
                        scoring=scoring,
                        refit=self.scoring_metric,
                        verbose=verbose * 4,
                        error_score="raise",
                    )

                else:
                    raise ValueError(
                        f"Invalid gridsearch_method specified: " f"{gridsearch_method}"
                    )

                search.fit(y_true, y=y_true)

            best_params = search.best_params_
            best_score = search.best_score_
            best_clf = search.best_estimator_

            fp = f"{self.prefix}_vae_cvresults.csv"

            cv_results = pd.DataFrame(search.cv_results_)
            cv_results.to_csv(fp, index=False)

        else:
            clf.fit(y_true, y=y_true)
            # model_params.pop("missing_mask")

            # vae = MakeVAEModel(
            #     **model_params,
            #     # optimizer=compile_params["optimizer"],
            #     # # optimizer__learning_rate=compile_params["learning_rate"],
            #     # loss=compile_params["loss"],
            #     # metrics=compile_params["metrics"],
            #     # epochs=fit_params["epochs"],
            #     # callbacks=fit_params["callbacks"],
            #     # verbose=0,
            #     # fit__validation_split=fit_params["validation_split"],
            #     # score__missing_mask=self.sim_missing_mask_,
            #     # score__scoring_metric=self.scoring_metric,
            # )

            # model = VAEModel(vae.encoder, vae.decoder)

            # # model.build((None, self.y_true.shape[1]))

            # model.compile(
            #     optimizer=compile_params["optimizer"](learning_rate=self.learning_rate),
            #     loss=compile_params["loss"],
            #     metrics=compile_params["metrics"],
            #     run_eagerly=False,
            # )

            # # model.build((None, y_train.shape[1], y_train.shape[2]))
            # # print(model.summary())

            # model.fit(
            #     y_train,
            #     y_train,
            #     batch_size=self.batch_size,
            #     epochs=self.epochs,
            #     callbacks=fit_params["callbacks"],
            #     validation_split=self.validation_split,
            # )

            # clf.fit(y_true, y=y_true)
            best_params = None
            best_score = None
            search = None
            best_clf = clf

        model = best_clf.model_
        best_history = best_clf.history_

        y_train = self.tt_.transform(y_true)
        y_pred_proba, z_mean, z_log_var, z = model(y_train, training=False)
        # y_pred = y_pred.numpy()
        y_pred = self.tt_.inverse_transform(y_pred_proba)

        # Get metric scores.
        metrics = Scorers.scorer(
            y_true,
            y_pred,
            missing_mask=self.sim_missing_mask_,
            testing=True,
        )

        return (
            model,
            best_history,
            best_params,
            best_score,
            best_clf,
            search,
            metrics,
        )

    def _initialize_parameters(self, y_train, missing_mask):
        """Initialize important parameters.

        Args:
            y_train (numpy.ndarray): Training subset of original input data.
            missing_mask (numpy.ndarray): Training missing data mask.

        Returns:
            List[int]: Hidden layer sizes of length num_hidden_layers.
            int: Number of hidden layers.
            tf.keras.optimizers: Gradient descent optimizer to use.
            str: Logfile for CSVLogger callback.
            List[tf.keras.callbacks]: List of callbacks for Keras model.
            Dict[str, Any]: Parameters to use for model.compile().
            Dict[str, Any]: UBP Phase 1 model parameters.
            Dict[str, Any]: UBP Phase 2 model parameters.
            Dict[str, Any]: UBP Phase 3 or NLPCA model parameters.
            Dict[str, Any]: Other parameters to pass to KerasClassifier().
            Dict[str, Any]: Parameters to pass to fit_params() in grid search.
        """
        # For CSVLogger() callback.
        append = False
        logfile = f"{self.prefix}_vae_log.csv"

        callbacks = [
            # VAECallbacks(),
            CSVLogger(filename=logfile, append=append),
            ReduceLROnPlateau(patience=self.lr_patience, min_lr=1e-6, min_delta=1e-6),
        ]

        search_mode = True if self.run_gridsearch_ else False

        if not self.disable_progressbar and not search_mode:
            callbacks.append(
                TqdmCallback(epochs=self.epochs, verbose=0, desc="Epoch: ")
            )

        compile_params = self.nn_.set_compile_params(self.optimizer, vae=True)
        compile_params["learning_rate"] = self.learning_rate

        model_params = {
            "y_train": y_train,
            "missing_mask": missing_mask,
            "output_shape": y_train.shape[1],
            "weights_initializer": self.weights_initializer,
            "n_components": self.n_components,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_activation": self.hidden_activation,
            "l1_penalty": self.l1_penalty,
            "l2_penalty": self.l2_penalty,
            "dropout_rate": self.dropout_rate,
            "num_classes": self.num_classes,
        }

        fit_verbose = 1 if self.verbose == 2 else 0

        fit_params = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "callbacks": callbacks,
            "validation_split": self.validation_split,
            "shuffle": True,
            "verbose": fit_verbose,
        }

        if self.sample_weights == "auto":
            # Get class weights for each column.
            sample_weights = self.nn_.get_class_weights(
                self.y_original_, self.original_missing_mask_
            )
            sample_weights = self.nn_.normalize_data(sample_weights)

        elif isinstance(self.sample_weights, dict):
            for i in range(self.num_classes):
                if self.sample_weights[i] == 0.0:
                    self.sim_missing_mask_[self.y_original_ == i] = False

            sample_weights = self.nn_.get_class_weights(
                self.y_original_, user_weights=self.sample_weights
            )

        else:
            sample_weights = None

        model_params["sample_weight"] = sample_weights

        if self.run_gridsearch_ and "learning_rate" in self.gridparams:
            self.gridparams["optimizer__learning_rate"] = self.gridparams[
                "learning_rate"
            ]

            self.gridparams.pop("learning_rate")

        return (
            logfile,
            callbacks,
            compile_params,
            model_params,
            fit_params,
        )

    # def _fit(self, batch_size=256, epochs=100):
    #     """Train a variational autoencoder model to impute missing data.

    #     Args:
    #         batch_size (int, optional): Number of data splits to train on per epoch. Defaults to 256.

    #         epochs (int, optional): Number of epochs (cycles through the data) to use. Defaults to 100.

    #     Returns:
    #         numpy.ndarray(float): Predicted values as numpy array.
    #     """

    #     missing_mask = self._create_missing_mask()
    #     self.fill(self.data, missing_mask, -1, self.num_classes)
    #     self.model = self._create_model()

    #     observed_mask = ~missing_mask

    #     for epoch in range(1, epochs + 1):
    #         X_pred = self._train_epoch(self.model, missing_mask, batch_size)
    #         observed_mse = self.masked_mse(
    #             X_true=self.data, X_pred=X_pred, mask=observed_mask
    #         )

    #         if epoch == 1:
    #             print(f"Initial MSE: {observed_mse}")

    #         elif epoch % 50 == 0:
    #             print(
    #                 f"Observed MSE ({epoch}/{epochs} epochs): "
    #                 f"{observed_mse}"
    #             )

    #         old_weight = 1.0 - self.recurrent_weight
    #         self.data[missing_mask] *= old_weight
    #         pred_missing = X_pred[missing_mask]
    #         self.data[missing_mask] += self.recurrent_weight * pred_missing

    #     return self.data.copy()

    def predict(self, X, complete_encoded):
        """Evaluate VAE predictions by calculating the highest predicted value.

        Calucalates highest predicted value for each row vector and each class, setting the most likely class to 1.0.

        Args:
            X (numpy.ndarray): Input one-hot encoded data.

            complete_encoded (numpy.ndarray): Output one-hot encoded data with the maximum predicted values for each class set to 1.0.

        Returns:
            numpy.ndarray: Imputed one-hot encoded values.

            pandas.DataFrame: One-hot encoded pandas DataFrame with no missing values.
        """

        df = self._encode_categorical(X)

        # Had to add dropna() to count unique classes while ignoring np.nan
        col_classes = [len(df[c].dropna().unique()) for c in df.columns]
        df_dummies = pd.get_dummies(df)
        mle_complete = None

        for i, cnt in enumerate(col_classes):
            start_idx = int(sum(col_classes[0:i]))
            col_completed = complete_encoded[:, start_idx : start_idx + cnt]

            mle_completed = np.apply_along_axis(self.mle, axis=1, arr=col_completed)

            if mle_complete is None:
                mle_complete = mle_completed

            else:
                mle_complete = np.hstack([mle_complete, mle_completed])

        return mle_complete, df_dummies

    @property
    def imputed(self):
        return self.imputed_df

    def _train_epoch(self, model, missing_mask, batch_size):
        """Train one cycle (epoch) of a variational autoencoder model.

        Args:
            model (tf.keras.Sequential object): VAE model object implemented in Keras with tensorflow.

            missing_mask (numpy.ndarray(bool)): Missing data boolean mask, with True corresponding to a missing value.

            batch_size (int): Batch size for one epoch.

        Returns:
            numpy.ndarray: VAE model predictions of the current epoch.
        """

        input_with_mask = np.hstack([self.data, missing_mask])
        n_samples = len(input_with_mask)

        n_batches = int(np.ceil(n_samples / batch_size))
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = input_with_mask[indices]

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            batch_data = X_shuffled[batch_start:batch_end, :]
            model.train_on_batch(batch_data, batch_data)

        return model.predict(input_with_mask)

    def _read_example_data(self):
        """Read in example mushrooms dataset.

        Returns:
            numpy.ndarray: One-hot encoded genotypes.
        """
        df = pd.read_csv("mushrooms_test_2.csv", header=None)

        df_incomplete = df.copy()

        df_incomplete.iat[1, 0] = np.nan
        df_incomplete.iat[2, 1] = np.nan

        missing_encoded = pd.get_dummies(df_incomplete)

        for col in df.columns:
            missing_cols = missing_encoded.columns.str.startswith(str(col) + "_")

            missing_encoded.loc[df_incomplete[col].isnull(), missing_cols] = np.nan

        return missing_encoded

    def _encode_categorical(self, X):
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


class UBP(BaseEstimator, TransformerMixin):
    """Class to impute missing data using unsupervised backpropagation (UBP) or inverse non-linear principal component analysis (NLPCA).

    Args:
        genotype_data (GenotypeData): Input GenotypeData instance.

        prefix (str, optional): Prefix for output files. Defaults to "output".

        gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If using GridSearchCV, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ``gridparams`` will be used for a randomized grid search with cross-validation. If using the genetic algorithm grid search (GASearchCV) by setting ``ga=True``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. If ``gridparams=None``\, a grid search is not performed. Defaults to None.

        disable_progressbar (bool, optional): Whether to disable the tqdm progress bar. Useful if you are doing the imputation on e.g. a high-performance computing cluster, where sometimes tqdm does not work correctly. If False, uses tqdm progress bar. If True, does not use tqdm. Defaults to False.

        nlpca (bool, optional): If True, then uses NLPCA model instead of UBP. Otherwise uses UBP. Defaults to False.

        batch_size (int, optional): Batch size per epoch to train the model with.

        n_components (int, optional): Number of components to use as the input data. Defaults to 3.

        early_stop_gen (int, optional): Early stopping criterion for epochs. Training will stop if the loss (error) does not decrease past the tolerance level for ``early_stop_gen`` epochs. Will save the optimal model and reload it once ``early_stop_gen`` has been reached. Defaults to 25.

        num_hidden_layers (int, optional): Number of hidden layers to use in the model. Adjust if overfitting occurs. Defaults to 3.

        hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of neurons to use in hidden layers. If string or a list of strings is supplied, the strings must be either "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units. If "log2" is supplied, the units will be calculated as ``log2(n_features)``. hidden_layer_sizes will calculate and set the number of output units for each hidden layer. If one string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list, which can differ. The list length must be equal to the ``num_hidden_layers``. Defaults to "midpoint".

        optimizer (str, optional): The optimizer to use with gradient descent. Possible value include: "adam", "sgd", "adagrad", "adadelta", "adamax", "ftrl", "nadam", and "rmsprop" are supported. See tf.keras.optimizers for more info. Defaults to "adam".

        hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Commonly used activation functions include "elu", "relu", and "sigmoid". Defaults to "elu".

        learning_rate (float, optional): The learning rate for the optimizers. Adjust if the loss is learning too slowly. Defaults to 0.1.

        lr_patience (int, optional): Number of epochs with no loss improvement to wait before reducing the learning rate.

        epochs (int, optional): Maximum number of epochs to run if the ``early_stop_gen`` criterion is not met.

        weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float, optional): L1 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

        l2_penalty (float, optional): L2 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

        dropout_rate (float, optional): Dropout rate during training to reduce overfitting. Must be a float between 0 and 1. Defaults to 0.2.

        sample_weights (str or Dict[int, float], optional): Whether to weight each genotype by its class frequency. If ``sample_weights='auto'`` then it automatically calculates sample weights based on genotype class frequencies per locus; for example, if there are a lot more 0s and fewer 2s, then it will balance out the classes by weighting each genotype accordingly. ``sample_weights`` can also be a dictionary with the genotypes (0, 1, and 2) as the keys and the weights as the keys. If ``sample_weights`` is anything else, then they are not calculated. Defaults to False.

        grid_iter (int, optional): Number of iterations for grid search. Defaults to 50.

        gridsearch_method (str, optional): Grid search method to use. Possible options include: 'gridsearch', 'randomized_gridsearch', and 'genetic_algorithm'. 'gridsearch' runs all possible permutations of parameters, 'randomized_gridsearch' runs a random subset of parameters, and 'genetic_algorithm' uses a genetic algorithm gridsearch (via GASearchCV). Defaults to 'gridsearch'.

        ga_kwargs (Dict[str, Any] or None): Keyword arguments to be passed to a Genetic Algorithm grid search. Only used if ``ga==True``\.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        verbose (int, optional): Verbosity setting ranging from 0 to 3 (least to most verbose). Defaults to 0.

        sim_strategy (str, optional): Strategy to use for simulating missing data. Only used to validate the accuracy of the imputation. The final model will be trained with the non-simulated dataset. Supported options include: "random", "nonrandom", and "nonrandom_weighted". "random" randomly simulates missing data. When set to "nonrandom", branches from ``GenotypeData.guidetree`` will be randomly sampled to generate missing data on descendant nodes. For "nonrandom_weighted", missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data distributed as might be the case with mutation-disruption of RAD sites). Defaults to "random".

        sim_prop_missing (float, optional): Proportion of missing data to simulate with the SimGenotypeDataTransformer. Defaults to 0.1.

        n_jobs (int, optional): Number of parallel jobs to use in the grid search if ``gridparams`` is not None. -1 means use all available processors. Defaults to 1.

        verbose (int, optional): Verbosity setting. Can be 0, 1, or 2. 0 is the least and 2 is the most verbose. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix="output",
        gridparams=None,
        disable_progressbar=False,
        nlpca=False,
        batch_size=32,
        n_components=3,
        early_stop_gen=25,
        num_hidden_layers=3,
        hidden_layer_sizes="midpoint",
        optimizer="adam",
        hidden_activation="elu",
        learning_rate=0.01,
        lr_patience=1,
        epochs=100,
        weights_initializer="glorot_normal",
        l1_penalty=0.0001,
        l2_penalty=0.0001,
        dropout_rate=0.2,
        sample_weights=False,
        grid_iter=80,
        gridsearch_method="gridsearch",
        ga_kwargs=None,
        scoring_metric="auc_macro",
        sim_strategy="random",
        sim_prop_missing=0.1,
        n_jobs=1,
        verbose=0,
    ):

        super().__init__()

        # CLF parameters.
        self.genotype_data = genotype_data
        self.prefix = prefix
        self.gridparams = gridparams
        self.disable_progressbar = disable_progressbar
        self.nlpca = nlpca
        self.batch_size = batch_size
        self.n_components = n_components
        self.early_stop_gen = early_stop_gen
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.hidden_activation = hidden_activation
        self.learning_rate = learning_rate
        self.lr_patience = lr_patience
        self.epochs = epochs
        self.weights_initializer = weights_initializer
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate
        self.sample_weights = sample_weights
        self.verbose = verbose

        # Grid search settings.
        self.gridsearch_method = gridsearch_method
        self.scoring_metric = scoring_metric
        self.sim_strategy = sim_strategy
        self.sim_prop_missing = sim_prop_missing
        self.grid_iter = grid_iter
        self.n_jobs = n_jobs
        self.ga_kwargs = ga_kwargs

        # TODO: Make estimators compatible with variable number of classes.
        # E.g., with morphological data.
        self.num_classes = 3

    @timer
    def fit(self, X):
        """Train a UBP or NLPCA model on input data X.

        Uses input data from GenotypeData object.

        Args:
            X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): 012-encoded input data.

        Returns:
            pandas.DataFrame: Imputated data.
        """
        self.run_gridsearch_ = False if self.gridparams is None else True

        y = X
        y = validate_input_type(y, return_type="array")

        self.nn_ = NeuralNetworkMethods()
        plotting = Plotting()

        if self.gridsearch_method == "genetic_algorithm":
            self.ga_ = True
        else:
            self.ga_ = False

        # Placeholder for V. Gets replaced in model.
        V = self.nn_.init_weights(y.shape[0], self.n_components)

        # Simulate missing data and get missing masks.
        sim = SimGenotypeDataTransformer(
            self.genotype_data,
            prop_missing=self.sim_prop_missing,
            strategy=self.sim_strategy,
            mask_missing=True,
        )

        self.y_original_ = y.copy()
        self.y_simulated_ = sim.fit_transform(self.y_original_)
        self.sim_missing_mask_ = sim.sim_missing_mask_
        self.original_missing_mask_ = sim.original_missing_mask_
        self.all_missing_ = sim.all_missing_mask_

        # In NLPCA and UBP, X and y are flipped.
        # X is the randomly initialized model input (V)
        # V is initialized with small, random values.
        # y is the actual input data, one-hot encoded.
        self.tt_ = TargetTransformer()
        y_train = self.tt_.fit_transform(self.y_original_)

        if self.gridparams is not None:
            self.scoring_metrics_ = [
                "precision_recall_macro",
                "precision_recall_micro",
                "auc_macro",
                "auc_micro",
                "accuracy",
            ]

        # testresults = pd.read_csv(f"{self.prefix}_nlpca_cvresults.csv")
        # plotting.plot_grid_search(testresults, self.prefix)
        # sys.exit()

        (
            logfile,
            callbacks,
            compile_params,
            model_params,
            fit_params,
        ) = self._initialize_parameters(y_train, self.original_missing_mask_)

        run_func = self._run_nlpca if self.nlpca else self._run_ubp

        (
            self.models_,
            self.histories_,
            self.best_params_,
            self.best_score_,
            self.best_estimator_,
            self.search_,
            self.metrics_,
        ) = run_func(
            self.y_original_,
            model_params,
            compile_params,
            fit_params,
        )

        if self.gridparams is not None:
            if self.verbose > 0:
                print("\nBest found parameters:")
                pprint.pprint(self.best_params_)
                print(f"\nBest score: {self.best_score_}")
            plotting.plot_grid_search(self.search_.cv_results_, self.prefix)

        nn_method = "NLPCA" if self.nlpca else "UBP"

        plotting.plot_history(self.histories_, nn_method)
        plotting.plot_metrics(self.metrics_, self.num_classes, self.prefix)

        if self.ga_:
            plot_fitness_evolution(self.search_)
            plt.savefig(f"{self.prefix}_fitness_evolution.pdf", bbox_inches="tight")
            plt.cla()
            plt.clf()
            plt.close()

            g = plotting.plot_search_space(self.search_)
            plt.savefig(f"{self.prefix}_search_space.pdf", bbox_inches="tight")
            plt.cla()
            plt.clf()
            plt.close()

        return self

    def transform(self, X):
        """Predict and decode imputations and return transformed array.

        Args:
            X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): Input data to transform.

        Returns:
            numpy.ndarray: Imputed data.
        """
        y = X
        y = validate_input_type(y, return_type="array")

        # Get last (i.e., most refined) model.
        if len(self.models_) == 1:
            model = self.models_[0]
        else:
            model = self.models_[-1]

        # Apply softmax acitvation.
        y_true = y.copy()
        y_true_1d = y_true.ravel()

        y_size = y_true.size
        y_missing_idx = np.flatnonzero(self.original_missing_mask_)

        y_pred_proba = model(model.V_latent, training=False)
        y_pred_proba = tf.nn.softmax(y_pred_proba).numpy()
        y_pred_decoded = self.nn_.decode_masked(y_pred_proba)
        y_pred_1d = y_pred_decoded.ravel()

        for i in np.arange(y_size):
            if i in y_missing_idx:
                y_true_1d[i] = y_pred_1d[i]

        return np.reshape(y_true_1d, y_true.shape)

    def _run_nlpca(
        self,
        y_true,
        model_params,
        compile_params,
        fit_params,
    ):
        """Run NLPCA using custom subclassed model.

        Args:
            y_true (numpy.ndarray): Original genotypes with known and missing values.

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
        scoring = None

        if self.run_gridsearch_:
            scoring = scorers.make_multimetric_scorer(
                self.scoring_metrics_, self.sim_missing_mask_
            )

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
            scoring=scoring,
            testing=True,
        )

        # Get metric scores.
        metrics = Scorers.scorer(
            y_true,
            y_pred,
            missing_mask=self.sim_missing_mask_,
            testing=True,
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

    def _run_ubp(
        self,
        y_true,
        model_params,
        compile_params,
        fit_params,
    ):
        """Run UBP using custom subclassed model.

        Args:
            y_true (numpy.ndarray): Original genotypes with known and missing values.

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
                self.scoring_metrics_, self.sim_missing_mask_
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
                scoring=scoring,
                testing=True,
            )

            if phase == 1:
                # Cannot have V input with different n_components
                # in other phases than are in phase 1.
                # So the n_components search has to happen in phase 1.
                if best_params is not None and search_n_components:
                    n_components_searched = best_params["n_components"]
                    model_params["V"] = {n_components_searched: model.V_latent.copy()}
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

    def run_clf(
        self,
        y_train,
        y_true,
        model_params,
        compile_params,
        fit_params,
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
        self.nn_.reset_seeds()

        model = None
        V = model_params.pop("V")

        if phase is not None:
            desc = f"Epoch (Phase {phase}): "
        else:
            desc = "Epoch: "

        if not self.disable_progressbar and not self.run_gridsearch_:
            fit_params["callbacks"][-1] = TqdmCallback(
                epochs=self.epochs, verbose=0, desc=desc
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
            score__missing_mask=self.sim_missing_mask_,
            score__scoring_metric=self.scoring_metric,
        )

        if self.run_gridsearch_:
            # Cannot do CV because there is no way to use test splits
            # given that the input gets refined. If using a test split,
            # then it would just be the randomly initialized values and
            # would not accurately represent the model.
            # Thus, we disable cross-validation for the grid searches.
            cross_val = DisabledCV()

            verbose = False if self.verbose == 0 else True

            if self.ga_:
                # Stop searching if GA sees no improvement.
                callback = [
                    ConsecutiveStopping(
                        generations=self.early_stop_gen, metric="fitness"
                    )
                ]

                if not self.disable_progressbar:
                    callback.append(ProgressBar())

                # Do genetic algorithm
                # with HiddenPrints():
                search = GASearchCV(
                    estimator=clf,
                    cv=cross_val,
                    scoring=scoring,
                    generations=self.grid_iter,
                    param_grid=self.gridparams,
                    n_jobs=self.n_jobs,
                    refit=self.scoring_metric,
                    verbose=verbose,
                    error_score="raise",
                    **self.ga_kwargs,
                )

                search.fit(V[self.n_components], y_true, callbacks=callback)

            else:
                if self.gridsearch_method.lower() == "gridsearch":
                    # Do GridSearchCV
                    search = GridSearchCV(
                        clf,
                        param_grid=self.gridparams,
                        n_jobs=self.n_jobs,
                        cv=cross_val,
                        scoring=scoring,
                        refit=self.scoring_metric,
                        verbose=verbose * 4,
                        error_score="raise",
                    )

                elif self.gridsearch_method.lower() == "randomized_gridsearch":
                    search = RandomizedSearchCV(
                        clf,
                        param_distributions=self.gridparams,
                        n_iter=self.grid_iter,
                        n_jobs=self.n_jobs,
                        cv=cross_val,
                        scoring=scoring,
                        refit=self.scoring_metric,
                        verbose=verbose * 4,
                        error_score="raise",
                    )

                else:
                    raise ValueError(
                        f"Invalid gridsearch_method specified: "
                        f"{self.gridsearch_method}"
                    )

                search.fit(V[self.n_components], y=y_true)

            best_params = search.best_params_
            best_score = search.best_score_
            best_clf = search.best_estimator_

            if phase is not None:
                fp = f"{self.prefix}_ubp_cvresults_phase{phase}.csv"
            else:
                fp = f"{self.prefix}_nlpca_cvresults.csv"

            cv_results = pd.DataFrame(search.cv_results_)
            cv_results.to_csv(fp, index=False)

        else:
            clf.fit(V[self.n_components], y=y_true)
            best_params = None
            best_score = None
            search = None
            best_clf = clf

        model = best_clf.model_
        best_history = best_clf.history_
        y_pred_proba = model(model.V_latent, training=False)
        y_pred = self.tt_.inverse_transform(y_pred_proba)

        # Get metric scores.
        metrics = Scorers.scorer(
            y_true,
            y_pred,
            missing_mask=self.sim_missing_mask_,
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

    def _initialize_parameters(self, y_train, missing_mask):
        """Initialize important parameters.

        Args:
            y_train (numpy.ndarray): Training subset of original input data to be used as target.

            missing_mask (numpy.ndarray): Training missing data mask.

        Returns:
            List[int]: Hidden layer sizes of length num_hidden_layers.
            int: Number of hidden layers.
            tf.keras.optimizers: Gradient descent optimizer to use.
            str: Logfile for CSVLogger callback.
            List[tf.keras.callbacks]: List of callbacks for Keras model.
            Dict[str, Any]: Parameters to use for model.compile().
            Dict[str, Any]: UBP Phase 1 model parameters.
            Dict[str, Any]: UBP Phase 2 model parameters.
            Dict[str, Any]: UBP Phase 3 or NLPCA model parameters.
            Dict[str, Any]: Other parameters to pass to KerasClassifier().
            Dict[str, Any]: Parameters to pass to fit_params() in grid search.
        """
        if self.nlpca:
            # For CSVLogger() callback.
            append = False
            logfile = f"{self.prefix}_nlpca_log.csv"
        else:
            append = True
            logfile = f"{self.prefix}_ubp_log.csv"

            # Remove logfile if exists, because UBP writes in append mode.
            try:
                os.remove(logfile)
            except:
                pass

        callbacks = [
            UBPCallbacks(),
            CSVLogger(filename=logfile, append=append),
            ReduceLROnPlateau(patience=self.lr_patience, min_lr=1e-6, min_delta=1e-6),
        ]

        search_mode = False if self.gridparams is None else True

        if not self.disable_progressbar and not search_mode:
            callbacks.append(
                TqdmCallback(epochs=self.epochs, verbose=0, desc="Epoch: ")
            )

        vinput = self._initV(y_train, search_mode)
        compile_params = self.nn_.set_compile_params(self.optimizer)
        compile_params["learning_rate"] = self.learning_rate

        model_params = {
            "V": vinput,
            "y_train": y_train,
            "batch_size": self.batch_size,
            "missing_mask": missing_mask,
            "output_shape": y_train.shape[1],
            "weights_initializer": self.weights_initializer,
            "n_components": self.n_components,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_activation": self.hidden_activation,
            "l1_penalty": self.l1_penalty,
            "l2_penalty": self.l2_penalty,
            "dropout_rate": self.dropout_rate,
            "num_classes": self.num_classes,
        }

        fit_verbose = 1 if self.verbose == 2 else 0

        fit_params = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "callbacks": callbacks,
            "validation_split": 0.0,
            "shuffle": False,
            "verbose": fit_verbose,
        }

        if self.sample_weights == "auto":
            # Get class weights for each column.
            sample_weights = self.nn_.get_class_weights(self.y_original_)
            sample_weights = self.nn_.normalize_data(sample_weights)

        elif isinstance(self.sample_weights, dict):
            for i in range(self.num_classes):
                if self.sample_weights[i] == 0.0:
                    self.sim_missing_mask_[self.y_original_ == i] = False

            sample_weights = self.nn_.get_class_weights(
                self.y_original_, user_weights=self.sample_weights
            )

        else:
            sample_weights = None

        model_params["sample_weight"] = sample_weights

        if self.gridparams is not None and "learning_rate" in self.gridparams:
            self.gridparams["optimizer__learning_rate"] = self.gridparams[
                "learning_rate"
            ]

            self.gridparams.pop("learning_rate")

        return (
            logfile,
            callbacks,
            compile_params,
            model_params,
            fit_params,
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
            if self.n_components < 2:
                raise ValueError("n_components must be >= 2.")

            elif self.n_components == 2:
                vinput[2] = self.nn_.init_weights(y_train.shape[0], self.n_components)

            else:
                for i in range(2, self.n_components + 1):
                    vinput[i] = self.nn_.init_weights(y_train.shape[0], i)

        else:
            vinput[self.n_components] = self.nn_.init_weights(
                y_train.shape[0], self.n_components
            )

        return vinput

    def _create_model(
        self,
        estimator,
        model_kwargs,
        compile_kwargs,
        permutation_kwargs=None,
        build=False,
    ):
        """Create a neural network model using the estimator to initialize.
        Model will be initialized, compiled, and built if ``build=True``\.
        Args:
            estimator (tf.keras.Model): Model to create. Can be a subclassed model.
            compile_params (Dict[str, Any]): Parameters passed to model.compile(). Key-value pairs should be the parameter names and their corresponding values.
            n_components (int): The number of principal components to use with NLPCA or UBP models. Not used if doing VAE. Defaults to 3.
            build (bool): Whether to build the model. Defaults to False.
            kwargs (Dict[str, Any]): Keyword arguments to pass to KerasClassifier.
        Returns:
            tf.keras.Model: Instantiated, compiled, and optionally built model.
        """
        if permutation_kwargs is not None:
            model = estimator(**model_kwargs, **permutation_kwargs)
        else:
            model = estimator(**model_kwargs)

        if build:
            model.build((None, model_kwargs["n_components"]))

        model.compile(**compile_kwargs)
        return model

    def _summarize_model(self, model):
        """Print model summary for debugging purposes.

        Args:
            model (tf.keras.Model): Model to summarize.
        """
        # For debugging.
        model.model().summary()
