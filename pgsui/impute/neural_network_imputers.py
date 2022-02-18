# Standard Library Imports
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

# Import tensorflow with reduced warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").disabled = True
warnings.filterwarnings("ignore", category=UserWarning)

# noinspection PyPackageRequirements
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow.python.util import deprecation

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

# Monkey patching deprecation utils to shut it up!
# noinspection PyUnusedLocal
def deprecated(
    date, instructions, warn_once=True
):  # pylint: disable=unused-argument
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
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2


from sklearn.metrics import accuracy_score

from sklearn.utils.class_weight import (
    compute_class_weight,
)

# Grid search imports
from sklearn.base import BaseEstimator, TransformerMixin

# Randomized grid search imports
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid

# Genetic algorithm grid search imports
from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import ConsecutiveStopping, ProgressBar
from sklearn_genetic.plots import plot_fitness_evolution

from scikeras.wrappers import KerasClassifier

# For development purposes
# from memory_profiler import memory_usage

# Custom module imports
try:
    from ..read_input.read_input import GenotypeData
    from ..utils.misc import timer
    from ..utils.misc import isnotebook
    from ..utils.misc import HiddenPrints
    from ..utils.misc import validate_input_type
    from .neural_network_methods import NeuralNetworkMethods
    from .nlpca_model import NLPCAModel
    from .ubp_model import UBPPhase1, UBPPhase2, UBPPhase3
    from ..data_processing.transformers import (
        UBPInputTransformer,
        MLPTargetTransformer,
        SimGenotypeDataTransformer,
        TargetTransformer,
    )
except (ModuleNotFoundError, ValueError):
    from read_input.read_input import GenotypeData
    from utils.misc import timer
    from utils.misc import isnotebook
    from utils.misc import HiddenPrints
    from utils.misc import validate_input_type
    from impute.neural_network_methods import NeuralNetworkMethods
    from impute.nlpca_model import NLPCAModel
    from impute.ubp_model import UBPPhase1, UBPPhase2, UBPPhase3
    from data_processing.transformers import (
        UBPInputTransformer,
        MLPTargetTransformer,
        SimGenotypeDataTransformer,
        TargetTransformer,
    )

is_notebook = isnotebook()

if is_notebook:
    from tqdm.notebook import tqdm as progressbar
else:
    from tqdm import tqdm as progressbar


class DisabledCV:
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        yield (np.arange(len(X)), np.arange(len(y)))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


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

        kwargs (Any): Other keyword arguments to route to fit, compile, callbacks, etc. Should have the routing prefix (e.g., optimizer__learning_rate=0.01).
    """

    def __init__(
        self,
        V,
        y_train,
        y_original,
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
            roc_auc = NeuralNetworkMethods.compute_roc_auc_micro_macro(
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
            pr_ap = NeuralNetworkMethods.compute_pr(
                y_true_masked, y_pred_masked
            )

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
        return UBPInputTransformer(self.phase, self.n_components, self.V)

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


class VAE(NeuralNetworkMethods):
    """Class to impute missing data using a Variational Autoencoder neural network.

    Args:
        genotype_data (GenotypeData object or None): Input data initialized as GenotypeData object. If value is None, then uses ``gt`` to get the genotypes. Either ``genotype_data`` or ``gt`` must be defined. Defaults to None.

        gt (numpy.ndarray or None): Input genotypes directly as a numpy array. If this value is None, ``genotype_data`` must be supplied instead. Defaults to None.

        prefix (str): Prefix for output files. Defaults to "output".

        cv (int): Number of cross-validation replicates to perform. Only used if ``validation_split`` is not None. Defaults to 5.

        initial_strategy (str): Initial strategy to impute missing data with for validation. Possible options include: "populations", "most_frequent", and "phylogeny", where "populations" imputes by the mode of each population at each site, "most_frequent" imputes by the overall mode of each site, and "phylogeny" uses an input phylogeny to inform the imputation. "populations" requires a population map file as input in the GenotypeData object, and "phylogeny" requires an input phylogeny and Rate Matrix Q (also instantiated in the GenotypeData object). Defaults to "populations".

        validation_split (float or None): Proportion of sites to use for the validation. If ``validation_split`` is None, then does not perform validation. Defaults to 0.2.

        disable_progressbar (bool): Whether to disable the tqdm progress bar. Useful if you are doing the imputation on e.g. a high-performance computing cluster, where sometimes tqdm does not work correctly. If False, uses tqdm progress bar. If True, does not use tqdm. Defaults to False.

        epochs (int): Number of epochs to train the VAE model with. Defaults to 100.

        batch_size (int): Batch size to train the model with.

        recurrent_weight (float): Weight to apply to recurrent network. Defaults to 0.5.

        optimizer (str): Gradient descent optimizer to use. See tf.keras.optimizers for more info. Defaults to "adam".

        dropout_rate (float): Dropout rate for neurons in the network. Can adjust to reduce overfitting. Defaults to 0.2.

        hidden_activation (str): Activation function to use for hidden layers. See tf.keras.activations for more info. Defaults to "relu".

        output_activation (str): Activation function to use for output layer. See tf.keras.activations for more info. Defaults to "sigmoid".

        kernel_initializer (str): Initializer to use for initializing model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float): L1 regularization penalty to apply. Adjust if overfitting is occurring. Defaults to 0.

        l2_penalty (float): L2 regularization penalty to apply. Adjust if overfitting is occurring. Defaults to 0.
    """

    def __init__(
        self,
        *,
        genotype_data=None,
        gt=None,
        prefix="output",
        cv=5,
        initial_strategy="populations",
        validation_split=0.2,
        disable_progressbar=False,
        epochs=100,
        batch_size=32,
        recurrent_weight=0.5,
        optimizer="adam",
        dropout_rate=0.2,
        hidden_activation="relu",
        output_activation="sigmoid",
        kernel_initializer="glorot_normal",
        l1_penalty=0,
        l2_penalty=0,
    ):
        super().__init__()

        self.prefix = prefix

        self.epochs = epochs
        self.initial_batch_size = batch_size
        self.recurrent_weight = recurrent_weight
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.kernel_initializer = kernel_initializer
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

        self.cv = cv
        self.validation_split = validation_split
        self.disable_progressbar = disable_progressbar
        self.num_classes = 1

        # Initialize variables common to all neural network classes.
        self.df = None
        self.data = None

    @timer
    def fit_transform(self, input_data):
        """Train the VAE model and predict missing values.

        Args:
            input_data (pandas.DataFrame, numpy.ndarray, or List[List[int]]): Input 012-encoded genotypes.

        Returns:
            pandas.DataFrame: Imputed data.

        Raises:
            TypeError: Must be either pandas.DataFrame, numpy.ndarray, or List[List[int]].
        """
        X = self.validate_input(input_data, out_type="pandas")
        self.batch_size = self.validate_batch_size(X, self.initial_batch_size)
        self.df = self._encode_onehot(X)

        # VAE needs a numpy array, not a dataframe
        self.data = self.df.copy().values

        imputed_enc = self.fit(epochs=self.epochs, batch_size=self.batch_size)

        imputed_enc, dummy_df = self.predict(X, imputed_enc)

        imputed_df = self._decode_onehot(
            pd.DataFrame(data=imputed_enc, columns=dummy_df.columns)
        )

        return imputed_df.to_numpy()

    def fit(self, batch_size=256, epochs=100):
        """Train a variational autoencoder model to impute missing data.

        Args:
            batch_size (int, optional): Number of data splits to train on per epoch. Defaults to 256.

            epochs (int, optional): Number of epochs (cycles through the data) to use. Defaults to 100.

        Returns:
            numpy.ndarray(float): Predicted values as numpy array.
        """

        missing_mask = self._create_missing_mask()
        self.fill(self.data, missing_mask, -1, self.num_classes)
        self.model = self._create_model()

        observed_mask = ~missing_mask

        for epoch in range(1, epochs + 1):
            X_pred = self._train_epoch(self.model, missing_mask, batch_size)
            observed_mse = self.masked_mse(
                X_true=self.data, X_pred=X_pred, mask=observed_mask
            )

            if epoch == 1:
                print(f"Initial MSE: {observed_mse}")

            elif epoch % 50 == 0:
                print(
                    f"Observed MSE ({epoch}/{epochs} epochs): "
                    f"{observed_mse}"
                )

            old_weight = 1.0 - self.recurrent_weight
            self.data[missing_mask] *= old_weight
            pred_missing = X_pred[missing_mask]
            self.data[missing_mask] += self.recurrent_weight * pred_missing

        return self.data.copy()

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

            mle_completed = np.apply_along_axis(
                self.mle, axis=1, arr=col_completed
            )

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
            missing_cols = missing_encoded.columns.str.startswith(
                str(col) + "_"
            )

            missing_encoded.loc[
                df_incomplete[col].isnull(), missing_cols
            ] = np.nan

        return missing_encoded

    def _encode_onehot(self, X):
        """Convert 012-encoded data to one-hot encodings.

        Args:
            X (numpy.ndarray): Input array with 012-encoded data and -9 as the missing data value.

        Returns:
            pandas.DataFrame: One-hot encoded data, ignoring missing values (np.nan).
        """

        df = self._encode_categorical(X)
        df_incomplete = df.copy()

        missing_encoded = pd.get_dummies(df_incomplete)

        for col in df.columns:
            missing_cols = missing_encoded.columns.str.startswith(
                str(col) + "_"
            )

            missing_encoded.loc[
                df_incomplete[col].isnull(), missing_cols
            ] = np.nan

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

    def _decode_onehot(self, df_dummies):
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

    def _get_hidden_layer_sizes(self):
        """Get dimensions of hidden layers.

        Returns:
            [int, int, int]: Number of dimensions in hidden layers.
        """
        n_dims = self.data.shape[1]
        return [
            min(2000, 8 * n_dims),
            min(500, 2 * n_dims),
            int(np.ceil(0.5 * n_dims)),
        ]

    def _create_model(self):
        """Create a variational autoencoder model with the following items: InputLayer -> DenseLayer1 -> Dropout1 -> DenseLayer2 -> Dropout2 -> DenseLayer3 -> Dropout3 -> DenseLayer4 -> OutputLayer.

        Returns:
            keras model object: Compiled Keras model.
        """
        hidden_layer_sizes = self._get_hidden_layer_sizes()
        first_layer_size = hidden_layer_sizes[0]
        n_dims = self.data.shape[1]

        model = Sequential()

        model.add(
            Dense(
                first_layer_size,
                input_dim=2 * n_dims,
                activation=self.hidden_activation,
                kernel_regularizer=l1_l2(self.l1_penalty, self.l2_penalty),
                kernel_initializer=self.kernel_initializer,
            )
        )

        model.add(Dropout(self.dropout_rate))

        for layer_size in hidden_layer_sizes[1:]:
            model.add(
                Dense(
                    layer_size,
                    activation=self.hidden_activation,
                    kernel_regularizer=l1_l2(self.l1_penalty, self.l2_penalty),
                    kernel_initializer=self.kernel_initializer,
                )
            )

            model.add(Dropout(self.dropout_rate))

        model.add(
            Dense(
                n_dims,
                activation=self.output_activation,
                kernel_regularizer=l1_l2(self.l1_penalty, self.l2_penalty),
                kernel_initializer=self.kernel_initializer,
            )
        )

        loss_function = self.make_reconstruction_loss(n_dims)
        model.compile(optimizer=self.optimizer, loss=loss_function)
        return model

    def _create_missing_mask(self):
        """Creates a missing data mask with boolean values.

        Returns:
            numpy.ndarray(bool): Boolean mask of missing values, with True corresponding to a missing data point.
        """
        if self.data.dtype != "f" and self.data.dtype != "d":
            self.data = self.data.astype(float)
        return np.isnan(self.data)


class UBP(BaseEstimator, TransformerMixin):
    """Class to impute missing data using unsupervised backpropagation (UBP) or inverse non-linear principal component analysis (NLPCA).

    Args:
        genotype_data (GenotypeData): Input GenotypeData instance.

        prefix (str, optional): Prefix for output files. Defaults to "output".

        gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If using GridSearchCV, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ``gridparams`` will be used for a randomized grid search with cross-validation. If using the genetic algorithm grid search (GASearchCV) by setting ``ga=True``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. If ``gridparams=None``\, a grid search is not performed. Defaults to None.

        do_validation (bool): Whether to validate the imputation if not doing a grid search. This validation method randomly replaces between 15% and 50% of the known, non-missing genotypes in ``n_features * column_subset`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None for ``do_validation`` to work. Calculating a validation score can be turned off altogether by setting ``do_validation`` to False. Defaults to True.

        write_output (bool, optional): If True, writes imputed data to file on disk. Otherwise just stores it as a class attribute.

        disable_progressbar (bool, optional): Whether to disable the tqdm progress bar. Useful if you are doing the imputation on e.g. a high-performance computing cluster, where sometimes tqdm does not work correctly. If False, uses tqdm progress bar. If True, does not use tqdm. Defaults to False.

        nlpca (bool, optional): If True, then uses NLPCA model instead of UBP. Otherwise uses UBP. Defaults to False.

        batch_size (int, optional): Batch size per epoch to train the model with.

        validation_split (float or None, optional): Proportion of samples (=rows) between 0 and 1 to use for the neural network training validation. Defaults to 0.3.

        n_components (int, optional): Number of components to use as the input data. Defaults to 3.

        early_stop_gen (int, optional): Early stopping criterion for epochs. Training will stop if the loss (error) does not decrease past the tolerance level ``tol`` for ``early_stop_gen`` epochs. Will save the optimal model and reload it once ``early_stop_gen`` has been reached. Defaults to 25.

        num_hidden_layers (int, optional): Number of hidden layers to use in the model. Adjust if overfitting occurs. Defaults to 3.

        hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of neurons to use in hidden layers. If string or a list of strings is supplied, the strings must be either "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units. If "log2" is supplied, the units will be calculated as ``log2(n_features)``. hidden_layer_sizes will calculate and set the number of output units for each hidden layer. If one string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list, which can differ. The list length must be equal to the ``num_hidden_layers``. Defaults to "midpoint".

        optimizer (str, optional): The optimizer to use with gradient descent. Possible value include: "adam", "sgd", and "adagrad" are supported. See tf.keras.optimizers for more info. Defaults to "adam".

        hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Commonly used activation functions include "elu", "relu", and "sigmoid". Defaults to "elu".

        learning_rate (float, optional): The learning rate for the optimizers. Adjust if the loss is learning too slowly. Defaults to 0.1.

        lr_patience (int, optional): Number of epochs with no loss improvement to wait before reducing the learning rate.

        epochs (int, optional): Maximum number of epochs to run if the ``early_stop_gen`` criterion is not met.

        tol (float, optional): Tolerance level to use for the early stopping criterion. If the loss does not improve past the tolerance level after ``early_stop_gen`` epochs, then training will halt. Defaults to 1e-3.

        weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float, optional): L1 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

        l2_penalty (float, optional): L2 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

        dropout_rate (float, optional): Dropout rate during training to reduce overfitting. Must be a float between 0 and 1. Defaults to 0.2.

        verbose (int, optional): Verbosity setting ranging from 0 to 3 (least to most verbose). Defaults to 0.

        cv (int, optional): Number of cross-validation replicates to perform. Defaults to 5.

        grid_iter (int, optional): Number of iterations for grid search. Defaults to 50.

        ga (bool, optional): Whether to use a genetic algorithm for the grid search. If False, a GridSearchCV is done instead. Defaults to False.

        ga_kwargs (Dict[str, Any] or None): Keyword arguments to be passed to a Genetic Algorithm grid search. Only used if ``ga==True``\.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        sim_strategy (str, optional): Strategy to use for simulating missing data. Only used to validate the accuracy of the imputation. The final model will be trained with the non-simulated dataset. Supported options include: "random", "nonrandom", and "nonrandom_weighted". "random" randomly simulates missing data. When set to "nonrandom", branches from ``GenotypeData.guidetree`` will be randomly sampled to generate missing data on descendant nodes. For "nonrandom_weighted", missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data distributed as might be the case with mutation-disruption of RAD sites). Defaults to "random".

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``\. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        n_jobs (int, optional): Number of parallel jobs to use in the grid search if ``gridparams`` is not None. -1 means use all available processors. Defaults to 1.

        verbose (int, optional): Verbosity setting. Can be 0, 1, or 2. 0 is the least and 2 is the most verbose. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix="output",
        gridparams=None,
        do_validation=True,
        write_output=True,
        disable_progressbar=False,
        nlpca=False,
        batch_size=32,
        validation_split=0.3,
        n_components=3,
        early_stop_gen=25,
        num_hidden_layers=3,
        hidden_layer_sizes="midpoint",
        optimizer="adam",
        hidden_activation="elu",
        learning_rate=0.01,
        lr_patience=1,
        epochs=100,
        tol=1e-3,
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
        str_encodings={
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        n_jobs=1,
        verbose=0,
        validation_mode=True,
    ):

        super().__init__()

        # CLF parameters.
        self.genotype_data = genotype_data
        self.prefix = prefix
        self.gridparams = gridparams
        self.do_validation = do_validation
        self.write_output = write_output
        self.disable_progressbar = disable_progressbar
        self.nlpca = nlpca
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.n_components = n_components
        self.early_stop_gen = early_stop_gen
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.hidden_activation = hidden_activation
        self.learning_rate = learning_rate
        self.lr_patience = lr_patience
        self.epochs = epochs
        self.tol = tol
        self.weights_initializer = weights_initializer
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate
        self.sample_weights = sample_weights
        self.verbose = verbose
        self.validation_mode = validation_mode

        # TODO: Make estimators compatible with variable number of classes.
        # E.g., with morphological data.
        self.num_classes = 3

        # Grid search settings.
        self.gridsearch_method = gridsearch_method
        self.scoring_metric = scoring_metric
        self.sim_strategy = sim_strategy
        self.sim_prop_missing = sim_prop_missing
        self.str_encodings = str_encodings
        self.grid_iter = grid_iter
        self.n_jobs = n_jobs
        self.ga_kwargs = ga_kwargs

    @timer
    def fit(self, X):
        """Train a UBP or NLPCA model and predict the output.

        Uses input data from GenotypeData object.

        Args:
            X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): 012-encoded input data.

        Returns:
            pandas.DataFrame: Imputated data.
        """
        y = X
        y = validate_input_type(y, return_type="array")

        self.nn_ = NeuralNetworkMethods()

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

        # testresults = pd.read_csv("testcvresults.csv")
        # self.nn_.plot_grid_search(testresults, self.prefix)
        # sys.exit()

        (
            logfile,
            callbacks,
            compile_params,
            model_params,
            fit_params,
        ) = self._initialize_parameters(y_train, self.original_missing_mask_)

        if self.nlpca:
            (
                self.models_,
                self.histories_,
                self.best_params_,
                self.best_score_,
                self.best_estimator_,
                self.search_,
                self.metrics_,
            ) = self._run_nlpca(
                self.y_original_,
                y_train,
                model_params,
                compile_params,
                fit_params,
            )

            if self.gridparams is not None:
                if self.verbose > 0:
                    print("\nBest found parameters:")
                    pprint.pprint(self.best_params_)
                    print(f"\nBest score: {self.best_score_}")
                self.nn_.plot_grid_search(self.search_.cv_results_, self.prefix)

        else:
            self.models_, self.histories_ = self._run_ubp(
                V, y_train, model_params, compile_params, fit_params, self.nn_
            )

        self._plot_history(self.histories_)
        self.nn_.plot_metrics(self.metrics_, self.num_classes, self.prefix)

        if self.ga_:
            plot_fitness_evolution(self.search_)
            plt.savefig(
                f"{self.prefix}_fitness_evolution.pdf", bbox_inches="tight"
            )
            plt.cla()
            plt.clf()
            plt.close()

            g = self.nn_.plot_search_space(self.search_)
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
        y_train,
        model_params,
        compile_params,
        fit_params,
    ):
        """Run NLPCA Model using custom subclassed model."""

        # Whether to do grid search.
        do_gridsearch = False if self.gridparams is None else True

        histories = list()
        models = list()

        # This reduces memory usage.
        # tensorflow builds graphs that
        # will stack if not cleared before
        # building a new model.
        tf.keras.backend.set_learning_phase(1)
        tf.keras.backend.clear_session()
        self.nn_.reset_seeds()

        if self.sample_weights == "auto":
            # Get class weights for each column.
            sample_weights = self._get_class_weights(y_true)
            sample_weights = self._normalize_data(sample_weights)

        elif isinstance(self.sample_weights, dict):
            sample_weights = self._get_class_weights(
                y_true, user_weights=self.sample_weights
            )

        else:
            sample_weights = None

        model_params["sample_weight"] = sample_weights

        if do_gridsearch:
            # Cannot do CV because there is no way to use test splits
            # given that the input gets refined. If using a test split,
            # then it would just be the randomly initialized values and
            # would not accurately represent the model.
            # Thus, we disable cross-validation for the grid searches.
            cross_val = DisabledCV()

            if "learning_rate" in self.gridparams:
                self.gridparams["optimizer__learning_rate"] = self.gridparams[
                    "learning_rate"
                ]
                self.gridparams.pop("learning_rate")

            V = model_params.pop("V")

            all_scoring_metrics = [
                "precision_recall_macro",
                "precision_recall_micro",
                "auc_macro",
                "auc_micro",
                "accuracy",
            ]

            scoring = self.nn_.make_multimetric_scorer(
                all_scoring_metrics, self.sim_missing_mask_
            )

            clf = MLPClassifier(
                V,
                model_params.pop("y_train"),
                y_true,
                **model_params,
                optimizer=compile_params["optimizer"],
                optimizer__learning_rate=compile_params["learning_rate"],
                loss=compile_params["loss"],
                metrics=compile_params["metrics"],
                epochs=fit_params["epochs"],
                phase=None,
                callbacks=fit_params["callbacks"],
                validation_split=fit_params["validation_split"],
                verbose=0,
            )

            if self.ga_:
                # Stop searching if GA sees no improvement.
                callback = [
                    ConsecutiveStopping(
                        generations=self.early_stop_gen, metric="fitness"
                    )
                ]

                if not self.disable_progressbar:
                    callback.append(ProgressBar())

                verbose = False if self.verbose == 0 else True

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
                if self.gridsearch_method == "gridsearch":
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

                elif self.gridsearch_method == "randomized_gridsearch":
                    search = RandomizedSearchCV(
                        clf,
                        param_distributions=self.gridparams,
                        n_iter=self.grid_iter,
                        n_jobs=self.n_jobs,
                        cv=cross_val,
                        scoring=scoring,
                        refit=self.scoring_metric,
                        verbose=self.verbose * 4,
                        error_score="raise",
                    )

                search.fit(V[self.n_components], y=y_true)

            best_params = search.best_params_
            best_score = search.best_score_
            best_index = search.best_index_

            cv_results = pd.DataFrame(search.cv_results_)
            cv_results.to_csv("testcvresults.csv", index=False)

            best_clf = search.best_estimator_
            histories.append(best_clf.history_)
            model = best_clf.model_

            y_pred_proba = model(model.V_latent, training=False)
            y_pred = self.tt_.inverse_transform(y_pred_proba)

            # Get metric scores.
            metrics = self.scorer(
                y_true,
                y_pred,
                missing_mask=self.sim_missing_mask_,
            )

        else:
            V = model_params.pop("V")

            clf = MLPClassifier(
                V,
                model_params.pop("y_train"),
                y_true,
                **model_params,
                optimizer=compile_params["optimizer"],
                optimizer__learning_rate=compile_params["learning_rate"],
                loss=compile_params["loss"],
                metrics=compile_params["metrics"],
                epochs=fit_params["epochs"],
                phase=None,
                callbacks=fit_params["callbacks"],
                validation_split=fit_params["validation_split"],
                verbose=0,
                score__missing_mask=self.sim_missing_mask_,
                score__scoring_metric=self.scoring_metric,
            )

            clf.fit(V[self.n_components], y=y_true)

            best_params = None
            best_score = None
            search = None
            best_clf = clf

            histories.append(clf.history_)
            model = clf.model_
            y_pred_proba = model(model.V_latent, training=False)
            y_pred = self.tt_.inverse_transform(y_pred_proba)

            # Get metric scores.
            metrics = self.scorer(
                y_true,
                y_pred,
                missing_mask=self.sim_missing_mask_,
            )

        models.append(model)

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
        self, V, y_train, model_params, compile_params, fit_params, nn
    ):

        # Whether to do grid search.
        do_gridsearch = False if self.gridparams is None else True

        histories = list()
        phases = [UBPPhase1, UBPPhase2, UBPPhase3]
        models = list()
        w = None
        for i, phase in enumerate(phases, start=1):

            # This reduces memory usage.
            # tensorflow builds graphs that
            # will stack if not cleared before
            # building a new model.
            tf.keras.backend.set_learning_phase(1)
            tf.keras.backend.clear_session()
            nn.reset_seeds()

            if do_gridsearch:
                compile_params["learning_rate"] = self.learning_rate

                search = self.grid_search(
                    model_params["V"],
                    y_train,
                    i,
                    self._create_model,
                    model_params,
                    compile_params,
                    self.epochs,
                    self.validation_split,
                    callbacks,
                    self.cv,
                    self.ga_,
                    self.early_stop_gen,
                    self.scoring_metric,
                    self.grid_iter,
                    self.gridparams,
                    self.n_jobs,
                    self.ga_kwargs,
                )
                pprint.ppprint(search.best_params_)
                print(search.best_score_)
                sys.exit()

            else:
                model = None

                model = self._create_model(
                    phase,
                    model_params,
                    compile_params,
                )

                if i == 3:
                    # Set the refined weights if doing phase 3.
                    # Phase 3 gets the refined weights from Phase 2.
                    model.set_weights(models[1].get_weights())

                history = model.fit(
                    x=model_params["V"],
                    y=y_train,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    callbacks=callbacks,
                    validation_split=self.validation_split,
                    shuffle=False,
                    verbose=self.verbose,
                )

                histories.append(history.history)
                models.append(model)

            if i == 1:
                model_params["V"] = model.V_latent.copy()

            del model

        return models, histories

    def _normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def _get_class_weights(self, y_true, user_weights=None):
        """Get class weights for each column in a 2D matrix.

        Args:
            y_true (numpy.ndarray): True target values.
            user_weights (Dict[int, float]): Class weights if user-provided.

        Returns:
            numpy.ndarray: Class weights per column of shape (n_samples, n_features).
        """
        # Get list of class_weights (per-column).
        class_weights = list()
        for i in np.arange(y_true.shape[1]):
            mm = ~self.original_missing_mask_[:, i]
            classes = np.unique(y_true[mm, i])

            if user_weights is None:
                cw = compute_class_weight(
                    "balanced",
                    classes=classes,
                    y=y_true[mm, i],
                )
            else:
                uw = user_weights.copy()
                cw = {
                    uw.pop(k)
                    for k in user_weights.copy().keys()
                    if k not in classes
                }

            class_weights.append({k: v for k, v in zip(classes, cw)})

        # Make sample_weight_matrix from per-column class_weights.
        sample_weight = np.zeros(y_true.shape)
        for i, w in enumerate(class_weights):
            for j in range(3):
                if j in w:
                    sample_weight[self.y_original_[:, i] == j, i] = w[j]

        print(pd.DataFrame(sample_weight))
        sys.exit()

        return sample_weight

    @staticmethod
    def scorer(y_true, y_pred, **kwargs):
        # Get missing mask if provided.
        # Otherwise default is all missing values (array all True).
        missing_mask = kwargs.get(
            "missing_mask", np.ones(y_true.shape, dtype=bool)
        )

        testing = kwargs.get("testing", False)

        y_true_masked = y_true[missing_mask]
        y_pred_masked = y_pred[missing_mask]

        nn = NeuralNetworkMethods()
        y_pred_masked_decoded = nn.decode_masked(y_pred_masked)

        roc_auc = NeuralNetworkMethods.compute_roc_auc_micro_macro(
            y_true_masked, y_pred_masked_decoded
        )

        pr_ap = NeuralNetworkMethods.compute_pr(y_true_masked, y_pred_masked)

        acc = accuracy_score(y_true_masked, y_pred_masked_decoded)

        if testing:
            np.set_printoptions(threshold=np.inf)
            print(y_true_masked)
            print(y_pred_masked_decoded)

        metrics = dict()
        metrics["accuracy"] = acc
        metrics["roc_auc"] = roc_auc
        metrics["precision_recall"] = pr_ap

        return metrics

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
            ReduceLROnPlateau(
                patience=self.lr_patience, min_lr=1e-6, min_delta=1e-6
            ),
        ]

        search_mode = False if self.gridparams is None else True
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
                vinput[2] = self.nn_.init_weights(
                    y_train.shape[0], self.n_components
                )

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

    def _plot_history(self, lod):
        """Plot model history traces. Will be saved to file.

        Args:
            lod (List[tf.keras.callbacks.History]): List of history objects.
        """
        if self.nlpca:
            title = "NLPCA"
            fn = "histplot_nlpca.pdf"
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle("NLPCA")
            fig.tight_layout(h_pad=2.0, w_pad=2.0)
            history = lod[0]

            # Plot accuracy
            ax1.plot(history["categorical_accuracy"])
            ax1.set_title("Model Accuracy")
            ax1.set_ylabel("Accuracy")
            ax1.set_xlabel("Epoch")
            ax1.set_ylim(bottom=0.0, top=1.0)
            ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax1.legend(["Train"], loc="best")

            # Plot model loss
            ax2.plot(history["loss"])
            ax2.set_title("Model Loss")
            ax2.set_ylabel("Loss (MSE)")
            ax2.set_xlabel("Epoch")
            ax2.legend(["Train"], loc="best")
            fig.savefig(fn, bbox_inches="tight")

            plt.close()
            plt.clf()

        else:
            fig = plt.figure(figsize=(12, 16))
            fig.suptitle("UBP")
            fig.tight_layout(h_pad=2.0, w_pad=2.0)
            fn = "histplot_ubp.pdf"

            idx = 1
            for i, history in enumerate(lod, start=1):
                plt.subplot(3, 2, idx)
                title = f"Phase {i}"

                # Plot model accuracy
                ax = plt.gca()
                ax.plot(history["accuracy"])
                ax.set_title(f"{title} Accuracy")
                ax.set_ylabel("Accuracy")
                ax.set_xlabel("Epoch")
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.legend(["Training"], loc="best")

                # Plot model loss
                plt.subplot(3, 2, idx + 1)
                ax = plt.gca()
                ax.plot(history["loss"])
                ax.set_title(f"{title} Loss")
                ax.set_ylabel("Loss (MSE)")
                ax.set_xlabel("Epoch")
                ax.legend(["Train"], loc="best")

                idx += 2

            plt.savefig(fn, bbox_inches="tight")

            plt.close()
            plt.clf()
