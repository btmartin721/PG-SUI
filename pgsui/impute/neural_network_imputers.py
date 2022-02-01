# Standard Library Imports
import logging
import math
import os
import pprint
import random
import shutil
import sys
from collections import defaultdict
from itertools import cycle

# Third-party Imports
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import (
    compute_class_weight,
    compute_sample_weight,
)

# Grid search imports
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, label_binarize

# Randomized grid search imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterGrid

# Genetic algorithm grid search imports
from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import ConsecutiveStopping
from sklearn_genetic.plots import plot_fitness_evolution

from scikeras.wrappers import KerasClassifier

# For development purposes
# from memory_profiler import memory_usage

# Ignore warnings, but still print errors.
# Set to 0 for debugging, 2 to ignore warnings, 3 to ignore all but fatal.errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2', '3'}

# Neural network imports
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.python.util import deprecation
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ProgbarLogger,
    ReduceLROnPlateau,
    CSVLogger,
)
from tensorflow.keras.layers import Dropout, Dense, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2

# Custom Modules
try:
    from ..read_input.read_input import GenotypeData
    from ..utils.misc import timer
    from ..utils.misc import isnotebook
    from ..utils.misc import validate_input_type
    from .neural_network_methods import NeuralNetworkMethods
    from .nlpca_model import NLPCAModel
    from .ubp_model import UBPPhase1, UBPPhase2, UBPPhase3
    from ..data_processing.transformers import (
        UBPInputTransformer,
        NNInputTransformer,
        NNOutputTransformer,
        RandomizeMissingTransformer,
        ImputePhyloTransformer,
        ImputeAlleleFreqTransformer,
        ImputeNMFTransformer,
        SimGenotypeDataTransformer,
        TargetTransformer,
    )
except (ModuleNotFoundError, ValueError):
    from read_input.read_input import GenotypeData
    from utils.misc import timer
    from utils.misc import isnotebook
    from utils.misc import validate_input_type
    from impute.neural_network_methods import NeuralNetworkMethods
    from impute.nlpca_model import NLPCAModel
    from impute.ubp_model import UBPPhase1, UBPPhase2, UBPPhase3
    from data_processing.transformers import (
        UBPInputTransformer,
        NNInputTransformer,
        NNOutputTransformer,
        RandomizeMissingTransformer,
        ImputePhyloTransformer,
        ImputeAlleleFreqTransformer,
        ImputeNMFTransformer,
        SimGenotypeDataTransformer,
        TargetTransformer,
    )


# Ignore warnings, but still print errors.
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.get_logger().setLevel("ERROR")

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

    def on_train_begin(self, logs=None):
        self.model.n_batches = self.params.get("steps")

    def on_epoch_begin(self, epoch, logs=None):
        # Shuffle input and target at start of epoch.
        # input_with_mask = np.hstack([self.model.y, self.model.missing_mask])
        # n_samples = len(input_with_mask)
        y = self.model.y.copy()
        sample_weight = self.model.sample_weight
        missing_mask = self.model.missing_mask
        n_samples = len(y)
        self.indices = np.arange(n_samples)
        np.random.shuffle(self.indices)
        self.model.y = y[self.indices]
        self.model.V_latent = self.model.V_latent[self.indices]
        self.model.missing_mask = missing_mask[self.indices]
        self.model.sample_weight = sample_weight[self.indices]

    def on_train_batch_begin(self, batch, logs=None):
        self.model.batch_idx = batch

    def on_test_batch_begin(self, batch, logs=None):
        self.model.batch_idx = batch

    def on_epoch_end(self, epoch, logs=None):
        # Unsort the row indices.
        unshuffled = np.argsort(self.indices)
        self.model.y = self.model.y[unshuffled]
        self.model.V_latent = self.model.V_latent[unshuffled]
        self.model.missing_mask = self.model.missing_mask[unshuffled]
        self.model.sample_weight = self.model.sample_weight[unshuffled]

        # y_pred = self.model.predict(self.model.V_latent)
        # y_true = self.model.y.copy()
        # mm = self.model.missing_mask
        # old_weight = 0.5
        # y_true[mm] *= old_weight
        # y_pred_missing = y_pred[mm]
        # y_true[mm] += 0.5 * y_pred_missing
        # self.model.y = y_true.copy()


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

        serach_mode (bool): Whether in grid search mode (True) or single estimator mode (False). Defaults to True.

        optimizer (str or callable): Optimizer to use during training. Should be either a str or tf.keras.optimizers callable. Defaults to "adam".

        loss (str or callable): Loss function to use. Should be a string or a callable if using a custom loss function. Defaults to "binary_crossentropy".

        metrics (List[Union[str, callable]]): Metrics to use. Should be a sstring or a callable if using a custom metrics function. Defaults to "accuracy".

        epochs (int): Number of epochs to train over. Defaults to 100.

        verbose (int): Verbosity mode ranging from 0 - 2. 0 = silent, 2 = most verbose.

        kwargs (Any): Other keyword arguments to route to fit, compile, callbacks, etc. Should have the routing prefix (e.g., optimizer__learning_rate=0.01).
    """

    def __init__(
        self,
        V=None,
        y_decoded=None,
        y=None,
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
        num_classes=1,
        phase=None,
        n_components=3,
        search_mode=True,
        optimizer="adam",
        loss="binary_crossentropy",
        metrics="accuracy",
        epochs=100,
        verbose=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.V = V
        self.y_decoded = y_decoded
        self.y = y
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
        self.n_components = n_components
        self.search_mode = search_mode
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
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
                y=self.y,
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

        elif self.phase == 1:
            model = UBPPhase1(
                V=self.V,
                y=self.y,
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
                y=self.y,
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
                y=self.y,
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
    def scorer(
        y_true,
        y_pred,
        **kwargs,
    ):
        """Scorer for grid search that masks missing data.

        To use this, do not specify a scoring metric when initializing the grid search object. By default if the scoring_metric option is left as None, then it uses the estimator's scoring metric (this one).

        Args:
            y_true (numpy.ndarray): True target values input to fit().
            y_pred (numpy.ndarray): Predicted target values from estimator. The predictions are modified by self.target_encoder().inverse_transform() before being sent here.
            kwargs (Any): Other parameters sent to sklearn scoring metric.

        Returns:
            float: Calculated score.
        """
        y_true = kwargs.get("y_orig").astype("int8").ravel()
        y_pred = y_pred.astype("int8").ravel()

        # Should be only simulated missing values.
        missing_mask = kwargs.get("missing_mask").ravel()

        # Get accuracy score.
        acc = accuracy_score(y_true[missing_mask], y_pred[missing_mask])
        print(acc)
        return acc

    @property
    def feature_encoder(self):
        """Handles feature input, X, before training.

        Returns:
            UBPInputTransformer: InputTransformer object that includes fit() and transform() methods to transform input before estimator fitting.
        """
        return UBPInputTransformer(
            self.phase, self.n_components, self.V, self.search_mode
        )

    @property
    def target_encoder(self):
        """Handles target input and output, y_true and y_pred, both before and after training.

        Returns:
            NNOutputTransformer: NNOutputTransformer object that includes fit(), transform(), and inverse_transform() methods.
        """
        return NNOutputTransformer(self.y_decoded)


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

        gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If using RandomizedSearchCV, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ``gridparams`` will be used for a randomized grid search with cross-validation. If using the genetic algorithm grid search (GASearchCV) by setting ``ga=True``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. If ``gridparams=None``\, a grid search is not performed. Defaults to None.

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

        ga (bool, optional): Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead. Defaults to False.

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
        missing_mask_val=None,
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
        cv=5,
        grid_iter=50,
        ga=False,
        ga_kwargs=None,
        scoring_metric="accuracy",
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
        self.missing_mask_val = missing_mask_val
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
        self.cv = cv
        self.verbose = verbose
        self.validation_mode = validation_mode

        # TODO: Make estimators compatible with variable number of classes.
        # E.g., with morphological data.
        self.num_classes = 3

        # Grid search settings.
        self.ga = ga
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
        self.y_original_ = y.copy()

        nn = NeuralNetworkMethods()
        self.nn_ = nn

        # Simulate missing data and get missing masks.
        sim = SimGenotypeDataTransformer(
            self.genotype_data,
            prop_missing=self.sim_prop_missing,
            strategy=self.sim_strategy,
            mask_missing=True,
        )

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
        # nn.plot_grid_search(testresults)
        # sys.exit()

        # Placeholder for V. Gets replaced in model.
        V = nn.init_weights(y.shape[0], self.n_components)

        (
            logfile,
            callbacks,
            compile_params,
            model_params,
            fit_params,
        ) = self._initialize_parameters(
            V, self.y_simulated_, y_train, self.original_missing_mask_, nn
        )

        if self.nlpca:
            (
                self.models_,
                self.histories_,
                self.best_params_,
                self.best_score_,
                self.metrics_,
            ) = self._run_nlpca(
                V,
                y_train,
                model_params,
                compile_params,
                fit_params,
                nn,
            )

            # if self.gridparams is not None:
            #     if self.verbose > 0:
            #         print("\nBest found parameters:")
            #         pprint.pprint(self.best_params_)
            #         print(f"\nBest score: {self.best_score_}")
            #     nn.plot_grid_search(self.search_.cv_results_)

        else:
            self.models_, self.histories_ = self._run_ubp(
                V, y_train, model_params, compile_params, fit_params, nn
            )

        self._plot_history(self.histories_)
        self._plot_metrics(self.metrics_)

        return self

    def transform(self, X):
        """Predict and decode imputations and return transformed array.

        Args:
            X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): Input data to transform.

        Returns:
            numpy.ndarray: Imputed data.
        """
        y = X
        nn = NeuralNetworkMethods()

        # Get last (i.e., most refined) model.
        if len(self.models_) == 1:
            imputed_enc = self.models_[0].y.copy()
        else:
            imputed_enc = self.models_[-1].y.copy()

        # Has to be y and not y_train because it's not supposed to be one-hot
        # encoded.
        imputed_enc, dummy_df = nn.predict(self.y_simulated_, imputed_enc)

        imputed_df = nn.decode_onehot(
            pd.DataFrame(data=imputed_enc, columns=dummy_df.columns)
        )

        return imputed_df.to_numpy()

    def dict_mean(self, dict_list):
        keys = list({k for d in dict_list for k in d.keys()})
        mean_dict = {}
        for key in keys:
            mean_dict[key] = float(
                sum(d[key] for d in dict_list if key in d)
            ) / len(dict_list)
        return mean_dict

    def _run_nlpca(
        self,
        V,
        y_train,
        model_params,
        compile_params,
        fit_params,
        nn,
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
        nn.reset_seeds()

        y_true = self.y_original_

        # Get class weights for each column.
        sample_weights = self._get_class_weights(y_true)
        model_params["sample_weight"] = sample_weights

        if do_gridsearch:
            compile_params["learning_rate"] = self.learning_rate

            # Cannot do CV because there is no way to use test splits
            # given that the input gets refined. If using a test split,
            # then it would just be the randomly initialized values and
            # would not accurately represent the model.
            # Thus, we disable cross-validation for the grid searches.
            # cross_val = DisabledCV()

            if "learning_rate" in self.gridparams:
                self.gridparams["optimizer__learning_rate"] = self.gridparams[
                    "learning_rate"
                ]
                self.gridparams.pop("learning_rate")

            if self.ga:
                # Stop searching if GA sees no improvement.
                callback = ConsecutiveStopping(
                    generations=self.early_stop_gen, metric="fitness"
                )

                # Do genetic algorithm
                with HiddenPrints():
                    search = GASearchCV(
                        estimator=clf,
                        cv=cross_val,
                        scoring=self.scoring_metric,
                        generations=self.grid_iter,
                        param_grid=self.gridparams,
                        n_jobs=self.n_jobs,
                        verbose=False,
                        **self.ga_kwargs,
                    )

                    search.fit(V, y_train, callbacks=callback)

                best_params = search.best_params_
                best_score = search.best_score_
                best_index = search.best_index_

                cv_results = pd.DataFrame(search.cv_results_)
                cv_results.to_csv("testcvresults.csv", index=False)

            else:
                # Get all possible permutations for grid search.
                grid = list(ParameterGrid(self.gridparams))
                total_tests = len(grid)
                optimizer_callable = compile_params["optimizer"]

                scores = list()
                for i, perms in enumerate(grid, start=1):
                    if self.verbose > 0:
                        print(
                            f"\nGrid search permutation {i} / {total_tests} "
                            f"({(i / total_tests) * 100})%\n"
                        )

                    (
                        model_params,
                        compile_params,
                        fit_params,
                        perms,
                    ) = self._prep_model_arguments(
                        model_params, compile_params, fit_params, perms
                    )

                    clf = self._create_model(
                        NLPCAModel, model_params, compile_params, perms
                    )

                    history = clf.fit(
                        V,
                        y_train,
                        sample_weight=sample_weights,
                        **fit_params,
                    )

                    y_pred_proba = clf(clf.V_latent, training=False)
                    y_pred = self.tt_.inverse_transform(y_pred_proba)

                    # Get metric scores.
                    metrics = self.scorer(
                        y_true,
                        y_pred,
                        missing_mask=self.sim_missing_mask_,
                    )

                    scores.append(metrics["roc_auc"]["macro"])

                    # Reset optimizer to callable.
                    compile_params["optimizer"] = optimizer_callable

                best_index = np.argmax(scores)
                best_score = scores[best_index]
                best_params = grid[best_index]
                print(best_index)
                print(best_params)

            model_params, compile_params, fit_params = self._set_best_arguments(
                model_params, compile_params, fit_params, best_params, y_train
            )

            model = self._create_model(
                NLPCAModel, model_params, compile_params, best_params
            )

            history = model.fit(
                V,
                y_train,
                sample_weight=sample_weights,
                **fit_params,
            )

            histories.append(history.history)

            y_pred_proba = model(model.V_latent, training=False)
            y_pred = self.tt_.inverse_transform(y_pred_proba)

            # Get metric scores.
            best_metrics = self.scorer(
                y_true,
                y_pred,
                missing_mask=self.sim_missing_mask_,
            )

        else:
            model = self._create_model(NLPCAModel, model_params, compile_params)

            history = model.fit(
                V,
                y_train,
                # sample_weight=sample_weights,
                **fit_params,
            )

            best_params = None
            best_score = None

            histories.append(history.history)

            y_pred_proba = model(model.V_latent.copy(), training=False)
            y_pred = self.tt_.inverse_transform(y_pred_proba)

            # Get metric scores.
            best_metrics = self.scorer(
                y_true,
                y_pred,
                missing_mask=self.sim_missing_mask_,
                testing=True,
            )

        models.append(model)
        return models, histories, best_params, best_score, best_metrics

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
                    self.ga,
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

    def _get_class_weights(self, y_true):
        """Get class weights for each column in a 2D matrix.

        Args:
            y_true (numpy.ndarray): True target values.

        Returns:
            numpy.ndarray: Class weights per column of shape (n_samples, n_features).
        """
        # Get list of class_weights (per-column).
        class_weights = list()
        for i in np.arange(y_true.shape[1]):
            mm = ~self.original_missing_mask_[:, i]
            classes = np.unique(y_true[mm, i]).astype(np.int)
            cw = compute_class_weight(
                "balanced",
                classes=classes,
                y=y_true[mm, i],
            )
            class_weights.append({k: v for k, v in zip(classes, cw)})

        # Make sample_weight_matrix from per-column class_weights.
        sample_weight = np.zeros(y_true.shape)
        for i, w in enumerate(class_weights):
            for j in range(3):
                if j in w:
                    sample_weight[self.y_original_[:, i] == j, i] = w[j]

        return sample_weight

    def _prep_model_arguments(
        self, model_params, compile_params, fit_params, perms
    ):
        """Prepare model, compile, and fit parameters for use with model.

        Sets arguments for current permutation and removes duplicate arguments when using perms dictionary.
        """
        # Avoid duplicated arguments.
        perm_searched = {
            model_params.pop(k) for k in model_params.copy() if k in perms
        }

        if "optimizer__learning_rate" in perms:
            if "optimizer" in perms:
                compile_params["optimizer"] = perms["optimizer"](
                    learning_rate=perms["optimizer__learning_rate"]
                )
            else:
                compile_params["optimizer"] = compile_params["optimizer"](
                    learning_rate=perms["optimizer__learning_rate"]
                )

            perms.pop("optimizer__learning_rate")

        if "epochs" in perms:
            fit_params["epochs"] = perms.pop("epochs")

        if "batch_size" in perms:
            fit_params["batch_size"] = perms.pop("batch_size")

        if "learning_rate" in compile_params:
            compile_params.pop("learning_rate")

        return model_params, compile_params, fit_params, perms

    def _set_best_arguments(
        self, model_params, compile_params, fit_params, best_params, y_train
    ):
        """Set model parameters to best found under grid search."""
        if "learning_rate" in best_params:
            compile_params["optimizer"] = compile_params["optimizer"](
                learning_rate=best_params["learning_rate"]
            )

        else:
            compile_params["optimizer"] = compile_params["optimizer"](
                learning_rate=self.learning_rate
            )

        if "epochs" in best_params:
            fit_params["epochs"] = best_params["epochs"]

        if "batch_size" in best_params:
            fit_params["batch_size"] = best_params["batch_size"]

        model_searched = {
            k: best_params[k] for k in best_params if k in model_params
        }

        model_params.update(model_searched)

        # Get new random initial V.
        vinput = self.nn_.init_weights(
            y_train.shape[0], model_params["n_components"]
        )

        model_params["V"] = vinput

        return model_params, compile_params, fit_params

    @staticmethod
    def scorer(y_true, y_pred, **kwargs):
        # Get missing mask if provided.
        # Otherwise default is no missing values (array all False).
        missing_mask = kwargs.get(
            "missing_mask", np.zeros(y_true.shape, dtype=bool)
        )
        testing = kwargs.get("testing", False)

        if testing:
            print(y_true[missing_mask])
            print(y_pred[missing_mask])
        acc = accuracy_score(y_true[missing_mask], y_pred[missing_mask])
        roc_auc = UBP.compute_roc_auc_micro_macro(y_true, y_pred, missing_mask)

        metrics = dict()
        metrics["accuracy"] = acc
        metrics["roc_auc"] = roc_auc

        return metrics

    @staticmethod
    def compute_roc_auc_micro_macro(y_true, y_pred, missing_mask):
        # Binarize the output fo use with ROC-AUC.
        y_true_bin = label_binarize(y_true[missing_mask], classes=[0, 1, 2])
        y_pred_bin = label_binarize(y_pred[missing_mask], classes=[0, 1, 2])
        n_classes = y_true_bin.shape[1]

        # Compute ROC curve and ROC area for each class.
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area.
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_bin.ravel(), y_pred_bin.ravel()
        )

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at these points.
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally, average it and compute AUC.
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr

        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        roc_auc["fpr_macro"] = fpr["macro"]
        roc_auc["tpr_macro"] = tpr["macro"]
        roc_auc["fpr_micro"] = fpr["micro"]
        roc_auc["tpr_micro"] = tpr["micro"]
        roc_auc["fpr_0"] = fpr[0]
        roc_auc["fpr_1"] = fpr[1]
        roc_auc["fpr_2"] = fpr[2]
        roc_auc["tpr_0"] = tpr[0]
        roc_auc["tpr_1"] = tpr[1]
        roc_auc["tpr_2"] = tpr[2]

        return roc_auc

    def _initialize_parameters(self, V, y, y_train, missing_mask, nn):
        """Initialize important parameters.

        Args:
            V (numpy.ndarray): Initial V with randomly initialized values, of shape (n_samples, n_components).

            y (numpy.ndarray): Original input data to be used as target.

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

        compile_params = nn.set_compile_params(
            self.optimizer,
            self.learning_rate,
            self.all_missing_,
            search_mode=search_mode,
        )

        if search_mode:
            vinput = dict()
            if self.n_components < 2:
                raise ValueError("n_components must be >= 2.")
            elif self.n_components == 2:
                vinput[2] = nn.init_weights(y_train.shape[0], self.n_components)
            else:
                for i in range(2, self.n_components + 1):
                    vinput[i] = nn.init_weights(y_train.shape[0], i)
        else:
            vinput = nn.init_weights(y_train.shape[0], self.n_components)

        model_params = {
            "V": vinput,
            "y": y_train,
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

    def _plot_metrics(self, metrics):
        fn = "metrics_plot.pdf"
        fig = plt.figure(figsize=(10, 5))
        ax = fig.subplots(nrows=1, ncols=1)

        # Line weight
        lw = 2

        roc_auc = metrics["roc_auc"]

        # Plot ROC curves.
        ax.plot(
            roc_auc["fpr_micro"],
            roc_auc["tpr_micro"],
            label=f"Micro-averaged ROC Curve (AUC = {roc_auc['micro']:.2f})",
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        ax.plot(
            roc_auc["fpr_macro"],
            roc_auc["tpr_macro"],
            label=f"Macro-averaged ROC Curve (AUC = {roc_auc['macro']:.2f})",
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i, color in zip(range(self.num_classes), colors):
            ax.plot(
                roc_auc[f"fpr_{i}"],
                roc_auc[f"tpr_{i}"],
                color=color,
                lw=lw,
                label=f"ROC Curve of class {i} (AUC = {roc_auc[i]:.2f})",
            )

        # Make center line.
        ax.plot([0, 1], [0, 1], "k--", lw=lw)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive rate")

        acc = round(metrics["accuracy"] * 100, 2)

        ax.set_title(
            f"Receiver Operating Characteristic (ROC)\nAccuracy = {acc}%",
        )
        ax.legend(loc="best")

        fig.savefig(fn, bbox_inches="tight")
        plt.close()
        plt.clf()
        plt.cla()
        sys.exit()

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
            ax2.legend(["Training"], loc="best")
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
