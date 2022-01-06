# Standard Library Imports
import math
import os
import random
import shutil
import sys
from collections import defaultdict

# Third-party Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

# Grid search imports
from sklearn.model_selection import StratifiedKFold

# Randomized grid search imports
from sklearn.model_selection import RandomizedSearchCV

# Genetic algorithm grid search imports
from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import ConsecutiveStopping
from sklearn_genetic.plots import plot_fitness_evolution

from keras.wrappers.scikit_learn import KerasClassifier

# For development purposes
# from memory_profiler import memory_usage

# Ignore warnings, but still print errors.
# Set to 0 for debugging, 2 to ignore warnings, 3 to ignore all but fatal.errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2', '3'}

# Neural network imports
import tensorflow as tf
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

import keras.backend as K

# from keras.objectives import mse

# Custom Modules
try:
    from ..read_input.read_input import GenotypeData
    from ..utils.misc import timer
    from ..utils.misc import isnotebook
except (ModuleNotFoundError, ValueError):
    from read_input.read_input import GenotypeData
    from utils.misc import timer
    from utils.misc import isnotebook

# Ignore warnings, but still print errors.
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.get_logger().setLevel("ERROR")

is_notebook = isnotebook()

if is_notebook:
    from tqdm.notebook import tqdm as progressbar
else:
    from tqdm import tqdm as progressbar


class NeuralNetwork:
    """Methods common to all neural network imputer classes and loss functions"""

    def __init__(self):
        self.data = None

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

    def make_reconstruction_loss(self, n_features):
        """Make loss function for use with a keras model.

        Args:
            n_features (int): Number of features in input dataset.

        Returns:
            callable: Function that calculates loss.
        """

        def reconstruction_loss(input_and_mask, y_pred):
            """Custom loss function for variational autoencoder model with missing mask.

            Ignores missing data in the calculation of the loss function.

            Args:
                n_features (int): Number of features (columns) in the dataset.

                input_and_mask (numpy.ndarray): Input one-hot encoded array with missing values also one-hot encoded and h-stacked.

                y_pred (numpy.ndarray): Predicted values.

            Returns:
                float: Mean squared error loss value with missing data masked.
            """
            X_values = input_and_mask[:, :n_features]
            missing_mask = input_and_mask[:, n_features:]
            observed_mask = 1 - missing_mask
            X_values_observed = X_values * observed_mask
            pred_observed = y_pred * observed_mask

            return tf.keras.metrics.mean_squared_error(
                y_true=X_values_observed, y_pred=pred_observed
            )

        return reconstruction_loss

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

    def categorical_crossentropy_masked(self, y_true, y_pred):
        """Calculates categorical crossentropy while ignoring missing values.

        Used for UBP and NLPCA. Missing values should be an array of length(n_categories). If data is missing, it should be encoded as [-1] * n_categories.

        Args:
            y_true (tf.Tensor): Known values from input data.
            y_pred (tf.Tensor): Values predicted from model.

        Returns:
            float: Loss value.
        """
        y_true_masked = tf.boolean_mask(
            y_true, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
        )

        y_pred_masked = tf.boolean_mask(
            y_pred, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
        )

        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        return loss_fn(y_true_masked, y_pred_masked)

    def categorical_accuracy_masked(self, y_true, y_pred):
        y_true_masked = tf.boolean_mask(
            y_true, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
        )

        y_pred_masked = tf.boolean_mask(
            y_pred, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
        )

        metric = tf.keras.metrics.CategoricalAccuracy()
        metric.update_state(y_true_masked, y_pred_masked)

        return metric.result().numpy()

    def categorical_mse_masked(self, y_true, y_pred):
        y_true_masked = tf.boolean_mask(
            y_true, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
        )

        y_pred_masked = tf.boolean_mask(
            y_pred, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
        )

        return mse(y_true_masked, y_pred_masked)

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

    def mle(self, row):
        """Get the Maximum Likelihood Estimation for the best prediction. Basically, it sets the index of the maxiumum value in a vector (row) to 1.0, since it is one-hot encoded.

        Args:
            row (numpy.ndarray(float)): Row vector with predicted values as floating points.

        Returns:
            numpy.ndarray(float): Row vector with the highest prediction set to 1.0 and the others set to 0.0.
        """
        res = np.zeros(row.shape[0])
        res[np.argmax(row)] = 1
        return res

    def fill(self, data, missing_mask, missing_value, num_classes):
        """Mask missing data as ``missing_value``.

        Args:
            data (numpy.ndarray): Input with missing values of shape (n_samples, n_features, num_classes).

            missing_mask (np.ndarray(bool)): Missing data mask with True corresponding to a missing value.

            missing_value (int): Value to set missing data to. If a list is provided, then its length should equal the number of one-hot classes.
        """
        if num_classes > 1:
            missing_value = [missing_value] * num_classes
        data[missing_mask] = missing_value

    def validate_extrakwargs(self, d):
        """Validate extra keyword arguments.

        Args:
            d (Dict[str, Any]): Dictionary with keys=keywords and values=passed setting.

        Returns:
            numpy.ndarray: Test categorical dataset.

        Raises:
            ValueError: If not a supported keyword argument.
        """
        supported_kwargs = ["test_categorical"]
        if kwargs is not None:
            for k in kwargs.keys():
                if k not in supported_kwargs:
                    raise ValueError(f"{k} is not a valid argument.")

        test_cat = kwargs.get("test_categorical", None)
        return test_cat

    def validate_input(self, input_data, out_type="numpy"):
        """Validate input data and ensure that it is of correct type.

        Args:
            input_data (List[List[int]], numpy.ndarray, or pandas.DataFrame): Input data to validate.

            out_type (str, optional): Type of object to convert input data to. Possible options include "numpy" and "pandas". Defaults to "numpy".

        Returns:
            numpy.ndarray: Input data as numpy array.

        Raises:
            TypeError: Must be of type pandas.DataFrame, numpy.ndarray, or List[List[int]].

            ValueError: astype must be either "numpy" or "pandas".
        """
        if out_type == "numpy":
            if isinstance(input_data, pd.DataFrame):
                X = input_data.to_numpy()
            elif isinstance(input_data, list):
                X = np.array(input_data)
            elif isinstance(input_data, np.ndarray):
                X = input_data.copy()
            else:
                raise TypeError(
                    f"input_data must be of type pd.DataFrame, np.ndarray, or "
                    f"list(list(int)), but got {type(input_data)}"
                )

        elif out_type == "pandas":
            if isinstance(input_data, pd.DataFrame):
                X = input_data.copy()
            elif isinstance(input_data, (list, np.ndarray)):
                X = pd.DataFrame(input_data)
            else:
                raise TypeError(
                    f"input_data must be of type pd.DataFrame, np.ndarray, or "
                    f"list(list(int)), but got {type(input_data)}"
                )
        else:
            raise ValueError("astype must be either 'numpy' or 'pandas'.")

        return X

    def set_optimizer(self):
        """Set optimizer to use.

        Returns:
            tf.keras.optimizers: Initialized optimizer.

        Raises:
            ValueError: Unsupported optimizer specified.
        """
        if self.optimizer == "adam":
            return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        elif self.optimizer == "adagrad":
            return tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)
        else:
            raise ValueError(
                f"Only 'adam', 'sgd', and 'adagrad' optimizers are supported, "
                f"but got {self.optimizer}."
            )

    def grid_search(
        self,
        X_train,
        y_train,
        estimator,
        grid_cv,
        ga,
        early_stop_gen,
        scoring_metric,
        grid_n_iter,
        search_space,
        grid_n_jobs,
        ga_kwargs,
    ):
        # Modified code
        cross_val = StratifiedKFold(n_splits=grid_cv, shuffle=False)

        if ga:
            callback = ConsecutiveStopping(
                generations=early_stop_gen, metric="fitness"
            )

            # Do genetic algorithm
            with HiddenPrints():
                search = GASearchCV(
                    estimator=estimator,
                    cv=cross_val,
                    scoring=scoring_metric,
                    generations=grid_n_iter,
                    param_grid=search_space,
                    n_jobs=grid_n_jobs,
                    verbose=False,
                    **ga_kwargs,
                )

                search.fit(X_train, y_train, callbacks=callback)

        else:
            # Do randomized grid search
            search = RandomizedSearchCV(
                estimator,
                param_distributions=search_space,
                n_iter=grid_n_iter,
                scoring=scoring_metric,
                n_jobs=grid_n_jobs,
                cv=cross_val,
            )

            search.fit(X_train, y_train)

        return search


class UBPCallbacks(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.model.n_batches = self.params.get("steps")
        # if self.model.phase == 3:
        #     self.model.set_weights(self.model.phase2_weights)

    def on_train_batch_begin(self, batch, logs=None):
        self.model.batch_idx = batch

    # def on_train_batch_end(self, batch, logs=None):
    #     if self.model.phase != 2:
    #         batch_size = self.model.batch_size

    # def on_train_end(self, logs=None):
    #     if self.model.phase == 2:
    #         # UBP model phase == 2: Save the weights from Phase 2.
    #         self.model.phase2_weights = self.model.get_weights()

    def on_test_batch_begin(self, batch, logs=None):
        self.model.batch_idx = batch


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
                self.best_input = self.model.V
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

                if self.phase != 2:
                    self.model.V = self.best_input


class NLPCAModel(tf.keras.Model, NeuralNetwork):
    """NLPCA model to train and use to predict imputations.

    UBPModel subclasses the tf.keras.Model and overrides the train_step() function, which does training for one batch in a single epoch.

    Args:
        V (numpy.ndarray(float)): If phase == 1 or nlpca == True, then V is randomly initialized and used as the input data that gets refined during training.

        output_shape (int): Output units for n_features dimension. Output will be of shape (batch_size, output_shape, num_classes).

        n_components (int): Number of principal components for V.

        weights_initializer (str): Kernel initializer to use for initializing model weights.

        hidden_layer_sizes (List[int]): Output units for each hidden layer. List should be of same length as the number of hidden layers.

        hidden_activation (str): Activation function to use for hidden layers.

        l1_penalty (float): L1 regularization penalty to use to reduce overfitting.

        l2_penalty (float): L2 regularization penalty to use to reduce overfitting.

        dropout_rate (float): Dropout rate during training to reduce overfitting. Must be a float between 0 and 1.

        num_classes (int, optional): Number of classes in output. Corresponds to the 3rd dimension of the output shape (batch_size, n_features, num_classes). Defaults to 3.

        phase (NoneType): Here for compatibility with UBP.


    Methods:
        call: Does forward pass for model.
        train_step: Does training for one batch in a single epoch.
        test_step: Does evaluation for one batch in a single epoch.

    Attributes:
        _V (numpy.ndarray(float)): Randomly initialized input that gets refined during training.

        hidden_layer_sizes (List[Union[int, str]]): Output units for each hidden layer. Length should be the same as the number of hidden layers.

        n_components (int): Number of principal components to use with _V.

        _batch_size (int): Batch size to use per epoch.

        _batch_idx (int): Index of current batch.

        _n_batches (int): Total number of batches per epoch.

    Example:
        >>>model = NLPCAModel(V, output_shape, n_components, weights_initializer, hidden_layer_sizes, hidden_activation, l1_penalty, l2_penalty, dropout_rate, num_classes=3)
        >>>model.compile(optimizer=optimizer, loss=loss_func, metrics=[my_metrics], run_eagerly=True)
        >>>history = model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[MyCallback()], validation_split=validation_split, shuffle=False)
    """

    def __init__(
        self,
        V,
        output_shape,
        n_components,
        weights_initializer,
        hidden_layer_sizes,
        hidden_activation,
        l1_penalty,
        l2_penalty,
        dropout_rate,
        num_classes=3,
        phase=None,
    ):
        super(NLPCAModel, self).__init__()

        self._V = V
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_components = n_components
        self.weights_initializer = weights_initializer
        self.phase = phase

        # Initialize parameters that are set via callbacks during fit().
        self._batch_size = 0
        self._batch_idx = 0
        self._n_batches = 0
        self._V_latent = self._V.copy()

        kernel_regularizer = l1_l2(l1_penalty, l2_penalty)
        self.kernel_regularizer = kernel_regularizer
        kernel_initializer = weights_initializer

        if len(hidden_layer_sizes) > 5:
            raise ValueError("The maximum number of hidden layers is 5.")

        self.dense2 = None
        self.dense3 = None
        self.dense4 = None
        self.dense5 = None

        # Construct multi-layer perceptron.
        # Add hidden layers dynamically.
        self.dense1 = Dense(
            hidden_layer_sizes[0],
            input_shape=(n_components,),
            activation=hidden_activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )

        if len(hidden_layer_sizes) >= 2:
            self.dense2 = Dense(
                hidden_layer_sizes[1],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if len(hidden_layer_sizes) >= 3:
            self.dense3 = Dense(
                hidden_layer_sizes[2],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if len(hidden_layer_sizes) >= 4:
            self.dense4 = Dense(
                hidden_layer_sizes[3],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if len(hidden_layer_sizes) == 5:
            self.dense5 = Dense(
                hidden_layer_sizes[4],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        self.dense6 = Dense(
            output_shape,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )

        # Expand dims to 3d to shape (batch_size, n_features, num_classes).
        self.lmda = Lambda(lambda x: tf.expand_dims(x, -1))
        self.output1 = Dense(num_classes, activation="softmax",)

        self.dropout_layer = Dropout(rate=dropout_rate)

    def call(self, inputs, training=None):
        """Forward propagates inputs through the model defined in __init__().

        Args:
            inputs (tf.keras.Input): Input tensor to forward propagate through the model.

            training (bool or None): Whether in training mode or not. Affects whether dropout is used.

        Returns:
            tf.keras.Model: Output tensor from forward propagation.
        """
        x = self.dense1(inputs)
        if self.dense2 is not None:
            x = self.dense2(x)
            x = self.dropout_layer(x, training=training)
        if self.dense3 is not None:
            x = self.dense3(x)
            x = self.dropout_layer(x, training=training)
        if self.dense4 is not None:
            x = self.dense4(x)
            x = self.dropout_layer(x, training=training)
        if self.dense5 is not None:
            x = self.dense5(x)
            x = self.dropout_layer(x, training=training)
        x = self.dense6(x)
        x = self.dropout_layer(x, training=training)

        x = self.lmda(x)
        return self.output1(x)

    def model(self):
        """Here so that mymodel.model().summary() can be called for debugging.
        """
        x = tf.keras.Input(shape=(self.n_components,))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def train_step(self, data):
        """Custom training loop for one step (=batch) in a single epoch.

        GradientTape records the weights and watched
        variables (usually tf.Variable objects), which
        in this case are the weights and the input (x),
        during the forward pass.
        This allows us to run gradient descent during
        backpropagation to refine the watched variables.

        This function will train on a batch of samples (rows), which can be adjusted with the ``batch_size`` parameter from the estimator.

        Args:
            data (Tuple[tf.EagerTensor, tf.EagerTensor]): Input tensorflow variables of shape (batch_size, n_components) and (batch_size, n_features, num_classes).

        Returns:
            Dict[str, float]: History object that gets returned from fit(). Contains the loss and any metrics specified in compile().

        ToDo:
            Obtain batch_size without using run_eagerly option in compile(). This will allow the step to be run in graph mode, thereby speeding up computation.
        """
        y_true = data[1]

        # Get current batch_size.
        # NOTE: run_eagerly must be set to True in the compile() method for this
        # to work. Otherwise it can't convert a Tensor object to a numpy array.
        # There is really no other way to get the batch_size in graph
        # mode as far as I know. eager execution is slower, so it would be nice
        # to find a way to obtain batch_size without converting to numpy.
        batch_size = y_true.numpy().shape[0]

        # self._batch_idx is set in the UBPCallbacks() callback
        # on_train_batch_begin() method.
        batch_start = self._batch_idx * batch_size
        batch_end = (self._batch_idx + 1) * batch_size

        # override v_batch. This model refines the input to fit the output, so
        # v_batch has to be overridden.
        v_batch = self._V_latent[batch_start:batch_end, :]

        # NOTE: Moved self.set_weights(self._phase2_weights[self._batch_idx])
        # to custom callback because it threw an error when it was here.

        # Initialize variable v as tensorflow variable.
        v = tf.Variable(
            tf.zeros([batch_size, self.n_components]),
            trainable=True,
            dtype=tf.float32,
        )

        # Assign current batch to v.
        v.assign(v_batch)

        src = [v]

        with tf.GradientTape() as tape:
            # Forward pass
            tape.watch(v)
            y_pred = self(v, training=True)

            loss = self.compiled_loss(
                tf.convert_to_tensor(y_true, dtype=tf.float32),
                y_pred,
                regularization_losses=self.losses,
            )

        src.extend(self.trainable_variables)

        # Refine the watched variables with
        # gradient descent backpropagation
        gradients = tape.gradient(loss, src)
        self.optimizer.apply_gradients(zip(gradients, src))

        self.compiled_metrics.update_state(
            tf.convert_to_tensor(y_true, dtype=tf.float32), y_pred
        )

        self._V_latent[batch_start:batch_end, :] = v.numpy()

        # history object that gets returned from fit().
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Validation step for one batch in a single epoch.

        Custom logic for the test step that gets sent back to on_train_batch_end() callback.

        Args:
            data (Tuple[tf.EagerTensor, tf.EagerTensor]): Batches of input data V and y_true.

        Returns:
            A dict containing values that will be passed to ``tf.keras.callbacks.CallbackList.on_train_batch_end``. Typically, the values of the Model's metrics are returned.
        """
        # Unpack the data. Don't need V here. Just X (y_true).
        y_true = data[1]

        batch_size = y_true.numpy().shape[0]

        # self._batch_idx is set in the UBPCallbacks() callback
        # on_train_batch_begin() method.
        batch_start = self._batch_idx * batch_size
        batch_end = (self._batch_idx + 1) * batch_size

        # Get the input that's being refined.
        v_batch = self._V_latent[batch_start:batch_end, :]

        v = tf.Variable(
            tf.zeros([batch_size, self.n_components]),
            trainable=False,
            dtype=tf.float32,
        )

        # Assign current batch to v.
        v.assign(v_batch)

        # Compute predictions
        y_pred = self(v, training=False)

        # Updates the metrics tracking the loss
        self.compiled_loss(
            tf.convert_to_tensor(y_true, dtype=tf.float32),
            y_pred,
            regularization_losses=self.losses,
        )

        # Update the metrics.
        self.compiled_metrics.update_state(
            tf.convert_to_tensor(y_true, dtype=tf.float32), y_pred
        )

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    @property
    def V(self):
        """Randomly initialized input variable that gets refined during training."""
        return self._V_latent

    @property
    def batch_size(self):
        """Batch (=step) size per epoch."""
        return self._batch_size

    @property
    def batch_idx(self):
        """Current batch (=step) index."""
        return self._batch_idx

    @property
    def n_batches(self):
        """Total number of batches per epoch."""
        return self._n_batches

    @V.setter
    def V(self, value):
        """Set randomly initialized input variable. Gets refined during training."""
        self._V = value

    @batch_size.setter
    def batch_size(self, value):
        """Set batch_size parameter."""
        self._batch_size = int(value)

    @batch_idx.setter
    def batch_idx(self, value):
        """Set current batch (=step) index."""
        self._batch_idx = int(value)

    @n_batches.setter
    def n_batches(self, value):
        """Set total number of batches (=steps) per epoch."""
        self._n_batches = int(value)


class UBPPhase1(tf.keras.Model, NeuralNetwork):
    """UBP Phase 1 single layer perceptron model to train predict imputations.

    UBPPhase1 subclasses the tf.keras.Model and overrides the train_step() and test_step() functions, which do training and evalutation for one batch in a single epoch.

    Args:
        V (numpy.ndarray(float)): If phase == 1 or nlpca == True, then V is randomly initialized and used as the input data that gets refined during training.

        output_shape (int): Output units for n_features dimension. Output will be of shape (batch_size, output_shape, num_classes).

        n_components (int): Number of principal components for V.

        weights_initializer (str): Kernel initializer to use for initializing model weights.

        hidden_layer_sizes (List[int]): Output units for each hidden layer. List should be of same length as the number of hidden layers.

        hidden_activation (str): Activation function to use for hidden layers.

        l1_penalty (float): L1 regularization penalty to use to reduce overfitting.

        l2_penalty (float): L2 regularization penalty to use to reduce overfitting.

        phase (int, optional): Current phase if doing UBP model. Defaults to 3.

        num_classes (int, optional): Number of classes in output. Corresponds to the 3rd dimension of the output shape (batch_size, n_features, num_classes). Defaults to 3.

    Methods:
        call: Does forward pass for model.
        train_step: Does training for one batch in a single epoch.
        test_step: Does evaluation for one batch in a single epoch.

    Attributes:
        _V (numpy.ndarray(float)): Randomly initialized input that gets refined during training.

        phase (int): Current phase if doing UBP model. Ignored if doing NLPCA model.

        hidden_layer_sizes (List[Union[int, str]]): Output units for each hidden layer. Length should be the same as the number of hidden layers.

        n_components (int): Number of principal components to use with _V.

        _batch_size (int): Batch size to use per epoch.

        _batch_idx (int): Index of current batch.

        _n_batches (int): Total number of batches per epoch.

    Example:
        >>>model = UBPPhase1(V, output_shape, n_components, weights_initializer, hidden_layer_sizes, hidden_activation, l1_penalty, l2_penalty, phase=3, num_classes=3)
        >>>model.compile(optimizer=optimizer, loss=loss_func, metrics=[my_metrics], run_eagerly=True)
        >>>history = model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[MyCallback()], validation_split=validation_split, shuffle=False)
    """

    def __init__(
        self,
        V,
        output_shape,
        n_components,
        weights_initializer,
        hidden_layer_sizes,
        hidden_activation,
        l1_penalty,
        l2_penalty,
        phase=3,
        num_classes=3,
    ):
        super(UBPPhase1, self).__init__()

        self._V = V
        self.phase = phase
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_components = n_components
        self.weights_initializer = weights_initializer

        ### NOTE: I tried using just _V as the input to be refined, but it
        # wasn't getting updated. So I copy it here and it works.
        self._V_latent = self._V.copy()

        # Initialize parameters that are set via callbacks during fit().
        self._batch_size = 0
        self._batch_idx = 0
        self._n_batches = 0

        kernel_regularizer = l1_l2(l1_penalty, l2_penalty)
        self.kernel_regularizer = kernel_regularizer
        kernel_initializer = weights_initializer

        # Construct single-layer perceptron.
        self.dense1 = Dense(
            output_shape,
            input_shape=(n_components,),
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            activation=hidden_activation,
        )

        # Expand dims to 3d (batch_size, n_features, num_classes)
        self.lmda = Lambda(lambda x: tf.expand_dims(x, -1))
        self.output1 = Dense(num_classes, activation="softmax",)

    def call(self, inputs):
        """Forward propagates inputs through the model defined in __init__().

        Args:
            inputs (tf.keras.Input): Input tensor to forward propagate through the model.

        Returns:
            tf.keras.Model: Output tensor from forward propagation.
        """
        x = self.dense1(inputs)
        x = self.lmda(x)
        return self.output1(x)

    def model(self):
        """Here so that mymodel.model().summary() can be called for debugging.
        """
        x = tf.keras.Input(shape=(self.n_components,))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def train_step(self, data):
        """Custom training loop for one step (=batch) in a single epoch.

        GradientTape records the weights and watched variables (usually tf.Variable objects), which in this case are the weights and the input (x), during the forward pass. This allows us to run gradient descent during backpropagation to refine the watched variables.

        This function will train on a batch of samples (rows), which can be adjusted with the ``batch_size`` parameter from the estimator.

        Args:
            data (Tuple[tf.EagerTensor, tf.EagerTensor]): Tuple of input tensors of shape (batch_size, n_components) and (batch_size, n_features, num_classes).

        Returns:
            Dict[str, float]: History object that gets returned from fit(). Contains the loss and any metrics specified in compile().

        ToDo:
            Obtain batch_size without using run_eagerly option in compile(). This will allow the step to be run in graph mode, thereby speeding up computation.
        """
        # y_true. Actual dataset that gets used to refine X.
        y_true = data[1]

        # Get current batch_size.
        # NOTE: run_eagerly must be set to True in the compile() method for this
        # to work. Otherwise it can't convert a Tensor object to a numpy array.
        # There is really no other way to get the batch_size in graph
        # mode as far as I know. eager execution is slower, so it would be nice
        # to find a way to obtain batch_size without converting to numpy.
        batch_size = y_true.numpy().shape[0]

        # self._batch_idx is set in the UBPCallbacks() callback
        # on_train_batch_begin() method.
        batch_start = self._batch_idx * batch_size
        batch_end = (self._batch_idx + 1) * batch_size

        # override v_batch. This model refines the input to fit the output, so
        # v_batch has to be overridden.
        v_batch = self._V_latent[batch_start:batch_end, :].copy()

        # NOTE: Moved self.set_weights(self._phase2_weights[self._batch_idx])
        # to custom callback because it threw an error when it was here.

        v = tf.Variable(
            tf.zeros([batch_size, self.n_components]),
            trainable=True,
            dtype=tf.float32,
        )

        # Assign current batch to tf.Variable v.
        v.assign(v_batch)
        src = [v]

        with tf.GradientTape() as tape:
            # Forward pass
            tape.watch(v)
            y_pred = self(v, training=True)
            loss = self.compiled_loss(
                tf.convert_to_tensor(y_true, dtype=tf.float32),
                y_pred,
                regularization_losses=self.losses,
            )

        src.extend(self.trainable_variables)

        # Refine the watched variables with
        # gradient descent backpropagation
        gradients = tape.gradient(loss, src)
        self.optimizer.apply_gradients(zip(gradients, src))

        self.compiled_metrics.update_state(
            tf.convert_to_tensor(y_true, dtype=tf.float32), y_pred
        )

        self._V_latent[batch_start:batch_end, :] = v.numpy().copy()

        # history object that gets returned from fit().
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Validation step for one batch in a single epoch.

        Custom logic for the test step that gets sent back to on_train_batch_end() callback.

        Args:
            data (Tuple[tf.EagerTensor, tf.EagerTensor]): Batches of input data V and y_true.

        Returns:
            A dict containing values that will be passed to ``tf.keras.callbacks.CallbackList.on_train_batch_end``. Typically, the values of the Model's metrics are returned.
        """
        # Unpack the data. Don't need V here. Just X (y_true).
        y_true = data[1]
        batch_size = y_true.numpy().shape[0]

        # self._batch_idx is set in the UBPCallbacks() callback
        # on_train_batch_begin() method.
        batch_start = self._batch_idx * batch_size
        batch_end = (self._batch_idx + 1) * batch_size

        # Get the input that's being refined.
        v_batch = self._V_latent[batch_start:batch_end, :].copy()

        v = tf.Variable(
            tf.zeros([batch_size, self.n_components]),
            trainable=False,
            dtype=tf.float32,
        )

        # Assign current batch to v.
        v.assign(v_batch)

        # Compute predictions
        y_pred = self(v, training=False)

        # Updates the metrics tracking the loss
        self.compiled_loss(
            tf.convert_to_tensor(y_true, dtype=tf.float32),
            y_pred,
            regularization_losses=self.losses,
        )

        # Update the metrics.
        self.compiled_metrics.update_state(
            tf.convert_to_tensor(y_true, dtype=tf.float32), y_pred
        )

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    @property
    def V(self):
        """Randomly initialized input variable that gets refined during training."""
        return self._V_latent

    @property
    def batch_size(self):
        """Batch (=step) size per epoch."""
        return self._batch_size

    @property
    def batch_idx(self):
        """Current batch (=step) index."""
        return self._batch_idx

    @property
    def n_batches(self):
        """Total number of batches per epoch."""
        return self._n_batches

    @V.setter
    def V(self, value):
        """Set randomly initialized input variable. Gets refined during training."""
        self._V = value

    @batch_size.setter
    def batch_size(self, value):
        """Set batch_size parameter."""
        self._batch_size = int(value)

    @batch_idx.setter
    def batch_idx(self, value):
        """Set current batch (=step) index."""
        self._batch_idx = int(value)

    @n_batches.setter
    def n_batches(self, value):
        """Set total number of batches (=steps) per epoch."""
        self._n_batches = int(value)


class UBPPhase2(tf.keras.Model, NeuralNetwork):
    """UBP Phase 2 model to train and use to predict imputations.

    UBPModel subclasses the tf.keras.Model and overrides the train_step() and test_step() functions, which do training for one batch in a single epoch.

    Args:
        output_shape (int): Output units for n_features dimension. Output will be of shape (batch_size, output_shape, num_classes).

        n_components (int): Number of principal components for V.

        weights_initializer (str): Kernel initializer to use for initializing model weights.

        hidden_layer_sizes (List[int]): Output units for each hidden layer. List should be of same length as the number of hidden layers.

        hidden_activation (str): Activation function to use for hidden layers.

        l1_penalty (float): L1 regularization penalty to use to reduce overfitting.

        l2_penalty (float): L2 regularization penalty to use to reduce overfitting.

        dropout_rate (float): Dropout rate during training to reduce overfitting. Must be a float between 0 and 1.

        phase (int, optional): Current phase if doing UBP model. Defaults to 3.

        num_classes (int, optional): Number of classes in output. Corresponds to the 3rd dimension of the output shape (batch_size, n_features, num_classes). Defaults to 3.

    Methods:
        call: Does forward pass for model.
        train_step: Does training for one batch in a single epoch.
        test_step: Does evalutation for one batch in a single epoch.

    Attributes:
        phase (int): Current phase if doing UBP model. Ignored if doing NLPCA model.

        hidden_layer_sizes (List[Union[int, str]]): Output units for each hidden layer. Length should be the same as the number of hidden layers.

        n_components (int): Number of principal components to use with _V.

        _batch_size (int): Batch size to use per epoch.

        _batch_idx (int): Index of current batch.

        _n_batches (int): Total number of batches per epoch.

    Example:
        >>>model = UBPPhase2(output_shape, n_components, weights_initializer, hidden_layer_sizes, hidden_activation, l1_penalty, l2_penalty, dropout_rate, phase=3, num_classes=3)
        >>>model.compile(optimizer=optimizer, loss=loss_func, metrics=[my_metrics], run_eagerly=True)
        >>>history = model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[MyCallback()], validation_split=validation_split, shuffle=False)
    """

    def __init__(
        self,
        output_shape,
        n_components,
        weights_initializer,
        hidden_layer_sizes,
        hidden_activation,
        l1_penalty,
        l2_penalty,
        dropout_rate,
        phase=3,
        num_classes=3,
    ):
        super(UBPPhase2, self).__init__()

        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_components = n_components
        self.weights_initializer = weights_initializer
        self.phase = phase

        # Initialize parameters that are set via callbacks during fit().
        self._batch_size = 0
        self._batch_idx = 0
        self._n_batches = 0

        kernel_regularizer = l1_l2(l1_penalty, l2_penalty)
        self.kernel_regularizer = kernel_regularizer
        kernel_initializer = weights_initializer

        if len(hidden_layer_sizes) > 5:
            raise ValueError("The maximum number of hidden layers is 5.")

        self.dense2 = None
        self.dense3 = None
        self.dense4 = None
        self.dense5 = None

        # Construct multi-layer perceptron.
        # Add hidden layers dynamically.
        self.dense1 = Dense(
            hidden_layer_sizes[0],
            input_shape=(n_components,),
            activation=hidden_activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )

        if len(hidden_layer_sizes) >= 2:
            self.dense2 = Dense(
                hidden_layer_sizes[1],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if len(hidden_layer_sizes) >= 3:
            self.dense3 = Dense(
                hidden_layer_sizes[2],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if len(hidden_layer_sizes) >= 4:
            self.dense4 = Dense(
                hidden_layer_sizes[3],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if len(hidden_layer_sizes) == 5:
            self.dense5 = Dense(
                hidden_layer_sizes[4],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        self.dense6 = Dense(
            output_shape,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )

        # Expand dims to 3d with shape (batch_size, n_features, num_classes)
        self.lmda = Lambda(lambda x: tf.expand_dims(x, -1))
        self.output1 = Dense(num_classes, activation="softmax",)
        self.dropout_layer = Dropout(rate=dropout_rate)

    def call(self, inputs, training=None):
        """Forward propagates inputs through the model defined in __init__().

        Args:
            inputs (tf.keras.Input): Input tensor to forward propagate through the model.

            training (bool or None): Whether in training mode or not. Affects whether dropout is used.

        Returns:
            tf.keras.Model: Output tensor from forward propagation.
        """
        x = self.dense1(inputs)
        if self.dense2 is not None:
            x = self.dense2(x)
            x = self.dropout_layer(x, training=training)
        if self.dense3 is not None:
            x = self.dense3(x)
            x = self.dropout_layer(x, training=training)
        if self.dense4 is not None:
            x = self.dense4(x)
            x = self.dropout_layer(x, training=training)
        if self.dense5 is not None:
            x = self.dense5(x)
            x = self.dropout_layer(x, training=training)
        x = self.dense6(x)
        x = self.dropout_layer(x, training=training)

        x = self.lmda(x)
        return self.output1(x)

    def model(self):
        """Here so that mymodel.model().summary() can be called for debugging.
        """
        x = tf.keras.Input(shape=(self.n_components,))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def train_step(self, data):
        """Custom training loop for one step (=batch) in a single epoch.

        GradientTape records the weights and watched
        variables (usually tf.Variable objects), which
        in this case are the weights, during the forward pass.
        This allows us to run gradient descent during
        backpropagation to refine the watched variables.

        This function will train on a batch of samples (rows), which can be adjusted with the ``batch_size`` parameter from the estimator.

        Args:
            data (Tuple[tf.EagerTensor, tf.EagerTensor]): Input tensorflow tensors of shape (batch_size, n_components) and (batch_size, n_features, num_classes).

        Returns:
            Dict[str, float]: History object that gets returned from fit(). Contains the loss and any metrics specified in compile().

        ToDo:
            Obtain batch_size without using run_eagerly option in compile(). This will allow the step to be run in graph mode, thereby speeding up computation.
        """
        v_train, y_true = data

        # Get current batch_size.
        # NOTE: run_eagerly must be set to True in the compile() method for this
        # to work. Otherwise it can't convert a Tensor object to a numpy array.
        # There is really no other way to get the batch_size in graph
        # mode as far as I know. eager execution is slower, so it would be nice
        # to find a way to obtain batch_size without converting to numpy.
        batch_size = y_true.numpy().shape[0]

        # self._batch_idx is set in the UBPCallbacks() callback
        # on_train_batch_begin() method.
        batch_start = self._batch_idx * batch_size
        batch_end = (self._batch_idx + 1) * batch_size

        # NOTE: Moved self.set_weights(self._phase2_weights[self._batch_idx])
        # to custom callback because it threw an error when it was here.

        v_batch = v_train.numpy()

        # v doesn't get refined in phase 2, so set trainable=False.
        v = tf.Variable(
            tf.zeros([batch_size, self.n_components]),
            trainable=False,
            dtype=tf.float32,
        )

        # Assign current batch to v.
        v.assign(v_batch)

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(v, training=True)
            loss = self.compiled_loss(
                tf.convert_to_tensor(y_true, dtype=tf.float32),
                y_pred,
                regularization_losses=self.losses,
            )

        # Phase == 2.
        src = self.trainable_variables

        # Refine the watched variables with
        # gradient descent backpropagation
        gradients = tape.gradient(loss, src)
        self.optimizer.apply_gradients(zip(gradients, src))

        self.compiled_metrics.update_state(
            tf.convert_to_tensor(y_true, dtype=tf.float32), y_pred
        )

        # history object that gets returned from fit().
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Validation step for one batch in a single epoch.

        Custom logic for the test step that gets sent back to on_train_batch_end() callback.

        Args:
            data (Tuple[tf.EagerTensor, tf.EagerTensor]): Batches of input data V and y.

        Returns:
            A dict containing values that will be passed to ``tf.keras.callbacks.CallbackList.on_train_batch_end``. Typically, the values of the Model's metrics are returned.
        """
        # Unpack the data. Don't need V here. Just X (y_true).
        v_test, y_true = data

        batch_size = y_true.numpy().shape[0]

        # self._batch_idx is set in the UBPCallbacks() callback
        # on_train_batch_begin() method.
        batch_start = self._batch_idx * batch_size
        batch_end = (self._batch_idx + 1) * batch_size

        v_batch = v_test.numpy()

        v = tf.Variable(
            tf.zeros([batch_size, self.n_components]),
            trainable=False,
            dtype=tf.float32,
        )

        # Assign current batch to v.
        v.assign(v_batch)

        # Compute predictions
        y_pred = self(v, training=False)

        # Updates the metrics tracking the loss
        self.compiled_loss(
            tf.convert_to_tensor(y_true, dtype=tf.float32),
            y_pred,
            regularization_losses=self.losses,
        )

        # Update the metrics.
        self.compiled_metrics.update_state(
            tf.convert_to_tensor(y_true, dtype=tf.float32), y_pred
        )

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    @property
    def batch_size(self):
        """Batch (=step) size per epoch."""
        return self._batch_size

    @property
    def batch_idx(self):
        """Current batch (=step) index."""
        return self._batch_idx

    @property
    def n_batches(self):
        """Total number of batches per epoch."""
        return self._n_batches

    @batch_size.setter
    def batch_size(self, value):
        """Set batch_size parameter."""
        self._batch_size = int(value)

    @batch_idx.setter
    def batch_idx(self, value):
        """Set current batch (=step) index."""
        self._batch_idx = int(value)

    @n_batches.setter
    def n_batches(self, value):
        """Set total number of batches (=steps) per epoch."""
        self._n_batches = int(value)


class UBPPhase3(tf.keras.Model, NeuralNetwork):
    """UBP Phase 3 model to train and use to predict imputations.

    UBPModel subclasses the tf.keras.Model and overrides the train_step() and test_step() functions, which do training and evaluation for one batch in a single epoch.

    Args:
        V (numpy.ndarray(float)): V is randomly initialized in UBPPhase3 and used as the input data to Phase 3. It gets refined during training.

        output_shape (int): Output units for n_features dimension. Output will be of shape (batch_size, output_shape, num_classes).

        n_components (int): Number of principal components for V.

        weights_initializer (str): Kernel initializer to use for initializing model weights.

        hidden_layer_sizes (List[int]): Output units for each hidden layer. List should be of same length as the number of hidden layers.

        hidden_activation (str): Activation function to use for hidden layers.

        l1_penalty (float): L1 regularization penalty to use to reduce overfitting.

        l2_penalty (float): L2 regularization penalty to use to reduce overfitting.

        dropout_rate (float): Dropout rate during training to reduce overfitting. Must be a float between 0 and 1.

        phase (int, optional): Current phase if doing UBP model. Defaults to 3.

        num_classes (int, optional): Number of classes in output. Corresponds to the 3rd dimension of the output shape (batch_size, n_features, num_classes). Defaults to 3.

    Methods:
        call: Does forward pass for model.
        train_step: Does training for one batch in a single epoch.
        test_step: Does evaluation for one batch in a single epoch.

    Attributes:
        _V (numpy.ndarray(float)): Randomly initialized input that gets refined during training.

        phase (int): Current phase if doing UBP model. Ignored if doing NLPCA model.

        hidden_layer_sizes (List[Union[int, str]]): Output units for each hidden layer. Length should be the same as the number of hidden layers.

        n_components (int): Number of principal components to use with _V.

        _batch_size (int): Batch size to use per epoch.

        _batch_idx (int): Index of current batch.

        _n_batches (int): Total number of batches per epoch.

    Example:
        >>>model = UBPPhase3(V, output_shape, n_components, weights_initializer, hidden_layer_sizes, hidden_activation, l1_penalty, l2_penalty, dropout_rate, phase=3, num_classes=3)
        >>>model.compile(optimizer=optimizer, loss=loss_func, metrics=[my_metrics], run_eagerly=True)
        >>>history = model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[MyCallback()], validation_split=validation_split, shuffle=False)
    """

    def __init__(
        self,
        V,
        output_shape,
        n_components,
        weights_initializer,
        hidden_layer_sizes,
        hidden_activation,
        l1_penalty,
        l2_penalty,
        dropout_rate,
        phase=3,
        num_classes=3,
    ):
        super(UBPPhase3, self).__init__()

        self._V = V
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_components = n_components
        self.weights_initializer = weights_initializer
        self.phase = phase

        # Initialize parameters that are set via callbacks during fit().
        self._batch_size = 0
        self._batch_idx = 0
        self._n_batches = 0
        self._V_latent = self._V.copy()

        kernel_regularizer = l1_l2(l1_penalty, l2_penalty)
        self.kernel_regularizer = kernel_regularizer
        kernel_initializer = weights_initializer

        if len(hidden_layer_sizes) > 5:
            raise ValueError("The maximum number of hidden layers is 5.")

        self.dense2 = None
        self.dense3 = None
        self.dense4 = None
        self.dense5 = None

        # Construct multi-layer perceptron.
        # Add hidden layers dynamically.
        self.dense1 = Dense(
            hidden_layer_sizes[0],
            input_shape=(n_components,),
            activation=hidden_activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )

        if len(hidden_layer_sizes) >= 2:
            self.dense2 = Dense(
                hidden_layer_sizes[1],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if len(hidden_layer_sizes) >= 3:
            self.dense3 = Dense(
                hidden_layer_sizes[2],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if len(hidden_layer_sizes) >= 4:
            self.dense4 = Dense(
                hidden_layer_sizes[3],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if len(hidden_layer_sizes) == 5:
            self.dense5 = Dense(
                hidden_layer_sizes[4],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        self.dense6 = Dense(
            output_shape,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )

        # Expand dims to 3d of shape (batch_size, n_features, num_classes).
        self.lmda = Lambda(lambda x: tf.expand_dims(x, -1))
        self.output1 = Dense(num_classes, activation="softmax",)

        self.dropout_layer = Dropout(rate=dropout_rate)

    def call(self, inputs, training=None):
        """Forward propagates inputs through the model defined in __init__().

        Model varies depending on which phase UBP is in.

        Args:
            inputs (tf.keras.Input): Input tensor to forward propagate through the model.
    
            training (bool or None): Whether in training mode or not. Affects whether dropout is used.

        Returns:
            tf.keras.Model: Output tensor from forward propagation.
        """
        x = self.dense1(inputs)
        if self.dense2 is not None:
            x = self.dense2(x)
            x = self.dropout_layer(x, training=training)
        if self.dense3 is not None:
            x = self.dense3(x)
            x = self.dropout_layer(x, training=training)
        if self.dense4 is not None:
            x = self.dense4(x)
            x = self.dropout_layer(x, training=training)
        if self.dense5 is not None:
            x = self.dense5(x)
            x = self.dropout_layer(x, training=training)
        x = self.dense6(x)
        x = self.dropout_layer(x, training=training)

        x = self.lmda(x)
        return self.output1(x)

    def model(self):
        """Here so that mymodel.model().summary() can be called for debugging.
        """
        x = tf.keras.Input(shape=(self.n_components,))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def train_step(self, data):
        """Custom training loop for one step (=batch) in a single epoch.

        GradientTape records the weights and watched
        variables (usually tf.Variable objects), which
        in this case are the weights and the input (y_true), during the forward pass.
        This allows us to run gradient descent during
        backpropagation to refine the watched variables.

        This function will train on a batch of samples (rows), which can be adjusted with the ``batch_size`` parameter from the estimator.

        Args:
            data (Tuple[tf.EagerTensor, tf.EagerTensor]): Input tensors of shape (batch_size, n_components) and (batch_size, n_features, num_classes).

        Returns:
            Dict[str, float]: History object that gets returned from fit(). Contains the loss and any metrics specified in compile().

        ToDo:
            Obtain batch_size without using run_eagerly option in compile(). This will allow the step to be run in graph mode, thereby speeding up computation.
        """
        y_true = data[1]

        # Get current batch_size.
        # NOTE: run_eagerly must be set to True in the compile() method for this
        # to work. Otherwise it can't convert a Tensor object to a numpy array.
        # There is really no other way to get the batch_size in graph
        # mode as far as I know. eager execution is slower, so it would be nice
        # to find a way to obtain batch_size without converting to numpy.
        batch_size = y_true.numpy().shape[0]

        # self._batch_idx is set in the UBPCallbacks() callback
        # on_train_batch_begin() method.
        batch_start = self._batch_idx * batch_size
        batch_end = (self._batch_idx + 1) * batch_size

        # override v_batch. This model refines the input to fit the output, so
        # v_batch has to be overridden.
        v_batch = self._V_latent[batch_start:batch_end, :].copy()

        # NOTE: Moved self.set_weights(self._phase2_weights[self._batch_idx])
        # to custom callback because it threw an error when it was here.

        # Initialize variable v as tensorflow variable.
        v = tf.Variable(
            tf.zeros([batch_size, self.n_components]),
            trainable=True,
            dtype=tf.float32,
        )

        # Assign current batch to v.
        v.assign(v_batch)

        src = [v]

        with tf.GradientTape() as tape:
            # Forward pass
            tape.watch(v)
            y_pred = self(v, training=True)

            loss = self.compiled_loss(
                tf.convert_to_tensor(y_true, dtype=tf.float32),
                y_pred,
                regularization_losses=self.losses,
            )

        src.extend(self.trainable_variables)

        # Refine the watched variables with
        # gradient descent backpropagation
        gradients = tape.gradient(loss, src)
        self.optimizer.apply_gradients(zip(gradients, src))

        self.compiled_metrics.update_state(
            tf.convert_to_tensor(y_true, dtype=tf.float32), y_pred
        )

        self._V_latent[batch_start:batch_end, :] = v.numpy().copy()

        # history object that gets returned from fit().
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Validation step for one batch in a single epoch.

        Custom logic for the test step that gets sent back to on_train_batch_end() callback.

        Args:
            data (Tuple[tf.EagerTensor, tf.EagerTensor]): Batches of input data V and y_true.

        Returns:
            A dict containing values that will be passed to ``tf.keras.callbacks.CallbackList.on_train_batch_end``. Typically, the values of the Model's metrics are returned.
        """
        # Unpack the data. Don't need V here. Just X (y_true).
        y_true = data[1]

        batch_size = y_true.numpy().shape[0]

        # self._batch_idx is set in the UBPCallbacks() callback
        # on_train_batch_begin() method.
        batch_start = self._batch_idx * batch_size
        batch_end = (self._batch_idx + 1) * batch_size

        # Get the input that's being refined.
        v_batch = self._V_latent[batch_start:batch_end, :].copy()

        v = tf.Variable(
            tf.zeros([batch_size, self.n_components]),
            trainable=False,
            dtype=tf.float32,
        )

        # Assign current batch to v.
        v.assign(v_batch)

        # Compute predictions
        y_pred = self(v, training=False)

        # Updates the metrics tracking the loss
        self.compiled_loss(
            tf.convert_to_tensor(y_true, dtype=tf.float32),
            y_pred,
            regularization_losses=self.losses,
        )

        # Update the metrics.
        self.compiled_metrics.update_state(
            tf.convert_to_tensor(y_true, dtype=tf.float32), y_pred
        )

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    @property
    def V(self):
        """Randomly initialized input variable that gets refined during training."""
        return self._V_latent

    @property
    def batch_size(self):
        """Batch (=step) size per epoch."""
        return self._batch_size

    @property
    def batch_idx(self):
        """Current batch (=step) index."""
        return self._batch_idx

    @property
    def n_batches(self):
        """Total number of batches per epoch."""
        return self._n_batches

    @V.setter
    def V(self, value):
        """Set randomly initialized input variable. Gets refined during training."""
        self._V = value

    @batch_size.setter
    def batch_size(self, value):
        """Set batch_size parameter."""
        self._batch_size = int(value)

    @batch_idx.setter
    def batch_idx(self, value):
        """Set current batch (=step) index."""
        self._batch_idx = int(value)

    @n_batches.setter
    def n_batches(self, value):
        """Set total number of batches (=steps) per epoch."""
        self._n_batches = int(value)


class VAE(NeuralNetwork):
    """Class to impute missing data using a Variational Autoencoder neural network.

    Args:
        genotype_data (GenotypeData object or None): Input data initialized as GenotypeData object. If value is None, then uses ``gt`` to get the genotypes. Either ``genotype_data`` or ``gt`` must be defined. Defaults to None.

        gt (numpy.ndarray or None): Input genotypes directly as a numpy array. If this value is None, ``genotype_data`` must be supplied instead. Defaults to None.

        prefix (str): Prefix for output files. Defaults to "output".

        cv (int): Number of cross-validation replicates to perform. Only used if ``validation_size`` is not None. Defaults to 5.

        initial_strategy (str): Initial strategy to impute missing data with for validation. Possible options include: "populations", "most_frequent", and "phylogeny", where "populations" imputes by the mode of each population at each site, "most_frequent" imputes by the overall mode of each site, and "phylogeny" uses an input phylogeny to inform the imputation. "populations" requires a population map file as input in the GenotypeData object, and "phylogeny" requires an input phylogeny and Rate Matrix Q (also instantiated in the GenotypeData object). Defaults to "populations".

        validation_size (float or None): Proportion of sites to use for the validation. If ``validation_size`` is None, then does not perform validation. Defaults to 0.2.

        disable_progressbar (bool): Whether to disable the tqdm progress bar. Useful if you are doing the imputation on e.g. a high-performance computing cluster, where sometimes tqdm does not work correctly. If False, uses tqdm progress bar. If True, does not use tqdm. Defaults to False.

        train_epochs (int): Number of epochs to train the VAE model with. Defaults to 100.

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
        validation_size=0.2,
        disable_progressbar=False,
        train_epochs=100,
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

        self.train_epochs = train_epochs
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
        self.validation_size = validation_size
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

        imputed_enc = self.fit(
            train_epochs=self.train_epochs, batch_size=self.batch_size
        )

        imputed_enc, dummy_df = self.predict(X, imputed_enc)

        imputed_df = self._decode_onehot(
            pd.DataFrame(data=imputed_enc, columns=dummy_df.columns)
        )

        return imputed_df.to_numpy()

    def fit(self, batch_size=256, train_epochs=100):
        """Train a variational autoencoder model to impute missing data.

        Args:
            batch_size (int, optional): Number of data splits to train on per epoch. Defaults to 256.

            train_epochs (int, optional): Number of epochs (cycles through the data) to use. Defaults to 100.

        Returns:
            numpy.ndarray(float): Predicted values as numpy array.
        """

        missing_mask = self._create_missing_mask()
        self.fill(missing_mask, -1, self.num_classes)
        self.model = self._create_model()

        observed_mask = ~missing_mask

        for epoch in range(1, train_epochs + 1):
            X_pred = self._train_epoch(self.model, missing_mask, batch_size)
            observed_mse = self.masked_mse(
                X_true=self.data, X_pred=X_pred, mask=observed_mask
            )

            if epoch == 1:
                print(f"Initial MSE: {observed_mse}")

            elif epoch % 50 == 0:
                print(
                    f"Observed MSE ({epoch}/{train_epochs} epochs): "
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


class UBP(NeuralNetwork):
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

        validation_size (float or None, optional): Proportion of samples (=rows) between 0 and 1 to use for the neural network training validation. Defaults to 0.3.

        n_components (int, optional): Number of components to use as the input data. Defaults to 3.

        early_stop_gen (int, optional): Early stopping criterion for epochs. Training will stop if the loss (error) does not decrease past the tolerance level ``tol`` for ``early_stop_gen`` epochs. Will save the optimal model and reload it once ``early_stop_gen`` has been reached. Defaults to 25.

        num_hidden_layers (int, optional): Number of hidden layers to use in the model. Adjust if overfitting occurs. Defaults to 3.

        hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of neurons to use in hidden layers. If string or a list of strings is supplied, the strings must be either "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units. If "log2" is supplied, the units will be calculated as ``log2(n_features)``. hidden_layer_sizes will calculate and set the number of output units for each hidden layer. If one string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list, which can differ. The list length must be equal to the ``num_hidden_layers``. Defaults to "midpoint".

        optimizer (str, optional): The optimizer to use with gradient descent. Possible value include: "adam", "sgd", and "adagrad" are supported. See tf.keras.optimizers for more info. Defaults to "adam".

        hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Commonly used activation functions include "elu", "relu", and "sigmoid". Defaults to "elu".

        learning_rate (float, optional): The learning rate for the optimizers. Adjust if the loss is learning too slowly. Defaults to 0.1.

        train_epochs (int, optional): Maximum number of epochs to run if the ``early_stop_gen`` criterion is not met.

        tol (float, optional): Tolerance level to use for the early stopping criterion. If the loss does not improve past the tolerance level after ``early_stop_gen`` epochs, then training will halt. Defaults to 1e-3.

        weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float, optional): L1 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

        l2_penalty (float, optional): L2 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

        dropout_rate (float, optional): Dropout rate during training to reduce overfitting. Must be a float between 0 and 1. Defaults to 0.2.

        verbose (int, optional): Verbosity setting ranging from 0 to 3 (least to most verbose). Defaults to 0.

        cv (int, optional): Number of cross-validation replicates to perform. Defaults to 5.

        grid_iter (int, optional): Number of iterations for grid search. Defaults to 50.

        ga (bool, optional): Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead. Defaults to False.

        population_size (int, optional): For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation. Defaults to 10.

        tournament_size (int, optional): For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation. Defaults to 0.1.

        algorithm (str, optional): For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation. Defaults to "eaMuPlusLambda".

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

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
        validation_size=0.3,
        n_components=3,
        early_stop_gen=25,
        num_hidden_layers=3,
        hidden_layer_sizes="midpoint",
        optimizer="adam",
        hidden_activation="elu",
        learning_rate=0.1,
        train_epochs=100,
        tol=1e-3,
        weights_initializer="glorot_normal",
        l1_penalty=0.01,
        l2_penalty=0.01,
        dropout_rate=0.2,
        cv=5,
        grid_iter=50,
        ga=False,
        population_size=10,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.1,
        algorithm="eaMuPlusLambda",
        scoring_metric="accuracy",
        n_jobs=1,
        verbose=0,
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
        self.validation_size = validation_size
        self.n_components = n_components
        self.early_stop_gen = early_stop_gen
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.hidden_activation = hidden_activation
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.tol = tol
        self.weights_initializer = weights_initializer
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate
        self.verbose = verbose

        # TODO: Make estimators compatible with variable number of classes.
        # E.g., with morphologial data.
        self.num_classes = 3

        # Grid search settings.
        self.ga = ga
        self.scoring_metric = scoring_metric
        self.grid_iter = grid_iter
        self.n_jobs = n_jobs
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.algorithm = algorithm

    @timer
    def fit_predict(self, y):
        """Train a UBP or NLPCA model and predict the output.

        Uses input data from GenotypeData object.

        Returns:
            pandas.DataFrame: Imputation predictions.
        """
        (
            self.hidden_layer_sizes,
            self.num_hidden_layers,
        ) = self._validate_hidden_layers(
            self.hidden_layer_sizes, self.num_hidden_layers
        )

        # In NLPCA and UBP, X and y are flipped.
        # X is the randomly initialized model input (V)
        # y is the actual input data.
        y = self.validate_input(y)

        # Initialize small random values.
        V = self._init_weights(y.shape[0], self.n_components)

        hl_sizes = self._get_hidden_layer_sizes(
            y.shape[1], self.n_components, self.hidden_layer_sizes
        )

        y_train = self._encode_onehot(y)

        missing_mask = self._create_missing_mask(y_train)
        observed_mask = ~missing_mask
        self.fill(y_train, missing_mask, -1, self.num_classes)

        # Define neural network models.
        if self.nlpca:
            # If using NLPCA model: Don't need phases 1 and 2.
            logfile = f"{self.prefix}_nlpca_log.csv"
        else:
            logfile = f"{self.prefix}_ubp_log.csv"

        histories = list()

        # Reset model states
        K.clear_session()
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        self.reset_seeds()

        lr_patience = 5

        callbacks = [
            UBPCallbacks(),
            CSVLogger(filename=logfile),
            ReduceLROnPlateau(patience=lr_patience),
        ]

        if self.nlpca:
            model1 = None
            model2 = None
            model3 = NLPCAModel(
                V,
                y_train.shape[1],
                self.n_components,
                self.weights_initializer,
                hl_sizes,
                self.hidden_activation,
                self.l1_penalty,
                self.l2_penalty,
                self.dropout_rate,
                num_classes=3,
            )

            model3.compile(
                optimizer=self.set_optimizer(),
                loss=self.categorical_crossentropy_masked,
                metrics=[self.categorical_accuracy_masked],
                run_eagerly=True,
            )

            callbacks3 = [
                UBPEarlyStopping(patience=self.early_stop_gen, phase=None,),
            ]

            callbacks3.extend(callbacks)

            V2 = V.copy()

        else:
            model1 = UBPPhase1(
                V,
                y_train.shape[1],
                self.n_components,
                self.weights_initializer,
                hl_sizes,
                self.hidden_activation,
                self.l1_penalty,
                self.l2_penalty,
                num_classes=3,
                phase=1,
            )

            model1.compile(
                optimizer=self.set_optimizer(),
                loss=self.categorical_crossentropy_masked,
                metrics=[self.categorical_accuracy_masked],
                run_eagerly=True,
            )

            callbacks1 = [
                UBPEarlyStopping(patience=self.early_stop_gen, phase=1,),
            ]

            callbacks1.extend(callbacks)

            history1 = model1.fit(
                x=V,
                y=y_train,
                batch_size=self.batch_size,
                epochs=self.train_epochs,
                callbacks=callbacks1,
                validation_split=self.validation_size,
                shuffle=False,
                verbose=self.verbose,
            )

            # model1.model().summary()

            histories.append(history1.history)

            V2 = model1.V.copy()

            # # Reset model states
            # K.clear_session()
            # tf.keras.backend.clear_session()
            # tf.compat.v1.reset_default_graph()
            # self.reset_seeds()

            model2 = UBPPhase2(
                y_train.shape[1],
                self.n_components,
                self.weights_initializer,
                hl_sizes,
                self.hidden_activation,
                self.l1_penalty,
                self.l2_penalty,
                self.dropout_rate,
                num_classes=3,
                phase=2,
            )

            model2.compile(
                optimizer=self.set_optimizer(),
                loss=self.categorical_crossentropy_masked,
                metrics=[self.categorical_accuracy_masked],
                run_eagerly=True,
            )

            callbacks2 = [
                UBPEarlyStopping(patience=self.early_stop_gen, phase=2,),
            ]

            callbacks2.extend(callbacks)

            history2 = model2.fit(
                x=V2,
                y=y_train,
                batch_size=self.batch_size,
                epochs=self.train_epochs,
                callbacks=callbacks2,
                validation_split=self.validation_size,
                shuffle=False,
                verbose=self.verbose,
            )

            # model2.model().summary()

            # # w = model2.phase2_weights
            # w = model2.get_weights()

            # print(w)

            histories.append(history2.history)

            # # Reset model states
            # K.clear_session()
            # tf.keras.backend.clear_session()
            # tf.compat.v1.reset_default_graph()
            # self.reset_seeds()

            model3 = UBPPhase3(
                V2,
                y_train.shape[1],
                self.n_components,
                self.weights_initializer,
                hl_sizes,
                self.hidden_activation,
                self.l1_penalty,
                self.l2_penalty,
                self.dropout_rate,
                num_classes=3,
                phase=3,
            )

            model3.build((None, self.n_components))

            model3.compile(
                optimizer=self.set_optimizer(),
                loss=self.categorical_crossentropy_masked,
                metrics=[self.categorical_accuracy_masked],
                run_eagerly=True,
            )

            model3.set_weights(model2.get_weights())

            callbacks3 = [
                UBPEarlyStopping(patience=self.early_stop_gen, phase=3,),
            ]

            callbacks3.extend(callbacks)

        history3 = model3.fit(
            x=V2,
            y=y_train,
            batch_size=self.batch_size,
            epochs=self.train_epochs,
            callbacks=callbacks3,
            validation_split=self.validation_size,
            shuffle=False,
            verbose=self.verbose,
        )

        # Reset model states
        K.clear_session()
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        self.reset_seeds()

        # For debugging.
        # model.model().summary()

        histories.append(history3.history)

        y_pred = self.predict(V, y, model3, observed_mask)
        self._plot_history(histories)

        sys.exit()

    def predict(self, V, X, model, observed_mask):
        """Predict imputations based on input X.

        Args:
            V (numpy.ndarray(float)): Refined reduced-dimensional random input of shape (n_samples, n_components).

            X (numpy.ndarray(float), pandas.DataFrame, or List[List[int]]): Original data to impute.

            model (tf.keras.Model): Trained Keras model to predict with.

            observed_mask (numpy.ndarray(bool)): Mask with True as non-missing values and False as missing.

        Returns:
            numpy.ndarray: Imputation predictions.
        """
        if X is None:
            if self.genotype_data is None:
                raise TypeError(
                    "genotype_data argument must be provided if X is "
                    "not provided in fit()"
                )
            else:
                X = self.genotype_data.genotypes012_array

        X = self.validate_input(X)
        return self._predict_imputed(X, V, model, observed_mask)

    def _predict_imputed(self, X, V, model, observed_mask):
        """Predict imputations based on a trained UBP or NLPCA model.

        Args:
            X (numpy.ndarray(float)): Original data to impute.

            V (numpy.ndarray(float)): Refined reduced-dimensional input for predicting imputations.

            model (tf.keras.Sequential): Trained model (phase 3 if doing UBP).

            observed_mask (numpy.ndarray(bool)): Mask with True as non-missing values and False as missing.

        Returns:
            numpy.ndarray: Imputation predictions.

        Raises:
            TypeError: V cannot be NoneType.
            TypeError: model cannot be NoneType.
        """
        predictions = model(V, training=False)
        Xprob = predictions.numpy()
        Xt = np.apply_along_axis(self.mle, axis=2, arr=Xprob)
        Xpred = np.argmax(Xt, axis=2)
        Xdecoded = np.zeros((Xpred.shape[0], Xpred.shape[1]))
        for idx, row in enumerate(Xdecoded):
            imputed_vals = np.zeros(len(row))
            known_vals = np.zeros(len(row))
            imputed_idx = np.where(observed_mask[idx] == 0)
            known_idx = np.nonzero(observed_mask[idx])
            Xdecoded[idx, imputed_idx] = Xpred[idx, imputed_idx]
            Xdecoded[idx, known_idx] = X[idx, known_idx]
        return Xdecoded

    def _plot_history(self, lod):
        """Plot model history traces.

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
            ax1.plot(history["categorical_accuracy_masked"])
            ax1.plot(history["val_categorical_accuracy_masked"])
            ax1.set_title("Model Accuracy")
            ax1.set_ylabel("Accuracy")
            ax1.set_xlabel("Epoch")
            ax1.set_ylim(bottom=0.0, top=1.0)
            ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax1.legend(["Train", "Validation"], loc="best")

            # Plot model loss
            ax2.plot(history["loss"])
            ax2.plot(history["val_loss"])
            ax2.set_title("Model Loss")
            ax2.set_ylabel("Loss (MSE)")
            ax2.set_xlabel("Epoch")
            ax2.legend(["Train", "Validation"], loc="best")
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
                plt.plot(history["categorical_accuracy_masked"])
                plt.plot(history["val_categorical_accuracy_masked"])
                plt.title(f"{title} Accuracy")
                plt.ylabel("Accuracy")
                plt.xlabel("Epoch")
                plt.ylim((0.0, 1.0))
                plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                plt.legend(["Train", "Validation"], loc="best")

                # Plot model loss
                plt.subplot(3, 2, idx + 1)
                plt.plot(history["loss"])
                plt.plot(history["val_loss"])
                plt.title(f"{title} Loss")
                plt.ylabel("Loss (MSE)")
                plt.xlabel("Epoch")
                plt.legend(["Train", "Validation"], loc="best")

                idx += 2

            plt.savefig(fn, bbox_inches="tight")

            plt.close()
            plt.clf()

    def train(self):
        """Train an unsupervised backpropagation (UBP) or NLPCA model.

        UBP runs over three phases.

        1. Train a single-layer perceptron to refine the input, V_latent.
        2. Train a multi-layer perceptron (MLP) to refine only the weights
        3. Train an MLP to refine both the weights and input, V_latent.

        NLPCA just does phase 3.

        Returns:
            numpy.ndarray(float): Predicted values as numpy array.
        """
        missing_mask = self._create_missing_mask()
        self.observed_mask = ~missing_mask
        self.fill(missing_mask, -1, self.num_classes)

        # Reset model states
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        self.reset_seeds()

        # Define neural network models.
        if self.nlpca:
            # If using NLPCA model: Don't need phases 1 and 2.
            model_single_layer = None
            model_mlp_phase2 = None
            start_phase = 3

        else:
            # Using UBP model over three phases
            phase1model = self._build_ubp(phase=1)
            phase2model = self._build_ubp(phase=2)

            # Initialize new model with new random weights.
            model_single_layer = tf.keras.models.clone_model(phase1model)
            model_mlp_phase2 = tf.keras.models.clone_model(phase2model)

            del phase1model
            del phase2model

            start_phase = 1

        phase3model = self._build_ubp(phase=3)
        model_mlp_phase3 = tf.keras.models.clone_model(phase3model)
        del phase3model

        # Number of batches based on rows in X and V_latent.
        n_batches = int(np.ceil(self.data.shape[0] / self.batch_size))

        if self.nlpca:
            model_dir3 = "optimal_nlpca"
            self._remove_dir(model_dir3)
        else:
            model_dir1 = f"{self.prefix}_optimal_ubp_phase1"
            model_dir2 = f"{self.prefix}_optimal_ubp_phase2"
            model_dir3 = f"{self.prefix}_optimal_ubp_phase3"
            self._remove_dir(model_dir1)
            self._remove_dir(model_dir2)
            self._remove_dir(model_dir3)

        # self._initialise_parameters()

        for phase in range(start_phase, 4):

            s_delta = None
            s_prime = np.inf
            acc_prime = np.inf
            final_s = 0
            final_acc = 0
            num_epochs = 0
            counter = 0
            checkpoint = 1
            criterion_met = False

            # While stopping criterion not met: keep doing more epochs.
            while (
                counter < self.early_stop_gen
                and num_epochs <= self.train_epochs
            ):
                # Train per epoch
                # s is error (loss)
                s, train_acc = self._train_epoch(
                    model_single_layer,
                    model_mlp_phase2,
                    model_mlp_phase3,
                    n_batches,
                    phase=phase,
                )

                num_epochs += 1

                if num_epochs % 50 == 0:
                    print(f"Epoch {num_epochs}...")
                    print(
                        f"Observed MSE: {s} (Accuracy: {round(train_acc * 100, 2)}%)"
                    )

                if num_epochs == 1:
                    s_prime = s
                    acc_prime = train_acc

                    if self.nlpca:
                        print("\nBeginning NLPCA training...\n")
                    else:
                        print(f"\nBeginning UBP Phase {phase} training...\n")
                    print(
                        f"Initial MSE: {s} (Accuracy: {round(train_acc * 100, 2)}%)"
                    )

                if not criterion_met and num_epochs > 1:
                    if s < s_prime:
                        s_delta = abs(s_prime - s)
                        if s_delta <= self.tol:
                            criterion_met = True
                            if phase == 1:
                                tf.keras.models.save_model(
                                    model_single_layer,
                                    model_dir1,
                                    include_optimizer=False,
                                    overwrite=True,
                                )
                            elif phase == 2:
                                tf.keras.models.save_model(
                                    model_mlp_phase2,
                                    model_dir2,
                                    include_optimizer=False,
                                    overwrite=True,
                                )
                            elif phase == 3:
                                tf.keras.models.save_model(
                                    model_mlp_phase3,
                                    model_dir3,
                                    include_optimizer=False,
                                    overwrite=True,
                                )
                            checkpoint = num_epochs
                        else:
                            counter = 0
                            s_prime = s
                            acc_prime = train_acc

                    else:
                        if phase == 1 and not os.path.isdir(model_dir1):
                            tf.keras.models.save_model(
                                model_single_layer,
                                model_dir1,
                                include_optimizer=False,
                                overwrite=True,
                            )
                        elif phase == 2 and not os.path.isdir(model_dir2):
                            tf.keras.models.save_model(
                                model_mlp_phase2,
                                model_dir2,
                                include_optimizer=False,
                                overwrite=True,
                            )
                        elif phase == 3 and not os.path.isdir(model_dir3):
                            tf.keras.models.save_model(
                                model_mlp_phase3,
                                model_dir3,
                                include_optimizer=False,
                                overwrite=True,
                            )
                        criterion_met = True
                        checkpoint = num_epochs

                elif criterion_met and num_epochs > 1:
                    if s < s_prime:
                        s_delta = abs(s_prime - s)
                        if s_delta > self.tol:
                            criterion_met = False
                            s_prime = s
                            acc_prime = train_acc
                            counter = 0
                        else:
                            counter += 1
                            if counter == self.early_stop_gen:
                                counter = 0
                                if phase == 1:
                                    model_single_layer = tf.keras.models.load_model(
                                        model_dir1, compile=False
                                    )
                                elif phase == 2:
                                    model_mlp_phase2 = tf.keras.models.load_model(
                                        model_dir2, compile=False
                                    )
                                elif phase == 3:
                                    model_mlp_phase3 = tf.keras.models.load_model(
                                        model_dir3, compile=False
                                    )
                                final_s = s_prime
                                final_acc = acc_prime
                                break
                    else:
                        counter += 1
                        if counter == self.early_stop_gen:
                            counter = 0
                            if phase == 1:
                                model_single_layer = tf.keras.models.load_model(
                                    model_dir1, compile=False
                                )
                            elif phase == 2:
                                model_mlp_phase2 = tf.keras.models.load_model(
                                    model_dir2, compile=False
                                )
                            elif phase == 3:
                                model_mlp_phase3 = tf.keras.models.load_model(
                                    model_dir3, compile=False
                                )
                            final_s = s_prime
                            final_acc = acc_prime
                            break

            print(f"Number of epochs used to train: {checkpoint}")
            print(
                f"Final MSE: {final_s} (Accuracy: {round(final_acc * 100, 2)}%)"
            )

        if not self.nlpca:
            self._remove_dir(model_dir1)
            self._remove_dir(model_dir2)
        self._remove_dir(model_dir3)

        del model_single_layer
        del model_mlp_phase2

        return model_mlp_phase3

    def _train_epoch(
        self,
        model_single_layer,
        model_mlp_phase2,
        model_mlp_phase3,
        n_batches,
        phase=3,
        num_classes=3,
    ):
        """Train UBP or NLPCA over one epoch.

        Args:
            model (tf.keras.models.Sequential): Keras model to train.

            n_batches (int): Number of batches to train.

            phase (int): Current phase. UBP has three phases. If doing NLPCA, it only does phase 3.

            num_classes (int): Number of categorical classes to predict (three for 012 encoding). Defaults to 3.
        """
        if phase > 3:
            raise ValueError(
                f"There can only be maximum 3 phases, but phase={phase}"
            )

        # Randomize the order the of the samples in the batches.
        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)

        losses = list()
        train_acc = list()
        val_loss = list()
        val_acc = list()
        for batch_idx in range(n_batches):
            if phase == 3 and not self.nlpca:
                # Set the refined weights from model 2.
                model_mlp_phase3.set_weights(self.phase2_model[batch_idx])

            batch_start = batch_idx * self.batch_size
            batch_end = (batch_idx + 1) * self.batch_size
            x_batch = self.data[batch_start:batch_end, :]
            v_batch = self.V_latent[batch_start:batch_end, :]

            # Initialize variable v as tensorflow variable.
            if phase != 2:
                v = tf.Variable(
                    tf.zeros([x_batch.shape[0], self.n_components]),
                    trainable=True,
                    dtype=tf.float32,
                )

            elif phase == 2:
                v = tf.Variable(
                    tf.zeros([x_batch.shape[0], self.n_components]),
                    trainable=False,
                    dtype=tf.float32,
                )

            # Assign current batch to v.
            v.assign(v_batch)

            if phase == 1:
                loss, refined, train_accuracy = self._train_on_batch(
                    v, x_batch, model_single_layer, phase
                )
            elif phase == 2:
                loss, refined, train_accuracy = self._train_on_batch(
                    v, x_batch, model_mlp_phase2, phase
                )
            elif phase == 3:
                loss, refined, train_accuracy = self._train_on_batch(
                    v, x_batch, model_mlp_phase3, phase
                )

            losses.append(loss.numpy())
            train_acc.append(train_accuracy)

            if phase != 2:
                self.V_latent[batch_start:batch_end, :] = refined.numpy()
            else:
                self.phase2_model.append(refined)

        return np.mean(losses), np.mean(train_acc)

    def _train_on_batch(self, x, y, model, phase):
        """Custom training loop for neural network.

        GradientTape records the weights and watched
        variables (usually tf.Variable objects), which
        in this case are the weights and the input (x)
        (if not phase 2), during the forward pass.
        This allows us to run gradient descent during
        backpropagation to refine the watched variables.

        This function will train on a batch of samples (rows).

        Args:
            x (tf.Variable): Input tensorflow variable of shape (batch_size, n_features).

            y (tf.Variable): Target variable to calculate loss.

            model (tf.keras.models.Sequential): Keras model to use.

            phase (int): UBP phase to run.

        Returns:
            tf.Tensor: Calculated loss of current batch.

            tf.Variable, conditional: Input refined by gradient descent. Only returned if phase != 2.

            List[np.ndarray], conditional: Refined weights from keras model, as returned with the tf.keras.models.Sequential.get_weights() function. Only returned if phase == 2.

            float: Training CategoricalAccuracy.

        """
        if phase != 2:
            src = [x]

        with tf.GradientTape() as tape:
            # Forward pass
            if phase != 2:
                tape.watch(x)
            pred = model(x, training=True)
            loss = self.categorical_crossentropy_masked(
                tf.convert_to_tensor(y, dtype=tf.float32), pred
            )

        if phase != 2:
            # Phase == 1 or 3.
            src.extend(model.trainable_variables)
        elif phase == 2:
            # Phase == 2.
            src = model.trainable_variables

        # Refine the watched variables with
        # gradient descent backpropagation
        gradients = tape.gradient(loss, src)
        self.opt.apply_gradients(zip(gradients, src))

        train_accuracy = self.categorical_accuracy_masked(
            tf.convert_to_tensor(y, dtype=tf.float32), pred
        )

        if phase != 2:
            return loss, x, train_accuracy
        elif phase == 2:
            return loss, model.get_weights(), train_accuracy

    def _remove_dir(self, dir_path):
        """Remove directory from disk.

        Will pass if directory doesn't exist.

        Args:
            dir_path (str): The directory to remove.
        """
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            pass

    def _build_ubp(self, phase=3, num_classes=3):
        """Create and train a UBP neural network model.

        If we are implementing a single layer perceptron, we want exactly one dense layer. So in that case, X_hat = v * w, where w is the weights of that one dense layer. If we are implementing NLPCA or UBP, then we should add more layers and x = f(v, w) in a multi-layer perceptron (MLP).

        Creates a network with the following structure:

        If phase > 1:
            InputLayer (V) -> DenseLayer1 -> ActivationFunction1 ... HiddenLayerN -> ActivationFunctionN ... DenseLayerN+1 -> Lambda (to expand shape) -> DenseOutputLayer -> Softmax

        If Phase == 1:
            InputLayer (V) -> DenseLayer1 -> Lambda (to expand shape) -> DenseOutputLayer -> Softmax

        Args:
            num_classes (int, optional): The number of classes in the vector. Defaults to 3.

        Returns:
            tf.keras.Sequential object: Keras model.
        """

        if phase == 1 or phase == 2:
            kernel_regularizer = l1_l2(self.l1_penalty, self.l2_penalty)
        elif phase == 3:
            kernel_regularizer = None
        else:
            raise ValueError(f"Phase must equal 1, 2, or 3, but got {phase}")

        if phase == 3:
            # Phase 3 uses weights from phase 2.
            kernel_initializer = None
        else:
            kernel_initializer = self.weights_initializer

        model = Sequential()

        model.add(tf.keras.Input(shape=(self.n_components,)))

        if phase > 1:
            # Construct multi-layer perceptron.
            # Add hidden layers dynamically.
            for layer_size in self.hidden_layer_sizes:
                model.add(
                    Dense(
                        layer_size,
                        activation=self.hidden_activation,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                    )
                )

            model.add(
                Dense(
                    self.data.shape[1],
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                )
            )

            model.add(Lambda(lambda x: tf.expand_dims(x, -1)))
            model.add(Dense(num_classes, activation="softmax"))

        else:
            # phase == 1.
            # Construct single-layer perceptron.
            model.add(
                Dense(
                    self.data.shape[1],
                    input_shape=(self.n_components,),
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                )
            )
            model.add(Lambda(lambda x: tf.expand_dims(x, -1)))
            model.add(Dense(num_classes, activation="softmax"))

        return model

    def _validate_hidden_layers(self, hidden_layer_sizes, num_hidden_layers):
        """Validate hidden_layer_sizes and verify that it is in the correct format.

        Args:
            hidden_layer_sizes (str, int, List[str], or List[int]): Output units for all the hidden layers.

            num_hidden_layers (int): Number of hidden layers to use.

        Returns:
            List[int] or List[str]: List of hidden layer sizes.
            int: Number of hidden layers to use.
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

        return hidden_layer_sizes, num_hidden_layers

    def _init_weights(self, dim1, dim2, w_mean=0, w_stddev=0.01):
        """Initialize random weights to use with the model.

        Args:
            dim1 (int): Size of first dimension.

            dim2 (int): Size of second dimension.

            w_mean (float, optional): Mean of normal distribution. Defaults to 0.

            w_stddev (float, optional): Standard deviation of normal distribution. Defaults to 0.01.
        """
        # Get reduced-dimension dataset.
        return np.random.normal(loc=w_mean, scale=w_stddev, size=(dim1, dim2))

    def _encode_onehot(self, X):
        """Convert 012-encoded data to one-hot encodings.

        Args:
            X (numpy.ndarray): Input array with 012-encoded data and -9 as the missing data value.

        Returns:
            pandas.DataFrame: One-hot encoded data, ignoring missing values (np.nan).
        """
        Xt = np.zeros(shape=(X.shape[0], X.shape[1], 3))
        mappings = {
            0: np.array([1, 0, 0]),
            1: np.array([0, 1, 0]),
            2: np.array([0, 0, 1]),
            -9: np.array([np.nan, np.nan, np.nan]),
        }
        for row in np.arange(X.shape[0]):
            Xt[row] = [mappings[enc] for enc in X[row]]
        return Xt

    def _get_hidden_layer_sizes(self, n_dims, n_components, hl_func):
        """Get dimensions of hidden layers.

        Args:
            n_dims (int): The number of feature dimensions (columns) (d).

            n_components (int): The number of reduced dimensions (t).

            hl_func (str): The function to use to calculate the hidden layer sizes. Possible options: "midpoint", "sqrt", "log2".

        Returns:
            [int, int, int]: [Number of dimensions in hidden layers]
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

    def _create_missing_mask(self, data):
        """Creates a missing data mask with boolean values.

        Args:
            data (numpy.ndarray): Data to generate missing mask from, of shape (n_samples, n_features, n_classes).

        Returns:
            numpy.ndarray(bool): Boolean mask of missing values of shape (n_samples, n_features), with True corresponding to a missing data point.
        """
        return np.isnan(data).all(axis=2)

    def _initialise_parameters(self):
        """Initialize important parameters."""
        self.current_eta = self.initial_eta
        self.final_s = 0
        self.s_prime = np.inf
        self.num_epochs = 0
        self.checkpoint = 1
