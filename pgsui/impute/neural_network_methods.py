import logging
import math
import os
import sys
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.utils.class_weight import (
    compute_class_weight,
)

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


class DisabledCV:
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        yield (np.arange(len(X)), np.arange(len(y)))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


# For VAE. Necessary to initialize outside of class for use with tf.function decorator.
cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
cce_proba = tf.keras.losses.CategoricalCrossentropy()
cca = tf.keras.metrics.CategoricalAccuracy()
ba = tf.keras.metrics.BinaryAccuracy()
bce = tf.keras.losses.BinaryCrossentropy()


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

    @staticmethod
    def encode_onehot(X, missing_val=np.nan):
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
            -9: np.array([missing_val, missing_val, missing_val]),
        }
        for row in np.arange(X.shape[0]):
            Xt[row] = [mappings[enc] for enc in X[row]]
        return Xt

    @staticmethod
    def encode_vae(X):
        """Encode 0-9 integer data in one-hot format.
        Args:
            X (numpy.ndarray): Input array with 012-encoded data and -9 as the missing data value.
        Returns:
            pandas.DataFrame: One-hot encoded data, ignoring missing values (np.nan). multi-label categories will be encoded as 0.5. Otherwise, it will be 1.0.
        """
        # return np.where(X >= 0.5, 1.0, 0.0)
        try:
            Xt = np.zeros(shape=(X.shape[0], X.shape[1], 4))
        except IndexError:
            Xt = np.zeros(shape=(X.shape[0],))
        mappings = {
            0: [1.0, 0.0, 0.0, 0.0],
            1: [0.0, 1.0, 0.0, 0.0],
            2: [0.0, 0.0, 1.0, 0.0],
            3: [0.0, 0.0, 0.0, 1.0],
            4: [1.0, 1.0, 0.0, 0.0],
            5: [1.0, 0.0, 1.0, 0.0],
            6: [1.0, 0.0, 0.0, 1.0],
            7: [0.0, 1.0, 1.0, 0.0],
            8: [0.0, 1.0, 0.0, 1.0],
            9: [0.0, 0.0, 1.0, 1.0],
            -9: [np.nan, np.nan, np.nan, np.nan],
        }
        try:
            for row in np.arange(X.shape[0]):
                Xt[row] = [mappings[enc] for enc in X[row]]
        except TypeError:
            Xt = [mappings[enc] for enc in X]
        return Xt

    @classmethod
    def decode_masked(
        cls,
        y_pred_proba,
        is_multiclass=False,
        threshold=0.5,
        return_proba=False,
        return_multilab=False,
    ):
        """Evaluate model predictions by decoding from one-hot encoding to integer-encoded format.

        Gets the index of the highest predicted value to obtain the integer encodings or integer encodings.

        Calucalates highest predicted value for each row vector and each class, setting the most likely class to 1.0.

        Args:
            y (numpy.ndarray): Model predictions of shape (n_samples * n_features,). Array should be flattened and masked.

            is_multiclass (bool, optional): True if using multiclass data with softmax activation. False if using multilabel data with sigmoid activation. Defaults to False.

            threshold (float, optional): If using multilabel, then set the threshold for determining 1 or 0 predictions. Defaults to 0.5.

            return_proba (bool, optional): If True, returns probabilities for unresolved values where all multilabel probabilities were below the threshold. Defaults to False.

            return_multilab (bool, optional): If True, returns the multilabel encodings instead of integer encodings. Defaults to False.

        Returns:
            numpy.ndarray: Imputed integer-encoded values.

            numpy.ndarray (optional): Probabilities for each call, with those above the threshold set to 1.0 and those below the threshold between 0 and 1.
        """
        y_unresolved_certainty = None
        if is_multiclass:
            y_pred = cls.decode_multiclass(y_pred_proba)
        else:
            # Call 0s and 1s based on threshold.
            pred_multilab = np.where(y_pred_proba >= threshold, 1.0, 0.0)

            pred_multilab = cls.zero_extra_categories(
                y_pred_proba, pred_multilab, threshold=threshold
            )

            pred_multilab_decoded = cls.decode_binary_multilab(pred_multilab)

            # Check if there are still any missing values.
            still_missing = np.all(pred_multilab == 0, axis=-1)

            if return_multilab:
                still_missing_bin = np.all(
                    pred_multilab == 0, axis=-1, keepdims=True
                )

            # Do multiclass prediction with argmax then get the probabilities
            # if any unresolved values.
            if np.any(still_missing):
                # Get the argmax with the highest probability if
                # all classes are below threshold.
                y_multi = cls.decode_multiclass(y_pred_proba)
                y_multi_bin = cls.decode_multiclass(
                    y_pred_proba, reduce_dim=False
                )

                y_pred = np.where(
                    still_missing, y_multi, pred_multilab_decoded
                )

                if return_multilab:
                    y_pred_bin = np.where(
                        still_missing_bin, y_multi_bin, pred_multilab
                    )

                if return_proba:
                    # Get max value as base call.
                    y_pred_proba_max = y_pred_proba.max(axis=-1)

                    # Get probability of max value that was < threshold.
                    y_unresolved_certainty = np.where(
                        still_missing, y_pred_proba_max, 1.0
                    )

            else:
                y_pred = pred_multilab_decoded

        y_pred = y_pred.astype(np.int)

        if return_multilab:
            y_pred = y_pred_bin

        if return_proba:
            return y_pred, y_unresolved_certainty
        else:
            return y_pred

    @staticmethod
    def zero_extra_categories(y_pred_proba, pred_multilab, threshold=0.5):
        """Check if any prediction probabilities have >2 values above threshold.

        If >2, then it sets the two with the lowest probabilities to 0.0.

        Args:
            y_pred_proba (numpy.ndarray): Prediction probabilities (sigmoid activation) of shape (n_samples, n_features, num_classes) or (n_samples * n_features, num_classes).

            pred_multilab (numpy.ndarray): Multi-label decodings. Inner arrays should have only 0s and 1s. Should be of shape (n_samples, n_features, num_classes) or (n_samples * n_features, num_classes).

            threshold (float, optional): Threshold to use to set decoded multilabel values to 0s (< threshold) or 1s (>= threshold). Defaults to 0.5.
        """
        # Check if multi-labels have >2 1s.
        multi_cats = np.where(
            np.count_nonzero(pred_multilab, axis=-1) > 2, True, False
        )

        # If >2 1s, set the extras with lowest probabilities to 0.0.
        if np.any(multi_cats):
            n_dims = len(y_pred_proba.shape)
            if n_dims == 2:
                for row in np.arange(y_pred_proba.shape[0]):
                    # Get lowest two values.
                    idx = np.argpartition(y_pred_proba[row], 2)[0:2]
                    y_pred_proba[row, idx] = 0.0

            elif n_dims == 3:
                for row in np.arange(y_pred_proba.shape[0]):
                    for col in np.arange(y_pred_proba.shape[1]):
                        idx = np.argpartition(y_pred_proba[row, col], 2)[0:2]
                        y_pred_proba[row, col, idx] = 0.0

            else:
                raise IndexError(
                    f"Incorrect number of dimensions for y_pred_proba: {n_dims}"
                )

            pred_multilab = np.where(y_pred_proba >= threshold, 1.0, 0.0)
        return pred_multilab

    @classmethod
    def decode_multiclass(cls, y_pred_proba, reduce_dim=True):
        yt = np.apply_along_axis(cls.mle, axis=-1, arr=y_pred_proba)
        if reduce_dim:
            return np.argmax(yt, axis=-1)
        else:
            return yt

    @classmethod
    def decode_binary_multilab(cls, y_pred):
        """Decode multi-label sigmoid probabilities to integer encodings.

        The predictions should have already undergone sigmoid activation and should be probabilities.

        If sigmoid activation output is >0.5, gets encoded as 1.0; else 0.0. If more than one category is > 0.5, then it is a heterozygote.

        Args:
            y_pred (numpy.ndarray): Model predictions of shape (n_samples * n_features, num_classes) or (n_samples, n_features, num_classes). A threshold should already have been applied to set each class to 0 or 1.

        Returns:
            numpy.ndarray: Integer-decoded multilabel predictions of shape (n_samples * n_features) or (n_samples, n_features).
        """
        bin_mapping = [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
        ]

        # Get integer mappings.
        int_mapping = list(range(10))
        int_mapping.append(-9)
        int_mapping = np.array(int_mapping)

        # Convert each inner array to a string.
        # E.g., ['[1, 0, 0, 0]', '[1, 1, 0, 0]']
        # This is so they can be used as dictionary keys.
        bin_mapping = [np.array2string(x) for x in bin_mapping]

        # Zip two lists to dictionary with bin_mapping and int_mapping
        # as keys and values
        bin_mapping = dict(zip(bin_mapping, int_mapping))

        # Convert each inner array to a string. E.g., ['[0, 1, 0, 0]'].
        y_pred_str = np.apply_along_axis(np.array2string, -1, y_pred)

        # Convert string arrays to integers using dictionary key.
        return np.vectorize(bin_mapping.get)(y_pred_str)

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
            if not all(
                [isinstance(x, (str, int)) for x in hidden_layer_sizes]
            ):
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

    def get_hidden_layer_sizes(self, n_dims, n_components, hl_func, vae=False):
        """Get dimensions of hidden layers.

        Args:
            n_dims (int): The number of feature dimensions (columns) (d).

            n_components (int): The number of reduced dimensions (t).

            hl_func (str): The function to use to calculate the hidden layer sizes. Possible options: "midpoint", "sqrt", "log2".

            vae (bool, optional): Whether using the VAE algorithm. If False, then the returned list gets reversed for NLPCA and UBP.

        Returns:
            [int, int, int, ...]: [Number of dimensions in hidden layers].

        Raises:
            ValueError: Too many hidden layers specified. Repeated reduction of layer sizes dips below n_components.
        """
        layers = list()
        if not isinstance(hl_func, list):
            raise TypeError(
                f"hl_func must be of type list, but got {type(hl_func)}."
            )

        units = n_dims
        for func in hl_func:
            if func == "midpoint":
                units = round((units + n_components) / 2)
            elif func == "sqrt":
                units = round(math.sqrt(units))
                print(units)
                sys.exit()
            elif func == "log2":
                units = round(math.log(units, 2))
            elif isinstance(func, int):
                units = func
            else:
                raise ValueError(
                    f"hidden_layer_sizes must be either integers or any of "
                    f"the following strings: 'midpoint', "
                    f"'sqrt', or 'log2', but got {func} of type {type(func)}"
                )

            if units <= n_components:
                print(
                    f"WARNING: hidden_layer_size reduction became less than n_components. Using only {len(layers)} hidden layers."
                )
                break

            assert units > 0 and units < n_dims, (
                f"The hidden layer sizes must be > 0 and < the number of "
                f"features (i.e., columns) in the dataset, but size was {units}"
            )

            layers.append(units)

        assert (
            layers
        ), "There was an error setting hidden layer sizes. Size list is empty. It is possible that the first 'sqrt' reduction caused units to be <= n_components."

        if not vae:
            layers.reverse()

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
        y_train=None,
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

            y_train (numpy.ndarray, optional): One-hot encoded target values of shape (n_samples, n_features, n_classes). Defaults to None.

        Returns:
            tf.Variable: Input tensor v with current batch assigned.
            numpy.ndarray: Current batch of target data (actual input) used to refine v.
            List[float]: Sample weights
            int: Batch starting index.
            int: Batch ending index.
            numpy.ndarray: Batch of y_train target data of shape (batch_size, n_features, n_classes). Only returned for VAE.
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

        if ubp:

            # override batches. This model refines the input to fit the output, so
            # v_batch and y_true have to be overridden.
            y_true = y[batch_start:batch_end, :]

            v_batch = V[batch_start:batch_end, :]
            missing_mask_batch = missing_mask[batch_start:batch_end, :]

            if sample_weight is not None:
                sample_weight_batch = sample_weight[batch_start:batch_end, :]
            else:
                sample_weight_batch = None

            v = tf.Variable(
                tf.zeros([batch_size, n_components]),
                trainable=trainable,
                dtype=tf.float32,
            )

            # Assign current batch to tf.Variable v.
            v.assign(v_batch)

            return (
                v,
                y_true,
                sample_weight_batch,
                missing_mask_batch,
                batch_start,
                batch_end,
            )

        else:
            # Using VAE.
            y_true = y[batch_start:batch_end, :]
            v = V[batch_start:batch_end, :]
            y_train_batch = y_train[batch_start:batch_end, :]
            missing_mask_batch = missing_mask[batch_start:batch_end, :]

            if sample_weight is not None:
                sample_weight_batch = sample_weight[batch_start:batch_end, :]
            else:
                sample_weight_batch = None

            return (
                v,
                y_true,
                sample_weight_batch,
                missing_mask_batch,
                batch_start,
                batch_end,
                y_train_batch,
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

    def set_compile_params(self, optimizer, vae=False):
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

        if vae:
            loss = NeuralNetworkMethods.make_masked_binary_crossentropy()
            metrics = [NeuralNetworkMethods.make_masked_binary_accuracy()]

        else:
            # Doing grid search. Params are callables.
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            metrics = [tf.keras.metrics.CategoricalAccuracy()]

        return {
            "optimizer": opt,
            "loss": loss,
            "metrics": metrics,
            "run_eagerly": True,
        }

    @staticmethod
    def init_weights(dim1, dim2, w_mean=0, w_stddev=0.01):
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

    @staticmethod
    def make_masked_binary_accuracy():
        """Make binary accuracy metric with missing mask.

        Returns:
            callable: Function that calculates categorical crossentropy loss.
        """

        @tf.function
        def masked_binary_accuracy(y_true, y_pred, sample_weight=None):
            """Custom neural network metric function with missing mask.

            Ignores missing data in the calculation of the loss function.

            Args:
                y_true (tensorflow.Tensor): Input multilabel encoded 3D tensor.
                y_pred (tensorflow.Tensor): Predicted values from model.
                sample_weight (numpy.ndarray): 2D matrix of sample weights.

            Returns:
                float: Binary accuracy calculated with missing data masked.
            """
            y_true_masked = tf.boolean_mask(
                y_true, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            )

            y_pred_masked = tf.boolean_mask(
                y_pred, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            )

            if sample_weight is not None:
                sample_weight_masked = tf.boolean_mask(
                    tf.convert_to_tensor(sample_weight),
                    tf.reduce_any(tf.not_equal(y_true, -1), axis=2),
                )
            else:
                sample_weight_masked = None

            return ba(
                y_true_masked,
                y_pred_masked,
                sample_weight=sample_weight_masked,
            )

        return masked_binary_accuracy

    @staticmethod
    def make_masked_binary_crossentropy():
        """Make binary crossentropy loss function with missing mask.

        Args:
            num_classes (int, optional): Number of classes used. Defaults to 2.

        Returns:
            callable: Function that calculates categorical crossentropy loss.
        """

        @tf.function
        def masked_binary_crossentropy(y_true, y_pred, sample_weight=None):
            """Custom loss function for with missing mask applied.

            Ignores missing data in the calculation of the loss function.

            Args:
                y_true (tensorflow.tensor): Input one-hot encoded 3D tensor.

                y_pred (tensorflow.tensor): Predicted values, should have undergone sigmoid activation.

                sample_weight (numpy.ndarray): 2D matrix of sample weights.

            Returns:
                float: Binary crossentropy loss value.
            """
            # Mask out missing values.
            y_true_masked = tf.boolean_mask(
                y_true, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            )

            y_pred_masked = tf.boolean_mask(
                y_pred, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            )

            if sample_weight is not None:
                sample_weight_masked = tf.boolean_mask(
                    tf.convert_to_tensor(sample_weight),
                    tf.reduce_any(tf.not_equal(y_true, -1), axis=2),
                )
            else:
                sample_weight_masked = None

            return bce(
                y_true_masked,
                y_pred_masked,
                sample_weight=sample_weight_masked,
            )

        return masked_binary_crossentropy

    @staticmethod
    def make_masked_categorical_accuracy():
        """Make categorical crossentropy loss function with missing mask.

        Returns:
            callable: Function that calculates categorical crossentropy loss.
        """

        # @tf.function
        def masked_categorical_accuracy(y_true, y_pred, sample_weight=None):
            """Custom loss function for neural network model with missing mask.
            Ignores missing data in the calculation of the loss function.
            Args:
                y_true (tensorflow.tensor): Input one-hot encoded 3D tensor.
                y_pred (tensorflow.tensor): Predicted values.
                sample_weight (numpy.ndarray): 2D matrix of sample weights.

            Returns:
                float: Mean squared error loss value with missing data masked.
            """
            y_true_masked = tf.boolean_mask(
                y_true, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            )

            y_pred_masked = tf.boolean_mask(
                y_pred, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            )

            return cca(y_true_masked, y_pred_masked)

        return masked_categorical_accuracy

    @staticmethod
    def make_masked_categorical_crossentropy():
        """Make categorical crossentropy loss function with missing mask.

        Args:
            kl_weight (float, optional): KL Divergence weight to use.

        Returns:
            callable: Function that calculates categorical crossentropy loss.
        """

        # @tf.function
        def masked_categorical_crossentropy(
            y_true, y_pred, sample_weight=None
        ):
            """Custom loss function for neural network model with missing mask.
            Ignores missing data in the calculation of the loss function.

            Args:
                y_true (tensorflow.tensor): Input one-hot encoded 3D tensor.
                y_pred (tensorflow.tensor): Predicted values.
                sample_weight (numpy.ndarray): 2D matrix of sample weights.

            Returns:
                float: Mean squared error loss value with missing data masked.
            """
            # Mask out missing values.
            y_true_masked = tf.boolean_mask(
                y_true, tf.reduce_any(tf.not_equal(y_true, -1), axis=-1)
            )

            y_pred_masked = tf.boolean_mask(
                y_pred, tf.reduce_any(tf.not_equal(y_true, -1), axis=-1)
            )

            if sample_weight is not None:
                sample_weight_masked = tf.boolean_mask(
                    tf.convert_to_tensor(sample_weight),
                    tf.reduce_any(tf.not_equal(y_true, -1), axis=-1),
                )
            else:
                sample_weight_masked = None

            return cce_proba(
                y_true_masked,
                y_pred_masked,
                sample_weight=sample_weight_masked,
            )

        return masked_categorical_crossentropy

    @staticmethod
    def kl_divergence(z_mean, z_log_var, kl_weight=0.5):
        kl_loss = -0.5 * (
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        return tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))

        # Another way of doing it.
        # TODO: Test both ways.
        # z_sigma = tf.math.exp(0.5 * z_log_var)
        # return tf.reduce_sum(
        #         tf.math.square(z_mean) + tf.math.square(z_sigma) - z_log_var - 1.0,
        #         axis=-1,
        #     )

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
                mm = ~original_missing_mask[:, i]
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
