import logging
import math
import os
import sys
import random
from collections import defaultdict
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from sklearn.preprocessing import label_binarize

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2', '3'}

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

# Custom module imports
try:
    from ..data_processing.transformers import SimGenotypeDataTransformer
except (ModuleNotFoundError, ValueError):
    from data_processing.transformers import SimGenotypeDataTransformer


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

    def simulate_missing(
        self, X, genotype_data, prop_missing, strategy, missing=-9
    ):
        """Simulate missing data and generate missing masks.

        Generates missing masks for original dataset, all missing values (simulated + original), and simulated missing values.

        Args:
            X (numpy.ndarray): Original dataset with original missing values.
            genotype_data (GenotypeData): Initialized GenotypeData object.
            prop_missing (float): Proportion of missing values to simulate.
            strategy (str): Strategy to use for simulation. Supported options include "random", "nonrandom", and "nonrandom_weighted".
            missing (int): Missing data value.

        Returns:
            numpy.ndarray: Dataset with simulated and original missing values.
            numpy.ndarray: Boolean mask of original missing values.
            numpy.ndarray: Boolean mask of simulated missing values.
            numpy.ndarray: Boolean mask of original + simulated missing values.
        """
        # Get original missing values.
        original_mask = (
            pd.DataFrame(X).replace(missing, np.nan).isna().to_numpy()
        )

        Xt = SimGenotypeDataTransformer(
            genotype_data,
            prop_missing=prop_missing,
            strategy=strategy,
        ).fit_transform(X)

        all_mask = pd.DataFrame(Xt).replace(missing, np.nan).isna().to_numpy()

        # Get values where original value was not missing and simulated missing
        # data is missing.
        sim_mask = np.logical_and(all_mask, original_mask == False)

        return (
            Xt,
            original_mask,
            sim_mask,
            all_mask,
        )

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
    ):
        """Prepare training batches in the custom training loop.

        Args:
            V (numpy.ndarray): Input to batch subset and refine, of shape (n_samples, n_components).

            y (numpy.ndarray): Target to use to refine input V. shape (n_samples, n_features).

            batch_size (int): Batch size to subset.

            batch_idx (int): Current batch index.

            trainable (bool): Whether tensor v should be trainable.

            n_components (int): Number of principal components used in V.

            sample_weight (List[float] or None): List of floats of shape (n_samples,) with sample weights. sample_weight argument must be passed to fit().

            missing_mask (numpy.ndarray): Boolean array with True for missing values and False for observed values.

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

    def make_masked_loss(self):
        """Make loss function for use with a keras model.

        Args:
            missing_mask (numpy.ndarray): Boolean missing mask of shape (n_samples, n_features).

        Returns:
            callable: Function that calculates loss.
        """

        def masked_loss(y_true, y_pred):
            """Custom loss function for neural network model with missing mask.

            Ignores missing data in the calculation of the loss function.

            Args:
                y_true (tensorflow.tensor): Input one-hot encoded 3D tensor.
                y_pred (tensorflow.tensor): Predicted values.

            Returns:
                float: Mean squared error loss value with missing data masked.
            """

            sample_weight = args[0]

            if sample_weight is not None:
                print("yes")
            else:
                print("no")
            sys.exit()

            # y_true_masked = tf.boolean_mask(
            #     y_true, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            # )

            # y_pred_masked = tf.boolean_mask(
            #     y_pred, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            # )

            # sample_weight_masked = tf.boolean_mask(
            #     tf.convert_to_tensor(sample_weight),
            #     tf.reduce_any(tf.not_equal(y_true, -1), axis=2),
            # )

            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

            return loss_fn(
                y_true_masked, y_pred_masked, sample_weight=sample_weight_masked
            )

        return masked_loss

    def make_masked_acc(self):
        """Make accuracy function for use with a keras model.

        Args:
            missing_mask (numpy.ndarray): Boolean missing mask of shape (n_samples, n_features).
            sample_weight (tensorflow.tensor or None, optional): Sample weights to multiply loss by.

        Returns:
            callable: Function that calculates loss.
        """

        def masked_acc(y_true, y_pred):
            """Custom loss function for neural network model with missing mask.

            Ignores missing data in the calculation of the loss function.

            Args:
                y_true (tensorflow.tensor): Input one-hot encoded 3D tensor.
                y_pred (tensorflow.tensor): Predicted values.

            Returns:
                float: Mean squared error loss value with missing data masked.
            """

            sample_weight = args[0]

            # y_true_masked = tf.boolean_mask(
            #     y_true, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            # )

            # y_pred_masked = tf.boolean_mask(
            #     y_pred, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            # )

            # sample_weight_masked = tf.boolean_mask(
            #     tf.convert_to_tensor(sample_weight),
            #     tf.reduce_any(tf.not_equal(y_true, -1), axis=2),
            # )

            metric = tf.keras.metrics.CategoricalAccuracy(
                name="masked_accuracy"
            )
            metric.update_state(
                y_true_masked,
                y_pred_masked,
                sample_weight=sample_weight_masked,
            )
            return metric.result()

        return masked_acc

    # def custom_loss(self, input_and_mask, y_pred):
    #     """Custom loss function for neural network model with missing mask.

    #     Ignores missing data in the calculation of the loss function.

    #     Args:
    #         input_and_mask (numpy.ndarray): Input one-hot encoded array with missing values also one-hot encoded and h-stacked.

    #         y_pred (numpy.ndarray): Predicted values.

    #     Returns:
    #         float: Mean squared error loss value with missing data masked.
    #     """
    #     n_features = y_pred.numpy().shape[1]

    #     true_indices = range(n_features)
    #     missing_indices = range(n_features, n_features * 2)

    #     # Split features and missing mask.
    #     y_true = tf.gather(input_and_mask, true_indices, axis=1)
    #     missing_mask = tf.gather(input_and_mask, missing_indices, axis=1)

    #     observed_mask = tf.subtract(1.0, missing_mask)
    #     y_true_observed = tf.multiply(y_true, observed_mask)
    #     pred_observed = tf.multiply(y_pred, observed_mask)

    #     return tf.keras.metrics.mean_squared_error(
    #         y_true=y_true_observed, y_pred=pred_observed
    #     )

    def val_loss(self, input_and_mask, y_pred):
        """Custom loss function for neural network model test dataset with missing mask.

        Ignores missing data in the calculation of the loss function.

        Args:
            input_and_mask (numpy.ndarray): Input one-hot encoded array with missing values also one-hot encoded and h-stacked.

            y_pred (numpy.ndarray): Predicted values.

        Returns:
            float: Mean squared error loss value with missing data masked.
        """

    def make_acc(self):
        """Make loss function for use with a keras model.

        Returns:
            callable: Function that calculates loss.
        """

        def accuracy_masked(input_and_mask, y_pred):
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

            metric = tf.keras.metrics.BinaryAccuracy(name="accuracy_masked")
            metric.update_state(y_true_observed, pred_observed)
            return metric.result()

        return accuracy_masked

    def accuracy(self, input_and_mask, y_pred):
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

        metric = tf.keras.metrics.BinaryAccuracy(name="accuracy_masked")
        metric.update_state(y_true_observed, pred_observed)
        return metric.result()

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

    @staticmethod
    def plot_grid_search(cv_results, prefix):
        """Plot cv_results_ from a grid search for each parameter.

        Saves a figure to disk.

        Args:
            cv_results (numpy.ndarray): the cv_results_ attribute from a trained grid search object.

            prefix (str): Prefix to use for saving the plot to file.
        """
        ## Results from grid search
        results = pd.DataFrame(cv_results)
        means_test = results["mean_test_score"]
        filter_col = [col for col in results if col.startswith("param_")]
        params_df = results[filter_col].astype(str)
        params_df["score"] = np.array(means_test)

        # Get number of needed subplot rows.
        tot = len(filter_col)
        cols = 4
        rows = int(np.ceil(tot / cols))
        remainder = tot % cols

        fig = plt.figure(1, figsize=(20, 10))
        fig.tight_layout(pad=3.0)

        # Set font properties.
        font = {"size": 12}
        plt.rc("font", **font)

        for i, p in enumerate(filter_col, start=1):
            # Get maximum score for each parameter setting.
            df_plot = params_df.groupby(p)["score"].agg("max")

            # Convert to float if not supposed to be string.
            try:
                df_plot.index = df_plot.index.astype(float)
            except TypeError:
                pass

            df_plot = df_plot.sort_index()

            ax = fig.add_subplot(rows, cols, i)
            ax.plot(df_plot.index.astype(str), df_plot.values, "-o")
            ax.set_xlabel(p.lstrip("param_").lower())
            ax.set_ylabel("Max Score")
            ax.set_ylim([0, 1])

        fig.savefig(f"{prefix}_gridsearch.pdf", bbox_inches="tight")

    @staticmethod
    def compute_roc_auc_micro_macro(y_true, y_pred):
        """Compute ROC curve with AUC scores.

        ROC (Receiver Operating Characteristic) curves and AUC (area under curve) scores are computed per-class and for micro and macro averages.

        Args:
            y_true (numpy.ndarray): Ravelled numpy array of shape (n_samples * n_features,).

            y_pred (numpy.ndarray): Ravelled numpy array of shape (n_samples * n_features,).

        Returns:
            Dict[str, Any]: Dictionary with true and false positive rates along probability threshold curve per class, micro and macro tpr and fpr curves averaged across classes, and AUC scores per-class and for micro and macro averages.
        """
        # Binarize the output fo use with ROC-AUC.
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2])
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

    @staticmethod
    def compute_pr(y_true, y_pred):
        """Compute precision-recall curve with Average Precision scores.

        PR and AP scores are computed per-class and for micro and macro averages.

        Args:
            y_true (numpy.ndarray): Ravelled numpy array of shape (n_samples * n_features,).

            y_pred (numpy.ndarray): Ravelled numpy array of shape (n_samples * n_features,).

        Returns:
            Dict[str, Any]: Dictionary with precision and recall curves per class and micro and macro averaged across classes, plus AP scores per-class and for micro and macro averages.
        """
        # Binarize the output fo use with ROC-AUC.
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2])
        n_classes = y_true_bin.shape[1]

        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_true_bin[:, i], y_pred_bin[:, i]
            )
            average_precision[i] = average_precision_score(
                y_true_bin[:, i], y_pred_bin[:, i]
            )

        # A "micro-average": quantifying score on all classes jointly.
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true_bin.ravel(), y_pred_bin.ravel()
        )

        average_precision["micro"] = average_precision_score(
            y_true_bin, y_pred_bin, average="micro"
        )

        average_precision["macro"] = average_precision_score(
            y_true_bin, y_pred_bin, average="macro"
        )

        # Aggregate all false positive rates
        all_recall = np.unique(
            np.concatenate([recall[i] for i in range(n_classes)])
        )

        # Then interpolate all ROC curves at these points.
        mean_precision = np.zeros_like(all_recall)
        for i in range(n_classes):
            mean_precision += np.interp(all_recall, recall[i], precision[i])

        # Finally, average it and compute AUC.
        mean_precision /= n_classes

        recall["macro"] = all_recall
        precision["macro"] = mean_precision

        results = dict()

        results["micro"] = average_precision["micro"]
        results["macro"] = average_precision["macro"]
        results["recall_macro"] = recall["macro"]
        results["precision_macro"] = precision["macro"]
        results["recall_micro"] = recall["micro"]
        results["precision_micro"] = precision["micro"]
        results["recall_0"] = recall[0]
        results["recall_1"] = recall[1]
        results["recall_2"] = recall[2]
        results["precision_0"] = precision[0]
        results["precision_1"] = precision[1]
        results["precision_2"] = precision[2]
        results[0] = average_precision[0]
        results[1] = average_precision[1]
        results[2] = average_precision[2]

        return results

    @staticmethod
    def plot_metrics(metrics, num_classes, prefix):
        """Plot performance metrics for classifier.

        Saves plot to PDF file on disk.

        Args:
            metrics (Dict[str, Any]): Per-class, micro, and macro-averaged metrics including accuracy, ROC-AUC, and Precision-Recall with Average Precision scores.

            num_classes (int): Number of classes evaluated.

            prefix (str): Prefix to use for output plot.
        """
        # Set font properties.
        font = {"size": 12}
        plt.rc("font", **font)

        fn = f"{prefix}_metrics_plot.pdf"
        fig = plt.figure(figsize=(20, 10))

        acc = round(metrics["accuracy"] * 100, 2)

        fig.suptitle(f"Performance Metrics\nAccuracy: {acc}")
        axs = fig.subplots(nrows=1, ncols=2)
        plt.subplots_adjust(hspace=0.5)

        # Line weight
        lw = 2

        roc_auc = metrics["roc_auc"]
        pr_ap = metrics["precision_recall"]

        metric_list = [roc_auc, pr_ap]

        for metric, ax in zip(metric_list, axs):

            if "fpr_micro" in metric:
                prefix1 = "fpr"
                prefix2 = "tpr"
                lab1 = "ROC"
                lab2 = "AUC"
                xlab = "False Positive Rate"
                ylab = "True Positive Rate"
                title = "Receiver Operating Characteristic (ROC)"
                baseline = [0, 1]

            elif "recall_micro" in metric:
                prefix1 = "recall"
                prefix2 = "precision"
                lab1 = "Precision-Recall"
                lab2 = "AP"
                xlab = "Recall"
                ylab = "Precision"
                title = "Precision-Recall"

                # Plot iso-f1 curves.
                f_scores = np.linspace(0.2, 0.8, num=4)
                for i, f_score in enumerate(f_scores):
                    x = np.linspace(0.01, 1)
                    y = f_score * x / (2 * x - f_score)
                    ax.plot(
                        x[y >= 0],
                        y[y >= 0],
                        color="gray",
                        alpha=0.2,
                        linewidth=lw,
                        label="Iso-F1 Curves" if i == 0 else "",
                    )
                    ax.annotate(f"F1={f_score:0.1f}", xy=(0.9, y[45] + 0.02))

            # Plot ROC curves.
            ax.plot(
                metric[f"{prefix1}_micro"],
                metric[f"{prefix2}_micro"],
                label=f"Micro-averaged {lab1} Curve ({lab2} = {metric['micro']:.2f})",
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            ax.plot(
                metric[f"{prefix1}_macro"],
                metric[f"{prefix2}_macro"],
                label=f"Macro-averaged {lab1} Curve ({lab2} = {metric['macro']:.2f})",
                color="navy",
                linestyle=":",
                linewidth=4,
            )

            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(num_classes), colors):
                ax.plot(
                    metric[f"{prefix1}_{i}"],
                    metric[f"{prefix2}_{i}"],
                    color=color,
                    lw=lw,
                    label=f"{lab1} Curve of class {i} ({lab2} = {metric[i]:.2f})",
                )

            if "fpr_micro" in metric:
                # Make center baseline
                ax.plot(
                    baseline,
                    baseline,
                    "k--",
                    linewidth=lw,
                    label="No Classification Skill",
                )

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.05)
            ax.set_xlabel(f"{xlab}")
            ax.set_ylabel(f"{ylab}")
            ax.set_title(f"{title}")
            ax.legend(loc="best")

        fig.savefig(fn, bbox_inches="tight")
        plt.close()
        plt.clf()
        plt.cla()

    @staticmethod
    def decode(y, missing_mask):
        """Evaluate VAE predictions by calculating the highest predicted value.

        Calucalates highest predicted value for each row vector and each class, setting the most likely class to 1.0.

        Args:
            y (numpy.ndarray): Input one-hot encoded data with predictions as last dimension.

            missing_mask (numpy.ndarray): Boolean mask of missing values.

        Returns:
            numpy.ndarray: Imputed one-hot encoded values.

            pandas.DataFrame: One-hot encoded pandas DataFrame with no missing values.
        """
        Xprob = y
        Xt = np.apply_along_axis(mle, axis=2, arr=Xprob)
        Xpred = np.argmax(Xt, axis=2)
        Xtrue = np.argmax(y, axis=2)
        Xdecoded = np.zeros((Xpred.shape[0], Xpred.shape[1]))
        for idx, row in enumerate(Xdecoded):
            imputed_vals = np.zeros(len(row))
            known_vals = np.zeros(len(row))
            imputed_idx = np.nonzero(missing_mask[idx])
            known_idx = np.where(missing_mask[idx] == 0)
            Xdecoded[idx, imputed_idx] = Xpred[idx, imputed_idx]
            Xdecoded[idx, known_idx] = Xtrue[idx, known_idx]
        return Xdecoded.astype("int8")
