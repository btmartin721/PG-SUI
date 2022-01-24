import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


class NeuralNetworkMethods:
    """Methods common to all neural network imputer classes and loss functions"""

    def __init__(self):
        self.data = None

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

    def validate_model_inputs(
        self, y, y_test, missing_mask, missing_mask_test, output_shape
    ):
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

        if y_test is None:
            raise TypeError("y_test must not be NoneType")

        if missing_mask is None:
            raise TypeError("missing_mask must not be NoneType.")

        if missing_mask_test is None:
            raise TypeError("missing_mask_test must not be NoneType.")

        if output_shape is None:
            raise TypeError("output_shape must not be NoneType.")

    def prepare_training_batches(
        self, V, y, batch_size, batch_idx, trainable, n_components
    ):
        """Prepare training batches in the custom training loop.

        Args:
            V (numpy.ndarray): Input to batch subset and refine, of shape (n_samples, n_components).

            y (numpy.ndarray): Target to use to refine input V. Has missing data mask horizontally concatenated (with np.hstack); shape (n_samples, n_features * 2).

            batch_size (int): Batch size to subset.

            batch_idx (int): Current batch index.

            trainable (bool): Whether tensor v should be trainable.

            n_components (int): Number of principal components used in V.

        Returns:
            tf.Variable: Input tensor v with current batch assigned.
            numpy.ndarray: Current batch of arget data (actual input) used to refine v.
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

        v = tf.Variable(
            tf.zeros([batch_size, n_components]),
            trainable=trainable,
            dtype=tf.float32,
        )

        # Assign current batch to tf.Variable v.
        v.assign(v_batch)

        return v, y_true, batch_start, batch_end

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

    def custom_loss(self, input_and_mask, y_pred):
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

        return tf.keras.metrics.mean_squared_error(
            y_true=y_true_observed, y_pred=pred_observed
        )

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

    def set_compile_params(self, optimizer, learning_rate, search_mode=False):
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

        if search_mode:
            # Doing grid search. Params are callables.
            optimizer = opt
            loss = self.custom_loss
            metrics = [self.accuracy]
        else:
            # No grid search. Optimizer params are initialized.
            optimizer = opt(learning_rate=learning_rate)
            loss = self.make_reconstruction_loss()
            metrics = [self.make_acc()]

        return {
            "optimizer": optimizer,
            "loss": loss,
            "metrics": metrics,
            "run_eagerly": True,
        }

    @staticmethod
    def plot_grid_search(cv_results):
        """Plot cv_results_ from a grid search.

        Saves a figure to disk.

        Args:
            cv_results: the cv_results_ attribute from a trained grid search object.
        """
        ## Results from grid search
        results = pd.DataFrame(cv_results)
        means_test = results["mean_test_score"]
        filter_col = [col for col in results if col.startswith("param_")]
        params_df = results[filter_col]

        # Get number of needed subplot rows.
        tot = len(filter_col)
        cols = 4
        rows = tot / cols
        rows += tot % cols
        position = range(1, tot + 1)

        fig = plt.figure(1, figsize=(20, 10))

        fig.suptitle("Parameter Scores")

        for i, p in enumerate(filter_col):
            x = np.array(params_df[p])
            y = np.array(means_test)
            ax = fig.add_subplot(rows, cols, position[i])
            ax.plot(x, y, "-o")
            ax.set_xlabel(p.lstrip("param_").lower())
            ax.set_ylabel("Mean Accuracy")

        # plt.legend()
        fig.savefig("gridsearch.pdf", bbox_inches="tight")
