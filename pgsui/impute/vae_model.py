import logging
import os
import sys
import warnings

# Import tensorflow with reduced warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").disabled = True
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import tensorflow as tf

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

from tensorflow.keras.layers import (
    Dropout,
    Dense,
    Reshape,
    LeakyReLU,
    PReLU,
)

from tensorflow.keras.regularizers import l1_l2

# Custom Modules
try:
    from .neural_network_methods import NeuralNetworkMethods
except (ModuleNotFoundError, ValueError):
    from impute.neural_network_methods import NeuralNetworkMethods


class VAEModel(tf.keras.Model):
    def __init__(
        self,
        y,
        y_test,
        n_components,
        missing_mask,
        output_shape,
        hidden_layer_sizes,
        num_hidden_layers,
        weights_initializer,
        dropout_rate,
        sample_weight,
        hidden_activation,
        l1_penalty,
        l2_penalty,
        kernel_initializer,
        num_classes,
        optimizer,
    ):
        """Create a variational autoencoder model with the following items: InputLayer -> DenseLayer1 -> Dropout1 -> DenseLayer2 -> Dropout2 -> DenseLayer3 -> Dropout3 -> DenseLayer4 -> OutputLayer.

        Returns:
            keras model object: Compiled Keras model.
        """
        super(VAEModel, self).__init__()

        self.n_components = n_components
        self._sample_weight = sample_weight
        self._missing_mask = missing_mask
        self.weights_initializer = weights_initializer
        self.dropout_rate = dropout_rate

        self.nn_ = NeuralNetworkMethods()

        self._x = y.copy()
        self._y = y.copy()
        self._x_test = y_test.copy()
        self._y_test = y_test.copy()

        hidden_layer_sizes = self.nn_.validate_hidden_layers(
            hidden_layer_sizes, num_hidden_layers
        )

        hidden_layer_sizes = self.nn_.get_hidden_layer_sizes(
            y.shape[1], self.n_components, hidden_layer_sizes
        )

        self.nn_.validate_model_inputs(y, missing_mask, output_shape)

        first_layer_size = hidden_layer_sizes[0]
        n_dims = self._x.shape[1]

        # hidden_layer_sizes = self._get_hidden_layer_sizes()
        # first_layer_size = hidden_layer_sizes[0]
        # n_dims = self.data.shape[1]

        if l1_penalty == 0.0 and l2_penalty == 0.0:
            kernel_regularizer = None
        else:
            kernel_regularizer = l1_l2(l1_penalty, l2_penalty)

        self.kernel_regularizer = kernel_regularizer
        kernel_initializer = weights_initializer

        if hidden_activation.lower() == "leaky_relu":
            activation = LeakyReLU(alpha=0.01)

        elif hidden_activation.lower() == "prelu":
            activation = PReLU()

        elif hidden_activation.lower() == "selu":
            activation = "selu"
            kernel_initializer = "lecun_normal"

        else:
            activation = hidden_activation

        if num_hidden_layers > 5:
            raise ValueError(
                f"The maximum number of hidden layers is 5, but got "
                f"{num_hidden_layers}"
            )

        self.dense2 = None
        self.dense3 = None
        self.dense4 = None
        self.dense5 = None
        self.dense6 = None
        self.dense7 = None
        self.dense8 = None
        self.dense9 = None
        self.dense10 = None

        # Construct multi-layer perceptron.
        # Add hidden layers dynamically.
        self.dense1 = Dense(
            hidden_layer_sizes[0],
            input_shape=(n_dims,),
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )

        if num_hidden_layers >= 2:
            self.dense2 = Dense(
                hidden_layer_sizes[1],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if num_hidden_layers >= 3:
            self.dense3 = Dense(
                hidden_layer_sizes[2],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if num_hidden_layers >= 4:
            self.dense4 = Dense(
                hidden_layer_sizes[3],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if num_hidden_layers == 5:
            self.dense5 = Dense(
                hidden_layer_sizes[4],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        self.reduced_ndim = Dense(
            self.n_components,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )

        hidden_layer_sizes.reverse()

        # Construct multi-layer perceptron.
        # Add hidden layers dynamically.
        self.dense6 = Dense(
            hidden_layer_sizes[0],
            input_shape=(n_dims,),
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )

        if num_hidden_layers >= 2:
            self.dense7 = Dense(
                hidden_layer_sizes[1],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if num_hidden_layers >= 3:
            self.dense8 = Dense(
                hidden_layer_sizes[2],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if num_hidden_layers >= 4:
            self.dense9 = Dense(
                hidden_layer_sizes[3],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if num_hidden_layers == 5:
            self.dense10 = Dense(
                hidden_layer_sizes[4],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        self.output1 = Dense(
            output_shape * num_classes,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )

        self.rshp = Reshape((output_shape, num_classes))

        self.dropout_layer = Dropout(rate=dropout_rate)

    def call(self, inputs, training=None):
        """Forward propagates inputs through the model defined in __init__().

        Args:
            inputs (tf.keras.Input): Input tensor to forward propagate through the model.

            training (bool or None): Whether in training mode or not. Affects whether dropout is used.

        Returns:
            tf.keras.Model: Output tensor from forward propagation.
        """
        if self.dropout_rate == 0.0:
            training = False
        x = self.dense1(inputs)
        if training:
            x = self.dropout_layer(x, training=training)
        if self.dense2 is not None:
            x = self.dense2(x)
            if training:
                x = self.dropout_layer(x, training=training)
        if self.dense3 is not None:
            x = self.dense3(x)
            if training:
                x = self.dropout_layer(x, training=training)
        if self.dense4 is not None:
            x = self.dense4(x)
            if training:
                x = self.dropout_layer(x, training=training)
        if self.dense5 is not None:
            x = self.dense5(x)
            if training:
                x = self.dropout_layer(x, training=training)
        x = self.reduced_ndim(x)
        if training:
            x = self.dropout_layer(x, training=training)
        x = self.dense6(x)
        if self.dense7 is not None:
            x = self.dense7(x)
            if training:
                x = self.dropout_layer(x, training=training)
        if self.dense8 is not None:
            x = self.dense8(x)
            if training:
                x = self.dropout_layer(x, training=training)
        if self.dense9 is not None:
            x = self.dense9(x)
            if training:
                x = self.dropout_layer(x, training=training)
        if self.dense10 is not None:
            x = self.dense10(x)
            if training:
                x = self.dropout_layer(x, training=training)

        x = self.output1(x)
        return self.rshp(x)

    def model(self):
        """Here so that mymodel.model().summary() can be called for debugging."""
        x = tf.keras.Input(shape=(self.y_train.shape[1],))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def set_model_outputs(self):
        x = tf.keras.Input(shape=(self.y_train.shape[1],))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        self.outputs = model.outputs

    def train_step(self, data):
        """Custom training loop for one step (=batch) in a single epoch.

        This function will train on a batch of samples (rows), which can be adjusted with the ``batch_size`` parameter from the estimator.

        Args:
            data (Tuple[tf.EagerTensor, tf.EagerTensor]): Input tensorflow variables of shape (batch_size, n_components) and (batch_size, n_features, num_classes).

        Returns:
            Dict[str, float]: History object that gets returned from fit(). Contains the loss and any metrics specified in compile().
        """
        x = self._x
        y = self._y

        (
            x,
            y_true,
            sample_weight,
            missing_mask,
            batch_start,
            batch_end,
        ) = self.nn.prepare_training_batches(
            x,
            y,
            self._batch_size,
            self._batch_idx,
            True,
            self.n_components,
            self._sample_weight,
            self._missing_mask,
        )

        if sample_weight is not None:
            sample_weight_masked = tf.convert_to_tensor(
                sample_weight[~missing_mask], dtype=tf.float32
            )
        else:
            sample_weight_masked = None

        x_masked = tf.boolean_mask(
            tf.convert_to_tensor(x, dtype=tf.float32),
            tf.reduce_any(tf.not_equal(x, -1), axis=2),
        )

        y_true_masked = tf.boolean_mask(
            tf.convert_to_tensor(y_true, dtype=tf.float32),
            tf.reduce_any(tf.not_equal(y_true, -1), axis=2),
        )

        with tf.GradientTape() as tape:
            y_pred = self(x_masked, training=True)  # Forward pass
            y_pred_masked = tf.boolean_mask(
                y_pred, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            )

            ### NOTE: If you get the error, "'tuple' object has no attribute
            ### 'rank'", then convert y_true to a tensor object."
            loss = self.compiled_loss(
                y_true_masked,
                y_pred_masked,
                sample_weight=sample_weight_masked,
                regularization_losses=self.losses,
            )

        # NOTE: Earlier model architectures incorrectly
        # applied one gradient to all the variables, including
        # the weights and v. Here we apply them separately, per
        # the UBP manuscript.
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass.
            y_pred = self(v, training=True)
            y_pred_masked = tf.boolean_mask(
                y_pred, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            )
            ### NOTE: If you get the error, "'tuple' object has no attribute
            ### 'rank'", then convert y_true to a tensor object."
            loss = self.compiled_loss(
                y_true_masked,
                y_pred_masked,
                sample_weight=sample_weight_masked,
                regularization_losses=self.losses,
            )

        # gradient descent backpropagation
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        ### NOTE: If you get the error, "'tuple' object has no attribute
        ### 'rank', then convert y_true to a tensor object."
        self.compiled_metrics.update_state(
            y_true_masked,
            y_pred_masked,
            sample_weight=sample_weight_masked,
        )

        # history object that gets returned from fit().
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x = self._x
        y = self._y

        (
            x,
            y_true,
            sample_weight,
            missing_mask,
            batch_start,
            batch_end,
        ) = self.nn.prepare_training_batches(
            x,
            y,
            self._batch_size,
            self._batch_idx,
            True,
            self.n_components,
            self._sample_weight,
            self._missing_mask,
        )

        x_masked = tf.boolean_mask(
            tf.convert_to_tensor(x, dtype=tf.float32),
            tf.reduce_any(tf.not_equal(x, -1), axis=2),
        )

        y_true_masked = tf.boolean_mask(
            tf.convert_to_tensor(y_true, dtype=tf.float32),
            tf.reduce_any(tf.not_equal(y_true, -1), axis=2),
        )

        with tf.GradientTape() as tape:
            y_pred = self(x_masked, training=True)  # Forward pass
            y_pred_masked = tf.boolean_mask(
                y_pred, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            )

            ### NOTE: If you get the error, "'tuple' object has no attribute
            ### 'rank'", then convert y_true to a tensor object."
            loss = self.compiled_loss(
                y_true_masked,
                y_pred_masked,
                sample_weight=sample_weight_masked,
                regularization_losses=self.losses,
            )

        # NOTE: Earlier model architectures incorrectly
        # applied one gradient to all the variables, including
        # the weights and v. Here we apply them separately, per
        # the UBP manuscript.
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass.
            y_pred = self(v, training=True)
            y_pred_masked = tf.boolean_mask(
                y_pred, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            )
            ### NOTE: If you get the error, "'tuple' object has no attribute
            ### 'rank'", then convert y_true to a tensor object."
            loss = self.compiled_loss(
                y_true_masked,
                y_pred_masked,
                sample_weight=sample_weight_masked,
                regularization_losses=self.losses,
            )

        # gradient descent backpropagation
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        ### NOTE: If you get the error, "'tuple' object has no attribute
        ### 'rank', then convert y_true to a tensor object."
        self.compiled_metrics.update_state(
            y_true_masked,
            y_pred_masked,
            sample_weight=sample_weight_masked,
        )

        # history object that gets returned from fit().
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
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def missing_mask(self):
        return self._missing_mask

    @property
    def sample_weight(self):
        return self._sample_weight

    @batch_size.setter
    def batch_size(self, value):
        """Set batch_size parameter."""
        self._batch_size = int(value)

    @batch_idx.setter
    def batch_idx(self, value):
        """Set current batch (=step) index."""
        self._batch_idx = int(value)

    @x.setter
    def x(self, value):
        """Set x after each epoch."""
        self._x = value

    @y.setter
    def y(self, value):
        """Set y after each epoch."""
        self._y = value

    @missing_mask.setter
    def missing_mask(self, value):
        """Set y after each epoch."""
        self._missing_mask = value

    @sample_weight.setter
    def sample_weight(self, value):
        self._sample_weight = value
