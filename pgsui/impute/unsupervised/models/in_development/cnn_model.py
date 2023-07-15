import logging
import os
import sys
import warnings
import math

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
    Activation,
    Flatten,
    BatchNormalization,
    LeakyReLU,
    PReLU,
)

from tensorflow.keras.regularizers import l1_l2

# Custom Modules
try:
    from ...neural_network_methods import NeuralNetworkMethods
except (ModuleNotFoundError, ValueError, ImportError):
    from impute.unsupervised.neural_network_methods import NeuralNetworkMethods


class SoftOrdering1DCNN(tf.keras.Model):
    def __init__(
        self,
        y=None,
        output_shape=None,
        weights_initializer="glorot_normal",
        hidden_layer_sizes="midpoint",
        num_hidden_layers=1,
        hidden_activation="elu",
        l1_penalty=1e-6,
        l2_penalty=1e-6,
        dropout_rate=0.2,
        num_classes=4,
        sample_weight=None,
        batch_size=32,
        missing_mask=None,
        activation=None,
        channel_increase_rate=2,
        initial_hidden_size=2048,
        num_groups=256,
    ):
        super(SoftOrdering1DCNN, self).__init__()

        self._y = y
        self._missing_mask = missing_mask
        self._sample_weight = sample_weight
        self._batch_idx = 0
        self._batch_size = batch_size
        self.output_activation = activation
        self.sample_weight = sample_weight

        self.nn_ = NeuralNetworkMethods()
        self.binary_accuracy = self.nn_.make_masked_binary_accuracy(
            is_vae=True
        )

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.accuracy_tracker = tf.keras.metrics.Mean(name="accuracy")

        # y_train[1] dimension.
        self.n_features = output_shape * num_classes

        self.weights_initializer = weights_initializer
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.channel_increase_rate = channel_increase_rate
        self.initial_hidden_size = initial_hidden_size
        self.channel_size1 = num_groups
        self.channel_size2 = num_groups * 2
        self.channel_size3 = num_groups * 2

        nn = NeuralNetworkMethods()

        # hidden_layer_sizes = nn.validate_hidden_layers(
        #     self.hidden_layer_sizes, self.num_hidden_layers
        # )

        # hidden_layer_sizes = nn.get_hidden_layer_sizes(
        #     self.n_features, self.n_components, hidden_layer_sizes, vae=True
        # )

        # hidden_layer_sizes = [h * self.num_classes for h in hidden_layer_sizes]

        if self.l1_penalty == 0.0 and self.l2_penalty == 0.0:
            kernel_regularizer = None
        else:
            kernel_regularizer = l1_l2(self.l1_penalty, self.l2_penalty)

        kernel_initializer = self.weights_initializer

        if self.hidden_activation.lower() == "leaky_relu":
            activation = LeakyReLU(alpha=0.01)

        elif self.hidden_activation.lower() == "prelu":
            activation = PReLU()

        elif self.hidden_activation.lower() == "selu":
            activation = "selu"
            kernel_initializer = "lecun_normal"

        else:
            activation = self.hidden_activation

        if num_hidden_layers > 5:
            raise ValueError(
                f"The maximum number of hidden layers is 5, but got "
                f"{num_hidden_layers}"
            )

        hidden_size = initial_hidden_size
        if self.n_features >= hidden_size:
            scaling_factor = int(math.ceil(self.n_features / hidden_size)) * 2
            hidden_size *= num_groups * int(
                math.ceil((scaling_factor / num_groups))
            )
        else:
            # If hidden_size is close in number to n_features
            if abs(hidden_size - self.n_features) <= (hidden_size // 2):
                hidden_size *= 2

        # Model adapted from: https://medium.com/spikelab/convolutional-neural-networks-on-tabular-datasets-part-1-4abdd67795b6

        signal_size1 = hidden_size // num_groups
        signal_size2 = signal_size1 // 2
        signal_size3 = signal_size1 // 4 * self.channel_size3

        self.signal_size1 = signal_size1
        self.signal_size2 = signal_size2
        self.signal_size3 = signal_size3

        self.batch_norm1 = BatchNormalization()
        self.dropout1 = Dropout(self.dropout_rate)
        self.dense1 = Dense(
            hidden_size,
            input_shape=(self.n_features,),
            activation=hidden_activation,
            kernel_initializer=kernel_initializer,
        )

        self.rshp = Reshape((num_groups, signal_size1))

        self.batch_norm_c1 = BatchNormalization()
        self.conv1 = tf.keras.layers.Conv1D(
            self.channel_size1 * self.channel_increase_rate,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=signal_size1,
            kernel_initializer=kernel_initializer,
            activation=hidden_activation,
        )

        self.avg_po_c1 = tf.keras.layers.AveragePooling1D(
            pool_size=4, padding="valid"
        )

        self.batch_norm_c2 = BatchNormalization()
        self.dropout_c2 = Dropout(self.dropout_rate)
        self.conv2 = tf.keras.layers.Conv1D(
            self.channel_size2,
            kernel_size=3,
            stride=1,
            padding=1,
            kernel_initializer=kernel_initializer,
            activation=hidden_activation,
        )

        self.batch_norm_c3 = BatchNormalization()
        self.dropout_c3 = Dropout(self.dropout_rate)
        self.conv3 = tf.keras.layers.Conv1D(
            self.channel_size2,
            kernel_size=3,
            stride=1,
            padding=1,
            kernel_initializer=kernel_initializer,
            activation=hidden_activation,
        )

        self.batch_norm_c4 = BatchNormalization()
        self.dropout_c4 = Dropout(self.dropout_rate)
        self.conv4 = tf.keras.layers.Conv1D(
            self.channel_size2,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=signal_size1,
            kernel_initializer=kernel_initializer,
            activation=None,
        )

        self.act_c4 = Activation(hidden_activation)

        self.max_po_c4 = tf.keras.layers.MaxPooling1D(
            pool_size=4, stride=2, padding=1
        )

        self.flatten = Flatten()

        self.batch_norm2 = BatchNormalization()
        self.dropout2 = Dropout(self.dropout_rate)
        self.dense2 = Dense(
            self.n_features, kernel_initializer=kernel_initializer
        )
        self.rshp2 = Reshape((output_shape, num_classes))
        self.act2 = Activation(activation)

    def call(self, inputs, training=None):
        """Call the model on a particular input.

        Args:
            input (tf.Tensor): Input tensor. Must be one-hot encoded.

        Returns:
            tf.Tensor: Output predictions. Will be one-hot encoded.
        """
        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.rshp(x)
        x = self.conv1(x)
        x = self.batch_norm_c1(x, training=training)
        x = self.avg_po_c1(x)
        x = self.conv2(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout_c2(x, training=training)
        x_s = x
        x = self.conv3(x)
        x = self.batch_norm_c3(x, training=training)
        x = self.dropout(x, training=training)
        x = self.conv4(x)
        x = self.batch_norm_c4(x, training=training)
        x += x_s
        x = self.act_c4(x)
        x = self.max_po_c4(x)
        x = self.dropout1(x)
        x = self.rshp(x)
        x = self.batch_norm_c1(x)
        x = self.conv1(x)
        x = self.avg_po_c1(x)
        x = self.flatten(x)
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.rshp2(x)
        return self.act2(x)

    def model(self):
        """Here so that mymodel.model().summary() can be called for debugging."""
        x = tf.keras.Input(shape=(self.n_features * self.num_classes,))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def set_model_outputs(self):
        x = tf.keras.Input(shape=(self.n_features * self.num_classes,))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        self.outputs = model.outputs

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.accuracy_tracker,
        ]

    @tf.function
    def train_step(self, data):
        # if isinstance(data, tuple):
        #     if len(data) == 2:
        #         x, y = data
        #         sample_weight = None
        #     else:
        #         x, y, sample_weight = data
        # else:
        #     raise TypeError("Target y must be supplied to fit for this model.")

        # Set in the UBPCallbacks() callback.
        y = self._y

        (
            y,
            y_true,
            sample_weight,
            missing_mask,
            batch_start,
            batch_end,
        ) = self.nn_.prepare_training_batches(
            y,
            y,
            self._batch_size,
            self._batch_idx,
            True,
            self.n_components,
            self._sample_weight,
            self._missing_mask,
            ubp=False,
        )

        if sample_weight is not None:
            sample_weight_masked = tf.convert_to_tensor(
                sample_weight[~missing_mask], dtype=tf.float32
            )
        else:
            sample_weight_masked = None

        y_true_masked = tf.boolean_mask(
            tf.convert_to_tensor(y_true, dtype=tf.float32),
            tf.reduce_any(tf.not_equal(y_true, -1), axis=2),
        )

        with tf.GradientTape() as tape:
            reconstruction = self(tf.convert_to_tensor(y), training=True)

            y_pred_masked = tf.boolean_mask(
                reconstruction, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            )

            # Returns binary crossentropy loss.
            loss = self.compiled_loss(
                y_true_masked,
                y_pred_masked,
                sample_weight=sample_weight_masked,
            )

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss)

        ### NOTE: If you get the error, "'tuple' object has no attribute
        ### 'rank', then convert y_true to a tensor object."
        # self.compiled_metrics.update_state(
        self.accuracy_tracker.update_state(
            self.binary_accuracy(
                y_true_masked,
                y_pred_masked,
                sample_weight=sample_weight_masked,
            )
        )

        return {
            "loss": self.total_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        if isinstance(data, tuple):
            if len(data) == 2:
                x, y = data
                sample_weight = None
            else:
                x, y, sample_weight = data
        else:
            raise TypeError("Target y must be supplied to fit in this model.")

        if sample_weight is not None:
            sample_weight_masked = tf.boolean_mask(
                tf.convert_to_tensor(sample_weight),
                tf.reduce_any(tf.not_equal(y, -1), axis=2),
            )
        else:
            sample_weight_masked = None

        reconstruction, z_mean, z_log_var, z = self(x, training=False)
        reconstruction_loss = self.compiled_loss(
            y,
            reconstruction,
            sample_weight=sample_weight_masked,
        )

        # Includes KL Divergence Loss.
        regularization_loss = sum(self.losses)

        total_loss = reconstruction_loss + regularization_loss

        self.accuracy_tracker.update_state(
            self.cateogrical_accuracy(
                y,
                reconstruction,
                sample_weight=sample_weight_masked,
            )
        )

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(regularization_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    @property
    def batch_size(self):
        """Batch (=step) size per epoch."""
        return self._batch_size

    @property
    def batch_idx(self):
        """Current batch (=step) index."""
        return self._batch_idx

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
