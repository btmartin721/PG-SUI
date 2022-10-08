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
    Activation,
    Flatten,
    BatchNormalization,
    LeakyReLU,
    PReLU,
)

from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import Sequential, Model
from tensorflow.keras import backend as K

# Custom Modules
try:
    from .neural_network_methods import NeuralNetworkMethods
except (ModuleNotFoundError, ValueError):
    from impute.neural_network_methods import NeuralNetworkMethods


class Sampling(tf.keras.layers.Layer):
    """Layer to calculate Z."""

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(Sampling, self).__init__(*args, **kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_sigma = tf.math.exp(0.5 * z_log_var)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + z_sigma * epsilon


class KLDivergenceLoss(tf.keras.layers.Layer):
    """Layer to calculate KL Divergence loss for VAE."""

    def __init__(self, *args, beta=1.0, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLoss, self).__init__(*args, **kwargs)
        self.beta = beta

    def call(self, inputs):
        z_mean, z_log_var = inputs

        kl_loss = self.beta * tf.reduce_mean(
            -0.5
            * tf.reduce_sum(
                z_log_var
                - tf.math.square(z_mean)
                - tf.math.exp(z_log_var)
                + 1,
                axis=-1,
            )
        )

        self.add_loss(kl_loss, inputs=inputs)
        return inputs


class Encoder(tf.keras.layers.Layer):
    """VAE encoder to Encode genotypes to (z_mean, z_log_var, z)."""

    def __init__(
        self,
        n_features,
        num_classes,
        latent_dim,
        hidden_layer_sizes,
        dropout_rate,
        activation,
        kernel_initializer,
        kernel_regularizer,
        beta=1.0,
        name="Encoder",
        **kwargs,
    ):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.beta = beta * latent_dim

        self.dense2 = None
        self.dense3 = None
        self.dense4 = None
        self.dense5 = None

        # for layer_size in hidden_layer_sizes:
        # self.dense_init = Dense(
        #     num_classes // 2,
        #     input_shape=(n_features, num_classes),
        #     activation=activation,
        #     kernel_initializer=kernel_initializer,
        #     kernel_regularizer=kernel_regularizer,
        #     name="Encoder1",
        # )

        # # n_features * num_classes.
        # self.flatten = Flatten()
        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = Dense(
            hidden_layer_sizes[0],
            input_shape=(n_features * num_classes,),
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="Encoder1",
        )

        if len(hidden_layer_sizes) >= 2:
            self.dense2 = Dense(
                hidden_layer_sizes[1],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="Encoder2",
            )

        if len(hidden_layer_sizes) >= 3:
            self.dense3 = Dense(
                hidden_layer_sizes[2],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="Encoder3",
            )

        if len(hidden_layer_sizes) >= 4:
            self.dense4 = Dense(
                hidden_layer_sizes[3],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="Encoder4",
            )

        if len(hidden_layer_sizes) == 5:
            self.dense5 = Dense(
                hidden_layer_sizes[4],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="Encoder5",
            )

        self.dense_z_mean = Dense(
            latent_dim,
            name="z_mean",
        )
        self.dense_z_log_var = Dense(
            latent_dim,
            name="z_log_var",
        )
        # # z_mean and z_log_var are inputs.
        self.sampling = Sampling(
            name="z",
        )

        self.kldivergence = KLDivergenceLoss(
            beta=self.beta, name="KLDivergence"
        )

        self.dense_latent = Dense(
            latent_dim,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="Encoder5",
        )

        self.dropout_layer = Dropout(dropout_rate)

        self.batch_norm_layer1 = BatchNormalization(center=False, scale=False)
        self.batch_norm_layer2 = BatchNormalization(center=False, scale=False)
        self.batch_norm_layer3 = BatchNormalization(center=False, scale=False)
        self.batch_norm_layer4 = BatchNormalization(center=False, scale=False)
        self.batch_norm_layer5 = BatchNormalization(center=False, scale=False)

    def call(self, inputs, training=None):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout_layer(x, training=training)
        # x = self.batch_norm_layer1(x, training=training)
        if self.dense2 is not None:
            x = self.dense2(x)
            x = self.dropout_layer(x, training=training)
            # x = self.batch_norm_layer2(x, training=training)
        if self.dense3 is not None:
            x = self.dense3(x)
            x = self.dropout_layer(x, training=training)
            # x = self.batch_norm_layer3(x, training=training)
        if self.dense4 is not None:
            x = self.dense4(x)
            x = self.dropout_layer(x, training=training)
            # x = self.batch_norm_layer4(x, training=training)
        if self.dense5 is not None:
            x = self.dense5(x)
            x = self.dropout_layer(x, training=training)
            # x = self.batch_norm_layer5(x, training=training)

        x = self.dense_latent(x)
        z_mean = self.dense_z_mean(x)
        z_log_var = self.dense_z_log_var(x)
        z = self.sampling([z_mean, z_log_var])
        z_mean, z_log_var = self.kldivergence([z_mean, z_log_var])

        return z_mean, z_log_var, z


class Decoder(tf.keras.layers.Layer):
    """Converts z, the encoded vector, back into the reconstructed output"""

    def __init__(
        self,
        n_features,
        num_classes,
        latent_dim,
        hidden_layer_sizes,
        dropout_rate,
        activation,
        kernel_initializer,
        kernel_regularizer,
        name="Decoder",
        **kwargs,
    ):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.dense2 = None
        self.dense3 = None
        self.dense4 = None
        self.dense5 = None

        self.dense1 = Dense(
            hidden_layer_sizes[0],
            input_shape=(latent_dim,),
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="Decoder1",
        )

        if len(hidden_layer_sizes) >= 2:
            self.dense2 = Dense(
                hidden_layer_sizes[1],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="Decoder2",
            )

        if len(hidden_layer_sizes) >= 3:
            self.dense3 = Dense(
                hidden_layer_sizes[2],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="Decoder3",
            )

        if len(hidden_layer_sizes) >= 4:
            self.dense4 = Dense(
                hidden_layer_sizes[3],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="Decoder4",
            )

        if len(hidden_layer_sizes) == 5:
            self.dense5 = Dense(
                hidden_layer_sizes[4],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="Decoder5",
            )

        # No activation for final layer.
        self.dense_output = Dense(
            n_features * num_classes,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="DecoderExpanded",
        )

        self.rshp = Reshape((n_features, num_classes))
        # self.dense_output = Dense(
        #     num_classes,
        #     kernel_initializer=kernel_initializer,
        #     kernel_regularizer=kernel_regularizer,
        #     activation="sigmoid",
        # )
        self.dropout_layer = Dropout(dropout_rate)

        self.batch_norm_layer1 = BatchNormalization(center=False, scale=False)
        self.batch_norm_layer2 = BatchNormalization(center=False, scale=False)
        self.batch_norm_layer3 = BatchNormalization(center=False, scale=False)
        self.batch_norm_layer4 = BatchNormalization(center=False, scale=False)
        self.batch_norm_layer5 = BatchNormalization(center=False, scale=False)

    def call(self, inputs, training=None):
        # x = self.flatten(inputs)
        x = self.dense1(inputs)
        x = self.dropout_layer(x, training=training)
        # x = self.batch_norm_layer1(x, training=training)
        if self.dense2 is not None:
            x = self.dense2(x)
            x = self.dropout_layer(x, training=training)
            # x = self.batch_norm_layer2(x, training=training)
        if self.dense3 is not None:
            x = self.dense3(x)
            x = self.dropout_layer(x, training=training)
            # x = self.batch_norm_layer3(x, training=training)
        if self.dense4 is not None:
            x = self.dense4(x)
            x = self.dropout_layer(x, training=training)
            # x = self.batch_norm_layer4(x, training=training)
        if self.dense5 is not None:
            x = self.dense5(x)
            x = self.dropout_layer(x, training=training)
            # x = self.batch_norm_layer5(x, training=training)

        x = self.dense_output(x)
        return self.rshp(x)
        # return self.dense_output(x)


class VAEModel(tf.keras.Model):
    def __init__(
        self,
        output_shape=None,
        n_components=3,
        weights_initializer="glorot_normal",
        hidden_layer_sizes="midpoint",
        num_hidden_layers=1,
        hidden_activation="elu",
        l1_penalty=1e-6,
        l2_penalty=1e-6,
        dropout_rate=0.2,
        kl_beta=1.0,
        num_classes=10,
        sample_weight=None,
    ):
        super(VAEModel, self).__init__()

        # self.kl_beta = K.variable(0.0)
        # self.kl_beta._trainable = False

        self.kl_beta = kl_beta
        self.sample_weight = sample_weight

        self.nn_ = NeuralNetworkMethods()
        self.categorical_accuracy = self.nn_.make_masked_categorical_accuracy(
            is_vae=True
        )

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.accuracy_tracker = tf.keras.metrics.Mean(name="accuracy")

        # y_train[1] dimension.
        self.n_features = output_shape

        self.n_components = n_components
        self.weights_initializer = weights_initializer
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        nn = NeuralNetworkMethods()

        hidden_layer_sizes = nn.validate_hidden_layers(
            self.hidden_layer_sizes, self.num_hidden_layers
        )

        hidden_layer_sizes = nn.get_hidden_layer_sizes(
            self.n_features, self.n_components, hidden_layer_sizes, vae=True
        )

        hidden_layer_sizes = [h * self.num_classes for h in hidden_layer_sizes]

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

        self.encoder = Encoder(
            self.n_features,
            self.num_classes,
            self.n_components,
            hidden_layer_sizes,
            self.dropout_rate,
            activation,
            kernel_initializer,
            kernel_regularizer,
            beta=self.kl_beta,
        )

        hidden_layer_sizes.reverse()

        self.decoder = Decoder(
            self.n_features,
            self.num_classes,
            self.n_components,
            hidden_layer_sizes,
            self.dropout_rate,
            activation,
            kernel_initializer,
            kernel_regularizer,
        )

    def call(self, inputs, training=None):
        """Call the model on a particular input.

        Args:
            input (tf.Tensor): Input tensor. Must be one-hot encoded.

        Returns:
            tf.Tensor: Output predictions. Will be one-hot encoded.
        """
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var, z

    def model(self):
        """Here so that mymodel.model().summary() can be called for debugging."""
        x = tf.keras.Input(shape=(self.n_features, self.num_classes))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def set_model_outputs(self):
        x = tf.keras.Input(shape=(self.n_features, self.num_classes))
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
        if isinstance(data, tuple):
            if len(data) == 2:
                x, y = data
                sample_weight = None
            else:
                x, y, sample_weight = data
        else:
            raise TypeError("Target y must be supplied to fit for this model.")

        with tf.GradientTape() as tape:
            reconstruction, z_mean, z_log_var, z = self(x, training=True)

            # Returns binary crossentropy loss.
            reconstruction_loss = self.compiled_loss(
                y,
                reconstruction,
                sample_weight=sample_weight,
            )

            # Doesn't include KL Divergence Loss.
            regularization_loss = sum(self.losses)

            total_loss = reconstruction_loss + regularization_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(regularization_loss)

        ### NOTE: If you get the error, "'tuple' object has no attribute
        ### 'rank', then convert y_true to a tensor object."
        # self.compiled_metrics.update_state(
        self.accuracy_tracker.update_state(
            self.categorical_accuracy(
                y,
                reconstruction,
                sample_weight=sample_weight,
            )
        )

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
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
