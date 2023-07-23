import logging
import os
import sys
import warnings

# Import tensorflow with reduced warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").disabled = True
warnings.filterwarnings("ignore", category=UserWarning)

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
    LeakyReLU,
    PReLU,
)

from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend as K

# Custom Modules
try:
    from ..neural_network_methods import NeuralNetworkMethods
except (ModuleNotFoundError, ValueError, ImportError):
    from impute.unsupervised.neural_network_methods import NeuralNetworkMethods


class Sampling(tf.keras.layers.Layer):
    """Layer to calculate Z to sample from latent dimension."""

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(Sampling, self).__init__(*args, **kwargs)

    def call(self, inputs):
        """Sampling during forward pass."""
        z_mean, z_log_var = inputs
        z_sigma = tf.math.exp(0.5 * z_log_var)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + z_sigma * epsilon


class Encoder(tf.keras.layers.Layer):
    """VAE encoder to Encode genotypes to (z_mean, z_log_var, z).

    Args:
        n_features (int): Number of featuresi in input dataset.

        num_classes (int): Number of classes in target data.

        latent_dim (int): Number of latent dimensions to use.

        hidden_layer_sizes (list of int): List of hidden layer sizes to use.

        dropout_rate (float): Dropout rate for Dropout layer.

        activation (str): Hidden activation function to use.

        kernel_initializer (str): Initializer to use for weights.

        kernel_regularizer (tf.keras.regularizers): L1 and/ or L2 objects.

        beta (float, optional): KL divergence beta to use. Defualts to 1.0.

        name (str): Name of model. Defaults to Encoder.

    """

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

        # # n_features * num_classes.
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
        # z_mean and z_log_var are inputs.
        self.sampling = Sampling(
            name="z",
        )

        self.dense_latent = Dense(
            latent_dim,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="Encoder5",
        )

        self.dropout_layer = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        """Forward pass for model."""
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout_layer(x, training=training)
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

        x = self.dense_latent(x)
        z_mean = self.dense_z_mean(x)
        z_log_var = self.dense_z_log_var(x)

        # Compute the KL divergence
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
        )
        # Add the KL divergence to the model's total loss
        self.add_loss(self.beta * tf.reduce_mean(kl_loss))

        z = self.sampling([z_mean, z_log_var])

        return z_mean, z_log_var, z


class Decoder(tf.keras.layers.Layer):
    """Converts z, the encoded vector, back into the reconstructed output.

    Args:
        n_features (int): Number of features in input dataset.

        num_classes (int): Number of classes in input dataset.

        latent_dim (int): Number of latent dimensions to use.

        hidden_layer_sizes (list of int): List of hidden layer sizes to use.

        dropout_rate (float): Dropout rate for Dropout layer.

        activation (str): Hidden activation function to use.

        kernel initializer (str): Function for initilizing weights.

        kernel_regularizer (tf.keras.regularizer): Initialized L1 and/ or L2 regularizer.

        name (str): Name of model. Defaults to "Decoder".
    """

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

        self.dropout_layer = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        """Forward pass for model."""
        x = self.dense1(inputs)
        x = self.dropout_layer(x, training=training)
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

        x = self.dense_output(x)
        return self.rshp(x)


class VAEModel(tf.keras.Model):
    """Variational Autoencoder model. Runs the encdoer and decoder and outputs the reconsruction.

    Args:
        output_shape (tuple): Shape of output. Defaults to None.

        n_components (int): Number of latent dimensions to use. Defaults to 3.

        weights_initializer (str, optional): kernel initializer to use. Defaults to "glorot_normal".

        hidden_layer_sizes (str or list, optional): List of hidden layer sizes to use, or can use "midpoint", "log2", or "sqrt". Defaults to "midpoint".

        num_hidden_layers (int, optional): Number of hidden layers to use. Defaults to 1.

        hidden_activation (str, optional): Hidden activation function to use. Defaults to "elu".

        l1_penalty (float, optional): L1 regularization penalty to use. Defaults to 1e-06.

        l2_penalty (float, optional): L2 regularization penalty. Defaults to 1e-06.

        dropout_rate (float, optional): Dropout rate to use for Dropout layers.

        kl_beta (float, optional): Beta to use for KL divergence loss. The KL divergence gets multiplied by this value. Defaults to 1.0 (no scaling).

        num_classes (int, optional): Number of classes in input data. Defaults to 10.

        sample_weight (list of float, optional): sample weights to use. Defaults to None.

        missing_mask (np.ndarray, optional): Missing mask to use in model for masking missing values. Defaults to None.

        batch_size (int, optional): Batch size to use for training. Defaults to 32.

        final_activation (str, optional): Final activation function to use. Should be either "sigmoid" (if doing multilabel classification) or "softmax" (if doing multiclass). If left None, then activation is not performed. Defaults to None.

        y (np.ndarray, optional): Input dataset y. Should be the full dataset. It will get subset by a callback to get batch_size samples. Default to None.

    """

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
        missing_mask=None,
        batch_size=32,
        final_activation=None,
        y=None,
    ):
        super(VAEModel, self).__init__()

        self.kl_beta = K.variable(0.0)
        self.kl_beta._trainable = False

        self._sample_weight = sample_weight
        self._missing_mask = missing_mask
        self._batch_idx = 0
        self._batch_size = batch_size
        self._y = y

        self._final_activation = final_activation
        if num_classes == 10 or num_classes == 3:
            self.acc_func = tf.keras.metrics.categorical_accuracy
        elif num_classes == 4:
            self.acc_func = tf.keras.metrics.binary_accuracy

        self.nn_ = NeuralNetworkMethods()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        # self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
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

        if final_activation is not None:
            self.act = Activation(final_activation)

    def call(self, inputs, training=None):
        """Forward pass for model."""
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        if self._final_activation is not None:
            reconstruction = self.act(reconstruction)
        return reconstruction

    def model(self):
        """Here so that mymodel.model().summary() can be called for debugging.
        :noindex:
        """
        x = tf.keras.Input(shape=(self.n_features, self.num_classes))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def set_model_outputs(self):
        """Set model output dimensions for building model.
        :noindex:
        """
        x = tf.keras.Input(shape=(self.n_features, self.num_classes))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        self.outputs = model.outputs

    @property
    def metrics(self):
        """Set metric trackers."""
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            # self.kl_loss_tracker,
            self.accuracy_tracker,
        ]

    @tf.function
    def train_step(self, data):
        y = self._y

        (
            y_true,
            sample_weight,
            missing_mask,
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
            tf.reduce_any(tf.not_equal(y_true, -1), axis=-1),
        )

        with tf.GradientTape() as tape:
            reconstruction = self(y_true, training=True)

            y_pred_masked = tf.boolean_mask(
                reconstruction,
                tf.reduce_any(tf.not_equal(y_true, -1), axis=-1),
            )

            # Returns binary crossentropy loss.
            reconstruction_loss = self.compiled_loss(
                y_true_masked,
                y_pred_masked,
                sample_weight=sample_weight_masked,
            )

            # Doesn't include KL Divergence Loss.
            regularization_loss = sum(self.losses)

            total_loss = reconstruction_loss + regularization_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        ### NOTE: If you get the error, "'tuple' object has no attribute
        ### 'rank', then convert y_true to a tensor object."
        # self.compiled_metrics.update_state(
        self.accuracy_tracker.update_state(
            self.acc_func(
                y_true_masked,
                y_pred_masked,
            )
        )

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    # @tf.function
    # def test_step(self, data):
    #     if isinstance(data, tuple):
    #         if len(data) == 2:
    #             x, y = data
    #             sample_weight = None
    #         else:
    #             x, y, sample_weight = data
    #     else:
    #         raise TypeError("Target y must be supplied to fit in this model.")

    #     if sample_weight is not None:
    #         sample_weight_masked = tf.boolean_mask(
    #             tf.convert_to_tensor(sample_weight),
    #             tf.reduce_any(tf.not_equal(y, -1), axis=2),
    #         )
    #     else:
    #         sample_weight_masked = None

    #     reconstruction, z_mean, z_log_var, z = self(x, training=False)
    #     reconstruction_loss = self.compiled_loss(
    #         y,
    #         reconstruction,
    #         sample_weight=sample_weight_masked,
    #     )

    #     # Includes KL Divergence Loss.
    #     regularization_loss = sum(self.losses)

    #     total_loss = reconstruction_loss + regularization_loss

    #     self.accuracy_tracker.update_state(
    #         self.cateogrical_accuracy(
    #             y,
    #             reconstruction,
    #             sample_weight=sample_weight_masked,
    #         )
    #     )

    #     self.total_loss_tracker.update_state(total_loss)
    #     self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    #     self.kl_loss_tracker.update_state(regularization_loss)

    #     return {
    #         "loss": self.total_loss_tracker.result(),
    #         "reconstruction_loss": self.reconstruction_loss_tracker.result(),
    #         "kl_loss": self.kl_loss_tracker.result(),
    #         "accuracy": self.accuracy_tracker.result(),
    #     }

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
