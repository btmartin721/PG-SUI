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
    Flatten,
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
        beta=K.variable(0.0),
        name="Encoder",
        **kwargs,
    ):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.beta = beta

        self.dense2 = None
        self.dense3 = None
        self.dense4 = None
        self.dense5 = None

        # n_features * num_classes.
        self.flatten = Flatten()

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

        self.dense_latent = Dense(
            latent_dim,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="Encoder5",
        )

        self.dropout_layer = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        """Forward pass through model."""
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

        return self.dense_latent(x)


class Decoder(tf.keras.layers.Layer):
    """Converts the encoded vector back into the reconstructed output"""

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
            activation=None,
            name="Decoder6",
        )

        self.rshp = Reshape((n_features, num_classes))
        self.dropout_layer = Dropout(dropout_rate)

    def call(self, inputs, training=None):
        """Forward pass through model."""
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


class AutoEncoderModel(tf.keras.Model):
    """Standard AutoEncoder model to impute missing data.

    Args:
        y (np.ndarray): Full input data.

        batch_size (int, optional): Batch size to use with model. Defaults to 32.

        output_shape (int, optional): Number of features in output. Defaults to None.

        n_components (int, optional): Number of principal components to encode. Defaults to 3.

        weights_initializer (str, optional): tf.keras function to use with initial weights. Defaults to 'glorot_normal'.

        hidden_layer_sizes (str, List[int], or int, optional): Number of nodes to use in hidden layers. If List[int] is provided, must be equal in length to the number of hidden layers. If a string is provided, a calculation will be performed to automatically estimate the hidden layer sizes, with possible options including {'midpoint' or 'sqrt'}. If an integer is provided, then the provided integer will be used for all hidden layers. Defaults to 'midpoint'.

        num_hidden_layers (int, optional): Number of hidden layers to use in model construction. Maximum number of layers is 5. Defaults to 1.

        hidden_activation (str, optional): Hidden activation function to use in hidden layers. Possible options include: {"elu", "relu", "selu", "leaky_relu", and "prelu"}. Defaults to "elu".

        l1_penalty (float, optional): l1_penalty to use for regularization. Defaults to 1e-6.

        l2_penalty (float, optional): l2_penalty to use fo regularization. Defaults to 1e-6.

        dropout_rate (float, optional): Dropout rate to use for Dropout() layer. Defaults to 0.2.

        sample_weight (numpy.ndarray, optional): Sample weight matrix for weighting class imbalance. Should be of shape (n_samples, n_features). Defaults to None.

        num_classes (int, optional): Number of classes in multiclass predictions. Defaults to 3.

    Raises:
        ValueError: Maximum number of hidden layers (5) was exceeded.
    """

    def __init__(
        self,
        y,
        batch_size=32,
        output_shape=None,
        n_components=3,
        weights_initializer="glorot_normal",
        hidden_layer_sizes="midpoint",
        num_hidden_layers=1,
        hidden_activation="elu",
        l1_penalty=1e-6,
        l2_penalty=1e-6,
        dropout_rate=0.2,
        sample_weight=None,
        missing_mask=None,
        num_classes=3,
    ):
        super(AutoEncoderModel, self).__init__()

        self.nn_ = NeuralNetworkMethods()
        self.categorical_accuracy = self.nn_.make_masked_categorical_accuracy()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.accuracy_tracker = tf.keras.metrics.Mean(name="accuracy")

        self._y = y
        self._batch_idx = 0
        self._batch_size = batch_size
        self._sample_weight = sample_weight
        self._missing_mask = missing_mask

        # y_train[1] dimension.
        self.n_features = output_shape
        n_features = self.n_features

        self.n_components = n_components
        self.weights_initializer = weights_initializer
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate
        self.sample_weight = sample_weight
        self.num_classes = num_classes

        nn = NeuralNetworkMethods()

        hidden_layer_sizes = nn.validate_hidden_layers(
            self.hidden_layer_sizes, self.num_hidden_layers
        )

        hidden_layer_sizes = nn.get_hidden_layer_sizes(
            n_features, self.n_components, hidden_layer_sizes, vae=True
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
            n_features,
            self.num_classes,
            self.n_components,
            hidden_layer_sizes,
            self.dropout_rate,
            activation,
            kernel_initializer,
            kernel_regularizer,
        )

        hidden_layer_sizes.reverse()

        self.decoder = Decoder(
            n_features,
            self.num_classes,
            self.n_components,
            hidden_layer_sizes,
            self.dropout_rate,
            activation,
            kernel_initializer,
            kernel_regularizer,
        )

    def call(self, inputs, training=None):
        """Forward pass through model."""
        x = self.encoder(inputs)
        return self.decoder(x)

    def model(self):
        """To allow model.summary().summar() to be called."""
        x = tf.keras.Input(shape=(self.n_features, self.num_classes))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def set_model_outputs(self):
        """Set expected model outputs."""
        x = tf.keras.Input(shape=(self.n_features, self.num_classes))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        self.outputs = model.outputs

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
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
            tf.reduce_any(tf.not_equal(y_true, -1), axis=2),
        )

        with tf.GradientTape() as tape:
            reconstruction = self(y_true, training=True)

            y_pred_masked = tf.boolean_mask(
                reconstruction, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
            )

            # Returns binary crossentropy loss.
            reconstruction_loss = self.compiled_loss(
                y_true_masked,
                y_pred_masked,
                sample_weight=sample_weight_masked,
            )

            regularization_loss = sum(self.losses)

            total_loss = reconstruction_loss + regularization_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        ### NOTE: If you get the error, "'tuple' object has no attribute
        ### 'rank', then convert y_true to a tensor object."
        # self.compiled_metrics.update_state(
        self.accuracy_tracker.update_state(
            self.categorical_accuracy(
                y_true_masked,
                y_pred_masked,
                sample_weight=sample_weight_masked,
            )
        )

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        """Custom evaluation loop for one step (=batch) in a single epoch.

        This function will evaluate on a batch of samples (rows), which can be adjusted with the ``batch_size`` parameter from the estimator.

        Args:
            data (Tuple[tf.EagerTensor, tf.EagerTensor]): Input tensorflow tensors of shape (batch_size, n_components) and (batch_size, n_features, num_classes).

        Returns:
            Dict[str, float]: History object that gets returned from fit(). Contains the loss and any metrics specified in compile().
        """
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
            tf.reduce_any(tf.not_equal(y_true, -1), axis=2),
        )

        reconstruction = self(y_true, training=False)

        y_pred_masked = tf.boolean_mask(
            reconstruction, tf.reduce_any(tf.not_equal(y_true, -1), axis=2)
        )

        reconstruction_loss = self.compiled_loss(
            y_true_masked,
            y_pred_masked,
            sample_weight=sample_weight_masked,
        )

        regularization_loss = sum(self.losses)

        total_loss = reconstruction_loss + regularization_loss

        self.accuracy_tracker.update_state(
            self.categorical_accuracy(
                y_true_masked,
                y_pred_masked,
                sample_weight=sample_weight_masked,
            )
        )

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    @property
    def batch_size(self):
        """Batch (=step) size per epoch.
        :noindex:
        """
        return self._batch_size

    @property
    def batch_idx(self):
        """Current batch (=step) index.
        :noindex:
        """
        return self._batch_idx

    @property
    def y(self):
        """Full input dataset.
        :noindex:
        """
        return self._y

    @property
    def missing_mask(self):
        """Missing mask of shape (y.shape[0], y.shape[1])
        :noindex:
        """
        return self._missing_mask

    @property
    def sample_weight(self):
        """Sample weights of shape (y.shape[0], y.shape[1])
        :noindex:
        """
        return self._sample_weight

    @batch_size.setter
    def batch_size(self, value):
        """Set batch_size parameter.
        :noindex:
        """
        self._batch_size = int(value)

    @batch_idx.setter
    def batch_idx(self, value):
        """Set current batch (=step) index.
        :noindex:
        """
        self._batch_idx = int(value)

    @y.setter
    def y(self, value):
        """Set y after each epoch.
        :noindex:
        """
        self._y = value

    @missing_mask.setter
    def missing_mask(self, value):
        """Set missing_mask after each epoch.
        :noindex:
        """
        self._missing_mask = value

    @sample_weight.setter
    def sample_weight(self, value):
        """Set sample_weight after each epoch.
        :noindex:
        """
        self._sample_weight = value
