import logging
import os
import sys
import warnings

# Import tensorflow with reduced warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").disabled = True
warnings.filterwarnings("ignore", category=UserWarning)

# noinspection PyPackageRequirements
import tensorflow as tf

# Disable can't find cuda .dll errors. Also turns off GPU support.
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
    from ..neural_network_methods import NeuralNetworkMethods
except (ModuleNotFoundError, ValueError, ImportError):
    from impute.unsupervised.neural_network_methods import NeuralNetworkMethods


class NLPCAModel(tf.keras.Model):
    """NLPCA model to train and use to predict imputations.

    NLPCAModel subclasses the tf.keras.Model and overrides the train_step function, which does training and evaluation for each batch in each epoch.

    Args:
        V (numpy.ndarray(float)): V should have been randomly initialized and will be used as the input data that gets refined during training. Defaults to None.

        y (numpy.ndarray): Target values to predict. Actual input data. Defaults to None.

        batch_size (int, optional): Batch size per epoch. Defaults to 32.

        missing_mask (numpy.ndarray): Missing data mask for y. Defaults to None.

        output_shape (int): Output units for n_features dimension. Output will be of shape (batch_size, n_features). Defaults to None.

        n_components (int, optional): Number of features in input V to use. Defaults to 3.

        weights_initializer (str, optional): Kernel initializer to use for initializing model weights. Defaults to "glorot_normal".

        hidden_layer_sizes (List[int], optional): Output units for each hidden layer. List should be of same length as the number of hidden layers. Defaults to "midpoint".

        num_hidden_layers (int, optional): Number of hidden layers to use. Defaults to 1.

        hidden_activation (str, optional): Activation function to use for hidden layers. Defaults to "elu".

        l1_penalty (float, optional): L1 regularization penalty to use to reduce overfitting. Defaults to 0.01.

        l2_penalty (float, optional): L2 regularization penalty to use to reduce overfitting. Defaults to 0.01.

        dropout_rate (float, optional): Dropout rate during training to reduce overfitting. Must be a float between 0 and 1. Defaults to 0.2.

        num_classes (int, optional): Number of classes in output. Corresponds to the 3rd dimension of the output shape (batch_size, n_features, num_classes). Defaults to 1.

        phase (NoneType): Here for compatibility with UBP.

        sample_weight (numpy.ndarray, optional): 2D sample weights of shape (n_samples, n_features). Should have values for each class weighted. Defaults to None.

    Example:
        >>>model = NLPCAModel(V=V, y=y, batch_size=32, missing_mask=missing_mask, output_shape, n_components, weights_initializer, hidden_layer_sizes, num_hidden_layers, hidden_activation, l1_penalty, l2_penalty, dropout_rate, num_classes=3)
        >>>
        >>>model.compile(optimizer=optimizer, loss=loss_func, metrics=[my_metrics], run_eagerly=True)
        >>>
        >>>history = model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[MyCallback()], validation_split=validation_split, shuffle=False)

    Raises:
        TypeError: V, y, missing_mask, output_shape must not be NoneType.
        ValueError: Maximum of 5 hidden layers.

    """

    def __init__(
        self,
        V=None,
        y=None,
        batch_size=32,
        missing_mask=None,
        output_shape=None,
        n_components=3,
        weights_initializer="glorot_normal",
        hidden_layer_sizes="midpoint",
        num_hidden_layers=1,
        hidden_activation="elu",
        l1_penalty=0.01,
        l2_penalty=0.01,
        dropout_rate=0.2,
        num_classes=3,
        phase=None,
        sample_weight=None,
    ):
        super(NLPCAModel, self).__init__()

        nn = NeuralNetworkMethods()
        self.nn = nn

        if V is None:
            self._V = nn.init_weights(y.shape[0], n_components)
        elif isinstance(V, dict):
            self._V = V[n_components]
        else:
            self._V = V

        self._y = y

        hidden_layer_sizes = nn.validate_hidden_layers(
            hidden_layer_sizes, num_hidden_layers
        )

        hidden_layer_sizes = nn.get_hidden_layer_sizes(
            y.shape[1], self._V.shape[1], hidden_layer_sizes
        )

        nn.validate_model_inputs(y, missing_mask, output_shape)

        self._missing_mask = missing_mask
        self.weights_initializer = weights_initializer
        self.phase = phase
        self.dropout_rate = dropout_rate
        self._sample_weight = sample_weight

        ### NOTE: I tried using just _V as the input to be refined, but it
        # wasn't getting updated. So I copy it here and it works.
        # V_latent is refined during train_step.
        self.V_latent_ = self._V.copy()

        # Initialize parameters used during train_step.
        self._batch_idx = 0
        self._batch_size = batch_size
        self.n_components = n_components

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

        # Construct multi-layer perceptron.
        # Add hidden layers dynamically.
        self.dense1 = Dense(
            hidden_layer_sizes[0],
            input_shape=(n_components,),
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

        self.output1 = Dense(
            output_shape * num_classes,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )

        self.rshp = Reshape((output_shape, num_classes))

        self.dropout_layer = Dropout(rate=dropout_rate)

    def call(self, inputs, training=None):
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

        x = self.output1(x)
        return self.rshp(x)

    def model(self):
        x = tf.keras.Input(shape=(self.n_components,))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def set_model_outputs(self):
        x = tf.keras.Input(shape=(self.n_components,))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        self.outputs = model.outputs

    def train_step(self, data):
        """Train step function. Parameters are set in UBPCallbacks callback."""
        y = self._y

        (
            v,
            y_true,
            sample_weight,
            missing_mask,
            batch_start,
            batch_end,
        ) = self.nn.prepare_training_batches(
            self.V_latent_,
            y,
            self._batch_size,
            self._batch_idx,
            True,
            self.n_components,
            self._sample_weight,
            self._missing_mask,
        )

        src = [v]

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

        # NOTE: Earlier model architectures incorrectly
        # applied one gradient to all the variables, including
        # the weights and v. Here we apply them separately, per
        # the UBP manuscript.
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass. Watch input tensor v.
            tape.watch(v)
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

        # Refine the watched variables with
        # gradient descent backpropagation
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

        # Apply separate gradients to v.
        vgrad = tape.gradient(loss, src)
        self.optimizer.apply_gradients(zip(vgrad, src))

        del tape

        ### NOTE: If you get the error, "'tuple' object has no attribute
        ### 'rank', then convert y_true to a tensor object."
        self.compiled_metrics.update_state(
            y_true_masked,
            y_pred_masked,
            sample_weight=sample_weight_masked,
        )

        # NOTE: run_eagerly must be set to True in the compile() method for this
        # to work. Otherwise it can't convert a Tensor object to a numpy array.
        # There is really no other way to set v back to V_latent_ in graph
        # mode as far as I know. eager execution is slower, so it would be nice
        # to find a way to do this without converting to numpy.
        self.V_latent_[batch_start:batch_end, :] = v.numpy()

        # history object that gets returned from fit().
        return {m.name: m.result() for m in self.metrics}

    @property
    def V_latent(self):
        """Randomly initialized input that gets refined during training.
        :noindex:
        """
        return self.V_latent_

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
        """Full dataset y.
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

    @V_latent.setter
    def V_latent(self, value):
        """Set randomly initialized input. Refined during training.
        :noindex:
        """
        self.V_latent_ = value

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
