import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2', '3'}

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Lambda
from tensorflow.keras.regularizers import l1_l2

# Custom Modules
try:
    from .neural_network_methods import NeuralNetworkMethods
except (ModuleNotFoundError, ValueError):
    from impute.neural_network_methods import NeuralNetworkMethods


class NLPCAModel(tf.keras.Model):
    """NLPCA model to train and use to predict imputations.

    NLPCAModel subclasses the tf.keras.Model and overrides the train_step() and test_step() functions, which do training and evaluation for each batch in each epoch.

    Args:
        V (numpy.ndarray(float)): V should have been randomly initialized and will be used as the input data that gets refined during training. Defaults to None.

        y (numpy.ndarray): Target values to predict. Actual input data. Defaults to None.

        batch_size (int, optional): Batch size per epoch. Defaults to 32.

        missing_mask (numpy.ndarray): Missing data mask for y. Defaults to None.

        output_shape (int): Output units for n_features dimension. Output will be of shape (batch_size, n_features). Defaults to None.

        weights_initializer (str, optional): Kernel initializer to use for initializing model weights. Defaults to "glorot_normal".

        hidden_layer_sizes (List[int], optional): Output units for each hidden layer. List should be of same length as the number of hidden layers. Defaults to "midpoint".

        num_hidden_layers (int, optional): Number of hidden layers to use. Defaults to 1.

        hidden_activation (str, optional): Activation function to use for hidden layers. Defaults to "elu".

        l1_penalty (float, optional): L1 regularization penalty to use to reduce overfitting. Defaults to 0.01.

        l2_penalty (float, optional): L2 regularization penalty to use to reduce overfitting. Defaults to 0.01.

        dropout_rate (float, optional): Dropout rate during training to reduce overfitting. Must be a float between 0 and 1. Defaults to 0.2.

        num_classes (int, optional): Number of classes in output. Corresponds to the 3rd dimension of the output shape (batch_size, n_features, num_classes). Defaults to 1.

        phase (NoneType): Here for compatibility with UBP.

    Methods:
        call: Does forward pass for model.
        train_step: Does training for one batch in a single epoch.
        test_step: Does evaluation for one batch in a single epoch.

    Attributes:
        V_latent_ (numpy.ndarray(float)): Randomly initialized input that gets refined during training to better predict the targets.

        hidden_layer_sizes (List[Union[int, str]]): Output units for each hidden layer. Length should be the same as the number of hidden layers.

        n_components (int): Number of principal components to use with _V.

        _batch_size (int): Batch size to use per epoch.

        _batch_idx (int): Index of current batch.

        _n_batches (int): Total number of batches per epoch.

        input_with_mask_ (numpy.ndarray): Target y with the missing data mask horizontally concatenated and shape (n_samples, n_features * 2). Gets refined in the UBPCallbacks() callback.

    Example:
        >>>model = NLPCAModel(V=V, y=y, batch_size=32, missing_mask=missing_mask, output_shape, n_components, weights_initializer, hidden_layer_sizes, num_hidden_layers, hidden_activation, l1_penalty, l2_penalty, dropout_rate, num_classes=3)
        >>>model.compile(optimizer=optimizer, loss=loss_func, metrics=[my_metrics], run_eagerly=True)
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
        num_classes=1,
        phase=None,
    ):
        super(NLPCAModel, self).__init__()

        nn = NeuralNetworkMethods()
        self.nn = nn

        if V is None:
            self._V = nn.init_weights(y.shape[0], n_components)
        else:
            self._V = V[n_components]

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

        ### NOTE: I tried using just _V as the input to be refined, but it
        # wasn't getting updated. So I copy it here and it works.
        # V_latent is refined during train_step().
        self.V_latent_ = self._V.copy()

        # Initialize parameters used during train_step() and test_step().
        # input_with_mask_ is set during the UBPCallbacks() execution.
        self._batch_idx = 0
        self._n_batches = 0
        self._batch_size = batch_size
        self.input_with_mask_ = None
        self.n_components = n_components

        if l1_penalty == 0.0 and l2_penalty == 0.0:
            kernel_regularizer = None
        else:
            kernel_regularizer = l1_l2(l1_penalty, l2_penalty)

        self.kernel_regularizer = kernel_regularizer
        kernel_initializer = weights_initializer

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
            activation=hidden_activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )

        if num_hidden_layers >= 2:
            self.dense2 = Dense(
                hidden_layer_sizes[1],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if num_hidden_layers >= 3:
            self.dense3 = Dense(
                hidden_layer_sizes[2],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if num_hidden_layers >= 4:
            self.dense4 = Dense(
                hidden_layer_sizes[3],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        if num_hidden_layers == 5:
            self.dense5 = Dense(
                hidden_layer_sizes[4],
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )

        self.output1 = Dense(
            output_shape,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            activation="sigmoid",
        )
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
        return self.output1(x)

    def model(self):
        """Here so that mymodel.model().summary() can be called for debugging."""
        x = tf.keras.Input(shape=(self.n_components,))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def set_model_outputs(self):
        x = tf.keras.Input(shape=(self.n_components,))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        # self.inputs = tf.convert_to_tensor(self.V_latent_)
        self.outputs = model.outputs

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
        # Set in the UBPCallbacks() callback.
        y = self.input_with_mask_

        v, y_true, batch_start, batch_end = self.nn.prepare_training_batches(
            self.V_latent_,
            y,
            self._batch_size,
            self._batch_idx,
            True,
            self.n_components,
        )

        src = [v]

        # NOTE: Earlier model architectures incorrectly
        # applied one gradient to all the variables, including
        # the weights and v. Here we apply them separately, per
        # the UBP manuscript.
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass. Watch input tensor v.
            tape.watch(v)
            y_pred = self(v, training=True)
            loss = self.compiled_loss(
                tf.convert_to_tensor(y_true, dtype=tf.float32),
                y_pred,
                regularization_losses=self.losses,
            )

        # Refine the watched variables with
        # gradient descent backpropagation
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Apply separate gradients to v.
        vgrad = tape.gradient(loss, src)
        self.optimizer.apply_gradients(zip(vgrad, src))

        del tape

        self.compiled_metrics.update_state(
            tf.convert_to_tensor(y_true, dtype=tf.float32),
            y_pred,
        )

        # NOTE: run_eagerly must be set to True in the compile() method for this
        # to work. Otherwise it can't convert a Tensor object to a numpy array.
        # There is really no other way to set v back to V_latent_ in graph
        # mode as far as I know. eager execution is slower, so it would be nice
        # to find a way to do this without converting to numpy.
        self.V_latent_[batch_start:batch_end, :] = v.numpy()

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
        y = self.input_with_mask_test_

        v, y_true, batch_start, batch_end = self.nn.prepare_training_batches(
            self.V_latent_,
            y,
            self._batch_size,
            self._batch_idx,
            False,
            self.n_components,
        )

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
            tf.convert_to_tensor(y_true, dtype=tf.float32),
            y_pred,
        )

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    @property
    def V_latent(self):
        """Randomly initialized input variable that gets refined during training."""
        return self.V_latent_

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

    @property
    def y(self):
        return self._y

    @property
    def missing_mask(self):
        return self._missing_mask

    @property
    def input_with_mask(self):
        return self.input_with_mask_

    @V_latent.setter
    def V_latent(self, value):
        """Set randomly initialized input variable. Gets refined during training."""
        self.V_latent_ = value

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

    @y.setter
    def y(self, value):
        """Set y after each epoch."""
        self._y = value

    @missing_mask.setter
    def missing_mask(self, value):
        """Set y after each epoch."""
        self._missing_mask = value

    @input_with_mask.setter
    def input_with_mask(self, value):
        self.input_with_mask_ = value
