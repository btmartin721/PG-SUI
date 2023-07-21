import math
import sys

import numpy as np
import tensorflow as tf


class CyclicalAnnealingCallback(tf.keras.callbacks.Callback):
    """Perform cyclical annealing with KL Divergence weights.

    The dynamically changing weight (beta) is multiplied with the KL Divergence loss.

    This process is supposed to improve the latent distribution sampling for the variational autoencoder model and eliminate the KL vanishing issue.

    Three types of cycle curves can be used that determine how the weight increases: 'linear', 'sigmoid', and 'cosine'..

    Code is adapted from: https://github.com/haofuml/cyclical_annealing

    The cyclical annealing process was first described in the following paper: https://aclanthology.org/N19-1021.pdf

    Args:
        n_iter (int): Number of iterations (epochs) being used in training.
        start (float, optional): Where to start cycles. Defaults to 0.0.
        stop (float, optional): Where to stop cycles. Defaults to 1.0.
        n_cycle (int, optional): How many cycles to use across all the epochs. Defaults to 4.
        ratio (float, optional): Ratio to determine proportion used to increase beta. Defaults to 0.5.
        schedule_type (str, optional): Type of curve to use for scheduler. Possible options include: 'linear', 'sigmoid', or 'cosine'. Defaults to 'linear'.

    """

    def __init__(
        self,
        n_iter,
        start=0.0,
        stop=1.0,
        n_cycle=4,
        ratio=0.5,
        schedule_type="linear",
    ):
        self.n_iter = n_iter
        self.start = start
        self.stop = stop
        self.n_cycle = n_cycle
        self.ratio = ratio
        self.schedule_type = schedule_type

        self.arr = None

    def on_train_begin(self, logs=None):
        """Executes on training begin.

        Here, the cycle curve is generated and stored as a class variable.
        """
        if self.schedule_type == "linear":
            cycle_func = self._linear_cycle_range
        elif self.schedule_type == "sigmoid":
            cycle_func = self._sigmoid_cycle_range
        elif self.schedule_type == "cosine":
            cycle_func = self._cosine_cycle_range
        else:
            raise ValueError(
                f"Invalid schedule_type value provided: {self.schedule_type}"
            )

        self.arr = cycle_func()

    def on_epoch_begin(self, epoch, logs=None):
        """Executes each time an epoch begins.

        Here, the new kl_beta weight is set.

        Args:
            epoch (int): Current epoch iteration.
            logs (None, optional): For compatibility. Not used. Defaults to None.
        """
        idx = epoch - 1
        new_weight = self.arr[idx]

        tf.keras.backend.set_value(self.model.kl_beta, new_weight)

    def _linear_cycle_range(self):
        """Get an array with a linear cycle curve ranging from 0 to 1 for n_iter epochs.

        The amount of time cycling and spent at 1.0 is determined by the ratio variable.

        Returns:
            numpy.ndarray: Linear cycle range.
        """
        L = np.ones(self.n_iter) * self.stop
        period = self.n_iter / self.n_cycle

        # Linear schedule
        step = (self.stop - self.start) / (
            period * self.ratio
        )  # linear schedule

        for c in range(self.n_cycle):
            v, i = self.start, 0
            while v <= self.stop and (int(i + c * period) < self.n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1

        return L

    def _sigmoid_cycle_range(self):
        """Get sigmoidal curve cycle ranging from 0 to 1 for n_iter epochs.

        The amount of time cycling and spent at 1.0 is determined by the ratio variable.

        Returns:
            numpy.ndarray: Sigmoidal cycle range.
        """
        L = np.ones(self.n_iter)
        period = self.n_iter / self.n_cycle
        step = (self.stop - self.start) / (
            period * self.ratio
        )  # step is in [0,1]

        for c in range(self.n_cycle):
            v, i = self.start, 0

            while v <= self.stop:
                L[int(i + c * period)] = 1.0 / (
                    1.0 + np.exp(-(v * 12.0 - 6.0))
                )
                v += step
                i += 1
        return L

    def _cosine_cycle_range(self):
        """Get cosine curve cycle ranging from 0 to 1 for n_iter epochs.

        The amount of time cycling and spent at 1.0 is determined by the ratio variable.

        Returns:
            numpy.ndarray: Cosine cycle range.
        """
        L = np.ones(self.n_iter)
        period = self.n_iter / self.n_cycle
        step = (self.stop - self.start) / (
            period * self.ratio
        )  # step is in [0,1]

        for c in range(self.n_cycle):
            v, i = self.start, 0

            while v <= self.stop:
                L[int(i + c * period)] = 0.5 - 0.5 * math.cos(v * math.pi)
                v += step
                i += 1
        return L


class VAECallbacks(tf.keras.callbacks.Callback):
    """Custom callbacks to use with subclassed VAE Keras model.

    Requires y, missing_mask, and sample_weight to be input variables to be properties with setters in the subclassed model.
    """

    def __init__(self):
        self.indices = None

    def on_epoch_begin(self, epoch, logs=None):
        """Shuffle input and target at start of epoch."""
        y = self.model.y.copy()
        missing_mask = self.model.missing_mask
        sample_weight = self.model.sample_weight

        n_samples = len(y)
        self.indices = np.arange(n_samples)
        np.random.shuffle(self.indices)

        self.model.y = y[self.indices]
        self.model.missing_mask = missing_mask[self.indices]

        if sample_weight is not None:
            self.model.sample_weight = sample_weight[self.indices]

    def on_train_batch_begin(self, batch, logs=None):
        """Get batch index."""
        self.model.batch_idx = batch

    def on_epoch_end(self, epoch, logs=None):
        """Unsort the row indices."""
        unshuffled = np.argsort(self.indices)

        self.model.y = self.model.y[unshuffled]
        self.model.missing_mask = self.model.missing_mask[unshuffled]

        if self.model.sample_weight is not None:
            self.model.sample_weight = self.model.sample_weight[unshuffled]


class UBPCallbacks(tf.keras.callbacks.Callback):
    """Custom callbacks to use with subclassed NLPCA/ UBP Keras models.

    Requires y, missing_mask, V_latent, and sample_weight to be input variables to be properties with setters in the subclassed model.
    """

    def __init__(self):
        self.indices = None

    def on_epoch_begin(self, epoch, logs=None):
        """Shuffle input and target at start of epoch."""
        y = self.model.y.copy()
        missing_mask = self.model.missing_mask
        sample_weight = self.model.sample_weight

        n_samples = len(y)
        self.indices = np.arange(n_samples)
        np.random.shuffle(self.indices)

        self.model.y = y[self.indices]
        self.model.V_latent = self.model.V_latent[self.indices]
        self.model.missing_mask = missing_mask[self.indices]

        if sample_weight is not None:
            self.model.sample_weight = sample_weight[self.indices]

    def on_train_batch_begin(self, batch, logs=None):
        """Get batch index."""
        self.model.batch_idx = batch

    def on_epoch_end(self, epoch, logs=None):
        """Unsort the row indices."""
        unshuffled = np.argsort(self.indices)

        self.model.y = self.model.y[unshuffled]
        self.model.V_latent = self.model.V_latent[unshuffled]
        self.model.missing_mask = self.model.missing_mask[unshuffled]

        if self.model.sample_weight is not None:
            self.model.sample_weight = self.model.sample_weight[unshuffled]


class UBPEarlyStopping(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Args:
        patience (int, optional): Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops. Defaults to 0.

        phase (int, optional): Current UBP Phase. Defaults to 3.
    """

    def __init__(self, patience=0, phase=3):
        super(UBPEarlyStopping, self).__init__()
        self.patience = patience
        self.phase = phase

        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

        # In UBP, the input gets refined during training.
        # So we have to revert it too.
        self.best_input = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()

            if self.phase != 2:
                # Only refine input in phase 2.
                self.best_input = self.model.V_latent
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

                if self.phase != 2:
                    self.model.V_latent = self.best_input
