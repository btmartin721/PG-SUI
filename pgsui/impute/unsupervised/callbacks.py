import logging
from pathlib import Path

import torch


class EarlyStopping:
    """Class to stop the training when a monitored metric has stopped improving.

    This class is used to stop the training of a model when a monitored metric has stopped improving. The monitored metric can be any metric that is being tracked during training, such as validation loss or accuracy. The class keeps track of the best score seen so far and saves the model when the monitored metric improves. If the monitored metric does not improve for a certain number of epochs, the training is stopped.

    Example:
    >>> early_stopping = EarlyStopping(patience=25, verbose=True)
    >>> for epoch in range(100):
    >>>     val_loss = train_epoch()
    >>>     early_stopping(val_loss, model)
    >>>     if early_stopping.early_stop:
    >>>         break

    Attributes:
        patience (int): Number of epochs to wait after the last time the monitored metric improved.
        delta (float): Minimum change in the monitored metric to qualify as an improvement.
        path (Path): pathlib.Path for saving the best model.
        verbose (int): If > 0, prints a message for each improvement. If > 1, prints a message for each epoch.
        mode (str): One of {"min", "max"}. Determines whether improvement is monitored as minimization or maximization.
        counter (int): Counter for the number of epochs since the monitored metric improved.
        best_score (float): Best score seen so far.
        early_stop (bool): Flag to stop training.
        val_loss_min (float): Minimum validation loss seen so far.
        val_acc_max (float): Maximum validation accuracy seen so far.
        best_model (torch.nn.Module): Best model seen so far.
        logger (logging.Logger): Logger object for logging messages.
        prefix (str): Prefix for the model checkpoint file.
        output_dir (Path): Directory for saving the model checkpoint file.
    """

    def __init__(
        self,
        patience: int = 25,
        delta: float = 0.0,
        path: Path = Path("checkpoint.pt"),
        verbose: int = 0,
        mode: str = "min",
        prefix: str = "pgsui",
        output_dir: Path = Path("output"),
        model_name: str = "model",
    ):
        """Initialize the EarlyStopping object.

        This class is used to stop the training of a model when a monitored metric has stopped improving. The monitored metric can be any metric that is being tracked during training, such as validation loss or accuracy. The class keeps track of the best score seen so far and saves the model when the monitored metric improves. If the monitored metric does not improve for a certain number of epochs, the training is stopped.

        Args:
            patience (int): Number of epochs to wait after the last time the monitored metric improved.
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
            path (Path): pathlib.Path for saving the best model.
            verbose (int): If > 0, prints a message for each improvement. If > 1, prints a message for each epoch.
            mode (str): One of {"min", "max"}. Determines whether improvement is monitored as minimization or maximization.
            prefix (str): Prefix for the model checkpoint file.
            output_dir (Path): Directory for saving the model checkpoint file.
            model_name (str): Name of the model being trained.

        Raises:
            ValueError: If mode is not one of {"min", "max"}.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.val_acc_max = 0
        self.best_model = None
        self.prefix = prefix
        self.output_dir = output_dir

        self.logger = logging.getLogger(__name__)

        if not isinstance(self.path, Path):
            self.path = Path(self.path)

        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

        self.path = (
            Path(f"{self.prefix}_{self.output_dir}")
            / "models"
            / "Unsupervised"
            / model_name
            / "checkpoint.pt"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Check to define whether improvement means lower (min) or higher (max)
        if mode == "min":
            self.monitor = lambda x, y: x < y - self.delta
        elif mode == "max":
            self.monitor = lambda x, y: x > y + self.delta
        else:
            msg = f"mode '{mode}' is unknown, use 'min' or 'max'"
            self.logger.error(msg)
            raise ValueError(msg)

    def __call__(self, score, model):
        """Checks if early stopping condition is met and updates model checkpoint.

        This method checks if the monitored metric has improved and updates the model checkpoint if it has. If the monitored metric has not improved for a certain number of epochs, the training is stopped.

        Args:
            score (float): The current value of the monitored metric (e.g., validation loss).
            model (torch.nn.Module): The model being trained.
        """

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif not self.monitor(score, self.best_score):
            self.counter += 1
            if self.verbose >= 2:
                self.logger.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Saves model when the monitored metric improves.

        This method saves the model to the path specified in the constructor when the monitored metric improves.

        Args:
            model (torch.nn.Module): The model being trained.
        """
        if self.verbose >= 2:
            self.logger.info(f"Metric improved. Saving model to {self.path}")
        torch.save(model.state_dict(), self.path)


# class UBPCallbacks(tf.keras.callbacks.Callback):
#     """Custom callbacks to use with subclassed NLPCA/ UBP Keras models.

#     Requires y, missing_mask, V_latent, and sample_weight to be input variables to be properties with setters in the subclassed model.
#     """

#     def __init__(self):
#         self.indices = None

#     def on_epoch_begin(self, epoch, logs=None):
#         """Shuffle input and target at start of epoch."""
#         y = self.model.y.copy()
#         missing_mask = self.model.missing_mask
#         sample_weight = self.model.sample_weight

#         n_samples = len(y)
#         self.indices = np.arange(n_samples)
#         np.random.shuffle(self.indices)

#         self.model.y = y[self.indices]
#         self.model.V_latent = self.model.V_latent[self.indices]
#         self.model.missing_mask = missing_mask[self.indices]

#         if sample_weight is not None:
#             self.model.sample_weight = sample_weight[self.indices]

#     def on_train_batch_begin(self, batch, logs=None):
#         """Get batch index."""
#         self.model.batch_idx = batch

#     def on_epoch_end(self, epoch, logs=None):
#         """Unsort the row indices."""
#         unshuffled = np.argsort(self.indices)

#         self.model.y = self.model.y[unshuffled]
#         self.model.V_latent = self.model.V_latent[unshuffled]
#         self.model.missing_mask = self.model.missing_mask[unshuffled]

#         if self.model.sample_weight is not None:
#             self.model.sample_weight = self.model.sample_weight[unshuffled]


# class UBPEarlyStopping(tf.keras.callbacks.Callback):
#     """Stop training when the loss is at its min, i.e. the loss stops decreasing.

#     Args:
#         patience (int, optional): Number of epochs to wait after min has been hit. After this
#         number of no improvement, training stops. Defaults to 0.

#         phase (int, optional): Current UBP Phase. Defaults to 3.
#     """

#     def __init__(self, patience=0, phase=3):
#         super(UBPEarlyStopping, self).__init__()
#         self.patience = patience
#         self.phase = phase

#         # best_weights to store the weights at which the minimum loss occurs.
#         self.best_weights = None

#         # In UBP, the input gets refined during training.
#         # So we have to revert it too.
#         self.best_input = None

#     def on_train_begin(self, logs=None):
#         # The number of epoch it has waited when loss is no longer minimum.
#         self.wait = 0
#         # The epoch the training stops at.
#         self.stopped_epoch = 0
#         # Initialize the best as infinity.
#         self.best = np.Inf

#     def on_epoch_end(self, epoch, logs=None):
#         current = logs.get("loss")
#         if np.less(current, self.best):
#             self.best = current
#             self.wait = 0
#             # Record the best weights if current results is better (less).
#             self.best_weights = self.model.get_weights()

#             if self.phase != 2:
#                 # Only refine input in phase 2.
#                 self.best_input = self.model.V_latent
#         else:
#             self.wait += 1
#             if self.wait >= self.patience:
#                 self.stopped_epoch = epoch
#                 self.model.stop_training = True
#                 self.model.set_weights(self.best_weights)

#                 if self.phase != 2:
#                     self.model.V_latent = self.best_input
