import numpy as np
from snpio.utils.logging import LoggerManager

from pgsui.utils.logging_utils import configure_logger


class EarlyStopping:
    """Class to stop the training when a monitored metric has stopped improving.

    This class is used to stop the training of a model when a monitored metric has stopped improving (such as validation loss or accuracy). If the metric does not improve for `patience` epochs, and we have already passed the `min_epochs` epoch threshold, training is halted. The best model checkpoint is reloaded when early stopping is triggered.

    Example:
        >>> early_stopping = EarlyStopping(patience=25, verbose=1, min_epochs=100)
        >>> for epoch in range(1, 1001):
        >>>     val_loss = train_epoch(...)
        >>>     early_stopping(val_loss, model)
        >>>     if early_stopping.early_stop:
        >>>         break
    """

    def __init__(
        self,
        patience: int = 25,
        delta: float = 0.0,
        verbose: int = 0,
        mode: str = "min",
        min_epochs: int = 150,
        prefix: str = "pgsui_output",
        debug: bool = False,
    ):
        """Early stopping callback for PyTorch training.

        This class is used to stop the training of a model when a monitored metric has stopped improving (such as validation loss or accuracy). If the metric does not improve for `patience` epochs, and we have already passed the `min_epochs` epoch threshold, training is halted. The best model checkpoint is reloaded when early stopping is triggered. The `mode` parameter can be set to "min" or "max" to indicate whether the metric should be minimized or maximized, respectively.

        Args:
            patience (int): Number of epochs to wait after the last time the monitored metric improved.
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
            verbose (int): Verbosity level (0 = silent, 1 = improvement messages, 2+ = more).
            mode (str): "min" or "max" to indicate how improvement is defined.
            prefix (str): Prefix for directory naming.
            output_dir (Path): Directory in which to create subfolders/checkpoints.
            min_epochs (int): Minimum epoch count before early stopping can take effect.
            debug (bool): Debug mode for logging messages

        Raises:
            ValueError: If an invalid mode is provided. Must be "min" or "max".
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose >= 2 or debug
        self.debug = debug
        self.mode = mode
        self.counter = 0
        self.epoch_count = 0
        self.best_score = float("inf") if mode == "min" else 0.0
        self.early_stop = False
        self.best_state_dict: dict | None = None
        self.min_epochs = min_epochs

        is_verbose = verbose >= 2 or debug
        logman = LoggerManager(name=__name__, prefix=prefix, verbose=is_verbose)
        self.logger = configure_logger(
            logman.get_logger(), verbose=is_verbose, debug=debug
        )

        # Define the comparison function for the monitored metric
        if mode == "min":
            self.monitor = lambda current, best: current < best - self.delta
        elif mode == "max":
            self.monitor = lambda current, best: current > best + self.delta
        else:
            msg = f"Invalid mode provided: '{mode}'. Use 'min' or 'max'."
            self.logger.error(msg)
            raise ValueError(msg)

    def __call__(self, score, model, *, epoch: int | None = None):
        """Update early stopping state.

        Args:
            score: Monitored metric value.
            model: Model to checkpoint.
            epoch: If provided, sets the internal epoch counter to the true epoch number.
        """
        if epoch is not None:
            self.epoch_count = int(epoch)
        else:
            self.epoch_count += 1

        # Treat non-finite scores as non-improvements
        try:
            score_f = float(score)
        except Exception:
            score_f = float("inf") if self.mode == "min" else float("-inf")

        if not np.isfinite(score_f):
            self.counter += 1
            if self.counter >= self.patience and self.epoch_count >= self.min_epochs:
                self.early_stop = True
            return

        if self.monitor(score_f, self.best_score):
            self.best_score = score_f
            # THIS is the real checkpoint:
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience and self.epoch_count >= self.min_epochs:
                self.early_stop = True
