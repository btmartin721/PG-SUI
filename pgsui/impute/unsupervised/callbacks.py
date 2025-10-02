from snpio.utils.logging import LoggerManager


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
        min_epochs: int = 100,
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
        self.best_score = None
        self.early_stop = False
        self.best_model = None
        self.min_epochs = min_epochs

        is_verbose = verbose >= 2 or debug
        logman = LoggerManager(name=__name__, prefix=prefix, verbose=is_verbose)
        self.logger = logman.get_logger()

        # Define the comparison function for the monitored metric
        if mode == "min":
            self.monitor = lambda current, best: current < best - self.delta
        elif mode == "max":
            self.monitor = lambda current, best: current > best + self.delta
        else:
            msg = f"Invalid mode provided: '{mode}'. Use 'min' or 'max'."
            self.logger.error(msg)
            raise ValueError(msg)

    def __call__(self, score, model):
        """Checks if early stopping condition is met and checkpoints model accordingly.

        Args:
            score (float): The current metric value (e.g., validation loss/accuracy).
            model (torch.nn.Module): The model being trained.
        """
        # Increment the epoch count each time we call this function
        self.epoch_count += 1

        # If this is the first epoch, initialize best_score and save model
        if self.best_score is None:
            self.best_score = score
            return

        # Check if there is improvement
        if self.monitor(score, self.best_score):
            # If improved, reset counter and update the best score/model
            self.best_score = score
            self.best_model = model
            self.counter = 0
        else:
            # No improvement: increase counter
            self.counter += 1

            if self.verbose:
                self.logger.info(
                    f"EarlyStopping counter: {self.counter}/{self.patience}"
                )

            # Now check if we surpass patience AND have reached min_epochs
            if self.counter >= self.patience and self.epoch_count >= self.min_epochs:

                if self.best_model is None:
                    self.best_model = model

                self.early_stop = True

                if self.verbose:
                    self.logger.info(
                        f"Early stopping triggered at epoch {self.epoch_count}"
                    )
