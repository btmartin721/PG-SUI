from typing import TYPE_CHECKING, Dict, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from snpio.utils.logging import LoggerManager
from torch import Tensor

from pgsui.utils.logging_utils import configure_logger
from pgsui.utils.misc import validate_input_type


class Scorer:
    """Class for evaluating the performance of a model using various metrics.

    This module provides a unified interface for computing common evaluation metrics. It supports accuracy, F1 score, precision, recall, ROC AUC, average precision, and macro-average precision. The class can handle both raw and one-hot encoded labels and includes options for logging and averaging methods.
    """

    def __init__(
        self,
        prefix: str,
        average: Literal["weighted", "macro", "micro"] = "macro",
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize a Scorer object.

        This class provides a unified interface for computing common evaluation metrics. It supports accuracy, F1 score, precision, recall, ROC AUC, average precision, and macro-average precision. The class can handle both raw and one-hot encoded labels and includes options for logging and averaging methods.

        Args:
            prefix (str): The prefix to use for logging.
            average (Literal["weighted", "macro", "micro"]): The averaging method to use for metrics. Must be one of 'micro', 'macro', or 'weighted'. Defaults to 'weighted'.
            verbose (bool): If True, enable verbose logging. Defaults to False.
            debug (bool): If True, enable debug logging. Defaults to False.
        """
        logman = LoggerManager(
            name=__name__, prefix=prefix, debug=debug, verbose=verbose
        )
        self.logger = configure_logger(
            logman.get_logger(), verbose=verbose, debug=debug
        )

        if average not in {"weighted", "micro", "macro"}:
            msg = f"Invalid average parameter: {average}. Must be one of 'micro', 'macro', or 'weighted'."
            self.logger.error(msg)
            raise ValueError(msg)

        self.average: Literal["micro", "macro", "weighted"] = average

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the accuracy score.

        Args:
            y_true (np.ndarray): Ground truth (correct) target values.
            y_pred (np.ndarray): Estimated target values.

        Returns:
            float: The accuracy score.
        """
        return float(accuracy_score(y_true, y_pred))

    def f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the F1 score.

        Args:
            y_true (np.ndarray): Ground truth (correct) target values.
            y_pred (np.ndarray): Estimated target values.

        Returns:
            float: The F1 score.
        """
        return float(f1_score(y_true, y_pred, average=self.average, zero_division=0))

    def precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the precision score.

        Args:
            y_true (np.ndarray): Ground truth (correct) target values.
            y_pred (np.ndarray): Estimated target values.

        Returns:
            float: The precision score.
        """
        return float(
            precision_score(y_true, y_pred, average=self.average, zero_division=0)
        )

    def recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the recall score.

        Args:
            y_true (np.ndarray): Ground truth (correct) target values.
            y_pred (np.ndarray): Estimated target values.

        Returns:
            float: The recall score.
        """
        return float(
            recall_score(y_true, y_pred, average=self.average, zero_division=0)
        )

    def roc_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute the ROC AUC score.

        Args:
            y_true (np.ndarray): Ground truth (correct) target values.
            y_pred_proba (np.ndarray): Predicted probabilities.

        Returns:
            float: The ROC AUC score.
        """
        if len(np.unique(y_true)) < 2:
            return 0.5

        if y_pred_proba.shape[-1] == 2:
            # Binary classification case
            # Use probabilities for the positive class
            # Otherwise it throws an error.
            y_pred_proba = y_pred_proba[:, 1]

        try:
            return float(
                roc_auc_score(
                    y_true, y_pred_proba, average=self.average, multi_class="ovr"
                )
            )
        except Exception:
            return float(roc_auc_score(y_true, y_pred_proba, average=self.average))

    # This method now correctly expects one-hot encoded true labels
    def average_precision(
        self, y_true_ohe: np.ndarray, y_pred_proba: np.ndarray
    ) -> float:
        """Compute the average precision score.

        Args:
            y_true_ohe (np.ndarray): One-hot encoded ground truth target values.
            y_pred_proba (np.ndarray): Predicted probabilities.

        Returns:
            float: The average precision score.
        """
        if y_pred_proba.shape[-1] == 2:
            # Binary classification case
            # Use probabilities for the positive class
            y_pred_proba = y_pred_proba[:, 1]

        if y_true_ohe.shape[1] == 2:
            # Binary classification case
            y_true_ohe = y_true_ohe[:, 1]

        return float(
            average_precision_score(y_true_ohe, y_pred_proba, average=self.average)
        )

    def pr_macro(self, y_true_ohe: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute the macro-average precision score.

        Args:
            y_true_ohe (np.ndarray): One-hot encoded ground truth target values.
            y_pred_proba (np.ndarray): Predicted probabilities.

        Returns:
            float: The macro-average precision score.
        """
        if y_pred_proba.shape[-1] == 2:
            # Binary classification case
            # Use probabilities for the positive class
            y_pred_proba = y_pred_proba[:, 1]

        if y_true_ohe.shape[1] == 2:
            # Binary classification case
            y_true_ohe = y_true_ohe[:, 1]

        return float(average_precision_score(y_true_ohe, y_pred_proba, average="macro"))

    def evaluate(
        self,
        y_true: pd.DataFrame | np.ndarray | Tensor | list,
        y_pred: pd.DataFrame | np.ndarray | Tensor | list,
        y_true_ohe: pd.DataFrame | np.ndarray | Tensor | list,
        y_pred_proba: pd.DataFrame | np.ndarray | Tensor | list,
        objective_mode: bool = False,
        tune_metric: Literal[
            "pr_macro",
            "roc_auc",
            "average_precision",
            "accuracy",
            "f1",
            "precision",
            "recall",
        ] = "pr_macro",
    ) -> Dict[str, float]:
        """Evaluate the model using various metrics.

        Args:
            y_true: Ground truth (correct) target values.
            y_pred: Estimated target values.
            y_true_ohe: One-hot encoded ground truth target values.
            y_pred_proba: Predicted probabilities.
            objective_mode: If True, only compute the metric specified by ``tune_metric``. Defaults to False.
            tune_metric: The metric to optimize during tuning. Defaults to "pr_macro".
        """
        y_true, y_pred, y_true_ohe, y_pred_proba = [
            validate_input_type(x) for x in (y_true, y_pred, y_true_ohe, y_pred_proba)
        ]

        if objective_mode:
            metric_calculators = {
                "pr_macro": lambda: self.pr_macro(
                    np.asarray(y_true_ohe), np.asarray(y_pred_proba)
                ),
                "roc_auc": lambda: self.roc_auc(
                    np.asarray(y_true), np.asarray(y_pred_proba)
                ),
                "average_precision": lambda: self.average_precision(
                    np.asarray(y_true_ohe), np.asarray(y_pred_proba)
                ),
                "accuracy": lambda: self.accuracy(
                    np.asarray(y_true), np.asarray(y_pred)
                ),
                "f1": lambda: self.f1(np.asarray(y_true), np.asarray(y_pred)),
                "precision": lambda: self.precision(
                    np.asarray(y_true), np.asarray(y_pred)
                ),
                "recall": lambda: self.recall(np.asarray(y_true), np.asarray(y_pred)),
            }
            if tune_metric not in metric_calculators:
                msg = f"Invalid tune_metric provided: '{tune_metric}'."
                self.logger.error(msg)
                raise ValueError(msg)

            metrics = {tune_metric: metric_calculators[tune_metric]()}
        else:
            metrics = {
                "accuracy": self.accuracy(np.asarray(y_true), np.asarray(y_pred)),
                "f1": self.f1(np.asarray(y_true), np.asarray(y_pred)),
                "precision": self.precision(np.asarray(y_true), np.asarray(y_pred)),
                "recall": self.recall(np.asarray(y_true), np.asarray(y_pred)),
                "roc_auc": self.roc_auc(np.asarray(y_true), np.asarray(y_pred_proba)),
                "average_precision": self.average_precision(
                    np.asarray(y_true_ohe), np.asarray(y_pred_proba)
                ),
                "pr_macro": self.pr_macro(
                    np.asarray(y_true_ohe), np.asarray(y_pred_proba)
                ),
            }
        return {k: float(v) for k, v in metrics.items()}
