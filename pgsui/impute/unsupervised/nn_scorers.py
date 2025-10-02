from typing import Dict, Literal

import numpy as np
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
        self.logger = logman.get_logger()

        if average not in {"micro", "macro", "weighted"}:
            msg = f"Invalid average parameter: {average}. Must be one of 'micro', 'macro', or 'weighted'."
            self.logger.error(msg)
            raise ValueError(msg)

        self.average = average

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the accuracy score.

        Args:
            y_true (np.ndarray): Ground truth (correct) target values.
            y_pred (np.ndarray): Estimated target values.

        Returns:
            float: The accuracy score.
        """
        return accuracy_score(y_true, y_pred)

    def f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the F1 score.

        Args:
            y_true (np.ndarray): Ground truth (correct) target values.
            y_pred (np.ndarray): Estimated target values.

        Returns:
            float: The F1 score.
        """
        return f1_score(y_true, y_pred, average=self.average, zero_division=0.0)

    def precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the precision score.

        Args:
            y_true (np.ndarray): Ground truth (correct) target values.
            y_pred (np.ndarray): Estimated target values.

        Returns:
            float: The precision score.
        """
        return precision_score(y_true, y_pred, average=self.average, zero_division=0.0)

    def recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the recall score.

        Args:
            y_true (np.ndarray): Ground truth (correct) target values.
            y_pred (np.ndarray): Estimated target values.

        Returns:
            float: The recall score.
        """
        return recall_score(y_true, y_pred, average=self.average, zero_division=0.0)

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
        return roc_auc_score(
            y_true, y_pred_proba, average=self.average, multi_class="ovr"
        )

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
        return average_precision_score(y_true_ohe, y_pred_proba, average=self.average)

    def pr_macro(self, y_true_ohe: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute the macro-average precision score.

        Args:
            y_true_ohe (np.ndarray): One-hot encoded ground truth target values.
            y_pred_proba (np.ndarray): Predicted probabilities.

        Returns:
            float: The macro-average precision score.
        """
        return average_precision_score(y_true_ohe, y_pred_proba, average="macro")

    def evaluate(
        self,
        y_true: np.ndarray | Tensor | list,
        y_pred: np.ndarray | Tensor | list,
        y_true_ohe: np.ndarray | Tensor | list,
        y_pred_proba: np.ndarray | Tensor | list,
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

        # NOTE: This is redundant because it's handled in the calling class
        # TODO: Remove redundancy
        valid_mask = np.logical_and(y_true >= 0, ~np.isnan(y_true))

        if not np.any(valid_mask):
            return {tune_metric: 0.0} if objective_mode else {}

        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        y_true_ohe = y_true_ohe[valid_mask]
        y_pred_proba = y_pred_proba[valid_mask]

        if objective_mode:
            metric_calculators = {
                "pr_macro": lambda: self.pr_macro(y_true_ohe, y_pred_proba),
                "roc_auc": lambda: self.roc_auc(y_true, y_pred_proba),
                "average_precision": lambda: self.average_precision(
                    y_true_ohe, y_pred_proba
                ),
                "accuracy": lambda: self.accuracy(y_true, y_pred),
                "f1": lambda: self.f1(y_true, y_pred),
                "precision": lambda: self.precision(y_true, y_pred),
                "recall": lambda: self.recall(y_true, y_pred),
            }
            if tune_metric not in metric_calculators:
                msg = f"Invalid tune_metric provided: '{tune_metric}'."
                self.logger.error(msg)
                raise ValueError(msg)

            metrics = {tune_metric: metric_calculators[tune_metric]()}
        else:
            metrics = {
                "accuracy": self.accuracy(y_true, y_pred),
                "f1": self.f1(y_true, y_pred),
                "precision": self.precision(y_true, y_pred),
                "recall": self.recall(y_true, y_pred),
                "roc_auc": self.roc_auc(y_true, y_pred_proba),
                "average_precision": self.average_precision(y_true_ohe, y_pred_proba),
                "pr_macro": self.pr_macro(y_true_ohe, y_pred_proba),
            }
        return {k: float(v) for k, v in metrics.items()}
