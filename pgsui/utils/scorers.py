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
from sklearn.preprocessing import label_binarize
from snpio.utils.logging import LoggerManager
from torch import Tensor


class Scorer:
    """Class for evaluating the performance of a model using various metrics.

    This class is used to evaluate the performance of a model using various metrics, such as accuracy, F1 score, precision, recall, average precision, and ROC AUC. The class can be used to evaluate the performance of a model on a dataset with ground truth labels. The class can also be used to evaluate the performance of a model in objective mode for hyperparameter tuning.
    """

    def __init__(
        self,
        prefix: str,
        average: Literal["micro", "macro", "weighted"] = "weighted",
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize a Scorer object.

        This class is used to evaluate the performance of a model using various metrics, such as accuracy, F1 score, precision, recall, average precision, and ROC AUC. The class can be used to evaluate the performance of a model on a dataset with ground truth labels. The class can also be used to evaluate the performance of a model in objective mode for hyperparameter tuning.

        Args:
            prefix (str): Prefix for logging messages.
            average (Literal["micro", "macro", "weighted"]): Average method for metrics. Must be one of 'micro', 'macro', or 'weighted'.
            verbose (bool): Verbosity level for logging messages. Default is False.
            debug (bool): Debug mode for logging messages. Default is False.

        Raises:
            ValueError: If the average parameter is invalid. Must be one of 'micro', 'macro', or 'weighted'.
        """
        logman = LoggerManager(
            name=__name__, prefix=prefix, debug=debug, verbose=verbose >= 1
        )
        self.logger = logman.get_logger()

        if average not in {"micro", "macro", "weighted"}:
            msg = f"Invalid average parameter: {average}. Must be one of 'micro', 'macro', or 'weighted'."
            self.logger.error(msg)
            raise ValueError(msg)

        self.average = average

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the accuracy of the model.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Accuracy score.
        """
        return accuracy_score(y_true, y_pred)

    def f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the F1 score of the model.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: F1 score.
        """
        return f1_score(y_true, y_pred, average=self.average, zero_division=0.0)

    def precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the precision of the model.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Precision score.
        """
        return precision_score(y_true, y_pred, average=self.average, zero_division=0.0)

    def recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate the recall of the model.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Recall score.
        """
        return recall_score(y_true, y_pred, average=self.average, zero_division=0.0)

    def roc_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Multiclass ROC-AUC with label targets.

        Args:
            y_true: 1D integer labels (shape: [n]).
                    If a one-hot/indicator matrix is supplied, we convert to labels.
            y_pred_proba: 2D probabilities (shape: [n, n_classes]).
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        if y_pred_proba.ndim == 3:
            y_pred_proba = y_pred_proba.reshape(-1, y_pred_proba.shape[-1])

        # If user passed indicator/one-hot, convert to labels.
        if y_true.ndim == 2 and y_true.shape[1] == y_pred_proba.shape[1]:
            y_true = y_true.argmax(axis=1)

        # Guard: need >1 class present for AUC
        if np.unique(y_true).size < 2:
            return 0.5

        return float(
            roc_auc_score(
                y_true,
                y_pred_proba,
                multi_class="ovr",
                average=self.average,
            )
        )

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

        This method evaluates the performance of a model using various metrics, such as accuracy, F1 score, precision, recall, average precision, and ROC AUC. The method can be used to evaluate the performance of a model on a dataset with ground truth labels. The method can also be used to evaluate the performance of a model in objective mode for hyperparameter tuning.

        Args:
            y_true (np.ndarray | torch.Tensor): Ground truth labels.
            y_pred (np.ndarray | torch.Tensor): Predicted labels.
            y_true_ohe (np.ndarray | torch.Tensor): One-hot encoded ground truth labels.
            y_pred_proba (np.ndarray | torch.Tensor): Predicted probabilities.
            objective_mode (bool): Whether to use objective mode for evaluation. Default is False.
            tune_metric (Literal["pr_macro", "roc_auc", "average_precision", "accuracy", "f1", "precision", "recall"]): Metric to use for tuning. Ignored if `objective_mode` is False. Default is 'pr_macro'.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics. Keys are 'accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'average_precision', and 'pr_macro'.

        Raises:
            ValueError: If the input data is invalid.
            ValueError: If an invalid tune_metric is provided.
        """
        if not y_true.ndim < 3:
            msg = "y_true must have 1 or 2 dimensions."
            self.logger.error(msg)
            raise ValueError(msg)

        if not y_pred.ndim < 3:
            msg = "y_pred must have 1 or 2 dimensions."
            self.logger.error(msg)
            raise ValueError(msg)

        if not y_true_ohe.ndim == 2:
            msg = "y_true_ohe must have 2 dimensions."
            self.logger.error(msg)
            raise ValueError(msg)

        if y_pred_proba.ndim != 2:
            y_pred_proba = y_pred_proba.reshape(-1, y_true_ohe.shape[-1])
            self.logger.debug(f"Reshaped y_pred_proba to {y_pred_proba.shape}")

        if objective_mode:
            if tune_metric == "pr_macro":
                metrics = {"pr_macro": self.pr_macro(y_true_ohe, y_pred_proba)}
            elif tune_metric == "roc_auc":
                metrics = {"roc_auc": self.roc_auc(y_true, y_pred_proba)}
            elif tune_metric == "average_precision":
                metrics = {
                    "average_precision": self.average_precision(y_true, y_pred_proba)
                }
            elif tune_metric == "accuracy":
                metrics = {"accuracy": self.accuracy(y_true, y_pred)}
            elif tune_metric == "f1":
                metrics = {"f1": self.f1(y_true, y_pred)}
            elif tune_metric == "precision":
                metrics = {"precision": self.precision(y_true, y_pred)}
            elif tune_metric == "recall":
                metrics = {"recall": self.recall(y_true, y_pred)}
            else:
                msg = f"Invalid tune_metric provided: '{tune_metric}'."
                self.logger.error(msg)
                raise ValueError(msg)
        else:
            metrics = {
                "accuracy": self.accuracy(y_true, y_pred),
                "f1": self.f1(y_true, y_pred),
                "precision": self.precision(y_true, y_pred),
                "recall": self.recall(y_true, y_pred),
                "roc_auc": self.roc_auc(y_true, y_pred_proba),
                "average_precision": self.average_precision(y_true, y_pred_proba),
                "pr_macro": self.pr_macro(y_true_ohe, y_pred_proba),
            }

        return {k: float(v) for k, v in metrics.items()}

    def average_precision(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Average precision with safe multiclass handling.

        If y_true is 1D of class indices, it is binarized against the number of columns in y_pred_proba.
        If y_true is already one-hot or indicator, it is used as-is.
        """
        y_true_arr = np.asarray(y_true)
        y_proba_arr = np.asarray(y_pred_proba)

        if y_proba_arr.ndim == 3:
            y_proba_arr = y_proba_arr.reshape(-1, y_proba_arr.shape[-1])

        # If y_true already matches proba columns (one-hot / indicator)
        if y_true_arr.ndim == 2 and y_true_arr.shape[1] == y_proba_arr.shape[1]:
            y_bin = y_true_arr
        else:
            # Interpret y_true as class indices
            n_classes = y_proba_arr.shape[1]
            y_bin = label_binarize(y_true_arr.ravel(), classes=np.arange(n_classes))

        return float(average_precision_score(y_bin, y_proba_arr, average=self.average))

    def pr_macro(self, y_true_ohe: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Macro-averaged average precision (precision-recall AUC) across classes."""
        y_true_arr = np.asarray(y_true_ohe)
        y_proba_arr = np.asarray(y_pred_proba)

        if y_proba_arr.ndim == 3:
            y_proba_arr = y_proba_arr.reshape(-1, y_proba_arr.shape[-1])

        # Ensure 2D indicator truth
        if y_true_arr.ndim == 1:
            n_classes = y_proba_arr.shape[1]
            y_true_arr = label_binarize(y_true_arr, classes=np.arange(n_classes))

        return float(average_precision_score(y_true_arr, y_proba_arr, average="macro"))
