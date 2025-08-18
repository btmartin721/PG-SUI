from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from snpio.utils.logging import LoggerManager
from torch import Tensor

from pgsui.utils.misc import validate_input_type


class Scorer:
    """Class for evaluating the performance of a model using various metrics.

    This class is used to evaluate the performance of a model using various metrics, such as accuracy, F1 score, precision, recall, average precision, and ROC AUC. The class can be used to evaluate the performance of a model on a dataset with ground truth labels. The class can also be used to evaluate the performance of a model in objective mode for hyperparameter tuning.

    Example:
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.model_selection import train_test_split
        >>> from pgsui.utils.scorer import Scorer
        >>> X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        >>> model = LogisticRegression(random_state=42)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
        >>> scorer = Scorer(average="macro", verbose=1, logger=logger)
        >>> print(scorer.evaluate(model, y_true, y_pred, y_true_ohe, y_pred_proba))
        {'accuracy': 0.95, 'f1': 0.95, 'precision': 0.95, 'recall': 0.95, 'roc_auc': 0.95, 'average_precision': 0.95, 'pr_macro': 0.95}
    """

    def __init__(
        self,
        average: str = "weighted",
        logger: LoggerManager | None = None,
        verbose: int = 0,
        debug: bool = False,
    ) -> None:
        """Initialize a Scorer object.

        This class is used to evaluate the performance of a model using various metrics, such as accuracy, F1 score, precision, recall, average precision, and ROC AUC. The class can be used to evaluate the performance of a model on a dataset with ground truth labels. The class can also be used to evaluate the performance of a model in objective mode for hyperparameter tuning.

        Args:
            average (str, optional): Average method for metrics. Must be one of 'micro', 'macro', or 'weighted'.
            logger (LoggerManager, optional): Logger for logging messages. If None, a new logger is created. Default is None.
            verbose (int, optional): Verbosity level for logging messages. Default is 0.
            debug (bool, optional): Debug mode for logging messages. Default is False.

        Raises:
            ValueError: If the average parameter is invalid. Must be one of 'micro', 'macro', or 'weighted'.
        """
        self.average = average

        if logger is not None:
            self.logger = logger
        else:
            prefix = "pgsui_output" if prefix == "pgsui" else prefix
            logman = LoggerManager(
                name=__name__, prefix=prefix, debug=debug, verbose=verbose >= 1
            )
            self.logger = logman.get_logger()

        if average not in {"micro", "macro", "weighted"}:
            msg = f"Invalid average parameter: {average}. Must be one of 'micro', 'macro', or 'weighted'."
            self.logger.error(msg)
            raise ValueError(msg)

    def accuracy(self, y_true, y_pred) -> float:
        """Calculate the accuracy of the model.

        Args:
            y_true (np.ndarray or torch.Tensor): Ground truth labels.
            y_pred (np.ndarray or torch.Tensor): Predicted labels.

        Returns:
            float: Accuracy score.
        """
        return accuracy_score(y_true, y_pred)

    def f1(self, y_true, y_pred) -> np.ndarray:
        """Calculate the F1 score of the model.

        Args:
            y_true (np.ndarray or torch.Tensor): Ground truth labels.
            y_pred (np.ndarray or torch.Tensor): Predicted labels.

        Returns:
            float: F1 score.
        """
        return f1_score(y_true, y_pred, average=self.average, zero_division=0.0)

    def precision(self, y_true, y_pred) -> np.ndarray:
        """Calculate the precision of the model.

        Args:
            y_true (np.ndarray or torch.Tensor): Ground truth labels.
            y_pred (np.ndarray or torch.Tensor): Predicted labels.

        Returns:
            float: Precision score.
        """
        return precision_score(y_true, y_pred, average=self.average, zero_division=0.0)

    def recall(self, y_true, y_pred) -> np.ndarray:
        """Calculate the recall of the model.

        Args:
            y_true (np.ndarray or torch.Tensor): Ground truth labels.
            y_pred (np.ndarray or torch.Tensor): Predicted labels.

        Returns:
            float: Recall score.
        """
        return recall_score(y_true, y_pred, average=self.average, zero_division=0.0)

    def roc_auc(self, y_true, y_pred_proba) -> np.ndarray:
        """Calculate the ROC AUC of the model.

        Args:
            y_true (np.ndarray or torch.Tensor): Ground truth labels.
            y_pred_proba (np.ndarray or torch.Tensor): Predicted probabilities.

        Returns:
            float: ROC AUC score.

        Notes:
            - This method uses the 'ovr' strategy for multi-class classification.
            - The input data must be properly formatted.
            - If all ground truth labels are 0, the ROC AUC score is 0.5.

        """
        if y_pred_proba.ndim == 3:
            y_pred_proba = y_pred_proba.reshape(-1, y_pred_proba.shape[-1])

        if y_true.shape[-1] == y_pred_proba.shape[-1]:
            y_true = np.argmax(y_true, axis=-1)

        if np.all(y_true == 0):
            return 0.5

        return roc_auc_score(
            y_true, y_pred_proba, average=self.average, multi_class="ovr"
        )

    def evaluate(
        self,
        y_true: np.ndarray | Tensor | list,
        y_pred: np.ndarray | Tensor | list,
        y_true_ohe: np.ndarray | Tensor | list,
        y_pred_proba: np.ndarray | Tensor | list,
        objective_mode: bool = False,
        tune_metric: str = "pr_macro",
    ) -> Dict[str, float]:
        """Evaluate the model using various metrics.

        This method evaluates the performance of a model using various metrics, such as accuracy, F1 score, precision, recall, average precision, and ROC AUC. The method can be used to evaluate the performance of a model on a dataset with ground truth labels. The method can also be used to evaluate the performance of a model in objective mode for hyperparameter tuning.

        Args:
            y_true (np.ndarray or torch.Tensor): Ground truth labels.
            y_pred (np.ndarray or torch.Tensor): Predicted labels.
            y_true_ohe (np.ndarray or torch.Tensor): One-hot encoded ground truth labels.
            y_pred_proba (np.ndarray or torch.Tensor): Predicted probabilities.
            objective_mode (bool, optional): Whether to use objective mode for evaluation. Default is False.
            tune_metric (str, optional): Metric to use for tuning. Ignored if `objective_mode` is False. Default is 'pr_macro'.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics. Keys are 'accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'average_precision', and 'pr_macro'.

        Raises:
            ValueError: If the input data is invalid.
            ValueError: If an invalid tune_metric is provided.
        """
        data = [y_true, y_pred, y_true_ohe, y_pred_proba]
        data = [validate_input_type(x) for x in data if x is not None]
        valid_mask = np.logical_and(data[0] >= 0, ~np.isnan(data[0]))
        data = [x[valid_mask] for x in data]
        y_true, y_pred, y_true_ohe, y_pred_proba = data

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

        if not y_pred_proba.ndim == 2:
            msg = "y_pred_proba must have 2 dimensions."
            self.logger.error(msg)
            raise ValueError(msg)

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

    def average_precision(self, y_true, y_pred_proba):
        """Calculate the average precision of the model.

        Args:
            y_true (np.ndarray or torch.Tensor): Ground truth labels.
            y_pred_proba (np.ndarray or torch.Tensor): Predicted probabilities.

        Returns:
            float: Average precision score.
        """
        if y_pred_proba.ndim == 3:
            y_pred_proba = y_pred_proba.reshape(-1, y_pred_proba.shape[-1])

        if y_true.ndim >= 2 and y_true.shape[-1] == y_pred_proba.shape[-1]:
            y_true = np.argmax(y_true, axis=-1)

        try:
            return average_precision_score(y_true, y_pred_proba, average=self.average)
        except ValueError:
            return average_precision_score(
                y_true, np.argmax(y_pred_proba, axis=-1), average=self.average
            )

    def pr_macro(self, y_true_ohe, y_pred_proba):
        """Calculate the average precision of the model.

        Args:
            y_true_ohe (np.ndarray or torch.Tensor): One-hot encoded ground truth labels.
            y_pred_proba (np.ndarray or torch.Tensor): Predicted probabilities.

        Returns:
            float: Average precision score (macro average).
        """
        if y_pred_proba.ndim == 3:
            y_pred_proba = y_pred_proba.reshape(-1, y_pred_proba.shape[-1])

        # Ensure y_true is properly binarized
        num_classes = y_pred_proba.shape[1]
        y_true = y_true_ohe

        # Initialize dictionaries for metrics
        fpr, tpr, roc_auc = {}, {}, {}
        precision, recall, average_precision = {}, {}, {}

        # Compute per-class ROC and PR metrics
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_proba[:, i], pos_label=1)
            roc_auc[i] = auc(fpr[i], tpr[i])
            precision[i], recall[i], _ = precision_recall_curve(
                y_true[:, i], y_pred_proba[:, i]
            )
            average_precision[i] = average_precision_score(
                y_true[:, i], y_pred_proba[:, i]
            )

        # Macro-average PR
        all_recall = np.unique(np.concatenate([recall[i] for i in range(num_classes)]))
        mean_precision = np.zeros_like(all_recall)
        for i in range(num_classes):
            mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
        mean_precision /= num_classes
        average_precision["macro"] = average_precision_score(
            y_true, y_pred_proba, average="macro"
        )

        return average_precision_score(y_true, y_pred_proba, average="macro")
