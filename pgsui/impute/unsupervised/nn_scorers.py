import pprint
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from snpio.utils.logging import LoggerManager


class Scorer:
    """Class for evaluating the performance of a model using various metrics.

    This class is used to evaluate the performance of a model using various metrics, such as accuracy, F1 score, precision, recall, average precision, and ROC AUC. The class can be used to evaluate the performance of a model on a dataset with ground truth labels. It also supports masking missing values in the evaluation.

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
        >>> scorer = Scorer(model, X_test, y_test, y_pred)
        >>> print(scorer.evaluate())
        {'accuracy': 0.95, 'f1': 0.95, 'precision': 0.95, 'recall': 0.95, 'roc_auc': 0.95, 'average_precision': 0.95}
        >>> # scorer is also a callable object.
        >>> print(scorer())

    Attributes:
        model (torch.nn.Module): Model to evaluate.
        X (np.ndarray or torch.Tensor): Input data.
        y_true (np.ndarray or torch.Tensor): Ground truth labels.
        y_pred (np.ndarray or torch.Tensor): Predicted labels.
        mask (np.ndarray or torch.Tensor, optional): Mask for missing values.
        average (str, optional): Average method for metrics. Must be one of 'micro', 'macro', or 'weighted'.
        logger (LoggerManager, optional): Logger for logging messages.
        verbose (int, optional): Verbosity level for logging messages.
        debug (bool, optional): Debug mode for logging messages.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        X: np.ndarray | torch.Tensor,
        y_true: np.ndarray | torch.Tensor,
        y_pred: np.ndarray | torch.Tensor,
        y_true_ohe: np.ndarray | torch.Tensor,
        y_pred_proba: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor = None,
        average: str = "weighted",
        logger: Optional[LoggerManager] = None,
        verbose: int = 0,
        debug: bool = False,
    ) -> None:
        """Initialize a Scorer object.

        This class is used to evaluate the performance of a model using various metrics, such as accuracy, F1 score, precision, recall, average precision, and ROC AUC. The class can be used to evaluate the performance of a model on a dataset with ground truth labels. It also supports masking missing values in the evaluation.

        Args:
            model (torch.nn.Module): Model to evaluate.
            X (np.ndarray or torch.Tensor): Input data.
            y_true (np.ndarray or torch.Tensor): Ground truth labels.
            y_pred (np.ndarray or torch.Tensor): Predicted labels.
            y_true_ohe (np.ndarray or torch.Tensor): Ground truth labels in one-hot encoded format.
            y_pred_proba (np.ndarray or torch.Tensor): Predicted probabilities.
            mask (np.ndarray or torch.Tensor, optional): Mask for missing values.
            average (str, optional): Average method for metrics. Must be one of 'micro', 'macro', or 'weighted'.
            logger (LoggerManager, optional): Logger for logging messages.
            verbose (int, optional): Verbosity level for logging messages.
            debug (bool, optional): Debug mode for logging messages.

        Raises:
            ValueError: If the average parameter is invalid.
        """
        self.model = model
        self.X = X
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_true_ohe = y_true_ohe
        self.y_pred_proba = y_pred_proba
        self.mask = mask
        self.average = average

        if logger is not None:
            self.logger = logger
        else:
            logman = LoggerManager(
                name=__name__, prefix="pgsui", debug=debug, verbose=verbose >= 1
            )
            self.logger = logman.get_logger()

        if average not in {"micro", "macro", "weighted"}:
            msg = f"Invalid average parameter: {average}. Must be one of 'micro', 'macro', or 'weighted'."
            self.logger.error(msg)
            raise ValueError(msg)

        data = (X, y_true, y_pred, y_true_ohe, y_pred_proba, mask)
        if any(isinstance(x, torch.Tensor) for x in data):
            if isinstance(X, torch.Tensor):
                self.X = X.cpu().detach().numpy()

            if isinstance(y_true, torch.Tensor):
                self.y_true = y_true.cpu().detach().numpy()

            if isinstance(y_pred, torch.Tensor):
                self.y_pred = y_pred.cpu().detach().numpy()

            if isinstance(y_true_ohe, torch.Tensor):
                self.y_true_ohe = y_true_ohe.cpu().detach().numpy()

            if isinstance(y_pred_proba, torch.Tensor):
                self.y_pred_proba = y_pred_proba.cpu().detach().numpy()

            if isinstance(mask, torch.Tensor):
                self.mask = mask.cpu().detach().numpy()

        if self.mask is not None:
            self.y_true = y_true[self.mask]
            self.y_pred = y_pred[self.mask]
            self.y_true_ohe = self.y_true_ohe[self.mask]
            self.y_pred_proba = self.y_pred_proba[self.mask]

            self.logger.debug(
                f"Masking missing values for evaluation. New shape: {self.y_true.shape}"
            )
        self.logger.debug(f"y_true: {self.y_true.shape}")
        self.logger.debug(f"y_pred: {self.y_pred.shape}")
        self.logger.debug(f"y_true_ohe: {self.y_true_ohe.shape}")
        self.logger.debug(f"y_pred_proba: {self.y_pred_proba.shape}")

    def accuracy(self) -> float:
        """Calculate the accuracy of the model."""
        return accuracy_score(self.y_true, self.y_pred)

    def f1(self) -> np.ndarray:
        """Calculate the F1 score of the model."""
        return f1_score(self.y_true, self.y_pred, average=self.average)

    def precision(self) -> np.ndarray:
        """Calculate the precision of the model."""
        return precision_score(self.y_true, self.y_pred, average=self.average)

    def recall(self) -> np.ndarray:
        """Calculate the recall of the model."""
        return recall_score(self.y_true, self.y_pred, average=self.average)

    def roc_auc(self) -> np.ndarray:
        """Calculate the ROC AUC of the model."""
        return roc_auc_score(
            self.y_true,
            self.y_pred_proba,
            average=self.average,
            multi_class="ovo",
        )

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model using various metrics."""
        return {
            "accuracy": self.accuracy(),
            "f1": self.f1(),
            "precision": self.precision(),
            "recall": self.recall(),
            "roc_auc": self.roc_auc(),
            "average_precision": self.average_precision(),
        }

    def pr_curve(self):
        """Calculate the precision-recall curve of the model."""
        precision, recall, _ = precision_recall_curve(
            self.y_true_ohe.ravel(), self.y_pred_proba.ravel()
        )
        return precision, recall

    def roc_curve(self):
        """Calculate the ROC curve of the model."""
        fpr, tpr, _ = roc_curve(self.y_true_ohe.ravel(), self.y_pred_proba.ravel())
        return fpr, tpr

    def average_precision(self):
        """Calculate the average precision of the model."""
        return average_precision_score(
            self.y_true, self.y_pred_proba, average=self.average
        )

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return pprint.pformat(self.evaluate(), indent=4)

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return pprint.pprint(self.evaluate(), indent=4)

    def __call__(self) -> Dict[str, float]:
        """Evaluate the model using various metrics."""
        return self.evaluate()

    def __getitem__(self, key: str) -> float:
        """Get the value of a metric."""
        return self.evaluate()[key]

    def __len__(self) -> int:
        """Return the number of metrics."""
        return len(self.evaluate())

    def __iter__(self):
        """Iterate over the metrics."""
        for key, value in self.evaluate().items():
            yield key, value

    def __contains__(self, key: str) -> bool:
        """Check if a metric is present."""
        return key in self.evaluate()

    def __eq__(self, other: Any) -> bool:
        """Check if two objects are equal."""

        if isinstance(other, Scorer):
            return self.evaluate() == other.evaluate()
        return False

    def __ne__(self, other: Any) -> bool:
        """Check if two objects are not equal."""
        if isinstance(other, Scorer):
            return self.evaluate() != other.evaluate()
        return False

    def __lt__(self, other: Any) -> bool:
        """Check if the object is less than another object."""
        if isinstance(other, Scorer):
            return self.evaluate() < other.evaluate()
        return False

    def __le__(self, other: Any) -> bool:
        """Check if the object is less than or equal to another object."""

        if isinstance(other, Scorer):
            return self.evaluate() <= other.evaluate()
        return False

    def __gt__(self, other: Any) -> bool:
        """Check if the object is greater than another object."""

        if isinstance(other, Scorer):
            return self.evaluate() > other.evaluate()
        return False

    def __ge__(self, other: Any) -> bool:
        """Check if the object is greater than or equal to another object."""

        if isinstance(other, Scorer):
            return self.evaluate() >= other.evaluate()
        return False
