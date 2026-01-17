from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from snpio.utils.logging import LoggerManager

from pgsui.utils.logging_utils import configure_logger


class Scorer:
    """Class for evaluating the performance of a model using various metrics.

    This module provides a unified interface for computing common evaluation metrics. It supports accuracy, F1 score, precision, recall, ROC AUC, average precision, and macro-average precision. The class can handle both raw and one-hot encoded labels and includes options for logging and averaging methods.
    """

    def __init__(
        self,
        prefix: str,
        average: Literal["weighted", "macro"] = "macro",
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize a Scorer object.

        This class provides a unified interface for computing common evaluation metrics. It supports accuracy, F1 score, precision, recall, ROC AUC, average precision, and macro-average precision. The class can handle both raw and one-hot encoded labels and includes options for logging and averaging methods.

        Args:
            prefix (str): The prefix to use for logging.
            average (Literal["weighted", "macro"]): The averaging method to use for metrics. Must be one of 'macro' or 'weighted'. Defaults to 'macro'.
            verbose (bool): If True, enable verbose logging. Defaults to False.
            debug (bool): If True, enable debug logging. Defaults to False.
        """
        logman = LoggerManager(
            name=__name__, prefix=prefix, debug=debug, verbose=verbose
        )
        self.logger = configure_logger(
            logman.get_logger(), verbose=verbose, debug=debug
        )

        if average not in {"weighted", "macro"}:
            msg = f"Invalid average parameter: {average}. Must be one of 'macro' or 'weighted'."
            self.logger.error(msg)
            raise ValueError(msg)

        self.average: Literal["macro", "weighted"] = average

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

    def _prepare_ohe_proba(
        self, y_true_ohe: np.ndarray, y_pred_proba: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coerce y_true_ohe / y_pred_proba into 2D (N_eval, K) arrays and drop invalid rows.

        Rules:
            - If inputs are 3D (N, L, K), flatten to (N*L, K).
            - Class dimension must be last.
            - Drop rows where y_true_ohe is not a valid one-hot vector (sum != 1). This protects against ignored/masked entries leaking into metrics.

        Args:
            y_true_ohe (np.ndarray): One-hot encoded ground truth; shape (N_eval, K) or (N, L, K).
            y_pred_proba (np.ndarray): Predicted probabilities; shape (N_eval, K) or (N, L, K).

        Returns:
            (y_true_2d, y_proba_2d): Filtered 2D arrays with matching first dimension.
        """
        yt = np.asarray(y_true_ohe)
        yp = np.asarray(y_pred_proba)

        if yt.ndim == 3:
            yt = yt.reshape(-1, yt.shape[-1])
        if yp.ndim == 3:
            yp = yp.reshape(-1, yp.shape[-1])

        if yt.ndim != 2 or yp.ndim != 2:
            msg = f"Expected 2D or 3D arrays; got y_true_ohe.ndim={yt.ndim}, y_pred_proba.ndim={yp.ndim}."
            self.logger.error(msg)
            raise ValueError(msg)

        if yt.shape[0] != yp.shape[0]:
            msg = f"Mismatched rows: y_true_ohe has {yt.shape[0]}, y_pred_proba has {yp.shape[0]}."
            self.logger.error(msg)
            raise ValueError(msg)

        if yt.shape[1] != yp.shape[1]:
            msg = f"Mismatched class dimension: y_true_ohe K={yt.shape[1]}, y_pred_proba K={yp.shape[1]}."
            self.logger.error(msg)
            raise ValueError(msg)

        # Valid one-hot rows: sum == 1 (and non-negative)
        row_sums = yt.sum(axis=1)
        valid = (row_sums == 1) & np.all(yt >= 0, axis=1)

        if not np.any(valid):
            # No valid rows to score
            return yt[:0], yp[:0]

        return yt[valid], yp[valid]

    def roc_auc(self, y_true_ohe: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute ROC AUC (binary or multiclass OVR) robustly."""
        yt, yp = self._prepare_ohe_proba(y_true_ohe, y_pred_proba)

        if yt.shape[0] == 0:
            self.logger.warning("No valid rows for ROC AUC; returning 0.0.")
            return 0.0

        K = yt.shape[1]

        # Determine which classes are present in truth
        present = np.where(yt.sum(axis=0) > 0)[0]
        if present.size < 2:
            msg = "ROC AUC: fewer than 2 classes in y_true returning 0.0."
            self.logger.warning(msg)
            return 0.0

        if K == 2:
            # Binary: score positive class only (class 1)
            y_true_bin = yt[:, 1]
            y_score = yp[:, 1]

            print(y_true_bin)
            print(y_score)
            try:
                return float(roc_auc_score(y_true_bin, y_score))
            except ValueError as e:
                msg = f"ROC AUC failed binary case; returning 0.0. Details: {e}"
                self.logger.warning(msg)
                return 0.0

        # Multiclass: one-vs-rest
        try:
            roc_auc = roc_auc_score(yt, yp, average=self.average, multi_class="ovr")
            return float(roc_auc)
        except ValueError as e:
            msg = f"ROC AUC failed multiclass case; returning 0.0. Details: {e}"
            self.logger.warning(msg)
            return 0.0

    def average_precision(
        self, y_true_ohe: np.ndarray, y_pred_proba: np.ndarray
    ) -> float:
        """Compute Average Precision (binary or multiclass) robustly."""
        yt, yp = self._prepare_ohe_proba(y_true_ohe, y_pred_proba)
        if yt.shape[0] == 0:
            msg = "No valid rows for Average Precision; returning 0.0."
            self.logger.warning(msg)
            return 0.0

        K = yt.shape[1]
        present = np.where(yt.sum(axis=0) > 0)[0]
        if present.size == 0:
            msg = "Average Precision undefined (no positives); returning 0.0."
            self.logger.warning(msg)
            return 0.0

        if K == 2:
            # Binary: positive class only
            y_true_bin = yt[:, 1]
            y_score = yp[:, 1]
            try:
                return float(average_precision_score(y_true_bin, y_score))
            except ValueError as e:
                msg = (
                    f"Average Precision failed binary case; returning 0.0. Details: {e}"
                )
                self.logger.warning(msg)
                return 0.0

        # Multiclass: sklearn expects indicator matrix + score matrix
        try:
            return float(average_precision_score(yt, yp, average=self.average))
        except ValueError as e:
            msg = (
                f"Average Precision failed multiclass case; returning 0.0. Details: {e}"
            )
            self.logger.warning(msg)
            return 0.0

    def pr_macro(self, y_true_ohe: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute macro-average precision (PR-AUC macro) robustly."""
        yt, yp = self._prepare_ohe_proba(y_true_ohe, y_pred_proba)
        if yt.shape[0] == 0:
            self.logger.warning("No valid rows for PR macro; returning 0.0.")
            return 0.0

        K = yt.shape[1]
        if K == 2:
            # Binary: macro is effectively binary AP; use positive class
            y_true_bin = yt[:, 1]
            y_score = yp[:, 1]
            try:
                return float(average_precision_score(y_true_bin, y_score))
            except ValueError as e:
                msg = f"PR macro failed binary case; returning 0.0. Details: {e}"
                self.logger.warning(msg)
                return 0.0

        try:
            return float(average_precision_score(yt, yp, average="macro"))
        except ValueError as e:
            msg = f"PR-macro failed multiclass case; returning 0. Details: {e}"
            self.logger.warning(msg)
            return 0.0

    def jaccard(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the Jaccard score.

        Args:
            y_true (np.ndarray): Ground truth (correct) target values.
            y_pred (np.ndarray): Estimated target values.

        Returns:
            float: The Jaccard score.
        """
        return float(
            jaccard_score(y_true, y_pred, average=self.average, zero_division=0)
        )

    def mcc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the Matthews correlation coefficient (MCC).

        MCC is a balanced measure that can be used even if the classes are of very different sizes. It returns a value between -1 and +1, where +1 indicates a perfect prediction, 0 indicates no better than random prediction, and -1 indicates total disagreement between prediction and observation.

        Args:
            y_true (np.ndarray): Ground truth (correct) target values.
            y_pred (np.ndarray): Estimated target values.

        Returns:
            float: The Matthews correlation coefficient.
        """
        return float(matthews_corrcoef(y_true, y_pred))

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_true_ohe: np.ndarray,
        y_pred_proba: np.ndarray,
        objective_mode: bool = False,
        tune_metric: (
            Literal[
                "pr_macro",
                "roc_auc",
                "average_precision",
                "accuracy",
                "f1",
                "precision",
                "recall",
                "mcc",
                "jaccard",
            ]
            | list[str]
            | tuple[str, ...]
        ) = "f1",
    ) -> dict[str, float]:
        """Evaluate the model using various metrics.

        Args:
            y_true (np.ndarray): Ground truth (correct) target values; shape (N_eval,).
            y_pred (np.ndarray): Estimated target values; shape (N_eval,).
            y_true_ohe (np.ndarray): One-hot encoded ground truth; shape (N_eval, K) or (N, L, K).
            y_pred_proba (np.ndarray): Predicted probabilities; shape (N_eval, K) or (N, L, K).
            objective_mode (bool): If True, compute only the tune_metric(s). Defaults to False.
            tune_metric (str | list[str] | tuple[str, ...]): The metric(s) to compute in objective mode.
                Valid options are: 'pr_macro', 'roc_auc', 'average_precision', 'accuracy', 'f1', 'precision', 'recall', 'mcc', 'jaccard'.
                Defaults to 'f1'.

        Returns:
            dict[str, float]: A dictionary mapping metric names to their computed values.
        """
        if not all(
            isinstance(y, np.ndarray)
            for y in (y_true, y_pred, y_true_ohe, y_pred_proba)
        ):
            msg = "y inputs to 'Scorer.evaluate()' must be numpy arrays."
            self.logger.error(msg)
            raise TypeError(msg)

        y_true = np.asarray(y_true).astype(np.int8, copy=False).reshape(-1)
        y_pred = np.asarray(y_pred).astype(np.int8, copy=False).reshape(-1)
        y_pred_proba = np.asarray(y_pred_proba).astype(np.float32, copy=False)

        if y_pred_proba.ndim == 3:
            y_pred_proba = y_pred_proba.reshape(-1, y_pred_proba.shape[-1])
        elif y_pred_proba.ndim != 2:
            msg = f"y_pred_proba must be 2D or 3D; got shape {y_pred_proba.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        K = int(y_pred_proba.shape[-1])

        y_true_ohe = np.asarray(y_true_ohe).astype(np.float32, copy=False)
        if y_true_ohe.ndim == 3:
            if y_true_ohe.shape[-1] != K:
                msg = f"y_true_ohe K={y_true_ohe.shape[-1]} does not match y_pred_proba K={K}."
                self.logger.error(msg)
                raise ValueError(msg)
            y_true_ohe = y_true_ohe.reshape(-1, K)

        elif y_true_ohe.ndim == 2:
            if y_true_ohe.shape[1] != K:
                msg = f"y_true_ohe K={y_true_ohe.shape[1]} does not match y_pred_proba K={K}."
                self.logger.error(msg)
                raise ValueError(msg)
        else:
            msg = f"y_true_ohe must be 2D or 3D; got shape {y_true_ohe.shape}."
            self.logger.error(msg)
            raise ValueError(msg)

        # Ensure valid onehots
        if sum(y_true_ohe.sum(axis=-1) != 1) > 0:
            msg = "y_true_ohe contains invalid one-hots that do not sum to 1."
            self.logger.error(msg)
            raise ValueError(msg)

        if y_true.shape[0] != y_pred.shape[0]:
            msg = f"y_true and y_pred length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}."
            self.logger.error(msg)
            raise ValueError(msg)

        if y_true_ohe.shape[0] != y_pred_proba.shape[0]:
            msg = f"y_true_ohe and y_pred_proba row mismatch: {y_true_ohe.shape[0]} vs {y_pred_proba.shape[0]}."
            self.logger.error(msg)
            raise ValueError(msg)

        if objective_mode:
            metric_calculators = {
                "pr_macro": lambda: self.pr_macro(y_true_ohe, y_pred_proba),
                "roc_auc": lambda: self.roc_auc(y_true_ohe, y_pred_proba),
                "average_precision": lambda: self.average_precision(
                    y_true_ohe, y_pred_proba
                ),
                "accuracy": lambda: self.accuracy(y_true, y_pred),
                "f1": lambda: self.f1(y_true, y_pred),
                "precision": lambda: self.precision(y_true, y_pred),
                "recall": lambda: self.recall(y_true, y_pred),
                "mcc": lambda: self.mcc(y_true, y_pred),
                "jaccard": lambda: self.jaccard(y_true, y_pred),
            }

            if isinstance(tune_metric, (list, tuple)):
                invalid = [tm for tm in tune_metric if tm not in metric_calculators]
                if invalid:
                    msg = f"Invalid tune_metric(s) provided: {invalid}. Valid options are: {list(metric_calculators.keys())}."
                    self.logger.error(msg)
                    raise ValueError(msg)
                metrics = {tm: metric_calculators[tm]() for tm in tune_metric}
            else:
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
                "roc_auc": self.roc_auc(y_true_ohe, y_pred_proba),
                "average_precision": self.average_precision(y_true_ohe, y_pred_proba),
                "pr_macro": self.pr_macro(y_true_ohe, y_pred_proba),
                "mcc": self.mcc(y_true, y_pred),
                "jaccard": self.jaccard(y_true, y_pred),
            }

        return {k: float(v) for k, v in metrics.items()}
