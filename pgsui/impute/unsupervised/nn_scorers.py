from __future__ import annotations

from typing import Literal, Optional, Tuple

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

    def _prepare_ohe_proba_with_mask(
        self, y_true_ohe: np.ndarray, y_pred_proba: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare (yt, yp) as (N_eval, K) and return the boolean mask of kept rows.

        This is identical in spirit to _prepare_ohe_proba(), but additionally returns the
        row mask so callers (notably evaluate()) can filter y_true/y_pred identically.

        Args:
            y_true_ohe (np.ndarray): Indicator labels; shape (N_eval, K) or (N, L, K).
            y_pred_proba (np.ndarray): Scores/probabilities; shape (N_eval, K) or (N, L, K).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                (yt2, yp2, mask) where yt2/yp2 are filtered 2D arrays and mask has shape (N_eval,).
                If nothing is valid, yt2/yp2 are empty with shape (0, K) and mask is all-False.
        """
        yt = np.asarray(y_true_ohe)
        yp = np.asarray(y_pred_proba)

        # Flatten (N, L, K) -> (N*L, K)
        if yt.ndim == 3:
            yt = yt.reshape(-1, yt.shape[-1])
        if yp.ndim == 3:
            yp = yp.reshape(-1, yp.shape[-1])

        if yt.ndim != 2 or yp.ndim != 2:
            msg = (
                "Expected 2D or 3D arrays (class dim last). "
                f"Got y_true_ohe.ndim={yt.ndim}, y_pred_proba.ndim={yp.ndim}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        if yt.shape[0] != yp.shape[0]:
            msg = (
                f"Mismatched rows: y_true_ohe has {yt.shape[0]}, "
                f"y_pred_proba has {yp.shape[0]}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        if yt.shape[1] != yp.shape[1]:
            msg = (
                f"Mismatched class dimension: y_true_ohe K={yt.shape[1]}, "
                f"y_pred_proba K={yp.shape[1]}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        K = yt.shape[1]
        if K == 0:
            mask = np.zeros((yt.shape[0],), dtype=bool)
            return yt[:0], yp[:0], mask

        # Finiteness: for object dtype, isfinite can raise -> treat as no valid rows
        try:
            finite_yt = np.isfinite(yt).all(axis=1)
            finite_yp = np.isfinite(yp).all(axis=1)
        except Exception:
            mask = np.zeros((yt.shape[0],), dtype=bool)
            return yt[:0, :], yp[:0, :], mask

        # Strict one-hot: entries ∈ {0,1} and row sum==1
        is01 = ((yt == 0) | (yt == 1)).all(axis=1)
        sum1 = yt.sum(axis=1) == 1

        mask = finite_yt & finite_yp & is01 & sum1

        if not np.any(mask):
            return yt[:0, :], yp[:0, :], mask

        yt2 = yt[mask]
        yp2 = yp[mask]

        # Dtypes for sklearn stability
        if yt2.dtype not in (np.int8, np.int32, np.int64):
            yt2 = yt2.astype(np.int8, copy=False)

        if not np.issubdtype(yp2.dtype, np.floating):
            yp2 = yp2.astype(np.float64, copy=False)

        return yt2, yp2, mask

    def _prepare_ohe_proba(
        self, y_true_ohe: np.ndarray, y_pred_proba: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Backwards-compatible wrapper that drops the mask."""
        yt2, yp2, _mask = self._prepare_ohe_proba_with_mask(y_true_ohe, y_pred_proba)
        return yt2, yp2

    def _prepare_1d_labels(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        ignore_labels: tuple[int, ...] = (-1, -9),
        metric_name: str = "Discrete metric",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare and filter 1D label arrays for discrete metrics.

        Filters:
            - coerces to 1D
            - drops non-finite entries (if float)
            - drops ignored labels (e.g., -1/-9)

        Args:
            y_true (np.ndarray): Ground truth labels; shape (N,) or (N, L).
            y_pred (np.ndarray): Pred labels; shape (N,) or (N, L).
            ignore_labels (tuple[int, ...]): Labels to drop from scoring.
            metric_name (str): Used for warnings.

        Returns:
            tuple[np.ndarray, np.ndarray]: Filtered (y_true, y_pred). May be empty.
        """
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)

        if yt.ndim == 2:
            yt = yt.reshape(-1)
        else:
            yt = yt.reshape(-1)

        if yp.ndim == 2:
            yp = yp.reshape(-1)
        else:
            yp = yp.reshape(-1)

        if yt.shape[0] != yp.shape[0]:
            msg = f"{metric_name}: y_true/y_pred length mismatch {yt.shape[0]} vs {yp.shape[0]}."
            self.logger.error(msg)
            raise ValueError(msg)

        # Build mask
        mask = np.ones((yt.shape[0],), dtype=bool)

        # Non-finite handling (float labels happen sometimes)
        if np.issubdtype(yt.dtype, np.floating):
            mask &= np.isfinite(yt)
        if np.issubdtype(yp.dtype, np.floating):
            mask &= np.isfinite(yp)

        if ignore_labels:
            # Compare after safe cast attempt; if object dtype, just try vectorized compare
            for lab in ignore_labels:
                mask &= (yt != lab) & (yp != lab)

        if not np.any(mask):
            self.logger.warning(
                f"{metric_name}: no valid rows after filtering; returning empty arrays."
            )
            return yt[:0].astype(np.int64, copy=False), yp[:0].astype(
                np.int64, copy=False
            )

        yt = yt[mask]
        yp = yp[mask]

        # Final cast: use int64 to avoid int8 overflow / unexpected behavior with many labels
        try:
            yt = yt.astype(np.int64, copy=False)
            yp = yp.astype(np.int64, copy=False)
        except Exception:
            self.logger.warning(
                f"{metric_name}: non-numeric labels after filtering; returning empty arrays."
            )
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

        return yt, yp

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy; returns 0.0 if undefined."""
        yt, yp = self._prepare_1d_labels(y_true, y_pred, metric_name="Accuracy")
        if yt.size == 0:
            return 0.0
        return float(accuracy_score(yt, yp))

    def f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute F1; returns 0.0 if undefined."""
        yt, yp = self._prepare_1d_labels(y_true, y_pred, metric_name="F1")
        if yt.size == 0:
            return 0.0
        return float(f1_score(yt, yp, average=self.average, zero_division=0))

    def precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute precision; returns 0.0 if undefined."""
        yt, yp = self._prepare_1d_labels(y_true, y_pred, metric_name="Precision")
        if yt.size == 0:
            return 0.0
        return float(precision_score(yt, yp, average=self.average, zero_division=0))

    def recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute recall; returns 0.0 if undefined."""
        yt, yp = self._prepare_1d_labels(y_true, y_pred, metric_name="Recall")
        if yt.size == 0:
            return 0.0
        return float(recall_score(yt, yp, average=self.average, zero_division=0))

    def jaccard(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Jaccard; returns 0.0 if undefined."""
        yt, yp = self._prepare_1d_labels(y_true, y_pred, metric_name="Jaccard")
        if yt.size == 0:
            return 0.0
        return float(jaccard_score(yt, yp, average=self.average, zero_division=0))

    def mcc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute MCC; returns 0.0 if undefined."""
        yt, yp = self._prepare_1d_labels(y_true, y_pred, metric_name="MCC")
        if yt.size == 0:
            return 0.0
        # matthews_corrcoef can be 0 when single-class; that's fine
        return float(matthews_corrcoef(yt, yp))

    def _validate_metric_inputs_ohe(
        self,
        yt: np.ndarray,
        yp: np.ndarray,
        *,
        metric_name: str,
        require_at_least_two_present_classes: bool = False,
        filter_to_present_classes: bool = False,
        allow_average_none: bool = True,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int, np.ndarray]]:
        """Validate prepared (N, K) indicator + score matrices for metrics.

        Args:
            yt: Prepared indicator matrix (N, K), expected strict 0/1 with row-sum==1.
            yp: Prepared score/proba matrix (N, K), expected finite.
            metric_name: For logging messages.
            require_at_least_two_present_classes: If True, require >=2 classes present in yt.
                (ROC AUC typically needs this; AP does not strictly, but you may want it.)
            filter_to_present_classes: If True, drop columns (classes) with zero positives in yt.
                Helpful for multiclass ROC AUC (OVR) to avoid sklearn errors.
            allow_average_none: If False, disallow average=None (rarely needed).

        Returns:
            (yt_v, yp_v, K, present) if valid; otherwise None (caller should return 0.0).
        """

        def _warn(msg: str) -> None:
            self.logger.warning(f"{metric_name}: {msg}")

        # Basic shape checks (should already be true, but keep defensive)
        if yt.ndim != 2 or yp.ndim != 2:
            _warn(
                f"expects 2D arrays; got yt.ndim={yt.ndim}, yp.ndim={yp.ndim}. Returning 0.0."
            )
            return None

        if yt.shape[0] == 0:
            _warn("no valid rows after filtering; returning 0.0.")
            return None

        if yt.shape != yp.shape:
            _warn(f"shape mismatch yt={yt.shape}, yp={yp.shape}; returning 0.0.")
            return None

        # Finiteness (prepare already filtered, but double-check)
        try:
            if not np.isfinite(yt).all() or not np.isfinite(yp).all():
                _warn("NaN/Inf detected in yt or yp; returning 0.0.")
                return None
        except Exception:
            _warn("non-numeric yt/yp detected; returning 0.0.")
            return None

        # Indicator sanity (strict 0/1 and row sum==1)
        # This should be guaranteed by _prepare_ohe_proba; keep in case it's bypassed.
        if not (((yt == 0) | (yt == 1)).all() and (yt.sum(axis=1) == 1).all()):
            _warn("yt is not a strict one-hot indicator matrix; returning 0.0.")
            return None

        K = yt.shape[1]
        if K < 2:
            _warn(f"K={K} < 2 classes; returning 0.0.")
            return None

        present = np.where(yt.sum(axis=0) > 0)[0]
        if present.size == 0:
            _warn("no positive labels in any class; returning 0.0.")
            return None

        if require_at_least_two_present_classes and present.size < 2:
            _warn("fewer than 2 classes present in y_true; returning 0.0.")
            return None

        # Validate average setting (sklearn-compatible set)
        avg = getattr(self, "average", None)
        allowed = {"micro", "macro", "weighted", "samples"}
        if avg is None:
            if not allow_average_none:
                _warn("average=None is not allowed here; returning 0.0.")
                return None
        elif avg not in allowed:
            _warn(
                f"invalid average='{avg}'. Must be one of {sorted(allowed)} or None; returning 0.0."
            )
            return None

        if filter_to_present_classes and present.size < K:
            yt = yt[:, present]
            yp = yp[:, present]
            K = yt.shape[1]

            # After filtering, still require >=2 classes when asked
            if require_at_least_two_present_classes and K < 2:
                _warn("fewer than 2 classes remain after filtering; returning 0.0.")
                return None

        return yt, yp, K, present

    def roc_auc(self, y_true_ohe: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute ROC AUC (binary or multiclass OVR) robustly.

        Args:
            y_true_ohe (np.ndarray): One-hot encoded ground truth; shape (N_eval, K) or (N, L, K).
            y_pred_proba (np.ndarray): Predicted probabilities; shape (N_eval, K) or (N, L, K).

        Returns:
            float: ROC AUC score, or 0.0 if undefined / invalid.
        """

        def _warn_and_return(msg: str, exc: Exception | None = None) -> float:
            if exc is None:
                self.logger.warning(msg)
            else:
                self.logger.warning(f"{msg} Details: {exc}")
            return 0.0

        try:
            yt, yp = self._prepare_ohe_proba(y_true_ohe, y_pred_proba)
        except Exception as e:
            return _warn_and_return(
                "ROC AUC failed in _prepare_ohe_proba(); returning 0.0.", e
            )

        # ROC AUC needs at least two present classes
        # Important for multiclass OVR stability
        v = self._validate_metric_inputs_ohe(
            yt,
            yp,
            metric_name="ROC AUC",
            require_at_least_two_present_classes=True,
            filter_to_present_classes=True,
        )
        if v is None:
            return 0.0

        # NOTE: if filtered, K/present reflect filtered version
        yt, yp, K, present = v

        if K == 2:
            y_true_bin = yt[:, 1]
            y_score = yp[:, 1]

            # ROC AUC undefined if only one class present
            if np.unique(y_true_bin).size < 2:
                return _warn_and_return(
                    "ROC AUC undefined in binary case (only one class present); returning 0.0."
                )

            try:
                return float(roc_auc_score(y_true_bin, y_score))
            except ValueError as e:
                return _warn_and_return("ROC AUC failed binary case; returning 0.0.", e)
            except Exception as e:
                return _warn_and_return(
                    "ROC AUC crashed unexpectedly (binary); returning 0.0.", e
                )

        try:
            return float(roc_auc_score(yt, yp, average=self.average, multi_class="ovr"))
        except ValueError as e:
            return _warn_and_return("ROC AUC failed multiclass case; returning 0.0.", e)
        except Exception as e:
            return _warn_and_return(
                "ROC AUC crashed unexpectedly (multiclass); returning 0.0.", e
            )

    def average_precision(
        self, y_true_ohe: np.ndarray, y_pred_proba: np.ndarray
    ) -> float:
        """Compute Average Precision (binary or multiclass) robustly.

        Args:
            y_true_ohe (np.ndarray): One-hot encoded ground truth; shape (N_eval, K) or (N, L, K).
            y_pred_proba (np.ndarray): Predicted probabilities; shape (N_eval, K) or (N, L, K).

        Returns:
            float: Average Precision score, or 0.0 if undefined / invalid.
        """

        def _warn_and_return(msg: str, exc: Exception | None = None) -> float:
            if exc is None:
                self.logger.warning(msg)
            else:
                self.logger.warning(f"{msg} Details: {exc}")
            return 0.0

        try:
            yt, yp = self._prepare_ohe_proba(y_true_ohe, y_pred_proba)
        except Exception as e:
            return _warn_and_return(
                "Average Precision failed in _prepare_ohe_proba(); returning 0.0.", e
            )

        # AP can be defined with 1 present class, so no need to require >=2.
        v = self._validate_metric_inputs_ohe(
            yt,
            yp,
            metric_name="Average Precision",
            require_at_least_two_present_classes=False,
            filter_to_present_classes=False,
        )
        if v is None:
            return 0.0
        yt, yp, K, _present = v

        if K == 2:
            y_true_bin = yt[:, 1]
            y_score = yp[:, 1]

            # AP undefined if no positives in chosen positive class
            if y_true_bin.sum() == 0:
                return _warn_and_return(
                    "Average Precision undefined in binary case (no positives in class=1); returning 0.0."
                )

            try:
                return float(average_precision_score(y_true_bin, y_score))
            except ValueError as e:
                return _warn_and_return(
                    "Average Precision failed binary case; returning 0.0.", e
                )
            except Exception as e:
                return _warn_and_return(
                    "Average Precision crashed unexpectedly (binary); returning 0.0.", e
                )

        try:
            return float(average_precision_score(yt, yp, average=self.average))
        except ValueError as e:
            return _warn_and_return(
                "Average Precision failed multiclass case; returning 0.0.", e
            )
        except Exception as e:
            return _warn_and_return(
                "Average Precision crashed unexpectedly (multiclass); returning 0.0.", e
            )

    def pr_macro(self, y_true_ohe: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute macro-average precision (PR macro) robustly."""

        def _warn_and_return(msg: str, exc: Exception | None = None) -> float:
            if exc is None:
                self.logger.warning(msg)
            else:
                self.logger.warning(f"{msg} Details: {exc}")
            return 0.0

        try:
            yt, yp = self._prepare_ohe_proba(y_true_ohe, y_pred_proba)
        except Exception as e:
            return _warn_and_return(
                "PR macro failed in _prepare_ohe_proba(); returning 0.0.", e
            )

        v = self._validate_metric_inputs_ohe(
            yt,
            yp,
            metric_name="PR macro",
            require_at_least_two_present_classes=False,
            filter_to_present_classes=False,
        )
        if v is None:
            return 0.0
        yt, yp, K, _present = v

        if K == 2:
            y_true_bin = yt[:, 1]
            y_score = yp[:, 1]
            if y_true_bin.sum() == 0:
                return _warn_and_return(
                    "PR macro undefined in binary case (no positives in class=1); returning 0.0."
                )
            try:
                return float(average_precision_score(y_true_bin, y_score))
            except Exception as e:
                return _warn_and_return(
                    "PR macro failed binary case; returning 0.0.", e
                )

        try:
            return float(average_precision_score(yt, yp, average="macro"))
        except Exception as e:
            return _warn_and_return(
                "PR macro failed multiclass case; returning 0.0.", e
            )

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
        """Evaluate the model using various metrics with robust filtering/alignment."""
        if not all(
            isinstance(y, np.ndarray)
            for y in (y_true, y_pred, y_true_ohe, y_pred_proba)
        ):
            msg = "y inputs to 'Scorer.evaluate()' must be numpy arrays."
            self.logger.error(msg)
            raise TypeError(msg)

        # Prepare OHE/proba + get mask of valid rows
        try:
            yt_ohe, yp_proba, mask = self._prepare_ohe_proba_with_mask(
                y_true_ohe, y_pred_proba
            )
        except Exception as e:
            # If OHE/proba are broken, discrete metrics can still be computed
            self.logger.warning(
                f"evaluate(): OHE/proba prep failed ({e}); scoring discrete metrics only."
            )
            yt_ohe = np.asarray([], dtype=np.int8).reshape(0, 0)
            yp_proba = np.asarray([], dtype=np.float32).reshape(0, 0)
            mask = None

        # Align y_true/y_pred to the same flattened indexing used by OHE/proba when possible
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        yt = yt.reshape(-1) if yt.ndim != 1 else yt
        yp = yp.reshape(-1) if yp.ndim != 1 else yp

        if yt.shape[0] != yp.shape[0]:
            msg = f"y_true and y_pred length mismatch: {yt.shape[0]} vs {yp.shape[0]}."
            self.logger.error(msg)
            raise ValueError(msg)

        if mask is not None:
            # If the user passed arrays that correspond to the flattened (N*L) layout,
            # we can co-filter discrete labels to match the valid OHE/proba rows.
            if mask.shape[0] == yt.shape[0]:
                yt = yt[mask]
                yp = yp[mask]
            else:
                # Don’t explode; just warn and proceed without alignment
                self.logger.warning(
                    "evaluate(): could not align y_true/y_pred to OHE/proba mask "
                    f"(mask={mask.shape[0]} vs labels={yt.shape[0]}). Proceeding without alignment."
                )

        # Discrete metrics will additionally drop ignored labels (-1/-9) internally.
        metric_calculators = {
            "pr_macro": lambda: self.pr_macro(yt_ohe, yp_proba) if yt_ohe.size else 0.0,
            "roc_auc": lambda: self.roc_auc(yt_ohe, yp_proba) if yt_ohe.size else 0.0,
            "average_precision": lambda: (
                self.average_precision(yt_ohe, yp_proba) if yt_ohe.size else 0.0
            ),
            "accuracy": lambda: self.accuracy(yt, yp),
            "f1": lambda: self.f1(yt, yp),
            "precision": lambda: self.precision(yt, yp),
            "recall": lambda: self.recall(yt, yp),
            "mcc": lambda: self.mcc(yt, yp),
            "jaccard": lambda: self.jaccard(yt, yp),
        }

        if objective_mode:
            if isinstance(tune_metric, (list, tuple)):
                invalid = [tm for tm in tune_metric if tm not in metric_calculators]
                if invalid:
                    msg = (
                        f"Invalid tune_metric(s) provided: {invalid}. "
                        f"Valid options are: {list(metric_calculators.keys())}."
                    )
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
            metrics = {name: fn() for name, fn in metric_calculators.items()}

        return {k: float(v) for k, v in metrics.items()}
