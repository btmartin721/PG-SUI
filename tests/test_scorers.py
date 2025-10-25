from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("snpio")

from pgsui.utils.scorers import Scorer


def test_scorer_rejects_invalid_average() -> None:
    with pytest.raises(ValueError):
        Scorer(prefix="unit", average="invalid")  # type: ignore[arg-type]


def test_scorer_evaluate_and_objective_mode() -> None:
    scorer = Scorer(prefix="unit", average="macro")

    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_true_ohe = np.eye(3, dtype=int)[y_true]
    y_pred_proba = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.2, 0.5, 0.3],
            [0.2, 0.6, 0.2],
        ]
    )

    metrics = scorer.evaluate(y_true, y_pred, y_true_ohe, y_pred_proba)

    expected_keys = {
        "accuracy",
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "average_precision",
        "pr_macro",
    }
    assert expected_keys.issubset(metrics)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    obj_metrics = scorer.evaluate(
        y_true,
        y_pred,
        y_true_ohe,
        y_pred_proba,
        objective_mode=True,
        tune_metric="accuracy",
    )
    assert obj_metrics == {"accuracy": pytest.approx(0.75)}


def test_scorer_roc_auc_single_class_defaults() -> None:
    scorer = Scorer(prefix="unit", average="macro")
    y_true = np.array([1, 1, 1])
    y_pred_proba = np.array([[0.4, 0.6], [0.3, 0.7], [0.1, 0.9]])

    assert scorer.roc_auc(y_true, y_pred_proba) == pytest.approx(0.5)
