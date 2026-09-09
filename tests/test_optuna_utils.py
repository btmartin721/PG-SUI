from __future__ import annotations

import logging

import optuna
import pandas as pd
import pytest

from pgsui.utils.optuna_utils import (
    representative_best_trial,
    trial_objective_values,
)
from pgsui.utils.plotting import Plotting


def test_representative_best_trial_single_objective() -> None:
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=4)

    trial = representative_best_trial(study)

    assert trial.number == study.best_trial.number
    assert trial_objective_values(trial) == (pytest.approx(study.best_value),)


def test_representative_best_trial_uses_primary_pareto_objective() -> None:
    study = optuna.create_study(directions=("maximize", "maximize", "maximize"))
    study.enqueue_trial({"x": 0.2})
    study.enqueue_trial({"x": 0.8})
    study.optimize(
        lambda trial: (
            trial.suggest_float("x", 0.0, 1.0),
            1.0 - trial.params["x"],
            0.5,
        ),
        n_trials=2,
    )

    trial = representative_best_trial(study)

    assert trial.params["x"] == pytest.approx(0.8)
    assert trial_objective_values(trial) == pytest.approx((0.8, 0.2, 0.5))


def test_multiobjective_study_without_completed_trials_raises() -> None:
    study = optuna.create_study(directions=("maximize", "maximize"))

    with pytest.raises(RuntimeError, match="successful Pareto trial"):
        representative_best_trial(study)


def test_queue_multiqc_tuning_accepts_three_objectives(monkeypatch) -> None:
    study = optuna.create_study(directions=("maximize", "maximize", "maximize"))
    study.enqueue_trial({"x": 0.75})
    study.optimize(
        lambda trial: (
            trial.suggest_float("x", 0.0, 1.0),
            0.6,
            0.7,
        ),
        n_trials=1,
    )

    queued: dict[str, object] = {}

    class MultiQCRecorder:
        @staticmethod
        def queue_linegraph(**kwargs) -> None:
            queued["linegraph"] = kwargs

        @staticmethod
        def queue_table(**kwargs) -> None:
            queued["table"] = kwargs

    monkeypatch.setattr("pgsui.utils.plotting.SNPioMultiQC", MultiQCRecorder)
    plotter = object.__new__(Plotting)
    plotter.use_multiqc = True
    plotter.model_name = "TestModel"
    plotter.multiqc_section = "Test section"
    plotter.logger = logging.getLogger("test.optuna.multiqc")

    plotter._queue_multiqc_tuning(
        study=study,
        model_name="TestModel",
        target_name="F1 value",
    )

    linegraph = queued["linegraph"]
    assert len(linegraph["data"]) == 3
    table = queued["table"]
    assert isinstance(table["df"], pd.Series)
    assert table["df"]["objective_1"] == pytest.approx(0.75)
    assert table["df"]["objective_2"] == pytest.approx(0.6)
    assert table["df"]["objective_3"] == pytest.approx(0.7)
