"""Helpers for handling single- and multi-objective Optuna studies."""

from __future__ import annotations

import optuna
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial


def representative_best_trial(study: optuna.Study) -> FrozenTrial:
    """Return one reproducible best trial from an Optuna study.

    Single-objective studies use Optuna's canonical ``best_trial``. For a
    multi-objective study, PG-SUI selects the Pareto-optimal trial with the
    best primary-objective value and then the lowest trial number. The primary
    objective is the first configured tuning metric.

    Args:
        study: Completed Optuna study.

    Returns:
        The selected completed trial.

    Raises:
        RuntimeError: If the study has no successful trial with objective
            values.
    """
    try:
        directions = tuple(study.directions)
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError("Optuna study has no optimization directions.") from exc

    if len(directions) == 1:
        try:
            return study.best_trial
        except (AttributeError, ValueError, RuntimeError) as exc:
            raise RuntimeError("Optuna study has no successful trial.") from exc

    try:
        pareto_trials = [trial for trial in study.best_trials if trial.values]
    except (AttributeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("Optuna study has no successful Pareto trial.") from exc

    if not pareto_trials:
        raise RuntimeError("Optuna study has no successful Pareto trial.")

    primary_direction = directions[0]
    if primary_direction is StudyDirection.MAXIMIZE:
        return min(pareto_trials, key=lambda trial: (-trial.values[0], trial.number))
    return min(pareto_trials, key=lambda trial: (trial.values[0], trial.number))


def trial_objective_values(trial: FrozenTrial) -> tuple[float, ...]:
    """Return a frozen trial's objective values as a nonempty float tuple.

    Args:
        trial: Completed Optuna trial.

    Returns:
        Objective values in configured order.

    Raises:
        RuntimeError: If the trial has no objective values.
    """
    if trial.values is not None:
        values = tuple(float(value) for value in trial.values)
    elif trial.value is not None:
        values = (float(trial.value),)
    else:
        values = ()

    if not values:
        raise RuntimeError(f"Optuna trial {trial.number} has no objective values.")
    return values
