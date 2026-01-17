from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, TextIO

import optuna
from optuna.trial import TrialState


def configure_logger(
    logger: logging.Logger,
    *,
    verbose: bool = False,
    debug: bool = False,
    quiet_level: int = logging.ERROR,
) -> logging.Logger:
    """Force a logger and its handlers to respect verbose/debug controls."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = quiet_level

    logger.setLevel(level)
    for handler in getattr(logger, "handlers", ()):
        handler.setLevel(level)
    return logger


class BestTrialOnlyHandler(logging.Handler):
    """Logging handler that emits output only for marked Optuna lifecycle events."""

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        *,
        starting_template: str = (
            "[Optuna] Starting study={study_name} | direction(s)={directions} | "
            "n_trials={n_trials} | sampler={sampler} | pruner={pruner}"
        ),
        best_template: str = (
            "[Optuna] New best | study={study_name} | trial={trial_number} | "
            "value(s)={best_value} | completed={completed_trials}/{total_trials} | "
            "elapsed_s={elapsed_s:.3f} | params={params}"
        ),
        overall_template: str = (
            "[Optuna] Best overall | study={study_name} | trial={trial_number} | "
            "value(s)={best_value} | completed={completed_trials}/{total_trials} | "
            "elapsed_s={elapsed_s:.3f} | params={params}"
        ),
        terminator: str = "\n",
    ) -> None:
        super().__init__(level=logging.INFO)
        self._stream = stream if stream is not None else sys.stdout
        self._starting_template = starting_template
        self._best_template = best_template
        self._overall_template = overall_template
        self._terminator = terminator

    @property
    def stream(self) -> TextIO:
        return self._stream

    @property
    def terminator(self) -> str:
        return self._terminator

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # We use getattr to safely extract custom fields added via the 'extra' dict.
            if getattr(record, "is_starting_study", False):
                msg = self._starting_template.format(
                    study_name=getattr(record, "study_name", "?"),
                    directions=getattr(record, "directions", "?"),
                    n_trials=getattr(record, "n_trials", "?"),
                    sampler=getattr(record, "sampler", "?"),
                    pruner=getattr(record, "pruner", "?"),
                )
            elif getattr(record, "is_new_best", False):
                msg = self._best_template.format(
                    study_name=getattr(record, "study_name", "?"),
                    trial_number=getattr(record, "trial_number", "?"),
                    best_value=getattr(record, "best_value", "N/A"),
                    params=getattr(record, "params", {}),
                    completed_trials=getattr(record, "completed_trials", "?"),
                    total_trials=getattr(record, "total_trials", "?"),
                    s_trial_mean=getattr(record, "s_trial_mean", float("nan")),
                    elapsed_s=getattr(record, "elapsed_s", float("nan")),
                    n_trials=getattr(record, "n_trials", "?"),
                    percent=getattr(record, "percent", float("nan")),
                )
            elif getattr(record, "is_best_overall", False):
                msg = self._overall_template.format(
                    study_name=getattr(record, "study_name", "?"),
                    trial_number=getattr(record, "trial_number", "?"),
                    best_value=getattr(record, "best_value", "N/A"),
                    completed_trials=getattr(record, "completed_trials", "?"),
                    total_trials=getattr(record, "total_trials", "?"),
                    elapsed_s=getattr(record, "elapsed_s", float("nan")),
                    params=getattr(record, "params", {}),
                )
            else:
                return

            self._stream.write(msg + self._terminator)
            self.flush()
        except Exception:
            # Prevent logging failures from crashing the optimization study
            self.handleError(record)


@dataclass(frozen=True)
class OptunaBestTrialLogger:
    """Façade that provides start/best/summary printing + an Optuna callback.

    Usage:
        logger = OptunaBestTrialLogger.configure(...)
        logger.start(study, n_trials)
        study.optimize(..., callbacks=[logger.callback])
        logger.finish(study)
    """

    logger: logging.Logger
    callback: Callable[[optuna.Study, optuna.trial.FrozenTrial], None]
    start: Callable[[optuna.Study, int], None]
    finish: Callable[[optuna.Study], None]


def configure_optuna_best_trial_logger(
    *,
    stream: Optional[TextIO] = None,
    logger_name: str = "optuna.best",
    disable_default_optuna_handler: bool = True,
    min_delta: float = 0.0,
    starting_template: str = (
        "[Optuna] Starting study={study_name} | "
        "direction(s)={directions} | "
        "n_trials={n_trials} | sampler={sampler} | pruner={pruner}"
    ),
    best_template: str = (
        "[Optuna] Trials completed = {completed_trials}/{n_trials} ({percent:.2f}%) | "
        "Avg trial (s) = {s_trial_mean:.3f} | "
        "New best: trial #{trial_number} -> {best_value}"
    ),
    overall_template: str = (
        "[Optuna] Best overall | study={study_name} | "
        "completed={completed_trials}/{total_trials} | "
        "elapsed_s={elapsed_s:.3f} | best_params={params} | -> best_value(s)={best_value}"
    ),
) -> OptunaBestTrialLogger:
    """Configures a custom logger for Optuna that highlights best trials."""
    if disable_default_optuna_handler:
        optuna.logging.disable_default_handler()

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Ensure we don't duplicate handlers if configured multiple times
    logger.handlers = [
        h for h in logger.handlers if not isinstance(h, BestTrialOnlyHandler)
    ]
    logger.addHandler(
        BestTrialOnlyHandler(
            stream=stream,
            starting_template=starting_template,
            best_template=best_template,
            overall_template=overall_template,
        )
    )

    lock = threading.Lock()

    @dataclass
    class _State:
        best_value_scalar: Optional[float] = (
            None  # Only used for single-objective optimization
        )
        t0: Optional[float] = None
        planned_trials: Optional[int] = None

    state = _State()

    def _format_value(value: float | Sequence[float]) -> str:
        """Helper to format scalars or lists of floats cleanly."""
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(f"{v:.6g}" for v in value) + "]"
        return f"{value:.6g}"

    def _is_scalar_improvement(
        direction: str, new: float, old: Optional[float]
    ) -> bool:
        if old is None:
            return True
        if direction == "minimize":
            return new <= (old - min_delta)
        return new >= (old + min_delta)

    def _get_handler() -> Optional[BestTrialOnlyHandler]:
        return next(
            (h for h in logger.handlers if isinstance(h, BestTrialOnlyHandler)), None
        )

    def start(study: optuna.Study, n_trials: int) -> None:
        handler = _get_handler()
        if handler is not None:
            for msg in [
                "",
                "---------------------------------------------------------",
                "* Begin Optuna Hyperparameter Tuning                    *",
                "---------------------------------------------------------",
                "",
            ]:
                handler.stream.write(msg + handler.terminator)

        with lock:
            state.t0 = time.time()
            state.planned_trials = int(n_trials)
            state.best_value_scalar = None

        sampler = getattr(study, "sampler", None)
        pruner = getattr(study, "pruner", None)
        # Unified access: 'directions' works for both single and multi-objective in recent Optuna
        directions_list = study.directions

        # Format directions string based on count
        if len(directions_list) == 1:
            directions_str = str(directions_list[0])
        else:
            directions_str = str(directions_list)

        logger.info(
            "starting",
            extra={
                "is_starting_study": True,
                "study_name": study.study_name,
                "directions": directions_str,
                "n_trials": n_trials,
                "sampler": type(sampler).__name__ if sampler is not None else "None",
                "pruner": type(pruner).__name__ if pruner is not None else "None",
            },
        )

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # 1. Basic validation: ignore pruned or incomplete trials
        if trial.state != TrialState.COMPLETE or trial.values is None:
            return

        directions = study.directions
        is_multiobj = len(directions) > 1

        # 2. Determine if this trial is a "New Best"
        is_new_best = False
        current_values = trial.values  # Always returns a list of floats

        if is_multiobj:
            # Multi-objective: Check if the current trial is in the Pareto front (best_trials)
            # This is O(N) where N is the size of the Pareto front, usually small.
            best_trials = study.best_trials
            if any(t.number == trial.number for t in best_trials):
                is_new_best = True
        else:
            # Single-objective: Check against the single best trial
            best_trial = study.best_trial
            if best_trial.number == trial.number:
                # Check min_delta logic (only applies to single objective scalar comparison)
                val = current_values[0]
                with lock:
                    if _is_scalar_improvement(
                        str(directions[0]), val, state.best_value_scalar
                    ):
                        state.best_value_scalar = val
                        is_new_best = True

        if not is_new_best:
            return

        # 3. Calculate statistics
        with lock:
            t0 = state.t0
            planned = state.planned_trials

        trials = study.trials
        completed = sum(1 for t in trials if t.state == TrialState.COMPLETE)
        total = len(trials)

        elapsed_s = time.time() - t0 if t0 is not None else 0.0
        avg_trial_s = (elapsed_s / completed) if completed > 0 else float("nan")
        percent = (
            (100.0 * completed / planned) if planned and planned > 0 else float("nan")
        )

        # 4. Format value for display (Scalar vs Vector)
        display_value = _format_value(
            current_values if is_multiobj else current_values[0]
        )

        logger.info(
            "new best",
            extra={
                "is_new_best": True,
                "study_name": study.study_name,
                "trial_number": trial.number,
                "best_value": display_value,  # String: "0.123" or "[0.123, 0.456]"
                "params": dict(trial.params),
                "completed_trials": completed,
                "total_trials": total,
                "s_trial_mean": avg_trial_s,
                "elapsed_s": elapsed_s,
                "n_trials": planned,
                "percent": percent,
            },
        )

    def finish(study: optuna.Study) -> None:
        trials = study.trials
        completed = sum(1 for t in trials if t.state == TrialState.COMPLETE)
        total = len(trials)

        directions = study.directions
        is_multiobj = len(directions) > 1

        # Determine best trial(s) to report
        best_params: Any = {}
        best_val_display: str = "?"
        best_number: Any = "?"

        if completed > 0:
            if is_multiobj:
                # For MO, we report the number of solutions on the Pareto front
                best_trials = study.best_trials
                best_number = f"{len(best_trials)} (Pareto Front Size)"
                # We cannot easily display params for all, so we display "N/A - See Front"
                # or perhaps the parameters of the *first* trial on the front as an example.
                if best_trials:
                    best_params = f"(Example from Trial #{best_trials[0].number}) {dict(best_trials[0].params)}"
                    # Format a list of value vectors? Too long. Just say 'Vector'
                    best_val_display = "Pareto Front Vectors"
            else:
                best_trial = study.best_trial
                best_number = best_trial.number
                best_params = dict(best_trial.params)
                best_val_display = _format_value(
                    best_trial.value if best_trial.value is not None else float("nan")
                )

        with lock:
            t0 = state.t0
            planned = state.planned_trials

        elapsed_s = (time.time() - t0) if isinstance(t0, (int, float)) else float("nan")
        total_trials_target = planned if planned is not None else total

        logger.info(
            "best overall",
            extra={
                "is_best_overall": True,
                "study_name": study.study_name,
                "trial_number": best_number,
                "best_value": best_val_display,
                "completed_trials": completed,
                "total_trials": total_trials_target,
                "elapsed_s": elapsed_s,
                "params": best_params,
            },
        )

        handler = _get_handler()
        if handler is not None:
            for msg in [
                "",
                "---------------------------------------------------------",
                "* Completed Optuna Hyperparameter Tuning!               *",
                "---------------------------------------------------------",
                "",
            ]:
                handler.stream.write(msg + handler.terminator)

    return OptunaBestTrialLogger(
        logger=logger, callback=callback, start=start, finish=finish
    )
