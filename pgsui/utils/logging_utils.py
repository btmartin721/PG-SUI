from __future__ import annotations

import logging
import math
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterator, Optional, Sequence, TextIO

import optuna
from optuna.trial import TrialState
from snpio.utils.logging import LoggerManager


PGSUI_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s | %(message)s"
)
PGSUI_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def model_logger_name(base_name: str, model_name: str | None = None) -> str:
    """Return a stable logger name, optionally scoped to a model class."""
    if not model_name:
        return base_name

    safe_model_name = str(model_name).strip().replace("/", "_").replace(" ", "_")
    if not safe_model_name or base_name.endswith(f".{safe_model_name}"):
        return base_name
    return f"{base_name}.{safe_model_name}"


def reset_logger_handlers(logger_or_name: logging.Logger | str) -> None:
    """Detach and close handlers for a logger before reconfiguring it."""
    logger = (
        logging.getLogger(logger_or_name)
        if isinstance(logger_or_name, str)
        else logger_or_name
    )
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def get_pgsui_logger(
    name: str,
    *,
    prefix: str | Path | None = "",
    model_name: str | None = None,
    verbose: bool = False,
    debug: bool = False,
    quiet_level: int = logging.ERROR,
    log_file: str | Path | None = None,
    propagate: bool = False,
    reset_handlers: bool = False,
    to_console: bool = True,
    to_file: bool = True,
) -> logging.Logger:
    """Create a PG-SUI logger with consistent formatting and levels."""
    logger_name = model_logger_name(name, model_name)
    if reset_handlers:
        reset_logger_handlers(logger_name)

    logman = LoggerManager(
        logger_name,
        prefix=str(prefix or ""),
        debug=debug,
        verbose=verbose,
        log_file=log_file,
        to_console=to_console,
        to_file=to_file,
        log_format=PGSUI_LOG_FORMAT,
        date_format=PGSUI_DATE_FORMAT,
    )
    logger = configure_logger(
        logman.get_logger(), verbose=verbose, debug=debug, quiet_level=quiet_level
    )
    logger.propagate = propagate
    return logger


def format_duration(seconds: float | int | None) -> str:
    """Format elapsed seconds as a compact human-readable duration."""
    if seconds is None:
        return "n/a"

    value = float(seconds)
    if not math.isfinite(value):
        return "n/a"

    value = max(0.0, value)
    hours, rem = divmod(value, 3600.0)
    minutes, secs = divmod(rem, 60.0)

    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {secs:05.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {secs:05.2f}s"
    return f"{secs:.3f}s"


def log_section(
    logger: logging.Logger,
    title: str,
    *,
    level: int = logging.INFO,
    width: int = 72,
    char: str = "=",
) -> None:
    """Log a compact section header."""
    clean_title = f" {title.strip()} "
    line = clean_title.center(width, char[:1] or "=")
    logger.log(level, line)


def log_key_values(
    logger: logging.Logger,
    rows: Sequence[tuple[str, Any]],
    *,
    title: str | None = None,
    level: int = logging.INFO,
) -> None:
    """Log aligned key/value rows."""
    if title:
        logger.log(level, "%s:", title)

    if not rows:
        return

    key_width = max(len(str(key)) for key, _ in rows)
    for key, value in rows:
        rendered = "n/a" if value is None else str(value)
        logger.log(level, "  %-*s : %s", key_width, str(key), rendered)


@dataclass
class RuntimeLog:
    """Mutable runtime record yielded by ``log_runtime``."""

    label: str
    started_at: float
    elapsed_s: float = 0.0

    def stop(self) -> float:
        """Stop the timer and return elapsed seconds."""
        self.elapsed_s = time.perf_counter() - self.started_at
        return self.elapsed_s


@contextmanager
def log_runtime(
    logger: logging.Logger,
    label: str,
    *,
    level: int = logging.INFO,
    failure_level: int = logging.ERROR,
    log_start: bool = True,
    log_finish: bool = True,
) -> Iterator[RuntimeLog]:
    """Log start/failure/completion timing around a block."""
    timer = RuntimeLog(label=label, started_at=time.perf_counter())
    if log_start:
        logger.log(level, "%s started.", label)

    try:
        yield timer
    except Exception:
        timer.stop()
        logger.log(
            failure_level,
            "%s failed after %s.",
            label,
            format_duration(timer.elapsed_s),
            exc_info=True,
        )
        raise
    else:
        timer.stop()
        if log_finish:
            logger.log(
                level,
                "%s completed in %s.",
                label,
                format_duration(timer.elapsed_s),
            )


@dataclass(frozen=True)
class OptunaStudyStats:
    """Compact Optuna execution statistics for logging."""

    planned_trials: int | None
    total_trials: int
    completed_trials: int
    pruned_trials: int
    failed_trials: int
    running_trials: int
    waiting_trials: int
    wall_time_s: float | None
    mean_trial_s: float | None
    std_trial_s: float | None
    min_trial_s: float | None
    max_trial_s: float | None

    @property
    def completed_percent(self) -> float | None:
        """Completed trials as a percentage of planned trials."""
        if not self.planned_trials:
            return None
        return 100.0 * self.completed_trials / self.planned_trials

    @property
    def trials_per_minute(self) -> float | None:
        """Completed-trial throughput."""
        if not self.wall_time_s or self.wall_time_s <= 0.0:
            return None
        return 60.0 * self.completed_trials / self.wall_time_s

    def as_log_rows(self) -> list[tuple[str, Any]]:
        """Return aligned rows for ``log_key_values``."""
        completed = str(self.completed_trials)
        if self.completed_percent is not None:
            completed = f"{completed} ({self.completed_percent:.1f}% of planned)"

        throughput = self.trials_per_minute
        return [
            ("planned trials", self.planned_trials),
            ("created trials", self.total_trials),
            ("completed trials", completed),
            ("pruned trials", self.pruned_trials),
            ("failed trials", self.failed_trials),
            ("running trials", self.running_trials),
            ("waiting trials", self.waiting_trials),
            ("wall time", format_duration(self.wall_time_s)),
            ("mean trial time", format_duration(self.mean_trial_s)),
            ("trial time stddev", format_duration(self.std_trial_s)),
            ("fastest trial", format_duration(self.min_trial_s)),
            ("slowest trial", format_duration(self.max_trial_s)),
            (
                "throughput",
                "n/a"
                if throughput is None
                else f"{throughput:.2f} completed trials/min",
            ),
        ]


def summarize_optuna_study(
    study: optuna.Study,
    *,
    planned_trials: int | None,
    wall_time_s: float | None,
) -> OptunaStudyStats:
    """Collect execution statistics from an Optuna study."""
    trials = list(study.trials)
    durations = [
        float(trial.duration.total_seconds())
        for trial in trials
        if trial.duration is not None
    ]

    return OptunaStudyStats(
        planned_trials=planned_trials,
        total_trials=len(trials),
        completed_trials=sum(1 for t in trials if t.state == TrialState.COMPLETE),
        pruned_trials=sum(1 for t in trials if t.state == TrialState.PRUNED),
        failed_trials=sum(1 for t in trials if t.state == TrialState.FAIL),
        running_trials=sum(1 for t in trials if t.state == TrialState.RUNNING),
        waiting_trials=sum(1 for t in trials if t.state == TrialState.WAITING),
        wall_time_s=wall_time_s,
        mean_trial_s=mean(durations) if durations else None,
        std_trial_s=pstdev(durations) if len(durations) > 1 else None,
        min_trial_s=min(durations) if durations else None,
        max_trial_s=max(durations) if durations else None,
    )


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
        forward_logger: Optional[logging.Logger] = None,
        echo: bool = True,
        terminator: str = "\n",
    ) -> None:
        super().__init__(level=logging.INFO)
        self._stream = stream if stream is not None else sys.stdout
        self._forward_logger = forward_logger
        self._echo = echo
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

    def write_line(self, message: str) -> None:
        """Write a rendered Optuna lifecycle message to configured sinks."""
        if self._echo:
            self._stream.write(message + self._terminator)
            self.flush()
        if self._forward_logger is not None:
            self._forward_logger.info(message)

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

            self.write_line(msg)
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
    forward_logger: Optional[logging.Logger] = None,
    echo: bool = True,
) -> OptunaBestTrialLogger:
    """Configures a custom logger for Optuna that highlights best trials."""
    if disable_default_optuna_handler:
        optuna.logging.disable_default_handler()

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if forward_logger is logger:
        forward_logger = None

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
            forward_logger=forward_logger,
            echo=echo,
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
                handler.write_line(msg)

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
                    best_params = (
                        f"(Example from Trial #{best_trials[0].number}) "
                        f"{dict(best_trials[0].params)}"
                    )
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
                handler.write_line(msg)

    return OptunaBestTrialLogger(
        logger=logger, callback=callback, start=start, finish=finish
    )
