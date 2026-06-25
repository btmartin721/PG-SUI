from __future__ import annotations

import io
import logging
from pathlib import Path

import optuna

from pgsui.impute.unsupervised.base import BaseNNImputer
from pgsui.utils.logging_utils import (
    configure_optuna_best_trial_logger,
    format_duration,
    get_pgsui_logger,
    reset_logger_handlers,
    summarize_optuna_study,
)


def test_model_scoped_logger_writes_model_specific_file(tmp_path: Path) -> None:
    prefix = tmp_path / "logger_run"
    logger = get_pgsui_logger(
        "pgsui.impute.unsupervised.imputers",
        prefix=prefix,
        model_name="ImputeVAE",
        verbose=True,
        reset_handlers=True,
        to_console=False,
    )

    try:
        logger.info("model scoped message")
        for handler in logger.handlers:
            handler.flush()

        log_dir = Path(f"{prefix}_output") / "logs"
        model_log = log_dir / "pgsui.impute.unsupervised.imputers.ImputeVAE.log"
        base_log = log_dir / "pgsui.impute.unsupervised.base.log"

        assert model_log.exists()
        assert "model scoped message" in model_log.read_text()
        assert not base_log.exists()
    finally:
        reset_logger_handlers(logger)


def test_base_nn_imputer_uses_model_scoped_logger(tmp_path: Path) -> None:
    prefix = tmp_path / "base_run"
    imputer = BaseNNImputer(
        model_name="ImputeAutoencoder",
        genotype_data=object(),
        prefix=str(prefix),
        verbose=True,
    )

    try:
        assert imputer.logger.name == (
            "pgsui.impute.unsupervised.imputers.ImputeAutoencoder"
        )

        log_dir = Path(f"{prefix}_output") / "logs"
        model_log = log_dir / "pgsui.impute.unsupervised.imputers.ImputeAutoencoder.log"
        base_log = log_dir / "pgsui.impute.unsupervised.base.log"

        assert model_log.exists()
        assert "Using PyTorch device" in model_log.read_text()
        assert not base_log.exists()
    finally:
        reset_logger_handlers(imputer.logger)


def test_format_duration_compacts_elapsed_seconds() -> None:
    assert format_duration(None) == "n/a"
    assert format_duration(3.25) == "3.250s"
    assert format_duration(65.5) == "1m 05.50s"
    assert format_duration(3665.5) == "1h 1m 05.50s"


def test_summarize_optuna_study_counts_trial_states() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        if trial.number == 1:
            raise optuna.exceptions.TrialPruned("expected prune")
        if trial.number == 2:
            raise RuntimeError("expected failure")
        return trial.suggest_float("x", 0.0, 1.0)

    study.optimize(objective, n_trials=3, catch=(RuntimeError,))
    stats = summarize_optuna_study(study, planned_trials=3, wall_time_s=1.5)

    assert stats.total_trials == 3
    assert stats.completed_trials == 1
    assert stats.pruned_trials == 1
    assert stats.failed_trials == 1
    assert stats.trials_per_minute == 40.0


def test_optuna_lifecycle_logger_can_forward_without_echo() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    echo_stream = io.StringIO()
    sink_stream = io.StringIO()
    sink_logger = logging.getLogger("tests.logging_utils.optuna.forward")
    reset_logger_handlers(sink_logger)
    sink_logger.setLevel(logging.INFO)
    sink_logger.propagate = False
    handler = logging.StreamHandler(sink_stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    sink_logger.addHandler(handler)

    try:
        optlog = configure_optuna_best_trial_logger(
            stream=echo_stream,
            logger_name="tests.logging_utils.optuna.best",
            forward_logger=sink_logger,
            echo=False,
        )
        study = optuna.create_study(direction="maximize")
        optlog.start(study, 2)
        study.optimize(
            lambda trial: trial.suggest_float("x", 0.0, 1.0),
            n_trials=2,
            callbacks=[optlog.callback],
        )
        optlog.finish(study)

        output = sink_stream.getvalue()
        assert echo_stream.getvalue() == ""
        assert "Begin Optuna Hyperparameter Tuning" in output
        assert "[Optuna] Starting study" in output
        assert "[Optuna] Best overall" in output
    finally:
        reset_logger_handlers(sink_logger)
        reset_logger_handlers("tests.logging_utils.optuna.best")
