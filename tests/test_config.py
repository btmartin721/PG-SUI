from __future__ import annotations

import os
from pathlib import Path

import pytest

from pgsui.data_processing.config import (
    apply_dot_overrides,
    dataclass_to_yaml,
    load_yaml_to_dataclass,
)
from pgsui.data_processing.containers import MostFrequentConfig, RefAlleleConfig
from pgsui.impute.deterministic.imputers.mode import ensure_mostfrequent_config
from pgsui.impute.deterministic.imputers.ref_allele import ensure_refallele_config


def test_apply_dot_overrides_coerces_and_updates() -> None:
    cfg = MostFrequentConfig()

    updated = apply_dot_overrides(
        cfg,
        {
            "io.prefix": "demo",
            "io.verbose": "true",
            "split.test_size": "0.3",
            "algo.default": "2",
        },
    )

    assert updated.io.prefix == "demo"
    assert updated.io.verbose is True
    assert pytest.approx(updated.split.test_size) == 0.3
    assert updated.algo.default == 2


def test_apply_dot_overrides_unknown_key_raises() -> None:
    cfg = MostFrequentConfig()

    with pytest.raises(KeyError):
        apply_dot_overrides(cfg, {"nonexistent.field": 1})


def test_load_yaml_to_dataclass_merges_with_env_and_overlays(tmp_path: Path) -> None:
    base = MostFrequentConfig.from_preset("fast")
    yaml_text = dataclass_to_yaml(base)
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml_text
        + "\n"
        + "\n".join(
            [
                "io:",
                "  prefix: ${PGSUI_PREFIX:default}",
                "  verbose: false",
                "split:",
                "  test_size: 0.45",
            ]
        ),
        encoding="utf-8",
    )

    os.environ["PGSUI_PREFIX"] = "env_overridden"

    cfg = load_yaml_to_dataclass(
        str(path),
        MostFrequentConfig,
        base=base,
        overlays={"plot": {"fmt": "png"}},
    )

    assert cfg.io.prefix == "env_overridden"
    assert cfg.io.verbose is False  # YAML overrides preset
    assert pytest.approx(cfg.split.test_size) == 0.45
    assert cfg.plot.fmt == "png"


def test_ensure_mostfrequent_config_accepts_dict() -> None:
    cfg = ensure_mostfrequent_config(
        {"preset": "fast", "split": {"test_size": 0.4}, "algo": {"default": 1}}
    )
    assert isinstance(cfg, MostFrequentConfig)
    assert pytest.approx(cfg.split.test_size) == 0.4
    assert cfg.algo.default == 1
    assert cfg.io.verbose is True  # carried over from preset


def test_ensure_refallele_config_path_roundtrip(tmp_path: Path) -> None:
    cfg = RefAlleleConfig()
    cfg.io.prefix = "from_yaml"
    cfg.algo.missing = -9

    yaml_path = tmp_path / "ref.yml"
    yaml_path.write_text(dataclass_to_yaml(cfg), encoding="utf-8")

    loaded = ensure_refallele_config(str(yaml_path))
    assert isinstance(loaded, RefAlleleConfig)
    assert loaded.io.prefix == "from_yaml"
    assert loaded.algo.missing == -9
