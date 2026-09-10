"""Regression tests for PG-SUI's SNPio encoding integration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pgsui import ImputeMostFrequent, ImputeRefAllele, cli
from pgsui.impute.unsupervised.base import BaseNNImputer


class RecordingEncoder:
    """Small test double that records canonical decode calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, bool]] = []

    def decode_012(self, values: np.ndarray, *, is_nuc: bool) -> np.ndarray:
        self.calls.append((values, is_nuc))
        return np.full(values.shape, "A", dtype="<U1")


@pytest.mark.parametrize(
    ("decoder", "encoder_attribute"),
    (
        (BaseNNImputer.decode_012, "pgenc"),
        (ImputeMostFrequent.decode_012, "encoder"),
        (ImputeRefAllele.decode_012, "encoder"),
    ),
)
def test_imputer_decoders_delegate_to_snpio(decoder, encoder_attribute: str) -> None:
    encoder = RecordingEncoder()
    holder = SimpleNamespace(**{encoder_attribute: encoder})
    values = np.asarray([[0, 1, 2]], dtype=np.int8)

    observed = decoder(holder, values, is_nuc=True)

    np.testing.assert_array_equal(observed, np.asarray([["A", "A", "A"]]))
    assert encoder.calls == [(values, True)]


def test_cli_uses_run_specific_snpio_prefix(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_reader(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(cli, "VCFReader", fake_reader)

    genotype_data, tree_parser = cli.build_genotype_data(
        input_path="dataset.vcf",
        fmt="vcf",
        popmap_path=None,
        treefile=None,
        qmatrix=None,
        siterates=None,
        force_popmap=False,
        debug=False,
        include_pops=None,
        prefix="results/dataset__random__cpu",
        plot_format="pdf",
    )

    assert genotype_data is not None
    assert tree_parser is None
    assert captured["prefix"] == "results/dataset__random__cpu_snpio"
