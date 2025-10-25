from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("snpio")
from snpio import GenotypeEncoder

from pgsui import (
    ImputeMostFrequent,
    ImputeRefAllele,
    MostFrequentConfig,
    RefAlleleConfig,
)


def _expected_shape(genotype_data) -> tuple[int, int]:
    encoder = GenotypeEncoder(genotype_data)
    return np.asarray(encoder.genotypes_012).shape


def _assert_decoded_strings(imputer, expected_shape: tuple[int, int]) -> None:
    decoded = imputer.transform()
    assert decoded.shape == expected_shape
    assert decoded.dtype.kind in {"U", "S", "O"}
    iupac_codes = ["A", "C", "G", "T", "R", "Y", "S", "W", "K", "M", "B", "D", "H", "V"]
    assert all(np.isin(np.unique(decoded), iupac_codes, assume_unique=True))
    assert np.count_nonzero(decoded == "N") == 0


def test_most_frequent_global(example_genotype_data, tmp_path) -> None:
    cfg = MostFrequentConfig.from_preset("fast")
    cfg.io.prefix = str(tmp_path / "mode_global")
    cfg.io.verbose = False
    cfg.plot.show = False
    cfg.algo.by_populations = False

    imputer = ImputeMostFrequent(example_genotype_data, config=cfg)
    imputer.fit()
    _assert_decoded_strings(imputer, _expected_shape(example_genotype_data))


def test_most_frequent_by_population(example_genotype_data, tmp_path) -> None:
    cfg = MostFrequentConfig.from_preset("fast")
    cfg.io.prefix = str(tmp_path / "mode_pops")
    cfg.io.verbose = False
    cfg.plot.show = False
    cfg.algo.by_populations = True

    imputer = ImputeMostFrequent(example_genotype_data, config=cfg)
    imputer.fit()
    _assert_decoded_strings(imputer, _expected_shape(example_genotype_data))


def test_ref_allele_imputer(example_genotype_data, tmp_path) -> None:
    cfg = RefAlleleConfig.from_preset("fast")
    cfg.io.prefix = str(tmp_path / "refallele")
    cfg.io.verbose = False
    cfg.plot.show = False

    imputer = ImputeRefAllele(example_genotype_data, config=cfg)
    imputer.fit()
    _assert_decoded_strings(imputer, _expected_shape(example_genotype_data))
