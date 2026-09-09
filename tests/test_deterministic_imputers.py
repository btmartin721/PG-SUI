from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("snpio")
from snpio import GenotypeEncoder

from pgsui import (
    ImputeAutoencoder,
    ImputeMostFrequent,
    ImputeNLPCA,
    ImputeRefAllele,
    ImputeUBP,
    ImputeVAE,
    MostFrequentConfig,
    RefAlleleConfig,
)
from pgsui.data_processing.splitting import train_validation_test_indices


def _expected_shape(genotype_data) -> tuple[int, int]:
    encoder = GenotypeEncoder(genotype_data)
    return np.asarray(encoder.genotypes_012).shape


def _assert_decoded_strings(imputer, expected_shape: tuple[int, int]) -> None:
    decoded = imputer.transform()
    assert decoded.shape == expected_shape
    assert decoded.dtype.kind in {"U", "S", "O"}
    iupac_codes = ["A", "C", "G", "T", "R", "Y", "S", "W", "K", "M"]
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


def test_all_imputers_use_canonical_seeded_test_rows(
    example_genotype_data, tmp_path
) -> None:
    """All six benchmark models must select identical validation/test rows."""
    n_samples = _expected_shape(example_genotype_data)[0]
    expected_train, expected_validation, expected_test = train_validation_test_indices(
        n_samples, validation_split=0.3, seed=42
    )

    neural_models = (ImputeUBP, ImputeNLPCA, ImputeVAE, ImputeAutoencoder)
    for model_class in neural_models:
        imputer = object.__new__(model_class)
        imputer.validation_split = 0.3
        imputer.seed = 42
        train_idx, validation_idx, test_idx = imputer._train_val_test_split(
            np.empty((n_samples, 1), dtype=np.float32)
        )
        np.testing.assert_array_equal(train_idx, expected_train)
        np.testing.assert_array_equal(validation_idx, expected_validation)
        np.testing.assert_array_equal(test_idx, expected_test)

    deterministic_configs = (
        (ImputeMostFrequent, MostFrequentConfig.from_preset("fast")),
        (ImputeRefAllele, RefAlleleConfig.from_preset("fast")),
    )
    deterministic_imputers = []
    for index, (model_class, config) in enumerate(deterministic_configs):
        config.io.prefix = str(tmp_path / f"deterministic_{index}")
        config.io.seed = 42
        config.io.verbose = False
        config.plot.show = False
        config.train.validation_split = 0.3
        config.sim.simulate_missing = True
        config.sim.sim_strategy = "random"
        config.sim.sim_prop = 0.3
        imputer = model_class(example_genotype_data, config=config)
        imputer.fit()
        deterministic_imputers.append(imputer)

        np.testing.assert_array_equal(imputer.train_idx_, expected_train)
        np.testing.assert_array_equal(imputer.val_idx_, expected_validation)
        np.testing.assert_array_equal(imputer.test_idx_, expected_test)

    np.testing.assert_array_equal(
        deterministic_imputers[0].sim_mask_global_,
        deterministic_imputers[1].sim_mask_global_,
    )
    np.testing.assert_array_equal(
        deterministic_imputers[0].sim_mask_test_only_,
        deterministic_imputers[1].sim_mask_test_only_,
    )


@pytest.mark.parametrize(
    ("model_class", "config_class"),
    (
        (ImputeMostFrequent, MostFrequentConfig),
        (ImputeRefAllele, RefAlleleConfig),
    ),
)
def test_deterministic_explicit_test_indices_remain_authoritative(
    model_class, config_class, example_genotype_data, tmp_path
) -> None:
    cfg = config_class.from_preset("fast")
    cfg.io.prefix = str(tmp_path / model_class.__name__)
    cfg.io.verbose = False
    cfg.plot.show = False
    cfg.split.test_indices = [0, 2]

    imputer = model_class(example_genotype_data, config=cfg)
    train_idx, validation_idx, test_idx = imputer._make_train_validation_test_split()

    np.testing.assert_array_equal(test_idx, np.array([0, 2]))
    assert validation_idx.size == 0
    assert not np.intersect1d(train_idx, test_idx).size
