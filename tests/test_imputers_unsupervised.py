from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

pytest.importorskip("snpio")
from snpio import GenotypeEncoder

from pgsui import (
    AutoencoderConfig,
    ImputeAutoencoder,
    ImputeNLPCA,
    ImputeUBP,
    ImputeVAE,
    NLPCAConfig,
    UBPConfig,
    VAEConfig,
)


def _expected_shape(genotype_data) -> tuple[int, int]:
    encoder = GenotypeEncoder(genotype_data)
    return np.asarray(encoder.genotypes_012).shape


def _assert_unsupervised_transform(model, expected_shape: tuple[int, int]) -> None:
    decoded = model.transform()
    assert decoded.shape == expected_shape
    assert decoded.dtype.kind in {"U", "S", "O"}

    iupac_codes = ["A", "C", "G", "T", "R", "Y", "S", "W", "K", "M", "B", "D", "H", "V"]
    assert all(np.isin(np.unique(decoded), iupac_codes, assume_unique=True))
    assert np.count_nonzero(decoded == "N") == 0


def _configure_common(cfg, tmp_path, prefix: str) -> None:
    cfg.io.prefix = str(tmp_path / prefix)
    cfg.io.verbose = False
    cfg.plot.show = False
    cfg.tune.enabled = False


def test_autoencoder_end_to_end(example_genotype_data, tmp_path) -> None:
    cfg = AutoencoderConfig.from_preset("fast")
    _configure_common(cfg, tmp_path, "ae_run")
    cfg.train.max_epochs = 5
    cfg.train.min_epochs = 1
    cfg.train.early_stop_gen = 2
    cfg.train.batch_size = 8

    model = ImputeAutoencoder(genotype_data=example_genotype_data, config=cfg)

    with pytest.raises(NotFittedError):
        model.transform()

    model.fit()
    _assert_unsupervised_transform(model, _expected_shape(example_genotype_data))


def test_vae_end_to_end(example_genotype_data, tmp_path) -> None:
    cfg = VAEConfig.from_preset("fast")
    _configure_common(cfg, tmp_path, "vae_run")
    cfg.train.max_epochs = 5
    cfg.train.min_epochs = 1
    cfg.train.early_stop_gen = 2
    cfg.train.batch_size = 8
    cfg.vae.kl_beta = 0.5

    model = ImputeVAE(genotype_data=example_genotype_data, config=cfg)

    with pytest.raises(NotFittedError):
        model.transform()

    model.fit()
    _assert_unsupervised_transform(model, _expected_shape(example_genotype_data))


def test_nlpca_end_to_end(example_genotype_data, tmp_path) -> None:
    cfg = NLPCAConfig.from_preset("fast")
    _configure_common(cfg, tmp_path, "nlpca_run")
    cfg.train.max_epochs = 3
    cfg.train.min_epochs = 1
    cfg.train.batch_size = 8
    cfg.model.latent_dim = 2

    model = ImputeNLPCA(genotype_data=example_genotype_data, config=cfg)

    with pytest.raises(NotFittedError):
        model.transform()

    model.fit()
    _assert_unsupervised_transform(model, _expected_shape(example_genotype_data))


def test_ubp_end_to_end(example_genotype_data, tmp_path) -> None:
    cfg = UBPConfig.from_preset("fast")
    _configure_common(cfg, tmp_path, "ubp_run")
    cfg.train.max_epochs = 5
    cfg.train.min_epochs = 1
    cfg.train.batch_size = 8
    cfg.model.latent_dim = 3

    model = ImputeUBP(genotype_data=example_genotype_data, config=cfg)

    with pytest.raises(NotFittedError):
        model.transform()

    model.fit()
    _assert_unsupervised_transform(model, _expected_shape(example_genotype_data))
