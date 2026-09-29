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

    iupac_codes = ["A", "C", "G", "T", "R", "Y", "S", "W", "K", "M"]
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


@pytest.mark.parametrize(
    ("imputer_cls", "config_cls"),
    [(ImputeNLPCA, NLPCAConfig), (ImputeUBP, UBPConfig)],
    ids=["nlpca", "ubp"],
)
def test_pca_embedding_init_hides_simulated_missing(
    imputer_cls, config_cls, example_genotype_data, tmp_path, monkeypatch
) -> None:
    """PCA warm start must not see genotypes at simulated-missing cells.

    Validation and test samples keep their PCA-initialized embeddings until
    evaluation-time projection, so computing the initialization from the
    uncorrupted matrix would leak the genotypes that are later scored.
    """
    captured: list[np.ndarray] = []
    original = imputer_cls._get_pca_embedding_init

    def _spy(self, X_full, train_idx, latent_dim):
        captured.append(np.array(X_full, copy=True))
        return original(self, X_full, train_idx, latent_dim)

    monkeypatch.setattr(imputer_cls, "_get_pca_embedding_init", _spy)

    cfg = config_cls.from_preset("fast")
    _configure_common(cfg, tmp_path, f"{imputer_cls.__name__}_pca_init")
    cfg.sim.simulate_missing = True
    cfg.sim.sim_strategy = "random"
    cfg.sim.sim_prop = 0.2
    cfg.train.max_epochs = 2
    cfg.train.min_epochs = 1
    cfg.train.early_stop_gen = 1
    cfg.train.batch_size = 8

    model = imputer_cls(genotype_data=example_genotype_data, config=cfg)
    model.fit()

    sim_mask = np.asarray(model.sim_mask_, dtype=bool)
    assert sim_mask.any()
    assert captured, "PCA embedding initialization was never called."
    for X_full in captured:
        # Simulated-missing genotypes are hidden from the warm start ...
        assert np.all(X_full[sim_mask] == -1)
        # ... and every other cell matches the input data.
        np.testing.assert_array_equal(X_full[~sim_mask], model.ground_truth_[~sim_mask])


def test_nlpca_trains_on_observed_cells_only(example_genotype_data, tmp_path) -> None:
    """NLPCA's training targets never expose simulated- or originally-missing cells.

    Training targets are the corrupted matrix, so held-out cells are -1 and
    masked out; observed cells match the clean genotypes. Also checks that the
    fitted model imputes every missing genotype.
    """
    cfg = NLPCAConfig.from_preset("fast")
    _configure_common(cfg, tmp_path, "nlpca_targets")
    cfg.sim.simulate_missing = True
    cfg.sim.sim_strategy = "random"
    cfg.sim.sim_prop = 0.2
    cfg.train.max_epochs = 3
    cfg.train.min_epochs = 1
    cfg.train.early_stop_gen = 2
    cfg.train.batch_size = 8

    model = ImputeNLPCA(genotype_data=example_genotype_data, config=cfg)
    model.fit()

    _, y, m = (t.numpy() for t in model.train_loader_.dataset.tensors)
    held_out = model.sim_mask_train_ | model.orig_mask_train_
    assert held_out.any()
    assert not m[held_out].any()
    assert np.all(y[held_out] == -1)
    np.testing.assert_array_equal(y[m], model.y_train_[m])

    _assert_unsupervised_transform(model, _expected_shape(example_genotype_data))
