## PG-SUI package by Bradley T. Martin and Tyler K. Chafin
## E-mail: evobio721@gmail.com

from importlib.metadata import PackageNotFoundError, version

from ._version import version as __version__

from pgsui.data_processing.containers import (
    AutoencoderConfig,
    HGBConfig,
    MostFrequentConfig,
    NLPCAConfig,
    RefAlleleConfig,
    RFConfig,
    UBPConfig,
    VAEConfig,
)
from pgsui.impute.deterministic.imputers.mode import ImputeMostFrequent
from pgsui.impute.deterministic.imputers.ref_allele import ImputeRefAllele
from pgsui.impute.supervised.imputers.hist_gradient_boosting import (
    ImputeHistGradientBoosting,
)
from pgsui.impute.supervised.imputers.random_forest import ImputeRandomForest
from pgsui.impute.unsupervised.imputers.autoencoder import ImputeAutoencoder
from pgsui.impute.unsupervised.imputers.nlpca import ImputeNLPCA
from pgsui.impute.unsupervised.imputers.ubp import ImputeUBP
from pgsui.impute.unsupervised.imputers.vae import ImputeVAE

__all__ = [
    "ImputeAutoencoder",  # Unsupervised imputer classes
    "ImputeVAE",
    "ImputeNLPCA",
    "ImputeUBP",
    "ImputeRandomForest",  # Supervised imputer classes
    "ImputeHistGradientBoosting",
    "ImputeRefAllele",  # Deterministic imputer classes
    "ImputeMostFrequent",
    "AutoencoderConfig",  # Unsupervised imputer configs
    "VAEConfig",
    "NLPCAConfig",
    "UBPConfig",
    "MostFrequentConfig",  # Deterministic imputer configs
    "RefAlleleConfig",
    "RFConfig",  # Supervised imputer configs
    "HGBConfig",
    "__version__",
]
