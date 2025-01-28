## PG-SUI package by Bradley T. Martin and Tyler K. Chafin
## E-mail: evobio721@gmail.com
## Version 0.1, completed 13-Dec-2021

import os
import warnings

from pgsui.utils.misc import get_processor_name

# Requires scikit-learn-intellex package
if get_processor_name().strip().startswith("Intel"):
    try:
        from sklearnex import patch_sklearn

        patch_sklearn()
        intelex = True
    except (ImportError, TypeError):
        warnings.warn(
            "Intel CPU detected but scikit-learn-intelex is not installed. We recommend installing it to speed up computation if your hardware supports it."
        )
        intelex = False
else:
    intelex = False

os.environ["intelex"] = str(intelex)

from pgsui.data_processing.transformers import SimGenotypeDataTransformer
from pgsui.impute.estimators import (
    ImputeKNN,
    ImputeNLPCA,
    ImputeRandomForest,
    ImputeStandardAutoEncoder,
    ImputeXGBoost,
)
from pgsui.impute.simple_imputers import (
    ImputeAlleleFreq,
    ImputeMF,
    ImputePhylo,
    ImputeRefAllele,
)
from pgsui.impute.unsupervised.imputers.cnn import ImputeCNN
from pgsui.impute.unsupervised.imputers.vae import ImputeVAE
from pgsui.impute.unsupervised.imputers.lstm import ImputeLSTM
from pgsui.impute.unsupervised.imputers.ubp import ImputeUBP

__all__ = [
    "ImputeUBP",
    "ImputeVAE",
    "ImputeCNN",
    "ImputeLSTM",
    "ImputeXGBoost",
    "ImputeStandardAutoEncoder",
    "ImputeRandomForest",
    "ImputeNLPCA",
    "ImputeKNN",
    "SimGenotypeDataTransformer",
    "ImputePhylo",
    "ImputeMF",
    "ImputeAlleleFreq",
    "ImputeRefAllele",
]
