import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from read_input.read_input import GenotypeData
from utils.misc import generate_012_genotypes


class ImputeVAE(GenotypeData):
    def __init__(self, *, genotype_data=None, gt=None):

        super().__init__()

        if genotype_data is None and gt is None:
            raise TypeError("genotype_data and gt cannot both be NoneType")

        if genotype_data is not None and gt is not None:
            raise TypeError("genotype_data and gt cannot both be used")

        if genotype_data is not None:
            self.X = genotype_data.genotypes_nparray

        elif gt is not None:
            self.X = gt

        np.nan_to_num(self.X, copy=False, nan=-9.0)
        self.X = self.X.astype(str)
        self.X[self.X == "-9.0"] = "none"

        self.X_enc = self.encode_onehot(self.X)
        print(self.X_enc)

    def encode_onehot(self, X):
        print(X)
        ohe = OneHotEncoder()
        Xenc = ohe.fit_transform(X).toarray()
        missing_mask = np.where(X == "none", 1.0, 0.0)

        Xmiss = np.zeros(Xenc.shape)
        for col in range(missing_mask.shape[1]):
            ncat = ohe.categories_[col]
            

        print(test)

        sys.exit()
        return missing_enc
