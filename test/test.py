import sys
import os

import unittest
from pgsui.impute.estimators import (
    ImputeKNN,
    ImputeRandomForest,
    ImputeGradientBoosting,
    ImputeXGBoost,
    ImputeLightGBM,
    ImputeVAE,
    ImputeStandardAutoEncoder,
    ImputeUBP,
    ImputeNLPCA,
)

from snpio import GenotypeData
from pgsui.data_processing.transformers import SimGenotypeDataTransformer
import numpy as np


class TestMyClasses(unittest.TestCase):
    def setUp(self):
        self.genotype_data = GenotypeData(
            filename="pgsui/example_data/phylip_files/test_n100.phy",
            popmapfile="pgsui/example_data/popmap_files/test.popmap",
            guidetree="pgsui/example_data/trees/test.tre",
            qmatrix_iqtree="pgsui/example_data/trees/test.qmat",
            siterates_iqtree="pgsui/example_data/trees/test.rate",
            prefix="test_imputer",
            force_popmap=True,
            plot_format="png",
        )
        # Create a SimGenotypeDataTransformer instance and use it to simulate missing data
        self.transformer = SimGenotypeDataTransformer(
            genotype_data=self.genotype_data, prop_missing=0.1
        )
        self.transformer.fit(self.genotype_data.snp_data)
        self.simulated_data = self.transformer.transform(
            self.genotype_data.snp_data
        )

    def test_class(self, class_instance):
        instance = class_instance(self.simulated_data)
        imputed_data = instance.imputer.snp_data

        # Test that there are no missing values in the imputed data
        self.assertFalse(np.isnan(imputed_data).any())

        # Test that the imputed values are close to the original values
        accuracy = self.transformer.accuracy(
            self.genotype_data.snp_data, imputed_data
        )
        self.assertGreaterEqual(accuracy, 0.9)  # adjust this as needed

    def test_ImputeKNN(self):
        self.test_class(ImputeKNN)

    def test_ImputeRandomForest(self):
        self.test_class(ImputeRandomForest)

    def test_ImputeGradientBoosting(self):
        self.test_class(ImputeGradientBoosting)

    def test_ImputeXGBoost(self):
        self.test_class(ImputeXGBoost)

    def test_ImputeLightGBM(self):
        self.test_class(ImputeLightGBM)

    def test_ImputeVAE(self):
        self.test_class(ImputeVAE)

    def test_ImputeStandardAutoEncoder(self):
        self.test_class(ImputeStandardAutoEncoder)

    def test_ImputeUBP(self):
        self.test_class(ImputeUBP)

    def test_ImputeNLPCA(self):
        self.test_class(ImputeNLPCA)


if __name__ == "__main__":
    unittest.main()
