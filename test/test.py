import sys
import os
import copy
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
from pgsui.impute.simple_imputers import (
    ImputePhylo,
    ImputeNMF, 
    ImputeAlleleFreq
)

from snpio import GenotypeData
from pgsui.data_processing.transformers import SimGenotypeDataTransformer
import numpy as np


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
class TestMyClasses(unittest.TestCase):
    def setUp(self):
        with HiddenPrints():
            self.genotype_data = GenotypeData(
                filename="pgsui/example_data/phylip_files/test_n100.phy",
                popmapfile="pgsui/example_data/popmaps/test.popmap",
                guidetree="pgsui/example_data/trees/test.tre",
                qmatrix="pgsui/example_data/trees/test.qmat",
                siterates="pgsui/example_data/trees/test_siterates_n100.txt",
                prefix="test_imputer",
                force_popmap=True,
                plot_format="png",
            )
        
            # Create a SimGenotypeDataTransformer instance and use it to simulate missing data
            self.transformer = SimGenotypeDataTransformer(
                genotype_data=self.genotype_data, prop_missing=0.2, 
                strategy="random"
            )
            self.transformer.fit(self.genotype_data.genotypes_012(fmt="numpy"))
            self.simulated_data = copy.deepcopy(self.genotype_data)
            
            self.simulated_data.genotypes_012 = self.transformer.transform(
                self.genotype_data.genotypes_012(fmt="numpy")
            )

    def _test_class(self, class_instance):
        print(f"METHOD: {class_instance.__name__}")
        #with HiddenPrints():
        instance = class_instance(self.simulated_data)
        imputed_data = instance.imputed.genotypes_012(fmt="numpy")
 
         # Test that the imputed values are close to the original values
        accuracy = self.transformer.accuracy(
            self.genotype_data.genotypes_012(fmt="numpy"), imputed_data
        )
        print(f"ACCURACY: {accuracy}")

    # def test_ImputeKNN(self):
    #     self._test_class(ImputeKNN)

    # def test_ImputeRandomForest(self):
    #     self._test_class(ImputeRandomForest)

    # def test_ImputeXGBoost(self):
    #     self._test_class(ImputeXGBoost)

    def test_ImputeVAE(self):
        self._test_class(ImputeVAE)

    def test_ImputeStandardAutoEncoder(self):
        self._test_class(ImputeStandardAutoEncoder)

    def test_ImputeUBP(self):
        self._test_class(ImputeUBP)

    def test_ImputeNLPCA(self):
        self._test_class(ImputeNLPCA)

    # def test_ImputePhylo(self):
    #     self._test_class(ImputePhylo)

    # def test_ImputeAlleleFreq(self):
    #     self._test_class(ImputeAlleleFreq)

    # def test_ImputeNMF(self):
    #     self._test_class(ImputeNMF)


if __name__ == "__main__":
    unittest.main()
