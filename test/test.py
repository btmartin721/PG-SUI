import sys
import os
import copy
import unittest
import pprint
from snpio import GenotypeData
from pgsui import *
from pgsui.utils.misc import HiddenPrints


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

            # Create a SimGenotypeDataTransformer instance and use it
            # to simulate missing data
            self.transformer = SimGenotypeDataTransformer(
                genotype_data=self.genotype_data,
                prop_missing=0.2,
                strategy="random",
            )
            self.transformer.fit(self.genotype_data.genotypes_012(fmt="numpy"))
            self.simulated_data = copy.deepcopy(self.genotype_data)

            self.simulated_data.genotypes_012 = self.transformer.transform(
                self.genotype_data.genotypes_012(fmt="numpy")
            )

    def _test_class(self, class_instance, do_gridsearch=False):
        print(f"\nMETHOD: {class_instance.__name__}\n")

        if do_gridsearch:
            # Do a simple test.
            if class_instance in [ImputeRandomForest, ImputeXGBoost]:
                param_grid = {"n_estimators": [50, 100]}  # Do a simple test
            elif class_instance in [
                ImputeVAE,
                ImputeStandardAutoEncoder,
                ImputeNLPCA,
                ImputeUBP,
            ]:
                param_grid = {"dropout_rate": [0.1, 0.2]}
            elif class_instance == ImputeKNN:
                param_grid = {"n_neighbors": [5, 8]}
        else:
            param_grid = None

        instance = class_instance(
            self.simulated_data,
            gridparams=param_grid,
        )
        imputed_data = instance.imputed.genotypes_012(fmt="numpy")

        # Test that the imputed values are close to the original values
        accuracy = self.transformer.accuracy(
            self.genotype_data.genotypes_012(fmt="numpy"), imputed_data
        )

        (
            auc_roc_scores,
            precision_scores,
            recall_scores,
            avg_precision_scores,
        ) = self.transformer.auc_roc_pr_ap(
            self.genotype_data.genotypes_012(fmt="numpy"), imputed_data
        )

        pprint.PrettyPrinter(indent=4, sort_dicts=True).pprint(
            f"OVERALL ACCURACY: {accuracy}"
        )
        pprint.PrettyPrinter(indent=4, sort_dicts=True).pprint(
            f"AUC-ROC PER CLASS: {dict(zip(range(3), auc_roc_scores))}"
        )
        pprint.PrettyPrinter(indent=4, sort_dicts=True).pprint(
            f"PRECISION PER CLASS: {dict(zip(range(3), precision_scores))}"
        )
        pprint.PrettyPrinter(indent=4, sort_dicts=True).pprint(
            f"RECALL PER CLASS: {dict(zip(range(3), recall_scores))}"
        )
        pprint.PrettyPrinter(indent=4, sort_dicts=True).pprint(
            f"AVERAGE PRECISION PER CLASS: {dict(zip(range(3), avg_precision_scores))}"
        )
        print("\n")

    def test_ImputeKNN(self):
        self._test_class(ImputeKNN)

    def test_ImputeRandomForest(self):
        self._test_class(ImputeRandomForest)

    def test_ImputeXGBoost(self):
        self._test_class(ImputeXGBoost)

    def test_ImputeVAE(self):
        self._test_class(ImputeVAE)

    def test_ImputeStandardAutoEncoder(self):
        self._test_class(ImputeStandardAutoEncoder)

    def test_ImputeUBP(self):
        self._test_class(ImputeUBP)

    def test_ImputeNLPCA(self):
        self._test_class(ImputeNLPCA)

    def test_ImputeKNN_grid(self):
        self._test_class(ImputeKNN, do_gridsearch=True)

    def test_ImputeRandomForest_grid(self):
        self._test_class(ImputeRandomForest, do_gridsearch=True)

    def test_ImputeXGBoost_grid(self):
        self._test_class(ImputeXGBoost, do_gridsearch=True)

    def test_ImputeVAE_grid(self):
        self._test_class(ImputeVAE, do_gridsearch=True)

    def test_ImputeStandardAutoEncoder_grid(self):
        self._test_class(ImputeStandardAutoEncoder, do_gridsearch=True)

    def test_ImputeUBP_grid(self):
        self._test_class(ImputeUBP, do_gridsearch=True)

    def test_ImputeNLPCA_grid(self):
        self._test_class(ImputeNLPCA, do_gridsearch=True)

    def test_ImputePhylo(self):
        self._test_class(ImputePhylo)

    def test_ImputeAlleleFreq(self):
        self._test_class(ImputeAlleleFreq)

    def test_ImputeMF(self):
        self._test_class(ImputeMF)

    def test_ImputeRefAllele(self):
        self._test_class(ImputeRefAllele)


if __name__ == "__main__":
    unittest.main()
