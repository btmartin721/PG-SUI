import unittest
import pprint
from snpio import GenotypeData
from pgsui import *
from pgsui.utils.misc import HiddenPrints
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    f1_score,
    average_precision_score,
    accuracy_score,
)

from sklearn.preprocessing import label_binarize

from sklearn.utils.class_weight import compute_class_weight


# Initialize dictionaries to store metrics for all methods
all_accuracies = {}
all_auc_rocs = {}
all_precisions = {}
all_recalls = {}
all_avg_precisions = {}
all_f1s = {}


def plot_scoring_metrics():
    """
    Plot the accumulated scoring metrics for all test methods in separate subplots.

    Args:
        None

    Returns:
        None: The function generates a grouped bar chart displaying the scoring metrics.
    """

    metrics = [
        "Accuracy",
        "AUC-ROC",
        "Precision",
        "Recall",
        "Average Precision",
        "F1 Score",
    ]
    metric_dicts = [
        all_accuracies,
        all_auc_rocs,
        all_precisions,
        all_recalls,
        all_avg_precisions,
        all_f1s,
    ]

    num_metrics = len(metrics)
    fig, axes = plt.subplots(2, num_metrics // 2, figsize=(20, 20))

    # Loop through each metric and its corresponding dictionary
    colcount = 0
    rowcount = 0

    for i, (metric, metric_dict) in enumerate(zip(metrics, metric_dicts)):
        if i > 0 and i % num_metrics // 2 == 0:
            rowcount += 1
            colcount = 0

        methods = list(metric_dict.keys())
        values = list(metric_dict.values())

        # Find the index of the highest bar
        highest_bar_idx = np.argmax(values)

        # Create the bar plot on the i-th subplot
        bars = axes[rowcount, colcount].bar(methods, values, color="gray")

        # Color the highest bar in orange
        bars[highest_bar_idx].set_color("orange")

        # Rotate x-axis labels
        axes[rowcount, colcount].tick_params(axis="x", rotation=90)

        # Annotate the bars with the actual values
        for j, v in enumerate(values):
            axes[rowcount, colcount].text(
                j, v, f"{v:.2f}", ha="center", va="bottom"
            )

        axes[rowcount, colcount].set_title(metric)
        axes[rowcount, colcount].set_ylabel("Score")
        colcount += 1

    plt.suptitle("Scoring Metrics for All Methods")

    fig.savefig("scores.png", facecolor="white", bbox_inches="tight")


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
            self.simulated_data = self.genotype_data.copy()

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
            sample_weights=None,
        )
        imputed_data = instance.imputed.genotypes_int

        # Test that the imputed values are close to the original values
        # accuracy = self.transformer.accuracy(
        #     self.genotype_data.genotypes_012(fmt="numpy"), imputed_data
        # )

        (
            accuracy,
            auc_roc_scores,
            precision_scores,
            recall_scores,
            avg_precision_scores,
            f1,
        ) = self._scoring_metrics(
            self.genotype_data.genotypes_int, imputed_data
        )

        pprint.PrettyPrinter(indent=4, sort_dicts=True).pprint(
            f"ACCURACY: {accuracy}"
        )
        pprint.PrettyPrinter(indent=4, sort_dicts=True).pprint(
            f"AUC-ROC: {auc_roc_scores}"
        )
        pprint.PrettyPrinter(indent=4, sort_dicts=True).pprint(
            f"PRECISION: {precision_scores}"
        )
        pprint.PrettyPrinter(indent=4, sort_dicts=True).pprint(
            f"RECALL: {recall_scores}"
        )
        pprint.PrettyPrinter(indent=4, sort_dicts=True).pprint(
            f"AVERAGE PRECISION: {avg_precision_scores}"
        )
        pprint.PrettyPrinter(indent=4, sort_dicts=True).pprint(
            f"F1 SCORE: {f1}"
        )
        print("\n")

        # Store metrics
        all_accuracies[class_instance.__name__] = accuracy
        all_auc_rocs[class_instance.__name__] = auc_roc_scores
        all_precisions[class_instance.__name__] = precision_scores
        all_recalls[class_instance.__name__] = recall_scores
        all_avg_precisions[class_instance.__name__] = avg_precision_scores
        all_f1s[class_instance.__name__] = f1
        plot_scoring_metrics()

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

    def _scoring_metrics(self, y_true, y_pred):
        """Calcuate AUC-ROC, Precision-Recall, and Average Precision (AP).

        Args:
            X_true (np.ndarray): True values.

            X_pred (np.ndarray): Imputed values.

        Returns:
            List[float]: List of AUC-ROC scores in order of: 0,1,2.
            List[float]: List of precision scores in order of: 0,1,2.
            List[float]: List of recall scores in order of: 0,1,2.
            List[float]: List of average precision scores in order of 0,1,2.

        """
        y_true = y_true[self.transformer.sim_missing_mask_]
        y_pred = y_pred[self.transformer.sim_missing_mask_]

        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2])

        accuracy = accuracy_score(y_true, y_pred)

        # AUC-ROC score
        auc_roc = roc_auc_score(y_true_bin, y_pred_bin, average="weighted")

        # Precision-recall score
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average="weighted"
        )

        # Average precision score
        avg_precision = average_precision_score(
            y_true_bin, y_pred_bin, average="weighted"
        )

        f1 = f1_score(y_true_bin, y_pred_bin, average="weighted")

        return (accuracy, auc_roc, precision, recall, avg_precision, f1)


if __name__ == "__main__":
    unittest.main()
