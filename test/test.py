import sys
import os
import copy
import unittest
import pprint
from snpio import GenotypeData
from pgsui import *
from pgsui.simulators.simulate import SNPulator, SNPulatorConfig, SNPulatoRate
from pgsui.utils.misc import HiddenPrints
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import demes

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    f1_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
)

import seaborn as sns

from sklearn.preprocessing import label_binarize


import matplotlib.pyplot as plt
import numpy as np

# Initialize dictionaries to store metrics for all methods
all_accuracies_overall = {}
all_accuracies = {}
all_auc_rocs = {}
all_precisions = {}
all_recalls = {}
all_avg_precisions = {}
all_f1s = {}

import matplotlib.pyplot as plt
import numpy as np


def plot_class_distribution(y_masked, filename="class_distribution.png"):
    """
    Plot and save the distribution of classes 0, 1, and 2 in the masked true labels.

    Args:
        y_masked (np.ndarray): Masked true labels, a 1D numpy array.
        filename (str): Filename to save the plot.

    Returns:
        None: The function saves the plot to disk.
    """

    # Count the occurrences of each class
    unique_elements, counts_elements = np.unique(y_masked, return_counts=True)

    # Create a bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(unique_elements, counts_elements, color=["red", "green", "blue"])

    # Annotate the bars with the actual counts
    for i, count in enumerate(counts_elements):
        plt.text(
            unique_elements[i], count, str(count), ha="center", va="bottom"
        )

    plt.xlabel("Class Label")
    plt.ylabel("Frequency")
    plt.title("Distribution of Classes in Masked y_true")

    # Save the plot
    plt.savefig(filename)


def plot_and_save_confusion_matrix(y_true, y_pred, filename="conf_mat.png"):
    """
    Plot and save the confusion matrix.

    Args:
        y_true (np.ndarray): True labels, a 1D numpy array.
        y_pred (np.ndarray): Predicted labels, a 1D numpy array.
        filename (str): Filename to save the plot.

    Returns:
        None: The function saves the plot to disk.
    """
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=[0, 1, 2],
        yticklabels=[0, 1, 2],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save to disk
    plt.savefig(filename)


def plot_scoring_metrics():
    """
    Plot the accumulated scoring metrics for all test methods in separate subplots.

    Args:
        None

    Returns:
        None: The function generates a grouped bar chart displaying the scoring metrics.
    """

    cmap = mpl.colormaps["Set1"]
    metrics = [
        "Overall Accuracy",
        "Accuracy",
        "AUC-ROC",
        "Precision",
        "Recall",
        "Average Precision",
        "F1 Score",
    ]
    metric_dicts = [
        all_accuracies_overall,
        all_accuracies,
        all_auc_rocs,
        all_precisions,
        all_recalls,
        all_avg_precisions,
        all_f1s,
    ]

    num_metrics = len(metrics)
    num_rows = -(-num_metrics // 4)  # This is a way to do "ceiling division"
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 20))

    n_classes = 3  # Number of classes (0, 1, 2)
    bar_width = 0.2  # Width of each bar
    index = np.arange(len(all_accuracies))  # The x locations for the groups

    # Loop through each metric and its corresponding dictionary
    colcount = 0
    rowcount = 0

    for i, (metric, metric_dict) in enumerate(zip(metrics, metric_dicts)):
        if i > 0 and i % 4 == 0:
            rowcount += 1
            colcount = 0

        methods = list(metric_dict.keys())
        values = np.array([metric_dict[method] for method in methods])

        for c in range(n_classes):
            bars = axes[rowcount, colcount].bar(
                index + c * bar_width,
                values[:, c],
                bar_width,
                label=f"Class {c}",
                color=cmap(c / float(n_classes)),
            )

        # Annotate the bars with the actual values
        for j, (method, vals) in enumerate(zip(methods, values)):
            for c, v in enumerate(vals):
                axes[rowcount, colcount].text(
                    j + c * bar_width,
                    v,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                )

        axes[rowcount, colcount].set_title(metric)
        axes[rowcount, colcount].set_xticks(index + bar_width)
        axes[rowcount, colcount].set_xticklabels(methods)
        axes[rowcount, colcount].legend(title="Class")
        axes[rowcount, colcount].set_ylabel("Score")
        axes[rowcount, colcount].tick_params(axis="x", rotation=90)

        colcount += 1

    plt.suptitle("Scoring Metrics for All Methods")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("scores_grouped.png", facecolor="white", bbox_inches="tight")


class TestMyClasses(unittest.TestCase):
    def setUp(self):
        with HiddenPrints():
            self.genotype_data = GenotypeData(
                filename="PG-SUI/pgsui/example_data/phylip_files/test.phy",
                popmapfile="PG-SUI/pgsui/example_data/popmaps/test.popmap",
                # filename="pgsui/example_data/phylip_files/nmt_ingroup_filt.phy",
                # popmapfile="filtered.popmap.K6.txt",
                prefix="test_sim",
                force_popmap=True,
                plot_format="png",
                guidetree="PG-SUI/pgsui/example_data/trees/test.tre",
                iqtree_filename="PG-SUI/pgsui/example_data/trees/test.iqtree",
                siterates_iqtree="PG-SUI/pgsui/example_data/trees/test.rate",
            )

            # Create a SimGenotypeDataTransformer instance and use it
            # to simulate missing data
            self.transformer = SimGenotypeDataTransformer(
                genotype_data=self.genotype_data,
                prop_missing=0.2,
                strategy="random_weighted",
            )
            self.transformer.fit(self.genotype_data.genotypes_int)
            self.simulated_data = self.genotype_data.copy()

            self.genotype_data.genotypes_int = self.transformer.transform(
                self.genotype_data.genotypes_int
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

        if class_instance in [
            ImputeVAE,
            ImputeStandardAutoEncoder,
            ImputeNLPCA,
            ImputeUBP,
        ]:
            instance = class_instance(
                self.simulated_data,
                sim_prop_missing=0.3,
                sim_strategy="random_weighted",
                disable_progressbar=True,
                prefix="test_impute_05",
                gridparams=param_grid,
                sample_weights="auto",
            )
        elif class_instance in [
            ImputeAlleleFreq,
            ImputeMF,
            ImputeRefAllele,
        ]:
            instance = class_instance(
                self.simulated_data2,
                prefix="test_impute_05",
            )
        else:
            instance = class_instance(
                self.simulated_data,
                prefix="test_impute_05",
                disable_progressbar=True,
                gridparams=param_grid,
                sample_weights=None,
            )

        imputed_data = instance.imputed.genotypes_012(fmt="numpy")

        metrics_dict = self._scoring_metrics(
            self.genotype_data.genotypes_012(fmt="numpy"),
            imputed_data,
            class_instance,
        )

        pprint.PrettyPrinter(indent=4, sort_dicts=True).pprint(metrics_dict)
        print("\n")

        # Store metrics
        all_accuracies_overall[class_instance.__name__] = metrics_dict[
            "accuracy_overall"
        ]
        all_accuracies[class_instance.__name__] = metrics_dict[
            "weighted_accuracy"
        ]
        all_auc_rocs[class_instance.__name__] = metrics_dict["auc_roc"]
        all_precisions[class_instance.__name__] = metrics_dict["precision"]
        all_recalls[class_instance.__name__] = metrics_dict["recall"]
        all_avg_precisions[class_instance.__name__] = metrics_dict[
            "avg_precision"
        ]
        all_f1s[class_instance.__name__] = metrics_dict["f1"]
        plot_scoring_metrics()

    def test_phylosim(self):
        # config = {
        #     "custom_model_path": "PG-SUI/pgsui/example_data/demography/custom_models.py",
        #     "custom_model_name": "two_pop_sym_mig_size",
        #     "mutation_rate": 2.17889154e-3,
        #     "recombination_rate": 1e-3,
        #     "sequence_length": 1e3,
        # }

        snpulator_jc = SNPulatoRate(
            self.genotype_data,
            1e7,
        )
        gtr = snpulator_jc.GTR(
            self.genotype_data.alignment, snpulator_jc.base_freq, None
        )
        print(gtr._calculate_gtr_Q(self.genotype_data.alignment))

        # mutation_rate_jc = snpulator_jc.calculate_rate(model="JC")
        # print(f"Jukes-Cantor Mutation Rate: {mutation_rate_jc}")

        # # For GTR Model
        # snpulator_gtr = SNPulatoRate(self.genotype_data, 1e7)
        # mutation_rate_gtr = snpulator_gtr.calculate_rate(model="GTR")
        # print(f"GTR Mutation Rate: {mutation_rate_gtr}")

        # snpulator_hky = SNPulatoRate(self.genotype_data, 1e7)
        # mutation_rate_hky = snpulator_hky.calculate_rate(model="HKY")
        # print(f"HKY Mutation Rate: {mutation_rate_hky}")

        # snpulator_f84 = SNPulatoRate(self.genotype_data, 1e7)
        # mutation_rate_f84 = snpulator_hky.calculate_rate(model="F84")
        # print(f"F84 Mutation Rate: {mutation_rate_f84}")

        # config = SNPulatorConfig(
        #     sequence_length=1e4,
        #     mutation_rate=mutation_rate_gtr,
        #     recombination_rate=1e-7,
        #     demes_graph="PG-SUI/pgsui/example_data/demography/example_demes.yaml",
        # )

        # snps = SNPulator(
        #     self.genotype_data,
        #     config,
        # )
        # snps.sim_ancestry(
        #     populations=["ON_T1", "ON_T2", "DS_T1", "DS_T2"],
        #     sample_sizes=[10] * 4,
        # )
        # snps.sim_mutations()

        # gt = snps.genotypes

    # def test_ImputeKNN(self):
    #     self._test_class(ImputeKNN)

    # def test_ImputeRandomForest(self):
    #     self._test_class(ImputeRandomForest)

    # def test_ImputeXGBoost(self):
    #     self._test_class(ImputeXGBoost)

    # def test_ImputeVAE(self):
    #     self._test_class(ImputeVAE)

    # def test_ImputeStandardAutoEncoder(self):
    #     self._test_class(ImputeStandardAutoEncoder)

    # def test_ImputeUBP(self):
    #     self._test_class(ImputeUBP)

    # def test_ImputeNLPCA(self):
    #     self._test_class(ImputeNLPCA)

    # def test_ImputeKNN_grid(self):
    #     self._test_class(ImputeKNN, do_gridsearch=True)

    # def test_ImputeRandomForest_grid(self):
    #     self._test_class(ImputeRandomForest, do_gridsearch=True)

    # def test_ImputeXGBoost_grid(self):
    #     self._test_class(ImputeXGBoost, do_gridsearch=True)

    # def test_ImputeVAE_grid(self):
    #     self._test_class(ImputeVAE, do_gridsearch=True)

    # def test_ImputeStandardAutoEncoder_grid(self):
    #     self._test_class(ImputeStandardAutoEncoder, do_gridsearch=True)

    # def test_ImputeUBP_grid(self):
    #     self._test_class(ImputeUBP, do_gridsearch=True)

    # def test_ImputeNLPCA_grid(self):
    #     self._test_class(ImputeNLPCA, do_gridsearch=True)

    # def test_ImputePhylo(self):
    #     self._test_class(ImputePhylo)

    # def test_ImputeAlleleFreq(self):
    #     self._test_class(ImputeAlleleFreq)

    # def test_ImputeMF(self):
    #     self._test_class(ImputeMF)

    # def test_ImputeRefAllele(self):
    #     self._test_class(ImputeRefAllele)

    def _scoring_metrics(self, y_true, y_pred, class_instance):
        """Calculate various scoring metrics for each class separately.

        Args:
            y_true (np.ndarray): True values, a 1D numpy array.
            y_pred (np.ndarray): Predicted values, a 1D numpy array.

        Returns:
            dict: Dictionary containing metrics, with the following keys:
                - 'accuracy_overall': Overall accuracy.
                - 'accuracy': List of accuracies in the order [0, 1, 2].
                - 'auc_roc': List of AUC-ROC scores in the order [0, 1, 2].
                - 'precision': List of precision scores in the order [0, 1, 2].
                - 'recall': List of recall scores in the order [0, 1, 2].
                - 'avg_precision': List of average precision scores in the order [0, 1, 2].
                - 'f1': List of F1 scores in the order [0, 1, 2].

        """

        if class_instance in [
            ImputeAlleleFreq,
            ImputeRefAllele,
            ImputeMF,
            ImputePhylo,
        ]:
            t = self.transformer2
        else:
            t = self.transformer

        # Apply mask
        y_true = y_true[t.sim_missing_mask_]
        y_pred = y_pred[t.sim_missing_mask_]

        plot_class_distribution(y_true, filename="class_distribution.png")

        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2])

        # Overall Accuracy
        accuracy_overall = accuracy_score(y_true, y_pred)

        # Class-wise accuracy
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        plot_and_save_confusion_matrix(y_true, y_pred, filename="conf_mat.png")

        # Weighted Accuracy
        total_samples = cm.sum(axis=1)
        true_positives = cm.diagonal()
        weighted_accuracies = np.where(
            total_samples == 0, 0, true_positives / total_samples
        )

        # AUC-ROC score
        auc_roc = roc_auc_score(y_true_bin, y_pred_bin, average=None)

        # Precision-Recall score
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average=None
        )

        # Average precision score
        avg_precision = average_precision_score(
            y_true_bin, y_pred_bin, average=None
        )

        # F1 score
        f1 = f1_score(y_true_bin, y_pred_bin, average=None)

        # Combine into dictionary for easier retrieval
        metrics_dict = {
            "accuracy_overall": [
                accuracy_overall,
                accuracy_overall,
                accuracy_overall,
            ],
            "weighted_accuracy": list(weighted_accuracies),
            "auc_roc": list(auc_roc),
            "precision": list(precision),
            "recall": list(recall),
            "avg_precision": list(avg_precision),
            "f1": list(f1),
        }

        return metrics_dict


if __name__ == "__main__":
    unittest.main()
