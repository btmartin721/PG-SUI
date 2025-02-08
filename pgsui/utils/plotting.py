import warnings
from pathlib import Path
from typing import Dict, List, Union

import matplotlib as mpl

# Use Agg backend for headless plotting
mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from optuna.exceptions import ExperimentalWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from snpio.utils.logging import LoggerManager

from pgsui.utils import misc


class Plotting:
    """Class for plotting imputer scoring and results.

    This class is used to plot the performance metrics of imputation models. It can plot ROC and Precision-Recall curves, model history, and the distribution of genotypes in the dataset.

    Example:
        >>> from pgsui import Plotting
        >>> plotter = Plotting(model_name="ImputeVAE")
        >>> plotter.plot_metrics(metrics, num_classes)
        >>> plotter.plot_history(history)
        >>> plotter.plot_certainty_heatmap(y_certainty)
        >>> plotter.plot_confusion_matrix(y_true_1d, y_pred_1d)
        >>> plotter.plot_gt_distribution(df)
        >>> plotter.plot_label_clusters(z_mean, z_log_var)

    Attributes:
        model_name (str): Name of the model.
        prefix (str): Prefix for the output directory.
        plot_format (str): Format for the plots (e.g., 'pdf', 'png', 'jpeg').
        plot_fontsize (int): Font size for the plots.
        plot_dpi (int): Dots per inch for the plots.
        title_fontsize (int): Font size for the plot titles.
        show_plots (bool): Whether to display the plots.
        output_dir (Path): Output directory for the plots.
        logger (LoggerManager): Logger object for logging messages.
    """

    def __init__(
        self,
        model_name: str,
        *,
        prefix: str = "pgsui",
        output_dir: str = "output",
        plot_format: str = "pdf",
        plot_fontsize: int = 18,
        plot_dpi: int = 300,
        title_fontsize: int = 20,
        despine: bool = True,
        show_plots: bool = False,
        verbose: int = 0,
        debug: bool = False,
    ) -> None:
        """Initialize the Plotting object.

        This class is used to plot the performance metrics of imputation models. It can plot ROC and Precision-Recall curves, model history, and the distribution of genotypes in the dataset.

        Args:
            model_name (str): Name of the model.
            prefix (str, optional): Prefix for the output directory. Defaults to 'pgsui'.
            output_dir (str, optional): Output directory for the plots. Defaults to 'output'.
            plot_format (str, optional): Format for the plots (e.g., 'pdf', 'png', 'jpeg'). Defaults to 'pdf'.
            plot_fontsize (int, optional): Font size for the plots. Defaults to 18.
            plot_dpi (int, optional): Dots per inch for the plots. Defaults to 300.
            title_fontsize (int, optional): Font size for the plot titles. Defaults to 20.
            despine (bool, optional): Whether to remove the top and right spines from the plots. Defaults to True.
            show_plots (bool, optional): Whether to display the plots. Defaults to False.
            verbose (int, optional): Verbosity level for logging. Defaults to 0.
            debug (bool, optional): Whether to enable debug mode. Defaults to False.
        """
        logman = LoggerManager(
            name=__name__, prefix=prefix, verbose=verbose, debug=debug
        )

        self.logger = logman.get_logger()

        self.model_name = model_name
        self.prefix = prefix
        self.output_dir = output_dir
        self.plot_format = plot_format
        self.plot_fontsize = plot_fontsize
        self.plot_dpi = plot_dpi
        self.title_fontsize = title_fontsize
        self.show_plots = show_plots

        if self.plot_format.startswith("."):
            self.plot_format = self.plot_format.lstrip(".")

        self.param_dict = {
            "axes.labelsize": self.plot_fontsize,
            "axes.titlesize": self.title_fontsize,
            "axes.spines.top": despine,
            "axes.spines.right": despine,
            "xtick.labelsize": self.plot_fontsize,
            "ytick.labelsize": self.plot_fontsize,
            "legend.fontsize": self.plot_fontsize,
            "legend.facecolor": "white",
            "figure.titlesize": self.title_fontsize,
            "figure.dpi": self.plot_dpi,
            "figure.facecolor": "white",
            "axes.linewidth": 2.0,
            "lines.linewidth": 2.0,
            "font.size": self.plot_fontsize,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.dpi": self.plot_dpi,
        }

        mpl.rcParams.update(self.param_dict)

        unsuper = {
            "ImputeVAE",
            "ImputeNLPCA",
            "ImputeAutoencoder",
            "ImputeUBP",
            "ImputeCNN",
            "ImputeLSTM",
        }
        plot_dir = "Unsupervised" if model_name in unsuper else "Supervised"
        self.output_dir = Path(f"{self.prefix}_{self.output_dir}", plot_dir)
        self.output_dir = self.output_dir / "plots" / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_tuning(
        self,
        study: optuna.study.Study,
        model_name: str,
        target_name: str = "Objective Value",
    ) -> None:
        """Plot the optimization history of a study.

        This method plots the optimization history of a study. The plot is saved to disk as a ``<plot_format>`` file.

        Args:
            study (optuna.study.Study): Optuna study object.
            model_name (str): Name of the model.
            target_name (str, optional): Name of the target value. Defaults to 'Objective Value'.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=ExperimentalWarning)

            od = self.output_dir / "optimize"
            target_name = target_name.title()

            ax = optuna.visualization.matplotlib.plot_optimization_history(
                study, target_name=target_name
            )
            ax.set_title(f"{model_name} Optimization History")
            ax.set_xlabel("Trial")
            ax.set_ylabel(target_name)
            ax.legend(
                loc="best",
                shadow=True,
                fancybox=True,
                fontsize=mpl.rcParamsDefault["legend.fontsize"],
            )

            fn = od / f"optuna_optimization_history.{self.plot_format}"

            if not fn.parent.exists():
                fn.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(fn)
            plt.close()

            ax = optuna.visualization.matplotlib.plot_edf(
                study, target_name=target_name
            )
            ax.set_title(f"{model_name} Empirical Distribution Function (EDF)")
            ax.set_xlabel(target_name)
            ax.set_ylabel(f"{model_name} Cumulative Probability")
            ax.legend(
                loc="best",
                shadow=True,
                fancybox=True,
                fontsize=mpl.rcParamsDefault["legend.fontsize"],
            )

            plt.savefig(fn.with_stem("optuna_edf_plot"))
            plt.close()

            ax = optuna.visualization.matplotlib.plot_param_importances(
                study, target_name=target_name
            )
            ax.set_xlabel("Parameter Importance")
            ax.set_ylabel("Parameter")
            ax.legend(loc="best", shadow=True, fancybox=True)

            plt.savefig(fn.with_stem("optuna_param_importances_plot"))
            plt.close()

            ax = optuna.visualization.matplotlib.plot_timeline(study)
            ax.set_title(f"{model_name} Timeline Plot")
            ax.set_xlabel("Datetime")
            ax.set_ylabel("Trial")
            plt.savefig(fn.with_stem("optuna_timeline_plot"))
            plt.close()

            # Reset the style from Optuna's plotting.
            sns.set_style("white", rc=self.param_dict)
            mpl.rcParams.update(self.param_dict)

    def plot_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metrics: Dict[str, float],
    ) -> None:
        """Plot multi-class ROC-AUC and Precision-Recall curves.

        Args:
            y_true (np.ndarray): Array of true labels.
            y_pred_proba (np.ndarray): Array of predicted probabilities.
            metrics (Dict[str, float]): Dictionary of metrics to display in the plot.

        Raises:
            ValueError: If the model_name is not one of the following: 'ImputeNLPCA', 'ImputeUBP', 'ImputeStandardAE', 'ImputeVAE', 'ImputeCNN', or 'ImputeLSTM'.
        """
        # Ensure y_true is properly binarized
        num_classes = y_pred_proba.shape[1]
        y_true = label_binarize(y_true, classes=np.arange(num_classes))

        # Initialize dictionaries for metrics
        fpr, tpr, roc_auc = {}, {}, {}
        precision, recall, average_precision = {}, {}, {}

        # Compute per-class ROC and PR metrics
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            precision[i], recall[i], _ = precision_recall_curve(
                y_true[:, i], y_pred_proba[:, i]
            )
            average_precision[i] = average_precision_score(
                y_true[:, i], y_pred_proba[:, i]
            )

        # Micro-average metrics
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true.ravel(), y_pred_proba.ravel()
        )
        average_precision["micro"] = average_precision_score(
            y_true, y_pred_proba, average="micro"
        )

        # Macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Macro-average PR
        all_recall = np.unique(np.concatenate([recall[i] for i in range(num_classes)]))
        mean_precision = np.zeros_like(all_recall)
        for i in range(num_classes):
            mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
        mean_precision /= num_classes
        average_precision["macro"] = average_precision_score(
            y_true, y_pred_proba, average="macro"
        )

        # Plot ROC and PR curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ROC curves
        axes[0].plot(
            fpr["micro"],
            tpr["micro"],
            label=f"Micro-average ROC (AUC = {roc_auc['micro']:.2f})",
            linestyle=":",
            linewidth=4,
        )
        axes[0].plot(
            fpr["macro"],
            tpr["macro"],
            label=f"Macro-average ROC (AUC = {roc_auc['macro']:.2f})",
            linestyle="--",
            linewidth=4,
        )
        for i in range(num_classes):
            axes[0].plot(
                fpr[i], tpr[i], label=f"Class {i} ROC (AUC = {roc_auc[i]:.2f})"
            )
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", label="Random")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("Multi-class ROC-AUC Curve")
        axes[0].legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True
        )

        # PR curves
        axes[1].plot(
            recall["micro"],
            precision["micro"],
            label=f"Micro-average PR (AP = {average_precision['micro']:.2f})",
            linestyle=":",
            linewidth=4,
        )
        axes[1].plot(
            all_recall,
            mean_precision,
            label=f"Macro-average PR (AP = {average_precision['macro']:.2f})",
            linestyle="--",
            linewidth=4,
        )
        for i in range(num_classes):
            axes[1].plot(
                recall[i],
                precision[i],
                label=f"Class {i} PR (AP = {average_precision[i]:.2f})",
            )
        axes[1].plot([0, 1], [1, 0], linestyle="--", color="black", label="Random")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Multi-class Precision-Recall Curve")
        axes[1].legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True
        )

        # Save and display
        fig.suptitle("\n".join([f"{k}: {v:.2f}" for k, v in metrics.items()]), y=1.35)

        if self.show_plots:
            plt.show()

        od = f"{self.model_name}_roc_pr_curves.{self.plot_format}"
        fn = self.output_dir / od
        fig.savefig(fn)

    def plot_history(self, history: Dict[str, List[float]]) -> None:
        """Plot model history traces. Will be saved to file.

        This method plots the model history traces. The plot is saved to disk as a ``<plot_format>`` file.

        Args:
            history (Dict[str, List[float]]): Dictionary with lists of history objects. Keys should be "Train" and "Validation".

        Raises:
            ValueError: nn_method must be either 'ImputeNLPCA', 'ImputeUBP', 'ImputeStandardAE', 'ImputeVAE', 'ImputeCNN', or 'ImputeLSTM'.
        """
        if self.model_name not in {
            "ImputeNLPCA",
            "ImputeVAE",
            "ImputeAutoencoder",
            "ImputeUBP",
            "ImputeLSTM",
            "ImputeCNN",
        }:
            msg = "nn_method must be either 'ImputeNLPCA', 'ImputeVAE', 'ImputeStandardAE', 'ImputeCNN', 'ImputeLSTM', or 'ImputeUBP'."
            self.logger.error(msg)
            raise ValueError(msg)

        if self.model_name in {
            "ImputeNLPCA",
            "ImputeVAE",
            "ImputeAutoencoder",
            "ImputeCNN",
            "ImputeLSTM",
        }:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            df = pd.DataFrame(history)
            df = df.iloc[1:]

            # Plot train accuracy
            ax.plot(df["Train"], c="blue", lw=3)

            ax.set_title(f"{self.model_name} Loss per Epoch")
            ax.set_ylabel("Loss")
            ax.set_xlabel("Epoch")
            ax.legend(["Train"], loc="best", shadow=True, fancybox=True)

        elif self.model_name == "ImputeUBP":
            fig, ax = plt.subplots(3, 1, figsize=(12, 8))

            for i, phase in enumerate(range(1, 4)):
                train = pd.Series(history["Train"][f"Phase {phase}"])
                train = train.iloc[1:]  # ignore first epoch

                # Plot train accuracy
                ax[i].plot(train, c="blue", lw=3)
                ax[i].set_title(f"{self.model_name}: Phase {phase} Loss per Epoch")
                ax[i].set_ylabel("Loss")
                ax[i].set_xlabel("Epoch")
                ax[i].legend([f"Phase {phase}"], loc="best", shadow=True, fancybox=True)

        fn = f"{self.model_name.lower()}_history_plot.{self.plot_format}"
        fn = self.output_dir / fn
        fig.savefig(fn)

        if self.show_plots:
            plt.show()

        plt.close()

    def plot_confusion_matrix(
        self, y_true_1d: np.ndarray, y_pred_1d: np.ndarray
    ) -> None:
        """Plot a confusion matrix.

        This method plots a confusion matrix. The plot is saved to disk as a ``<plot_format>`` file.

        Args:
            y_true_1d (np.ndarray): Array of true labels. The array should be 1D. If the array is multi-dimensional, it will be flattened.
            y_pred_1d (np.ndarray): Array of predicted labels. The array should be 1D. If the array is multi-dimensional, it will be flattened.
        """
        y_true_1d = misc.validate_input_type(y_true_1d, return_type="array")
        y_pred_1d = misc.validate_input_type(y_pred_1d, return_type="array")

        if y_true_1d.ndim > 1:
            y_true_1d = y_true_1d.flatten()

        if y_pred_1d.ndim > 1:
            y_pred_1d = y_pred_1d.flatten()

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ConfusionMatrixDisplay.from_predictions(
            y_true=y_true_1d, y_pred=y_pred_1d, ax=ax, cmap="viridis"
        )

        fn = f"{self.model_name.lower()}_confusion_matrix.{self.plot_format}"
        fn = self.output_dir / fn
        fig.savefig(fn)

        if self.show_plots:
            plt.show()

        plt.close()

    def plot_gt_distribution(
        self,
        X: Union[np.ndarray, pd.DataFrame, List[List[int]], torch.Tensor],
        is_imputed: bool = False,
    ) -> None:
        """Plot the distribution of genotypes in the dataset.

        This method plots the distribution of genotypes in the dataset. The plot is saved to disk as a ``<plot_format>`` file.

        Args:
            X (pd.DataFrame): DataFrame containing genotype data. Columns should be samples and rows should be genotypes.
            is_imputed (bool): Whether the data has been imputed. Defaults to False.
        """
        arr = misc.validate_input_type(X, "array")
        values, counts = np.unique(arr, return_counts=True, axis=None)
        df = pd.DataFrame({"Genotype": values, "Count": counts})
        df = df.sort_values(by="Genotype", ascending=True)
        title = "Imputed Genotype Counts" if is_imputed else "Genotype Counts"

        fig, ax = plt.subplots(1, 1, figsize=(8, 12))
        g = sns.barplot(x="Genotype", y="Count", data=df, ax=ax, palette="Set1")
        g.set_xlabel("Integer Encoded Genotype")
        g.set_ylabel("Count")
        g.set_title(title)
        for p in g.patches:
            g.annotate(
                f"{p.get_height():.1f}",
                (p.get_x() + 0.25, p.get_height() + 0.01),
                xytext=(0, 1),
                textcoords="offset points",
                va="bottom",
            )

        suffix = "imputed" if is_imputed else "original"
        fn = self.output_dir / f"gt_distributions_{suffix}.{self.plot_format}"
        fig.savefig(fn)

        if self.show_plots:
            plt.show()

        plt.close()
