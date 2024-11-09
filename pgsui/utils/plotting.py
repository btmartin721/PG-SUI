from pathlib import Path
from typing import Dict, List, Union

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc,
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
        self.plot_format = plot_format
        self.plot_fontsize = plot_fontsize
        self.plot_dpi = plot_dpi
        self.title_fontsize = title_fontsize
        self.show_plots = show_plots

        param_dict = {
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

        mpl.rcParams.update(param_dict)

        if model_name in {"ImputeVAE", "ImputeNLPCA", "ImputeStandardAE"}:
            plot_dir = "Unsupervised"
        else:
            plot_dir = "Supervised"

        self.output_dir = (
            Path(f"{self.prefix}_output") / "plots" / plot_dir / model_name
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metrics: Dict[str, float],
    ) -> None:
        """Plot multi-class ROC-AUC and Precision-Recall curves.

        Args:
            y_true (np.ndarray): Array of true labels (class indices).
            y_pred_proba (np.ndarray): Array of predicted probabilities for each class.
            metrics (Dict[str, float]): Dictionary of performance metrics.
        """
        # Binarize the labels if they are not already one-hot encoded
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            num_classes = y_pred_proba.shape[1]
            y_true = label_binarize(y_true, classes=range(num_classes))
        else:
            num_classes = y_true.shape[1]

        yt = np.reshape(y_true, (-1, num_classes))
        yp = np.reshape(y_pred_proba, (-1, num_classes))

        # Initialize dictionaries to hold per-class metrics
        fpr = {}
        tpr = {}
        roc_auc = {}
        precision = {}
        recall = {}
        average_precision = {}

        # Compute per-class ROC and PR curves
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(yt[:, i], yp[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            precision[i], recall[i], _ = precision_recall_curve(yt[:, i], yp[:, i])
            average_precision[i] = average_precision_score(yt[:, i], yp[:, i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(yt.ravel(), yp.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= num_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Compute micro-average PR curve and PR area
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            yt.ravel(), yp.ravel()
        )
        average_precision["micro"] = average_precision_score(yt, yp, average="micro")

        # Compute macro-average PR curve and PR area
        average_precision["macro"] = average_precision_score(yt, yp, average="macro")

        # Create a figure with subplots for ROC and Precision-Recall curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot all ROC curves
        axes[0].plot(
            fpr["micro"],
            tpr["micro"],
            label=f"Micro-average ROC curve (area = {roc_auc['micro']:.2f})",
            linestyle=":",
            linewidth=4,
        )
        axes[0].plot(
            fpr["macro"],
            tpr["macro"],
            label=f"Macro-average ROC curve (area = {roc_auc['macro']:.2f})",
            linestyle="--",
            linewidth=4,
        )

        for i in range(num_classes):
            axes[0].plot(
                fpr[i],
                tpr[i],
                label=f"Class {i} ROC curve (area = {roc_auc[i]:.2f})",
            )

        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("Multi-class ROC-AUC Curve")
        axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", label="Random")
        axes[0].legend(bbox_to_anchor=(0.5, -0.05), loc="upper center")

        # Plot all Precision-Recall curves
        axes[1].plot(
            recall["micro"],
            precision["micro"],
            label=f"Micro-average PR curve (AP = {average_precision['micro']:.2f})",
            linestyle=":",
            linewidth=4,
        )

        for i in range(num_classes):
            axes[1].plot(
                recall[i],
                precision[i],
                label=f"Class {i} PR curve (AP = {average_precision[i]:.2f})",
            )

        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Multi-class Precision-Recall Curve")
        axes[1].plot([0, 1], [1, 0], color="black", linestyle="--", label="Random")
        axes[1].legend(bbox_to_anchor=(0.5, -0.05), loc="upper center")

        # Add summary metrics (accuracy, F1 score, precision, recall, etc.) in
        # the figure's title
        summary_metrics = "\n".join([f"{k}: {v:.2f}" for k, v in metrics.items()])

        # Adding the summary metrics as the figure title or below the plot
        fig.suptitle(summary_metrics, y=1.3)

        fn = f"{self.model_name.lower()}_metrics_plot.{self.plot_format}"
        fn = self.output_dir / fn
        fig.savefig(fn)

        if self.show_plots:
            plt.show()

        plt.close()

    def plot_history(self, history: Dict[str, List[float]]) -> None:
        """Plot model history traces. Will be saved to file.

        This method plots the model history traces. The plot is saved to disk as a ``<plot_format>`` file.

        Args:
            history (Dict[str, List[float]]): Dictionary with lists of history objects. Keys should be "Train" and "Val".

        Raises:
            ValueError: nn_method must be either 'ImputeNLPCA', 'ImputeUBP', 'ImputeStandardAE', or 'ImputeVAE'.
        """
        if self.model_name not in {
            "ImputeNLPCA",
            "ImputeVAE",
            "ImputeStandardAE",
            "ImputeUBP",
        }:
            msg = "nn_method must be either 'ImputeNLPCA', 'ImputeVAE', 'ImputeStandardAE', or 'ImputeUBP'."
            self.logger.error(msg)
            raise ValueError()

        if self.model_name in {"ImputeNLPCA", "ImputeVAE", "ImputeStandardAE"}:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            df = pd.DataFrame(history)

            # Plot train accuracy
            ax = sns.lineplot(
                data=df,
                x=np.arange(len(df)),
                y="Train",
                color="blue",
                linewidth=3,
                ax=ax,
            )
            ax = sns.lineplot(
                data=df,
                x=np.arange(len(df)),
                y="Val",
                color="orange",
                linewidth=3,
                ax=ax,
            )
            ax.set_title("Model Loss per Epoch")
            ax.set_ylabel("Loss")
            ax.set_xlabel("Epoch")

            ymax = max(1.0, df["Train"].max(), df["Val"].max())
            ax.set_ylim(bottom=0.0, top=ymax)
            ax.set_yticks(np.linspace(0, ymax, num=int(ymax / 4), endpoint=True))
            ax.legend(["Train", "Val"], labels=["Train", "Validation"], loc="best")

        elif self.model_name == "ImputeUBP":
            fig = plt.figure(figsize=(12, 16))
            fig.suptitle(self.model_name)
            fig.tight_layout(h_pad=2.0, w_pad=2.0)

            idx = 1
            for i, history in enumerate(history, start=1):
                plt.subplot(3, 2, idx)
                title = f"Phase {i}"

                # Plot model accuracy
                ax = plt.gca()
                ax.plot(history["binary_accuracy"])
                ax.set_title(f"{title} Accuracy")
                ax.set_ylabel("Accuracy")
                ax.set_xlabel("Epoch")
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

                # Plot validation accuracy
                ax.plot(history["val_binary_accuracy"])
                ax.legend(["Train", "Validation"], loc="best")

                # Plot model loss
                plt.subplot(3, 2, idx + 1)
                ax = plt.gca()
                ax.plot(history["loss"])
                ax.set_title(f"{title} Loss")
                ax.set_ylabel("Loss (MSE)")
                ax.set_xlabel("Epoch")

                # Plot validation loss
                ax.plot(history["val_loss"])
                ax.legend(["Train", "Validation"], loc="best")

                idx += 2

        fn = f"{self.model_name.lower()}_history_plot.{self.plot_format}"
        fn = self.output_dir / fn
        plt.savefig(fn)

        if self.show_plots:
            plt.show()

        plt.close()

    def plot_certainty_heatmap(self, y_certainty: np.ndarray) -> None:
        """Plot a heatmap of the probabilities of uncertain sites.

        This method plots a heatmap of the probabilities of uncertain sites. The plot is saved to disk as a ``<plot_format>`` file.

        Args:
            y_certainty (np.ndarray): Array of probabilities of uncertain sites.
        """
        fig = plt.figure()
        hm = sns.heatmap(
            data=y_certainty,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "Prob."},
        )
        hm.set_xlabel("Site")
        hm.set_ylabel("Sample")
        hm.set_title("Probabilities of Uncertain Sites")
        fig.tight_layout()

        fn = f"{self.model_name.lower()}_certainty_heatmap.{self.plot_format}"
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
            y_true=y_true_1d, y_pred=y_pred_1d, ax=ax
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
        arr = misc.validate_input_type(X, return_type="array")
        values, counts = np.unique(arr, return_counts=True, axis=None)
        df = pd.DataFrame({"Genotype": values, "Count": counts})
        df = df.sort_values(by="Genotype", ascending=True)
        title = "Imputed Genotype Counts" if is_imputed else "Genotype Counts"

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        g = sns.barplot(x="Genotype", y="Count", data=df, ax=ax)
        g.set_xlabel("Integer-encoded Genotype")
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

        fn = f"{self.model_name.lower()}_gt_distributions.{self.plot_format}"
        fn = self.output_dir / fn

        fig.savefig(fn)

        if self.show_plots:
            plt.show()

        plt.close()

    def plot_label_clusters(
        self, z_mean: np.ndarray | torch.Tensor, z_log_var: np.ndarray | torch.Tensor
    ) -> None:
        """Display a 2D plot of the classes in the latent space.

        This method displays a 2D plot of the classes in the latent space. The plot is saved to disk as a ``<plot_format>`` file.

        Args:
            z_mean (Union[np.ndarray, torch.Tensor]): Array or torch.Tensor of latent means.
            z_log_var (Union[np.ndarray, torch.Tensor): Array or torch.Tensor of latent log variances.

        ToDo:
            * Add z_log_var to the plot.
        """
        if isinstance(z_mean, torch.Tensor):
            z_mean = z_mean.detach().cpu().numpy()

        if isinstance(z_log_var, torch.Tensor):
            z_log_var = z_log_var.detach().cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))

        # ToDo: Add z_log_var to the plot.
        sns.scatterplot(x=z_mean[:, 0], y=z_mean[:, 1], ax=ax)
        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")

        fn = f"{self.model_name.lower()}_label_clusters.{self.plot_format}"
        fn = self.output_dir / fn

        fig.savefig(fn)

        if self.show_plots:
            plt.show()

        plt.close()

    def plot_pr_curve(
        self, precision: np.ndarray, recall: np.ndarray, average_precision: float
    ) -> None:
        """Plot a Precision-Recall curve.

        This method plots a Precision-Recall curve. The plot is saved to disk as a ``<plot_format>`` file.

        Args:
            precision (np.ndarray): Array of precision values.
            recall (np.ndarray): Array of recall values.
            average_precision (float): Average precision score.
        """
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.plot(recall, precision, lw=2, label=f"AP = {average_precision:.2f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{self.model_name} Precision-Recall Curve")
        ax.legend(loc="best")

        fn = f"{self.model_name.lower()}_pr_curve.{self.plot_format}"
        fn = self.output_dir / fn

        fig.savefig(fn)

        if self.show_plots:
            plt.show()

        plt.close()

    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> None:
        """Plot a ROC curve.

        This method plots a ROC curve. The plot is saved to disk as a ``<plot_format>`` file.

        Args:
            fpr (np.ndarray): Array of false positive rates.
            tpr (np.ndarray): Array of true positive rates.
            roc_auc (float): ROC AUC score.
        """
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.2f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{self.model_name} ROC Curve")

        fn = f"{self.model_name.lower()}_roc_curve.{self.plot_format}"
        fn = self.output_dir / fn

        fig.savefig(fn)

        if self.show_plots:
            plt.show()

        plt.close()
