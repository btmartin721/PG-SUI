import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

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
        output_dir (Path): Output directory for the plots.
        plot_format (Literal["pdf", "png", "jpeg", "jpg"]): Format for the plots ('pdf', 'png', 'jpeg', 'jpg').
        plot_fontsize (int): Font size for the plots.
        plot_dpi (int): Dots per inch for the plots.
        title_fontsize (int): Font size for the plot titles.
        show_plots (bool): Whether to display the plots.
        logger (LoggerManager): Logger object for logging messages.
    """

    def __init__(
        self,
        model_name: str,
        *,
        prefix: str = "pgsui",
        plot_format: Literal["pdf", "png", "jpeg", "jpg"] = "pdf",
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
            plot_format (Literal["pdf", "png", "jpeg", "jpg"]): Format for the plots ('pdf', 'png', 'jpeg', 'jpg'). Defaults to 'pdf'.
            plot_fontsize (int): Font size for the plots. Defaults to 18.
            plot_dpi (int): Dots per inch for the plots. Defaults to 300.
            title_fontsize (int): Font size for the plot titles. Defaults to 20.
            despine (bool): Whether to remove the top and right spines from the plots. Defaults to True.
            show_plots (bool): Whether to display the plots. Defaults to False.
            verbose (int): Verbosity level for logging. Defaults to 0.
            debug (bool): Whether to enable debug mode. Defaults to False.
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

        unsuper = {"ImputeVAE", "ImputeNLPCA", "ImputeAutoencoder", "ImputeUBP"}
        det = {
            "ImputeRefAllele",
            "ImputeMostFrequent",
            "ImputeMostFrequentPerPop",
            "ImputePhylo",
        }
        sup = {"ImputeRandomForest", "ImputeHistGradientBoosting"}

        if model_name in unsuper:
            plot_dir = "Unsupervised"
        elif model_name in det:
            plot_dir = "Deterministic"
        elif model_name in sup:
            plot_dir = "Supervised"
        else:
            msg = f"model_name '{model_name}' not recognized."
            self.logger.error(msg)
            raise ValueError(msg)

        self.output_dir = Path(f"{self.prefix}_output", plot_dir)
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
        label_names: Optional[Sequence[str]] = None,
        prefix: str = "",
    ) -> None:
        """Plot multi-class ROC-AUC and Precision-Recall curves.

        This method plots the multi-class ROC-AUC and Precision-Recall curves. The plot is saved to disk as a ``<plot_format>`` file.

        Args:
            y_true (np.ndarray): 1D array of true integer labels in [0, n_classes-1].
            y_pred_proba (np.ndarray): (n_samples, n_classes) array of predicted probabilities.
            metrics (Dict[str, float]): Dict of summary metrics to annotate the figure.
            label_names (Optional[Sequence[str]]): Optional sequence of class names (length must equal n_classes).
                If provided, legends will use these names instead of 'Class i'.
            prefix (str): Optional prefix for the output filename.

        Raises:
            ValueError: If model_name is not recognized (legacy guard).
        """
        num_classes = y_pred_proba.shape[1]

        # Validate/normalize label names
        if label_names is not None and len(label_names) != num_classes:
            self.logger.warning(
                f"plot_metrics: len(label_names)={len(label_names)} "
                f"!= n_classes={num_classes}. Ignoring label_names."
            )
            label_names = None
        if label_names is None:
            label_names = [f"Class {i}" for i in range(num_classes)]

        # Binarize y_true for one-vs-rest curves
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

        # Containers
        fpr, tpr, roc_auc = {}, {}, {}
        precision, recall, average_precision = {}, {}, {}

        # Per-class ROC & PR
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            precision[i], recall[i], _ = precision_recall_curve(
                y_true_bin[:, i], y_pred_proba[:, i]
            )
            average_precision[i] = average_precision_score(
                y_true_bin[:, i], y_pred_proba[:, i]
            )

        # Micro-averages
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_bin.ravel(), y_pred_proba.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true_bin.ravel(), y_pred_proba.ravel()
        )
        average_precision["micro"] = average_precision_score(
            y_true_bin, y_pred_proba, average="micro"
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
            # recall[i] increases, but precision[i] is given over decreasing thresholds
            mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
        mean_precision /= num_classes
        average_precision["macro"] = average_precision_score(
            y_true_bin, y_pred_proba, average="macro"
        )

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ROC
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
                fpr[i], tpr[i], label=f"{label_names[i]} ROC (AUC = {roc_auc[i]:.2f})"
            )
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", label="Random")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("Multi-class ROC-AUC Curve")
        axes[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            shadow=True,
            ncol=2,
        )

        # PR
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
                label=f"{label_names[i]} PR (AP = {average_precision[i]:.2f})",
            )
        axes[1].plot([0, 1], [1, 0], linestyle="--", color="black", label="Random")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Multi-class Precision-Recall Curve")
        axes[1].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            shadow=True,
            ncol=2,
        )

        # Title & save
        fig.suptitle("\n".join([f"{k}: {v:.2f}" for k, v in metrics.items()]), y=1.35)

        if prefix != "":
            prefix = f"{prefix}_"

        out_name = f"{self.model_name}_{prefix}roc_pr_curves.{self.plot_format}"
        fig.savefig(self.output_dir / out_name, bbox_inches="tight")
        if self.show_plots:
            plt.show()
        plt.close(fig)

    def plot_history(self, history: Dict[str, List[float]]) -> None:
        """Plot model history traces. Will be saved to file.

        This method plots the deep learning model history traces. The plot is saved to disk as a ``<plot_format>`` file.

        Args:
            history (Dict[str, List[float]]): Dictionary with lists of history objects. Keys should be "Train" and "Validation".

        Raises:
            ValueError: nn_method must be either 'ImputeNLPCA', 'ImputeUBP', 'ImputeAutoencoder', 'ImputeVAE'.
        """
        if self.model_name not in {
            "ImputeNLPCA",
            "ImputeVAE",
            "ImputeAutoencoder",
            "ImputeUBP",
        }:
            msg = "nn_method must be either 'ImputeNLPCA', 'ImputeVAE', 'ImputeAutoencoder', 'ImputeUBP'."
            self.logger.error(msg)
            raise ValueError(msg)

        if self.model_name != "ImputeUBP":
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            df = pd.DataFrame(history)
            df = df.iloc[1:]

            # Plot train accuracy
            ax.plot(df["Train"], c="blue", lw=3)

            ax.set_title(f"{self.model_name} Loss per Epoch")
            ax.set_ylabel("Loss")
            ax.set_xlabel("Epoch")
            ax.legend(["Train"], loc="best", shadow=True, fancybox=True)

        else:
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
        plt.close(fig)

    def plot_confusion_matrix(
        self,
        y_true_1d: np.ndarray,
        y_pred_1d: np.ndarray,
        label_names: Sequence[str] | None = None,
        prefix: str = "",
    ) -> None:
        """Plot a confusion matrix with optional class labels.

        This method plots a confusion matrix using true and predicted labels. The plot is saved to disk as a ``<plot_format>`` file.

        Args:
            y_true_1d (np.ndarray): 1D array of true integer labels in [0, n_classes-1].
            y_pred_1d (np.ndarray): 1D array of predicted integer labels in [0, n_classes-1].
            label_names (Sequence[str] | None): Optional sequence of class names (length must equal n_classes). If provided, both the internal label order and displayed tick labels will respect this order (assumed to be 0..n-1).
            prefix (str): Optional prefix for the output filename.

        Notes:
            - If `label_names` is None, the display labels default to the numeric class indices inferred from `y_true_1d ∪ y_pred_1d`.
        """
        y_true_1d = misc.validate_input_type(y_true_1d, return_type="array")
        y_pred_1d = misc.validate_input_type(y_pred_1d, return_type="array")
        if y_true_1d.ndim > 1:
            y_true_1d = y_true_1d.flatten()
        if y_pred_1d.ndim > 1:
            y_pred_1d = y_pred_1d.flatten()

        # Determine class count/order
        if label_names is not None:
            n_classes = len(label_names)
            labels = np.arange(n_classes)  # our y_* are ints 0..n-1
            display_labels = list(map(str, label_names))
        else:
            # Infer labels from data to keep matrix tight
            labels = np.unique(np.concatenate([y_true_1d, y_pred_1d]))
            display_labels = labels  # sklearn will convert to strings

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ConfusionMatrixDisplay.from_predictions(
            y_true=y_true_1d,
            y_pred=y_pred_1d,
            labels=labels,
            display_labels=display_labels,
            ax=ax,
            cmap="viridis",
            colorbar=True,
        )

        if prefix != "":
            prefix = f"{prefix}_"

        out_name = (
            f"{self.model_name.lower()}_{prefix}confusion_matrix.{self.plot_format}"
        )
        fig.savefig(self.output_dir / out_name, bbox_inches="tight")
        if self.show_plots:
            plt.show()
        plt.close(fig)

    def plot_gt_distribution(
        self,
        X: np.ndarray | pd.DataFrame | list | torch.Tensor,
        is_imputed: bool = False,
    ) -> None:
        """Plot genotype distribution (IUPAC or integer-encoded).

        This plots counts for all genotypes present in X. It supports IUPAC single-letter genotypes and integer encodings. Missing markers '-', '.', '?' are normalized to 'N'. Bars are annotated with counts and percentages.

        Args:
            X: Array-like genotype matrix. Rows=loci, cols=samples (any orientation is OK). Elements are IUPAC one-letter genotypes (e.g., 'A','C','G','T','N','R',...) or integers (e.g., 0/1/2[/3]).
            is_imputed: Whether these genotypes are imputed. Affects the title only. Defaults to False.
        """
        # --- Flatten X to a 1D Series ---
        if isinstance(X, pd.DataFrame):
            arr = X.values
        elif torch.is_tensor(X):
            arr = X.detach().cpu().numpy()
        else:
            arr = np.asarray(X)

        s = pd.Series(arr.ravel())

        # Detect string vs numeric encodings and normalize
        if s.dtype.kind in ("O", "U", "S"):  # string-like → IUPAC path
            s = s.astype(str).str.upper().replace({"-": "N", ".": "N", "?": "N"})
            x_label = "Genotype (IUPAC)"
            # Define canonical order: N, A/C/T/G, then IUPAC ambiguity codes.
            canonical = ["A", "C", "T", "G"]
            iupac_ambiguity = sorted(["M", "R", "W", "S", "Y", "K", "V", "H", "D", "B"])
            base_order = ["N"] + canonical + iupac_ambiguity
        else:  # numeric path (e.g., 0/1/2/[3], -1 for missing)
            # Map common missing sentinels to 'N', keep others as strings for
            # labeling
            s = s.astype(float)  # allow NaN comparisons
            s = s.where(~np.isin(s, [-1, np.nan]), other=np.nan)
            s = s.fillna("N").astype(int, errors="ignore").astype(str)

            x_label = "Genotype (Integer-encoded)"

            # Support both ternary and quaternary encodings; keep a stable order
            base_order = ["N", "0", "1", "2", "3"]

        # Include any unexpected symbols at the end (sorted) so nothing is
        # dropped
        extras = sorted(set(s.unique()) - set(base_order))
        full_order = base_order + [e for e in extras if e not in base_order]

        # Count and reindex to show zero-count categories
        counts = s.value_counts().reindex(full_order, fill_value=0)
        df = counts.rename_axis("Genotype").reset_index(name="Count")
        df["Percent"] = df["Count"] / df["Count"].sum() * 100

        title = "Imputed Genotype Counts" if is_imputed else "Genotype Counts"

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.despine(fig=fig)

        ax = sns.barplot(
            data=df,
            x="Genotype",
            y="Percent",
            hue="Genotype",
            order=full_order,
            errorbar=None,
            ax=ax,
            palette="Set1",
            legend=False,
            fill=True,
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Percent")
        ax.set_title(title)
        ax.set_ylim([0, 50])

        fig.tight_layout()

        suffix = "imputed" if is_imputed else "original"
        fn = self.output_dir / f"gt_distributions_{suffix}.{self.plot_format}"
        fig.savefig(fn, dpi=300)

        if self.show_plots:
            plt.show()
        plt.close(fig)
