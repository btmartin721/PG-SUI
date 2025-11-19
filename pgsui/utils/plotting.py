import logging
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, cast

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
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from snpio import SNPioMultiQC
from snpio.utils.logging import LoggerManager

from pgsui.utils import misc
from pgsui.utils.logging_utils import configure_logger

# Quiet Matplotlib/fontTools INFO logging when saving PDF/SVG
for name in (
    "fontTools",
    "fontTools.subset",
    "fontTools.ttLib",
    "matplotlib.font_manager",
):
    lg = logging.getLogger(name)
    lg.setLevel(logging.WARNING)
    lg.propagate = False


class Plotting:
    """Class for plotting imputer scoring and results.

    This class is used to plot the performance metrics of imputation models. It can plot ROC and Precision-Recall curves, model history, and the distribution of genotypes in the dataset.

    Example:
        >>> from pgsui import Plotting
        >>> plotter = Plotting(model_name="ImputeVAE", prefix="pgsui_test", plot_format="png")
        >>> plotter.plot_metrics(metrics, num_classes)
        >>> plotter.plot_history(history)
        >>> plotter.plot_confusion_matrix(y_true_1d, y_pred_1d)
        >>> plotter.plot_tuning(study, model_name, optimize_dir, target_name="Objective Value")
        >>> plotter.plot_gt_distribution(df)

    Attributes:
        model_name (str): Name of the model.
        prefix (str): Prefix for the output directory.
        plot_format (Literal["pdf", "png", "jpeg", "jpg", "svg"]): Format for the plots ('pdf', 'png', 'jpeg', 'jpg', 'svg').
        plot_fontsize (int): Font size for the plots.
        plot_dpi (int): Dots per inch for the plots.
        title_fontsize (int): Font size for the plot titles.
        show_plots (bool): Whether to display the plots inline or during execution.
        output_dir (Path): Directory where plots will be saved.
        logger (logging.Logger): Logger instance for logging messages.
    """

    def __init__(
        self,
        model_name: str,
        *,
        prefix: str = "pgsui",
        plot_format: Literal["pdf", "png", "jpeg", "jpg", "svg"] = "pdf",
        plot_fontsize: int = 18,
        plot_dpi: int = 300,
        title_fontsize: int = 20,
        despine: bool = True,
        show_plots: bool = False,
        verbose: int = 0,
        debug: bool = False,
        multiqc: bool = False,
        multiqc_section: Optional[str] = None,
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
            multiqc (bool): Whether to queue plots for a MultiQC HTML report. Defaults to False.
            multiqc_section (Optional[str]): Section name to use in MultiQC. Defaults to 'PG-SUI (<model_name>)'.
        """
        logman = LoggerManager(
            name=__name__, prefix=prefix, verbose=bool(verbose), debug=bool(debug)
        )
        self.logger = configure_logger(
            logman.get_logger(), verbose=bool(verbose), debug=bool(debug)
        )

        self.model_name = model_name
        self.prefix = prefix
        self.plot_format = plot_format
        self.plot_fontsize = plot_fontsize
        self.plot_dpi = plot_dpi
        self.title_fontsize = title_fontsize
        self.show_plots = show_plots

        # MultiQC configuration
        self.use_multiqc: bool = bool(multiqc)

        self.multiqc_section: str = (
            multiqc_section if multiqc_section is not None else f"PG-SUI ({model_name})"
        )

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

    # --------------------------------------------------------------------- #
    # Core plotting methods                                                #
    # --------------------------------------------------------------------- #
    def plot_tuning(
        self,
        study: optuna.study.Study,
        model_name: str,
        optimize_dir: Path,
        target_name: str = "Objective Value",
    ) -> None:
        """Plot the optimization history of a study.

        This method plots the optimization history of a study. The plot is saved to disk as a ``<plot_format>`` file.

        Args:
            study (optuna.study.Study): Optuna study object.
            model_name (str): Name of the model.
            optimize_dir (Path): Directory to save the optimization plots.
            target_name (str, optional): Name of the target value. Defaults to 'Objective Value'.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=ExperimentalWarning)

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

            od = optimize_dir
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

        # ---- MultiQC: Optuna tuning line graph + best-params table --------
        if self._multiqc_enabled():
            try:
                self._queue_multiqc_tuning(
                    study=study, model_name=model_name, target_name=target_name
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning(f"Failed to queue MultiQC tuning plots: {exc}")

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
        y_true_bin = np.asarray(label_binarize(y_true, classes=np.arange(num_classes)))

        # Containers
        fpr, tpr, roc_auc_vals = {}, {}, {}
        precision, recall, average_precision_vals = {}, {}, {}

        # Per-class ROC & PR
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc_vals[i] = auc(fpr[i], tpr[i])
            precision[i], recall[i], _ = precision_recall_curve(
                y_true_bin[:, i], y_pred_proba[:, i]
            )
            average_precision_vals[i] = average_precision_score(
                y_true_bin[:, i], y_pred_proba[:, i]
            )

        # Micro-averages
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_bin.ravel(), y_pred_proba.ravel()
        )
        roc_auc_vals["micro"] = auc(fpr["micro"], tpr["micro"])
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true_bin.ravel(), y_pred_proba.ravel()
        )
        average_precision_vals["micro"] = average_precision_score(
            y_true_bin, y_pred_proba, average="micro"
        )

        # Macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
        roc_auc_vals["macro"] = auc(fpr["macro"], tpr["macro"])

        # Macro-average PR
        all_recall = np.unique(np.concatenate([recall[i] for i in range(num_classes)]))
        mean_precision = np.zeros_like(all_recall)
        for i in range(num_classes):
            # recall[i] increases, but precision[i] is given over decreasing thresholds
            mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
        mean_precision /= num_classes
        average_precision_vals["macro"] = average_precision_score(
            y_true_bin, y_pred_proba, average="macro"
        )

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ROC
        axes[0].plot(
            fpr["micro"],
            tpr["micro"],
            label=f"Micro-average ROC (AUC = {roc_auc_vals['micro']:.2f})",
            linestyle=":",
            linewidth=4,
        )
        axes[0].plot(
            fpr["macro"],
            tpr["macro"],
            label=f"Macro-average ROC (AUC = {roc_auc_vals['macro']:.2f})",
            linestyle="--",
            linewidth=4,
        )
        for i in range(num_classes):
            axes[0].plot(
                fpr[i],
                tpr[i],
                label=f"{label_names[i]} ROC (AUC = {roc_auc_vals[i]:.2f})",
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
            label=f"Micro-average PR (AP = {average_precision_vals['micro']:.2f})",
            linestyle=":",
            linewidth=4,
        )
        axes[1].plot(
            all_recall,
            mean_precision,
            label=f"Macro-average PR (AP = {average_precision_vals['macro']:.2f})",
            linestyle="--",
            linewidth=4,
        )
        for i in range(num_classes):
            axes[1].plot(
                recall[i],
                precision[i],
                label=f"{label_names[i]} PR (AP = {average_precision_vals[i]:.2f})",
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

        prefix_for_name = f"{prefix}_" if prefix != "" else ""
        out_name = (
            f"{self.model_name}_{prefix_for_name}roc_pr_curves.{self.plot_format}"
        )
        fig.savefig(self.output_dir / out_name, bbox_inches="tight")
        if self.show_plots:
            plt.show()
        plt.close(fig)

        # ---- MultiQC: metrics table + per-class AUC/AP heatmap ------------
        if self._multiqc_enabled():
            try:
                self._queue_multiqc_metrics(
                    metrics=metrics,
                    roc_auc=roc_auc_vals,
                    average_precision=average_precision_vals,
                    label_names=label_names,
                    panel_prefix=prefix,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning(f"Failed to queue MultiQC metrics plots: {exc}")

            try:
                self._queue_multiqc_roc_curves(
                    fpr=fpr,
                    tpr=tpr,
                    label_names=label_names,
                    panel_prefix=prefix,
                )
                self._queue_multiqc_pr_curves(
                    precision=precision,
                    recall=recall,
                    label_names=label_names,
                    panel_prefix=prefix,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning(f"Failed to queue MultiQC ROC/PR curves: {exc}")

    def plot_history(
        self,
        history: Dict[str, List[float] | Dict[str, List[float]] | None] | None,
    ) -> None:
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

            # Ensure history is the nested dictionary type for ImputeUBP
            if not (
                isinstance(history, dict)
                and "Train" in history
                and isinstance(history["Train"], dict)
            ):
                msg = "For ImputeUBP, history must be a nested dictionary with phases."
                self.logger.error(msg)
                raise TypeError(msg)

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

        # ---- MultiQC: training-loss vs epoch linegraphs -------------------
        if self._multiqc_enabled():
            try:
                self._queue_multiqc_history(history=history)
            except Exception as exc:  # pragma: no cover
                self.logger.warning(f"Failed to queue MultiQC history plot: {exc}")

    def plot_confusion_matrix(
        self,
        y_true_1d: np.ndarray | pd.DataFrame | List[str | int] | torch.Tensor,
        y_pred_1d: np.ndarray | pd.DataFrame | List[str | int] | torch.Tensor,
        label_names: Sequence[str] | Dict[str, int] | None = None,
        prefix: str = "",
    ) -> None:
        """Plot a confusion matrix with optional class labels.

        This method plots a confusion matrix using true and predicted labels. The plot is saved to disk as a ``<plot_format>`` file.

        Args:
            y_true_1d (np.ndarray | pd.DataFrame | list | torch.Tensor): 1D array of true integer labels in [0, n_classes-1].
            y_pred_1d (np.ndarray | pd.DataFrame | list | torch.Tensor): 1D array of predicted integer labels in [0, n_classes-1].
            label_names (Sequence[str] | None): Optional sequence of class names (length must equal n_classes). If provided, both the internal label order and displayed tick labels will respect this order (assumed to be 0..n-1).
            prefix (str): Optional prefix for the output filename.

        Notes:
            - If `label_names` is None, the display labels default to the numeric class indices inferred from `y_true_1d ∪ y_pred_1d`.
        """
        y_true_1d = misc.validate_input_type(y_true_1d, return_type="array")
        y_pred_1d = misc.validate_input_type(y_pred_1d, return_type="array")

        if not isinstance(y_true_1d, np.ndarray) or y_true_1d.ndim != 1:
            msg = "y_true_1d must be a 1D array-like of true labels."
            self.logger.error(msg)
            raise TypeError(msg)

        if not isinstance(y_pred_1d, np.ndarray) or y_pred_1d.ndim != 1:
            msg = "y_pred_1d must be a 1D array-like of predicted labels."
            self.logger.error(msg)
            raise TypeError(msg)

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

        # Build a stable panel id before mutating prefix
        panel_suffix = f"{prefix}_" if prefix else ""
        panel_id = f"{self.model_name.lower()}_{panel_suffix}confusion_matrix"

        if prefix != "":
            prefix = f"{prefix}_"

        out_name = (
            f"{self.model_name.lower()}_{prefix}confusion_matrix.{self.plot_format}"
        )
        fig.savefig(self.output_dir / out_name, bbox_inches="tight")
        if self.show_plots:
            plt.show()
        plt.close(fig)

        # ---- MultiQC: confusion-matrix heatmap ----------------------------
        if self._multiqc_enabled():
            try:
                self._queue_multiqc_confusion(
                    y_true=y_true_1d,
                    y_pred=y_pred_1d,
                    labels=labels,
                    display_labels=display_labels,
                    panel_id=panel_id,
                )
            except Exception as exc:  # pragma: no cover
                self.logger.warning(f"Failed to queue MultiQC confusion matrix: {exc}")

    def plot_gt_distribution(
        self,
        X: np.ndarray | pd.DataFrame | list | torch.Tensor,
        is_imputed: bool = False,
    ) -> None:
        """Plot genotype distribution (IUPAC or integer-encoded).

        This plots counts for all genotypes present in X. It supports IUPAC single-letter genotypes and integer encodings. Missing markers '-', '.', '?' are normalized to 'N'. Bars are annotated with counts and percentages.

        Args:
            X (np.ndarray | pd.DataFrame | list | torch.Tensor): Array-like genotype matrix. Rows=loci, cols=samples (any orientation is OK). Elements are IUPAC one-letter genotypes (e.g., 'A','C','G','T','N','R',...) or integers (e.g., 0/1/2[/3]).
            is_imputed (bool): Whether these genotypes are imputed. Affects the title only. Defaults to False.
        """
        # Flatten X to a 1D Series
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
        ax.set_ylim((0.0, 50.0))

        fig.tight_layout()

        suffix = "imputed" if is_imputed else "original"
        fn = self.output_dir / f"gt_distributions_{suffix}.{self.plot_format}"
        fig.savefig(fn, dpi=300)

        if self.show_plots:
            plt.show()
        plt.close(fig)

        # ---- MultiQC: genotype-distribution barplot -----------------------
        if self._multiqc_enabled():
            try:
                self._queue_multiqc_gt_distribution(df=df, is_imputed=is_imputed)
            except Exception as exc:  # pragma: no cover
                self.logger.warning(
                    f"Failed to queue MultiQC genotype distribution: {exc}"
                )

    # --------------------------------------------------------------------- #
    # MultiQC helper methods                                                #
    # --------------------------------------------------------------------- #
    def _multiqc_enabled(self) -> bool:
        """Return True if MultiQC integration is active."""
        return bool(self.use_multiqc)

    def _queue_multiqc_tuning(
        self,
        *,
        study: optuna.study.Study,
        model_name: str,
        target_name: str,
    ) -> None:
        """Queue Optuna tuning results for MultiQC.

        Args:
            study (optuna.study.Study): Optuna study object.
            model_name (str): Name of the model.
            target_name (str): Name of the target value.
        """
        if not self._multiqc_enabled():
            return

        # trial number vs objective value line graph
        try:
            df_trials = study.trials_dataframe(attrs=("number", "value"))
        except Exception as exc:  # pragma: no cover
            self.logger.warning(
                f"Could not extract trials_dataframe for MultiQC: {exc}"
            )
            return

        if df_trials.empty or "value" not in df_trials:
            return

        data: Dict[str, Dict[int, int]] = {
            model_name: {
                row["number"]: row["value"]
                for _, row in df_trials.iterrows()
                if row["value"] is not None
            }
        }

        if not data[model_name]:
            return

        SNPioMultiQC.queue_linegraph(
            data=data,
            panel_id=f"{self.model_name}_optuna_history",
            section=self.multiqc_section,
            title=f"{self.model_name} Optuna Optimization History",
            index_label="Trial",
            description=f"Optuna optimization history for {self.model_name} "
            f"(target={target_name}).",
        )

        # best-params table
        try:
            best_value = study.best_value
            best_params = study.best_params
        except Exception:
            return

        if best_params:
            series = pd.Series(best_params, name="Best Value")
            series["objective"] = best_value
            SNPioMultiQC.queue_table(
                df=series,
                panel_id=f"{self.model_name}_optuna_best_params",
                section=self.multiqc_section,
                title=f"{self.model_name} Best Optuna Parameters",
                index_label="Parameter",
                description="Best Optuna hyperparameters and objective value.",
            )

    def _queue_multiqc_roc_curves(
        self,
        *,
        fpr: dict,
        tpr: dict,
        label_names: Sequence[str],
        panel_prefix: str,
    ) -> None:
        """Queue ROC and Precision-Recall curves for MultiQC.

        Args:
            fpr (dict): False positive rates for each class.
            tpr (dict): True positive rates for each class.
            label_names (Sequence[str]): Class names.
            panel_prefix (str): Optional prefix for panel IDs.
        """
        if not self._multiqc_enabled():
            return

        def _curve_to_mapping(
            x_vals: Sequence[float], y_vals: Sequence[float]
        ) -> Dict[float, float]:
            """Return {x: y} mapping expected by MultiQC linegraphs."""
            return {float(x): float(y) for x, y in zip(x_vals, y_vals)}

        data: Dict[str, Dict[float, float]] = {}

        # Only report the first three classes (MultiQC plot readability) plus micro/macro averages
        class_keys = sorted(k for k in fpr.keys() if isinstance(k, int))
        for idx in class_keys[:3]:
            label = label_names[idx] if idx < len(label_names) else f"Class {idx}"
            data[label] = _curve_to_mapping(fpr[idx], tpr[idx])

        for agg in ("micro", "macro"):
            if agg in fpr and agg in tpr:
                pretty_name = f"{agg.title()} Average"
                data[pretty_name] = _curve_to_mapping(fpr[agg], tpr[agg])

        if not data:
            return

        # ROC curves
        curve_data = cast(Dict[str, Dict[int, int]], data)

        SNPioMultiQC.queue_linegraph(
            data=curve_data,
            panel_id=(
                f"{self.model_name}_{panel_prefix}_roc_curves"
                if panel_prefix
                else f"{self.model_name}_roc_curves"
            ),
            section=self.multiqc_section,
            title=f"{self.model_name} ROC Curves",
            index_label="False Positive Rate",
            description="Multi-class ROC curves for PG-SUI predictions.",
        )

    def _queue_multiqc_pr_curves(
        self,
        *,
        precision: dict,
        recall: dict,
        label_names: Sequence[str],
        panel_prefix: str,
    ) -> None:
        """Queue Precision-Recall curves for MultiQC."""
        if not self._multiqc_enabled():
            return

        def _curve_to_mapping(
            x_vals: Sequence[float], y_vals: Sequence[float]
        ) -> Dict[float, float]:
            """Return {recall: precision} mapping expected by MultiQC linegraphs."""
            return {float(x): float(y) for x, y in zip(x_vals, y_vals)}

        data: Dict[str, Dict[float, float]] = {}

        # Only report the first three classes (MultiQC plot readability) plus micro/macro averages
        class_keys = sorted(k for k in recall.keys() if isinstance(k, int))
        for idx in class_keys[:3]:
            if idx not in precision or idx not in recall:
                continue
            label = label_names[idx] if idx < len(label_names) else f"Class {idx}"
            data[label] = _curve_to_mapping(recall[idx], precision[idx])

        for agg in ("micro", "macro"):
            if agg in precision and agg in recall:
                pretty_name = f"{agg.title()} Average"
                data[pretty_name] = _curve_to_mapping(recall[agg], precision[agg])

        if not data:
            return

        curve_data = cast(Dict[str, Dict[int, int]], data)

        SNPioMultiQC.queue_linegraph(
            data=curve_data,
            panel_id=(
                f"{self.model_name}_{panel_prefix}_pr_curves"
                if panel_prefix
                else f"{self.model_name}_pr_curves"
            ),
            section=self.multiqc_section,
            title=f"{self.model_name} Precision-Recall Curves",
            index_label="Recall",
            description="Multi-class Precision-Recall curves for PG-SUI predictions.",
        )

    def _queue_multiqc_metrics(
        self,
        *,
        metrics: Dict[str, float],
        roc_auc: Dict[object, float],
        average_precision: Dict[object, float],
        label_names: Sequence[str],
        panel_prefix: str,
    ) -> None:
        """Queue summary metrics and per-class AUC/AP for MultiQC.

        Args:
            metrics (Dict[str, float]): Summary metrics (accuracy, F1, etc.).
            roc_auc (Dict[object, float]): Per-class and aggregate ROC-AUC values.
            average_precision (Dict[object, float]): Per-class and aggregate average precision values.
            label_names (Sequence[str]): Class names.
            panel_prefix (str): Optional prefix for panel IDs.
        """
        if not self._multiqc_enabled():
            return

        # Summary metrics table (accuracy, F1, etc.)
        if metrics:
            series = pd.Series(metrics, name="Value")
            SNPioMultiQC.queue_table(
                df=series,
                panel_id=f"{self.model_name}_summary_metrics",
                section=self.multiqc_section,
                title=f"{self.model_name} Summary Metrics",
                index_label="Metric",
                description="Global evaluation metrics produced by PG-SUI.",
            )

        # Per-class ROC-AUC and AP heatmap
        rows: List[Dict[str, float | str]] = []

        # integer keys are classes; others are 'micro', 'macro'
        class_keys = [k for k in roc_auc.keys() if isinstance(k, int)]
        class_keys_sorted = sorted(class_keys)

        for i in class_keys_sorted:
            class_name = label_names[i] if i < len(label_names) else f"Class {i}"
            rows.append(
                {
                    "Class": str(class_name),
                    "ROC_AUC": float(roc_auc.get(i, np.nan)),
                    "AveragePrecision": float(average_precision.get(i, np.nan)),
                }
            )

        for agg in ("micro", "macro"):
            if agg in roc_auc:
                rows.append(
                    {
                        "Class": agg,
                        "ROC_AUC": float(roc_auc.get(agg, np.nan)),
                        "AveragePrecision": float(average_precision.get(agg, np.nan)),
                    }
                )

        if not rows:
            return

        df = pd.DataFrame(rows).set_index("Class")
        suffix = f"{panel_prefix}_" if panel_prefix else ""
        panel_id = f"{self.model_name}_{suffix}roc_pr_summary"

        SNPioMultiQC.queue_heatmap(
            df=df,
            panel_id=panel_id,
            section=self.multiqc_section,
            title=f"{self.model_name} ROC-AUC and Average Precision",
            index_label="Class",
            description=(
                "Per-class ROC-AUC and average precision for PG-SUI predictions (including micro/macro averages where available)."
            ),
        )

    def _queue_multiqc_history(
        self,
        *,
        history: Dict[str, List[float] | Dict[str, List[float]] | None] | None,
    ) -> None:
        """Queue training history (loss vs epoch) for MultiQC.

        Args:
            history (Dict[str, List[float]] | None): Dictionary with lists of history objects. Keys should be "Train" and "Validation".
        """
        if not self._multiqc_enabled() or history is None:
            return

        data: Dict[str, Dict[int, int]] = {}

        if self.model_name != "ImputeUBP":
            if not isinstance(history, dict) or "Train" not in history:
                return

            train_vals = pd.Series(history["Train"]).iloc[1:]

            data["Train"] = {
                epoch: val for epoch, val in enumerate(train_vals.values, start=1)
            }
        else:
            if not (
                isinstance(history, dict)
                and "Train" in history
                and isinstance(history["Train"], dict)
            ):
                return
            for phase in range(1, 4):
                key = f"Phase {phase}"
                if key not in history["Train"]:
                    continue
                series = pd.Series(history["Train"][key]).iloc[1:]
                data[key] = {
                    epoch: val for epoch, val in enumerate(series.values, start=1)
                }

        if not data:
            return

        SNPioMultiQC.queue_linegraph(
            data=data,
            panel_id=f"{self.model_name}_training_history",
            section=self.multiqc_section,
            title=f"{self.model_name} Training Loss per Epoch",
            index_label="Epoch",
            description="Training loss trajectory by epoch as recorded by PG-SUI.",
        )

    def _queue_multiqc_confusion(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: np.ndarray,
        display_labels: List[str] | np.ndarray,
        panel_id: str,
    ) -> None:
        """Queue confusion-matrix heatmap for MultiQC.

        Args:
            y_true (np.ndarray): 1D array of true integer labels.
            y_pred (np.ndarray): 1D array of predicted integer labels.
            labels (np.ndarray): Array of label indices to index the confusion matrix.
            display_labels (List[str] | np.ndarray): Labels to display on axes.
            panel_id (str): Panel ID for MultiQC.
        """
        if not self._multiqc_enabled():
            return

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        df_cm = pd.DataFrame(cm, index=display_labels, columns=display_labels)

        SNPioMultiQC.queue_heatmap(
            df=df_cm,
            panel_id=panel_id,
            section=self.multiqc_section,
            title=f"{self.model_name} Confusion Matrix",
            index_label="True Label",
            description=(
                "Confusion matrix for PG-SUI predictions. Rows correspond to true "
                "labels; columns correspond to predicted labels."
            ),
        )

    def _queue_multiqc_gt_distribution(
        self,
        *,
        df: pd.DataFrame,
        is_imputed: bool,
    ) -> None:
        """Queue genotype-distribution barplot for MultiQC.

        Args:
            df (pd.DataFrame): DataFrame with 'Genotype' and 'Percent' columns
            is_imputed (bool): Whether these genotypes are imputed.
        """
        if not self._multiqc_enabled():
            return

        if "Genotype" not in df.columns or "Percent" not in df.columns:
            return

        series = df.set_index("Genotype")["Percent"]
        suffix = "imputed" if is_imputed else "original"
        title = (
            f"{self.model_name} Imputed Genotype Distribution"
            if is_imputed
            else f"{self.model_name} Genotype Distribution"
        )

        SNPioMultiQC.queue_barplot(
            df=series,
            panel_id=f"{self.model_name}_gt_distribution_{suffix}",
            section=self.multiqc_section,
            title=title,
            index_label="Genotype",
            value_label="Percent",
            description=(
                "Genotype frequency distribution (percent of calls per genotype) computed by PG-SUI."
            ),
        )
