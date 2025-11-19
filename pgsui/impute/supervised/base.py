import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from snpio.utils.logging import LoggerManager

from pgsui.utils.classification_viz import ClassificationReportVisualizer
from pgsui.utils.logging_utils import configure_logger


class BaseImputer:
    """A base class for supervised, iterative imputer models.

    This class provides a common framework and shared functionality for imputers that use scikit-learn's `IterativeImputer`. It is not intended for direct instantiation. Child classes should inherit from this class and provide a specific estimator model (e.g., RandomForest, GradientBoosting).

    Notes:
        - A hyperparameter tuning workflow using Optuna.
        - Standardized data splitting, model training, and evaluation methods.
        - Utilities for creating output directories and handling model state.
    """

    def __init__(self, verbose: bool = False, debug: bool = False) -> None:
        """Initializes the BaseImputer class.

        This class sets up logging and verbosity/debug settings. It also contains methods that all supervised imputers will share.

        Note:
            Inheriting child classes must define `self.prefix` before calling `super().__init__()`, as it is required for logger initialization.

        Args:
            verbose (bool): If True, enables detailed logging output. Defaults to False.
            debug (bool): If True, enables debug mode. Defaults to False.
        """
        self.verbose = verbose
        self.debug = debug

        logman = LoggerManager(
            __name__, prefix=self.prefix, verbose=self.verbose, debug=self.debug
        )
        self.logger = configure_logger(
            logman.get_logger(), verbose=self.verbose, debug=self.debug
        )

    def _create_model_directories(self, prefix: str, outdirs: List[str]) -> None:
        """Creates the output directory structure for the imputer.

        This method sets up a standardized folder hierarchy for saving models, plots, metrics, and optimization results, organized by the model's name.

        Args:
            prefix (str): The prefix for the main output directory.
            outdirs (List[str]): A list of subdirectories to create (e.g., 'models', 'plots').
        """
        base_dir = Path(f"{prefix}_output") / "Supervised"
        for d in outdirs:
            subdir = base_dir / d / self.model_name
            setattr(self, f"{d}_dir", subdir)
            subdir.mkdir(parents=True, exist_ok=True)

    def _make_class_reports(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Dict[str, float],
        y_pred_proba: np.ndarray | None = None,
        labels: List[str] = ["REF", "HET", "ALT"],
    ) -> None:
        """Generate and save detailed classification reports and visualizations.

        3-class (zygosity) or 10-class (IUPAC) depending on `labels` length.

        Args:
            y_true (np.ndarray): True labels (1D array).
            y_pred (np.ndarray): Predicted labels (1D array).
            metrics (Dict[str, float]): Computed metrics.
            y_pred_proba (np.ndarray | None): Predicted probabilities (2D array). Defaults to None.
            labels (List[str], optional): Class label names
                (default: ["REF", "HET", "ALT"] for 3-class).
        """
        report_name = "zygosity" if len(labels) == 3 else "iupac"
        middle = "IUPAC" if report_name == "iupac" else "Zygosity"

        msg = f"{middle} Report (on {y_true.size} total genotypes)"
        self.logger.info(msg)

        if y_pred_proba is not None:
            self.plotter_.plot_metrics(
                y_true,
                y_pred_proba,
                metrics,
                label_names=labels,
                prefix=report_name,
            )

        self.plotter_.plot_confusion_matrix(
            y_true, y_pred, label_names=labels, prefix=report_name
        )

        self.logger.info(
            "\n"
            + classification_report(
                y_true,
                y_pred,
                labels=list(range(len(labels))),
                target_names=labels,
                zero_division=0,
            )
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
            output_dict=True,
        )

        with open(self.metrics_dir / f"{report_name}_report.json", "w") as f:
            json.dump(report, f, indent=4)

        viz = ClassificationReportVisualizer(reset_kwargs=self.plotter_.param_dict)

        plots = viz.plot_all(
            report,
            title_prefix=f"{self.model_name} {middle} Report",
            show=getattr(self, "show_plots", False),
            heatmap_classes_only=True,
        )

        for name, fig in plots.items():
            fout = self.plots_dir / f"{report_name}_report_{name}.{self.plot_format}"
            if hasattr(fig, "savefig"):
                fig.savefig(fout, dpi=300, facecolor="#111122")
                plt.close(fig)
            else:
                fig.write_html(file=fout.with_suffix(".html"))

        viz._reset_mpl_style()

    def _evaluate_012_and_plot(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """0/1/2 zygosity report & confusion matrix.

        This method generates a classification report and confusion matrix for genotypes encoded as 0, 1, or 2. If the data is haploid, it treats genotypes 1 and 2 as equivalent (presence of the alternate allele).

        Args:
            y_true (np.ndarray): True genotypes (0/1/2) for masked
            y_pred (np.ndarray): Predicted genotypes (0/1/2) for masked

        Raises:
            NotFittedError: If fit() and transform() have not been called.
        """
        labels = [0, 1, 2]
        # Haploid parity: fold ALT (2) into ALT/Present (1)
        if self.is_haploid_:
            y_true[y_true == 2] = 1
            y_pred[y_pred == 2] = 1
            labels = [0, 1]

        metrics = {
            "n_masked_test": int(y_true.size),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(
                y_true, y_pred, average="macro", labels=labels, zero_division=0
            ),
            "precision": precision_score(
                y_true, y_pred, average="macro", labels=labels, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, average="macro", labels=labels, zero_division=0
            ),
        }

        metrics.update({f"zygosity_{k}": v for k, v in metrics.items()})

        report_names = ["REF", "HET"] if self.is_haploid_ else ["REF", "HET", "ALT"]

        self.logger.info(
            f"\n{classification_report(y_true, y_pred, labels=labels, target_names=report_names, zero_division=0)}"
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=report_names,
            zero_division=0,
            output_dict=True,
        )

        viz = ClassificationReportVisualizer(reset_kwargs=self.plotter_.param_dict)

        plots = viz.plot_all(
            report,
            title_prefix=f"{self.model_name} Zygosity Report",
            show=getattr(self, "show_plots", False),
            heatmap_classes_only=True,
        )

        for name, fig in plots.items():
            fout = self.plots_dir / f"zygosity_report_{name}.{self.plot_format}"
            if hasattr(fig, "savefig"):
                fig.savefig(fout, dpi=300, facecolor="#111122")
                plt.close(fig)
            else:
                fig.write_html(file=fout.with_suffix(".html"))

        viz._reset_mpl_style()

        # Save JSON
        self._save_report(report, suffix="zygosity")

        # Confusion matrix
        self.plotter_.plot_confusion_matrix(
            y_true, y_pred, label_names=report_names, prefix="zygosity"
        )

    def _evaluate_iupac10_and_plot(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """10-class IUPAC report & confusion matrix.

        This method generates a classification report and confusion matrix for genotypes encoded using the 10 IUPAC codes (0-9). The IUPAC codes represent various nucleotide combinations, including ambiguous bases.

        Args:
            y_true (np.ndarray): True genotypes (0-9) for masked
            y_pred (np.ndarray): Predicted genotypes (0-9) for masked

        Raises:
            NotFittedError: If fit() and transform() have not been called.
        """
        labels_idx = list(range(10))
        labels_names = ["A", "C", "G", "T", "W", "R", "M", "K", "Y", "S"]

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(
                y_true, y_pred, average="macro", labels=labels_idx, zero_division=0
            ),
            "precision": precision_score(
                y_true, y_pred, average="macro", labels=labels_idx, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, average="macro", labels=labels_idx, zero_division=0
            ),
        }
        metrics.update({f"iupac_{k}": v for k, v in metrics.items()})

        self.logger.info(
            f"\n{classification_report(y_true, y_pred, labels=labels_idx, target_names=labels_names, zero_division=0)}"
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=labels_idx,
            target_names=labels_names,
            zero_division=0,
            output_dict=True,
        )

        viz = ClassificationReportVisualizer(reset_kwargs=self.plotter_.param_dict)

        plots = viz.plot_all(
            report,
            title_prefix=f"{self.model_name} IUPAC Report",
            show=getattr(self, "show_plots", False),
            heatmap_classes_only=True,
        )

        # Reset the style from Optuna's plotting.
        plt.rcParams.update(self.plotter_.param_dict)

        for name, fig in plots.items():
            fout = self.plots_dir / f"iupac_report_{name}.{self.plot_format}"
            if hasattr(fig, "savefig"):
                fig.savefig(fout, dpi=300, facecolor="#111122")
                plt.close(fig)
            else:
                fig.write_html(file=fout.with_suffix(".html"))

        # Reset the style
        viz._reset_mpl_style()

        # Save JSON
        self._save_report(report, suffix="iupac")

        # Confusion matrix
        self.plotter_.plot_confusion_matrix(
            y_true, y_pred, label_names=labels_names, prefix="iupac"
        )

    def _save_report(self, report_dict: Dict[str, float], suffix: str) -> None:
        """Save classification report dictionary as a JSON file.

        This method saves the provided classification report dictionary to a JSON file in the metrics directory, appending the specified suffix to the filename.

        Args:
            report_dict (Dict[str, float]): The classification report dictionary to save.
            suffix (str): Suffix to append to the filename (e.g., 'zygosity' or 'iupac').

        Raises:
            NotFittedError: If fit() and transform() have not been called.
        """
        if not self.is_fit_:
            msg = "No report to save. Ensure fit() has been called."
            raise NotFittedError(msg)

        out_fp = self.metrics_dir / f"classification_report_{suffix}.json"

        with open(out_fp, "w") as f:
            json.dump(report_dict, f, indent=4)

        self.logger.info(f"{self.model_name} {suffix} report saved to {out_fp}.")

    def _save_best_params(self, best_params: Dict[str, Any]) -> None:
        """Save the best hyperparameters to a JSON file.

        This method saves the best hyperparameters found during hyperparameter tuning to a JSON file in the optimization directory. The filename includes the model name for easy identification.

        Args:
            best_params (Dict[str, Any]): A dictionary of the best hyperparameters to save.
        """
        if not hasattr(self, "parameters_dir"):
            msg = "Attribute 'parameters_dir' not found. Ensure _create_model_directories() has been called."
            self.logger.error(msg)
            raise AttributeError(msg)

        fout = self.parameters_dir / "best_parameters.json"

        with open(fout, "w") as f:
            json.dump(best_params, f, indent=4)
