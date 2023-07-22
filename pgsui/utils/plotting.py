import os
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_genetic.utils import logbook_to_pandas
from sklearn.metrics import ConfusionMatrixDisplay

try:
    from . import misc
except (ModuleNotFoundError, ValueError, ImportError):
    from utils import misc


class Plotting:
    """Functions for plotting imputer scoring and results."""

    @staticmethod
    def plot_grid_search(cv_results, nn_method, prefix):
        """Plot cv_results\_ from a grid search for each parameter.

        Saves a figure to disk.

        Args:
            cv_results (np.ndarray): the cv_results\_ attribute from a trained grid search object.

            nn_method (str): Neural network algorithm name.

            prefix (str): Prefix to use for saving the plot to file.
        """
        ## Results from grid search
        results = pd.DataFrame(cv_results)
        means_test = [col for col in results if col.startswith("mean_test_")]
        filter_col = [col for col in results if col.startswith("param_")]
        params_df = results[filter_col].astype(str)
        for i, col in enumerate(means_test):
            params_df[col] = results[means_test[i]]

        # Get number of needed subplot rows.
        tot = len(filter_col)
        cols = 4
        rows = int(np.ceil(tot / cols))

        fig = plt.figure(1, figsize=(20, 10))
        fig.tight_layout(pad=3.0)

        # Set font properties.
        font = {"size": 12}
        plt.rc("font", **font)

        for i, p in enumerate(filter_col, start=1):
            ax = fig.add_subplot(rows, cols, i)

            # Plot each metric.
            for col in means_test:
                # Get maximum score for each parameter setting.
                df_plot = params_df.groupby(p)[col].agg("max")

                # Convert to float if not supposed to be string.
                try:
                    df_plot.index = df_plot.index.astype(float)
                except TypeError:
                    pass

                # Sort by index (numerically if possible).
                df_plot = df_plot.sort_index()

                # Remove prefix from score name.
                col_new_name = col[len("mean_test_") :]

                ax.plot(
                    df_plot.index.astype(str),
                    df_plot.values,
                    "-o",
                    label=col_new_name,
                )

                ax.legend(loc="best")

            param_new_name = p[len("param_") :]
            ax.set_xlabel(param_new_name.lower())
            ax.set_ylabel("Max Score")
            ax.set_ylim([0, 1])

        fig.savefig(
            os.path.join(
                f"{prefix}_output",
                "plots",
                "Unsupervised",
                nn_method,
                "gridsearch_metrics.pdf",
            ),
            bbox_inches="tight",
            facecolor="white",
        )

    @staticmethod
    def plot_metrics(metrics, num_classes, prefix, nn_method):
        """Plot AUC-ROC and Precision-Recall performance metrics for neural network classifier.

        Saves plot to PDF file on disk.

        Args:
            metrics (Dict[str, Any]): Per-class, micro, and macro-averaged metrics including accuracy, ROC-AUC, and Precision-Recall with Average Precision scores.

            num_classes (int): Number of classes evaluated.

            prefix (str): Prefix to use for output plot.

            nn_method (str): Neural network algorithm being used.
        """
        # Set font properties.
        font = {"size": 12}
        plt.rc("font", **font)

        fn = os.path.join(
            f"{prefix}_output",
            "plots",
            "Unsupervised",
            nn_method,
            f"auc_pr_curves.pdf",
        )
        fig = plt.figure(figsize=(20, 10))

        acc = round(metrics["accuracy"] * 100, 2)
        ham = round(metrics["hamming"], 2)

        fig.suptitle(
            f"Performance Metrics\nAccuracy: {acc}\nHamming Loss: {ham}"
        )
        axs = fig.subplots(nrows=1, ncols=2)
        plt.subplots_adjust(hspace=0.5)

        # Line weight
        lw = 2

        roc_auc = metrics["roc_auc"]
        pr_ap = metrics["precision_recall"]

        metric_list = [roc_auc, pr_ap]

        for metric, ax in zip(metric_list, axs):
            if "fpr_micro" in metric:
                prefix1 = "fpr"
                prefix2 = "tpr"
                lab1 = "ROC"
                lab2 = "AUC"
                xlab = "False Positive Rate"
                ylab = "True Positive Rate"
                title = "Receiver Operating Characteristic (ROC)"
                baseline = [0, 1]

            elif "recall_micro" in metric:
                prefix1 = "recall"
                prefix2 = "precision"
                lab1 = "Precision-Recall"
                lab2 = "AP"
                xlab = "Recall"
                ylab = "Precision"
                title = "Precision-Recall"
                baseline = [metric["baseline"], metric["baseline"]]

                # Plot iso-f1 curves.
                f_scores = np.linspace(0.2, 0.8, num=4)
                for i, f_score in enumerate(f_scores):
                    x = np.linspace(0.01, 1)
                    y = f_score * x / (2 * x - f_score)
                    ax.plot(
                        x[y >= 0],
                        y[y >= 0],
                        color="gray",
                        alpha=0.2,
                        linewidth=lw,
                        label="Iso-F1 Curves" if i == 0 else "",
                    )
                    ax.annotate(f"F1={f_score:0.1f}", xy=(0.9, y[45] + 0.02))

            # Plot ROC curves.
            ax.plot(
                metric[f"{prefix1}_micro"],
                metric[f"{prefix2}_micro"],
                label=f"Micro-averaged {lab1} Curve ({lab2} = {metric['micro']:.2f})",
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            ax.plot(
                metric[f"{prefix1}_macro"],
                metric[f"{prefix2}_macro"],
                label=f"Macro-averaged {lab1} Curve ({lab2} = {metric['macro']:.2f})",
                color="navy",
                linestyle=":",
                linewidth=4,
            )

            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(num_classes), colors):
                if f"{prefix1}_{i}" in metric:
                    ax.plot(
                        metric[f"{prefix1}_{i}"],
                        metric[f"{prefix2}_{i}"],
                        color=color,
                        lw=lw,
                        label=f"{lab1} Curve of class {i} ({lab2} = {metric[i]:.2f})",
                    )

            if "fpr_micro" in metric:
                # Make center baseline
                ax.plot(
                    baseline,
                    baseline,
                    "k--",
                    linewidth=lw,
                    label="No Classification Skill",
                )
            else:
                ax.plot(
                    [0, 1],
                    baseline,
                    "k--",
                    linewidth=lw,
                    label="No Classification Skill",
                )

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.05)
            ax.set_xlabel(f"{xlab}")
            ax.set_ylabel(f"{ylab}")
            ax.set_title(f"{title}")
            ax.legend(loc="best")

        fig.savefig(fn, bbox_inches="tight", facecolor="white")
        plt.close()
        plt.clf()
        plt.cla()

    @staticmethod
    def plot_search_space(
        estimator,
        height=2,
        s=25,
        features=None,
    ):
        """Make density and contour plots for showing search space during grid search.

        Modified from sklearn-genetic-opt function to implement exception handling.

        Args:
            estimator (sklearn estimator object): A fitted estimator from :class:`~sklearn_genetic.GASearchCV`.

            height (float, optional): Height of each facet. Defaults to 2.

            s (float, optional): Size of the markers in scatter plot. Defaults to 5.

            features (list, optional): Subset of features to plot, if ``None`` it plots all the features by default. Defaults to None.

        Returns:
            g (seaborn.PairGrid): Pair plot of the used hyperparameters during the search.
        """
        sns.set_style("white")

        df = logbook_to_pandas(estimator.logbook)
        if features:
            _stats = df[features]
        else:
            variables = [*estimator.space.parameters, "score"]
            _stats = df[variables]

        g = sns.PairGrid(_stats, diag_sharey=False, height=height)

        g = g.map_upper(sns.scatterplot, s=s, color="r", alpha=0.2)

        try:
            g = g.map_lower(
                sns.kdeplot,
                shade=True,
                cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True),
            )
        except np.linalg.LinAlgError as err:
            if "singular matrix" in str(err).lower():
                g = g.map_lower(sns.scatterplot, s=s, color="b", alpha=1.0)
            else:
                raise

        try:
            g = g.map_diag(
                sns.kdeplot,
                shade=True,
                palette="crest",
                alpha=0.2,
                color="red",
            )
        except np.linalg.LinAlgError as err:
            if "singular matrix" in str(err).lower():
                g = g.map_diag(sns.histplot, color="red", alpha=1.0, kde=False)

        return g

    @staticmethod
    def visualize_missingness(
        genotype_data,
        df,
        zoom=True,
        prefix="imputer",
        horizontal_space=0.6,
        vertical_space=0.6,
        bar_color="gray",
        heatmap_palette="magma",
        plot_format="pdf",
        dpi=300,
    ):
        """Make multiple plots to visualize missing data.

        Args:
            genotype_data (GenotypeData): Initialized GentoypeData object.

            df (pandas.DataFrame): DataFrame with snps to visualize.

            zoom (bool, optional): If True, zooms in to the missing proportion range on some of the plots. If False, the plot range is fixed at [0, 1]. Defaults to True.

            prefix (str, optional): Prefix for output directory and files. Plots and files will be written to a directory called <prefix>_reports. The report directory will be created if it does not already exist. If prefix is None, then the reports directory will not have a prefix. Defaults to 'imputer'.

            horizontal_space (float, optional): Set width spacing between subplots. If your plot are overlapping horizontally, increase horizontal_space. If your plots are too far apart, decrease it. Defaults to 0.6.

            vertical_space (float, optioanl): Set height spacing between subplots. If your plots are overlapping vertically, increase vertical_space. If your plots are too far apart, decrease it. Defaults to 0.6.

            bar_color (str, optional): Color of the bars on the non-stacked barplots. Can be any color supported by matplotlib. See matplotlib.pyplot.colors documentation. Defaults to 'gray'.

            heatmap_palette (str, optional): Palette to use for heatmap plot. Can be any palette supported by seaborn. See seaborn documentation. Defaults to 'magma'.

            plot_format (str, optional): Format to save plots. Can be any of the following: "pdf", "png", "svg", "ps", "eps". Defaults to "pdf".

            dpi (int): The resolution in dots per inch. Defaults to 300.

        Returns:
            pandas.DataFrame: Per-locus missing data proportions.
            pandas.DataFrame: Per-individual missing data proportions.
            pandas.DataFrame: Per-population + per-locus missing data proportions.
            pandas.DataFrame: Per-population missing data proportions.
            pandas.DataFrame: Per-individual and per-population missing data proportions.
        """

        loc, ind, poploc, poptotal, indpop = genotype_data.calc_missing(df)

        ncol = 3
        nrow = 1 if genotype_data.pops is None else 2

        fig, axes = plt.subplots(nrow, ncol, figsize=(8, 11))
        plt.subplots_adjust(wspace=horizontal_space, hspace=vertical_space)
        fig.suptitle("Missingness Report")

        ax = axes[0, 0]

        ax.set_title("Per-Individual")
        ax.barh(genotype_data.samples, ind, color=bar_color, height=1.0)
        if not zoom:
            ax.set_xlim([0, 1])
        ax.set_ylabel("Sample")
        ax.set_xlabel("Missing Prop.")
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False,
        )

        ax = axes[0, 1]

        ax.set_title("Per-Locus")
        ax.barh(
            range(genotype_data.num_snps), loc, color=bar_color, height=1.0
        )
        if not zoom:
            ax.set_xlim([0, 1])
        ax.set_ylabel("Locus")
        ax.set_xlabel("Missing Prop.")
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False,
        )

        id_vars = ["SampleID"]
        if poptotal is not None:
            ax = axes[0, 2]

            ax.set_title("Per-Population Total")
            ax.barh(poptotal.index, poptotal, color=bar_color, height=1.0)
            if not zoom:
                ax.set_xlim([0, 1])
            ax.set_xlabel("Missing Prop.")
            ax.set_ylabel("Population")

            ax = axes[1, 0]

            ax.set_title("Per-Population + Per-Locus")
            npops = len(poploc.columns)

            vmax = None if zoom else 1.0

            sns.heatmap(
                poploc,
                vmin=0.0,
                vmax=vmax,
                cmap=sns.color_palette(heatmap_palette, as_cmap=True),
                yticklabels=False,
                cbar_kws={"label": "Missing Prop."},
                ax=ax,
            )
            ax.set_xlabel("Population")
            ax.set_ylabel("Locus")

            id_vars.append("Population")

        melt_df = indpop.isna()
        melt_df["SampleID"] = genotype_data.samples
        indpop["SampleID"] = genotype_data.samples

        if poptotal is not None:
            melt_df["Population"] = genotype_data.pops
            indpop["Population"] = genotype_data.pops

        melt_df = melt_df.melt(value_name="Missing", id_vars=id_vars)
        melt_df.sort_values(by=id_vars[::-1], inplace=True)
        melt_df["Missing"].replace(False, "Present", inplace=True)
        melt_df["Missing"].replace(True, "Missing", inplace=True)

        ax = axes[0, 2] if poptotal is None else axes[1, 1]

        ax.set_title("Per-Individual")
        g = sns.histplot(
            data=melt_df,
            y="variable",
            hue="Missing",
            multiple="fill",
            ax=ax,
        )
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False,
        )
        g.get_legend().set_title(None)

        if poptotal is not None:
            ax = axes[1, 2]

            ax.set_title("Per-Population")
            g = sns.histplot(
                data=melt_df,
                y="Population",
                hue="Missing",
                multiple="fill",
                ax=ax,
            )
            g.get_legend().set_title(None)

        fig.savefig(
            os.path.join(
                f"{prefix}_output", "plots", f"missingness.{plot_format}"
            ),
            bbox_inches="tight",
            facecolor="white",
        )
        plt.cla()
        plt.clf()
        plt.close()

        return loc, ind, poploc, poptotal, indpop

    @staticmethod
    def run_and_plot_pca(
        original_genotype_data,
        imputer_object,
        prefix="imputer",
        n_components=3,
        center=True,
        scale=False,
        n_axes=2,
        point_size=15,
        font_size=15,
        plot_format="pdf",
        bottom_margin=0,
        top_margin=0,
        left_margin=0,
        right_margin=0,
        width=1088,
        height=700,
    ):
        """Runs PCA and makes scatterplot with colors showing missingness.

        Genotypes are plotted as separate shapes per population and colored according to missingness per individual.

        This function is run at the end of each imputation method, but can be run independently to change plot and PCA parameters such as ``n_axes=3`` or ``scale=True``.

        The imputed and original GenotypeData objects need to be passed to the function as positional arguments.

        PCA (principal component analysis) scatterplot can have either two or three axes, set with the n_axes parameter.

        The plot is saved as both an interactive HTML file and as a static image. Each population is represented by point shapes. The interactive plot has associated metadata when hovering over the points.

        Files are saved to a reports directory as <prefix>_output/imputed_pca.<plot_format|html>. Supported image formats include: "pdf", "svg", "png", and "jpeg" (or "jpg").

        Args:
            original_genotype_data (GenotypeData): Original GenotypeData object that was input into the imputer.

            imputer_object (Any imputer instance): Imputer object created when imputing. Can be any of the imputers, such as: ``ImputePhylo()``, ``ImputeUBP()``, and ``ImputeRandomForest()``.

            original_012 (pandas.DataFrame, numpy.ndarray, or List[List[int]], optional): Original 012-encoded genotypes (before imputing). Missing values are encoded as -9. This object can be obtained as ``df = GenotypeData.genotypes012_df``.

            prefix (str, optional): Prefix for report directory. Plots will be save to a directory called <prefix>_output/imputed_pca<html|plot_format>. Report directory will be created if it does not already exist. Defaults to "imputer".

            n_components (int, optional): Number of principal components to include in the PCA. Defaults to 3.

            center (bool, optional): If True, centers the genotypes to the mean before doing the PCA. If False, no centering is done. Defaults to True.

            scale (bool, optional): If True, scales the genotypes to unit variance before doing the PCA. If False, no scaling is done. Defaults to False.

            n_axes (int, optional): Number of principal component axes to plot. Must be set to either 2 or 3. If set to 3, a 3-dimensional plot will be made. Defaults to 2.

            point_size (int, optional): Point size for scatterplot points. Defaults to 15.

            plot_format (str, optional): Plot file format to use. Supported formats include: "pdf", "svg", "png", and "jpeg" (or "jpg"). An interactive HTML file is also created regardless of this setting. Defaults to "pdf".

            bottom_margin (int, optional): Adjust bottom margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            top (int, optional): Adjust top margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            left_margin (int, optional): Adjust left margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            right_margin (int, optional): Adjust right margin. If whitespace cuts off some of your plot, lower the corresponding margins. The default corresponds to that of plotly update_layout(). Defaults to 0.

            width (int, optional): Width of plot space. If your plot is cut off at the edges, even after adjusting the margins, increase the width and height. Try to keep the aspect ratio similar. Defaults to 1088.

            height (int, optional): Height of plot space. If your plot is cut off at the edges, even after adjusting the margins, increase the width and height. Try to keep the aspect ratio similar. Defaults to 700.

        Returns:
            numpy.ndarray: PCA data as a numpy array with shape (n_samples, n_components).

            sklearn.decomposision.PCA: Scikit-learn PCA object from sklearn.decomposision.PCA. Any of the sklearn.decomposition.PCA attributes can be accessed from this object. See sklearn documentation.

        Examples:
            >>>data = GenotypeData(
            >>>    filename="snps.str",
            >>>    filetype="structure2row",
            >>>    popmapfile="popmap.txt",
            >>>)
            >>>
            >>>ubp = ImputeUBP(genotype_data=data)
            >>>
            >>>components, pca = run_and_plot_pca(
            >>>    data,
            >>>    ubp,
            >>>    scale=True,
            >>>    center=True,
            >>>    plot_format="png"
            >>>)
            >>>
            >>>explvar = pca.explained_variance_ratio\_
        """
        report_path = os.path.join(f"{prefix}_output", "plots")
        Path(report_path).mkdir(parents=True, exist_ok=True)

        if n_axes > 3:
            raise ValueError(
                ">3 axes is not supported; n_axes must be either 2 or 3."
            )
        if n_axes < 2:
            raise ValueError(
                "<2 axes is not supported; n_axes must be either 2 or 3."
            )

        imputer = imputer_object.imputed

        df = misc.validate_input_type(
            imputer.genotypes012_df, return_type="df"
        )

        original_df = misc.validate_input_type(
            original_genotype_data.genotypes_012(fmt="pandas"),
            return_type="df",
        )

        original_df.replace(-9, np.nan, inplace=True)

        if center or scale:
            # Center data to mean. Scaling to unit variance is off.
            scaler = StandardScaler(with_mean=center, with_std=scale)
            pca_df = scaler.fit_transform(df)
        else:
            pca_df = df.copy()

        # Run PCA.
        model = PCA(n_components=n_components)
        components = model.fit_transform(pca_df)

        df_pca = pd.DataFrame(
            components[:, [0, 1, 2]], columns=["Axis1", "Axis2", "Axis3"]
        )

        df_pca["SampleID"] = original_genotype_data.samples
        df_pca["Population"] = original_genotype_data.pops
        df_pca["Size"] = point_size

        _, ind, _, _, _ = imputer.calc_missing(original_df, use_pops=False)
        df_pca["missPerc"] = ind

        my_scale = [("rgb(19, 43, 67)"), ("rgb(86,177,247)")]  # ggplot default

        z = "Axis3" if n_axes == 3 else None
        labs = {
            "Axis1": f"PC1 ({round(model.explained_variance_ratio_[0] * 100, 2)}%)",
            "Axis2": f"PC2 ({round(model.explained_variance_ratio_[1] * 100, 2)}%)",
            "missPerc": "Missing Prop.",
            "Population": "Population",
        }

        if z is not None:
            labs[
                "Axis3"
            ] = f"PC3 ({round(model.explained_variance_ratio_[2] * 100, 2)}%)"
            fig = px.scatter_3d(
                df_pca,
                x="Axis1",
                y="Axis2",
                z="Axis3",
                color="missPerc",
                symbol="Population",
                color_continuous_scale=my_scale,
                custom_data=["Axis3", "SampleID", "Population", "missPerc"],
                size="Size",
                size_max=point_size,
                labels=labs,
            )
        else:
            fig = px.scatter(
                df_pca,
                x="Axis1",
                y="Axis2",
                color="missPerc",
                symbol="Population",
                color_continuous_scale=my_scale,
                custom_data=["Axis3", "SampleID", "Population", "missPerc"],
                size="Size",
                size_max=point_size,
                labels=labs,
            )
        fig.update_traces(
            hovertemplate="<br>".join(
                [
                    "Axis 1: %{x}",
                    "Axis 2: %{y}",
                    "Axis 3: %{customdata[0]}",
                    "Sample ID: %{customdata[1]}",
                    "Population: %{customdata[2]}",
                    "Missing Prop.: %{customdata[3]}",
                ]
            ),
        )
        fig.update_layout(
            showlegend=True,
            margin=dict(
                b=bottom_margin,
                t=top_margin,
                l=left_margin,
                r=right_margin,
            ),
            width=width,
            height=height,
            legend_orientation="h",
            legend_title="Population",
            legend_title_font=dict(size=font_size),
            legend_title_side="top",
            font=dict(size=font_size),
        )
        fig.write_html(os.path.join(report_path, "imputed_pca.html"))
        fig.write_image(
            os.path.join(report_path, f"imputed_pca.{plot_format}"),
        )

        return components, model

    @staticmethod
    def plot_history(lod, nn_method, prefix="imputer"):
        """Plot model history traces. Will be saved to file.

        Args:
            lod (List[tf.keras.callbacks.History]): List of history objects.
            nn_method (str): Neural network method to plot. Possible options include: 'NLPCA', 'UBP', or 'VAE'. NLPCA and VAE get plotted the same, but UBP does it differently due to its three phases.
            prefix (str, optional): Prefix to use for output directory. Defaults to 'imputer'.

        Raises:
            ValueError: nn_method must be either 'NLPCA', 'UBP', or 'VAE'.
        """
        if nn_method == "NLPCA" or nn_method == "VAE" or nn_method == "SAE":
            title = nn_method
            fn = os.path.join(
                f"{prefix}_output",
                "plots",
                "Unsupervised",
                nn_method,
                "histplot.pdf",
            )

            if nn_method == "VAE":
                fig, axes = plt.subplots(2, 2)
                ax1 = axes[0, 0]
                ax2 = axes[0, 1]
                # ax3 = axes[1, 0]
                # ax4 = axes[1, 1]
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(title)
            fig.tight_layout(h_pad=3.0, w_pad=3.0)
            history = lod[0]

            acctrain = (
                "categorical_accuracy" if nn_method == "NLPCA" else "accuracy"
            )

            # if nn_method == "VAE":
            #     accval = "val_accuracy"
            #     # recon_loss = "reconstruction_loss"
            #     # kl_loss = "kl_loss"
            #     # val_recon_loss = "val_reconstruction_loss"
            #     # val_kl_loss = "val_kl_loss"
            #     lossval = "val_loss"

            if nn_method == "SAE":
                accval = "val_accuracy"
                lossval = "val_loss"

            # Plot train accuracy
            ax1.plot(history[acctrain])
            ax1.set_title("Model Accuracy")
            ax1.set_ylabel("Accuracy")
            ax1.set_xlabel("Epoch")
            ax1.set_ylim(bottom=0.0, top=1.0)
            ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

            labels = ["Train"]
            if nn_method == "SAE":
                # Plot validation accuracy
                ax1.plot(history[accval])
                labels.append("Validation")

            ax1.legend(labels, loc="best")

            # Plot model loss
            # if nn_method == "VAE":
            #     # Reconstruction loss only.
            #     ax2.plot(history["loss"])
            # ax2.plot(history[val_recon_loss])

            # # KL Loss
            # ax3.plot(history[kl_loss])
            # ax3.plot(history[val_kl_loss])
            # ax3.set_title("KL Divergence Loss")
            # ax3.set_ylabel("Loss")
            # ax3.set_xlabel("Epoch")
            # ax3.legend(labels, loc="best")

            # Total Loss (Reconstruction Loss + KL Loss)
            # ax4.plot(history["loss"])
            # ax4.plot(history[lossval])
            # ax4.set_title("Total Loss (Recon. + KL)")
            # ax4.set_ylabel("Loss")
            # ax4.set_xlabel("Epoch")
            # ax4.legend(labels, loc="best")

            # else:
            ax2.plot(history["loss"])

            if nn_method == "SAE":
                ax2.plot(history[lossval])

            ax2.set_title("Total Loss")
            ax2.set_ylabel("Loss")
            ax2.set_xlabel("Epoch")
            ax2.legend(labels, loc="best")

            fig.savefig(fn, bbox_inches="tight", facecolor="white")

            plt.close()
            plt.clf()

        elif nn_method == "UBP":
            fig = plt.figure(figsize=(12, 16))
            fig.suptitle(nn_method)
            fig.tight_layout(h_pad=2.0, w_pad=2.0)
            fn = os.path.join(
                f"{prefix}_output",
                "plots",
                "Unsupervised",
                nn_method,
                "histplot.pdf",
            )

            idx = 1
            for i, history in enumerate(lod, start=1):
                plt.subplot(3, 2, idx)
                title = f"Phase {i}"

                # Plot model accuracy
                ax = plt.gca()
                ax.plot(history["categorical_accuracy"])
                ax.set_title(f"{title} Accuracy")
                ax.set_ylabel("Accuracy")
                ax.set_xlabel("Epoch")
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.legend(["Training"], loc="best")

                # Plot model loss
                plt.subplot(3, 2, idx + 1)
                ax = plt.gca()
                ax.plot(history["loss"])
                ax.set_title(f"{title} Loss")
                ax.set_ylabel("Loss (MSE)")
                ax.set_xlabel("Epoch")
                ax.legend(["Train"], loc="best")

                idx += 2

            plt.savefig(fn, bbox_inches="tight", facecolor="white")

            plt.close()
            plt.clf()

        else:
            raise ValueError(
                f"nn_method must be either 'NLPCA', 'UBP', or 'VAE', but got {nn_method}"
            )

    @staticmethod
    def plot_certainty_heatmap(
        y_certainty, sample_ids=None, nn_method="VAE", prefix="imputer"
    ):
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
        fig.savefig(
            os.path.join(
                f"{prefix}_output",
                "plots",
                "Unsupervised",
                nn_method,
                "uncertainty_plot.png",
            ),
            bbox_inches="tight",
            facecolor="white",
        )

    @staticmethod
    def plot_confusion_matrix(
        y_true_1d, y_pred_1d, nn_method, prefix="imputer"
    ):
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ConfusionMatrixDisplay.from_predictions(
            y_true=y_true_1d, y_pred=y_pred_1d, ax=ax
        )

        outfile = os.path.join(
            f"{prefix}_output",
            "plots",
            "Unsupervised",
            nn_method,
            f"confusion_matrix_{nn_method}.png",
        )

        if os.path.isfile(outfile):
            os.remove(outfile)

        fig.savefig(outfile, facecolor="white")

    @staticmethod
    def plot_gt_distribution(df, plot_path):
        df = misc.validate_input_type(df, return_type="df")
        df_melt = pd.melt(df, value_name="Count")
        cnts = df_melt["Count"].value_counts()
        cnts.index.names = ["Genotype"]
        cnts = pd.DataFrame(cnts).reset_index()
        cnts.sort_values(by="Genotype", inplace=True)
        cnts["Genotype"] = cnts["Genotype"].astype(str)

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        g = sns.barplot(x="Genotype", y="Count", data=cnts, ax=ax)
        g.set_xlabel("Integer-encoded Genotype")
        g.set_ylabel("Count")
        g.set_title("Genotype Counts")
        for p in g.patches:
            g.annotate(
                f"{p.get_height():.1f}",
                (p.get_x() + 0.25, p.get_height() + 0.01),
                xytext=(0, 1),
                textcoords="offset points",
                va="bottom",
            )

        fig.savefig(
            os.path.join(plot_path, "genotype_distributions.png"),
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

    @staticmethod
    def plot_label_clusters(z_mean, labels, prefix="imputer"):
        """Display a 2D plot of the classes in the latent space."""
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))

        sns.scatterplot(x=z_mean[:, 0], y=z_mean[:, 1], ax=ax)
        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")

        outfile = os.path.join(
            f"{prefix}_output",
            "plots",
            "Unsupervised",
            "VAE",
            "label_clusters.png",
        )

        if os.path.isfile(outfile):
            os.remove(outfile)

        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
