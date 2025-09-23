# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


@dataclass
class ClassificationReportVisualizer:
    """Pretty plotting for scikit-learn classification reports (output_dict=True).

    Adds neon cyberpunk aesthetics, a per-class support overlay, and optional bootstrap confidence intervals.

    Methods:
        - to_dataframe(): standardize sklearn dict -> tidy DataFrame
        - plot_heatmap(): seaborn heatmap (+ right-hand support strip)
        - plot_grouped_bars(): grouped bars with support markers and CI error bars
        - plot_radar(): interactive radar with optional CI bands and top-k classes
        - compute_ci(): aggregate bootstrap report dicts to CI bounds (per metric)
        - plot_all(): convenience wrapper

    Attributes:
        retro_palette: Hex colors for neon vibe.
        background_hex: Matplotlib/Plotly dark background.
        grid_hex: Gridline color for dark theme.
        reset_kwargs: Keyword args for resetting Matplotlib rcParams.
    """

    retro_palette: List[str] = field(
        default_factory=lambda: [
            "#ff00ff",
            "#9400ff",
            "#00f0ff",
            "#00ff9f",
            "#ff6ec7",
            "#7d00ff",
            "#39ff14",
            "#00bcd4",
        ]
    )
    background_hex: str = "#0a0a15"
    grid_hex: str = "#2a2a3a"
    reset_kwargs: Dict[str, bool | str] | None = None

    # ---------- Core data prep ----------
    def to_dataframe(self, report: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Convert sklearn classification_report output_dict to a tidy DataFrame.

        This method standardizes the output of scikit-learn's classification_report function.

        Args:
            report (Dict[str, Dict[str, float]]): Dictionary from `classification_report(..., output_dict=True)`.

        Returns:
            pd.DataFrame: Index are classes/avg rows (str). Columns include ["precision", "recall", "f1-score", "support"]. The "accuracy" scalar (if present) is stored in df.attrs["accuracy"], and the row is removed.
        """
        df = pd.DataFrame(report).T
        for col in ["precision", "recall", "f1-score", "support"]:
            if col not in df.columns:
                df[col] = np.nan

        if "accuracy" in df.index:
            # sklearn puts accuracy scalar in "accuracy" row, usually in 'precision'
            try:
                df.attrs["accuracy"] = float(df.loc["accuracy", "precision"])
            except Exception:
                df.attrs["accuracy"] = float(df.loc["accuracy"].squeeze())
            df = df.drop(index="accuracy", errors="ignore")

        df.index = df.index.astype(str)

        is_avg = df.index.str.contains("avg", case=False, regex=True)
        class_df = df.loc[~is_avg].copy()
        avg_df = df.loc[is_avg].copy()

        num_cols = ["precision", "recall", "f1-score", "support"]
        class_df[num_cols] = class_df[num_cols].apply(pd.to_numeric, errors="coerce")
        avg_df[num_cols] = avg_df[num_cols].apply(pd.to_numeric, errors="coerce")

        class_df = class_df.sort_index()
        tidy = pd.concat([class_df, avg_df], axis=0)
        return tidy

    def compute_ci(
        self,
        boot_reports: List[Dict[str, Dict[str, float]]],
        ci: float = 0.95,
        metrics: Tuple[str, ...] = ("precision", "recall", "f1-score"),
    ) -> pd.DataFrame:
        """Compute per-class bootstrap CIs from multiple report dicts.

        Args:
            boot_reports (List[Dict[str, Dict[str, float]]]): List of `output_dict=True` results over bootstrap repeats.
            ci (float): Confidence level (e.g., 0.95 for 95%).
            metrics (Tuple[str, ...]): Metrics to compute bounds for.

        Returns:
            pd.DataFrame: Multi-index columns with (metric, ["lower","upper","mean"]). Index contains any class/avg labels present in the bootstrap reports.
        """
        if not boot_reports:
            raise ValueError("boot_reports is empty; provide at least one dict.")

        # Gather frames; union of indices (classes/avg rows) across repeats
        frames = []
        for rep in boot_reports:
            df = self.to_dataframe(rep)
            frames.append(df)

        # Align on index, stack into 3D array (repeat x class x metric)
        common_index = sorted(set().union(*[f.index for f in frames]))
        arrs = []
        for f in frames:
            sub = f.reindex(common_index)
            arrs.append(sub[[m for m in metrics]].to_numpy(dtype=float))
        arr = np.stack(arrs, axis=0)  # shape: (B, C, M)

        alpha = (1 - ci) / 2
        lower_q = 100 * alpha
        upper_q = 100 * (1 - alpha)

        lower = np.nanpercentile(arr, lower_q, axis=0)  # (C, M)
        upper = np.nanpercentile(arr, upper_q, axis=0)  # (C, M)
        mean = np.nanmean(arr, axis=0)  # (C, M)

        out = pd.DataFrame(index=common_index)
        for j, m in enumerate(metrics):
            out[(m, "lower")] = lower[:, j]
            out[(m, "upper")] = upper[:, j]
            out[(m, "mean")] = mean[:, j]

        out.columns = pd.MultiIndex.from_tuples(out.columns)
        return out

    # ---------- Palettes & styles ----------
    def _retro_cmap(self, n: int = 256) -> LinearSegmentedColormap:
        """Create a neon gradient colormap.

        Args:
            n (int): Number of discrete colors in the colormap. Defaults to 256.

        Returns:
            LinearSegmentedColormap: The generated colormap.
        """
        anchors = ["#241937", "#7d00ff", "#ff00ff", "#ff6ec7", "#00f0ff", "#00ff9f"]
        return LinearSegmentedColormap.from_list("retro_neon", anchors, N=n)

    def _set_mpl_style(self) -> None:
        """Apply a dark neon Matplotlib theme.

        This method modifies global rcParams; call before plotting.
        """
        plt.rcParams.update(
            {
                "figure.facecolor": self.background_hex,
                "axes.facecolor": self.background_hex,
                "axes.edgecolor": self.grid_hex,
                "axes.labelcolor": "#e8e8ff",
                "xtick.color": "#d7d7ff",
                "ytick.color": "#d7d7ff",
                "grid.color": self.grid_hex,
                "text.color": "#f7f7ff",
                "axes.grid": True,
                "grid.linestyle": "--",
                "grid.linewidth": 0.5,
                "legend.facecolor": "#121222",
                "legend.edgecolor": self.grid_hex,
            }
        )

    def _reset_mpl_style(self) -> None:
        """Reset Matplotlib rcParams to default."""
        plt.rcParams.update(plt.rcParamsDefault)
        mpl.rcParams.update(plt.rcParamsDefault)

        if self.reset_kwargs is not None:
            plt.rcParams.update(self.reset_kwargs)
            mpl.rcParams.update(self.reset_kwargs)

    def plot_heatmap(
        self,
        df: pd.DataFrame,
        title: str = "Classification Report — Per-Class Metrics",
        classes_only: bool = True,
        figsize: Tuple[int, int] = (12, 6),
        annot_decimals: int = 3,
        vmax: float = 1.0,
        vmin: float = 0.0,
        show_support_strip: bool = False,
    ):
        """Plot a per-class heatmap with an optional right-hand support strip.

        This visualizes the classification metrics for each class.

        Args:
            df (pd.DataFrame): DataFrame from `to_dataframe()`.
            title (str): Plot title.
            classes_only (bool): If True, exclude avg rows.
            figsize (Tuple[int, int]): Matplotlib figure size.
            annot_decimals (int): Decimal places for annotations.
            vmax (float): Max heatmap value.
            vmin (float): Min heatmap value.
            show_support_strip (bool): If True, draw normalized support strip at right.

        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        self._set_mpl_style()

        work = df.copy()
        if classes_only:
            work = work[~work.index.str.contains("avg", case=False, regex=True)]

        metric_cols = ["precision", "recall", "f1-score"]
        heat = work[metric_cols].astype(float)

        fig, ax = plt.subplots(figsize=figsize)
        cmap = self._retro_cmap()
        sns.heatmap(
            heat,
            annot=True,
            fmt=f".{annot_decimals}f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor=self.grid_hex,
            cbar_kws={"label": "Score"},
            ax=ax,
        )
        ax.set_title(title, pad=12, fontweight="bold")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Class")

        # Optional support strip (normalized 0..1) as an inset axis
        if show_support_strip and "support" in work.columns:
            supports = work["support"].astype(float).fillna(0.0).values
            sup_norm = (supports - supports.min()) / (np.ptp(supports) + 1e-9)
            ax_strip = inset_axes(
                ax,
                width="2%",
                height="100%",
                loc="right",
                bbox_to_anchor=(0.03, 0.0, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )

            strip_data = sup_norm[:, None]  # (n_classes, 1)

            sns.heatmap(
                strip_data,
                cmap=self._retro_cmap(),
                cbar=True,
                cbar_kws={"label": "Support (normalized)"},
                xticklabels=False,
                yticklabels=False,
                vmin=0.0,
                vmax=1.0,
                linewidths=0.0,
                ax=ax_strip,
            )

            # Align strip y-limits to main heatmap
            ax_strip.set_ylim(ax.get_ylim())

        return fig

    def plot_grouped_bars(
        self,
        df: pd.DataFrame,
        title: str = "Per-Class Metrics (Grouped Bars)",
        classes_only: bool = True,
        figsize: Tuple[int, int] = (14, 7),
        bar_alpha: float = 0.9,
        ci_df: Optional[pd.DataFrame] = None,
    ):
        """Plot grouped bars for P/R/F1 with support markers and optional CI.

        Args:
            df (pd.DataFrame): DataFrame from `to_dataframe()`.
            title (str): Plot title.
            classes_only (bool): If True, exclude avg rows.
            figsize (Tuple[int, int]): Figure size.
            bar_alpha (float): Bar alpha.
            ci_df (Optional[pd.DataFrame]): Output of `compute_ci()`; adds error bars if provided.

        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        self._set_mpl_style()
        work = df.copy()
        if classes_only:
            work = work[~work.index.str.contains("avg", case=False, regex=True)]

        metric_cols = ["precision", "recall", "f1-score"]

        lng = (
            work[metric_cols]
            .reset_index(names="class")
            .melt(id_vars="class", var_name="metric", value_name="score")
            .dropna(subset=["score"])
        )

        homozygote_order = ["A", "C", "G", "T"]
        classes = homozygote_order + [
            c for c in lng["class"].unique().tolist() if c not in homozygote_order
        ]

        metrics = metric_cols
        palette = self.retro_palette[: len(metrics)]

        x = np.arange(len(classes))
        width = 0.25
        offsets = np.linspace(-width, width, num=len(metrics))

        fig, ax = plt.subplots(figsize=figsize)

        # Secondary axis for support markers
        ax2 = ax.twinx()
        supports = work.reindex(classes)["support"].astype(float).fillna(0.0).values

        ax2.plot(
            x,
            supports,
            linestyle="None",
            marker="o",
            markersize=6,
            markerfacecolor="#39ff14",
            markeredgecolor="#ffffff",
            alpha=0.9,
            label="Support",
        )

        # Plot bars with optional CI error bars
        for i, m in enumerate(metrics):
            vals = (
                lng.loc[lng["metric"].eq(m)]
                .set_index("class")
                .reindex(classes)["score"]
                .values
            )

            yerr = None
            if ci_df is not None and (m, "lower") in ci_df.columns:
                lows = ci_df.loc[classes, (m, "lower")].to_numpy(dtype=float)
                ups = ci_df.loc[classes, (m, "upper")].to_numpy(dtype=float)

                # Convert to symmetric error around the point estimate
                center = vals
                yerr = np.vstack([center - lows, ups - center])

            ax.bar(
                x + offsets[i],
                vals,
                width=width * 0.95,
                label=m.title(),
                color=palette[i % len(palette)],
                alpha=bar_alpha,
                edgecolor="#ffffff",
                linewidth=0.4,
                yerr=yerr,
                error_kw=dict(ecolor="#ffffff", elinewidth=0.9, capsize=3),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(title, pad=12, fontweight="bold")
        ax.legend(ncols=3, frameon=True, loc="upper left")

        # Configure secondary (support) axis
        ax2.set_ylabel("Support")
        ax2.grid(False)
        ax2.set_ylim(0, max(1.0, supports.max() * 1.15))
        ax2.legend(loc="upper right", frameon=True)

        ax.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        return fig

    def plot_radar(
        self,
        df: pd.DataFrame,
        title: str = "Macro/Weighted Averages & Top-K Class Radar",
        top_k: int = 5,
        include_micro: bool = True,
        include_macro: bool = True,
        include_weighted: bool = True,
        ci_df: Optional[pd.DataFrame] = None,
    ) -> go.Figure:
        """Interactive radar chart of averages + top-k classes; optional CI bands.

        Args:
            df (pd.DataFrame): DataFrame from `to_dataframe()`.
            title (str): Figure title.
            top_k (int): Include up to top_k classes by support (descending).
            include_micro (bool): Include micro avg trace if available.
            include_macro (bool): Include macro avg trace.
            include_weighted (bool): Include weighted avg trace.
            ci_df (Optional[pd.DataFrame]): Output of `compute_ci()`; draws semi-transparent CI bands.

        Returns:
            plotly.graph_objects.Figure: The interactive radar chart.
        """
        work = df.copy()

        is_avg = work.index.str.contains("avg", case=False, regex=True)
        classes = work.loc[~is_avg].copy().sort_values("support", ascending=False)
        if top_k is not None and top_k > 0:
            classes = classes.head(top_k)

        avgs = []
        if include_macro and "macro avg" in work.index:
            avgs.append(("macro avg", work.loc["macro avg"]))
        if include_weighted and "weighted avg" in work.index:
            avgs.append(("weighted avg", work.loc["weighted avg"]))
        if include_micro and "micro avg" in work.index:
            avgs.append(("micro avg", work.loc["micro avg"]))

        metrics = ["precision", "recall", "f1-score"]
        theta = metrics + [metrics[0]]

        fig = go.Figure()

        def _add_ci_band(name: str, color: str):
            if ci_df is None:
                return
            if not all([(m, "lower") in ci_df.columns for m in metrics]):
                return
            if name not in ci_df.index:
                return
            lows = [float(ci_df.loc[name, (m, "lower")]) for m in metrics]
            ups = [float(ci_df.loc[name, (m, "upper")]) for m in metrics]
            lows.append(lows[0])
            ups.append(ups[0])

            # Plotly polar CI band: plot upper path, then lower reversed with fill
            fig.add_trace(
                go.Scatterpolar(
                    r=ups,
                    theta=theta,
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatterpolar(
                    r=lows[::-1],
                    theta=theta[::-1],
                    mode="lines",
                    line=dict(width=0),
                    fill="toself",
                    fillcolor=(
                        color.replace("#", "rgba(") if False else None
                    ),  # placeholder
                    hoverinfo="skip",
                    name=f"{name} CI",
                    showlegend=False,
                    opacity=0.20,
                )
            )
            # Workaround: directly set fillcolor via marker color on last trace
            fig.data[-1].fillcolor = f"{color}33"  # add ~20% alpha

        # Add average traces with CI first
        for i, (name, row) in enumerate(avgs):
            r = [float(row.get(m, np.nan)) for m in metrics]
            r.append(r[0])
            color = self.retro_palette[i % len(self.retro_palette)]
            _add_ci_band(name, color)
            fig.add_trace(
                go.Scatterpolar(
                    r=r,
                    theta=theta,
                    name=name.title(),
                    mode="lines+markers",
                    line=dict(width=3, color=color),
                    marker=dict(size=7, color=color),
                    opacity=0.95,
                )
            )

        # Add class traces (top-k) with optional CI
        base_idx = len(avgs)
        for i, (cls, row) in enumerate(classes[metrics].iterrows()):
            r = [float(row.get(m, np.nan)) for m in metrics]
            r.append(r[0])
            color = self.retro_palette[(base_idx + i) % len(self.retro_palette)]
            _add_ci_band(str(cls), color)
            fig.add_trace(
                go.Scatterpolar(
                    r=r,
                    theta=theta,
                    name=str(cls),
                    mode="lines+markers",
                    line=dict(width=2, color=color),
                    marker=dict(size=6, color=color),
                    opacity=0.85,
                )
            )

        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor=self.background_hex,
            plot_bgcolor=self.background_hex,
            polar=dict(
                bgcolor="#111122",
                radialaxis=dict(range=[0, 1.05], showline=True, gridcolor="#33334d"),
                angularaxis=dict(gridcolor="#33334d"),
            ),
            legend=dict(
                bgcolor="#121222",
                bordercolor="#2a2a3a",
                borderwidth=1,
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                x=0.5,
                xanchor="center",
            ),
        )
        return fig

    def plot_all(
        self,
        report: Dict[str, Dict[str, float]],
        title_prefix: str = "Classification Report",
        heatmap_classes_only: bool = True,
        radar_top_k: int = 5,
        boot_reports: Optional[List[Dict[str, Dict[str, float]]]] = None,
        ci: float = 0.95,
        show: bool = True,
    ) -> Dict[str, Union[plt.Figure, go.Figure]]:
        """Generate all visuals, with optional CI from bootstrap reports.

        Args:
            report (Dict[str, Dict[str, float]]): The `output_dict=True` classification report (single run).
            title_prefix (str): Common prefix for titles.
            heatmap_classes_only (bool): Exclude averages in heatmap if True.
            radar_top_k (int): Number of top classes (by support) on radar.
            boot_reports (Optional[List[Dict[str, Dict[str, float]]]]): Optional list of bootstrap report dicts for CI.
            ci (float): Confidence level (e.g., 0.95).
            show (bool): If True, call plt.show() for Matplotlib figures.

        Returns:
            Dict[str, Union[matplotlib.figure.Figure, plotly.graph_objects.Figure]]: Keys: {"heatmap_fig", "bars_fig", "radar_fig"}.
        """
        df = self.to_dataframe(report)
        acc = df.attrs.get("accuracy", None)
        acc_str = f" (Accuracy: {acc:.3f})" if isinstance(acc, float) else ""

        ci_df = None
        if boot_reports:
            ci_df = self.compute_ci(boot_reports, ci=ci)

        heatmap_fig = self.plot_heatmap(
            df,
            title=f"{title_prefix} — Heatmap{acc_str}",
            classes_only=heatmap_classes_only,
            show_support_strip=False,
        )
        bars_fig = self.plot_grouped_bars(
            df,
            title=f"{title_prefix} — Grouped Bars{acc_str}",
            classes_only=True,
            ci_df=ci_df,
        )
        radar_fig = self.plot_radar(
            df,
            title=f"{title_prefix} — Averages & Top-{radar_top_k} Classes",
            top_k=radar_top_k,
            ci_df=ci_df,
        )

        if show:
            plt.show()

        return {
            "heatmap_fig": heatmap_fig,
            "bars_fig": bars_fig,
            "radar_fig": radar_fig,
        }
