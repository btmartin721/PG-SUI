# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass
class ClassificationReportVisualizer:
    """Pretty plotting for scikit-learn classification reports (output_dict=True).

    Adds neon cyberpunk aesthetics, a per-class support overlay, and optional bootstrap confidence intervals.

    Attributes:
        retro_palette: Hex colors for neon vibe.
        background_hex: Matplotlib/Plotly dark background.
        grid_hex: Gridline color for dark theme.
        reset_kwargs: Keyword args for resetting Matplotlib rcParams.
        genotype_order: Canonical ordering for genotype/IUPAC class labels in plots.
        avg_order: Canonical ordering for average rows (when present).
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

    # Canonical label order used everywhere.
    # Edit/extend this if you want additional IUPAC or special tokens
    # ordered explicitly.
    genotype_order: List[str] = field(
        default_factory=lambda: ["A", "C", "G", "T", "K", "M", "R", "S", "W", "Y", "N"]
    )
    avg_order: List[str] = field(
        default_factory=lambda: [
            "micro avg",
            "macro avg",
            "weighted avg",
            "samples avg",
        ]
    )

    # ---------- Ordering helpers ----------
    @staticmethod
    def _normalize_label(label: str) -> str:
        """Normalize class labels for ordering comparisons (case-insensitive)."""
        return str(label).strip().upper()

    @staticmethod
    def _normalize_avg(label: str) -> str:
        """Normalize avg labels for ordering comparisons (case-insensitive)."""
        return str(label).strip().lower()

    @staticmethod
    def _natural_sort_key(s: str):
        """Natural sort key so '10' sorts after '2'."""
        parts = re.split(r"(\d+)", str(s))
        key = []
        for p in parts:
            if p.isdigit():
                key.append((0, int(p)))
            else:
                key.append((1, p.lower()))
        return key

    def _ordered_class_labels(self, labels: Union[pd.Index, List[str]]) -> List[str]:
        """Order non-avg class labels with genotype_order first, then natural-sorted remainder."""
        labels_list = [str(x) for x in list(labels)]
        if not labels_list:
            return []

        # Map normalized -> first-seen original label to preserve original
        #  formatting.
        norm_to_orig: Dict[str, str] = {}
        for lab in labels_list:
            n = self._normalize_label(lab)
            norm_to_orig.setdefault(n, lab)

        desired_norm = [self._normalize_label(x) for x in self.genotype_order]
        desired_set = set(desired_norm)

        ordered = [norm_to_orig[n] for n in desired_norm if n in norm_to_orig]

        # Append everything not in genotype_order
        # (natural sort; stable + de-dup)
        seen = set(self._normalize_label(x) for x in ordered)
        remainder = [
            lab
            for lab in labels_list
            if self._normalize_label(lab) not in desired_set
            and self._normalize_label(lab) not in seen
        ]
        remainder_sorted = sorted(remainder, key=self._natural_sort_key)
        return ordered + remainder_sorted

    def _ordered_avg_labels(self, labels: Union[pd.Index, List[str]]) -> List[str]:
        """Order avg labels with avg_order first, then alpha remainder.

        Args:
            labels (Union[pd.Index, List[str]]): List of avg labels.

        Returns:
            List[str]: Ordered list of avg labels.
        """
        labels_list = [str(x) for x in list(labels)]
        if not labels_list:
            return []

        norm_to_orig: Dict[str, str] = {}
        for lab in labels_list:
            n = self._normalize_avg(lab)
            norm_to_orig.setdefault(n, lab)

        preferred = []
        preferred_set = set(self.avg_order)
        for pref in self.avg_order:
            if pref in norm_to_orig:
                preferred.append(norm_to_orig[pref])

        seen = set(self._normalize_avg(x) for x in preferred)
        remainder = [
            lab
            for lab in labels_list
            if self._normalize_avg(lab) not in preferred_set
            and self._normalize_avg(lab) not in seen
        ]
        remainder_sorted = sorted(remainder, key=lambda x: x.lower())
        return preferred + remainder_sorted

    def _ordered_report_index(self, idx: Union[pd.Index, List[str]]) -> List[str]:
        """Order full report index: classes first (genotype_order), avg rows last (avg_order).

        Args:
            idx (Union[pd.Index, List[str]]): Index from classification report DataFrame.

        Returns:
            List[str]: Ordered list of index labels.
        """
        labels = [str(x) for x in list(idx)]
        is_avg = [("avg" in lab.lower()) for lab in labels]
        class_labels = [lab for lab, a in zip(labels, is_avg) if not a]
        avg_labels = [lab for lab, a in zip(labels, is_avg) if a]
        return self._ordered_class_labels(class_labels) + self._ordered_avg_labels(
            avg_labels
        )

    def _apply_ordering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex df to canonical ordering (classes then avgs).

        Args:
            df (pd.DataFrame): DataFrame from classification report.

        Returns:
            pd.DataFrame: Reindexed DataFrame.
        """
        ordered = self._ordered_report_index(df.index)
        # Only keep labels that exist (avoid introducing all genotype_order labels as NaN rows)
        ordered = [x for x in ordered if x in df.index]
        return df.reindex(ordered)

    # ---------- Core data prep ----------
    def to_dataframe(self, report: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Convert sklearn classification_report output_dict to a tidy DataFrame.

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
                acc_val = df.loc["accuracy", "precision"]
                if pd.api.types.is_number(acc_val):
                    df.attrs["accuracy"] = float(str(acc_val))
            except Exception:
                squeezed_val = df.loc["accuracy"].squeeze()
                if pd.api.types.is_number(squeezed_val):
                    df.attrs["accuracy"] = float(str(squeezed_val))
            df = df.drop(index="accuracy", errors="ignore")

        df.index = df.index.astype(str)

        num_cols = ["precision", "recall", "f1-score", "support"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

        # Apply canonical ordering (classes then avg rows)
        df = self._apply_ordering(df)
        return df

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
            msg = "boot_reports is empty; provide at least one dict."
            raise ValueError(msg)

        frames = [self.to_dataframe(rep) for rep in boot_reports]

        # Union of indices across repeats, ordered canonically.
        union_idx = set().union(*[set(f.index) for f in frames])
        common_index = self._ordered_report_index(list(union_idx))
        common_index = [x for x in common_index if x in union_idx]

        arrs = []
        for f in frames:
            sub = f.reindex(common_index)
            arrs.append(sub[[m for m in metrics]].to_numpy(dtype=float))
        arr = np.stack(arrs, axis=0)  # (B, C, M)

        alpha = (1 - ci) / 2
        lower_q = 100 * alpha
        upper_q = 100 * (1 - alpha)

        lower = np.nanpercentile(arr, lower_q, axis=0)  # (C, M)
        upper = np.nanpercentile(arr, upper_q, axis=0)  # (C, M)
        mean = np.nanmean(arr, axis=0)  # (C, M)

        out = pd.DataFrame(index=common_index)
        column_tuples = []
        for j, m in enumerate(metrics):
            out[(m, "lower")] = lower[:, j]
            out[(m, "upper")] = upper[:, j]
            out[(m, "mean")] = mean[:, j]
            column_tuples.extend([(m, "lower"), (m, "upper"), (m, "mean")])

        out.columns = pd.MultiIndex.from_tuples(column_tuples)
        return out

    # ---------- Palettes & styles ----------
    def _retro_cmap(self, n: int = 256) -> LinearSegmentedColormap:
        """Create a neon gradient colormap.

        Args:
            n (int): Number of discrete colors in the colormap.

        Returns:
            LinearSegmentedColormap: Neon-themed colormap.
        """
        anchors = ["#241937", "#7d00ff", "#ff00ff", "#ff6ec7", "#00f0ff", "#00ff9f"]
        return LinearSegmentedColormap.from_list("retro_neon", anchors, N=n)

    def _set_mpl_style(self) -> None:
        """Apply a dark neon Matplotlib theme."""
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
        plt.rcParams.update(mpl.rcParamsDefault)
        mpl.rcParams.update(mpl.rcParamsDefault)

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

        Args:
            df (pd.DataFrame): DataFrame from to_dataframe().
            title (str): Plot title.
            classes_only (bool): Whether to include only classes (exclude avg rows).
            figsize (Tuple[int, int]): Figure size.
            annot_decimals (int): Decimal places for annotations.
            vmax (float): Max value for colormap scaling.
            vmin (float): Min value for colormap scaling.
            show_support_strip (bool): Whether to show a support strip on the right.

        Returns:
            Figure: Matplotlib figure.
        """
        self._set_mpl_style()

        work = df.copy()
        # Ensure canonical ordering even if caller didn't use to_dataframe().
        work = self._apply_ordering(work)

        if classes_only:
            work = work[~work.index.str.contains("avg", case=False, regex=True)]
            # Re-apply class ordering after filtering
            work = work.reindex(self._ordered_class_labels(work.index))

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

        if show_support_strip and "support" in work.columns:
            supports = work["support"].astype(float).fillna(0.0).to_numpy()
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
            df (pd.DataFrame): DataFrame from to_dataframe().
            title (str): Plot title.
            classes_only (bool): Whether to include only classes (exclude avg rows).
            figsize (Tuple[int, int]): Figure size.
            bar_alpha (float): Alpha transparency for bars.
            ci_df (Optional[pd.DataFrame]): DataFrame from compute_ci() for CI bars (optional).

        Returns:
            Figure: Matplotlib figure.
        """
        self._set_mpl_style()

        work = df.copy()
        work = self._apply_ordering(work)

        if classes_only:
            work = work[~work.index.str.contains("avg", case=False, regex=True)]
            classes = self._ordered_class_labels(work.index)
            work = work.reindex(classes)
        else:
            # If including avgs, only plot classes on x-axis for bars.
            classes = self._ordered_class_labels(
                work.loc[~work.index.str.contains("avg", case=False, regex=True)].index
            )

        metric_cols = ["precision", "recall", "f1-score"]
        lng = (
            work[metric_cols]
            .reset_index(names="class")
            .melt(id_vars="class", var_name="metric", value_name="score")
            .dropna(subset=["score"])
        )

        metrics = metric_cols
        palette = self.retro_palette[: len(metrics)]

        x = np.arange(len(classes))
        width = 0.25
        offsets = np.linspace(-width, width, num=len(metrics))

        fig, ax = plt.subplots(figsize=figsize)

        ax2 = ax.twinx()
        supports = (
            work.reindex(classes)["support"].astype(float).fillna(0.0).to_numpy()
            if "support" in work.columns
            else np.zeros(len(classes), dtype=float)
        )

        ax2.plot(
            x,
            np.asarray(supports),
            linestyle="None",
            marker="o",
            markersize=6,
            markerfacecolor="#39ff14",
            markeredgecolor="#ffffff",
            alpha=0.9,
            label="Support",
        )

        for i, m in enumerate(metrics):
            vals = (
                lng.loc[lng["metric"].eq(m)]
                .set_index("class")
                .reindex(classes)["score"]
                .to_numpy(dtype=float)
            )

            yerr = None
            if ci_df is not None and (m, "lower") in ci_df.columns:
                ci_reindexed = ci_df.reindex(classes)
                lows = ci_reindexed[(m, "lower")].to_numpy(dtype=float)
                ups = ci_reindexed[(m, "upper")].to_numpy(dtype=float)
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

        ax2.set_ylabel("Support")
        ax2.grid(False)
        ax2.set_ylim(0, max(1.0, float(np.asarray(supports).max()) * 1.15))
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
            df (pd.DataFrame): DataFrame from to_dataframe().
            title (str): Plot title.
            top_k (int): Number of top classes by support to include.
            include_micro (bool): Whether to include micro avg.
            include_macro (bool): Whether to include macro avg.
            include_weighted (bool): Whether to include weighted avg.
            ci_df (Optional[pd.DataFrame]): DataFrame from compute_ci() for CI bands (optional).

        Returns:
            go.Figure: Plotly radar figure.
        """
        work = df.copy()
        work = self._apply_ordering(work)

        is_avg = work.index.str.contains("avg", case=False, regex=True)

        # --- choose top-k by support, but order those chosen by canonical genotype order ---
        class_block = work.loc[~is_avg].copy()
        if top_k is not None and top_k > 0 and "support" in class_block.columns:
            top_labels = (
                (
                    class_block["support"]
                    .astype(float)
                    .fillna(0.0)
                    .sort_values(ascending=False)
                )
                .head(top_k)
                .index.tolist()
            )
            ordered_top = self._ordered_class_labels(top_labels)
            classes = class_block.reindex(
                [x for x in ordered_top if x in class_block.index]
            )
        else:
            classes = class_block.reindex(self._ordered_class_labels(class_block.index))

        # --- averages in canonical order ---
        include_map = {
            "micro avg": include_micro,
            "macro avg": include_macro,
            "weighted avg": include_weighted,
            "samples avg": True,  # keep if present; user can ignore via flags by removing from avg_order
        }
        avgs = []
        for name in self.avg_order:
            if include_map.get(name, True) and name in work.index:
                avgs.append((name, work.loc[name]))

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
            lows = [
                float(pd.to_numeric(ci_df.loc[name, (m, "lower")], errors="coerce"))
                for m in metrics
            ]
            ups = [
                float(pd.to_numeric(ci_df.loc[name, (m, "upper")], errors="coerce"))
                for m in metrics
            ]
            lows.append(lows[0])
            ups.append(ups[0])

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
                    hoverinfo="skip",
                    name=f"{name} CI",
                    showlegend=False,
                    opacity=0.20,
                )
            )
            fig.data[-1].fillcolor = f"{color}33"  # 8-digit hex w/ alpha

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
        radar_top_k: int = 10,
        boot_reports: Optional[List[Dict[str, Dict[str, float]]]] = None,
        ci: float = 0.95,
        show: bool = True,
    ) -> Dict[str, Union["Figure", go.Figure]]:
        """Generate all visuals, with optional CI from bootstrap reports.

        Args:
            report (Dict[str, Dict[str, float]]): Dictionary from `classification_report(..., output_dict=True)`.
            title_prefix (str): Prefix for plot titles.
            heatmap_classes_only (bool): Whether to only plot classes (exclude avg rows) in heatmap.
            radar_top_k (int): Number of top classes by support to include in radar plot.
            boot_reports (Optional[List[Dict[str, Dict[str, float]]]]): Optional list of bootstrap report dicts for CI computation.
            ci (float): Confidence level for CIs.
            show (bool): Whether to display the plots via plt.show().

        Returns:
            Dict[str, Union[Figure, go.Figure]]: Dictionary with keys:
                "heatmap_fig", "bars_fig", "radar_fig".
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.show()

        return {
            "heatmap_fig": heatmap_fig,
            "bars_fig": bars_fig,
            "radar_fig": radar_fig,
        }
