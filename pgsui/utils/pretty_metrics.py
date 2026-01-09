from __future__ import annotations

import json
import math
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Optional Rich console; falls back to ASCII if not installed.
try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    _HAS_RICH = True
    _CONSOLE = Console()
except Exception:
    _HAS_RICH = False
    _CONSOLE = None


class PrettyMetrics:
    """Pretty-print and export nested metric dictionaries.

    Handles scalars, 1D sequences, and nested dicts. Summarizes sequences with
    mean ± std and last value. Uses `rich` colors if available.

    Attributes:
        metrics (Mapping[str, Any]): Metrics payload.
        precision (int): Decimal precision for numeric formatting.
        title (Optional[str]): Optional table title.
    """

    def __init__(
        self,
        metrics: Mapping[str, Any],
        *,
        precision: int = 4,
        title: Optional[str] = "Metrics",
    ) -> None:
        """Initialize the printer.

        Args:
            metrics (Mapping[str, Any]): Mapping of metric names to values. Values can be scalars, 1D sequences (lists or 1D numpy arrays), or nested dicts.
            precision (int): Decimal places for numeric formatting.
            title (Optional[str]): Optional table title shown when rendering.
        """
        self.metrics = metrics
        self.precision = precision
        self.title = title

    # ------------------------- Public API ---------------------------------

    def render(self) -> None:
        """Print the table to stdout.

        Uses a Rich table if Rich is installed. Otherwise prints a clean ASCII table.
        """
        rows = self._rows()

        if _HAS_RICH:
            table = Table(
                title=self.title or None, header_style="bold", show_lines=False
            )
            table.add_column("Metric", no_wrap=True)
            table.add_column("Value", justify="right")
            for metric, value, last_val in rows:
                table.add_row(metric, self._color_val_rich(metric, value, last_val))

            if _CONSOLE is not None:
                _CONSOLE.print(table)
            return

        # ASCII fallback
        m_w = max(len("Metric"), *(len(r[0]) for r in rows)) if rows else len("Metric")
        v_w = max(len("Value"), *(len(r[1]) for r in rows)) if rows else len("Value")
        title = (self.title or "Metrics").strip()
        line = "=" * (m_w + v_w + 5)
        print(title)
        print(line)
        print(f"{'Metric':<{m_w}} | {'Value':>{v_w}}")
        print("-" * (m_w + v_w + 5))
        for metric, value, _ in rows:
            print(f"{metric:<{m_w}} | {value:>{v_w}}")
        print(line)

    def to_text(self) -> str:
        """Return the rendered table as plain text.

        Uses Rich capture when available, else builds the ASCII table used by render().

        Returns:
            str: Pretty-printed metrics table.
        """
        rows = self._rows()
        if _HAS_RICH:
            table = Table(
                title=self.title or None, header_style="bold", show_lines=False
            )
            table.add_column("Metric", no_wrap=True)
            table.add_column("Value", justify="right")
            for metric, value, last_val in rows:
                table.add_row(metric, self._color_val_rich(metric, value, last_val))
            console = Console(record=True)
            console.print(table)
            return console.export_text(clear=False)

        m_w = max(len("Metric"), *(len(r[0]) for r in rows)) if rows else len("Metric")
        v_w = max(len("Value"), *(len(r[1]) for r in rows)) if rows else len("Value")
        title = (self.title or "Metrics").strip()
        line = "=" * (m_w + v_w + 5)
        parts = [
            title,
            line,
            f"{'Metric':<{m_w}} | {'Value':>{v_w}}",
            "-" * (m_w + v_w + 5),
        ]
        parts += [f"{metric:<{m_w}} | {value:>{v_w}}" for metric, value, _ in rows]
        parts.append(line)
        return "\n".join(parts)

    def to_dataframe(self):
        """Return a tidy pandas DataFrame of flattened metrics.

        Returns:
            pandas.DataFrame: Columns ['metric', 'value'] with scalars and sequence elements.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except Exception as e:  # pragma: no cover
            raise ImportError("pandas is required for to_dataframe()") from e

        out: List[Tuple[str, Any]] = []
        for k, v in self._flatten(self.metrics):
            if self._is_numeric(v):
                out.append((k, float(v)))  # type: ignore[arg-type]
            elif self._is_num_seq(v):
                seq = self._to_float_seq(v)
                out.extend((f"{k}[{i}]", float(x)) for i, x in enumerate(seq))
            else:
                out.append((k, str(v)))
        return pd.DataFrame(out, columns=["metric", "value"])

    def to_json(self) -> str:
        """Return a compact JSON string of the metrics.

        Returns:
            str: Compact JSON representation, suitable for logging artifacts.
        """
        return json.dumps(self.metrics, separators=(",", ":"), ensure_ascii=False)

    # ----------------------- Internal helpers -----------------------------

    def _rows(self) -> List[Tuple[str, str, Optional[float]]]:
        """Build rows as (metric_name, formatted_value, last_numeric_for_coloring)."""
        rows: List[Tuple[str, str, Optional[float]]] = []
        for name, val in self._flatten(self.metrics):
            if self._is_numeric(val):
                val_num = float(val)
                rows.append((name, self._format_scalar(val_num), val_num))
            elif self._is_num_seq(val):
                seq = self._to_float_seq(val)
                summary = self._fmt_mean_std(seq)
                last_val = seq[-1] if seq else None
                rows.append((name, summary, last_val))
            else:
                rows.append((name, str(val), None))
        return rows

    @staticmethod
    def _flatten(d: Mapping[str, Any], prefix: str = "") -> Iterable[Tuple[str, Any]]:
        for k, v in d.items():
            name = f"{prefix} → {k}" if prefix else str(k)
            if isinstance(v, Mapping):
                yield from PrettyMetrics._flatten(v, name)
            else:
                yield name, v

    def _fmt_mean_std(self, seq: Sequence[float]) -> str:
        """Format mean ± std and last value for a numeric sequence."""
        if np is not None:
            arr = np.asarray(seq, dtype=float)
            mean = float(arr.mean()) if arr.size else float("nan")
            std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            last = float(arr[-1]) if arr.size else float("nan")
        else:
            n = len(seq)
            if n == 0:
                mean = std = last = float("nan")
            else:
                mean = sum(seq) / n
                var = sum((x - mean) ** 2 for x in seq) / (n - 1) if n > 1 else 0.0
                std = var**0.5
                last = seq[-1]
        return f"{mean:.{self.precision}f} ± {std:.{self.precision}f}  (last {last:.{self.precision}f})"

    @staticmethod
    def _to_float_seq(val: Any) -> List[float]:
        if np is not None and hasattr(val, "tolist"):
            return list(map(float, val.tolist()))
        return list(map(float, val))

    def _format_scalar(self, v: float) -> str:
        if abs(v) >= 1000 or (0 < abs(v) < 1e-3):
            return f"{v:.{self.precision}e}"
        return f"{v:.{self.precision}f}"

    @staticmethod
    def _is_numeric(x: Any) -> bool:
        return (
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and math.isfinite(float(x))
        )

    @staticmethod
    def _is_num_seq(x: Any) -> bool:
        if (
            np is not None
            and isinstance(x, np.ndarray)
            and getattr(x, "ndim", 0) == 1
            and x.size > 0
        ):
            return np.issubdtype(x.dtype, np.number)
        return (
            isinstance(x, Sequence)
            and len(x) > 0
            and all(isinstance(v, (int, float)) for v in x)
        )

    @staticmethod
    def _better_is_higher(metric_name: str) -> Optional[bool]:
        name = metric_name.lower()
        higher = (
            "acc",
            "f1",
            "auc",
            "precision",
            "recall",
            "specificity",
            "r2",
            "matthews",
            "iou",
            "dice",
        )
        lower = (
            "loss",
            "mae",
            "mse",
            "rmse",
            "nll",
            "perplexity",
            "ece",
            "brier",
            "cross-entropy",
        )
        if any(k in name for k in higher):
            return True
        if any(k in name for k in lower):
            return False
        return None

    def _color_val_rich(
        self, metric: str, value_text: str, value_num: Optional[float]
    ) -> "Text | str":
        if not _HAS_RICH:
            return value_text
        t = Text(value_text)
        pref = self._better_is_higher(metric)
        if value_num is None or pref is None:
            return t
        if 0.0 <= value_num <= 1.0:
            good = value_num if pref else 1.0 - value_num
            if good >= 0.8:
                t.stylize("bold green")
            elif good >= 0.6:
                t.stylize("green")
            elif good <= 0.3:
                t.stylize("red")
        else:
            if pref is False and value_num <= 0.1:
                t.stylize("bold green")
        return t
