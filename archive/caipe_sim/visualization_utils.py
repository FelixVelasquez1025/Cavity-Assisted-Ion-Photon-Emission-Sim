"""
Plotting helpers for the interactive Yb171+ visualizations.

These utilities isolate the matplotlib setup so the main script can focus on
physics and orchestration. Nothing is removed from the existing example; this
is a reusable toolkit for population and coherence plots.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def set_y_limits(ax: plt.Axes, data: np.ndarray | None, default: Tuple[float, float], padding: float = 0.1) -> None:
    """Adjust y-axis bounds with padding while handling empty or degenerate data."""
    if data is None or len(data) == 0:
        ax.set_ylim(*default)
        return
    low = float(np.nanmin(data))
    high = float(np.nanmax(data))
    if not np.isfinite(low) or not np.isfinite(high):
        ax.set_ylim(*default)
        return
    span = high - low
    pad = max(abs(high) * padding, abs(low) * padding, span * padding, 1e-6)
    if span < 1e-12:
        ax.set_ylim(low - pad, high + pad)
    else:
        ax.set_ylim(low - pad, high + pad)


def init_population_plot(t_us: np.ndarray, populations: Sequence[np.ndarray], labels: Sequence[str], title: str = "Populations") -> Tuple[plt.Figure, plt.Axes, List[plt.Line2D]]:
    """Create a population-vs-time plot and return figure, axis, and line handles."""
    fig, ax = plt.subplots(figsize=(9, 6))
    lines = []
    for y, label in zip(populations, labels):
        (ln,) = ax.plot(t_us, y, label=label)
        lines.append(ln)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Population")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    return fig, ax, lines


def init_coherence_plot(taulist_us: np.ndarray, values: np.ndarray | None, ylabel: str, label: str) -> Tuple[plt.Figure, plt.Axes, plt.Line2D]:
    """Create a single coherence plot (g^(1) or g^(2))."""
    fig, ax = plt.subplots(figsize=(8, 3))
    if values is not None and len(values) > 0:
        (ln,) = ax.plot(taulist_us, values, label=label)
        ax.set_xlim(taulist_us[0], taulist_us[-1])
    else:
        (ln,) = ax.plot([], [], label=label)
    ax.set_xlabel("Delay τ (µs)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    return fig, ax, ln


__all__ = [
    "set_y_limits",
    "init_population_plot",
    "init_coherence_plot",
]
