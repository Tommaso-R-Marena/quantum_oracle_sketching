"""Shared plotting utilities for real-dataset experiments.

Provides parametric hybrid plots (accuracy vs. machine size) and common
visualization helpers used across all dataset experiments.
"""

from __future__ import annotations

from typing import Any

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "sans",
        "font.serif": ["Google Sans"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.figsize": (3.5, 2.5),
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "legend.frameon": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }
)

COLORS: dict[str, str] = {
    "quantum": "#CD591A",
    "streaming": "#2657AF",
    "sparse": "#606060",
}

LABELS: dict[str, str] = {
    "streaming": "Classical streaming",
    "sparse": "Classical sparse / QRAM",
    "quantum": "Quantum oracle sketching",
}

MARKERS: dict[str, str] = {"streaming": "P", "sparse": "X", "quantum": "D"}

FIGSIZE: tuple[float, float] = (3.5, 3.5)

MARKERSIZE: dict[str, int] = {"streaming": 50, "sparse": 50, "quantum": 30}

LINEWIDTH_MARKER: dict[str, int] = {"streaming": 0, "sparse": 0, "quantum": 0}

NUM_MARKERS: int = 40


def plot_parametric_hybrid(
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    color: str,
    marker: str,
    label: str,
    linewidth: int,
    marker_size: int,
    num_markers: int = NUM_MARKERS,
) -> None:
    """Plot a parametric curve with horizontal error bars and scattered markers.

    Args:
        x_mean: Mean accuracy values.
        x_std: Standard errors of the mean for accuracy.
        y_mean: Machine size values (log scale).
        color: Line/marker color.
        marker: Marker style.
        label: Legend label.
        linewidth: Marker edge width.
        marker_size: Marker size in points.
        num_markers: Number of evenly spaced markers to show.
    """
    y_vals = np.array(y_mean)
    x_vals = np.array(x_mean)
    x_errs = np.array(x_std)

    plt.fill_betweenx(
        y_vals,
        x_vals - x_errs,
        x_vals + x_errs,
        color=color,
        alpha=0.2,
        edgecolor="none",
    )

    plt.plot(x_vals, y_vals, linestyle="-", color=color, linewidth=1.5, alpha=0.9)

    x_min, x_max = np.min(x_vals), np.max(x_vals)
    target_x = np.linspace(x_min, x_max, num=num_markers)
    marker_indices = []
    for tx in target_x:
        idx = int((np.abs(x_vals - tx)).argmin())
        if idx not in marker_indices:
            marker_indices.append(idx)

    plt.scatter(
        x_vals[marker_indices],
        y_vals[marker_indices],
        marker=marker,
        color=color,
        label=label,
        alpha=0.9,
        s=marker_size,
        linewidth=linewidth,
    )


def get_sorted_arrays(
    x_mean: list[float] | np.ndarray,
    x_std: list[float] | np.ndarray,
    y_mean: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort parametric data by the y-axis metric (machine size)."""
    data = sorted(zip(x_mean, x_std, y_mean), key=lambda item: item[2])
    return (
        np.array([d[0] for d in data]),
        np.array([d[1] for d in data]),
        np.array([d[2] for d in data]),
    )


def add_text_annotations(
    positions: dict[str, tuple[float, float]],
    ha: dict[str, str] | None = None,
) -> None:
    """Add text annotations for the three machine-size regimes.

    Args:
        positions: Dict with keys ``sparse``, ``streaming``, ``quantum`` mapping
            to ``(x, y)`` data coordinates.
        ha: Optional horizontal alignment overrides.
    """
    halo = [pe.withStroke(linewidth=3, foreground="white")]
    default_ha = {"sparse": "left", "streaming": "right", "quantum": "right"}
    ha = ha or default_ha

    for key in ("sparse", "streaming", "quantum"):
        x_pos, y_pos = positions[key]
        plt.text(
            x_pos,
            y_pos,
            LABELS[key],
            color=COLORS[key],
            fontsize=10,
            path_effects=halo,
            ha=ha.get(key, "left"),
        )


def finalize_accuracy_plot(
    title: str,
    xlabel: str = "Accuracy",
    ylabel: str = "Machine size",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] = (1e1, 1e7),
    xticks: list[float] | None = None,
    xtick_labels: list[str] | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Apply common formatting and save an accuracy-vs-size plot."""
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(*ylim)
    plt.tick_params(direction="in", which="both", top=False, right=True)
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title(title)
    plt.tight_layout()

    if xticks is not None:
        plt.xticks(xticks, xtick_labels or [f"{100 * x:.0f}%" for x in xticks])
    if xlim is not None:
        plt.xlim(*xlim)

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    plt.close()
