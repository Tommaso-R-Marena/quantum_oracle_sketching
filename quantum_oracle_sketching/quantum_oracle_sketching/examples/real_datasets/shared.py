"""Shared real-dataset experiment utilities and runner."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from qos.experiments.plotting import (
    COLORS,
    LABELS,
    MARKERS,
    add_text_annotations,
    finalize_accuracy_plot,
    get_sorted_arrays,
    plot_parametric_hybrid,
)


def compute_space_metrics(
    X_shape: tuple[int, int],
    row_sparsity: int,
    col_sparsity: int,
    nnz: int,
) -> dict[str, float]:
    """Compute machine-size estimates for streaming, sparse, and quantum access.

    Args:
        X_shape: ``(num_samples, feature_dim)``.
        row_sparsity: Maximum non-zero entries per row.
        col_sparsity: Maximum non-zero entries per column.
        nnz: Total non-zero entries.

    Returns:
        Dict with keys ``streaming``, ``sparse``, ``quantum``.
    """
    num_samples, feature_dim = X_shape
    sparsity = max(row_sparsity, col_sparsity)

    space_stream = float(feature_dim)
    space_sparse = float(nnz)
    space_quantum = float(
        2 * np.ceil(np.log2(num_samples + 2 * feature_dim))
        + np.ceil(np.log2(sparsity + 1))
        + 4
    )

    return {
        "streaming": space_stream,
        "sparse": space_sparse,
        "quantum": space_quantum,
    }


def run_classification_experiment(
    X,
    y,
    sweep_values: list[int],
    filter_fn,
    dataset_name: str,
    output_prefix: str,
    clf_alpha: float = 10.0,
    cv_folds: int = 5,
    clf_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a classification sweep over feature-filter thresholds.

    Args:
        X: Feature matrix (dense or sparse).
        y: Labels.
        sweep_values: List of threshold values to sweep.
        filter_fn: Callable ``(X, threshold) -> (X_filtered, info_dict)``.
        dataset_name: Human-readable dataset name.
        output_prefix: Prefix for output JSON/PDF files.
        clf_alpha: Ridge regularization strength.
        cv_folds: Number of cross-validation folds.
        clf_kwargs: Extra classifier kwargs.

    Returns:
        Nested dict of results suitable for JSON serialization.
    """
    results = {
        "thresholds": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "accuracies_mean": [],
        "accuracies_std": [],
    }

    tqdm.write(f"Running classification sweep on {dataset_name}...")
    for threshold in tqdm(sweep_values, desc="Sweep"):
        X_filtered, info = filter_fn(X, threshold)
        if X_filtered is None or (hasattr(X_filtered, "shape") and X_filtered.shape[1] == 0):
            continue

        shape = X_filtered.shape if hasattr(X_filtered, "shape") else X_filtered.get_shape()
        num_samples = shape[0]
        feature_dim = shape[1]

        if hasattr(X_filtered, "getnnz"):
            row_sparsity = int(X_filtered.getnnz(axis=1).max())
            col_sparsity = int(X_filtered.getnnz(axis=0).max())
            nnz = int(X_filtered.getnnz())
        else:
            row_sparsity = int(np.max(np.sum(X_filtered != 0, axis=1)))
            col_sparsity = int(np.max(np.sum(X_filtered != 0, axis=0)))
            nnz = int(np.count_nonzero(X_filtered))

        spaces = compute_space_metrics(
            (num_samples, feature_dim), row_sparsity, col_sparsity, nnz
        )

        kwargs = {"random_state": 42, "alpha": clf_alpha, "solver": "auto"}
        if clf_kwargs:
            kwargs.update(clf_kwargs)
        clf = RidgeClassifier(**kwargs)

        try:
            scores = cross_val_score(clf, X_filtered, y, cv=cv_folds)
        except Exception as exc:
            tqdm.write(f"Skipping threshold={threshold} due to error: {exc}")
            continue

        acc_mean = float(scores.mean())
        acc_sem = float(scores.std() / np.sqrt(len(scores)))

        results["thresholds"].append(threshold)
        results["space_streaming"].append(spaces["streaming"])
        results["space_sparse"].append(spaces["sparse"])
        results["space_quantum"].append(spaces["quantum"])
        results["accuracies_mean"].append(acc_mean)
        results["accuracies_std"].append(acc_sem)

    # Save JSON
    json_path = f"{output_prefix}_size_vs_accuracy.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    tqdm.write(f"Saved raw data to {json_path}")
    return results


def run_pca_experiment(
    X,
    y,
    sweep_values: list[int],
    filter_fn,
    dataset_name: str,
    output_prefix: str,
    n_components: int = 2,
    target_variance: float = 0.9,
) -> dict[str, Any]:
    """Run a PCA variance-recovery sweep over feature-filter thresholds.

    Args:
        X: Feature matrix.
        y: Labels (unused but kept for API consistency).
        sweep_values: Threshold values.
        filter_fn: ``(X, threshold) -> (X_filtered, info_dict)``.
        dataset_name: Dataset name.
        output_prefix: File prefix.
        n_components: Number of principal components to retain.
        target_variance: Target explained variance fraction.

    Returns:
        Results dict.
    """
    results = {
        "thresholds": [],
        "space_streaming": [],
        "space_sparse": [],
        "space_quantum": [],
        "variance_recovered": [],
    }

    tqdm.write(f"Running PCA sweep on {dataset_name}...")
    for threshold in tqdm(sweep_values, desc="PCA Sweep"):
        X_filtered, info = filter_fn(X, threshold)
        if X_filtered is None or (hasattr(X_filtered, "shape") and X_filtered.shape[1] == 0):
            continue

        shape = X_filtered.shape if hasattr(X_filtered, "shape") else X_filtered.get_shape()
        num_samples = shape[0]
        feature_dim = shape[1]

        if hasattr(X_filtered, "getnnz"):
            row_sparsity = int(X_filtered.getnnz(axis=1).max())
            col_sparsity = int(X_filtered.getnnz(axis=0).max())
            nnz = int(X_filtered.getnnz())
        else:
            row_sparsity = int(np.max(np.sum(X_filtered != 0, axis=1)))
            col_sparsity = int(np.max(np.sum(X_filtered != 0, axis=0)))
            nnz = int(np.count_nonzero(X_filtered))

        spaces = compute_space_metrics(
            (num_samples, feature_dim), row_sparsity, col_sparsity, nnz
        )

        try:
            if hasattr(X_filtered, "toarray"):
                X_dense = X_filtered.toarray()
            else:
                X_dense = np.asarray(X_filtered)
            svd = TruncatedSVD(n_components=min(n_components, feature_dim - 1), random_state=42)
            X_reduced = svd.fit_transform(X_dense)
            variance = float(np.sum(svd.explained_variance_ratio_))
        except Exception as exc:
            tqdm.write(f"Skipping threshold={threshold} due to SVD error: {exc}")
            continue

        results["thresholds"].append(threshold)
        results["space_streaming"].append(spaces["streaming"])
        results["space_sparse"].append(spaces["sparse"])
        results["space_quantum"].append(spaces["quantum"])
        results["variance_recovered"].append(variance)

    json_path = f"{output_prefix}_size_vs_variance.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    tqdm.write(f"Saved raw data to {json_path}")
    return results


def plot_experiment_results(
    results: dict[str, Any],
    title: str,
    output_prefix: str,
    xlabel: str = "Accuracy",
    xticks: list[float] | None = None,
    xtick_labels: list[str] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] = (1e1, 1e7),
    text_positions: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Plot classification or PCA results with the shared parametric style."""
    keys = ["streaming", "sparse", "quantum"]

    plt.figure(figsize=(3.5, 3.5))
    for k in keys:
        xm, xs, ym = get_sorted_arrays(
            results["accuracies_mean"],
            results["accuracies_std"],
            results[f"space_{k}"],
        )
        plot_parametric_hybrid(
            xm, xs, ym, COLORS[k], MARKERS[k], LABELS[k],
            LINEWIDTH_MARKER[k], MARKERSIZE[k]
        )

    if text_positions is not None:
        add_text_annotations(text_positions)

    finalize_accuracy_plot(
        title=title,
        xlabel=xlabel,
        ylim=ylim,
        xticks=xticks,
        xtick_labels=xtick_labels,
        xlim=xlim,
        save_path=f"{output_prefix}_size_vs_accuracy.pdf",
    )


def plot_pca_results(
    results: dict[str, Any],
    title: str,
    output_prefix: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] = (1e1, 1e7),
) -> None:
    """Plot PCA variance recovery results."""
    keys = ["streaming", "sparse", "quantum"]

    plt.figure(figsize=(3.5, 3.5))
    for k in keys:
        data = sorted(
            zip(results["variance_recovered"], results[f"space_{k}"]),
            key=lambda item: item[1],
        )
        x_vals = np.array([d[0] for d in data])
        y_vals = np.array([d[1] for d in data])

        plt.plot(x_vals, y_vals, linestyle="-", color=COLORS[k], linewidth=1.5, alpha=0.9)
        plt.scatter(
            x_vals, y_vals, marker=MARKERS[k], color=COLORS[k],
            label=LABELS[k], alpha=0.9, s=MARKERSIZE[k], linewidth=LINEWIDTH_MARKER[k]
        )

    plt.yscale("log")
    plt.xlabel("Variance recovered")
    plt.ylabel("Machine size")
    plt.ylim(*ylim)
    plt.tick_params(direction="in", which="both", top=False, right=True)
    plt.grid(True, which="major", ls="-", alpha=0.1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_size_vs_variance.pdf")
    print(f"Saved {output_prefix}_size_vs_variance.pdf")
    plt.close()
