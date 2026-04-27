"""Shared real-dataset experiment utilities and runner."""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
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
    X_sparse,
) -> dict[str, int]:
    """
    Compute machine size for all four algorithms.
    Implements Zhao et al. (2025) Appendix A, Equations A1–A3 verbatim.

    Parameters
    ----------
    X_sparse : scipy.sparse matrix, shape (N, D)

    Returns
    -------
    dict:
        'space_sparse'    : int  — S_{c,sparse} = N_nnz          (Eq. A1)
        'space_qram'      : int  — S_QRAM = N_nnz                (Eq. A1)
        'space_streaming' : int  — S_{c,streaming} = D           (Eq. A2)
        'space_quantum'   : int  — S^{LS-SVM}_{QOS}              (Eq. A3)
        '_N', '_D', '_N_nnz', '_s' : raw values for debugging
    """
    X_sparse = sp.csr_matrix(X_sparse)
    N, D = X_sparse.shape
    N_nnz = X_sparse.nnz

    # Eq. A1
    space_sparse = N_nnz
    space_qram = N_nnz

    # Eq. A2
    space_streaming = D

    # Eq. A3
    # s = max nnz in any single row or column
    row_nnz = np.diff(X_sparse.tocsr().indptr)
    col_nnz = np.diff(X_sparse.tocsc().indptr)
    s = int(max(row_nnz.max(), col_nnz.max()))

    term1 = 2 * math.ceil(math.log2(max(N + 2 * D, 2)))
    term2 = math.ceil(math.log2(s + 1)) if s > 0 else 0
    # +3 ancilla qubits for block-encoding + 1 Hadamard test readout qubit
    space_quantum = term1 + term2 + 3 + 1

    return {
        "space_sparse": space_sparse,
        "space_qram": space_qram,
        "space_streaming": space_streaming,
        "space_quantum": space_quantum,
        "_N": N,
        "_D": D,
        "_N_nnz": N_nnz,
        "_s": s,
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
        "space_qram": [],
        "space_quantum": [],
        "accuracies_mean": [],
        "accuracies_std": [],
    }

    tqdm.write(f"Running classification sweep on {dataset_name}...")
    for threshold in tqdm(sweep_values, desc="Sweep"):
        X_filtered, info = filter_fn(X, threshold)
        if X_filtered is None or (hasattr(X_filtered, "shape") and X_filtered.shape[1] == 0):
            continue

        spaces = compute_space_metrics(X_filtered)

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
        results["space_streaming"].append(spaces["space_streaming"])
        results["space_sparse"].append(spaces["space_sparse"])
        results["space_qram"].append(spaces["space_qram"])
        results["space_quantum"].append(spaces["space_quantum"])
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
        "space_qram": [],
        "space_quantum": [],
        "variance_recovered": [],
    }

    tqdm.write(f"Running PCA sweep on {dataset_name}...")
    for threshold in tqdm(sweep_values, desc="PCA Sweep"):
        X_filtered, info = filter_fn(X, threshold)
        if X_filtered is None or (hasattr(X_filtered, "shape") and X_filtered.shape[1] == 0):
            continue

        spaces = compute_space_metrics(X_filtered)

        try:
            if hasattr(X_filtered, "toarray"):
                X_dense = X_filtered.toarray()
            else:
                X_dense = np.asarray(X_filtered)
            feature_dim = X_dense.shape[1]
            svd = TruncatedSVD(n_components=min(n_components, feature_dim - 1), random_state=42)
            X_reduced = svd.fit_transform(X_dense)
            variance = float(np.sum(svd.explained_variance_ratio_))
        except Exception as exc:
            tqdm.write(f"Skipping threshold={threshold} due to SVD error: {exc}")
            continue

        results["thresholds"].append(threshold)
        results["space_streaming"].append(spaces["space_streaming"])
        results["space_sparse"].append(spaces["space_sparse"])
        results["space_qram"].append(spaces["space_qram"])
        results["space_quantum"].append(spaces["space_quantum"])
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
    curve_specs = [
        ("quantum", "Quantum oracle sketching", "#CD591A", "-", "D", 0, 30),
        ("qram", "QRAM-based quantum", "#808080", "--", "o", 0, 20),
        ("sparse", "Classical sparse-matrix", "#606060", "-", "X", 0, 50),
        ("streaming", "Classical streaming", "#2657AF", "-", "P", 0, 50),
    ]

    plt.figure(figsize=(3.5, 3.5))
    for key, label, color, linestyle, marker, linewidth, marker_size in curve_specs:
        xm, xs, ym = get_sorted_arrays(
            results["accuracies_mean"],
            results["accuracies_std"],
            results[f"space_{key}"],
        )
        plt.fill_betweenx(ym, xm - xs, xm + xs, color=color, alpha=0.2, edgecolor="none")
        plt.plot(xm, ym, linestyle=linestyle, color=color, linewidth=1.5, alpha=0.9, label=label)
        plt.scatter(xm, ym, marker=marker, color=color, alpha=0.9, s=marker_size, linewidth=linewidth)

    if text_positions is not None:
        add_text_annotations(text_positions)

    plt.legend()

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
    keys = ["streaming", "sparse", "qram", "quantum"]

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
