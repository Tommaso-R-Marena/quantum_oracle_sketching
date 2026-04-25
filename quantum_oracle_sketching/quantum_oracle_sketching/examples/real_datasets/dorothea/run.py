"""Dorothea drug-discovery classification experiment."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared import plot_experiment_results, run_classification_experiment

DOROTHEA_MIN_DFS = list(np.unique(np.logspace(np.log10(1), np.log10(500), 40).astype(int)))


def load_dorothea_data(data_dir="./data_cache/dorothea", feature_dim=100000):
    import scipy.sparse as sp

    def _load_file(subset):
        data_path = os.path.join(data_dir, f"dorothea_{subset}.data")
        labels_path = os.path.join(data_dir, f"dorothea_{subset}.labels")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        row_ind, col_ind, data_val = [], [], []
        with open(data_path, "r") as f:
            for r, line in enumerate(f):
                indices = [int(x) for x in line.strip().split()]
                for c in indices:
                    row_ind.append(r)
                    col_ind.append(c - 1)
                    data_val.append(1)
        num_samples = r + 1
        X = sp.csr_matrix((data_val, (row_ind, col_ind)), shape=(num_samples, feature_dim))
        if os.path.exists(labels_path):
            y = np.loadtxt(labels_path, dtype=int)
        else:
            y = np.zeros(num_samples)
        return X, y

    X_train, y_train = _load_file("train")
    try:
        X_valid, y_valid = _load_file("valid")
        X = sp.vstack([X_train, X_valid])
        y = np.concatenate([y_train, y_valid])
    except FileNotFoundError:
        X, y = X_train, y_train
    return X, y


def filter_fn(X, min_df):
    import scipy.sparse as sp
    if sp.issparse(X):
        feature_counts = np.asarray((X != 0).sum(axis=0)).ravel()
    else:
        feature_counts = np.sum(X != 0, axis=0)
        if hasattr(feature_counts, "ravel"):
            feature_counts = feature_counts.ravel()
    keep = np.where(feature_counts >= min_df)[0]
    X_filtered = X[:, keep]
    return X_filtered, {"features_kept": len(keep)}


def run_svm():
    X, y = load_dorothea_data()
    results = run_classification_experiment(
        X, y,
        sweep_values=DOROTHEA_MIN_DFS,
        filter_fn=filter_fn,
        dataset_name="Dorothea",
        output_prefix="dorothea",
        clf_alpha=1.0,
        cv_folds=5,
    )
    plot_experiment_results(
        results,
        title="Binary classification (Dorothea)",
        output_prefix="dorothea",
        xlabel="Accuracy",
        xticks=[0.80, 0.85, 0.90, 0.95],
        xtick_labels=["80%", "85%", "90%", "95%"],
        xlim=(0.78, 0.97),
        ylim=(1e1, 1e7),
        text_positions={
            "sparse": (0.80, 2e6),
            "streaming": (0.86, 2e4),
            "quantum": (0.88, 2e1),
        },
    )


def run_pca():
    from shared import run_pca_experiment, plot_pca_results
    X, y = load_dorothea_data()
    results = run_pca_experiment(
        X, y,
        sweep_values=DOROTHEA_MIN_DFS,
        filter_fn=filter_fn,
        dataset_name="Dorothea",
        output_prefix="dorothea",
        n_components=2,
    )
    plot_pca_results(
        results,
        title="Dimension reduction (Dorothea)",
        output_prefix="dorothea",
        ylim=(1e1, 1e7),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dorothea QOS experiment")
    parser.add_argument("--task", choices=["svm", "pca", "both"], default="both")
    args = parser.parse_args()
    if args.task in ("svm", "both"):
        run_svm()
    if args.task in ("pca", "both"):
        run_pca()
