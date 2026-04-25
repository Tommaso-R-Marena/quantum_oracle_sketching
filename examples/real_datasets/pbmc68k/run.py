"""PBMC68k single-cell RNA classification experiment."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared import plot_experiment_results, run_classification_experiment

PBMC_MIN_SAMPLES = list(np.unique(np.logspace(np.log10(1), np.log10(20000), 30).astype(int)))


def load_pbmc68k_data(binary: bool = True, normalize: bool = True):
    import scvelo as scv
    from sklearn.preprocessing import LabelEncoder

    adata = scv.datasets.pbmc68k(file_path="./data_cache/pbmc68k.h5ad")
    X = adata.X
    labels = adata.obs["celltype"].values

    if binary:
        unique, counts = np.unique(labels, return_counts=True)
        top2_idx = np.argsort(counts)[-2:]
        top2_classes = unique[top2_idx]
        mask = np.isin(labels, top2_classes)
        X = X[mask]
        labels = labels[mask]

    le = LabelEncoder()
    y = le.fit_transform(labels)

    if normalize:
        import scipy.sparse as sp
        if sp.issparse(X):
            X = X.toarray()
        X = np.log1p(X)
        X = sp.csr_matrix(X)

    return X, y


def filter_fn(X, min_samples):
    import scipy.sparse as sp
    if sp.issparse(X):
        gene_counts = np.asarray((X != 0).sum(axis=0)).ravel()
    else:
        gene_counts = np.sum(X != 0, axis=0)
        if hasattr(gene_counts, "ravel"):
            gene_counts = gene_counts.ravel()
    keep = np.where(gene_counts >= min_samples)[0]
    X_filtered = X[:, keep]
    return X_filtered, {"genes_kept": len(keep)}


def run_svm():
    X, y = load_pbmc68k_data()
    results = run_classification_experiment(
        X, y,
        sweep_values=PBMC_MIN_SAMPLES,
        filter_fn=filter_fn,
        dataset_name="PBMC68k",
        output_prefix="pbmc68k",
        clf_alpha=1.0,
        cv_folds=5,
    )
    plot_experiment_results(
        results,
        title="Binary classification (PBMC68k)",
        output_prefix="pbmc68k",
        xlabel="Accuracy",
        xticks=[0.70, 0.80, 0.90, 0.95],
        xtick_labels=["70%", "80%", "90%", "95%"],
        xlim=(0.68, 0.97),
        ylim=(1e1, 1e7),
        text_positions={
            "sparse": (0.70, 2e6),
            "streaming": (0.78, 2e4),
            "quantum": (0.80, 2e1),
        },
    )


def run_pca():
    from shared import run_pca_experiment, plot_pca_results
    X, y = load_pbmc68k_data()
    results = run_pca_experiment(
        X, y,
        sweep_values=PBMC_MIN_SAMPLES,
        filter_fn=filter_fn,
        dataset_name="PBMC68k",
        output_prefix="pbmc68k",
        n_components=2,
    )
    plot_pca_results(
        results,
        title="Dimension reduction (PBMC68k)",
        output_prefix="pbmc68k",
        ylim=(1e1, 1e7),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PBMC68k QOS experiment")
    parser.add_argument("--task", choices=["svm", "pca", "both"], default="both")
    args = parser.parse_args()
    if args.task in ("svm", "both"):
        run_svm()
    if args.task in ("pca", "both"):
        run_pca()
