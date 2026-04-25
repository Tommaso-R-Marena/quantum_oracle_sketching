"""Splice Junction classification experiment (k-mer features)."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared import plot_experiment_results, run_classification_experiment

SPLICE_MIN_SAMPLES = list(np.unique(np.logspace(np.log10(1), np.log10(1000), 55).astype(int)))


def load_splice_data(binary: bool = True):
    from ucimlrepo import fetch_ucirepo
    from sklearn.preprocessing import LabelEncoder
    import scipy.sparse as sp
    from collections import Counter

    dataset = fetch_ucirepo(id=69)
    X_df = dataset.data.features
    y_df = dataset.data.targets

    sequences = []
    for idx, row in X_df.iterrows():
        seq = ''.join(str(v).upper() for v in row.values)
        sequences.append(seq)

    labels = y_df['class'].values

    if binary:
        mask = (labels == 'EI') | (labels == 'IE')
        sequences = [s for s, m in zip(sequences, mask) if m]
        labels = labels[mask]

    # k-mer features
    k = 6
    NUCLEOTIDES = "ACGT"
    kmer_to_idx = {}
    idx = 0
    for seq in sequences:
        seq_clean = ''.join(c for c in seq if c in NUCLEOTIDES)
        for i in range(len(seq_clean) - k + 1):
            kmer = seq_clean[i:i+k]
            if kmer not in kmer_to_idx:
                kmer_to_idx[kmer] = idx
                idx += 1

    n_features = len(kmer_to_idx)
    n_samples = len(sequences)
    row_ind, col_ind, data_val = [], [], []
    for r, seq in enumerate(sequences):
        seq_clean = ''.join(c for c in seq if c in NUCLEOTIDES)
        kmer_counts = Counter()
        for i in range(len(seq_clean) - k + 1):
            kmer = seq_clean[i:i+k]
            if kmer in kmer_to_idx:
                kmer_counts[kmer] += 1
        total = sum(kmer_counts.values())
        if total > 0:
            for kmer, count in kmer_counts.items():
                row_ind.append(r)
                col_ind.append(kmer_to_idx[kmer])
                data_val.append(count / total)

    X = sp.csr_matrix((data_val, (row_ind, col_ind)), shape=(n_samples, n_features))
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return X, y


def filter_fn(X, min_samples):
    import scipy.sparse as sp
    if sp.issparse(X):
        feature_counts = np.asarray((X != 0).sum(axis=0)).ravel()
    else:
        feature_counts = np.sum(X != 0, axis=0)
        if hasattr(feature_counts, "ravel"):
            feature_counts = feature_counts.ravel()
    keep = np.where(feature_counts >= min_samples)[0]
    X_filtered = X[:, keep]
    return X_filtered, {"features_kept": len(keep)}


def run_svm():
    X, y = load_splice_data(binary=True)
    results = run_classification_experiment(
        X, y,
        sweep_values=SPLICE_MIN_SAMPLES,
        filter_fn=filter_fn,
        dataset_name="Splice Junction (EI vs IE)",
        output_prefix="splice",
        clf_alpha=1.0,
        cv_folds=5,
        clf_kwargs={"class_weight": "balanced"},
    )
    plot_experiment_results(
        results,
        title="Binary classification (Splice)",
        output_prefix="splice",
        xlabel="Accuracy",
        xticks=[0.78, 0.80, 0.82, 0.84, 0.86],
        xtick_labels=["78%", "80%", "82%", "84%", "86%"],
        xlim=(0.77, 0.875),
        ylim=(1e1, 2e5),
        text_positions={
            "sparse": (0.78, 8e4),
            "streaming": (0.845, 1e3),
            "quantum": (0.86, 1.5e1),
        },
    )


def run_pca():
    from shared import run_pca_experiment, plot_pca_results
    X, y = load_splice_data(binary=True)
    results = run_pca_experiment(
        X, y,
        sweep_values=SPLICE_MIN_SAMPLES,
        filter_fn=filter_fn,
        dataset_name="Splice Junction",
        output_prefix="splice",
        n_components=2,
    )
    plot_pca_results(
        results,
        title="Dimension reduction (Splice)",
        output_prefix="splice",
        ylim=(1e1, 2e5),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splice QOS experiment")
    parser.add_argument("--task", choices=["svm", "pca", "both"], default="both")
    args = parser.parse_args()
    if args.task in ("svm", "both"):
        run_svm()
    if args.task in ("pca", "both"):
        run_pca()
