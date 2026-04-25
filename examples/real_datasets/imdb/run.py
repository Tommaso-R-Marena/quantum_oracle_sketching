"""IMDb sentiment classification experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared import plot_experiment_results, run_classification_experiment

IMDB_MIN_DFS = [
    2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 16, 19, 21, 24, 28, 32, 36, 42, 48,
    55, 62, 71, 81, 93, 106, 122, 139, 159, 181, 207, 236, 270, 308, 352, 402,
    459, 524, 599, 684, 781, 891, 1018, 1162, 1327, 1515, 1730, 1976, 2256,
    2576, 2941, 3358, 3835, 4379, 5000,
]


def load_imdb_data():
    """Auto-download and load IMDb reviews."""
    from imdb_utils import load_imdb_data as _load
    return _load()


def filter_fn(X_raw, min_df):
    """TF-IDF vectorize with min_df threshold."""
    vectorizer = TfidfVectorizer(min_df=min_df, stop_words="english")
    X = vectorizer.fit_transform(X_raw)
    X.eliminate_zeros()
    return X, {"vocab_size": X.shape[1]}


def run_svm():
    X_all_raw, y_all = load_imdb_data()
    results = run_classification_experiment(
        X_all_raw,
        y_all,
        sweep_values=IMDB_MIN_DFS,
        filter_fn=filter_fn,
        dataset_name="IMDb Full",
        output_prefix="imdb",
        clf_alpha=10.0,
        cv_folds=5,
    )
    plot_experiment_results(
        results,
        title="Binary classification",
        output_prefix="imdb",
        xlabel="Accuracy",
        xticks=[0.70, 0.75, 0.80, 0.85, 0.90],
        xtick_labels=["70%", "75%", "80%", "85%", "90%"],
        xlim=(0.69, 0.91),
        ylim=(1e1, 1e7),
        text_positions={
            "sparse": (0.70, 4e6),
            "streaming": (0.88, 9e4),
            "quantum": (0.90, 1.9e1),
        },
    )


def run_pca():
    from shared import run_pca_experiment, plot_pca_results
    X_all_raw, y_all = load_imdb_data()
    results = run_pca_experiment(
        X_all_raw,
        y_all,
        sweep_values=IMDB_MIN_DFS,
        filter_fn=filter_fn,
        dataset_name="IMDb Full",
        output_prefix="imdb",
        n_components=2,
    )
    plot_pca_results(
        results,
        title="Dimension reduction (IMDb)",
        output_prefix="imdb",
        ylim=(1e1, 1e7),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMDb QOS experiment")
    parser.add_argument("--task", choices=["svm", "pca", "both"], default="both")
    args = parser.parse_args()
    if args.task in ("svm", "both"):
        run_svm()
    if args.task in ("pca", "both"):
        run_pca()
