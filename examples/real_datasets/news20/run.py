"""20 Newsgroups classification experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared import plot_experiment_results, run_classification_experiment

NEWS20_MIN_DFS = [
    1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 27, 33, 40, 49, 60, 73, 89, 109,
    133, 162, 198, 242, 295, 360, 439, 536, 654, 799, 976, 1191, 1454, 1775,
    2167, 2646, 3232, 3947, 4820, 5888, 7192, 8786, 10733, 13112, 16017, 19572,
]


def load_20news_data():
    """Load the full 20 newsgroups dataset (all categories)."""
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    return data.data, data.target


def filter_fn(X_raw, min_df):
    vectorizer = TfidfVectorizer(min_df=min_df, stop_words="english")
    X = vectorizer.fit_transform(X_raw)
    X.eliminate_zeros()
    return X, {"vocab_size": X.shape[1]}


def run_svm():
    X_all_raw, y_all = load_20news_data()
    results = run_classification_experiment(
        X_all_raw,
        y_all,
        sweep_values=NEWS20_MIN_DFS,
        filter_fn=filter_fn,
        dataset_name="20 Newsgroups",
        output_prefix="20news",
        clf_alpha=10.0,
        cv_folds=5,
    )
    plot_experiment_results(
        results,
        title="Multi-class classification",
        output_prefix="20news",
        xlabel="Accuracy",
        xticks=[0.40, 0.50, 0.60, 0.70],
        xtick_labels=["40%", "50%", "60%", "70%"],
        xlim=(0.38, 0.72),
        ylim=(1e1, 1e7),
        text_positions={
            "sparse": (0.40, 2e6),
            "streaming": (0.50, 2e4),
            "quantum": (0.52, 2e1),
        },
    )


def run_pca():
    from shared import run_pca_experiment, plot_pca_results
    X_all_raw, y_all = load_20news_data()
    results = run_pca_experiment(
        X_all_raw,
        y_all,
        sweep_values=NEWS20_MIN_DFS,
        filter_fn=filter_fn,
        dataset_name="20 Newsgroups",
        output_prefix="20news",
        n_components=2,
    )
    plot_pca_results(
        results,
        title="Dimension reduction (20 Newsgroups)",
        output_prefix="20news",
        ylim=(1e1, 1e7),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="20 Newsgroups QOS experiment")
    parser.add_argument("--task", choices=["svm", "pca", "both"], default="both")
    args = parser.parse_args()
    if args.task in ("svm", "both"):
        run_svm()
    if args.task in ("pca", "both"):
        run_pca()
