"""Dorothea drug-discovery dataset loader.

Downloads the Dorothea dataset from the UCI ML Repository on first call
and caches it locally.

Reference: Guyon et al. (2005), NIPS 2003 Feature Selection Challenge.
URL: https://archive.ics.uci.edu/ml/datasets/dorothea

Follows Zhao et al. (2025) Appendix A / Figure 4b preprocessing:
  - Binary feature matrix (1900 samples x 100k sparse features)
  - Labels: +1 -> 1  (active compound), -1 -> 0 (inactive)
"""
from __future__ import annotations

import os
import io
import urllib.request
import numpy as np
import scipy.sparse as sp

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "qos", "dorothea")
_BASE_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/dorothea/DOROTHEA/"


def _download(filename: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, filename)
    if not os.path.exists(local):
        url = _BASE_URL + filename
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, local)
    return local


def _parse_sparse(path: str, n_features: int = 100_000) -> sp.csr_matrix:
    """Parse Dorothea sparse feature file (each line = space-separated col indices)."""
    rows, cols = [], []
    with open(path) as fh:
        for i, line in enumerate(fh):
            for tok in line.split():
                j = int(tok) - 1  # 1-indexed -> 0-indexed
                rows.append(i)
                cols.append(j)
    n_samples = i + 1
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix(
        (data, (rows, cols)), shape=(n_samples, n_features)
    )


def load_dorothea(
    cache_dir: str = _CACHE_DIR,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Return (X, y) for the Dorothea dataset.

    Returns
    -------
    X : scipy.sparse.csr_matrix, shape (1950, 100_000)
        Binary feature matrix (train + valid concatenated, matching Zhao et al.).
    y : np.ndarray, shape (1950,), dtype int
        0 = inactive, 1 = active.
    """
    # Train split
    train_data_path  = _download("dorothea_train.data",   cache_dir)
    train_label_path = _download("dorothea_train.labels", cache_dir)
    # Validation split (Zhao et al. concatenate train+valid for the full sweep)
    valid_data_path  = _download("dorothea_valid.data",   cache_dir)
    valid_label_path = _download("dorothea_valid.labels", cache_dir)

    X_train = _parse_sparse(train_data_path)
    X_valid = _parse_sparse(valid_data_path)
    X = sp.vstack([X_train, X_valid], format="csr")

    def _load_labels(path: str) -> np.ndarray:
        with open(path) as fh:
            vals = [int(v.strip()) for v in fh if v.strip()]
        return np.array([(1 if v > 0 else 0) for v in vals], dtype=np.int64)

    y = np.concatenate([_load_labels(train_label_path),
                        _load_labels(valid_label_path)])
    return X, y
