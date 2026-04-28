"""Dorothea drug-discovery dataset loader.

Downloads the Dorothea dataset from the UCI ML Repository on first call
and caches it locally.

Reference: Guyon et al. (2005), NIPS 2003 Feature Selection Challenge.

Note: The Dorothea validation labels were never publicly released (competition
holdout). We use the 800-sample training set only, which matches the labelled
portion used by Zhao et al. (2025) Figure 4b.

Follows Zhao et al. (2025) Appendix A / Figure 4b preprocessing:
  - Binary feature matrix (800 samples x 100k sparse features)
  - Labels: +1 -> 1  (active compound), -1 -> 0 (inactive)
"""
from __future__ import annotations

import os
import urllib.request
import numpy as np
import scipy.sparse as sp

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "qos", "dorothea")

# UCI moved to a new URL structure; try both in order
_URL_CANDIDATES = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/dorothea/DOROTHEA/{filename}",
    "https://archive.ics.uci.edu/static/public/116/data/{filename}",
]


def _download(filename: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, filename)
    if os.path.exists(local):
        return local
    last_err = None
    for url_template in _URL_CANDIDATES:
        url = url_template.format(filename=filename)
        try:
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, local)
            return local
        except Exception as e:
            last_err = e
            if os.path.exists(local):
                os.remove(local)  # remove partial download
    raise RuntimeError(
        f"Could not download {filename} from any known UCI URL. "
        f"Last error: {last_err}"
    )


def _parse_sparse(path: str, n_features: int = 100_000) -> sp.csr_matrix:
    """Parse Dorothea sparse feature file (each line = space-separated 1-indexed col indices)."""
    rows, cols = [], []
    i = 0
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
    """Return (X, y) for the Dorothea dataset (train split only, 800 samples).

    Returns
    -------
    X : scipy.sparse.csr_matrix, shape (800, 100_000)
        Binary feature matrix.
    y : np.ndarray, shape (800,), dtype int
        0 = inactive compound, 1 = active compound.
    """
    train_data_path  = _download("dorothea_train.data",   cache_dir)
    train_label_path = _download("dorothea_train.labels", cache_dir)

    X = _parse_sparse(train_data_path)

    with open(train_label_path) as fh:
        vals = [int(v.strip()) for v in fh if v.strip()]
    y = np.array([(1 if v > 0 else 0) for v in vals], dtype=np.int64)

    assert X.shape[0] == len(y), (
        f"Shape mismatch: X has {X.shape[0]} rows but y has {len(y)} labels"
    )
    return X, y
