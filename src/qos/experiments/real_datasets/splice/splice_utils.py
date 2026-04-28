"""Splice-junction DNA dataset loader.

Downloads the UCI Molecular Biology (Splice-junction Gene Sequences)
dataset on first call and caches it locally.

URL: https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)

This is a novel dataset not studied in Zhao et al. (2025) — labelled
as Marena 2026 in the paper.

Preprocessing:
  - One-hot encode each nucleotide position (60 positions x 4 nucleotides
    = 240 binary features per sample after dropping ambiguous bases).
  - 3-class -> binary: EI (exon-intron boundary) vs non-EI (IE + neither).
"""
from __future__ import annotations

import os
import io
import urllib.request
import numpy as np
import scipy.sparse as sp

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "qos", "splice")
_DATA_URL  = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "molecular-biology/splice-junction-gene-sequences/splice.data"
)
_NUCLEOTIDES = list("ACGT")


def _download(cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, "splice.data")
    if not os.path.exists(local):
        print(f"  Downloading {_DATA_URL} ...")
        urllib.request.urlretrieve(_DATA_URL, local)
    return local


def load_splice(
    cache_dir: str = _CACHE_DIR,
    binary: bool = True,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Return (X, y) for the Splice dataset.

    Parameters
    ----------
    cache_dir:
        Local cache directory.
    binary:
        If True (default), collapse to binary: EI=1 vs IE+N=0.
        If False, returns 3-class labels (0=EI, 1=IE, 2=N).

    Returns
    -------
    X : scipy.sparse.csr_matrix, shape (3190, 240)
        One-hot encoded binary feature matrix.
    y : np.ndarray, shape (3190,), dtype int
    """
    local = _download(cache_dir)

    class_map = {"EI": 0, "IE": 1, "N": 2}
    rows, cols, labels = [], [], []
    row_idx = 0

    with open(local) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            cls_str = parts[0].strip()
            seq     = parts[2].strip().upper()

            label_3 = class_map.get(cls_str)
            if label_3 is None:
                continue  # skip malformed rows

            col_offset = 0
            for ch in seq:
                if ch in _NUCLEOTIDES:
                    cols.append(col_offset + _NUCLEOTIDES.index(ch))
                    rows.append(row_idx)
                col_offset += 4  # 4 nucleotide columns per position

            labels.append(label_3)
            row_idx += 1

    n_samples  = row_idx
    n_features = 4 * 60  # 240
    data = np.ones(len(rows), dtype=np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features))

    y_raw = np.array(labels, dtype=np.int64)
    if binary:
        y = (y_raw == 0).astype(np.int64)  # EI=1, everything else=0
    else:
        y = y_raw

    return X, y
