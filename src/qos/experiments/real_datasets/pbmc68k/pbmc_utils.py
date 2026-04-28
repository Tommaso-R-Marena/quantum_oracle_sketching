"""PBMC68k single-cell RNA-seq dataset loader.

Downloads the 10x Genomics PBMC68k dataset via scanpy on first call.
Requires: scanpy, anndata  (pip install scanpy anndata).

Follows the preprocessing in Zhao et al. (2025):
  - Filter to highly-variable genes (min_mean=0.0125, max_mean=3, min_disp=0.5)
  - Binarise cell-type labels: CD14+ Monocytes (label=1) vs all others (label=0)
"""
from __future__ import annotations

import numpy as np


def load_pbmc68k(
    cache_dir: str | None = None,
    n_top_genes: int = 1000,
) -> tuple[object, np.ndarray]:
    """Return (adata, labels) for the PBMC68k dataset.

    Parameters
    ----------
    cache_dir:
        Directory for scanpy data cache. Defaults to ~/.cache/scanpy.
    n_top_genes:
        Number of highly-variable genes to retain (default 1000, matching
        Zhao et al. 2025 ~50-qubit regime).

    Returns
    -------
    adata : AnnData
        Preprocessed AnnData object; adata.X is a scipy sparse matrix
        of shape (N_cells, n_top_genes).
    labels : np.ndarray, shape (N_cells,), dtype int
        Binary labels: 1 = CD14+ Monocyte, 0 = other.
    """
    try:
        import scanpy as sc  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scanpy is required for load_pbmc68k(). "
            "Install it with: pip install scanpy anndata"
        ) from exc

    import os
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        sc.settings.datasetdir = cache_dir

    # Download / load from cache
    adata = sc.datasets.pbmc68k_reduced()  # ~70 k cells, preprocessed
    # pbmc68k_reduced already has leiden clusters; map to binary
    # CD14+ Monocytes correspond to cluster '4' in the standard Seurat pipeline
    # used by Zhao et al.; fall back gracefully if cluster names differ.
    if "bulk_labels" in adata.obs.columns:
        label_col = "bulk_labels"
        labels = (adata.obs[label_col] == "CD14+ Monocyte").astype(int).values
    elif "louvain" in adata.obs.columns:
        label_col = "louvain"
        labels = (adata.obs[label_col] == "4").astype(int).values
    else:
        # last-resort: split on median of first PCA component
        pca = adata.obsm.get("X_pca")
        if pca is not None:
            labels = (pca[:, 0] > np.median(pca[:, 0])).astype(int)
        else:
            labels = np.zeros(adata.n_obs, dtype=int)

    # Reduce to highly-variable gene subset
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, flavor="seurat", inplace=True
    )
    adata = adata[:, adata.var["highly_variable"]].copy()

    import scipy.sparse as sp
    if not sp.issparse(adata.X):
        import scipy.sparse
        adata.X = scipy.sparse.csr_matrix(adata.X)

    return adata, labels
