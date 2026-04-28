"""PBMC single-cell RNA-seq dataset loaders.

Provides two loaders:

  load_pbmc3k()   -- 2,700 cells, fast, good for development (~seconds)
  load_pbmc68k()  -- 68,579 cells, full dataset matching Zhao et al. (2025)
                     Figure 2b (~10-20 min download on first run)

Both return (adata, labels) where labels are binary:
  1 = CD14+ Monocyte, 0 = all other cell types

Requires: scanpy, anndata  (pip install scanpy anndata)
"""
from __future__ import annotations

import os
import numpy as np


def _binarise_labels(adata) -> np.ndarray:
    """Extract binary CD14+ Monocyte labels from an AnnData object."""
    for col in ("bulk_labels", "cell_type", "celltype", "CellType", "louvain", "leiden"):
        if col in adata.obs.columns:
            vals = adata.obs[col].astype(str)
            # CD14+ Monocytes appear under several naming conventions
            mask = vals.str.contains("CD14", case=False, na=False)
            if mask.sum() > 0:
                return mask.astype(int).values
    # Last-resort: median split on first PCA component
    pca = adata.obsm.get("X_pca")
    if pca is not None:
        return (pca[:, 0] > np.median(pca[:, 0])).astype(int)
    return np.zeros(adata.n_obs, dtype=int)


def _ensure_sparse(adata):
    import scipy.sparse as sp
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    return adata


def load_pbmc3k(
    cache_dir: str | None = None,
    n_top_genes: int = 500,
) -> tuple[object, np.ndarray]:
    """Return (adata, labels) for the PBMC3k dataset (2,700 cells).

    Fast standin for development; ~2 seconds on A100.
    Downloads ~7 MB via scanpy on first call.

    Parameters
    ----------
    cache_dir : optional scanpy data cache directory
    n_top_genes : highly-variable genes to retain (default 500)

    Returns
    -------
    adata : AnnData, adata.X shape (2700, n_top_genes)
    labels : np.ndarray shape (2700,), dtype int
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("pip install scanpy anndata") from exc

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        sc.settings.datasetdir = cache_dir

    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    adata = _ensure_sparse(adata)

    # pbmc3k comes with louvain clusters from the scanpy tutorial
    # CD14+ Monocytes = cluster '4' in the standard tutorial pipeline
    if "louvain" not in adata.obs.columns:
        sc.pp.pca(adata, n_comps=10)
        sc.pp.neighbors(adata)
        sc.tl.louvain(adata)

    labels = _binarise_labels(adata)
    return adata, labels


def load_pbmc68k(
    cache_dir: str | None = None,
    n_top_genes: int = 1000,
) -> tuple[object, np.ndarray]:
    """Return (adata, labels) for the full PBMC68k dataset (68,579 cells).

    Downloads the dataset from HuggingFace (scverse/pbmc68k_reduced is the
    scanpy toy subset; the full dataset is fetched via the 10x h5 file).
    First download is ~150 MB; cached locally afterwards.

    Matches Zhao et al. (2025) Figure 2b: ~50 qubits, ~4-6 OOM advantage.

    Parameters
    ----------
    cache_dir : local cache directory (default ~/.cache/qos/pbmc68k)
    n_top_genes : highly-variable genes to retain (default 1000)

    Returns
    -------
    adata : AnnData, adata.X shape (~68k, n_top_genes)
    labels : np.ndarray shape (~68k,), dtype int
    """
    try:
        import scanpy as sc
        import anndata as ad
    except ImportError as exc:
        raise ImportError("pip install scanpy anndata") from exc

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "qos", "pbmc68k")
    os.makedirs(cache_dir, exist_ok=True)

    h5_path = os.path.join(cache_dir, "pbmc68k_raw.h5ad")

    if not os.path.exists(h5_path):
        # Download pre-processed 68k PBMC h5ad from a stable HuggingFace mirror
        import urllib.request
        url = (
            "https://huggingface.co/datasets/scverse/pbmc68k/resolve/main/"
            "pbmc68k_raw.h5ad"
        )
        print(f"  Downloading PBMC68k (~150 MB) from {url} ...")
        urllib.request.urlretrieve(url, h5_path)
        print("  Download complete.")

    adata = ad.read_h5ad(h5_path)

    # Preprocessing pipeline matching Zhao et al. (2025)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, flavor="seurat", inplace=True
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    adata = _ensure_sparse(adata)

    labels = _binarise_labels(adata)
    return adata, labels
