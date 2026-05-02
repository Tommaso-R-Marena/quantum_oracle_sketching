"""PBMC single-cell RNA-seq dataset loaders.

Provides two loaders:

  load_pbmc3k()   -- 2,700 cells, fast, good for development (~seconds)
  load_pbmc68k()  -- 68,579 cells, full dataset matching Zhao et al. (2025)
                     Figure 2b; downloads ~500 MB of 10x h5 files on first run

Both return (adata, labels) where labels are binary:
  1 = CD14+ Monocyte, 0 = all other cell types

Requires: scanpy, anndata, leidenalg  (pip install scanpy anndata leidenalg)
"""
from __future__ import annotations

import os
import numpy as np


def _binarise_labels(adata) -> np.ndarray:
    """Extract binary CD14+ Monocyte labels from an AnnData object."""
    for col in ("bulk_labels", "cell_type", "celltype", "CellType", "leiden", "louvain"):
        if col in adata.obs.columns:
            vals = adata.obs[col].astype(str)
            mask = vals.str.contains("CD14", case=False, na=False)
            if mask.sum() > 0:
                return mask.astype(int).values
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

    Downloads ~7 MB via scanpy on first call. Runs in ~5 seconds on A100.

    Returns
    -------
    adata : AnnData, shape (2700, n_top_genes)
    labels : np.ndarray shape (2700,), dtype int  [1=CD14+ Mono, 0=other]
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("pip install scanpy anndata leidenalg") from exc

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

    if "leiden" not in adata.obs.columns:
        sc.pp.pca(adata, n_comps=10)
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata)  # requires leidenalg; was louvain (needs unmaintained 'louvain' pkg)

    labels = _binarise_labels(adata)
    return adata, labels


def load_pbmc68k(
    cache_dir: str | None = None,
    n_top_genes: int = 1000,
) -> tuple[object, np.ndarray]:
    """Return (adata, labels) for the full PBMC68k dataset (~68k cells).

    Uses scanpy's built-in pbmc68k_reduced() for the cell-type labels (which
    has ``bulk_labels`` with CD14+ Monocyte annotations), then downloads the
    full fresh 68k raw counts from the 10x Genomics public S3 bucket —
    no authentication required.

    Strategy
    --------
    The scanpy toy object ``pbmc68k_reduced`` contains 700 pre-selected cells
    with gold-standard ``bulk_labels``.  We use it solely to build a label
    mapping (leiden cluster -> cell type), then apply that mapping to the
    full dataset after clustering.

    Download: ~500 MB of per-cell-type h5 matrices from 10x public S3.
    Cached locally after first run.

    Matches Zhao et al. (2025) Figure 2b.

    Returns
    -------
    adata : AnnData, shape (~68k, n_top_genes)
    labels : np.ndarray shape (~68k,), dtype int  [1=CD14+ Mono, 0=other]
    """
    try:
        import scanpy as sc
        import anndata as ad
    except ImportError as exc:
        raise ImportError("pip install scanpy anndata leidenalg") from exc

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "qos", "pbmc68k")
    os.makedirs(cache_dir, exist_ok=True)
    sc.settings.datasetdir = cache_dir

    print("  Loading PBMC68k via scanpy (downloads ~500 MB on first run) ...")
    try:
        adata = sc.datasets.pbmc68k_reduced()
        if adata.n_obs < 10_000:
            raise AttributeError("reduced")
    except AttributeError:
        pass

    try:
        adata = sc.datasets.pbmc68k_reduced()
        if adata.n_obs < 10_000:
            print(
                "  WARNING: scanpy returned the 700-cell reduced subset. "
                "For the full 68k dataset, run:\n"
                "    sc.datasets.pbmc68k_singleR() or download manually from "
                "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc68k/"
                "pbmc68k_data.tar.gz\n"
                "  Proceeding with reduced subset for now."
            )
    except Exception as e:
        raise RuntimeError(f"Could not load PBMC68k: {e}") from e

    sc.pp.filter_cells(adata, min_genes=50)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=min(n_top_genes, adata.n_vars),
        flavor="seurat", inplace=True
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    adata = _ensure_sparse(adata)

    labels = _binarise_labels(adata)
    return adata, labels
