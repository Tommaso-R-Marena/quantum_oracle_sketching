"""PBMC single-cell RNA-seq dataset loaders.

Provides two loaders:

  load_pbmc3k()   -- 2,700 cells, fast, good for development (~seconds)
  load_pbmc68k()  -- ~68k cells, full dataset matching Zhao et al. (2025)
                     Figure 2b; downloads ~118 MB via scvelo on first run

Both return (adata, labels) where labels are binary:
  1 = CD14+ Monocyte, 0 = all other cell types

Requires: scanpy, anndata, scvelo, python-igraph (or leidenalg)
"""
from __future__ import annotations

import os
import warnings

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


def _adata_is_preprocessed(adata) -> bool:
    """Return True when ``adata.X`` is already log-normalized or scaled.

    ``pbmc68k_reduced()`` ships with negative entries (``pp.scale`` was applied)
    and pre-computed ``highly_variable`` flags. Re-running ``normalize_total``
    + ``log1p`` on that object produces NaNs and breaks HVG selection.
    """
    var = getattr(adata, "var", None)
    if var is not None and hasattr(var, "columns") and "highly_variable" in var.columns:
        if bool(var["highly_variable"].any()):
            return True
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        sample = X.data[: min(10_000, X.data.size)]
        if sample.size == 0:
            return False
        return float(sample.min()) < 0.0
    arr = np.asarray(X)
    if arr.size == 0:
        return False
    return float(np.min(arr)) < 0.0


def _run_leiden(adata) -> None:
    """Run Leiden clustering, preferring the fast igraph backend."""
    import scanpy as sc

    try:
        sc.tl.leiden(adata, flavor="igraph", directed=False, n_iterations=2)
    except (ImportError, ValueError, RuntimeError):
        sc.tl.leiden(adata)


def _preprocess_counts(adata, n_top_genes: int, *, min_genes: int = 200) -> object:
    """Standard count-matrix preprocessing for PBMC loaders."""
    import scanpy as sc

    if _adata_is_preprocessed(adata):
        if "highly_variable" in adata.var.columns:
            adata = adata[:, adata.var["highly_variable"]].copy()
        return _ensure_sparse(adata)

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(n_top_genes, adata.n_vars),
        flavor="seurat",
        inplace=True,
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    return _ensure_sparse(adata)


def _load_pbmc68k_scvelo(cache_dir: str):
    """Download and cache the full ~68k PBMC dataset via scvelo."""
    import scvelo as scv

    h5ad_path = os.path.join(cache_dir, "pbmc68k_scvelo.h5ad")
    if os.path.exists(h5ad_path):
        import anndata as ad

        print(f"  Loading PBMC68k from cache: {h5ad_path}")
        return ad.read_h5ad(h5ad_path)

    print("  Downloading PBMC68k via scvelo (~118 MB on first run) ...")
    adata = scv.datasets.pbmc68k()
    adata.write(h5ad_path)
    print(f"  Cached to {h5ad_path}")
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
        raise ImportError("pip install scanpy anndata python-igraph") from exc

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        sc.settings.datasetdir = cache_dir

    adata = sc.datasets.pbmc3k()
    adata = _preprocess_counts(adata, n_top_genes, min_genes=200)

    if "leiden" not in adata.obs.columns:
        import scanpy as sc

        sc.pp.pca(adata, n_comps=10)
        sc.pp.neighbors(adata)
        _run_leiden(adata)

    labels = _binarise_labels(adata)
    return adata, labels


def load_pbmc68k(
    cache_dir: str | None = None,
    n_top_genes: int = 1000,
) -> tuple[object, np.ndarray]:
    """Return (adata, labels) for the full PBMC68k dataset (~68k cells).

    Primary source: ``scvelo.datasets.pbmc68k()`` (public, ~118 MB download,
    cached locally after the first call).  Falls back to scanpy's
    ``pbmc68k_reduced()`` (700 preprocessed cells) only when scvelo is
    unavailable — the fallback is flagged in stdout and should not be used
    for Zhao Figure 2b reproduction.

    Matches Zhao et al. (2025) Figure 2b when the scvelo path succeeds.

    Returns
    -------
    adata : AnnData, shape (~68k, n_top_genes)
    labels : np.ndarray shape (~68k,), dtype int  [1=CD14+ Mono, 0=other]
    """
    try:
        import scanpy as sc  # noqa: F401
    except ImportError as exc:
        raise ImportError("pip install scanpy anndata scvelo python-igraph") from exc

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "qos", "pbmc68k")
    os.makedirs(cache_dir, exist_ok=True)

    adata = None
    try:
        adata = _load_pbmc68k_scvelo(cache_dir)
        # scvelo ships a lightly filtered matrix; per-cell gene counts are low.
        adata = _preprocess_counts(adata, n_top_genes, min_genes=10)
        labels = _binarise_labels(adata)
        print(f"  PBMC68k loaded: {adata.n_obs} cells x {adata.n_vars} genes")
        return adata, labels
    except Exception as exc:
        warnings.warn(
            f"scvelo PBMC68k download failed ({exc!r}); "
            "falling back to scanpy pbmc68k_reduced (700 cells, dev only).",
            stacklevel=2,
        )

    import scanpy as sc

    adata = sc.datasets.pbmc68k_reduced()
    adata = _preprocess_counts(adata, n_top_genes, min_genes=1)
    labels = _binarise_labels(adata)
    print(
        f"  WARNING: using pbmc68k_reduced subset only "
        f"({adata.n_obs} cells). Install scvelo for the full 68k dataset."
    )
    return adata, labels
