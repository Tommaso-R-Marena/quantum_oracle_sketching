"""Tests for PBMC dataset loaders (preprocessing guards and label logic)."""

from __future__ import annotations

import numpy as np
import pytest


def test_adata_is_preprocessed_detects_scaled_matrix():
    from qos.experiments.real_datasets.pbmc68k.pbmc_utils import _adata_is_preprocessed

    class _Var:
        def __init__(self):
            self.columns = []

    class _Fake:
        var = _Var()
        X = np.array([[-1.0, 0.5], [0.2, -0.3]])

    assert _adata_is_preprocessed(_Fake()) is True


def test_adata_is_preprocessed_detects_hvg_column():
    from qos.experiments.real_datasets.pbmc68k.pbmc_utils import _adata_is_preprocessed

    class _Var:
        columns = ["highly_variable"]

        @staticmethod
        def __getitem__(key):
            assert key == "highly_variable"
            return np.array([True, False, True])

    class _Fake:
        var = _Var()
        X = np.array([[1.0, 2.0, 3.0]])

    assert _adata_is_preprocessed(_Fake()) is True


def test_preprocess_counts_skips_renormalize_on_scaled_data():
    from qos.experiments.real_datasets.pbmc68k.pbmc_utils import _preprocess_counts
    import pandas as pd

    X = np.array([[-0.5, 1.2, 0.0], [0.3, -1.1, 0.8]])

    class _Fake:
        n_vars = 3

        def __init__(self):
            self.X = X.copy()
            self.var = pd.DataFrame({"highly_variable": [True, True, False]})

        def copy(self):
            out = _Fake()
            out.X = self.X.copy()
            out.var = self.var.copy()
            return out

        def __getitem__(self, idx):
            cols = self.var.index[self.var["highly_variable"]].tolist()
            out = _Fake()
            out.X = self.X[:, : len(cols)]
            out.var = pd.DataFrame({"highly_variable": np.ones(len(cols), dtype=bool)})
            return out

    out = _preprocess_counts(_Fake(), n_top_genes=2)
    import scipy.sparse as sp

    arr = out.X.toarray() if sp.issparse(out.X) else np.asarray(out.X)
    assert not np.isnan(arr).any()
    assert arr.shape[1] == 2


def test_binarise_labels_finds_cd14_in_bulk_labels():
    from qos.experiments.real_datasets.pbmc68k.pbmc_utils import _binarise_labels
    import pandas as pd

    class _Obs:
        columns = ["bulk_labels"]

        def __getitem__(self, key):
            return pd.Series(["CD14+ Monocyte", "B cell", "CD14+ Monocyte"])

    class _Fake:
        obs = _Obs()
        obsm = {}

    labels = _binarise_labels(_Fake())
    np.testing.assert_array_equal(labels, [1, 0, 1])


def test_load_pbmc68k_reduced_fallback_runs_without_nan():
  """Smoke test: reduced fallback must not re-normalize into NaNs."""
  pytest.importorskip("scanpy")
  from qos.experiments.real_datasets.pbmc68k.pbmc_utils import load_pbmc68k

  # Force fallback by breaking scvelo import inside loader.
  import builtins

  real_import = builtins.__import__

  def _fake_import(name, *args, **kwargs):
      if name == "scvelo":
          raise ImportError("forced fallback")
      return real_import(name, *args, **kwargs)

  builtins.__import__ = _fake_import
  try:
      adata, labels = load_pbmc68k(n_top_genes=100)
  finally:
      builtins.__import__ = real_import

  import scipy.sparse as sp

  X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
  assert not np.isnan(X).any()
  assert labels.shape[0] == adata.n_obs
  assert set(np.unique(labels)).issubset({0, 1})
