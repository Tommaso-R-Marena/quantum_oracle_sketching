"""IMDb binary sentiment dataset loader.

Downloads the Stanford Large Movie Review Dataset (Maas et al., 2011)
from Hugging Face datasets on first call; subsequent calls use cache.
"""
from __future__ import annotations

import numpy as np


def load_imdb_data(
    split: str = "all",
    cache_dir: str | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return (texts, labels) for the IMDb dataset.

    Parameters
    ----------
    split:
        'train', 'test', or 'all' (default). 'all' concatenates both
        splits to match Zhao et al. (2025) which use 50,000 reviews.
    cache_dir:
        Optional path for HuggingFace dataset cache.

    Returns
    -------
    texts : list[str]
        Raw review strings.
    labels : np.ndarray, shape (N,), dtype int
        0 = negative, 1 = positive.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'datasets' package is required for load_imdb_data(). "
            "Install it with: pip install datasets"
        ) from exc

    def _col_to_list(col):
        """Safely convert a HF Column or list to a plain Python list."""
        return col.to_pylist() if hasattr(col, "to_pylist") else list(col)

    if split == "all":
        ds_train = load_dataset("imdb", split="train", cache_dir=cache_dir)
        ds_test  = load_dataset("imdb", split="test",  cache_dir=cache_dir)
        texts  = _col_to_list(ds_train["text"])  + _col_to_list(ds_test["text"])
        labels = _col_to_list(ds_train["label"]) + _col_to_list(ds_test["label"])
    else:
        ds = load_dataset("imdb", split=split, cache_dir=cache_dir)
        texts  = _col_to_list(ds["text"])
        labels = _col_to_list(ds["label"])

    return texts, np.asarray(labels, dtype=np.int64)
