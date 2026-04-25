"""IMDb dataset utilities (auto-download and load)."""

from __future__ import annotations

import os
import tarfile
import urllib.request

from tqdm import tqdm


IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_ARCHIVE_NAME = "aclImdb_v1.tar.gz"


def _safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    abs_path = os.path.abspath(path)
    for member in tar.getmembers():
        member_path = os.path.abspath(os.path.join(path, member.name))
        if not member_path.startswith(abs_path + os.sep) and member_path != abs_path:
            raise RuntimeError("Unsafe path detected in tar archive.")
    tar.extractall(path)


def download_imdb_data(data_root: str = "data_cache") -> str:
    """Download and extract the IMDb dataset."""
    os.makedirs(data_root, exist_ok=True)
    archive_path = os.path.join(data_root, IMDB_ARCHIVE_NAME)
    imdb_path = os.path.join(data_root, "aclImdb")

    if not os.path.exists(archive_path):
        tqdm.write(f"Downloading IMDb dataset to {archive_path}...")
        urllib.request.urlretrieve(IMDB_URL, archive_path)
    else:
        tqdm.write(f"IMDb archive already exists at {archive_path}, skipping download.")

    if not os.path.exists(imdb_path):
        tqdm.write(f"Extracting IMDb dataset to {data_root}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract_tar(tar, data_root)

    return imdb_path


def load_imdb_data(download_if_missing: bool = True, data_root: str = "data_cache"):
    """Load the full IMDb dataset.

    Returns:
        ``(data, target)`` where data is a list of strings and target is a numpy array of labels.
    """
    from sklearn.datasets import load_files

    potential_paths = ["aclImdb", os.path.join(data_root, "aclImdb")]
    imdb_path = None
    for p in potential_paths:
        if os.path.exists(p):
            imdb_path = p
            break

    if imdb_path is None and download_if_missing:
        download_imdb_data(data_root=data_root)
        for p in potential_paths:
            if os.path.exists(p):
                imdb_path = p
                break

    if imdb_path is None:
        raise FileNotFoundError(
            f"IMDb dataset not found in {potential_paths}. Please download it from {IMDB_URL}."
        )

    tqdm.write("Loading IMDb Train Data...")
    train_data = load_files(
        os.path.join(imdb_path, "train"), categories=["pos", "neg"], encoding="utf-8"
    )
    tqdm.write("Loading IMDb Test Data...")
    test_data = load_files(
        os.path.join(imdb_path, "test"), categories=["pos", "neg"], encoding="utf-8"
    )

    all_data = train_data.data + test_data.data
    all_target = np.concatenate([train_data.target, test_data.target])
    return all_data, all_target
