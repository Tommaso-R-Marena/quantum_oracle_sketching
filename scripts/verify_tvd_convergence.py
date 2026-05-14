"""TVD convergence vs. sample count for uniform-cold oracle sketching.

Reproduces the canonical "TVD shrinks as ~1/sqrt(M)" curve for several
N values and writes the data + figure to ``results/``.

The metric plotted is the Hadamard-induced measurement-distribution TVD
(``hadamard_distribution_tvd`` below), which is the convergence target
of the uniform sketch. The raw-diagonal-L1 metric ``tvd_diag`` (see
``tests/test_tvd_core.py``) is a separate notion and would give a
different (typically larger) numerical value here; it is not used in
this script.

Outputs
-------
results/raw_data/tvd_convergence.csv    one row per (N, M, trial)
results/figures/tvd_convergence.png     log-log plot, slope -1/2 reference
results/figures/tvd_convergence.pdf

Run with:
    PYTHONPATH=src python scripts/verify_tvd_convergence.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qos.core.oracle_sketch import q_oracle_sketch_boolean

OUT_RAW = Path("results/raw_data")
OUT_FIG = Path("results/figures")
OUT_RAW.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


def hadamard(n: int) -> np.ndarray:
    H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    Hn = H.copy()
    for _ in range(n - 1):
        Hn = np.kron(Hn, H)
    return Hn


def hadamard_distribution_tvd(d_approx, d_ideal, Hn):
    """Hadamard-induced measurement-distribution TVD; ``Hn`` is precomputed."""
    _N = len(d_ideal)

    def probs(d):
        d_arr = np.real(np.array(d, dtype=np.complex128)).astype(np.float64)
        s = Hn @ (d_arr / np.sqrt(_N))
        p = np.abs(s) ** 2
        return p / p.sum()

    return 0.5 * float(np.sum(np.abs(probs(d_approx) - probs(d_ideal))))


def main() -> None:
    N_BITS_LIST = [4, 5, 6]
    M_GRID = [50, 100, 200, 500, 1000, 2000, 5000]
    NUM_TRIALS = 3
    SEED = 0

    rows = []
    for n_bits in N_BITS_LIST:
        N = 2 ** n_bits
        Hn = hadamard(n_bits)
        rng = np.random.default_rng(SEED + n_bits)
        for trial in range(NUM_TRIALS):
            # 50%-density random truth table (worst case for cold sketch)
            tt = rng.integers(0, 2, size=N).astype(np.float64)
            tt_j = jnp.array(tt, dtype=jnp.float64)
            d_ideal = (-1.0) ** tt
            for M in M_GRID:
                d, _ = q_oracle_sketch_boolean(tt_j, M)
                t = hadamard_distribution_tvd(d, d_ideal, Hn)
                rows.append({"n_bits": n_bits, "N": N, "M": M, "trial": trial, "tvd": t})

    csv_path = OUT_RAW / "tvd_convergence.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_bits", "N", "M", "trial", "tvd"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}  ({len(rows)} rows)")

    # Aggregate: mean TVD per (N, M)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for n_bits in N_BITS_LIST:
        N = 2 ** n_bits
        means = []
        for M in M_GRID:
            vals = [r["tvd"] for r in rows if r["n_bits"] == n_bits and r["M"] == M]
            means.append(np.mean(vals))
        ax.loglog(M_GRID, means, "o-", label=f"N={N}")
    M_arr = np.array(M_GRID, dtype=float)
    ref = 0.5 / np.sqrt(M_arr / M_arr[0]) * np.mean(
        [r["tvd"] for r in rows if r["M"] == M_GRID[0]]
    )
    ax.loglog(M_GRID, ref, "k--", alpha=0.6, label="$M^{-1/2}$ reference")
    ax.set_xlabel("Sample count M")
    ax.set_ylabel("TVD")
    ax.set_title("Cold uniform oracle sketch: TVD vs. M")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    for ext in ("png", "pdf"):
        out = OUT_FIG / f"tvd_convergence.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
