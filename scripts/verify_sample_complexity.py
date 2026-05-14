"""Sample complexity M(epsilon, N) for uniform cold oracle sketching.

For a 50%-density random truth table, finds the smallest M such that the
Hadamard-induced TVD between sketch and ideal is below epsilon.  Repeats
across N to verify the O(N) scaling.

Outputs
-------
results/raw_data/sample_complexity.csv
results/figures/sample_complexity.png / .pdf
"""

from __future__ import annotations

import csv
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


def tvd_diag(d_approx, d_ideal, Hn):
    _N = len(d_ideal)

    def probs(d):
        d_arr = np.real(np.array(d, dtype=np.complex128)).astype(np.float64)
        s = Hn @ (d_arr / np.sqrt(_N))
        p = np.abs(s) ** 2
        return p / p.sum()

    return 0.5 * float(np.sum(np.abs(probs(d_approx) - probs(d_ideal))))


def find_M(tt, d_ideal, Hn, epsilon, m_max):
    tt_j = jnp.array(tt, dtype=jnp.float64)
    lo, hi = 10, m_max
    while lo < hi - 1:
        mid = (lo + hi) // 2
        d, _ = q_oracle_sketch_boolean(tt_j, mid)
        if tvd_diag(d, d_ideal, Hn) < epsilon:
            hi = mid
        else:
            lo = mid
    return hi


def main() -> None:
    N_BITS_LIST = [3, 4, 5, 6]
    EPSILON = 0.10
    M_MAX = 8000
    NUM_TRIALS = 3
    SEED = 0

    rows = []
    for n_bits in N_BITS_LIST:
        N = 2 ** n_bits
        Hn = hadamard(n_bits)
        rng = np.random.default_rng(SEED + n_bits)
        for trial in range(NUM_TRIALS):
            tt = rng.integers(0, 2, size=N).astype(np.float64)
            d_ideal = (-1.0) ** tt
            M_star = find_M(tt, d_ideal, Hn, EPSILON, M_MAX)
            rows.append({"n_bits": n_bits, "N": N, "trial": trial, "M_star": M_star, "epsilon": EPSILON})
            print(f"N={N}, trial={trial}: M*={M_star}")

    csv_path = OUT_RAW / "sample_complexity.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_bits", "N", "trial", "M_star", "epsilon"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    # Mean M* vs N
    Ns = sorted({r["N"] for r in rows})
    means = [np.mean([r["M_star"] for r in rows if r["N"] == n]) for n in Ns]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.loglog(Ns, means, "o-", label="empirical M*")
    ref = np.array(means[0]) * np.array(Ns) / Ns[0]
    ax.loglog(Ns, ref, "k--", alpha=0.6, label="O(N) reference")
    ax.set_xlabel("N")
    ax.set_ylabel(f"M* (TVD < {EPSILON})")
    ax.set_title("Sample complexity vs. N (uniform cold sketch)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    for ext in ("png", "pdf"):
        out = OUT_FIG / f"sample_complexity.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
