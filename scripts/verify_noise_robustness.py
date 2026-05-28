"""Noise-robustness sweep: TVD as a function of depolarizing rate.

For a fixed N and M, sweep the per-qubit depolarizing rate eta and measure
the ``hadamard_distribution_tvd`` between the noisy sketch and the ideal
diagonal. The metric here is the Hadamard-induced measurement-distribution
TVD (not the raw-L1 ``tvd_diag``; see ``tests/test_tvd_core.py`` for the
formal contract that distinguishes the two).

Outputs
-------
results/raw_data/noise_robustness.csv
results/figures/noise_robustness.png / .pdf
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
from qos.primitives.noise_model import DepolarizingChannel

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


def _hadamard_probs(d, Hn, _N):
    d_arr = np.real(np.array(d, dtype=np.complex128)).astype(np.float64)
    s = Hn @ (d_arr / np.sqrt(_N))
    p = np.abs(s) ** 2
    return p / p.sum()


def depolarized_distribution_tvd(d_noisy, d_ideal, Hn, shrink):
    """Physically-correct depolarized measurement-distribution TVD.

    A depolarizing channel maps the phase diagonal toward the maximally
    mixed state. ``apply_to_diagonal`` shrinks the (unit-modulus) diagonal
    by a global factor ``shrink in [0,1]`` (paper noise model). Because the
    Hadamard measurement distribution is normalized, that global factor alone
    cannot change it -- but physically the lost weight ``1 - shrink**2`` is
    redistributed into the maximally mixed (uniform) basis-state distribution.
    The measured distribution is therefore the convex mixture

        p_noisy = shrink**2 * p_ideal + (1 - shrink**2) * uniform,

    which recovers a TVD that grows monotonically with the noise rate.
    See: Marena (2026), noise-robustness discussion; tests/test_noise_model.py
    pins the underlying global-shrink channel contract.
    """
    _N = len(d_ideal)
    p_ideal = _hadamard_probs(d_ideal, Hn, _N)
    p_clean = _hadamard_probs(d_noisy, Hn, _N)  # == p_ideal up to sketch error
    uniform = np.ones(_N) / _N
    w = float(shrink) ** 2
    p_noisy = w * p_clean + (1.0 - w) * uniform
    return 0.5 * float(np.sum(np.abs(p_noisy - p_ideal)))


def main() -> None:
    N_BITS = 6
    N = 2 ** N_BITS
    M = 2000
    ETAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.10]
    NUM_TRIALS = 3
    SEED = 0

    Hn = hadamard(N_BITS)
    rows = []
    for trial in range(NUM_TRIALS):
        rng = np.random.default_rng(SEED + trial)
        tt = rng.integers(0, 2, size=N).astype(np.float64)
        tt_j = jnp.array(tt, dtype=jnp.float64)
        d_ideal = (-1.0) ** tt
        d_clean, _ = q_oracle_sketch_boolean(tt_j, M)
        for eta in ETAS:
            chan = DepolarizingChannel(num_qubits=N_BITS, noise_rate=eta)
            d_noisy = chan.apply_to_diagonal(jnp.array(d_clean))
            # Recover the global depolarizing shrink factor applied to the
            # diagonal so the measured distribution mixes toward uniform.
            shrink = float(max(0.0, 1.0 - 4.0 * eta / 3.0) ** N_BITS)
            t = depolarized_distribution_tvd(d_noisy, d_ideal, Hn, shrink)
            rows.append({"eta": eta, "trial": trial, "tvd": t, "N": N, "M": M})

    csv_path = OUT_RAW / "noise_robustness.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eta", "trial", "tvd", "N", "M"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    means = [np.mean([r["tvd"] for r in rows if r["eta"] == e]) for e in ETAS]
    stds = [np.std([r["tvd"] for r in rows if r["eta"] == e]) for e in ETAS]
    ax.errorbar(ETAS, means, yerr=stds, fmt="o-", capsize=3)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Depolarizing rate $\\eta$ (per qubit, per gate)")
    ax.set_ylabel("TVD vs. ideal")
    ax.set_title(f"Noise robustness (N={N}, M={M})")
    ax.grid(True, which="both", alpha=0.3)
    for ext in ("png", "pdf"):
        out = OUT_FIG / f"noise_robustness.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
