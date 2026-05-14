"""Standalone reproduction of the warmstart-ablation binary search with
per-iteration TVD diagnostics.

Runs ``find_M_cold`` and ``find_M_warm`` on a small synthetic dataset (N=64,
sparse support) and writes:

  results/raw_data/warmstart_ablation_iterations.csv
      one row per binary-search iteration with columns:
      method, trial, lo, mid, hi, tvd, accepted

  results/raw_data/warmstart_ablation_summary.csv
      one row per dataset/trial with M_cold, M_warm, speedup.

The binary search gates on ``hadamard_distribution_tvd`` (the
Hadamard-induced measurement-distribution TVD) because that matches the
downstream basis-state measurement semantics of the sketched oracle. The
raw-diagonal-L1 metric ``tvd_diag`` is a different notion and is not used
inside this binary search; see ``tests/test_tvd_core.py`` for the formal
contract that pins both metrics separately.

Deterministic: seeds the JAX, NumPy, and dataset trials.

Run from repo root with:
    PYTHONPATH=src python scripts/verify_warmstart_ablation.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from qos.core.oracle_sketch import q_oracle_sketch_boolean
from qos.theory.variational_warmstart import VariationalWarmstart

# ---------------------------------------------------------------------------
# Config (smaller than the publication notebook so the script runs in <2 min)
# ---------------------------------------------------------------------------
N_BITS = 6
N = 2 ** N_BITS
EPSILON = 0.10
M_MAX = 2000
NUM_FOURIER = 16
WARMSTART_STEPS = 200
WARMSTART_LR = 0.03
NUM_TRIALS = 3
SEED = 0

OUT_DIR = Path("results/raw_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def hadamard_matrix(n: int) -> np.ndarray:
    H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    Hn = H.copy()
    for _ in range(n - 1):
        Hn = np.kron(Hn, H)
    return Hn


def hadamard_distribution_tvd(diag_approx, diag_ideal) -> float:
    """Hadamard-induced measurement-distribution TVD (matches notebook).

    ``s_i = H_n d_i / sqrt(N)``, ``p_i = |s_i|^2``,
    ``TVD = 0.5 * ||p_approx - p_ideal||_1``.

    Globally invariant under ``d -> -d`` -- this is the right metric for
    convergence gating because two oracle diagonals that differ only by a
    global sign produce identical basis-state measurement statistics.
    For the raw-L1 metric on ±1 diagonals (``tvd_diag``) see
    ``tests/test_tvd_core.py``.
    """
    _N = len(diag_ideal)
    n = int(np.log2(_N))
    Hn = hadamard_matrix(n)

    def probs(d):
        d_arr = np.real(np.array(d, dtype=np.complex128)).astype(np.float64)
        s = Hn @ (d_arr / np.sqrt(_N))
        p = np.abs(s) ** 2
        return p / p.sum()

    return 0.5 * float(np.sum(np.abs(probs(diag_approx) - probs(diag_ideal))))


def find_m_cold(
    tt: jax.Array,
    *,
    epsilon: float = EPSILON,
    m_max: int = M_MAX,
    trial: int = 0,
    iter_log: list | None = None,
) -> int:
    d_ideal = (-1.0) ** tt
    lo, hi = 10, m_max
    while lo < hi - 1:
        mid = (lo + hi) // 2
        d, _ = q_oracle_sketch_boolean(tt, mid)
        t = hadamard_distribution_tvd(d, d_ideal)
        accepted = t < epsilon
        if iter_log is not None:
            iter_log.append(
                {
                    "method": "cold",
                    "trial": trial,
                    "lo": lo,
                    "mid": mid,
                    "hi": hi,
                    "tvd": t,
                    "accepted": int(accepted),
                }
            )
        if accepted:
            hi = mid
        else:
            lo = mid
    return hi


def find_m_warm(
    tt: jax.Array,
    *,
    epsilon: float = EPSILON,
    m_max: int = M_MAX,
    trial: int = 0,
    trial_seed: int = SEED,
    iter_log: list | None = None,
) -> int:
    d_ideal = (-1.0) ** tt
    _N = int(tt.shape[0])
    rng = np.random.default_rng(trial_seed)
    lo, hi = 10, m_max
    while lo < hi - 1:
        mid = (lo + hi) // 2
        n_queries = min(mid, _N)
        idx = rng.choice(_N, size=n_queries, replace=False)
        tt_sub = jnp.zeros(_N, dtype=jnp.float64).at[idx].set(tt[idx])
        vw = VariationalWarmstart(
            tt_sub,
            num_fourier_modes=NUM_FOURIER,
            learning_rate=WARMSTART_LR,
            num_steps=WARMSTART_STEPS,
            key=jax.random.PRNGKey(trial_seed),
        )
        vw.fit(unit_num_samples=mid * 4)
        d_warm = jnp.sign(jnp.real(vw.predict()))
        t = hadamard_distribution_tvd(d_warm, d_ideal)
        accepted = t < epsilon
        if iter_log is not None:
            iter_log.append(
                {
                    "method": "warm",
                    "trial": trial,
                    "lo": lo,
                    "mid": mid,
                    "hi": hi,
                    "tvd": t,
                    "accepted": int(accepted),
                }
            )
        if accepted:
            hi = mid
        else:
            lo = mid
    return hi


def main() -> None:
    rng = np.random.default_rng(SEED)
    # Sparse truth table (K=4 / N=64) -- the regime where warmstart helps
    base_tt = np.zeros(N, dtype=np.float64)
    base_tt[:4] = 1.0

    iter_log: list[dict] = []
    summary_rows: list[dict] = []

    t0 = time.time()
    for trial in range(NUM_TRIALS):
        # 5% bit-flip noise per trial
        noise_mask = rng.random(N) < 0.05
        tt = base_tt.copy()
        tt[noise_mask] = 1.0 - tt[noise_mask]
        tt_j = jnp.array(tt, dtype=jnp.float64)
        mc = find_m_cold(tt_j, trial=trial, iter_log=iter_log)
        mw = find_m_warm(tt_j, trial=trial, trial_seed=SEED + trial, iter_log=iter_log)
        speedup = mc / max(mw, 1)
        print(
            f"trial {trial}: M_cold={mc}  M_warm={mw}  speedup={speedup:.2f}x"
        )
        summary_rows.append(
            {
                "trial": trial,
                "m_cold": mc,
                "m_warm": mw,
                "speedup": speedup,
                "epsilon": EPSILON,
                "n_bits": N_BITS,
            }
        )

    elapsed = time.time() - t0

    # Write per-iteration CSV
    iter_csv = OUT_DIR / "warmstart_ablation_iterations.csv"
    with open(iter_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "trial", "lo", "mid", "hi", "tvd", "accepted"])
        w.writeheader()
        w.writerows(iter_log)
    print(f"wrote {iter_csv}  ({len(iter_log)} rows)")

    # Write summary CSV
    sum_csv = OUT_DIR / "warmstart_ablation_summary.csv"
    with open(sum_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "m_cold", "m_warm", "speedup", "epsilon", "n_bits"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"wrote {sum_csv}")
    print(f"total time: {elapsed:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
