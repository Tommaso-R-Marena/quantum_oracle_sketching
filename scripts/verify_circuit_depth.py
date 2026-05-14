"""Circuit-depth scaling: M* vs. depth for a fixed noise rate.

Uses the analytic crossover formula from
``qos.primitives.noise_model.crossover_sample_count`` to plot how the
minimum sample budget grows as the noisy circuit depth grows for several
fixed dimensions N.

Outputs
-------
results/raw_data/circuit_depth.csv
results/figures/circuit_depth.png / .pdf
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qos.primitives.noise_model import crossover_sample_count

OUT_RAW = Path("results/raw_data")
OUT_FIG = Path("results/figures")
OUT_RAW.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


def main() -> None:
    N_LIST = [16, 64, 256]
    DEPTHS = [1, 2, 5, 10, 20, 50, 100]
    NOISE = 1e-3
    EPS = 0.10

    rows = []
    for N in N_LIST:
        for d in DEPTHS:
            m_star = crossover_sample_count(N, NOISE, d, EPS)
            rows.append({"N": N, "depth": d, "M_star": int(m_star), "eta": NOISE, "epsilon": EPS})

    csv_path = OUT_RAW / "circuit_depth.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "depth", "M_star", "eta", "epsilon"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for N in N_LIST:
        ms = [r["M_star"] for r in rows if r["N"] == N]
        ax.semilogy(DEPTHS, ms, "o-", label=f"N={N}")
    ax.set_xlabel("Circuit depth (layers)")
    ax.set_ylabel("M* (crossover sample count)")
    ax.set_title(f"Sample budget vs. depth ($\\eta$={NOISE}, $\\varepsilon$={EPS})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    for ext in ("png", "pdf"):
        out = OUT_FIG / f"circuit_depth.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
