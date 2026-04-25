"""Noise crossover benchmark for QOS.

# Copyright (c) 2026 Tommaso R. Marena. MIT License.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import pandas as pd
from matplotlib import pyplot as plt

from qos.core.oracle_sketch import q_oracle_sketch_boolean
from qos.primitives.noise_model import DepolarizingChannel

__all__ = ["run_noise_crossover_experiment", "main"]


def run_noise_crossover_experiment(dims: list[int], noise_rates: list[float], num_trials: int) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    sample_grid = [64, 128, 256, 512, 1024]
    for dim in dims:
        truth = jnp.mod(jnp.arange(dim), 2)
        exact_diag = jnp.exp(1j * jnp.pi * truth)
        num_qubits = int(jnp.log2(dim))
        for eta in noise_rates:
            channel = DepolarizingChannel(num_qubits=num_qubits, noise_rate=eta)
            for _ in range(num_trials):
                for m in sample_grid:
                    sketch_diag, _ = q_oracle_sketch_boolean(truth, m)
                    noisy = channel.apply_to_diagonal(sketch_diag)
                    sketch_error = float(jnp.linalg.norm(sketch_diag - exact_diag, ord=jnp.inf))
                    total_error = float(jnp.linalg.norm(noisy - exact_diag, ord=jnp.inf))
                    rows.append({
                        "dim": dim,
                        "noise_rate": eta,
                        "num_samples": m,
                        "sketch_error": sketch_error,
                        "noise_error": total_error - sketch_error,
                        "total_error": total_error,
                    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="./results/")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = run_noise_crossover_experiment([args.dim], [0.0, 0.01, 0.03, 0.05], args.num_trials)
    fig, ax = plt.subplots(figsize=(7, 4))
    for eta, g in df.groupby("noise_rate"):
        s = g.groupby("num_samples", as_index=False)["total_error"].mean()
        ax.plot(s["num_samples"], s["total_error"], marker="o", label=f"eta={eta:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("num_samples")
    ax.set_ylabel("total_error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "noise_crossover.pdf")
    df.to_csv(out / "noise_crossover.csv", index=False)


if __name__ == "__main__":
    main()
