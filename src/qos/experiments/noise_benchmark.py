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

__all__ = ["run_noise_benchmark", "run_noise_crossover_experiment", "main"]


def run_noise_crossover_experiment(
    dims: list[int],
    noise_rates: list[float],
    num_trials: int,
) -> pd.DataFrame:
    """Run the depolarizing noise crossover experiment.

    Args:
        dims: List of oracle dimensions (must be powers of 2).
        noise_rates: List of depolarizing noise rates eta in [0, 1).
        num_trials: Number of independent trials per (dim, eta, M) cell.

    Returns:
        DataFrame with columns: dim, noise_rate, num_samples,
        sketch_error, noise_error, total_error.
    """
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


def run_noise_benchmark(
    dim: int = 256,
    num_trials: int = 3,
    output_dir: str | Path = "./results/",
    noise_rates: list[float] | None = None,
    save: bool = True,
) -> dict[str, object]:
    """High-level entry point for the Appendix A depolarizing noise benchmark.

    Runs ``run_noise_crossover_experiment`` over a default grid of noise rates,
    optionally saves a plot and CSV to ``output_dir``, and returns a results
    dictionary suitable for inline notebook display.

    Args:
        dim: Oracle dimension (power of 2, e.g. 256 or 1024).
        num_trials: Independent trials per configuration.
        output_dir: Directory for saved artefacts (created if absent).
        noise_rates: Depolarizing rates to sweep.  Defaults to
            ``[0.0, 0.01, 0.03, 0.05, 0.10]``.
        save: Whether to save the CSV and plot.  Set False in unit tests.

    Returns:
        Dictionary with keys:

        - ``dataframe``:   full ``pd.DataFrame`` of raw results.
        - ``summary``:     per-(noise_rate, num_samples) mean/std table.
        - ``noise_rates``: list of noise rates swept.
        - ``dim``:         oracle dimension used.
        - ``num_trials``:  number of trials per cell.
        - ``output_dir``:  resolved output directory path (str).
    """
    if noise_rates is None:
        noise_rates = [0.0, 0.01, 0.03, 0.05, 0.10]

    out = Path(output_dir)
    if save:
        out.mkdir(parents=True, exist_ok=True)

    df = run_noise_crossover_experiment([dim], noise_rates, num_trials)

    summary = (
        df.groupby(["noise_rate", "num_samples"])[["sketch_error", "noise_error", "total_error"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    if save:
        df.to_csv(out / "noise_crossover.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        for eta, g in df.groupby("noise_rate"):
            s = g.groupby("num_samples", as_index=False)["total_error"].mean()
            ax.plot(s["num_samples"], s["total_error"], marker="o", label=f"eta={eta:.3f}")
        ax.set_xscale("log")
        ax.set_xlabel("num_samples")
        ax.set_ylabel("total_error (sup-norm)")
        ax.set_title(f"Depolarizing noise crossover  (dim={dim})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "noise_crossover.pdf")
        plt.close(fig)

    return {
        "dataframe": df,
        "summary": summary,
        "noise_rates": noise_rates,
        "dim": dim,
        "num_trials": num_trials,
        "output_dir": str(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="./results/")
    args = parser.parse_args()
    run_noise_benchmark(dim=args.dim, num_trials=args.num_trials, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
