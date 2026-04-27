"""Non-IID repetition scaling experiment.

# Copyright (c) 2026 Tommaso R. Marena. MIT License.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from qos.core.sampling import q_oracle_sketch_boolean

__all__ = ["run_non_iid_scaling", "run_non_iid_scaling_experiment", "fit_non_iid_exponent", "main"]


def _non_iid_stream(
    dim: int,
    r: int,
    m: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    epochs = int(jnp.ceil(m / r))
    idx = jax.random.randint(key, shape=(epochs,), minval=0, maxval=dim)
    idx_rep = jnp.repeat(idx, r)[:m]
    values = jnp.mod(idx_rep, 2)
    return idx_rep.astype(jnp.int32), values.astype(jnp.float32)


def run_non_iid_scaling_experiment(
    dim: int,
    repetition_numbers: list[int],
    total_samples_list: list[int],
    num_trials: int,
    key: jax.Array,
) -> pd.DataFrame:
    """Sweep over (R, M) for non-IID stream oracle sketching.

    Args:
        dim: Oracle dimension.
        repetition_numbers: Repetition counts R (number of times each index
            is repeated consecutively in the stream).
        total_samples_list: Total sample budgets M.
        num_trials: Independent trials per (R, M) cell.
        key: JAX PRNGKey.

    Returns:
        DataFrame with columns: R, M, M_per_R, error_mean, error_std.
    """
    rows = []
    exact = jnp.exp(1j * jnp.pi * jnp.mod(jnp.arange(dim), 2))
    for r in repetition_numbers:
        for m in total_samples_list:
            errs = []
            for _ in range(num_trials):
                key, sub = jax.random.split(key)
                idx, vals = _non_iid_stream(dim, r, m, sub)
                diag = q_oracle_sketch_boolean((idx, vals), dim)
                errs.append(float(jnp.linalg.norm(diag - exact, ord=jnp.inf)))
            rows.append({
                "R": r,
                "M": m,
                "M_per_R": m / r,
                "error_mean": float(np.mean(errs)),
                "error_std": float(np.std(errs)),
            })
    return pd.DataFrame(rows)


def fit_non_iid_exponent(df: pd.DataFrame) -> dict:
    """Fit log-linear exponents: error ~ C * M^alpha * R^beta."""
    x = np.column_stack([
        np.log(df["M"].values),
        np.log(df["R"].values),
        np.ones(len(df)),
    ])
    y = np.log(df["error_mean"].values + 1e-12)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {
        "M_exponent": float(beta[0]),
        "R_exponent": float(beta[1]),
        "intercept": float(beta[2]),
    }


def run_non_iid_scaling(
    dim: int = 256,
    num_trials: int = 3,
    output_dir: str | Path = "./results/",
    repetition_numbers: list[int] | None = None,
    total_samples_list: list[int] | None = None,
    seed: int = 0,
    save: bool = True,
) -> dict[str, object]:
    """High-level entry point for Appendix D Non-IID Scaling benchmark.

    Runs ``run_non_iid_scaling_experiment`` over a default grid of
    repetition numbers R and sample budgets M, optionally saves a plot
    and CSV, and returns a results dictionary for inline notebook display.

    Args:
        dim: Oracle dimension.
        num_trials: Independent trials per (R, M) cell.
        output_dir: Directory for saved artefacts (created if absent).
        repetition_numbers: Stream repetition counts R.  Defaults to
            ``[1, 2, 4, 8]``.
        total_samples_list: Total sample budgets M.  Defaults to
            ``[64, 128, 256, 512, 1024]``.
        seed: JAX PRNG seed.
        save: Whether to save CSV and plot.  Set False in unit tests.

    Returns:
        Dictionary with keys:

        - ``dataframe``:            full ``pd.DataFrame`` of raw results.
        - ``fit``:                  fitted exponents from
                                    ``fit_non_iid_exponent``.
        - ``repetition_numbers``:   list of R values swept.
        - ``total_samples_list``:   list of M values swept.
        - ``dim``:                  oracle dimension used.
        - ``num_trials``:           trials per cell.
        - ``output_dir``:           resolved output directory (str).
    """
    if repetition_numbers is None:
        repetition_numbers = [1, 2, 4, 8]
    if total_samples_list is None:
        total_samples_list = [64, 128, 256, 512, 1024]

    out = Path(output_dir)
    if save:
        out.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    df = run_non_iid_scaling_experiment(
        dim, repetition_numbers, total_samples_list, num_trials, key
    )
    fit = fit_non_iid_exponent(df)

    if save:
        df.to_csv(out / "non_iid_scaling.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for r, g in df.groupby("R"):
            axes[0].plot(g["M"], g["error_mean"], marker="o", label=f"R={r}")
            axes[1].plot(g["M_per_R"], g["error_mean"], marker="o", label=f"R={r}")
        for ax, xlabel in zip(axes, ["M (total samples)", "M/R (effective IID samples)"]):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("error (sup-norm)")
            ax.legend()
        axes[0].set_title(f"Non-IID scaling  (dim={dim})")
        axes[1].set_title("Rescaled by R")
        fig.tight_layout()
        fig.savefig(out / "non_iid_scaling.pdf")
        plt.close(fig)

    return {
        "dataframe": df,
        "fit": fit,
        "repetition_numbers": repetition_numbers,
        "total_samples_list": total_samples_list,
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
    run_non_iid_scaling(
        dim=args.dim,
        num_trials=args.num_trials,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
