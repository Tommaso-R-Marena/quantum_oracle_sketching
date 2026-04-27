"""k-Forrelation benchmark experiments.

# Copyright (c) 2026 Tommaso R. Marena. MIT License.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
from matplotlib import pyplot as plt

from qos.core.oracle_sketch import q_oracle_sketch_boolean
from qos.data.generation import k_forrelation_data
from qos.utils.numerical import unnormalized_hadamard_transform

__all__ = ["run_forrelation_benchmark", "run_k_forrelation_sweep", "main"]


def run_k_forrelation_sweep(
    n_values: list[int],
    k_values: list[int],
    num_samples_list: list[int],
    num_trials: int,
) -> pd.DataFrame:
    """Sweep over (n, k, M) configurations for the k-Forrelation problem.

    Args:
        n_values: Log2 of oracle dimension, e.g. [8, 10].
        k_values: Number of Forrelation levels, e.g. [2, 3, 4].
        num_samples_list: Sample budgets M to evaluate.
        num_trials: Independent trials per configuration.

    Returns:
        DataFrame with columns: n, k, M, qos_error, classical_complexity_bound.
    """
    rows: list[dict[str, float]] = []
    key = jax.random.PRNGKey(0)
    for n in n_values:
        dim = 2**n
        for k in k_values:
            generator = k_forrelation_data(n=n, k=k, key=key)
            for m in num_samples_list:
                errs = []
                for _ in range(num_trials):
                    key, *subs = jax.random.split(key, k + 2)
                    funcs = [
                        jax.random.choice(subs[i], jnp.array([-1.0, 1.0]), shape=(dim,))
                        for i in range(k)
                    ]
                    exact = generator.compute_exact_forrelation(funcs)
                    hadamard = unnormalized_hadamard_transform(n)
                    state = funcs[-1].astype(jnp.float32)
                    for fi in reversed(funcs[:-1]):
                        state = (hadamard @ state) / generator.dim
                        state = fi.astype(jnp.float32) * state
                    truth = ((1.0 - jnp.sign(state)) / 2.0).astype(jnp.int32)
                    diag, _ = q_oracle_sketch_boolean(truth, m)
                    est = generator.quantum_query_algorithm(diag)
                    errs.append(abs(est - exact))
                rows.append({
                    "n": n,
                    "k": k,
                    "M": m,
                    "qos_error": float(jnp.mean(jnp.array(errs))),
                    "classical_complexity_bound": generator.classical_streaming_complexity(0.1),
                })
    return pd.DataFrame(rows)


def run_forrelation_benchmark(
    dim: int = 256,
    num_trials: int = 3,
    output_dir: str | Path = "./results/",
    k_values: list[int] | None = None,
    num_samples_list: list[int] | None = None,
    save: bool = True,
) -> dict[str, object]:
    """High-level entry point for Appendix B k-Forrelation lower-bound benchmark.

    Calls ``run_k_forrelation_sweep`` over a default grid of k values and
    sample budgets, optionally saves a plot and CSV, and returns a results
    dictionary for inline notebook display.

    Args:
        dim: Oracle dimension (power of 2, e.g. 256 or 1024).
        num_trials: Independent trials per (n, k, M) cell.
        output_dir: Directory for saved artefacts (created if absent).
        k_values: Forrelation levels to sweep.  Defaults to ``[2, 3, 4]``.
        num_samples_list: Sample budgets.  Defaults to ``[64, 128, 256, 512, 1024]``.
        save: Whether to save CSV and plot.  Set False in unit tests.

    Returns:
        Dictionary with keys:

        - ``dataframe``:         full ``pd.DataFrame`` of raw results.
        - ``summary``:           per-(k, M) mean error table.
        - ``k_values``:          list of k values swept.
        - ``num_samples_list``:  list of M values swept.
        - ``dim``:               oracle dimension used.
        - ``num_trials``:        trials per cell.
        - ``output_dir``:        resolved output directory (str).
    """
    if k_values is None:
        k_values = [2, 3, 4]
    if num_samples_list is None:
        num_samples_list = [64, 128, 256, 512, 1024]

    # dim must be a power of 2; derive n = log2(dim).
    n = int(round(jnp.log2(float(dim)).item()))
    if 2**n != dim:
        raise ValueError(f"dim={dim} is not a power of 2.")

    out = Path(output_dir)
    if save:
        out.mkdir(parents=True, exist_ok=True)

    df = run_k_forrelation_sweep([n], k_values, num_samples_list, num_trials)

    summary = (
        df.groupby(["k", "M"])[["qos_error", "classical_complexity_bound"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    if save:
        df.to_csv(out / "forrelation_k_sweep.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        for k_val, g in df.groupby("k"):
            s = g.groupby("M", as_index=False)["qos_error"].mean()
            ax.plot(s["M"], s["qos_error"], marker="o", label=f"k={k_val}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("M (samples)")
        ax.set_ylabel("QOS error")
        ax.set_title(f"k-Forrelation lower-bound benchmark  (dim={dim})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "forrelation_k_sweep.pdf")
        plt.close(fig)

    return {
        "dataframe": df,
        "summary": summary,
        "k_values": k_values,
        "num_samples_list": num_samples_list,
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
    run_forrelation_benchmark(
        dim=args.dim,
        num_trials=args.num_trials,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
