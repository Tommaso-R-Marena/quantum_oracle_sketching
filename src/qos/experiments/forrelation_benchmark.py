"""k-Forrelation benchmark experiments.

Copyright 2026 The Quantum Oracle Sketching Authors.
Licensed under the Apache License, Version 2.0.
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

__all__ = ["run_k_forrelation_sweep", "main"]


def run_k_forrelation_sweep(n_values: list[int], k_values: list[int], num_samples_list: list[int], num_trials: int) -> pd.DataFrame:
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
                    funcs = [jax.random.choice(subs[i], jnp.array([-1.0, 1.0]), shape=(dim,)) for i in range(k)]
                    exact = generator.compute_exact_forrelation(funcs)
                    truth = ((1.0 - funcs[0]) / 2.0).astype(jnp.int32)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="./results/")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = int(jnp.log2(args.dim))
    df = run_k_forrelation_sweep([n], [2, 3, 4], [64, 128, 256, 512], args.num_trials)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, g in df.groupby("k"):
        ax.plot(g["M"], g["qos_error"], marker="o", label=f"k={k}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("M")
    ax.set_ylabel("qos_error")
    fig.tight_layout()
    fig.savefig(out / "forrelation_k_sweep.pdf")
    df.to_csv(out / "forrelation_k_sweep.csv", index=False)


if __name__ == "__main__":
    main()
