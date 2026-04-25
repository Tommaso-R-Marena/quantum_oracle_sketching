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

__all__ = ["run_non_iid_scaling_experiment", "fit_non_iid_exponent", "main"]


def _non_iid_stream(dim: int, r: int, m: int, key: jax.Array) -> tuple[jax.Array, jax.Array]:
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
            rows.append({"R": r, "M": m, "M_per_R": m / r, "error_mean": float(np.mean(errs)), "error_std": float(np.std(errs))})
    return pd.DataFrame(rows)


def fit_non_iid_exponent(df: pd.DataFrame) -> dict:
    x = np.column_stack([np.log(df["M"].values), np.log(df["R"].values), np.ones(len(df))])
    y = np.log(df["error_mean"].values + 1e-12)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {"M_exponent": float(beta[0]), "R_exponent": float(beta[1]), "intercept": float(beta[2])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="./results/")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(0)
    df = run_non_iid_scaling_experiment(args.dim, [1, 2, 4, 8], [64, 128, 256, 512, 1024], args.num_trials, key)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for r, g in df.groupby("R"):
        axes[0].plot(g["M"], g["error_mean"], marker="o", label=f"R={r}")
        axes[1].plot(g["M_per_R"], g["error_mean"], marker="o", label=f"R={r}")
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "non_iid_scaling.pdf")
    df.to_csv(out / "non_iid_scaling.csv", index=False)


if __name__ == "__main__":
    main()
