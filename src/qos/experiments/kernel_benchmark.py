"""Kernel-vs-linear interferometric benchmark.

# Copyright (c) 2026 Tommaso R. Marena. MIT License.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

from qos.core.state_sketch import (
    fit_kernel_svm_from_states,
    q_interferometric_kernel_shadow,
    q_state_sketch_flat,
)

__all__ = ["run_kernel_vs_linear_benchmark", "main"]


def run_kernel_vs_linear_benchmark(dataset_name: str, n_samples_list: list[int], dim: int) -> pd.DataFrame:
    del dataset_name
    x, y = make_classification(n_samples=max(n_samples_list) * 2, n_features=dim, n_informative=min(8, dim), random_state=0)
    y = 2 * y - 1
    x = jnp.array(jnp.where(x >= 0, 1.0, -1.0))
    y = jnp.array(y)
    rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for n_train in n_samples_list:
        acc_k, acc_l = [], []
        xsub, ysub = x[:n_train], y[:n_train]
        for train_idx, test_idx in kf.split(xsub):
            xtr, xte = xsub[train_idx], xsub[test_idx]
            ytr, yte = ysub[train_idx], ysub[test_idx]
            tr_states = jax.vmap(lambda v: q_state_sketch_flat(v, 256)[0])(xtr)
            te_states = jax.vmap(lambda v: q_state_sketch_flat(v, 256)[0])(xte)
            alpha = fit_kernel_svm_from_states(tr_states, ytr)
            pred_k = jnp.array([q_interferometric_kernel_shadow(tr_states, ytr, alpha, s) for s in te_states])
            acc_k.append(float(jnp.mean((pred_k == yte).astype(jnp.float32))))

            w = jnp.linalg.lstsq(xtr, ytr, rcond=None)[0]
            pred_l = jnp.where(xte @ w >= 0, 1, -1)
            acc_l.append(float(jnp.mean((pred_l == yte).astype(jnp.float32))))
        rows.append({"n_train": n_train, "kernel_acc": float(jnp.mean(jnp.array(acc_k))), "linear_acc": float(jnp.mean(jnp.array(acc_l))), "kernel_memory": n_train, "linear_memory": dim})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="./results/")
    args = parser.parse_args()
    del args.num_trials
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = run_kernel_vs_linear_benchmark("synthetic", [32, 64, 96, 128], args.dim)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["n_train"], df["kernel_acc"], marker="o", label="kernel")
    ax.plot(df["n_train"], df["linear_acc"], marker="s", label="linear")
    ax.legend()
    ax.set_xlabel("n_train")
    ax.set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(out / "kernel_vs_linear_accuracy.pdf")
    df.to_csv(out / "kernel_vs_linear_accuracy.csv", index=False)


if __name__ == "__main__":
    main()
