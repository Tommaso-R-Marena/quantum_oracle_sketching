"""Synthetic benchmark suite for quantum oracle and state sketching.

Provides vectorized benchmark runners and fitting utilities to extract
sample-complexity scaling laws.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from matplotlib.lines import Line2D
from tqdm import tqdm

from qos.config import DEFAULT_CONFIG, real_dtype
from qos.core.oracle_sketch import (
    q_oracle_sketch_boolean,
    q_oracle_sketch_matrix_element,
    q_oracle_sketch_matrix_row_index,
)
from qos.core.state_sketch import q_state_sketch_flat, q_state_sketch
from qos.qsvt.angles import get_qsvt_angles
from qos.utils.numerical import (
    random_flat_vector,
    random_sparse_matrix_constant_magnitude,
    random_sparse_matrix_given_row_sparsity,
    random_unit_vector,
)

plt.rcParams.update(
    {
        "font.family": "sans",
        "font.serif": ["Google Sans"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.figsize": (3.5, 2.5),
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "legend.frameon": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }
)

_cmap = plt.get_cmap("Oranges")
_color_indices = jnp.linspace(0.35, 1.0, 4)
_colors = [_cmap(float(i)) for i in _color_indices]

# Pre-compute arcsin angle set for general vector sketching.
_arcsin_degree = DEFAULT_CONFIG.arcsin_degree
_arcsin_angle_set = get_qsvt_angles(
    func=lambda x: jnp.arcsin(x) / jnp.arcsin(1.0),
    degree=_arcsin_degree,
    rescale=1.0,
    cheb_domain=(-jnp.sin(1.0), jnp.sin(1.0)),
    ensure_bounded=False,
    parity=1,
)
_target_norm = 1.0 / (jnp.arcsin(1.0) * 5.0)

# QSVT overhead multiplier: total_samples = unit_num_samples * _qsvt_overhead.
# angle_set has shape (degree+1,), so overhead = degree (number of SU(2) gates).
_qsvt_overhead: int = int(_arcsin_angle_set.shape[0] - 1)

# Vectorized wrappers.
_q_state_sketch_vec = jax.vmap(
    partial(q_state_sketch, angle_set=_arcsin_angle_set, degree=_arcsin_degree),
    in_axes=(0, 0, None),
    out_axes=(0, 0),
)
_q_state_sketch_flat_vec = jax.vmap(
    q_state_sketch_flat, in_axes=(0, None), out_axes=(0, 0)
)
_q_oracle_boolean_vec = jax.vmap(
    q_oracle_sketch_boolean, in_axes=(0, None), out_axes=(0, 0)
)
_q_oracle_matrix_element_vec = jax.vmap(
    q_oracle_sketch_matrix_element, in_axes=(0, None), out_axes=(0, 0)
)


def _unit_normalize(v: jax.Array) -> jax.Array:
    """Normalize rows of v to unit L2 norm, safe against zero vectors."""
    norms = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.where(norms > 0, norms, jnp.ones_like(norms))


def benchmark_random_vector(
    key: random.PRNGKeyArray,
    dim: int,
    unit_num_samples: int,
    repetition: int = 10,
) -> dict[str, float]:
    """Benchmark general-vector state sketching.

    ``num_samples`` in the returned dict is set to ``unit_num_samples``
    (the *input* sample budget), **not** the QSVT-inflated total returned by
    ``q_state_sketch``.  The QSVT overhead is a fixed multiplicative constant
    (``_qsvt_overhead``) that does not vary with the input; folding it into the
    x-axis makes the regression matrix singular and destroys the fit.  The raw
    total is stored separately under ``qsvt_total_samples`` for reference.

    Error is the L2 distance between unit-normalized sketch output and
    unit-normalized target, measuring pure angular/directional fidelity.
    """
    key, subkey = random.split(key)
    vector = random_unit_vector(subkey, dim, batch_size=repetition)

    keys = random.split(key, repetition)
    state_sketch, qsvt_total = _q_state_sketch_vec(vector, keys, unit_num_samples)

    # Unit-normalize both sides to measure angular error independently of
    # the q_state_sketch internal rescaling.
    sketch_unit = _unit_normalize(jnp.real(state_sketch))
    target_unit = _unit_normalize(vector)
    error = jnp.linalg.norm(sketch_unit - target_unit, axis=-1)

    norm_error = jnp.abs(jnp.linalg.norm(state_sketch, axis=-1) - _target_norm)

    return {
        "error_mean": float(jnp.mean(error)),
        "error_std": float(jnp.std(error) / jnp.sqrt(repetition)),
        "norm_error_mean": float(jnp.mean(norm_error)),
        "norm_error_std": float(jnp.std(norm_error) / jnp.sqrt(repetition)),
        # Use unit_num_samples as the canonical x-axis for fitting.
        # QSVT overhead (_qsvt_overhead) is constant and must NOT be folded in.
        "num_samples": float(unit_num_samples),
        "qsvt_total_samples": float(jnp.mean(qsvt_total)),
    }


def benchmark_random_flat_vector(
    key: random.PRNGKeyArray,
    dim: int,
    unit_num_samples: int,
    repetition: int = 10,
) -> dict[str, float]:
    """Benchmark flat-vector state sketching."""
    key, subkey = random.split(key)
    vector = random_flat_vector(subkey, dim, batch_size=repetition)

    state_sketch, num_samples = _q_state_sketch_flat_vec(vector, unit_num_samples)

    error = jnp.linalg.norm(
        state_sketch - vector / jnp.linalg.norm(vector, axis=-1, keepdims=True),
        axis=-1,
    )

    return {
        "error_mean": float(jnp.mean(error)),
        "error_std": float(jnp.std(error) / jnp.sqrt(repetition)),
        "num_samples": float(jnp.mean(num_samples)),
    }


def benchmark_random_boolean_function(
    key: random.PRNGKeyArray,
    dim: int,
    unit_num_samples: int,
    repetition: int = 10,
) -> dict[str, float]:
    """Benchmark Boolean phase-oracle sketching."""
    key, subkey = random.split(key)
    truth_table = random.randint(
        subkey, (repetition, dim), minval=0, maxval=2, dtype=jnp.int32
    )

    oracle_sketch, num_samples = _q_oracle_boolean_vec(truth_table, unit_num_samples)
    target_diag = jnp.exp(1j * jnp.pi * truth_table)
    error = jnp.max(jnp.abs(oracle_sketch - target_diag), axis=-1)

    return {
        "error_mean": float(jnp.mean(error)),
        "error_std": float(jnp.std(error) / jnp.sqrt(repetition)),
        "num_samples": float(jnp.mean(num_samples)),
    }


def benchmark_random_sparse_matrix_element(
    key: random.PRNGKeyArray,
    dim: int,
    nnz: int,
    unit_num_samples: int,
    repetition: int = 10,
) -> dict[str, float]:
    """Benchmark sparse matrix element oracle sketching."""
    key, subkey = random.split(key)
    sparse_matrix = random_sparse_matrix_constant_magnitude(
        subkey, (dim, dim), nnz=nnz, magnitude=1.0, batch_size=repetition
    )

    oracle_sketch, num_samples = _q_oracle_matrix_element_vec(
        sparse_matrix, unit_num_samples
    )
    target_oracle = sparse_matrix.reshape(repetition, -1)
    error = jnp.max(jnp.abs(oracle_sketch - target_oracle), axis=-1)

    return {
        "error_mean": float(jnp.mean(error)),
        "error_std": float(jnp.std(error) / jnp.sqrt(repetition)),
        "num_samples": float(jnp.mean(num_samples)),
    }


def benchmark_random_sparse_matrix_row_index(
    key: random.PRNGKeyArray,
    dim: int,
    unit_num_samples: int,
    repetition: int = 10,
) -> dict[str, float]:
    """Benchmark sparse matrix row-index oracle sketching."""
    key, subkey = random.split(key)
    sparse_matrix = random_sparse_matrix_given_row_sparsity(
        subkey, (dim, dim), row_sparsity=8, batch_size=repetition
    )

    errors = []
    for idx in tqdm(range(repetition), desc="Repetitions", leave=False):
        oracle_sketch, num_samples = q_oracle_sketch_matrix_row_index(
            sparse_matrix[idx], unit_num_samples
        )
        row_sparsity = oracle_sketch.shape[1]
        target_oracle = jnp.zeros((dim, row_sparsity, dim), dtype=real_dtype)
        err_max = 0.0
        for i in range(dim):
            row = sparse_matrix[idx][i]
            non_zero_indices = jnp.nonzero(row)[0]
            for j in range(min(row_sparsity, non_zero_indices.shape[0])):
                col_idx = int(non_zero_indices[j])
                target_oracle = target_oracle.at[i, j, col_idx].set(1.0)
                err_max = max(
                    err_max,
                    float(jnp.sqrt(jnp.sum(jnp.abs(oracle_sketch[i, j] - target_oracle[i, j]) ** 2))),
                )
        errors.append(err_max)

    error = jnp.array(errors)
    return {
        "error_mean": float(jnp.mean(error)),
        "error_std": float(jnp.std(error) / jnp.sqrt(repetition)),
        "num_samples": float(jnp.mean(num_samples)),
    }


def run_benchmark_sweep(
    key: random.PRNGKeyArray,
    benchmark_fn: Callable[..., dict[str, float]],
    dim_list: Sequence[int],
    unit_num_samples_list: Sequence[int],
    repetition: int = 10,
    matrix_dim: int | None = None,
    verbose: bool = False,
) -> dict[int, dict[int, dict[str, float]]]:
    """Run a full sweep over dimensions and sample counts."""
    results: dict[int, dict[int, dict[str, float]]] = {}
    for dim in tqdm(dim_list, desc="Dimensions", position=0):
        results[int(dim)] = {}
        for unit_num_samples in tqdm(
            unit_num_samples_list, desc="Samples", position=1, leave=False
        ):
            key, subkey = random.split(key)
            if matrix_dim is not None:
                res = benchmark_fn(subkey, matrix_dim, dim, unit_num_samples, repetition)
            else:
                res = benchmark_fn(subkey, dim, unit_num_samples, repetition)
            results[int(dim)][int(unit_num_samples)] = res
            if verbose:
                tqdm.write(
                    f"dim={dim}, samples={unit_num_samples}: "
                    + ", ".join(f"{k}={v:.3e}" for k, v in res.items())
                )
    return results


def fit_sample_complexity(
    results: dict[int, dict[int, dict[str, float]]],
    dim_transform: Callable[[int], float] = lambda x: float(x),
) -> dict[str, float]:
    """Fit the ansatz ``M = C * dim^alpha / epsilon^beta`` via log-space least squares."""
    num_samples_list: list[float] = []
    dims_list: list[float] = []
    errors_list: list[float] = []

    for dim, dim_results in results.items():
        for unit_num_samples, res in dim_results.items():
            num_samples_list.append(res["num_samples"])
            dims_list.append(dim_transform(dim))
            errors_list.append(res["error_mean"])

    ln_sample = jnp.log(jnp.array(num_samples_list))
    ln_dim = jnp.log(jnp.array(dims_list))
    ln_error = jnp.log(jnp.array(errors_list))

    ones = jnp.ones_like(ln_sample)
    A = jnp.stack([ones, ln_dim, ln_error], axis=1)
    coeffs, residuals, rank, s = jnp.linalg.lstsq(A, ln_sample)

    ln_C, alpha, neg_beta = coeffs
    cov_matrix = jnp.linalg.pinv(A.T @ A) * residuals[0] / (A.shape[0] - A.shape[1])
    std = jnp.sqrt(jnp.diag(cov_matrix))
    ln_C_std, alpha_std, neg_beta_std = std

    fit = {
        "C": float(jnp.exp(ln_C)),
        "alpha": float(alpha),
        "beta": float(-neg_beta),
        "rmse": float(jnp.sqrt(residuals / (A.shape[0] - A.shape[1]))[0]),
        "C_std": float(jnp.exp(ln_C) * ln_C_std),
        "alpha_std": float(alpha_std),
        "beta_std": float(neg_beta_std),
    }

    print(
        f"Fit: C={fit['C']:.3e}\u00b1{fit['C_std']:.1e}, "
        f"alpha={fit['alpha']:.3f}\u00b1{fit['alpha_std']:.1e}, "
        f"beta={fit['beta']:.3f}\u00b1{fit['beta_std']:.1e}, "
        f"rmse={fit['rmse']:.3e}"
    )
    return fit


def plot_benchmark_results(
    results: dict[int, dict[int, dict[str, float]]],
    title: str,
    dim_list: Sequence[int] | None = None,
    fit: dict[str, float] | None = None,
    dim_transform: Callable[[int], float] = lambda x: float(x),
    dim_label: str = "N",
    dim_fit_label: str = "N",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot benchmark sweep results with optional fit curve."""
    plt.figure(figsize=(4, 3))

    color_counter = 0
    for dim, dim_results in results.items():
        if dim_list is not None and dim not in dim_list:
            continue

        num_samples_arr = []
        error_mean_arr = []
        error_std_arr = []
        for unit_num_samples, res in dim_results.items():
            num_samples_arr.append(res["num_samples"])
            error_mean_arr.append(res["error_mean"])
            error_std_arr.append(res["error_std"])

        num_samples_arr = jnp.array(num_samples_arr)
        error_mean_arr = jnp.array(error_mean_arr)
        error_std_arr = jnp.array(error_std_arr)

        plt.errorbar(
            num_samples_arr,
            error_mean_arr,
            yerr=error_std_arr,
            fmt="o",
            label=rf"${dim_label} = {dim}$",
            color=_colors[color_counter % len(_colors)],
            capsize=5,
        )

        if fit is not None:
            error_fit = (
                fit["C"] * (dim_transform(dim) ** fit["alpha"]) / (num_samples_arr ** fit["beta"])
            )
            plt.plot(num_samples_arr, error_fit, "-", color=_colors[color_counter % len(_colors)])
            color_counter += 1

    fit_handles = None
    if fit is not None:
        label_str = rf"Fit: $M = {fit['C']:.1f} {dim_fit_label}^{{{fit['alpha']:.2f}}}/\epsilon^{{{fit['beta']:.2f}}}$"
        rmse_str = rf"RMS rel. err.: ${fit['rmse'] * 100:.1f}\%$"
        fit_handles = [
            Line2D([], [], color="grey", linestyle="-", label=label_str),
            Line2D([], [], color="none", label=rmse_str),
        ]

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Number of samples $M$")
    plt.ylabel(r"Error $\epsilon$")
    plt.grid(True, which="major", linestyle="-", linewidth=0.8, color="gray", alpha=0.4)
    plt.grid(True, which="minor", linestyle=":", linewidth=0.5, color="gray", alpha=0.3)
    data_legend = plt.legend(loc="upper right")
    if fit_handles is not None:
        plt.gca().add_artist(data_legend)
        plt.legend(handles=fit_handles, loc="lower left", handlelength=1)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    plt.close()


def main() -> None:
    key = random.PRNGKey(42)
    repetition = 10

    # 1. Flat vector state sketching
    key, subkey = random.split(key)
    flat_results = run_benchmark_sweep(
        subkey,
        benchmark_random_flat_vector,
        dim_list=[100, 1000, 10000],
        unit_num_samples_list=[
            10_000, 100_000, 1_000_000, 10_000_000, 100_000_000
        ],
        repetition=repetition,
    )
    flat_fit = fit_sample_complexity(flat_results)
    plot_benchmark_results(
        flat_results,
        title="Flat vector state sketching",
        dim_list=[100, 1000, 10000],
        fit=flat_fit,
        save_path="benchmark_flat_vector.pdf",
    )

    # 2. General vector state sketching
    key, subkey = random.split(key)
    vector_results = run_benchmark_sweep(
        subkey,
        benchmark_random_vector,
        dim_list=[100, 1000, 10000],
        unit_num_samples_list=[
            10_000, 100_000, 1_000_000, 10_000_000, 100_000_000
        ],
        repetition=repetition,
    )
    vector_fit = fit_sample_complexity(vector_results)
    plot_benchmark_results(
        vector_results,
        title="General vector state sketching",
        dim_list=[100, 1000, 10000],
        fit=vector_fit,
        save_path="benchmark_general_vector.pdf",
    )

    # 3. Boolean function oracle sketching
    key, subkey = random.split(key)
    boolean_results = run_benchmark_sweep(
        subkey,
        benchmark_random_boolean_function,
        dim_list=[100, 1000, 10000],
        unit_num_samples_list=[
            10_000, 100_000, 1_000_000, 10_000_000, 100_000_000
        ],
        repetition=repetition,
    )
    boolean_fit = fit_sample_complexity(boolean_results)
    plot_benchmark_results(
        boolean_results,
        title="Boolean function oracle sketching",
        dim_list=[100, 1000, 10000],
        fit=boolean_fit,
        save_path="benchmark_boolean_function.pdf",
    )

    # 4. Sparse matrix element oracle sketching
    key, subkey = random.split(key)
    element_results = run_benchmark_sweep(
        subkey,
        benchmark_random_sparse_matrix_element,
        dim_list=[100, 1000, 10000],
        unit_num_samples_list=[
            10_000, 100_000, 1_000_000, 10_000_000, 100_000_000
        ],
        repetition=repetition,
        matrix_dim=1000,
    )
    element_fit = fit_sample_complexity(
        element_results,
        dim_transform=lambda x: float(x),
    )
    plot_benchmark_results(
        element_results,
        title="Sparse matrix element oracle sketching",
        dim_list=[100, 1000, 10000],
        fit=element_fit,
        dim_label="nnz",
        dim_fit_label="nnz",
        save_path="benchmark_matrix_element.pdf",
    )

    # 5. Sparse matrix row-index oracle sketching
    key, subkey = random.split(key)
    row_index_results = run_benchmark_sweep(
        subkey,
        benchmark_random_sparse_matrix_row_index,
        dim_list=[100, 1000, 10000],
        unit_num_samples_list=[
            10_000, 100_000, 1_000_000, 10_000_000, 100_000_000
        ],
        repetition=repetition,
    )
    row_index_fit = fit_sample_complexity(row_index_results)
    plot_benchmark_results(
        row_index_results,
        title="Sparse matrix row-index oracle sketching",
        dim_list=[100, 1000, 10000],
        fit=row_index_fit,
        save_path="benchmark_matrix_row_index.pdf",
    )


if __name__ == "__main__":
    main()
