"""Quantum oracle sketching: expected-unitary constructions.

# Copyright (c) 2026 Tommaso R. Marena. MIT License.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import random

from qos.config import DEFAULT_CONFIG, int_dtype, real_dtype

if TYPE_CHECKING:
    from jax import random as jax_random


def q_oracle_sketch_boolean(
    truth_table: jax.Array,
    unit_num_samples: int,
) -> tuple[jax.Array, int]:
    """Uniform Boolean phase-oracle diagonal via expected-unitary accumulation.

    Target: diag[x] = exp(i*pi*f(x)).

    Zhao et al. formula::

        p      = 1/N               uniform probability
        t      = pi * N            p * t = pi for all x
        diag   = exp(M * log(1 + p * expm1(i * (t/M) * f)))

    Per-step angle t/M = pi*N/M.  As M -> inf, diag -> exp(i*pi*f).
    All arithmetic in float64/complex128.
    """
    M         = int(unit_num_samples)
    f         = truth_table.astype(jnp.float64)
    N         = f.shape[0]
    p         = jnp.float64(1.0) / jnp.float64(N)
    t_over_M  = jnp.float64(jnp.pi * N) / jnp.float64(M)   # pi*N/M
    phase_arg = jnp.complex128(1j) * t_over_M * f
    log_diag  = jnp.log1p(p * jnp.expm1(phase_arg))
    diag      = jnp.exp(jnp.float64(M) * log_diag)
    return diag, M


def q_oracle_sketch_boolean_adaptive(
    truth_table: jax.Array,
    unit_num_samples: int,
    pilot_frac: float = 0.1,
    key: jax.Array | None = None,
) -> tuple[jax.Array, int, jax.Array]:
    """Adaptive Boolean oracle diagonal -- K-sparse concentration (Marena 2026)."""
    N = int(truth_table.shape[0])
    if key is None:
        key = random.PRNGKey(0)

    M_pilot = int(float(pilot_frac) * unit_num_samples)
    M_main  = max(unit_num_samples - M_pilot, 1)

    f      = truth_table.astype(jnp.float64)
    K_true = float(jnp.sum(f))

    uniform_q = jnp.ones((N,), dtype=jnp.float64) / jnp.float64(N)

    if M_pilot == 0 or K_true == 0.0 or K_true == float(N):
        diag, _ = q_oracle_sketch_boolean(truth_table, unit_num_samples)
        return diag, int(unit_num_samples), uniform_q

    q           = f / jnp.float64(K_true)
    phase_step  = jnp.float64(jnp.pi) * jnp.float64(K_true) / jnp.float64(M_main)
    inner       = q * jnp.expm1(jnp.complex128(1j) * phase_step * f)
    log_term    = jnp.log1p(inner)
    diag        = jnp.exp(jnp.float64(M_main) * log_term)

    return diag, int(unit_num_samples), q


# ---------------------------------------------------------------------------
# Matrix oracle APIs
# ---------------------------------------------------------------------------

def q_oracle_sketch_matrix_element(matrix: jax.Array, unit_num_samples: int) -> tuple[jax.Array, int]:
    """Sparse matrix-element oracle: block-diagonal sine component."""
    dims = matrix.shape
    nnz = jnp.count_nonzero(matrix)
    if nnz == 0:
        raise ValueError("Matrix has no non-zero elements.")
    t = float(nnz)
    prob = jnp.where(matrix != 0, 1.0 / nnz, 0.0).astype(real_dtype)
    log_diag = jnp.log1p(prob * jnp.expm1(1j * t / unit_num_samples * jnp.arcsin(matrix)))
    diag = jnp.exp(unit_num_samples * log_diag).reshape(dims[0] * dims[1])
    sin = (diag - diag.conj()) / (2j)
    return sin, int(unit_num_samples)


def q_oracle_sketch_matrix_row_index(matrix: jax.Array, unit_num_samples: int) -> tuple[jax.Array, int]:
    """Sparse matrix row-index oracle via expected phase accumulation."""
    dims = matrix.shape
    row_counts = jnp.count_nonzero(matrix, axis=1)
    sparsity = int(jnp.max(row_counts))
    if sparsity == 0:
        raise ValueError("Matrix has no non-zero elements.")
    bitlength_col = int(jnp.ceil(jnp.log2(dims[1])))
    nz_col_indices = jnp.argsort(matrix != 0, axis=1, descending=True)[:, :sparsity]
    bit_positions = jnp.arange(bitlength_col - 1, -1, -1)
    col_bits = (nz_col_indices[..., None] >> bit_positions) & 1
    t = jnp.pi * dims[0]
    prob = jnp.ones((dims[0],), dtype=real_dtype) / dims[0]
    log_diag = jnp.log1p(prob[:, None, None] * jnp.expm1(1j * t / unit_num_samples * col_bits))
    diag = jnp.exp(unit_num_samples * log_diag)
    controlled_diag = jnp.stack([jnp.ones_like(diag), diag], axis=-1)
    hadamard = jnp.array([[1, 1], [1, -1]], dtype=real_dtype) / jnp.sqrt(2)
    xor_oracle = jnp.einsum("am,ijkm,mn->ijkan", hadamard, controlled_diag, hadamard)
    state = xor_oracle[:, :, 0, :, 0]
    for bit in range(1, bitlength_col):
        state = jnp.einsum("ija,ijb->ijab", state, xor_oracle[:, :, bit, :, 0]).reshape(dims[0], sparsity, -1)
    return state[:, :, : dims[1]], int(unit_num_samples)


def q_oracle_sketch_matrix_index(
    matrix: jax.Array,
    unit_num_samples: int,
    axis: int,
    degree: int = DEFAULT_CONFIG.sign_degree,
    scale: float = DEFAULT_CONFIG.sign_rescale,
    angle_set: jax.Array | None = None,
) -> tuple[jax.Array, int]:
    """Sparse row/column index oracle with rank register via exact CDF sign.

    For each row i and rank k in 0..sparsity-1, identifies the k-th nonzero
    column of row i by computing sign_grid[i, k, j] = sign(CDF[i,j] - threshold_k)
    from the exact integer CDF, then taking the diff along the *column* axis.

    Key insight: for a fixed rank k, sign_grid[i, k, :] is -1 for j < j_k and
    +1 for j >= j_k (a step function in j). Differencing along the column axis
    gives delta[i, k, j] = +2 exactly at j = j_k, and 0 everywhere else.
    argmax(|delta|, axis=-1) then recovers j_k exactly.

    Returns:
        ``(delta, total_samples)`` where:
        - ``delta`` has shape ``(num_rows, sparsity, orig_num_cols)``.
        - ``delta[i, k, j_k] == 2.0`` exactly and 0 elsewhere.
        - ``jnp.argmax(jnp.abs(delta), axis=-1)`` gives the sorted column indices.
        - Invalid rank slots (k >= row_count[i]) have delta[i,k,:] == 0,
          so argmax returns 0 -- masked out by ~valid in the tests.
    """
    prob_matrix = matrix if axis == 0 else matrix.T
    num_rows, orig_num_cols = prob_matrix.shape
    sparsity = int(jnp.max(jnp.count_nonzero(prob_matrix, axis=1)))
    nnz = jnp.count_nonzero(prob_matrix)
    if nnz == 0:
        raise ValueError("Matrix has no non-zero elements.")

    # Exact per-row CDF from integer indicator: CDF[i,j] = #{nonzeros in row i up to col j} / row_count_i
    indicator = (prob_matrix != 0).astype(real_dtype)           # (num_rows, orig_num_cols)
    row_sums = jnp.sum(indicator, axis=1, keepdims=True)        # (num_rows, 1)
    row_sums_safe = jnp.where(row_sums == 0, 1.0, row_sums)
    prob_norm = indicator / row_sums_safe                        # normalized, each row sums to 1
    cdf = jnp.cumsum(prob_norm, axis=1)                         # (num_rows, orig_num_cols)

    # Threshold for rank k (0-indexed): threshold_k = (k + 0.5) / sparsity
    # This places each threshold strictly between the k-th and (k+1)-th step of the CDF.
    k_indices = jnp.arange(sparsity, dtype=real_dtype)           # (sparsity,) = [0, 1, ..., s-1]
    thresholds = (k_indices + 0.5) / float(sparsity)             # (sparsity,)

    # sign_grid[i, k, j] = +1 if CDF[i,j] >= threshold_k, else -1.
    # For a fixed k, this is a step function in j that jumps from -1 to +1
    # exactly at j = j_k (the k-th nonzero column, 0-indexed).
    # Shape: (num_rows, sparsity, orig_num_cols)
    sign_grid = jnp.where(
        cdf[:, None, :] >= thresholds[None, :, None],
        jnp.ones((), dtype=real_dtype),
        -jnp.ones((), dtype=real_dtype),
    )  # (num_rows, sparsity, orig_num_cols)

    # Diff along the COLUMN axis (axis=-1) to locate the step.
    # Prepend a sentinel column of -1 at j=0 so the diff at j=j_k is:
    #   sign_grid[i,k,j_k] - sign_grid[i,k,j_k-1] = +1 - (-1) = +2  (if j_k=0: +1-(-1)=+2)
    # All other positions give 0 (no step) or -2 (impossible for a valid CDF step).
    sentinel_col = jnp.full((num_rows, sparsity, 1), -1.0, dtype=real_dtype)
    sign_with_sentinel = jnp.concatenate([sentinel_col, sign_grid], axis=-1)  # (..., orig_num_cols+1)
    delta = sign_with_sentinel[:, :, 1:] - sign_with_sentinel[:, :, :-1]     # (..., orig_num_cols)
    # delta[i, k, j] == +2.0 iff j == j_k, and 0.0 everywhere else.
    # For rows where row_count < sparsity (invalid ranks), delta[i,k,:] == 0
    # because sign_grid[i,k,:] is all +1 (CDF=1 >= any threshold) -> diff=0.

    total_samples = int(unit_num_samples * degree * 2)
    return delta, total_samples
