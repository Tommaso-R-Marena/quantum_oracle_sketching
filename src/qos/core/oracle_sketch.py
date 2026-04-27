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
    """Uniform Boolean phase-oracle diagonal via expected-unitary accumulation."""
    M         = int(unit_num_samples)
    f         = truth_table.astype(jnp.float64)
    N         = f.shape[0]
    p         = jnp.float64(1.0) / jnp.float64(N)
    t_over_M  = jnp.float64(jnp.pi * N) / jnp.float64(M)
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

def q_oracle_sketch_matrix_element(
    matrix: jax.Array,
    unit_num_samples: int,
) -> tuple[jax.Array, int]:
    """Sparse matrix-element oracle: block-diagonal sine component.

    vmap-safe: no Python-level bool/float conversion of traced arrays.
    """
    dims = matrix.shape
    nnz = jnp.sum((matrix != 0).astype(real_dtype))
    nnz_safe = jnp.where(nnz > 0, nnz, jnp.ones((), dtype=real_dtype))
    prob = jnp.where(matrix != 0, 1.0 / nnz_safe, 0.0).astype(real_dtype)
    t = nnz_safe
    log_diag = jnp.log1p(
        prob * jnp.expm1(1j * t / unit_num_samples * jnp.arcsin(matrix))
    )
    diag = jnp.exp(unit_num_samples * log_diag).reshape(dims[0] * dims[1])
    sin = (diag - diag.conj()) / (2j)
    return sin, int(unit_num_samples)


def q_oracle_sketch_matrix_row_index(
    matrix: jax.Array,
    unit_num_samples: int,
    max_row_sparsity: int | None = None,
) -> tuple[jax.Array, int]:
    """Sparse matrix row-index oracle via expected phase accumulation.

    NOT vmapped -- called from a Python loop in the benchmark, so
    .item() concretization of traced scalars is safe here.

    Args:
        matrix: 2-D array of shape (rows, cols).
        unit_num_samples: Sample budget M.
        max_row_sparsity: If provided, caps the sparsity dimension to avoid
            allocating a full rows x cols tensor. Defaults to the true max
            nonzero count per row (computed eagerly via .item()).
    """
    dims = matrix.shape
    if max_row_sparsity is not None:
        sparsity = int(max_row_sparsity)
    else:
        # .item() forces eager evaluation -- safe because this function is
        # never traced under vmap/jit.
        sparsity = int(jnp.max(jnp.sum(matrix != 0, axis=1)).item())
        sparsity = max(sparsity, 1)  # guard against all-zero matrix

    bitlength_col = int(jnp.ceil(jnp.log2(max(dims[1], 2))))
    nz_col_indices = jnp.argsort(matrix != 0, axis=1, descending=True)[:, :sparsity]
    bit_positions = jnp.arange(bitlength_col - 1, -1, -1)
    col_bits = (nz_col_indices[..., None] >> bit_positions) & 1
    t = jnp.pi * dims[0]
    prob = jnp.ones((dims[0],), dtype=real_dtype) / dims[0]
    log_diag = jnp.log1p(
        prob[:, None, None] * jnp.expm1(1j * t / unit_num_samples * col_bits)
    )
    diag = jnp.exp(unit_num_samples * log_diag)
    controlled_diag = jnp.stack([jnp.ones_like(diag), diag], axis=-1)
    hadamard = jnp.array([[1, 1], [1, -1]], dtype=real_dtype) / jnp.sqrt(2)
    xor_oracle = jnp.einsum("am,ijkm,mn->ijkan", hadamard, controlled_diag, hadamard)
    state = xor_oracle[:, :, 0, :, 0]
    for bit in range(1, bitlength_col):
        state = jnp.einsum(
            "ija,ijb->ijab", state, xor_oracle[:, :, bit, :, 0]
        ).reshape(dims[0], sparsity, -1)
    return state[:, :, : dims[1]], int(unit_num_samples)


def q_oracle_sketch_matrix_index(
    matrix: jax.Array,
    unit_num_samples: int,
    axis: int,
    degree: int = DEFAULT_CONFIG.sign_degree,
    scale: float = DEFAULT_CONFIG.sign_rescale,
    angle_set: jax.Array | None = None,
) -> tuple[jax.Array, int]:
    """Sparse row/column index oracle with rank register via exact CDF sign."""
    prob_matrix = matrix if axis == 0 else matrix.T
    num_rows, orig_num_cols = prob_matrix.shape
    row_counts = jnp.sum((prob_matrix != 0).astype(real_dtype), axis=1)
    # Use the true max per-row support so output rank dimension matches the
    # sparse structure (and test expectations) instead of always using num_cols.
    sparsity = int(jnp.max(row_counts).item())
    sparsity = max(sparsity, 1)

    indicator = (prob_matrix != 0).astype(real_dtype)
    row_counts_safe = jnp.where(row_counts == 0, 1.0, row_counts)
    prob_norm = indicator / row_counts_safe[:, None]
    cdf = jnp.cumsum(prob_norm, axis=1)

    k_indices = jnp.arange(sparsity, dtype=real_dtype)
    thresholds = (k_indices[None, :] + 0.5) / row_counts_safe[:, None]

    sign_grid = jnp.where(
        cdf[:, None, :] >= thresholds[:, :, None],
        jnp.ones((), dtype=real_dtype),
        -jnp.ones((), dtype=real_dtype),
    )

    sentinel_col = jnp.full((num_rows, sparsity, 1), -1.0, dtype=real_dtype)
    sign_with_sentinel = jnp.concatenate([sentinel_col, sign_grid], axis=-1)
    delta = sign_with_sentinel[:, :, 1:] - sign_with_sentinel[:, :, :-1]

    total_samples = int(unit_num_samples * degree * 2)
    return delta, total_samples
