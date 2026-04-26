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
    """Sparse row/column index oracle with rank register via QSVT sign function.

    For each row i and rank k (1..sparsity), this oracle identifies the k-th
    nonzero column of row i. It works by:
      1. Building the per-row CDF: CDF[i,j] = #{nonzeros in row i at cols 0..j}
                                              / #{nonzeros in row i}
      2. The QSVT sign function evaluates sign(CDF[i,j] - (k-0.5)/s) for each
         (i, k, j). This is +1 for j >= j_k and -1 for j < j_k.
      3. The difference sign_k[i,j] - sign_{k-1}[i,j] is +2 at column j=j_k
         and 0 elsewhere (since the step moves up by exactly 1/s at each
         nonzero column). Taking argmax over j gives j_k directly.
    """
    from qos.qsvt.angles import get_qsvt_angles_sign
    from qos.qsvt.transform import apply_qsvt_diag

    prob_matrix = matrix if axis == 0 else matrix.T
    num_rows, orig_num_cols = prob_matrix.shape
    bitlength_col = int(jnp.ceil(jnp.log2(orig_num_cols)))
    num_cols = 2**bitlength_col
    sparsity = int(jnp.max(jnp.count_nonzero(prob_matrix, axis=1)))
    nnz = jnp.count_nonzero(prob_matrix)
    if nnz == 0:
        raise ValueError("Matrix has no non-zero elements.")

    # Per-row CDF: normalize each row by its own row-sum, then cumsum.
    indicator = (prob_matrix != 0).astype(real_dtype)  # (num_rows, orig_num_cols)
    indicator_pad = jnp.pad(indicator, ((0, 0), (0, num_cols - orig_num_cols)), constant_values=0.0)
    row_sums = jnp.sum(indicator_pad, axis=1, keepdims=True)
    row_sums_safe = jnp.where(row_sums == 0, 1.0, row_sums)
    prob_norm = indicator_pad / row_sums_safe          # each row sums to 1
    cdf = jnp.cumsum(prob_norm, axis=1)               # (num_rows, num_cols), values in [0,1]

    # For each rank k in 1..sparsity, threshold = (k - 0.5) / sparsity.
    # sign_val[i, k, j] = sign(cdf[i,j] - threshold_k).
    # We encode this as a flat block-encoding over (num_rows * sparsity * num_cols) elements.
    k_indices = jnp.arange(1, sparsity + 1, dtype=real_dtype)          # (sparsity,)
    thresholds = (k_indices - 0.5) / float(sparsity)                   # (sparsity,)
    # cdf_expanded: (num_rows, sparsity, num_cols)
    cdf_expanded = cdf[:, None, :] - thresholds[None, :, None]         # centered at threshold

    # Build block encoding of the centered CDF values.
    # We map each entry x -> sin(pi/2 * x) approximately via log1p trick.
    t_scale = jnp.pi / 2.0
    flat = cdf_expanded.reshape(-1)  # (num_rows * sparsity * num_cols,)
    log_diag = jnp.log1p(flat.astype(jnp.complex128) * jnp.expm1(1j * t_scale / unit_num_samples))
    diag = jnp.exp(unit_num_samples * log_diag)
    sin_val = ((diag - jnp.conj(diag)) / (2j)).real
    cos_val = ((diag + jnp.conj(diag)) / 2).real
    block_encoding = jnp.stack(
        [jnp.stack([sin_val, cos_val], axis=0),
         jnp.stack([cos_val, -sin_val], axis=0)],
        axis=0
    )

    if angle_set is None:
        threshold_qsvt = 0.5 / float(sparsity) * DEFAULT_CONFIG.sign_threshold_factor
        angle_set, _ = get_qsvt_angles_sign(
            degree=degree, threshold=float(threshold_qsvt), rescale=scale
        )
        angle_set = angle_set.astype(real_dtype)

    block_encoding = apply_qsvt_diag(block_encoding, num_ancilla=1, angle_set=angle_set)
    # sign_grid[i, k, j] = sign(cdf[i,j] - threshold_k), shape (num_rows, sparsity, num_cols)
    sign_grid = jnp.real(block_encoding[0, 0]).reshape(num_rows, sparsity, num_cols)

    # Column indicator: delta_k[i, k, j] = sign_k[i,j] - sign_{k-1}[i,j].
    # Prepend a row of -1 (below the lowest threshold) as the k=0 sentinel.
    sentinel = jnp.full((num_rows, 1, num_cols), -1.0, dtype=sign_grid.dtype)
    sign_with_sentinel = jnp.concatenate([sentinel, sign_grid], axis=1)  # (num_rows, sparsity+1, num_cols)
    delta = sign_with_sentinel[:, 1:, :] - sign_with_sentinel[:, :-1, :]  # (num_rows, sparsity, num_cols)

    # delta[i, k, j] == +2 at j = j_k (the k-th nonzero column of row i), 0 elsewhere.
    # Clip to orig_num_cols to remove padding.
    delta = delta[:, :, :orig_num_cols]

    total_samples = int(unit_num_samples * (angle_set.shape[0] - 1))
    return delta, total_samples
