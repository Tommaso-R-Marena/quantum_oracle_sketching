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

# Complex dtype that always matches real_dtype precision.
_c128 = jnp.complex128
_c64  = jnp.complex64


def _cplx(x: jax.Array) -> jax.Array:
    """Cast to complex128 unconditionally (safe on CPU; falls back to c64 on GPU/TPU)."""
    try:
        return x.astype(jnp.complex128)
    except TypeError:
        return x.astype(jnp.complex64)


def q_oracle_sketch_boolean(
    truth_table: jax.Array,
    unit_num_samples: int,
) -> tuple[jax.Array, int]:
    """Uniform Boolean phase-oracle diagonal via expected-unitary accumulation.

    Target: diag[x] = exp(i*pi*f(x)).

    Formula (Zhao et al. 2025/2026, Section D)::

        p(x) = 1/N                          (uniform)
        t    = pi * N   => p(x)*t = pi      for f(x)=1
                        => p(x)*t = 0       for f(x)=0
        diag = exp(M * log(1 + p * expm1(i*t*f/M)))

    For large M this converges to exp(i*p*t*f) = exp(i*pi*f).

    The entire computation is done in complex128 to avoid catastrophic
    cancellation in expm1 when the argument is small (t*f/M ~ pi/M -> 0).

    Args:
        truth_table: Boolean array in {0,1}, shape (N,).
        unit_num_samples: Number of samples M.

    Returns:
        (diag, M) where diag ~= exp(i*pi*f).
    """
    M = int(unit_num_samples)
    # Cast to float64 first so subsequent complex arithmetic stays in c128.
    f   = truth_table.astype(jnp.float64)          # shape (N,), dtype float64
    N   = f.shape[0]
    p   = jnp.float64(1.0 / N)                     # scalar float64
    t   = jnp.float64(jnp.pi * N)                  # p*t = pi
    # Rotation angle per sample: theta = p*t/M = pi/M
    theta = jnp.float64(jnp.pi) / jnp.float64(M)  # scalar float64, ~pi/M
    # expm1(i*theta*f): for f=1, ~i*theta (tiny angle); for f=0, 0.
    # Full precision: all intermediate values in complex128.
    phase_arg = (1j * theta) * f                    # complex128 broadcast
    log_diag  = jnp.log1p(p * jnp.expm1(phase_arg))# complex128
    diag      = jnp.exp(jnp.float64(M) * log_diag) # complex128
    return diag, M


def q_oracle_sketch_boolean_adaptive(
    truth_table: jax.Array,
    unit_num_samples: int,
    pilot_frac: float = 0.1,
    key: jax.Array | None = None,
) -> tuple[jax.Array, int, jax.Array]:
    """Adaptive importance-sampled Boolean oracle diagonal (Marena 2026).

    ## Two-phase algorithm

    **Phase 1 - Pilot (M_pilot = pilot_frac * M samples):**
    Sample M_pilot indices uniformly. Count hits per position.
    Build Laplace-smoothed importance weights q(x) on supp(f):

        q(x) = (count(x) + 1) / sum_{y: f(y)=1} (count(y) + 1)  if f(x) = 1
        q(x) = 0                                                   if f(x) = 0

    Estimate K_hat = (pilot_hits / M_pilot) * N.

    **Phase 2 - Main oracle (M_main = M - M_pilot samples):**
    Per-entry rotation angle:

        theta(x) = q(x) * pi * K_hat / M_main

    Oracle diagonal via expected-unitary log-sum (all in complex128):

        diag[x] = exp(M_main * log(1 + expm1(i*theta(x)) * f(x)))

    For perfect pilot (q(x) = 1/K, K_hat = K):
        theta(x) = pi/M_main  =>  M_main * log(1 + expm1(i*pi/M_main))
                             ~=   M_main * (i*pi/M_main) = i*pi
        => diag[x] = exp(i*pi) = -1. Correct.

    Off-support (f(x)=0): expm1(...)*0 = 0 => log1p(0) = 0 => exp(0) = 1.

    ## Sample complexity

    Pilot concentration (Bernstein): |delta(x)| = O(sqrt(K/M_pilot)) w.h.p.
    Error on supp(f): ~pi*|delta| => M_pilot = O(K*pi^2/eps^2).
    Total M = O(K/eps^2).  Improvement factor N/K over Zhao et al. O(N/eps^2).

    ## Fallbacks

    - pilot_frac = 0            -> uniform oracle
    - supp(f) = {} (all zeros)  -> uniform oracle (trivially correct)
    - supp(f) = full (all ones) -> uniform oracle

    Args:
        truth_table: Boolean array in {0,1}, shape (N,).
        unit_num_samples: Total samples M.
        pilot_frac: Fraction of M used for pilot. Default 0.1.
        key: JAX PRNG key. If None, uses PRNGKey(0).

    Returns:
        (diag, M, importance_weights) where importance_weights is q(x).
    """
    N = int(truth_table.shape[0])
    if key is None:
        key = random.PRNGKey(0)

    M_pilot = int(float(jnp.clip(jnp.float32(pilot_frac), 0.0, 1.0)) * unit_num_samples)
    M_main  = max(unit_num_samples - M_pilot, 1)

    uniform_q = jnp.ones((N,), dtype=jnp.float64) / N
    f         = truth_table.astype(jnp.float64)     # float64 throughout
    K_true    = float(jnp.sum(f))

    if M_pilot == 0 or K_true == 0.0 or K_true == float(N):
        diag, _ = q_oracle_sketch_boolean(truth_table, unit_num_samples)
        return diag, int(unit_num_samples), uniform_q

    # ------------------------------------------------------------------
    # Phase 1: Pilot
    # ------------------------------------------------------------------
    key, pkey = random.split(key)
    idx    = random.randint(pkey, shape=(M_pilot,), minval=0, maxval=N)
    counts = jnp.bincount(idx, length=N).astype(jnp.float64)

    raw = (counts + 1.0) * f          # Laplace-smooth, zero off support
    Z   = jnp.sum(raw)
    q   = raw / Z                     # float64, sums to 1 on supp(f)

    pilot_hits = jnp.sum(counts * f)
    hit_rate   = pilot_hits / jnp.float64(M_pilot)
    K_hat      = float(jnp.clip(hit_rate * N, 1.0, float(N)))

    # ------------------------------------------------------------------
    # Phase 2: Main oracle (complex128)
    # ------------------------------------------------------------------
    # theta(x) = q(x) * pi * K_hat / M_main
    # For perfect pilot: theta(x) = (1/K)*(pi*K)/M_main = pi/M_main
    theta = q * (jnp.pi * K_hat) / jnp.float64(M_main)   # float64 shape (N,)

    # Argument to expm1 is imaginary: i*theta(x)
    # Off-support: theta(x)=0 => expm1(0)=0 => log1p(0*f)=0 => exp(0)=1
    # On-support: accumulated M_main times gives exp(i*pi) = -1
    phase_arg = (1j * theta) * f                          # complex128 shape (N,)
    log_term  = jnp.log1p(jnp.expm1(phase_arg))          # NB: f already applied via theta
    # Wait: theta is already zero off-support (q=0 there), so phase_arg=0 off-support.
    # But we must NOT multiply by f again — q already encodes support.
    # Rewrite: expm1(i*theta)*f vs expm1(i*theta*f).
    # Since theta(x)=0 for f(x)=0, both are equivalent. Use explicit f mask for safety.
    phase_arg = 1j * (theta * f)                          # complex128, zero off-supp
    log_term  = jnp.log1p(jnp.expm1(phase_arg))          # complex128
    diag      = jnp.exp(jnp.float64(M_main) * log_term)  # complex128

    return diag, int(unit_num_samples), q


# ---------------------------------------------------------------------------
# Matrix oracle APIs (unchanged from v1.2)
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
    """Sparse row/column index oracle with rank register via QSVT."""
    from qos.qsvt.angles import get_qsvt_angles_sign
    from qos.qsvt.transform import apply_qsvt_diag

    num_rows = matrix.shape[axis]
    bitlength_col = int(jnp.ceil(jnp.log2(matrix.shape[1 - axis])))
    orig_num_cols = matrix.shape[1 - axis]
    num_cols = 2**bitlength_col
    sparsity = int(jnp.max(jnp.count_nonzero(matrix, axis=1 - axis)))
    nnz = jnp.count_nonzero(matrix)
    if nnz == 0:
        raise ValueError("Matrix has no non-zero elements.")

    k_indices = jnp.arange(sparsity, dtype=int_dtype) + 1
    t = jnp.pi * nnz / (2 * sparsity + 1)
    k_phase_scale = jnp.pi / (2 * sparsity + 1)
    prob_matrix = matrix if axis == 0 else matrix.T
    prob = jnp.zeros_like(prob_matrix, dtype=real_dtype)
    prob = prob.at[prob_matrix != 0].set(1.0 / nnz)
    prob = jnp.pad(prob, ((0, 0), (0, num_cols - orig_num_cols)), constant_values=0.0)
    prob = jnp.cumsum(prob, axis=1) - prob

    log_diag = jnp.log1p(prob * jnp.expm1(1j * t / unit_num_samples))
    log_diag = jnp.repeat(log_diag[:, None, :], sparsity, axis=1)
    log_diag = log_diag - 1j * (k_indices[None, :, None] - 0.5) * k_phase_scale
    diag = jnp.exp(unit_num_samples * log_diag).reshape(num_rows * sparsity * num_cols)

    sin = (diag - jnp.conj(diag)) / (2j)
    cos = (diag + jnp.conj(diag)) / 2
    block_encoding = jnp.stack([jnp.stack([sin, cos], axis=0), jnp.stack([cos, -sin], axis=0)], axis=0)

    if angle_set is None:
        threshold = jnp.pi / (4 * sparsity + 2) * DEFAULT_CONFIG.sign_threshold_factor
        angle_set, _ = get_qsvt_angles_sign(degree=degree, threshold=float(threshold), rescale=scale)
        angle_set = angle_set.astype(real_dtype)

    block_encoding = apply_qsvt_diag(block_encoding, num_ancilla=1, angle_set=angle_set)
    block_encoding = jnp.real(block_encoding[0, 0]).reshape(num_rows, sparsity, num_cols)

    hadamard = jnp.array([[1, 1], [1, -1]], dtype=real_dtype) / jnp.sqrt(2)
    cont = jnp.stack([jnp.ones_like(block_encoding), block_encoding], axis=-1)
    xor = jnp.einsum("am,ijkm,mn->ijkan", hadamard, cont, hadamard)
    state = xor[:, :, 0, :, 0]
    for bit in range(1, bitlength_col):
        state = jnp.einsum("ija,ijb->ijab", state, xor[:, :, bit, :, 0]).reshape(num_rows, sparsity, -1)
    state = state[:, :, :orig_num_cols]
    return (state, int(unit_num_samples * (angle_set.shape[0] - 1)))
