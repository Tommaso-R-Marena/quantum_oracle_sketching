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

    Per-step angle t/M = pi*N/M. As M -> inf, diag -> exp(i*pi*f).
    All arithmetic in float64/complex128 to avoid expm1 cancellation.
    """
    M = int(unit_num_samples)
    f        = truth_table.astype(jnp.float64)
    N        = f.shape[0]
    p        = jnp.float64(1.0) / jnp.float64(N)
    t_over_M = jnp.float64(jnp.pi * N) / jnp.float64(M)   # pi*N/M
    phase_arg = jnp.complex128(1j) * t_over_M * f           # i*(pi*N/M)*f
    log_diag  = jnp.log1p(p * jnp.expm1(phase_arg))        # log(1 + p*expm1(...))
    diag      = jnp.exp(jnp.float64(M) * log_diag)
    return diag, M


def q_oracle_sketch_boolean_adaptive(
    truth_table: jax.Array,
    unit_num_samples: int,
    pilot_frac: float = 0.1,
    key: jax.Array | None = None,
) -> tuple[jax.Array, int, jax.Array]:
    """Adaptive importance-sampled Boolean oracle diagonal (Marena 2026).

    The Zhao et al. formula generalises to any probability distribution p:

        diag[x] = exp(M * log(1 + p(x) * expm1(i * t(x) / M * f(x))))

    where t(x) = pi / p(x) so that p(x)*t(x) = pi for all x in supp(f).

    Adaptive choice: p(x) = q(x) = importance weight concentrated on supp(f).
    Then t(x) = pi / q(x) per entry.  Per-step angle = t(x)/M = pi/(q(x)*M).

    Equivalently, setting t_global = pi * K_hat and distributing:
        q(x) * t_global / M_main = q(x) * pi * K_hat / M_main
    For perfect pilot (q=1/K, K_hat=K): = pi/M_main.  Correct.

    The log-sum formula is::

        phase_step(x) = pi * K_hat / M_main          (scalar, same for all supp entries
                                                       when q is uniform on supp)
        diag[x] = exp(M_main * log(1 + q(x) * expm1(i * phase_step * f(x))))

    Note: q(x) must be INSIDE the log1p, not multiplied onto the exponent.
    The formula log1p(expm1(z)) = z is a tautology and gives no accumulation.

    Off-support: q(x)=0 => log1p(0)=0 => exp(0)=1. Exact.
    On-support (perfect pilot): q(x)*expm1(i*pi/M_main) accumulated M_main
    times gives exp(i*pi) = -1.

    Two-phase algorithm
    -------------------
    Phase 1 (Pilot, M_pilot samples):
      - Draw uniform indices, count hits per x.
      - Laplace-smooth restricted to supp(f): q(x) = (count+1)/Z * f(x)
      - Estimate K_hat from pilot hit rate, bias-corrected.

    Phase 2 (Main, M_main samples):
      - phase_step = pi * K_hat / M_main
      - diag[x] = exp(M_main * log1p(q(x) * expm1(1j * phase_step * f(x))))

    Sample complexity: O(K/eps^2), N/K improvement over Zhao et al. O(N/eps^2).
    """
    N = int(truth_table.shape[0])
    if key is None:
        key = random.PRNGKey(0)

    M_pilot = int(float(pilot_frac) * unit_num_samples)
    M_main  = max(unit_num_samples - M_pilot, 1)

    f         = truth_table.astype(jnp.float64)
    uniform_q = jnp.ones((N,), dtype=jnp.float64) / jnp.float64(N)
    K_true    = float(jnp.sum(f))

    if M_pilot == 0 or K_true == 0.0 or K_true == float(N):
        diag, _ = q_oracle_sketch_boolean(truth_table, unit_num_samples)
        return diag, int(unit_num_samples), uniform_q

    # ------------------------------------------------------------------
    # Phase 1: Pilot -- estimate support distribution
    # ------------------------------------------------------------------
    key, pkey = random.split(key)
    idx    = random.randint(pkey, shape=(M_pilot,), minval=0, maxval=N)
    counts = jnp.bincount(idx, length=N).astype(jnp.float64)

    # Laplace-smooth importance weights on supp(f) only
    raw = (counts + jnp.float64(1.0)) * f
    Z   = jnp.sum(raw)
    q   = raw / Z                       # float64, sums to 1 on supp(f)

    # Bias-corrected K_hat: subtract Laplace pseudo-counts
    # raw_hits = sum of actual counts on support (before +1 smoothing)
    pilot_hits = jnp.sum(counts * f)    # raw hits on supp(f)
    # Unbiased estimate: K_hat = pilot_hits / M_pilot * N
    # Clip to [1, N] to prevent degenerate angles
    K_hat = float(jnp.clip(
        pilot_hits / jnp.float64(M_pilot) * jnp.float64(N),
        jnp.float64(1.0),
        jnp.float64(float(N))
    ))

    # ------------------------------------------------------------------
    # Phase 2: Main oracle -- Zhao log-sum formula with p = q
    # ------------------------------------------------------------------
    # Per-step angle (scalar): pi * K_hat / M_main
    # For perfect pilot: q(x) = 1/K, K_hat = K
    #   => M_main * log1p(q(x) * expm1(i * pi*K/M_main))
    #   => M_main * log1p((1/K) * expm1(i * pi*K/M_main))
    #   -> exp(i * (1/K) * pi*K) = exp(i*pi) = -1 as M_main -> inf. Correct.
    #
    # KEY: q(x) must be INSIDE log1p, just like p is inside log1p in Zhao.
    # This is the log-sum (expected-unitary) formula, not a direct formula.
    phase_step = jnp.float64(jnp.pi) * jnp.float64(K_hat) / jnp.float64(M_main)
    # phase_arg[x] = i * phase_step * f(x)  (same for all supp entries if q were omitted,
    # but q IS the weight inside log1p, not an angle modifier)
    inner     = q * jnp.expm1(jnp.complex128(1j) * phase_step * f)  # q(x)*expm1(i*ps*f)
    log_term  = jnp.log1p(inner)                                      # log(1 + q*expm1(...))
    diag      = jnp.exp(jnp.float64(M_main) * log_term)

    return diag, int(unit_num_samples), q


# ---------------------------------------------------------------------------
# Matrix oracle APIs (unchanged)
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
