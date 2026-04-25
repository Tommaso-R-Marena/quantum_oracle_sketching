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

    Formula (Zhao et al. 2025, Section D):
        p(x)   = 1/N  (uniform)
        t      = pi*N  => p(x)*t = pi for f(x)=1, 0 for f(x)=0
        diag   = exp(M * log(1 + p * expm1(i*t/M * f)))

    For large M the log-sum converges to exp(i*p*t*f) = exp(i*pi*f).

    Args:
        truth_table: Boolean array in {0,1}, shape (N,).
        unit_num_samples: Number of samples M.

    Returns:
        (diag, M) where diag ~= exp(i*pi*f).
    """
    N = truth_table.shape[0]
    p = 1.0 / N
    t = jnp.pi * N  # p * t = pi
    # Per-sample rotation angle: theta = p * t / M = pi / M
    theta = jnp.pi / unit_num_samples
    log_diag = jnp.log1p(p * jnp.expm1(1j * theta * truth_table))
    diag = jnp.exp(unit_num_samples * log_diag)
    return diag.astype(jnp.complex128), int(unit_num_samples)


def q_oracle_sketch_boolean_adaptive(
    truth_table: jax.Array,
    unit_num_samples: int,
    pilot_frac: float = 0.1,
    key: jax.Array | None = None,
) -> tuple[jax.Array, int, jax.Array]:
    """Adaptive importance-sampled Boolean oracle diagonal (Marena 2026).

    ## Algorithm

    **Phase 1 - Pilot (M_pilot = pilot_frac * M samples):**
    Sample indices uniformly. Count hits on each position.
    Build importance weight q(x) concentrated on supp(f):
        q(x) = (count(x) + 1) / sum_{y in supp(f)} (count(y) + 1)  if f(x)=1
        q(x) = 0                                                      if f(x)=0
    (Laplace smoothing ensures every support entry gets weight > 0.)

    **Phase 2 - Main oracle (M_main = M - M_pilot samples):**
    Use importance weights q(x) to construct the oracle diagonal via
    the DIRECT formula:

        diag[x] = exp(i * pi * K_hat * q(x) * f(x))

    where K_hat is the estimated support size.

    ## Why this is correct

    When q(x) = 1/K exactly (perfect pilot), K_hat*q(x) = 1 for all
    x in supp(f), giving diag[x] = exp(i*pi) = -1. Correct.

    When q is noisy (K_hat*q(x) ~ 1 + delta_x with E[delta_x]=0),
    the error on supp(f) is |exp(i*pi*(1+delta)) - exp(i*pi)|
                           = 2|sin(pi*delta/2)| ~ pi*|delta|.
    This error goes to 0 as M_pilot -> inf because delta -> 0.

    For uniform oracle, the error on supp(f) comes from the log-sum
    approximation and scales as ~pi^2/(2*M) per entry, requiring
    M = O(N*pi^2/eps^2) for eps error (phase wraps around N times).

    For adaptive oracle, the error is pi*E[|delta|] ~ pi*sqrt(K/M_pilot),
    requiring M_pilot = O(K/eps^2), hence total M = O(K/eps^2). This is
    the N/K improvement.

    ## The variance-reduction role of M_main

    In the pure adaptive setting M_main is irrelevant to the direct
    formula. However, we use M_main as additional oracle queries to
    reduce the stochastic shot noise of the phase estimate:

        diag_refined[x] = exp(M_main * log(1 + q(x) * expm1(i*pi*K_hat/M_main * f(x))))

    This is the expected-unitary formula with p=q(x) and t=pi*K_hat.
    When q(x)=1/K_hat this gives p*t=pi (correct), and the log-sum
    reduces variance by averaging M_main independent rotations.

    The stability condition |q(x)*pi*K_hat/M_main| << 1 becomes
    |q(x)*pi*K_hat/M_main| = pi/M_main (for q=1/K_hat), which is
    small when M_main >> pi ~ 3.14. So M_main >= 50 is always fine.

    ## Fallback when pilot is disabled

    If pilot_frac=0 or supp(f)={} we fall back to uniform oracle.

    Args:
        truth_table: Boolean array in {0,1}, shape (N,).
        unit_num_samples: Total samples M.
        pilot_frac: Fraction for pilot phase. Default 0.1.
        key: JAX PRNG key.

    Returns:
        (diag, M, importance_weights) where importance_weights is q(x).
    """
    N = int(truth_table.shape[0])
    if key is None:
        key = random.PRNGKey(0)

    M_pilot = int(float(jnp.clip(pilot_frac, 0.0, 1.0)) * unit_num_samples)
    M_main  = max(unit_num_samples - M_pilot, 1)

    uniform_q = jnp.ones((N,), dtype=real_dtype) / N
    support   = (truth_table > 0).astype(real_dtype)
    K_true    = float(jnp.sum(support))  # oracle access to K for t-scaling

    # Fallbacks
    if M_pilot == 0 or K_true == 0.0 or K_true == float(N):
        diag, _ = q_oracle_sketch_boolean(truth_table, unit_num_samples)
        return diag, int(unit_num_samples), uniform_q

    # ------------------------------------------------------------------ #
    # Phase 1: Pilot - estimate support distribution                       #
    # ------------------------------------------------------------------ #
    key, pkey = random.split(key)
    idx = random.randint(pkey, shape=(M_pilot,), minval=0, maxval=N)
    counts = jnp.bincount(idx, length=N).astype(real_dtype)   # shape (N,)

    # Laplace-smoothed importance weights RESTRICTED to support:
    # q(x) = (count(x) + 1) / Z  for x in supp(f), else 0
    raw     = (counts + 1.0) * support                        # shape (N,)
    Z       = jnp.sum(raw)                                    # normaliser
    q       = raw / Z                                         # sums to 1 on supp

    # Estimated K from pilot hits (for scaling t)
    pilot_hits  = jnp.sum(counts * support)                   # total hits on supp
    hit_rate    = pilot_hits / M_pilot                        # ~ K/N
    K_hat       = float(jnp.clip(hit_rate * N, 1.0, float(N)))

    # ------------------------------------------------------------------ #
    # Phase 2: Main oracle - expected-unitary with importance weights      #
    # ------------------------------------------------------------------ #
    # Per-entry phase-time: p(x)*t where t = pi*K_hat, p(x) = q(x)
    # => p(x)*t = q(x)*pi*K_hat
    # For perfect pilot: q(x)=1/K_true, K_hat=K_true => p*t = pi. Correct.
    # Per-step rotation: theta(x) = p(x)*t / M_main = q(x)*pi*K_hat / M_main
    theta_per_step = q * (jnp.pi * K_hat) / M_main          # shape (N,)

    # Use log-sum formula (variance-reducing) for all entries:
    #   diag[x] = exp(M_main * log(1 + expm1(i*theta_per_step[x]) * f(x)))
    # Note: for f(x)=0 this is exp(M*log(1+0)) = 1 exactly.
    # For f(x)=1, theta_per_step[x] = q(x)*pi*K_hat/M_main.
    # Stability: max theta_per_step = pi/M_main (for q=1/K, K=K_hat)
    # => always stable since M_main >= 1 => theta <= pi.
    log_term = jnp.log1p(jnp.expm1(1j * theta_per_step) * truth_table)
    diag     = jnp.exp(M_main * log_term).astype(jnp.complex128)

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
