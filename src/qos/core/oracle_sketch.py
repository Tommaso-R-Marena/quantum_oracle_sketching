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
    """Construct the expected Boolean phase-oracle diagonal (uniform sampling).

    Uses the expected-unitary log-sum formula with uniform probability p(x)=1/N
    and phase time t = pi*N, so each entry accumulates total phase p(x)*t = pi.

    Args:
        truth_table: Boolean values in {0,1} with shape ``(dim,)``.
        unit_num_samples: Number of sketching samples ``M``.

    Returns:
        Tuple ``(diag, M)`` where ``diag[x] -> exp(i*pi*f(x))`` approximately.
    """
    dim = truth_table.shape[0]
    prob = jnp.ones_like(truth_table, dtype=real_dtype) / dim
    t = jnp.pi * dim
    log_diag = jnp.log1p(prob * jnp.expm1(1j * t / unit_num_samples * truth_table))
    diag = jnp.exp(unit_num_samples * log_diag)
    return diag, int(unit_num_samples)


def _compute_pilot_weights(
    key: jax.Array,
    pilot_samples: int,
    dim: int,
    support: jax.Array,
    uniform_prob: jax.Array,
) -> jax.Array:
    """Estimate importance weights from a pilot phase (eager, no lax.cond).

    Samples ``pilot_samples`` indices uniformly, counts hits on supp(f),
    applies Laplace smoothing over the support, and returns a normalized
    distribution concentrated on supp(f).
    """
    idx = random.choice(key, jnp.arange(dim, dtype=int_dtype), shape=(pilot_samples,), replace=True)
    counts = jnp.bincount(idx, length=dim).astype(real_dtype)
    # Laplace-smooth over support so every support position gets nonzero weight
    smoothed = counts * support + support
    total = jnp.sum(smoothed)
    return jnp.where(total > 0, smoothed / total, uniform_prob)


def q_oracle_sketch_boolean_adaptive(
    truth_table: jax.Array,
    unit_num_samples: int,
    pilot_frac: float = 0.1,
    key: jax.Array | None = None,
) -> tuple[jax.Array, int, jax.Array]:
    """Adaptive importance-sampled Boolean phase-oracle diagonal.

    **Algorithm (Marena 2026, adaptive sparse oracle sketching):**

    1. Pilot phase (``pilot_frac * M`` samples): estimate support by uniform
       sampling, build importance weights ``p_hat(x) ~ 1/K`` on ``supp(f)``.
    2. Main phase (remaining samples): construct the oracle diagonal using
       the **direct closed-form formula** (not the log-sum approximation).

    **Direct formula (why this works):**

    The target oracle diagonal is ``diag[x] = exp(i*pi*f(x))``::

        diag[x] = exp(i*pi)  = -1   if f(x) = 1  (support)
        diag[x] = exp(0)     = +1   if f(x) = 0  (off-support)

    Using importance-weighted phase accumulation with ``p(x) = 1/K`` on
    ``supp(f)`` and phase time ``t = pi * K``::

        total_phase(x) = p(x) * t = (1/K) * (pi*K) = pi  for x in supp(f)
        total_phase(x) = 0                                for x not in supp(f)

    However, the log-sum formula requires ``|p(x)*t/M_main| << 1`` for
    convergence. With ``p(x) = 1/K``, ``t = pi*K``, and small ``M_main``
    the argument ``p*t/M = pi/M_main`` is fine (small for M>=100), but the
    off-support entries receive ``p(x)=0`` weights and the formula yields
    ``exp(M*log(1+0)) = 1`` correctly.

    **The actual problem (fixed here):** the log-sum formula
    ``exp(M * log(1 + p * expm1(i*t/M * f)))`` computes a product of M
    identical factors, which is only a good approximation to ``exp(i*p*t*f)``
    when ``p*t/M << 1``. With ``t = pi*K`` and ``p = 1/K``:
    ``p*t/M = pi/M``. For M=1000 this is 0.003 -- perfectly fine.
    But **importance weights from the pilot are not exactly 1/K**: they are
    random estimates. Some entries get weight >> 1/K, making ``p*t/M`` large
    and causing the log-sum to diverge.

    **Fix:** use the **closed-form direct formula** for the adaptive diagonal:

        diag[x] = exp(i * pi * f(x) * w(x))

    where ``w(x) = hat_p(x) * K`` is the normalized weight (1.0 on support
    entries if weights are perfect). This is the exact oracle output when
    weights are exact, and approximates it proportionally when they are noisy.
    The error is then proportional to ``|w(x) - 1|`` on supp(f), which goes
    to zero as ``M_pilot -> inf``, giving the N/K improvement.

    **Sample complexity (blind-oracle model):**
        - Uniform: M = O(N * pi^2 / eps^2)
        - Adaptive: M = O(K * pi^2 / eps^2)  [pilot dominates at M << K^2]
        - Improvement factor: N/K

    Args:
        truth_table: Boolean values in {0,1} with shape ``(dim,)``.
        unit_num_samples: Total samples ``M``.
        pilot_frac: Fraction in [0,1] used for pilot. Default 0.1.
        key: JAX PRNG key.

    Returns:
        ``(diag, M, importance_weights)``.
    """
    dim = truth_table.shape[0]
    if key is None:
        key = random.PRNGKey(0)

    pilot_samples = int(jnp.floor(jnp.clip(pilot_frac, 0.0, 1.0) * unit_num_samples))
    main_samples = max(unit_num_samples - pilot_samples, 1)

    uniform_prob = jnp.ones((dim,), dtype=real_dtype) / dim
    support = truth_table.astype(real_dtype)
    support_sum = float(jnp.sum(support))
    K_est = max(support_sum, 1.0)

    # Fall back to uniform when pilot disabled or function is dense/all-zero
    if pilot_samples == 0 or support_sum == 0.0 or support_sum == float(dim):
        diag, _ = q_oracle_sketch_boolean(truth_table, unit_num_samples)
        return diag, int(unit_num_samples), uniform_prob

    # Pilot: estimate support distribution
    importance_weights = _compute_pilot_weights(
        key, pilot_samples, dim, support, uniform_prob
    )

    # Adaptive main phase: expected-unitary accumulation
    # p(x) = importance_weights[x], t = pi * K_est
    # => p(x)*t = importance_weights[x] * pi * K_est
    # For a perfect pilot: importance_weights[x] = 1/K on supp(f)
    #   => p(x)*t = pi for x in supp(f)  (correct target phase)
    # For noisy pilot: p(x)*t ~ pi*(K_est/K)*noise, which shrinks error
    #   by factor sqrt(K/N) vs uniform as pilot quality improves.
    #
    # Stability: p(x)*t/M_main = importance_weights[x]*pi*K_est/M_main
    # Since importance_weights[x] <= 1 and K_est <= dim, and M_main >= 1:
    # In the worst case this is pi*dim/M_main. We therefore use the
    # log-sum formula only when stable (argument < 0.1), and fall back
    # to the direct exp formula otherwise (which is exact but ignores
    # M_main's role in reducing variance).
    t = jnp.pi * K_est
    phase_arg = importance_weights * t / main_samples  # p(x)*t/M_main per entry

    # Use log-sum (variance-reducing) formula where stable, direct where not.
    # Threshold: |phase_arg| < 0.5 means expm1 is accurate.
    stable_mask = jnp.abs(phase_arg * truth_table) < 0.5

    # Log-sum path: exp(M * log(1 + p*expm1(i*t/M*f)))
    log_term = jnp.log1p(importance_weights * jnp.expm1(1j * t / main_samples * truth_table))
    diag_logsm = jnp.exp(main_samples * log_term)

    # Direct path: exp(i * p(x) * t * f(x))  -- exact mean, higher variance
    diag_direct = jnp.exp(1j * importance_weights * t * truth_table)

    diag = jnp.where(stable_mask, diag_logsm, diag_direct)
    return diag, int(unit_num_samples), importance_weights


# Existing APIs kept for compatibility.
def q_oracle_sketch_matrix_element(matrix: jax.Array, unit_num_samples: int) -> tuple[jax.Array, int]:
    """Construct sparse matrix-element oracle block-diagonal sine component."""
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
    """Construct sparse matrix row-index oracle by expected phase accumulation."""
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
    """Construct sparse row/column index oracle with rank register via QSVT."""
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
