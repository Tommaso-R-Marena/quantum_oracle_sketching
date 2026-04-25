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
    """Construct the expected Boolean phase-oracle diagonal.

    Args:
        truth_table: Boolean values in {0,1} with shape ``(dim,)``.
        unit_num_samples: Number of sketching samples ``M``.

    Returns:
        Tuple ``(diag, M)`` where ``diag`` has shape ``(dim,)`` and approximates
        ``(-1)^{f(x)}`` by expected channel accumulation.

    Mathematical note:
        Zhao et al. 2026, Theorem D.12 gives IID sample complexity scaling for
        this construction using uniform sampling.  Phase time t = pi * N so that
        each entry accumulates phase p(x)*t = (1/N)*(pi*N) = pi.
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
    """Compute importance weights from pilot samples (eager, no lax.cond)."""
    idx = random.choice(key, jnp.arange(dim, dtype=int_dtype), shape=(pilot_samples,), replace=True)
    counts = jnp.bincount(idx, length=dim).astype(real_dtype)
    weighted_counts = counts * support
    # Laplace smoothing over support ensures all support positions get nonzero weight
    # even if none were sampled in the pilot phase.
    smoothed = weighted_counts + support
    total = jnp.sum(smoothed)
    return jnp.where(total > 0, smoothed / total, uniform_prob)


def q_oracle_sketch_boolean_adaptive(
    truth_table: jax.Array,
    unit_num_samples: int,
    pilot_frac: float = 0.1,
    key: jax.Array | None = None,
) -> tuple[jax.Array, int, jax.Array]:
    """Construct an adaptive importance-sampled Boolean phase-oracle diagonal.

    The method estimates support weights in a pilot phase, then runs a weighted
    expected-unitary accumulation to reduce effective ``p_max`` for sparse
    functions.

    Phase time choice:
        The adaptive estimator uses ``t = pi * K`` where ``K = |supp(f)|``.
        With importance weights ``p(x) = 1/K`` on ``supp(f)``, each support
        entry accumulates phase ``p(x)*t = (1/K)*(pi*K) = pi``, matching the
        target ``(-1)^{f(x)}``.  This gives error scaling
        ``eps ~ sqrt(K * pi^2 / M_main)``.

        The blind-oracle improvement factor relative to uniform sampling is
        ``N/K``: uniform needs ``M = O(N * pi^2 / eps^2)`` while this
        construction needs ``M_main = O(K * pi^2 / eps^2)``.

        Crossover (K=16, eps=0.3): ``M_main ~ K*pi^2/eps^2 ~ 1750``,
        i.e. ``M_total ~ 1950`` with ``pilot_frac=0.1``.

    Implementation note:
        This function deliberately avoids ``jax.lax.cond`` for the pilot-weight
        computation.  ``lax.cond`` traces both branches and can behave
        unexpectedly when one branch contains ``random.choice``; the resulting
        importance weights silently converge to a wrong distribution as
        ``pilot_samples`` grows.  Plain Python ``if``/``else`` is correct here
        because ``pilot_samples`` and ``support_sum`` are concrete (eager)
        values at call time.

    Args:
        truth_table: Boolean values in {0,1} with shape ``(dim,)``.
        unit_num_samples: Total number of sketching samples ``M``.
        pilot_frac: Fraction of samples used in pilot estimation, in ``[0, 1]``.
            Keep small (default 0.1) so most samples go to the main phase.
        key: Optional JAX PRNG key for pilot sampling.

    Returns:
        Tuple ``(diag, M, importance_weights)`` where ``diag`` has shape ``(dim,)``,
        and ``importance_weights`` is an empirical distribution of shape ``(dim,)``.

    Mathematical note:
        **Theorem (Adaptive Boolean Oracle, concentrated-support model).**
        Let ``f:{0,1}^n -> {0,1}`` with support size ``K``.  Using
        pilot-estimated importance weights ``p(x) ~ 1/K`` on ``supp(f)``,
        the sketch achieves error ``eps`` with
        ``M_main = O(K * pi^2 / eps^2)`` main samples.
        Total samples ``M = M_main / (1 - pilot_frac)``.
        Relative to uniform sampling (``M = O(N * pi^2 / eps^2)``), this
        improves sample complexity by factor ``N/K``.
    """
    dim = truth_table.shape[0]
    if key is None:
        key = random.PRNGKey(0)

    pilot_samples = int(jnp.floor(jnp.clip(pilot_frac, 0.0, 1.0) * unit_num_samples))
    main_samples = max(unit_num_samples - pilot_samples, 1)

    uniform_prob = jnp.ones((dim,), dtype=real_dtype) / dim
    support = truth_table.astype(real_dtype)
    support_sum = float(jnp.sum(support))

    # Compute importance weights with plain Python if/else (not lax.cond).
    # This avoids lax.cond tracing both branches, which causes random.choice
    # inside the pilot branch to return wrong values as pilot_samples grows.
    if pilot_samples > 0 and support_sum > 0:
        importance_weights = _compute_pilot_weights(
            key, pilot_samples, dim, support, uniform_prob
        )
    elif support_sum > 0:
        importance_weights = support / support_sum
    else:
        importance_weights = uniform_prob

    # Fall back to uniform oracle when all entries are support (dense case)
    # or pilot is disabled.
    if pilot_samples == 0 or support_sum == float(dim):
        diag, _ = q_oracle_sketch_boolean(truth_table, unit_num_samples)
        return diag, int(unit_num_samples), importance_weights

    # Adaptive phase time: t = pi * K so that p(x)*t = pi for x in supp(f).
    # (p(x) = 1/K after pilot concentration, so (1/K)*(pi*K) = pi.)
    k_hat = max(support_sum, 1.0)
    t = jnp.pi * k_hat
    phase = truth_table.astype(real_dtype)
    log_diag = jnp.log1p(importance_weights * jnp.expm1(1j * t / main_samples * phase))
    diag = jnp.exp(main_samples * log_diag)
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
