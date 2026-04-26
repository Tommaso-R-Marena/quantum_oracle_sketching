"""Quantum state sketching: prepare quantum states from classical vector samples.

This module implements the expected-unitary (deterministic) QOS path used for
benchmarking. It provides a conservative (pessimistic) upper bound on the error
of the real-world random-channel scenario (see ``qos.core.sampling`` for the
latter).

All functions are JAX-transformable (jit/vmap-friendly) and use 64-bit precision
by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import random

from qos.config import DEFAULT_CONFIG, complex_dtype, int_dtype, real_dtype
from qos.utils.numerical import bitwise_parity_matrix, unnormalized_hadamard_transform

if TYPE_CHECKING:
    from jax import random as jax_random


def q_state_sketch_flat(
    vector: jax.Array,
    unit_num_samples: int,
) -> tuple[jax.Array, int]:
    """Construct the quantum state sketch of a flat (\u00b11) vector.

    Uses 1 ancilla qubit. The sketch is a diagonal phase gate applied to the
    uniform superposition |+^n>, yielding the state vector proportional to
    the input flat vector.

    Args:
        vector: Input flat vector of shape ``(dim,)`` with entries \u00b11.
        unit_num_samples: Number of effective samples in the sketch.

    Returns:
        ``(state, num_samples)`` where ``state`` has shape ``(dim,)`` and
        ``num_samples == unit_num_samples``.

    Example:
        >>> v = jnp.array([1, -1, 1, -1])
        >>> state, m = q_state_sketch_flat(v, 100_000)
        >>> float(jnp.linalg.norm(state))  # \u2248 1
    """
    dim = vector.shape[0]
    prob = jnp.ones_like(vector, dtype=real_dtype) / dim
    t = jnp.pi * dim

    # Expected single gate via log1p for numerical stability.
    log_diag = jnp.log1p(prob * jnp.expm1(1j * t / unit_num_samples * (1 - vector) / 2))
    log_diag = unit_num_samples * log_diag
    diag = jnp.exp(log_diag)

    state = diag / jnp.sqrt(dim)
    return state, int(unit_num_samples)


def q_state_sketch(
    vector: jax.Array,
    key: jax_random.PRNGKeyArray,
    unit_num_samples: int,
    angle_set: jax.Array | None = None,
    degree: int = DEFAULT_CONFIG.arcsin_degree,
) -> tuple[jax.Array, int]:
    """Construct the quantum state sketch of a general real vector.

    Uses 2 ancilla qubits:
        1. LCU + QSVT for arcsin inversion.
        2. Second LCU to extract the imaginary part (odd-parity polynomial).

    The dimension is padded to the next power of 2 internally to support the
    Walsh-Hadamard randomization.

    Args:
        vector: Input vector of shape ``(dim,)``; will be L2-normalized internally.
        key: JAX PRNGKey for random sign randomization.
        unit_num_samples: Number of effective samples per elementary gate.
        angle_set: Pre-computed QSVT angles. If ``None``, they are generated
            internally (adds overhead).
        degree: Polynomial degree for the arcsin(x)/(pi/2) approximation.
            Must be odd (parity=1).

    Returns:
        ``(state, total_samples)`` where ``state`` has shape ``(dim,)`` and
        ``total_samples`` accounts for QSVT multiplicative overhead.

    Mathematical outline:
        1. Pad ``vector`` to power-of-2 dimension ``D``.
        2. Choose t = 1/(norm * general_vector_time_scale) so that
           |t * v_j| << 1 for all j, keeping sin(t*v_j) in the arcsin domain.
        3. Random-sign Walsh-Hadamard: w = diag(signs) @ H @ v  (in freq domain).
        4. Construct expected phase oracle U = diag(exp(i * B)) where
           B_u = sum_j prob_j * exp(i * t * signs_j * v_j * chi(j,u)).
        5. LCU: sin(B) = (U - U^dag)/(2i).
        6. QSVT (parity=1, odd): block[0,0] = i * arcsin(sin(B)) * (2/pi).
           = i * B * (2/pi)  when |B| << pi/2.
           Signal is in imag(block[0,0]).
        7. Inverse Walsh-Hadamard: v_reconstructed = signs * H @ imag / dim.
           Divide by t * (2/pi) to normalize.
        8. Truncate to orig_dim.
    """
    from qos.qsvt.angles import get_qsvt_angles

    orig_dim = vector.shape[0]
    dim = int(2 ** jnp.ceil(jnp.log2(orig_dim)))
    prob = jnp.ones_like(vector, dtype=real_dtype) / orig_dim

    vector = jnp.pad(
        vector, (0, int(dim) - orig_dim), mode="constant", constant_values=0.0
    )
    prob = jnp.pad(prob, (0, int(dim) - orig_dim), mode="constant", constant_values=0.0)
    norm = jnp.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Input vector has zero norm.")

    # t chosen so that |t * v_j| <= 1/(general_vector_time_scale) for all j
    # when vector is unit-norm, keeping sin(t*v_j) well inside [-1, 1].
    t = 1.0 / (float(norm) * DEFAULT_CONFIG.general_vector_time_scale)

    # Random sign O_h: w_u = sum_j (-1)^{popcount(j & u)} * signs_j * v_j
    key, subkey = random.split(key)
    random_signs = random.choice(
        subkey, jnp.array([1, -1], dtype=int_dtype), shape=(dim,)
    )

    # Bitwise interaction matrix (-1)^(j . u), shape (dim, dim)
    inner_prod_signs = bitwise_parity_matrix(dim)

    # Expected single-gate with stable log1p accumulation.
    # phase_arg[j, u] = i * t * signs_j * v_j * chi(j, u)
    # log_diag[u] = log(1 + sum_j prob_j * expm1(phase_arg[j, u]))
    log_diag = jnp.log1p(
        jnp.sum(
            prob[:, None]
            * jnp.expm1(
                1j
                * (random_signs * vector)[:, None]
                * inner_prod_signs
                * t
                / unit_num_samples
            ),
            axis=0,
        )
    )
    log_diag = unit_num_samples * log_diag
    diag = jnp.exp(log_diag)

    # LCU: extract sin(B) from phase oracle.
    sin_b = (diag - jnp.conj(diag)) / (2j)   # real, shape (dim,)
    cos_b = (diag + jnp.conj(diag)) / 2
    block_encoding = jnp.stack([sin_b, cos_b, cos_b, -sin_b], axis=0).reshape(2, 2, dim)

    # QSVT with parity=1 (odd): approximates arcsin(x)/(pi/2) on [-1, 1].
    # Output: block[0, 0] = i * (2/pi) * arcsin(sin_b) ≈ i * (2/pi) * t * w
    # where w = diag(signs) @ H @ v / sqrt(dim) (randomized WHT of v).
    # Use imag(block[0, 0]) = (2/pi) * arcsin(sin_b).
    if angle_set is None:
        angle_set = get_qsvt_angles(
            func=lambda x: jnp.arcsin(x) / (jnp.pi / 2),
            degree=degree,
            rescale=1.0,
            cheb_domain=(-1.0, 1.0),
            ensure_bounded=False,
            parity=1,
        )

    from qos.qsvt.transform import apply_qsvt_diag

    block_encoding = apply_qsvt_diag(block_encoding, num_ancilla=1, angle_set=angle_set)

    # imag(block[0,0]) ≈ (2/pi) * t * (diag(signs) @ H @ v / sqrt(dim))_u
    # = (2/pi) * t * w_u  where w = randomized WHT of v.
    qsvt_out = jnp.imag(block_encoding[0, 0])  # shape (dim,), still in WHT basis

    # Inverse WHT: H @ w = H @ diag(signs) @ H @ v = dim * diag(signs) @ v
    # since H @ H = dim * I.
    # unnormalized_hadamard_transform(n) returns H_{2^n} (without any normalization).
    hadamard = unnormalized_hadamard_transform(int(jnp.round(jnp.log2(dim))))
    inv_wht = hadamard @ qsvt_out   # = dim * diag(signs) @ v (up to t * 2/pi factor)

    # Now: inv_wht ≈ dim * (2/pi) * t * diag(signs) @ v
    # Undo sign randomization and normalization.
    state = random_signs * inv_wht / (dim * t * (2.0 / jnp.pi))
    state = state[:orig_dim]

    total_samples = unit_num_samples * (angle_set.shape[0] - 1)
    return state, int(total_samples)


def q_kernel_estimate(
    state_x: jax.Array,
    state_z: jax.Array,
) -> float:
    """Estimate the quantum kernel value ``K(x,z)=|<psi_x|psi_z>|^2``.

    Args:
        state_x: Prepared state ``|psi(x)>`` with shape ``(dim,)``.
        state_z: Prepared state ``|psi(z)>`` with shape ``(dim,)``.

    Returns:
        Real scalar kernel value in ``[0,1]``.

    Mathematical note:
        Extends Zhao et al. 2026 Theorem F.16 from linear prediction to
        interferometric kernel prediction.
    """
    nx = jnp.linalg.norm(state_x)
    nz = jnp.linalg.norm(state_z)
    denom = jnp.maximum(nx * nz, 1e-12)
    return float(jnp.abs(jnp.vdot(state_x, state_z) / denom) ** 2)


def fit_kernel_svm_from_states(
    train_states: jax.Array,
    train_labels: jax.Array,
    regularization: float = 1e-4,
) -> jax.Array:
    """Fit kernel dual weights from QOS states using Tikhonov-regularized solve.

    Args:
        train_states: Training states of shape ``(n_train, dim)``.
        train_labels: Labels in ``{-1,+1}`` with shape ``(n_train,)``.
        regularization: Ridge/Tikhonov term ``lambda``.

    Returns:
        Dual vector ``alpha`` of shape ``(n_train,)``.

    Mathematical note:
        Solves ``(K + lambda I) alpha = y`` where
        ``K_ij = |<psi_i|psi_j>|^2``.
    """
    gram = jnp.einsum('id,jd->ij', train_states.conj(), train_states)
    kernel = jnp.abs(gram) ** 2
    reg_eye = regularization * jnp.eye(train_states.shape[0], dtype=kernel.dtype)
    return jnp.linalg.solve(kernel + reg_eye, train_labels.astype(kernel.dtype))


def q_interferometric_kernel_shadow(
    train_states: jax.Array,
    train_labels: jax.Array,
    alpha: jax.Array,
    test_state: jax.Array,
    regularization: float = 1e-4,
) -> float:
    """Predict a binary label using an interferometric kernel-shadow rule.

    Args:
        train_states: Array with shape ``(n_train, dim)``.
        train_labels: Label vector with shape ``(n_train,)`` and entries \u00b11.
        alpha: Dual weights with shape ``(n_train,)`` from fit_kernel_svm_from_states.
        test_state: Test state with shape ``(dim,)``.
        regularization: Additive stabilizer applied to the sign of the score.

    Returns:
        Predicted class as ``+1.0`` or ``-1.0``.

    Mathematical note:
        Implements score = sum_i alpha_i * |<psi_i|psi_test>|^2, then returns
        sign(score). Generalizes Zhao et al. 2026 interferometric shadow by
        replacing linear inner products with quantum kernel evaluations
        K(x,z) = |<psi(x)|psi(z)>|^2.
    """
    del train_labels
    overlaps = jnp.einsum('id,d->i', train_states.conj(), test_state)
    kernels = jnp.abs(overlaps) ** 2
    score = jnp.dot(alpha, kernels)
    return float(jnp.sign(score + regularization * jnp.sign(score)))
