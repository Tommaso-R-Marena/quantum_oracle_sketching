"""Quantum state sketching: prepare quantum states from classical vector samples.

This module implements state sketching primitives for benchmarking.

For general vectors, ``q_state_sketch`` uses an explicit finite-sample Monte Carlo
phase-oracle construction so the reconstruction error properly scales with the
input sample budget ``unit_num_samples``.

All functions are JAX-transformable (jit/vmap-friendly) and use 64-bit precision
by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import random

from qos.config import DEFAULT_CONFIG, complex_dtype, int_dtype, real_dtype
from qos.utils.numerical import bitwise_parity_matrix, fwht, unnormalized_hadamard_transform

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

    Uses a finite-sample random-channel phase oracle (Monte Carlo over sampled
    coordinates) so output quality depends on ``unit_num_samples``. The QSVT
    arcsin path then inverts the sine nonlinearity to recover the vector.

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

    Mathematical outline
    --------------------
    1.  Pad ``vector`` to power-of-2 dimension ``D``.
    2.  Random-sign scrambling: ``sv = signs * vector``.
    3.  Build phase oracle diagonal: ``diag[u] = exp(i * B[u])`` where
        ``B[u] = t * (H @ sv)[u] / D`` and ``t = pi/4`` (keeps |B| < pi/2).
    4.  LCU block encoding (unitary!)::

            U_be = [[sin(B),   i*cos(B)],
                    [i*cos(B),  sin(B) ]]

        Top-left block encodes ``sin(B)``.
    5.  QSVT (parity=1): approximates ``arcsin(x)/(pi/2)`` on ``[-1,1]``.
        Output block ``[0,0]`` encodes ``(2/pi)*arcsin(sin(B)) = (2/pi)*B``
        when ``|B| < pi/2``.
    6.  Take ``imag(block[0,0])`` -- the imaginary part carries the signal
        since B is real: ``(2/pi)*B[u] = (2/pi)*t*(H@sv)[u]/D``.
    7.  Inverse WHT + undo sign scrambling::

            v_recon = signs * fwht(qsvt_out) / D  /  (t * 2/pi / D)
                    = signs * fwht(qsvt_out) / (t * 2/pi)

    8.  Truncate to orig_dim.
    """
    from qos.qsvt.angles import get_qsvt_angles
    from qos.qsvt.transform import apply_qsvt_diag

    orig_dim = vector.shape[0]
    dim = int(2 ** jnp.ceil(jnp.log2(jnp.maximum(orig_dim, 2))))

    # Pad to power of 2.
    vector_padded = jnp.pad(
        vector.astype(real_dtype),
        (0, dim - orig_dim),
        mode="constant",
        constant_values=0.0,
    )
    # Use jnp.linalg.norm directly — do NOT call float() inside vmap/jit.
    norm = jnp.linalg.norm(vector_padded)
    # Safe normalize: if norm==0 return zeros (jnp.where keeps vmap-compatible).
    vector_padded = jnp.where(norm > 0, vector_padded / norm, vector_padded)

    # Random sign scrambling to spread energy across WHT frequencies.
    key, subkey = random.split(key)
    random_signs = random.choice(
        subkey, jnp.array([1, -1], dtype=int_dtype), shape=(dim,)
    ).astype(real_dtype)
    sv = random_signs * vector_padded  # scrambled vector

    t = jnp.pi / 4.0

    # Finite-sample random channel: sample indices ~ Uniform([0, dim)).
    # This preserves the intended M-dependent concentration behavior.
    counts = random.multinomial(key, n=unit_num_samples, p=jnp.ones(dim) / dim)
    inner_prod_signs = bitwise_parity_matrix(dim).astype(real_dtype)  # (-1)^{popcount(j&u)}
    weighted_sv = counts.astype(real_dtype) * sv
    phase = (t / (unit_num_samples * dim)) * (weighted_sv @ inner_prod_signs)
    diag = jnp.exp(1j * phase)

    # Unitary LCU block encoding of sin(B):
    sin_b = jnp.imag(diag)   # sin(B)
    cos_b = jnp.real(diag)   # cos(B)
    block_encoding = jnp.stack(
        [sin_b, 1j * cos_b, 1j * cos_b, sin_b], axis=0
    ).reshape(2, 2, dim).astype(complex_dtype)

    # QSVT angles for arcsin(x)/(pi/2), parity=1 (odd polynomial).
    if angle_set is None:
        angle_set = get_qsvt_angles(
            func=lambda x: jnp.arcsin(x) / (jnp.pi / 2),
            degree=degree,
            rescale=1.0,
            cheb_domain=(-1.0, 1.0),
            ensure_bounded=False,
            parity=1,
        )

    block_out = apply_qsvt_diag(block_encoding, num_ancilla=1, angle_set=angle_set)
    qsvt_out = jnp.imag(block_out[0, 0]).astype(real_dtype)  # (2/pi)*B[u], shape (dim,)

    # Inverse WHT and undo sign scrambling + t*(2/pi) scale.
    inv_wht = fwht(qsvt_out)  # ~ (2/pi) * t * sv, shape (dim,)
    state = -random_signs * inv_wht / (t * (2.0 / jnp.pi))
    # Re-apply original norm so the output lives in the same space as input.
    state = state[:orig_dim].astype(real_dtype) * norm

    total_samples = int(unit_num_samples) * int(angle_set.shape[0] - 1)
    return state, total_samples


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
