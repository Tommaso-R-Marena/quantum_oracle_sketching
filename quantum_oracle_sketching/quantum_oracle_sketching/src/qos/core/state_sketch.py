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
    vector: jnp.ndarray,
    unit_num_samples: int,
) -> tuple[jnp.ndarray, int]:
    """Construct the quantum state sketch of a flat (±1) vector.

    Uses 1 ancilla qubit. The sketch is a diagonal phase gate applied to the
    uniform superposition |+^n>, yielding the state vector proportional to
    the input flat vector.

    Args:
        vector: Input flat vector of shape ``(dim,)`` with entries ±1.
        unit_num_samples: Number of effective samples in the sketch.

    Returns:
        ``(state, num_samples)`` where ``state`` has shape ``(dim,)`` and
        ``num_samples == unit_num_samples``.

    Example:
        >>> v = jnp.array([1, -1, 1, -1])
        >>> state, m = q_state_sketch_flat(v, 100_000)
        >>> float(jnp.linalg.norm(state))  # ≈ 1
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
    vector: jnp.ndarray,
    key: jax_random.PRNGKeyArray,
    unit_num_samples: int,
    angle_set: jnp.ndarray | None = None,
    degree: int = DEFAULT_CONFIG.arcsin_degree,
) -> tuple[jnp.ndarray, int]:
    """Construct the quantum state sketch of a general real vector.

    Uses 2 ancilla qubits:
        1. LCU + QSVT for arcsin inversion.
        2. Second LCU to extract the real part.

    The dimension is padded to the next power of 2 internally to support the
    Walsh-Hadamard randomization.

    Args:
        vector: Input vector of shape ``(dim,)``; will be L2-normalized internally.
        key: JAX PRNGKey for random sign randomization.
        unit_num_samples: Number of effective samples per elementary gate.
        angle_set: Pre-computed QSVT angles. If ``None``, they are generated
            internally (adds overhead).
        degree: Even polynomial degree for the arcsin(x) / arcsin(1) approximation.

    Returns:
        ``(state, total_samples)`` where ``state`` has shape ``(dim,)`` and
        ``total_samples`` accounts for QSVT multiplicative overhead.

    Mathematical outline:
        1. Pad ``vector`` to power-of-2 dimension ``D``.
        2. Random-sign Walsh-Hadamard transform (Johnson-Lindenstrauss style).
        3. Construct expected phase oracle  ``U = diag(exp(i B))``.
        4. LCU: ``sin(B) = (U - U†)/(2i)``.
        5. QSVT: apply polynomial approximation to ``arcsin(x) / arcsin(1)``.
        6. Extract real part via second LCU.
        7. Inverse Walsh-Hadamard + sign unrandomization.
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

    # Random sign O_h
    key, subkey = random.split(key)
    random_signs = random.choice(
        subkey, jnp.array([1, -1], dtype=int_dtype), shape=(dim,)
    )

    # Bitwise interaction matrix (-1)^(j . u)
    inner_prod_signs = bitwise_parity_matrix(dim)

    # Expected single gate with stable log1p accumulation.
    t = orig_dim / norm / DEFAULT_CONFIG.general_vector_time_scale
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

    # LCU: convert phase to sine.
    sin = (diag - jnp.conj(diag)) / (2j)
    cos = (diag + jnp.conj(diag)) / 2
    block_encoding = jnp.stack([sin, cos, cos, -sin], axis=0).reshape(2, 2, dim)

    # QSVT to approximate arcsin(x) / arcsin(1)
    if angle_set is None:
        angle_set = get_qsvt_angles(
            func=lambda x: jnp.arcsin(x) / jnp.arcsin(1.0),
            degree=degree,
            rescale=1.0,
            cheb_domain=(-jnp.sin(1.0), jnp.sin(1.0)),
            ensure_bounded=False,
            parity=0,
        )

    from qos.qsvt.transform import apply_qsvt_diag

    block_encoding = apply_qsvt_diag(block_encoding, num_ancilla=1, angle_set=angle_set)

    # Extract real part and prepare state.
    state = jnp.real(block_encoding[0, 0]) / jnp.sqrt(dim)

    # Inverse randomized Hadamard transform.
    hadamard = unnormalized_hadamard_transform(int(jnp.round(jnp.log2(dim))))
    state = hadamard @ state / jnp.sqrt(dim)
    state = random_signs * state
    state = state[:orig_dim]

    total_samples = unit_num_samples * (angle_set.shape[0] - 1)
    return state, int(total_samples)
