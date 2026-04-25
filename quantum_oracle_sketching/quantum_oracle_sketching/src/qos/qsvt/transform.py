"""QSVT application: transform block-encoded matrices via quantum signal processing.

Provides efficient implementations for:
    - Dense Hermitian unitary block encodings.
    - Diagonal-block-structured unitaries (exploiting direct-sum structure).
    - Imperfect (noisy) gate sequences.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from qos.config import complex_dtype, real_dtype


def apply_qsvt(
    U: jnp.ndarray,
    num_ancilla: int,
    angle_set: jnp.ndarray,
) -> jnp.ndarray:
    """Apply QSVT to a Hermitian unitary block encoding.

    Args:
        U: Hermitian unitary of shape ``(dim * 2^num_ancilla, dim * 2^num_ancilla)``.
        num_ancilla: Number of ancilla qubits.
        angle_set: QSVT angles.

    Returns:
        The QSVT-transformed unitary.
    """
    dim = U.shape[-1] // (2**num_ancilla)

    mask = jnp.concatenate([jnp.array([1.0]), -jnp.ones((2**num_ancilla) - 1)])
    qsp_op_phase_pattern = jnp.repeat(mask, dim).astype(real_dtype)

    # Global phase to turn imaginary blocks into real polynomials.
    circ = jnp.exp(1j * (-jnp.pi / 2) * (angle_set.shape[0])) * jnp.diag(
        jnp.exp(1j * angle_set[0] * qsp_op_phase_pattern)
    )

    for angle in angle_set[1:]:
        circ = circ @ U * jnp.exp(1j * angle * qsp_op_phase_pattern)[None, :]

    return circ


apply_qsvt_vectorized = jax.vmap(apply_qsvt, in_axes=(0, None, None))


def apply_qsvt_diag(
    U: jnp.ndarray,
    num_ancilla: int,
    angle_set: jnp.ndarray,
) -> jnp.ndarray:
    """Apply QSVT to a Hermitian unitary with diagonal block structure.

    Args:
        U: Array of shape ``(2^num_ancilla, 2^num_ancilla, dim)`` representing
            a block-diagonal unitary with diagonal blocks.
        num_ancilla: Number of ancilla qubits.
        angle_set: QSVT angles.

    Returns:
        QSVT-transformed unitary of the same shape.
    """
    U = U.transpose((2, 0, 1))
    U = apply_qsvt_vectorized(U, num_ancilla, angle_set)
    U = U.transpose((1, 2, 0))
    return U


def apply_qsvt_imperfect(
    U_sequence: jnp.ndarray,
    num_ancilla: int,
    angle_set: jnp.ndarray,
) -> jnp.ndarray:
    """Apply QSVT to a sequence of imperfect unitary implementations.

    Args:
        U_sequence: Shape ``(num_gates, dim * 2^num_ancilla, dim * 2^num_ancilla)``.
        num_ancilla: Number of ancilla qubits.
        angle_set: Shape ``(num_gates + 1,)``.

    Returns:
        The QSVT circuit unitary.
    """
    dim = U_sequence.shape[-1] // (2**num_ancilla)

    if U_sequence.shape[0] != angle_set.shape[0] - 1:
        raise ValueError(
            f"Number of imperfect gates ({U_sequence.shape[0]}) must match "
            f"number of angles minus one ({angle_set.shape[0] - 1})."
        )

    mask = jnp.concatenate([jnp.array([1.0]), -jnp.ones((2**num_ancilla) - 1)])
    qsp_op_phase_pattern = jnp.repeat(mask, dim).astype(real_dtype)

    circ = jnp.exp(1j * (-jnp.pi / 2) * (angle_set.shape[0])) * jnp.diag(
        jnp.exp(1j * angle_set[0] * qsp_op_phase_pattern)
    )

    for angle, U in zip(angle_set[1:], U_sequence):
        circ = circ @ U * jnp.exp(1j * angle * qsp_op_phase_pattern)[None, :]

    return circ


apply_qsvt_imperfect_vectorized = jax.vmap(
    apply_qsvt_imperfect, in_axes=(0, None, None)
)


@partial(jax.jit, static_argnums=(1,))
def apply_qsvt_imperfect_diag(
    U_sequence: jnp.ndarray,
    num_ancilla: int,
    angle_set: jnp.ndarray,
) -> jnp.ndarray:
    """Apply QSVT to a sequence of imperfect diagonal-block unitaries.

    Args:
        U_sequence: Shape ``(num_gates, 2^num_ancilla, 2^num_ancilla, dim)``.
        num_ancilla: Number of ancilla qubits.
        angle_set: QSVT angles.

    Returns:
        QSVT-transformed unitary of shape ``(2^num_ancilla, 2^num_ancilla, dim)``.
    """
    U_sequence = U_sequence.transpose((3, 0, 1, 2))
    U = apply_qsvt_imperfect_vectorized(U_sequence, num_ancilla, angle_set)
    U = U.transpose((1, 2, 0))
    return U
