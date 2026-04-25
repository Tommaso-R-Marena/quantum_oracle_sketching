"""Quantum primitives: amplitude amplification and related utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from qos.config import DEFAULT_CONFIG, complex_dtype, real_dtype
from qos.qsvt.angles import get_qsvt_angles_sign
from qos.qsvt.transform import apply_qsvt, apply_qsvt_imperfect
from qos.utils.numerical import get_block_encoded, halmos_dilation

if TYPE_CHECKING:
    from jax import random as jax_random


def amplitude_amplification(
    unnormalized_state: jnp.ndarray,
    degree: int = DEFAULT_CONFIG.amplitude_amplification_degree,
    target_norm: float = DEFAULT_CONFIG.amplitude_amplification_target_norm,
    threshold: float | None = None,
) -> jnp.ndarray:
    """Amplify the amplitude of an unnormalized quantum state using QSVT.

    Implicitly increases the ancilla count by 1. The garbage blocks from the
    block encoding are ignored because QOS+QSVT have a fixed circuit structure
    where only the data are random; the garbage is always the canonical Halmos
    dilation up to a data-independent unitary.

    Args:
        unnormalized_state: Shape ``(dim,)`` or ``(degree, dim)`` for imperfect
            instantiations.
        degree: Odd polynomial degree for the sign-function QSVT.
        target_norm: Desired norm after amplification.
        threshold: Sign threshold. Defaults to half the minimum norm, clamped
            to at least ``1e-3``.

    Returns:
        Amplified state vector of shape ``(dim,)``.

    Raises:
        ValueError: If the input state has zero norm.
    """
    imperfect = len(unnormalized_state.shape) == 2

    if imperfect:
        norms = jnp.sqrt(jnp.sum(jnp.abs(unnormalized_state) ** 2, axis=-1))
        norm = jnp.min(norms)
    else:
        norm = jnp.linalg.norm(unnormalized_state)

    if norm == 0:
        raise ValueError("Input state has zero norm and cannot be amplified.")

    if threshold is None:
        threshold = max(float(norm) * 0.5, 1e-3)

    angle_set, scale = get_qsvt_angles_sign(
        degree=degree, threshold=float(threshold), rescale=target_norm
    )

    def _embed(state: jnp.ndarray) -> jnp.ndarray:
        dim = state.shape[-1]
        hermitian_embed = jnp.block(
            [
                [jnp.zeros((dim, dim)), state[:, None]],
                [state[None, :].conj(), jnp.zeros((1, 1))],
            ]
        )
        return halmos_dilation(hermitian_embed)

    halmos_block_encoding = (
        jax.vmap(_embed)(unnormalized_state)
        if imperfect
        else _embed(unnormalized_state)
    )

    if imperfect:
        transformed_block_encoding = apply_qsvt_imperfect(
            halmos_block_encoding, num_ancilla=1, angle_set=angle_set
        )
    else:
        transformed_block_encoding = apply_qsvt(
            halmos_block_encoding, num_ancilla=1, angle_set=angle_set
        )

    transformed_matrix = get_block_encoded(transformed_block_encoding, num_ancilla=1)
    transformed_matrix = (transformed_matrix + transformed_matrix.conj().T) / 2

    transformed_state = transformed_matrix[:-1, -1]
    return transformed_state
