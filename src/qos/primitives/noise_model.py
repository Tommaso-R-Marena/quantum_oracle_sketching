"""Depolarizing noise model for post-sketching quantum circuit simulation.

# Copyright (c) 2026 Tommaso R. Marena. MIT License.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = [
    "DepolarizingChannel",
    "compose_sketch_and_noise_error",
    "crossover_sample_count",
]


@dataclass
class DepolarizingChannel:
    """Independent single-qubit depolarizing channel approximation.

    Args:
        num_qubits: Number of qubits in the target register.
        noise_rate: Per-gate, per-qubit depolarizing parameter ``η``.
        seed: Stored seed value for reproducibility metadata.
    """

    num_qubits: int
    noise_rate: float
    seed: int = 42

    def apply_to_diagonal(self, diag: jax.Array) -> jax.Array:
        """Apply depolarizing shrinkage to a phase-diagonal unitary.

        Args:
            diag: Complex phase diagonal with shape ``(2**num_qubits,)``.

        Returns:
            Noisy diagonal with same shape.

        Mathematical note:
            Under Pauli twirling, coherences shrink by ``(1-4η/3)`` per qubit.
        """
        eta = jnp.clip(self.noise_rate, 0.0, 1.0)
        shrink = jnp.maximum(0.0, 1.0 - 4.0 * eta / 3.0) ** self.num_qubits
        return diag * shrink

    def apply_to_block_encoding(self, be: jax.Array, num_ancilla: int) -> jax.Array:
        """Apply depolarizing shrinkage to a diagonal block encoding.

        Args:
            be: Block encoding tensor with shape ``(2, 2, dim)``.
            num_ancilla: Number of ancilla qubits in the full encoding.

        Returns:
            Noisy block encoding tensor with the same shape.

        Mathematical note:
            First-order channel composition gives multiplicative attenuation
            across signal+ancilla registers.
        """
        total_q = self.num_qubits + int(num_ancilla)
        eta = jnp.clip(self.noise_rate, 0.0, 1.0)
        shrink = jnp.maximum(0.0, 1.0 - 4.0 * eta / 3.0) ** total_q
        return be * shrink

    def diamond_norm_degradation(self, circuit_depth: int) -> float:
        """Compute first-order diamond-norm noise upper bound.

        Args:
            circuit_depth: Number of noisy gate layers.

        Returns:
            Upper bound ``depth * num_qubits * η`` as float.
        """
        return float(circuit_depth * self.num_qubits * self.noise_rate)


def compose_sketch_and_noise_error(
    sketch_error: float,
    noise_rate: float,
    circuit_depth: int,
    num_qubits: int,
) -> float:
    """Compose sketching and depolarizing errors by triangle inequality.

    Args:
        sketch_error: Sketch-only approximation error ``ε_sketch``.
        noise_rate: Depolarizing rate ``η``.
        circuit_depth: Number of layers.
        num_qubits: Number of qubits.

    Returns:
        Total error upper bound ``ε_sketch + min(2, depth*num_qubits*η)``.
    """
    eps_noise = min(2.0, float(circuit_depth * num_qubits * noise_rate))
    return float(sketch_error + eps_noise)


def crossover_sample_count(
    dim: int,
    noise_rate: float,
    circuit_depth: int,
    epsilon_target: float,
) -> int:
    """Find minimal sample count where sketching error meets residual budget.

    Args:
        dim: Problem dimension ``N``.
        noise_rate: Depolarizing rate ``η``.
        circuit_depth: Circuit depth ``d``.
        epsilon_target: Desired total error tolerance.

    Returns:
        Minimum integer ``M*`` satisfying ``N/sqrt(M*) <= ε_target-ε_noise``.
    """
    num_qubits = int(jnp.log2(dim))
    eps_noise = min(2.0, float(circuit_depth * num_qubits * noise_rate))
    budget = epsilon_target - eps_noise
    if budget <= 0:
        return 1
    m_star = jnp.ceil((dim / budget) ** 2)
    return int(jnp.maximum(1, m_star))
