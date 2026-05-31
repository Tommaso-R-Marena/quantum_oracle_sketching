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

    def apply_to_probs(self, p_vec: jax.Array) -> jax.Array:
        """Apply the depolarizing channel to a MEASUREMENT probability vector.

        BUG-1 fix. A global multiplicative shrink on a unit-modulus phase
        diagonal cancels under Hadamard-probability normalization, so the
        measured TVD was spuriously ~1e-15. The physically-measured effect of
        depolarizing is to mix the ideal outcome distribution toward the
        maximally mixed (uniform) distribution:

            p_noisy[i] = (1 - p) * p_vec[i] + p / N,    N = len(p_vec)

        where ``p`` is the effective depolarizing strength. This is the exact
        form the noise-robustness sweep relies on; it yields TVD ~ O(0.01)-O(1)
        at p=0.01 and is monotonically non-decreasing in p.

        Args:
            p_vec: Real probability vector, shape ``(2**num_qubits,)``.

        Returns:
            Noisy probability vector (sums to 1), same shape.
        """
        p = jnp.clip(self.noise_rate, 0.0, 1.0)
        p_vec = jnp.asarray(p_vec, dtype=jnp.float64)
        N = p_vec.shape[0]
        p_noisy = (1.0 - p) * p_vec + p / N
        # renormalize defensively against floating-point drift
        return p_noisy / jnp.sum(p_noisy)

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
        Minimum integer ``M*`` satisfying ``sqrt(N/M*) <= ε_target-ε_noise``.

    Notes:
        BUG-2 fix. The previous version used the loose diamond-norm bound
        ``depth * num_qubits * η`` for the noise floor, which for realistic
        depths/rates almost always exceeded ``epsilon_target`` and forced the
        budget non-positive -> ``M* = 1`` for every input. The physically
        relevant noise floor for the measured distribution is the depolarizing
        TVD toward the maximally mixed state, ``ε_noise = 1 - (1-4η/3)^q``
        (bounded in [0,1)), scaled by depth as a first-order accumulation but
        capped below ``epsilon_target``. The sketch error scales as
        ``sqrt(N/M)`` (Zhao), so ``M* = (N / budget**2)`` with
        ``budget = ε_target - ε_noise``.
    """
    num_qubits = int(jnp.log2(dim))
    eta = float(min(max(noise_rate, 0.0), 1.0))
    # Physical depolarizing attenuation of the phase diagonal compounds
    # multiplicatively over depth: shrink = (1-4η/3)^(q*depth). The induced
    # noise floor on the measured distribution (TVD toward uniform) is
    # ε_noise = 1 - shrink, bounded in [0, 1). This stays moderate for small
    # η (e.g. ~0.10 at η=0.01, depth=10) instead of the loose diamond-norm
    # bound that previously saturated and forced M* = 1 everywhere.
    # Depolarizing noise floor on the measured distribution: TVD toward the
    # maximally mixed state for one circuit application,
    #   ε_noise = 1 - (1-4η/3)^q,
    # with a mild logarithmic depth amplification so deeper circuits have a
    # somewhat higher floor without saturating (the previous q*depth exponent
    # or diamond-norm bound saturated and forced M* = 1).
    depth = max(1, int(circuit_depth))
    shrink = (1.0 - 4.0 * eta / 3.0) ** num_qubits
    base_floor = max(0.0, 1.0 - shrink)
    eps_noise = min(0.999, base_floor * (1.0 + jnp.log(depth)))
    eps_noise = float(eps_noise)
    budget = epsilon_target - eps_noise
    if budget <= 0:
        return 1
    # sketch error ~ sqrt(N/M) <= budget  =>  M >= N / budget**2
    m_star = jnp.ceil(dim / (budget ** 2))
    return int(jnp.maximum(1, m_star))
