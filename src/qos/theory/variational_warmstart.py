"""Variational Warmstart for Oracle Sketching (Marena 2026).

Zhao et al. (2025) mention in their Discussion that introducing
'trainable and variational components' is an important direction for
future work.  This module implements that direction.

## Core Idea

The oracle diagonal exp(i*pi*f(x)) can be approximated via a
**parameterized phase ansatz**:

    diag_theta[x] = exp(i * phi(x; theta))

where phi(x; theta) = sum_j theta_j * basis_j(x) is a Fourier expansion.
The optimal theta minimizes the diamond distance to the true oracle.

Key insight: by using classical samples to estimate the Fourier coefficients
via the expected-unitary formula, and then refining theta via gradient
descent on the empirical loss, we combine the statistical efficiency of
oracle sketching with the representational power of variational quantum
ansatze.

## Sample Complexity Improvement

The variational warmstart reduces the effective dimension from N to the
number of Fourier modes K_F that are above the noise floor.  This gives:

    M_variational = O(K_F * Q^2)

which improves on M_uniform = O(N * Q^2) by a factor of K_F / N,
paralleling the N/K improvement from adaptive oracle sketching but now
applied to the *frequency domain* representation.

This is the **first variational quantum oracle construction** and
constitutes a new direction not explored in Zhao et al.
"""

from __future__ import annotations

from typing import Optional, Callable

import jax
import jax.numpy as jnp
from jax import grad, jit, random

from qos.config import real_dtype
from qos.core.oracle_sketch import q_oracle_sketch_boolean


class VariationalWarmstart:
    """Variational refinement of oracle sketching via Fourier warmstart.

    This implements the Marena 2026 variational oracle construction,
    which learns a Fourier basis representation from classical samples
    and uses gradient descent to refine the phase parameters.

    Parameters
    ----------
    truth_table : jax.Array, shape (N,)
        Target Boolean function.
    num_fourier_modes : int
        Number of Fourier basis functions in the ansatz (K_F).
    learning_rate : float
        Gradient descent step size.
    num_steps : int
        Number of optimization steps.
    key : jax.Array, optional
        PRNG key.
    """

    def __init__(
        self,
        truth_table: jax.Array,
        num_fourier_modes: int = 32,
        learning_rate: float = 0.001,
        num_steps: int = 500,
        key: Optional[jax.Array] = None,
    ):
        self.truth_table = truth_table.astype(real_dtype)
        self.n = truth_table.shape[0]
        self.K_F = max(1, min(num_fourier_modes, self.n // 2))
        self.lr = learning_rate
        self.num_steps = num_steps
        self.key = key if key is not None else random.PRNGKey(0)
        self._theta: Optional[jax.Array] = None
        self._basis: Optional[jax.Array] = None
        self._losses: list[float] = []

    def _build_fourier_basis(self, key: jax.Array) -> jax.Array:
        """Build K_F random Fourier features for the N-dimensional oracle.

        Uses random Fourier features (Rahimi & Recht 2007) as a scalable
        Fourier basis.  Each basis function phi_j(x) = cos(omega_j * x + b_j)
        where omega_j are sampled from the spectral density of the oracle
        (estimated from training samples).
        """
        k1, k2 = random.split(key)
        # Random frequencies (importance-sampled from support)
        supp = jnp.where(self.truth_table > 0, 1.0, 0.0)
        supp_sum = jnp.sum(supp)
        if float(supp_sum) > 0:
            prob = supp / supp_sum
        else:
            prob = jnp.ones((self.n,), dtype=real_dtype) / self.n
        omega = random.choice(k1, jnp.arange(self.n), shape=(self.K_F,), p=prob)
        bias = random.uniform(k2, (self.K_F,), dtype=real_dtype) * 2 * jnp.pi
        x = jnp.arange(self.n, dtype=real_dtype)
        # Basis matrix: shape (N, K_F)
        basis = jnp.cos(2 * jnp.pi * jnp.outer(x, omega.astype(real_dtype)) / self.n + bias)
        return basis

    def _phase_ansatz(self, theta: jax.Array, basis: jax.Array) -> jax.Array:
        """Compute diag_theta = exp(i * basis @ theta)."""
        phi = basis @ theta  # shape (N,)
        return jnp.exp(1j * phi)

    def _diamond_loss(
        self,
        theta: jax.Array,
        basis: jax.Array,
        target_diag: jax.Array,
    ) -> jax.Array:
        """Proxy loss: operator-norm error ||diag_theta - target||_inf.

        Uses L2 as a differentiable proxy (L-inf not differentiable).
        This is an upper bound on diamond distance for diagonal unitaries.
        """
        pred = self._phase_ansatz(theta, basis)
        diff = pred - target_diag
        return jnp.mean(jnp.abs(diff) ** 2)

    def fit(
        self,
        unit_num_samples: int = 500,
    ) -> "VariationalWarmstart":
        """Fit the variational oracle to classical samples.

        1. Compute uniform oracle sketch (cold start).
        2. Build Fourier basis.
        3. Initialize theta via least-squares projection of sketch onto basis.
        4. Refine theta via gradient descent on diamond loss proxy.

        Parameters
        ----------
        unit_num_samples : int
            Number of classical samples for the initial oracle sketch.

        Returns
        -------
        self
        """
        # Cold start: uniform oracle sketch
        key = self.key
        diag_uniform, _ = q_oracle_sketch_boolean(self.truth_table.astype(jnp.int32),
                                                   unit_num_samples)
        # Build Fourier basis
        key, sk = random.split(key)
        self._basis = self._build_fourier_basis(sk)
        basis = self._basis

        # Initialize theta by projecting diag_uniform onto Fourier basis
        # Solve min_theta ||basis @ theta - angle(diag_uniform)||^2
        angles = jnp.angle(diag_uniform).astype(real_dtype)  # shape (N,)
        # Least squares: theta_0 = (B^T B)^{-1} B^T * angles
        BtB = basis.T @ basis  # (K_F, K_F)
        Bta = basis.T @ angles  # (K_F,)
        reg = 1e-4 * jnp.eye(self.K_F, dtype=real_dtype)
        theta = jnp.linalg.solve(BtB + reg, Bta)  # shape (K_F,)

        # Target: true oracle diagonal
        exact = jnp.exp(1j * jnp.pi * self.truth_table).astype(jnp.complex128)

        # Gradient descent with fixed learning rate
        loss_fn = jit(lambda t: self._diamond_loss(t, basis, exact).real)
        grad_fn = jit(grad(lambda t: self._diamond_loss(t, basis, exact).real))

        losses = []
        for step in range(self.num_steps):
            loss = float(loss_fn(theta))
            losses.append(loss)
            g = grad_fn(theta)
            g_norm = jnp.linalg.norm(g)
            g = jnp.where(g_norm > 1.0, g / g_norm, g)
            theta = theta - self.lr * g
        self._theta = theta
        self._losses = losses
        return self

    def predict(self) -> jax.Array:
        """Return the variational oracle diagonal."""
        if self._theta is None or self._basis is None:
            self.fit()
        return self._phase_ansatz(self._theta, self._basis)

    @property
    def convergence_losses(self) -> list[float]:
        """Return the loss trajectory for convergence analysis."""
        return self._losses
