"""Interferometric / Clifford Classical Shadow (Zhao et al. 2025, Theorem F.16).

This module implements a **classical shadow** readout primitive for the QOS
pipeline: given a quantum weight state ``|w>``, it estimates the complex inner
products ``<w|x_j>`` for arbitrary (sparse) test vectors ``x_j`` from a compact
set of randomized single-shot measurements, without re-running circuits per
test vector.

## Algorithm (random Clifford classical shadows)

For each of ``T`` shots:
  1. Sample a Haar-random Clifford unitary ``U`` (the Clifford group is a
     unitary 3-design, hence a 2-design -- the property the estimator needs).
  2. Apply ``U`` to ``rho = |w><w|`` and measure in the computational basis,
     obtaining a 0/1 basis index ``b`` with probability ``|<b|U|w>|^2``.
  3. Form the single-shot snapshot of ``rho`` via the standard Clifford
     reconstruction (inverse measurement channel):

         rho_hat = (2^n + 1) * U^dagger |b><b| U  -  I.

The shadow estimate of any linear functional ``Tr(O rho)`` is the average of
``Tr(O rho_hat)`` over the ``T`` shots (the **1/T normalization** is essential).
For the complex overlap we use ``<w|x> = <w|rho|x>`` (since ``rho = |w><w|``),
so the per-shot estimator of ``<w|x>`` is

    o_hat = (2^n + 1) * conj(<b|U|w>) * <b|U|x>  -  <w|x>,

whose average over shots is unbiased and whose error decays as ``O(1/sqrt(T))``.

## Novel Extension (Marena 2026)

The estimator returns both ``Re<w|x_j>`` and ``Im<w|x_j>`` from the same shadow
(a single randomized-measurement ensemble), supporting complex-valued test
vectors needed for quantum-chemistry / protein-folding applications.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from qos.config import real_dtype


class InterferometricClassicalShadow:
    """Random-Clifford classical shadow for compact inner-product readout.

    Estimates the complex overlaps ``<w|x_j>`` for arbitrary test vectors from
    a random-Clifford classical shadow of ``rho = |w><w|``. The Clifford group
    is a unitary 2-design, so the reconstruction
    ``rho_hat = (2^n+1) U^dagger |b><b| U - I`` is an unbiased single-shot
    estimator of ``rho`` and the prediction error decays as ``O(1/sqrt(T))``.

    See: Marena (2026), section 7 (sec:shadow).

    Parameters
    ----------
    weight_state : jax.Array, shape (N,)
        The quantum weight state |w> as a complex amplitude vector, N = 2^n.
        It is renormalized to ||w||_2 = 1.
    num_shadows : int
        Number of randomized measurements (shadow budget T).
    key : jax.Array, optional
        JAX PRNG key (used to derive the NumPy / Clifford seed deterministically).
    """

    def __init__(
        self,
        weight_state: jax.Array,
        num_shadows: int = 1000,
        key: Optional[jax.Array] = None,
    ):
        self.weight_state = weight_state / jnp.linalg.norm(weight_state)
        self.num_shadows = num_shadows
        self.key = key if key is not None else random.PRNGKey(0)
        # number of qubits n such that N = 2^n
        self._N = int(self.weight_state.shape[0])
        self._n = int(round(np.log2(self._N)))
        if 2 ** self._n != self._N:
            raise ValueError(
                f"weight_state length {self._N} must be a power of two (2^n)."
            )
        # per-shot measurement rows  r_b = <b|U   (shape (T, N), complex)
        self._shadow_rows: Optional[np.ndarray] = None
        self._shadow_built = False

    def build_shadow(self) -> "InterferometricClassicalShadow":
        """Sample the random-Clifford shadow of ``|w>``.

        For each shot, draws a random Clifford ``U`` (Qiskit
        ``random_clifford``), measures ``U|w>`` in the computational basis to
        get a 0/1 basis index ``b``, and stores the measurement row
        ``r_b = <b|U`` (a length-N complex vector). Storing only the row keeps
        memory at ``O(T * N)`` and lets ``predict`` form the rank-1 snapshot
        ``(2^n+1) U^dagger |b><b| U`` implicitly.

        Returns
        -------
        self
        """
        from qiskit.quantum_info import random_clifford

        N, n = self._N, self._n
        w = np.asarray(self.weight_state, dtype=np.complex128)
        # Derive a deterministic NumPy seed from the JAX key for reproducibility.
        base_seed = int(jax.random.randint(self.key, (), 0, 2**31 - 1))
        rng = np.random.default_rng(base_seed)

        rows = np.empty((self.num_shadows, N), dtype=np.complex128)
        for t in range(self.num_shadows):
            # BUG-14a: random Clifford ensemble (a 2-design), not fixed Hadamard.
            cl = random_clifford(n, seed=rng)
            U = cl.to_matrix()                      # (N, N) unitary
            Uw = U @ w                              # amplitudes of U|w>
            probs = np.abs(Uw) ** 2
            probs = probs / probs.sum()             # guard tiny renorm drift
            # BUG-14c: outcome must be a 0/1 BASIS INDEX (not +/-1), so that
            # <b|U is row b of U. An int index in [0, N) cannot cause the
            # +/-1 division-by-zero pitfall.
            b = int(rng.choice(N, p=probs))
            rows[t] = U[b, :]                        # r_b = <b|U  (row b of U)
        self._shadow_rows = rows
        self._shadow_built = True
        return self

    def predict(
        self,
        test_vectors: jax.Array,
        epsilon: float = 0.1,
    ) -> jax.Array:
        """Predict ``Re<w|x_j>`` and ``Im<w|x_j>`` for each test vector x_j.

        Uses the stored random-Clifford shadow to estimate the complex overlap
        ``<w|x> = <w|rho|x>`` offline, at ``O(T * N)`` cost per test vector and
        with no further quantum measurements.

        Parameters
        ----------
        test_vectors : jax.Array, shape (m, N)
            m test vectors (each 2-norm ~1).
        epsilon : float
            Target additive error (informational; not used in the estimate).

        Returns
        -------
        predictions : jax.Array, shape (m, 2)
            Column 0: Re<w|x_j>, Column 1: Im<w|x_j>.
        """
        if not self._shadow_built:
            self.build_shadow()
        N = self._N
        rows = self._shadow_rows                    # (T, N), r_b = <b|U
        w = np.asarray(self.weight_state, dtype=np.complex128)
        X = np.asarray(test_vectors, dtype=np.complex128)
        if X.ndim == 1:
            X = X[None, :]

        # <b|U|w> = rows . w   (per shot);  <b|U|x> = rows . x
        bw = rows @ w                               # (T,)   <b|U|w>
        true_wx = np.conj(w) @ X.T                  # (m,)   <w|x>  (exact term)

        preds = np.empty((X.shape[0], 2), dtype=np.float64)
        coeff = (2.0 ** self._n + 1.0)
        for j in range(X.shape[0]):
            bx = rows @ X[j]                        # (T,)   <b|U|x_j>
            # BUG-14a reconstruction (per shot):
            #   o_hat = (2^n+1) * conj(<b|U|w>) * <b|U|x>  -  <w|x>
            snapshots = coeff * np.conj(bw) * bx - true_wx[j]
            # BUG-14b: average over T shots (the 1/T normalization).
            est = snapshots.mean()
            # BUG-14c: guard against NaN/Inf from a malformed outcome encoding.
            shadow_estimate = np.array([est.real, est.imag], dtype=np.float64)
            assert np.all(np.isfinite(shadow_estimate)), (
                "Shadow estimate contains NaN/Inf -- check measurement "
                "outcome encoding (must be 0/1 basis index, not +/-1)"
            )
            preds[j] = shadow_estimate
        return jnp.asarray(preds, dtype=real_dtype)

    def prediction_error_bound(self, sparsity: int) -> float:
        """Upper bound on prediction error given sparsity s.

        From Theorem F.16 of Zhao et al.: error <= sqrt(s / num_shadows).
        """
        return float(jnp.sqrt(sparsity / self.num_shadows))
