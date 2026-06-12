"""Convergence + correctness tests for the random-Clifford classical shadow
estimator (BUG-14 fix) in qos.theory.interferometric_shadow.

These tests document the estimator's intended behavior:
  1. smoke      -- finite output of the correct shape;
  2. convergence-- mean abs error strictly decreases from T=100 to T=1000;
  3. normalization -- the shadow mean matches a true Pauli-Z expectation.

If the estimator regresses to a bias floor (the prior non-2-design ensemble),
tests 2 and 3 fail -- so this file also guards against that failure mode.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from qos.theory.interferometric_shadow import InterferometricClassicalShadow


def _random_state(n_qubits: int, seed: int) -> jnp.ndarray:
    """Return a normalized random complex state of dimension 2**n_qubits."""
    rng = np.random.default_rng(seed)
    N = 2 ** n_qubits
    w = rng.normal(size=N) + 1j * rng.normal(size=N)
    w = w / np.linalg.norm(w)
    return jnp.asarray(w, dtype=jnp.complex128)


def test_shadow_smoke_finite_and_shape():
    """T=100 on a 2-qubit random state: output is finite with shape (1, 2)."""
    w = _random_state(2, seed=0)
    rng = np.random.default_rng(1)
    x = rng.normal(size=4) + 1j * rng.normal(size=4)
    x = jnp.asarray(x / np.linalg.norm(x), dtype=jnp.complex128)

    shadow = InterferometricClassicalShadow(
        w, num_shadows=100, key=jax.random.PRNGKey(0)
    ).build_shadow()
    preds = shadow.predict(jnp.stack([x]))

    assert preds.shape == (1, 2)
    assert np.all(np.isfinite(np.asarray(preds))), "shadow output must be finite"


def test_shadow_converges_T100_vs_T1000():
    """Mean abs error at T=1000 must be strictly smaller than at T=100."""
    n_qubits = 3
    w = _random_state(n_qubits, seed=2)
    rng = np.random.default_rng(3)
    # average error over several fixed test vectors for a stable comparison
    N = 2 ** n_qubits
    xs = []
    for i in range(5):
        v = rng.normal(size=N) + 1j * rng.normal(size=N)
        xs.append(v / np.linalg.norm(v))
    X = jnp.asarray(np.stack(xs), dtype=jnp.complex128)

    w_np = np.asarray(w)
    true = np.conj(w_np) @ np.asarray(X).T  # <w|x_j>

    def mean_abs_err(T: int) -> float:
        shadow = InterferometricClassicalShadow(
            w, num_shadows=T, key=jax.random.PRNGKey(123)
        ).build_shadow()
        preds = np.asarray(shadow.predict(X))
        est = preds[:, 0] + 1j * preds[:, 1]
        return float(np.mean(np.abs(est - true)))

    err_100 = mean_abs_err(100)
    err_1000 = mean_abs_err(1000)
    assert err_1000 < err_100, (
        f"estimator not converging: err(T=1000)={err_1000:.4f} "
        f">= err(T=100)={err_100:.4f}"
    )


def test_shadow_normalization_pauli_z():
    """Mean shadow estimate of <Z> for a 1-qubit state is within 0.01 of truth.

    For a single qubit, <Z> = <w|Z|w> = |w_0|^2 - |w_1|^2. Using the shadow of
    rho = |w><w|, this equals <w | (Z|w>) > = predict([Z @ w])[0] (real part),
    since the estimator returns <w|x> for an (un-normalized) test vector x.
    """
    w = _random_state(1, seed=4)
    w_np = np.asarray(w)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    true_z = float((np.conj(w_np) @ (Z @ w_np)).real)

    x = jnp.asarray(Z @ w_np, dtype=jnp.complex128)  # un-normalized on purpose
    shadow = InterferometricClassicalShadow(
        w, num_shadows=10000, key=jax.random.PRNGKey(7)
    ).build_shadow()
    est_z = float(np.asarray(shadow.predict(jnp.stack([x])))[0, 0])

    assert abs(est_z - true_z) < 0.01, (
        f"shadow <Z>={est_z:.4f} not within 0.01 of true {true_z:.4f}"
    )
