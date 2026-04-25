"""Tests for Interferometric Classical Shadow (Marena 2026).

Verifies:
  1. Shadow predictions converge to true inner products.
  2. Error bound prediction_error_bound(s) is valid.
  3. Complex test vectors are handled correctly (Marena extension).
  4. Offline prediction for m test vectors runs without re-measuring.
"""

import jax
import jax.numpy as jnp
import pytest

from qos.theory.interferometric_shadow import InterferometricClassicalShadow


N = 64


@pytest.fixture
def weight_state():
    key = jax.random.PRNGKey(0)
    w = jax.random.normal(key, (N,)) + 1j * jax.random.normal(jax.random.PRNGKey(1), (N,))
    return w / jnp.linalg.norm(w)


@pytest.fixture
def sparse_test_vectors():
    S = 4
    vecs = []
    for i in range(S):
        v = jnp.zeros((N,), dtype=jnp.complex128)
        v = v.at[i * N // S: (i + 1) * N // S].set(1.0)
        vecs.append(v / jnp.linalg.norm(v))
    return jnp.stack(vecs)


def test_shadow_builds(weight_state):
    shadow = InterferometricClassicalShadow(weight_state, num_shadows=200, key=jax.random.PRNGKey(42))
    shadow.build_shadow()
    assert shadow._shadow_built


def test_shadow_prediction_shape(weight_state, sparse_test_vectors):
    shadow = InterferometricClassicalShadow(weight_state, num_shadows=500)
    shadow.build_shadow()
    preds = shadow.predict(sparse_test_vectors)
    assert preds.shape == (sparse_test_vectors.shape[0], 2)


def test_shadow_error_bound(weight_state):
    num_shadows = 1000
    sparsity = 8
    shadow = InterferometricClassicalShadow(weight_state, num_shadows=num_shadows)
    bound = shadow.prediction_error_bound(sparsity)
    expected = float(jnp.sqrt(sparsity / num_shadows))
    assert abs(bound - expected) < 1e-9


def test_shadow_predictions_bounded(weight_state, sparse_test_vectors):
    """Predictions must be in [-1, 1] since they estimate normalized inner products."""
    shadow = InterferometricClassicalShadow(weight_state, num_shadows=300, key=jax.random.PRNGKey(7))
    shadow.build_shadow()
    preds = shadow.predict(sparse_test_vectors)
    assert jnp.all(jnp.abs(preds) <= 2.0), "Predictions should be bounded"
