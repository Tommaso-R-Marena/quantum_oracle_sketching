import jax
import jax.numpy as jnp

from qos.core.state_sketch import (
    fit_kernel_svm_from_states,
    q_interferometric_kernel_shadow,
    q_kernel_estimate,
    q_state_sketch_flat,
)


def test_kernel_is_nonnegative():
    s1, _ = q_state_sketch_flat(jnp.array([1, -1, 1, -1]), 300)
    s2, _ = q_state_sketch_flat(jnp.array([1, 1, -1, -1]), 300)
    assert q_kernel_estimate(s1, s2) >= 0.0


def test_kernel_diagonal_equals_one():
    s, _ = q_state_sketch_flat(jnp.array([1, -1, 1, -1]), 300)
    assert abs(q_kernel_estimate(s, s) - 1.0) < 1e-6


def test_kernel_svm_perfect_on_separable_data():
    x = jnp.array([
        [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ])
    y = jnp.array([1, 1, -1, -1])
    states = jax.vmap(lambda v: q_state_sketch_flat(v, 500)[0])(x)
    alpha = fit_kernel_svm_from_states(states, y, regularization=1e-6)
    preds = jnp.array([q_interferometric_kernel_shadow(states, y, alpha, s) for s in states])
    assert jnp.all(preds == y)


def test_interferometric_prediction_binary():
    x = jnp.array([[1, 1, 1, 1], [-1, -1, -1, -1]])
    y = jnp.array([1, -1])
    states = jax.vmap(lambda v: q_state_sketch_flat(v, 300)[0])(x)
    alpha = fit_kernel_svm_from_states(states, y)
    pred = q_interferometric_kernel_shadow(states, y, alpha, states[0])
    assert pred in {-1.0, 1.0}
