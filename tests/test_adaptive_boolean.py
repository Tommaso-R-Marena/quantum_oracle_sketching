import jax
import jax.numpy as jnp

from qos.core.oracle_sketch import q_oracle_sketch_boolean, q_oracle_sketch_boolean_adaptive


def test_adaptive_reduces_samples_for_sparse_function():
    n = 1024
    truth = jnp.zeros((n,), dtype=jnp.int32).at[:10].set(1)
    exact = jnp.exp(1j * jnp.pi * truth)
    uni, _ = q_oracle_sketch_boolean(truth, 1000)
    ada, _, _ = q_oracle_sketch_boolean_adaptive(truth, 1000, key=jax.random.PRNGKey(0))
    err_u = float(jnp.linalg.norm(uni - exact, ord=jnp.inf))
    err_a = float(jnp.linalg.norm(ada - exact, ord=jnp.inf))
    assert err_a < err_u


def test_adaptive_exact_for_uniform_function():
    truth = jnp.ones((128,), dtype=jnp.int32)
    uni, _ = q_oracle_sketch_boolean(truth, 1000)
    ada, _, _ = q_oracle_sketch_boolean_adaptive(truth, 1000, key=jax.random.PRNGKey(1))
    assert jnp.allclose(uni, ada, atol=1e-6)


def test_adaptive_importance_weights_sum_to_one():
    truth = jnp.zeros((64,), dtype=jnp.int32).at[:7].set(1)
    _, _, weights = q_oracle_sketch_boolean_adaptive(truth, 200, key=jax.random.PRNGKey(2))
    assert jnp.isclose(jnp.sum(weights), 1.0, atol=1e-6)


def test_pilot_frac_zero_falls_back_to_uniform():
    truth = jnp.mod(jnp.arange(64), 2)
    uni, _ = q_oracle_sketch_boolean(truth, 500)
    ada, _, _ = q_oracle_sketch_boolean_adaptive(truth, 500, pilot_frac=0.0, key=jax.random.PRNGKey(0))
    assert jnp.allclose(uni, ada, atol=1e-6)
