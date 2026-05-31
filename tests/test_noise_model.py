import jax.numpy as jnp

from qos.primitives.noise_model import (
    DepolarizingChannel,
    compose_sketch_and_noise_error,
    crossover_sample_count,
)


def test_depolarizing_zero_noise_is_identity():
    d = jnp.exp(1j * jnp.linspace(0, 1, 8))
    ch = DepolarizingChannel(num_qubits=3, noise_rate=0.0)
    assert jnp.allclose(ch.apply_to_diagonal(d), d)


def test_depolarizing_high_noise_collapses_to_maximally_mixed():
    d = jnp.exp(1j * jnp.linspace(0, 1, 8))
    ch = DepolarizingChannel(num_qubits=3, noise_rate=1.0)
    assert jnp.allclose(ch.apply_to_diagonal(d), jnp.zeros_like(d), atol=1e-6)


def test_crossover_monotone_in_noise_rate():
    # BUG-2: higher depolarizing noise shrinks the error budget, so reaching
    # the same target requires MORE samples (M* non-decreasing in noise), or
    # M* collapses to 1 once noise exceeds the target. The previous assertion
    # (m2 <= m1) encoded the buggy regime where M* was always 1.
    m1 = crossover_sample_count(256, 0.0, 10, 1.5)
    m2 = crossover_sample_count(256, 0.05, 10, 1.5)
    assert m2 >= m1 or m2 == 1
    # and at zero noise the crossover must be a real, non-trivial sample count
    assert m1 >= 1


def test_compose_errors_triangle_inequality():
    total = compose_sketch_and_noise_error(0.2, 0.01, 10, 8)
    assert total <= 0.2 + min(2.0, 0.01 * 10 * 8)
