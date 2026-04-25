"""Tests for quantum primitives (amplitude amplification)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import random

from qos.primitives.amplification import amplitude_amplification
from qos.utils.numerical import is_hermitian, is_unitary


def test_amplitude_amplification_exact():
    key = random.PRNGKey(42)
    dim = 500
    initial_norm = 0.2
    target_norm = 0.99
    degree = 51

    key, subkey = random.split(key)
    v = random.normal(subkey, (dim,))
    v = v / jnp.linalg.norm(v) * initial_norm

    state_aa = amplitude_amplification(
        v, degree=degree, target_norm=target_norm
    )
    final_norm = float(jnp.linalg.norm(state_aa))
    assert pytest.approx(target_norm, abs=1e-2) == final_norm

    error = float(jnp.linalg.norm(v / jnp.linalg.norm(v) - state_aa / target_norm))
    assert error < 1e-2


def test_amplitude_amplification_imperfect():
    key = random.PRNGKey(42)
    dim = 100
    initial_norm = 0.2
    target_norm = 0.99
    degree = 51
    noise_level = 0.01

    key, subkey = random.split(key)
    v = random.normal(subkey, (dim,))
    v = v / jnp.linalg.norm(v) * initial_norm

    key, subkey = random.split(key)
    noise = noise_level * random.normal(subkey, (degree, dim)) * jnp.linalg.norm(v)
    v_imperfect = jnp.tile(v, (degree, 1)) + noise

    state_aa = amplitude_amplification(
        v_imperfect, degree=degree, target_norm=target_norm
    )
    final_norm = float(jnp.linalg.norm(state_aa))
    assert pytest.approx(target_norm, abs=1e-2) == final_norm

    error = float(jnp.linalg.norm(v / jnp.linalg.norm(v) - state_aa / target_norm))
    assert error / noise_level < 10.0


def test_amplitude_amplification_zero_norm_raises():
    v = jnp.zeros(10)
    with pytest.raises(ValueError, match="zero norm"):
        amplitude_amplification(v, degree=11)
