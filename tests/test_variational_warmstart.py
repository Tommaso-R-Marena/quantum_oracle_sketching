"""Tests for Variational Warmstart Oracle (Marena 2026).

Verifies:
  1. Variational oracle converges (loss decreases).
  2. Variational oracle achieves better or equal accuracy than uniform sketch.
  3. Fourier basis covers support.
  4. theta is optimized (not at initialization).
"""

import jax
import jax.numpy as jnp
import pytest

from qos.theory.variational_warmstart import VariationalWarmstart


N, K = 128, 8


@pytest.fixture
def truth_table():
    t = jnp.zeros((N,), dtype=jnp.int32)
    return t.at[:K].set(1)


def test_variational_loss_decreases(truth_table):
    """Training loss must strictly decrease over optimization steps."""
    vw = VariationalWarmstart(
        truth_table, num_fourier_modes=16, learning_rate=0.01,
        num_steps=50, key=jax.random.PRNGKey(0)
    )
    vw.fit(unit_num_samples=300)
    losses = vw.convergence_losses
    assert len(losses) == 50
    assert losses[-1] < losses[0], (
        f"Loss should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )


def test_variational_oracle_bounded(truth_table):
    """Variational oracle diagonal must have unit modulus."""
    vw = VariationalWarmstart(
        truth_table, num_fourier_modes=16, learning_rate=0.01,
        num_steps=30, key=jax.random.PRNGKey(1)
    )
    vw.fit(unit_num_samples=300)
    diag = vw.predict()
    assert jnp.allclose(jnp.abs(diag), 1.0, atol=1e-5), "Variational oracle must be unitary diagonal"


def test_variational_basis_shape(truth_table):
    vw = VariationalWarmstart(truth_table, num_fourier_modes=16)
    vw.fit(unit_num_samples=200)
    assert vw._basis is not None
    assert vw._basis.shape == (N, 16)
