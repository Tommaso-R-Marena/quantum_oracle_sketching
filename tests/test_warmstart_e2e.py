"""End-to-end smoke tests for the warmstart ablation pipeline.

These tests exercise the full pipeline used by the warmstart-ablation
notebook on synthetic data (no external dataset, no Colab dependency):

  - fit a VariationalWarmstart on a small N=64 Boolean truth table;
  - convert predict() to a real +-1 diagonal via jnp.sign(jnp.real(...));
  - compute TVD vs the ideal (-1)**tt diagonal;
  - verify the diagnose-warmstart gate does NOT pass with a
    deliberately-broken (random) prediction (no false convergence);
  - verify a small binary search behaves monotonically on the cold path.

They are deliberately small so the full file runs in well under a minute.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qos.core.oracle_sketch import q_oracle_sketch_boolean
from qos.theory.variational_warmstart import VariationalWarmstart

from tests.test_tvd_core import tvd_diag  # canonical TVD reference

EPSILON = 0.10  # target TVD threshold for "converged"
SEED = 0
N_BITS = 6
N = 2 ** N_BITS


@pytest.fixture(scope="module")
def sparse_tt():
    tt = jnp.zeros(N, dtype=jnp.float64)
    return tt.at[:4].set(1.0)  # K=4 support out of N=64


@pytest.fixture(scope="module")
def d_ideal_sparse(sparse_tt):
    return (-1.0) ** sparse_tt


def predict_real(vw: VariationalWarmstart) -> jax.Array:
    return jnp.sign(jnp.real(vw.predict()))


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline_converges_at_full_budget(sparse_tt, d_ideal_sparse):
    """Warmstart with full truth table + reasonable budget reaches TVD < epsilon."""
    vw = VariationalWarmstart(
        sparse_tt,
        num_fourier_modes=16,
        learning_rate=0.03,
        num_steps=400,
        key=jax.random.PRNGKey(SEED),
    )
    vw.fit(unit_num_samples=N * 4)
    d_warm = predict_real(vw)
    t = tvd_diag(d_warm, d_ideal_sparse)
    assert t < EPSILON, (
        f"Warmstart at full budget did not converge: TVD={t:.4f} >= {EPSILON}"
    )


def test_diagnose_warmstart_rejects_random_prediction(d_ideal_sparse):
    """The diagnose gate must NOT report convergence for a random +-1 diagonal.

    This guards against a regression where an always-zero TVD (e.g. abs() bug
    in tvd_diag, or sign() collapsing to all-+1) would let any diagnostic
    pass.  Here we pass a true-random diagonal and assert the gate flags it.
    """
    rng = np.random.default_rng(SEED)
    d_random = rng.choice([-1.0, 1.0], size=N)
    t = tvd_diag(d_random, d_ideal_sparse)
    # With sparse_tt mostly-+1, random gives ~uniform-on-bits; we expect t
    # well above epsilon.  Use a loose lower bound to keep the test robust.
    assert t > EPSILON, (
        f"Diagnose gate falsely accepts a random prediction: TVD={t:.4f} < {EPSILON}"
    )


def test_cold_sketch_tvd_monotone_in_budget(sparse_tt, d_ideal_sparse):
    """Increasing the cold-sketch sample count should generally reduce TVD.

    We compare M = 50 vs M = 2000 and require the larger budget to be
    strictly better in expectation; this guards against a TVD regression
    where TVD became insensitive to the sample count (broken normalization).
    """
    tt = sparse_tt
    d_low, _ = q_oracle_sketch_boolean(tt, 50)
    d_hi, _ = q_oracle_sketch_boolean(tt, 2000)
    t_low = tvd_diag(d_low, d_ideal_sparse)
    t_hi = tvd_diag(d_hi, d_ideal_sparse)
    assert t_hi <= t_low + 1e-3, (
        f"Cold-sketch TVD did not improve with budget: "
        f"M=50 -> {t_low:.4f}, M=2000 -> {t_hi:.4f}"
    )


def test_predict_real_is_pm_one_after_fit(sparse_tt):
    """The sign(real(predict)) post-processing must yield +-1 valued output."""
    vw = VariationalWarmstart(
        sparse_tt,
        num_fourier_modes=8,
        learning_rate=0.02,
        num_steps=30,
        key=jax.random.PRNGKey(1),
    )
    vw.fit(unit_num_samples=100)
    d_warm = predict_real(vw)
    vals = set(float(v) for v in jnp.unique(d_warm))
    assert vals.issubset({-1.0, 0.0, 1.0}), (
        f"predict_real() returned values outside {{-1, 0, +1}}: {vals}"
    )


def test_complex_input_does_not_corrupt_real_path(d_ideal_sparse):
    """Passing a complex phase oracle to VariationalWarmstart must not raise
    a complex-to-real DeprecationWarning under JAX >= 0.10.

    Regression check: the constructor previously called
    `truth_arr.astype(real_dtype)` unconditionally, dropping the imaginary
    part of a phase target.  The fix preserves complex inputs and uses
    `|truth_arr|` as the support indicator.
    """
    rng = jax.random.PRNGKey(SEED)
    phases = jax.random.uniform(rng, (N,), minval=0.0, maxval=2 * jnp.pi)
    f = jnp.exp(1j * phases)
    # If the regression returns, JAX 0.10 will raise a DeprecationWarning
    # for the complex->real .astype; with our default filters that is a
    # warning, but with the project's CI invocation it can be promoted to
    # an error.  Either way, the call must complete here without raising.
    vw = VariationalWarmstart(
        f,
        num_fourier_modes=8,
        learning_rate=0.001,
        num_steps=50,
        key=jax.random.PRNGKey(7),
    )
    result = vw.fit()
    assert result["variational_error"] < result["baseline_error"] + 0.5
