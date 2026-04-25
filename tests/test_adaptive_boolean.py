"""Tests for adaptive Boolean oracle sketching (Marena 2026).

Key invariants under test:
  1. On supp(f), adaptive error <= uniform error at equal M (sparsity advantage).
  2. On all-ones f, adaptive and uniform produce the same diagonal.
  3. Importance weights sum to 1.
  4. pilot_frac=0 falls back to uniform.
  5. Error on supp(f) decreases monotonically with M (convergence).

Note on error metric:
  All error comparisons use L-inf restricted to supp(f), i.e.
      max_{x: f(x)=1} |diag[x] - exp(i*pi*f(x))|
  Off-support entries are trivially correct for both methods and must be
  excluded to observe the N/K improvement on the support.
"""

import jax
import jax.numpy as jnp
import pytest

from qos.core.oracle_sketch import q_oracle_sketch_boolean, q_oracle_sketch_boolean_adaptive


def _support_linf_error(diag: jax.Array, exact: jax.Array, mask: jax.Array) -> float:
    """L-inf error restricted to supp(f)."""
    return float(jnp.max(jnp.abs((diag - exact)[mask])))


def test_adaptive_reduces_error_on_support():
    """Adaptive oracle achieves lower L-inf error on supp(f) than uniform at equal M.

    Setup: N=1024, K=10 (sparsity ratio 1%), M=5000.
    Expected: adaptive error << uniform error on supp(f).
    Adaptive crossover: M ~ K*pi^2/eps^2 ~ 10*10/eps^2; at M=5000 we are well
    past crossover for eps ~ 0.14.
    """
    n, K = 1024, 10
    truth = jnp.zeros((n,), dtype=jnp.int32).at[:K].set(1)
    exact = jnp.exp(1j * jnp.pi * truth.astype(jnp.float64))
    mask = truth.astype(bool)
    M = 5000
    errors_u, errors_a = [], []
    for seed in range(8):
        uni, _ = q_oracle_sketch_boolean(truth, M)
        ada, _, _ = q_oracle_sketch_boolean_adaptive(truth, M, pilot_frac=0.1, key=jax.random.PRNGKey(seed))
        errors_u.append(_support_linf_error(uni, exact, mask))
        errors_a.append(_support_linf_error(ada, exact, mask))
    mean_u = float(jnp.mean(jnp.array(errors_u)))
    mean_a = float(jnp.mean(jnp.array(errors_a)))
    assert mean_a < mean_u, (
        f"Adaptive ({mean_a:.4f}) should beat uniform ({mean_u:.4f}) on supp(f) "
        f"at N={n}, K={K}, M={M}."
    )


def test_adaptive_exact_for_uniform_function():
    """When f is all-ones, adaptive and uniform give the same diagonal."""
    truth = jnp.ones((128,), dtype=jnp.int32)
    uni, _ = q_oracle_sketch_boolean(truth, 1000)
    ada, _, _ = q_oracle_sketch_boolean_adaptive(truth, 1000, key=jax.random.PRNGKey(1))
    assert jnp.allclose(uni, ada, atol=1e-4)


def test_adaptive_importance_weights_sum_to_one():
    """Importance weights must form a valid probability distribution."""
    truth = jnp.zeros((64,), dtype=jnp.int32).at[:7].set(1)
    _, _, weights = q_oracle_sketch_boolean_adaptive(truth, 500, key=jax.random.PRNGKey(2))
    assert jnp.isclose(jnp.sum(weights), 1.0, atol=1e-5)


def test_pilot_frac_zero_falls_back_to_uniform():
    """pilot_frac=0 must produce identical output to the uniform oracle."""
    truth = jnp.mod(jnp.arange(64), 2).astype(jnp.int32)
    uni, _ = q_oracle_sketch_boolean(truth, 500)
    ada, _, _ = q_oracle_sketch_boolean_adaptive(truth, 500, pilot_frac=0.0, key=jax.random.PRNGKey(0))
    assert jnp.allclose(uni, ada, atol=1e-6)


def test_adaptive_error_decreases_with_M():
    """Adaptive support error must decrease as M grows (convergence check)."""
    n, K = 512, 8
    truth = jnp.zeros((n,), dtype=jnp.int32).at[:K].set(1)
    exact = jnp.exp(1j * jnp.pi * truth.astype(jnp.float64))
    mask = truth.astype(bool)
    Ms = [500, 2000, 10000]
    prev_err = float('inf')
    for M in Ms:
        errs = []
        for seed in range(6):
            ada, _, _ = q_oracle_sketch_boolean_adaptive(
                truth, M, pilot_frac=0.1, key=jax.random.PRNGKey(seed)
            )
            errs.append(_support_linf_error(ada, exact, mask))
        err = float(jnp.mean(jnp.array(errs)))
        assert err < prev_err, f"Error did not decrease: M={M}, err={err:.4f}, prev={prev_err:.4f}"
        prev_err = err


def test_adaptive_nk_improvement_factor():
    """Adaptive needs ~N/K fewer samples than uniform to reach same error on supp(f).

    We verify that at M=K*C the adaptive error is comparable to uniform at M=N*C
    (i.e., the adaptive method is N/K more sample-efficient on the support).
    """
    n, K, C = 512, 8, 100
    M_adaptive = K * C        # 800 samples
    M_uniform  = n * C        # 51200 samples
    truth = jnp.zeros((n,), dtype=jnp.int32).at[:K].set(1)
    exact = jnp.exp(1j * jnp.pi * truth.astype(jnp.float64))
    mask = truth.astype(bool)
    errs_a, errs_u = [], []
    for seed in range(10):
        ada, _, _ = q_oracle_sketch_boolean_adaptive(
            truth, M_adaptive, pilot_frac=0.1, key=jax.random.PRNGKey(seed)
        )
        uni, _ = q_oracle_sketch_boolean(truth, M_uniform)
        errs_a.append(_support_linf_error(ada, exact, mask))
        errs_u.append(_support_linf_error(uni, exact, mask))
    mean_a = float(jnp.mean(jnp.array(errs_a)))
    mean_u = float(jnp.mean(jnp.array(errs_u)))
    # Both should be in the same ballpark -- adaptive at 800 samples should
    # achieve similar or better error than uniform at 51200 samples.
    assert mean_a <= mean_u * 2.0, (
        f"N/K improvement not observed: adaptive@M={M_adaptive} err={mean_a:.4f}, "
        f"uniform@M={M_uniform} err={mean_u:.4f}"
    )
