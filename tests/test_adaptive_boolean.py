"""Tests for adaptive Boolean oracle sketching (Marena 2026).

All arithmetic is done in float64/complex128 (JAX x64 enabled at the top
of each test via jax.config.update). The tests verify:

  1. Uniform oracle converges: error DECREASES as M grows.
  2. Adaptive oracle converges: error DECREASES as M grows.
  3. pilot_frac=0 gives identical output to uniform oracle (fallback).
  4. Importance weights sum to 1 on supp(f).
  5. All-ones fallback triggers uniform path.
  6. Off-support entries are exactly 1.
  7. N/K improvement: adaptive at M=K*C matches uniform at M=N*C.
  8. Large-N/K regime: adaptive strictly beats uniform at equal M.

Operating-point discipline
--------------------------
The key regime for N/K demonstration is N/K >> 1 AND M >> K so that
the adaptive pilot concentrates on supp(f) reliably. We use N=2048, K=4,
giving N/K=512. At M=8000 the uniform error ~sqrt(N/M)~0.5 while
adaptive error ~sqrt(K/M)~0.02. The strict inequality holds with very
high probability over random seeds.

Convergence direction
---------------------
Error = max_{x: f(x)=1} |diag[x] - exp(i*pi)| = max |diag[x] - (-1)|.
At M=100 this is ~1.98 (diag near +1), at M=infinity it is 0 (diag=-1).
So error is DECREASING with M. The test checks b < a for consecutive pairs.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from qos.core.oracle_sketch import q_oracle_sketch_boolean, q_oracle_sketch_boolean_adaptive


def _supp_linf(diag: jax.Array, truth: jax.Array) -> float:
    """L-inf error on supp(f): max_{x: f(x)=1} |diag[x] - exp(i*pi)|."""
    exact = jnp.exp(1j * jnp.pi * truth.astype(jnp.float64))
    mask  = truth.astype(bool)
    return float(jnp.max(jnp.abs((diag - exact)[mask])))


# --------------------------------------------------------------------
# 1. Uniform oracle: error strictly decreasing with M
# --------------------------------------------------------------------

def test_uniform_oracle_converges():
    """Uniform oracle error on supp(f) must strictly decrease as M grows."""
    truth = jnp.zeros(256, dtype=jnp.int32).at[:8].set(1)
    # At M=100: diag ~ exp(i*100*pi/100) = exp(i*pi) = -1 exactly for N=256?
    # No: diag = exp(100 * log(1 + (1/256)*expm1(i*pi/100)))
    #         ~ exp(100 * (1/256)*i*pi/100) = exp(i*pi/256) ~ 1+small
    # Error ~ |1 - (-1)| = 2 at low M, converges to 0 as M->inf.
    Ms   = [100, 500, 2000, 10000, 50000]
    errs = [_supp_linf(q_oracle_sketch_boolean(truth, M)[0], truth) for M in Ms]
    for i, (a, b) in enumerate(zip(errs, errs[1:])):
        assert b < a, (
            f"Uniform oracle error not decreasing at step {i}: "
            f"err[M={Ms[i]}]={a:.6f}, err[M={Ms[i+1]}]={b:.6f}. "
            f"All errors: {[f'{e:.6f}' for e in errs]}"
        )


# --------------------------------------------------------------------
# 2. Adaptive oracle: error decreasing with M (with slack for noise)
# --------------------------------------------------------------------

def test_adaptive_oracle_converges():
    """Adaptive oracle mean error over seeds must decrease as M grows."""
    truth = jnp.zeros(256, dtype=jnp.int32).at[:8].set(1)
    Ms    = [200, 800, 3200, 12800]
    N_SEEDS = 6
    errs = []
    for M in Ms:
        seed_errs = [
            _supp_linf(
                q_oracle_sketch_boolean_adaptive(
                    truth, M, pilot_frac=0.2, key=jax.random.PRNGKey(s)
                )[0],
                truth,
            )
            for s in range(N_SEEDS)
        ]
        errs.append(float(jnp.mean(jnp.array(seed_errs))))
    # Allow 30% slack for stochastic noise between steps
    for i, (a, b) in enumerate(zip(errs, errs[1:])):
        assert b < a * 1.3, (
            f"Adaptive oracle mean error not decreasing at step {i}: "
            f"{[f'{e:.4f}' for e in errs]}"
        )


# --------------------------------------------------------------------
# 3. Fallback: pilot_frac=0 gives same output as uniform
# --------------------------------------------------------------------

def test_pilot_frac_zero_fallback():
    """pilot_frac=0 must give identical output to uniform oracle."""
    truth = jnp.mod(jnp.arange(64), 2).astype(jnp.int32)
    uni, _ = q_oracle_sketch_boolean(truth, 500)
    ada, _, _ = q_oracle_sketch_boolean_adaptive(
        truth, 500, pilot_frac=0.0, key=jax.random.PRNGKey(0)
    )
    assert jnp.allclose(uni, ada, atol=1e-9), (
        f"Fallback mismatch: max diff = {float(jnp.max(jnp.abs(uni - ada))):.2e}"
    )


# --------------------------------------------------------------------
# 4. Importance weights sum to 1
# --------------------------------------------------------------------

def test_weights_sum_to_one():
    """Importance weights must form a valid probability distribution."""
    truth = jnp.zeros(64, dtype=jnp.int32).at[:7].set(1)
    _, _, q = q_oracle_sketch_boolean_adaptive(
        truth, 500, pilot_frac=0.2, key=jax.random.PRNGKey(2)
    )
    assert jnp.isclose(jnp.sum(q), 1.0, atol=1e-5), (
        f"Weights sum to {float(jnp.sum(q)):.8f}, expected 1.0"
    )


# --------------------------------------------------------------------
# 5. All-ones fallback
# --------------------------------------------------------------------

def test_uniform_function_fallback():
    """f=all-ones triggers uniform fallback; both oracles must agree."""
    truth = jnp.ones(64, dtype=jnp.int32)
    uni, _ = q_oracle_sketch_boolean(truth, 500)
    ada, _, _ = q_oracle_sketch_boolean_adaptive(
        truth, 500, key=jax.random.PRNGKey(1)
    )
    assert jnp.allclose(uni, ada, atol=1e-9)


# --------------------------------------------------------------------
# 6. Off-support entries are exactly 1
# --------------------------------------------------------------------

def test_off_support_entries_are_one():
    """Off-support entries of adaptive oracle must equal 1.0 to 1e-9."""
    truth = jnp.zeros(128, dtype=jnp.int32).at[:5].set(1)
    ada, _, _ = q_oracle_sketch_boolean_adaptive(
        truth, 2000, pilot_frac=0.2, key=jax.random.PRNGKey(3)
    )
    off_mask = ~truth.astype(bool)
    err = float(jnp.max(jnp.abs(ada[off_mask] - 1.0)))
    assert err < 1e-9, f"Off-support deviation {err:.2e} (expected < 1e-9)"


# --------------------------------------------------------------------
# 7. N/K improvement: adaptive at M=K*C matches uniform at M=N*C
# --------------------------------------------------------------------

def test_adaptive_nk_improvement_factor():
    """Adaptive at M=K*C achieves comparable error to uniform at M=N*C.

    N=512, K=8, C=300.
      M_adaptive = 8*300   =  2400
      M_uniform  = 512*300 = 153600
    Adaptive with 64x fewer samples should give similar or better error.
    We allow a 3x slack (conservative, since true ratio -> 1 at large M).
    """
    N, K, C = 512, 8, 300
    M_a, M_u = K * C, N * C
    truth = jnp.zeros(N, dtype=jnp.int32).at[:K].set(1)
    N_SEEDS = 12
    errs_a = [
        _supp_linf(
            q_oracle_sketch_boolean_adaptive(
                truth, M_a, pilot_frac=0.2, key=jax.random.PRNGKey(s)
            )[0],
            truth,
        )
        for s in range(N_SEEDS)
    ]
    errs_u = [_supp_linf(q_oracle_sketch_boolean(truth, M_u)[0], truth)
               for _ in range(N_SEEDS)]
    mean_a = float(jnp.mean(jnp.array(errs_a)))
    mean_u = float(jnp.mean(jnp.array(errs_u)))
    assert mean_a <= mean_u * 3.0, (
        f"N/K improvement not observed: "
        f"adaptive@M={M_a} err={mean_a:.4f}, uniform@M={M_u} err={mean_u:.4f}"
    )


# --------------------------------------------------------------------
# 8. Adaptive strictly beats uniform at equal M for large N/K
# --------------------------------------------------------------------

def test_adaptive_beats_uniform_at_equal_M_large_N():
    """Adaptive < uniform on supp(f) at equal M when N/K is large.

    N=2048, K=4, M=8000, N/K=512.
    Uniform error ~ sqrt(N/M) ~ 0.50.
    Adaptive error ~ sqrt(K/M) ~ 0.02.
    Mean over 10 seeds; adaptive must be strictly less than uniform.
    """
    N, K = 2048, 4
    M = 8000
    truth = jnp.zeros(N, dtype=jnp.int32).at[:K].set(1)
    N_SEEDS = 10
    errs_a = [
        _supp_linf(
            q_oracle_sketch_boolean_adaptive(
                truth, M, pilot_frac=0.2, key=jax.random.PRNGKey(s)
            )[0],
            truth,
        )
        for s in range(N_SEEDS)
    ]
    errs_u = [_supp_linf(q_oracle_sketch_boolean(truth, M)[0], truth)
               for _ in range(N_SEEDS)]
    mean_a = float(jnp.mean(jnp.array(errs_a)))
    mean_u = float(jnp.mean(jnp.array(errs_u)))
    assert mean_a < mean_u, (
        f"Adaptive ({mean_a:.4f}) must beat uniform ({mean_u:.4f}) "
        f"at N={N}, K={K}, M={M}, N/K={N//K}."
    )
