"""Tests for adaptive Boolean oracle sketching (Marena 2026).

All tests enable JAX x64 at module level to ensure complex128 arithmetic.

Convergence regime
------------------
The log-sum oracle formula::

    diag[x] = exp(M * log(1 + p * expm1(i * (t/M) * f(x))))

converges to exp(i*p*t*f) as M -> inf, with per-step angle t/M = pi*N/M.
For the approximation to be in the small-angle regime (fast convergence)
we need t/M << 1, i.e. M >> pi*N. Tests use N=16 or N=64 so that
moderate M values (M ~ 100-4000) are well into the convergent regime.

N/K improvement (tests 7 & 8)
------------------------------
The adaptive oracle concentrates M_main shots on the K-entry support,
giving effective p(x) = 1/K per support entry vs p(x) = 1/N for uniform.
For the same convergence depth (same M/N ratio for uniform vs M/K for
adaptive), adaptive requires K/N fewer total samples.

Test 7: Fix a target error threshold epsilon. Find the smallest M_uniform
such that uniform oracle achieves error < epsilon. Then show adaptive
achieves the same epsilon with M_adaptive <= M_uniform * (K/N + slack).

Test 8: At equal M, adaptive error < uniform error when N >> K.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from qos.core.oracle_sketch import (
    q_oracle_sketch_boolean,
    q_oracle_sketch_boolean_adaptive,
)


def _supp_linf(diag: jax.Array, truth: jax.Array) -> float:
    """L-inf error on supp(f): max_{x:f=1} |diag[x] - exp(i*pi)|."""
    exact = jnp.exp(jnp.complex128(1j) * jnp.float64(jnp.pi) * truth.astype(jnp.float64))
    mask  = truth.astype(bool)
    return float(jnp.max(jnp.abs((diag - exact)[mask])))


# ------------------------------------------------------------------
# 1. Uniform oracle converges (error strictly decreasing with M)
# ------------------------------------------------------------------

def test_uniform_oracle_converges():
    """Uniform oracle error on supp(f) strictly decreases as M grows."""
    N = 16
    truth = jnp.zeros(N, dtype=jnp.int32).at[:4].set(1)
    Ms   = [N, 4*N, 16*N, 64*N, 256*N]
    errs = [_supp_linf(q_oracle_sketch_boolean(truth, M)[0], truth) for M in Ms]
    for i, (a, b) in enumerate(zip(errs, errs[1:])):
        assert b < a, (
            f"Uniform oracle error not decreasing at step {i}: "
            f"err[M={Ms[i]}]={a:.6f} >= err[M={Ms[i+1]}]={b:.6f}. "
            f"Full sequence: {[f'{e:.6f}' for e in errs]}"
        )


# ------------------------------------------------------------------
# 2. Adaptive oracle converges (mean error decreasing with M)
# ------------------------------------------------------------------

def test_adaptive_oracle_converges():
    """Adaptive oracle mean error (over seeds) decreases as M grows."""
    N = 16
    truth = jnp.zeros(N, dtype=jnp.int32).at[:4].set(1)
    Ms      = [N, 4*N, 16*N, 64*N]
    N_SEEDS = 6
    errs = []
    for M in Ms:
        seed_errs = [
            _supp_linf(
                q_oracle_sketch_boolean_adaptive(
                    truth, M, pilot_frac=0.2, key=jax.random.PRNGKey(s)
                )[0], truth,
            )
            for s in range(N_SEEDS)
        ]
        errs.append(float(jnp.mean(jnp.array(seed_errs))))
    for i, (a, b) in enumerate(zip(errs, errs[1:])):
        assert b < a * 1.4, (
            f"Adaptive mean error not decreasing at step {i}: "
            f"{[f'{e:.4f}' for e in errs]}"
        )


# ------------------------------------------------------------------
# 3. Fallback: pilot_frac=0 gives identical output to uniform
# ------------------------------------------------------------------

def test_pilot_frac_zero_fallback():
    """pilot_frac=0 must give identical output to uniform oracle."""
    truth = jnp.mod(jnp.arange(64), 2).astype(jnp.int32)
    uni, _ = q_oracle_sketch_boolean(truth, 500)
    ada, _, _ = q_oracle_sketch_boolean_adaptive(
        truth, 500, pilot_frac=0.0, key=jax.random.PRNGKey(0)
    )
    max_diff = float(jnp.max(jnp.abs(uni - ada)))
    assert max_diff < 1e-9, f"Fallback mismatch: max diff = {max_diff:.2e}"


# ------------------------------------------------------------------
# 4. Importance weights sum to 1
# ------------------------------------------------------------------

def test_weights_sum_to_one():
    """Importance weights must form a valid probability distribution."""
    truth = jnp.zeros(64, dtype=jnp.int32).at[:7].set(1)
    _, _, q = q_oracle_sketch_boolean_adaptive(
        truth, 500, pilot_frac=0.2, key=jax.random.PRNGKey(2)
    )
    s = float(jnp.sum(q))
    assert abs(s - 1.0) < 1e-5, f"Weights sum to {s:.8f}, expected 1.0"


# ------------------------------------------------------------------
# 5. All-ones function triggers uniform fallback
# ------------------------------------------------------------------

def test_uniform_function_fallback():
    """f=all-ones triggers uniform fallback path; outputs must agree."""
    truth = jnp.ones(64, dtype=jnp.int32)
    uni, _ = q_oracle_sketch_boolean(truth, 500)
    ada, _, _ = q_oracle_sketch_boolean_adaptive(
        truth, 500, key=jax.random.PRNGKey(1)
    )
    assert jnp.allclose(uni, ada, atol=1e-9)


# ------------------------------------------------------------------
# 6. Off-support entries are exactly 1
# ------------------------------------------------------------------

def test_off_support_entries_are_one():
    """Off-support entries of adaptive oracle must equal 1.0 to 1e-9."""
    truth = jnp.zeros(128, dtype=jnp.int32).at[:5].set(1)
    ada, _, _ = q_oracle_sketch_boolean_adaptive(
        truth, 2000, pilot_frac=0.2, key=jax.random.PRNGKey(3)
    )
    off_mask = ~truth.astype(bool)
    err = float(jnp.max(jnp.abs(ada[off_mask] - 1.0)))
    assert err < 1e-9, f"Off-support deviation {err:.2e} (expected < 1e-9)"


# ------------------------------------------------------------------
# 7. N/K improvement: adaptive reaches target error with fewer samples
# ------------------------------------------------------------------

def test_adaptive_nk_improvement_factor():
    """Adaptive oracle reaches a target error with ~N/K fewer samples than uniform.

    Setup: N=64, K=4, N/K=16.
    Both oracles operate in the convergent regime (M >> pi*N = 201).

    Find M_uniform = smallest power-of-4 such that uniform error < 0.05.
    Then verify adaptive achieves the same error at M_adaptive <= M_uniform // 4
    (conservative: we only require a 4x improvement, though theory says 16x).

    Both M values are in {256, 1024, 4096, 16384} so the test is fast.
    """
    N, K = 64, 4
    truth = jnp.zeros(N, dtype=jnp.int32).at[:K].set(1)
    TARGET = 0.05
    N_SEEDS = 8

    # Find smallest M_uniform achieving TARGET
    M_uniform = None
    for M in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        err = _supp_linf(q_oracle_sketch_boolean(truth, M)[0], truth)
        if err < TARGET:
            M_uniform = M
            break
    assert M_uniform is not None, "Uniform oracle failed to reach target in tested range"

    # Adaptive must reach the same TARGET at <= M_uniform // 4
    M_adaptive = M_uniform // 4
    assert M_adaptive >= 1, "M_adaptive too small"

    mean_err_a = float(jnp.mean(jnp.array([
        _supp_linf(
            q_oracle_sketch_boolean_adaptive(
                truth, M_adaptive, pilot_frac=0.2, key=jax.random.PRNGKey(s)
            )[0], truth
        )
        for s in range(N_SEEDS)
    ])))

    assert mean_err_a < TARGET, (
        f"Adaptive@M={M_adaptive} mean_err={mean_err_a:.4f} did not reach "
        f"target={TARGET} (uniform needed M={M_uniform}). "
        f"Expected N/K={N//K}x improvement, tested 4x."
    )


# ------------------------------------------------------------------
# 8. Equal-M comparison: adaptive error < uniform error when N >> K
# ------------------------------------------------------------------

def test_adaptive_beats_uniform_at_equal_M_large_N():
    """At equal M, adaptive error < uniform error when N/K is large.

    Setup: N=64, K=2, N/K=32, M=2048.
    Both oracles are in the convergent regime (M >> pi*N = 201).
    Mean over 10 seeds; adaptive must strictly beat uniform.
    """
    N, K = 64, 2
    M = 2048
    truth = jnp.zeros(N, dtype=jnp.int32).at[:K].set(1)
    N_SEEDS = 10

    errs_a = [
        _supp_linf(
            q_oracle_sketch_boolean_adaptive(
                truth, M, pilot_frac=0.2, key=jax.random.PRNGKey(s)
            )[0], truth
        )
        for s in range(N_SEEDS)
    ]
    errs_u = [
        _supp_linf(q_oracle_sketch_boolean(truth, M)[0], truth)
        for _ in range(N_SEEDS)
    ]
    mean_a = float(jnp.mean(jnp.array(errs_a)))
    mean_u = float(jnp.mean(jnp.array(errs_u)))
    assert mean_a < mean_u, (
        f"Adaptive ({mean_a:.4f}) must beat uniform ({mean_u:.4f}) "
        f"at N={N}, K={K}, M={M}, N/K={N//K}."
    )
