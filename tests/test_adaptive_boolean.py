"""Tests for adaptive Boolean oracle sketching (Marena 2026).

All tests enable JAX x64 at module level to ensure complex128 arithmetic.

Convergence regime
------------------
The adaptive oracle formula (Zhao generalised with p=1/K)::

    diag[x] = exp(M_main * log(1 + (1/K) * expm1(i * pi*K/M_main * f(x))))

converges to exp(i*pi*f) as M_main -> inf.  Convergence threshold: M_main >> pi*K.
Error on supp(f): O(sqrt(K/M_main)).  Sample improvement over uniform: N/K.

All test parameters are chosen so that M >> pi*K, placing both oracles
firmly in the convergent small-angle regime.
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
            f"err[M={Ms[i]}]={a:.6f} >= err[M={Ms[i+1]}]={b:.6f}."
        )


# ------------------------------------------------------------------
# 2. Adaptive oracle converges (error strictly decreasing with M)
# ------------------------------------------------------------------

def test_adaptive_oracle_converges():
    """Adaptive oracle error strictly decreases as M grows.

    Uses exact q=1/K weights (deterministic), so no seed averaging needed.
    N=16, K=4, convergence threshold M_main >> pi*K ~ 12.6.
    All M values in [16..1024] satisfy this; error must strictly decrease.
    """
    N, K = 16, 4
    truth = jnp.zeros(N, dtype=jnp.int32).at[:K].set(1)
    # M_main = M * (1 - pilot_frac) = M * 0.8; threshold pi*K ~ 12.6
    # M=20 -> M_main=16 >> 12.6 barely; use M=64 as minimum for safety
    Ms = [64, 256, 1024, 4096]
    errs = [
        _supp_linf(
            q_oracle_sketch_boolean_adaptive(
                truth, M, pilot_frac=0.2, key=jax.random.PRNGKey(0)
            )[0], truth
        )
        for M in Ms
    ]
    for i, (a, b) in enumerate(zip(errs, errs[1:])):
        assert b < a, (
            f"Adaptive error not strictly decreasing at step {i}: "
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
    """Adaptive oracle reaches target error with N/K fewer samples than uniform.

    Setup: N=64, K=4.  Both oracles are in the convergent regime (M >> pi*N=201).

    Uniform error formula: O(sqrt(N/M)).  To reach eps=0.05: M_u ~ N/eps^2 = 25600.
    Adaptive error formula: O(sqrt(K/M_main)).  Same eps: M_a ~ K/eps^2 = 1600.
    Ratio: M_u/M_a = N/K = 16.  We test a conservative 4x improvement.

    Adaptive uses exact q=1/K weights so the result is deterministic
    (no seed variance); we average over seeds only for robustness.
    """
    N, K = 64, 4
    truth = jnp.zeros(N, dtype=jnp.int32).at[:K].set(1)
    TARGET = 0.05
    N_SEEDS = 8

    # Find smallest M_uniform achieving TARGET
    M_uniform = None
    for M in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        err = _supp_linf(q_oracle_sketch_boolean(truth, M)[0], truth)
        if err < TARGET:
            M_uniform = M
            break
    assert M_uniform is not None, "Uniform oracle failed to reach target in tested range"

    # Adaptive must reach TARGET at <= M_uniform // 4  (conservative 4x; theory says 16x)
    M_adaptive = max(M_uniform // 4, 64)   # floor at 64 to stay above convergence threshold
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

    N=64, K=2, M=2048.  Both oracles in convergent regime (M >> pi*N=201).
    Adaptive uses exact q=1/K; result is deterministic.
    """
    N, K = 64, 2
    M     = 2048
    truth = jnp.zeros(N, dtype=jnp.int32).at[:K].set(1)
    N_SEEDS = 4   # averaging for robustness; actually deterministic

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
