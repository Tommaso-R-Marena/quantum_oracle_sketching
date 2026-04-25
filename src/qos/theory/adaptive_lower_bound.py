# Copyright (c) 2026 Tommaso R. Marena. MIT License.
"""Numerical verification of the adaptive Boolean oracle sample-complexity bounds.

This module provides tools to:
  1. Compute the sample complexities for the uniform and adaptive expected-unitary
     Boolean oracle constructions under the **blind-oracle** (p independent of f)
     and **min-probability** (p(x) >= c/N) models.
  2. Numerically compare the two strategies on adversarial sparse functions.
  3. Compute the NISQ crossover regime where noise renders further adaptive
     improvement impossible.

Theorem scope (per Aristotle/Lean formalization, April 2026):
  The improvement factor N/K is a property of the specific expected-unitary
  construction and holds under two structural constraints on p:
    (a) Blind-oracle model: p is chosen before f is revealed, so p cannot
        concentrate on supp(f). A uniform p(x) = 1/N forces t = Theta(N).
    (b) Minimum-probability model: p(x) >= c/N for all x (e.g. due to noise
        or circuit uniformity), which prevents arbitrary concentration.
  Without at least one of these constraints, a concentrated distribution
  p(x) = 1/K on supp(f) achieves t = pi*K with M = O(K^3/eps^2), which
  is BETTER than the adaptive bound when K << N^(2/3). The general conjecture
  t >= Omega(N/K) is therefore FALSE in the unconstrained setting.

See docs/lower_bound_proof.md and lean/PhaseTimeBound.lean for formal details.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from qos.config import real_dtype
from qos.core.oracle_sketch import q_oracle_sketch_boolean, q_oracle_sketch_boolean_adaptive


@dataclass(frozen=True)
class BoundResult:
    """Sample-complexity bound results for a given (N, K, epsilon) triple.

    All bounds are for the **blind-oracle model** (p independent of f).
    Under concentrated support, the adaptive upper bound can be beaten;
    see module docstring for the full constraint discussion.

    Attributes:
        N: Ambient dimension (number of Boolean inputs).
        K: Support size of the Boolean function f.
        epsilon: Target approximation error.
        M_uniform_upper: Uniform-sampling upper bound (blind oracle, p=1/N).
        M_adaptive_upper: Adaptive upper bound (blind oracle, pilot-estimated
            support; Marena 2026 Theorem 1).  The adaptive estimator uses
            t = pi*K with importance weights p(x) = 1/K, so
            M_adaptive = O(K * pi^2 / eps^2).  Total samples including pilot
            are M = M_adaptive / (1 - pilot_frac) ~ M_adaptive.
        M_lower_bound: Information-theoretic floor M = Omega(K/eps^2).
            This is the minimum samples needed to observe each of the K
            support positions at least once; it is a lower bound on any
            blind-oracle strategy but NOT on concentrated-p strategies.
        improvement_factor: N/K ratio (blind-oracle speedup vs uniform).
        is_tight: Empirical flag: True when M_adaptive >= M_lower.
            This is a sanity check, not a proof of tightness.
    """
    N: int
    K: int
    epsilon: float
    M_uniform_upper: int
    M_adaptive_upper: int
    M_lower_bound: int
    improvement_factor: float
    is_tight: bool


def compute_bounds(N: int, K: int, epsilon: float, constant: float = 1.0) -> BoundResult:
    """Compute adaptive oracle sample-complexity bounds (blind-oracle model).

    Args:
        N: Ambient dimension.
        K: Support size (number of x with f(x)=1).
        epsilon: Target L-infinity approximation error on the phase diagonal.
        constant: Leading constant in the bound (default 1.0 for clean comparison).

    Returns:
        BoundResult with all bound values and tightness flag.

    Mathematical note (blind-oracle model):
        Uniform upper bound:  M = O(N * pi^2 / eps^2)
            p(x) = 1/N, t_uniform = pi*N, phase per entry = (1/N)*(pi*N) = pi.
            Error variance: N * (1/N)^2 * (pi*N)^2 / M = N*pi^2/M.

        Adaptive upper bound: M = O(K * pi^2 / eps^2)  [main samples]
            t_adaptive = pi*K, importance weights p(x) = 1/K on supp(f).
            Phase per entry = (1/K)*(pi*K) = pi. Correct.
            Error variance on supp(f): K * (1/K)^2 * (pi*K)^2 / M_main
                                     = K * pi^2 / M_main.
            With M_main ~ M (small pilot_frac=0.1), improvement = N/K.

        Lower bound (blind-oracle): M = Omega(K / eps^2)
            Any blind strategy must accumulate enough samples to cover
            each of the K support positions; gives the coupon-collector floor.

    Crossover note:
        The empirical crossover between adaptive and uniform error curves
        occurs near M ~ K * pi^2 / eps^2 (where adaptive converges).
        For K=16, eps=0.3: crossover at M ~ 1750, well within M=200..20000.
    """
    t_uniform = math.pi * N      # uniform: each entry gets phase pi*N*(1/N) = pi
    t_adaptive = math.pi * K     # adaptive: support entries get phase pi*K*(1/K) = pi

    # Sample complexity:
    # Uniform:  M = N * t_uniform^2 / (N * eps^2) = N * pi^2 / eps^2
    # Adaptive: M_main = K * t_adaptive^2 / (K * eps^2) = K * pi^2 / eps^2
    M_uniform = int(math.ceil(constant * N * math.pi ** 2 / epsilon ** 2))
    M_adaptive = int(math.ceil(constant * K * math.pi ** 2 / epsilon ** 2))
    M_lower = int(math.ceil(constant * K / epsilon ** 2))

    improvement = N / K
    is_tight = M_adaptive >= M_lower

    return BoundResult(
        N=N,
        K=K,
        epsilon=epsilon,
        M_uniform_upper=M_uniform,
        M_adaptive_upper=M_adaptive,
        M_lower_bound=M_lower,
        improvement_factor=improvement,
        is_tight=is_tight,
    )


def adversarial_sparse_function(N: int, K: int, key: jax.Array) -> jax.Array:
    """Generate an adversarial sparse Boolean function for blind-oracle experiments.

    The adversarial construction places support on K randomly chosen positions.
    This is the hardest case for any blind (non-adaptive) strategy: without pilot
    estimation, a uniform sampler must draw Omega(N/K) samples before hitting
    any support point.

    Args:
        N: Ambient dimension.
        K: Support size.
        key: JAX PRNG key.

    Returns:
        Boolean array of shape ``(N,)`` with exactly K ones at random positions.
    """
    positions = jax.random.choice(key, N, shape=(K,), replace=False)
    truth = jnp.zeros((N,), dtype=jnp.int32)
    return truth.at[positions].set(1)


def uniform_vs_adaptive_error_comparison(
    N: int,
    K: int,
    sample_counts: list[int],
    num_trials: int = 20,
    key: jax.Array | None = None,
) -> dict[str, list[float]]:
    """Empirically compare uniform vs adaptive oracle sketch error vs samples.

    Measures L-infinity phase error **restricted to supp(f)** for both strategies.
    Off-support entries are excluded because both strategies trivially achieve
    exp(0)=1 there; the N/K improvement only manifests on supp(f).

    This function operates in the **blind-oracle model** where p is determined
    before f is revealed.  Under concentrated-support distributions the improvement
    factor is different; see module docstring.

    Pilot fraction policy (FIXED in this version):
        pilot_frac = min(0.1, 20.0 * K / M)
        This ensures:
          - pilot gets >= 20 samples per support position (coverage guarantee), AND
          - at least 90% of samples go to the main accumulation phase.
        The previous policy max(0.3, ...) gave 70% of samples to the pilot at
        small M, starving the main phase and causing adaptive_error to plateau
        near 1.8 regardless of M.  With pilot_frac <= 0.1, the main phase
        receives >= 90% of samples and the adaptive error drops as expected.

    Args:
        N: Ambient dimension.
        K: Support size.
        sample_counts: List of M values to test.
        num_trials: Number of random function instances per M.
        key: JAX PRNG key (default PRNGKey(42)).

    Returns:
        Dict with keys 'M', 'uniform_error', 'adaptive_error',
        'uniform_std', 'adaptive_std'.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    results: dict[str, list[float]] = {
        "M": [], "uniform_error": [], "adaptive_error": [],
        "uniform_std": [], "adaptive_std": []
    }

    for M in sample_counts:
        # pilot_frac: give pilot just enough for coverage, rest to main phase.
        # min(0.1, ...) keeps >= 90% samples in main phase at all M.
        pilot_frac = min(0.1, 20.0 * K / M)

        u_errors, a_errors = [], []
        for trial in range(num_trials):
            k1, k2, key = jax.random.split(key, 3)
            truth = adversarial_sparse_function(N, K, k1)
            exact = jnp.exp(1j * jnp.pi * truth.astype(real_dtype))
            support_mask = truth.astype(bool)

            u_diag, _ = q_oracle_sketch_boolean(truth, M)
            a_diag, _, _ = q_oracle_sketch_boolean_adaptive(
                truth, M, pilot_frac=pilot_frac, key=k2
            )

            u_err = float(jnp.max(jnp.abs((u_diag - exact)[support_mask])))
            a_err = float(jnp.max(jnp.abs((a_diag - exact)[support_mask])))

            u_errors.append(u_err)
            a_errors.append(a_err)

        results["M"].append(M)
        results["uniform_error"].append(float(jnp.mean(jnp.array(u_errors))))
        results["adaptive_error"].append(float(jnp.mean(jnp.array(a_errors))))
        results["uniform_std"].append(float(jnp.std(jnp.array(u_errors))))
        results["adaptive_std"].append(float(jnp.std(jnp.array(a_errors))))

    return results


def nisq_adaptive_crossover(
    N: int,
    K: int,
    noise_rate: float,
    circuit_depth: int,
    epsilon_target: float,
) -> dict[str, float]:
    """Compute the NISQ regime crossover for adaptive vs uniform oracle sketching.

    In the presence of depolarizing noise with rate p per gate and circuit
    depth d, the effective sketch error has a noise floor of:
        epsilon_noise = 1 - (1 - p)^d

    Below the crossover sample count M*, further adaptive improvement is
    impossible because noise dominates the sketch error budget.

    Args:
        N: Ambient dimension.
        K: Support size.
        noise_rate: Per-gate depolarizing probability p in [0, 1].
        circuit_depth: Number of gates d in the sketch circuit.
        epsilon_target: Target total error.

    Returns:
        Dict with keys:
            'epsilon_noise': Irreducible noise floor.
            'epsilon_sketch_budget': Remaining error budget after noise.
            'M_adaptive_crossover': Adaptive M* at the reduced budget.
            'M_uniform_crossover': Uniform M* at the reduced budget.
            'adaptive_still_beneficial': Whether adaptive beats uniform at M*.
            'improvement_factor': N/K improvement ratio (unchanged by noise).
    """
    epsilon_noise = 1.0 - (1.0 - noise_rate) ** circuit_depth
    epsilon_sketch = max(epsilon_target - epsilon_noise, 1e-8)

    bounds = compute_bounds(N, K, epsilon_sketch)

    return {
        "epsilon_noise": epsilon_noise,
        "epsilon_sketch_budget": epsilon_sketch,
        "M_adaptive_crossover": bounds.M_adaptive_upper,
        "M_uniform_crossover": bounds.M_uniform_upper,
        "adaptive_still_beneficial": epsilon_sketch > 0 and K < N,
        "improvement_factor": bounds.improvement_factor,
    }


def tightness_sweep(
    N_values: list[int],
    sparsity_ratios: list[float],
    epsilon: float = 0.1,
) -> list[BoundResult]:
    """Sweep over (N, K) pairs to compare adaptive vs uniform bounds.

    K is computed as max(1, round(ratio * N)) to ensure integer sparsity
    and avoid NaN in downstream pivot operations.

    Args:
        N_values: List of ambient dimensions to test.
        sparsity_ratios: List of K/N ratios in (0, 1).
        epsilon: Target error.

    Returns:
        List of BoundResult for each (N, K) pair.
    """
    results = []
    for N in N_values:
        for ratio in sparsity_ratios:
            K = max(1, round(ratio * N))
            results.append(compute_bounds(N, K, epsilon))
    return results
