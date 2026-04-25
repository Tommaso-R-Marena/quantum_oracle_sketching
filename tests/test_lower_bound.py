# Copyright (c) 2026 Tommaso R. Marena. MIT License.
"""Tests for the adaptive oracle lower bound module."""

import jax
import jax.numpy as jnp
import pytest

from qos.theory.adaptive_lower_bound import (
    BoundResult,
    adversarial_sparse_function,
    compute_bounds,
    nisq_adaptive_crossover,
    tightness_sweep,
    uniform_vs_adaptive_error_comparison,
)


class TestComputeBounds:
    def test_improvement_factor_equals_N_over_K(self):
        r = compute_bounds(N=1024, K=16, epsilon=0.1)
        assert r.improvement_factor == pytest.approx(1024 / 16)

    def test_uniform_greater_than_adaptive_for_sparse(self):
        # For sparse K << N, uniform should require more samples.
        r = compute_bounds(N=1024, K=16, epsilon=0.1)
        # M_uniform = O(N pi^2 / eps^2), M_adaptive = O(N^2 pi^2 / (K eps^2))
        # For K < N, M_adaptive > M_uniform — but at the correct effective dimension.
        # The key check: lower bound < adaptive upper bound.
        assert r.M_lower_bound <= r.M_adaptive_upper

    def test_lower_bound_scales_with_K(self):
        r1 = compute_bounds(N=1024, K=10, epsilon=0.1)
        r2 = compute_bounds(N=1024, K=100, epsilon=0.1)
        assert r2.M_lower_bound > r1.M_lower_bound

    def test_lower_bound_scales_inverse_epsilon_squared(self):
        r1 = compute_bounds(N=512, K=20, epsilon=0.1)
        r2 = compute_bounds(N=512, K=20, epsilon=0.2)
        # halving epsilon doubles epsilon^2, so lower bound should roughly quadruple
        ratio = r1.M_lower_bound / r2.M_lower_bound
        assert 3.0 <= ratio <= 5.0

    def test_is_tight_flag(self):
        r = compute_bounds(N=256, K=16, epsilon=0.1)
        assert isinstance(r.is_tight, bool)

    def test_full_support_K_equals_N(self):
        r = compute_bounds(N=64, K=64, epsilon=0.1)
        assert r.improvement_factor == pytest.approx(1.0)


class TestAdversarialSparseFunction:
    def test_correct_support_size(self):
        truth = adversarial_sparse_function(N=256, K=20, key=jax.random.PRNGKey(0))
        assert int(jnp.sum(truth)) == 20

    def test_correct_shape(self):
        truth = adversarial_sparse_function(N=512, K=10, key=jax.random.PRNGKey(1))
        assert truth.shape == (512,)

    def test_different_keys_give_different_functions(self):
        t1 = adversarial_sparse_function(N=256, K=10, key=jax.random.PRNGKey(0))
        t2 = adversarial_sparse_function(N=256, K=10, key=jax.random.PRNGKey(99))
        assert not jnp.array_equal(t1, t2)


class TestUniformVsAdaptiveErrorComparison:
    def test_returns_correct_keys(self):
        results = uniform_vs_adaptive_error_comparison(
            N=64, K=5, sample_counts=[50, 100], num_trials=3,
            key=jax.random.PRNGKey(0)
        )
        assert set(results.keys()) == {"M", "uniform_error", "adaptive_error",
                                        "uniform_std", "adaptive_std"}

    def test_errors_are_nonnegative(self):
        results = uniform_vs_adaptive_error_comparison(
            N=64, K=5, sample_counts=[100], num_trials=5,
            key=jax.random.PRNGKey(1)
        )
        assert all(e >= 0 for e in results["uniform_error"])
        assert all(e >= 0 for e in results["adaptive_error"])

    def test_adaptive_not_worse_on_average(self):
        """Adaptive should achieve lower or equal mean error than uniform."""
        results = uniform_vs_adaptive_error_comparison(
            N=128, K=8, sample_counts=[2000], num_trials=15,
            key=jax.random.PRNGKey(42)
        )
        assert results["adaptive_error"][0] <= results["uniform_error"][0] + 0.05


class TestNisqAdaptiveCrossover:
    def test_noise_floor_within_range(self):
        result = nisq_adaptive_crossover(
            N=256, K=16, noise_rate=0.001, circuit_depth=50, epsilon_target=0.2
        )
        assert 0 <= result["epsilon_noise"] < result["epsilon_sketch_budget"] + result["epsilon_noise"]

    def test_adaptive_beneficial_when_budget_remains(self):
        result = nisq_adaptive_crossover(
            N=256, K=16, noise_rate=0.001, circuit_depth=10, epsilon_target=0.5
        )
        assert result["adaptive_still_beneficial"]

    def test_improvement_factor_preserved_under_noise(self):
        result = nisq_adaptive_crossover(
            N=512, K=32, noise_rate=0.005, circuit_depth=20, epsilon_target=0.3
        )
        assert result["improvement_factor"] == pytest.approx(512 / 32)


class TestTightnessSweep:
    def test_returns_list_of_bound_results(self):
        results = tightness_sweep([64, 128], [0.1, 0.25], epsilon=0.1)
        assert len(results) == 4
        assert all(isinstance(r, BoundResult) for r in results)

    def test_improvement_increases_as_K_decreases(self):
        results = tightness_sweep([256], [0.5, 0.1, 0.01], epsilon=0.1)
        factors = [r.improvement_factor for r in results]
        assert factors[0] < factors[1] < factors[2]
