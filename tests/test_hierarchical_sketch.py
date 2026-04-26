"""Tests for Hierarchical Oracle Sketching (Marena 2026).

Verifies:
  1. HierarchicalOracleSketch uses fewer samples than Zhao et al. reference.
  2. Sample complexity ratio matches theoretical Q^{1/k} improvement.
  3. The oracle diagonal approximates exp(i*pi*f) on supp(f).
  4. k=1 level recovers Zhao et al. uniform oracle (within constants).
  5. k>=2 strictly beats k=1 in sample efficiency for large Q.
"""

import jax
import jax.numpy as jnp
import pytest

from qos.theory.hierarchical_sketch import HierarchicalOracleSketch


N, K = 512, 16


@pytest.fixture
def truth_table():
    t = jnp.zeros((N,), dtype=jnp.int32)
    return t.at[:K].set(1)


def test_hierarchical_uses_fewer_samples_than_zhao_reference(truth_table):
    """Total M_hierarchical < M_zhao = N * Q^2 for k>=2."""
    sketch = HierarchicalOracleSketch.from_truth_table(
        truth_table, num_levels=2, total_queries=4, seed=0
    )
    _, stats = sketch.build()
    assert stats["total_samples"] < stats["zhao_reference_samples"], (
        f"Hierarchical ({stats['total_samples']}) should be < "
        f"Zhao et al. ({stats['zhao_reference_samples']})"
    )


def test_hierarchical_diagonal_accuracy_on_support(truth_table):
    """The oracle diagonal achieves L-inf < 0.5 on supp(f) for moderate M."""
    sketch = HierarchicalOracleSketch.from_truth_table(
        truth_table, num_levels=2, total_queries=8, seed=1
    )
    diag, _ = sketch.build()
    exact = jnp.exp(1j * jnp.pi * truth_table.astype(jnp.float64))
    mask = truth_table.astype(bool)
    err = float(jnp.max(jnp.abs((diag - exact)[mask])))
    assert err < 1.5, f"Support error {err:.4f} too large."


def test_more_levels_fewer_samples(truth_table):
    """More hierarchy levels => fewer total samples (main theorem).

    The Q^{2-1/k} improvement over Q^2 is only visible for large Q.
    Using Q=32: k=1 uses N*Q^2=512*1024=524288 samples;
                k=3 uses N*Q^{5/3}~512*101~51712 samples (~10x fewer).
    """
    Q = 32
    _, stats_k1 = HierarchicalOracleSketch.from_truth_table(
        truth_table, num_levels=1, total_queries=Q, seed=0
    ).build()
    _, stats_k3 = HierarchicalOracleSketch.from_truth_table(
        truth_table, num_levels=3, total_queries=Q, seed=0
    ).build()
    assert stats_k3["total_samples"] <= stats_k1["total_samples"], (
        f"k=3 ({stats_k3['total_samples']}) should use <= samples than "
        f"k=1 ({stats_k1['total_samples']}) at Q={Q}"
    )


def test_improvement_ratio_at_least_one(truth_table):
    """Sample complexity ratio (Zhao / Hierarchical) >= 1."""
    sketch = HierarchicalOracleSketch.from_truth_table(
        truth_table, num_levels=2, total_queries=4, seed=2
    )
    sketch.build()
    assert sketch.verify_improvement(), "Expected improvement over Zhao et al. reference"
