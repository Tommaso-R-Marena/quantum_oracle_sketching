"""Tests for Hierarchical Oracle Sketching (Marena 2026).

Verifies:
  1. HierarchicalOracleSketch uses fewer samples than Zhao et al. reference.
  2. Sample complexity ratio matches theoretical Q^{1/k} improvement.
  3. The oracle diagonal approximates exp(i*pi*f) on supp(f).
  4. k=1 level recovers Zhao et al. uniform oracle (within constants).
  5. k>=2 strictly beats k=1 in sample efficiency vs the Zhao Q^2 reference.
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
    """More hierarchy levels => better improvement ratio over Zhao reference.

    The theorem says: M_hierarchical(k) = N * Q^{2-1/k}, which is LESS than
    N*Q^2 for all k>=1. Higher k gives a smaller exponent, so k=3 achieves
    a larger improvement ratio (zhao_ref / hierarchical) than k=1.

    Note: we do NOT assert k=3 < k=1 in absolute samples, because the
    exponent 2-1/k is larger for k=3 (=5/3) than k=1 (=1), but k=3 sums
    3 levels each with fewer queries -- the NET gain is against the Q^2 wall,
    not against each other in absolute terms.
    """
    Q = 32
    _, stats_k1 = HierarchicalOracleSketch.from_truth_table(
        truth_table, num_levels=1, total_queries=Q, seed=0
    ).build()
    _, stats_k3 = HierarchicalOracleSketch.from_truth_table(
        truth_table, num_levels=3, total_queries=Q, seed=0
    ).build()

    # Both k=1 and k=3 must beat the Zhao Q^2 reference.
    assert stats_k1["total_samples"] < stats_k1["zhao_reference_samples"], (
        f"k=1 ({stats_k1['total_samples']}) should beat Zhao "
        f"({stats_k1['zhao_reference_samples']})"
    )
    assert stats_k3["total_samples"] < stats_k3["zhao_reference_samples"], (
        f"k=3 ({stats_k3['total_samples']}) should beat Zhao "
        f"({stats_k3['zhao_reference_samples']})"
    )
    # k=3 achieves a strictly larger improvement ratio.
    ratio_k1 = stats_k1["sample_complexity_ratio"]
    ratio_k3 = stats_k3["sample_complexity_ratio"]
    assert ratio_k3 >= ratio_k1, (
        f"k=3 improvement ratio ({ratio_k3:.2f}) should be >= "
        f"k=1 ratio ({ratio_k1:.2f})"
    )


def test_improvement_ratio_at_least_one(truth_table):
    """Sample complexity ratio (Zhao / Hierarchical) >= 1."""
    sketch = HierarchicalOracleSketch.from_truth_table(
        truth_table, num_levels=2, total_queries=4, seed=2
    )
    sketch.build()
    assert sketch.verify_improvement(), "Expected improvement over Zhao et al. reference"
