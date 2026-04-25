"""Marena 2026 novel theoretical extensions to Quantum Oracle Sketching.

Public API:
    HierarchicalOracleSketch    -- Q^{2-1/k} sample complexity (Contribution 2)
    InterferometricClassicalShadow -- first open-source interferometric shadow (Contribution 4)
    VariationalWarmstart        -- Fourier-sparse variational oracle (Contribution 3)

Example::

    from qos.theory import HierarchicalOracleSketch, VariationalWarmstart

    sketch = HierarchicalOracleSketch.from_truth_table(
        truth_table, num_levels=3, total_queries=16, seed=0
    )
    diag, stats = sketch.build()

    vw = VariationalWarmstart(
        truth_table, num_fourier_modes=32, learning_rate=0.02, num_steps=100
    )
    vw.fit(unit_num_samples=2000)
    diag = vw.predict()
"""

from __future__ import annotations

from qos.theory.hierarchical_sketch import HierarchicalOracleSketch
from qos.theory.interferometric_shadow import InterferometricClassicalShadow
from qos.theory.variational_warmstart import VariationalWarmstart

__all__ = [
    "HierarchicalOracleSketch",
    "InterferometricClassicalShadow",
    "VariationalWarmstart",
]
