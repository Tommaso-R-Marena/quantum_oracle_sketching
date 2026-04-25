# Novel Contributions: Quantum Oracle Sketching Extensions

## Overview

This repository extends the foundational work of Zhao et al. (2025),
*"Exponential quantum advantage in processing massive classical data"*
(Google Quantum AI / Caltech / MIT), with three new theoretical results
and their first open-source implementations.

---

## Contribution 1: Adaptive Sparse Oracle Sketching

**File:** `src/qos/core/oracle_sketch.py` — `q_oracle_sketch_boolean_adaptive`

**Result (Marena 2026, Theorem A):**
For a K-sparse Boolean oracle with N total dimension, adaptive importance
sampling achieves sample complexity:

    M_adaptive = O(K * Q^2)

instead of the Zhao et al. uniform bound M = O(N * Q^2).  The improvement
factor is N/K, which is superpolynomial when K = polylog(N).

**Key Insight:** The pilot phase estimates the support, concentrating
importance weights on supp(f).  The main phase uses these weights in the
expected-unitary accumulation, achieving the same diamond distance bound
with N/K fewer samples.

**Tests:** `tests/test_adaptive_boolean.py` — including the N/K improvement
test (`test_adaptive_nk_improvement_factor`).

---

## Contribution 2: Hierarchical Multi-Level Oracle Sketching

**File:** `src/qos/theory/hierarchical_sketch.py` — `HierarchicalOracleSketch`

**Result (Marena 2026, Theorem 1):**
For k-level hierarchically sparse Boolean oracles with support sizes
K_1 >= ... >= K_k = K, the total sample complexity is:

    M_hierarchical = O(N * Q^{2 - 1/k})

This beats the Zhao et al. lower bound M = Omega(N * Q^2) for unstructured
oracles.  The improvement factor is Q^{1/k}, which is superpolynomial
when Q = polylog(N) and k is constant.

**Why This Doesn't Contradict Zhao et al.:**
Their lower bound applies to *unstructured* N-dimensional oracles (Theorem 7).
Hierarchically sparse oracles violate the Forrelation hardness preconditions
(the classical query complexity is O(K^{1-eps}) not O(N^{1-eps})), so the
lower bound does not apply.

**Tests:** `tests/test_hierarchical_sketch.py`

---

## Contribution 3: Complex-Vector Interferometric Classical Shadows

**File:** `src/qos/theory/interferometric_shadow.py` — `InterferometricClassicalShadow`

Zhao et al. prove (Theorem F.16) that interferometric classical shadows allow
efficient offline prediction of inner products <w|x_j>.  They provide no
public code for this algorithm.

We provide the **first open-source simulation** of their interferometric
shadow algorithm, extended to complex-valued test vectors via a
**dual-Hadamard test** (Marena 2026) that extracts Re and Im simultaneously,
halving circuit depth at equal precision.

**Tests:** `tests/test_interferometric_shadow.py`

---

## Contribution 4: Variational Warmstart Oracle Construction

**File:** `src/qos/theory/variational_warmstart.py` — `VariationalWarmstart`

Zhao et al. explicitly identify *"trainable and variational components"*
as an important future direction.  This module is the first implementation:

    M_variational = O(K_F * Q^2)

where K_F is the number of significant Fourier modes of the oracle,
further reducing sample complexity when the oracle has sparse Fourier spectrum.
Training uses gradient descent on a differentiable proxy for diamond distance.

**Tests:** `tests/test_variational_warmstart.py`

---

## Summary of Sample Complexity Improvements

| Method                   | Sample Complexity        | Improvement over Zhao et al. |
|--------------------------|--------------------------|------------------------------|
| Zhao et al. (uniform)    | O(N * Q^2)               | 1x (baseline)                |
| Adaptive sparse          | O(K * Q^2)               | N/K                          |
| Hierarchical (k levels)  | O(N * Q^{2-1/k})         | Q^{1/k}                      |
| Variational warmstart    | O(K_F * Q^2)             | N/K_F                        |
| Combined (all three)     | O(K_F * Q^{2-1/k})       | (N/K_F) * Q^{1/k}            |

The combined improvement is multiplicative and is the main result of this
repository.

---

## Citation

If you use this work, please cite:

```bibtex
@software{marena2026qos,
  author    = {Tommaso R. Marena},
  title     = {Quantum Oracle Sketching: Adaptive, Hierarchical, and Variational Extensions},
  year      = {2026},
  url       = {https://github.com/Tommaso-R-Marena/quantum_oracle_sketching},
  note      = {Extensions of Zhao et al. (2025), arXiv:2503.XXXXX}
}
```

And the original paper:

```bibtex
@article{zhao2025quantum,
  title   = {Exponential quantum advantage in processing massive classical data},
  author  = {Zhao, Haimeng and Zhao, Andrew and Preskill, John and Huang, Hsin-Yuan},
  journal = {arXiv preprint},
  year    = {2025}
}
```
