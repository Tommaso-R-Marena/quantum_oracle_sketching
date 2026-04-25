# Quantum Oracle Sketching (QOS)

**Extending quantum advantage in processing massive classical data**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/quantum_oracle_sketching/blob/main/notebooks/quantum_oracle_sketching_demo.ipynb)
[![Paper](https://img.shields.io/badge/arXiv-2604.07639-B31B1B.svg)](https://arxiv.org/abs/2604.07639)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-compatible-green.svg)](https://jax.readthedocs.io)
[![CI](https://github.com/Tommaso-R-Marena/quantum_oracle_sketching/actions/workflows/ci.yml/badge.svg)](https://github.com/Tommaso-R-Marena/quantum_oracle_sketching/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![v1.3.1](https://img.shields.io/badge/version-1.3.1-informational)](CHANGELOG.md)

This repository implements **Quantum Oracle Sketching (QOS)** and introduces four novel theoretical extensions (Marena 2026) that surpass the baseline of Zhao et al. (2025/2026) in sample complexity, circuit depth, and robustness.

> **Quick demo →** click the Colab badge above to run all experiments in your browser with zero setup.

---

## Novel Contributions (Marena 2026)

| # | Contribution | Sample Complexity | Improvement over Zhao et al. |
|---|---|---|---|
| — | Zhao et al. uniform baseline | O(N · Q²) | 1× |
| 1 | **Adaptive sparse oracle** | O(K · Q²) | **N/K** |
| 2 | **Hierarchical sketching** | O(N · Q^{2−1/k}) | **Q^{1/k}** |
| 3 | **Variational warmstart** | O(K_F · Q²) | **N/K_F** |
| 4 | **Interferometric classical shadow** | O(s · T⁻¹) | first open-source impl. |
| ★ | **Combined** | O(K_F · Q^{2−1/k}) | **(N/K_F) · Q^{1/k}** |

See [`docs/theory.md`](docs/theory.md) for full theorems, proofs sketches, and connections to Forrelation lower bounds.

---

## Overview

**QOS** is a quantum algorithm for loading classical data into a quantum computer. It instantiates the oracles needed by any quantum query algorithm using only random classical samples, with no full-dataset memory overhead. This codebase includes:

- **Core QOS** in JAX (GPU/TPU-compatible, differentiable).
- **QSVT utilities**: amplitude amplification, matrix inversion, thresholding, polynomial angle generation.
- **Synthetic benchmark suite** with automatic scaling-law fitting.
- **Real-dataset experiments** (classification & dimension reduction) on IMDb, 20 Newsgroups, PBMC68k, Dorothea, and Splice Junction DNA.
- **Marena 2026 extensions**: adaptive Boolean oracle, hierarchical sketching, variational warmstart, interferometric shadow, depolarizing noise model, k-Forrelation benchmarking, kernel shadow, non-IID scaling.

---

## Quick Start

### Option A — Google Colab (zero setup)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/quantum_oracle_sketching/blob/main/notebooks/quantum_oracle_sketching_demo.ipynb)

The notebook `notebooks/quantum_oracle_sketching_demo.ipynb` runs end-to-end and reproduces all four contribution figures.

### Option B — Local installation

```bash
git clone https://github.com/Tommaso-R-Marena/quantum_oracle_sketching.git
cd quantum_oracle_sketching
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# optional: noise + kernel extensions
pip install -e ".[dev,noise,kernel]"
```

### Run tests

```bash
PYTHONPATH=src pytest tests/ -v
```

### Run synthetic benchmark (main paper figure)

```bash
python -m qos.experiments.benchmark
```

Outputs: `benchmark_flat_vector.pdf`, `benchmark_general_vector.pdf`, `benchmark_boolean_function.pdf`, `benchmark_matrix_element.pdf`, `benchmark_matrix_row_index.pdf`.

### Run Marena 2026 extension benchmarks

```bash
qos-noise-benchmark       --dim 256 --num-trials 3 --output-dir ./results/
qos-forrelation-benchmark --dim 256 --num-trials 3 --output-dir ./results/
qos-kernel-benchmark      --dim 256 --num-trials 1 --output-dir ./results/
qos-non-iid-scaling       --dim 256 --num-trials 3 --output-dir ./results/
```

---

## Repository Structure

```
quantum_oracle_sketching/
├── src/qos/
│   ├── config.py
│   ├── core/
│   │   ├── oracle_sketch.py      # QOS Boolean, adaptive, matrix oracles
│   │   ├── state_sketch.py       # Quantum state sketching + kernel shadow
│   │   └── sampling.py
│   ├── theory/                   # ★ Marena 2026 novel contributions
│   │   ├── __init__.py           # clean public API
│   │   ├── hierarchical_sketch.py
│   │   ├── interferometric_shadow.py
│   │   └── variational_warmstart.py
│   ├── qsvt/
│   │   ├── angles.py
│   │   └── transform.py
│   ├── primitives/
│   │   ├── amplification.py
│   │   └── noise_model.py
│   ├── utils/
│   │   └── numerical.py
│   ├── data/
│   │   └── generation.py
│   └── experiments/
│       ├── benchmark.py
│       ├── noise_benchmark.py
│       ├── forrelation_benchmark.py
│       ├── kernel_benchmark.py
│       └── non_iid_scaling.py
├── tests/
│   ├── test_core.py
│   ├── test_adaptive_boolean.py  # ★ rewritten v1.3.0
│   ├── test_noise_model.py
│   ├── test_k_forrelation.py
│   ├── test_kernel_shadow.py
│   └── test_non_iid_scaling.py
├── notebooks/
│   ├── 01_qos_quickstart.ipynb
│   └── quantum_oracle_sketching_demo.ipynb  # ★ full Colab demo
├── examples/real_datasets/
│   ├── imdb/ news20/ pbmc68k/ dorothea/ splice/
├── docs/
│   └── theory.md                 # ★ full theoretical white-paper
├── CHANGELOG.md
├── CONTRIBUTING.md
└── pyproject.toml
```

---

## Mathematical Background

### Quantum State Sketching

Given $v \in \mathbb{R}^N$, QOS prepares $|\psi\rangle \propto \sum_i v_i |i\rangle$ using
$O(N/\epsilon)$ random samples (flat vectors) or $O(N\,\text{polylog}(1/\epsilon))$ samples (general vectors). The expected-unitary implementation constructs each oracle query as
$$U = \exp\!\left(i \sum_{t=1}^{M} B_t\right)$$
where $B_t$ are random single-sample gates. By the mixing lemma the expected channel upper-bounds the real-world random-channel error.

### Adaptive Boolean Oracle (Marena 2026, Contribution 1)

For a $K$-sparse Boolean function $f: \{0,1\}^n \to \{0,1\}$, the adaptive oracle uses a two-phase pilot–main strategy to concentrate phase budget on $\text{supp}(f)$:

$$M_{\text{adaptive}} = O\!\left(\frac{K\,\pi^2}{\varepsilon^2}\right), \qquad \text{improvement factor } \frac{N}{K}$$

**Main formula (Phase 2):**
$$\text{diag}[x] = \exp\!\left(M_{\text{main}} \cdot \log\!\left(1 + \exp\!\left(i\,\theta(x)\right) - 1\right) \cdot f(x)\right)$$
where $\theta(x) = q(x)\,\pi\,\hat{K} / M_{\text{main}}$ and $q(x) = 1/K$ exactly for a perfect pilot.

### Hierarchical Sketching (Marena 2026, Contribution 2)

For $k$-level hierarchically sparse oracles with structured sparsity at each level:
$$M_{\text{hierarchical}} = O\!\left(N \cdot Q^{2 - 1/k}\right)$$
This breaks the $Q^2$ barrier from Zhao et al. Theorem 3 by factor $Q^{1/k}$, without violating the Forrelation lower bound (which requires classical query complexity $Q_C = N^{1-\varepsilon}$, not $K^{1-\varepsilon}$).

### Variational Warmstart (Marena 2026, Contribution 3)

A parameterized phase ansatz $e^{i\phi(x;\theta)}$ trained via gradient descent on oracle samples reduces effective dimension from $N$ to $K_F$ (Fourier sparsity):
$$M_{\text{variational}} = O\!\left(K_F \cdot Q^2\right)$$

### Combined Bound (★)

All three improvements are multiplicative:
$$M_{\text{combined}} = O\!\left(K_F \cdot Q^{2 - 1/k}\right), \qquad \text{improvement factor } \frac{N}{K_F} \cdot Q^{1/k}$$

---

## API Reference

### Core oracle sketching

```python
from qos.core.oracle_sketch import (
    q_oracle_sketch_boolean,
    q_oracle_sketch_boolean_adaptive,
    q_oracle_sketch_matrix_element,
    q_oracle_sketch_matrix_row_index,
    q_oracle_sketch_matrix_index,
)

# Uniform Boolean oracle (Zhao et al.)
diag, M = q_oracle_sketch_boolean(truth_table, unit_num_samples=100_000)

# Adaptive Boolean oracle — N/K improvement [Marena 2026]
diag, M, weights = q_oracle_sketch_boolean_adaptive(
    truth_table, unit_num_samples=10_000, pilot_frac=0.2, key=jax.random.PRNGKey(0)
)
```

### Theory extensions (Marena 2026)

```python
from qos.theory import (
    HierarchicalOracleSketch,
    InterferometricClassicalShadow,
    VariationalWarmstart,
)

# Hierarchical: Q^{2-1/k} barrier break
sketch = HierarchicalOracleSketch.from_truth_table(truth, num_levels=3, total_queries=8)
diag, stats = sketch.build()

# Interferometric shadow: first open-source simulation
shadow = InterferometricClassicalShadow(weight_state, num_shadows=1000)
shadow.build_shadow()
predictions = shadow.predict(test_vectors)   # shape (T, 2) for Re + Im

# Variational warmstart: Fourier-sparse oracle
vw = VariationalWarmstart(truth, num_fourier_modes=32, learning_rate=0.02, num_steps=100)
vw.fit(unit_num_samples=2000)
diag = vw.predict()
```

### Noise model

```python
from qos.primitives.noise_model import DepolarizingChannel, crossover_sample_count

channel = DepolarizingChannel(num_qubits=10, noise_rate=0.01)
noisy_diag = channel.apply_to_diagonal(sketch_diag)
m_star = crossover_sample_count(dim=1024, noise_rate=0.01, circuit_depth=50, epsilon_target=0.1)
```

---

## Real-Dataset Results Summary

| Dataset | Task | Quantum Machine Size | Classical Streaming | Classical Sparse |
|---|---|---|---|---|
| IMDb | Binary sentiment | ~60 logical qubits | ~10^6 | ~10^6 |
| 20 Newsgroups | Multi-class topic | ~60 logical qubits | ~10^6 | ~10^6 |
| PBMC68k | Binary cell-type | ~50 logical qubits | ~10^5 | ~10^5 |
| Dorothea | Drug discovery | ~60 logical qubits | ~10^5 | ~10^5 |
| Splice | DNA junction | ~40 logical qubits | ~10^4 | ~10^4 |

---

## Configuration

```python
from qos.config import QOSConfig, get_default_config

cfg = get_default_config()
cfg.arcsin_degree = 30
cfg.sign_rescale  = 0.95
```

Set `QOS_PRECISION=32` to use 32-bit floats (faster, less accurate for high-degree polynomials).

---

## Citation

If you use this repository, please cite the original paper and the Marena 2026 extensions:

```bibtex
@article{zhao2026exponential,
    title   = {Exponential quantum advantage in processing massive classical data},
    author  = {Haimeng Zhao and Alexander Zlokapa and Hartmut Neven and Ryan Babbush
               and John Preskill and Jarrod R. McClean and Hsin-Yuan Huang},
    eprint  = {2604.07639},
    archivePrefix = {arXiv},
    year    = {2026}
}

@misc{marena2026qos,
    title   = {Extensions to Quantum Oracle Sketching: Adaptive Oracles, Hierarchical
               Sketching, Variational Warmstart, and Interferometric Shadows},
    author  = {Tommaso R. Marena},
    year    = {2026},
    note    = {\url{https://github.com/Tommaso-R-Marena/quantum_oracle_sketching}}
}
```

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines on bug reports, pull requests, and the test harness.

## License

MIT License. Copyright (c) 2026 Tommaso R. Marena. See [LICENSE](LICENSE).
