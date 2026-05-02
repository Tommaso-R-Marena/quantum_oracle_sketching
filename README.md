# Quantum Oracle Sketching (QOS)

**Extending quantum advantage in processing massive classical data**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/quantum_oracle_sketching/blob/main/notebooks/quantum_oracle_sketching_demo.ipynb)
[![Real Datasets](https://img.shields.io/badge/Colab-Real%20Datasets-orange?logo=googlecolab)](https://colab.research.google.com/github/Tommaso-R-Marena/quantum_oracle_sketching/blob/main/notebooks/real_datasets_colab.ipynb)
[![IBM Hardware](https://img.shields.io/badge/Colab-IBM%20Hardware-purple?logo=googlecolab)](https://colab.research.google.com/github/Tommaso-R-Marena/quantum_oracle_sketching/blob/main/notebooks/hardware_ibm_colab.ipynb)
[![Paper](https://img.shields.io/badge/arXiv-2604.07639-B31B1B.svg)](https://arxiv.org/abs/2604.07639)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-compatible-green.svg)](https://jax.readthedocs.io)
[![CI](https://github.com/Tommaso-R-Marena/quantum_oracle_sketching/actions/workflows/ci.yml/badge.svg)](https://github.com/Tommaso-R-Marena/quantum_oracle_sketching/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![v1.3.1](https://img.shields.io/badge/version-1.3.1-informational)](CHANGELOG.md)

This repository implements **Quantum Oracle Sketching (QOS)** and introduces four novel theoretical extensions (Marena 2026) that surpass the baseline of Zhao et al. (2025/2026) in sample complexity, circuit depth, and robustness.

> **Quick demo →** click the Colab badge above to run all experiments in your browser with zero setup.

---

## ⚡ Reproduce Figure 4 in 5 Minutes

```bash
# 1. Clone and install
git clone https://github.com/Tommaso-R-Marena/quantum_oracle_sketching.git
cd quantum_oracle_sketching
pip install -e ".[dev,noise,kernel]"

# 2. Reproduce Figure 4 (warmstart ablation, fast mode)
python -m qos.experiments.benchmark --dataset splice --fast
```

Or run everything in Colab (no install needed):

| Notebook | What it runs | Runtime |
|---|---|---|
| [📓 Demo](https://colab.research.google.com/github/Tommaso-R-Marena/quantum_oracle_sketching/blob/main/notebooks/quantum_oracle_sketching_demo.ipynb) | All four contribution figures | ~15 min (T4) |
| [📓 Real Datasets](https://colab.research.google.com/github/Tommaso-R-Marena/quantum_oracle_sketching/blob/main/notebooks/real_datasets_colab.ipynb) | IMDb, News20, PBMC3k, PBMC68k, Dorothea, Splice | ~2–4 hr (A100) |
| [📓 IBM Hardware](https://colab.research.google.com/github/Tommaso-R-Marena/quantum_oracle_sketching/blob/main/notebooks/hardware_ibm_colab.ipynb) | QOS oracle on real QPU + ZNE | ~3 min QPU time |
| [📓 Noise Sweep](https://colab.research.google.com/github/Tommaso-R-Marena/quantum_oracle_sketching/blob/main/notebooks/noise_robustness_sweep.ipynb) | Depolarizing noise robustness | ~20 min (T4) |
| [📓 Warmstart Ablation](https://colab.research.google.com/github/Tommaso-R-Marena/quantum_oracle_sketching/blob/main/notebooks/warmstart_ablation.ipynb) | Warmstart speedup per dataset | ~30 min (T4) |

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
- **Real-dataset experiments** (classification & dimension reduction) on IMDb, 20 Newsgroups, PBMC3k, PBMC68k, Dorothea, and Splice Junction DNA.
- **IBM hardware benchmarks**: n=3–5 qubit oracle circuits with ZNE error mitigation.
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
│   │   ├── __init__.py
│   │   ├── hierarchical_sketch.py
│   │   ├── interferometric_shadow.py
│   │   └── variational_warmstart.py
│   ├── qsvt/
│   ├── primitives/
│   │   └── noise_model.py
│   └── experiments/
│       └── real_datasets/
├── notebooks/
│   ├── quantum_oracle_sketching_demo.ipynb  # ★ full demo
│   ├── real_datasets_colab.ipynb            # ★ 6 datasets
│   ├── hardware_ibm_colab.ipynb             # ★ IBM QPU
│   ├── noise_robustness_sweep.ipynb         # ★ noise robustness
│   ├── warmstart_ablation.ipynb             # ★ warmstart ablation
│   └── circuit_depth_scaling.ipynb          # ★ circuit depth vs n
├── paper/
│   └── appendix_B_hardware_template.tex     # ★ Appendix B LaTeX
├── CITATION.cff
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

### Hierarchical Sketching (Marena 2026, Contribution 2)

$$M_{\text{hierarchical}} = O\!\left(N \cdot Q^{2 - 1/k}\right)$$

### Variational Warmstart (Marena 2026, Contribution 3)

$$M_{\text{variational}} = O\!\left(K_F \cdot Q^2\right)$$

### Combined Bound (★)

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

sketch = HierarchicalOracleSketch.from_truth_table(truth, num_levels=3, total_queries=8)
diag, stats = sketch.build()

shadow = InterferometricClassicalShadow(weight_state, num_shadows=1000)
shadow.build_shadow()
predictions = shadow.predict(test_vectors)

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
| PBMC3k | Cell-type (2.7k cells) | ~40 logical qubits | ~10^5 | ~10^5 |
| PBMC68k | Cell-type (68k cells) | ~50 logical qubits | ~10^5 | ~10^5 |
| Dorothea | Drug discovery | ~60 logical qubits | ~10^5 | ~10^5 |
| Splice | DNA junction | ~40 logical qubits | ~10^4 | ~10^4 |

---

## Citation

If you use this repository, please cite:

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
