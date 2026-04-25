# Quantum Oracle Sketching (QOS)

**Exponential quantum advantage in processing massive classical data**

[![Paper](https://img.shields.io/badge/arXiv-2604.07639-B31B1B.svg)](https://arxiv.org/abs/2604.07639)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-compatible-green.svg)](https://jax.readthedocs.io)

This repository implements **Quantum Oracle Sketching**, a framework that enables quantum computers to access classical data in superposition using only random classical samples—without full-dataset memory overhead. Combined with classical shadows, QOS constructs succinct classical models from massive data, a task provably impossible for any classical machine that is not exponentially larger than the quantum machine.

## Overview

**Quantum Oracle Sketching (QOS)** is a quantum algorithm for loading classical data into a quantum computer. It instantiates the oracles needed by any quantum query algorithm using only random classical samples, with no full-dataset memory overhead. This codebase includes:

- **Core QOS implementations** in JAX (GPU/TPU-compatible, automatically differentiable).
- **QSVT utilities** including amplitude amplification, matrix inversion, thresholding, and polynomial angle generation via `pyqsp`.
- **Synthetic benchmark suite** for quantum oracle and state sketching with automatic scaling-law fitting.
- **Real-dataset experiments** (classification and dimension reduction) demonstrating exponential memory advantage on:
  - IMDb sentiment (text TF-IDF)
  - 20 Newsgroups topic data (text TF-IDF)
  - PBMC68k single-cell RNA (UMI)
  - Dorothea drug-discovery dataset
  - Splice Junction DNA (k-mer)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/quantum-frontiers/quantum-oracle-sketching.git
cd quantum-oracle-sketching

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 1. Run the test suite

```bash
pytest tests/ -v
```

### 2. Run the synthetic benchmark (main paper figure)

```bash
python -m qos.experiments.benchmark
```

This generates:
- `benchmark_flat_vector.pdf`
- `benchmark_general_vector.pdf`
- `benchmark_boolean_function.pdf`
- `benchmark_matrix_element.pdf`
- `benchmark_matrix_row_index.pdf`

### 3. Run a real-dataset experiment

```bash
cd examples/real_datasets/imdb
python run.py --task both
```

### 4. Interactive tutorial

Open `notebooks/01_qos_quickstart.ipynb` in JupyterLab for a step-by-step introduction.

## Repository Structure

```
quantum_oracle_sketching/
├── src/qos/                    # Core package
│   ├── config.py               # Global configuration and dtypes
│   ├── core/                   # QOS implementations
│   │   ├── state_sketch.py     # Quantum state sketching (flat + general)
│   │   ├── oracle_sketch.py    # Quantum oracle sketching (Boolean, matrix element, index)
│   │   └── sampling.py         # Active random-sampling QOS
│   ├── qsvt/                   # Quantum Singular Value Transform
│   │   ├── angles.py           # Phase angle generation (poly fit + pyqsp)
│   │   └── transform.py        # QSVT application (dense / diagonal / imperfect)
│   ├── primitives/             # Quantum primitives
│   │   └── amplification.py    # Amplitude amplification via QSVT
│   ├── utils/                  # Utilities
│   │   └── numerical.py        # Random generators, block encodings, fidelity
│   ├── data/                   # Data sampling
│   │   └── generation.py       # matrix_data, vector_data, boolean_data
│   └── experiments/            # Benchmarks and plotting
│       ├── benchmark.py        # Synthetic benchmark suite
│       └── plotting.py         # Shared plotting utilities
├── tests/                      # Pytest test suite
├── examples/real_datasets/     # Real-world experiments
│   ├── imdb/
│   ├── news20/
│   ├── pbmc68k/
│   ├── dorothea/
│   └── splice/
├── notebooks/                  # Jupyter tutorials
├── docs/                       # Documentation
└── pyproject.toml              # Modern Python packaging
```

## Mathematical Background

### Quantum State Sketching

Given a vector $v \in \mathbb{R}^N$, QOS prepares a quantum state $|\psi\rangle \propto \sum_i v_i |i\rangle$ using only $O(N/\epsilon)$ random samples (for flat vectors) or $O(N \text{ polylog}(1/\epsilon))$ samples (for general vectors), where $\epsilon$ is the Euclidean error. This avoids the $\Omega(N)$ memory required by classical streaming algorithms.

The expected-unitary implementation constructs each oracle query as
$$U = \exp\left(i \sum_{t=1}^{M} B_t\right)$$
where $B_t$ are random single-sample gates. By the mixing lemma, the expected channel upper-bounds the real-world random-channel error.

### Quantum Oracle Sketching

For Boolean functions $f: \{0,1\}^n \to \{0,1\}$, QOS constructs the phase oracle
$$O_f |x\rangle = (-1)^{f(x)} |x\rangle$$
from $O(2^n / \epsilon)$ random queries.

For sparse matrix element oracles $|i\rangle|j\rangle \to A_{ij}|i\rangle|j\rangle$, QOS uses $O(\text{nnz}/\epsilon)$ samples. For sparse index oracles, QOS combines cumulative counters with QSVT sign-function amplification to encode the binary threshold predicate.

### QSVT Integration

QOS leverages the Quantum Singular Value Transform to:
1. **Invert** phases via $\arcsin(x)$ polynomials (for general state sketching).
2. **Amplify** amplitudes via sign-function polynomials (for amplitude amplification).
3. **Threshold** cumulative counts via sign functions (for index oracles).

All QSVT angles are generated via `pyqsp` with Chebyshev interpolation for numerical stability.

## Configuration

Numerical precision and default hyperparameters are managed by `qos.config.QOSConfig`:

```python
from qos.config import QOSConfig, get_default_config

cfg = get_default_config()
cfg.arcsin_degree = 30          # higher-degree polynomial
cfg.sign_rescale = 0.95         # tighter sign-function bound
```

Set the environment variable `QOS_PRECISION=32` to use 32-bit floats (faster but less accurate for high-degree polynomials).

## API Reference (Selected)

### State Sketching

```python
from qos.core.state_sketch import q_state_sketch_flat, q_state_sketch

# Flat vector (entries ±1)
state, samples = q_state_sketch_flat(vector, unit_num_samples=100_000)

# General vector (uses QSVT arcsin inversion)
state, samples = q_state_sketch(vector, key, unit_num_samples=100_000, degree=20)
```

### Oracle Sketching

```python
from qos.core.oracle_sketch import (
    q_oracle_sketch_boolean,
    q_oracle_sketch_matrix_element,
    q_oracle_sketch_matrix_row_index,
    q_oracle_sketch_matrix_index,
)

# Boolean phase oracle
diag, samples = q_oracle_sketch_boolean(truth_table, unit_num_samples=100_000)

# Sparse matrix element oracle
diag, samples = q_oracle_sketch_matrix_element(matrix, unit_num_samples=1_000_000)

# Row-index oracle with rank register
oracle, samples = q_oracle_sketch_matrix_index(matrix, unit_num_samples, axis=0)
```

### Amplitude Amplification

```python
from qos.primitives.amplification import amplitude_amplification

amplified = amplitude_amplification(
    unnormalized_state, degree=51, target_norm=0.99
)
```

### QSVT

```python
from qos.qsvt.angles import get_qsvt_angles, get_qsvt_angles_sign, get_qsvt_angles_inverse
from qos.qsvt.transform import apply_qsvt, apply_qsvt_diag

# General angles
angles = get_qsvt_angles(func=lambda x: x**3, degree=9, rescale=0.9)

# Sign function
angles, scale = get_qsvt_angles_sign(degree=101, threshold=0.1, rescale=0.9)

# Inverse function
angles, scale = get_qsvt_angles_inverse(kappa=5.0, epsilon=0.1)

# Apply to block encoding
V_qsvt = apply_qsvt(V, num_ancilla=2, angle_set=angles)
```

## Real-Dataset Results Summary

| Dataset | Task | Quantum Machine Size | Classical Streaming | Classical Sparse |
|---------|------|---------------------|--------------------|-----------------|
| IMDb | Binary classification | ~60 logical qubits | ~10^6 | ~10^6 |
| 20 Newsgroups | Multi-class | ~60 logical qubits | ~10^6 | ~10^6 |
| PBMC68k | Binary cell-type | ~50 logical qubits | ~10^5 | ~10^5 |
| Dorothea | Drug discovery | ~60 logical qubits | ~10^5 | ~10^5 |
| Splice | DNA junction | ~40 logical qubits | ~10^4 | ~10^4 |

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{zhao2026exponential,
    title={Exponential quantum advantage in processing massive classical data},
    author={Haimeng Zhao and Alexander Zlokapa and Hartmut Neven and Ryan Babbush and John Preskill and Jarrod R. McClean and Hsin-Yuan Huang},
    journal={},
    eprint={2604.07639},
    archivePrefix={arXiv},
    year={2026}
}
```

## License

Apache 2.0

## Novel Contributions (Marena 2026)

We add an adaptive Boolean oracle sketch that estimates support-aware importance weights from a pilot stage and then reweights phase accumulation to reduce effective worst-case sampling mass for sparse predicates. The new theorem states that sparse support size \(K\) replaces ambient size \(N\) in the leading sample-complexity term after pilot concentration.

For post-sketch hardware effects, we include a depolarizing-noise composition model that quantifies additive diamond-norm degradation and computes the crossover sample count where additional sketching no longer improves end-to-end error because gate noise dominates.

We extend hard-instance benchmarking from 2-Forrelation to k-Forrelation, including exact Walsh-Hadamard evaluation, a constant-query quantum estimator surrogate, and classical lower-bound scaling annotations to expose k-dependent separation trends.

We extend interferometric shadows from linear predictors to kernel predictors with quantum overlap kernels \(K(x,z)=|\langle\psi(x)|\psi(z)\rangle|^2\), a dual solver over state Gram matrices, and benchmark plumbing comparing memory/accuracy regimes against linear models.

We add a non-IID repetition-scaling experiment to test whether the theoretical repetition factor \(R\) is tight by jointly fitting exponents in \(\log(\mathrm{error}) = a\log M + b\log R + c\).

\[
\textbf{Adaptive Boolean Oracle Theorem:}\quad
M = O\!\left(\frac{K t^2}{\epsilon^2}\right),\quad t = \pi N/K,
\]
with improvement factor \(N/K\) over uniform-sampling bounds.

| Method | Sample complexity (Boolean) | Notes |
|---|---:|---|
| Uniform QOS | \(O(N t^2/\epsilon^2)\) | Zhao et al. baseline |
| Adaptive QOS (this work) | \(O(K t^2/\epsilon^2)\) | Pilot-estimated support weights |
| Classical streaming baseline | \(\Omega(N/\epsilon^2)\) | No coherent phase access |

Run the new experiments:

```bash
qos-noise-benchmark --dim 256 --num-trials 3 --output-dir ./results/
qos-forrelation-benchmark --dim 256 --num-trials 3 --output-dir ./results/
qos-kernel-benchmark --dim 256 --num-trials 1 --output-dir ./results/
qos-non-iid-scaling --dim 256 --num-trials 3 --output-dir ./results/
```
