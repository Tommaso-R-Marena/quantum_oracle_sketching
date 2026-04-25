# Contributing to Quantum Oracle Sketching

Thank you for your interest in contributing. This document covers the development workflow, test conventions, and pull request process.

## Development Setup

```bash
git clone https://github.com/Tommaso-R-Marena/quantum_oracle_sketching.git
cd quantum_oracle_sketching
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,noise,kernel]"
```

## Running Tests

```bash
# Full suite
PYTHONPATH=src pytest tests/ -v

# Single file
PYTHONPATH=src pytest tests/test_adaptive_boolean.py -v --tb=short

# With coverage
PYTHONPATH=src pytest tests/ --cov=qos --cov-report=term-missing
```

## Test Conventions

### Operating-point discipline
Each test must operate in a regime where the claim being tested is actually true. For the adaptive oracle N/K improvement:
- Use `N/K >= 128` (strong ratio) so uniform error ~√(N/M) >> √(K/M).
- Use `M >= 4*N` for the uniform oracle to be out of its pre-convergence regime.
- Use at least 8 random seeds and compare means, not single runs.

### Tolerance conventions
- Hard assertions (e.g., off-support == 1.0): `atol=1e-6`.
- Convergence monotonicity with slack: `next < prev * 1.5` (allow 50% noise).
- Relative improvement comparisons: `mean_adaptive <= mean_uniform * 3.0` (conservative bound).

### Naming
- `test_<module>_<what_is_being_tested>` for unit tests.
- `test_<module>_<property>_<condition>` for parameterized properties.

## Adding a New Theory Module

1. Add `src/qos/theory/my_module.py`.
2. Export the public class/function from `src/qos/theory/__init__.py`.
3. Add tests in `tests/test_my_module.py`.
4. Document the theorem in `docs/theory.md` with sample complexity and proof sketch.
5. Add a Colab cell in `notebooks/quantum_oracle_sketching_demo.ipynb`.

## Pull Request Process

1. Fork the repository and create a feature branch: `git checkout -b feat/my-contribution`.
2. Ensure all existing tests pass: `PYTHONPATH=src pytest tests/ -v`.
3. Add tests for your new code (coverage should not decrease).
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Open a PR with a clear description of the theoretical contribution and empirical evidence.

## Code Style

- Python 3.10+ type hints throughout.
- JAX-native: no NumPy in hot paths, use `jnp` everywhere.
- Docstrings: Google style with Args/Returns sections.
- Line length: 100 characters.

## Theoretical Contributions

For contributions that claim sample complexity improvements:
- State the theorem explicitly in the module docstring.
- Provide a proof sketch in `docs/theory.md`.
- Identify which assumption from Zhao et al. is relaxed (e.g., unstructured oracle → hierarchically sparse oracle).
- Show the improvement is not vacuously compatible with the Forrelation lower bound.
