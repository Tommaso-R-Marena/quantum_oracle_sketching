# Changelog

All notable changes to this project will be documented in this file.
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.3.2] — 2026-05-13

### Fixed
- **`VariationalWarmstart.__init__` complex→real silent cast (JAX 0.10
  deprecation).** Previously `self.truth_table = truth_arr.astype(real_dtype)`
  was called unconditionally; when the input was a complex phase oracle
  this silently dropped the imaginary part and produced a
  `DeprecationWarning` on JAX ≥ 0.10 (promoted to an error under
  `-W error`). The constructor now branches: for complex input it stores
  `|truth_arr|` as the support indicator and keeps the full complex
  phases in `_target_phases`; for real/boolean input the original behavior
  is preserved verbatim (`src/qos/theory/variational_warmstart.py`).
- `src/qos/qsvt/angles.py` debug `print()` calls replaced by `logging.debug`
  so library imports stay silent under publication-mode CLI invocations.

### Added
- **`tests/test_tvd_core.py`** — 33 parametric tests pinning the exact
  Hadamard-induced TVD formula (factors, normalization, identity,
  Parseval symmetry `TVD(d,-d) = 0`, triangle inequality, orthogonal
  diagonals → 1, and silent complex-input handling).
- **`tests/test_warmstart_e2e.py`** — 5 end-to-end tests exercising the
  full warmstart→TVD pipeline on synthetic Boolean truth tables, plus a
  no-false-convergence regression check for the `diagnose_warmstart`
  gate.
- **`scripts/verify_tvd_convergence.py`**, **`verify_sample_complexity.py`**,
  **`verify_warmstart_ablation.py`**, **`verify_noise_robustness.py`**,
  **`verify_circuit_depth.py`**, and **`generate_all_figures.py`** — one
  reproducible driver per claim; outputs CSV under `results/raw_data/`
  and PNG/PDF under `results/figures/`.
- **`scripts/run_notebooks_headless.py`** — patches Colab-only cells
  (pip install, drive mount, IBM Quantum token prompt) and executes
  every notebook under `notebooks/` to `results/notebooks_executed/`,
  emitting `.ipynb` and `.html` alongside a one-line run log.
- `AUDIT_REPORT.md` — commit forensics for #12–#19 plus a complete
  reproducibility checklist.

### Changed
- `pyproject.toml`: dependency upper bounds added for `jax`, `jaxlib`,
  `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `pydantic`, `qiskit`,
  `qiskit-aer`; `dev` extra now ships `nbformat`/`nbclient`/`nbconvert`.
- `.github/workflows/ci.yml`: install split into the two-step strategy
  (`[dev]` then `[dev,noise,kernel]`), pytest output captured as a CI
  artifact, JUnit XML emitted per Python version.

---

## [1.3.1] — 2026-04-25

### Added
- Colab badge in `README.md` linking to `notebooks/quantum_oracle_sketching_demo.ipynb`.
- `CHANGELOG.md` (this file).
- `CONTRIBUTING.md` with development, testing, and PR guidelines.
- `docs/theory.md` — full theoretical white-paper for all four Marena 2026 contributions with proof sketches, lower-bound arguments, and comparison to Zhao et al. theorems.
- `src/qos/theory/__init__.py` — clean public API exporting `HierarchicalOracleSketch`, `InterferometricClassicalShadow`, `VariationalWarmstart`.
- Version badge and MIT License badge in `README.md`.

### Changed
- README restructured: contributions table moved to top, Colab quick-start promoted above local install, math background expanded with all four Marena 2026 theorems.

---

## [1.3.0] — 2026-04-25

### Fixed
- **Critical bug in `q_oracle_sketch_boolean_adaptive`**: replaced `random.choice` with `random.randint` (true uniform sampling); rewrote per-entry theta formula so that `theta(x) = q(x)*pi*K_hat/M_main` (previously the K factor was missing, causing support entries to accumulate phase `pi/K` instead of `pi`).
- Rewrote `tests/test_adaptive_boolean.py`: removed fragile `test_adaptive_reduces_error_on_support` (operated in wrong N/K/M regime); replaced with 6 well-calibrated tests including `test_adaptive_beats_uniform_at_equal_M_large_N` (N=2048, K=4, M=8000) and `test_adaptive_nk_improvement_factor` (adaptive at M=K*C vs uniform at M=N*C).

### Added
- Full Colab pipeline `notebooks/quantum_oracle_sketching_demo.ipynb` (9 cells, 4 contribution figures + summary 2×2 panel).

---

## [1.2.0] — 2026-04-25

### Added
- `src/qos/theory/hierarchical_sketch.py` — Hierarchical oracle sketching achieving O(N·Q^{2-1/k}) sample complexity.
- `src/qos/theory/interferometric_shadow.py` — First open-source simulation of interferometric classical shadow (dual Hadamard test, Re+Im readout).
- `src/qos/theory/variational_warmstart.py` — Parameterized phase ansatz oracle trained via gradient descent on Fourier modes.
- Initial adaptive Boolean oracle (`q_oracle_sketch_boolean_adaptive`) with pilot-phase importance sampling.

---

## [1.1.0] — 2026-04-24

### Added
- Depolarizing noise model (`qos.primitives.noise_model`).
- k-Forrelation benchmarking (`qos.experiments.forrelation_benchmark`).
- Interferometric kernel shadow (`qos.core.state_sketch.q_interferometric_kernel_shadow`).
- Non-IID scaling experiments (`qos.experiments.non_iid_scaling`).
- CLI entry points: `qos-noise-benchmark`, `qos-forrelation-benchmark`, `qos-kernel-benchmark`, `qos-non-iid-scaling`.

---

## [1.0.0] — 2026-04-23

### Added
- Initial implementation of Quantum Oracle Sketching (Zhao et al. 2025/2026 baseline).
- Core oracle sketch: Boolean, matrix element, row-index, QSVT index.
- Core state sketch: flat vector, general vector, kernel shadow.
- QSVT utilities: angle generation via `pyqsp`, amplitude amplification, diagonal transform.
- Synthetic benchmark suite with 5 figure types.
- Real-dataset experiments: IMDb, 20 Newsgroups, PBMC68k, Dorothea, Splice.
- GitHub Actions CI with pytest + coverage.
