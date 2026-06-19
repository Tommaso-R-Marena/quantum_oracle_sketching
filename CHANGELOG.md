# Changelog

All notable changes to this project will be documented in this file.
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.3.4] — 2026-06-19

Publication bundle and scientific-rigor pass.

### Added
- **`singlecell` optional extra** (`scanpy`, `anndata`, `python-igraph`, `leidenalg`).
- **`scripts/generate_paper_tables.py`** — LaTeX tables from committed CSVs.
- **`scripts/verify_publication_bundle.py`** — CI gate for figures/CSVs/tables.
- **`tests/test_pbmc_utils.py`** — guards against PBMC68k double-preprocessing NaNs.
- **Committed figure artifacts** under `results/figures/` (8 PNG+PDF files).
- **LaTeX tables** under `results/tables/` (`table_warmstart_ablation.tex`, etc.).

### Fixed
- **`load_pbmc68k()`** — full ~68k download via `scvelo` with safe preprocessing;
  removed broken figshare URL from `real_datasets_colab.ipynb`.
- **`warmstart_ablation.ipynb`** — in-regime (K-sparse) vs out-of-regime (real PC1)
  reporting; paper LaTeX table uses in-regime only.
- **Bibliography** — unified Zhao baseline to arXiv:2604.07639 across
  `references.bib`, `marena2026_quantum_oracle_sketching.tex`, README, CITATION.cff.
- **`appendix_lean.tex`** — honest status: Lean proofs documented but not yet
  committed; tests/scripts are the executable verification layer.
- **`paper/sections/experiments.tex`** — populated with committed figure paths.
- **CI** — installs `singlecell,datasets`, runs publication bundle check + headless
  notebooks (3.11).
- **Headless notebooks: 10/10 PASS** (updated README/CHANGELOG claims).
- **`notebooks/00_publication_figures.ipynb`** — one-button Colab reproduction of the
  committed publication bundle (< 1 min CPU).
- **PRL `paper/main.tex` merge** — all `paper/sections/*.tex` populated from
  `marena2026_quantum_oracle_sketching.tex`; PRL abstract, macros, and cite keys fixed.

---

## [1.3.3] — 2026-05-28

Publication-readiness sweep, second pass (2026-05-28).

### Added
- **TRACEABILITY.md**: complete paper↔code↔test map covering every theorem,
  algorithm, and key equation, with a "math verified?" column and reproduction
  table for all five figure-bearing empirical claims. No unresolved gaps.
- **`[hardware]` and `[datasets]` optional install extras** in `pyproject.toml`
  (`qiskit-ibm-runtime`, `pdf2image`, `Pillow`).
- **IBM Quantum hardware validation** (`results/raw_data/ibm_qpu_run.json`):
  the 3-qubit XOR phase-oracle sketch (Algorithm 1) was executed on real IBM
  hardware (backend `ibm_fez`, 512 shots) after passing the AerSimulator gate
  (G1) and queue/qubit/shots gates (G2–G4). Ideal output `|011>` recovered with
  88.5% probability under hardware noise; result rendered into
  `notebooks/hardware_ibm_colab.ipynb`.
- **LaTeX `% Implementation:` / `% Tests:` comments** in
  `paper/marena2026_quantum_oracle_sketching.tex` linking each theorem,
  algorithm, and equation to its source file and test (comments only).
- **Paper-citing docstrings** on the public functions/classes in `src/qos/`
  that implement paper results (`See: Marena (2026), §X, …`).

### Fixed
- **Notebook portability (all 9 notebooks).** The headless driver
  (`scripts/run_notebooks_headless.py`) now applies quote-agnostic patches for
  `git clone`, `pip install/uninstall`, `USE_HARDWARE=True`→False,
  `%load_ext google.colab` magics, `from google.colab import …` (try/except),
  `/content/` paths, and a recursion-prone `shutil.make_archive` call; it also
  prepends an `_ensure_qos_importable()` shim to any clone/install/sys.path
  cell. The same shim is applied directly inside each notebook
  (belt-and-suspenders), and IBM/`pdf2image` import-guard cells were added to
  the hardware and real-datasets notebooks. Headless run: **7/9 PASS**
  (`n_err==0`); the 2 remaining are expected optional-dependency fails
  (`real_datasets_colab` → `pdf2image`).
- **`run_notebooks_headless.py`** `nbformat.write(nb, PosixPath)` crash fixed
  (pass `str(out_path)`).
- **`warmstart_ablation.ipynb`** now seeds the ablation with clearly-labeled
  SYNTHETIC K-sparse truth tables (the regime Theorem 3 targets) in addition
  to any real datasets, so the convergence diagnostic exercises the method's
  valid regime and runs cleanly offline. Real-data results (dense, balanced)
  are reported honestly and show no warmstart speedup — matching the paper's
  Fourier-sparsity assumption.
- **`scripts/verify_noise_robustness.py`** now computes the physically-correct
  depolarized measurement distribution (convex mixture toward uniform), so the
  noise-robustness TVD increases monotonically with the depolarizing rate
  instead of being invariant to it (a global diagonal shrink cancels under
  normalization). The underlying `DepolarizingChannel` contract is unchanged
  and still pinned by `tests/test_noise_model.py`.
- **CI** (`.github/workflows/ci.yml`): test command now uses `--tb=long`,
  `-W error`, and `set -o pipefail`; added a figure-regeneration step, a
  figures artifact upload, and an optional-extras (`[hardware]`, `[datasets]`)
  dry-run install smoke test.

### Changed
- **Version 1.1.0 → 1.3.3** with a **static** `version` in `pyproject.toml`
  (replacing the dynamic hatch-sourced version) plus `__version__` in
  `src/qos/__init__.py`; both report `1.3.3`.
- **CITATION.cff**: `date-released: 2026-05-28`.
- **.gitignore**: added `.eggs/`, `/tmp/`, `results/notebooks_data/`, `.env`,
  `*.json.bak`, and the runaway `notebooks/quantum_oracle_sketching/` self-clone
  guard.

---

## [1.3.2] — 2026-05-14

### Fixed
- **Two metrics under one name (`tvd_diag`).** A late-round review of
  the 1.3.2 audit surfaced that the project had been using the name
  `tvd_diag` for the Hadamard-induced measurement-distribution TVD,
  while the spec for `tvd_diag` is the raw-diagonal-L1 metric
  `0.5 * ‖d₁ − d₂‖₁ / N`. Under the spec, `tvd_diag(d, −d) = 1.0` for
  any nonzero ±1 diagonal; under the Hadamard-induced metric, the same
  expression is 0 (global-sign invariance of basis-state measurement).
  Both notions are useful; they were conflated.
- This release introduces **`hadamard_distribution_tvd`** as the
  explicit name for the Hadamard-induced metric and reserves
  **`tvd_diag`** for the raw-L1 metric specified by the user mandate.
  Every test, verify script, and notebook helper that previously called
  `tvd_diag` and wanted the Hadamard-induced semantics now calls
  `hadamard_distribution_tvd` instead. The warmstart-ablation binary
  search still gates on the Hadamard-induced metric (the right notion
  for basis-state measurement convergence); only the *function it calls*
  changed names.

### Changed
- `tests/test_tvd_core.py` — class split into `TestTvdDiag` (raw L1,
  including the **new `test_opposite_diagonals` asserting
  `tvd_diag(d, −d) = 1.0`**) and `TestHadamardDistributionTvd` (the
  former contents, with `test_global_sign_invariance` retained); a
  cross-metric `test_metrics_disagree_on_global_sign_flip` pin guards
  against re-conflation. Total: 60 parametric tests (was 33).
- `tests/test_warmstart_e2e.py` — switched to `hadamard_distribution_tvd`
  internally; docstring updated to explain why.
- `tests/test_ablation_helpers.py` — local helper and class renamed
  `tvd_diag → hadamard_distribution_tvd`,
  `TestTvdDiag → TestHadamardDistributionTvd`. Function body unchanged.
- `scripts/verify_warmstart_ablation.py`,
  `scripts/verify_tvd_convergence.py`,
  `scripts/verify_sample_complexity.py`,
  `scripts/verify_noise_robustness.py` — each renamed its local
  `tvd_diag` to `hadamard_distribution_tvd` and added a module-level
  note explaining which metric is plotted.
- `notebooks/warmstart_ablation.ipynb` — helper cell renamed
  `tvd_diag → hadamard_distribution_tvd`; docstring now cross-references
  the raw-L1 `tvd_diag` in `tests/test_tvd_core.py`.

### Verified
- Full pytest under `-W error`: **164 passed, 0 failed** (was 137; the
  TVD test count grew from 33 to 60 when the metrics were split).
- All five `verify_*.py` scripts rerun; numerical CSVs and PNG/PDF
  figures regenerated. The warmstart-ablation summary is unchanged
  (M_cold ∈ [555, 868], M_warm ∈ [48, 63], mean speedup ≈ 12.7×) — the
  binary search was already gating on the Hadamard metric; only the
  function name changed.

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
