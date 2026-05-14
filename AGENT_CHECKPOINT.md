# Agent Checkpoint — Milestone 1

**Date:** 2026-05-14
**Agent:** autonomous coding subagent (resume-after-checkpoint protocol)
**Working directory:** `/home/user/workspace/quantum_oracle_sketching-7acd62f3`
**Repo:** https://github.com/Tommaso-R-Marena/quantum_oracle_sketching
**Branch:** `main` (clean working tree at checkpoint time)
**HEAD:** `3352732` — `fix(notebook): avoid NumPy/SciPy version conflict on Colab (#19)`

## 1. Setup completed

| Step | Status | Notes |
|------|--------|-------|
| Repo present (already cloned) | OK | Located at primary working dir |
| Git identity configured | OK | `Tommaso R. Marena <marena@cua.edu>` (local repo scope) |
| `results/{figures,tables,raw_data,notebooks_executed}` created | OK | |
| Python venv created at `.venv/` | OK | Using **Python 3.12.8** — see "Python version" below |
| Two-step install `pip install -e .[dev]` then `pip install -e .[dev,noise,kernel]` | OK | Both steps completed without conflicts |
| `git log --oneline -10` → `results/raw_data/git_log_last10.txt` | OK | |
| Initial `pytest -v --tb=long -W error` → `results/raw_data/pytest_initial.txt` | OK | 98 passed, 1 failed (see §4) |

## 2. Python version — Python 3.10 NOT available

The environment provides **Python 3.12.8** and Python 3.13.x only. Python 3.10 is not available on this host.

- `pyproject.toml` declares `requires-python = ">=3.10"` and classifiers list 3.10/3.11/3.12, so 3.12 is supported by the project metadata.
- `[tool.mypy] python_version = "3.10"` and `[tool.ruff] target-version = "py310"` remain unchanged — only the runtime interpreter differs.
- Venv interpreter: `.venv/bin/python` → Python 3.12.8 (built with GCC 14.2.0, Apr 28 2026).

## 3. Installed dependency versions (key packages)

Captured to `results/raw_data/env_versions.txt`:

```
Python   : 3.12.8
jax      : 0.10.0
jaxlib   : 0.10.0
numpy    : 2.4.4
scipy    : 1.17.1
sklearn  : 1.8.0
matplotlib: 3.10.9
qiskit   : 2.4.1
qiskit-aer: 0.17.2
pyqsp    : (no __version__ attr; installed from PyPI ≥0.2.0)
```

Notable: this is on **NumPy 2.x** and **JAX 0.10**, which is significantly newer than the JAX versions assumed in earlier `fix(tests): version-agnostic ComplexWarning shim` commit (`7953894`). The single failing test is caused by a JAX 0.10 deprecation (complex→real `.astype`) that is promoted to an error by `-W error`.

## 4. Initial test result — 98 passed / 1 failed (with `-W error`)

**Summary line (from `results/raw_data/pytest_initial.txt`):**

```
1 failed, 98 passed in 97.72s (0:01:37)
```

**Failing test:**

```
FAILED tests/test_theory_fixes.py::test_variational_warmstart_beats_baseline
- DeprecationWarning: Casting from complex to real dtypes will soon raise a
  ValueError. Please first use jnp.real or jnp.imag to take the real/imaginary
  component of your input.
```

This is **not a logic regression** — it is a JAX 0.10 deprecation surfaced as an error solely because `-W error` is in the pytest invocation. The full pytest output shows the test reaches the warmstart predict path, which performs an implicit complex→real `.astype` cast inside JAX. Two avenues for the fix (to evaluate in milestone 2):

1. Change the relevant call site (likely in `src/qos/...` along the warmstart predict path, or the test's `find_M_warm` shim per commit `19ba3e5`) to take `jnp.real(...)` or `jnp.imag(...)` before casting.
2. Add `DeprecationWarning` filters scoped to JAX in `pyproject.toml`'s `filterwarnings` (already filters DeprecationWarning for `jax.*`, but `-W error` on the CLI overrides those filters — that's actually expected).

Without `-W error`, the same suite is expected to be **99/99 passing** (existing project CI relies on the default pytest config). This will be verified in milestone 2.

## 5. Files produced this milestone

```
results/raw_data/git_log_last10.txt       (10 lines)
results/raw_data/env_versions.txt         (key package versions)
results/raw_data/pytest_initial.txt       (full pytest -W error output, ~98 passed + 1 failed traceback)
AGENT_CHECKPOINT.md                       (this file)
```

No source code has been modified. No commits have been made. Working tree status:

```
Untracked:
  .venv/                       (local venv, should NOT be committed; add to .gitignore if not already)
  AGENT_CHECKPOINT.md          (this checkpoint, intentionally untracked for handoff)
  results/                     (output directory tree, currently has only raw_data/* files)
```

## 6. Next steps (for resumed run — full publication-ready mandate)

In priority order:

1. **Commit forensics** — review commits since the last green CI run; map each `fix(...)` commit (#13–#19) to the test/notebook it addressed and confirm no orphaned regressions.
2. **Source audit** — sweep `src/qos/**` for:
   - the JAX 0.10 complex→real cast that triggers the test failure under `-W error`;
   - NumPy 2.x compatibility (e.g. `np.complex_`, `np.float_`, `np.product`, deprecated aliases);
   - SciPy 1.17 API changes if used.
3. **Tests/fixes** — apply minimal fix(es) so the suite passes both with and without `-W error`; rerun `pytest tests/ -v --tb=long` and capture to `results/raw_data/pytest_post_fix.txt`.
4. **Numerical scripts/figures** — run the four `qos-*-benchmark` entry-point scripts (`noise_benchmark`, `forrelation_benchmark`, `kernel_benchmark`, `non_iid_scaling`) at publication settings; emit figures to `results/figures/` and tables to `results/tables/`.
5. **Notebooks** — execute `notebooks/quantum_oracle_sketching_demo.ipynb` (and warmstart ablation notebook) headlessly; save executed copies to `results/notebooks_executed/`.
6. **Docs/CI/AUDIT_REPORT** — update `CHANGELOG.md`, regenerate `docs/`, ensure `.github/workflows/` is consistent with Python 3.12 runtime; write `AUDIT_REPORT.md` capturing what was fabricated-vs-reproduced (nothing should be fabricated).
7. **Final commit + PR** — single feature branch, sensible commit graph, PR opened against `main` with full reproducibility summary.

## 7. Constraints reaffirmed

- **No fabrication of numerical results.** All figures/tables in `results/` must come from actually executed code, with seed and command captured.
- **Be explicit about blockers.** If qiskit-aer noise simulation or scvelo data fetch fails, document the failure in `AUDIT_REPORT.md` and leave the corresponding figure missing rather than synthesizing one.
- **NEVER approve PRs.** Only create and comment.
- **Always write final response to `/tmp/claude_code_output.md`** before exiting (user-global instruction).

---

**Checkpoint status: READY FOR RESUME.** Pausing here per protocol; awaiting confirmation before proceeding to milestone 2.
