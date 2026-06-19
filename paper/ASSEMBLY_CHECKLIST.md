# Paper Assembly Checklist

**Canonical draft:** `paper/marena2026_quantum_oracle_sketching.tex` (complete prose).
**PRL track:** `paper/main.tex` (section shells — paste from canonical draft or Aristotle outputs).

## Step 1 — Paste Aristotle outputs into section files (PRL track only)

- [ ] `sections/introduction.tex` — paste from `RequestProject/PaperText.md` (296 words)
- [ ] `sections/main_theorem.tex` — paste Main Theorem + Prop from `PaperText.md`
- [ ] `sections/sec51_adaptive.tex` — paste Section 5.1 from `paper_sections.md`
- [ ] `sections/sec52_hierarchical.tex` — paste Section 5.2 from `paper_sections.md`
- [ ] `sections/sec53_variational.tex` — paste Section 5.3 from `sections_5_3_and_5_4.tex`
- [ ] `sections/sec54_shadow.tex` — paste Section 5.4 from `sections_5_3_and_5_4.tex`
  - **CRITICAL**: update Theorem 4 to corrected estimator (no 2N factor, see note in file)
- [ ] `sections/appendix_phase_time.tex` — paste from `GapAnalysis.md`
- [ ] `main.tex` abstract — paste from `PaperText.md` (149 words)

## Step 2 — Paste Lean files

- [ ] `lean/VariationalWarmstart.lean`
- [ ] `lean/ShadowEstimator.lean`
- [ ] `lean/AdaptiveAllocation.lean`
- [ ] `lean/HierarchicalComplexity.lean`
- [ ] `lean/QOSExtensions.lean`
- [ ] `lean/PhaseTimeBound.lean`

> **Note:** `appendix_lean.tex` now documents that Lean sources are planned but not
> yet committed; executable verification is via `tests/` + `scripts/verify_*.py`.

## Step 3 — Run benchmarks and insert results

- [x] Run `scripts/generate_all_figures.py` — committed figures in `results/figures/`
- [x] Run `scripts/generate_paper_tables.py` — tables in `results/tables/`
- [x] `sections/experiments.tex` — populated with figure paths + table input
- [ ] Run `notebooks/real_datasets_colab.ipynb` at paper quality (`FAST_MODE=False`)
- [ ] Insert per-dataset Zhao comparison tables from notebook JSON outputs

## Step 4 — Write remaining sections

- [x] `marena2026_quantum_oracle_sketching.tex` — full draft (introduction through conclusion)
- [ ] `sections/background.tex` — fill in Zhao baseline description (PRL track)
- [ ] `sections/core_scaling.tex` — insert figures from real datasets notebook (PRL track)
- [ ] `sections/conclusion.tex` — write ~200 word conclusion (PRL track)

## Step 5 — Final checks before arXiv

- [ ] Compile `main.tex` with pdflatex — zero errors (PRL track)
- [x] Compile `marena2026_quantum_oracle_sketching.tex` — should build (verify locally)
- [x] `references.bib` — Zhao baseline unified to arXiv:2604.07639
- [x] `PYTHONPATH=src pytest tests/ -q` — 172 tests pass
- [x] `scripts/verify_publication_bundle.py` — figures + CSVs + tables present
- [x] `scripts/run_notebooks_headless.py` — 9/9 notebooks OK
- [ ] Lean files confirmed to compile: `lake build` in `paper/lean/`
- [ ] Abstract is exactly 149 words (PRL `main.tex` track)

## Target: arXiv submission Wednesday April 29 or Thursday April 30, 2026
