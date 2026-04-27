# Paper Assembly Checklist

## Step 1 — Paste Aristotle outputs into section files

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

## Step 3 — Run benchmarks and insert results

- [ ] Run `notebooks/full_benchmark_suite.ipynb` — get Section 5.3 + 5.4 actual numbers
- [ ] Run `notebooks/real_datasets_colab.ipynb` — get Section 6 figures and tables
- [ ] Run `scripts/generate_paper_tables.py` — get Table 1 and Table 2 LaTeX
- [ ] Insert figures into `sections/experiments.tex`
- [ ] Update benchmark numbers in `sec53_variational.tex` and `sec54_shadow.tex`

## Step 4 — Write remaining sections

- [ ] `sections/background.tex` — fill in Zhao baseline description
- [ ] `sections/core_scaling.tex` — insert figures from real datasets notebook
- [ ] `sections/conclusion.tex` — write ~200 word conclusion

## Step 5 — Final checks before arXiv

- [ ] Compile `main.tex` with pdflatex — zero errors
- [ ] All theorems cited correctly (\cref)
- [ ] All figures have captions matching Aristotle output
- [ ] Abstract is exactly 149 words
- [ ] References complete in `references.bib`
- [ ] Lean files confirmed to compile: `lake build` in `paper/lean/`
- [ ] `PYTHONPATH=src pytest tests/ -q` — all tests pass

## Target: arXiv submission Wednesday April 29 or Thursday April 30, 2026
