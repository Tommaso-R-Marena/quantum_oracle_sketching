# Paper ↔ Code ↔ Test Traceability Map
## quantum_oracle_sketching — Marena (2026)

This map links every theorem, algorithm, and key equation in
`paper/marena2026_quantum_oracle_sketching.tex` to its implementation in
`src/qos/` and to at least one test that checks the *mathematical claim*
(not merely that the code runs). Each paper element also carries a
`% Implementation:` / `% Tests:` LaTeX comment at its definition site in the
`.tex` source (comments only; they do not appear in the compiled PDF).

**"Math verified?"** = the cited test asserts that the function produces
output consistent with the mathematical bound/claim stated in the paper.

**Status legend:** "covered" (✅) means the paper claim has code AND a
mathematically-checking test. Other possible states (none occur below) would be
"no test", "no code", or "unclear".

| Paper element | Section / Eq / Thm | Source file :: function | Test file :: test | Math verified? | Status |
|---|---|---|---|---|---|
| Zhao log-sum expected-unitary oracle | §2.1, Eq. (1) `eq:logsum` | `src/qos/core/oracle_sketch.py::q_oracle_sketch_boolean` | `tests/test_core.py::test_phase_oracle`, `test_reconstruction` | Asserts the reconstructed phase diagonal matches `e^{iπf(x)}` within tolerance for the documented budget. | ✅ |
| Zhao error bound `|d̂_M(x)−e^{iπ}| ≤ C√(N/M)` | §2.1, Eq. (2) `eq:zhao_bound` | `src/qos/core/oracle_sketch.py::q_oracle_sketch_boolean` | `tests/test_core.py::test_success_probability_and_error`, `tests/test_adaptive_boolean.py::test_uniform_oracle_converges` | Asserts the L∞ support error shrinks with M consistent with the √(N/M) scaling. | ✅ |
| Algorithm 1: Adaptive Oracle Sketch (K-sparse) | §3, Alg. 1 `alg:adaptive` | `src/qos/core/oracle_sketch.py::q_oracle_sketch_boolean_adaptive` | `tests/test_adaptive_boolean.py::test_adaptive_oracle_converges`, `test_off_support_entries_are_one`, `test_weights_sum_to_one`, `test_pilot_frac_zero_fallback` | Asserts support error ≤ ε at the documented sample count, exact `d̂(x)=1` off-support, and importance weights normalize. | ✅ |
| Theorem 1: Adaptive Oracle Sketching bound `Pr[max err > ε] ≤ δ`, `M=O(K/ε²)` | §3, Thm. `thm:adaptive` | `src/qos/core/oracle_sketch.py::q_oracle_sketch_boolean_adaptive`; `src/qos/theory/adaptive_lower_bound.py::compute_bounds` | `tests/test_adaptive_boolean.py::test_adaptive_oracle_converges`; `tests/test_lower_bound.py::test_lower_bound_scales_with_K`, `test_lower_bound_scales_inverse_epsilon_squared` | Asserts adaptive sample bound scales O(K/ε²) and beats uniform; lower-bound scaling in K and 1/ε² verified numerically. | ✅ |
| Remark: matching lower bound `Ω(K/ε²)` (Assouad) | §3, Remark | `src/qos/theory/adaptive_lower_bound.py::compute_bounds` (`is_tight`) | `tests/test_lower_bound.py::test_is_tight_flag`, `test_improvement_factor_equals_N_over_K`, `test_uniform_greater_than_adaptive_for_sparse` | Asserts upper/lower bounds coincide up to constants (tightness flag) and N/K improvement factor. | ✅ |
| Definition + Theorem 2: Hierarchical query reduction `M=O(N·k·Q^{2−1/k})` | §4, Def. + Thm. `thm:hier` | `src/qos/theory/hierarchical_sketch.py::HierarchicalOracleSketch.build`, `::compute_hierarchical_sample_complexity` | `tests/test_hierarchical_sketch.py::test_hierarchical_uses_fewer_samples_than_zhao_reference`, `test_more_levels_fewer_samples`, `test_hierarchical_diagonal_accuracy_on_support` | Asserts total samples < Zhao O(NQ²) reference, monotone decrease with k, and on-support accuracy preserved. | ✅ |
| Corollary: sub-quadratic query cost `O(NQ^{2−ε_Q})` | §4, Cor. `cor:hier` | `src/qos/theory/hierarchical_sketch.py::compute_hierarchical_sample_complexity` | `tests/test_hierarchical_sketch.py::test_improvement_ratio_at_least_one`, `test_more_levels_fewer_samples` | Asserts the Q^{1/k} improvement factor ≥ 1 and grows with k (sub-quadratic regime). | ✅ |
| Theorem 3: Variational warmstart (Fourier-sparse) | §5, Thm. `thm:var` | `src/qos/theory/variational_warmstart.py::VariationalWarmstart` (`fit`, `predict`) | `tests/test_variational_warmstart.py::test_variational_loss_decreases`, `test_variational_oracle_bounded`; `tests/test_warmstart_e2e.py::test_full_pipeline_converges_at_full_budget`, `test_diagnose_warmstart_rejects_random_prediction`, `test_cold_sketch_tvd_monotone_in_budget` | Asserts proxy loss decreases, output on the unit circle, full-budget convergence below ε, and rejection of a random (non-converged) diagonal. | ✅ |
| Corollary: combined bound `O(K_F·Q^{2−1/k}/ε²)` | §6, Cor. `cor:combined` | `src/qos/theory/hierarchical_sketch.py::compute_hierarchical_sample_complexity` (composed with adaptive + variational scaling) | `tests/test_hierarchical_sketch.py::test_hierarchical_uses_fewer_samples_than_zhao_reference`; `tests/test_theory_fixes.py::test_variational_warmstart_beats_baseline` | Asserts the composed sample complexity beats the Zhao baseline and the warmstart beats the uniform baseline (the two factors of the combined bound). | ✅ |
| §7: Interferometric classical shadow, `Var[Ô_T]=O(s/T)`, unbiased | §7 `sec:shadow` | `src/qos/theory/interferometric_shadow.py::InterferometricClassicalShadow` (`build_shadow`, `predict`, `prediction_error_bound`) | `tests/test_interferometric_shadow.py::test_shadow_error_bound`, `test_shadow_predictions_bounded`, `test_shadow_prediction_shape`; `tests/test_kernel_shadow.py::test_interferometric_prediction_binary`, `test_prediction_uses_alpha_not_nearest_neighbor` | Asserts the empirical prediction error obeys the O(√(s/T)) bound and the estimator is unbiased/bounded. | ✅ |
| TVD metric contract (raw-L1 `tvd_diag` vs Hadamard-induced) | §empirical / methods | (test-module helpers in `tests/test_tvd_core.py`) used by `scripts/verify_*` and notebooks | `tests/test_tvd_core.py::test_identity`, `test_opposite_diagonals`, `test_single_bit_flip_is_1_over_N`, `test_metrics_disagree_on_global_sign_flip` | Asserts both TVD notions independently (raw-L1 `TVD(d,−d)=1`; Hadamard `TVD(d,−d)=0`), pinning the metric separation. | ✅ |
| Depolarizing noise model (post-sketch robustness) | §7 / Discussion; noise sweep | `src/qos/primitives/noise_model.py::DepolarizingChannel`, `compose_sketch_and_noise_error`, `crossover_sample_count` | `tests/test_noise_model.py::test_depolarizing_zero_noise_is_identity`, `test_depolarizing_high_noise_collapses_to_maximally_mixed`, `test_crossover_monotone_in_noise_rate`, `test_compose_errors_triangle_inequality` | Asserts zero-noise = identity, high-noise → maximally mixed, crossover monotone in η, and triangle-inequality error composition. | ✅ |

## Reproduction of empirical findings (§8)

| Empirical claim (§8) | Reproduction script | Output | Spot-check (Phase 4) |
|---|---|---|---|
| Adaptive vs uniform sample complexity, sub-linear M*(N) | `scripts/verify_sample_complexity.py` | `results/raw_data/sample_complexity.csv`, `results/figures/sample_complexity.{png,pdf}` | log-log slope of M* vs N ≈ 0.94 < 1 (sub-linear) ✅ |
| TVD convergence vs budget M | `scripts/verify_tvd_convergence.py` | `results/raw_data/tvd_convergence.csv`, `results/figures/tvd_convergence.{png,pdf}` | TVD decreases with M ✅ |
| Warmstart speedup (M_cold/M_warm) | `scripts/verify_warmstart_ablation.py` | `results/raw_data/warmstart_ablation_{iterations,summary}.csv`, `results/tables/table_warmstart_ablation.tex` | mean speedup ≈ 12.7× ≥ 5× ✅ |
| Hierarchical circuit-depth / sample crossover | `scripts/verify_circuit_depth.py` | `results/raw_data/circuit_depth.csv`, `results/figures/circuit_depth.{png,pdf}` | crossover exists at finite depth/N ✅ |
| Noise robustness (TVD vs depolarizing rate η) | `scripts/verify_noise_robustness.py` | `results/raw_data/noise_robustness.csv`, `results/figures/noise_robustness.{png,pdf}` | TVD monotonically non-decreasing in η ✅ |

## Notes
- **Paper assembly status:** `paper/marena2026_quantum_oracle_sketching.tex` is the
  canonical prose draft; `paper/main.tex` (PRL track) still has section shells.
  Committed figure/table artifacts live under `results/figures/` and
  `results/tables/`; regenerate via `scripts/generate_all_figures.py`.
- **Formal verification:** Lean 4 sources are documented in `paper/lean/README.md`
  but not yet committed; executable contracts are in `tests/` (see
  `paper/sections/appendix_lean.tex`).
- Docstrings on the public functions/classes above cite the corresponding
  paper section/theorem/equation (`See: Marena (2026), §X, …`).
- The interferometric-shadow `Var=O(s/T)` claim is checked via the empirical
  error-bound test rather than a direct variance estimate; the test asserts the
  prediction error stays within the theoretical `O(√(s/T))` envelope across the
  shadow-budget sweep.
