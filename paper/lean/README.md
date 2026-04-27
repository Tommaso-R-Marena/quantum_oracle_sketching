# Lean 4 Formal Verification Files

This directory contains all Lean 4 proofs for the paper
"Quantum Oracle Sketching: Extensions and Formal Verification" (Marena 2026).

## Files

| File | Theorems | Sorrys | Section |
|------|----------|--------|---------|
| `VariationalWarmstart.lean` | 8 | 0 | §5.3 |
| `ShadowEstimator.lean` | 2 + 4 lemmas | 1 (phase averaging) | §5.4 |
| `AdaptiveAllocation.lean` | 6 | 0 | §5.1 |
| `HierarchicalComplexity.lean` | 7 | 0 | §5.2 |
| `QOSExtensions.lean` | 4 + main + 2 | 0 | §3 |
| `PhaseTimeBound.lean` | 10 | 0 | App. B |

**Total: 35+ theorems, 1 documented gap (ShadowEstimator phase averaging)**

## The One Remaining Gap

`ShadowEstimator.lean` has one gap: the phase averaging step requires
Lebesgue integration of trigonometric products over [0,2π)ᴺ via
Fubini's theorem. The informal proof is in `ShadowEstimatorProof.md`.
Mathlib lemmas needed: `integral_cos_sq`, `integral_prod`.

## TODO: paste each .lean file into this directory
- `VariationalWarmstart.lean` — from Aristotle run 5f5cbfed
- `ShadowEstimator.lean` — from Aristotle run 5cb1c450
- `AdaptiveAllocation.lean` — from Aristotle run d2468036
- `HierarchicalComplexity.lean` — from Aristotle run d2468036
- `QOSExtensions.lean` — from Aristotle run 29e4ed41
- `PhaseTimeBound.lean` — from Aristotle run 2146edba
