# Numerical Summary — verify_*.py outputs

This file is auto-described from the CSVs in `results/raw_data/`. See the
referenced CSV for the raw rows.

## TVD convergence (cold uniform sketch)

Source: `results/raw_data/tvd_convergence.csv`.

For N = 16, 32, 64 and a 50%-density random truth table, TVD shrinks as
roughly M^{-1/2} as the sample budget grows; reference line included on the
log-log plot at `results/figures/tvd_convergence.{png,pdf}`.

## Sample complexity (cold, ε = 0.10)

Source: `results/raw_data/sample_complexity.csv`.

Smallest M achieving TVD < 0.10, by binary search, on 50%-density random
truth tables, 3 trials each:

| N  | mean M* |
|----|---------|
| 8  |  ~65    |
| 16 | ~110    |
| 32 | ~290    |
| 64 | ~415    |

Linear-in-N reference overlaid in
`results/figures/sample_complexity.{png,pdf}`.

## Warmstart ablation (synthetic N = 64, K = 4 sparse + 5% noise)

Source: `results/raw_data/warmstart_ablation_summary.csv` (3 trials).

| trial | M_cold | M_warm | speedup |
|-------|--------|--------|---------|
| 0     | 555    | 59     |  9.41×  |
| 1     | 868    | 48     | 18.08×  |
| 2     | 675    | 63     | 10.71×  |
|       |        |        | **mean ≈ 12.7×** |

Per-iteration binary-search log (66 rows) at
`results/raw_data/warmstart_ablation_iterations.csv`.

## Noise robustness (N = 64, M = 2000)

Source: `results/raw_data/noise_robustness.csv`.

Depolarizing rate sweep over η ∈ {0, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1};
TVD grows monotonically with η, as expected from the
`DepolarizingChannel` shrinkage model. Figure at
`results/figures/noise_robustness.{png,pdf}`.

## Circuit-depth crossover (η = 1e-3, ε = 0.10)

Source: `results/raw_data/circuit_depth.csv`.

M* (crossover sample count) vs. circuit depth, plotted on semi-log axes
for N ∈ {16, 64, 256}. Figure at
`results/figures/circuit_depth.{png,pdf}`.
