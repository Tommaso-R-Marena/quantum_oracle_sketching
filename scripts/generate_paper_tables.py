"""Generate LaTeX tables for the paper from committed CSV/JSON artifacts.

Reads reproducibility outputs under ``results/raw_data/`` and writes:

  results/tables/table_warmstart_ablation.tex
  results/tables/table_sample_complexity.tex
  results/tables/table_empirical_summary.tex

Run from repo root:

    PYTHONPATH=src python scripts/generate_paper_tables.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

OUT_DIR = Path("results/tables")
RAW = Path("results/raw_data")


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    print(f"wrote {path}")


def table_warmstart_ablation() -> None:
    csv_path = RAW / "warmstart_ablation_summary.csv"
    rows = list(csv.DictReader(csv_path.open()))
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Trial & $M_{\mathrm{cold}}$ & $M_{\mathrm{warm}}$ & Speedup \\",
        r"\midrule",
    ]
    speedups = []
    for r in rows:
        mc, mw = int(r["m_cold"]), int(r["m_warm"])
        sp = float(r["speedup"])
        speedups.append(sp)
        lines.append(f"{r['trial']} & {mc} & {mw} & {sp:.2f}$\\times$ \\\\")
    mean_sp = sum(speedups) / len(speedups) if speedups else 0.0
    lines += [
        r"\midrule",
        f"Mean & --- & --- & {mean_sp:.2f}$\\times$ \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    _write(OUT_DIR / "table_warmstart_ablation.tex", "\n".join(lines) + "\n")


def table_sample_complexity() -> None:
    csv_path = RAW / "sample_complexity.csv"
    by_n: dict[str, list[int]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            by_n.setdefault(row["N"], []).append(int(row["M_star"]))
    lines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"$N$ & Mean $M^\star$ ($\varepsilon=0.10$) \\",
        r"\midrule",
    ]
    for n in sorted(by_n, key=lambda x: int(x)):
        vals = by_n[n]
        mean_m = sum(vals) / len(vals)
        lines.append(f"{n} & {mean_m:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(OUT_DIR / "table_sample_complexity.tex", "\n".join(lines) + "\n")


def table_empirical_summary() -> None:
    """High-level summary table tying scripts to paper claims."""
    claims = [
        ("TVD convergence", "verify_tvd_convergence.py", r"$TVD \propto M^{-1/2}$"),
        ("Sample complexity", "verify_sample_complexity.py", r"$M^\star \propto N$ (cold)"),
        ("Warmstart ablation", "verify_warmstart_ablation.py", r"$\approx 12\times$ speedup (K-sparse)"),
        ("Noise robustness", "verify_noise_robustness.py", r"TVD monotone in $\eta$"),
        ("Circuit depth", "verify_circuit_depth.py", "Finite-depth crossover"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Empirical summary of publication-bundle claims.}",
        r"\label{tab:empirical-summary}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Claim & Script & Key finding \\",
        r"\midrule",
    ]
    for claim, script, finding in claims:
        lines.append(f"{claim} & \\texttt{{{script}}} & {finding} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(OUT_DIR / "table_empirical_summary.tex", "\n".join(lines) + "\n")


def main() -> None:
    table_warmstart_ablation()
    table_sample_complexity()
    table_empirical_summary()

    # Optional: ingest notebook Zhao comparison JSON if present.
    zhao_json = Path("notebooks/results/pbmc68k_zhao_comparison.json")
    if not zhao_json.exists():
        zhao_json = Path("results/notebooks_data/pbmc68k_zhao_comparison.json")
    if zhao_json.exists():
        data = json.loads(zhao_json.read_text())
        snippet = (
            f"% Auto-generated from {zhao_json}\n"
            f"% reproduced={data.get('reproduced')} "
            f"acc_delta={data.get('accuracy_delta')}\n"
        )
        _write(OUT_DIR / "zhao_comparison_pbmc68k.tex", snippet)


if __name__ == "__main__":
    main()
