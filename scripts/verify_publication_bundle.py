"""Verify that all publication artifacts exist and are non-empty.

Exit 0 when the bundle is complete; exit 1 with a concise report otherwise.

    PYTHONPATH=src python scripts/verify_publication_bundle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_CSV = [
    "results/raw_data/tvd_convergence.csv",
    "results/raw_data/sample_complexity.csv",
    "results/raw_data/warmstart_ablation_summary.csv",
    "results/raw_data/warmstart_ablation_iterations.csv",
    "results/raw_data/noise_robustness.csv",
    "results/raw_data/circuit_depth.csv",
]

REQUIRED_FIGURES = [
    "results/figures/tvd_convergence.png",
    "results/figures/tvd_convergence.pdf",
    "results/figures/sample_complexity.png",
    "results/figures/sample_complexity.pdf",
    "results/figures/noise_robustness.png",
    "results/figures/noise_robustness.pdf",
    "results/figures/circuit_depth.png",
    "results/figures/circuit_depth.pdf",
]

REQUIRED_TABLES = [
    "results/tables/summary.md",
    "results/tables/table_warmstart_ablation.tex",
    "results/tables/table_sample_complexity.tex",
    "results/tables/table_empirical_summary.tex",
]


def _check(paths: list[str]) -> list[str]:
    missing = []
    for p in paths:
        path = Path(p)
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(p)
    return missing


def main() -> int:
    missing = []
    missing += _check(REQUIRED_CSV)
    missing += _check(REQUIRED_FIGURES)
    missing += _check(REQUIRED_TABLES)

    log = Path("results/raw_data/notebook_run_log.txt")
    if log.is_file():
        text = log.read_text()
        if "FAIL" in text and "0/9" not in text:
            print("WARN: notebook_run_log.txt contains FAIL entries")
    else:
        missing.append(str(log))

    if missing:
        print("Publication bundle INCOMPLETE. Missing or empty:")
        for m in missing:
            print(f"  - {m}")
        return 1

    print("Publication bundle OK:")
    print(f"  {len(REQUIRED_CSV)} CSVs, {len(REQUIRED_FIGURES)} figures, tables present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
