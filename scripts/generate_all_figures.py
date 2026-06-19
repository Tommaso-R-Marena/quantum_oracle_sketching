"""One-button driver: run every verify_*.py script in this directory.

Each script writes its own CSV(s) under ``results/raw_data`` and figure(s)
under ``results/figures``.  This driver exists so a reviewer can reproduce
all figures with a single command:

    PYTHONPATH=src python scripts/generate_all_figures.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = [
    "scripts/verify_tvd_convergence.py",
    "scripts/verify_sample_complexity.py",
    "scripts/verify_warmstart_ablation.py",
    "scripts/verify_noise_robustness.py",
    "scripts/verify_circuit_depth.py",
    "scripts/generate_paper_tables.py",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / "src"
    env = {"PYTHONPATH": str(env_path)}
    import os

    env.update(os.environ)

    total = 0.0
    failures: list[tuple[str, int]] = []
    for s in SCRIPTS:
        full = repo_root / s
        print(f"\n===== {s} =====")
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(full)],
            cwd=str(repo_root),
            env=env,
        )
        dt = time.time() - t0
        total += dt
        print(f"({dt:.1f}s, rc={proc.returncode})")
        if proc.returncode != 0:
            failures.append((s, proc.returncode))

    print(f"\nTotal: {total:.1f}s")
    if failures:
        print("FAILURES:")
        for s, rc in failures:
            print(f"  {s}  (rc={rc})")
        return 1
    print("OK: all figures regenerated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
