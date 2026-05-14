"""Headless executor for notebooks/ that patches Colab-only cells.

Strategy:
  1. Copy each notebook to ``results/notebooks_executed/<name>.ipynb``.
  2. Patch:
       - ``subprocess.run([..., 'pip', 'install', ...])`` cells -> no-op
         (the local venv already has everything installed editable).
       - ``MOUNT_DRIVE = True`` -> ``MOUNT_DRIVE = False`` so the Colab
         drive.mount path is skipped.
       - any ``getpass(...)`` IBM token prompt -> set IBM_TOKEN='' and
         force the AerSimulator path (best effort).
  3. Execute via nbclient with a long per-cell timeout.
  4. Also write a sibling .html file for browser viewing.

If a notebook fails, its traceback is captured in the executed .ipynb
(`allow_errors=True`); a one-line summary is printed and added to
``results/raw_data/notebook_run_log.txt``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

NB_DIR = Path("notebooks")
OUT_DIR = Path("results/notebooks_executed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = Path("results/raw_data/notebook_run_log.txt")

# Per-cell timeout (s).
# Long enough for an honest run, short enough that a hang doesn't burn the
# whole session.  Heavy real-dataset / IBM-hardware notebooks may legitimately
# exceed this — the runner records the failure and moves on instead of
# blocking forever.
CELL_TIMEOUT = 300

INSTALL_NOOP_HEADER = (
    "# [patched]: in-notebook pip install skipped (local venv already has the package).\n"
)

# Stand-in for a subprocess.CompletedProcess so cells that later do
# `result.stdout` / `result.returncode` keep working.
INSTALL_NOOP_VALUE = (
    "type('_PatchedResult', (), "
    "{'stdout': b'', 'stderr': b'', 'returncode': 0})()"
)

# Heuristic detectors
RE_PIP_SUBPROC = re.compile(
    r"subprocess\.run\(\s*\[\s*sys\.executable\s*,\s*'-m'\s*,\s*'pip'[^)]*?\)\s*",
    re.M | re.S,
)
RE_PIP_BANG = re.compile(r"^\s*!pip\s+install[^\n]*", re.M)
RE_SHUTIL_RMTREE_REPO = re.compile(
    r"shutil\.rmtree\(\s*_repo\s*\)|if\s+os\.path\.isdir\(\s*_repo\s*\)\s*:\s*shutil\.rmtree\(\s*_repo\s*\)",
    re.M,
)
RE_PIP_INSTALL_GENERIC = re.compile(
    r"subprocess\.run\(\s*\[[^\]]*?'pip'[^\]]*?\][^)]*\)\s*", re.M | re.S
)
RE_MOUNT_TRUE = re.compile(r"MOUNT_DRIVE\s*=\s*True")
RE_DRIVE_MOUNT = re.compile(r"from\s+google\.colab\s+import\s+drive")
RE_GETPASS = re.compile(r"getpass\.getpass\(|from\s+getpass\s+import\s+getpass")


def patch_source(src: str) -> tuple[str, list[str]]:
    """Return (patched_source, list_of_applied_patches).

    Strategy: replace only the *offending statements* (pip-install calls,
    repo-rmtree calls), never the whole cell -- because cells often mix
    install logic with imports / setup that we need to keep.
    """
    applied: list[str] = []
    new_src = src

    # 1) Replace each pip-install statement / line, preserving the rest of
    #    the cell so any imports or setup logic survives.
    #    NOTE: we replace with `None` (an expression) rather than `pass`
    #    (a statement) so that `result = subprocess.run(...)` stays valid
    #    Python after the substitution.
    if RE_PIP_SUBPROC.search(new_src):
        new_src = RE_PIP_SUBPROC.sub(INSTALL_NOOP_VALUE + "  # [patched: pip install skipped]\n", new_src)
        applied.append("pip subprocess -> None")
    if RE_PIP_BANG.search(new_src):
        new_src = RE_PIP_BANG.sub("# [patched: !pip install skipped]", new_src)
        applied.append("!pip -> comment")
    if RE_PIP_INSTALL_GENERIC.search(new_src):
        new_src = RE_PIP_INSTALL_GENERIC.sub(
            INSTALL_NOOP_VALUE + "  # [patched: pip install skipped]\n", new_src
        )
        applied.append("generic pip subprocess -> None")
    # Also no-op pip uninstall calls -- if a notebook tries to uninstall
    # the local editable install, downstream cells break.
    new_src = re.sub(
        r"subprocess\.run\(\s*\[[^\]]*?'uninstall'[^\]]*?\][^)]*\)\s*",
        INSTALL_NOOP_VALUE + "  # [patched: pip uninstall skipped]\n",
        new_src,
        flags=re.M | re.S,
    )
    # Also no-op git clone calls -- networked clones don't work in sandboxes.
    new_src = re.sub(
        r"subprocess\.(?:run|check_call|check_output)\(\s*\[[^\]]*?'git'\s*,\s*'clone'[^\]]*?\][^)]*\)\s*",
        INSTALL_NOOP_VALUE + "  # [patched: git clone skipped]\n",
        new_src,
        flags=re.M | re.S,
    )
    if RE_SHUTIL_RMTREE_REPO.search(new_src):
        new_src = RE_SHUTIL_RMTREE_REPO.sub(
            "None  # [patched: would have rm -rf the local repo clone]", new_src
        )
        applied.append("repo rmtree -> None")

    # 2) Colab drive mount
    if RE_MOUNT_TRUE.search(new_src):
        new_src = RE_MOUNT_TRUE.sub("MOUNT_DRIVE = False  # patched (no Colab)", new_src)
        applied.append("MOUNT_DRIVE=False")

    # 3) Rewrite Colab '/content/...' paths to a local results/notebooks_data/<name>
    if "'/content/" in new_src or '"/content/' in new_src:
        new_src = new_src.replace("'/content/", "'./results/notebooks_data/")
        new_src = new_src.replace('"/content/', '"./results/notebooks_data/')
        applied.append("/content/ -> ./results/notebooks_data/")

    # 4) Hard-fail if a getpass IBM token prompt would otherwise block.
    if RE_GETPASS.search(new_src):
        new_src = (
            "import os\n"
            "os.environ.setdefault('IBMQ_TOKEN', '')  # patched: no interactive prompt\n"
            "USE_HARDWARE = False  # patched: force AerSimulator path\n"
            + new_src
        )
        applied.append("disable IBM hardware prompt")

    return new_src, applied


def execute_notebook(nb_path: Path) -> dict:
    """Patch and execute one notebook.  Returns metadata dict."""
    nb = nbformat.read(nb_path, as_version=4)
    patches: list[tuple[int, list[str]]] = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        new_src, applied = patch_source(cell.source)
        if applied:
            cell.source = new_src
            patches.append((i, applied))

    out_path = OUT_DIR / nb_path.name
    t0 = time.time()
    client = NotebookClient(
        nb,
        timeout=CELL_TIMEOUT,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(nb_path.parent.resolve())}},
    )
    err: str | None = None
    try:
        client.execute()
    except Exception as e:  # pragma: no cover -- surfaces kernel-level failures
        err = repr(e)
    dt = time.time() - t0

    nbformat.write(nb, out_path)

    # Count cells with errors
    n_err = 0
    first_err = None
    for cell in nb.cells:
        if cell.cell_type == "code":
            for o in cell.get("outputs", []):
                if o.get("output_type") == "error":
                    n_err += 1
                    if first_err is None:
                        first_err = ":".join(o.get("traceback", [""])[-1:])[:200]

    # Export HTML
    html_path = OUT_DIR / (nb_path.stem + ".html")
    try:
        html_exporter = HTMLExporter()
        body, _ = html_exporter.from_notebook_node(nb)
        html_path.write_text(body)
    except Exception as e:  # pragma: no cover
        html_path.write_text(f"<html><body><pre>HTML export failed: {e}</pre></body></html>")

    return {
        "name": nb_path.name,
        "elapsed_s": round(dt, 1),
        "n_err": n_err,
        "first_err": first_err,
        "n_patched": len(patches),
        "exec_err": err,
    }


def main() -> int:
    nbs = sorted(NB_DIR.glob("*.ipynb"))
    log_lines = []
    summary = []
    for nb in nbs:
        print(f"\n=== {nb.name} ===")
        meta = execute_notebook(nb)
        status = "OK" if meta["n_err"] == 0 and not meta["exec_err"] else "FAIL"
        line = (
            f"{nb.name:42s}  {status:4s}  {meta['elapsed_s']:>6.1f}s  "
            f"patches={meta['n_patched']}  errors={meta['n_err']}"
        )
        if meta["first_err"]:
            line += f"  first_err={meta['first_err']!r}"
        print(line)
        log_lines.append(line)
        summary.append(meta)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("\n".join(log_lines) + "\n")
    print(f"\nlog -> {LOG_FILE}")
    n_fail = sum(1 for s in summary if s["n_err"] > 0 or s["exec_err"])
    print(f"{len(summary) - n_fail}/{len(summary)} notebooks ran without cell errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
