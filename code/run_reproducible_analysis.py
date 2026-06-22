#!/usr/bin/env python3
"""Run all analysis scripts supported by the supplied derived data.

The repository includes nine analysis scripts that can be rerun without
restricted raw-source intermediates. This runner executes those scripts in the
same order as the code README and writes a concise run summary to
``_rebuild_outputs/reports``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve()
for parent in PACKAGE_ROOT.parents:
    if (parent / "data").is_dir():
        PACKAGE_ROOT = parent
        break

SCRIPT_DIR = PACKAGE_ROOT / "code" / "analysis_scripts"
REPORT_DIR = PACKAGE_ROOT / "_rebuild_outputs" / "reports"

RUNNABLE_SCRIPTS = [
    "34_schelling_full_rule_adoption_simulation_v2.py",
    "38_build_rule_liminal_threshold_model_v4.py",
    "40_build_regional_emergence_suppression_indices_v5.py",
    "43_build_primary_source_sensitivity_v1.py",
    "46_build_controlled_grid_model_v6.py",
    "47_build_causal_upgrade_event_audit_v1.py",
    "48_run_final_systematic_models_v7.py",
    "66_repair_sparse_rule_scaling_v51.py",
    "67_build_controlled_grid_model_v61_sparse_repaired.py",
]


def run_script(script_name: str) -> dict[str, object]:
    script = SCRIPT_DIR / script_name
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    start = time.time()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PACKAGE_ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=900,
    )
    stdout = proc.stdout.replace(str(PACKAGE_ROOT), ".")
    stderr = proc.stderr.replace(str(PACKAGE_ROOT), ".")
    return {
        "script": script_name,
        "returncode": proc.returncode,
        "seconds": round(time.time() - start, 2),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for script_name in RUNNABLE_SCRIPTS:
        print(f"RUN {script_name}", flush=True)
        result = run_script(script_name)
        results.append(result)
        print(f"  returncode={result['returncode']} seconds={result['seconds']}", flush=True)
        if result["returncode"] != 0:
            break

    summary_path = REPORT_DIR / "reproducible_analysis_run_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    failures = [r for r in results if r["returncode"] != 0]
    if failures:
        print(f"FAILED: {failures[0]['script']}", file=sys.stderr)
        print(f"Run summary: {summary_path}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Completed {len(results)} reproducible analysis scripts.")
    print(f"Run summary: {summary_path.relative_to(PACKAGE_ROOT)}")


if __name__ == "__main__":
    main()
