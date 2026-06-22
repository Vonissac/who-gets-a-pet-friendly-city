"""Verify that the analysis scripts can locate their input files inside this package.

This walks every script in ``analysis_scripts/``, statically resolves the input
files it reads (``read_csv`` / ``read_file`` / ``open`` targets that are built
from the package data root) and checks whether each file is present.

Run from anywhere:

    python code/verify_package_inputs.py

A script is reported as FULLY RUNNABLE when every input it reads is present in
``data/derived_data``. A small number of upstream scripts also need
intermediate tables that are derived directly from licensed raw inputs (raw POI
archive, raw source captures); those tables are not redistributed and are listed
separately so the reproduction boundary is explicit rather than hidden.
"""

import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

try:
    from config import PACKAGE_ROOT
except Exception:  # allow running from any working directory
    PACKAGE_ROOT = Path(__file__).resolve()
    for _p in PACKAGE_ROOT.parents:
        if (_p / "data").is_dir():
            PACKAGE_ROOT = _p
            break

ANALYSIS = PACKAGE_ROOT / "code" / "analysis_scripts"

# Inputs that are derived directly from licensed raw inputs and are intentionally
# not redistributed (see the reproduction boundary in CODE_README.md).
RESTRICTED_INTERMEDIATES = {
    "pet_service_core_A_deduplicated.csv",
    "full_silent_rule_venues_2025_v1.csv",
    "hilton_shenzhen_structured_pet_policies_v1.csv",
}

READ_CALL = re.compile(r"(?:read_csv|read_file|read_parquet|read_excel|open)\(\s*([A-Z_][A-Z0-9_]*)")
VAR_PATH = re.compile(r'^([A-Z_][A-Z0-9_]*) = PROJECT_ROOT((?: / "[^"]+")+)', re.M)


def resolve_inputs(src):
    varpath = {}
    for m in VAR_PATH.finditer(src):
        parts = re.findall(r'"([^"]+)"', m.group(2))
        varpath[m.group(1)] = PACKAGE_ROOT.joinpath(*parts)
    reads = set(READ_CALL.findall(src))
    return {v: varpath[v] for v in reads if v in varpath}


def main():
    runnable, partial, hard_missing = [], [], []
    for f in sorted(ANALYSIS.glob("*.py")):
        inputs = resolve_inputs(f.read_text(encoding="utf-8"))
        missing = [p for p in inputs.values() if not p.exists()]
        if not missing:
            runnable.append(f.name)
        elif all(p.name in RESTRICTED_INTERMEDIATES for p in missing):
            partial.append((f.name, [p.name for p in missing]))
        else:
            hard_missing.append((f.name, [str(p.relative_to(PACKAGE_ROOT)) for p in missing]))

    print(f"Package root: {PACKAGE_ROOT}")
    print(f"\nFully runnable from the package ({len(runnable)} scripts):")
    for n in runnable:
        print(f"  OK   {n}")
    if partial:
        print(f"\nNeed a restricted raw-derived intermediate ({len(partial)} scripts):")
        for n, miss in partial:
            print(f"  SKIP {n}  (needs: {', '.join(miss)})")
    if hard_missing:
        print(f"\nUNEXPECTED missing inputs ({len(hard_missing)} scripts):")
        for n, miss in hard_missing:
            print(f"  FAIL {n}")
            for m in miss:
                print(f"        - {m}")
        raise SystemExit(1)
    print("\nInput check passed: all non-restricted analysis inputs are present.")


if __name__ == "__main__":
    main()
