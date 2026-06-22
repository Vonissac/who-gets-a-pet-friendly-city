#!/usr/bin/env python3
"""Repair sparse rule-signal scaling without overwriting v5.

The original v5 robust scaling uses q05-q95. That is suitable for continuous
urban signals, but fails for very sparse rule variables whose 95th percentile is
still zero. This script creates a v5.1 audit layer with sparse-aware rule
normalisation and recomputed diagnostic indices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve()
for _p in PROJECT_ROOT.parents:
    if (_p / "data").is_dir():
        PROJECT_ROOT = _p
        break
(PROJECT_ROOT / "_rebuild_outputs" / "reports").mkdir(parents=True, exist_ok=True)
V5 = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_emergence_suppression_indices_500m_2025_v5.csv"
OUT = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_emergence_suppression_indices_500m_2025_v51_sparse_repaired.csv"
AUDIT = PROJECT_ROOT / "data" / "derived_data" / "model" / "sparse_rule_scaling_audit_v51.csv"
REPORT = PROJECT_ROOT / "_rebuild_outputs" / "reports" / "66_sparse_rule_scaling_repair_v51_report.md"


def robust_scale(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    lo = s.quantile(0.05)
    hi = s.quantile(0.95)
    if hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s.clip(lo, hi) - lo) / (hi - lo)).clip(0, 1)


def sparse_scale(series: pd.Series) -> pd.Series:
    """Scale sparse non-negative signals while preserving zero as zero.

    For variables with many zeros, q95 may be zero. Use the positive
    distribution's 95th percentile and log compression; if that also fails,
    fall back to the positive maximum.
    """

    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(float).clip(lower=0)
    pos = s[s > 0]
    if pos.empty:
        return pd.Series(np.zeros(len(s)), index=s.index)
    hi = float(pos.quantile(0.95))
    if hi <= 0:
        hi = float(pos.max())
    if hi <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (np.log1p(s.clip(upper=hi)) / np.log1p(hi)).clip(0, 1)


def classify(row: pd.Series) -> str:
    e = row["emergence_index_v51"]
    s = row["suppression_index_v51"]
    if e >= 0.70 and s < 0.45:
        return "emergent_capability_core"
    if e >= 0.55 and s >= 0.55:
        return "suppressed_emergence_frontier"
    if e < 0.45 and s >= 0.60:
        return "ecology_without_rule_readiness"
    if e >= 0.50 and row["positive_rule_norm_v51"] > row["pet_ecology_norm"]:
        return "rule_first_demonstration_zone"
    if e < 0.35 and s < 0.35:
        return "low_pet_city_signal"
    return "mixed_or_transitional_zone"


def main() -> None:
    df = pd.read_csv(V5, encoding="utf-8-sig")
    rule_signal = df["positive_rule_count"] + df["mean_explicit_open_rule_exposure"]
    liminal_exposure = df["mean_liminal_rule_exposure"]

    df["positive_rule_norm_v51"] = sparse_scale(rule_signal)
    df["rule_liminal_exposure_norm_v51"] = sparse_scale(liminal_exposure)
    df["positive_rule_present"] = (df["positive_rule_count"] > 0).astype(int)
    df["explicit_rule_present"] = (df["explicit_rule_count"] > 0).astype(int)
    df["explicit_rule_gap_norm_v51"] = (
        (df["pet_ecology_norm"] + df["liminal_host_norm"]) / 2 - df["positive_rule_norm_v51"]
    ).clip(lower=0, upper=1)

    df["emergence_index_v51"] = (
        0.28 * df["pet_ecology_norm"]
        + 0.24 * df["liminal_host_norm"]
        + 0.18 * df["liminal_potential_norm"]
        + 0.14 * df["rule_liminal_exposure_norm_v51"]
        + 0.10 * df["topology_norm"]
        + 0.06 * df["positive_rule_norm_v51"]
    ).clip(0, 1)
    open_exposure_norm = sparse_scale(df["mean_explicit_open_rule_exposure"])
    df["suppression_index_v51"] = (
        0.30 * df["explicit_rule_gap_norm_v51"]
        + 0.25 * df["rule_silence_norm"]
        + 0.18 * df["restrictive_norm"]
        + 0.15 * df["pet_ecology_norm"] * (1 - df["positive_rule_norm_v51"])
        + 0.12 * df["liminal_host_norm"] * (1 - open_exposure_norm)
    ).clip(0, 1)
    df["emergence_suppression_gap_v51"] = df["emergence_index_v51"] - df["suppression_index_v51"]
    df["readiness_deficit_index_v51"] = (df["suppression_index_v51"] - df["emergence_index_v51"]).clip(lower=0)
    df["grid_emergence_type_v51"] = df.apply(classify, axis=1)

    audit_rows = []
    for col in [
        "positive_rule_count",
        "mean_explicit_open_rule_exposure",
        "positive_rule_norm",
        "positive_rule_norm_v51",
        "mean_liminal_rule_exposure",
        "rule_liminal_exposure_norm",
        "rule_liminal_exposure_norm_v51",
        "emergence_index",
        "emergence_index_v51",
        "suppression_index",
        "suppression_index_v51",
    ]:
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)
        audit_rows.append(
            {
                "field": col,
                "nonzero": int((s != 0).sum()),
                "min": float(s.min()),
                "p50": float(s.quantile(0.5)),
                "p95": float(s.quantile(0.95)),
                "max": float(s.max()),
                "mean": float(s.mean()),
            }
        )
    audit = pd.DataFrame(audit_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    audit.to_csv(AUDIT, index=False, encoding="utf-8-sig")

    report = f"""# 66 Sparse Rule Scaling Repair V5.1

## Why This Repair Exists

The v5 grid table contained non-zero `positive_rule_count` and
`mean_liminal_rule_exposure`, but their robust-scaled fields
`positive_rule_norm` and `rule_liminal_exposure_norm` were all zero. The cause
is not missing rule data. It is a scaling failure: the original q05-q95 robust
scale collapses highly sparse variables when the 95th percentile is still zero.

## Outputs

- Repaired grid table: `{OUT.relative_to(PROJECT_ROOT)}`
- Scaling audit: `{AUDIT.relative_to(PROJECT_ROOT)}`

## Audit Summary

`{audit.to_dict("records")}`

## Use Rule

For manuscript figures and model-audit graphics, use the v5.1 repaired fields
for sparse rule signals:

- `positive_rule_norm_v51`
- `rule_liminal_exposure_norm_v51`
- `emergence_index_v51`
- `suppression_index_v51`
- `grid_emergence_type_v51`

Keep the original v5 fields as provenance, but do not use all-zero sparse
normalised fields as evidence in final figures.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(audit.to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()
