#!/usr/bin/env python3
"""Build Round 3 primary-source sensitivity layer.

This does not rebuild the full v5 model. It tests whether the current rule
support layer is dominated by weak sources by recomputing grid/district support
using A/B primary sources only and comparing it with the existing v5 pattern.
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
GRID = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_emergence_suppression_indices_500m_2025_v5.csv"
RULES = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_semantic_records_v1.csv"
LEDGER = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_source_ledger_v1.csv"
OUT_GRID = PROJECT_ROOT / "data" / "derived_data" / "model" / "primary_source_sensitivity_grid_v1.csv"
OUT_DISTRICT = PROJECT_ROOT / "data" / "derived_data" / "model" / "primary_source_sensitivity_district_v1.csv"
REPORT = PROJECT_ROOT / "_rebuild_outputs" / "reports" / "59_primary_source_sensitivity_v1_report.md"

TITLE = "Pet-Service Ecologies Outpace Urban Readiness in Shenzhen: Rule-Liminal Venues and the Uneven Emergence of Companion-Animal Urban Capability"
CHINESE_TITLE = "宠物城市生态已到，城市尚未准备好：深圳伴侣动物空间生态的规则临界与共处能力涌现机制"


def robust_scale(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    lo = s.quantile(0.05)
    hi = s.quantile(0.95)
    if hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s.clip(lo, hi) - lo) / (hi - lo)).clip(0, 1)


def source_group(row: pd.Series) -> str:
    cls = str(row.get("source_class", ""))
    status = str(row.get("publication_use_status", ""))
    if cls in {"A_official_primary", "B_operator_primary"} and status in {"main_model", "main_model_with_caution"}:
        return "primary_main_or_caution"
    if cls in {"A_official_primary", "B_operator_primary"}:
        return "primary_not_main"
    if cls.startswith("C_"):
        return "reported_operator_detail"
    return "weaker_or_context"


def main() -> None:
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(GRID, encoding="utf-8-sig")
    rules = pd.read_csv(RULES, encoding="utf-8-sig")
    ledger = pd.read_csv(LEDGER, encoding="utf-8-sig")
    ledger["source_group"] = ledger.apply(source_group, axis=1)
    rules = rules.merge(
        ledger[
            [
                "source_id",
                "source_class",
                "publication_use_status",
                "source_weight_for_model",
                "source_group",
                "manual_verification_status",
                "publication_claim_boundary",
            ]
        ],
        on="source_id",
        how="left",
    )
    primary = rules[rules["source_group"] == "primary_main_or_caution"].copy()
    primary["is_positive_primary"] = primary["access_semantic_score"].fillna(0) > 0
    primary["is_restrictive_primary"] = primary["restriction_intensity"].fillna(0) > 0
    primary["is_conditional_primary"] = primary["conditionality_intensity"].fillna(0) > 0

    by_grid = (
        primary.groupby("grid_id")
        .agg(
            primary_rule_count=("source_id", "nunique"),
            primary_positive_rule_count=("is_positive_primary", "sum"),
            primary_restrictive_rule_count=("is_restrictive_primary", "sum"),
            primary_conditional_rule_count=("is_conditional_primary", "sum"),
            primary_mean_access_semantic_score=("access_semantic_score", "mean"),
            primary_mean_weight=("source_weight_for_model", "mean"),
            primary_source_ids=("source_id", lambda s: "|".join(sorted(set(map(str, s)))[:20])),
        )
        .reset_index()
    )
    out = grid.merge(by_grid, on="grid_id", how="left")
    fill_cols = [
        "primary_rule_count",
        "primary_positive_rule_count",
        "primary_restrictive_rule_count",
        "primary_conditional_rule_count",
        "primary_mean_access_semantic_score",
        "primary_mean_weight",
    ]
    out[fill_cols] = out[fill_cols].fillna(0)
    out["primary_source_ids"] = out["primary_source_ids"].fillna("")

    out["primary_positive_rule_norm"] = robust_scale(
        out["primary_positive_rule_count"] + out["primary_mean_access_semantic_score"].clip(lower=0)
    )
    out["primary_rule_gap_norm"] = ((out["pet_ecology_norm"] + out["liminal_host_norm"]) / 2 - out["primary_positive_rule_norm"]).clip(
        lower=0, upper=1
    )
    out["primary_sensitive_suppression_index"] = (
        0.35 * out["primary_rule_gap_norm"]
        + 0.25 * out["rule_silence_norm"]
        + 0.20 * out["restrictive_norm"]
        + 0.20 * out["pet_ecology_norm"] * (1 - out["primary_positive_rule_norm"])
    ).clip(0, 1)
    out["primary_support_delta"] = out["positive_rule_norm"] - out["primary_positive_rule_norm"]
    out["suppression_delta_primary_minus_v5"] = out["primary_sensitive_suppression_index"] - out["suppression_index"]
    out["primary_source_support_class"] = np.select(
        [
            (out["positive_rule_count"] > 0) & (out["primary_positive_rule_count"] > 0),
            (out["positive_rule_count"] > 0) & (out["primary_positive_rule_count"] <= 0),
            (out["positive_rule_count"] <= 0) & (out["primary_rule_count"] > 0),
        ],
        [
            "positive_rule_supported_by_primary",
            "positive_rule_depends_on_nonprimary_or_unmatched",
            "primary_rule_present_but_not_positive",
        ],
        default="no_primary_rule_signal",
    )

    out.to_csv(OUT_GRID, index=False, encoding="utf-8-sig")

    district_rows = []
    for district, g in out.groupby("district_name"):
        weights = g["valid_area_km2"].clip(lower=0.0001).to_numpy()
        district_rows.append(
            {
                "district_name": district,
                "grid_count": int(len(g)),
                "primary_rule_count": int(g["primary_rule_count"].sum()),
                "primary_positive_rule_count": int(g["primary_positive_rule_count"].sum()),
                "v5_positive_rule_count": int(g["positive_rule_count"].sum()),
                "mean_v5_suppression_index": float(np.average(g["suppression_index"], weights=weights)),
                "mean_primary_sensitive_suppression_index": float(
                    np.average(g["primary_sensitive_suppression_index"], weights=weights)
                ),
                "mean_suppression_delta_primary_minus_v5": float(
                    np.average(g["suppression_delta_primary_minus_v5"], weights=weights)
                ),
                "suppressed_frontier_grids_v5": int((g["grid_emergence_type"] == "suppressed_emergence_frontier").sum()),
                "primary_supported_positive_grids": int(
                    (g["primary_source_support_class"] == "positive_rule_supported_by_primary").sum()
                ),
                "positive_depends_on_nonprimary_grids": int(
                    (g["primary_source_support_class"] == "positive_rule_depends_on_nonprimary_or_unmatched").sum()
                ),
            }
        )
    district = pd.DataFrame(district_rows)
    district["primary_sensitive_suppression_rank"] = district["mean_primary_sensitive_suppression_index"].rank(
        ascending=False, method="dense"
    ).astype(int)
    district.to_csv(OUT_DISTRICT, index=False, encoding="utf-8-sig")

    counts = {
        "rules_total": int(len(rules)),
        "primary_main_or_caution_rules": int(len(primary)),
        "grid_rows": int(len(out)),
        "primary_source_support_class": out["primary_source_support_class"].value_counts().to_dict(),
        "district_rows": int(len(district)),
    }
    district_summary = district.sort_values("primary_sensitive_suppression_rank").to_dict("records")
    fragile_grids = out.sort_values("primary_support_delta", ascending=False).head(30)[
        [
            "grid_id",
            "district_name",
            "grid_emergence_type",
            "positive_rule_count",
            "primary_positive_rule_count",
            "positive_rule_norm",
            "primary_positive_rule_norm",
            "primary_support_delta",
            "suppression_index",
            "primary_sensitive_suppression_index",
            "primary_source_support_class",
        ]
    ].to_dict("records")
    REPORT.write_text(
        f"""# 59 Primary-Source Sensitivity V1 Report

Locked title: {TITLE}

Chinese locked title: {CHINESE_TITLE}

## Purpose

Round 3 tests whether the rule-readiness pattern survives a stricter evidence
filter. Only A official primary and B operator primary sources with main-model
or caution status are counted as primary support.

## Outputs

- Grid sensitivity: `{OUT_GRID.relative_to(PROJECT_ROOT)}`
- District sensitivity: `{OUT_DISTRICT.relative_to(PROJECT_ROOT)}`

## Counts

`{counts}`

## District Sensitivity Summary

`{district_summary}`

## Most Source-Fragile Grids

`{fragile_grids}`

## Interpretation

If suppression increases under the primary-source-only layer, that does not
invalidate the mechanism. It indicates that visible positive rule support is
thin after weak sources are removed, which is exactly the rule-readiness deficit
the paper must measure carefully.

## Boundary

This sensitivity test is not a final causal model. It is an evidence-quality
stress test for the rule layer.
""",
        encoding="utf-8",
    )
    print(counts)
    print(OUT_GRID)
    print(OUT_DISTRICT)


if __name__ == "__main__":
    main()
