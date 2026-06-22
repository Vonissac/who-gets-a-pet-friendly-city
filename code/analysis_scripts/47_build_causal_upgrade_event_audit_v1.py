#!/usr/bin/env python3
"""Audit event candidates for quasi-causal upgrade.

The audit scores existing rule/evidence records for event-study, DID and
matched-comparison feasibility. It does not estimate causal effects yet.
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
RULES = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_semantic_records_v1.csv"
LEDGER = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_source_ledger_v1.csv"
EVIDENCE_V2 = PROJECT_ROOT / "data" / "derived_data" / "platform" / "round_mining_evidence_ledger_v2.csv"
GRID_YEAR = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_service_background_500m_2019_2025.csv"
GRID_V6 = PROJECT_ROOT / "data" / "derived_data" / "model" / "controlled_grid_model_500m_2025_v6.csv"

OUT_EVENTS = PROJECT_ROOT / "data" / "derived_data" / "model" / "causal_upgrade_event_candidates_v1.csv"
OUT_DESIGNS = PROJECT_ROOT / "data" / "derived_data" / "model" / "causal_upgrade_design_options_v1.csv"
REPORT = PROJECT_ROOT / "_rebuild_outputs" / "reports" / "63_causal_upgrade_event_audit_v1_report.md"

TITLE = "Who Gets a Pet-Friendly City? Rule-Liminal Venues, Pet-Service Ecologies and Companion-Animal Urban Capability in Shenzhen"
CHINESE_TITLE = "谁获得宠物友好城市？深圳规则临界场所、宠物服务生态与伴侣动物城市能力研究"
MECHANISM = "Pet-service ecologies outpace urban readiness in Shenzhen."


EVENT_KEYWORDS = {
    "pet_park_or_public_space_opening": ["park", "trail", "public_space", "公园", "乐园", "郊野径", "小游园"],
    "mobility_or_transport_service": ["bus", "airport", "route", "transport", "候机", "巴士", "机场"],
    "mall_or_commercial_policy": ["mall", "shopping", "commercial", "商场", "购物", "天虹", "COCO", "京基"],
    "hotel_policy": ["hotel", "marriott", "hyatt", "hilton", "酒店"],
    "district_strategy": ["district", "strategy", "target", "罗湖", "政策", "城区", "产业"],
    "standard_or_guideline": ["standard", "guideline", "规范", "标准"],
}


def parse_year(value: object) -> float:
    text = str(value)
    year = pd.to_numeric(text[:4], errors="coerce")
    if pd.isna(year):
        return np.nan
    if int(year) < 2019 or int(year) > 2026:
        return np.nan
    return float(year)


def infer_event_type(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(c, ""))
        for c in [
            "venue_or_area",
            "venue_name",
            "venue_type",
            "sector",
            "rule_position",
            "access_position",
            "rule_dimensions",
            "rule_features",
            "source_title",
            "evidence_summary",
        ]
    ).lower()
    hits = []
    for event_type, kws in EVENT_KEYWORDS.items():
        if any(kw.lower() in text for kw in kws):
            hits.append(event_type)
    return "|".join(hits) if hits else "other_rule_event"


def design_family(event_type: str, geo_level: str) -> str:
    if "standard_or_guideline" in event_type:
        return "methods_context_not_causal"
    if "district_strategy" in event_type or geo_level == "district_strategy":
        return "district_event_study_or_synthetic_control_candidate"
    if "mobility_or_transport_service" in event_type or geo_level == "route":
        return "route_buffer_event_study_candidate"
    if geo_level in {"venue", "public_space", "transport_facility", "venue_complex"}:
        return "point_buffer_event_study_or_matched_grid_candidate"
    if "hotel_policy" in event_type:
        return "sector_matched_comparison_candidate"
    return "matched_comparison_or_sensitivity_candidate"


def read_candidates() -> pd.DataFrame:
    rules = pd.read_csv(RULES, encoding="utf-8-sig")
    ledger = pd.read_csv(LEDGER, encoding="utf-8-sig")
    rules = rules.merge(
        ledger[["source_id", "source_class", "publication_use_status", "source_weight_for_model"]],
        on="source_id",
        how="left",
    )
    r = pd.DataFrame(
        {
            "candidate_id": rules["source_id"],
            "source_table": "rule_semantic_records_v1",
            "event_name": rules["venue_name"],
            "district_or_area": rules["district_or_area"],
            "url": rules["url"],
            "source_date": rules["source_date"],
            "source_class": rules["source_class"],
            "publication_use_status": rules["publication_use_status"],
            "rule_position": rules["access_position"],
            "rule_dimensions": rules["rule_features"],
            "geo_level": rules["geo_level"],
            "grid_id": rules["grid_id"],
            "lon": rules["lon"],
            "lat": rules["lat"],
            "evidence_summary": rules["rule_summary"],
        }
    )
    ev = pd.read_csv(EVIDENCE_V2, encoding="utf-8-sig")
    e = pd.DataFrame(
        {
            "candidate_id": ev["source_id_candidate"],
            "source_table": "round_mining_evidence_ledger_v2",
            "event_name": ev["venue_or_area"],
            "district_or_area": ev["district_or_area"],
            "url": ev["url"],
            "source_date": ev["source_date"],
            "source_class": ev["source_class"],
            "publication_use_status": ev["promotion_status"],
            "rule_position": ev["rule_position"],
            "rule_dimensions": ev["rule_dimensions"],
            "geo_level": ev["geo_level"],
            "grid_id": "",
            "lon": np.nan,
            "lat": np.nan,
            "evidence_summary": ev["evidence_summary"],
        }
    )
    return pd.concat([r, e], ignore_index=True)


def score_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    grid_year = pd.read_csv(GRID_YEAR, encoding="utf-8-sig")
    available_years = sorted(grid_year["year"].dropna().astype(int).unique().tolist())
    min_year, max_year = min(available_years), max(available_years)
    v6 = pd.read_csv(GRID_V6, encoding="utf-8-sig")
    district_grid_counts = v6.groupby("district_name")["grid_id"].nunique().to_dict()

    df = candidates.copy()
    df["event_year"] = df["source_date"].map(parse_year)
    df["has_event_year"] = df["event_year"].notna()
    df["event_type"] = df.apply(infer_event_type, axis=1)
    df["design_family"] = df.apply(lambda r: design_family(str(r["event_type"]), str(r["geo_level"])), axis=1)
    df["has_point_geo"] = df["lon"].notna() & df["lat"].notna()
    df["has_grid_geo"] = df["grid_id"].astype(str).str.startswith("SZ500_")
    df["has_geography"] = df["has_point_geo"] | df["has_grid_geo"] | df["geo_level"].astype(str).isin(
        ["district_strategy", "route", "operator_network_city", "citywide_public_space_list"]
    )
    df["pre_year_count_available"] = df["event_year"].apply(
        lambda y: int(sum(year < y for year in available_years)) if pd.notna(y) else 0
    )
    df["post_year_count_available"] = df["event_year"].apply(
        lambda y: int(sum(year >= y for year in available_years)) if pd.notna(y) else 0
    )
    df["source_quality_score"] = df["source_class"].map(
        {
            "A_official_primary": 3,
            "B_operator_primary": 3,
            "C_reported_operator_detail": 2,
            "D_standard_context": 1,
            "D_secondary_industry_media": 1,
            "E_discovery_or_context": 0,
        }
    ).fillna(1)
    df["district_grid_count"] = df["district_or_area"].map(lambda x: district_grid_counts.get(str(x), np.nan))
    df["has_plausible_controls"] = (
        df["has_grid_geo"]
        | df["has_point_geo"]
        | df["district_grid_count"].fillna(0).ge(30)
        | df["geo_level"].astype(str).isin(["route", "operator_network_city"])
    )
    df["feasibility_score"] = (
        df["has_event_year"].astype(int) * 2
        + df["has_geography"].astype(int) * 2
        + (df["pre_year_count_available"] >= 2).astype(int)
        + (df["post_year_count_available"] >= 1).astype(int)
        + df["has_plausible_controls"].astype(int)
        + df["source_quality_score"]
    )
    df["causal_upgrade_class"] = np.select(
        [
            df["design_family"].eq("methods_context_not_causal"),
            df["feasibility_score"].ge(9),
            df["feasibility_score"].between(7, 8),
            df["feasibility_score"].between(5, 6),
        ],
        [
            "not_causal_context_only",
            "high_feasibility_quasi_causal_candidate",
            "medium_feasibility_needs_geocoding_or_timing",
            "low_feasibility_sensitivity_only",
        ],
        default="not_ready_for_causal_design",
    )
    df["minimum_next_step"] = np.select(
        [
            ~df["has_event_year"],
            ~df["has_geography"],
            df["pre_year_count_available"] < 2,
            ~df["has_plausible_controls"],
            df["causal_upgrade_class"].eq("high_feasibility_quasi_causal_candidate"),
        ],
        [
            "find_or_confirm_event_date",
            "geocode_event_or_define_route_district_unit",
            "event_too_early_for_pre_window_or_needs_other_outcome",
            "construct_matched_controls",
            "build_treated_control_panel_and_run_event_study",
        ],
        default="manual_review_then_design_selection",
    )
    return df.sort_values(["feasibility_score", "source_quality_score"], ascending=False)


def design_options(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    top = events[events["causal_upgrade_class"].isin(
        ["high_feasibility_quasi_causal_candidate", "medium_feasibility_needs_geocoding_or_timing"]
    )].copy()
    for _, row in top.head(50).iterrows():
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "event_name": row["event_name"],
                "event_year": row["event_year"],
                "design_family": row["design_family"],
                "recommended_treated_unit": "500m_grid_buffer_or_district_unit",
                "candidate_outcomes": "pet_core_service_density_per_km2|positive_rule_presence|suppressed_frontier_status|rule_liminal_host_count",
                "candidate_controls": "pre_event_pet_service_density|morphology|district|edge_cell|venue_type_mix|source_quality",
                "main_risk": "event timing may be source publication date rather than true intervention date",
                "allowed_claim": "quasi-causal evidence consistent with rule-emergence mechanism if assumptions pass",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_EVENTS.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    events = score_candidates(read_candidates())
    designs = design_options(events)
    events.to_csv(OUT_EVENTS, index=False, encoding="utf-8-sig")
    designs.to_csv(OUT_DESIGNS, index=False, encoding="utf-8-sig")

    counts = {
        "candidate_rows": int(len(events)),
        "causal_upgrade_class": events["causal_upgrade_class"].value_counts().to_dict(),
        "design_family": events["design_family"].value_counts().head(12).to_dict(),
        "event_years": events["event_year"].dropna().astype(int).value_counts().sort_index().to_dict(),
        "design_option_rows": int(len(designs)),
    }
    top_rows = events.head(30)[
        [
            "candidate_id",
            "event_name",
            "district_or_area",
            "event_year",
            "event_type",
            "design_family",
            "source_class",
            "feasibility_score",
            "causal_upgrade_class",
            "minimum_next_step",
        ]
    ].to_dict("records")
    REPORT.write_text(
        f"""# 63 Causal Upgrade Event Audit V1 Report

Locked title: {TITLE}

Chinese locked title: {CHINESE_TITLE}

Core mechanism sentence: {MECHANISM}

## Outputs

- Event candidates: `{OUT_EVENTS.relative_to(PROJECT_ROOT)}`
- Design options: `{OUT_DESIGNS.relative_to(PROJECT_ROOT)}`

## Counts

`{counts}`

## Top Candidates

`{top_rows}`

## Interpretation

This audit identifies which existing records can support event-study,
difference-in-differences or matched-comparison designs. It does not estimate
causal effects. A candidate is high feasibility only when timing, geography,
pre/post data and plausible controls are all present enough to justify a next
design step.

## Claim Boundary

Until a treated-control panel is built and identification assumptions are
checked, the paper should use causal language only as a methodological ambition,
not as a result claim.
""",
        encoding="utf-8",
    )
    print(counts)
    print(OUT_EVENTS)
    print(OUT_DESIGNS)


if __name__ == "__main__":
    main()
