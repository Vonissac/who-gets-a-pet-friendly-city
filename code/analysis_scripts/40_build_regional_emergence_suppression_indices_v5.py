#!/usr/bin/env python3
"""Build grid- and district-level emergence/suppression indices.

V5 converts the v3/v4 mechanism layers into regional indices:

- Emergence index: pet-service ecology, rule-liminal hosts, network position and
  rule-liminal exposure combine to signal urban capability emergence.
- Suppression index: pet ecology and liminal hosts are present but explicit
  rules are thin, restrictive pressure exists, or rule silence dominates.

The indices are diagnostic and comparative, not causal estimates.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve()
for _p in PROJECT_ROOT.parents:
    if (_p / "data").is_dir():
        PROJECT_ROOT = _p
        break
(PROJECT_ROOT / "_rebuild_outputs" / "reports").mkdir(parents=True, exist_ok=True)
GRID_LIMINAL = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_rule_liminality_500m_2025_v4.csv"
TOPO = PROJECT_ROOT / "data" / "derived_data" / "network" / "grid_pet_rule_topology_2025_v3.csv"
GRID_GEO = PROJECT_ROOT / "data" / "derived_data" / "geo" / "shenzhen_grid_500m.gpkg"

OUT_GRID = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_emergence_suppression_indices_500m_2025_v5.csv"
OUT_GRID_GEO = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_emergence_suppression_indices_500m_2025_v5.geojson"
OUT_DISTRICT = PROJECT_ROOT / "data" / "derived_data" / "model" / "district_emergence_suppression_indices_2025_v5.csv"
REPORT = PROJECT_ROOT / "_rebuild_outputs" / "reports" / "56_regional_emergence_suppression_indices_v5_report.md"
METHOD = PROJECT_ROOT / "REGIONAL_EMERGENCE_SUPPRESSION_INDICES_v5.md"

TITLE = "Pet-Service Ecologies Outpace Urban Readiness in Shenzhen: Rule-Liminal Venues and the Uneven Emergence of Companion-Animal Urban Capability"
CHINESE_TITLE = "宠物城市生态已到，城市尚未准备好：深圳伴侣动物空间生态的规则临界与共处能力涌现机制"


def robust_scale(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    lo = s.quantile(0.05)
    hi = s.quantile(0.95)
    if hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s.clip(lo, hi) - lo) / (hi - lo)).clip(0, 1)


def classify_region(row: pd.Series) -> str:
    e = row["emergence_index"]
    s = row["suppression_index"]
    if e >= 0.70 and s < 0.45:
        return "emergent_capability_core"
    if e >= 0.55 and s >= 0.55:
        return "suppressed_emergence_frontier"
    if e < 0.45 and s >= 0.60:
        return "ecology_without_rule_readiness"
    if e >= 0.50 and row["positive_rule_norm"] > row["pet_ecology_norm"]:
        return "rule_first_demonstration_zone"
    if e < 0.35 and s < 0.35:
        return "low_pet_city_signal"
    return "mixed_or_transitional_zone"


def build_grid_indices() -> pd.DataFrame:
    grid = pd.read_csv(GRID_LIMINAL)
    topo = pd.read_csv(TOPO)[["grid_id", "pet_rule_pagerank", "weighted_topological_degree", "topology_ecology_class"]]
    df = grid.merge(topo, on="grid_id", how="left", suffixes=("", "_topo"))
    df["topology_ecology_class"] = df["topology_ecology_class"].fillna("low_network_pet_rule_signal")

    df["pet_ecology_norm"] = robust_scale(
        np.log1p(df["pet_service_exposure_500m"]) + 0.35 * df["pet_service_diversity_entropy_2025"]
    )
    df["liminal_host_norm"] = robust_scale(
        df["liminal_or_frontier_hosts"] + 0.35 * df["liminal_threshold_core_hosts"]
    )
    df["liminal_potential_norm"] = robust_scale(df["p90_liminal_potential"])
    df["rule_liminal_exposure_norm"] = robust_scale(df["mean_liminal_rule_exposure"])
    df["positive_rule_norm"] = robust_scale(df["positive_rule_count"] + df["mean_explicit_open_rule_exposure"])
    df["topology_norm"] = robust_scale(
        df["pet_rule_pagerank"].fillna(0) + 0.00001 * df["weighted_topological_degree"].fillna(0)
    )
    df["rule_silence_norm"] = robust_scale(
        df["ecology_silent_hosts"] + np.where(df["positive_rule_count"] <= 0, df["pet_service_exposure_500m"], 0)
    )
    df["restrictive_norm"] = robust_scale(df["restricted_hosts"] + df["restrictive_rule_count"])
    df["explicit_rule_gap_norm"] = (
        (df["pet_ecology_norm"] + df["liminal_host_norm"]) / 2 - df["positive_rule_norm"]
    ).clip(lower=0, upper=1)

    df["emergence_index"] = (
        0.28 * df["pet_ecology_norm"]
        + 0.24 * df["liminal_host_norm"]
        + 0.18 * df["liminal_potential_norm"]
        + 0.14 * df["rule_liminal_exposure_norm"]
        + 0.10 * df["topology_norm"]
        + 0.06 * df["positive_rule_norm"]
    ).clip(0, 1)
    df["suppression_index"] = (
        0.30 * df["explicit_rule_gap_norm"]
        + 0.25 * df["rule_silence_norm"]
        + 0.18 * df["restrictive_norm"]
        + 0.15 * df["pet_ecology_norm"] * (1 - df["positive_rule_norm"])
        + 0.12 * df["liminal_host_norm"] * (1 - df["mean_explicit_open_rule_exposure"].pipe(robust_scale))
    ).clip(0, 1)
    df["emergence_suppression_gap"] = df["emergence_index"] - df["suppression_index"]
    df["readiness_deficit_index"] = (df["suppression_index"] - df["emergence_index"]).clip(lower=0)
    df["grid_emergence_type"] = df.apply(classify_region, axis=1)
    return df


def build_district_indices(grid: pd.DataFrame) -> pd.DataFrame:
    weight = grid["valid_area_km2"].clip(lower=0.0001)
    work = grid.copy()
    work["_w"] = weight
    rows = []
    for district, g in work.groupby("district_name"):
        w = g["_w"].to_numpy(dtype=float)
        rows.append(
            {
                "district_name": district,
                "grid_count": int(len(g)),
                "valid_area_km2": float(g["valid_area_km2"].sum()),
                "pet_service_points": int(g["pet_service_count_2025"].sum()),
                "host_candidates": int(g["host_candidates"].sum()),
                "liminal_or_frontier_hosts": int(g["liminal_or_frontier_hosts"].sum()),
                "ecology_silent_hosts": int(g["ecology_silent_hosts"].sum()),
                "positive_rule_count": int(g["positive_rule_count"].sum()),
                "restrictive_rule_count": int(g["restrictive_rule_count"].sum()),
                "emergence_index_mean": float(np.average(g["emergence_index"], weights=w)),
                "suppression_index_mean": float(np.average(g["suppression_index"], weights=w)),
                "readiness_deficit_index_mean": float(np.average(g["readiness_deficit_index"], weights=w)),
                "p90_emergence_index": float(g["emergence_index"].quantile(0.9)),
                "p90_suppression_index": float(g["suppression_index"].quantile(0.9)),
                "emergent_core_grids": int((g["grid_emergence_type"] == "emergent_capability_core").sum()),
                "suppressed_frontier_grids": int((g["grid_emergence_type"] == "suppressed_emergence_frontier").sum()),
                "ecology_without_rule_grids": int((g["grid_emergence_type"] == "ecology_without_rule_readiness").sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["emergence_rank"] = out["emergence_index_mean"].rank(ascending=False, method="dense").astype(int)
    out["suppression_rank"] = out["suppression_index_mean"].rank(ascending=False, method="dense").astype(int)
    out["district_mechanism_type"] = np.select(
        [
            (out["emergence_index_mean"] >= out["emergence_index_mean"].quantile(0.70))
            & (out["suppression_index_mean"] < out["suppression_index_mean"].median()),
            (out["emergence_index_mean"] >= out["emergence_index_mean"].median())
            & (out["suppression_index_mean"] >= out["suppression_index_mean"].median()),
            (out["emergence_index_mean"] < out["emergence_index_mean"].median())
            & (out["suppression_index_mean"] >= out["suppression_index_mean"].median()),
        ],
        [
            "district_emergence_core",
            "district_suppressed_emergence",
            "district_ecology_rule_deficit",
        ],
        default="district_low_or_mixed_signal",
    )
    return out.sort_values(["emergence_index_mean", "suppression_index_mean"], ascending=False)


def write_outputs() -> None:
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    grid = build_grid_indices()
    district = build_district_indices(grid)
    grid.to_csv(OUT_GRID, index=False, encoding="utf-8-sig")
    district.to_csv(OUT_DISTRICT, index=False, encoding="utf-8-sig")
    geo = gpd.read_file(GRID_GEO)
    geo_out = geo.merge(grid, on=["grid_id", "district_name", "edge_cell", "intersect_area_m2"], how="left")
    geo_out.to_crs("EPSG:4326").to_file(OUT_GRID_GEO, driver="GeoJSON")

    counts = {
        "grid_rows": int(len(grid)),
        "grid_emergence_type": grid["grid_emergence_type"].value_counts().to_dict(),
        "district_rows": int(len(district)),
        "district_mechanism_type": district["district_mechanism_type"].value_counts().to_dict(),
        "mean_emergence_index": float(grid["emergence_index"].mean()),
        "mean_suppression_index": float(grid["suppression_index"].mean()),
        "p90_emergence_index": float(grid["emergence_index"].quantile(0.9)),
        "p90_suppression_index": float(grid["suppression_index"].quantile(0.9)),
    }
    top_emergence = grid.sort_values("emergence_index", ascending=False).head(30)[
        [
            "grid_id",
            "district_name",
            "grid_emergence_type",
            "emergence_index",
            "suppression_index",
            "pet_ecology_norm",
            "liminal_host_norm",
            "positive_rule_norm",
            "rule_silence_norm",
            "pet_rule_ecology_class",
            "grid_liminal_type",
        ]
    ].to_dict("records")
    top_suppression = grid.sort_values("suppression_index", ascending=False).head(30)[
        [
            "grid_id",
            "district_name",
            "grid_emergence_type",
            "emergence_index",
            "suppression_index",
            "readiness_deficit_index",
            "pet_ecology_norm",
            "positive_rule_norm",
            "rule_silence_norm",
            "restrictive_norm",
            "pet_rule_ecology_class",
        ]
    ].to_dict("records")
    district_rows = district.to_dict("records")
    REPORT.write_text(
        f"""# 56 Regional Emergence And Suppression Indices V5 Report

Locked title: {TITLE}

Chinese locked title: {CHINESE_TITLE}

## Outputs

- Grid indices: `{OUT_GRID.relative_to(PROJECT_ROOT)}`
- Grid indices GeoJSON: `{OUT_GRID_GEO.relative_to(PROJECT_ROOT)}`
- District indices: `{OUT_DISTRICT.relative_to(PROJECT_ROOT)}`

## Counts

`{counts}`

## District Summary

`{district_rows}`

## Top Emergence Grids

`{top_emergence}`

## Top Suppression Grids

`{top_suppression}`

## Interpretation

The emergence index captures whether pet-service ecology, rule-liminal hosts, liminal rule exposure and spatial-network centrality are jointly present. The suppression index captures the opposite tension: pet ecology and liminal venues are present, but explicit positive rules are thin or restrictive/silent signals dominate.

The most interesting manuscript mechanism is not high emergence alone. It is the combination of high emergence and high suppression: these are areas where the pet city is already active but urban rule readiness has not caught up.

## Claim Boundary

These indices are diagnostic composites. They should be interpreted as regional mechanism signals, not as causal effects or direct measurements of actual pet access.
""",
        encoding="utf-8",
    )
    METHOD.write_text(
        f"""# Regional Emergence And Suppression Indices V5

Locked title: **{TITLE}**

Chinese locked title: **{CHINESE_TITLE}**

## Emergence Index

The emergence index combines:

- pet-service ecology intensity and diversity;
- rule-liminal and frontier host concentration;
- p90 liminal potential;
- liminal rule exposure;
- grid topology centrality;
- explicit positive rules as partial institutionalization.

## Suppression Index

The suppression index combines:

- explicit rule gap between pet ecology and visible positive rules;
- rule-silent host concentration;
- restrictive rule pressure;
- pet ecology without rule support;
- liminal host concentration without explicit rule support.

## Typology

| Type | Meaning |
|---|---|
| `emergent_capability_core` | high emergence, low suppression |
| `suppressed_emergence_frontier` | high emergence and high suppression |
| `ecology_without_rule_readiness` | ecology exists but rules lag strongly |
| `rule_first_demonstration_zone` | rules visible before strong ecology |
| `low_pet_city_signal` | low signal on both dimensions |
| `mixed_or_transitional_zone` | intermediate pattern |

## Boundary

The indices are designed for spatial diagnosis and figure-making. They do not confirm venue-level pet access.
""",
        encoding="utf-8",
    )
    print(counts)
    print(OUT_GRID)


if __name__ == "__main__":
    write_outputs()
