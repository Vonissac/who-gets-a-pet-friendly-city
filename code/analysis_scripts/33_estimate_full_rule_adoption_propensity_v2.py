#!/usr/bin/env python3
"""Estimate point-level and grid-level adoption propensity for full POI agents.

This v2 model replaces the 150-record sample universe with city-scale 2025 POI
silent agents. It keeps the same publication boundary: a high score is a
collection/model signal, not evidence that the venue is actually pet-friendly.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.special import expit


PROJECT_ROOT = Path(__file__).resolve()
for _p in PROJECT_ROOT.parents:
    if (_p / "data").is_dir():
        PROJECT_ROOT = _p
        break
(PROJECT_ROOT / "_rebuild_outputs" / "reports").mkdir(parents=True, exist_ok=True)
RULES = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_semantic_records_v1.csv"
LEDGER = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_source_ledger_v1.csv"
SILENT = PROJECT_ROOT / "data" / "derived_data" / "platform" / "full_silent_rule_venues_2025_v1.csv"
GRID = PROJECT_ROOT / "data" / "derived_data" / "geo" / "shenzhen_grid_500m.gpkg"
OUT = PROJECT_ROOT / "data" / "derived_data" / "platform" / "full_silent_rule_adoption_propensity_2025_v2.csv"
OUT_GRID = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_silent_rule_adoption_surface_500m_2025_v2.csv"
OUT_GRID_GEO = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_silent_rule_adoption_surface_500m_2025_v2.geojson"
REPORT = PROJECT_ROOT / "_rebuild_outputs" / "reports" / "48_full_rule_adoption_propensity_v2_report.md"

TYPE_BASELINE = {
    "shopping_mall": -1.00,
    "restaurant": -1.25,
    "hotel": -1.55,
    "park_or_recreation": -0.95,
    "residential_property": -1.80,
}

RULE_TYPE_MAP = {
    "shopping_mall": {
        "shopping_mall",
        "shopping_mall_group",
        "shopping_mall_network",
        "shopping_mall_sample",
        "shopping_mall_standard",
        "shopping_mall_pet_park",
    },
    "restaurant": {"restaurant", "restaurant_network_metric"},
    "hotel": {"hotel"},
    "park_or_recreation": {"pet_friendly_park", "pet_governance_education_base", "public_pet_park_code"},
    "residential_property": {"residential_community"},
}


def primary_rule_type(venue_type: str) -> str:
    venue_type = str(venue_type)
    if "shopping_mall" in venue_type:
        return "shopping_mall"
    if "restaurant" in venue_type or "pet_cafe" in venue_type:
        return "restaurant"
    if "hotel" in venue_type:
        return "hotel"
    if "park" in venue_type or "pet_governance" in venue_type:
        return "park_or_recreation"
    if "residential" in venue_type:
        return "residential_property"
    return "unknown"


def compatibility(rule_type: str, silent_type: str) -> float:
    if rule_type in RULE_TYPE_MAP.get(silent_type, set()):
        return 1.0
    if silent_type == "restaurant" and "shopping_mall" in rule_type:
        return 0.35
    if silent_type == "shopping_mall" and rule_type == "restaurant":
        return 0.25
    if silent_type == "hotel" and rule_type in {"restaurant", "shopping_mall"}:
        return 0.20
    if silent_type == "residential_property" and rule_type == "park_or_recreation":
        return 0.30
    if silent_type == "park_or_recreation" and rule_type == "residential_property":
        return 0.20
    return 0.10


def load_rules() -> pd.DataFrame:
    rules = pd.read_csv(RULES)
    if LEDGER.exists():
        ledger = pd.read_csv(LEDGER)[["source_id", "publication_use_status", "source_weight_for_model"]]
        rules = rules.merge(ledger, on="source_id", how="left")
        rules = rules[
            rules["publication_use_status"].isin(
                ["main_model", "main_model_with_caution", "supplement_or_sensitivity"]
            )
        ].copy()
    rules = rules[rules["geo_level"].isin(["venue", "district", "citywide"])].copy()
    rules = rules[np.isfinite(rules["lon"]) & np.isfinite(rules["lat"])].copy()
    rules["source_weight_for_model"] = pd.to_numeric(rules["source_weight_for_model"], errors="coerce").fillna(0.40)
    rules["rule_primary_type"] = rules["venue_type"].map(primary_rule_type)
    return rules


def transform_xy(lon: pd.Series, lat: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    x, y = transformer.transform(lon.to_numpy(dtype=float), lat.to_numpy(dtype=float))
    return np.asarray(x), np.asarray(y)


def score_chunk(chunk: pd.DataFrame, rules: pd.DataFrame, rule_x: np.ndarray, rule_y: np.ndarray) -> pd.DataFrame:
    sx, sy = transform_xy(chunk["lon_wgs84"], chunk["lat_wgs84"])
    stype = chunk["primary_venue_type"].astype(str).to_numpy()
    baseline = np.array([TYPE_BASELINE.get(t, -1.20) for t in stype], dtype=float)
    same_100 = np.zeros(len(chunk), dtype=float)
    same_250 = np.zeros(len(chunk), dtype=float)
    same_500 = np.zeros(len(chunk), dtype=float)
    cross_250 = np.zeros(len(chunk), dtype=float)
    positive_500 = np.zeros(len(chunk), dtype=float)
    restrictive_500 = np.zeros(len(chunk), dtype=float)
    nearest_positive = np.full(len(chunk), np.nan, dtype=float)

    for ridx, rrow in rules.reset_index(drop=True).iterrows():
        dx = sx - rule_x[ridx]
        dy = sy - rule_y[ridx]
        dist = np.sqrt(dx * dx + dy * dy)
        within_500 = dist <= 500.0
        if not within_500.any():
            continue
        acc = float(rrow["access_semantic_score"])
        ew = float(rrow["source_weight_for_model"])
        rtype = rrow["rule_primary_type"]
        comp = np.array([compatibility(rtype, t) for t in stype], dtype=float)
        dist_w_100 = np.where(dist <= 100.0, 1.0 / (1.0 + (dist / 100.0) ** 2), 0.0)
        dist_w_250 = np.where(dist <= 250.0, 1.0 / (1.0 + (dist / 100.0) ** 2), 0.0)
        dist_w_500 = np.where(within_500, 1.0 / (1.0 + (dist / 100.0) ** 2), 0.0)
        same = comp >= 0.9
        if acc > 0:
            val = acc * ew * comp
            pos = dist_w_500 * val
            positive_500 += pos
            same_100 += np.where(same, dist_w_100 * val, 0.0)
            same_250 += np.where(same, dist_w_250 * val, 0.0)
            same_500 += np.where(same, dist_w_500 * val, 0.0)
            cross_250 += np.where(~same, dist_w_250 * val, 0.0)
            nearest_positive = np.where(
                within_500 & (np.isnan(nearest_positive) | (dist < nearest_positive)),
                dist,
                nearest_positive,
            )
        elif acc < 0:
            restrictive_500 += np.where(within_500, abs(acc) * ew * (1.0 / (1.0 + (dist / 150.0) ** 2)), 0.0)

    logit = (
        baseline
        + 1.80 * same_100
        + 1.10 * same_250
        + 0.55 * same_500
        + 0.45 * cross_250
        + 0.20 * positive_500
        - 0.85 * restrictive_500
    )
    prob = expit(logit)
    classes = np.where(
        (prob >= 0.65) & (positive_500 > 0.03),
        "high_latent_adoption_propensity",
        np.where(
            (prob >= 0.45) & (positive_500 > 0.03),
            "medium_latent_adoption_propensity",
            np.where(
                (prob >= 0.30) & (positive_500 > 0.03),
                "low_but_plausible_adoption_propensity",
                "low_or_unknown_adoption_propensity",
            ),
        ),
    )

    out = chunk[
        [
            "id",
            "source_file",
            "source_row_number",
            "name",
            "observed_poi_name",
            "primary_venue_type",
            "potential_venue_type",
            "classification_confidence",
            "address",
            "adname",
            "lon_wgs84",
            "lat_wgs84",
            "grid_id",
            "district_name",
            "collection_priority",
        ]
    ].copy()
    out["baseline_logit"] = baseline
    out["same_type_exposure_100m"] = same_100
    out["same_type_exposure_250m"] = same_250
    out["same_type_exposure_500m"] = same_500
    out["cross_type_exposure_250m"] = cross_250
    out["positive_rule_exposure_500m"] = positive_500
    out["restrictive_rule_exposure_500m"] = restrictive_500
    out["nearest_positive_rule_m"] = nearest_positive
    out["adoption_propensity"] = prob
    out["adoption_propensity_class"] = classes
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    rules = load_rules()
    rule_x, rule_y = transform_xy(rules["lon"], rules["lat"])

    first = True
    chunks = 0
    total = 0
    class_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    for chunk in pd.read_csv(SILENT, chunksize=25000, low_memory=False):
        scored = score_chunk(chunk, rules, rule_x, rule_y)
        scored.to_csv(OUT, index=False, mode="w" if first else "a", header=first, encoding="utf-8-sig")
        first = False
        chunks += 1
        total += len(scored)
        class_counter.update(scored["adoption_propensity_class"])
        type_counter.update(scored["primary_venue_type"])
        print(f"scored chunk {chunks}: rows={len(scored)} total={total}")

    out = pd.read_csv(OUT)
    grid = (
        out.groupby("grid_id", dropna=False)
        .agg(
            silent_agents=("id", "count"),
            mean_adoption_propensity=("adoption_propensity", "mean"),
            p90_adoption_propensity=("adoption_propensity", lambda s: float(s.quantile(0.9))),
            high_propensity_agents=("adoption_propensity_class", lambda s: int((s == "high_latent_adoption_propensity").sum())),
            medium_or_high_propensity_agents=(
                "adoption_propensity_class",
                lambda s: int(s.isin(["medium_latent_adoption_propensity", "high_latent_adoption_propensity"]).sum()),
            ),
            positive_rule_exposure_500m_mean=("positive_rule_exposure_500m", "mean"),
            restrictive_rule_exposure_500m_mean=("restrictive_rule_exposure_500m", "mean"),
        )
        .reset_index()
    )
    grid["medium_or_high_share"] = grid["medium_or_high_propensity_agents"] / grid["silent_agents"]
    grid["high_share"] = grid["high_propensity_agents"] / grid["silent_agents"]
    grid.to_csv(OUT_GRID, index=False, encoding="utf-8-sig")

    grid_geo = gpd.read_file(GRID)
    grid_joined = grid_geo.merge(grid, on="grid_id", how="left")
    for col in [
        "silent_agents",
        "mean_adoption_propensity",
        "p90_adoption_propensity",
        "high_propensity_agents",
        "medium_or_high_propensity_agents",
        "positive_rule_exposure_500m_mean",
        "restrictive_rule_exposure_500m_mean",
        "medium_or_high_share",
        "high_share",
    ]:
        grid_joined[col] = grid_joined[col].fillna(0)
    grid_joined.to_crs("EPSG:4326").to_file(OUT_GRID_GEO, driver="GeoJSON")

    counts = {
        "silent_agents_scored": int(len(out)),
        "rules_used": int(len(rules)),
        "positive_rules": int((rules["access_semantic_score"] > 0).sum()),
        "restrictive_rules": int((rules["access_semantic_score"] < 0).sum()),
        "agent_type_counts": dict(type_counter),
        "propensity_class_counts": dict(class_counter),
        "mean_probability": float(out["adoption_propensity"].mean()),
        "p90_probability": float(out["adoption_propensity"].quantile(0.9)),
        "max_probability": float(out["adoption_propensity"].max()),
        "grid_rows_with_agents": int((grid["silent_agents"] > 0).sum()),
        "grid_rows_with_medium_or_high_agents": int((grid["medium_or_high_propensity_agents"] > 0).sum()),
    }
    top_grids = (
        grid.sort_values(["medium_or_high_propensity_agents", "mean_adoption_propensity"], ascending=False)
        .head(20)
        .to_dict("records")
    )
    REPORT.write_text(
        f"""# 48 Full Rule Adoption Propensity V2 Report

Locked title: Pet-Service Ecologies Outpace Urban Readiness in Shenzhen: Rule-Liminal Venues and the Uneven Emergence of Companion-Animal Urban Capability

## Execution

- Silent-agent input: `{SILENT.relative_to(PROJECT_ROOT)}`
- Point-level output: `{OUT.relative_to(PROJECT_ROOT)}`
- Grid-level output: `{OUT_GRID.relative_to(PROJECT_ROOT)}`
- Grid GeoJSON: `{OUT_GRID_GEO.relative_to(PROJECT_ROOT)}`

## Counts

`{counts}`

## Top Candidate Grids

`{top_grids}`

## Model Meaning

This is a rule-adoption propensity model. It estimates whether a silent venue is locally exposed to explicit pet-friendly or restrictive rule signals from nearby same-type and cross-type venues. It does not label silent venues as pet-friendly.

## Why This Moves Beyond Kernel Decay

The score is venue-agent based. It combines source-weighted rule evidence, semantic access direction, same-type peer exposure, cross-type spillover, restrictive counter-signals, and type-specific baselines. The 100m term addresses the study question of whether a nearby pet-friendly restaurant or mall plausibly changes the probability that an adjacent semi-public venue adopts or publicizes a pet-friendly rule.

## Publication Boundary

Use point-level scores for hypothesis generation and targeted verification. Use grid-level aggregates for spatial diagnosis. Manuscript claims must describe "latent rule-adoption propensity" and "rule-environment exposure", not observed pet friendliness unless confirmed by primary sources.
""",
        encoding="utf-8",
    )
    print(counts)
    print(OUT)


if __name__ == "__main__":
    main()
