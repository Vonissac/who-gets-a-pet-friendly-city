#!/usr/bin/env python3
"""Build a rule-liminal threshold model for Shenzhen host venues.

The v4 layer treats conditional, silent, unpublicized, and operator-discretion
venues as the central empirical object. A high score is not a pet-friendly
claim. It is a prioritized signal that a venue is near the threshold where
pet-service ecology, rule exposure, and network position may make explicit
pet-access rules emerge.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.special import expit


PROJECT_ROOT = Path(__file__).resolve()
for _p in PROJECT_ROOT.parents:
    if (_p / "data").is_dir():
        PROJECT_ROOT = _p
        break
(PROJECT_ROOT / "_rebuild_outputs" / "reports").mkdir(parents=True, exist_ok=True)
RULES = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_semantic_records_v1.csv"
LEDGER = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_source_ledger_v1.csv"
HOST_V3 = PROJECT_ROOT / "data" / "derived_data" / "platform" / "host_venue_pet_rule_ecology_2025_v3.csv"
GRID_V3 = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_pet_rule_ecology_500m_2025_v3.csv"
TOPO = PROJECT_ROOT / "data" / "derived_data" / "network" / "grid_pet_rule_topology_2025_v3.csv"
GRID = PROJECT_ROOT / "data" / "derived_data" / "geo" / "shenzhen_grid_500m.gpkg"

OUT_HOST = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_liminal_host_candidates_2025_v4.csv"
OUT_GRID = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_rule_liminality_500m_2025_v4.csv"
OUT_GRID_GEO = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_rule_liminality_500m_2025_v4.geojson"
OUT_EDGES = PROJECT_ROOT / "data" / "derived_data" / "network" / "rule_liminal_threshold_edges_2025_v4.csv"
REPORT = PROJECT_ROOT / "_rebuild_outputs" / "reports" / "54_rule_liminal_threshold_model_v4_report.md"
METHOD = PROJECT_ROOT / "_rebuild_outputs" / "reports" / "RULE_LIMINAL_THRESHOLD_MODEL_v4.md"

TITLE = "Pet-Service Ecologies Outpace Urban Readiness in Shenzhen: Rule-Liminal Venues and the Uneven Emergence of Companion-Animal Urban Capability"
CHINESE_TITLE = "宠物城市生态已到，城市尚未准备好：深圳伴侣动物空间生态的规则临界与共处能力涌现机制"
PROJECTED_CRS = "EPSG:32649"

TYPE_BASELINE = {
    "shopping_mall": -0.15,
    "restaurant": -0.30,
    "hotel": -0.55,
    "park_or_recreation": -0.10,
    "residential_property": -0.70,
}


def transform_xy(lon: pd.Series, lat: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    x, y = transformer.transform(lon.to_numpy(dtype=float), lat.to_numpy(dtype=float))
    return np.asarray(x), np.asarray(y)


def decay(distance: np.ndarray, scale: float) -> np.ndarray:
    return 1.0 / (1.0 + (distance / scale) ** 2)


def load_rules() -> pd.DataFrame:
    rules = pd.read_csv(RULES)
    if LEDGER.exists():
        ledger = pd.read_csv(LEDGER)[
            ["source_id", "publication_use_status", "source_weight_for_model", "source_class"]
        ]
        rules = rules.merge(ledger, on="source_id", how="left")
    rules = rules[np.isfinite(rules["lon"]) & np.isfinite(rules["lat"])].copy()
    rules["source_weight_for_model"] = pd.to_numeric(rules["source_weight_for_model"], errors="coerce").fillna(0.35)
    rules["publication_use_status"] = rules["publication_use_status"].fillna("hold_until_classified")
    rules["source_class"] = rules["source_class"].fillna("Z_unclassified")
    rules["is_model_eligible"] = rules["publication_use_status"].isin(
        ["main_model", "main_model_with_caution", "supplement_or_sensitivity"]
    )
    ap = rules["access_position"].astype(str)
    score = rules["access_semantic_score"].astype(float)
    evidence_caution = rules["publication_use_status"].isin(["supplement_or_sensitivity", "main_model_with_caution"])
    rules["rule_liminal_state"] = np.select(
        [
            score <= -0.40,
            (score >= 0.70) & ap.str.startswith("allowed_"),
            ap.str.contains("operator_discretion|design_guidance|district_pet|policy_change|absent", regex=True),
            (score > -0.10) & (score < 0.70),
            evidence_caution & (score > 0),
        ],
        [
            "restrictive_rule",
            "explicit_open_rule",
            "governance_or_operator_liminal_rule",
            "conditional_liminal_rule",
            "cautious_liminal_rule",
        ],
        default="other_rule_signal",
    )
    rules["is_liminal_rule"] = rules["rule_liminal_state"].isin(
        ["governance_or_operator_liminal_rule", "conditional_liminal_rule", "cautious_liminal_rule"]
    )
    return rules.reset_index(drop=True)


def rule_exposure_by_state(
    host: pd.DataFrame, rules: pd.DataFrame, radius: float = 800.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = rules.reset_index(drop=True).copy()
    hx, hy = transform_xy(host["lon_wgs84"], host["lat_wgs84"])
    rx, ry = transform_xy(rules["lon"], rules["lat"])
    tree = cKDTree(np.column_stack([rx, ry]))
    explicit = np.zeros(len(host), dtype=float)
    liminal = np.zeros(len(host), dtype=float)
    restrictive = np.zeros(len(host), dtype=float)
    nearest_liminal = np.full(len(host), np.nan, dtype=float)
    nearest_explicit = np.full(len(host), np.nan, dtype=float)
    nearest_restrictive = np.full(len(host), np.nan, dtype=float)
    edge_rows: list[dict] = []
    rule_state = rules["rule_liminal_state"].to_numpy()
    access = rules["access_semantic_score"].to_numpy(dtype=float)
    src_weight = rules["source_weight_for_model"].to_numpy(dtype=float)

    sampled_host = set(host.index[host["v3_propensity_class"].isin(["medium_pet_rule_propensity", "high_pet_rule_propensity"])])
    sampled_host.update(host.nlargest(12000, "pet_service_exposure_500m").index.tolist())

    for idx, xy in enumerate(np.column_stack([hx, hy])):
        neighbors = tree.query_ball_point(xy, r=radius)
        if not neighbors:
            continue
        n = np.asarray(neighbors, dtype=int)
        dist = np.sqrt((rx[n] - xy[0]) ** 2 + (ry[n] - xy[1]) ** 2)
        w = decay(dist, scale=220.0) * src_weight[n]
        exp_mask = rule_state[n] == "explicit_open_rule"
        lim_mask = np.isin(rule_state[n], ["governance_or_operator_liminal_rule", "conditional_liminal_rule", "cautious_liminal_rule"])
        neg_mask = rule_state[n] == "restrictive_rule"
        if exp_mask.any():
            explicit[idx] = float((w[exp_mask] * np.maximum(access[n][exp_mask], 0)).sum())
            nearest_explicit[idx] = float(dist[exp_mask].min())
        if lim_mask.any():
            liminal[idx] = float((w[lim_mask] * np.maximum(access[n][lim_mask], 0.15)).sum())
            nearest_liminal[idx] = float(dist[lim_mask].min())
        if neg_mask.any():
            restrictive[idx] = float((w[neg_mask] * np.abs(access[n][neg_mask])).sum())
            nearest_restrictive[idx] = float(dist[neg_mask].min())
        if idx in sampled_host and (lim_mask.any() or exp_mask.any() or neg_mask.any()):
            for ridx, d, ww in zip(n, dist, w):
                state = rule_state[ridx]
                if state not in {"explicit_open_rule", "restrictive_rule", "governance_or_operator_liminal_rule", "conditional_liminal_rule", "cautious_liminal_rule"}:
                    continue
                edge_rows.append(
                    {
                        "host_key": host.at[idx, "host_key"],
                        "host_name": host.at[idx, "host_name"],
                        "host_grid_id": host.at[idx, "grid_id"],
                        "rule_source_id": rules.at[ridx, "source_id"],
                        "rule_venue_name": rules.at[ridx, "venue_name"],
                        "rule_state": state,
                        "distance_m": float(d),
                        "weight": float(ww),
                    }
                )

    exposure = pd.DataFrame(
        {
            "liminal_rule_exposure_800m": liminal,
            "explicit_open_rule_exposure_800m": explicit,
            "restrictive_rule_exposure_800m": restrictive,
            "nearest_liminal_rule_m": nearest_liminal,
            "nearest_explicit_open_rule_m": nearest_explicit,
            "nearest_restrictive_rule_m": nearest_restrictive,
        }
    )
    return exposure, pd.DataFrame(edge_rows)


def classify_liminal(row: pd.Series) -> str:
    if row["restrictive_rule_exposure_800m"] > 0.25 and row["liminal_potential_score"] < 0.45:
        return "restricted_or_negative_pressure"
    if row["liminal_potential_score"] >= 0.70 and row["liminal_rule_exposure_800m"] > 0:
        return "rule_liminal_threshold_core"
    if row["liminal_potential_score"] >= 0.55 and row["pet_service_exposure_500m"] > 0:
        return "ecology_ready_rule_liminal_frontier"
    if row["pet_service_exposure_500m"] > 0 and row["liminal_rule_exposure_800m"] == 0 and row["explicit_open_rule_exposure_800m"] == 0:
        return "ecology_present_rule_silent"
    if row["liminal_rule_exposure_800m"] > 0 or row["explicit_open_rule_exposure_800m"] > 0:
        return "rule_exposed_low_ecology"
    return "low_signal_unknown"


def build_host_liminal() -> tuple[pd.DataFrame, pd.DataFrame]:
    host = pd.read_csv(HOST_V3)
    rules = load_rules()
    topo = pd.read_csv(TOPO)[["grid_id", "pet_rule_pagerank", "topology_ecology_class"]]
    host = host.merge(topo, on="grid_id", how="left")
    exposure, edges = rule_exposure_by_state(host, rules[rules["is_model_eligible"]].copy())
    out = pd.concat([host.reset_index(drop=True), exposure], axis=1)

    pagerank = out["pet_rule_pagerank"].fillna(0)
    if pagerank.max() > pagerank.min():
        topo_z = (pagerank - pagerank.min()) / (pagerank.max() - pagerank.min())
    else:
        topo_z = pagerank
    service = np.log1p(out["pet_service_exposure_500m"].fillna(0))
    diversity = np.log1p(
        (
            out[
                [
                    "a_medical_exposure_500m",
                    "a_retail_exposure_500m",
                    "a_grooming_exposure_500m",
                    "a_boarding_exposure_500m",
                    "a_training_exposure_500m",
                ]
            ].fillna(0)
            > 0
        ).sum(axis=1)
    )
    baseline = out["primary_venue_type"].map(TYPE_BASELINE).fillna(-0.35)
    prior = pd.to_numeric(out["pet_rule_threshold_propensity_v3"], errors="coerce").fillna(0.0)
    prior_logit = np.log(np.clip(prior, 0.001, 0.999) / np.clip(1 - prior, 0.001, 0.999))
    logit = (
        baseline
        + 0.30 * prior_logit
        + 0.55 * service
        + 0.24 * diversity
        + 1.10 * out["liminal_rule_exposure_800m"].fillna(0)
        + 0.55 * out["explicit_open_rule_exposure_800m"].fillna(0)
        + 0.65 * topo_z
        - 0.90 * out["restrictive_rule_exposure_800m"].fillna(0)
    )
    out["liminal_potential_score"] = expit(logit)
    out["liminal_threshold_shift"] = (
        0.55 * service
        + 0.24 * diversity
        + 1.10 * out["liminal_rule_exposure_800m"].fillna(0)
        + 0.65 * topo_z
    )
    out["liminal_class"] = out.apply(classify_liminal, axis=1)
    out["publication_status"] = np.select(
        [
            out["liminal_class"].isin(["rule_liminal_threshold_core", "ecology_ready_rule_liminal_frontier"]),
            out["liminal_class"].eq("ecology_present_rule_silent"),
        ],
        [
            "priority_verification_candidate_not_rule_claim",
            "latent_ecology_candidate_not_rule_claim",
        ],
        default="context_or_low_priority",
    )
    out["interpretation_boundary_v4"] = (
        "Liminal scores estimate threshold proximity under pet-service ecology, nearby rule signals, and topology. "
        "They do not confirm that the host currently permits pets."
    )
    return out, edges


def build_grid_liminal(host: pd.DataFrame) -> pd.DataFrame:
    grid_v3 = pd.read_csv(GRID_V3)
    grouped = host.groupby("grid_id").agg(
        host_candidates=("host_key", "count"),
        mean_liminal_potential=("liminal_potential_score", "mean"),
        p90_liminal_potential=("liminal_potential_score", lambda s: float(s.quantile(0.9))),
        liminal_threshold_core_hosts=("liminal_class", lambda s: int((s == "rule_liminal_threshold_core").sum())),
        liminal_frontier_hosts=("liminal_class", lambda s: int((s == "ecology_ready_rule_liminal_frontier").sum())),
        ecology_silent_hosts=("liminal_class", lambda s: int((s == "ecology_present_rule_silent").sum())),
        restricted_hosts=("liminal_class", lambda s: int((s == "restricted_or_negative_pressure").sum())),
        mean_liminal_rule_exposure=("liminal_rule_exposure_800m", "mean"),
        mean_explicit_open_rule_exposure=("explicit_open_rule_exposure_800m", "mean"),
    ).reset_index()
    out = grid_v3.merge(grouped, on="grid_id", how="left")
    fill_cols = [c for c in grouped.columns if c != "grid_id"]
    out[fill_cols] = out[fill_cols].fillna(0)
    out["liminal_or_frontier_hosts"] = out["liminal_threshold_core_hosts"] + out["liminal_frontier_hosts"]
    out["liminal_frontier_share"] = np.where(
        out["host_candidates"] > 0,
        out["liminal_or_frontier_hosts"] / out["host_candidates"],
        0,
    )
    out["grid_liminal_type"] = np.select(
        [
            (out["liminal_threshold_core_hosts"] > 0) & (out["pet_rule_ecology_class"] == "high_pet_ecology_with_positive_rule"),
            (out["liminal_or_frontier_hosts"] > 0) & (out["pet_rule_ecology_class"] == "high_pet_ecology_rule_silent"),
            (out["ecology_silent_hosts"] > 0) & (out["pet_service_count_2025"] > 0),
            (out["restricted_hosts"] > 0),
        ],
        [
            "confirmed_ecology_rule_liminal_core",
            "high_ecology_liminal_frontier",
            "latent_ecology_rule_silent_zone",
            "restriction_pressure_zone",
        ],
        default="low_or_diffuse_liminal_signal",
    )
    return out


def build_grid_geo(grid_liminal: pd.DataFrame) -> None:
    grid_geo = gpd.read_file(GRID)
    out = grid_geo.merge(grid_liminal, on=["grid_id", "district_name", "edge_cell", "intersect_area_m2"], how="left")
    out.to_crs("EPSG:4326").to_file(OUT_GRID_GEO, driver="GeoJSON")


def build_rule_state_network(host: pd.DataFrame, edges: pd.DataFrame) -> dict[str, float]:
    if edges.empty:
        return {}
    g = nx.Graph()
    host_keep = host[
        host["liminal_class"].isin(
            ["rule_liminal_threshold_core", "ecology_ready_rule_liminal_frontier", "ecology_present_rule_silent"]
        )
    ][["host_key", "liminal_potential_score"]]
    for row in host_keep.itertuples():
        g.add_node(f"host:{row.host_key}", node_type="host", score=float(row.liminal_potential_score))
    for row in edges.itertuples():
        source = f"host:{row.host_key}"
        target = f"rule:{row.rule_source_id}"
        if source in g:
            g.add_node(target, node_type="rule", rule_state=row.rule_state)
            g.add_edge(source, target, weight=float(row.weight))
    if g.number_of_edges() == 0:
        return {}
    comp_sizes = [len(c) for c in nx.connected_components(g)]
    return {
        "nodes": float(g.number_of_nodes()),
        "edges": float(g.number_of_edges()),
        "components": float(nx.number_connected_components(g)),
        "largest_component_nodes": float(max(comp_sizes) if comp_sizes else 0),
        "mean_degree": float(np.mean([d for _, d in g.degree()])),
    }


def write_outputs() -> None:
    OUT_HOST.parent.mkdir(parents=True, exist_ok=True)
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    OUT_EDGES.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    host, edges = build_host_liminal()
    grid = build_grid_liminal(host)
    host.to_csv(OUT_HOST, index=False, encoding="utf-8-sig")
    grid.to_csv(OUT_GRID, index=False, encoding="utf-8-sig")
    edges.to_csv(OUT_EDGES, index=False, encoding="utf-8-sig")
    build_grid_geo(grid)
    network_stats = build_rule_state_network(host, edges)
    counts = {
        "host_rows": int(len(host)),
        "liminal_class": host["liminal_class"].value_counts().to_dict(),
        "grid_rows": int(len(grid)),
        "grid_liminal_type": grid["grid_liminal_type"].value_counts().to_dict(),
        "threshold_edges": int(len(edges)),
        "network_stats": network_stats,
        "mean_liminal_potential": float(host["liminal_potential_score"].mean()),
        "p90_liminal_potential": float(host["liminal_potential_score"].quantile(0.9)),
    }
    top_hosts = host.sort_values("liminal_potential_score", ascending=False).head(40)[
        [
            "host_name",
            "primary_venue_type",
            "district_name",
            "grid_id",
            "liminal_potential_score",
            "liminal_class",
            "pet_service_exposure_500m",
            "liminal_rule_exposure_800m",
            "explicit_open_rule_exposure_800m",
            "restrictive_rule_exposure_800m",
            "publication_status",
        ]
    ].to_dict("records")
    top_grids = grid.sort_values(
        ["liminal_or_frontier_hosts", "p90_liminal_potential"], ascending=False
    ).head(30)[
        [
            "grid_id",
            "district_name",
            "grid_liminal_type",
            "host_candidates",
            "liminal_or_frontier_hosts",
            "ecology_silent_hosts",
            "p90_liminal_potential",
            "pet_rule_ecology_class",
        ]
    ].to_dict("records")
    REPORT.write_text(
        f"""# 54 Rule-Liminal Threshold Model V4 Report

Locked title: {TITLE}

Chinese locked title: {CHINESE_TITLE}

## Outputs

- Host-level liminal candidates: `{OUT_HOST.relative_to(PROJECT_ROOT)}`
- Grid-level liminality table: `{OUT_GRID.relative_to(PROJECT_ROOT)}`
- Grid-level liminality GeoJSON: `{OUT_GRID_GEO.relative_to(PROJECT_ROOT)}`
- Host-rule threshold edges: `{OUT_EDGES.relative_to(PROJECT_ROOT)}`

## Counts

`{counts}`

## Top Rule-Liminal Host Candidates

`{top_hosts}`

## Top Grid Frontiers

`{top_grids}`

## Interpretation

V4 separates three things that were previously blurred: explicit pet-friendly rules, conditional or operator-discretion rules, and silent venues under strong pet-service ecology. This makes the ambiguous middle the paper's empirical center rather than residual noise.

The strongest mechanism is threshold proximity: a venue can be rule-liminal when pet-service exposure, nearby conditional rules, and network position jointly make explicit access rules plausible, even if no current primary source confirms pet access.

## Publication Boundary

`rule_liminal_threshold_core`, `ecology_ready_rule_liminal_frontier`, and `ecology_present_rule_silent` are not observed pet-friendly labels. They are model-prioritized verification classes. Manuscript claims should use them to discuss latent capability and threshold pressure, not confirmed access.
""",
        encoding="utf-8",
    )
    METHOD.write_text(
        f"""# Rule-Liminal Threshold Model V4

Locked title: **{TITLE}**

Chinese locked title: **{CHINESE_TITLE}**

## Purpose

The v4 model is designed around the project's central pivot: the most important venues are not only those with explicit pet-friendly rules, but those that sit between silence, conditional openness, operator discretion, and emerging ecological pressure.

## Concept

Rule-liminal venues are semi-public or private hosts whose pet-access status is not fully stabilized. They may be silent, conditional, event-based, operator-dependent, mall-dependent, or spatially zoned.

## Model Logic

The model estimates threshold proximity using:

- v3 pet-service ecology and host threshold propensity;
- distance-weighted exposure to explicit open rules;
- distance-weighted exposure to conditional, operator-discretion, district-target, or cautious liminal rules;
- restrictive rule pressure;
- pet-service diversity;
- grid topology centrality.

## Output Classes

| Class | Meaning |
|---|---|
| `rule_liminal_threshold_core` | High liminal potential with nearby liminal rule evidence |
| `ecology_ready_rule_liminal_frontier` | High pet ecology and threshold pressure, likely verification priority |
| `ecology_present_rule_silent` | Pet-service ecology exists but no nearby explicit or liminal rule seed |
| `rule_exposed_low_ecology` | Rule signal exists without strong local pet ecology |
| `restricted_or_negative_pressure` | Restrictive pressure dominates |
| `low_signal_unknown` | Insufficient evidence |

## Claim Boundary

V4 is a mechanism and verification model. It should not be used to say a venue permits pets unless a primary source confirms that rule.
""",
        encoding="utf-8",
    )
    print(counts)
    print(OUT_HOST)


if __name__ == "__main__":
    write_outputs()
