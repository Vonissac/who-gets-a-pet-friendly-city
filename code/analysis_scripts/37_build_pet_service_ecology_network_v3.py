#!/usr/bin/env python3
"""Build v3 pet-service ecology and multilayer topology signals.

This script upgrades pet-service POI from a background count to an ecological
pressure layer. It links core pet-service points, explicit rule seeds, host
venue candidates and 500m grid adjacency into publishable diagnostic tables.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point


PROJECT_ROOT = Path(__file__).resolve()
for _p in PROJECT_ROOT.parents:
    if (_p / "data").is_dir():
        PROJECT_ROOT = _p
        break
(PROJECT_ROOT / "_rebuild_outputs" / "reports").mkdir(parents=True, exist_ok=True)
PET = PROJECT_ROOT / "data" / "derived_data" / "pet_service_core_A_deduplicated.csv"
HOST = PROJECT_ROOT / "data" / "derived_data" / "platform" / "host_venue_rule_agents_2025_v2.csv"
RULES = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_semantic_records_v1.csv"
LEDGER = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_source_ledger_v1.csv"
GRID = PROJECT_ROOT / "data" / "derived_data" / "geo" / "shenzhen_grid_500m.gpkg"

OUT_PET_POINTS = PROJECT_ROOT / "data" / "derived_data" / "platform" / "pet_service_ecology_points_2025_v3.csv"
OUT_PET_POINTS_GEO = PROJECT_ROOT / "data" / "derived_data" / "platform" / "pet_service_ecology_points_2025_v3.geojson"
OUT_GRID = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_pet_rule_ecology_500m_2025_v3.csv"
OUT_GRID_GEO = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_pet_rule_ecology_500m_2025_v3.geojson"
OUT_HOST = PROJECT_ROOT / "data" / "derived_data" / "platform" / "host_venue_pet_rule_ecology_2025_v3.csv"
OUT_EDGES = PROJECT_ROOT / "data" / "derived_data" / "network" / "pet_rule_multilayer_edges_2025_v3.csv"
OUT_GRID_TOPO = PROJECT_ROOT / "data" / "derived_data" / "network" / "grid_pet_rule_topology_2025_v3.csv"
REPORT = PROJECT_ROOT / "_rebuild_outputs" / "reports" / "52_pet_service_ecology_network_v3_report.md"
METHOD = PROJECT_ROOT / "PET_SERVICE_ECOLOGY_NETWORK_MODEL_v3.md"

PROJECTED_CRS = "EPSG:32649"

PET_CLASS_WEIGHT = {
    "A_MEDICAL": 1.15,
    "A_RETAIL": 1.00,
    "A_GROOMING": 0.95,
    "A_BOARDING": 1.10,
    "A_TRAINING": 1.05,
}

HOST_BASELINE = {
    "shopping_mall": -0.95,
    "restaurant": -1.25,
    "hotel": -1.50,
    "park_or_recreation": -0.95,
    "residential_property": -1.85,
}


def class_flags(classes: str) -> dict[str, int]:
    parts = set(str(classes).split("|"))
    return {cls: int(cls in parts) for cls in PET_CLASS_WEIGHT}


def pet_weight(classes: str) -> float:
    flags = class_flags(classes)
    if not any(flags.values()):
        return 0.5
    return max(PET_CLASS_WEIGHT[k] for k, v in flags.items() if v)


def transform_xy(lon: pd.Series, lat: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    x, y = transformer.transform(lon.to_numpy(dtype=float), lat.to_numpy(dtype=float))
    return np.asarray(x), np.asarray(y)


def distance_decay(distance: np.ndarray, scale: float = 180.0) -> np.ndarray:
    return 1.0 / (1.0 + (distance / scale) ** 2)


def service_exposure(
    target_x: np.ndarray,
    target_y: np.ndarray,
    pet: pd.DataFrame,
    pet_x: np.ndarray,
    pet_y: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    tree = cKDTree(np.column_stack([pet_x, pet_y]))
    exposure = np.zeros(len(target_x), dtype=float)
    nearest = np.full(len(target_x), np.nan, dtype=float)
    class_exp = {cls: np.zeros(len(target_x), dtype=float) for cls in PET_CLASS_WEIGHT}
    weights = pet["pet_service_weight"].to_numpy(dtype=float)
    pet_classes = pet["candidate_classes"].astype(str).to_numpy()
    for idx, xy in enumerate(np.column_stack([target_x, target_y])):
        neighbors = tree.query_ball_point(xy, r=radius)
        if not neighbors:
            continue
        n = np.asarray(neighbors, dtype=int)
        dist = np.sqrt((pet_x[n] - xy[0]) ** 2 + (pet_y[n] - xy[1]) ** 2)
        nearest[idx] = float(dist.min())
        val = distance_decay(dist) * weights[n]
        exposure[idx] = float(val.sum())
        for cls in PET_CLASS_WEIGHT:
            mask = np.array([cls in pet_classes[j].split("|") for j in n])
            if mask.any():
                class_exp[cls][idx] = float(val[mask].sum())
    return exposure, class_exp, nearest


def rule_primary_type(venue_type: str) -> str:
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
    rules = rules[np.isfinite(rules["lon"]) & np.isfinite(rules["lat"])].copy()
    rules["source_weight_for_model"] = pd.to_numeric(rules["source_weight_for_model"], errors="coerce").fillna(0.40)
    rules["rule_primary_type"] = rules["venue_type"].map(rule_primary_type)
    return rules.reset_index(drop=True)


def rule_exposure(
    host_x: np.ndarray,
    host_y: np.ndarray,
    rules: pd.DataFrame,
    rule_x: np.ndarray,
    rule_y: np.ndarray,
    radius: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(np.column_stack([rule_x, rule_y]))
    positive = np.zeros(len(host_x), dtype=float)
    restrictive = np.zeros(len(host_x), dtype=float)
    nearest_pos = np.full(len(host_x), np.nan, dtype=float)
    access = rules["access_semantic_score"].to_numpy(dtype=float)
    weights = rules["source_weight_for_model"].to_numpy(dtype=float)
    for idx, xy in enumerate(np.column_stack([host_x, host_y])):
        neighbors = tree.query_ball_point(xy, r=radius)
        if not neighbors:
            continue
        n = np.asarray(neighbors, dtype=int)
        dist = np.sqrt((rule_x[n] - xy[0]) ** 2 + (rule_y[n] - xy[1]) ** 2)
        val = distance_decay(dist, scale=150.0) * weights[n]
        pos_mask = access[n] > 0
        neg_mask = access[n] < 0
        if pos_mask.any():
            positive[idx] = float((val[pos_mask] * access[n][pos_mask]).sum())
            nearest_pos[idx] = float(dist[pos_mask].min())
        if neg_mask.any():
            restrictive[idx] = float((val[neg_mask] * np.abs(access[n][neg_mask])).sum())
    return positive, restrictive, nearest_pos


def entropy_from_counts(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[arr > 0]
    if len(arr) == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(values))) if len(values) > 1 else 0.0


def build_pet_points(grid: gpd.GeoDataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    pet = pd.read_csv(PET)
    pet = pet[
        (pet["year"] == 2025)
        & pet["coordinate_status"].eq("in_shenzhen_bounds")
        & np.isfinite(pet["lon_wgs84"])
        & np.isfinite(pet["lat_wgs84"])
    ].copy()
    pet["pet_service_weight"] = pet["candidate_classes"].map(pet_weight)
    for cls in PET_CLASS_WEIGHT:
        pet[cls.lower() + "_flag"] = pet["candidate_classes"].map(lambda s, c=cls: int(c in str(s).split("|")))
    gdf = gpd.GeoDataFrame(
        pet,
        geometry=gpd.points_from_xy(pet["lon_wgs84"], pet["lat_wgs84"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(gdf.to_crs(grid.crs), grid[["grid_id", "district_name", "geometry"]], how="left", predicate="within")
    joined = joined.to_crs("EPSG:4326")
    keep = [
        "candidate_id",
        "year",
        "name",
        "candidate_classes",
        "candidate_confidence",
        "category_primary",
        "category_secondary",
        "category_tertiary",
        "address",
        "district",
        "lon_wgs84",
        "lat_wgs84",
        "pet_service_weight",
        "grid_id",
        "district_name",
    ] + [cls.lower() + "_flag" for cls in PET_CLASS_WEIGHT]
    out_gdf = joined[keep + ["geometry"]].copy()
    OUT_PET_POINTS.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.drop(columns="geometry").to_csv(OUT_PET_POINTS, index=False, encoding="utf-8-sig")
    out_gdf.to_file(OUT_PET_POINTS_GEO, driver="GeoJSON")
    pet_out = out_gdf.drop(columns="geometry").copy().reset_index(drop=True)
    px, py = transform_xy(pet_out["lon_wgs84"], pet_out["lat_wgs84"])
    return pet_out, px, py


def build_grid_layer(grid: gpd.GeoDataFrame, pet: pd.DataFrame, pet_x: np.ndarray, pet_y: np.ndarray, rules: pd.DataFrame) -> pd.DataFrame:
    cent = grid.to_crs(PROJECTED_CRS).copy()
    cx = cent.geometry.centroid.x.to_numpy()
    cy = cent.geometry.centroid.y.to_numpy()
    exp500, class_exp, nearest = service_exposure(cx, cy, pet, pet_x, pet_y, radius=500.0)
    exp1000, _, _ = service_exposure(cx, cy, pet, pet_x, pet_y, radius=1000.0)

    out = grid[["grid_id", "district_name", "edge_cell", "intersect_area_m2"]].copy()
    out["valid_area_km2"] = out["intersect_area_m2"].astype(float) / 1_000_000
    out["pet_service_exposure_500m"] = exp500
    out["pet_service_exposure_1000m"] = exp1000
    out["nearest_pet_service_m"] = nearest
    for cls, arr in class_exp.items():
        out[f"{cls.lower()}_exposure_500m"] = arr

    counts = pet.groupby("grid_id").agg(
        pet_service_count_2025=("candidate_id", "count"),
        pet_service_weight_sum_2025=("pet_service_weight", "sum"),
        pet_medical_count_2025=("a_medical_flag", "sum"),
        pet_retail_count_2025=("a_retail_flag", "sum"),
        pet_grooming_count_2025=("a_grooming_flag", "sum"),
        pet_boarding_count_2025=("a_boarding_flag", "sum"),
        pet_training_count_2025=("a_training_flag", "sum"),
    ).reset_index()
    out = out.merge(counts, on="grid_id", how="left")
    count_cols = [c for c in out.columns if c.endswith("_2025")]
    out[count_cols] = out[count_cols].fillna(0)
    out["pet_service_density_2025_per_km2"] = (
        out["pet_service_count_2025"] / out["valid_area_km2"].replace(0, np.nan)
    ).fillna(0)
    out["pet_service_diversity_entropy_2025"] = out[
        [
            "pet_medical_count_2025",
            "pet_retail_count_2025",
            "pet_grooming_count_2025",
            "pet_boarding_count_2025",
            "pet_training_count_2025",
        ]
    ].apply(lambda row: entropy_from_counts(list(row)), axis=1)

    rule_grid = rules.groupby("grid_id").agg(
        explicit_rule_count=("source_id", "count"),
        positive_rule_count=("access_semantic_score", lambda s: int((s > 0).sum())),
        restrictive_rule_count=("access_semantic_score", lambda s: int((s < 0).sum())),
        mean_access_semantic_score=("access_semantic_score", "mean"),
    ).reset_index()
    out = out.merge(rule_grid, on="grid_id", how="left")
    for col in ["explicit_rule_count", "positive_rule_count", "restrictive_rule_count", "mean_access_semantic_score"]:
        out[col] = out[col].fillna(0)
    out["pet_service_high"] = out["pet_service_exposure_500m"] >= out.loc[out["pet_service_exposure_500m"] > 0, "pet_service_exposure_500m"].quantile(0.75)
    out["rule_seed_present"] = out["explicit_rule_count"] > 0
    out["pet_rule_ecology_class"] = np.select(
        [
            out["pet_service_high"] & (out["positive_rule_count"] > 0),
            out["pet_service_high"] & (out["positive_rule_count"] == 0),
            ~out["pet_service_high"] & (out["positive_rule_count"] > 0),
            (out["pet_service_count_2025"] > 0),
        ],
        [
            "high_pet_ecology_with_positive_rule",
            "high_pet_ecology_rule_silent",
            "positive_rule_without_high_pet_ecology",
            "low_pet_ecology_with_services",
        ],
        default="low_pet_ecology_rule_silent",
    )
    return out


def build_host_layer(host: pd.DataFrame, pet: pd.DataFrame, pet_x: np.ndarray, pet_y: np.ndarray, rules: pd.DataFrame) -> pd.DataFrame:
    host = host.reset_index(drop=True).copy()
    hx, hy = transform_xy(host["lon_wgs84"], host["lat_wgs84"])
    pet500, class_exp, nearest_pet = service_exposure(hx, hy, pet, pet_x, pet_y, radius=500.0)
    pet1000, _, _ = service_exposure(hx, hy, pet, pet_x, pet_y, radius=1000.0)
    rx, ry = transform_xy(rules["lon"], rules["lat"])
    pos_rule, neg_rule, nearest_pos_rule = rule_exposure(hx, hy, rules, rx, ry, radius=500.0)

    out = host.copy()
    out["pet_service_exposure_500m"] = pet500
    out["pet_service_exposure_1000m"] = pet1000
    out["nearest_pet_service_m"] = nearest_pet
    for cls, arr in class_exp.items():
        out[f"{cls.lower()}_exposure_500m"] = arr
    out["rule_positive_exposure_500m_v3"] = pos_rule
    out["rule_restrictive_exposure_500m_v3"] = neg_rule
    out["nearest_positive_rule_m_v3"] = nearest_pos_rule

    baseline = out["primary_venue_type"].map(HOST_BASELINE).fillna(-1.2)
    service_signal = np.log1p(out["pet_service_exposure_500m"])
    service_diverse_signal = np.log1p(
        (out[[f"{cls.lower()}_exposure_500m" for cls in PET_CLASS_WEIGHT]] > 0).sum(axis=1)
    )
    logit = (
        baseline
        + 0.35 * service_signal
        + 0.22 * service_diverse_signal
        + 1.35 * out["rule_positive_exposure_500m_v3"]
        - 0.95 * out["rule_restrictive_exposure_500m_v3"]
    )
    out["pet_rule_threshold_propensity_v3"] = 1.0 / (1.0 + np.exp(-logit))
    out["pet_ecology_threshold_shift"] = 0.35 * service_signal + 0.22 * service_diverse_signal
    out["v3_propensity_class"] = pd.cut(
        out["pet_rule_threshold_propensity_v3"],
        bins=[-1, 0.30, 0.45, 0.65, 1.0],
        labels=[
            "low_or_unknown_pet_rule_propensity",
            "low_but_plausible_pet_rule_propensity",
            "medium_pet_rule_propensity",
            "high_pet_rule_propensity",
        ],
    ).astype(str)
    out["v3_mechanism_class"] = np.select(
        [
            (out["pet_service_exposure_500m"] > 0) & (out["rule_positive_exposure_500m_v3"] > 0),
            (out["pet_service_exposure_500m"] > 0) & (out["rule_positive_exposure_500m_v3"] == 0),
            (out["pet_service_exposure_500m"] == 0) & (out["rule_positive_exposure_500m_v3"] > 0),
            (out["rule_restrictive_exposure_500m_v3"] > 0),
        ],
        [
            "pet_ecology_plus_positive_rule",
            "pet_ecology_without_observed_rule",
            "positive_rule_without_local_pet_ecology",
            "restrictive_rule_pressure",
        ],
        default="low_signal_silent",
    )
    return out


def build_edges(host: pd.DataFrame, pet: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    host = host.reset_index(drop=True).copy()
    pet = pet.reset_index(drop=True).copy()
    rules = rules.reset_index(drop=True).copy()
    rows: list[dict] = []
    hx, hy = transform_xy(host["lon_wgs84"], host["lat_wgs84"])
    px, py = transform_xy(pet["lon_wgs84"], pet["lat_wgs84"])
    rx, ry = transform_xy(rules["lon"], rules["lat"])
    pet_tree = cKDTree(np.column_stack([px, py]))
    rule_tree = cKDTree(np.column_stack([rx, ry]))
    host_sample = host[
        (host["pet_service_exposure_500m"] > 0)
        | (host["rule_positive_exposure_500m_v3"] > 0)
        | (host["v3_propensity_class"].isin(["medium_pet_rule_propensity", "high_pet_rule_propensity"]))
    ].copy()
    host_indices = host_sample.index.to_numpy()
    for idx in host_indices:
        hid = f"host:{host.at[idx, 'host_key']}"
        for pidx in pet_tree.query_ball_point([hx[idx], hy[idx]], r=500.0):
            dist = float(math.hypot(hx[idx] - px[pidx], hy[idx] - py[pidx]))
            rows.append(
                {
                    "source": hid,
                    "target": f"pet:{pet.at[pidx, 'candidate_id']}",
                    "edge_type": "host_pet_service_proximity",
                    "distance_m": dist,
                    "weight": float(distance_decay(np.asarray([dist]))[0] * pet.at[pidx, "pet_service_weight"]),
                    "source_grid_id": host.at[idx, "grid_id"],
                    "target_grid_id": pet.at[pidx, "grid_id"],
                }
            )
        for ridx in rule_tree.query_ball_point([hx[idx], hy[idx]], r=500.0):
            dist = float(math.hypot(hx[idx] - rx[ridx], hy[idx] - ry[ridx]))
            rows.append(
                {
                    "source": hid,
                    "target": f"rule:{rules.at[ridx, 'source_id']}",
                    "edge_type": "host_rule_seed_proximity",
                    "distance_m": dist,
                    "weight": float(distance_decay(np.asarray([dist]), scale=150.0)[0] * rules.at[ridx, "source_weight_for_model"]),
                    "source_grid_id": host.at[idx, "grid_id"],
                    "target_grid_id": rules.at[ridx, "grid_id"],
                }
            )
    return pd.DataFrame(rows)


def grid_topology(grid: gpd.GeoDataFrame, grid_layer: pd.DataFrame) -> pd.DataFrame:
    g = nx.Graph()
    attrs = grid_layer.set_index("grid_id").to_dict("index")
    for grid_id, attr in attrs.items():
        g.add_node(grid_id, **attr)
    projected = grid[["grid_id", "geometry"]].to_crs(PROJECTED_CRS).copy()
    sindex = projected.sindex
    for idx, row in projected.iterrows():
        geom = row.geometry.buffer(1)
        for cand_idx in list(sindex.intersection(geom.bounds)):
            if cand_idx <= idx:
                continue
            other = projected.iloc[cand_idx]
            if geom.intersects(other.geometry):
                a = row["grid_id"]
                b = other["grid_id"]
                a_attr = attrs.get(a, {})
                b_attr = attrs.get(b, {})
                pet_weight = 1.0 + min(
                    float(a_attr.get("pet_service_exposure_500m", 0)) + float(b_attr.get("pet_service_exposure_500m", 0)),
                    10.0,
                ) / 10.0
                rule_weight = 1.0 + min(
                    float(a_attr.get("positive_rule_count", 0)) + float(b_attr.get("positive_rule_count", 0)),
                    5.0,
                ) / 5.0
                g.add_edge(a, b, weight=pet_weight * rule_weight)
    degree = dict(g.degree())
    weighted_degree = dict(g.degree(weight="weight"))
    try:
        pagerank = nx.pagerank(g, weight="weight", max_iter=200)
    except Exception:
        pagerank = {n: 0.0 for n in g.nodes}
    out = pd.DataFrame(
        {
            "grid_id": list(g.nodes),
            "topological_degree": [degree[n] for n in g.nodes],
            "weighted_topological_degree": [weighted_degree[n] for n in g.nodes],
            "pet_rule_pagerank": [pagerank[n] for n in g.nodes],
        }
    )
    out = out.merge(grid_layer, on="grid_id", how="left")
    out["topology_ecology_class"] = np.select(
        [
            (out["pet_rule_pagerank"] >= out["pet_rule_pagerank"].quantile(0.90)) & out["pet_service_high"],
            (out["pet_rule_pagerank"] >= out["pet_rule_pagerank"].quantile(0.90)) & ~out["pet_service_high"],
            (out["pet_rule_pagerank"] < out["pet_rule_pagerank"].quantile(0.90)) & out["pet_service_high"],
        ],
        [
            "network_core_with_pet_ecology",
            "network_core_without_high_pet_ecology",
            "pet_ecology_peripheral_or_local",
        ],
        default="low_network_pet_rule_signal",
    )
    return out


def main() -> None:
    OUT_EDGES.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    grid = gpd.read_file(GRID).to_crs(PROJECTED_CRS)
    rules = load_rules()
    pet, pet_x, pet_y = build_pet_points(grid)
    grid_layer = build_grid_layer(grid, pet, pet_x, pet_y, rules)
    grid_layer.to_csv(OUT_GRID, index=False, encoding="utf-8-sig")
    grid_geo = grid.to_crs("EPSG:4326").merge(grid_layer, on=["grid_id", "district_name", "edge_cell", "intersect_area_m2"], how="left")
    grid_geo.to_file(OUT_GRID_GEO, driver="GeoJSON")

    host = pd.read_csv(HOST)
    host_v3 = build_host_layer(host, pet, pet_x, pet_y, rules)
    host_v3.to_csv(OUT_HOST, index=False, encoding="utf-8-sig")

    edges = build_edges(host_v3, pet, rules)
    edges.to_csv(OUT_EDGES, index=False, encoding="utf-8-sig")
    topo = grid_topology(grid, grid_layer)
    topo.to_csv(OUT_GRID_TOPO, index=False, encoding="utf-8-sig")

    counts = {
        "pet_service_points_2025": int(len(pet)),
        "pet_service_classes": dict(Counter("|".join(pet["candidate_classes"].astype(str)).split("|"))),
        "grid_rows": int(len(grid_layer)),
        "grid_rows_with_pet_service": int((grid_layer["pet_service_count_2025"] > 0).sum()),
        "grid_pet_rule_ecology_class": grid_layer["pet_rule_ecology_class"].value_counts().to_dict(),
        "host_rows": int(len(host_v3)),
        "host_v3_propensity_class": host_v3["v3_propensity_class"].value_counts().to_dict(),
        "host_v3_mechanism_class": host_v3["v3_mechanism_class"].value_counts().to_dict(),
        "network_edges": int(len(edges)),
        "edge_types": edges["edge_type"].value_counts().to_dict() if len(edges) else {},
        "topology_classes": topo["topology_ecology_class"].value_counts().to_dict(),
    }
    top_hosts = host_v3.sort_values("pet_rule_threshold_propensity_v3", ascending=False).head(30)[
        [
            "host_name",
            "primary_venue_type",
            "district_name",
            "grid_id",
            "pet_rule_threshold_propensity_v3",
            "v3_propensity_class",
            "v3_mechanism_class",
            "pet_service_exposure_500m",
            "rule_positive_exposure_500m_v3",
            "rule_restrictive_exposure_500m_v3",
            "nearest_pet_service_m",
            "nearest_positive_rule_m_v3",
        ]
    ].to_dict("records")
    top_grids = topo.sort_values("pet_rule_pagerank", ascending=False).head(30)[
        [
            "grid_id",
            "district_name",
            "pet_rule_pagerank",
            "topology_ecology_class",
            "pet_service_count_2025",
            "pet_service_exposure_500m",
            "positive_rule_count",
            "pet_rule_ecology_class",
        ]
    ].to_dict("records")

    REPORT.write_text(
        f"""# 52 Pet Service Ecology Network V3 Report

Locked title: Pet-Service Ecologies Outpace Urban Readiness in Shenzhen: Rule-Liminal Venues and the Uneven Emergence of Companion-Animal Urban Capability

## Outputs

- Pet service points: `{OUT_PET_POINTS.relative_to(PROJECT_ROOT)}`
- Grid pet-rule ecology: `{OUT_GRID.relative_to(PROJECT_ROOT)}`
- Grid pet-rule ecology GeoJSON: `{OUT_GRID_GEO.relative_to(PROJECT_ROOT)}`
- Host pet-rule ecology: `{OUT_HOST.relative_to(PROJECT_ROOT)}`
- Multilayer edge list: `{OUT_EDGES.relative_to(PROJECT_ROOT)}`
- Grid topology table: `{OUT_GRID_TOPO.relative_to(PROJECT_ROOT)}`

## Counts

`{counts}`

## Top Host Candidates

`{top_hosts}`

## Top Network Grids

`{top_grids}`

## Interpretation

This v3 layer treats pet-service POI as an ecological pressure field rather than a passive service-background control. The model asks whether nearby pet medical, retail, grooming, boarding and training services lower the threshold for host venues to adopt or publicize pet-friendly rules.

The topology table adds a spatial-network view: grid cells are connected by adjacency and weighted by pet-service and rule signals. This supports a new mechanism claim: pet-friendly capability may emerge not only from local density but from a grid's position in a pet-service/rule topology.

## Claim Boundary

The v3 score is not observed pet-friendliness. It is a candidate mechanism layer: pet-service ecology plus rule exposure may make some host venues more likely to cross a rule-adoption threshold. Primary-source verification is still required before any host is claimed as explicitly pet-friendly.
""",
        encoding="utf-8",
    )
    METHOD.write_text(
        """# Pet Service Ecology Network Model V3

This method upgrade responds to four project needs:

1. More potential venue mining is useful, but candidate volume alone is not enough. Each point must enter as a typed agent with provenance and verification status.
2. Direct companion-animal POI should be part of the model, not only a descriptive background layer. Pet services are interpreted as ecological pressure: they indicate routine pet presence, care infrastructure and consumer normalization.
3. The model should move beyond kernel decay. V3 uses a multilayer spatial network: pet-service nodes, explicit rule-seed nodes, host-venue nodes and 500m grid nodes.
4. The research can stand without social media or street view if the claim is reframed around open-data rule geography, service ecology and threshold urban capability.

Core mechanism:

pet service ecology + positive rule exposure - restrictive rule exposure -> lower or raise host rule-adoption threshold.

Topology mechanism:

Adjacent 500m grids form a spatial graph. Grids with high pet-service ecology and high network centrality may become capability cores, while high-service but rule-silent grids become candidate tipping zones.

Publication boundary:

V3 estimates latent threshold propensity and topological position. It does not observe real pet-friendly access unless confirmed by official or operator source evidence.
""",
        encoding="utf-8",
    )
    print(counts)
    print(OUT_HOST)


if __name__ == "__main__":
    main()
