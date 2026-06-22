#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import math
import warnings

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import box

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "submission_package" / "figures_v30_map_evidence_vignettes"
PANELS = OUT / "panels"
COMP = OUT / "composites"
PREVIEWS = OUT / "previews"
SRC = OUT / "source_data"
for d in [PANELS, COMP, PREVIEWS, SRC]:
    d.mkdir(parents=True, exist_ok=True)

INK = "#24211f"
TEXT = "#504a43"
MUTED = "#7e766c"
FAINT = "#eee9df"
LAND = "#fbfaf7"
GRID = "#e9e3d8"
BUILDING = "#d7d0c3"
ROAD = "#b5b1a8"
PED = "#7fa6a4"
PUBLIC = "#dbe8dc"
BLUE = "#315f7d"
BLUE_DARK = "#214f6d"
GREEN = "#4f8d69"
GOLD = "#efcf73"
GOLD_LIGHT = "#f6dfa1"
RUST = "#d9806d"
RUST_DARK = "#ab5a52"
SILENT = "#b8b0a5"
GREY = "#c9c3ba"

DISTRICT_EN = {
    "宝安区": "Baoan",
    "龙岗区": "Longgang",
    "龙华区": "Longhua",
    "南山区": "Nanshan",
    "福田区": "Futian",
    "罗湖区": "Luohu",
    "光明区": "Guangming",
    "坪山区": "Pingshan",
    "盐田区": "Yantian",
}

TYPE_LABELS = {
    "shopping_mall": "Malls",
    "hotel": "Hotels",
    "residential_property": "Residential",
    "park_or_recreation": "Parks",
}

TYPE_COLORS = {
    "shopping_mall": RUST,
    "hotel": BLUE,
    "residential_property": GREY,
    "park_or_recreation": GREEN,
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 5.4,
        "axes.titlesize": 6.0,
        "axes.labelsize": 4.8,
        "xtick.labelsize": 4.0,
        "ytick.labelsize": 4.0,
        "legend.fontsize": 4.0,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.linewidth": 0.42,
    }
)


def save(fig: plt.Figure, stem: Path, dpi: int = 600, tight: bool = False) -> None:
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.025} if tight else {}
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, **kwargs)
    fig.savefig(stem.with_suffix(".pdf"), **kwargs)
    fig.savefig(stem.with_suffix(".svg"), **kwargs)
    plt.close(fig)


def read_gdf(rel: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(ROOT / rel)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    return g.to_crs("EPSG:32649")


def read_csv_points(rel: str, lon: str, lat: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(ROOT / rel)
    df = df.dropna(subset=[lon, lat]).copy()
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326").to_crs("EPSG:32649")


def load_data() -> dict[str, gpd.GeoDataFrame | pd.DataFrame]:
    grid = read_gdf("data_processed/model/grid_emergence_suppression_indices_500m_2025_v51_sparse_repaired.geojson")
    boundary = read_gdf("data_processed/geo/shenzhen_boundary_verified.geojson")
    roads = read_gdf("data_processed/osm/shenzhen_osm_roads.gpkg")
    buildings = read_gdf("data_processed/osm/shenzhen_osm_buildings_3d.gpkg")
    ped = read_gdf("data_processed/osm/shenzhen_osm_pedestrian_paths.gpkg")
    public = read_gdf("data_processed/osm/shenzhen_osm_public_space.gpkg")
    pet = read_gdf("data_processed/platform/pet_service_ecology_points_2025_v3.geojson")
    rules = read_gdf("data_processed/platform/rule_semantic_geocoded_points_v1.geojson")
    hosts = read_csv_points("data_processed/platform/rule_liminal_host_candidates_2025_v4.csv", "lon_wgs84", "lat_wgs84")
    queue = read_csv_points("data_processed/platform/rule_liminal_verification_queue_2025_v4.csv", "lon_wgs84", "lat_wgs84")
    return {
        "grid": grid,
        "boundary": boundary,
        "roads": roads,
        "buildings": buildings,
        "ped": ped,
        "public": public,
        "pet": pet,
        "rules": rules,
        "hosts": hosts,
        "queue": queue,
    }


def norm01(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((v - lo) / (hi - lo), 0, 1))


def local_extent(geom, width_m: float = 3200, aspect: float = 1.08) -> tuple[float, float, float, float]:
    c = geom.centroid
    w = width_m
    h = width_m / aspect
    return c.x - w / 2, c.y - h / 2, c.x + w / 2, c.y + h / 2


def select_cases(data: dict[str, gpd.GeoDataFrame | pd.DataFrame]) -> pd.DataFrame:
    grid = data["grid"].copy()
    rows = []
    selected_grid_ids: set[str] = set()

    def take(case_id: str, district: str, role: str, query: pd.Series, score_cols: list[str], note: str) -> None:
        sub = grid[grid["district_name"].eq(district)].copy()
        sub = sub[query.loc[sub.index]]
        sub = sub[~sub["grid_id"].isin(selected_grid_ids)]
        if sub.empty:
            sub = grid[grid["district_name"].eq(district)].copy()
            sub = sub[~sub["grid_id"].isin(selected_grid_ids)]
        if sub.empty:
            return
        score = np.zeros(len(sub), dtype=float)
        for col in score_cols:
            if col in sub.columns:
                score += pd.to_numeric(sub[col], errors="coerce").fillna(0).to_numpy()
        row = sub.assign(_case_score=score).sort_values("_case_score", ascending=False).iloc[0]
        selected_grid_ids.add(str(row["grid_id"]))
        rows.append(
            {
                "case_id": case_id,
                "district_name": district,
                "district_en": DISTRICT_EN.get(district, district),
                "evidence_role": role,
                "focal_grid_id": row["grid_id"],
                "selection_note": note,
                "selection_score": float(row["_case_score"]),
                "suppression_index_v51": float(row.get("suppression_index_v51", 0)),
                "emergence_index_v51": float(row.get("emergence_index_v51", 0)),
                "pet_ecology_norm": float(row.get("pet_ecology_norm", 0)),
                "liminal_host_norm": float(row.get("liminal_host_norm", 0)),
                "positive_rule_count": float(row.get("positive_rule_count", 0)),
            }
        )

    zero_rule = pd.to_numeric(grid["positive_rule_count"], errors="coerce").fillna(0).eq(0)
    positive = pd.to_numeric(grid["positive_rule_count"], errors="coerce").fillna(0).gt(0)
    explicit_rule = pd.to_numeric(grid["explicit_rule_count"], errors="coerce").fillna(0).gt(0)
    restrictive = pd.to_numeric(grid["restrictive_rule_count"], errors="coerce").fillna(0).gt(0)
    high_queue_proxy = pd.to_numeric(grid["liminal_or_frontier_hosts"], errors="coerce").fillna(0).gt(25)
    take(
        "baoan_high_host_low_rule",
        "宝安区",
        "liminal substrate",
        zero_rule & grid["pet_ecology_norm"].gt(0.55),
        ["liminal_host_norm", "pet_ecology_norm", "suppression_index_v51"],
        "high host capacity and pet ecology with no visible positive rule in focal grid",
    )
    take(
        "longgang_suppressed_frontier",
        "龙岗区",
        "diagnostic frontier",
        zero_rule & grid["isna_dummy"].isna() if "isna_dummy" in grid.columns else zero_rule,
        ["suppression_index_v51", "pet_ecology_norm", "liminal_host_norm"],
        "high suppression and high pet ecology with no visible positive rule in focal grid",
    )
    take(
        "futian_visible_rule_core",
        "福田区",
        "traceable rule evidence",
        positive,
        ["positive_rule_count", "pet_ecology_norm", "liminal_host_norm"],
        "positive-rule visibility core with high local rule count",
    )
    take(
        "nanshan_mixed_contrast",
        "南山区",
        "mixed urban core",
        grid["pet_ecology_norm"].gt(0.45),
        ["positive_rule_count", "pet_ecology_norm", "suppression_index_v51"],
        "service-rich urban core where visible and silent rule conditions sit near each other",
    )
    take(
        "longhua_emerging_pressure",
        "龙华区",
        "emerging liminality",
        zero_rule,
        ["liminal_potential_norm", "liminal_host_norm", "pet_ecology_norm"],
        "high liminal potential and host capacity before rule visibility is widespread",
    )
    take(
        "guangming_transitional_edge",
        "光明区",
        "peripheral transition",
        zero_rule,
        ["mean_liminal_potential", "liminal_host_norm", "pet_ecology_norm"],
        "peripheral transition case with weaker rule visibility and emerging substrate",
    )
    take(
        "luohu_compact_rule_visibility",
        "罗湖区",
        "compact rule trace",
        positive,
        ["positive_rule_count", "liminal_host_norm", "pet_ecology_norm"],
        "dense older-core case where visible rule traces sit close to host and service points",
    )
    take(
        "yantian_public_space_edge",
        "盐田区",
        "public-space edge",
        grid["pet_ecology_norm"].gt(0.20),
        ["pet_ecology_norm", "positive_rule_count", "mean_liminal_potential"],
        "coastal public-space edge with thinner service ecology and localized rule traces",
    )
    take(
        "pingshan_ecology_silent_edge",
        "坪山区",
        "silent ecology edge",
        zero_rule,
        ["pet_ecology_norm", "ecology_silent_hosts", "suppression_index_v51"],
        "peripheral ecology-present grid where host candidates are visible before positive rules",
    )
    take(
        "baoan_verification_pressure",
        "宝安区",
        "verification pressure",
        zero_rule & high_queue_proxy,
        ["liminal_or_frontier_hosts", "liminal_potential_norm", "pet_ecology_norm"],
        "large host substrate with many candidates requiring venue-level verification",
    )
    take(
        "longgang_service_rule_gap",
        "龙岗区",
        "service-rule gap",
        zero_rule & grid["pet_ecology_norm"].gt(0.35),
        ["pet_ecology_norm", "explicit_rule_gap_norm_v51", "suppression_index_v51"],
        "service ecology is present but positive rule visibility remains locally thin",
    )
    take(
        "nanshan_restrictive_contact_zone",
        "南山区",
        "restrictive contact",
        restrictive | explicit_rule,
        ["restrictive_rule_count", "explicit_rule_count", "pet_ecology_norm"],
        "visible rule contact zone where service ecology meets explicit or restrictive rule traces",
    )
    out = pd.DataFrame(rows)
    out.to_csv(SRC / "v30_vignette_case_selection.csv", index=False, encoding="utf-8-sig")
    return out


def subset_extent(gdf: gpd.GeoDataFrame, extent: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    b = box(*extent)
    return gdf[gdf.intersects(b)].copy()


def draw_texture(ax: plt.Axes, data: dict[str, gpd.GeoDataFrame | pd.DataFrame], extent: tuple[float, float, float, float]) -> None:
    ax.set_facecolor(LAND)
    layers = [
        ("public", PUBLIC, 0.0, 0.52, 1),
        ("buildings", BUILDING, 0.0, 0.48, 2),
        ("roads", ROAD, 0.18, 0.34, 3),
        ("ped", PED, 0.24, 0.48, 4),
    ]
    for key, color, lw, alpha, z in layers:
        sub = subset_extent(data[key], extent)
        if sub.empty:
            continue
        if key in {"roads", "ped"}:
            sub.plot(ax=ax, color=color, linewidth=lw, alpha=alpha, zorder=z)
        else:
            sub.plot(ax=ax, color=color, edgecolor="none", alpha=alpha, zorder=z)


def local_stats(data: dict[str, gpd.GeoDataFrame | pd.DataFrame], extent: tuple[float, float, float, float], focal_grid_id: str) -> dict[str, object]:
    grid = subset_extent(data["grid"], extent)
    pet = subset_extent(data["pet"], extent)
    hosts = subset_extent(data["hosts"], extent)
    queue = subset_extent(data["queue"], extent)
    rules = subset_extent(data["rules"], extent)
    priority = hosts[hosts["liminal_class"].isin(["rule_liminal_threshold_core", "ecology_ready_rule_liminal_frontier"])]
    silent = hosts[hosts["liminal_class"].eq("ecology_present_rule_silent")]
    type_counts = hosts["primary_venue_type"].value_counts().to_dict()
    stats = {
        "focal_grid_id": focal_grid_id,
        "local_grid_cells": int(len(grid)),
        "pet_service_points": int(len(pet)),
        "host_candidates": int(len(hosts)),
        "priority_hosts": int(len(priority)),
        "silent_hosts": int(len(silent)),
        "verification_queue_candidates": int(len(queue)),
        "visible_rule_records": int(len(rules)),
        "positive_rule_grid_count": int((pd.to_numeric(grid.get("positive_rule_count", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()) if len(grid) else 0,
        "mean_pet_ecology_norm": float(pd.to_numeric(grid.get("pet_ecology_norm", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if len(grid) else 0,
        "mean_liminal_host_norm": float(pd.to_numeric(grid.get("liminal_host_norm", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if len(grid) else 0,
        "mean_rule_visibility_norm": float(pd.to_numeric(grid.get("positive_rule_norm_v51", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if len(grid) else 0,
        "mean_suppression_index": float(pd.to_numeric(grid.get("suppression_index_v51", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if len(grid) else 0,
    }
    for typ in TYPE_LABELS:
        stats[f"host_type_{typ}"] = int(type_counts.get(typ, 0))
    return stats


def draw_local_map(ax: plt.Axes, data: dict[str, gpd.GeoDataFrame | pd.DataFrame], extent: tuple[float, float, float, float], focal_grid_id: str, case_color: str) -> None:
    draw_texture(ax, data, extent)
    grid = subset_extent(data["grid"], extent)
    focal = grid[grid["grid_id"].eq(focal_grid_id)]
    if len(grid):
        grid.plot(ax=ax, color="none", edgecolor=GRID, linewidth=0.13, alpha=0.50, zorder=6)
        suppressed = grid[grid["grid_emergence_type_v51"].eq("suppressed_emergence_frontier")]
        if len(suppressed):
            suppressed.plot(ax=ax, color=RUST, edgecolor="none", alpha=0.065, zorder=7)
        transitional = grid[grid["grid_emergence_type_v51"].eq("mixed_or_transitional_zone")]
        if len(transitional):
            transitional.plot(ax=ax, color=GOLD, edgecolor="none", alpha=0.055, zorder=7)
    if len(focal):
        focal.boundary.plot(ax=ax, color=INK, linewidth=0.78, alpha=0.86, zorder=20)
    hosts = subset_extent(data["hosts"], extent)
    priority = hosts[hosts["liminal_class"].isin(["rule_liminal_threshold_core", "ecology_ready_rule_liminal_frontier"])]
    silent = hosts[hosts["liminal_class"].eq("ecology_present_rule_silent")]
    pet = subset_extent(data["pet"], extent)
    queue = subset_extent(data["queue"], extent)
    rules = subset_extent(data["rules"], extent)
    if len(silent):
        pts = silent.sample(min(len(silent), 700), random_state=16)
        pts.plot(ax=ax, color="white", markersize=8.8, alpha=0.52, linewidth=0, zorder=10)
        pts.plot(ax=ax, color=SILENT, markersize=3.0, alpha=0.48, edgecolor="none", linewidth=0, zorder=11)
    if len(priority):
        pts = priority.sample(min(len(priority), 820), random_state=17)
        pts.plot(ax=ax, color="white", markersize=17.0, alpha=0.86, linewidth=0, zorder=12)
        pts.plot(ax=ax, color=RUST, markersize=7.8, alpha=0.80, edgecolor="#fffdf8", linewidth=0.38, zorder=13)
    if len(pet):
        pet.plot(ax=ax, color="white", markersize=17.0, alpha=0.78, linewidth=0, zorder=14)
        pet.plot(ax=ax, color=BLUE_DARK, markersize=7.5, alpha=0.72, edgecolor="#fffdf8", linewidth=0.34, zorder=15)
    if len(queue):
        pts = queue.sample(min(len(queue), 280), random_state=18)
        pts.plot(ax=ax, color="white", markersize=22.0, alpha=0.82, linewidth=0, zorder=16)
        pts.plot(ax=ax, color=GOLD, markersize=10.0, alpha=0.86, edgecolor="#fffdf8", linewidth=0.40, zorder=17)
    if len(rules):
        rules.plot(ax=ax, color="white", markersize=56, alpha=0.96, linewidth=0, zorder=18)
        rules.plot(ax=ax, color=GREEN, markersize=36, alpha=0.96, edgecolor="#fffdf8", linewidth=0.48, zorder=19)
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#d8d3ca")
        spine.set_linewidth(0.42)


def draw_metric_strips(ax: plt.Axes, stats: dict[str, object]) -> None:
    metrics = [
        ("service ecology", stats["mean_pet_ecology_norm"], BLUE),
        ("host liminality", stats["mean_liminal_host_norm"], RUST),
        ("rule visibility", stats["mean_rule_visibility_norm"], GREEN),
        ("suppression", stats["mean_suppression_index"], RUST_DARK),
    ]
    ax.text(0.055, 0.770, "local indices", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.1, fontweight="bold", color=INK)
    y0 = 0.708
    for i, (label, val, color) in enumerate(metrics):
        y = y0 - i * 0.060
        ax.text(0.055, y + 0.017, label, transform=ax.transAxes, ha="left", va="center", fontsize=4.6, color=TEXT)
        ax.add_patch(Rectangle((0.455, y), 0.365, 0.034, transform=ax.transAxes, facecolor="#eee9df", edgecolor="none"))
        ax.add_patch(Rectangle((0.455, y), 0.365 * norm01(float(val)), 0.034, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=0.80))
        ax.text(0.875, y + 0.017, f"{float(val):.2f}", transform=ax.transAxes, ha="right", va="center", fontsize=4.8, fontweight="bold", color=INK)


def draw_count_stack(ax: plt.Axes, stats: dict[str, object]) -> None:
    counts = [
        ("pet services", stats["pet_service_points"], BLUE),
        ("host candidates", stats["host_candidates"], RUST),
        ("verify queue", stats["verification_queue_candidates"], GOLD),
        ("rule records", stats["visible_rule_records"], GREEN),
    ]
    maxv = max([int(v) for _, v, _ in counts] + [1])
    ax.text(0.055, 0.478, "local evidence counts", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.1, fontweight="bold", color=INK)
    for i, (label, val, color) in enumerate(counts):
        col = i % 2
        row = i // 2
        x = 0.055 + col * 0.450
        y = 0.358 - row * 0.120
        ax.add_patch(Rectangle((x, y), 0.405, 0.092, transform=ax.transAxes, facecolor="#fbfaf7", edgecolor="#e4ded4", linewidth=0.35))
        width = 0.345 * (math.log1p(int(val)) / max(math.log1p(maxv), 1e-6))
        ax.add_patch(Rectangle((x + 0.030, y + 0.018), width, 0.012, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=0.86))
        ax.text(x + 0.030, y + 0.069, f"{int(val):,}", transform=ax.transAxes, ha="left", va="center", fontsize=6.0, fontweight="bold", color=INK)
        ax.text(x + 0.030, y + 0.044, label, transform=ax.transAxes, ha="left", va="center", fontsize=4.4, color=MUTED)


def draw_host_type_mini(ax: plt.Axes, stats: dict[str, object]) -> None:
    vals = [(TYPE_LABELS[t], int(stats[f"host_type_{t}"]), TYPE_COLORS[t]) for t in TYPE_LABELS]
    total = max(sum(v for _, v, _ in vals), 1)
    left = 0.055
    y = 0.104
    ax.text(0.055, 0.170, "host mix", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.1, fontweight="bold", color=INK)
    for label, val, color in vals:
        w = 0.840 * val / total
        if w > 0:
            ax.add_patch(Rectangle((left, y), w, 0.038, transform=ax.transAxes, facecolor=color, edgecolor="white", linewidth=0.25, alpha=0.82))
        left += w
    ranked = sorted(vals, key=lambda x: x[1], reverse=True)
    top = ranked[0]
    second = ranked[1]
    ax.text(0.055, 0.082, f"top: {top[0]} {top[1]:,}", transform=ax.transAxes, ha="left", va="top", fontsize=4.2, color=TEXT)
    ax.text(0.895, 0.082, f"next: {second[0]} {second[1]:,}", transform=ax.transAxes, ha="right", va="top", fontsize=4.2, color=MUTED)


def draw_evidence_panel(ax: plt.Axes, case: pd.Series, stats: dict[str, object]) -> None:
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="white", edgecolor="#ddd6cb", linewidth=0.55))
    ax.text(0.055, 0.945, "matched local evidence", transform=ax.transAxes, ha="left", va="top", fontsize=6.8, fontweight="bold", color=INK)
    ax.text(
        0.055,
        0.875,
        case["evidence_role"],
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.0,
        color="white",
        bbox={"facecolor": RUST_DARK if "frontier" in case["evidence_role"] else BLUE, "edgecolor": "none", "pad": 1.3},
    )
    ax.text(0.895, 0.875, f"n={stats['local_grid_cells']} cells", transform=ax.transAxes, ha="right", va="center", fontsize=5.0, fontweight="bold", color=TEXT)
    rule_share = int(stats["positive_rule_grid_count"]) / max(int(stats["local_grid_cells"]), 1)
    ax.text(0.055, 0.824, f"same map window; rule-visible cells {rule_share:.0%}", transform=ax.transAxes, ha="left", va="top", fontsize=4.55, color=MUTED)
    draw_metric_strips(ax, stats)
    draw_count_stack(ax, stats)
    draw_host_type_mini(ax, stats)
    ax.text(
        0.055,
        0.018,
        "absence of visible rule is not proof of prohibition",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.05,
        color=MUTED,
    )


def draw_legend(ax: plt.Axes) -> None:
    items = [
        ("patch", PUBLIC, "public space"),
        ("patch", BUILDING, "building texture"),
        ("line", ROAD, "road network"),
        ("line", PED, "pedestrian path"),
        ("point", RUST, "rule-liminal host"),
        ("point", BLUE_DARK, "pet-service POI"),
        ("point", GOLD, "validation queue"),
        ("point", GREEN, "visible rule"),
        ("frame", INK, "focal 500 m grid"),
    ]
    for i, (kind, color, label) in enumerate(items):
        x = 0.02 + (i % 5) * 0.185
        y = 0.40 if i < 5 else 0.12
        if kind == "patch":
            ax.add_patch(Rectangle((x, y - 0.06), 0.035, 0.12, transform=ax.transAxes, facecolor=color, edgecolor="#d2ccc2", linewidth=0.25, alpha=0.70))
        elif kind == "line":
            ax.plot([x, x + 0.042], [y, y], transform=ax.transAxes, color=color, linewidth=1.4, alpha=0.75)
        elif kind == "frame":
            ax.add_patch(Rectangle((x, y - 0.05), 0.04, 0.10, transform=ax.transAxes, fill=False, edgecolor=color, linewidth=0.80))
        else:
            ax.scatter([x + 0.020], [y], transform=ax.transAxes, s=28, color=color, edgecolor="white", linewidth=0.35)
    ax.text(x + 0.050, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=4.4, color=MUTED)


def draw_composite_header(ax_title: plt.Axes, page_label: str) -> None:
    ax_title.set_axis_off()
    ax_title.text(
        0.0,
        0.97,
        f"Fig. X | Local map-evidence vignettes make rule-liminal urban capability visible ({page_label})",
        transform=ax_title.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=INK,
    )
    ax_title.text(
        0.0,
        0.74,
        "Each card uses the same local window for OSM texture, 500 m grids, pet services, host candidates, validation queue and traceable rule records.",
        transform=ax_title.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color=TEXT,
    )
    draw_legend(ax_title)


def render_composite_page(
    data: dict[str, gpd.GeoDataFrame | pd.DataFrame],
    cases: pd.DataFrame,
    start_idx: int,
    page_label: str,
    out_stem: str,
) -> Path:
    fig = plt.figure(figsize=(12.8, 12.0))
    outer = GridSpec(5, 2, figure=fig, height_ratios=[0.36, 1, 1, 1, 0.12], hspace=0.34, wspace=0.12, left=0.035, right=0.985, bottom=0.035, top=0.94)
    ax_title = fig.add_subplot(outer[0, :])
    draw_composite_header(ax_title, page_label)
    for j, (_, case) in enumerate(cases.iterrows()):
        sub = outer[1 + j // 2, j % 2].subgridspec(1, 2, width_ratios=[1.28, 0.96], wspace=0.035)
        ax_map = fig.add_subplot(sub[0, 0])
        ax_ev = fig.add_subplot(sub[0, 1])
        grid = data["grid"]
        focal = grid[grid["grid_id"].eq(case["focal_grid_id"])].iloc[0]
        extent = local_extent(focal.geometry, 3300 if case["case_id"] != "futian_visible_rule_core" else 2600)
        stats = local_stats(data, extent, case["focal_grid_id"])
        color = GREEN if "rule" in case["evidence_role"] else (RUST if "frontier" in case["evidence_role"] or "liminal" in case["evidence_role"] else BLUE)
        draw_local_map(ax_map, data, extent, case["focal_grid_id"], color)
        draw_evidence_panel(ax_ev, case, stats)
        letter = chr(ord("a") + start_idx + j)
        ax_map.text(0.0, 1.035, letter, transform=ax_map.transAxes, ha="left", va="bottom", fontsize=8.2, fontweight="bold", color=INK)
        ax_map.text(0.065, 1.040, f"{case['district_en']} | {case['evidence_role']}", transform=ax_map.transAxes, ha="left", va="bottom", fontsize=5.9, fontweight="bold", color=INK)
    ax_note = fig.add_subplot(outer[4, :])
    ax_note.set_axis_off()
    ax_note.text(0.0, 0.58, "Claim boundary: these cards diagnose visible rule-ecology mismatch and validation priorities; they do not prove hidden prohibition or observed pet access at every venue.", ha="left", va="center", fontsize=5.0, color=MUTED)
    stem = COMP / out_stem
    save(fig, stem, dpi=520, tight=False)
    return stem.with_suffix(".png")


def render_card(data: dict[str, gpd.GeoDataFrame | pd.DataFrame], case: pd.Series, letter: str, composite: bool = False) -> tuple[plt.Figure, dict[str, object]]:
    grid = data["grid"]
    focal = grid[grid["grid_id"].eq(case["focal_grid_id"])].iloc[0]
    extent = local_extent(focal.geometry, 3300 if case["case_id"] != "futian_visible_rule_core" else 2600)
    stats = local_stats(data, extent, case["focal_grid_id"])
    color = GREEN if "rule" in case["evidence_role"] else (RUST if "frontier" in case["evidence_role"] or "liminal" in case["evidence_role"] else BLUE)

    fig = plt.figure(figsize=(6.2, 3.05))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.28, 0.96], wspace=0.035, left=0.035, right=0.985, bottom=0.075, top=0.82)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_ev = fig.add_subplot(gs[0, 1])
    draw_local_map(ax_map, data, extent, case["focal_grid_id"], color)
    draw_evidence_panel(ax_ev, case, stats)
    title = f"{case['district_en']} | {case['case_id'].replace('_', ' ')}"
    fig.text(0.035, 0.965, letter, ha="left", va="top", fontsize=9.0, fontweight="bold", color=INK)
    fig.text(0.070, 0.960, title, ha="left", va="top", fontsize=7.0, fontweight="bold", color=INK)
    fig.text(0.070, 0.910, case["selection_note"], ha="left", va="top", fontsize=4.6, color=MUTED)
    stats.update(
        {
            "case_id": case["case_id"],
            "district_name": case["district_name"],
            "district_en": case["district_en"],
            "evidence_role": case["evidence_role"],
            "focal_grid_id": case["focal_grid_id"],
            "extent_minx": extent[0],
            "extent_miny": extent[1],
            "extent_maxx": extent[2],
            "extent_maxy": extent[3],
            "selection_note": case["selection_note"],
        }
    )
    return fig, stats


def render_overview(data: dict[str, gpd.GeoDataFrame | pd.DataFrame], cases: pd.DataFrame) -> None:
    grid = data["grid"]
    boundary = data["boundary"]
    fig, ax = plt.subplots(figsize=(4.8, 3.45))
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.05, top=0.86)
    grid.plot(ax=ax, color=LAND, edgecolor=GRID, linewidth=0.010, alpha=0.66, zorder=1, rasterized=True)
    grid[grid["grid_emergence_type_v51"].eq("suppressed_emergence_frontier")].plot(ax=ax, color=RUST, edgecolor="none", alpha=0.52, zorder=4, rasterized=True)
    grid[pd.to_numeric(grid["positive_rule_count"], errors="coerce").fillna(0).gt(0)].plot(ax=ax, color=GREEN, edgecolor="none", alpha=0.72, zorder=5, rasterized=True)
    boundary.boundary.plot(ax=ax, color="#5f5a53", linewidth=0.38, zorder=10)
    for i, row in cases.iterrows():
        focal = grid[grid["grid_id"].eq(row["focal_grid_id"])].iloc[0]
        extent = local_extent(focal.geometry, 3300 if row["case_id"] != "futian_visible_rule_core" else 2600)
        ax.add_patch(Rectangle((extent[0], extent[1]), extent[2] - extent[0], extent[3] - extent[1], fill=False, edgecolor=INK, linewidth=0.62, zorder=20))
        c = focal.geometry.centroid
        ax.text(c.x, c.y, chr(ord("a") + i), ha="center", va="center", fontsize=5.2, fontweight="bold", color="white", bbox={"facecolor": INK, "edgecolor": "white", "linewidth": 0.25, "boxstyle": "circle,pad=0.18"}, zorder=25)
    minx, miny, maxx, maxy = boundary.total_bounds
    ax.set_xlim(minx - 4500, maxx + 4500)
    ax.set_ylim(miny - 4500, maxy + 4500)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.text(0.03, 0.97, "overview | local rule-ecology vignette windows", ha="left", va="top", fontsize=7.2, fontweight="bold", color=INK)
    fig.text(0.03, 0.915, "coral = diagnostic suppressed frontier; green = visible positive-rule grid; boxes locate local evidence cards", ha="left", va="top", fontsize=4.5, color=MUTED)
    save(fig, PANELS / "figX_v30_panel_overview_case_windows", dpi=600, tight=False)


def render_all() -> None:
    data = load_data()
    cases = select_cases(data)
    render_overview(data, cases)
    source_rows = []
    card_files = []
    for i, (_, case) in enumerate(cases.iterrows()):
        letter = chr(ord("a") + i)
        fig, stats = render_card(data, case, letter)
        stem = PANELS / f"figX_v30_card_{letter}_{case['case_id']}"
        save(fig, stem, dpi=600, tight=False)
        source_rows.append(stats)
        card_files.append(stem.with_suffix(".png"))
    pd.DataFrame(source_rows).to_csv(SRC / "v30_vignette_card_sources.csv", index=False, encoding="utf-8-sig")

    composite_files = []
    for page_no, start in enumerate(range(0, len(cases), 6), start=1):
        page_cases = cases.iloc[start : start + 6]
        composite_files.append(
            render_composite_page(
                data,
                page_cases,
                start,
                f"page {page_no}",
                f"figX_v30_local_map_evidence_vignettes_composite_p{page_no}",
            )
        )

    files = [PANELS / "figX_v30_panel_overview_case_windows.png"] + card_files + composite_files
    thumbs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        im.thumbnail((470, 320), Image.LANCZOS)
        can = Image.new("RGB", (500, 370), "white")
        can.paste(im, ((500 - im.width) // 2, 12))
        ImageDraw.Draw(can).text((12, 345), f.name, fill=(35, 35, 35))
        thumbs.append(can)
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 500, rows * 370), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * 500, (i // cols) * 370))
    sheet.save(PREVIEWS / "v30_map_evidence_vignettes_contact_sheet.png", dpi=(180, 180))

    manifest = []
    for f in files:
        manifest.append({"asset": f.relative_to(ROOT).as_posix(), "role": "composite" if f.parent == COMP else "panel"})
    pd.DataFrame(manifest).to_csv(SRC / "v30_map_evidence_vignettes_manifest.csv", index=False)
    print("rendered v30 map evidence vignettes")


if __name__ == "__main__":
    render_all()
