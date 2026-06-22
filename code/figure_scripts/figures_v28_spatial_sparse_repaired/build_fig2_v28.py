#!/usr/bin/env python3
"""Build v28 Fig. 2 with single-panel exports.

Figure claim:
Pet-service ecology is already spatially extensive in Shenzhen, while explicit
pet-friendly rule visibility is sparse; the publication-relevant frontier lies
in rule-liminal host venues embedded in real urban texture.

This script intentionally renders each panel separately before assembling the
composite. Composite-only delivery is incomplete for this project.
"""

from __future__ import annotations

from pathlib import Path
import math
import warnings

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import Point, box


warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "submission_package" / "figures_v28_spatial_sparse_repaired"
COMP = OUT / "composites"
PANELS = OUT / "panels"
SRC = OUT / "source_data"
PREVIEWS = OUT / "previews"

for p in [COMP, PANELS, SRC, PREVIEWS]:
    p.mkdir(parents=True, exist_ok=True)

INK = "#24211f"
TEXT = "#4f4a45"
MUTED = "#7c756d"
LAND = "#f5f2ec"
GRID = "#e6e0d5"
BUILDING = "#d6d0c3"
ROAD = "#b6b2a8"
PED = "#789d9d"
PUBLIC = "#dbe7d8"
BLUE = "#3e7899"
BLUE_DARK = "#22516f"
GREEN = "#4d8a69"
GOLD = "#e4bf68"
RUST = "#d88973"
PURPLE = "#74679f"
SILENT = "#b7b0a5"

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

HOST_LABELS = {
    "residential_property": "Residential",
    "hotel": "Hotels",
    "shopping_mall": "Malls",
    "park_or_recreation": "Parks",
}

LIMINAL_ORDER = [
    "rule_liminal_threshold_core",
    "ecology_ready_rule_liminal_frontier",
    "ecology_present_rule_silent",
    "rule_exposed_low_ecology",
    "restricted_or_negative_pressure",
    "low_signal_unknown",
]

LIMINAL_LABELS = {
    "rule_liminal_threshold_core": "Threshold core",
    "ecology_ready_rule_liminal_frontier": "Ready frontier",
    "ecology_present_rule_silent": "Ecology silent",
    "rule_exposed_low_ecology": "Rule exposed",
    "restricted_or_negative_pressure": "Restricted",
    "low_signal_unknown": "Low signal",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 5.2,
        "axes.titlesize": 5.8,
        "axes.labelsize": 4.9,
        "xtick.labelsize": 4.1,
        "ytick.labelsize": 4.1,
        "legend.fontsize": 4.2,
        "axes.linewidth": 0.42,
        "axes.edgecolor": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

SERVICE_CMAP = LinearSegmentedColormap.from_list(
    "service_ecology", ["#f5f2ec", "#dce9e7", "#abcbd3", "#6fa6bb", "#2f7192", "#164762"]
)
SUPPRESS_CMAP = LinearSegmentedColormap.from_list(
    "suppression", ["#fffaf0", "#f5e9c9", "#e8c76f", "#de987c", "#b96a5d"]
)


def read_gdf(rel: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(ROOT / rel)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    return g.to_crs("EPSG:32649")


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def load_data() -> dict[str, object]:
    hosts = read_csv("data_processed/platform/rule_liminal_host_candidates_2025_v4.csv")
    host_g = gpd.GeoDataFrame(
        hosts,
        geometry=gpd.points_from_xy(hosts["lon_wgs84"], hosts["lat_wgs84"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32649")
    return {
        "grid": read_gdf("data_processed/model/grid_emergence_suppression_indices_500m_2025_v51_sparse_repaired.geojson"),
        "boundary": read_gdf("data_processed/geo/shenzhen_boundary_verified.geojson"),
        "pet": read_gdf("data_processed/platform/pet_service_ecology_points_2025_v3.geojson"),
        "rules": read_gdf("data_processed/platform/rule_semantic_geocoded_points_v1.geojson"),
        "roads": read_gdf("data_processed/osm/shenzhen_osm_roads.gpkg"),
        "ped": read_gdf("data_processed/osm/shenzhen_osm_pedestrian_paths.gpkg"),
        "buildings": read_gdf("data_processed/osm/shenzhen_osm_buildings_3d.gpkg"),
        "public": read_gdf("data_processed/osm/shenzhen_osm_public_space.gpkg"),
        "hosts": hosts,
        "host_g": host_g,
        "matched": read_csv("data_processed/model/final_matched_frontier_comparison_v7.csv"),
    }


def city_bounds(boundary: gpd.GeoDataFrame, pad: float = 4300) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = boundary.total_bounds
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def lonlat(lon: float, lat: float) -> tuple[float, float]:
    p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:32649").iloc[0]
    return float(p.x), float(p.y)


def zoom_extent(lon: float, lat: float, w: float, h: float) -> tuple[float, float, float, float]:
    x, y = lonlat(lon, lat)
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def panel_label(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        0.0,
        1.018,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.055,
        1.024,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
        fontweight="bold",
        color=INK,
    )


def clean_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.38)
    ax.spines["bottom"].set_linewidth(0.38)
    ax.tick_params(length=2.0, width=0.4, color=MUTED, labelcolor=TEXT)
    if grid:
        ax.grid(axis="x", color="#ece8df", lw=0.35)


def save_figure(fig: plt.Figure, path: Path, dpi: int = 600, tight: bool = True) -> None:
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.025} if tight else {}
    fig.savefig(path.with_suffix(".png"), dpi=dpi, **kwargs)
    fig.savefig(path.with_suffix(".pdf"), **kwargs)
    fig.savefig(path.with_suffix(".svg"), **kwargs)
    plt.close(fig)


def draw_city_base(ax: plt.Axes, data: dict[str, object]) -> None:
    grid = data["grid"]
    boundary = data["boundary"]
    grid.plot(ax=ax, color=LAND, edgecolor=GRID, linewidth=0.018, zorder=0, rasterized=True)
    boundary.boundary.plot(ax=ax, color=INK, linewidth=0.54, zorder=20)
    ax.set_xlim(*city_bounds(boundary)[0::2])
    minx, miny, maxx, maxy = city_bounds(boundary)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_axis_off()


def draw_urban_texture(ax: plt.Axes, data: dict[str, object], extent: tuple[float, float, float, float]) -> None:
    minx, miny, maxx, maxy = extent
    bbox = box(minx, miny, maxx, maxy)
    ax.set_facecolor("#fbfaf7")
    layers = [
        ("public", PUBLIC, 0.0, 0.60, 1),
        ("buildings", BUILDING, 0.0, 0.54, 2),
        ("roads", ROAD, 0.24, 0.43, 3),
        ("ped", PED, 0.32, 0.58, 4),
    ]
    for key, color, lw, alpha, z in layers:
        layer = data[key]
        sub = layer[layer.intersects(bbox)]
        if len(sub) == 0:
            continue
        if key in {"roads", "ped"}:
            sub.plot(ax=ax, color=color, linewidth=lw, alpha=alpha, zorder=z)
        else:
            sub.plot(ax=ax, color=color, edgecolor="none", alpha=alpha, zorder=z)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_axis_off()


def point_size(values: pd.Series, min_size: float = 2.0, max_size: float = 18.0) -> np.ndarray:
    v = numeric(values)
    if v.max() <= 0:
        return np.full(len(v), min_size)
    return min_size + (np.sqrt(v / v.max()) * (max_size - min_size))


def render_panel_a(data: dict[str, object], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.65, 3.25))
    grid = data["grid"].copy()
    boundary = data["boundary"]
    pet = data["pet"]
    rules = data["rules"]
    draw_city_base(ax, data)

    grid["service"] = numeric(grid["pet_service_exposure_500m"])
    grid[grid["service"] > 0].plot(
        ax=ax,
        column="service",
        cmap=SERVICE_CMAP,
        linewidth=0,
        alpha=0.96,
        norm=Normalize(vmin=0, vmax=np.nanpercentile(grid["service"], 99)),
        rasterized=True,
        zorder=4,
    )
    pet.plot(ax=ax, color=BLUE_DARK, markersize=1.35, alpha=0.23, linewidth=0, zorder=8)
    if len(rules):
        rules.plot(ax=ax, color=GREEN, markersize=18, alpha=0.86, edgecolor="white", linewidth=0.22, zorder=11)

    zooms = [
        ("Futian-Nanshan", zoom_extent(114.043, 22.535, 17000, 12500), RUST),
        ("Baoan-Longhua", zoom_extent(113.890, 22.610, 18000, 13200), PURPLE),
    ]
    for name, ext, color in zooms:
        minx, miny, maxx, maxy = ext
        ax.add_patch(Rectangle((minx, miny), maxx - minx, maxy - miny, fill=False, ec=color, lw=0.92, zorder=30))
        ax.text(minx, maxy + 1200, name, fontsize=4.4, color=color, fontweight="bold", zorder=31)

    panel_label(ax, "a", "service ecology outpaces visible rules")
    ax.text(
        0.018,
        0.035,
        "500 m cells; blue density = pet-service exposure; green dots = explicit rule records",
        transform=ax.transAxes,
        fontsize=4.0,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    cax = fig.add_axes([0.64, 0.102, 0.22, 0.018])
    sm = mpl.cm.ScalarMappable(cmap=SERVICE_CMAP, norm=Normalize(vmin=0, vmax=np.nanpercentile(grid["service"], 99)))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=3.4, length=1.0, pad=1.2)
    cb.outline.set_linewidth(0.25)
    cb.set_label("service exposure", fontsize=3.9, labelpad=1.2, color=MUTED)

    grid[["grid_id", "district_name", "pet_service_exposure_500m", "positive_rule_count", "suppression_index"]].to_csv(
        SRC / "fig2_panel_a_source.csv", index=False, encoding="utf-8-sig"
    )
    save_figure(fig, path)


def render_zoom_panel(
    data: dict[str, object],
    path: Path,
    letter: str,
    title: str,
    extent: tuple[float, float, float, float],
    border_color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3.0, 2.18))
    grid = data["grid"].copy()
    pet = data["pet"]
    rules = data["rules"]
    hosts = data["host_g"]
    bbox = box(*extent)
    draw_urban_texture(ax, data, extent)

    grid["suppression"] = numeric(grid["suppression_index"])
    gsub = grid[grid.intersects(bbox)]
    gsub[gsub["suppression"] > 0].plot(
        ax=ax,
        column="suppression",
        cmap=SUPPRESS_CMAP,
        linewidth=0,
        alpha=0.50,
        norm=Normalize(vmin=0, vmax=np.nanpercentile(grid["suppression"], 98)),
        zorder=6,
        rasterized=True,
    )
    hsub = hosts[hosts.intersects(bbox)].copy()
    priority = hsub[hsub["liminal_class"].isin(["rule_liminal_threshold_core", "ecology_ready_rule_liminal_frontier"])]
    silent = hsub[hsub["liminal_class"].eq("ecology_present_rule_silent")]
    if len(silent):
        silent.sample(min(len(silent), 650), random_state=12).plot(
            ax=ax, color=SILENT, markersize=2.6, alpha=0.18, linewidth=0, zorder=8
        )
    if len(priority):
        priority.plot(ax=ax, color=RUST, markersize=5.6, alpha=0.42, edgecolor="white", linewidth=0.32, zorder=10)
    psub = pet[pet.intersects(bbox)]
    if len(psub):
        psub.plot(ax=ax, color=BLUE_DARK, markersize=3.4, alpha=0.28, linewidth=0, zorder=9)
    rsub = rules[rules.intersects(bbox)]
    if len(rsub):
        rsub.plot(ax=ax, color=GREEN, markersize=22, alpha=0.94, edgecolor="white", linewidth=0.22, zorder=12)
    ax.add_patch(Rectangle((extent[0], extent[1]), extent[2] - extent[0], extent[3] - extent[1], fill=False, ec=border_color, lw=0.85, zorder=30))
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_axis_off()
    ax.set_aspect("equal")
    panel_label(ax, letter, title)
    ax.text(
        0.01,
        0.035,
        f"hosts {len(hsub):,} | priority frontier {len(priority):,} | rules {len(rsub):,}",
        transform=ax.transAxes,
        fontsize=3.8,
        color=MUTED,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.45},
    )
    pd.DataFrame(
        {
            "zoom": [title],
            "host_candidates": [len(hsub)],
            "priority_frontier_hosts": [len(priority)],
            "silent_hosts": [len(silent)],
            "pet_service_points": [len(psub)],
            "rule_points": [len(rsub)],
        }
    ).to_csv(SRC / f"fig2_panel_{letter}_source.csv", index=False, encoding="utf-8-sig")
    save_figure(fig, path)


def render_panel_d(data: dict[str, object], path: Path) -> None:
    hosts = data["hosts"].copy()
    counts = hosts.groupby(["primary_venue_type", "liminal_class"]).size().reset_index(name="n")
    pivot = counts.pivot_table(index="primary_venue_type", columns="liminal_class", values="n", fill_value=0)
    pivot = pivot.reindex(index=["residential_property", "hotel", "shopping_mall", "park_or_recreation"]).fillna(0)
    pivot = pivot.reindex(columns=LIMINAL_ORDER).fillna(0)
    share = pivot.div(pivot.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(3.0, 1.95))
    colors = {
        "rule_liminal_threshold_core": RUST,
        "ecology_ready_rule_liminal_frontier": GOLD,
        "ecology_present_rule_silent": "#c8c0b5",
        "rule_exposed_low_ecology": "#9db7be",
        "restricted_or_negative_pressure": "#7d5550",
        "low_signal_unknown": "#dedbd2",
    }
    y = np.arange(len(share.index))
    left = np.zeros(len(share))
    for col in LIMINAL_ORDER:
        vals = share[col].values
        ax.barh(y, vals, left=left, color=colors[col], height=0.62, edgecolor="white", linewidth=0.22, label=LIMINAL_LABELS[col])
        left += vals
    labels = [HOST_LABELS.get(i, i) for i in share.index]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50", "100"])
    ax.set_xlabel("share of host candidates (%)")
    ax.invert_yaxis()
    clean_axis(ax)
    panel_label(ax, "d", "where rule-liminal venues concentrate")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.37),
        ncol=2,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.8,
        handletextpad=0.35,
        fontsize=3.65,
    )
    out = share.reset_index()
    out.to_csv(SRC / "fig2_panel_d_source.csv", index=False, encoding="utf-8-sig")
    save_figure(fig, path)


def render_panel_e(data: dict[str, object], path: Path) -> None:
    grid = data["grid"].copy()
    grid["district"] = grid["district_name"].map(DISTRICT_EN).fillna(grid["district_name"])
    district = (
        grid.groupby("district", as_index=False)
        .agg(
            service=("pet_service_count_2025", "sum"),
            positive_rules=("positive_rule_count", "sum"),
            frontier=("liminal_frontier_hosts", "sum"),
            silent=("ecology_silent_hosts", "sum"),
            suppression=("suppression_index", "mean"),
        )
        .sort_values("frontier", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(3.0, 1.95))
    y = np.arange(len(district))
    scale = district["service"].max() / max(district["frontier"].max(), 1)
    ax.barh(y, district["service"], height=0.55, color=BLUE, alpha=0.72, label="pet-service POI")
    ax.scatter(
        district["frontier"] * scale,
        y,
        s=point_size(district["silent"], 10, 54),
        color=GOLD,
        edgecolor=INK,
        linewidth=0.22,
        alpha=0.86,
        label="frontier hosts",
        zorder=5,
    )
    ax.scatter(
        district["positive_rules"] * max(district["service"].max() / max(district["positive_rules"].max(), 1), 1),
        y,
        s=18,
        marker="D",
        color=GREEN,
        edgecolor="white",
        linewidth=0.18,
        alpha=0.95,
        label="visible rules",
        zorder=6,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(district["district"])
    ax.set_xlabel("service count; scaled rule/frontier overlays")
    clean_axis(ax)
    panel_label(ax, "e", "district-scale readiness gap")
    ax.legend(loc="lower right", frameon=False, fontsize=3.75, handletextpad=0.35)
    district.to_csv(SRC / "fig2_panel_e_source.csv", index=False, encoding="utf-8-sig")
    save_figure(fig, path)


def draw_panel_a_comp(ax: plt.Axes, fig: plt.Figure, data: dict[str, object]) -> None:
    grid = data["grid"].copy()
    pet = data["pet"]
    rules = data["rules"]
    draw_city_base(ax, data)
    grid["service"] = numeric(grid["pet_service_exposure_500m"])
    grid[grid["service"] > 0].plot(
        ax=ax,
        column="service",
        cmap=SERVICE_CMAP,
        linewidth=0,
        alpha=0.96,
        norm=Normalize(vmin=0, vmax=np.nanpercentile(grid["service"], 99)),
        rasterized=True,
        zorder=4,
    )
    pet.plot(ax=ax, color=BLUE_DARK, markersize=1.2, alpha=0.23, linewidth=0, zorder=8)
    rules.plot(ax=ax, color=GREEN, markersize=15, alpha=0.86, edgecolor="white", linewidth=0.20, zorder=11)
    for name, ext, color in [
        ("Futian-Nanshan", zoom_extent(114.043, 22.535, 17000, 12500), RUST),
        ("Baoan-Longhua", zoom_extent(113.890, 22.610, 18000, 13200), PURPLE),
    ]:
        minx, miny, maxx, maxy = ext
        ax.add_patch(Rectangle((minx, miny), maxx - minx, maxy - miny, fill=False, ec=color, lw=0.82, zorder=30))
        ax.text(minx, maxy + 1200, name, fontsize=4.0, color=color, fontweight="bold", zorder=31)
    panel_label(ax, "a", "service ecology outpaces visible rules")
    ax.text(
        0.012,
        0.018,
        "500 m cells; blue density = pet-service exposure; green dots = explicit rule records",
        transform=ax.transAxes,
        fontsize=3.65,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    box = ax.get_position()
    cax = fig.add_axes([box.x0 + box.width * 0.68, box.y0 + box.height * 0.015, box.width * 0.22, 0.009])
    sm = mpl.cm.ScalarMappable(cmap=SERVICE_CMAP, norm=Normalize(vmin=0, vmax=np.nanpercentile(grid["service"], 99)))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=3.1, length=0.8, pad=0.8)
    cb.outline.set_linewidth(0.22)
    cb.set_label("service exposure", fontsize=3.3, labelpad=0.6, color=MUTED)


def draw_zoom_comp(
    ax: plt.Axes,
    data: dict[str, object],
    letter: str,
    title: str,
    extent: tuple[float, float, float, float],
    border_color: str,
) -> None:
    grid = data["grid"].copy()
    pet = data["pet"]
    rules = data["rules"]
    hosts = data["host_g"]
    bbox = box(*extent)
    draw_urban_texture(ax, data, extent)
    grid["suppression"] = numeric(grid["suppression_index"])
    gsub = grid[grid.intersects(bbox)]
    gsub[gsub["suppression"] > 0].plot(
        ax=ax,
        column="suppression",
        cmap=SUPPRESS_CMAP,
        linewidth=0,
        alpha=0.48,
        norm=Normalize(vmin=0, vmax=np.nanpercentile(grid["suppression"], 98)),
        zorder=6,
        rasterized=True,
    )
    hsub = hosts[hosts.intersects(bbox)].copy()
    priority = hsub[hsub["liminal_class"].isin(["rule_liminal_threshold_core", "ecology_ready_rule_liminal_frontier"])]
    silent = hsub[hsub["liminal_class"].eq("ecology_present_rule_silent")]
    if len(silent):
        silent.sample(min(len(silent), 650), random_state=12).plot(ax=ax, color=SILENT, markersize=2.1, alpha=0.17, linewidth=0, zorder=8)
    if len(priority):
        priority.plot(ax=ax, color=RUST, markersize=4.6, alpha=0.42, edgecolor="white", linewidth=0.30, zorder=10)
    psub = pet[pet.intersects(bbox)]
    if len(psub):
        psub.plot(ax=ax, color=BLUE_DARK, markersize=2.8, alpha=0.26, linewidth=0, zorder=9)
    rsub = rules[rules.intersects(bbox)]
    if len(rsub):
        rsub.plot(ax=ax, color=GREEN, markersize=18, alpha=0.94, edgecolor="white", linewidth=0.20, zorder=12)
    ax.add_patch(Rectangle((extent[0], extent[1]), extent[2] - extent[0], extent[3] - extent[1], fill=False, ec=border_color, lw=0.70, zorder=30))
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_axis_off()
    ax.set_aspect("equal")
    panel_label(ax, letter, title)
    ax.text(
        0.01,
        0.035,
        f"hosts {len(hsub):,} | priority frontier {len(priority):,} | rules {len(rsub):,}",
        transform=ax.transAxes,
        fontsize=3.55,
        color=MUTED,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
    )


def draw_panel_d_comp(ax: plt.Axes, data: dict[str, object]) -> None:
    hosts = data["hosts"].copy()
    counts = hosts.groupby(["primary_venue_type", "liminal_class"]).size().reset_index(name="n")
    pivot = counts.pivot_table(index="primary_venue_type", columns="liminal_class", values="n", fill_value=0)
    pivot = pivot.reindex(index=["residential_property", "hotel", "shopping_mall", "park_or_recreation"]).fillna(0)
    pivot = pivot.reindex(columns=LIMINAL_ORDER).fillna(0)
    share = pivot.div(pivot.sum(axis=1), axis=0)
    colors = {
        "rule_liminal_threshold_core": RUST,
        "ecology_ready_rule_liminal_frontier": GOLD,
        "ecology_present_rule_silent": "#c8c0b5",
        "rule_exposed_low_ecology": "#9db7be",
        "restricted_or_negative_pressure": "#7d5550",
        "low_signal_unknown": "#dedbd2",
    }
    y = np.arange(len(share.index))
    left = np.zeros(len(share))
    for col in LIMINAL_ORDER:
        vals = share[col].values
        ax.barh(y, vals, left=left, color=colors[col], height=0.58, edgecolor="white", linewidth=0.22, label=LIMINAL_LABELS[col])
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([HOST_LABELS.get(i, i) for i in share.index])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50", "100"])
    ax.set_xlabel("share of host candidates (%)")
    ax.invert_yaxis()
    clean_axis(ax)
    panel_label(ax, "d", "where rule-liminal venues concentrate")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=3, frameon=False, handlelength=1.0, columnspacing=0.55, handletextpad=0.3, fontsize=3.25)


def draw_panel_e_comp(ax: plt.Axes, data: dict[str, object]) -> None:
    grid = data["grid"].copy()
    grid["district"] = grid["district_name"].map(DISTRICT_EN).fillna(grid["district_name"])
    district = (
        grid.groupby("district", as_index=False)
        .agg(
            service=("pet_service_count_2025", "sum"),
            positive_rules=("positive_rule_count", "sum"),
            frontier=("liminal_frontier_hosts", "sum"),
            silent=("ecology_silent_hosts", "sum"),
        )
        .sort_values("frontier", ascending=True)
    )
    y = np.arange(len(district))
    scale = district["service"].max() / max(district["frontier"].max(), 1)
    ax.barh(y, district["service"], height=0.52, color=BLUE, alpha=0.72, label="pet-service POI")
    ax.scatter(district["frontier"] * scale, y, s=point_size(district["silent"], 8, 42), color=GOLD, edgecolor="white", linewidth=0.52, alpha=0.86, label="frontier hosts", zorder=5)
    ax.scatter(
        district["positive_rules"] * max(district["service"].max() / max(district["positive_rules"].max(), 1), 1),
        y,
        s=15,
        marker="D",
        color=GREEN,
        edgecolor="white",
        linewidth=0.16,
        alpha=0.95,
        label="visible rules",
        zorder=6,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(district["district"])
    ax.set_xlabel("service count; scaled rule/frontier overlays")
    clean_axis(ax)
    panel_label(ax, "e", "district-scale readiness gap")
    ax.legend(loc="lower right", frameon=False, fontsize=3.25, handletextpad=0.3)


def assemble_composite(panel_paths: dict[str, Path], data: dict[str, object]) -> None:
    fig = plt.figure(figsize=(7.25, 5.35))
    gs = GridSpec(
        3,
        6,
        figure=fig,
        width_ratios=[1.04, 1.04, 0.52, 0.28, 0.96, 1.12],
        height_ratios=[1.08, 1.00, 1.08],
        wspace=0.16,
        hspace=0.13,
    )
    slots = {
        "a": gs[0:2, 0:3],
        "b": gs[0, 3:6],
        "c": gs[1, 3:6],
        "d": gs[2, 0:3],
        "e": gs[2, 4:6],
    }
    draw_panel_a_comp(fig.add_subplot(slots["a"]), fig, data)
    draw_zoom_comp(fig.add_subplot(slots["b"]), data, "b", "Futian-Nanshan rule-liminal texture", zoom_extent(114.043, 22.535, 17000, 12500), RUST)
    draw_zoom_comp(fig.add_subplot(slots["c"]), data, "c", "Baoan-Longhua silent frontier texture", zoom_extent(113.890, 22.610, 18000, 13200), PURPLE)
    draw_panel_d_comp(fig.add_subplot(slots["d"]), data)
    draw_panel_e_comp(fig.add_subplot(slots["e"]), data)
    fig.text(
        0.018,
        0.985,
        "Fig. 2 | Pet-service ecology is spatially mature; rule visibility remains selective",
        ha="left",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.018,
        0.955,
        "City-scale exposure, urban-texture windows and host-candidate composition jointly locate the rule-liminal frontier.",
        ha="left",
        va="top",
        fontsize=5.1,
        color=TEXT,
    )
    save_figure(fig, COMP / "fig2_composite_v28", dpi=520, tight=True)


def make_contact_sheet() -> None:
    files = sorted(PANELS.glob("fig2_panel_*.png")) + [COMP / "fig2_composite_v28.png"]
    thumbs = []
    for f in files:
        img = Image.open(f).convert("RGB")
        img.thumbnail((560, 420), Image.LANCZOS)
        canvas = Image.new("RGB", (580, 455), "white")
        canvas.paste(img, ((580 - img.width) // 2, 18))
        d = ImageDraw.Draw(canvas)
        d.text((12, 430), f.name, fill=(40, 40, 40))
        thumbs.append(canvas)
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 580, rows * 455), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * 580, (i // cols) * 455))
    sheet.save(PREVIEWS / "fig2_v28_contact_sheet.png", dpi=(180, 180))


def main() -> None:
    data = load_data()
    panel_paths = {
        "a": PANELS / "fig2_panel_a_city_service_rule_v28",
        "b": PANELS / "fig2_panel_b_futian_nanshan_texture_v28",
        "c": PANELS / "fig2_panel_c_baoan_longhua_texture_v28",
        "d": PANELS / "fig2_panel_d_host_liminal_composition_v28",
        "e": PANELS / "fig2_panel_e_district_readiness_gap_v28",
    }
    render_panel_a(data, panel_paths["a"])
    render_zoom_panel(
        data,
        panel_paths["b"],
        "b",
        "Futian-Nanshan rule-liminal texture",
        zoom_extent(114.043, 22.535, 17000, 12500),
        RUST,
    )
    render_zoom_panel(
        data,
        panel_paths["c"],
        "c",
        "Baoan-Longhua silent frontier texture",
        zoom_extent(113.890, 22.610, 18000, 13200),
        PURPLE,
    )
    render_panel_d(data, panel_paths["d"])
    render_panel_e(data, panel_paths["e"])
    assemble_composite(panel_paths, data)
    make_contact_sheet()
    audit = pd.DataFrame(
        [
            {"asset": p.with_suffix(ext).relative_to(ROOT).as_posix(), "role": key}
            for key, p in panel_paths.items()
            for ext in [".png", ".pdf", ".svg"]
        ]
        + [
            {"asset": (COMP / f"fig2_composite_v28{ext}").relative_to(ROOT).as_posix(), "role": "composite"}
            for ext in [".png", ".pdf", ".svg"]
        ]
    )
    audit.to_csv(SRC / "fig2_v28_export_manifest.csv", index=False, encoding="utf-8-sig")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
