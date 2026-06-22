#!/usr/bin/env python3
"""Build v14 Fig. 3: rule-liminal venues as Shenzhen's institutional middle terrain.

Figure claim:
Rule-liminal venues are candidate transition sites, not confirmed pet-friendly
venues. They form a measurable middle terrain where service ecology, host
capacity and partial rule signals make rule publicisation plausible but
unverified.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys
import warnings

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import box

warnings.filterwarnings("ignore", category=UserWarning)

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from build_fig2_v28 import (  # noqa: E402
    BLUE_DARK,
    BUILDING,
    COMP,
    DISTRICT_EN,
    GOLD,
    GREEN,
    GRID,
    INK,
    LAND,
    MUTED,
    PANELS,
    PED,
    PREVIEWS,
    PUBLIC,
    ROAD,
    ROOT,
    RUST,
    SRC,
    TEXT,
    clean_axis,
    numeric,
    panel_label,
    read_csv,
    read_gdf,
    save_figure,
)
from build_fig2_district_atlas_v28 import draw_texture_light  # noqa: E402


OUT_STEM = "fig3_rule_liminal_venues_v28"
PROJECT_CRS = "EPSG:32649"
LIMINAL_GOLD = "#e4bf68"
LIMINAL_DARK = "#d88973"
LIMINAL_BLUE = "#9ab7bd"
RESTRICTED = "#9d6d67"
LOW_SIGNAL = "#dedbd2"
FINGER_CMAP = LinearSegmentedColormap.from_list("fig3_fingerprint", ["#fffaf0", "#f5e9c9", "#e4bf68", "#d88973"])
LIMINAL_CMAP = LinearSegmentedColormap.from_list("fig3_liminal", ["#fffaf0", "#f5e9c9", "#e8c76f", "#de987c", "#b96a5d"])

DISTRICT_ORDER = ["福田区", "龙华区", "宝安区", "龙岗区", "南山区", "罗湖区", "光明区", "坪山区", "盐田区"]
HOST_ORDER = ["residential_property", "hotel", "shopping_mall", "park_or_recreation"]
HOST_LABELS = {
    "residential_property": "Residential",
    "hotel": "Hotels",
    "shopping_mall": "Malls",
    "park_or_recreation": "Parks",
}
CLASS_ORDER = [
    "rule_liminal_threshold_core",
    "ecology_ready_rule_liminal_frontier",
    "ecology_present_rule_silent",
    "rule_exposed_low_ecology",
    "restricted_or_negative_pressure",
    "low_signal_unknown",
]
CLASS_LABELS = {
    "rule_liminal_threshold_core": "Threshold core",
    "ecology_ready_rule_liminal_frontier": "Ready frontier",
    "ecology_present_rule_silent": "Ecology silent",
    "rule_exposed_low_ecology": "Rule exposed",
    "restricted_or_negative_pressure": "Restricted",
    "low_signal_unknown": "Low signal",
}
CLASS_COLORS = {
    "rule_liminal_threshold_core": RUST,
    "ecology_ready_rule_liminal_frontier": LIMINAL_GOLD,
    "ecology_present_rule_silent": "#c8c0b5",
    "rule_exposed_low_ecology": LIMINAL_BLUE,
    "restricted_or_negative_pressure": RESTRICTED,
    "low_signal_unknown": LOW_SIGNAL,
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
        "legend.fontsize": 4.0,
        "axes.linewidth": 0.42,
        "axes.edgecolor": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def panel_path(letter: str, name: str) -> Path:
    return PANELS / f"{OUT_STEM}_panel_{letter}_{name}"


def load() -> dict[str, object]:
    hosts = read_csv("data_processed/platform/rule_liminal_host_candidates_2025_v4.csv")
    host_g = gpd.GeoDataFrame(
        hosts,
        geometry=gpd.points_from_xy(hosts["lon_wgs84"], hosts["lat_wgs84"]),
        crs="EPSG:4326",
    ).to_crs(PROJECT_CRS)
    return {
        "grid": read_gdf("data_processed/model/grid_rule_liminality_500m_2025_v4.geojson").to_crs(PROJECT_CRS),
        "boundary": read_gdf("data_processed/geo/shenzhen_boundary_verified.geojson").to_crs(PROJECT_CRS),
        "rules": read_gdf("data_processed/platform/rule_semantic_geocoded_points_v1.geojson").to_crs(PROJECT_CRS),
        "rule_records": read_csv("data_processed/platform/rule_semantic_records_v1.csv"),
        "rule_ledger": read_csv("data_processed/platform/rule_source_ledger_v1.csv"),
        "pet": read_gdf("data_processed/platform/pet_service_ecology_points_2025_v3.geojson").to_crs(PROJECT_CRS),
        "roads": read_gdf("data_processed/osm/shenzhen_osm_roads.gpkg").to_crs(PROJECT_CRS),
        "ped": read_gdf("data_processed/osm/shenzhen_osm_pedestrian_paths.gpkg").to_crs(PROJECT_CRS),
        "buildings": read_gdf("data_processed/osm/shenzhen_osm_buildings_3d.gpkg").to_crs(PROJECT_CRS),
        "public": read_gdf("data_processed/osm/shenzhen_osm_public_space.gpkg").to_crs(PROJECT_CRS),
        "hosts": hosts,
        "host_g": host_g,
        "verification": read_csv("data_processed/platform/rule_liminal_verification_queue_2025_v4.csv"),
    }


def city_extent(boundary: gpd.GeoDataFrame, pad: float = 4200) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = boundary.total_bounds
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def fixed_panel() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.15, top=0.82)
    return fig, ax


def light_frame(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#d8d3ca")
        spine.set_linewidth(0.42)


def draw_fig3_legend(ax: plt.Axes) -> None:
    ax.set_axis_off()
    items = [
        ("patch", "#dbe7d8", "public space"),
        ("patch", BUILDING, "building texture"),
        ("line", ROAD, "road network"),
        ("line", PED, "pedestrian path"),
        ("patch", LIMINAL_GOLD, "rule-liminal grid"),
        ("point", LIMINAL_DARK, "priority host candidate"),
        ("point", BLUE_DARK, "pet-service point"),
        ("point", GREEN, "traceable rule evidence"),
    ]
    xs = [0.02, 0.265, 0.505, 0.745]
    ys = [0.68, 0.28]
    for idx, (kind, color, label) in enumerate(items):
        x, y = xs[idx % 4], ys[idx // 4]
        if kind == "patch":
            ax.add_patch(Rectangle((x, y - 0.095), 0.034, 0.19, transform=ax.transAxes, facecolor=color, edgecolor="#d0cbc2", linewidth=0.25, alpha=0.78))
        elif kind == "line":
            ax.plot([x, x + 0.038], [y, y], transform=ax.transAxes, color=color, linewidth=1.35, alpha=0.70)
        else:
            ax.scatter([x + 0.019], [y], transform=ax.transAxes, s=21, color=color, edgecolor="white", linewidth=0.30, alpha=0.92)
        ax.text(x + 0.050, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=4.1, color=MUTED)


def draw_hero_map(ax: plt.Axes, data: dict[str, object], letter: str = "a") -> pd.DataFrame:
    grid = data["grid"].copy()
    boundary = data["boundary"]
    rules = data["rules"]
    hosts = data["host_g"]
    priority = hosts[hosts["liminal_class"].isin(["rule_liminal_threshold_core", "ecology_ready_rule_liminal_frontier"])].copy()
    grid["liminal_surface"] = numeric(grid["liminal_or_frontier_hosts"]) + 0.35 * numeric(grid["ecology_silent_hosts"])
    grid.plot(ax=ax, color=LAND, edgecolor=GRID, linewidth=0.006, alpha=0.48, zorder=0, rasterized=True)
    surf = grid[grid["liminal_surface"] > 0].copy()
    surf.plot(
        ax=ax,
        column="liminal_surface",
        cmap=LIMINAL_CMAP,
        linewidth=0,
        alpha=0.75,
        norm=Normalize(vmin=0, vmax=max(np.nanpercentile(surf["liminal_surface"], 98), 1.0)),
        zorder=7,
        rasterized=True,
    )
    boundary.boundary.plot(ax=ax, color="#5d5750", linewidth=0.38, alpha=0.70, zorder=20)
    if len(priority):
        p = priority.sample(min(len(priority), 2600), random_state=23)
        p.plot(ax=ax, color=LIMINAL_DARK, markersize=2.2, alpha=0.28, linewidth=0, zorder=14)
    positive = rules[rules["access_semantic_score"].fillna(0) > 0.55]
    if len(positive):
        positive.plot(ax=ax, color=GREEN, markersize=12, alpha=0.88, edgecolor="white", linewidth=0.12, zorder=30)
    minx, miny, maxx, maxy = city_extent(boundary)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    light_frame(ax)
    ax.text(0.0, 1.02, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=INK)
    ax.text(0.055, 1.025, "citywide rule-liminal middle terrain", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, fontweight="bold", color=INK)
    ax.text(
        0.012,
        0.025,
        f"{len(hosts):,} host candidates; {len(priority):,} priority candidates; green points are traceable positive rule evidence",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=3.65,
        color=MUTED,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.32},
        zorder=40,
    )
    source = grid[["grid_id", "district_name", "liminal_surface", "liminal_or_frontier_hosts", "ecology_silent_hosts", "positive_rule_count"]].copy()
    source.to_csv(SRC / f"{OUT_STEM}_panel_a_source.csv", index=False, encoding="utf-8-sig")
    return source


def district_extent_for_grid(grid: gpd.GeoDataFrame, district: str, pad_ratio: float = 0.08, target_aspect: float = 1.18) -> tuple[float, float, float, float]:
    dg = grid[grid["district_name"].eq(district)]
    minx, miny, maxx, maxy = dg.total_bounds
    w, h = maxx - minx, maxy - miny
    pad = max(w, h) * pad_ratio
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
    w, h = maxx - minx, maxy - miny
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    if w / h > target_aspect:
        new_h = w / target_aspect
        miny, maxy = cy - new_h / 2, cy + new_h / 2
    else:
        new_w = h * target_aspect
        minx, maxx = cx - new_w / 2, cx + new_w / 2
    return minx, miny, maxx, maxy


def draw_local_zoom(ax: plt.Axes, data: dict[str, object], district: str, letter: str = "b") -> pd.DataFrame:
    grid = data["grid"].copy()
    hosts = data["host_g"]
    pet = data["pet"]
    rules = data["rules"]
    boundary = data["boundary"]
    ext = district_extent_for_grid(grid, district, 0.09)
    draw_texture_light(ax, data, ext)
    bbox = box(*ext)
    grid[grid.intersects(bbox)].plot(ax=ax, color=LAND, edgecolor=GRID, linewidth=0.006, alpha=0.20, zorder=5, rasterized=True)
    dg = grid[grid["district_name"].eq(district)].copy()
    dg["liminal_surface"] = numeric(dg["liminal_or_frontier_hosts"]) + 0.35 * numeric(dg["ecology_silent_hosts"])
    dg[dg["liminal_surface"] > 0].plot(
        ax=ax,
        column="liminal_surface",
        cmap=LIMINAL_CMAP,
        linewidth=0,
        alpha=0.46,
        norm=Normalize(vmin=0, vmax=max(np.nanpercentile(grid["liminal_or_frontier_hosts"] + 0.35 * grid["ecology_silent_hosts"], 98), 1.0)),
        zorder=7,
        rasterized=True,
    )
    boundary.boundary.plot(ax=ax, color="#aaa49b", linewidth=0.12, alpha=0.38, zorder=12)
    dg.dissolve().boundary.plot(ax=ax, color="#5d5750", linewidth=0.30, alpha=0.70, zorder=30)
    h = hosts[hosts["district_name"].eq(district)].copy()
    silent = h[h["liminal_class"].eq("ecology_present_rule_silent")]
    priority = h[h["liminal_class"].isin(["rule_liminal_threshold_core", "ecology_ready_rule_liminal_frontier"])]
    if len(silent):
        silent.sample(min(len(silent), 900), random_state=31).plot(ax=ax, color="#b7b0a5", markersize=1.7, alpha=0.14, linewidth=0, zorder=13)
    if len(priority):
        priority.plot(ax=ax, color=LIMINAL_DARK, markersize=4.5, alpha=0.36, edgecolor="white", linewidth=0.30, zorder=15)
    psub = pet[pet["district_name"].eq(district)]
    if len(psub):
        psub.plot(ax=ax, color=BLUE_DARK, markersize=3.2, alpha=0.20, linewidth=0, zorder=16)
    rsub = rules[rules["grid_id"].isin(dg["grid_id"])]
    if len(rsub):
        rsub.plot(ax=ax, color=GREEN, markersize=22, alpha=0.88, edgecolor="white", linewidth=0.20, zorder=18)
    ax.set_xlim(ext[0], ext[2])
    ax.set_ylim(ext[1], ext[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    light_frame(ax)
    name = DISTRICT_EN.get(district, district)
    ax.text(0.0, 1.018, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=INK)
    ax.text(0.080, 1.024, f"{name}: local rule-liminal texture", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, fontweight="bold", color=INK)
    stats = pd.DataFrame(
        [
            {
                "district_name": district,
                "district_en": name,
                "host_candidates": len(h),
                "priority_candidates": len(priority),
                "silent_hosts": len(silent),
                "rule_records": len(rsub),
                "pet_services": len(psub),
            }
        ]
    )
    ax.text(
        0.012,
        0.025,
        f"hosts {len(h):,} | priority {len(priority):,} | rules {len(rsub):,}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=3.7,
        color=MUTED,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 0.32},
        zorder=40,
    )
    stats.to_csv(SRC / f"{OUT_STEM}_panel_b_source.csv", index=False, encoding="utf-8-sig")
    return stats


def draw_host_type_distribution(ax: plt.Axes, data: dict[str, object], letter: str = "c") -> pd.DataFrame:
    """Dense host-type by liminal-class evidence matrix using all candidates."""
    hosts = data["hosts"].copy()
    hosts = hosts[hosts["primary_venue_type"].isin(HOST_ORDER)].copy()
    counts = (
        hosts.groupby(["primary_venue_type", "liminal_class"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    mat = (
        counts.pivot_table(index="primary_venue_type", columns="liminal_class", values="n", fill_value=0)
        .reindex(index=HOST_ORDER, columns=CLASS_ORDER, fill_value=0)
        .fillna(0)
        .astype(int)
    )
    share = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    scaled = np.log1p(mat.values) / max(np.log1p(mat.values).max(), 1e-9)
    ax.imshow(scaled, cmap=FINGER_CMAP, aspect="auto", vmin=0, vmax=1)
    totals = mat.sum(axis=1)
    ax.set_yticks(np.arange(len(HOST_ORDER)))
    ax.set_yticklabels([f"{HOST_LABELS[h]}\nn={totals.loc[h]:,}" for h in HOST_ORDER])
    ax.set_xticks(np.arange(len(CLASS_ORDER)))
    ax.set_xticklabels([CLASS_LABELS[c].replace(" ", "\n") for c in CLASS_ORDER], rotation=0)
    ax.tick_params(length=0, axis="x", labeltop=False, labelbottom=True, pad=2, labelsize=3.10)
    ax.tick_params(length=0, axis="y", labelsize=3.65)
    for i, htype in enumerate(HOST_ORDER):
        for j, cls in enumerate(CLASS_ORDER):
            n = int(mat.loc[htype, cls])
            pct = share.loc[htype, cls] * 100
            if n >= 10000:
                label = f"{n/1000:.1f}k\n{pct:.0f}%"
            else:
                label = f"{n:,}\n{pct:.0f}%"
            ax.text(j, i, label, ha="center", va="center", fontsize=3.05, color=INK if scaled[i, j] < 0.70 else "white", linespacing=0.82)
    ax.set_xticks(np.arange(-0.5, len(CLASS_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(HOST_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.60)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, letter, "57,994 candidates: host type by liminal state")
    ax.text(1.0, -0.22, "cells show raw count and within-host-type share; shade uses log-scaled counts", transform=ax.transAxes, ha="right", va="top", fontsize=3.30, color=MUTED)
    out = mat.reset_index().rename(columns={"primary_venue_type": "host_type"})
    for cls in CLASS_ORDER:
        out[f"share_{cls}"] = out[cls] / out[CLASS_ORDER].sum(axis=1)
    out.to_csv(SRC / f"{OUT_STEM}_panel_c_source.csv", index=False, encoding="utf-8-sig")
    return out

def draw_class_composition(
    ax: plt.Axes,
    data: dict[str, object],
    letter: str = "d",
    legend_anchor: tuple[float, float] = (0.5, -0.38),
) -> pd.DataFrame:
    hosts = data["hosts"].copy()
    counts = hosts.groupby(["primary_venue_type", "liminal_class"]).size().reset_index(name="n")
    pivot = counts.pivot_table(index="primary_venue_type", columns="liminal_class", values="n", fill_value=0).reindex(index=HOST_ORDER).fillna(0)
    pivot = pivot.reindex(columns=CLASS_ORDER).fillna(0)
    share = pivot.div(pivot.sum(axis=1), axis=0)
    y = np.arange(len(share.index))
    left = np.zeros(len(share))
    for col in CLASS_ORDER:
        vals = share[col].values
        ax.barh(y, vals, left=left, color=CLASS_COLORS[col], height=0.58, edgecolor="white", linewidth=0.22, label=CLASS_LABELS[col])
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([HOST_LABELS.get(i, i) for i in share.index])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50", "100"])
    ax.set_xlabel("share of host candidates (%)")
    ax.invert_yaxis()
    clean_axis(ax)
    panel_label(ax, letter, "candidate classes are not access claims")
    ax.legend(loc="lower center", bbox_to_anchor=legend_anchor, ncol=3, frameon=False, handlelength=1.0, columnspacing=0.55, handletextpad=0.3, fontsize=3.25)
    share.reset_index().to_csv(SRC / f"{OUT_STEM}_panel_d_source.csv", index=False, encoding="utf-8-sig")
    return share.reset_index()


def draw_rule_semantic_matrix(ax: plt.Axes, data: dict[str, object], letter: str = "e") -> pd.DataFrame:
    records = data["rule_records"].copy()
    ledger = data["rule_ledger"].copy()
    rules = records.merge(ledger[["source_id", "source_class", "publication_use_status"]], on="source_id", how="left")

    def family(cls: str) -> str:
        cls = str(cls)
        if cls.startswith("A1") or cls.startswith("A2") or cls.startswith("A5"):
            return "public/open\nacceptance"
        if cls.startswith("A3") or cls.startswith("A4"):
            return "commercial\npositive"
        if cls.startswith("B"):
            return "conditional/\nzoned"
        if cls.startswith("C1") or cls.startswith("C2"):
            return "restriction/\nstandard"
        if cls.startswith("C3"):
            return "ambiguous/\nunresolved"
        if cls.startswith("D"):
            return "governance\nsupport"
        return "other"

    def source_group(sc: str) -> str:
        sc = str(sc)
        if sc.startswith("A_"):
            return "official\nprimary"
        if sc.startswith("B_"):
            return "operator\nprimary"
        if sc.startswith("C_"):
            return "reported\ndetail"
        if sc.startswith("D_") or sc.startswith("E_"):
            return "secondary/\ncontext"
        return "identity/\nexternal"

    family_order = ["public/open\nacceptance", "commercial\npositive", "conditional/\nzoned", "restriction/\nstandard", "ambiguous/\nunresolved", "governance\nsupport", "other"]
    source_order = ["official\nprimary", "operator\nprimary", "reported\ndetail", "secondary/\ncontext", "identity/\nexternal"]
    rules["rule_family"] = rules["rule_granular_class"].map(family)
    rules["source_group"] = rules["source_class"].map(source_group)
    mat = rules.pivot_table(index="rule_family", columns="source_group", values="source_id", aggfunc="count", fill_value=0).reindex(index=family_order, columns=source_order, fill_value=0)
    mat = mat.loc[mat.sum(axis=1) > 0].astype(int)
    scaled = np.log1p(mat.values) / max(np.log1p(mat.values).max(), 1e-9)
    ax.imshow(scaled, cmap=FINGER_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(np.arange(len(source_order)))
    ax.set_xticklabels(source_order)
    ax.tick_params(axis="x", labeltop=False, labelbottom=True, length=0, pad=2, labelsize=3.25)
    ax.tick_params(axis="y", length=0, labelsize=3.55)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = int(mat.iloc[i, j])
            label = str(val) if val else "·"
            ax.text(j, i, label, ha="center", va="center", fontsize=4.05, color=INK if scaled[i, j] < 0.68 else "white")
    row_totals = mat.sum(axis=1)
    for i, total in enumerate(row_totals):
        ax.text(len(source_order) - 0.08, i - 0.38, f"n={int(total)}", ha="right", va="top", fontsize=2.95, color=MUTED)
    ax.set_xticks(np.arange(-0.5, len(source_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mat.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.60)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, letter, "132 traceable rule records by semantic family and source")
    main_model_n = int(rules["publication_use_status"].fillna("").str.contains("main_model").sum())
    ax.text(1.0, -0.24, f"source-grounded records only; main-model/caution ledger rows n={main_model_n}", transform=ax.transAxes, ha="right", va="top", fontsize=3.30, color=MUTED)
    out = mat.reset_index()
    out.to_csv(SRC / f"{OUT_STEM}_panel_e_source.csv", index=False, encoding="utf-8-sig")
    return out

def draw_district_fingerprint(
    ax: plt.Axes,
    data: dict[str, object],
    letter: str = "f",
    title_y: float = 1.13,
) -> pd.DataFrame:
    grid = data["grid"].copy()
    hosts = data["hosts"].copy()
    district = (
        grid.groupby("district_name", as_index=False)
        .agg(
            grids=("grid_id", "count"),
            liminal_grids=("grid_liminal_type", lambda s: int(s.isin(["confirmed_ecology_rule_liminal_core", "high_ecology_liminal_frontier", "latent_ecology_rule_silent_zone"]).sum())),
            liminal_or_frontier_hosts=("liminal_or_frontier_hosts", "sum"),
            ecology_silent_hosts=("ecology_silent_hosts", "sum"),
            p90_liminal_potential=("p90_liminal_potential", "mean"),
            positive_rule_count=("positive_rule_count", "sum"),
        )
    )
    h = hosts.groupby("district_name", as_index=False).size().rename(columns={"size": "host_candidates"})
    district = district.merge(h, on="district_name", how="left").set_index("district_name").reindex(DISTRICT_ORDER).reset_index()
    district["district_en"] = district["district_name"].map(DISTRICT_EN).fillna(district["district_name"])
    cols = [
        ("host_candidates", "host\ncandidates"),
        ("liminal_or_frontier_hosts", "liminal/\nfrontier"),
        ("ecology_silent_hosts", "ecology\nsilent"),
        ("liminal_grids", "liminal\ngrids"),
        ("p90_liminal_potential", "mean p90\npotential"),
        ("positive_rule_count", "visible\nrules"),
    ]
    raw = district[[c for c, _ in cols]].astype(float)
    scaled = raw.copy()
    for c in raw.columns:
        if "potential" in c:
            scaled[c] = raw[c] / max(raw[c].max(), 1e-9)
        else:
            scaled[c] = np.log1p(raw[c]) / max(np.log1p(raw[c]).max(), 1e-9)
    ax.imshow(scaled.values, cmap=FINGER_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(district)))
    ax.set_yticklabels(district["district_en"])
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([label for _, label in cols])
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, length=0, pad=7, labelsize=3.45)
    ax.tick_params(axis="y", length=0)
    for i in range(len(district)):
        for j, (col, _) in enumerate(cols):
            val = raw.iloc[i][col]
            text = f"{val:.2f}" if "potential" in col else f"{int(round(val)):,}"
            ax.text(j, i, text, ha="center", va="center", fontsize=3.15, color=INK if scaled.iloc[i][col] < 0.72 else "white")
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(district), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.55)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.0, title_y, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=INK)
    ax.text(0.075, title_y + 0.01, "district evidence fingerprint", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, fontweight="bold", color=INK)
    ax.text(1.0, -0.19, "cell shade uses within-column log/relative scaling; numbers are raw counts except potential", transform=ax.transAxes, ha="right", va="top", fontsize=3.35, color=MUTED)
    district.to_csv(SRC / f"{OUT_STEM}_panel_f_source.csv", index=False, encoding="utf-8-sig")
    return district


def render_single_panels(data: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.09, top=0.86)
    draw_hero_map(ax, data, "a")
    save_figure(fig, panel_path("a", "city_liminal_surface"), dpi=600, tight=False)

    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.07, top=0.89)
    draw_local_zoom(ax, data, "福田区", "b")
    save_figure(fig, panel_path("b", "futian_local_texture"), dpi=600, tight=False)

    fig, ax = fixed_panel()
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.27, top=0.78)
    draw_host_type_distribution(ax, data, "c")
    save_figure(fig, panel_path("c", "host_type_score_distribution"), dpi=600, tight=False)

    fig, ax = fixed_panel()
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.30, top=0.84)
    draw_class_composition(ax, data, "d")
    save_figure(fig, panel_path("d", "liminal_class_composition"), dpi=600, tight=False)

    fig, ax = fixed_panel()
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.25, top=0.78)
    draw_rule_semantic_matrix(ax, data, "e")
    save_figure(fig, panel_path("e", "rule_semantic_matrix"), dpi=600, tight=False)

    fig, ax = fixed_panel()
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.78)
    draw_district_fingerprint(ax, data, "f")
    save_figure(fig, panel_path("f", "district_evidence_fingerprint"), dpi=600, tight=False)


def render_composite(data: dict[str, object]) -> None:
    fig = plt.figure(figsize=(7.25, 9.05))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.065, top=0.895)
    gs = GridSpec(
        5,
        6,
        figure=fig,
        height_ratios=[0.22, 1.18, 1.0, 0.94, 1.12],
        width_ratios=[1, 1, 1, 1, 1, 1],
        wspace=0.12,
        hspace=0.82,
    )
    axleg = fig.add_subplot(gs[0, :])
    draw_fig3_legend(axleg)
    axa = fig.add_subplot(gs[1:3, :3])
    draw_hero_map(axa, data, "a")
    axb = fig.add_subplot(gs[1:3, 3:])
    draw_local_zoom(axb, data, "福田区", "b")
    axc = fig.add_subplot(gs[3, :3])
    draw_host_type_distribution(axc, data, "c")
    axd = fig.add_subplot(gs[3, 3:])
    draw_class_composition(axd, data, "d", legend_anchor=(0.5, -0.22))
    axe = fig.add_subplot(gs[4, :3])
    draw_rule_semantic_matrix(axe, data, "e")
    axf = fig.add_subplot(gs[4, 3:])
    draw_district_fingerprint(axf, data, "f", title_y=1.20)
    fig.text(0.015, 0.985, "Fig. 3 | Rule-liminal venues as Shenzhen's institutional middle terrain", ha="left", va="top", fontsize=8.0, fontweight="bold", color=INK)
    fig.text(0.015, 0.960, "Candidate venues are transition signals, not confirmed pet-friendly places; observed rules remain separately marked as traceable evidence.", ha="left", va="top", fontsize=5.0, color=TEXT)
    save_figure(fig, COMP / f"{OUT_STEM}_composite", dpi=520, tight=False)


def contact_sheet() -> None:
    files = [
        panel_path("a", "city_liminal_surface").with_suffix(".png"),
        panel_path("b", "futian_local_texture").with_suffix(".png"),
        panel_path("c", "host_type_score_distribution").with_suffix(".png"),
        panel_path("d", "liminal_class_composition").with_suffix(".png"),
        panel_path("e", "rule_semantic_matrix").with_suffix(".png"),
        panel_path("f", "district_evidence_fingerprint").with_suffix(".png"),
        COMP / f"{OUT_STEM}_composite.png",
    ]
    thumbs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        im.thumbnail((430, 350), Image.LANCZOS)
        canvas = Image.new("RGB", (450, 382), "white")
        canvas.paste(im, ((450 - im.width) // 2, 12))
        ImageDraw.Draw(canvas).text((10, 360), f.name, fill=(35, 35, 35))
        thumbs.append(canvas)
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 450, rows * 382), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * 450, (i // cols) * 382))
    sheet.save(PREVIEWS / f"{OUT_STEM}_contact_sheet.png", dpi=(180, 180))


def main() -> None:
    data = load()
    render_single_panels(data)
    render_composite(data)
    contact_sheet()
    rows = []
    for f in sorted(PANELS.glob(f"{OUT_STEM}_panel_*.*")):
        rows.append({"asset": f.relative_to(ROOT).as_posix(), "role": "panel"})
    for f in sorted(COMP.glob(f"{OUT_STEM}_composite.*")):
        rows.append({"asset": f.relative_to(ROOT).as_posix(), "role": "composite"})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(SRC / f"{OUT_STEM}_export_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
