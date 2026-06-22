#!/usr/bin/env python3
"""Rebuild HSSC Fig. 7 with source-locked cases and higher map/text fidelity.

This is a conservative source-level rebuild of the local evidence vignette
atlas. Case selection and data layers are imported from the v30 source script;
layout, typography and OSM drawing hierarchy are rebuilt for HSSC readability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
V30_CODE = ROOT / "submission_package" / "code" / "figures_v30_map_evidence_vignettes"
sys.path.insert(0, str(V30_CODE))
import build_v30_map_evidence_vignettes as v30  # noqa: E402


OUT = ROOT / "hssc_fig7_quality_v7_20260616"
FORMAL = OUT / "formal_upload_figures"
AUDIT = OUT / "audit"
SRC = OUT / "source_data"
for folder in [FORMAL, AUDIT, SRC]:
    folder.mkdir(parents=True, exist_ok=True)

INK = "#24211f"
TEXT = "#4f4a43"
MUTED = "#756e66"
FAINT = "#e8e2d8"
LAND = "#fbfaf7"
GRID = "#e5ded2"
BUILDING = "#d9d2c5"
ROAD_LIGHT = "#c2beb4"
ROAD_MID = "#9d9991"
ROAD_DARK = "#6e6a64"
PED = "#79a6a4"
PUBLIC = "#dce9df"
BLUE = "#315f7d"
GREEN = "#4f8d69"
GOLD = "#e4bf68"
RUST = "#d9806d"
RUST_DARK = "#a95b52"
SILENT = "#b8b0a5"
WHITE = "#ffffff"

MAJOR_ROADS = {"motorway", "trunk", "primary"}
MID_ROADS = {"secondary", "tertiary", "primary_link", "trunk_link", "motorway_link"}
LOCAL_ROADS = {"residential", "service", "unclassified", "secondary_link", "tertiary_link"}

TYPE_LABELS = v30.TYPE_LABELS
TYPE_COLORS = v30.TYPE_COLORS
OVERVIEW_LABEL_OFFSETS = {
    "baoan_high_host_low_rule": (-1800, -1250),
    "baoan_verification_pressure": (-2100, 950),
    "futian_visible_rule_core": (-450, 1550),
    "nanshan_mixed_contrast": (-1900, -950),
    "nanshan_restrictive_contact_zone": (1850, -1200),
    "longgang_service_rule_gap": (1600, 950),
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.4,
        "axes.titlesize": 8.4,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.55,
        "axes.edgecolor": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def save_all(fig: plt.Figure, stem: Path) -> dict[str, str]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=600)
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)
    rgb = Image.open(png).convert("RGB")
    rgb.save(png, dpi=(600, 600))
    return {"png": str(png), "pdf": str(pdf), "svg": str(svg)}


def normalize_extent(extent: tuple[float, float, float, float], target_aspect: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = extent
    w = maxx - minx
    h = maxy - miny
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    if h <= 0:
        return extent
    if w / h > target_aspect:
        new_h = w / target_aspect
        miny = cy - new_h / 2
        maxy = cy + new_h / 2
    else:
        new_w = h * target_aspect
        minx = cx - new_w / 2
        maxx = cx + new_w / 2
    return minx, miny, maxx, maxy


def local_extent(geom, width_m: float = 3600, aspect: float = 1.62) -> tuple[float, float, float, float]:
    c = geom.centroid
    h = width_m / aspect
    return c.x - width_m / 2, c.y - h / 2, c.x + width_m / 2, c.y + h / 2


def subset(gdf: gpd.GeoDataFrame, extent: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    return v30.subset_extent(gdf, extent)


def plot_roads(ax: plt.Axes, roads: gpd.GeoDataFrame, extent: tuple[float, float, float, float] | None, city: bool) -> None:
    r = roads if extent is None else subset(roads, extent)
    if r.empty:
        return
    if city:
        major = r[r["fclass"].isin(MAJOR_ROADS | {"secondary"})]
        mid = r[r["fclass"].isin({"tertiary", "primary_link", "secondary_link", "trunk_link", "motorway_link"})]
        if not mid.empty:
            mid.plot(ax=ax, color=ROAD_LIGHT, linewidth=0.18, alpha=0.28, zorder=6)
        if not major.empty:
            major.plot(ax=ax, color=ROAD_MID, linewidth=0.28, alpha=0.42, zorder=7)
        return
    local = r[r["fclass"].isin(LOCAL_ROADS)]
    mid = r[r["fclass"].isin(MID_ROADS)]
    major = r[r["fclass"].isin(MAJOR_ROADS)]
    if not local.empty:
        local.plot(ax=ax, color=ROAD_LIGHT, linewidth=0.20, alpha=0.42, zorder=5)
    if not mid.empty:
        mid.plot(ax=ax, color=ROAD_MID, linewidth=0.34, alpha=0.56, zorder=6)
    if not major.empty:
        major.plot(ax=ax, color=ROAD_DARK, linewidth=0.52, alpha=0.70, zorder=7)


def draw_local_texture(ax: plt.Axes, data: dict[str, object], extent: tuple[float, float, float, float]) -> None:
    ax.set_facecolor(LAND)
    for key, color, alpha, z in [
        ("public", PUBLIC, 0.55, 1),
        ("buildings", BUILDING, 0.36, 2),
    ]:
        layer = subset(data[key], extent)
        if not layer.empty:
            layer.plot(ax=ax, color=color, edgecolor="none", alpha=alpha, zorder=z)
    plot_roads(ax, data["roads"], extent, city=False)
    ped = subset(data["ped"], extent)
    if not ped.empty:
        ped.plot(ax=ax, color=PED, linewidth=0.20, alpha=0.46, zorder=8)


def draw_north_arrow_and_scale(ax: plt.Axes, extent: tuple[float, float, float, float], scale_m: int, compact: bool = False) -> None:
    minx, miny, maxx, maxy = extent
    w = maxx - minx
    h = maxy - miny
    sx0 = minx + (0.065 if compact else 0.055) * w
    sy = miny + (0.068 if compact else 0.055) * h
    sx1 = sx0 + scale_m
    ax.plot([sx0, sx1], [sy, sy], color=INK, linewidth=0.75 if compact else 0.95, zorder=45, solid_capstyle="butt")
    ax.plot([sx0, sx0], [sy - 0.010 * h, sy + 0.010 * h], color=INK, linewidth=0.60, zorder=45)
    ax.plot([sx1, sx1], [sy - 0.010 * h, sy + 0.010 * h], color=INK, linewidth=0.60, zorder=45)
    label = f"{scale_m//1000} km" if scale_m >= 1000 else f"{scale_m} m"
    ax.text(
        (sx0 + sx1) / 2,
        sy + 0.018 * h,
        label,
        ha="center",
        va="bottom",
        fontsize=4.8 if compact else 5.4,
        color=INK,
        zorder=46,
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
    )
    arrow_x = 0.932 if compact else 0.952
    arrow_top = 0.880 if compact else 0.930
    arrow_bottom = 0.795 if compact else 0.845
    ax.annotate(
        "",
        xy=(arrow_x, arrow_top),
        xytext=(arrow_x, arrow_bottom),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "lw": 0.64 if compact else 0.86, "color": INK, "shrinkA": 0, "shrinkB": 0},
        zorder=46,
    )
    ax.text(
        arrow_x,
        arrow_top + (0.018 if compact else 0.018),
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=4.9 if compact else 5.9,
        color=INK,
        fontweight="bold",
        zorder=46,
    )


def draw_local_map(ax: plt.Axes, data: dict[str, object], extent: tuple[float, float, float, float], focal_grid_id: str) -> None:
    draw_local_texture(ax, data, extent)
    grid = subset(data["grid"], extent)
    if not grid.empty:
        grid.plot(ax=ax, color="none", edgecolor=GRID, linewidth=0.18, alpha=0.66, zorder=10)
        suppressed = grid[grid["grid_emergence_type_v51"].eq("suppressed_emergence_frontier")]
        transitional = grid[grid["grid_emergence_type_v51"].eq("mixed_or_transitional_zone")]
        if not suppressed.empty:
            suppressed.plot(ax=ax, color=RUST, edgecolor="none", alpha=0.11, zorder=11)
        if not transitional.empty:
            transitional.plot(ax=ax, color=GOLD, edgecolor="none", alpha=0.09, zorder=11)
    hosts = subset(data["hosts"], extent)
    silent = hosts[hosts["liminal_class"].eq("ecology_present_rule_silent")]
    priority = hosts[hosts["liminal_class"].isin(["rule_liminal_threshold_core", "ecology_ready_rule_liminal_frontier"])]
    pet = subset(data["pet"], extent)
    queue = subset(data["queue"], extent)
    rules = subset(data["rules"], extent)
    if not silent.empty:
        pts = silent.sample(min(len(silent), 520), random_state=16)
        pts.plot(ax=ax, color=SILENT, markersize=4.0, alpha=0.38, linewidth=0, zorder=14)
    if not priority.empty:
        pts = priority.sample(min(len(priority), 620), random_state=17)
        pts.plot(ax=ax, color=WHITE, markersize=18, alpha=0.90, linewidth=0, zorder=15)
        pts.plot(ax=ax, color=RUST, markersize=8.0, alpha=0.82, edgecolor=WHITE, linewidth=0.32, zorder=16)
    if not pet.empty:
        pet.plot(ax=ax, color=WHITE, markersize=20, alpha=0.90, linewidth=0, zorder=17)
        pet.plot(ax=ax, color=BLUE, markersize=8.0, alpha=0.78, edgecolor=WHITE, linewidth=0.35, zorder=18)
    if not queue.empty:
        pts = queue.sample(min(len(queue), 240), random_state=18)
        pts.plot(ax=ax, color=WHITE, markersize=24, alpha=0.92, linewidth=0, zorder=19)
        pts.plot(ax=ax, color=GOLD, markersize=10.0, alpha=0.88, edgecolor=WHITE, linewidth=0.36, zorder=20)
    if not rules.empty:
        rules.plot(ax=ax, color=WHITE, markersize=50, alpha=0.96, linewidth=0, zorder=21)
        rules.plot(ax=ax, color=GREEN, markersize=27, alpha=0.96, edgecolor=WHITE, linewidth=0.45, zorder=22)
    draw_north_arrow_and_scale(ax, extent, 500, compact=True)
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#d8d2c7")
        spine.set_linewidth(0.52)


def draw_bar(ax: plt.Axes, x: float, y: float, width: float, value: float, color: str, label: str) -> None:
    ax.text(x, y + 0.018, label, transform=ax.transAxes, ha="left", va="center", fontsize=5.4, color=TEXT)
    ax.add_patch(Rectangle((x + 0.42, y), width, 0.034, transform=ax.transAxes, facecolor="#eee9df", edgecolor="none"))
    ax.add_patch(Rectangle((x + 0.42, y), width * float(np.clip(value, 0, 1)), 0.034, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=0.84))
    ax.text(x + 0.42 + width + 0.010, y + 0.017, f"{value:.2f}", transform=ax.transAxes, ha="left", va="center", fontsize=5.6, fontweight="bold", color=INK)


def fmt_count(value: object) -> str:
    n = int(value)
    if n >= 1000:
        return f"{n / 1000:.2f}k"
    return f"{n:,}"


def draw_evidence_box(ax: plt.Axes, case: pd.Series, stats: dict[str, object]) -> None:
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=WHITE, edgecolor="#d8d2c7", linewidth=0.65))
    ax.text(0.055, 0.945, "Matched evidence", transform=ax.transAxes, ha="left", va="top", fontsize=7.0, fontweight="bold", color=INK)
    role_color = RUST_DARK if "frontier" in case["evidence_role"] or "liminal" in case["evidence_role"] else BLUE
    ax.text(0.055, 0.865, case["evidence_role"], transform=ax.transAxes, ha="left", va="center", fontsize=5.6, color=WHITE, bbox={"facecolor": role_color, "edgecolor": "none", "pad": 1.5})
    ax.text(0.930, 0.905, f"n={stats['local_grid_cells']} cells", transform=ax.transAxes, ha="right", va="center", fontsize=5.6, fontweight="bold", color=TEXT)
    rule_share = int(stats["positive_rule_grid_count"]) / max(int(stats["local_grid_cells"]), 1)
    ax.text(0.055, 0.800, f"same window; rule-visible cells {rule_share:.0%}", transform=ax.transAxes, ha="left", va="top", fontsize=5.0, color=MUTED)
    ax.text(0.055, 0.720, "Local indices", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.9, fontweight="bold", color=INK)
    draw_bar(ax, 0.055, 0.652, 0.300, float(stats["mean_pet_ecology_norm"]), BLUE, "service ecology")
    draw_bar(ax, 0.055, 0.596, 0.300, float(stats["mean_liminal_host_norm"]), RUST, "host liminality")
    draw_bar(ax, 0.055, 0.540, 0.300, float(stats["mean_rule_visibility_norm"]), GREEN, "rule visibility")
    draw_bar(ax, 0.055, 0.484, 0.300, float(stats["mean_suppression_index"]), RUST_DARK, "suppression")
    counts = [
        ("pet services", stats["pet_service_points"], BLUE),
        ("host candidates", stats["host_candidates"], RUST),
        ("verify queue", stats["verification_queue_candidates"], GOLD),
        ("rule records", stats["visible_rule_records"], GREEN),
    ]
    ax.text(0.055, 0.384, "Evidence counts", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.9, fontweight="bold", color=INK)
    maxv = max([int(v) for _, v, _ in counts] + [1])
    for i, (label, val, color) in enumerate(counts):
        x = 0.055 + (i % 2) * 0.455
        y = 0.270 - (i // 2) * 0.104
        ax.add_patch(Rectangle((x, y), 0.390, 0.082, transform=ax.transAxes, facecolor="#fbfaf7", edgecolor="#e4ded4", linewidth=0.40))
        ax.text(x + 0.025, y + 0.057, f"{int(val):,}", transform=ax.transAxes, ha="left", va="center", fontsize=6.2, fontweight="bold", color=INK)
        ax.text(x + 0.025, y + 0.030, label, transform=ax.transAxes, ha="left", va="center", fontsize=4.8, color=MUTED)
        w = 0.300 * np.log1p(int(val)) / max(np.log1p(maxv), 1e-6)
        ax.add_patch(Rectangle((x + 0.025, y + 0.010), w, 0.010, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=0.86))
    vals = [(TYPE_LABELS[t], int(stats[f"host_type_{t}"]), TYPE_COLORS[t]) for t in TYPE_LABELS]
    total = max(sum(v for _, v, _ in vals), 1)
    ax.text(0.055, 0.105, "Host mix", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.9, fontweight="bold", color=INK)
    left = 0.055
    for _, val, color in vals:
        w = 0.830 * val / total
        if w:
            ax.add_patch(Rectangle((left, 0.060), w, 0.035, transform=ax.transAxes, facecolor=color, edgecolor=WHITE, linewidth=0.25, alpha=0.86))
        left += w
    top = sorted(vals, key=lambda x: x[1], reverse=True)[0]
    ax.text(0.055, 0.030, f"top: {top[0]} ({top[1]:,})", transform=ax.transAxes, ha="left", va="bottom", fontsize=4.7, color=MUTED)


def draw_evidence_strip(ax: plt.Axes, case: pd.Series, stats: dict[str, object]) -> None:
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=WHITE, edgecolor="#d8d2c7", linewidth=0.62))
    rule_share = int(stats["positive_rule_grid_count"]) / max(int(stats["local_grid_cells"]), 1)
    ax.text(0.030, 0.660, f"rule-vis {rule_share:.0%}", transform=ax.transAxes, ha="left", va="center", fontsize=4.05, color=MUTED)
    ax.text(0.955, 0.660, f"count, n={stats['local_grid_cells']}", transform=ax.transAxes, ha="right", va="center", fontsize=4.05, color=MUTED)
    ax.plot([0.500, 0.500], [0.230, 0.635], transform=ax.transAxes, color="#ebe5dc", lw=0.42)

    metrics = [
        ("svc", float(stats["mean_pet_ecology_norm"]), BLUE),
        ("host", float(stats["mean_liminal_host_norm"]), RUST),
        ("rule", float(stats["mean_rule_visibility_norm"]), GREEN),
        ("sup", float(stats["mean_suppression_index"]), RUST_DARK),
    ]
    metric_xs = [0.075, 0.180, 0.285, 0.390]
    for i, (label, value, color) in enumerate(metrics):
        x = metric_xs[i]
        ax.text(x, 0.505, label, transform=ax.transAxes, ha="center", va="center", fontsize=4.15, color=TEXT)
        ax.plot([x - 0.028, x + 0.028], [0.410, 0.410], transform=ax.transAxes, color="#eee9df", lw=0.85, solid_capstyle="butt")
        ax.plot([x - 0.028, x - 0.028 + 0.056 * float(np.clip(value, 0, 1))], [0.410, 0.410], transform=ax.transAxes, color=color, lw=0.85, solid_capstyle="butt")
        ax.text(x, 0.285, f"{value:.2f}", transform=ax.transAxes, ha="center", va="center", fontsize=4.75, fontweight="bold", color=INK)

    counts = [
        ("svc", stats["pet_service_points"], BLUE),
        ("host", stats["host_candidates"], RUST),
        ("queue", stats["verification_queue_candidates"], GOLD),
        ("rule", stats["visible_rule_records"], GREEN),
    ]
    count_xs = [0.590, 0.705, 0.820, 0.935]
    for i, (label, val, color) in enumerate(counts):
        x = count_xs[i]
        ax.add_patch(Rectangle((x - 0.032, 0.390), 0.014, 0.036, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=0.86))
        ax.text(x, 0.285, fmt_count(val), transform=ax.transAxes, ha="center", va="center", fontsize=4.75, fontweight="bold", color=INK)


def draw_header(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.text(0.0, 0.43, "Fig. 7 | Local evidence vignette atlas", transform=ax.transAxes, ha="left", va="center", fontsize=9.0, fontweight="bold", color=INK)


def draw_legend_strip(ax: plt.Axes) -> None:
    ax.set_axis_off()
    legend = [
        ("patch", PUBLIC, "public space"),
        ("line", ROAD_DARK, "major road"),
        ("line", PED, "pedestrian path"),
        ("point", RUST, "rule-liminal host"),
        ("point", BLUE, "pet service"),
        ("point", GOLD, "validation queue"),
        ("point", GREEN, "visible rule"),
    ]
    ax.add_patch(
        Rectangle(
            (0.0, 0.04),
            1.0,
            0.86,
            transform=ax.transAxes,
            facecolor=WHITE,
            edgecolor="#ddd7ce",
            linewidth=0.40,
            zorder=1,
        )
    )
    xs = [0.018, 0.164, 0.312, 0.466, 0.615, 0.748, 0.885]
    y = 0.48
    for i, (kind, color, label) in enumerate(legend):
        x = xs[i]
        if kind == "patch":
            ax.add_patch(Rectangle((x, y - 0.125), 0.020, 0.250, transform=ax.transAxes, facecolor=color, edgecolor="#d2ccc2", linewidth=0.35, zorder=2))
        elif kind == "line":
            ax.plot([x, x + 0.026], [y, y], transform=ax.transAxes, color=color, linewidth=1.0, zorder=2)
        else:
            ax.scatter([x + 0.012], [y], transform=ax.transAxes, s=18, color=color, edgecolor=WHITE, linewidth=0.35, zorder=2)
        ax.text(x + 0.030, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=4.45, color=TEXT, zorder=2)


def draw_overview(ax: plt.Axes, data: dict[str, object], cases: pd.DataFrame) -> None:
    boundary = data["boundary"]
    grid = data["grid"]
    public = data["public"]
    minx, miny, maxx, maxy = boundary.total_bounds
    extent = normalize_extent((minx - 1200, miny - 1200, maxx + 1200, maxy + 1200), 2.15)
    ax.set_facecolor(LAND)
    p = subset(public, extent)
    if not p.empty:
        p.plot(ax=ax, color=PUBLIC, edgecolor="none", alpha=0.38, zorder=1)
    grid.plot(ax=ax, color="#fbfaf7", edgecolor=GRID, linewidth=0.012, alpha=0.70, zorder=2, rasterized=True)
    suppressed = grid[grid["grid_emergence_type_v51"].eq("suppressed_emergence_frontier")]
    visible = grid[pd.to_numeric(grid["positive_rule_count"], errors="coerce").fillna(0).gt(0)]
    if not suppressed.empty:
        suppressed.plot(ax=ax, color=RUST, edgecolor="none", alpha=0.46, zorder=3, rasterized=True)
    if not visible.empty:
        visible.plot(ax=ax, color=GREEN, edgecolor="none", alpha=0.70, zorder=4, rasterized=True)
    plot_roads(ax, data["roads"], extent, city=True)
    boundary.boundary.plot(ax=ax, color="#59554f", linewidth=0.52, alpha=0.92, zorder=10)
    for i, row in cases.iterrows():
        focal = grid[grid["grid_id"].eq(row["focal_grid_id"])].iloc[0]
        case_extent = local_extent(focal.geometry, 3600 if row["case_id"] != "futian_visible_rule_core" else 2900)
        ax.add_patch(Rectangle((case_extent[0], case_extent[1]), case_extent[2] - case_extent[0], case_extent[3] - case_extent[1], fill=False, edgecolor=INK, linewidth=0.72, zorder=20))
        c = focal.geometry.centroid
        dx, dy = OVERVIEW_LABEL_OFFSETS.get(row["case_id"], (0, 0))
        ax.plot([c.x, c.x + dx], [c.y, c.y + dy], color=INK, linewidth=0.35, alpha=0.55, zorder=24)
        ax.text(c.x + dx, c.y + dy, chr(ord("b") + i), ha="center", va="center", fontsize=7.4, fontweight="bold", color=WHITE, bbox={"facecolor": INK, "edgecolor": WHITE, "linewidth": 0.35, "boxstyle": "circle,pad=0.16"}, zorder=25)
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal", adjustable="box")
    draw_north_arrow_and_scale(ax, extent, 10000, compact=False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.0, 1.018, "a", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.8, fontweight="bold", color=INK)
    ax.text(0.037, 1.022, "citywide case-window overview with OSM road structure", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.6, fontweight="bold", color=INK)


def render() -> None:
    data = v30.load_data()
    cases = v30.select_cases(data).reset_index(drop=True)
    cases.to_csv(SRC / "Figure_07_case_selection_source_locked.csv", index=False, encoding="utf-8-sig")

    fig_w, fig_h = 7.15, 11.69
    fig = plt.figure(figsize=(fig_w, fig_h))

    left, right = 0.050, 0.970
    full_w = right - left
    ax_header = fig.add_axes([left, 0.960, full_w, 0.030])
    draw_header(ax_header)

    overview_aspect = 2.15
    overview_h = (full_w * fig_w / overview_aspect) / fig_h
    overview_y = 0.675
    ax_overview = fig.add_axes([left, overview_y, full_w, overview_h])
    draw_overview(ax_overview, data, cases)
    legend_h = 0.022
    legend_y = overview_y - 0.032
    ax_legend = fig.add_axes([left, legend_y, full_w, legend_h])
    draw_legend_strip(ax_legend)

    stats_rows: list[dict[str, object]] = []
    gutter = 0.044
    col_w = (full_w - 2 * gutter) / 3
    map_h = (col_w * fig_w / 1.62) / fig_h
    strip_h = 0.031
    title_pad = 0.013
    map_strip_gap = 0.004
    row_h = title_pad + map_h + map_strip_gap + strip_h
    row_top_start = legend_y - 0.014
    row_gap = max((row_top_start - 0.012 - 4 * row_h) / 3, 0.002)
    for i, (_, case) in enumerate(cases.iterrows()):
        row = i // 3
        col = i % 3
        x0 = left + col * (col_w + gutter)
        row_top = row_top_start - row * (row_h + row_gap)
        map_y = row_top - title_pad - map_h
        strip_y = map_y - map_strip_gap - strip_h
        ax_map = fig.add_axes([x0, map_y, col_w, map_h])
        ax_ev = fig.add_axes([x0, strip_y, col_w, strip_h])
        focal = data["grid"][data["grid"]["grid_id"].eq(case["focal_grid_id"])].iloc[0]
        extent = local_extent(focal.geometry, 3600 if case["case_id"] != "futian_visible_rule_core" else 2900)
        stats = v30.local_stats(data, extent, case["focal_grid_id"])
        draw_local_map(ax_map, data, extent, case["focal_grid_id"])
        draw_evidence_strip(ax_ev, case, stats)
        letter = chr(ord("b") + i)
        fig.text(x0, row_top - 0.002, f"{letter}  {case['district_en']} | {case['evidence_role']}", ha="left", va="top", fontsize=6.1, fontweight="bold", color=INK)
        stats.update(
            {
                "panel": letter,
                "case_id": case["case_id"],
                "district_name": case["district_name"],
                "district_en": case["district_en"],
                "evidence_role": case["evidence_role"],
                "focal_grid_id": case["focal_grid_id"],
                "extent_minx": extent[0],
                "extent_miny": extent[1],
                "extent_maxx": extent[2],
                "extent_maxy": extent[3],
            }
        )
        stats_rows.append(stats)

    pd.DataFrame(stats_rows).to_csv(SRC / "Figure_07_panel_stats_source_locked.csv", index=False, encoding="utf-8-sig")
    outputs = save_all(fig, FORMAL / "Figure_07")
    manifest = {
        "figure": "Figure_07",
        "source_script": str(V30_CODE / "build_v30_map_evidence_vignettes.py"),
        "source_data": [
            "data_processed/model/grid_emergence_suppression_indices_500m_2025_v51_sparse_repaired.geojson",
            "data_processed/platform/rule_liminal_host_candidates_2025_v4.csv",
            "data_processed/platform/rule_liminal_verification_queue_2025_v4.csv",
            "data_processed/platform/pet_service_ecology_points_2025_v3.geojson",
            "data_processed/platform/rule_semantic_geocoded_points_v1.geojson",
            "data_processed/osm/shenzhen_osm_roads.gpkg",
            "data_processed/osm/shenzhen_osm_pedestrian_paths.gpkg",
            "data_processed/osm/shenzhen_osm_buildings_3d.gpkg",
            "data_processed/osm/shenzhen_osm_public_space.gpkg",
        ],
        "layout": "double-column-width tall figure; citywide overview plus three-column four-row source-locked local evidence cards",
        "outputs": outputs,
    }
    (SRC / "Figure_07_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    render()
