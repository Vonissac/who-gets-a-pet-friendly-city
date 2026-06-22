#!/usr/bin/env python3
"""Build v28 Fig. 4: suppressed-emergence frontier.

Design standard inherited from the v14 district atlas:
- Python-only rendering.
- Every panel exported separately before the composite.
- Fixed standalone panel canvas for flexible later layout.
- Visual argument first: the figure explains a diagnostic spatial mechanism,
  not a dashboard of model outputs.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import box

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from build_fig2_v28 import (  # noqa: E402
    BLUE,
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
from build_fig2_district_atlas_v28 import (  # noqa: E402
    DISTRICT_LABEL,
    DISTRICT_PANEL_ASPECT,
    SILENT,
    district_extent,
    draw_atlas_legend,
    draw_texture_light,
)


OUT_STEM = "fig4_suppressed_frontier_v28"
FIXED_PANEL_SIZE = (3.3, 2.75)
DISTRICT_ORDER = ["福田区", "龙华区", "宝安区", "龙岗区", "南山区", "罗湖区", "光明区", "坪山区", "盐田区"]

FRONTIER = "#d88973"
TRANSITION = "#e8c76f"
LOW = "#dedbd2"
RULE = "#4f8a69"
SERVICE = "#3e7899"
FINGER_CMAP = LinearSegmentedColormap.from_list("frontier_fingerprint", ["#fffaf0", "#f5e9c9", "#e4bf68", "#d88973"])


def panel_path(letter: str, name: str) -> Path:
    return PANELS / f"{OUT_STEM}_panel_{letter}_{name}"


def load() -> dict[str, object]:
    hosts = read_csv("data_processed/platform/rule_liminal_host_candidates_2025_v4.csv")
    host_g = gpd.GeoDataFrame(
        hosts,
        geometry=gpd.points_from_xy(hosts["lon_wgs84"], hosts["lat_wgs84"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32649")
    grid = read_gdf("data_processed/model/grid_emergence_suppression_indices_500m_2025_v51_sparse_repaired.geojson")
    grid["grid_emergence_type"] = grid["grid_emergence_type_v51"]
    grid["emergence_index"] = grid["emergence_index_v51"]
    grid["suppression_index"] = grid["suppression_index_v51"]
    grid["positive_rule_norm"] = grid["positive_rule_norm_v51"]
    return {
        "grid": grid,
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
        "district": read_csv("data_processed/model/district_emergence_suppression_indices_2025_v5.csv"),
        "findings": read_csv("data/derived_data/model/final_model_summary_v7.csv"),
    }


def city_extent(boundary: gpd.GeoDataFrame, pad: float = 4200) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = boundary.total_bounds
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def fixed_fig() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=FIXED_PANEL_SIZE)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.14, top=0.84)
    return fig, ax


def light_frame(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#d8d3ca")
        spine.set_linewidth(0.42)


def render_panel_a(data: dict[str, object]) -> None:
    """Overview anchor + high-granularity district zoom atlas."""
    render_frontier_overview_panel(data)
    render_frontier_district_zoom_panels(data)
    render_frontier_district_atlas_composite(data)


def render_frontier_overview_panel(data: dict[str, object]) -> None:
    fig, ax = fixed_fig()
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.09, top=0.86)
    grid = data["grid"].copy()
    boundary = data["boundary"]
    grid.plot(ax=ax, color=LAND, edgecolor=GRID, linewidth=0.010, zorder=0, rasterized=True)
    grid[grid["grid_emergence_type"].eq("mixed_or_transitional_zone")].plot(
        ax=ax, color=TRANSITION, edgecolor="none", alpha=0.28, zorder=4, rasterized=True
    )
    grid[grid["grid_emergence_type"].eq("suppressed_emergence_frontier")].plot(
        ax=ax, color=FRONTIER, edgecolor="none", alpha=0.72, zorder=8, rasterized=True
    )
    boundary.boundary.plot(ax=ax, color="#514b45", linewidth=0.44, alpha=0.78, zorder=20)
    # Mark district windows to make this an overview anchor, not just another map.
    for district in DISTRICT_ORDER:
        dg = grid[grid["district_name"].eq(district)]
        if len(dg) == 0:
            continue
        minx, miny, maxx, maxy = district_extent(dg, 0.10)
        ax.add_patch(Rectangle((minx, miny), maxx - minx, maxy - miny, fill=False, edgecolor="#8c8177", linewidth=0.22, alpha=0.52, zorder=25))
    minx, miny, maxx, maxy = city_extent(boundary)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    light_frame(ax)
    panel_label(ax, "a0", "city overview: suppressed frontier anchor")
    ax.text(0.012, 0.025, "coral = 503 sparse-repaired frontier grids; grey boxes = district zoom panels", transform=ax.transAxes, fontsize=3.8, color=MUTED, ha="left", va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35})
    save_figure(fig, panel_path("a0", "city_overview"), dpi=600, tight=False)


def draw_frontier_zoom_panel(
    ax: plt.Axes,
    data: dict[str, object],
    district: str,
    letter: str | None,
    compact: bool = False,
) -> dict[str, int | str]:
    grid = data["grid"].copy()
    boundary = data["boundary"]
    hosts = data["host_g"]
    pet = data["pet"]
    rules = data["rules"]
    dg = grid[grid["district_name"].eq(district)].copy()
    if len(dg) == 0:
        raise ValueError(f"Missing district: {district}")
    ext = district_extent(dg, 0.08 if compact else 0.10)
    draw_texture_light(ax, data, ext)
    bbox = box(*ext)
    grid[grid.intersects(bbox)].plot(
        ax=ax, color=LAND, edgecolor=GRID, linewidth=0.006, alpha=0.20, zorder=5, rasterized=True
    )
    dg[dg["grid_emergence_type"].eq("mixed_or_transitional_zone")].plot(ax=ax, color=TRANSITION, edgecolor="none", alpha=0.18, zorder=7, rasterized=True)
    dg[dg["grid_emergence_type"].eq("suppressed_emergence_frontier")].plot(ax=ax, color=FRONTIER, edgecolor="none", alpha=0.36, zorder=9, rasterized=True)
    boundary.boundary.plot(ax=ax, color="#aaa49b", linewidth=0.10 if compact else 0.15, alpha=0.40, zorder=12)
    dg.dissolve().boundary.plot(ax=ax, color="#5d5750", linewidth=0.24 if compact else 0.32, alpha=0.72, zorder=30)
    hsub = hosts[hosts["district_name"].eq(district)].copy()
    priority = hsub[hsub["liminal_class"].isin(["rule_liminal_threshold_core", "ecology_ready_rule_liminal_frontier"])]
    silent = hsub[hsub["liminal_class"].eq("ecology_present_rule_silent")]
    if len(silent):
        silent.sample(min(len(silent), 500 if compact else 760), random_state=34).plot(ax=ax, color=SILENT, markersize=1.5 if compact else 2.0, alpha=0.11, linewidth=0, zorder=13)
    if len(priority):
        priority.sample(min(len(priority), 680 if compact else 980), random_state=35).plot(ax=ax, color=FRONTIER, markersize=3.0 if compact else 4.1, alpha=0.30, edgecolor="white", linewidth=0.36, zorder=14)
    psub = pet[pet["district_name"].eq(district)]
    if len(psub):
        psub.plot(ax=ax, color=BLUE_DARK, markersize=2.6 if compact else 3.5, alpha=0.20, linewidth=0, zorder=15)
    rsub = rules[rules["grid_id"].isin(dg["grid_id"])]
    if len(rsub):
        rsub.plot(ax=ax, color=GREEN, markersize=20 if compact else 30, alpha=0.88, edgecolor="white", linewidth=0.20, zorder=18)
    ax.set_xlim(ext[0], ext[2])
    ax.set_ylim(ext[1], ext[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#d8d3ca")
        spine.set_linewidth(0.34 if compact else 0.42)
    name = DISTRICT_LABEL.get(district, DISTRICT_EN.get(district, district))
    if letter:
        ax.text(0.0, 1.018, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.4, fontweight="bold", color=INK)
        ax.text(0.085, 1.024, name, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.4, fontweight="bold", color=INK)
    else:
        ax.text(0.01, 1.02, name, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.4, fontweight="bold", color=INK)
    n_frontier = int((dg["grid_emergence_type"] == "suppressed_emergence_frontier").sum())
    stats = {
        "district_name": district,
        "district_en": name,
        "frontier_grids": n_frontier,
        "priority_frontier_hosts": int(len(priority)),
        "pet_service_points": int(len(psub)),
        "positive_rule_points": int(len(rsub)),
    }
    ax.text(0.012, 0.025, f"frontier grids {n_frontier} | hosts {len(priority):,} | rules {len(rsub)}", transform=ax.transAxes, ha="left", va="bottom", fontsize=3.6 if compact else 4.0, color=MUTED, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.32}, zorder=40)
    return stats


def render_frontier_district_zoom_panels(data: dict[str, object]) -> None:
    letters = list("abcdefghi")
    rows = []
    for district, letter in zip(DISTRICT_ORDER, letters):
        fig, ax = plt.subplots(figsize=FIXED_PANEL_SIZE)
        fig.subplots_adjust(left=0.035, right=0.985, bottom=0.070, top=0.890)
        stats = draw_frontier_zoom_panel(ax, data, district, letter, compact=False)
        rows.append(stats)
        stem = panel_path(f"a_{letter}", f"{stats['district_en'].lower()}_frontier_zoom")
        save_figure(fig, stem, dpi=600, tight=False)
        pd.DataFrame([stats]).to_csv(SRC / f"{OUT_STEM}_panel_a_{letter}_{stats['district_en'].lower()}_source.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(SRC / f"{OUT_STEM}_panel_a_district_zoom_sources.csv", index=False, encoding="utf-8-sig")


def render_frontier_district_atlas_composite(data: dict[str, object]) -> None:
    fig = plt.figure(figsize=(7.25, 10.95))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.055, top=0.900)
    gs = GridSpec(
        7,
        6,
        figure=fig,
        height_ratios=[0.22, 1.03, 1, 1, 1, 0.86, 0.86],
        width_ratios=[1, 1, 1, 1, 1, 1],
        wspace=0.10,
        hspace=0.42,
    )
    axleg = fig.add_subplot(gs[0, :])
    draw_atlas_legend(axleg, "fig4")
    ax0 = fig.add_subplot(gs[1, :])
    grid = data["grid"]
    boundary = data["boundary"]
    grid.plot(ax=ax0, color=LAND, edgecolor=GRID, linewidth=0.006, alpha=0.55, zorder=0, rasterized=True)
    grid[grid["grid_emergence_type"].eq("suppressed_emergence_frontier")].plot(ax=ax0, color=FRONTIER, edgecolor="none", alpha=0.70, zorder=8, rasterized=True)
    boundary.boundary.plot(ax=ax0, color="#514b45", linewidth=0.36, alpha=0.72, zorder=20)
    for district in DISTRICT_ORDER:
        dg = grid[grid["district_name"].eq(district)]
        if len(dg):
            minx, miny, maxx, maxy = district_extent(dg, 0.10)
            ax0.add_patch(Rectangle((minx, miny), maxx - minx, maxy - miny, fill=False, edgecolor="#8c8177", linewidth=0.20, alpha=0.50, zorder=25))
    minx, miny, maxx, maxy = city_extent(boundary)
    ax0.set_xlim(minx, maxx); ax0.set_ylim(miny, maxy); ax0.set_aspect("equal"); ax0.set_xticks([]); ax0.set_yticks([])
    light_frame(ax0)
    ax0.text(0.0, 1.02, "a0", transform=ax0.transAxes, ha="left", va="bottom", fontsize=7.4, fontweight="bold", color=INK)
    ax0.text(0.105, 1.025, "Shenzhen overview: 549 suppressed-frontier grids", transform=ax0.transAxes, ha="left", va="bottom", fontsize=5.4, fontweight="bold", color=INK)
    letters = list("abcdefghi")
    for i, (district, letter) in enumerate(zip(DISTRICT_ORDER, letters)):
        ax = fig.add_subplot(gs[(i // 3) + 2, (i % 3) * 2 : (i % 3 + 1) * 2])
        draw_frontier_zoom_panel(ax, data, district, letter, compact=True)
    axj = fig.add_subplot(gs[5:, :3])
    draw_phase_space_axis(axj, data, "j")
    axk = fig.add_subplot(gs[5:, 3:])
    draw_district_burden_axis(axk, data, "k", title_y=1.08)
    fig.text(0.015, 0.985, "Fig. 4A atlas | Suppressed-frontier texture by district", ha="left", va="top", fontsize=8.0, fontweight="bold", color=INK)
    fig.text(0.015, 0.964, "City overview first; each district zoom then shows OSM texture, frontier grids, rule-liminal hosts, pet-service points and visible-rule records.", ha="left", va="top", fontsize=5.0, color=TEXT)
    save_figure(fig, COMP / f"{OUT_STEM}_panel_a_district_zoom_atlas", dpi=520, tight=False)


def render_panel_a(data: dict[str, object]) -> None:
    render_frontier_overview_panel(data)
    render_frontier_district_zoom_panels(data)
    render_frontier_district_atlas_composite(data)
    grid = data["grid"]
    grid[["grid_id", "district_name", "grid_emergence_type", "emergence_index", "suppression_index"]].to_csv(
        SRC / f"{OUT_STEM}_panel_a_source.csv", index=False, encoding="utf-8-sig"
    )


def draw_phase_space_axis(ax: plt.Axes, data: dict[str, object], letter: str = "b") -> None:
    grid = data["grid"].copy()
    grid["emergence_index"] = numeric(grid["emergence_index"])
    grid["suppression_index"] = numeric(grid["suppression_index"])
    grid["frontier"] = grid["grid_emergence_type"].eq("suppressed_emergence_frontier")
    rng = np.random.default_rng(18)
    sample_non = grid[~grid["frontier"]].sample(min(3600, (~grid["frontier"]).sum()), random_state=18)
    ax.scatter(
        sample_non["emergence_index"] + rng.normal(0, 0.003, len(sample_non)),
        sample_non["suppression_index"] + rng.normal(0, 0.003, len(sample_non)),
        s=5,
        color="#c9c5bb",
        alpha=0.32,
        linewidth=0,
        label="other grids",
    )
    f = grid[grid["frontier"]]
    ax.scatter(
        f["emergence_index"],
        f["suppression_index"],
        s=14,
        color=FRONTIER,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.16,
        label="suppressed frontier",
    )
    ax.axhline(grid["suppression_index"].quantile(0.90), color=FRONTIER, lw=0.55, alpha=0.55)
    ax.axvline(grid["emergence_index"].quantile(0.90), color=SERVICE, lw=0.55, alpha=0.55)
    ax.set_xlabel("emergence index")
    ax.set_ylabel("suppression index")
    ax.set_xlim(-0.02, 0.84)
    ax.set_ylim(-0.02, 0.86)
    clean_axis(ax)
    panel_label(ax, letter, "phase space: ecology present, rule translation delayed")
    ax.legend(loc="lower right", frameon=False, fontsize=3.7, handletextpad=0.35)
    grid[["grid_id", "district_name", "emergence_index", "suppression_index", "grid_emergence_type"]].to_csv(
        SRC / f"{OUT_STEM}_panel_b_source.csv", index=False, encoding="utf-8-sig"
    )


def render_panel_b(data: dict[str, object]) -> None:
    fig, ax = fixed_fig()
    draw_phase_space_axis(ax, data, "b")
    save_figure(fig, panel_path("b", "phase_space"), dpi=600, tight=False)


def render_panel_c(data: dict[str, object]) -> None:
    fig, ax = fixed_fig()
    fig.subplots_adjust(left=0.29, right=0.975, bottom=0.16, top=0.82)
    matched = data["matched"].copy()
    order = [
        "suppression_index",
        "rule_silence_norm",
        "has_positive_rule",
        "primary_positive_rule_count",
    ]
    labels = {
        "suppression_index": "suppression",
        "rule_silence_norm": "rule silence",
        "has_positive_rule": "visible rule",
        "primary_positive_rule_count": "primary rule\ncount",
    }
    matched = matched[matched["outcome"].isin(order)].copy()
    matched["outcome"] = pd.Categorical(matched["outcome"], categories=order, ordered=True)
    matched = matched.sort_values("outcome")
    y = np.arange(len(matched))
    ax.barh(y - 0.16, matched["matched_control_mean"], height=0.27, color="#9fb1b6", alpha=0.68, label="matched controls")
    ax.barh(y + 0.16, matched["treated_mean"], height=0.27, color=FRONTIER, alpha=0.76, label="frontier grids")
    ax.hlines(y, matched["matched_control_mean"], matched["treated_mean"], color="#7a7168", lw=0.65, alpha=0.55, zorder=4)
    for _, row in matched.iterrows():
        yi = list(matched["outcome"]).index(row["outcome"])
        diff = row["difference_treated_minus_control"]
        ax.text(
            max(row["matched_control_mean"], row["treated_mean"]) + 0.018,
            yi,
            f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}",
            ha="left",
            va="center",
            fontsize=3.9,
            color=FRONTIER if diff > 0 else MUTED,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([labels[o] for o in matched["outcome"]])
    ax.set_xlabel("mean value after nearest-neighbour matching")
    ax.set_xlim(-0.02, max(matched[["treated_mean", "matched_control_mean"]].max()) + 0.18)
    ax.invert_yaxis()
    clean_axis(ax)
    panel_label(ax, "c", "matched diagnostic: rule-silence burden")
    ax.text(
        0.02,
        -0.22,
        "diagnostic comparison; not a causal effect estimate",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=3.45,
        color=MUTED,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=3.6, handletextpad=0.35)
    matched.to_csv(SRC / f"{OUT_STEM}_panel_c_source.csv", index=False, encoding="utf-8-sig")
    save_figure(fig, panel_path("c", "matched_diagnostic"), dpi=600, tight=False)


def draw_district_burden_axis(
    ax: plt.Axes,
    data: dict[str, object],
    letter: str = "d",
    title_y: float = 1.12,
) -> None:
    grid = data["grid"].copy()
    hosts = data["host_g"].copy()
    rules = data["rules"].copy()
    district = (
        grid.groupby("district_name")
        .agg(
            grid_count=("grid_id", "count"),
            suppressed_frontier_grids=("grid_emergence_type", lambda s: int((s == "suppressed_emergence_frontier").sum())),
            suppression_index_mean=("suppression_index", "mean"),
            p90_suppression_index=("suppression_index", lambda s: float(np.nanpercentile(s, 90))),
        )
        .reset_index()
    )
    host_summary = (
        hosts[hosts["liminal_class"].isin(["rule_liminal_threshold_core", "ecology_ready_rule_liminal_frontier"])]
        .groupby("district_name")
        .size()
        .rename("liminal_or_frontier_hosts")
        .reset_index()
    )
    grid_district = grid[["grid_id", "district_name"]].drop_duplicates()
    rule_summary = (
        rules[["grid_id"]]
        .merge(grid_district, on="grid_id", how="left")
        .groupby("district_name")
        .size()
        .rename("positive_rule_count")
        .reset_index()
    )
    district = district.merge(host_summary, on="district_name", how="left").merge(rule_summary, on="district_name", how="left")
    district[["liminal_or_frontier_hosts", "positive_rule_count"]] = district[["liminal_or_frontier_hosts", "positive_rule_count"]].fillna(0)
    district = district.set_index("district_name").reindex(DISTRICT_ORDER).reset_index()
    district["district_en"] = district["district_name"].map(DISTRICT_EN).fillna(district["district_name"])
    cols = [
        ("suppressed_frontier_grids", "frontier grids"),
        ("suppression_index_mean", "mean suppression"),
        ("p90_suppression_index", "p90 suppression"),
        ("liminal_or_frontier_hosts", "liminal/frontier hosts"),
        ("positive_rule_count", "visible rules"),
    ]
    raw = district[[c for c, _ in cols]].astype(float)
    scaled = raw.copy()
    for c in raw.columns:
        if "index" in c:
            scaled[c] = raw[c] / max(raw[c].max(), 1e-9)
        else:
            scaled[c] = np.log1p(raw[c]) / max(np.log1p(raw[c]).max(), 1e-9)
    ax.imshow(scaled.values, cmap=FINGER_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(district)))
    ax.set_yticklabels(district["district_en"])
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([label for _, label in cols])
    ax.tick_params(axis="x", labeltop=False, labelbottom=True, length=0, pad=2, labelsize=3.45)
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)
        tick.set_ha("right")
    ax.tick_params(axis="y", length=0)
    for i in range(len(district)):
        for j, (col, _) in enumerate(cols):
            val = raw.iloc[i][col]
            text = f"{val:.2f}" if "index" in col else f"{int(round(val)):,}"
            ax.text(j, i, text, ha="center", va="center", fontsize=3.15, color=INK if scaled.iloc[i][col] < 0.72 else "white")
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(district), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.55)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.0, title_y, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(
        0.075,
        title_y + 0.005,
        "district burden of suppressed emergence",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        1.0,
        -0.18,
        "cell shade uses within-column log/relative scaling; numbers are raw counts or means",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=3.35,
        color=MUTED,
    )
    district.to_csv(SRC / f"{OUT_STEM}_panel_d_source.csv", index=False, encoding="utf-8-sig")


def render_panel_d(data: dict[str, object]) -> None:
    fig, ax = fixed_fig()
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.18, top=0.78)
    draw_district_burden_axis(ax, data, "d")
    save_figure(fig, panel_path("d", "district_burden"), dpi=600, tight=False)


def render_composite() -> None:
    # The standalone panels are the official single-panel exports. For the
    # composite, use a fixed direct layout rather than a screenshot contact sheet.
    files = [
        panel_path("a0", "city_overview").with_suffix(".png"),
        panel_path("b", "phase_space").with_suffix(".png"),
        panel_path("c", "matched_diagnostic").with_suffix(".png"),
        panel_path("d", "district_burden").with_suffix(".png"),
    ]
    imgs = [Image.open(f).convert("RGB") for f in files]
    w, h = imgs[0].size
    title_h = 220
    gap = 60
    margin = 70
    canvas = Image.new("RGB", (2 * w + gap + 2 * margin, 2 * h + gap + title_h + margin), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 28), "Fig. 4 | Suppressed-emergence frontiers diagnose delayed rule translation", fill=(35, 33, 31))
    draw.text(
        (margin, 78),
        "Frontier grids combine companion-animal ecology, rule-liminal host potential and rule silence; matched diagnostics bound the claim.",
        fill=(78, 74, 69),
    )
    positions = [
        (margin, title_h),
        (margin + w + gap, title_h),
        (margin, title_h + h + gap),
        (margin + w + gap, title_h + h + gap),
    ]
    for img, pos in zip(imgs, positions):
        canvas.paste(img, pos)
    base = COMP / f"{OUT_STEM}_composite"
    canvas.save(base.with_suffix(".png"), dpi=(520, 520))
    # Keep vector/PDF outputs for package consistency; they contain the raster
    # composite because the panel-level SVG/PDF files preserve editability.
    fig, ax = plt.subplots(figsize=(7.25, 5.7))
    ax.imshow(canvas)
    ax.set_axis_off()
    save_figure(fig, base, dpi=520, tight=True)


def contact_sheet() -> None:
    files = [
        panel_path("a0", "city_overview").with_suffix(".png"),
        COMP / f"{OUT_STEM}_panel_a_district_zoom_atlas.png",
        panel_path("b", "phase_space").with_suffix(".png"),
        panel_path("c", "matched_diagnostic").with_suffix(".png"),
        panel_path("d", "district_burden").with_suffix(".png"),
        COMP / f"{OUT_STEM}_composite.png",
    ]
    thumbs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        im.thumbnail((440, 360), Image.LANCZOS)
        canvas = Image.new("RGB", (460, 392), "white")
        canvas.paste(im, ((460 - im.width) // 2, 12))
        ImageDraw.Draw(canvas).text((10, 368), f.name, fill=(35, 35, 35))
        thumbs.append(canvas)
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 460, rows * 392), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * 460, (i // cols) * 392))
    sheet.save(PREVIEWS / f"{OUT_STEM}_contact_sheet.png", dpi=(180, 180))


def main() -> None:
    data = load()
    render_panel_a(data)
    render_panel_b(data)
    render_panel_c(data)
    render_panel_d(data)
    render_composite()
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
