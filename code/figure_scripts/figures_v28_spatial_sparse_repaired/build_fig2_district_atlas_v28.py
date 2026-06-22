#!/usr/bin/env python3
"""Build a district-by-district zoom atlas for v28 Fig. 2.

The figure-facing rule is strict: every district panel is exported separately,
and the 3x3 atlas composite is an additional asset rather than a replacement.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import box

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from build_fig2_v28 import (  # noqa: E402
    BLUE_DARK,
    BUILDING,
    COMP,
    DISTRICT_EN,
    GOLD,
    GREEN,
    HOST_LABELS,
    GRID,
    INK,
    LAND,
    LIMINAL_LABELS,
    LIMINAL_ORDER,
    MUTED,
    PANELS,
    PED,
    PREVIEWS,
    PUBLIC,
    ROAD,
    ROOT,
    RUST,
    SERVICE_CMAP,
    SILENT,
    SRC,
    SUPPRESS_CMAP,
    TEXT,
    clean_axis,
    load_data,
    numeric,
    panel_label,
    point_size,
    save_figure,
)


DISTRICT_ORDER = ["光明区", "龙华区", "龙岗区", "宝安区", "南山区", "坪山区", "福田区", "罗湖区", "盐田区"]
DISTRICT_LABEL = {k: DISTRICT_EN.get(k, k) for k in DISTRICT_ORDER}
FINGER_CMAP = LinearSegmentedColormap.from_list("fingerprint", ["#fffaf0", "#f5e9c9", "#e4bf68", "#d88973"])
DISTRICT_PANEL_ASPECT = 1.18


def draw_atlas_legend(ax: plt.Axes, mode: str = "fig2") -> None:
    ax.set_axis_off()
    items = [
        ("patch", "#dbe7d8", "public space"),
        ("patch", BUILDING, "building texture"),
        ("line", ROAD, "road network"),
        ("line", PED, "pedestrian path"),
        ("point", "#d88973" if mode == "fig4" else RUST, "rule-liminal/frontier host"),
        ("point", BLUE_DARK, "pet-service point"),
        ("point", GREEN, "visible rule record"),
    ]
    if mode == "fig4":
        items.insert(4, ("patch", "#d88973", "suppressed-frontier grid"))
    else:
        items.insert(4, ("patch", "#9fb8c5", "service/suppression grid"))
    x_positions = [0.020, 0.265, 0.505, 0.745]
    y_positions = [0.68, 0.28]
    for idx, (kind, color, label) in enumerate(items):
        x = x_positions[idx % 4]
        y = y_positions[idx // 4]
        if kind == "patch":
            ax.add_patch(Rectangle((x, y - 0.095), 0.034, 0.19, transform=ax.transAxes, facecolor=color, edgecolor="#d0cbc2", linewidth=0.25, alpha=0.72))
        elif kind == "line":
            ax.plot([x, x + 0.038], [y, y], transform=ax.transAxes, color=color, linewidth=1.35, alpha=0.70)
        else:
            ax.scatter([x + 0.019], [y], transform=ax.transAxes, s=21, color=color, edgecolor="white", linewidth=0.30, alpha=0.92)
        ax.text(x + 0.050, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=4.15, color=MUTED)


def district_extent(district_grid: gpd.GeoDataFrame, pad_ratio: float = 0.08) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = district_grid.total_bounds
    w, h = maxx - minx, maxy - miny
    pad = max(w, h) * pad_ratio
    return normalize_extent_aspect((minx - pad, miny - pad, maxx + pad, maxy + pad), DISTRICT_PANEL_ASPECT)


def normalize_extent_aspect(
    extent: tuple[float, float, float, float],
    target_aspect: float,
) -> tuple[float, float, float, float]:
    """Expand an extent so every exported district panel uses the same frame ratio."""
    minx, miny, maxx, maxy = extent
    w = maxx - minx
    h = maxy - miny
    current = w / h if h else target_aspect
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    if current > target_aspect:
        new_h = w / target_aspect
        miny = cy - new_h / 2
        maxy = cy + new_h / 2
    else:
        new_w = h * target_aspect
        minx = cx - new_w / 2
        maxx = cx + new_w / 2
    return minx, miny, maxx, maxy


def draw_texture_light(ax: plt.Axes, data: dict[str, object], extent: tuple[float, float, float, float]) -> None:
    minx, miny, maxx, maxy = extent
    bbox = box(minx, miny, maxx, maxy)
    ax.set_facecolor("#fbfaf7")
    for key, color, lw, alpha, z in [
        ("public", PUBLIC, 0.0, 0.58, 1),
        ("buildings", BUILDING, 0.0, 0.46, 2),
        ("roads", ROAD, 0.20, 0.36, 3),
        ("ped", PED, 0.25, 0.45, 4),
    ]:
        layer = data[key]
        sub = layer[layer.intersects(bbox)]
        if len(sub) == 0:
            continue
        if key in {"roads", "ped"}:
            sub.plot(ax=ax, color=color, linewidth=lw, alpha=alpha, zorder=z)
        else:
            sub.plot(ax=ax, color=color, edgecolor="none", alpha=alpha, zorder=z)


def district_stats(data: dict[str, object]) -> pd.DataFrame:
    grid = data["grid"].copy()
    hosts = data["hosts"].copy()
    pet = data["pet"].copy()
    rules = data["rules"].copy()

    priority_classes = ["rule_liminal_threshold_core", "ecology_ready_rule_liminal_frontier"]
    g = (
        grid.groupby("district_name", as_index=False)
        .agg(
            cells=("grid_id", "count"),
            suppression=("suppression_index", "mean"),
        )
    )
    h = hosts.groupby("district_name", as_index=False).size().rename(columns={"size": "host_candidates"})
    frontier = (
        hosts[hosts["liminal_class"].isin(priority_classes)]
        .groupby("district_name", as_index=False)
        .size()
        .rename(columns={"size": "frontier"})
    )
    silent = (
        hosts[hosts["liminal_class"].eq("ecology_present_rule_silent")]
        .groupby("district_name", as_index=False)
        .size()
        .rename(columns={"size": "silent"})
    )
    pet_count = pet.groupby("district_name", as_index=False).size().rename(columns={"size": "service"})
    rule_count = (
        rules.groupby("grid_id", as_index=False)
        .size()
        .merge(grid[["grid_id", "district_name"]], on="grid_id", how="left")
        .dropna(subset=["district_name"])
        .groupby("district_name", as_index=False)["size"]
        .sum()
        .rename(columns={"size": "rules"})
    )
    out = (
        g.merge(h, on="district_name", how="left")
        .merge(frontier, on="district_name", how="left")
        .merge(silent, on="district_name", how="left")
        .merge(pet_count, on="district_name", how="left")
        .merge(rule_count, on="district_name", how="left")
    )
    for col in ["host_candidates", "frontier", "silent", "service", "rules"]:
        out[col] = out[col].fillna(0).astype(int)
    out["district_en"] = out["district_name"].map(DISTRICT_LABEL).fillna(out["district_name"])
    out.to_csv(SRC / "fig2_district_atlas_summary_source.csv", index=False, encoding="utf-8-sig")
    return out


def draw_district_panel(
    ax: plt.Axes,
    data: dict[str, object],
    district: str,
    letter: str | None,
    title_prefix: str = "",
    compact: bool = False,
) -> dict[str, float | int | str]:
    grid = data["grid"].copy()
    boundary = data["boundary"]
    pet = data["pet"]
    rules = data["rules"]
    hosts = data["host_g"]

    dg = grid[grid["district_name"].eq(district)].copy()
    if len(dg) == 0:
        raise ValueError(f"Missing district: {district}")
    ext = district_extent(dg, 0.08 if compact else 0.10)
    bbox = box(*ext)

    draw_texture_light(ax, data, ext)
    grid[grid.intersects(bbox)].plot(ax=ax, color=LAND, edgecolor=GRID, linewidth=0.012, alpha=0.46, zorder=5, rasterized=True)

    dg["service"] = numeric(dg["pet_service_exposure_500m"])
    dg[dg["service"] > 0].plot(
        ax=ax,
        column="service",
        cmap=SERVICE_CMAP,
        linewidth=0,
        alpha=0.70,
        norm=Normalize(vmin=0, vmax=max(np.nanpercentile(grid["pet_service_exposure_500m"], 99), 0.1)),
        zorder=7,
        rasterized=True,
    )
    dg["suppression"] = numeric(dg["suppression_index"])
    dg[dg["suppression"] > 0].plot(
        ax=ax,
        column="suppression",
        cmap=SUPPRESS_CMAP,
        linewidth=0,
        alpha=0.32,
        norm=Normalize(vmin=0, vmax=max(np.nanpercentile(grid["suppression_index"], 98), 0.1)),
        zorder=8,
        rasterized=True,
    )

    boundary.boundary.plot(ax=ax, color="#aaa49b", linewidth=0.11 if compact else 0.16, alpha=0.46, zorder=12)
    dg.dissolve().boundary.plot(ax=ax, color="#5d5750", linewidth=0.26 if compact else 0.34, alpha=0.76, zorder=30)

    hsub = hosts[hosts["district_name"].eq(district)].copy()
    priority = hsub[hsub["liminal_class"].isin(["rule_liminal_threshold_core", "ecology_ready_rule_liminal_frontier"])]
    silent = hsub[hsub["liminal_class"].eq("ecology_present_rule_silent")]
    if len(silent):
        silent.sample(min(len(silent), 850 if compact else 1300), random_state=21).plot(
            ax=ax, color=SILENT, markersize=2.2 if compact else 3.0, alpha=0.22, linewidth=0, zorder=13
        )
    if len(priority):
        priority.plot(
            ax=ax,
            color=RUST,
            markersize=4.2 if compact else 5.6,
            alpha=0.46,
            edgecolor="white",
            linewidth=0.28,
            zorder=14,
        )
    psub = pet[pet["district_name"].eq(district)]
    if len(psub):
        psub.plot(ax=ax, color=BLUE_DARK, markersize=3.5 if compact else 4.6, alpha=0.28, linewidth=0, zorder=15)
    rsub = rules[rules["grid_id"].isin(dg["grid_id"])]
    if len(rsub):
        rsub.plot(
            ax=ax,
            color=GREEN,
            markersize=24 if compact else 34,
            alpha=0.94,
            edgecolor="white",
            linewidth=0.22,
            zorder=18,
        )

    ax.set_xlim(ext[0], ext[2])
    ax.set_ylim(ext[1], ext[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#d8d3ca")
        spine.set_linewidth(0.34 if compact else 0.42)
    name = DISTRICT_LABEL[district]
    if letter:
        ax.text(0.0, 1.018, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=INK)
        ax.text(0.088, 1.024, f"{title_prefix}{name}", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, fontweight="bold", color=INK)
    else:
        ax.text(0.01, 1.02, f"{title_prefix}{name}", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2, fontweight="bold", color=INK)

    stats = {
        "district_name": district,
        "district_en": name,
        "grid_cells": int(len(dg)),
        "pet_service_points": int(len(psub)),
        "positive_rule_points": int(len(rsub)),
        "host_candidates": int(len(hsub)),
        "priority_frontier_hosts": int(len(priority)),
        "silent_hosts": int(len(silent)),
    }
    ax.text(
        0.012,
        0.025,
        f"hosts {stats['host_candidates']:,} | frontier {stats['priority_frontier_hosts']:,} | rules {stats['positive_rule_points']:,}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=3.9 if compact else 4.2,
        color=MUTED,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
        zorder=40,
    )
    return stats


def render_single_district_panels(data: dict[str, object]) -> pd.DataFrame:
    rows = []
    letters = list("abcdefghi")
    for district, letter in zip(DISTRICT_ORDER, letters):
        fig, ax = plt.subplots(figsize=(3.3, 2.75))
        fig.subplots_adjust(left=0.035, right=0.985, bottom=0.070, top=0.890)
        stats = draw_district_panel(ax, data, district, letter, compact=False)
        rows.append(stats)
        stem = f"fig2_district_panel_{letter}_{DISTRICT_LABEL[district].lower()}_v28"
        save_figure(fig, PANELS / stem, dpi=600, tight=False)
        pd.DataFrame([stats]).to_csv(SRC / f"{stem}_source.csv", index=False, encoding="utf-8-sig")
    out = pd.DataFrame(rows)
    out.to_csv(SRC / "fig2_district_atlas_panel_sources.csv", index=False, encoding="utf-8-sig")
    return out


def city_extent(boundary: gpd.GeoDataFrame, pad: float = 4200) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = boundary.total_bounds
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def draw_city_overview_panel(ax: plt.Axes, data: dict[str, object], letter: str = "a0") -> None:
    grid = data["grid"].copy()
    boundary = data["boundary"]
    rules = data["rules"]
    grid["service"] = numeric(grid["pet_service_exposure_500m"])
    grid["suppression"] = numeric(grid["suppression_index"])
    grid.plot(ax=ax, color=LAND, edgecolor=GRID, linewidth=0.008, alpha=0.45, zorder=0, rasterized=True)
    grid[grid["service"] > 0].plot(
        ax=ax,
        column="service",
        cmap=SERVICE_CMAP,
        linewidth=0,
        alpha=0.62,
        norm=Normalize(vmin=0, vmax=max(np.nanpercentile(grid["pet_service_exposure_500m"], 99), 0.1)),
        zorder=5,
        rasterized=True,
    )
    grid[grid["suppression"] > 0].plot(
        ax=ax,
        column="suppression",
        cmap=SUPPRESS_CMAP,
        linewidth=0,
        alpha=0.36,
        norm=Normalize(vmin=0, vmax=max(np.nanpercentile(grid["suppression_index"], 98), 0.1)),
        zorder=8,
        rasterized=True,
    )
    boundary.boundary.plot(ax=ax, color="#514b45", linewidth=0.36, alpha=0.70, zorder=20)
    for district in DISTRICT_ORDER:
        dg = grid[grid["district_name"].eq(district)]
        if len(dg):
            minx, miny, maxx, maxy = district_extent(dg, 0.10)
            ax.add_patch(Rectangle((minx, miny), maxx - minx, maxy - miny, fill=False, edgecolor="#8c8177", linewidth=0.18, alpha=0.45, zorder=25))
    if len(rules):
        rules.plot(ax=ax, color=GREEN, markersize=10, alpha=0.85, edgecolor="white", linewidth=0.10, zorder=30)
    minx, miny, maxx, maxy = city_extent(boundary)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#d8d3ca")
        spine.set_linewidth(0.38)
    ax.text(0.0, 1.02, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.4, fontweight="bold", color=INK)
    ax.text(0.105, 1.025, "Shenzhen overview: service ecology, rule visibility and district zoom windows", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.4, fontweight="bold", color=INK)
    ax.text(0.012, 0.025, "blue = pet-service exposure; rust = suppression; green = visible rule records; grey boxes = district zoom panels", transform=ax.transAxes, ha="left", va="bottom", fontsize=3.6, color=MUTED, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.32}, zorder=40)


def render_city_overview_panel(data: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.09, top=0.86)
    draw_city_overview_panel(ax, data, "a0")
    save_figure(fig, PANELS / "fig2_panel_a0_city_overview_v28", dpi=600, tight=False)


def draw_fingerprint_strip(
    ax: plt.Axes,
    summary: pd.DataFrame,
    letter: str = "j",
    title_y: float = 1.24,
) -> None:
    summary = summary.copy()
    summary["district_en"] = summary["district_name"].map(DISTRICT_LABEL)
    summary = summary.set_index("district_name").loc[DISTRICT_ORDER].reset_index()
    cols = [
        ("host_candidates", "host\ncandidates"),
        ("frontier", "frontier\nhosts"),
        ("silent", "silent\nhosts"),
        ("service", "pet-service\nPOI"),
        ("rules", "visible\nrules"),
        ("suppression", "mean\nsuppression"),
    ]
    raw = summary[[c for c, _ in cols]].astype(float)
    scaled = raw.copy()
    for c in raw.columns:
        if c == "suppression":
            mx = max(raw[c].max(), 1e-9)
            scaled[c] = raw[c] / mx
        else:
            scaled[c] = np.log1p(raw[c]) / max(np.log1p(raw[c]).max(), 1e-9)
    ax.imshow(scaled.values, cmap=FINGER_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(summary)))
    ax.set_yticklabels(summary["district_en"])
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([label for _, label in cols])
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, length=0, pad=2)
    ax.tick_params(axis="y", length=0)
    for i in range(len(summary)):
        for j, (col, _) in enumerate(cols):
            val = raw.iloc[i][col]
            text = f"{val:.2f}" if col == "suppression" else f"{int(round(val)):,}"
            ax.text(j, i, text, ha="center", va="center", fontsize=3.25, color=INK if scaled.iloc[i][col] < 0.72 else "white")
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(summary), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.55)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.0, title_y, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=INK)
    ax.text(0.075, title_y + 0.01, "district evidence fingerprint", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, fontweight="bold", color=INK)
    ax.text(
        1.0,
        -0.24,
        "cell shade uses within-column log/relative scaling; numbers are raw counts except mean suppression",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=3.5,
        color=MUTED,
    )


def draw_stacked_host_panel(
    ax: plt.Axes,
    data: dict[str, object],
    letter: str = "j",
    legend_anchor: tuple[float, float] = (0.5, -0.38),
) -> None:
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
        ax.barh(
            y,
            vals,
            left=left,
            color=colors[col],
            height=0.58,
            edgecolor="white",
            linewidth=0.22,
            label=LIMINAL_LABELS[col],
        )
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([HOST_LABELS.get(i, i) for i in share.index])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50", "100"])
    ax.set_xlabel("share of host candidates (%)")
    ax.invert_yaxis()
    clean_axis(ax)
    panel_label(ax, letter, "host-type rule-liminal composition")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=legend_anchor,
        ncol=3,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.58,
        handletextpad=0.32,
        fontsize=3.35,
    )
    share.reset_index().to_csv(SRC / "fig2_district_atlas_stacked_host_source.csv", index=False, encoding="utf-8-sig")


def render_stacked_host_panel(data: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.30, top=0.84)
    draw_stacked_host_panel(ax, data, "j")
    save_figure(fig, PANELS / "fig2_panel_j_host_liminal_stacked_v28", dpi=600, tight=False)


def render_fingerprint_panel(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.78)
    draw_fingerprint_strip(ax, summary, "k", title_y=1.12)
    save_figure(fig, PANELS / "fig2_panel_k_district_evidence_fingerprint_v28", dpi=600, tight=False)


def render_composite(data: dict[str, object], summary: pd.DataFrame) -> None:
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
    draw_atlas_legend(axleg, "fig2")
    ax0 = fig.add_subplot(gs[1, :])
    draw_city_overview_panel(ax0, data, "a0")
    letters = list("abcdefghi")
    for i, (district, letter) in enumerate(zip(DISTRICT_ORDER, letters)):
        ax = fig.add_subplot(gs[(i // 3) + 2, (i % 3) * 2 : (i % 3 + 1) * 2])
        draw_district_panel(ax, data, district, letter, compact=True)
    axj = fig.add_subplot(gs[5:, :3])
    draw_stacked_host_panel(axj, data, "j", legend_anchor=(0.5, -0.22))
    axk = fig.add_subplot(gs[5:, 3:])
    draw_fingerprint_strip(axk, summary, "k", title_y=1.08)
    fig.text(
        0.015,
        0.985,
        "Fig. 2 district atlas | rule-liminal urban texture across Shenzhen",
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.015,
        0.962,
        "All districts are shown at local scale; red-brown points mark rule-liminal frontier hosts, blue points pet services and green points visible rules.",
        ha="left",
        va="top",
        fontsize=5.0,
        color=TEXT,
    )
    save_figure(fig, COMP / "fig2_district_atlas_composite_v28", dpi=520, tight=False)


def contact_sheet() -> None:
    files = sorted(PANELS.glob("fig2_district_panel_*_v28.png")) + [COMP / "fig2_district_atlas_composite_v28.png"]
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
    sheet.save(PREVIEWS / "fig2_district_atlas_v28_contact_sheet.png", dpi=(180, 180))


def main() -> None:
    data = load_data()
    render_city_overview_panel(data)
    summary = district_stats(data)
    panel_stats = render_single_district_panels(data)
    render_stacked_host_panel(data)
    render_fingerprint_panel(summary)
    render_composite(data, summary)
    contact_sheet()
    manifest_rows = []
    for f in sorted(PANELS.glob("fig2_panel_a0_city_overview_v28.*")):
        manifest_rows.append({"asset": f.relative_to(ROOT).as_posix(), "role": "city_overview_panel"})
    for f in sorted(PANELS.glob("fig2_district_panel_*_v28.*")):
        manifest_rows.append({"asset": f.relative_to(ROOT).as_posix(), "role": "district_panel"})
    for f in sorted(PANELS.glob("fig2_panel_j_host_liminal_stacked_v28.*")):
        manifest_rows.append({"asset": f.relative_to(ROOT).as_posix(), "role": "stacked_host_panel"})
    for f in sorted(PANELS.glob("fig2_panel_k_district_evidence_fingerprint_v28.*")):
        manifest_rows.append({"asset": f.relative_to(ROOT).as_posix(), "role": "district_evidence_fingerprint_panel"})
    for f in sorted(COMP.glob("fig2_district_atlas_composite_v28.*")):
        manifest_rows.append({"asset": f.relative_to(ROOT).as_posix(), "role": "district_atlas_composite"})
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(SRC / "fig2_district_atlas_export_manifest_v28.csv", index=False, encoding="utf-8-sig")
    print(panel_stats.to_string(index=False))
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
