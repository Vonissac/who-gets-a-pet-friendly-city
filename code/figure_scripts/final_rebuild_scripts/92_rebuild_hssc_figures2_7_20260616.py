#!/usr/bin/env python3
"""Rebuild HSSC Figures 2-7 from located source panels/functions.

This script does not recompute study metrics or edit source data. It only
re-composes already located figure panels from the project figure scripts into
HSSC-facing figure files with corrected numbering, unified typography, white
canvas, readable legends, and explicit source mapping.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hssc_figures_quality_v3_20260616"
FIGS = OUT / "figures"
PANELS = OUT / "panels"
SRC = OUT / "source_maps"
AUDIT = OUT / "audit"
for folder in [FIGS, PANELS, SRC, AUDIT]:
    folder.mkdir(parents=True, exist_ok=True)

V28_CODE = ROOT / "submission_package" / "code" / "figures_v28_spatial_sparse_repaired"
V29_CODE = ROOT / "submission_package" / "code" / "figures_v29_expanded_system"
V30_CODE = ROOT / "submission_package" / "code" / "figures_v30_map_evidence_vignettes"
V32_SCRIPT = ROOT / "scripts" / "82_build_completed_validation_figure_v32.py"
V27_CODE = ROOT / "submission_package" / "code" / "figures_v27_experiment_adjusted_dense"


def import_module(name: str, path: Path, prepend: Path | None = None):
    if prepend is not None:
        sys.path.insert(0, str(prepend))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fig2_base = import_module("fig2_v28_base", V28_CODE / "build_fig2_v28.py", V28_CODE)
fig2_atlas = import_module("fig2_atlas_v28", V28_CODE / "build_fig2_district_atlas_v28.py", V28_CODE)
fig3_rule = import_module("fig3_rule_v28", V28_CODE / "build_fig3_rule_liminal_venues_v28.py", V28_CODE)
fig4_frontier = import_module("fig4_frontier_v28", V28_CODE / "build_fig4_suppressed_frontier_v28.py", V28_CODE)
fig5_topology = import_module("fig5_topology_v28", V28_CODE / "build_fig5_topology_threshold_v28.py", V28_CODE)
fig27_model = import_module("fig27_model_v27", V27_CODE / "build_fig1_fig6_v27_experiment_adjusted_dense.py", V27_CODE)
fig29 = import_module("fig29_expanded", V29_CODE / "build_v29_expanded_system.py", V29_CODE)
fig30 = import_module("fig30_vignettes", V30_CODE / "build_v30_map_evidence_vignettes.py", V30_CODE)
fig32 = import_module("fig32_validation", V32_SCRIPT, ROOT / "scripts")


INK = "#242321"
TEXT = "#504a43"
MUTED = "#766f66"
RULE = "#2b6777"
WHITE = "white"


def set_hssc_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.4,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.58,
            "axes.edgecolor": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.dpi": 600,
        }
    )


def hssc_header(fig: plt.Figure, number: int, title: str, subtitle: str) -> None:
    fig.text(
        0.018,
        0.984,
        f"Fig. {number} | {title}",
        ha="left",
        va="top",
        fontsize=13.2,
        fontweight="bold",
        color=INK,
    )
    fig.text(0.018, 0.958, subtitle, ha="left", va="top", fontsize=7.0, color=TEXT)
    fig.lines.append(
        plt.Line2D([0.018, 0.982], [0.992, 0.992], transform=fig.transFigure, color=RULE, lw=0.75)
    )


def section_label(fig: plt.Figure, y: float, text: str) -> None:
    fig.text(0.018, y, text, ha="left", va="center", fontsize=7.4, fontweight="bold", color=INK)
    fig.patches.append(
        plt.Rectangle(
            (0.018, y - 0.010),
            0.006,
            0.020,
            transform=fig.transFigure,
            facecolor=RULE,
            edgecolor="none",
            zorder=5,
        )
    )


def save_fig(fig: plt.Figure, stem: Path, dpi: int = 600) -> dict[str, Any]:
    paths = {}
    for ext in ["png", "pdf", "svg"]:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=dpi)
        paths[ext] = path
    tif = stem.with_suffix(".tif")
    plt.close(fig)
    rgb = Image.open(paths["png"]).convert("RGB")
    rgb.save(paths["png"], dpi=(dpi, dpi))
    rgb.save(tif, dpi=(dpi, dpi), compression="tiff_lzw")
    return image_info(paths["png"]) | {"pdf": str(paths["pdf"]), "svg": str(paths["svg"]), "tif": str(tif)}


def image_info(path: Path) -> dict[str, Any]:
    im = Image.open(path)
    return {"png": str(path), "width_px": im.width, "height_px": im.height, "dpi": im.info.get("dpi")}


def add_figure_legend(ax: plt.Axes, items: list[tuple[str, str]]) -> None:
    ax.set_axis_off()
    ax.text(0, 0.92, "Figure legend", ha="left", va="top", fontsize=6.2, fontweight="bold", color=INK)
    cols = 3 if len(items) > 5 else 2
    for i, (color, label) in enumerate(items):
        x = (i % cols) / cols
        y = 0.62 - 0.28 * (i // cols)
        ax.add_patch(plt.Rectangle((x, y - 0.055), 0.024, 0.11, transform=ax.transAxes, facecolor=color, edgecolor="#d4cec3", lw=0.35))
        ax.text(x + 0.034, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=5.0, color=TEXT)


def draw_frontier_overview_on_axis(ax: plt.Axes, data: dict[str, object], letter: str = "a") -> None:
    """Draw Fig. 4 overview using the v28 source logic, but on caller axis."""
    grid = data["grid"].copy()
    boundary = data["boundary"]
    grid.plot(ax=ax, color=fig4_frontier.LAND, edgecolor=fig4_frontier.GRID, linewidth=0.010, zorder=0, rasterized=True)
    grid[grid["grid_emergence_type"].eq("mixed_or_transitional_zone")].plot(
        ax=ax, color=fig4_frontier.TRANSITION, edgecolor="none", alpha=0.28, zorder=4, rasterized=True
    )
    grid[grid["grid_emergence_type"].eq("suppressed_emergence_frontier")].plot(
        ax=ax, color=fig4_frontier.FRONTIER, edgecolor="none", alpha=0.72, zorder=8, rasterized=True
    )
    boundary.boundary.plot(ax=ax, color="#514b45", linewidth=0.44, alpha=0.78, zorder=20)
    for district in fig4_frontier.DISTRICT_ORDER:
        dg = grid[grid["district_name"].eq(district)]
        if len(dg) == 0:
            continue
        minx, miny, maxx, maxy = fig4_frontier.district_extent(dg, 0.10)
        ax.add_patch(
            plt.Rectangle(
                (minx, miny),
                maxx - minx,
                maxy - miny,
                fill=False,
                edgecolor="#8c8177",
                linewidth=0.22,
                alpha=0.52,
                zorder=25,
            )
        )
    minx, miny, maxx, maxy = fig4_frontier.city_extent(boundary)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig4_frontier.light_frame(ax)
    fig4_frontier.panel_label(ax, letter, "citywide frontier anchor")
    ax.text(
        0.012,
        0.025,
        "coral = 503 sparse-repaired frontier grids; grey boxes = district zoom panels",
        transform=ax.transAxes,
        fontsize=4.2,
        color=fig4_frontier.MUTED,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
    )


def draw_matched_diagnostic_on_axis(ax: plt.Axes, data: dict[str, object], letter: str = "c") -> None:
    """Draw Fig. 4 matched diagnostic using the v28 source logic."""
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
    y = range(len(matched))
    ax.barh(
        [v - 0.16 for v in y],
        matched["matched_control_mean"],
        height=0.27,
        color="#9fb1b6",
        alpha=0.68,
        label="matched controls",
    )
    ax.barh(
        [v + 0.16 for v in y],
        matched["treated_mean"],
        height=0.27,
        color=fig4_frontier.FRONTIER,
        alpha=0.76,
        label="frontier grids",
    )
    ax.hlines(list(y), matched["matched_control_mean"], matched["treated_mean"], color="#7a7168", lw=0.65, alpha=0.55, zorder=4)
    outcomes = list(matched["outcome"])
    for _, row in matched.iterrows():
        yi = outcomes.index(row["outcome"])
        diff = row["difference_treated_minus_control"]
        ax.text(
            max(row["matched_control_mean"], row["treated_mean"]) + 0.018,
            yi,
            f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}",
            ha="left",
            va="center",
            fontsize=4.2,
            color=fig4_frontier.FRONTIER if diff > 0 else fig4_frontier.MUTED,
        )
    ax.set_yticks(list(y))
    ax.set_yticklabels([labels[o] for o in matched["outcome"]])
    ax.set_xlabel("mean value after nearest-neighbour matching")
    ax.set_xlim(-0.02, max(matched[["treated_mean", "matched_control_mean"]].max()) + 0.18)
    ax.invert_yaxis()
    fig4_frontier.clean_axis(ax)
    fig4_frontier.panel_label(ax, letter, "matched diagnostic: rule-silence burden")
    ax.text(
        0.02,
        -0.22,
        "diagnostic comparison; not a causal effect estimate",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=3.7,
        color=fig4_frontier.MUTED,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=3.9, handletextpad=0.35)


def source_record(figure_id: str, panels: list[dict[str, str]]) -> None:
    (SRC / f"{figure_id}_source_map.json").write_text(json.dumps(panels, indent=2), encoding="utf-8")


def figure_2() -> dict[str, Any]:
    set_hssc_style()
    data = fig2_base.load_data()
    summary = fig2_atlas.district_stats(data)
    fig = plt.figure(figsize=(7.25, 10.55))
    gs = GridSpec(
        7,
        6,
        figure=fig,
        height_ratios=[0.22, 1.04, 1, 1, 1, 0.82, 0.82],
        left=0.055,
        right=0.985,
        bottom=0.045,
        top=0.902,
        wspace=0.11,
        hspace=0.46,
    )
    hssc_header(
        fig,
        2,
        "District service-rule atlas across Shenzhen",
        "Citywide substrate, district summaries and local panels separate service ecology, candidate hosts and visible rule records.",
    )
    section_label(fig, 0.910, "Citywide overview and district summary")
    axleg = fig.add_subplot(gs[0, :])
    fig2_atlas.draw_atlas_legend(axleg, "fig2")
    ax0 = fig.add_subplot(gs[1, :2])
    fig2_atlas.draw_city_overview_panel(ax0, data, "a")
    axj = fig.add_subplot(gs[1, 2:4])
    fig2_atlas.draw_stacked_host_panel(axj, data, "b", legend_anchor=(0.5, -0.24))
    axk = fig.add_subplot(gs[1, 4:])
    fig2_atlas.draw_fingerprint_strip(axk, summary, "c", title_y=1.10)
    section_label(fig, 0.575, "District atlas: every district appears once")
    for i, district in enumerate(fig2_atlas.DISTRICT_ORDER):
        ax = fig.add_subplot(gs[(i // 3) + 2, (i % 3) * 2 : (i % 3 + 1) * 2])
        fig2_atlas.draw_district_panel(ax, data, district, chr(ord("d") + i), compact=True)
    stem = FIGS / "Figure_02_district_service_rule_atlas_HSSC_20260616"
    source_record(
        "Figure_02",
        [
            {"panel": "a", "source": "build_fig2_district_atlas_v28.draw_city_overview_panel"},
            {"panel": "b", "source": "build_fig2_district_atlas_v28.draw_stacked_host_panel"},
            {"panel": "c", "source": "build_fig2_district_atlas_v28.draw_fingerprint_strip"},
            {"panel": "d-l", "source": "build_fig2_district_atlas_v28.draw_district_panel; all 9 district panels"},
        ],
    )
    return {"figure": "Figure_02", **save_fig(fig, stem)}


def figure_3() -> dict[str, Any]:
    set_hssc_style()
    data = fig3_rule.load()
    fig = plt.figure(figsize=(7.25, 10.80))
    gs = GridSpec(
        5,
        6,
        figure=fig,
        height_ratios=[1.55, 1.0, 1.0, 1.0, 0.06],
        left=0.070,
        right=0.985,
        bottom=0.055,
        top=0.905,
        wspace=0.18,
        hspace=0.58,
    )
    hssc_header(
        fig,
        3,
        "Rule-liminal field and semantic rule evidence",
        "Candidate venues, host-type states and traceable rule records are shown without equating candidate signals with confirmed access.",
    )
    section_label(fig, 0.914, "Citywide candidate surface")
    axa = fig.add_subplot(gs[0, :])
    fig3_rule.draw_hero_map(axa, data, "a")
    section_label(fig, 0.612, "Candidate composition and rule-source structure")
    fig3_rule.draw_host_type_distribution(fig.add_subplot(gs[1, :3]), data, "b")
    fig3_rule.draw_class_composition(fig.add_subplot(gs[1, 3:]), data, "c", legend_anchor=(0.5, -0.24))
    fig3_rule.draw_rule_semantic_matrix(fig.add_subplot(gs[2, :3]), data, "d")
    fig3_rule.draw_district_fingerprint(fig.add_subplot(gs[2:, 3:]), data, "e", title_y=1.10)
    axleg = fig.add_subplot(gs[4, :])
    add_figure_legend(
        axleg,
        [
            (fig3_rule.LIMINAL_GOLD, "rule-liminal grid"),
            (fig3_rule.LIMINAL_DARK, "priority host candidate"),
            (fig3_rule.BLUE_DARK, "pet-service point"),
            (fig3_rule.GREEN, "traceable rule evidence"),
            (fig3_rule.LOW_SIGNAL, "low-signal host class"),
        ],
    )
    stem = FIGS / "Figure_03_rule_liminal_field_HSSC_20260616"
    source_record(
        "Figure_03",
        [
            {"panel": "a", "source": "build_fig3_rule_liminal_venues_v28.draw_hero_map"},
            {"panel": "b", "source": "build_fig3_rule_liminal_venues_v28.draw_host_type_distribution"},
            {"panel": "c", "source": "build_fig3_rule_liminal_venues_v28.draw_class_composition"},
            {"panel": "d", "source": "build_fig3_rule_liminal_venues_v28.draw_rule_semantic_matrix"},
            {"panel": "e", "source": "build_fig3_rule_liminal_venues_v28.draw_district_fingerprint"},
        ],
    )
    return {"figure": "Figure_03", **save_fig(fig, stem)}


def figure_4() -> dict[str, Any]:
    set_hssc_style()
    front = fig4_frontier.load()
    model = fig27_model.load_data()
    fig = plt.figure(figsize=(7.25, 9.20))
    gs = GridSpec(
        3,
        6,
        figure=fig,
        height_ratios=[1.08, 1.02, 0.82],
        left=0.075,
        right=0.985,
        bottom=0.065,
        top=0.900,
        wspace=0.18,
        hspace=0.58,
    )
    hssc_header(
        fig,
        4,
        "Suppressed-emergence frontier and evidence boundary",
        "Frontier diagnostics locate grid cells where service ecology and host potential are not matched by visible rule signals.",
    )
    section_label(fig, 0.910, "Frontier anchor and diagnostic contrast")
    draw_frontier_overview_on_axis(fig.add_subplot(gs[0, :3]), front, "a")
    fig4_frontier.draw_phase_space_axis(fig.add_subplot(gs[0, 3:]), front, "b")
    draw_matched_diagnostic_on_axis(fig.add_subplot(gs[1, :3]), front, "c")
    fig4_frontier.draw_district_burden_axis(fig.add_subplot(gs[1, 3:]), front, "d")
    section_label(fig, 0.300, "Large-sample boundary checks")
    fig27_model.p6a(fig.add_subplot(gs[2, :2]), model, "e")
    fig27_model.p6b(fig.add_subplot(gs[2, 2:4]), model, "f")
    fig27_model.p6c(fig.add_subplot(gs[2, 4:]), model, "g")
    stem = FIGS / "Figure_04_suppressed_frontier_boundary_HSSC_20260616"
    source_record(
        "Figure_04",
        [
            {"panel": "a", "source": "build_fig4_suppressed_frontier_v28.render_frontier_overview_panel logic, redrawn on composite axis"},
            {"panel": "b", "source": "build_fig4_suppressed_frontier_v28.draw_phase_space_axis"},
            {"panel": "c", "source": "build_fig4_suppressed_frontier_v28.render_panel_c logic, redrawn on composite axis"},
            {"panel": "d", "source": "build_fig4_suppressed_frontier_v28.draw_district_burden_axis"},
            {"panel": "e", "source": "build_fig1_fig6_v27_experiment_adjusted_dense.p6a"},
            {"panel": "f", "source": "build_fig1_fig6_v27_experiment_adjusted_dense.p6b"},
            {"panel": "g", "source": "build_fig1_fig6_v27_experiment_adjusted_dense.p6c"},
        ],
    )
    return {"figure": "Figure_04", **save_fig(fig, stem)}


def figure_5() -> dict[str, Any]:
    set_hssc_style()
    summary, dirs, groups, *_ = fig32.load()
    calls = [
        ("a", fig32.panel_a, summary),
        ("b", fig32.panel_b, dirs),
        ("c", fig32.panel_c, groups),
        ("d", fig32.panel_d, groups),
        ("e", fig32.panel_e, dirs),
        ("f", fig32.panel_f, summary),
    ]
    fig = plt.figure(figsize=(7.25, 8.65))
    gs = GridSpec(3, 2, figure=fig, wspace=0.28, hspace=0.58, left=0.080, right=0.985, bottom=0.075, top=0.895)
    hssc_header(
        fig,
        5,
        "Manual validation narrows the claim",
        "Validation upgrades the curated rule-source layer while showing that candidate queues mostly represent rule non-publicity.",
    )
    section_label(fig, 0.905, "Validation outcome and rule meaning")
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for ax, (_, func, data) in zip(axes, calls):
        func(ax, data)
    stem = FIGS / "Figure_05_manual_validation_HSSC_20260616"
    source_record(
        "Figure_05",
        [{"panel": letter, "source": f"82_build_completed_validation_figure_v32.{func.__name__}"} for letter, func, _ in calls],
    )
    return {"figure": "Figure_05", **save_fig(fig, stem)}


def figure_6() -> dict[str, Any]:
    set_hssc_style()
    topo = fig5_topology.load()
    cluster, morph_model, morph_grid, morph_boundary = fig29.load_morphology()
    fig = plt.figure(figsize=(7.25, 8.90))
    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[1.03, 1.00, 1.00, 1.22],
        left=0.075,
        right=0.985,
        bottom=0.065,
        top=0.900,
        wspace=0.20,
        hspace=0.58,
    )
    hssc_header(
        fig,
        6,
        "Network, threshold and morphology diagnostics",
        "Network exposure, threshold behaviour and morphology are retained as boundary diagnostics rather than a second main claim.",
    )
    section_label(fig, 0.910, "Network substrate")
    fig5_topology.draw_topology_surface(fig.add_subplot(gs[0, :2]), topo, "a")
    fig5_topology.draw_topology_fingerprint(fig.add_subplot(gs[0, 2:4]), topo, "b")
    fig5_topology.draw_edge_distance_structure(fig.add_subplot(gs[0, 4:]), topo, "c")
    section_label(fig, 0.655, "Threshold and model diagnostics")
    fig5_topology.draw_threshold_trajectories(fig.add_subplot(gs[1, :2]), topo, "d")
    fig5_topology.draw_scenario_type_composition(fig.add_subplot(gs[1, 2:4]), topo, "e")
    fig5_topology.draw_model_diagnostic(fig.add_subplot(gs[1, 4:]), topo, "f")
    section_label(fig, 0.395, "Morphology boundary condition")
    old_label = fig29.label

    def mapped_morphology_label(ax, letter, title, x=0, y=1.02):
        mapped = {"a": "g", "e": "h"}.get(letter, letter)
        old_label(ax, mapped, title, x=x, y=y)

    fig29.label = mapped_morphology_label
    try:
        fig29.panel9a(fig.add_subplot(gs[2:, :3]), cluster)
        fig29.panel9e(fig.add_subplot(gs[2:, 3:]), morph_grid, morph_boundary)
    finally:
        fig29.label = old_label
    stem = FIGS / "Figure_06_network_threshold_morphology_HSSC_20260616"
    source_record(
        "Figure_06",
        [
            {"panel": "a", "source": "build_fig5_topology_threshold_v28.draw_topology_surface"},
            {"panel": "b", "source": "build_fig5_topology_threshold_v28.draw_topology_fingerprint"},
            {"panel": "c", "source": "build_fig5_topology_threshold_v28.draw_edge_distance_structure"},
            {"panel": "d", "source": "build_fig5_topology_threshold_v28.draw_threshold_trajectories"},
            {"panel": "e", "source": "build_fig5_topology_threshold_v28.draw_scenario_type_composition"},
            {"panel": "f", "source": "build_fig5_topology_threshold_v28.draw_model_diagnostic"},
            {"panel": "g", "source": "build_v29_expanded_system.fig9a"},
            {"panel": "h", "source": "build_v29_expanded_system.fig9e"},
        ],
    )
    return {"figure": "Figure_06", **save_fig(fig, stem)}


def figure_7() -> dict[str, Any]:
    set_hssc_style()
    data = fig30.load_data()
    cases = fig30.select_cases(data)
    fig = plt.figure(figsize=(7.25, 13.50))
    gs = GridSpec(
        7,
        2,
        figure=fig,
        height_ratios=[1.18, 1, 1, 1, 1, 1, 1],
        left=0.052,
        right=0.985,
        bottom=0.045,
        top=0.910,
        wspace=0.10,
        hspace=0.38,
    )
    hssc_header(
        fig,
        7,
        "Local evidence vignette atlas",
        "The full set of local cards is retained as a compact two-column atlas so labels remain readable at double-column width.",
    )
    section_label(fig, 0.920, "Citywide case windows")
    ax_over = fig.add_subplot(gs[0, :])
    fig30.render_overview(data, cases)
    overview = Image.open(fig30.PANELS / "figX_v30_panel_overview_case_windows.png").convert("RGB")
    ax_over.imshow(overview)
    ax_over.set_axis_off()
    section_label(fig, 0.785, "Local evidence cards: two columns by six rows")
    source_rows = []
    for i, (_, case) in enumerate(cases.iterrows()):
        letter = chr(ord("b") + i)
        fig_card, stats = fig30.render_card(data, case, letter)
        card_path = PANELS / f"Figure_07_card_{letter}_{case['case_id']}.png"
        fig_card.savefig(card_path, dpi=600)
        plt.close(fig_card)
        ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        ax.imshow(Image.open(card_path).convert("RGB"))
        ax.set_axis_off()
        stats.update({"panel": letter})
        source_rows.append(stats)
    pd.DataFrame(source_rows).to_csv(SRC / "Figure_07_vignette_card_sources.csv", index=False, encoding="utf-8-sig")
    stem = FIGS / "Figure_07_local_evidence_vignette_atlas_HSSC_20260616"
    source_record(
        "Figure_07",
        [
            {"panel": "a", "source": "build_v30_map_evidence_vignettes.render_overview"},
            {"panel": "b-m", "source": "build_v30_map_evidence_vignettes.render_card; 12 selected local cases"},
        ],
    )
    return {"figure": "Figure_07", **save_fig(fig, stem)}


def make_contact_sheet(rows: list[dict[str, Any]]) -> None:
    images = []
    for row in rows:
        im = Image.open(row["png"]).convert("RGB")
        im.thumbnail((520, 720), Image.LANCZOS)
        can = Image.new("RGB", (560, 790), "white")
        can.paste(im, ((560 - im.width) // 2, 20))
        ImageDraw.Draw(can).text((20, 748), Path(row["png"]).name, fill=(35, 35, 35))
        images.append(can)
    cols = 3
    sheet = Image.new("RGB", (cols * 560, math.ceil(len(images) / cols) * 790), "white")
    for i, im in enumerate(images):
        sheet.paste(im, ((i % cols) * 560, (i // cols) * 790))
    sheet.save(AUDIT / "Figure_02_to_07_contact_sheet.png", dpi=(180, 180))


def write_manifest(rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(OUT / "HSSC_Figure_02_to_07_manifest.csv", index=False)
    (OUT / "HSSC_Figure_02_to_07_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    rows = [figure_2(), figure_3(), figure_4(), figure_5(), figure_6(), figure_7()]
    write_manifest(rows)
    make_contact_sheet(rows)
    # Provide a formal folder with neutral file names only.
    formal = OUT / "formal_upload_figures"
    formal.mkdir(exist_ok=True)
    for row in rows:
        n = row["figure"].replace("Figure_", "")
        for ext in ["png", "pdf", "tif", "svg"]:
            src = Path(row[ext])
            shutil.copy2(src, formal / f"Figure_{n}.{ext}")
    print(OUT)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
