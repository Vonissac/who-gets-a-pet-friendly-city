#!/usr/bin/env python3
"""Rebuild HSSC Fig. 4 with a quality-gated publication layout.

This wrapper preserves the v28/v27 scientific payload while correcting the
current composite's spatial controls, title hierarchy, panel balance, and
PDF/PNG/SVG-only formal package.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
V28_CODE = ROOT / "submission_package" / "code" / "figures_v28_spatial_sparse_repaired"
V27_CODE = ROOT / "submission_package" / "code" / "figures_v27_experiment_adjusted_dense"

OUT = ROOT / "hssc_fig4_quality_v1_20260617"
FORMAL = OUT / "formal_upload_figures"
AUDIT = OUT / "audit"
SRC = OUT / "source_data"
for folder in [FORMAL, AUDIT, SRC]:
    folder.mkdir(parents=True, exist_ok=True)


def import_module(name: str, path: Path, prepend: Path | None = None):
    if prepend is not None:
        sys.path.insert(0, str(prepend))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fig4 = import_module("fig4_frontier_v28_quality", V28_CODE / "build_fig4_suppressed_frontier_v28.py", V28_CODE)
fig27 = import_module("fig27_dense_quality", V27_CODE / "build_fig1_fig6_v27_experiment_adjusted_dense.py", V27_CODE)


INK = "#24211f"
TEXT = "#4f4a45"
MUTED = "#756e66"
RULE = "#2b6777"
WHITE = "#ffffff"


def set_hssc_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.titlesize": 7.8,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.3,
            "ytick.labelsize": 5.3,
            "legend.fontsize": 5.0,
            "axes.linewidth": 0.50,
            "axes.edgecolor": INK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.dpi": 600,
        }
    )


def hssc_header(fig: plt.Figure) -> None:
    fig.text(
        0.022,
        0.985,
        "Fig. 4 | Suppressed-emergence frontier and evidence boundary",
        ha="left",
        va="top",
        fontsize=12.3,
        fontweight="bold",
        color=INK,
    )


def panel_title(ax: plt.Axes, letter: str, title: str, y: float = 1.025, x_title: float = 0.060) -> None:
    ax.text(0.0, y, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2, fontweight="bold", color=INK)
    ax.text(x_title, y + 0.004, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, fontweight="bold", color=INK)


def remove_matching_text(ax: plt.Axes, patterns: tuple[str, ...], exact: tuple[str, ...] = ()) -> None:
    for text in ax.texts:
        value = text.get_text()
        if value in exact or any(p in value for p in patterns):
            text.set_visible(False)


def draw_north_arrow_and_scale(ax: plt.Axes, scale_m: int = 20000) -> None:
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    w = maxx - minx
    h = maxy - miny

    sx0 = minx + 0.720 * w
    sx1 = min(sx0 + scale_m, minx + 0.925 * w)
    sy = miny + 0.105 * h
    tick = 0.012 * h
    ax.plot([sx0, sx1], [sy, sy], color=INK, linewidth=0.78, zorder=88, solid_capstyle="butt")
    ax.plot([sx0, sx0], [sy - tick, sy + tick], color=INK, linewidth=0.55, zorder=89)
    ax.plot([sx1, sx1], [sy - tick, sy + tick], color=INK, linewidth=0.55, zorder=89)
    ax.text(
        (sx0 + sx1) / 2,
        sy + 0.020 * h,
        f"{scale_m // 1000} km",
        ha="center",
        va="bottom",
        fontsize=4.4,
        color=INK,
        zorder=90,
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.75, "pad": 0.22},
    )

    x = 0.915
    y0 = 0.745
    y1 = 0.850
    ax.annotate(
        "",
        xy=(x, y1),
        xytext=(x, y0),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "lw": 0.72, "color": INK, "shrinkA": 0, "shrinkB": 0},
        zorder=91,
    )
    ax.text(
        x,
        y1 + 0.022,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=4.9,
        fontweight="bold",
        color=INK,
        zorder=92,
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.58, "pad": 0.18},
    )


def draw_map_legend(ax: plt.Axes) -> None:
    x0, y0 = 0.020, 0.032
    w, h = 0.455, 0.078
    ax.add_patch(
        Rectangle(
            (x0, y0),
            w,
            h,
            transform=ax.transAxes,
            facecolor=WHITE,
            edgecolor="#e7e1d8",
            linewidth=0.25,
            alpha=0.88,
            zorder=80,
        )
    )
    items = [
        ("patch", fig4.FRONTIER, "frontier grid n=503"),
        ("patch", fig4.TRANSITION, "transition grid"),
        ("box", "#8c8177", "district zoom frame"),
    ]
    step = w / 3
    for idx, (kind, color, label) in enumerate(items):
        x = x0 + 0.020 + idx * step
        y = y0 + 0.044
        if kind == "box":
            ax.add_patch(
                Rectangle(
                    (x, y - 0.010),
                    0.024,
                    0.020,
                    transform=ax.transAxes,
                    fill=False,
                    edgecolor=color,
                    linewidth=0.35,
                    alpha=0.72,
                    zorder=82,
                )
            )
        else:
            ax.add_patch(
                Rectangle(
                    (x, y - 0.011),
                    0.024,
                    0.022,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="#d5cec4",
                    linewidth=0.18,
                    alpha=0.72,
                    zorder=82,
                )
            )
        ax.text(x + 0.032, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=3.35, color=MUTED, zorder=83)


def draw_frontier_overview(ax: plt.Axes, data: dict[str, object]) -> None:
    grid = data["grid"].copy()
    boundary = data["boundary"]
    grid.plot(ax=ax, color=fig4.LAND, edgecolor=fig4.GRID, linewidth=0.010, zorder=0, rasterized=True)
    grid[grid["grid_emergence_type"].eq("mixed_or_transitional_zone")].plot(
        ax=ax, color=fig4.TRANSITION, edgecolor="none", alpha=0.28, zorder=4, rasterized=True
    )
    grid[grid["grid_emergence_type"].eq("suppressed_emergence_frontier")].plot(
        ax=ax, color=fig4.FRONTIER, edgecolor="none", alpha=0.72, zorder=8, rasterized=True
    )
    boundary.boundary.plot(ax=ax, color="#514b45", linewidth=0.44, alpha=0.78, zorder=20)
    for district in fig4.DISTRICT_ORDER:
        dg = grid[grid["district_name"].eq(district)]
        if len(dg) == 0:
            continue
        minx, miny, maxx, maxy = fig4.district_extent(dg, 0.10)
        ax.add_patch(Rectangle((minx, miny), maxx - minx, maxy - miny, fill=False, edgecolor="#8c8177", linewidth=0.22, alpha=0.52, zorder=25))
    minx, miny, maxx, maxy = fig4.city_extent(boundary, pad=2200)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig4.light_frame(ax)
    panel_title(ax, "a", "citywide frontier anchor", y=1.020)
    draw_map_legend(ax)
    draw_north_arrow_and_scale(ax, 20000)


def draw_phase_space(ax: plt.Axes, data: dict[str, object]) -> None:
    fig4.draw_phase_space_axis(ax, data, "b")
    remove_matching_text(ax, ("phase space:",), exact=("b",))
    panel_title(ax, "b", "emergence-suppression phase space", y=1.020, x_title=0.055)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((0.985, 0.030))
        leg._loc = 4
        for text in leg.get_texts():
            text.set_fontsize(4.4)


def draw_matched_diagnostic(ax: plt.Axes, data: dict[str, object], letter: str = "e") -> None:
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
    ax.barh(y + 0.16, matched["treated_mean"], height=0.27, color=fig4.FRONTIER, alpha=0.76, label="frontier grids")
    ax.hlines(y, matched["matched_control_mean"], matched["treated_mean"], color="#7a7168", lw=0.65, alpha=0.55, zorder=4)
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
            color=fig4.FRONTIER if diff > 0 else fig4.MUTED,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([labels[o] for o in matched["outcome"]])
    ax.set_xlabel("matched mean value")
    ax.set_xlim(-0.02, max(matched[["treated_mean", "matched_control_mean"]].max()) + 0.18)
    ax.invert_yaxis()
    fig4.clean_axis(ax)
    panel_title(ax, letter, "matched rule-silence diagnostic", y=1.020, x_title=0.055)
    ax.legend(loc="lower right", frameon=False, fontsize=4.2, handletextpad=0.35, borderaxespad=0.15)
    matched.to_csv(SRC / f"Figure_04_panel_{letter}_matched_diagnostic_source.csv", index=False, encoding="utf-8-sig")


def draw_district_burden(ax: plt.Axes, data: dict[str, object], letter: str = "c") -> None:
    fig4.draw_district_burden_axis(ax, data, letter, title_y=1.020)
    remove_matching_text(ax, ("district burden of", "cell shade uses"), exact=(letter,))
    panel_title(ax, letter, "district frontier burden", y=1.020, x_title=0.055)
    ax.tick_params(axis="x", labelsize=3.85, pad=4)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_ha("center")
    ax.tick_params(axis="y", labelsize=5.2, pad=2)


def draw_district_scores(ax: plt.Axes, data: dict[str, pd.DataFrame], letter: str = "d") -> None:
    fig27.p6a(ax, data, letter)
    remove_matching_text(ax, ("district mean scores", "matrix form makes"), exact=(letter,))
    panel_title(ax, letter, "district mean scores", y=1.025, x_title=0.055)
    ax.tick_params(axis="x", labelsize=3.95, pad=4)
    ax.tick_params(axis="y", labelsize=5.1, pad=2)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_ha("center")
    for text in ax.texts:
        if text.get_visible() and text.get_text().startswith("rank "):
            text.set_visible(False)


def draw_grid_typology(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    before_axes = set(ax.figure.axes)
    fig27.p6b(ax, data, "f")
    remove_matching_text(ax, ("grid typology is", "each cell reports"), exact=("f",))
    panel_title(ax, "f", "grid typology by district", y=1.025, x_title=0.055)
    ax.tick_params(axis="x", labelsize=4.3, pad=2)
    ax.tick_params(axis="y", labelsize=5.1, pad=2)
    short = {
        "low pet city signal": "low signal",
        "mixed or transitional zone": "mixed",
        "suppressed emergence frontier": "frontier",
        "emergent capability core": "core",
        "rule first demonstration zone": "rule first",
    }
    new_labels = []
    for tick in ax.get_xticklabels():
        value = tick.get_text()
        head, _, tail = value.partition("\n")
        new_labels.append(f"{short.get(head, head)}\n{tail}" if tail else short.get(head, head))
        tick.set_rotation(0)
        tick.set_ha("center")
    ax.set_xticklabels(new_labels)
    for other in set(ax.figure.axes) - before_axes:
        other.tick_params(labelsize=0, length=0, pad=0)
        other.set_yticklabels([])
        other.set_visible(False)
    for child in getattr(ax, "child_axes", []):
        child.set_visible(False)


def draw_feature_ranking(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    fig27.p6c(ax, data, "g")
    remove_matching_text(ax, ("frontier feature ranking", "coefficients use"), exact=("g",))
    panel_title(ax, "g", "frontier feature ranking", y=1.025, x_title=0.055)
    ax.tick_params(axis="y", labelsize=5.0, pad=3)
    ax.tick_params(axis="x", labelsize=5.1, pad=2)
    for text in ax.texts:
        if text.get_visible() and text.get_text().startswith(("+", "-")):
            text.set_fontsize(3.8)
    label_map = {
        "edge cell no.": "edge cell",
        "liminal potential": "liminal potential",
        "liminal host": "liminal host",
        "positive rule": "positive rule",
        "pet ecology": "pet ecology",
        "rule silence": "rule silence",
        "topology": "topology",
        "edge cell yes": "edge cell yes",
        "residential property host count": "residential hosts",
        "primary positive rule count": "rule count",
        "public space share": "public space",
        "rule liminal exposure": "liminal exposure",
    }
    ax.set_yticklabels([label_map.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()])
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin - 0.42, xmax + 0.40)


def save_all(fig: plt.Figure) -> dict[str, Any]:
    stem = FORMAL / "Figure_04"
    paths = {}
    for ext in ["png", "pdf", "svg"]:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=600)
        paths[ext] = path
    plt.close(fig)
    rgb = Image.open(paths["png"]).convert("RGB")
    rgb.save(paths["png"], dpi=(600, 600))
    return {
        "figure": "Figure_04",
        "png": str(paths["png"]),
        "pdf": str(paths["pdf"]),
        "svg": str(paths["svg"]),
        "width_px": rgb.width,
        "height_px": rgb.height,
    }


def render_audit_crops(png_path: Path) -> None:
    im = Image.open(png_path).convert("RGB")
    w, h = im.size
    crops = {
        "whole": (0, 0, w, h),
        "top_panels": (0, 0, w, int(h * 0.41)),
        "middle_panels": (0, int(h * 0.32), w, int(h * 0.72)),
        "bottom_panels": (0, int(h * 0.65), w, h),
        "map_panel_a": (0, int(h * 0.07), int(w * 0.51), int(h * 0.40)),
        "feature_panel_g": (int(w * 0.58), int(h * 0.66), w, h),
    }
    for name, box in crops.items():
        im.crop(box).save(AUDIT / f"Figure_04_{name}_crop.png")


def package(row: dict[str, Any]) -> Path:
    zip_path = OUT / "HSSC_Figure_04_PDF_PNG_SVG.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for ext in ["pdf", "png", "svg"]:
            p = Path(row[ext])
            zf.write(p, p.name)
    return zip_path


def write_manifest(row: dict[str, Any], zip_path: Path) -> None:
    manifest = {
        **row,
        "package": str(zip_path),
        "source_scripts": [
            "code/figure_scripts/figures_v28_spatial_sparse_repaired/build_fig4_suppressed_frontier_v28.py",
            "code/figure_scripts/final_rebuild_scripts/100_rebuild_hssc_fig6_quality_v1.py",
        ],
        "layout_notes": [
            "removed decorative top rule and explanatory bands",
            "added inset north arrow and 20 km scale bar to the citywide map",
            "removed non-essential micro-notes below panels c and d",
            "restored the requested 2-2-3 reading rhythm: map/phase, matrix/matrix, bar/matrix/coefficient",
            "laid out lower evidence panels by the full axis-label envelope, not only the plot-body rectangle",
            "expanded bottom boundary-check panels and gave panel g extra label space",
        ],
        "formal_package_contents": ["Figure_04.pdf", "Figure_04.png", "Figure_04.svg"],
    }
    (OUT / "Figure_04_quality_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    set_hssc_style()
    front = fig4.load()
    model = fig27.load_data()

    fig = plt.figure(figsize=(10.40, 8.35))
    hssc_header(fig)

    # Manual axes use the full visual envelope, including tick labels and axis
    # labels, as the layout unit. This keeps the 2-2-3 rhythm without allowing
    # axis text to collide across panels.
    draw_frontier_overview(fig.add_axes([0.055, 0.640, 0.455, 0.265]), front)
    draw_phase_space(fig.add_axes([0.555, 0.665, 0.390, 0.225]), front)
    draw_district_burden(fig.add_axes([0.075, 0.378, 0.405, 0.225]), front)
    draw_district_scores(fig.add_axes([0.555, 0.378, 0.390, 0.225]), model)
    draw_matched_diagnostic(fig.add_axes([0.075, 0.100, 0.275, 0.225]), front)
    draw_grid_typology(fig.add_axes([0.410, 0.100, 0.280, 0.225]), model)
    draw_feature_ranking(fig.add_axes([0.765, 0.100, 0.205, 0.225]), model)

    row = save_all(fig)
    render_audit_crops(Path(row["png"]))
    zip_path = package(row)
    write_manifest(row, zip_path)
    print(json.dumps({**row, "package": str(zip_path)}, indent=2))


if __name__ == "__main__":
    main()
