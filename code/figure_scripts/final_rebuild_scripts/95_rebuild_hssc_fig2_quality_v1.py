#!/usr/bin/env python3
"""Rebuild HSSC Fig. 2 with source-locked data and tighter map controls.

This script preserves the v28 Fig. 2 data and drawing semantics while fixing
the HSSC-facing layout: compact canvas occupancy, global alignment, north
arrows, scale bars and PDF/PNG/SVG formal outputs.
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
V28_CODE = ROOT / "submission_package" / "code" / "figures_v28_spatial_sparse_repaired"
sys.path.insert(0, str(V28_CODE))

import build_fig2_v28 as fig2_base  # noqa: E402
import build_fig2_district_atlas_v28 as fig2_atlas  # noqa: E402


OUT = ROOT / "hssc_fig2_quality_v1_20260616"
FORMAL = OUT / "formal_upload_figures"
AUDIT = OUT / "audit"
SRC = OUT / "source_data"
for folder in [FORMAL, AUDIT, SRC]:
    folder.mkdir(parents=True, exist_ok=True)

INK = "#24211f"
TEXT = "#504a43"
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
            "font.size": 7.1,
            "axes.titlesize": 7.8,
            "axes.labelsize": 6.7,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.52,
            "axes.edgecolor": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def hssc_header(fig: plt.Figure) -> None:
    fig.text(
        0.020,
        0.982,
        "Fig. 2 | District service-rule atlas across Shenzhen",
        ha="left",
        va="top",
        fontsize=12.4,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.020,
        0.952,
        "Citywide substrate, district summaries and local panels separate service ecology, candidate hosts and visible rule records.",
        ha="left",
        va="top",
        fontsize=6.6,
        color=TEXT,
    )
    fig.lines.append(plt.Line2D([0.020, 0.980], [0.992, 0.992], transform=fig.transFigure, color=RULE, lw=0.72))


def section_label(fig: plt.Figure, y: float, text: str) -> None:
    fig.text(0.023, y, text, ha="left", va="center", fontsize=7.0, fontweight="bold", color=INK)
    fig.patches.append(
        plt.Rectangle(
            (0.020, y - 0.012),
            0.006,
            0.024,
            transform=fig.transFigure,
            facecolor=RULE,
            edgecolor="none",
            zorder=5,
        )
    )


def draw_north_arrow_and_scale(ax: plt.Axes, scale_m: int, compact: bool) -> None:
    """Add quiet cartographic controls inside the panel safe area."""
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    w = maxx - minx
    h = maxy - miny
    sx0 = minx + (0.060 if compact else 0.055) * w
    sy = miny + (0.145 if compact else 0.118) * h
    sx1 = min(sx0 + scale_m, minx + 0.340 * w)
    ax.plot([sx0, sx1], [sy, sy], color=INK, linewidth=0.72 if compact else 0.82, zorder=80, solid_capstyle="butt")
    tick = 0.010 * h
    ax.plot([sx0, sx0], [sy - tick, sy + tick], color=INK, linewidth=0.54, zorder=81)
    ax.plot([sx1, sx1], [sy - tick, sy + tick], color=INK, linewidth=0.54, zorder=81)
    label = f"{scale_m // 1000} km" if scale_m >= 1000 else f"{scale_m} m"
    ax.text(
        (sx0 + sx1) / 2,
        sy + 0.018 * h,
        label,
        ha="center",
        va="bottom",
        fontsize=4.0 if compact else 4.3,
        color=INK,
        zorder=82,
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.74, "pad": 0.28},
    )
    x = 0.925 if compact else 0.935
    y0 = 0.790 if compact else 0.805
    y1 = 0.875 if compact else 0.905
    ax.annotate(
        "",
        xy=(x, y1),
        xytext=(x, y0),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "lw": 0.58 if compact else 0.70, "color": INK, "shrinkA": 0, "shrinkB": 0},
        zorder=82,
    )
    ax.text(
        x,
        y1 + 0.020,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=4.3 if compact else 4.8,
        fontweight="bold",
        color=INK,
        zorder=82,
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.58, "pad": 0.20},
    )


def draw_city(ax: plt.Axes, data: dict[str, object]) -> None:
    fig2_atlas.draw_city_overview_panel(ax, data, "a")
    if len(ax.texts) >= 2:
        ax.texts[1].set_text("Shenzhen overview: service ecology and visible rules")
        ax.texts[1].set_fontsize(5.0)
    draw_north_arrow_and_scale(ax, 10000, compact=False)


def draw_district(ax: plt.Axes, data: dict[str, object], district: str, letter: str) -> None:
    fig2_atlas.draw_district_panel(ax, data, district, letter, compact=True)
    draw_north_arrow_and_scale(ax, 5000, compact=True)


def draw_host_panel(ax: plt.Axes, data: dict[str, object]) -> None:
    fig2_atlas.draw_stacked_host_panel(ax, data, "b", legend_anchor=(0.5, -0.155))
    if len(ax.texts) >= 2:
        ax.texts[1].set_text("host-type composition")
        ax.texts[1].set_fontsize(5.3)
    ax.set_xlabel("share of host candidates (%)", labelpad=1.5)


def draw_fingerprint(ax: plt.Axes, summary: pd.DataFrame) -> None:
    fig2_atlas.draw_fingerprint_strip(ax, summary, "c", title_y=1.220)
    ax.tick_params(axis="x", pad=1, labelsize=5.0)
    ax.tick_params(axis="y", labelsize=5.0)


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


def write_source_manifest() -> None:
    manifest = {
        "figure": "Figure_02",
        "source_locked": True,
        "source_code": "code/figure_scripts/figures_v28_spatial_sparse_repaired/build_fig2_district_atlas_v28.py",
        "source_data": [
            "data/figure_source_data/hssc_fig2_quality_v2_20260616/Figure_02_source_map.json",
            "data/figure_source_data/hssc_fig2_quality_v2_20260616/Figure_02_source_map.json",
        ],
        "layout_changes": [
            "HSSC-facing canvas tightened to remove accidental bottom whitespace",
            "All map panels receive north arrow and scale bar inside panel safe area",
            "Top summary row and district atlas rows rebalanced on a common visual grid",
            "Formal outputs restricted to PDF, PNG and SVG",
        ],
        "cartographic_controls": {
            "city_panel": {"north_arrow": True, "scale_bar": "10 km"},
            "district_panels": {"north_arrow": True, "scale_bar": "5 km"},
        },
    }
    (SRC / "Figure_02_source_map.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def render_audit_crops(png_path: Path) -> None:
    im = Image.open(png_path).convert("RGB")
    w, h = im.size
    crops = {
        "Figure_02_top_summary_review.png": (0, 0, w, int(h * 0.375)),
        "Figure_02_district_grid_review.png": (0, int(h * 0.325), w, int(h * 0.900)),
        "Figure_02_bottom_margin_review.png": (0, int(h * 0.820), w, h),
    }
    for name, box in crops.items():
        im.crop(box).save(AUDIT / name, dpi=(220, 220))


def package(outputs: dict[str, str]) -> Path:
    zip_path = OUT / "HSSC_Figure_02_PDF_PNG_SVG.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for ext in ["pdf", "png", "svg"]:
            p = Path(outputs[ext])
            zf.write(p, arcname=p.name)
    return zip_path


def main() -> None:
    set_hssc_style()
    data = fig2_base.load_data()
    summary = fig2_atlas.district_stats(data)

    fig = plt.figure(figsize=(7.25, 9.35))
    gs = GridSpec(
        6,
        6,
        figure=fig,
        height_ratios=[0.16, 1.04, 0.74, 1.02, 1.02, 1.02],
        left=0.060,
        right=0.982,
        bottom=0.058,
        top=0.855,
        wspace=0.20,
        hspace=0.545,
    )
    hssc_header(fig)
    section_label(fig, 0.872, "Citywide overview and district summary")
    axleg = fig.add_subplot(gs[0, :])
    fig2_atlas.draw_atlas_legend(axleg, "fig2")

    draw_city(fig.add_subplot(gs[1, :3]), data)
    draw_host_panel(fig.add_subplot(gs[1, 3:]), data)
    draw_fingerprint(fig.add_subplot(gs[2, :]), summary)

    section_label(fig, 0.515, "District atlas: every district appears once")
    for i, district in enumerate(fig2_atlas.DISTRICT_ORDER):
        ax = fig.add_subplot(gs[(i // 3) + 3, (i % 3) * 2 : (i % 3 + 1) * 2])
        draw_district(ax, data, district, chr(ord("d") + i))

    outputs = save_all(fig, FORMAL / "Figure_02")
    write_source_manifest()
    render_audit_crops(Path(outputs["png"]))
    zip_path = package(outputs)
    info = Image.open(outputs["png"])
    audit = {
        "figure": "Figure_02",
        "outputs": outputs,
        "zip": str(zip_path),
        "png_size": [info.width, info.height],
        "dpi": info.info.get("dpi"),
        "formal_package_files": ["Figure_02.pdf", "Figure_02.png", "Figure_02.svg"],
        "hard_gate_notes": [
            "all map panels include north arrow and scale bar",
            "cartographic controls are inset from map edges",
            "bottom whitespace reduced by canvas/layout rebuild",
            "PDF/PNG/SVG only",
        ],
    }
    (AUDIT / "Figure_02_quality_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
