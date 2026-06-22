#!/usr/bin/env python3
"""Rebuild HSSC Fig. 2 with source-locked data and atlas-first layout.

This script preserves the v28 Fig. 2 data and drawing semantics while fixing
the HSSC-facing layout: the nine high-granularity district maps become the
main evidence field, each district receives a locator inset, and the two
summary panels are kept as bottom evidence panels.
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


OUT = ROOT / "hssc_fig2_quality_v2_20260616"
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


def district_extent_tight(district_grid, pad_ratio: float = 0.028) -> tuple[float, float, float, float]:
    """Tighter district extent for atlas-first panels.

    The v28 source used a conservative atlas padding that was safe for
    standalone panels but leaves too much dead map body inside the 3x3 layout.
    This keeps projected, equal-aspect geography while reducing accidental
    whitespace around the district geometry.
    """
    minx, miny, maxx, maxy = district_grid.total_bounds
    w, h = maxx - minx, maxy - miny
    pad = max(w, h) * pad_ratio
    return fig2_atlas.normalize_extent_aspect(
        (minx - pad, miny - pad, maxx + pad, maxy + pad),
        fig2_atlas.DISTRICT_PANEL_ASPECT,
    )


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


def draw_compact_atlas_legend(ax: plt.Axes) -> None:
    ax.set_axis_off()
    items = [
        ("patch", "#dbe7d8", "public space"),
        ("patch", fig2_atlas.BUILDING, "building texture"),
        ("patch", "#9fb8c5", "service/suppression grid"),
        ("line", fig2_atlas.ROAD, "road network"),
        ("line", fig2_atlas.PED, "pedestrian path"),
        ("point", fig2_atlas.RUST, "frontier host"),
        ("point", fig2_atlas.BLUE_DARK, "pet-service point"),
        ("point", fig2_atlas.GREEN, "visible rule record"),
    ]
    xs = [0.010, 0.140, 0.275, 0.425, 0.555, 0.685, 0.800, 0.910]
    y = 0.48
    for x, (kind, color, label) in zip(xs, items):
        if kind == "patch":
            ax.add_patch(
                plt.Rectangle(
                    (x, y - 0.085),
                    0.030,
                    0.170,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="#d5cec4",
                    linewidth=0.25,
                    alpha=0.72,
                )
            )
        elif kind == "line":
            ax.plot([x, x + 0.034], [y, y], transform=ax.transAxes, color=color, linewidth=1.25, alpha=0.72)
        else:
            ax.scatter([x + 0.017], [y], transform=ax.transAxes, s=18, color=color, edgecolor=WHITE, linewidth=0.28, alpha=0.94)
        ax.text(x + 0.040, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=4.20, color=MUTED)


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


def fmt_short(value: int) -> str:
    if value >= 10_000:
        return f"{value / 1000:.1f}k"
    if value >= 1000:
        return f"{value / 1000:.2f}k"
    return f"{value:,}"


def fmt_pct(part: int, whole: int) -> str:
    if whole <= 0:
        return "0%"
    return f"{part / whole * 100:.0f}%"


def draw_locator_inset(ax: plt.Axes, data: dict[str, object], district: str) -> None:
    """Small not-to-scale locator: Shenzhen outline plus current district."""
    inset = ax.inset_axes([0.035, 0.690, 0.235, 0.255], zorder=90)
    inset.set_facecolor((1, 1, 1, 0.82))
    grid = data["grid"]
    boundary = data["boundary"]
    current = grid[grid["district_name"].eq(district)]
    boundary.plot(ax=inset, color="#f7f5ef", edgecolor="#9b958c", linewidth=0.34, alpha=0.96, zorder=1)
    grid.dissolve(by="district_name").boundary.plot(ax=inset, color="#d4cec4", linewidth=0.18, alpha=0.82, zorder=2)
    if len(current):
        current.dissolve().plot(ax=inset, color="#d88973", edgecolor="#8f5a50", linewidth=0.34, alpha=0.86, zorder=4)
    minx, miny, maxx, maxy = fig2_atlas.city_extent(boundary, pad=2600)
    inset.set_xlim(minx, maxx)
    inset.set_ylim(miny, maxy)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color("#d8d2c7")
        spine.set_linewidth(0.34)


def draw_metric_strip(ax: plt.Axes, stats: dict[str, object]) -> None:
    """Open, panel-local evidence readout without a pasted white data box."""
    if ax.texts:
        ax.texts[-1].set_visible(False)

    host_n = int(stats["host_candidates"])
    frontier_n = int(stats["priority_frontier_hosts"])
    items = [
        ("host", fmt_short(host_n), fig2_atlas.RUST),
        ("frontier", fmt_pct(frontier_n, host_n), fig2_atlas.GOLD),
        ("pet POI", fmt_short(int(stats["pet_service_points"])), fig2_atlas.BLUE_DARK),
        ("rules", fmt_short(int(stats["positive_rule_points"])), fig2_atlas.GREEN),
    ]
    positions = [(0.074, 0.083), (0.320, 0.083), (0.074, 0.041), (0.320, 0.041)]
    halo = {"facecolor": WHITE, "edgecolor": "none", "alpha": 0.54, "pad": 0.22}
    for (x, y), (label, value, color) in zip(positions, items):
        ax.scatter([x], [y], transform=ax.transAxes, s=13, color=color, edgecolor=WHITE, linewidth=0.26, zorder=88)
        ax.text(
            x + 0.025,
            y,
            f"{label} {value}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=3.95,
            fontweight="bold",
            color=INK,
            bbox=halo,
            zorder=88,
        )


def draw_district(ax: plt.Axes, data: dict[str, object], district: str, letter: str) -> None:
    stats = fig2_atlas.draw_district_panel(ax, data, district, letter, compact=True)
    if len(ax.texts) >= 2:
        ax.texts[1].set_fontsize(6.2)
    draw_locator_inset(ax, data, district)
    draw_metric_strip(ax, stats)
    draw_north_arrow_and_scale(ax, 5000, compact=True)


def draw_host_panel(ax: plt.Axes, data: dict[str, object], letter: str = "j") -> None:
    fig2_atlas.draw_stacked_host_panel(ax, data, letter, legend_anchor=(0.5, -0.155))
    if len(ax.texts) >= 2:
        ax.texts[0].set_visible(False)
        ax.texts[1].set_visible(False)
    ax.text(0.0, 1.245, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.4, fontweight="bold", color=INK)
    ax.text(
        0.075,
        1.255,
        "host-type composition",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        fontweight="bold",
        color=INK,
    )
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(1.000, 1.112),
        ncol=6,
        frameon=False,
        handlelength=0.78,
        columnspacing=0.30,
        handletextpad=0.20,
        fontsize=3.25,
        borderaxespad=0.0,
    )
    ax.set_xlabel("share of host candidates (%)", labelpad=2.0)
    ax.tick_params(axis="y", labelsize=5.4, pad=1)


def draw_fingerprint(ax: plt.Axes, summary: pd.DataFrame, letter: str = "k") -> None:
    fig2_atlas.draw_fingerprint_strip(ax, summary, letter, title_y=1.245)
    if len(ax.texts) >= 2:
        ax.texts[-1].set_visible(False)
        ax.texts[-3].set_fontsize(7.4)
        ax.texts[-2].set_fontsize(7.4)
    ax.tick_params(axis="x", pad=1, labelsize=4.7)
    ax.tick_params(axis="y", labelsize=4.7, pad=1)


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
            "Atlas-first layout: nine high-granularity district maps become the primary evidence field",
            "Each district map receives a muted Shenzhen locator inset highlighting district position",
            "All main map panels receive north arrow and scale bar inside panel safe area",
            "Each district map includes an open 2x2 evidence readout for host count, frontier share, pet-service POI and visible rules",
            "Non-core OSM/global legend removed so atlas panels occupy more of the figure body",
            "Host-type composition and district evidence fingerprint moved to bottom summary panels",
            "Formal outputs restricted to PDF, PNG and SVG",
        ],
        "cartographic_controls": {
            "district_panels": {"north_arrow": True, "scale_bar": "5 km"},
            "locator_insets": {"north_arrow": False, "scale_bar": False, "reason": "not-to-scale district locator insets"},
        },
    }
    (SRC / "Figure_02_source_map.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def render_audit_crops(png_path: Path) -> None:
    im = Image.open(png_path).convert("RGB")
    w, h = im.size
    crops = {
        "Figure_02_top_atlas_review.png": (0, 0, w, int(h * 0.380)),
        "Figure_02_district_grid_review.png": (0, int(h * 0.180), w, int(h * 0.820)),
        "Figure_02_bottom_panels_review.png": (0, int(h * 0.740), w, h),
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
    fig2_atlas.district_extent = district_extent_tight
    data = fig2_base.load_data()
    summary = fig2_atlas.district_stats(data)

    fig = plt.figure(figsize=(7.25, 8.80))
    hssc_header(fig)

    col_w = 0.300
    row_h = 0.210
    x_gap = 0.014
    y_gap = 0.016
    left = 0.050
    bottom0 = 0.236
    for i, district in enumerate(fig2_atlas.DISTRICT_ORDER):
        row = i // 3
        col = i % 3
        x = left + col * (col_w + x_gap)
        y = bottom0 + (2 - row) * (row_h + y_gap)
        ax = fig.add_axes([x, y, col_w, row_h])
        draw_district(ax, data, district, chr(ord("a") + i))

    draw_host_panel(fig.add_axes([0.082, 0.034, 0.400, 0.145]), data, "j")
    draw_fingerprint(fig.add_axes([0.570, 0.034, 0.390, 0.145]), summary, "k")

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
            "nine district maps are primary evidence panels",
            "all main map panels include north arrow and scale bar",
            "locator insets are not-to-scale and exempt from scale bars",
            "cartographic controls are inset from map edges",
            "district open evidence readouts clarify host count, frontier share, pet-service POI and visible rules",
            "non-core OSM/global legend removed",
            "PDF/PNG/SVG only",
        ],
    }
    (AUDIT / "Figure_02_quality_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
