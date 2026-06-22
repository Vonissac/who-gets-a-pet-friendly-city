#!/usr/bin/env python3
"""Rebuild HSSC Fig. 3 with source-locked data and quality-gated layout.

The v28 Fig. 3 scientific semantics are preserved: rule-liminal hosts remain
candidate transition signals, while traceable rules stay separate evidence.
This wrapper rebuilds the assembled figure with tighter HSSC spacing, map
controls, cleaner title hierarchy, and PDF/PNG/SVG-only formal outputs.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
V28_CODE = ROOT / "submission_package" / "code" / "figures_v28_spatial_sparse_repaired"
sys.path.insert(0, str(V28_CODE))

import build_fig3_rule_liminal_venues_v28 as fig3_v28  # noqa: E402


OUT = ROOT / "hssc_fig3_quality_v1_20260617"
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

FIGSIZE = (8.20, 7.82)
LEFT_COL_X = 0.055
RIGHT_COL_X = 0.535
COL_W = 0.420
AXES_LAYOUT = {
    "a": [LEFT_COL_X, 0.568, COL_W, 0.324],
    "b": [RIGHT_COL_X, 0.568, COL_W, 0.372],
    "c": [LEFT_COL_X, 0.356, COL_W, 0.154],
    "d": [RIGHT_COL_X, 0.356, COL_W, 0.154],
    "e": [LEFT_COL_X, 0.082, COL_W, 0.218],
    "f": [RIGHT_COL_X, 0.082, COL_W, 0.218],
}


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
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
            "legend.fontsize": 5.0,
            "axes.linewidth": 0.50,
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
        "Fig. 3 | Rule-liminal venues and semantic rule evidence",
        ha="left",
        va="top",
        fontsize=12.4,
        fontweight="bold",
        color=INK,
    )


def draw_core_legend(ax: plt.Axes) -> None:
    ax.set_axis_off()
    items = [
        ("patch", fig3_v28.LIMINAL_GOLD, "rule-liminal grid"),
        ("point", fig3_v28.LIMINAL_DARK, "priority host candidate"),
        ("point", fig3_v28.BLUE_DARK, "pet-service point"),
        ("point", fig3_v28.GREEN, "traceable rule evidence"),
        ("line", fig3_v28.ROAD, "road network"),
        ("line", fig3_v28.PED, "pedestrian path"),
    ]
    xs = [0.020, 0.185, 0.385, 0.560, 0.755, 0.885]
    y = 0.50
    for x, (kind, color, label) in zip(xs, items):
        if kind == "patch":
            ax.add_patch(
                Rectangle(
                    (x, y - 0.090),
                    0.032,
                    0.180,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="#d5cec4",
                    linewidth=0.25,
                    alpha=0.72,
                )
            )
        elif kind == "line":
            ax.plot([x, x + 0.035], [y, y], transform=ax.transAxes, color=color, linewidth=1.15, alpha=0.72)
        else:
            ax.scatter([x + 0.016], [y], transform=ax.transAxes, s=18, color=color, edgecolor=WHITE, linewidth=0.28, alpha=0.94)
        ax.text(x + 0.042, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=4.3, color=MUTED)


def hide_map_source_note(ax: plt.Axes) -> None:
    for text in ax.texts:
        value = text.get_text()
        if "host candidates" in value or value.startswith("hosts "):
            text.set_visible(False)


def draw_map_panel_legend(ax: plt.Axes, items: list[tuple[str, str, str]]) -> None:
    """Panel-local legend for map semantics; excludes non-core basemap items."""
    box_w = 0.355 if len(items) <= 3 else 0.475
    box_h = 0.090
    x0 = 0.985 - box_w
    y0 = 0.030
    ax.add_patch(
        Rectangle(
            (x0, y0),
            box_w,
            box_h,
            transform=ax.transAxes,
            facecolor=WHITE,
            edgecolor="#ebe5dc",
            linewidth=0.25,
            alpha=0.88,
            zorder=88,
        )
    )
    step = box_w / len(items)
    for idx, (kind, color, label) in enumerate(items):
        x = x0 + 0.026 + idx * step
        y = y0 + 0.055
        if kind == "patch":
            ax.add_patch(
                Rectangle(
                    (x, y - 0.012),
                    0.022,
                    0.024,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="#d5cec4",
                    linewidth=0.18,
                    alpha=0.72,
                    zorder=90,
                )
            )
        else:
            ax.scatter([x + 0.011], [y], transform=ax.transAxes, s=11, color=color, edgecolor=WHITE, linewidth=0.22, zorder=90)
        ax.text(x + 0.028, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=3.15, color=MUTED, zorder=90)


def draw_map_metric_note(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.012,
        0.026,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=3.55,
        color=MUTED,
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.72, "pad": 0.25},
        zorder=90,
    )


def draw_map_evidence_strip(
    ax: plt.Axes,
    metrics: list[tuple[str, str]],
    legend_items: list[tuple[str, str, str]],
    *,
    width: float = 0.760,
    x0: float = 0.220,
    y0: float = 0.028,
) -> None:
    """Compact map evidence strip with quantitative readouts and semantic keys."""
    mx = x0 + 0.020
    metric_step = min(0.115, width * 0.455 / max(len(metrics), 1))
    for idx, (value, label) in enumerate(metrics):
        x = mx + idx * metric_step
        ax.text(
            x,
            y0 + 0.071,
            value,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=4.25,
            fontweight="bold",
            color=INK,
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.46, "pad": 0.10},
            zorder=91,
        )
        ax.text(
            x,
            y0 + 0.031,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=3.15,
            color=MUTED,
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.34, "pad": 0.06},
            zorder=91,
        )

    lx = x0 + width * 0.525
    legend_step = (x0 + width - lx - 0.014) / max(len(legend_items), 1)
    for idx, (kind, color, label) in enumerate(legend_items):
        x = lx + idx * legend_step
        y = y0 + 0.055
        if kind == "patch":
            ax.add_patch(
                Rectangle((x, y - 0.014), 0.020, 0.028, transform=ax.transAxes, facecolor=color, edgecolor=WHITE, linewidth=0.26, alpha=0.82, zorder=91)
            )
        else:
            ax.scatter([x + 0.010], [y], transform=ax.transAxes, s=12, color=color, edgecolor=WHITE, linewidth=0.22, zorder=91)
        ax.text(
            x + 0.026,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=3.15,
            color=MUTED,
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.32, "pad": 0.05},
            zorder=91,
        )


def draw_north_arrow_and_scale(ax: plt.Axes, scale_m: int, compact: bool) -> None:
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    w = maxx - minx
    h = maxy - miny
    sx0 = minx + (0.060 if compact else 0.055) * w
    sy = miny + (0.130 if compact else 0.095) * h
    sx1 = min(sx0 + scale_m, minx + 0.340 * w)
    ax.plot([sx0, sx1], [sy, sy], color=INK, linewidth=0.72 if compact else 0.82, zorder=82, solid_capstyle="butt")
    tick = 0.010 * h
    ax.plot([sx0, sx0], [sy - tick, sy + tick], color=INK, linewidth=0.54, zorder=83)
    ax.plot([sx1, sx1], [sy - tick, sy + tick], color=INK, linewidth=0.54, zorder=83)
    label = f"{scale_m // 1000} km" if scale_m >= 1000 else f"{scale_m} m"
    ax.text(
        (sx0 + sx1) / 2,
        sy + 0.018 * h,
        label,
        ha="center",
        va="bottom",
        fontsize=4.0 if compact else 4.4,
        color=INK,
        zorder=84,
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.74, "pad": 0.28},
    )
    x = 0.930 if compact else 0.940
    y0 = 0.790 if compact else 0.800
    y1 = 0.875 if compact else 0.900
    ax.annotate(
        "",
        xy=(x, y1),
        xytext=(x, y0),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "lw": 0.60 if compact else 0.72, "color": INK, "shrinkA": 0, "shrinkB": 0},
        zorder=84,
    )
    ax.text(
        x,
        y1 + 0.020,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=4.4 if compact else 4.9,
        fontweight="bold",
        color=INK,
        zorder=85,
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.58, "pad": 0.20},
    )


def align_panel_title(ax: plt.Axes, letter: str, title: str, title_y: float = 1.105) -> None:
    """Hide source title text and redraw a shared title baseline."""
    for text in ax.texts:
        value = text.get_text()
        if value == letter or title in value or "candidates:" in value or "traceable rule records" in value:
            text.set_visible(False)
    ax.text(0.0, title_y, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2, fontweight="bold", color=INK)
    ax.text(0.062, title_y + 0.004, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, fontweight="bold", color=INK)


def trim_note_texts(ax: plt.Axes, keep_last: bool = False) -> None:
    """Remove source-note microtext that costs layout space in the composite."""
    if keep_last:
        return
    for text in ax.texts:
        value = text.get_text()
        if "cell shade" in value or "cells show" in value or "source-grounded" in value or value.startswith("n="):
            text.set_visible(False)


def draw_map_a(ax: plt.Axes, data: dict[str, object]) -> None:
    fig3_v28.draw_hero_map(ax, data, "a")
    minx, miny, maxx, maxy = fig3_v28.city_extent(data["boundary"], pad=1500)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    hide_map_source_note(ax)
    draw_north_arrow_and_scale(ax, 20000, compact=False)
    hosts = data["host_g"]
    priority = hosts[hosts["liminal_class"].isin(["rule_liminal_threshold_core", "ecology_ready_rule_liminal_frontier"])]
    positive = data["rules"][data["rules"]["access_semantic_score"].fillna(0) > 0.55]
    verification = data["verification"]
    draw_map_evidence_strip(
        ax,
        [
            (f"{len(hosts):,}", "hosts"),
            (f"{len(priority):,}", "priority"),
            (f"{len(positive):,}", "positive rules"),
            (f"{len(verification):,}", "verify queue"),
        ],
        [
            ("patch", fig3_v28.LIMINAL_GOLD, "liminal grid"),
            ("point", fig3_v28.LIMINAL_DARK, "host"),
            ("point", fig3_v28.GREEN, "rule"),
        ],
        width=0.700,
        x0=0.275,
        y0=0.026,
    )


def draw_map_b(ax: plt.Axes, data: dict[str, object]) -> None:
    fig3_v28.draw_local_zoom(ax, data, "福田区", "b")
    hide_map_source_note(ax)
    draw_north_arrow_and_scale(ax, 5000, compact=True)
    hosts = data["hosts"]
    h = hosts[hosts["district_name"].eq("福田区")]
    priority = h[h["liminal_class"].isin(["rule_liminal_threshold_core", "ecology_ready_rule_liminal_frontier"])]
    rules = data["rules"]
    grid = data["grid"]
    dg = grid[grid["district_name"].eq("福田区")]
    rsub = rules[rules["grid_id"].isin(dg["grid_id"])]
    psub = data["pet"][data["pet"]["district_name"].eq("福田区")]
    draw_map_evidence_strip(
        ax,
        [
            (f"{len(h):,}", "hosts"),
            (f"{len(priority):,}", "priority"),
            (f"{len(rsub):,}", "rules"),
            (f"{len(psub):,}", "services"),
        ],
        [
            ("patch", fig3_v28.LIMINAL_GOLD, "liminal grid"),
            ("point", fig3_v28.LIMINAL_DARK, "host"),
            ("point", fig3_v28.BLUE_DARK, "service"),
            ("point", fig3_v28.GREEN, "rule"),
        ],
        width=0.710,
        x0=0.245,
        y0=0.030,
    )


def draw_matrix_c(ax: plt.Axes, data: dict[str, object]) -> None:
    fig3_v28.draw_host_type_distribution(ax, data, "c")
    align_panel_title(ax, "c", "host type by liminal state", title_y=1.135)
    trim_note_texts(ax)
    ax.tick_params(axis="x", labelsize=3.20, pad=1)
    ax.tick_params(axis="y", labelsize=4.2, pad=1)


def draw_composition_d(ax: plt.Axes, data: dict[str, object]) -> None:
    fig3_v28.draw_class_composition(ax, data, "d", legend_anchor=(0.50, 1.115))
    align_panel_title(ax, "d", "candidate classes are not access claims", title_y=1.135)
    if ax.get_legend() is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        ax.legend(
            handles,
            labels,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.035),
            ncol=6,
            frameon=False,
            handlelength=0.72,
            columnspacing=0.26,
            handletextpad=0.16,
            fontsize=3.20,
            borderaxespad=0,
        )
    ax.set_xlabel("")
    ax.xaxis.label.set_visible(False)
    ax.xaxis.label.set_text("")
    ax.tick_params(axis="y", labelsize=4.5, pad=1)
    ax.tick_params(axis="x", labelsize=4.2, pad=1)


def draw_rule_e(ax: plt.Axes, data: dict[str, object]) -> None:
    fig3_v28.draw_rule_semantic_matrix(ax, data, "e")
    align_panel_title(ax, "e", "traceable rule records by semantic family and source", title_y=1.130)
    trim_note_texts(ax)
    ax.tick_params(axis="x", labelsize=3.65, pad=2)
    ax.tick_params(axis="y", labelsize=4.0, pad=1)


def draw_fingerprint_f(ax: plt.Axes, data: dict[str, object]) -> None:
    fig3_v28.draw_district_fingerprint(ax, data, "f", title_y=1.130)
    trim_note_texts(ax)
    if len(ax.texts) >= 3:
        ax.texts[-3].set_fontsize(7.2)
        ax.texts[-2].set_fontsize(6.5)
    ax.tick_params(axis="x", labelsize=3.80, pad=3)
    ax.tick_params(axis="y", labelsize=4.3, pad=1)


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


def render_audit_crops(png_path: Path) -> None:
    im = Image.open(png_path).convert("RGB")
    w, h = im.size
    crops = {
        "Figure_03_top_maps_review.png": (0, 0, w, int(h * 0.545)),
        "Figure_03_mid_panels_review.png": (0, int(h * 0.455), w, int(h * 0.760)),
        "Figure_03_bottom_panels_review.png": (0, int(h * 0.690), w, h),
    }
    for name, box in crops.items():
        im.crop(box).save(AUDIT / name, dpi=(220, 220))

    overlay = im.copy()
    from PIL import ImageDraw

    draw = ImageDraw.Draw(overlay)
    for panel, (x, y, width, height) in AXES_LAYOUT.items():
        color = (210, 48, 48) if panel in {"a", "b"} else (32, 86, 210)
        for xpos in [x, x + width]:
            xx = int(xpos * w)
            draw.line([(xx, 0), (xx, h)], fill=color, width=2)
        for ypos in [y, y + height]:
            yy = int((1 - ypos) * h)
            draw.line([(0, yy), (w, yy)], fill=(160, 160, 160), width=1)
    overlay.save(AUDIT / "Figure_03_hidden_alignment_overlay.png", dpi=(220, 220))


def write_source_manifest(outputs: dict[str, str], zip_path: Path) -> None:
    manifest = {
        "figure": "Figure_03",
        "source_locked": True,
        "source_code": "code/figure_scripts/figures_v28_spatial_sparse_repaired/build_fig3_rule_liminal_venues_v28.py",
        "source_data": [
            "data_processed/model/grid_rule_liminality_500m_2025_v4.geojson",
            "data_processed/platform/rule_liminal_host_candidates_2025_v4.csv",
            "data_processed/platform/rule_semantic_records_v1.csv",
            "data_processed/platform/rule_semantic_geocoded_points_v1.geojson",
            "data_processed/platform/rule_source_ledger_v1.csv",
            "data_processed/osm/shenzhen_osm_roads.gpkg",
            "data_processed/osm/shenzhen_osm_pedestrian_paths.gpkg",
            "data_processed/osm/shenzhen_osm_buildings_3d.gpkg",
            "data_processed/osm/shenzhen_osm_public_space.gpkg",
        ],
        "layout_changes": [
            "Removed top explanatory line and oversized global legend",
            "Rebuilt as source-locked HSSC quality composite with two map panels and four evidence panels",
            "Added north arrow and scale bar to both evidentiary map panels",
            "Expanded the citywide map frame to visually align with the local zoom frame",
            "Added panel-local map legends for core semantic layers",
            "Aligned lower evidence-panel title baselines and removed non-essential micro-footnotes",
            "Formal outputs restricted to PDF, PNG and SVG",
            "Panel a enlarged by widening the top-left map frame and adapting the mother canvas",
            "Axes layout recorded for hidden-alignment review",
        ],
        "canvas_inches": FIGSIZE,
        "axes_layout": AXES_LAYOUT,
        "cartographic_controls": {
            "citywide_map": {"north_arrow": True, "scale_bar": "20 km"},
            "futian_local_map": {"north_arrow": True, "scale_bar": "5 km"},
        },
        "outputs": outputs,
        "zip": str(zip_path),
    }
    (SRC / "Figure_03_source_map.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def package(outputs: dict[str, str]) -> Path:
    zip_path = OUT / "HSSC_Figure_03_PDF_PNG_SVG.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for ext in ["pdf", "png", "svg"]:
            p = Path(outputs[ext])
            zf.write(p, arcname=p.name)
    return zip_path


def main() -> None:
    set_hssc_style()
    fig3_v28.SRC = SRC
    data = fig3_v28.load()

    fig = plt.figure(figsize=FIGSIZE)
    hssc_header(fig)
    draw_map_a(fig.add_axes(AXES_LAYOUT["a"]), data)
    draw_map_b(fig.add_axes(AXES_LAYOUT["b"]), data)
    draw_matrix_c(fig.add_axes(AXES_LAYOUT["c"]), data)
    draw_composition_d(fig.add_axes(AXES_LAYOUT["d"]), data)
    draw_rule_e(fig.add_axes(AXES_LAYOUT["e"]), data)
    draw_fingerprint_f(fig.add_axes(AXES_LAYOUT["f"]), data)

    outputs = save_all(fig, FORMAL / "Figure_03")
    render_audit_crops(Path(outputs["png"]))
    zip_path = package(outputs)
    write_source_manifest(outputs, zip_path)
    info = Image.open(outputs["png"])
    audit = {
        "figure": "Figure_03",
        "outputs": outputs,
        "zip": str(zip_path),
        "png_size": [info.width, info.height],
        "canvas_inches": FIGSIZE,
        "axes_layout": AXES_LAYOUT,
        "dpi": info.info.get("dpi"),
        "formal_package_files": ["Figure_03.pdf", "Figure_03.png", "Figure_03.svg"],
        "hard_gate_notes": [
            "citywide and local evidentiary maps include north arrow and scale bar",
            "citywide and local maps have panel-local semantic legends",
            "citywide and local map frames share a matched vertical extent",
            "evidence-panel titles use aligned baselines",
            "non-essential explanatory micro-footnotes hidden",
            "PDF/PNG/SVG only",
        ],
    }
    (AUDIT / "Figure_03_quality_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
