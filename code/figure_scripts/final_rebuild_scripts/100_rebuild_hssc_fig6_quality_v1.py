#!/usr/bin/env python3
"""Rebuild HSSC Fig. 6 with formal-figure hard gates.

This wrapper preserves the v28 network/threshold payload and the v29
morphology payload while removing decorative header elements, tightening the
panel layout, adding cartographic controls to map panels, and exporting a
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
V29_CODE = ROOT / "submission_package" / "code" / "figures_v29_expanded_system"

OUT = ROOT / "hssc_fig6_quality_v1_20260617"
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


fig5_topology = import_module("fig5_topology_v28_fig6_quality", V28_CODE / "build_fig5_topology_threshold_v28.py", V28_CODE)
fig29 = import_module("fig29_morphology_fig6_quality", V29_CODE / "build_v29_expanded_system.py", V29_CODE)


INK = "#24211f"
TEXT = "#4f4a45"
MUTED = "#766f66"
WHITE = "#ffffff"
RULE = "#2b6777"


def set_hssc_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.2,
            "axes.titlesize": 6.7,
            "axes.labelsize": 5.8,
            "xtick.labelsize": 4.8,
            "ytick.labelsize": 4.8,
            "legend.fontsize": 4.3,
            "axes.linewidth": 0.46,
            "axes.edgecolor": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.dpi": 600,
        }
    )


def hssc_header(fig: plt.Figure) -> None:
    fig.text(
        0.022,
        0.976,
        "Fig. 6 | Network, threshold and morphology diagnostics",
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        color=INK,
    )


def remove_microtext(ax: plt.Axes) -> None:
    """Remove explanatory footnotes while preserving data labels and legends."""
    patterns = (
        "grids;",
        "numbers inside",
        "vertical ticks",
        "pale bars",
        "each row",
        "signed coefficients",
        "segment labels",
        "diagnostic evidence",
        "hexagons summarize",
        "highlighted points",
        "surface =",
        "AUC",
        "counts are",
        "not causal",
    )
    for text in list(ax.texts):
        value = text.get_text()
        if any(p in value for p in patterns):
            text.set_visible(False)


def panel_title(ax: plt.Axes, letter: str, title: str, y: float = 1.018, x_title: float = 0.065) -> None:
    ax.text(0.0, y, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0, fontweight="bold", color=INK)
    ax.text(x_title, y + 0.003, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.35, fontweight="bold", color=INK)


def replace_panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    for text in list(ax.texts):
        _x, y = text.get_position()
        if text.get_transform() == ax.transAxes and y >= 0.96:
            text.set_visible(False)
    panel_title(ax, letter, title)


def compact_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6f6961")
    ax.spines["bottom"].set_color("#6f6961")
    ax.spines["left"].set_linewidth(0.46)
    ax.spines["bottom"].set_linewidth(0.46)
    ax.tick_params(colors="#67605a", width=0.38, length=2.1, pad=2)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#ece8df", lw=0.35)


def draw_panel_b_clean(ax: plt.Axes, data: dict[str, object], letter: str = "b") -> pd.DataFrame:
    grid = data["grid"].copy()
    counts = grid.groupby(["district_name", "topology_ecology_class"]).size().reset_index(name="n")
    pivot = counts.pivot_table(index="district_name", columns="topology_ecology_class", values="n", fill_value=0)
    pivot = pivot.reindex(fig5_topology.DISTRICT_ORDER).fillna(0)
    pivot = pivot.reindex(columns=fig5_topology.TOPO_ORDER, fill_value=0)
    share = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
    core_share = share["network_core_with_pet_ecology"] + share["network_core_without_high_pet_ecology"]
    order = core_share.sort_values(ascending=False).index.tolist()
    share = share.reindex(order)
    pivot = pivot.reindex(order)
    core_share = core_share.reindex(order)

    y = np.arange(len(share.index))
    left = np.zeros(len(share))
    for klass in fig5_topology.TOPO_ORDER:
        vals = share[klass].values
        ax.barh(
            y,
            vals,
            left=left,
            height=0.58,
            color=fig5_topology.TOPO_COLORS[klass],
            edgecolor="white",
            linewidth=0.24,
            alpha=0.90,
        )
        for yi, lft, val, district in zip(y, left, vals, share.index):
            n = int(pivot.loc[district, klass])
            if val >= 0.10 and klass != "low_network_pet_rule_signal":
                color = "white" if klass in ["network_core_with_pet_ecology", "network_core_without_high_pet_ecology"] else INK
                ax.text(lft + val / 2, yi, f"{val * 100:.0f}", ha="center", va="center", fontsize=4.0, color=color)
            elif klass == "low_network_pet_rule_signal" and val >= 0.78:
                ax.text(lft + val - 0.018, yi, f"{n:,}", ha="right", va="center", fontsize=3.7, color="#8a8277")
        left += vals

    ax2 = ax.inset_axes([0.735, 0.09, 0.205, 0.78])
    ax2.scatter(
        core_share.values,
        y,
        s=20 + 105 * core_share.values / max(core_share.max(), 1e-9),
        color="#9f7b45",
        edgecolor="white",
        linewidth=0.34,
        zorder=4,
    )
    for yi, val in zip(y, core_share.values):
        ax2.hlines(yi, 0, val, color="#d8c7a9", lw=0.58, zorder=2)
    ax2.set_xlim(0, max(0.22, core_share.max() * 1.16))
    ax2.set_ylim(-0.6, len(y) - 0.4)
    ax2.invert_yaxis()
    ax2.set_yticks([])
    ax2.set_xticks([0, round(float(core_share.max()), 2)])
    ax2.set_xticklabels(["0", f"{core_share.max() * 100:.0f}%"], fontsize=3.6)
    ax2.text(0.98, 0.98, "core\nshare", transform=ax2.transAxes, ha="right", va="top", fontsize=3.8, color=MUTED)
    compact_axis(ax2)

    ax.set_yticks(y)
    ax.set_yticklabels([fig5_topology.DISTRICT_EN.get(i, i) for i in share.index])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50", "100"])
    ax.set_xlabel("share of district grids (%)")
    ax.set_ylim(-0.65, len(share.index) - 0.35)
    ax.invert_yaxis()
    compact_axis(ax, "x")
    handles = [Rectangle((0, 0), 1, 1, facecolor=fig5_topology.TOPO_COLORS[k], edgecolor="none") for k in fig5_topology.TOPO_ORDER]
    labels = ["Core+ecology", "Core low ecology", "Local pet ecology", "Low signal"]
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.015, 1.010),
        ncol=4,
        frameon=False,
        fontsize=3.55,
        columnspacing=0.38,
        handlelength=0.65,
        handletextpad=0.22,
        borderaxespad=0.0,
    )
    panel_title(ax, letter, "district topology profile")
    return share.reset_index()


def draw_panel_e_clean(ax: plt.Axes, data: dict[str, object], letter: str = "e") -> pd.DataFrame:
    scenarios = data["scenarios"].copy()
    rows = []
    for _, row in scenarios.iterrows():
        for host, n in fig5_topology.parse_dict(row["adopters_by_type"]).items():
            rows.append({"scenario": row["scenario"], "host_type": host, "n": int(n), "adoption_share": float(row["adoption_share"])})
    out = pd.DataFrame(rows)
    order = list(fig5_topology.SCENARIO_LABELS.keys())
    host_order = ["restaurant", "shopping_mall", "hotel", "park_or_recreation", "residential_property"]
    pivot = out.pivot_table(index="scenario", columns="host_type", values="n", aggfunc="sum", fill_value=0).reindex(order).fillna(0)
    pivot = pivot.reindex(columns=host_order, fill_value=0)
    share = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
    y = np.arange(len(share.index))
    left = np.zeros(len(share))
    for host in host_order:
        vals = share[host].values
        ax.barh(y, vals, left=left, height=0.56, color=fig5_topology.HOST_COLORS[host], edgecolor="white", linewidth=0.24, alpha=0.90)
        for yi, lft, val, scenario in zip(y, left, vals, share.index):
            n = int(pivot.loc[scenario, host])
            if n > 0 and val >= 0.075:
                color = "white" if host in ["restaurant", "shopping_mall", "hotel"] else INK
                ax.text(lft + val / 2, yi, f"{n:,}", ha="center", va="center", fontsize=3.9, color=color)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([fig5_topology.SCENARIO_LABELS[s] for s in share.index])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "50", "100"])
    ax.set_xlabel("composition of simulated adopters (%)")
    ax.set_ylim(-0.58, len(share.index) - 0.34)
    ax.invert_yaxis()
    compact_axis(ax, "x")
    handles = [Rectangle((0, 0), 1, 1, facecolor=fig5_topology.HOST_COLORS[h], edgecolor="none") for h in host_order]
    labels = [fig5_topology.HOST_LABELS[h] for h in host_order]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, -0.245), ncol=5, frameon=False, fontsize=4.1, columnspacing=0.58, handlelength=0.88)
    panel_title(ax, letter, "scenario composition")
    out.to_csv(SRC / "Figure_06_panel_e_long_source.csv", index=False)
    return share.reset_index()


def draw_panel_f_clean(ax: plt.Axes, data: dict[str, object], letter: str = "f") -> pd.DataFrame:
    coef = data["coefficients"].copy()
    metrics = data["metrics"].copy()
    keep_features = [
        "topology_norm",
        "pet_ecology_norm",
        "liminal_potential_norm",
        "positive_rule_norm_v51",
        "rule_liminal_exposure_norm_v51",
        "restrictive_norm",
        "edge_cell_no",
    ]
    label_map = {
        "topology_norm": "Topology",
        "pet_ecology_norm": "Pet ecology",
        "liminal_potential_norm": "Liminal potential",
        "positive_rule_norm_v51": "Positive rule",
        "rule_liminal_exposure_norm_v51": "Rule-liminal exposure",
        "restrictive_norm": "Restriction",
        "edge_cell_no": "Non-edge cell",
    }
    target_map = {"is_suppressed_frontier": "Suppressed frontier", "has_positive_rule": "Visible positive rule"}
    sub = coef[coef["spec"].eq("core_plus_controls") & coef["feature"].isin(keep_features)].copy()
    sub = sub[sub["target"].isin(target_map)].copy()
    sub["feature_label"] = sub["feature"].map(label_map)
    sub["target_label"] = sub["target"].map(target_map)
    target_order = ["Visible positive rule", "Suppressed frontier"]
    feature_order = [label_map[f] for f in keep_features]
    sub["target_label"] = pd.Categorical(sub["target_label"], categories=target_order, ordered=True)
    sub["feature_label"] = pd.Categorical(sub["feature_label"], categories=feature_order, ordered=True)
    sub = sub.sort_values(["target_label", "feature_label"])
    offsets = {"Visible positive rule": -0.14, "Suppressed frontier": 0.14}
    colors = {"Visible positive rule": RULE, "Suppressed frontier": "#d88973"}
    ybase = np.arange(len(feature_order))[::-1]
    ymap = {feat: y for feat, y in zip(feature_order, ybase)}
    ax.axvline(0, color="#6f6961", lw=0.56, alpha=0.75, zorder=1)
    for target in target_order:
        ss = sub[sub["target_label"].astype(str).eq(target)]
        yy = np.array([ymap[str(f)] + offsets[target] for f in ss["feature_label"]])
        xx = ss["coefficient"].to_numpy()
        ax.hlines(yy, 0, xx, color=colors[target], lw=0.90, alpha=0.80)
        ax.scatter(xx, yy, s=20 + 10 * np.clip(np.abs(xx), 0, 4), facecolor="white", edgecolor=colors[target], linewidth=0.68, zorder=4, label=target)
        for xval, yval in zip(xx, yy):
            if abs(xval) >= 3.0 or target == "Suppressed frontier":
                ha = "left" if xval >= 0 else "right"
                xpos = xval + (0.14 if xval >= 0 else -0.14)
                ax.text(xpos, yval, f"{xval:+.2f}", ha=ha, va="center", fontsize=3.7, color=colors[target])
    ax.set_yticks(ybase)
    ax.set_yticklabels(feature_order)
    ax.tick_params(axis="y", pad=3, labelsize=4.55)
    ax.set_xlim(-7.05, 5.20)
    ax.set_xticks([-6, -3, 0, 3])
    ax.set_xlabel("standardised model coefficient")
    compact_axis(ax, "x")
    ax.legend(loc="lower right", bbox_to_anchor=(0.985, 0.015), ncol=1, frameon=False, fontsize=3.9, handletextpad=0.32, borderaxespad=0.0)
    panel_title(ax, letter, "controlled diagnostic coefficients")
    out = sub.merge(metrics[["target", "roc_auc", "average_precision", "balanced_accuracy_at_0_5", "spec"]], on=["target", "spec"], how="left")
    return out


def clean_panel_c_left_labels(ax: plt.Axes) -> None:
    ax.set_yticklabels([])
    labels = [("Pet-service\nedges", 0.72), ("Rule-seed\nedges", 0.28)]
    for lab, y in labels:
        ax.text(0.035, y, lab, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=4.6, color=TEXT, bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.70, "pad": 0.18}, zorder=20)


def clean_panel_d_endpoint_labels(ax: plt.Axes) -> None:
    for text in ax.texts:
        value = text.get_text()
        if value.startswith("Permissive"):
            text.set_position((4.86, 0.777))
            text.set_ha("left")
            text.set_fontsize(3.7)
        elif value.startswith("Balanced"):
            text.set_position((3.35, 0.646))
            text.set_ha("left")
            text.set_fontsize(3.7)
        elif value.startswith("Conservative"):
            text.set_position((2.45, 0.034))
            text.set_ha("left")
            text.set_fontsize(3.7)


def add_north_arrow_and_scale(ax: plt.Axes, scale_m: int = 20000, x: float = 0.815, y: float = 0.100) -> None:
    """Add quiet cartographic controls in projected map coordinates."""
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    w = maxx - minx
    h = maxy - miny
    sx0 = minx + x * w
    sx1 = min(sx0 + scale_m, minx + 0.965 * w)
    sy = miny + y * h
    tick = 0.014 * h
    ax.plot([sx0, sx1], [sy, sy], color=INK, lw=0.70, zorder=90, solid_capstyle="butt")
    ax.plot([sx0, sx0], [sy - tick, sy + tick], color=INK, lw=0.48, zorder=91)
    ax.plot([sx1, sx1], [sy - tick, sy + tick], color=INK, lw=0.48, zorder=91)
    ax.text(
        (sx0 + sx1) / 2,
        sy + 0.022 * h,
        f"{scale_m // 1000} km",
        ha="center",
        va="bottom",
        fontsize=4.0,
        color=INK,
        zorder=92,
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.72, "pad": 0.16},
    )
    ax.annotate(
        "",
        xy=(0.940, 0.895),
        xytext=(0.940, 0.790),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "lw": 0.72, "color": INK, "mutation_scale": 7.2},
        zorder=93,
    )
    ax.text(
        0.940,
        0.915,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=4.3,
        fontweight="bold",
        color=INK,
        zorder=94,
    )


def add_panel_a_reading_key(ax: plt.Axes, source: pd.DataFrame) -> None:
    core_classes = {
        "network_core_with_pet_ecology",
        "network_core_without_high_pet_ecology",
    }
    total = len(source)
    top_quartile = int((source["topology_surface"] >= source["topology_surface"].quantile(0.75)).sum())
    core = int(source["topology_ecology_class"].isin(core_classes).sum())
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    h = maxy - miny
    ax.set_ylim(miny - 0.090 * h, maxy - 0.010 * h)
    key = ax.inset_axes([0.030, 0.022, 0.395, 0.185])
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.set_axis_off()
    key.add_patch(Rectangle((0, 0), 1, 1, facecolor=WHITE, edgecolor="#d8d3ca", linewidth=0.38, alpha=0.88, zorder=0))
    key.text(0.06, 0.84, "Network exposure percentile", ha="left", va="center", fontsize=4.0, fontweight="bold", color=INK)
    grad = np.linspace(0, 1, 128).reshape(1, -1)
    key.imshow(grad, extent=(0.07, 0.93, 0.60, 0.71), cmap=fig5_topology.TOPO_CMAP, aspect="auto", zorder=1)
    key.text(0.07, 0.50, "low", ha="left", va="center", fontsize=3.5, color=MUTED)
    key.text(0.93, 0.50, "high", ha="right", va="center", fontsize=3.5, color=MUTED)
    key.text(0.06, 0.340, f"{total:,} 500 m grids", ha="left", va="center", fontsize=3.6, color=TEXT)
    key.text(0.06, 0.205, f"{top_quartile:,} top-quartile exposure", ha="left", va="center", fontsize=3.6, color=TEXT)
    key.text(0.06, 0.075, f"{core:,} topology-core grids", ha="left", va="center", fontsize=3.6, color=TEXT)


def map_panel_frame(ax: plt.Axes) -> None:
    ax.set_axis_on()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#d8d3ca")
        spine.set_linewidth(0.46)


def draw_morphology_map(ax: plt.Axes, grid, boundary, letter: str = "h") -> pd.DataFrame:
    g = grid.copy()
    b = boundary.copy()
    if g.crs and str(g.crs).upper() != "EPSG:32649":
        g = g.to_crs("EPSG:32649")
        b = b.to_crs("EPSG:32649")
    color_map = {
        "suppressed_emergence_frontier": fig29.RUST,
        "emergent_capability_core": fig29.GREEN,
        "mixed_or_transitional_zone": fig29.GOLD,
        "rule_first_demonstration_zone": fig29.BLUE,
        "low_pet_city_signal": "#efebe3",
    }
    g.plot(
        ax=ax,
        color=g["grid_emergence_type_v51"].map(color_map).fillna("#efebe3"),
        edgecolor="#fffdf9",
        linewidth=0.010,
        alpha=0.92,
        rasterized=True,
        zorder=4,
    )
    b.boundary.plot(ax=ax, color="#625c55", linewidth=0.40, alpha=0.88, zorder=12)
    minx, miny, maxx, maxy = b.total_bounds
    pad_x = 1200
    pad_y = 1600
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - 0.090 * (maxy - miny) - pad_y, maxy + 0.002 * (maxy - miny) + pad_y)
    map_panel_frame(ax)
    ax.set_aspect("equal")
    items = [
        ("suppressed frontier", fig29.RUST),
        ("capability core", fig29.GREEN),
        ("mixed/transitional", fig29.GOLD),
        ("rule-first zone", fig29.BLUE),
    ]
    for i, (lab, color) in enumerate(items):
        ax.add_patch(Rectangle((0.030, 0.088 + i * 0.046), 0.020, 0.020, transform=ax.transAxes, facecolor=color, edgecolor="white", lw=0.18, zorder=30))
        ax.text(0.057, 0.098 + i * 0.046, lab, transform=ax.transAxes, ha="left", va="center", fontsize=3.75, color=MUTED, zorder=31)
    add_north_arrow_and_scale(ax, 20000, x=0.785, y=0.095)
    panel_title(ax, letter, "spatial typology map")
    out = g[["grid_id", "district_name", "grid_emergence_type_v51", "suppression_index_v51", "emergence_index_v51"]].copy()
    out.to_csv(SRC / "Figure_06_panel_h_spatial_typology.csv", index=False)
    return out


def save_outputs(fig: plt.Figure) -> dict[str, Any]:
    paths: dict[str, str] = {}
    for ext in ["pdf", "png", "svg"]:
        path = FORMAL / f"Figure_06.{ext}"
        fig.savefig(path, dpi=600)
        paths[ext] = str(path)
    plt.close(fig)
    im = Image.open(paths["png"]).convert("RGB")
    im.save(paths["png"], dpi=(600, 600))
    return {"width_px": im.width, "height_px": im.height, "outputs": paths}


def package_outputs() -> Path:
    zip_path = OUT / "HSSC_Figure_06_PDF_PNG_SVG.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for ext in ["pdf", "png", "svg"]:
            zf.write(FORMAL / f"Figure_06.{ext}", arcname=f"Figure_06.{ext}")
    return zip_path


LOWER_COL_LEFTS = [0.082, 0.392, 0.702]
LOWER_COL_WIDTH = 0.272
AXES_LAYOUT = {
    "a": [0.054, 0.596, 0.438, 0.300],
    "b": [0.507, 0.596, 0.438, 0.300],
    "c": [LOWER_COL_LEFTS[0], 0.342, LOWER_COL_WIDTH, 0.212],
    "d": [LOWER_COL_LEFTS[1], 0.342, LOWER_COL_WIDTH, 0.212],
    "e": [LOWER_COL_LEFTS[2], 0.342, LOWER_COL_WIDTH, 0.212],
    "f": [LOWER_COL_LEFTS[0], 0.076, LOWER_COL_WIDTH, 0.220],
    "g": [LOWER_COL_LEFTS[1], 0.076, LOWER_COL_WIDTH, 0.220],
    "h": [LOWER_COL_LEFTS[2], 0.076, LOWER_COL_WIDTH, 0.220],
}


def write_audit_preview() -> None:
    src = FORMAL / "Figure_06.png"
    im = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(im)
    w, h = im.size
    for y in [int(h * 0.325), int(h * 0.612)]:
        draw.line([(0, y), (w, y)], fill=(220, 220, 220), width=1)
    im.save(AUDIT / "Figure_06_layout_audit_overlay.png")

    aligned = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(aligned)
    for panel, (x, y, width, height) in AXES_LAYOUT.items():
        color = (210, 48, 48) if panel in {"a", "b"} else (32, 86, 210)
        for xpos in [x, x + width]:
            xx = int(xpos * w)
            draw.line([(xx, 0), (xx, h)], fill=color, width=2)
        for ypos in [y, y + height]:
            yy = int((1 - ypos) * h)
            draw.line([(0, yy), (w, yy)], fill=(160, 160, 160), width=1)
    aligned.save(AUDIT / "Figure_06_hidden_alignment_overlay.png")


def render() -> None:
    set_hssc_style()
    topo = fig5_topology.load()
    cluster, _morph_model, morph_grid, morph_boundary = fig29.load_morphology()
    for old_csv in SRC.glob("Figure_06_panel_*.csv"):
        old_csv.unlink()

    fig = plt.figure(figsize=(8.60, 7.80))
    hssc_header(fig)

    # Explicit axes positions enforce the hidden-alignment gate. Do not shift
    # individual lower-row panels to solve label pressure; adjust margins/fonts.
    ax_a = fig.add_axes(AXES_LAYOUT["a"])
    ax_b = fig.add_axes(AXES_LAYOUT["b"])
    ax_c = fig.add_axes(AXES_LAYOUT["c"])
    ax_d = fig.add_axes(AXES_LAYOUT["d"])
    ax_e = fig.add_axes(AXES_LAYOUT["e"])
    ax_f = fig.add_axes(AXES_LAYOUT["f"])
    ax_g = fig.add_axes(AXES_LAYOUT["g"])
    ax_h = fig.add_axes(AXES_LAYOUT["h"])

    source_tables: dict[str, pd.DataFrame] = {}
    source_tables["panel_a_network_exposure"] = fig5_topology.draw_topology_surface(ax_a, topo, "a")
    add_panel_a_reading_key(ax_a, source_tables["panel_a_network_exposure"])
    add_north_arrow_and_scale(ax_a, 20000, x=0.745, y=0.115)
    map_panel_frame(ax_a)
    replace_panel_title(ax_a, "a", "network exposure surface")

    source_tables["panel_b_spatial_typology"] = draw_morphology_map(ax_b, morph_grid, morph_boundary, "b")

    source_tables["panel_c_topology_profile"] = draw_panel_b_clean(ax_c, topo, "c")
    ax_c.set_xlabel("")

    source_tables["panel_d_edge_distance"] = fig5_topology.draw_edge_distance_structure(ax_d, topo, "d")
    replace_panel_title(ax_d, "d", "edge-distance structure")
    ax_d.set_ylabel("")
    ax_d.set_xlabel("")
    ax_d.tick_params(axis="y", pad=5)
    clean_panel_c_left_labels(ax_d)

    source_tables["panel_e_threshold"] = fig5_topology.draw_threshold_trajectories(ax_e, topo, "e")
    clean_panel_d_endpoint_labels(ax_e)
    leg_d = ax_e.get_legend()
    if leg_d is not None:
        leg_d.remove()
        ax_e.legend(loc="upper left", bbox_to_anchor=(0.015, 0.865), frameon=False, fontsize=3.8, handlelength=1.0, borderaxespad=0.0)
    replace_panel_title(ax_e, "e", "threshold trajectories")
    ax_e.set_xlabel("")

    source_tables["panel_f_model"] = draw_panel_f_clean(ax_f, topo, "f")

    source_tables["panel_g_scenario"] = draw_panel_e_clean(ax_g, topo, "g")

    fig29.panel9a(ax_h, cluster)
    replace_panel_title(ax_h, "h", "service-morphology phase space")
    source_tables["panel_h_morphology_phase"] = cluster[["grid_id", "district_name", "baseline_mismatch_type", "service_score", "affordance_morphology_score"]].copy()

    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h]:
        remove_microtext(ax)

    for name, table in source_tables.items():
        table.to_csv(SRC / f"Figure_06_{name}.csv", index=False)

    info = save_outputs(fig)
    zip_path = package_outputs()
    write_audit_preview()
    manifest = {
        "figure": "Figure_06",
        "title": "Network, threshold and morphology diagnostics",
        "canvas_inches": [8.60, 7.80],
        "formal_outputs": info["outputs"],
        "package": str(zip_path),
        "width_px": info["width_px"],
        "height_px": info["height_px"],
        "source_data": sorted(p.name for p in SRC.glob("Figure_06_panel_*.csv")),
        "axes_layout": AXES_LAYOUT,
        "quality_gates": [
            "PDF/PNG/SVG-only formal package",
            "no decorative top rule or explanatory subtitle",
            "non-essential explanatory microtext removed",
            "map panels include north arrow and scale bar",
            "data-panel envelope spacing includes tick labels, legends, values, titles, and panel letters",
            "whole-figure regression after local panel redrawing",
            "hidden column alignment: c/f, d/g and e/h share exact axes left boundaries",
            "top map frames are equal-sized and optically paired with a controlled gutter",
        ],
    }
    (OUT / "Figure_06_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(OUT / "Figure_06_manifest.json", AUDIT / "Figure_06_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    render()
