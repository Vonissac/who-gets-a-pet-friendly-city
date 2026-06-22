#!/usr/bin/env python3
"""Rebuild HSSC Fig. 5 with envelope-aware validation panels.

The script keeps the completed manual-validation payload but redraws the
five-panel composite with explicit label envelopes, balanced panel occupancy,
and a PDF/PNG/SVG-only formal package.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data_processed" / "validation"
OUT = ROOT / "hssc_fig5_quality_v1_20260617"
FORMAL = OUT / "formal_upload_figures"
AUDIT = OUT / "audit"
SRC = OUT / "source_data"
for folder in [FORMAL, AUDIT, SRC]:
    folder.mkdir(parents=True, exist_ok=True)


INK = "#24211f"
TEXT = "#4f4a45"
MUTED = "#766f66"
GRID = "#eee9e5"
RULE = "#2b6777"
RUST = "#c9655a"
GOLD = "#edc96b"
GREEN = "#5f967a"
BLUE = "#376a86"
TEAL = "#8db8b3"
GREY = "#c8c1b7"
PALE = "#f8f5ef"
WHITE = "#ffffff"

TYPE_EN = {
    "shopping_mall": "Malls",
    "hotel": "Hotels",
    "residential_property": "Residential",
    "park_or_recreation": "Parks",
}
DISTRICT_EN = {
    "福田区": "Futian",
    "罗湖区": "Luohu",
    "南山区": "Nanshan",
    "宝安区": "Baoan",
    "龙岗区": "Longgang",
    "龙华区": "Longhua",
    "坪山区": "Pingshan",
    "光明区": "Guangming",
    "盐田区": "Yantian",
}


def set_hssc_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.titlesize": 7.3,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.2,
            "axes.linewidth": 0.52,
            "axes.edgecolor": INK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.dpi": 600,
        }
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(VAL / "manual_validation_summary_v1.csv")
    dirs = pd.read_csv(VAL / "manual_validation_direction_classes_v1.csv")
    groups = pd.read_csv(VAL / "manual_validation_grouped_pass_rates_v1.csv")
    return summary, dirs, groups


def clean_axis(ax: plt.Axes, grid_axis: str | None = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.2, width=0.42, color=MUTED, labelcolor=TEXT, pad=2.5)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.45)
        ax.set_axisbelow(True)


def panel_title(ax: plt.Axes, letter: str, title: str, y: float = 1.018, x_title: float = 0.060) -> None:
    ax.text(0.0, y, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2, fontweight="bold", color=INK)
    ax.text(x_title, y + 0.003, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.7, fontweight="bold", color=INK)


def hssc_header(fig: plt.Figure) -> None:
    fig.text(
        0.023,
        0.975,
        "Fig. 5 | Manual validation narrows the claim",
        ha="left",
        va="top",
        fontsize=12.4,
        fontweight="bold",
        color=INK,
    )


def draw_validation_funnel(ax: plt.Axes, summary: pd.DataFrame, letter: str = "a") -> pd.DataFrame:
    df = summary.copy()
    labels = {
        "rule_sources_132": "Curated\nrule sources",
        "round1_300_priority": "Priority\ncandidates",
        "top1000_stratified": "Stratified\ncandidates",
    }
    colors = {
        "rule_sources_132": GREEN,
        "round1_300_priority": GOLD,
        "top1000_stratified": RUST,
    }
    df["label"] = df["sample"].map(labels)
    y = np.arange(len(df))
    ax.barh(y, df["rows"], color=PALE, edgecolor="#ddd5cc", lw=0.55, height=0.56)
    ax.barh(y, df["pass_n"], color=[colors[s] for s in df["sample"]], edgecolor=WHITE, lw=0.55, height=0.56)
    for i, row in df.iterrows():
        x = row["pass_n"] + 22 if row["pass_n"] < 150 else row["pass_n"] + 18
        ax.text(
            x,
            i,
            f"{int(row['pass_n'])}/{int(row['rows'])}  {row['pass_rate'] * 100:.1f}%",
            ha="left",
            va="center",
            fontsize=5.4,
            color=TEXT,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("records reviewed")
    ax.set_xlim(0, 1125)
    clean_axis(ax, "x")
    panel_title(ax, letter, "source reliability versus candidate visibility")
    return df[["sample", "rows", "pass_n", "pass_rate"]]


def draw_rule_mix(ax: plt.Axes, dirs: pd.DataFrame, letter: str = "b") -> pd.DataFrame:
    df = dirs[dirs["sample"].eq("rule_sources_132")].copy()
    order = [
        "verified_open_or_designated",
        "verified_conditional",
        "verified_restrictive",
        "verified_ambiguous_or_discretionary",
        "not_verified",
    ]
    labels = ["open or designated", "conditional", "restrictive", "ambiguous", "not verified"]
    short = ["open/\ndesignated", "conditional", "restrictive", "ambiguous", "not\nverified"]
    colors = [GREEN, GOLD, BLUE, GREY, RUST]
    vals = np.array([int(df.loc[df["verified_rule_direction_class"].eq(o), "n"].sum()) for o in order])
    total = vals.sum()
    left = 0.0
    for val, color in zip(vals, colors):
        share = val / total
        ax.barh([0], [share], left=left, height=0.46, color=color, edgecolor=WHITE, lw=0.5)
        if share >= 0.09:
            ax.text(
                left + share / 2,
                0,
                f"{val}\n{share * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=5.0,
                color=WHITE if color in [GREEN, BLUE, RUST] else INK,
                linespacing=0.86,
            )
        left += share
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.36, 0.48)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("share of 132 curated source records (%)")
    clean_axis(ax, None)
    panel_title(ax, letter, "verified rules are mixed, not simply permissive")
    for i, (lab, color) in enumerate(zip(short, colors)):
        x = 0.015 + i * 0.190
        ax.add_patch(Rectangle((x, 0.090), 0.018, 0.055, transform=ax.transAxes, facecolor=color, edgecolor="none"))
        ax.text(x + 0.026, 0.117, lab, transform=ax.transAxes, ha="left", va="center", fontsize=4.2, color=TEXT, linespacing=0.82)
    return pd.DataFrame({"class": order, "label": labels, "n": vals, "share": vals / total})


def draw_sector_pass_rates(ax: plt.Axes, groups: pd.DataFrame, letter: str = "d") -> pd.DataFrame:
    df = groups[groups["primary_venue_type"].notna()].copy()
    df = df[df["sample"].isin(["round1_300_priority", "top1000_stratified"])]
    df["type"] = df["primary_venue_type"].map(TYPE_EN)
    order = ["Malls", "Hotels", "Residential", "Parks"]
    sample_order = ["round1_300_priority", "top1000_stratified"]
    sample_labels = {"round1_300_priority": "priority 300", "top1000_stratified": "stratified 1000"}
    colors = {"round1_300_priority": GOLD, "top1000_stratified": RUST}
    x = np.arange(len(order))
    width = 0.33
    for j, sample in enumerate(sample_order):
        sub = df[df["sample"].eq(sample)].set_index("type").reindex(order)
        vals = sub["pass_rate"].fillna(0).to_numpy() * 100
        ns = sub["n"].fillna(0).astype(int).to_numpy()
        pos = x + (j - 0.5) * width
        ax.bar(pos, vals, width=width, color=colors[sample], edgecolor=WHITE, lw=0.45, alpha=0.90, label=sample_labels[sample])
        for px, val, n in zip(pos, vals, ns):
            ax.text(px, val + 0.85, f"{val:.1f}%\nn={n}", ha="center", va="bottom", fontsize=4.25, color=TEXT, linespacing=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylabel("source-verification pass rate (%)")
    ax.set_ylim(0, 30.2)
    clean_axis(ax, "y")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.99, 1.00), ncol=2, handlelength=1.1, columnspacing=0.9)
    panel_title(ax, letter, "malls publicise rules more often than other hosts")
    return df[["sample", "primary_venue_type", "n", "pass_n", "pass_rate"]]


def draw_district_pass_rates(ax: plt.Axes, groups: pd.DataFrame, letter: str = "e") -> pd.DataFrame:
    df = groups[groups["district_name"].notna()].copy()
    df = df[df["sample"].eq("top1000_stratified")].copy()
    df["district"] = df["district_name"].map(DISTRICT_EN)
    df = df.sort_values("pass_rate")
    y = np.arange(len(df))
    vals = df["pass_rate"].to_numpy() * 100
    ax.hlines(y, 0, vals, color="#dcd5cc", lw=2.2)
    ax.scatter(vals, y, s=32, color=RUST, edgecolor=WHITE, lw=0.55, zorder=3)
    for yi, (_, row) in enumerate(df.iterrows()):
        ax.text(row["pass_rate"] * 100 + 0.62, yi, f"{int(row['pass_n'])}/{int(row['n'])}", ha="left", va="center", fontsize=4.55, color=TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(df["district"])
    ax.set_ylim(-0.55, len(df) - 0.25)
    ax.set_xlim(0, 15.5)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("stratified queue pass rate (%)")
    clean_axis(ax, "x")
    panel_title(ax, letter, "rule-publicity scarcity is uneven across districts")
    return df[["district_name", "district", "n", "pass_n", "pass_rate"]]


def draw_direction_classes(ax: plt.Axes, dirs: pd.DataFrame, letter: str = "c") -> pd.DataFrame:
    df = dirs[dirs["sample"].eq("top1000_stratified")].copy()
    order = [
        "not_source_verifiable",
        "verified_conditional_or_zoned_access",
        "verified_open_or_event_access",
        "verified_restrictive",
        "verified_pet_service_identity_not_ordinary_host_access",
    ]
    labels = ["not\nverified", "conditional", "open /\nevent", "restrictive", "pet-service"]
    colors = [GREY, GOLD, GREEN, BLUE, TEAL]
    vals = np.array([int(df.loc[df["verified_rule_direction_class"].eq(o), "n"].sum()) for o in order])
    total = vals.sum()
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, edgecolor=WHITE, lw=0.5, width=0.72)
    for xi, val in zip(x, vals):
        offset = 20 if val > 100 else 16
        ax.text(xi, val + offset, f"{val}\n{val / total * 100:.1f}%", ha="center", va="bottom", fontsize=4.65, color=TEXT, linespacing=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("records", labelpad=2)
    ax.set_ylim(0, 1115)
    clean_axis(ax, "y")
    panel_title(ax, letter, "the middle terrain is mostly rule non-publicity")
    return pd.DataFrame({"class": order, "label": labels, "n": vals, "share": vals / total})


def draw_claim_upgrade(ax: plt.Axes, summary: pd.DataFrame) -> pd.DataFrame:
    ax.set_axis_off()
    panel_title(ax, "f", "claim upgrade after validation")
    rows = [
        ("Curated rule ledger", "upgrade", "highly reliable source layer", GREEN),
        ("Rule-liminal hosts", "reframe", "publicity frontier;\nnot hidden openness", GOLD),
        ("Suppressed emergence", "retain", "service-rule conversion gap", RUST),
        ("Causality", "bound", "diagnostic sequence;\nnot strong causal effect", BLUE),
    ]
    y0 = 0.78
    for i, (left, tag, right, color) in enumerate(rows):
        y = y0 - i * 0.175
        ax.add_patch(Rectangle((0.025, y - 0.054), 0.320, 0.108, transform=ax.transAxes, facecolor=PALE, edgecolor="#ddd5cc", lw=0.58))
        ax.text(0.044, y, left, transform=ax.transAxes, ha="left", va="center", fontsize=5.10, color=INK, fontweight="bold")
        ax.add_patch(Rectangle((0.372, y - 0.044), 0.130, 0.088, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=0.90))
        ax.text(0.437, y, tag, transform=ax.transAxes, ha="center", va="center", fontsize=4.65, color=WHITE if color in [GREEN, RUST, BLUE] else INK, fontweight="bold")
        ax.text(0.550, y, right, transform=ax.transAxes, ha="left", va="center", fontsize=5.00, color=TEXT, linespacing=0.92)
    return pd.DataFrame(rows, columns=["evidence_object", "claim_operation", "revised_claim", "color"])


def save_outputs(fig: plt.Figure) -> dict[str, Any]:
    paths: dict[str, str] = {}
    for ext in ["pdf", "png", "svg"]:
        path = FORMAL / f"Figure_05.{ext}"
        fig.savefig(path, dpi=600)
        paths[ext] = str(path)
    plt.close(fig)
    im = Image.open(paths["png"]).convert("RGB")
    im.save(paths["png"], dpi=(600, 600))
    return {"width_px": im.width, "height_px": im.height, "outputs": paths}


def package_outputs() -> Path:
    zip_path = OUT / "HSSC_Figure_05_PDF_PNG_SVG.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for ext in ["pdf", "png", "svg"]:
            zf.write(FORMAL / f"Figure_05.{ext}", arcname=f"Figure_05.{ext}")
    return zip_path


def write_audit_preview() -> None:
    src = FORMAL / "Figure_05.png"
    im = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(im)
    w, h = im.size
    for y in [int(h * 0.112), int(h * 0.545)]:
        draw.line([(0, y), (w, y)], fill=(220, 220, 220), width=1)
    im.save(AUDIT / "Figure_05_layout_audit_overlay.png")


def render() -> None:
    set_hssc_style()
    summary, dirs, groups = load_data()
    fig = plt.figure(figsize=(7.45, 5.55))
    gs = GridSpec(
        2,
        6,
        figure=fig,
        left=0.073,
        right=0.985,
        bottom=0.092,
        top=0.865,
        wspace=0.42,
        hspace=0.455,
        height_ratios=[1.0, 1.05],
    )
    hssc_header(fig)
    for old_csv in SRC.glob("Figure_05_panel_*.csv"):
        old_csv.unlink()
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])
    source_tables = {
        "panel_a_validation_funnel": draw_validation_funnel(ax_a, summary, "a"),
        "panel_b_rule_direction_mix": draw_rule_mix(ax_b, dirs, "b"),
        "panel_c_direction_classes": draw_direction_classes(ax_c, dirs, "c"),
        "panel_d_venue_type_pass_rates": draw_sector_pass_rates(ax_d, groups, "d"),
        "panel_e_district_pass_rates": draw_district_pass_rates(ax_e, groups, "e"),
    }
    for name, table in source_tables.items():
        table.to_csv(SRC / f"Figure_05_{name}.csv", index=False)
    info = save_outputs(fig)
    zip_path = package_outputs()
    write_audit_preview()
    manifest = {
        "figure": "Figure_05",
        "title": "Manual validation narrows the claim",
        "canvas_inches": [7.45, 5.55],
        "formal_outputs": info["outputs"],
        "package": str(zip_path),
        "width_px": info["width_px"],
        "height_px": info["height_px"],
        "source_data": sorted(p.name for p in SRC.glob("Figure_05_*.csv")),
        "quality_gates": [
            "PDF/PNG/SVG-only formal package",
            "English-only figure-facing text",
            "Helvetica/Arial-like sans-serif typography",
            "data-panel envelope spacing includes ticks, labels, legends, values, titles, and panel letters",
            "panel letters renumbered in visible reading order",
            "low-evidence claim-translation panel removed",
            "whole-figure regression after local panel redrawing",
        ],
    }
    (OUT / "Figure_05_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(OUT / "Figure_05_manifest.json", AUDIT / "Figure_05_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    render()
