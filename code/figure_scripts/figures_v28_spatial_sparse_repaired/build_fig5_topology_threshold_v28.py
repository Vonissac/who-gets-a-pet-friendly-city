#!/usr/bin/env python3
"""Build v14 Fig. 5: spatial topology and threshold dynamics.

Figure claim:
Companion-animal urban capability is not reducible to local pet-service
density. It is a relational threshold condition shaped by topology, rule
exposure and venue-type thresholds. The threshold simulation is reported as a
mechanism diagnostic, not as observed diffusion or a policy prediction.
"""

from __future__ import annotations

import ast
import math
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore", category=UserWarning)

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from build_fig2_v28 import (  # noqa: E402
    BLUE,
    BLUE_DARK,
    COMP,
    DISTRICT_EN,
    GOLD,
    GREEN,
    GRID,
    INK,
    LAND,
    MUTED,
    PANELS,
    PREVIEWS,
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


OUT_STEM = "fig5_topology_threshold_v28"
FIXED_PANEL_SIZE = (3.3, 2.75)

DISTRICT_ORDER = ["福田区", "龙华区", "宝安区", "龙岗区", "南山区", "罗湖区", "光明区", "坪山区", "盐田区"]
TOPO_ORDER = [
    "network_core_with_pet_ecology",
    "network_core_without_high_pet_ecology",
    "pet_ecology_peripheral_or_local",
    "low_network_pet_rule_signal",
]
TOPO_LABELS = {
    "network_core_with_pet_ecology": "Topology core\nwith ecology",
    "network_core_without_high_pet_ecology": "Topology core\nlow ecology",
    "pet_ecology_peripheral_or_local": "Pet ecology\nlocal only",
    "low_network_pet_rule_signal": "Low topology\nsignal",
}
TOPO_COLORS = {
    "network_core_with_pet_ecology": RUST,
    "network_core_without_high_pet_ecology": GOLD,
    "pet_ecology_peripheral_or_local": BLUE,
    "low_network_pet_rule_signal": "#dedbd2",
}
SCENARIO_LABELS = {
    "conservative_same_type": "Conservative\nsame-type",
    "balanced_threshold": "Balanced\nthreshold",
    "permissive_tipping": "Permissive\ntipping",
}
SCENARIO_COLORS = {
    "conservative_same_type": "#9d6d67",
    "balanced_threshold": "#e4bf68",
    "permissive_tipping": "#4e8b68",
}
SCENARIO_SOFT = {
    "conservative_same_type": "#f0dfda",
    "balanced_threshold": "#f8e8bd",
    "permissive_tipping": "#d7e6d8",
}
HOST_LABELS = {
    "restaurant": "Restaurants",
    "shopping_mall": "Malls",
    "hotel": "Hotels",
    "park_or_recreation": "Parks",
    "residential_property": "Residential",
}
HOST_COLORS = {
    "restaurant": "#e4bf68",
    "shopping_mall": "#d88973",
    "hotel": "#3e7899",
    "park_or_recreation": "#4d8a69",
    "residential_property": "#b4aaa0",
}

TOPO_CMAP = LinearSegmentedColormap.from_list("topology_surface", ["#fffaf0", "#f5e9c9", "#e4bf68", "#de987c", "#b96a5d"])
EDGE_CMAP = LinearSegmentedColormap.from_list("edge_density", ["#fffaf0", "#f5e9c9", "#e4bf68", "#d88973"])
FINGER_CMAP = LinearSegmentedColormap.from_list("fig5_fingerprint", ["#fffaf0", "#f5e9c9", "#e4bf68", "#d88973"])
SIGNED_CMAP = LinearSegmentedColormap.from_list("fig5_signed", ["#2f7192", "#fffaf0", "#d88973"])

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 5.2,
        "axes.titlesize": 5.8,
        "axes.labelsize": 4.9,
        "xtick.labelsize": 4.0,
        "ytick.labelsize": 4.0,
        "legend.fontsize": 3.8,
        "axes.linewidth": 0.42,
        "axes.edgecolor": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def panel_path(letter: str, name: str) -> Path:
    return PANELS / f"{OUT_STEM}_panel_{letter}_{name}"


def fixed_fig() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=FIXED_PANEL_SIZE)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.14, top=0.84)
    return fig, ax


def light_frame(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#d8d3ca")
        spine.set_linewidth(0.42)


def city_extent(boundary: gpd.GeoDataFrame, pad: float = 4200) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = boundary.total_bounds
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def parse_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return {}


def load() -> dict[str, object]:
    grid = read_gdf("data_processed/model/grid_emergence_suppression_indices_500m_2025_v51_sparse_repaired.geojson")
    topology = read_csv("data_processed/network/grid_pet_rule_topology_2025_v3.csv")
    topo_cols = ["grid_id", "topological_degree", "weighted_topological_degree", "pet_rule_pagerank", "topology_ecology_class"]
    grid = grid.drop(columns=[c for c in topo_cols if c in grid.columns and c != "grid_id"], errors="ignore")
    grid = grid.merge(topology[topo_cols], on="grid_id", how="left")
    return {
        "grid": grid,
        "boundary": read_gdf("data_processed/geo/shenzhen_boundary_verified.geojson"),
        "edges": read_csv("data_processed/network/pet_rule_multilayer_edges_2025_v3.csv"),
        "threshold_edges": read_csv("data_processed/network/rule_liminal_threshold_edges_2025_v4.csv"),
        "steps": read_csv("data_processed/platform/schelling_full_rule_adoption_steps_2025_v2.csv"),
        "scenarios": read_csv("data_processed/platform/schelling_full_rule_adoption_scenarios_2025_v2.csv"),
        "coefficients": read_csv("data_processed/model/controlled_grid_model_coefficients_v61_sparse_repaired.csv"),
        "metrics": read_csv("data_processed/model/controlled_grid_model_metrics_v61_sparse_repaired.csv"),
    }


def draw_legend(ax: plt.Axes) -> None:
    ax.set_axis_off()
    items = [
        ("patch", RUST, "topology core + pet ecology"),
        ("patch", GOLD, "topology core, low ecology"),
        ("patch", BLUE, "local pet ecology"),
        ("line", SCENARIO_COLORS["conservative_same_type"], "conservative threshold"),
        ("line", SCENARIO_COLORS["balanced_threshold"], "balanced threshold"),
        ("line", SCENARIO_COLORS["permissive_tipping"], "permissive threshold"),
    ]
    xs = [0.02, 0.335, 0.66]
    ys = [0.68, 0.28]
    for idx, (kind, color, label) in enumerate(items):
        x, y = xs[idx % 3], ys[idx // 3]
        if kind == "patch":
            ax.add_patch(Rectangle((x, y - 0.09), 0.034, 0.18, transform=ax.transAxes, facecolor=color, edgecolor="#d0cbc2", linewidth=0.25, alpha=0.78))
        else:
            ax.plot([x, x + 0.042], [y, y], transform=ax.transAxes, color=color, linewidth=1.4)
        ax.text(x + 0.052, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=4.1, color=MUTED)


def draw_topology_surface(ax: plt.Axes, data: dict[str, object], letter: str = "a") -> pd.DataFrame:
    grid = data["grid"].copy()
    boundary = data["boundary"]
    grid["topology_surface"] = numeric(grid["pet_rule_pagerank"]).rank(pct=True) * 0.55 + numeric(grid["weighted_topological_degree"]).rank(pct=True) * 0.45
    grid.plot(ax=ax, color=LAND, edgecolor=GRID, linewidth=0.007, alpha=0.50, zorder=0, rasterized=True)
    surf = grid[grid["topology_surface"] > grid["topology_surface"].quantile(0.25)].copy()
    surf.plot(ax=ax, column="topology_surface", cmap=TOPO_CMAP, linewidth=0, alpha=0.66, norm=Normalize(0.25, 1.0), zorder=6, rasterized=True)
    core = grid[grid["topology_ecology_class"].isin(["network_core_with_pet_ecology", "network_core_without_high_pet_ecology"])].copy()
    if len(core):
        core.centroid.plot(ax=ax, color="#8a6c5f", markersize=1.25, alpha=0.24, linewidth=0, zorder=12)
    boundary.boundary.plot(ax=ax, color="#514b45", linewidth=0.42, alpha=0.78, zorder=20)
    minx, miny, maxx, maxy = city_extent(boundary)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    light_frame(ax)
    panel_label(ax, letter, "network exposure surface")
    ax.text(
        0.012,
        0.025,
        f"{len(grid):,} grids; {len(core):,} topology-core grids; surface = PageRank rank + weighted-degree rank",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=3.6,
        color=MUTED,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.32},
        zorder=30,
    )
    source = grid[["grid_id", "district_name", "topology_surface", "weighted_topological_degree", "pet_rule_pagerank", "topology_ecology_class", "pet_service_count_2025", "positive_rule_count"]].copy()
    source.to_csv(SRC / f"{OUT_STEM}_panel_a_source.csv", index=False, encoding="utf-8-sig")
    return source


def draw_topology_fingerprint(ax: plt.Axes, data: dict[str, object], letter: str = "b") -> pd.DataFrame:
    grid = data["grid"].copy()
    counts = grid.groupby(["district_name", "topology_ecology_class"]).size().reset_index(name="n")
    pivot = counts.pivot_table(index="district_name", columns="topology_ecology_class", values="n", fill_value=0).reindex(DISTRICT_ORDER).fillna(0)
    pivot = pivot.reindex(columns=TOPO_ORDER, fill_value=0)
    share = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
    core_share = share["network_core_with_pet_ecology"] + share["network_core_without_high_pet_ecology"]
    order = core_share.sort_values(ascending=False).index.tolist()
    share = share.reindex(order)
    pivot = pivot.reindex(order)
    core_share = core_share.reindex(order)
    y = np.arange(len(share.index))
    left = np.zeros(len(share))
    for klass in TOPO_ORDER:
        vals = share[klass].values
        ax.barh(y, vals, left=left, height=0.58, color=TOPO_COLORS[klass], edgecolor="white", linewidth=0.24, alpha=0.88)
        for yi, lft, val, district in zip(y, left, vals, share.index):
            n = int(pivot.loc[district, klass])
            if val >= 0.08 and klass != "low_network_pet_rule_signal":
                color = "white" if klass in ["network_core_with_pet_ecology", "network_core_without_high_pet_ecology"] else INK
                ax.text(lft + val / 2, yi, f"{val*100:.0f}", ha="center", va="center", fontsize=3.05, color=color)
            if klass == "low_network_pet_rule_signal" and val >= 0.72:
                ax.text(lft + val - 0.012, yi, f"{n:,}", ha="right", va="center", fontsize=2.85, color="#8a8277")
        left += vals
    ax2 = ax.inset_axes([0.76, 0.08, 0.19, 0.80])
    ax2.scatter(core_share.values, y, s=17 + 110 * core_share.values / max(core_share.max(), 1e-9), color="#9f7b45", edgecolor="white", linewidth=0.32, zorder=4)
    for yi, val in zip(y, core_share.values):
        ax2.hlines(yi, 0, val, color="#d8c7a9", lw=0.62, zorder=2)
    ax2.set_xlim(0, max(0.22, core_share.max() * 1.14))
    ax2.set_ylim(-0.6, len(y) - 0.4)
    ax2.invert_yaxis()
    ax2.set_yticks([])
    ax2.set_xticks([0, round(float(core_share.max()), 2)])
    ax2.set_xticklabels(["0", f"{core_share.max()*100:.0f}%"], fontsize=2.9)
    ax2.set_title("core\nshare", fontsize=3.0, color=MUTED, pad=1.5)
    clean_axis(ax2, grid=False)
    ax.set_yticks(y)
    ax.set_yticklabels([DISTRICT_EN.get(i, i) for i in share.index])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50", "100"])
    ax.set_xlabel("share of district grids (%)")
    ax.set_ylim(-0.65, len(share.index) - 0.35)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#eee8df", lw=0.36)
    clean_axis(ax, grid=True)
    panel_label(ax, letter, "district topology profile")
    handles = [Rectangle((0, 0), 1, 1, facecolor=TOPO_COLORS[k], edgecolor="none") for k in TOPO_ORDER]
    labels = ["Core+ecology", "Core low ecology", "Local pet ecology", "Low signal"]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, -0.34), ncol=2, frameon=False, fontsize=3.1, columnspacing=0.7, handlelength=1.0)
    ax.text(0.99, -0.20, "numbers inside coloured segments are within-district shares; grey numerals are low-signal grid counts", transform=ax.transAxes, ha="right", va="top", fontsize=3.05, color=MUTED)
    out = share.reset_index()
    out.to_csv(SRC / f"{OUT_STEM}_panel_b_source.csv", index=False, encoding="utf-8-sig")
    return out


def draw_edge_distance_structure(ax: plt.Axes, data: dict[str, object], letter: str = "c") -> pd.DataFrame:
    edges = data["edges"].copy()
    edges["edge_family"] = edges["edge_type"].map(
        {
            "host_pet_service_proximity": "host-pet service",
            "host_rule_seed_proximity": "host-rule seed",
        }
    ).fillna(edges["edge_type"])
    bins = np.arange(0, 525, 25)
    rows = []
    for family, sub in edges.groupby("edge_family"):
        hist, bin_edges = np.histogram(sub["distance_m"].clip(0, 500), bins=bins, weights=sub["weight"])
        rows.extend({"edge_family": family, "distance_mid_m": (bin_edges[i] + bin_edges[i + 1]) / 2, "weighted_edges": hist[i], "raw_edges": len(sub)} for i in range(len(hist)))
    out = pd.DataFrame(rows)
    y_offsets = {"host-pet service": 0.72, "host-rule seed": 0.28}
    for family, color in [("host-pet service", BLUE_DARK), ("host-rule seed", GOLD)]:
        sub = out[out["edge_family"].eq(family)].copy()
        if sub.empty:
            continue
        denom = max(sub["weighted_edges"].max(), 1e-9)
        x = sub["distance_mid_m"].to_numpy()
        density = (sub["weighted_edges"] / denom).to_numpy()
        base = y_offsets[family]
        y = base + density * 0.28
        ax.fill_between(x, base, y, color=color, alpha=0.16, linewidth=0)
        ax.plot(x, y, color="white", lw=3.0, alpha=0.82, solid_capstyle="round", zorder=3)
        ax.plot(x, y, color=color, lw=1.35, label=family, solid_capstyle="round", zorder=4)
        raw = edges.loc[edges["edge_family"].eq(family), "distance_m"].clip(0, 500)
        qs = raw.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        for q, val in qs.items():
            lw = 0.82 if q == 0.5 else 0.45
            ax.vlines(val, base - 0.045, base + 0.34, color=color, lw=lw, alpha=0.76 if q == 0.5 else 0.38)
        ax.text(
            492,
            base + 0.16,
            f"{family}\nn={len(raw):,}\nmedian {qs[0.5]:.0f} m",
            ha="right",
            va="center",
            fontsize=3.35,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.66, "pad": 0.25},
        )
    ax.set_xlim(0, 500)
    ax.set_ylim(0.10, 1.08)
    ax.set_xlabel("proximity edge distance (m)")
    ax.set_ylabel("weighted edge-density ridge")
    ax.set_yticks([y_offsets["host-rule seed"], y_offsets["host-pet service"]])
    ax.set_yticklabels(["Rule-seed\nedges", "Pet-service\nedges"])
    clean_axis(ax)
    panel_label(ax, letter, "edge-distance ridges with quantile ticks")
    ax.text(0.01, -0.27, f"{len(edges):,} multilayer proximity edges; vertical ticks mark p10/p25/median/p75/p90 within edge family", transform=ax.transAxes, ha="left", va="top", fontsize=3.35, color=MUTED)
    out.to_csv(SRC / f"{OUT_STEM}_panel_c_source.csv", index=False, encoding="utf-8-sig")
    return out


def draw_threshold_evidence_rail(ax: plt.Axes, data: dict[str, object]) -> None:
    summary = data["scenarios"].copy().set_index("scenario")
    order = ["conservative_same_type", "balanced_threshold", "permissive_tipping"]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.add_patch(Rectangle((0.0, 0.0), 1, 1, facecolor="white", edgecolor="#dedbd2", linewidth=0.38, alpha=0.92))
    ax.text(0.06, 0.94, "scenario evidence", ha="left", va="top", fontsize=3.45, color=TEXT)
    ax.text(0.06, 0.845, "rule-seed base", ha="left", va="center", fontsize=2.95, color=MUTED)
    if len(summary):
        seed_row = summary.iloc[0]
        pos = float(seed_row["positive_rule_seeds"])
        res = float(seed_row["restrictive_rule_seeds"])
        other = max(float(seed_row["rules_used"]) - pos - res, 0.0)
        total_rules = max(float(seed_row["rules_used"]), 1.0)
        x0, w, y0, h = 0.34, 0.48, 0.790, 0.075
        left = x0
        for val, color, txt, txt_color in [
            (pos, GREEN, "53+", "white"),
            (res, "#7f665f", "28-", "white"),
            (other, "#ddd7cc", "", INK),
        ]:
            ww = w * val / total_rules
            ax.add_patch(Rectangle((left, y0), ww, h, facecolor=color, edgecolor="white", linewidth=0.25, alpha=0.92))
            if txt:
                ax.text(left + ww / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=2.65, color=txt_color)
            left += ww
        ax.text(x0 + w + 0.035, y0 + h / 2, "82", ha="left", va="center", fontsize=2.75, color=MUTED)
    ax.plot([0.06, 0.94], [0.705, 0.705], color="#e8e2d8", lw=0.40)
    ax.text(0.06, 0.655, "final state", ha="left", va="center", fontsize=2.95, color=MUTED)
    for i, scenario in enumerate(order):
        if scenario not in summary.index:
            continue
        row = summary.loc[scenario]
        adopted = float(row["simulated_adopters"])
        remain = float(row["silent_agents"] - row["simulated_adopters"])
        total = float(row["silent_agents"])
        yrow = 0.560 - i * 0.135
        label = {"conservative_same_type": "cons", "balanced_threshold": "bal", "permissive_tipping": "perm"}[scenario]
        ax.text(0.06, yrow + 0.026, label, ha="left", va="center", fontsize=2.75, color=MUTED)
        x0, w, h = 0.205, 0.56, 0.058
        ax.add_patch(Rectangle((x0, yrow), w * adopted / total, h, facecolor=SCENARIO_COLORS[scenario], edgecolor="white", linewidth=0.20, alpha=0.82))
        ax.add_patch(Rectangle((x0 + w * adopted / total, yrow), w * remain / total, h, facecolor="#e5e0d8", edgecolor="white", linewidth=0.20, alpha=0.86))
        ax.text(x0 + w + 0.035, yrow + h / 2, f"{int(adopted):,}", ha="left", va="center", fontsize=2.55, color=MUTED)
    ax.text(0.205, 0.115, "0", ha="center", va="center", fontsize=2.45, color=MUTED)
    ax.text(0.765, 0.115, "154k", ha="center", va="center", fontsize=2.45, color=MUTED)
    ax.text(0.205, 0.155, "|", ha="center", va="center", fontsize=3.2, color="#bcb6ad")
    ax.text(0.765, 0.155, "|", ha="center", va="center", fontsize=3.2, color="#bcb6ad")


def draw_threshold_trajectories(ax: plt.Axes, data: dict[str, object], letter: str = "d") -> pd.DataFrame:
    scenarios = data["scenarios"].copy()
    rows = []
    for _, row in scenarios.iterrows():
        scenario = row["scenario"]
        cumulative = 0
        steps = parse_dict(row["adopted_by_step"])
        for step in sorted(int(k) for k in steps.keys()):
            cumulative += int(steps.get(step, steps.get(str(step), 0)))
            rows.append({"scenario": scenario, "step": step, "new_adopters": int(steps.get(step, steps.get(str(step), 0))), "cumulative_adopters": cumulative, "adoption_share": cumulative / float(row["silent_agents"])})
    out = pd.DataFrame(rows)
    max_new = max(out["new_adopters"].max(), 1)
    # Mechanism background: seed formation, cascade window and saturation tail.
    phases = [
        (-0.15, 1.05, "#f7f4ee", "seed"),
        (1.05, 2.35, "#efe6d2", "cascade window"),
        (2.35, max(out["step"]) + 0.35, "#f2f5ef", "saturation tail"),
    ]
    for x0, x1, color, label in phases:
        ax.axvspan(x0, x1, color=color, alpha=0.75, zorder=0)
        ax.text((x0 + x1) / 2, 0.835, label, ha="center", va="top", fontsize=3.35, color="#8b8176", zorder=1)
    ax.hlines([0.25, 0.50, 0.75], -0.15, max(out["step"]) + 0.35, colors="white", lw=0.65, zorder=1)
    for scenario, sub in out.groupby("scenario"):
        sub = sub.sort_values("step")
        x = sub["step"].to_numpy()
        y = sub["adoption_share"].to_numpy()
        new_scaled = 0.10 * sub["new_adopters"].to_numpy() / max_new
        ax.bar(x, new_scaled, width=0.26, color=SCENARIO_SOFT.get(scenario, "#e9e5dc"), alpha=0.72, edgecolor="white", linewidth=0.20, zorder=2)
        ax.plot(x, y, color="white", lw=2.35, alpha=0.88, solid_capstyle="round", zorder=3)
        ax.plot(x, y, color=SCENARIO_SOFT.get(scenario, "#e9e5dc"), lw=1.55, alpha=0.86, solid_capstyle="round", zorder=4)
        ax.plot(x, y, color=SCENARIO_COLORS.get(scenario, MUTED), lw=0.82, marker="o", ms=2.2, markerfacecolor="white", markeredgecolor=SCENARIO_COLORS.get(scenario, MUTED), markeredgewidth=0.58, label=SCENARIO_LABELS.get(scenario, scenario).replace("\n", " "), solid_capstyle="round", zorder=5)
        ax.scatter(x, y, facecolor="white", edgecolor=SCENARIO_COLORS.get(scenario, MUTED), s=9, linewidth=0.55, zorder=6)
        short_label = {
            "conservative_same_type": "Conservative",
            "balanced_threshold": "Balanced",
            "permissive_tipping": "Permissive",
        }.get(scenario, scenario)
        y_nudge = {
            "conservative_same_type": 0.018,
            "balanced_threshold": 0.010,
            "permissive_tipping": 0.004,
        }.get(scenario, 0.0)
        ax.text(
            x[-1] - 0.08,
            y[-1] + y_nudge,
            f"{short_label}  {y[-1]*100:.1f}%",
            ha="right",
            va="center",
            fontsize=3.35,
            color=SCENARIO_COLORS.get(scenario, MUTED),
            zorder=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.50, "pad": 0.20},
        )
        peak = sub.loc[sub["new_adopters"].idxmax()]
        peak_y = 0.10 * peak["new_adopters"] / max_new
        ax.scatter([peak["step"]], [peak_y], s=20, marker="v", color=SCENARIO_COLORS.get(scenario, MUTED), edgecolor="white", linewidth=0.32, zorder=6)
        if scenario in ["balanced_threshold", "permissive_tipping"]:
            peak_label_y = 0.136 if scenario == "permissive_tipping" else 0.092
            ax.text(
                peak["step"] + 0.11,
                peak_label_y,
                f"+{int(peak['new_adopters']):,}",
                ha="left",
                va="bottom",
                fontsize=3.15,
                color=SCENARIO_COLORS.get(scenario, MUTED),
                zorder=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.58, "pad": 0.18},
            )
    ax.set_xlim(-0.15, max(out["step"]) + 0.78)
    ax.set_ylim(0, min(0.88, max(out["adoption_share"]) * 1.14))
    ax.set_xlabel("threshold iteration")
    ax.set_ylabel("simulated adoption share")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.75), frameon=False, fontsize=3.25, handlelength=1.0)
    clean_axis(ax, grid=False)
    panel_label(ax, letter, "threshold trajectories with cascade backdrop")
    ax.text(0.99, -0.25, "pale bars = normalised new adopters per step; shaded bands are interpretive phases, not observed diffusion", transform=ax.transAxes, ha="right", va="top", fontsize=3.25, color=MUTED)
    out.to_csv(SRC / f"{OUT_STEM}_panel_d_source.csv", index=False, encoding="utf-8-sig")
    return out


def draw_scenario_type_composition(ax: plt.Axes, data: dict[str, object], letter: str = "e") -> pd.DataFrame:
    scenarios = data["scenarios"].copy()
    rows = []
    for _, row in scenarios.iterrows():
        for host, n in parse_dict(row["adopters_by_type"]).items():
            rows.append({"scenario": row["scenario"], "host_type": host, "n": int(n), "adoption_share": float(row["adoption_share"])})
    out = pd.DataFrame(rows)
    order = list(SCENARIO_LABELS.keys())
    host_order = ["restaurant", "shopping_mall", "hotel", "park_or_recreation", "residential_property"]
    pivot = out.pivot_table(index="scenario", columns="host_type", values="n", aggfunc="sum", fill_value=0).reindex(order).fillna(0)
    pivot = pivot.reindex(columns=host_order, fill_value=0)
    share = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
    y = np.arange(len(share.index))
    left = np.zeros(len(share))
    for host in host_order:
        vals = share[host].values
        ax.barh(y, vals, left=left, height=0.58, color=HOST_COLORS[host], edgecolor="white", linewidth=0.24, alpha=0.90)
        for yi, lft, val, scenario in zip(y, left, vals, share.index):
            n = int(pivot.loc[scenario, host])
            if n <= 0:
                continue
            if val >= 0.045:
                color = "white" if host in ["restaurant", "shopping_mall", "hotel"] else INK
                ax.text(lft + val / 2, yi, f"{n:,}", ha="center", va="center", fontsize=3.0, color=color)
            elif n > 0:
                ax.text(lft + val + 0.008, yi, f"{n:,}", ha="left", va="center", fontsize=2.75, color=HOST_COLORS[host])
        left += vals
    scenario_share = scenarios.set_index("scenario").reindex(order)["adoption_share"]
    ax2 = ax.inset_axes([0.82, 0.16, 0.16, 0.72])
    ax2.barh(np.arange(len(order)), scenario_share.values, height=0.42, color=[SCENARIO_COLORS[s] for s in order], alpha=0.80)
    ax2.set_xlim(0, 0.82)
    ax2.set_yticks([])
    ax2.set_xticks([0, 0.8])
    ax2.set_xticklabels(["0", "80%"], fontsize=3.1)
    ax2.invert_yaxis()
    clean_axis(ax2, grid=False)
    ax.set_yticks(y)
    ax.set_yticklabels([SCENARIO_LABELS[s] for s in share.index])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "50", "100"])
    ax.set_xlabel("composition of simulated adopters (%)")
    ax.set_ylim(-0.58, len(share.index) - 0.32)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#ece8df", lw=0.35)
    clean_axis(ax, grid=True)
    panel_label(ax, letter, "venue-type tipping composition")
    handles = [Rectangle((0, 0), 1, 1, facecolor=HOST_COLORS[h], edgecolor="none") for h in host_order]
    labels = [HOST_LABELS[h] for h in host_order]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, -0.36), ncol=3, frameon=False, fontsize=3.0, columnspacing=0.65, handlelength=1.0)
    ax.text(0.99, -0.20, "segment labels are simulated adopter counts; right strip = total adoption share", transform=ax.transAxes, ha="right", va="top", fontsize=3.15, color=MUTED)
    share.reset_index().to_csv(SRC / f"{OUT_STEM}_panel_e_source.csv", index=False, encoding="utf-8-sig")
    out.to_csv(SRC / f"{OUT_STEM}_panel_e_long_source.csv", index=False, encoding="utf-8-sig")
    return share.reset_index()


def draw_model_diagnostic(ax: plt.Axes, data: dict[str, object], letter: str = "f") -> pd.DataFrame:
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
    sub = coef[coef["spec"].eq("core_plus_controls") & coef["feature"].isin(keep_features)].copy()
    sub = sub[sub["target"].isin(["is_suppressed_frontier", "has_positive_rule"])].copy()
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
    sub["feature_label"] = sub["feature"].map(label_map)
    sub["target_label"] = sub["target"].map(target_map)
    target_order = ["Visible positive rule", "Suppressed frontier"]
    feature_order = [label_map[f] for f in keep_features]
    sub = sub[sub["target_label"].isin(target_order)].copy()
    sub["target_label"] = pd.Categorical(sub["target_label"], categories=target_order, ordered=True)
    sub["feature_label"] = pd.Categorical(sub["feature_label"], categories=feature_order, ordered=True)
    sub = sub.sort_values(["target_label", "feature_label"])
    offsets = {"Visible positive rule": -0.15, "Suppressed frontier": 0.15}
    colors = {"Visible positive rule": BLUE_DARK, "Suppressed frontier": RUST}
    ybase = np.arange(len(feature_order))[::-1]
    ymap = {feat: y for feat, y in zip(feature_order, ybase)}
    ax.axvline(0, color="#6f6961", lw=0.55, alpha=0.75, zorder=1)
    for target in target_order:
        ss = sub[sub["target_label"].astype(str).eq(target)]
        yy = np.array([ymap[str(f)] + offsets[target] for f in ss["feature_label"]])
        xx = ss["coefficient"].to_numpy()
        ax.hlines(yy, 0, xx, color=colors[target], lw=0.85, alpha=0.80)
        ax.scatter(xx, yy, s=20 + 10 * np.clip(np.abs(xx), 0, 4), facecolor="white", edgecolor=colors[target], linewidth=0.62, zorder=4, label=target)
        for xval, yval in zip(xx, yy):
            if abs(xval) >= 3.0 or str(target) == "Suppressed frontier":
                if xval < -4.8:
                    ha = "left"
                    xpos = xval + 0.18
                else:
                    ha = "left" if xval >= 0 else "right"
                    xpos = xval + (0.10 if xval >= 0 else -0.10)
                ax.text(xpos, yval, f"{xval:+.2f}", ha=ha, va="center", fontsize=3.0, color=colors[target])
    ax.set_yticks(ybase)
    ax.set_yticklabels(feature_order)
    ax.set_xlim(-6.55, 5.05)
    ax.set_xticks([-6, -3, 0, 3])
    ax.set_xlabel("standardised model coefficient")
    ax.grid(axis="x", color="#ece8df", lw=0.36)
    clean_axis(ax, grid=False)
    ax.legend(loc="lower right", frameon=False, fontsize=3.2, handletextpad=0.30)
    panel_label(ax, letter, "controlled diagnostic: signed coefficient forest")
    m = metrics[metrics["spec"].eq("core_plus_controls") & metrics["target"].isin(["is_suppressed_frontier", "has_positive_rule"])].copy()
    m = m.drop_duplicates(subset=["target", "model", "n", "positive", "roc_auc", "average_precision"])
    if len(m):
        metric_lines = []
        for r in m.sort_values("target").itertuples():
            metric_lines.append(f"{target_map.get(r.target, r.target)}: AUC {r.roc_auc:.3f}")
        ax.text(
            1.0,
            -0.20,
            " | ".join(metric_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=3.05,
            color=MUTED,
        )
    ax.text(
        0.0,
        -0.31,
        "signed coefficients from one controlled specification; diagnostic evidence, not causal estimation",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=3.25,
        color=MUTED,
    )
    out = sub.merge(metrics[["target", "roc_auc", "average_precision", "balanced_accuracy_at_0_5", "spec"]], on=["target", "spec"], how="left")
    out.to_csv(SRC / f"{OUT_STEM}_panel_f_source.csv", index=False, encoding="utf-8-sig")
    return out


def render_single_panels(data: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=FIXED_PANEL_SIZE)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.09, top=0.86)
    draw_topology_surface(ax, data, "a")
    save_figure(fig, panel_path("a", "network_exposure_surface"), dpi=600, tight=False)

    fig, ax = fixed_fig()
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.31, top=0.84)
    draw_topology_fingerprint(ax, data, "b")
    save_figure(fig, panel_path("b", "topology_ecology_classes"), dpi=600, tight=False)

    fig, ax = fixed_fig()
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.22, top=0.84)
    draw_edge_distance_structure(ax, data, "c")
    save_figure(fig, panel_path("c", "edge_distance_structure"), dpi=600, tight=False)

    fig, ax = fixed_fig()
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.22, top=0.84)
    draw_threshold_trajectories(ax, data, "d")
    save_figure(fig, panel_path("d", "threshold_trajectories"), dpi=600, tight=False)

    fig, ax = fixed_fig()
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.32, top=0.84)
    draw_scenario_type_composition(ax, data, "e")
    save_figure(fig, panel_path("e", "scenario_type_composition"), dpi=600, tight=False)

    fig, ax = fixed_fig()
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.30, top=0.78)
    draw_model_diagnostic(ax, data, "f")
    save_figure(fig, panel_path("f", "controlled_model_diagnostic"), dpi=600, tight=False)


def render_composite(data: dict[str, object]) -> None:
    fig = plt.figure(figsize=(7.25, 9.95))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.065, top=0.910)
    gs = GridSpec(
        6,
        6,
        figure=fig,
        height_ratios=[0.16, 1.20, 0.98, 1.22, 0.98, 1.02],
        width_ratios=[1, 1, 1, 1, 1, 1],
        wspace=0.16,
        hspace=0.88,
    )
    draw_legend(fig.add_subplot(gs[0, :]))
    draw_topology_surface(fig.add_subplot(gs[1:3, :3]), data, "a")
    draw_topology_fingerprint(fig.add_subplot(gs[1, 3:]), data, "b")
    draw_edge_distance_structure(fig.add_subplot(gs[2, 3:]), data, "c")
    draw_threshold_trajectories(fig.add_subplot(gs[3, :]), data, "d")
    draw_scenario_type_composition(fig.add_subplot(gs[4, :]), data, "e")
    draw_model_diagnostic(fig.add_subplot(gs[5, :]), data, "f")
    fig.text(0.015, 0.985, "Fig. 5 | Spatial topology and threshold dynamics", ha="left", va="top", fontsize=8.0, fontweight="bold", color=INK)
    fig.text(
        0.015,
        0.960,
        "Topology and threshold diagnostics show why pet-friendly capability is a relational condition, not simply a count of nearby services.",
        ha="left",
        va="top",
        fontsize=5.0,
        color=TEXT,
    )
    save_figure(fig, COMP / f"{OUT_STEM}_composite", dpi=520, tight=False)


def contact_sheet() -> None:
    files = [
        panel_path("a", "network_exposure_surface").with_suffix(".png"),
        panel_path("b", "topology_ecology_classes").with_suffix(".png"),
        panel_path("c", "edge_distance_structure").with_suffix(".png"),
        panel_path("d", "threshold_trajectories").with_suffix(".png"),
        panel_path("e", "scenario_type_composition").with_suffix(".png"),
        panel_path("f", "controlled_model_diagnostic").with_suffix(".png"),
        COMP / f"{OUT_STEM}_composite.png",
    ]
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
    sheet.save(PREVIEWS / f"{OUT_STEM}_contact_sheet.png", dpi=(180, 180))


def main() -> None:
    data = load()
    render_single_panels(data)
    render_composite(data)
    contact_sheet()
    rows = []
    for f in sorted(PANELS.glob(f"{OUT_STEM}_panel_*.*")):
        rows.append({"asset": f.relative_to(ROOT).as_posix(), "role": "panel"})
    for f in sorted(COMP.glob(f"{OUT_STEM}_composite.*")):
        rows.append({"asset": f.relative_to(ROOT).as_posix(), "role": "composite"})
    for f in sorted(SRC.glob(f"{OUT_STEM}_panel_*source.csv")):
        rows.append({"asset": f.relative_to(ROOT).as_posix(), "role": "source_data"})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(SRC / f"{OUT_STEM}_export_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
