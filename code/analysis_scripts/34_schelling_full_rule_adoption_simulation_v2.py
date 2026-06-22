#!/usr/bin/env python3
"""Schelling-style threshold simulation over full 2025 silent venue agents."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve()
for _p in PROJECT_ROOT.parents:
    if (_p / "data").is_dir():
        PROJECT_ROOT = _p
        break
(PROJECT_ROOT / "_rebuild_outputs" / "reports").mkdir(parents=True, exist_ok=True)
RULES = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_semantic_records_v1.csv"
LEDGER = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_source_ledger_v1.csv"
AGENTS = PROJECT_ROOT / "data" / "derived_data" / "platform" / "full_silent_rule_adoption_propensity_2025_v2.csv"
GRID = PROJECT_ROOT / "data" / "derived_data" / "geo" / "shenzhen_grid_500m.gpkg"
OUT_FINAL = PROJECT_ROOT / "data" / "derived_data" / "platform" / "schelling_full_rule_adoption_final_agents_2025_v2.csv"
OUT_STEPS = PROJECT_ROOT / "data" / "derived_data" / "platform" / "schelling_full_rule_adoption_steps_2025_v2.csv"
OUT_SCENARIOS = PROJECT_ROOT / "data" / "derived_data" / "platform" / "schelling_full_rule_adoption_scenarios_2025_v2.csv"
OUT_GRID = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_schelling_rule_adoption_500m_2025_v2.csv"
OUT_GRID_GEO = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_schelling_rule_adoption_500m_2025_v2.geojson"
REPORT = PROJECT_ROOT / "_rebuild_outputs" / "reports" / "49_schelling_full_rule_adoption_simulation_v2_report.md"

TYPE_THRESHOLD = {
    "shopping_mall": 0.22,
    "restaurant": 0.35,
    "hotel": 0.42,
    "park_or_recreation": 0.28,
    "residential_property": 0.55,
}

SCENARIOS = {
    "conservative_same_type": {
        "threshold_multiplier": 1.20,
        "adopter_threshold_ratio": 0.86,
        "high_prob_auto_seed": 0.72,
        "adopter_seed_weight": 0.30,
        "max_steps": 4,
    },
    "balanced_threshold": {
        "threshold_multiplier": 1.00,
        "adopter_threshold_ratio": 0.74,
        "high_prob_auto_seed": 0.65,
        "adopter_seed_weight": 0.45,
        "max_steps": 5,
    },
    "permissive_tipping": {
        "threshold_multiplier": 0.82,
        "adopter_threshold_ratio": 0.62,
        "high_prob_auto_seed": 0.58,
        "adopter_seed_weight": 0.55,
        "max_steps": 6,
    },
}

TYPE_COMPAT = {
    ("shopping_mall", "shopping_mall"): 1.0,
    ("restaurant", "restaurant"): 1.0,
    ("hotel", "hotel"): 1.0,
    ("park_or_recreation", "park_or_recreation"): 1.0,
    ("residential_property", "residential_property"): 1.0,
    ("restaurant", "shopping_mall"): 0.35,
    ("shopping_mall", "restaurant"): 0.25,
    ("hotel", "restaurant"): 0.20,
    ("hotel", "shopping_mall"): 0.20,
    ("residential_property", "park_or_recreation"): 0.30,
    ("park_or_recreation", "residential_property"): 0.20,
}


def transform_xy(lon: pd.Series, lat: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    x, y = transformer.transform(lon.to_numpy(dtype=float), lat.to_numpy(dtype=float))
    return np.asarray(x), np.asarray(y)


def rule_primary_type(venue_type: str) -> str:
    venue_type = str(venue_type)
    if "shopping_mall" in venue_type:
        return "shopping_mall"
    if "restaurant" in venue_type or "pet_cafe" in venue_type:
        return "restaurant"
    if "hotel" in venue_type:
        return "hotel"
    if "park" in venue_type or "pet_governance" in venue_type:
        return "park_or_recreation"
    if "residential" in venue_type:
        return "residential_property"
    return "unknown"


def load_rule_seed_summary() -> dict:
    rules = pd.read_csv(RULES)
    if LEDGER.exists():
        ledger = pd.read_csv(LEDGER)[["source_id", "publication_use_status", "source_weight_for_model"]]
        rules = rules.merge(ledger, on="source_id", how="left")
        rules = rules[
            rules["publication_use_status"].isin(
                ["main_model", "main_model_with_caution", "supplement_or_sensitivity"]
            )
        ].copy()
    rules = rules[np.isfinite(rules["lon"]) & np.isfinite(rules["lat"])].copy()
    rules["rule_primary_type"] = rules["venue_type"].map(rule_primary_type)
    return {
        "rules_used": int(len(rules)),
        "positive_rule_seeds": int((rules["access_semantic_score"] > 0).sum()),
        "restrictive_rule_seeds": int((rules["access_semantic_score"] < 0).sum()),
        "positive_by_type": dict(Counter(rules.loc[rules["access_semantic_score"] > 0, "rule_primary_type"])),
        "restrictive_by_type": dict(Counter(rules.loc[rules["access_semantic_score"] < 0, "rule_primary_type"])),
    }


def distance_weight(distance: np.ndarray) -> np.ndarray:
    return np.where(distance <= 100, 1.0, np.where(distance <= 250, 0.55, np.where(distance <= 500, 0.25, 0.0)))


def add_adopter_influence(
    adopter_idx: np.ndarray,
    agents: pd.DataFrame,
    tree: cKDTree,
    x: np.ndarray,
    y: np.ndarray,
    signal_sum: np.ndarray,
    norm_sum: np.ndarray,
    adopter_seed_weight: float,
) -> None:
    """Accumulate influence only within 500m of newly adopted agents."""
    for aidx in adopter_idx:
        neighbors = tree.query_ball_point([x[aidx], y[aidx]], r=500.0)
        if not neighbors:
            continue
        neighbors_arr = np.asarray(neighbors, dtype=int)
        dx = x[neighbors_arr] - x[aidx]
        dy = y[neighbors_arr] - y[aidx]
        dist = np.sqrt(dx * dx + dy * dy)
        w = distance_weight(dist)
        atype = agents.at[aidx, "primary_venue_type"]
        ntypes = agents.loc[neighbors_arr, "primary_venue_type"].to_numpy()
        comp = np.array([TYPE_COMPAT.get((ntype, atype), 0.10) for ntype in ntypes], dtype=float)
        signal_sum[neighbors_arr] += w * comp * adopter_seed_weight
        norm_sum[neighbors_arr] += w * comp


def run_scenario(
    base_agents: pd.DataFrame,
    scenario_name: str,
    scenario: dict,
    x: np.ndarray,
    y: np.ndarray,
    tree: cKDTree,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    agents = base_agents.copy()
    agents["threshold"] = agents["primary_venue_type"].map(TYPE_THRESHOLD).fillna(0.45) * float(scenario["threshold_multiplier"])
    agents["schelling_state"] = "silent"
    agents["adopted_step"] = -1
    agents["schelling_signal_at_adoption"] = 0.0
    signal_sum = np.zeros(len(agents), dtype=float)
    norm_sum = np.zeros(len(agents), dtype=float)
    adopted_mask = np.zeros(len(agents), dtype=bool)

    auto_seed = agents["adoption_propensity"] >= float(scenario["high_prob_auto_seed"])
    adopted_mask[auto_seed.to_numpy()] = True
    agents.loc[auto_seed, "schelling_state"] = "simulated_adopter"
    agents.loc[auto_seed, "adopted_step"] = 0
    agents.loc[auto_seed, "schelling_signal_at_adoption"] = agents.loc[auto_seed, "adoption_propensity"]
    add_adopter_influence(
        np.where(auto_seed.to_numpy())[0],
        agents,
        tree,
        x,
        y,
        signal_sum,
        norm_sum,
        float(scenario["adopter_seed_weight"]),
    )

    records: list[dict] = []
    if auto_seed.any():
        for _, row in agents.loc[auto_seed].iterrows():
            records.append(
                {
                    "scenario": scenario_name,
                    "step": 0,
                    "id": row["id"],
                    "name": row["name"],
                    "primary_venue_type": row["primary_venue_type"],
                    "signal": row["adoption_propensity"],
                    "threshold": row["threshold"],
                    "grid_id": row["grid_id"],
                    "district_name": row["district_name"],
                    "note": "high_propensity_initial_seed",
                }
            )

    for step in range(1, int(scenario["max_steps"]) + 1):
        candidates = np.where(~adopted_mask)[0]
        local_signal = np.divide(
            signal_sum[candidates],
            norm_sum[candidates],
            out=np.zeros(len(candidates), dtype=float),
            where=norm_sum[candidates] > 0,
        )
        combined_signal = (
            agents.loc[candidates, "adoption_propensity"].to_numpy() * 0.70
            + local_signal * 0.30
        )
        threshold = agents.loc[candidates, "threshold"].to_numpy() * float(scenario["adopter_threshold_ratio"])
        newly = candidates[combined_signal >= threshold]
        if len(newly) == 0:
            records.append(
                {
                    "scenario": scenario_name,
                    "step": step,
                    "id": "",
                    "name": "",
                    "primary_venue_type": "",
                    "signal": 0,
                    "threshold": 0,
                    "grid_id": "",
                    "district_name": "",
                    "note": "no_new_adoptions",
                }
            )
            break
        adopted_mask[newly] = True
        agents.loc[newly, "schelling_state"] = "simulated_adopter"
        agents.loc[newly, "adopted_step"] = step
        signal_series = pd.Series(combined_signal, index=candidates)
        agents.loc[newly, "schelling_signal_at_adoption"] = signal_series.loc[newly].to_numpy()
        for idx in newly:
            row = agents.loc[idx]
            records.append(
                {
                    "scenario": scenario_name,
                    "step": step,
                    "id": row["id"],
                    "name": row["name"],
                    "primary_venue_type": row["primary_venue_type"],
                    "signal": row["schelling_signal_at_adoption"],
                    "threshold": row["threshold"],
                    "grid_id": row["grid_id"],
                    "district_name": row["district_name"],
                    "note": "threshold_tipping",
                }
            )
        add_adopter_influence(
            newly,
            agents,
            tree,
            x,
            y,
            signal_sum,
            norm_sum,
            float(scenario["adopter_seed_weight"]),
        )

    counts = {
        "scenario": scenario_name,
        "silent_agents": int(len(agents)),
        "simulated_adopters": int(agents["schelling_state"].eq("simulated_adopter").sum()),
        "adoption_share": float(agents["schelling_state"].eq("simulated_adopter").mean()),
        "adopted_by_step": agents.loc[agents["adopted_step"] >= 0, "adopted_step"].value_counts().sort_index().to_dict(),
        "adopters_by_type": dict(Counter(agents.loc[agents["schelling_state"].eq("simulated_adopter"), "primary_venue_type"])),
        "adopters_by_district_top10": dict(Counter(agents.loc[agents["schelling_state"].eq("simulated_adopter"), "district_name"]).most_common(10)),
    }
    agents["scenario"] = scenario_name
    return pd.DataFrame(records), agents, counts


def main() -> None:
    OUT_FINAL.parent.mkdir(parents=True, exist_ok=True)
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(AGENTS)
    base = base.reset_index(drop=True)
    x, y = transform_xy(base["lon_wgs84"], base["lat_wgs84"])
    tree = cKDTree(np.column_stack([x, y]))
    rule_summary = load_rule_seed_summary()

    all_steps: list[pd.DataFrame] = []
    scenario_rows: list[dict] = []
    balanced_agents = None
    for name, scenario in SCENARIOS.items():
        steps, agents, counts = run_scenario(base, name, scenario, x, y, tree)
        counts.update(rule_summary)
        all_steps.append(steps)
        scenario_rows.append(counts)
        if name == "balanced_threshold":
            balanced_agents = agents

    if balanced_agents is None:
        raise RuntimeError("Balanced scenario did not run.")

    pd.concat(all_steps, ignore_index=True).to_csv(OUT_STEPS, index=False, encoding="utf-8-sig")
    pd.DataFrame(scenario_rows).to_csv(OUT_SCENARIOS, index=False, encoding="utf-8-sig")
    balanced_agents.to_csv(OUT_FINAL, index=False, encoding="utf-8-sig")

    grid = (
        balanced_agents.groupby("grid_id", dropna=False)
        .agg(
            silent_agents=("id", "count"),
            simulated_adopters=("schelling_state", lambda s: int((s == "simulated_adopter").sum())),
            mean_adoption_propensity=("adoption_propensity", "mean"),
            mean_signal_at_adoption=("schelling_signal_at_adoption", "mean"),
        )
        .reset_index()
    )
    grid["simulated_adoption_share"] = grid["simulated_adopters"] / grid["silent_agents"]
    grid.to_csv(OUT_GRID, index=False, encoding="utf-8-sig")
    grid_geo = gpd.read_file(GRID).merge(grid, on="grid_id", how="left")
    for col in [
        "silent_agents",
        "simulated_adopters",
        "mean_adoption_propensity",
        "mean_signal_at_adoption",
        "simulated_adoption_share",
    ]:
        grid_geo[col] = grid_geo[col].fillna(0)
    grid_geo.to_crs("EPSG:4326").to_file(OUT_GRID_GEO, driver="GeoJSON")

    top_grids = grid.sort_values(["simulated_adopters", "simulated_adoption_share"], ascending=False).head(20).to_dict("records")
    REPORT.write_text(
        f"""# 49 Schelling Full Rule Adoption Simulation V2 Report

Locked title: Pet-Service Ecologies Outpace Urban Readiness in Shenzhen: Rule-Liminal Venues and the Uneven Emergence of Companion-Animal Urban Capability

## Execution

- Input point propensities: `{AGENTS.relative_to(PROJECT_ROOT)}`
- Step output: `{OUT_STEPS.relative_to(PROJECT_ROOT)}`
- Balanced final agents: `{OUT_FINAL.relative_to(PROJECT_ROOT)}`
- Scenario table: `{OUT_SCENARIOS.relative_to(PROJECT_ROOT)}`
- Grid output: `{OUT_GRID.relative_to(PROJECT_ROOT)}`
- Grid GeoJSON: `{OUT_GRID_GEO.relative_to(PROJECT_ROOT)}`

## Scenario Counts

`{scenario_rows}`

## Top Balanced-Scenario Grids

`{top_grids}`

## Model Meaning

The v2 Schelling model treats silent POI venues as agents. Initial adopters are high-propensity venues from source-weighted rule exposure. Further adoption occurs only when a venue's own rule-exposure propensity and nearby simulated adopters jointly exceed a type-specific threshold.

This keeps the model elegant: explicit rule seeds generate local signals; silent venues have thresholds; a few high-propensity local switches can produce clustered rule tolerance. It is still a hypothesis-generating model and needs targeted verification before being described as observed adoption.

## Publication Boundary

Use the balanced scenario as the main diagnostic surface. Conservative and permissive scenarios are robustness bounds. Claims should be phrased as "threshold-based latent adoption clusters" and "candidate next-verification zones", not confirmed pet-friendly venues.
""",
        encoding="utf-8",
    )
    print(scenario_rows)
    print(OUT_SCENARIOS)


if __name__ == "__main__":
    main()
