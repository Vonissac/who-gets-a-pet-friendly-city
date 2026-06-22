#!/usr/bin/env python3
"""Run final systematic v7 model synthesis.

V7 combines the established diagnostic models with quasi-causal/event-study
screens. It produces final result tables and a synthesis report. The causal
language remains conservative: event-study outputs are mechanism-consistent
diagnostics, not definitive causal estimates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve()
for _p in PROJECT_ROOT.parents:
    if (_p / "data").is_dir():
        PROJECT_ROOT = _p
        break
(PROJECT_ROOT / "_rebuild_outputs" / "reports").mkdir(parents=True, exist_ok=True)
GRID_YEAR = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_service_background_500m_2019_2025.csv"
MORPH = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_morphology_500m.csv"
V5 = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_emergence_suppression_indices_500m_2025_v5.csv"
V6 = PROJECT_ROOT / "data" / "derived_data" / "model" / "controlled_grid_model_500m_2025_v6.csv"
EVENTS = PROJECT_ROOT / "data" / "derived_data" / "model" / "causal_upgrade_event_candidates_v1.csv"
RULES = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_semantic_records_v1.csv"
LEDGER = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_source_ledger_v1.csv"

OUT_PANEL = PROJECT_ROOT / "data" / "derived_data" / "model" / "final_balanced_grid_year_panel_2019_2025_v7.csv"
OUT_EVENT_PANEL = PROJECT_ROOT / "data" / "derived_data" / "model" / "final_event_study_panel_v7.csv"
OUT_EVENT_RESULTS = PROJECT_ROOT / "data" / "derived_data" / "model" / "final_event_study_results_v7.csv"
OUT_MATCH = PROJECT_ROOT / "data" / "derived_data" / "model" / "final_matched_frontier_comparison_v7.csv"
OUT_MODELS = PROJECT_ROOT / "data" / "derived_data" / "model" / "final_model_summary_v7.csv"
OUT_CONCLUSIONS = PROJECT_ROOT / "_rebuild_outputs" / "reports" / "SUPPORTED_FINDINGS_REGISTER_v7.csv"
REPORT = PROJECT_ROOT / "_rebuild_outputs" / "reports" / "64_final_systematic_models_and_results_v7.md"

TITLE = "Who Gets a Pet-Friendly City? Rule-Liminal Venues, Pet-Service Ecologies and Companion-Animal Urban Capability in Shenzhen"
CHINESE_TITLE = "谁获得宠物友好城市？深圳规则临界场所、宠物服务生态与伴侣动物城市能力研究"


def robust_scale(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(0).astype(float)
    lo, hi = x.quantile(0.05), x.quantile(0.95)
    if hi <= lo:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).clip(0, 1)


def build_balanced_panel() -> pd.DataFrame:
    yearly = pd.read_csv(GRID_YEAR, encoding="utf-8-sig")
    morph = pd.read_csv(MORPH, encoding="utf-8-sig")
    grids = morph[["grid_id", "district_name", "edge_cell", "valid_area_km2"]].drop_duplicates()
    years = pd.DataFrame({"year": list(range(2019, 2026))})
    panel = grids.merge(years, how="cross")
    panel = panel.merge(
        yearly.drop(columns=["district_name", "edge_cell", "valid_area_km2"], errors="ignore"),
        on=["grid_id", "year"],
        how="left",
    )
    count_cols = [
        "pet_medical_count",
        "pet_retail_count",
        "pet_grooming_count",
        "pet_boarding_count",
        "pet_training_count",
        "pet_core_service_count",
        "pet_service_diversity_count",
    ]
    panel[count_cols] = panel[count_cols].fillna(0)
    panel["pet_core_service_density_per_km2"] = (
        panel["pet_core_service_count"] / panel["valid_area_km2"].replace(0, np.nan)
    ).fillna(0)
    panel = panel.merge(
        morph[
            [
                "grid_id",
                "boundary_share",
                "road_density_m_per_km2",
                "ped_path_density_m_per_km2",
                "public_space_share",
                "building_coverage_share",
                "building_count_per_km2",
            ]
        ],
        on="grid_id",
        how="left",
    )
    panel["public_space_share"] = panel["public_space_share"].clip(lower=0, upper=1)
    panel["log_service_density"] = np.log1p(panel["pet_core_service_density_per_km2"])
    return panel


def classify_event_scope(row: pd.Series) -> str:
    event_type = str(row["event_type"])
    geo = str(row["geo_level"])
    if "standard" in event_type or geo == "standard_not_geocoded":
        return "exclude_standard_context"
    if geo in {"venue", "public_space", "transport_facility", "venue_complex"} and str(row["grid_id"]).startswith("SZ500_"):
        return "point_grid_event"
    if "district_strategy" in str(row["design_family"]):
        return "district_event"
    return "exclude_for_v7_event_panel"


def build_event_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(EVENTS, encoding="utf-8-sig")
    high = events[
        (events["causal_upgrade_class"] == "high_feasibility_quasi_causal_candidate")
        & events["event_year"].notna()
    ].copy()
    high["event_scope"] = high.apply(classify_event_scope, axis=1)
    point = high[high["event_scope"] == "point_grid_event"].copy()
    # Keep source-quality and timing windows conservative: events after 2025
    # cannot be evaluated with current 2019-2025 POI panel.
    point = point[(point["event_year"] >= 2021) & (point["event_year"] <= 2025)]
    first_event = (
        point.groupby("grid_id")
        .agg(
            first_event_year=("event_year", "min"),
            event_count=("candidate_id", "nunique"),
            event_source_quality=("source_quality_score", "max"),
            event_names=("event_name", lambda s: "|".join(sorted(set(map(str, s)))[:8])),
        )
        .reset_index()
    )
    out = panel.merge(first_event, on="grid_id", how="left")
    out["treated_event_grid"] = out["first_event_year"].notna().astype(int)
    out["post_event"] = ((out["treated_event_grid"] == 1) & (out["year"] >= out["first_event_year"])).astype(int)
    out["event_time"] = out["year"] - out["first_event_year"]
    out["event_time_band"] = out["event_time"].where(out["treated_event_grid"] == 1, np.nan)
    return out, point


def did_event_model(event_panel: pd.DataFrame) -> pd.DataFrame:
    # Limit controls to never-treated grids plus treated grids with usable pre/post.
    df = event_panel.copy()
    usable_treated = df.groupby("grid_id").agg(
        treated=("treated_event_grid", "max"),
        pre=("post_event", lambda x: int((x == 0).any())),
        post=("post_event", lambda x: int((x == 1).any())),
    )
    usable_ids = usable_treated[
        (usable_treated["treated"] == 0)
        | ((usable_treated["treated"] == 1) & (usable_treated["pre"] == 1) & (usable_treated["post"] == 1))
    ].index
    df = df[df["grid_id"].isin(usable_ids)].copy()
    df["year_c"] = df["year"].astype(str)
    results = []
    formulas = {
        "log_service_density": "log_service_density ~ post_event + treated_event_grid + C(year) + C(district_name) + boundary_share + road_density_m_per_km2 + ped_path_density_m_per_km2 + public_space_share + building_coverage_share",
        "pet_service_diversity_count": "pet_service_diversity_count ~ post_event + treated_event_grid + C(year) + C(district_name) + boundary_share + road_density_m_per_km2 + ped_path_density_m_per_km2 + public_space_share + building_coverage_share",
    }
    for target, formula in formulas.items():
        try:
            model = smf.ols(formula=formula, data=df).fit(cov_type="HC3")
            results.append(
                {
                    "model_id": f"event_did_{target}",
                    "outcome": target,
                    "n": int(model.nobs),
                    "treated_grid_count": int(df.groupby("grid_id")["treated_event_grid"].max().sum()),
                    "coef_post_event": float(model.params.get("post_event", np.nan)),
                    "p_post_event": float(model.pvalues.get("post_event", np.nan)),
                    "r2": float(model.rsquared),
                    "interpretation": "post-event association after year, district and morphology controls; not definitive causality",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "model_id": f"event_did_{target}",
                    "outcome": target,
                    "n": int(len(df)),
                    "treated_grid_count": int(df.groupby("grid_id")["treated_event_grid"].max().sum()),
                    "coef_post_event": np.nan,
                    "p_post_event": np.nan,
                    "r2": np.nan,
                    "interpretation": f"model_failed: {exc}",
                }
            )
    return pd.DataFrame(results)


def matched_frontier_comparison() -> pd.DataFrame:
    v6 = pd.read_csv(V6, encoding="utf-8-sig")
    v5 = pd.read_csv(V5, encoding="utf-8-sig")[["grid_id", "pet_service_count_2025", "positive_rule_count"]]
    df = v6.merge(v5, on="grid_id", how="left")
    df["treated_frontier"] = df["is_suppressed_frontier"].astype(int)
    controls = df[(df["treated_frontier"] == 0) & (df["pet_ecology_norm"] > 0)].copy()
    treated = df[df["treated_frontier"] == 1].copy()
    features = [
        "pet_ecology_norm",
        "valid_area_km2",
        "boundary_share",
        "road_density_m_per_km2",
        "ped_path_density_m_per_km2",
        "public_space_share",
        "building_coverage_share",
        "building_count_per_km2",
    ]
    pre = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)],
        remainder="drop",
    )
    Xc = pre.fit_transform(controls[features])
    Xt = pre.transform(treated[features])
    nn = NearestNeighbors(n_neighbors=1).fit(Xc)
    dist, idx = nn.kneighbors(Xt)
    matched_controls = controls.iloc[idx.ravel()].copy().reset_index(drop=True)
    treated = treated.reset_index(drop=True)
    rows = []
    for outcome in ["suppression_index", "positive_rule_norm", "rule_silence_norm", "has_positive_rule", "primary_positive_rule_count"]:
        t = pd.to_numeric(treated[outcome], errors="coerce").fillna(0)
        c = pd.to_numeric(matched_controls[outcome], errors="coerce").fillna(0)
        rows.append(
            {
                "model_id": "nearest_neighbor_matched_frontier",
                "outcome": outcome,
                "treated_n": int(len(treated)),
                "control_n_unique": int(matched_controls["grid_id"].nunique()),
                "treated_mean": float(t.mean()),
                "matched_control_mean": float(c.mean()),
                "difference_treated_minus_control": float((t - c).mean()),
                "mean_match_distance": float(dist.mean()),
                "interpretation": "matched comparison on pet ecology and morphology; diagnostic, not causal",
            }
        )
    return pd.DataFrame(rows)


def time_order_rule_model(panel: pd.DataFrame) -> pd.DataFrame:
    rules = pd.read_csv(RULES, encoding="utf-8-sig")
    ledger = pd.read_csv(LEDGER, encoding="utf-8-sig")[["source_id", "source_class", "publication_use_status"]]
    rules = rules.merge(ledger, on="source_id", how="left")
    rules["event_year"] = pd.to_numeric(rules["source_date"].astype(str).str[:4], errors="coerce")
    rules = rules[
        rules["grid_id"].astype(str).str.startswith("SZ500_")
        & rules["event_year"].between(2020, 2025)
        & rules["source_class"].isin(["A_official_primary", "B_operator_primary"])
    ].copy()
    first_rule = rules.groupby("grid_id").agg(first_primary_rule_year=("event_year", "min")).reset_index()
    base = panel.merge(first_rule, on="grid_id", how="left")
    # Use grid-year before or at rule year: lagged service predicts whether a primary rule appears next/current year.
    base = base.sort_values(["grid_id", "year"])
    base["lag_log_service_density"] = base.groupby("grid_id")["log_service_density"].shift(1).fillna(0)
    base["primary_rule_onset"] = ((base["first_primary_rule_year"].notna()) & (base["year"] == base["first_primary_rule_year"])).astype(int)
    work = base[base["year"].between(2020, 2025)].copy()
    numeric = [
        "lag_log_service_density",
        "boundary_share",
        "road_density_m_per_km2",
        "ped_path_density_m_per_km2",
        "public_space_share",
        "building_coverage_share",
    ]
    X = pd.get_dummies(work[numeric + ["district_name", "year"]], columns=["district_name", "year"], drop_first=True)
    y = work["primary_rule_onset"].astype(int)
    clf = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
            ("logit", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42)),
        ]
    )
    clf.fit(X, y)
    pred = clf.predict_proba(X)[:, 1]
    coef = clf.named_steps["logit"].coef_[0][list(X.columns).index("lag_log_service_density")]
    return pd.DataFrame(
        [
            {
                "model_id": "time_order_primary_rule_onset",
                "outcome": "primary_rule_onset",
                "n": int(len(work)),
                "positive": int(y.sum()),
                "lag_service_coef": float(coef),
                "roc_auc_in_sample": float(roc_auc_score(y, pred)) if y.nunique() > 1 else np.nan,
                "average_precision_in_sample": float(average_precision_score(y, pred)) if y.nunique() > 1 else np.nan,
                "interpretation": "lagged pet-service density association with primary rule onset; time-order diagnostic only",
            }
        ]
    )


def supported_findings(model_summary: pd.DataFrame, event_panel: pd.DataFrame, point_events: pd.DataFrame) -> pd.DataFrame:
    v5 = pd.read_csv(V5, encoding="utf-8-sig")
    v6 = pd.read_csv(V6, encoding="utf-8-sig")
    rules = pd.read_csv(RULES, encoding="utf-8-sig")
    ledger = pd.read_csv(LEDGER, encoding="utf-8-sig")
    rows = [
        {
            "finding_id": "F1",
            "finding": "Shenzhen has a substantial observable pet-service ecology.",
            "support_level": "strong_descriptive",
            "evidence": f"{int(v5['pet_service_count_2025'].sum())} grid-counted pet service points in v5 layer; 1,902 point records in v3 ecology layer.",
            "claim_boundary": "Service ecology is not demand, sentiment or welfare.",
        },
        {
            "finding_id": "F2",
            "finding": "Visible positive pet-access rules are sparse relative to the service ecology.",
            "support_level": "strong_descriptive_with_primary_source_sensitivity",
            "evidence": f"{int(v5['positive_rule_count'].sum())} positive rule grid-counts across {len(v5)} grids; {int(v6['has_primary_positive_rule'].sum())} grids with primary-source-supported positive rules.",
            "claim_boundary": "No observed positive rule is not proof of prohibition.",
        },
        {
            "finding_id": "F3",
            "finding": "Suppressed emergence frontiers are a major spatial mechanism.",
            "support_level": "strong_diagnostic",
            "evidence": f"{int(v6['is_suppressed_frontier'].sum())} suppressed frontier grids; v6 controls retain strong pet-ecology and rule-silence associations.",
            "claim_boundary": "Diagnostic index, not causal effect.",
        },
        {
            "finding_id": "F4",
            "finding": "Rule-liminal venues are a meaningful empirical object.",
            "support_level": "strong_candidate_layer",
            "evidence": "57,994 host candidates and 300 round-1 verification candidates; rule-liminal status is separated from confirmed access.",
            "claim_boundary": "Candidate status does not prove current pet access.",
        },
        {
            "finding_id": "F5",
            "finding": "Quasi-causal upgrading is feasible but not yet definitive.",
            "support_level": "moderate_methodological",
            "evidence": f"{int(point_events['grid_id'].nunique())} point-event grids in v7 event panel; event-study and time-order models run as diagnostics.",
            "claim_boundary": "Use mechanism-consistent language only until event timing and controls are manually verified.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    panel = build_balanced_panel()
    event_panel, point_events = build_event_panel(panel)
    event_results = did_event_model(event_panel)
    match_results = matched_frontier_comparison()
    time_results = time_order_rule_model(panel)
    model_summary = pd.concat([event_results, match_results, time_results], ignore_index=True, sort=False)
    findings = supported_findings(model_summary, event_panel, point_events)

    panel.to_csv(OUT_PANEL, index=False, encoding="utf-8-sig")
    event_panel.to_csv(OUT_EVENT_PANEL, index=False, encoding="utf-8-sig")
    event_results.to_csv(OUT_EVENT_RESULTS, index=False, encoding="utf-8-sig")
    match_results.to_csv(OUT_MATCH, index=False, encoding="utf-8-sig")
    model_summary.to_csv(OUT_MODELS, index=False, encoding="utf-8-sig")
    findings.to_csv(OUT_CONCLUSIONS, index=False, encoding="utf-8-sig")

    counts = {
        "balanced_panel_rows": int(len(panel)),
        "balanced_panel_grids": int(panel["grid_id"].nunique()),
        "event_panel_rows": int(len(event_panel)),
        "treated_event_grids": int(event_panel.groupby("grid_id")["treated_event_grid"].max().sum()),
        "point_event_records_used": int(len(point_events)),
        "model_summary_rows": int(len(model_summary)),
    }
    model_records = model_summary.to_dict("records")
    finding_records = findings.to_dict("records")
    REPORT.write_text(
        f"""# 64 Final Systematic Models And Results V7

Locked title: {TITLE}

Chinese locked title: {CHINESE_TITLE}

## Outputs

- Balanced grid-year panel: `{OUT_PANEL.relative_to(PROJECT_ROOT)}`
- Event-study panel: `{OUT_EVENT_PANEL.relative_to(PROJECT_ROOT)}`
- Event-study results: `{OUT_EVENT_RESULTS.relative_to(PROJECT_ROOT)}`
- Matched frontier comparison: `{OUT_MATCH.relative_to(PROJECT_ROOT)}`
- Model summary: `{OUT_MODELS.relative_to(PROJECT_ROOT)}`
- Supported findings: `{OUT_CONCLUSIONS.relative_to(PROJECT_ROOT)}`

## Counts

`{counts}`

## Model Results

`{model_records}`

## Supported Findings

`{finding_records}`

## Final Interpretation

The final model suite supports the manuscript's central mechanism: pet-service
ecology is spatially extensive, while visible rule readiness is sparse and
uneven. The most important empirical geography is the suppressed emergence
frontier: areas where pet services, rule-liminal hosts and rule silence meet.

The v7 event-study and time-order outputs upgrade the method layer, but they
remain diagnostic. They show patterns consistent with thresholded rule
emergence, not definitive causal proof.

## Claim Boundary

Do not claim actual pet-owner experience, social sentiment, street-level
co-presence or strong causality from these outputs. The supported claim is a
rule-geography and open-data mechanism claim.
""",
        encoding="utf-8",
    )
    print(counts)
    print(model_summary.to_string(index=False))
    print(findings.to_string(index=False))


if __name__ == "__main__":
    main()
