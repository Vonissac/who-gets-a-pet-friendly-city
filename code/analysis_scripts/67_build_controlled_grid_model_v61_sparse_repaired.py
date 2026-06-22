#!/usr/bin/env python3
"""Build controlled grid model V6.1 sparse-repaired.

V6.1 tests whether the v5.1 emergence/suppression pattern is merely a byproduct of
urban morphology, boundary cells, district context or source-quality exposure.
It is a diagnostic association model, not a causal model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve()
for _p in PROJECT_ROOT.parents:
    if (_p / "data").is_dir():
        PROJECT_ROOT = _p
        break
(PROJECT_ROOT / "_rebuild_outputs" / "reports").mkdir(parents=True, exist_ok=True)
V5 = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_emergence_suppression_indices_500m_2025_v51_sparse_repaired.csv"
MORPH = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_morphology_500m.csv"
SENS = PROJECT_ROOT / "data" / "derived_data" / "model" / "primary_source_sensitivity_grid_v1.csv"
HOST = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_liminal_host_candidates_2025_v4.csv"

OUT_GRID = PROJECT_ROOT / "data" / "derived_data" / "model" / "controlled_grid_model_500m_2025_v61_sparse_repaired.csv"
OUT_METRICS = PROJECT_ROOT / "data" / "derived_data" / "model" / "controlled_grid_model_metrics_v61_sparse_repaired.csv"
OUT_COEFS = PROJECT_ROOT / "data" / "derived_data" / "model" / "controlled_grid_model_coefficients_v61_sparse_repaired.csv"
REPORT = PROJECT_ROOT / "_rebuild_outputs" / "reports" / "67_controlled_grid_model_v61_sparse_repaired_report.md"

TITLE = "Pet-Service Ecologies Outpace Urban Readiness in Shenzhen: Rule-Liminal Venues and the Uneven Emergence of Companion-Animal Urban Capability"
CHINESE_TITLE = "宠物城市生态已到，城市尚未准备好：深圳伴侣动物空间生态的规则临界与共处能力涌现机制"


CORE_NUMERIC = [
    "pet_ecology_norm",
    "liminal_host_norm",
    "liminal_potential_norm",
    "rule_liminal_exposure_norm_v51",
    "positive_rule_norm_v51",
    "topology_norm",
    "rule_silence_norm",
    "restrictive_norm",
]

CONTROL_NUMERIC = [
    "valid_area_km2",
    "boundary_share",
    "road_density_m_per_km2",
    "ped_path_density_m_per_km2",
    "public_space_share",
    "building_coverage_share",
    "building_count_per_km2",
    "primary_rule_count",
    "primary_positive_rule_count",
    "primary_restrictive_rule_count",
    "shopping_mall_host_count",
    "hotel_host_count",
    "restaurant_host_count",
    "residential_property_host_count",
    "park_or_recreation_host_count",
]

CATEGORICAL = ["district_name", "edge_cell"]


def safe_clip_public_space_share(series: pd.Series) -> pd.Series:
    # OSM polygons can overlap; cap for modeling while preserving raw value.
    return pd.to_numeric(series, errors="coerce").clip(lower=0, upper=1)


def host_type_counts() -> pd.DataFrame:
    host = pd.read_csv(HOST, encoding="utf-8-sig")
    piv = (
        host.pivot_table(index="grid_id", columns="primary_venue_type", values="host_key", aggfunc="count", fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    rename = {
        "shopping_mall": "shopping_mall_host_count",
        "hotel": "hotel_host_count",
        "restaurant": "restaurant_host_count",
        "residential_property": "residential_property_host_count",
        "park_or_recreation": "park_or_recreation_host_count",
    }
    piv = piv.rename(columns=rename)
    for col in rename.values():
        if col not in piv.columns:
            piv[col] = 0
    return piv[["grid_id", *rename.values()]]


def build_frame() -> pd.DataFrame:
    v5 = pd.read_csv(V5, encoding="utf-8-sig")
    morph = pd.read_csv(MORPH, encoding="utf-8-sig")[
        [
            "grid_id",
            "boundary_share",
            "road_density_m_per_km2",
            "ped_path_density_m_per_km2",
            "public_space_share",
            "building_coverage_share",
            "building_count_per_km2",
        ]
    ]
    sens = pd.read_csv(SENS, encoding="utf-8-sig")[
        [
            "grid_id",
            "primary_rule_count",
            "primary_positive_rule_count",
            "primary_restrictive_rule_count",
            "primary_source_support_class",
            "primary_sensitive_suppression_index",
        ]
    ]
    df = v5.merge(morph, on="grid_id", how="left").merge(sens, on="grid_id", how="left").merge(
        host_type_counts(), on="grid_id", how="left"
    )
    host_cols = [c for c in df.columns if c.endswith("_host_count")]
    df[host_cols] = df[host_cols].fillna(0)
    df["public_space_share_raw"] = df["public_space_share"]
    df["public_space_share"] = safe_clip_public_space_share(df["public_space_share"])
    df["is_suppressed_frontier"] = (df["grid_emergence_type_v51"] == "suppressed_emergence_frontier").astype(int)
    df["has_positive_rule"] = (df["positive_rule_count"] > 0).astype(int)
    df["has_primary_positive_rule"] = (df["primary_positive_rule_count"] > 0).astype(int)
    return df


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def feature_names(pre: ColumnTransformer, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    names = list(numeric_cols)
    if not categorical_cols:
        return names
    cat_pipe = pre.named_transformers_["cat"]
    onehot = cat_pipe.named_steps["onehot"]
    names.extend(onehot.get_feature_names_out(categorical_cols).tolist())
    return names


def fit_logit_cv(df: pd.DataFrame, target: str, numeric_cols: list[str], categorical_cols: list[str]) -> tuple[dict, pd.DataFrame, np.ndarray]:
    y = df[target].astype(int).to_numpy()
    X = df[numeric_cols + categorical_cols]
    pipe = Pipeline(
        [
            ("pre", make_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        return {"target": target, "status": "skipped_low_class_count"}, pd.DataFrame(), np.zeros(len(df))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    pipe.fit(X, y)
    metrics = {
        "target": target,
        "model": "balanced_logistic_regression_cv5",
        "n": int(len(df)),
        "positive": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, pred)),
        "average_precision": float(average_precision_score(y, pred)),
        "accuracy_at_0_5": float(accuracy_score(y, pred >= 0.5)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, pred >= 0.5)),
    }
    names = feature_names(pipe.named_steps["pre"], numeric_cols, categorical_cols)
    coefs = pd.DataFrame(
        {
            "target": target,
            "feature": names,
            "coefficient": pipe.named_steps["model"].coef_[0],
            "abs_coefficient": np.abs(pipe.named_steps["model"].coef_[0]),
        }
    ).sort_values("abs_coefficient", ascending=False)
    return metrics, coefs, pred


def fit_ridge_cv(df: pd.DataFrame, target: str, numeric_cols: list[str], categorical_cols: list[str]) -> tuple[dict, pd.DataFrame, np.ndarray]:
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).to_numpy()
    X = df[numeric_cols + categorical_cols]
    pipe = Pipeline(
        [
            ("pre", make_preprocessor(numeric_cols, categorical_cols)),
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    cv = 5
    pred = cross_val_predict(pipe, X, y, cv=cv)
    pipe.fit(X, y)
    metrics = {
        "target": target,
        "model": "ridge_regression_cv5",
        "n": int(len(df)),
        "positive": "",
        "roc_auc": "",
        "average_precision": "",
        "accuracy_at_0_5": "",
        "balanced_accuracy_at_0_5": "",
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
    }
    names = feature_names(pipe.named_steps["pre"], numeric_cols, categorical_cols)
    coefs = pd.DataFrame(
        {
            "target": target,
            "feature": names,
            "coefficient": pipe.named_steps["model"].coef_,
            "abs_coefficient": np.abs(pipe.named_steps["model"].coef_),
        }
    ).sort_values("abs_coefficient", ascending=False)
    return metrics, coefs, pred


def ablation_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("controls_only", CONTROL_NUMERIC, CATEGORICAL),
        ("core_only", CORE_NUMERIC, []),
        ("core_plus_controls", CORE_NUMERIC + CONTROL_NUMERIC, CATEGORICAL),
    ]
    for label, nums, cats in specs:
        metrics, _, _ = fit_logit_cv(df, "is_suppressed_frontier", nums, cats)
        metrics["spec"] = label
        rows.append(metrics)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    df = build_frame()
    metrics_rows = []
    coef_frames = []

    full_numeric = CORE_NUMERIC + CONTROL_NUMERIC
    for target in ["is_suppressed_frontier", "has_positive_rule", "has_primary_positive_rule"]:
        metrics, coefs, pred = fit_logit_cv(df, target, full_numeric, CATEGORICAL)
        metrics["spec"] = "core_plus_controls"
        metrics_rows.append(metrics)
        if not coefs.empty:
            coef_frames.append(coefs.assign(spec="core_plus_controls"))
            df[f"pred_{target}_v61"] = pred
    metrics, coefs, pred = fit_ridge_cv(df, "suppression_index_v51", full_numeric, CATEGORICAL)
    metrics["spec"] = "core_plus_controls"
    metrics_rows.append(metrics)
    coef_frames.append(coefs.assign(spec="core_plus_controls"))
    df["pred_suppression_index_v61"] = pred

    ablation = ablation_models(df)
    metrics_df = pd.concat([pd.DataFrame(metrics_rows), ablation], ignore_index=True, sort=False)
    coefs_df = pd.concat(coef_frames, ignore_index=True, sort=False)

    key_cols = [
        "grid_id",
        "district_name",
        "edge_cell",
        "grid_emergence_type_v51",
        "emergence_index_v51",
        "suppression_index_v51",
        "is_suppressed_frontier",
        "has_positive_rule",
        "has_primary_positive_rule",
        "pred_is_suppressed_frontier_v61",
        "pred_has_positive_rule_v61",
        "pred_has_primary_positive_rule_v61",
        "pred_suppression_index_v61",
        "positive_rule_count",
        "positive_rule_norm",
        "rule_liminal_exposure_norm",
        *CORE_NUMERIC,
        *CONTROL_NUMERIC,
        "primary_source_support_class",
        "public_space_share_raw",
    ]
    df[key_cols].to_csv(OUT_GRID, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
    coefs_df.to_csv(OUT_COEFS, index=False, encoding="utf-8-sig")

    counts = {
        "grid_rows": int(len(df)),
        "suppressed_frontier": int(df["is_suppressed_frontier"].sum()),
        "positive_rule_grids": int(df["has_positive_rule"].sum()),
        "primary_positive_rule_grids": int(df["has_primary_positive_rule"].sum()),
        "edge_cells": int((df["edge_cell"].astype(str) == "yes").sum()),
    }
    top_coef = (
        coefs_df[coefs_df["target"] == "is_suppressed_frontier"]
        .head(25)[["feature", "coefficient", "abs_coefficient"]]
        .to_dict("records")
    )
    metrics_records = metrics_df.to_dict("records")
    REPORT.write_text(
        f"""# 67 Controlled Grid Model V6.1 Sparse-Repaired Report

Locked title: {TITLE}

Chinese locked title: {CHINESE_TITLE}

## Outputs

- Controlled grid predictions: `{OUT_GRID.relative_to(PROJECT_ROOT)}`
- Metrics: `{OUT_METRICS.relative_to(PROJECT_ROOT)}`
- Coefficients: `{OUT_COEFS.relative_to(PROJECT_ROOT)}`

## Counts

`{counts}`

## Model Design

V6.1 adds morphology, boundary/area, district fixed context, host venue-type mix
and primary-source support controls to the v5 rule-ecology model. It tests
whether suppressed-emergence frontiers and rule visibility remain associated
with pet ecology and liminality after controls.

## Metrics

`{metrics_records}`

## Top Coefficients For Suppressed Frontier

`{top_coef}`

## Interpretation Boundary

These are controlled association models, not causal identification. They test
whether the mechanism survives obvious confounds; they do not prove that pet
services cause rule adoption.
""",
        encoding="utf-8",
    )
    print(counts)
    print(metrics_df.to_string(index=False))
    print(OUT_GRID)


if __name__ == "__main__":
    main()
