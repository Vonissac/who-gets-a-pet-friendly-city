#!/usr/bin/env python3
"""Align public-private pet rules to geographic space using semantic similarity.

This first version uses a reproducible local TF-IDF semantic proxy because no
sentence-transformer model is installed. It creates record-level semantic
dimensions, conservative geocoding, GeoJSON outputs, and a grid-level rule
surface.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve()
for _p in PROJECT_ROOT.parents:
    if (_p / "data").is_dir():
        PROJECT_ROOT = _p
        break
(PROJECT_ROOT / "_rebuild_outputs" / "reports").mkdir(parents=True, exist_ok=True)
RULES = PROJECT_ROOT / "data" / "derived_data" / "platform" / "public_private_pet_friendly_rules_v1.csv"
GRID = PROJECT_ROOT / "data" / "derived_data" / "geo" / "shenzhen_grid_500m.gpkg"
PUBLIC_SPACE = PROJECT_ROOT / "data" / "derived_data" / "osm" / "shenzhen_osm_public_space.gpkg"
BUILDINGS = PROJECT_ROOT / "data" / "derived_data" / "osm" / "shenzhen_osm_buildings_3d.gpkg"
HILTON_STRUCTURED = PROJECT_ROOT / "data" / "derived_data" / "platform" / "hilton_shenzhen_structured_pet_policies_v1.csv"

OUT_RECORDS = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_semantic_records_v1.csv"
OUT_POINTS = PROJECT_ROOT / "data" / "derived_data" / "platform" / "rule_semantic_geocoded_points_v1.geojson"
OUT_GRID = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_rule_semantics_500m_v1.csv"
OUT_GRID_GEO = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_rule_semantics_500m_v1.geojson"
REPORT = PROJECT_ROOT / "_rebuild_outputs" / "reports" / "39_rule_semantic_spatial_alignment_report.md"


DIMENSIONS = {
    "full_acceptance": "allowed accepted welcome pet-friendly all merchants accept open pet park pet allowed",
    "conditional_acceptance": "conditional leash vaccine proof height under weight under cleaning feces carrier dog tag behavior norms responsibility",
    "operator_discretion": "operator decides manager decides restaurant shop market rules by operator discretion publicized rules",
    "spatial_zoning": "pet zone pet elevator pet area guestroom only no restaurant no food processing no serving area park list designated area",
    "restriction_refusal": "no pets prohibited banned not accepted no offline pet reception stop pilot service dogs exception",
    "safety_hygiene_control": "safety hygiene feces waste disinfection cleaning injury leash vaccine rabies public health",
    "service_facility_support": "pet toilet handwashing drinking facility stroller waste bags activity facility exercise equipment resting area",
    "commercialization": "hotel package mall merchant restaurant dining store commercial shopping package fee damage fee",
    "reversal_contestation": "pilot stopped complaints contested public debate concern reversal controversy safety hygiene complaints",
    "governance_formalization": "official regulation standard list registration administrative management publicized municipal government",
}

ACCESS_SCORE = {
    "allowed_designated_pet_park": 1.0,
    "allowed_governance_education_site": 0.8,
    "pet_friendly_with_entry_rules": 0.65,
    "pet_friendly_with_zoned_facilities": 0.6,
    "pet_friendly_but_operationally_contested": 0.25,
    "pilot_pet_friendly_restaurant": 0.35,
    "pet_dog_allowed_package_only": 0.25,
    "conditional_access": 0.2,
    "operator_discretion_under_municipal_rule": 0.0,
    "design_guidance": 0.15,
    "pilot_reversal_restrictive": -0.65,
    "generally_no_pets": -0.85,
    "no_pets_except_service_animals": -0.75,
    "pet_friendly_citywide_commercial_network": 0.55,
    "standardized_pet_friendly_commercial_operation": 0.50,
    "pet_dog_allowed_with_cleaning_fee": 0.25,
    "pet_friendly_transport_service": 0.45,
    "administrative_service_support": 0.10,
    "pet_friendly_but_low_evidence_platform_story": 0.25,
    "operator_policy_change_or_inconsistency": 0.0,
    "pet_friendly_with_entry_rules_unverified_directory": 0.30,
    "pet_friendly_hotel_structured_allowed": 0.40,
    "hotel_structured_pets_not_allowed": -0.80,
    "hotel_structured_pet_policy_absent": 0.0,
    "allowed_pet_friendly_activity_space": 0.75,
    "pet_friendly_commercial_demonstration_point": 0.70,
    "district_pet_friendly_development_target": 0.35,
    "allowed_pet_friendly_trail": 0.80,
}

ACCESS_CLASS = {
    "allowed_designated_pet_park": "同意-指定开放",
    "allowed_governance_education_site": "同意-治理教育空间",
    "pet_friendly_with_entry_rules": "半同意-规则准入",
    "pet_friendly_with_zoned_facilities": "半同意-分区准入",
    "pet_friendly_but_operationally_contested": "半同意-争议准入",
    "pilot_pet_friendly_restaurant": "半同意-试点准入",
    "pet_dog_allowed_package_only": "半同意-套餐限定",
    "conditional_access": "半同意-行为条件",
    "operator_discretion_under_municipal_rule": "不确定-经营者裁量",
    "design_guidance": "不确定-空间标准",
    "pilot_reversal_restrictive": "不同意-试点反转",
    "generally_no_pets": "不同意-一般禁宠",
    "no_pets_except_service_animals": "不同意-服务动物例外",
    "pet_friendly_citywide_commercial_network": "半同意-商业网络友好",
    "standardized_pet_friendly_commercial_operation": "半同意-商业标准化",
    "pet_dog_allowed_with_cleaning_fee": "半同意-清洁费限定",
    "pet_friendly_transport_service": "半同意-交通服务支持",
    "administrative_service_support": "不确定-行政服务支持",
    "pet_friendly_but_low_evidence_platform_story": "半同意-低证据叙事",
    "operator_policy_change_or_inconsistency": "不确定-政策时序冲突",
    "pet_friendly_with_entry_rules_unverified_directory": "半同意-目录待核验",
    "pet_friendly_hotel_structured_allowed": "半同意-酒店结构化允许",
    "hotel_structured_pets_not_allowed": "不同意-酒店结构化禁宠",
    "hotel_structured_pet_policy_absent": "不确定-酒店结构化缺失",
    "allowed_pet_friendly_activity_space": "同意-活动空间开放",
    "pet_friendly_commercial_demonstration_point": "半同意-商业示范开放",
    "district_pet_friendly_development_target": "不确定-区级建设目标",
    "allowed_pet_friendly_trail": "同意-郊野径开放",
}

MANUAL_POINTS = {
    "景田北六街宠物公园": (114.0487, 22.5608),
    "洪湖东宠物乐园": (114.1272, 22.5654),
    "春天海社区公园": (114.2465, 22.5595),
    "四海公园宠物乐园": (113.9217, 22.4895),
    "灵芝街心宠物公园": (113.9057, 22.5827),
    "龙岗河干流宠物乐园": (114.2540, 22.7190),
    "龙华区宠物公园": (114.0295, 22.6560),
    "坪山养犬教育科普基地": (114.3560, 22.6890),
    "光明区宠物公园": (113.9360, 22.7550),
    "大鹏宠物公园": (114.4790, 22.5960),
    "沙井京基百纳广场": (113.8233, 22.7328),
    "深圳来福士广场": (113.9257, 22.5006),
    "海底捞深圳上河坊店宠物友好试点": (114.0315, 22.6629),
    "深圳君悦酒店 Dogcation 宠物犬客房套餐": (114.1110, 22.5415),
    "深圳君悦酒店一般酒店政策": (114.1110, 22.5415),
    "深圳柏悦酒店": (114.0579, 22.5355),
    "宝安天虹5家宠物友好商场": (113.9000, 22.5900),
    "观澜湖新城MH MALL": (114.0720, 22.7190),
    "深圳佳兆业万豪酒店": (114.5470, 22.5490),
    "深圳机场宠物候机厅": (113.8100, 22.6390),
    "Baia Burger Concept Bar Restaurant": (114.0545, 22.5372),
    "Fannou Bistro": (113.9300, 22.5000),
    "La Maison": (113.9250, 22.4930),
    "PAW HUB OCT PARK": (113.9860, 22.5350),
    "JW Marriott Hotel Shenzhen Bao'an": (113.8920, 22.5500),
    "Benji Bistro Dutenews": (113.9300, 22.4930),
    "深圳蛇口希尔顿南海酒店": (113.9130, 22.4795),
    "深业上城 Apple Maps pet-friendly review": (114.0598, 22.5560),
    "Benji Bistro Reddit experience": (113.9300, 22.4930),
    "深圳中洲万豪酒店": (113.9437, 22.5277),
    "深圳福田JW万豪酒店": (114.0508, 22.5395),
    "深圳盐田凯悦酒店": (114.2369, 22.5562),
    "深圳南山科技园希尔顿惠庭酒店": (113.9470, 22.5470),
    "深圳国际会展中心希尔顿酒店": (113.7750, 22.7080),
    "深圳万象天地": (113.9539, 22.5455),
    "深圳湾万象城": (113.9456, 22.5127),
    "福田星河COCO Park": (114.0535, 22.5340),
    "Renaissance Shenzhen Bay Hotel": (113.9480, 22.5200),
    "Four Points by Sheraton Shenzhen": (114.0575, 22.5158),
    "Four Points by Sheraton Shenzhen Airport": (113.8100, 22.6390),
    "Four Points by Sheraton Shenzhen Bao'an": (113.8830, 22.5560),
    "The Ritz-Carlton Shenzhen": (114.0585, 22.5362),
    "Shenzhen Marriott Hotel Golden Bay current policy": (114.5470, 22.5490),
    "Shenzhen Marriott Hotel Golden Bay historical policy inconsistency": (114.5470, 22.5490),
    "宝安天虹宠物友好五店": (113.8900, 22.5900),
    "深圳天虹25家宠物友好空间升级": (114.0579, 22.5431),
    "绿景佐阾红树林购物中心": (114.0110, 22.5310),
    "KK ONE购物中心": (114.0240, 22.5290),
    "中洲湾 C Future City 宠物市集": (114.0220, 22.5250),
    "深圳欢乐谷宠物友好季": (113.9800, 22.5370),
    "华润城万象天地": (113.9539, 22.5455),
    "深圳宜家家居北环大道店": (113.9410, 22.5660),
    "海岸城购物中心": (113.9368, 22.5177),
    "沙井京基百纳广场年度报告规则证据": (113.8233, 22.7328),
    "福田绿谷公园周边宠物友好空间": (114.0570, 22.5060),
    "香蜜湖中旅花园二期宠物友好活动区域": (114.0270, 22.5460),
    "沙头新洲路紫薇公园宠物友好空间": (114.0410, 22.5220),
    "深圳国际创新中心A座周边宠物友好空间": (114.0610, 22.5430),
    "华强北鹏怡苑社区小公园宠物友好空间": (114.0850, 22.5500),
    "梅林梅京社区街心小公园宠物友好空间": (114.0550, 22.5700),
    "园岭三街雕塑公园宠物友好空间": (114.0980, 22.5550),
    "莲花彩田村三层坪台宠物友好空间": (114.0650, 22.5620),
    "南园巴登社区万德居花园宠物友好空间": (114.0960, 22.5410),
    "福保桂花都市公园篮球场旁宠物友好空间": (114.0630, 22.5080),
    "福田星河COCO Park COCO Paw体系": (114.0535, 22.5340),
    "东门雅园立交桥下宠物友好活动空间": (114.1240, 22.5480),
    "翠竹德弘社区公园宠物友好活动空间": (114.1290, 22.5680),
    "东湖罗安居社区公园宠物友好活动空间": (114.1410, 22.5790),
    "东晓笋岗西路转角公园宠物友好活动空间": (114.1250, 22.5740),
    "桂园地王大厦对面街心公园宠物友好活动空间": (114.1100, 22.5450),
    "黄贝社区公园宠物友好活动空间": (114.1390, 22.5520),
    "莲塘莲花社区公园宠物友好活动空间": (114.1640, 22.5600),
    "南湖玛雅城右侧草地宠物友好活动空间": (114.1160, 22.5360),
    "清水河环湖三路金湖路银湖二街宠物友好活动空间": (114.0960, 22.5780),
    "笋岗田贝社区44号大院文化广场宠物友好活动空间": (114.1210, 22.5700),
    "罗湖区宠物经济高质量发展实施方案空间目标": (114.1310, 22.5480),
    "罗芳公园郊野径宠物友好线路": (114.1550, 22.5520),
    "ZoNe丨宠域知汇宠物生态城": (114.1055, 22.5644),
    "天河城绿洲·宠物友好生活聚场": (114.1055, 22.5644),
    "深圳巴士H239宠物友好线路": (113.9300, 22.5420),
    "深圳宠物友好空间官方媒体统计": (114.0579, 22.5431),
    "前海欢乐港湾宠物友好商业街区": (113.8800, 22.5465),
    "深圳欢乐谷宠物友好季服务体系": (113.9800, 22.5370),
    "龙华区2025宠物友好社区与商场目标": (114.0440, 22.6960),
}

if HILTON_STRUCTURED.exists():
    _hilton = pd.read_csv(HILTON_STRUCTURED)
    HILTON_COORDS = {
        str(row["venue_name"]): (float(row["lon"]), float(row["lat"]))
        for _, row in _hilton.iterrows()
        if np.isfinite(row.get("lon", np.nan)) and np.isfinite(row.get("lat", np.nan))
    }
else:
    HILTON_COORDS = {}

DISTRICT_CENTROIDS = {
    "福田区": (114.055, 22.522),
    "罗湖区": (114.131, 22.548),
    "南山区": (113.930, 22.533),
    "宝安区": (113.884, 22.555),
    "龙岗区": (114.247, 22.720),
    "龙华区": (114.044, 22.696),
    "坪山区": (114.346, 22.690),
    "光明区": (113.936, 22.748),
    "盐田区": (114.237, 22.557),
    "大鹏新区": (114.477, 22.596),
}


def tokenize(text: str) -> str:
    text = text.lower()
    parts = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
    return " ".join(parts)


def classify_granular(row: pd.Series, sims: dict[str, float]) -> str:
    access_position = str(row["access_position"])
    if access_position in {"pet_friendly_citywide_commercial_network", "standardized_pet_friendly_commercial_operation"}:
        return "A3_networked_commercial_pet_friendly_rule"
    if access_position == "pet_friendly_commercial_demonstration_point":
        return "A4_demonstration_commercial_pet_friendly_rule"
    if access_position == "district_pet_friendly_development_target":
        return "D2_district_pet_friendly_development_target"
    if access_position == "allowed_pet_friendly_activity_space":
        return "A1_full_acceptance_designated_public_pet_space"
    if access_position == "allowed_pet_friendly_trail":
        return "A5_pet_friendly_trail_or_mobility_space"
    if access_position in {"administrative_service_support"}:
        return "D1_citywide_administrative_support"
    if access_position in {"pet_friendly_transport_service"}:
        return "B5_mobility_service_pet_friendly_support"
    if access_position in {"pet_dog_allowed_with_cleaning_fee"}:
        return "B2_zoned_or_package_limited_acceptance"
    if access_position == "allowed_designated_pet_park":
        return "A1_full_acceptance_designated_public_pet_space"
    if access_position == "allowed_governance_education_site":
        return "A2_acceptance_governance_education_space"
    if access_position in {"pet_friendly_with_entry_rules", "conditional_access"}:
        return "B1_conditional_acceptance_behavior_rules"
    if access_position in {"pet_friendly_with_zoned_facilities", "pet_dog_allowed_package_only"}:
        return "B2_zoned_or_package_limited_acceptance"
    if access_position == "pilot_pet_friendly_restaurant":
        return "B3_experimental_commercial_acceptance"
    if access_position == "pet_friendly_but_operationally_contested":
        return "B4_contested_pet_friendly_claim"
    if access_position == "operator_discretion_under_municipal_rule":
        return "C1_operator_discretion_governance_delegation"
    if access_position == "design_guidance":
        return "C2_spatial_standard_without_access_commitment"
    if row["access_semantic_score"] < -0.5:
        return "C1_restrictive_or_reversal"
    if sims["reversal_contestation"] >= 0.18:
        return "B4_contested_pet_friendly_claim"
    return "C3_ambiguous_or_unresolved_rule"


def geocode_rule(row: pd.Series, public_space: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> tuple[float, float, str, str]:
    venue = str(row["venue_name"])
    if str(row.get("source_id", "")).startswith("PPR_HILTON_") and HILTON_COORDS:
        if venue in HILTON_COORDS:
            lon, lat = HILTON_COORDS[venue]
            return lon, lat, "hilton_structured_coordinate", "venue"
    if venue in MANUAL_POINTS:
        lon, lat = MANUAL_POINTS[venue]
        return lon, lat, "manual_verified_place_seed", "venue"

    # OSM name match for public spaces/buildings when available.
    for gdf, layer in [(public_space, "osm_public_space"), (buildings, "osm_building")]:
        if "name" in gdf.columns:
            matches = gdf[gdf["name"].fillna("").astype(str).apply(lambda x: bool(x) and (venue in x or x in venue))]
            if not matches.empty:
                pt = matches.to_crs("EPSG:4326").geometry.iloc[0].representative_point()
                return float(pt.x), float(pt.y), f"{layer}_name_match", "venue"

    area = str(row["district_or_area"])
    for district, xy in DISTRICT_CENTROIDS.items():
        if district in area:
            return xy[0], xy[1], "district_centroid", "district"
    if "citywide" in area or "multi_area" in area:
        return 114.0579, 22.5431, "citywide_centroid", "citywide"
    return math.nan, math.nan, "unmatched", "unmatched"


def influence_radius_m(row: pd.Series) -> float:
    venue_type = str(row.get("venue_type", ""))
    sector = str(row.get("sector", ""))
    geo_level = str(row.get("geo_level", ""))
    if geo_level == "citywide":
        return 20000.0
    if "shopping_mall_network" in venue_type or "shopping_mall_standard" in venue_type:
        return 12000.0
    if "shopping_mall_group" in venue_type:
        return 8000.0
    if "airport" in venue_type or "mobility" in sector:
        return 10000.0
    if "hotel" in venue_type:
        return 3000.0
    if "restaurant" in venue_type:
        return 2000.0
    if "pet_cafe" in venue_type or "pet_care" in venue_type:
        return 2500.0
    if "district_pet_friendly_target" in venue_type:
        return 9000.0
    if "pet_friendly_activity_space" in venue_type:
        return 1800.0
    if "pet_friendly_trail" in venue_type:
        return 3000.0
    if "shopping_mall_demonstration" in venue_type or "pet_economy_complex" in venue_type or "pet_social_space" in venue_type:
        return 5000.0
    if "pet_friendly_bus_route" in venue_type:
        return 12000.0
    if "residential" in venue_type:
        return 5000.0
    if "pet_friendly_park" in venue_type or "pet_governance" in venue_type:
        return 2500.0
    return 3000.0


def influence_weight(distance_m: float, radius_m: float) -> float:
    if distance_m > radius_m * 3:
        return 0.0
    return math.exp(-0.5 * (distance_m / max(radius_m, 1.0)) ** 2)


def main() -> None:
    OUT_RECORDS.parent.mkdir(parents=True, exist_ok=True)
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RULES)
    texts = (
        df["access_position"].fillna("")
        + " "
        + df["rule_features"].fillna("")
        + " "
        + df["rule_summary"].fillna("")
        + " "
        + df["sector"].fillna("")
        + " "
        + df["venue_type"].fillna("")
    )
    dim_names = list(DIMENSIONS)
    corpus = [tokenize(t) for t in texts.tolist()] + [tokenize(DIMENSIONS[k]) for k in dim_names]
    vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=1)
    mat = vectorizer.fit_transform(corpus)
    sim = cosine_similarity(mat[: len(df)], mat[len(df) :])
    for i, name in enumerate(dim_names):
        df[f"sem_{name}"] = np.round(sim[:, i], 4)
    df["access_semantic_score"] = df["access_position"].map(ACCESS_SCORE).fillna(0.0)
    df["access_class_cn"] = df["access_position"].map(ACCESS_CLASS).fillna("不确定-未分类")
    df["restriction_intensity"] = np.round(np.maximum(df["sem_restriction_refusal"], 0) + (df["access_semantic_score"] < 0).astype(float) * 0.35, 4)
    df["conditionality_intensity"] = np.round(df["sem_conditional_acceptance"] + df["sem_spatial_zoning"] + df["sem_safety_hygiene_control"], 4)
    df["facility_support_intensity"] = np.round(df["sem_service_facility_support"], 4)
    df["commercial_rule_intensity"] = np.round(df["sem_commercialization"], 4)
    granular = []
    for _, row in df.iterrows():
        sims = {name: float(row[f"sem_{name}"]) for name in dim_names}
        granular.append(classify_granular(row, sims))
    df["rule_granular_class"] = granular

    grid = gpd.read_file(GRID)
    grid_wgs = grid.to_crs("EPSG:4326")
    public_space = gpd.read_file(PUBLIC_SPACE)
    buildings = gpd.read_file(BUILDINGS)
    geocoded = [geocode_rule(row, public_space, buildings) for _, row in df.iterrows()]
    df["lon"] = [x[0] for x in geocoded]
    df["lat"] = [x[1] for x in geocoded]
    df["geo_method"] = [x[2] for x in geocoded]
    df["geo_level"] = [x[3] for x in geocoded]
    df["influence_radius_m"] = df.apply(influence_radius_m, axis=1)
    points = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) if not math.isnan(xy[0]) else None for xy in zip(df["lon"], df["lat"])],
        crs="EPSG:4326",
    )
    valid_points = points[points.geometry.notna()].copy()
    joined = gpd.sjoin(valid_points.to_crs(grid.crs), grid[["grid_id", "district_name", "geometry"]], how="left", predicate="within")
    df = df.merge(joined[["source_id", "grid_id", "district_name"]], on="source_id", how="left")
    df.to_csv(OUT_RECORDS, index=False, encoding="utf-8-sig")
    valid_points = valid_points.merge(df[["source_id", "grid_id"]], on="source_id", how="left")
    valid_points.to_file(OUT_POINTS, driver="GeoJSON")

    assigned = df[df["grid_id"].fillna("") != ""].copy()
    if assigned.empty:
        grid_scores = pd.DataFrame(columns=["grid_id"])
    else:
        assigned["is_allowed"] = assigned["access_semantic_score"] > 0.45
        assigned["is_conditional"] = assigned["access_class_cn"].str.contains("半同意|裁量|标准", regex=True)
        assigned["is_restrictive"] = assigned["access_semantic_score"] < -0.4
        assigned["is_private"] = assigned["sector"].str.contains("private", regex=False)
        assigned["is_official"] = assigned["sector"].str.contains("official", regex=False)
        grid_scores = assigned.groupby("grid_id").agg(
            rule_record_count=("source_id", "nunique"),
            rule_access_score_mean=("access_semantic_score", "mean"),
            rule_allowed_count=("is_allowed", "sum"),
            rule_conditional_count=("is_conditional", "sum"),
            rule_restrictive_count=("is_restrictive", "sum"),
            rule_private_count=("is_private", "sum"),
            rule_official_count=("is_official", "sum"),
            rule_restriction_intensity_mean=("restriction_intensity", "mean"),
            rule_conditionality_intensity_mean=("conditionality_intensity", "mean"),
            rule_facility_support_intensity_mean=("facility_support_intensity", "mean"),
            rule_commercial_intensity_mean=("commercial_rule_intensity", "mean"),
        ).reset_index()
    grid_scores.to_csv(OUT_GRID, index=False, encoding="utf-8-sig")
    # Influence surface: each geocoded rule contributes to nearby grids through
    # a Gaussian distance-decay kernel with a radius set by venue/rule scope.
    grid_centers = grid.copy()
    grid_centers["geometry"] = grid_centers.geometry.centroid
    valid_projected = valid_points.to_crs(grid.crs)
    inf_rows = []
    for _, rule in df[df["geo_level"].ne("unmatched")].iterrows():
        match = valid_projected[valid_projected["source_id"].eq(rule["source_id"])]
        if match.empty:
            continue
        point = match.geometry.iloc[0]
        radius = float(rule["influence_radius_m"])
        candidate = grid_centers[grid_centers.geometry.distance(point) <= radius * 3].copy()
        if candidate.empty:
            continue
        distances = candidate.geometry.distance(point)
        weights = distances.apply(lambda d: influence_weight(float(d), radius))
        for grid_id, weight in zip(candidate["grid_id"], weights):
            if weight <= 0:
                continue
            inf_rows.append(
                {
                    "grid_id": grid_id,
                    "source_id": rule["source_id"],
                    "weight": float(weight),
                    "weighted_access": float(weight) * float(rule["access_semantic_score"]),
                    "weighted_restriction": float(weight) * float(rule["restriction_intensity"]),
                    "weighted_conditionality": float(weight) * float(rule["conditionality_intensity"]),
                    "weighted_facility": float(weight) * float(rule["facility_support_intensity"]),
                    "weighted_commercial": float(weight) * float(rule["commercial_rule_intensity"]),
                }
            )
    influence_df = pd.DataFrame(inf_rows)
    if not influence_df.empty:
        influence_grid = influence_df.groupby("grid_id").agg(
            rule_influence_weight_sum=("weight", "sum"),
            rule_influence_access_sum=("weighted_access", "sum"),
            rule_influence_restriction_sum=("weighted_restriction", "sum"),
            rule_influence_conditionality_sum=("weighted_conditionality", "sum"),
            rule_influence_facility_sum=("weighted_facility", "sum"),
            rule_influence_commercial_sum=("weighted_commercial", "sum"),
            rule_influence_source_count=("source_id", "nunique"),
        ).reset_index()
        influence_grid["rule_influence_access_mean"] = influence_grid["rule_influence_access_sum"] / influence_grid["rule_influence_weight_sum"]
        influence_grid["rule_influence_restriction_mean"] = influence_grid["rule_influence_restriction_sum"] / influence_grid["rule_influence_weight_sum"]
    else:
        influence_grid = pd.DataFrame(columns=["grid_id"])
    influence_out = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_rule_influence_surface_500m_v1.csv"
    influence_geo_out = PROJECT_ROOT / "data" / "derived_data" / "model" / "grid_rule_influence_surface_500m_v1.geojson"
    influence_grid.to_csv(influence_out, index=False, encoding="utf-8-sig")
    grid_geo = grid_wgs.merge(grid_scores, on="grid_id", how="left").merge(influence_grid, on="grid_id", how="left")
    for col in grid_geo.columns:
        if col.startswith("rule_"):
            grid_geo[col] = grid_geo[col].fillna(0)
    grid_geo.to_file(OUT_GRID_GEO, driver="GeoJSON")
    grid_geo.to_file(influence_geo_out, driver="GeoJSON")

    report_counts = {
        "records": int(len(df)),
        "geo_level": df["geo_level"].value_counts(dropna=False).to_dict(),
        "geo_method": df["geo_method"].value_counts(dropna=False).to_dict(),
        "access_class_cn": df["access_class_cn"].value_counts(dropna=False).to_dict(),
        "rule_granular_class": df["rule_granular_class"].value_counts(dropna=False).to_dict(),
        "grid_rows_with_rules": int(len(grid_scores)),
        "grid_rows_with_influence": int(len(influence_grid)),
    }
    REPORT.write_text(
        f"""# 39 Rule Semantic Spatial Alignment Report

Locked title: Pet-Service Ecologies Outpace Urban Readiness in Shenzhen: Rule-Liminal Venues and the Uneven Emergence of Companion-Animal Urban Capability

## Execution

- Rule records aligned: `{len(df)}`
- Semantic method: local TF-IDF cosine similarity over rule summaries, access positions, sectors, venue types and rule features.
- Output record table: `{OUT_RECORDS.relative_to(PROJECT_ROOT)}`
- Output geocoded points: `{OUT_POINTS.relative_to(PROJECT_ROOT)}`
- Output grid table: `{OUT_GRID.relative_to(PROJECT_ROOT)}`
- Output grid GeoJSON: `{OUT_GRID_GEO.relative_to(PROJECT_ROOT)}`
- Output influence surface CSV: `{influence_out.relative_to(PROJECT_ROOT)}`
- Output influence surface GeoJSON: `{influence_geo_out.relative_to(PROJECT_ROOT)}`

## Counts

`{report_counts}`

## Semantic Dimensions

`{list(DIMENSIONS.keys())}`

## Interpretation Boundary

This is a first-pass semantic alignment layer. Venue points include manually verified place seeds and conservative OSM/district/citywide fallback. Citywide and district rules are represented by centroids for visualization but should not be interpreted as exact site-level rule locations. Replace TF-IDF with a multilingual sentence-embedding model when available.
""",
        encoding="utf-8",
    )
    print(report_counts)
    print(OUT_RECORDS)
    print(OUT_GRID_GEO)


if __name__ == "__main__":
    main()
