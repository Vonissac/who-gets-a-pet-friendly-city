#!/usr/bin/env python3
"""HSSC-facing polish for the six approved high-info master figures.

The figures in hssc图片排版 are treated as fixed master layouts. This script
does not reposition subfigures or redraw analytical marks. It removes the old
internal figure header, adds corrected Fig. 2-Fig. 7 HSSC headers, writes
600-DPI submission PNG/PDF/TIFF files, and records exact source mapping.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "hssc图片排版"
OUT = ROOT / "hssc_figure_polished_20260616"
FORMAL = OUT / "formal_upload_figures"
AUDIT = OUT / "audit"
SRC = OUT / "source_maps"
for folder in [FORMAL, AUDIT, SRC]:
    folder.mkdir(parents=True, exist_ok=True)

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
INK = (36, 34, 31)
TEXT = (80, 74, 68)
RULE = (43, 103, 119)
WHITE = (255, 255, 255)

HEADER_H = 250
CROP_TOP = 240
DPI = (600, 600)
BOTTOM_NOTE_CLEAN_H = 230

SPECS = [
    {
        "figure": "Figure_02",
        "number": "2",
        "input": "Fig01_service_rule_atlas_high_info.png",
        "title": "District service-rule atlas across Shenzhen",
        "subtitle": "Citywide substrate, district summaries and local panels separate service ecology, host candidates and visible rule records.",
        "source_map": [
            "Master figure: hssc图片排版/Fig01_service_rule_atlas_high_info.png",
            "Primary source functions: build_fig2_district_atlas_v28.py; build_fig2_v28.py",
            "Source data: grid_emergence_suppression_indices_500m_2025_v51_sparse_repaired.geojson; rule_liminal_host_candidates_2025_v4.csv; pet_service_ecology_points_2025_v3.geojson; rule_semantic_geocoded_points_v1.geojson; OSM layers",
        ],
    },
    {
        "figure": "Figure_03",
        "number": "3",
        "input": "Fig02_rule_liminal_field_high_info.png",
        "title": "Rule-liminal field and semantic rule evidence",
        "subtitle": "Candidate venues, host-type states and traceable rule records are shown without equating candidate signals with confirmed access.",
        "source_map": [
            "Master figure: hssc图片排版/Fig02_rule_liminal_field_high_info.png",
            "Primary source functions: build_fig3_rule_liminal_venues_v28.py",
            "Source data: grid_rule_liminality_500m_2025_v4.geojson; rule_liminal_host_candidates_2025_v4.csv; rule_source_ledger_v1.csv; rule_semantic_records_v1.csv; OSM layers",
        ],
    },
    {
        "figure": "Figure_04",
        "number": "4",
        "input": "Fig03_frontier_evidence_boundary_high_info.png",
        "title": "Suppressed-emergence frontier and evidence boundary",
        "subtitle": "Frontier diagnostics locate grid cells where service ecology and host potential are not matched by visible rule signals.",
        "source_map": [
            "Master figure: hssc图片排版/Fig03_frontier_evidence_boundary_high_info.png",
            "Primary source functions: build_fig4_suppressed_frontier_v28.py; build_fig1_fig6_v27_experiment_adjusted_dense.py",
            "Source data: grid_emergence_suppression_indices_500m_2025_v51_sparse_repaired.geojson; final_matched_frontier_comparison_v7.csv; controlled_grid_model_500m_2025_v61_sparse_repaired.csv",
        ],
    },
    {
        "figure": "Figure_05",
        "number": "5",
        "input": "Fig04_manual_validation_high_info.png",
        "title": "Manual validation narrows the claim",
        "subtitle": "Validation upgrades the curated rule-source layer while showing that candidate queues mostly represent rule non-publicity.",
        "source_map": [
            "Master figure: hssc图片排版/Fig04_manual_validation_high_info.png",
            "Primary source functions: 82_build_completed_validation_figure_v32.py",
            "Source data: data_processed/validation/manual_validation_*_completed_utf8.csv; manual_validation_summary_v1.csv; manual_validation_direction_classes_v1.csv",
        ],
    },
    {
        "figure": "Figure_06",
        "number": "6",
        "input": "Fig05_network_threshold_morphology_high_info.png",
        "title": "Network, threshold and morphology diagnostics",
        "subtitle": "Network exposure, threshold behaviour and morphology are retained as boundary diagnostics rather than a second main claim.",
        "source_map": [
            "Master figure: hssc图片排版/Fig05_network_threshold_morphology_high_info.png",
            "Primary source functions: build_fig5_topology_threshold_v28.py; build_v29_expanded_system.py",
            "Source data: topology/edge datasets, schelling adoption scenarios, controlled model coefficients, advanced_cluster_spatial_diagnostics_500m.csv",
        ],
    },
    {
        "figure": "Figure_07",
        "number": "7",
        "input": "Fig06_local_evidence_vignette_atlas_high_info.png",
        "title": "Local evidence vignette atlas",
        "subtitle": "Local cards diagnose rule-ecology mismatch and validation priorities without treating candidate signals as observed access.",
        "source_map": [
            "Master figure: hssc图片排版/Fig06_local_evidence_vignette_atlas_high_info.png",
            "Primary source functions: build_v30_map_evidence_vignettes.py",
            "Source data: grid_emergence_suppression_indices_500m_2025_v51_sparse_repaired.geojson; pet/rule/host/queue layers; v30_vignette_case_selection.csv",
        ],
    },
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size)


def polish(spec: dict) -> dict:
    src = IN / spec["input"]
    im = Image.open(src).convert("RGB")
    body = im.crop((0, CROP_TOP, im.width, im.height))
    body_draw = ImageDraw.Draw(body)
    body_draw.rectangle((0, body.height - BOTTOM_NOTE_CLEAN_H, body.width, body.height), fill=WHITE)
    out = Image.new("RGB", (im.width, body.height + HEADER_H), WHITE)
    draw = ImageDraw.Draw(out)
    draw.line((115, 60, im.width - 115, 60), fill=RULE, width=5)
    draw.text((115, 95), f"Fig. {spec['number']} | {spec['title']}", fill=INK, font=font(78, True))
    draw.text((115, 178), spec["subtitle"], fill=TEXT, font=font(31, False))
    out.paste(body, (0, HEADER_H))
    png = FORMAL / f"{spec['figure']}.png"
    pdf = FORMAL / f"{spec['figure']}.pdf"
    tif = FORMAL / f"{spec['figure']}.tif"
    out.save(png, dpi=DPI)
    out.save(pdf, resolution=600)
    out.save(tif, dpi=DPI, compression="tiff_lzw")
    (SRC / f"{spec['figure']}_source_map.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "figure": spec["figure"],
        "input_master": str(src),
        "png": str(png),
        "pdf": str(pdf),
        "tif": str(tif),
        "width_px": out.width,
        "height_px": out.height,
        "dpi": "600",
        "crop_top_px": CROP_TOP,
        "header_px": HEADER_H,
    }


def contact(rows: list[dict]) -> None:
    thumbs = []
    for row in rows:
        im = Image.open(row["png"]).convert("RGB")
        im.thumbnail((560, 720), Image.LANCZOS)
        can = Image.new("RGB", (600, 790), WHITE)
        can.paste(im, ((600 - im.width) // 2, 20))
        ImageDraw.Draw(can).text((20, 748), Path(row["png"]).name, fill=INK, font=font(18, False))
        thumbs.append(can)
    sheet = Image.new("RGB", (1800, 1580), WHITE)
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % 3) * 600, (i // 3) * 790))
    sheet.save(AUDIT / "Figure_02_to_07_polished_contact_sheet.png", dpi=(180, 180))


def main() -> None:
    rows = [polish(spec) for spec in SPECS]
    pd.DataFrame(rows).to_csv(OUT / "HSSC_polished_figure_manifest.csv", index=False)
    (OUT / "HSSC_polished_figure_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    contact(rows)
    print(OUT)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
