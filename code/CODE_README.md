# Code package

This folder contains custom Python scripts used for rule-space alignment, model estimation, robustness checks, network construction, validation summaries and figure assembly.

## Environment

Recommended runtime: Python 3.10 or newer.

Core packages: pandas, numpy, scipy, statsmodels, scikit-learn, geopandas, shapely, pyproj, networkx, matplotlib, seaborn, pillow and python-docx.

## How paths are resolved

Every analysis script locates the package automatically by walking up from its own
location until it finds the folder that contains `data`. Inputs are then
read from `data/derived_data/` and there is no setup step or environment
variable to configure. `config.py` exposes the same paths (`PACKAGE_ROOT`,
`DERIVED_DATA`, `FIGURE_SOURCE_DATA`, `REVIEW_SCREENS`) for ad-hoc use, and
`HSSC_SUBMISSION_ROOT` can override the root if you relocate parts of the package.

Run the input check from anywhere after unpacking:

    python code/verify_package_inputs.py

It statically resolves the inputs each analysis script reads and reports which
scripts are fully runnable from the package.

## Reproduction boundary

`analysis_scripts/` reproduces the reported numerical claims from the shareable
derived-data layer:

- **Fully runnable from this package (9 scripts):** the regional emergence/
  suppression indices (40), sparse-rule repair (66), controlled grid models
  (46, 67), rule-liminal threshold model (38), primary-source sensitivity (43),
  causal-upgrade event audit (47), final systematic models (48) and the Schelling
  adoption simulation (34). These read only the included derived data.
- **Need a licensed raw-derived intermediate (3 scripts):** `24` (rule-space
  alignment) needs `hilton_shenzhen_structured_pet_policies_v1.csv`, `33`
  (silent-rule adoption propensity) needs `full_silent_rule_venues_2025_v1.csv`,
  and `37` (service-ecology network) needs `pet_service_core_A_deduplicated.csv`.
  These intermediates are produced directly from the licensed raw POI archive and
  raw source captures and are not redistributed; their schemas and the upstream
  steps are documented in `data/metadata`.

`figure_scripts/` is provided for transparency. The scripts document the figure
assembly logic and should be read together with the supplied panel-source tables,
figure-source manifests and final figure files. Some full figure-rendering steps
depend on the complete local figure-production environment and are therefore
provided as provenance documentation rather than as the primary reproducibility
path. Final figure files are supplied separately in the manuscript submission
package; this repository provides panel source tables and figure manifests under
`data/figure_source_data/`.

Running a script writes its outputs back into `data/derived_data/`
(regenerating the shipped derived files) and writes reports and registers to
`code/_rebuild_outputs/`. To keep the shipped derived data untouched,
copy `data/` before re-running the pipeline.
