# Data and code repository

This repository contains shareable derived data, metadata, editable tables and custom code for the Shenzhen pet-friendly city study.

## Repository structure

- `data/`: derived data, figure source data, validation summaries, metadata, source manifests and review-screen tables.
- `code/`: custom Python analysis scripts, figure-assembly scripts, path configuration and input checker.
- `tables/`: editable Word and CSV versions of the manuscript tables.
- `DATA_AND_CODE_AVAILABILITY_NOTE.md`: data and code availability statement.
- `DATA_CODE_FIGURE_TABLE_CROSSWALK.md`: manuscript-element crosswalk linking figures, tables and methods to supporting data and code.
- `data_code_figure_table_crosswalk.csv`: editable crosswalk.

## Availability boundary

The repository provides shareable derived outputs used to support Figs. 1-7, Tables 1-4, Supplementary Table S1 and the methods. Restricted raw commercial POI archives, raw full-text source captures, raw platform or web captures and raw downloaded OSM packages are not redistributed because they may be governed by original licences, access terms or copyright restrictions. The supplied source manifests, metadata tables and derived layers document the processing boundary.

## Code validation

After cloning, run:

```bash
python code/verify_package_inputs.py
```

The input checker reports which analysis scripts run directly from the supplied derived data and which upstream scripts require restricted raw-derived intermediates. In the validated package, nine analysis scripts run from the supplied derived data: scripts 34, 38, 40, 43, 46, 47, 48, 66 and 67. Three upstream scripts document restricted steps requiring licensed raw-derived intermediates: scripts 24, 33 and 37.

To rerun the full shareable derived-layer analysis, use:

```bash
python code/run_reproducible_analysis.py
```

This command reruns the nine scripts supported by the supplied data and writes a run summary to `_rebuild_outputs/reports/`.

## Figure and table support

Use `DATA_CODE_FIGURE_TABLE_CROSSWALK.md` or `data_code_figure_table_crosswalk.csv` to identify which files support each figure, table and analytical step.
