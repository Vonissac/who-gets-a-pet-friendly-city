from pathlib import Path
import os

PACKAGE_ROOT = Path(os.environ.get("HSSC_SUBMISSION_ROOT", Path(__file__).resolve().parents[1]))
DATA_ROOT = PACKAGE_ROOT / "data"
DERIVED_DATA = DATA_ROOT / "derived_data"
FIGURE_SOURCE_DATA = DATA_ROOT / "figure_source_data"
REVIEW_SCREENS = DATA_ROOT / "review_screens"
OUTPUT_ROOT = PACKAGE_ROOT / "_rebuild_outputs"
