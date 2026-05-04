"""Project-wide configuration constants for Kairo Health."""

from pathlib import Path

# Project paths — derived from this file's location so it works on any machine
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
IMAGE_DIR = DATA_DIR / "images"
RESULTS_DIR = PROJECT_ROOT / "results"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"
OCR_RESULTS_PATH = DATA_DIR / "ocr_results.json"

# Experiment parameters
# N_FORMS = 10 during pipeline development (30 OCR docs across 3 noise tiers)
# N_FORMS = 30 for the final paper run (90 OCR docs total, matching proposal)
N_FORMS = 10
NOISE_LEVELS = ["clean", "moderate", "heavy"]
RANDOM_SEED = 42

# Field schema — both extractors emit JSON with exactly these keys
FIELD_SCHEMA = [
    "date", "time", "ampm", "name", "age", "sex",
    "presenting_complaint", "triage_color",
    "rr", "hr", "sat", "temperature", "weight", "muac",
]

# Make sure output directories exist whenever this module is imported
for d in [PDF_DIR, IMAGE_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
for level in NOISE_LEVELS:
    (IMAGE_DIR / level).mkdir(parents=True, exist_ok=True)
