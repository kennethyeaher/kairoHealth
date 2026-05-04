"""
Generate synthetic MSF Aweil Pediatric Triage forms as PDFs.

Each form is filled with medically plausible patient data. The
ground truth field values used to fill each form are saved to
ground_truth.json so the evaluator can score predictions later.
"""

import json
import random

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from config import GROUND_TRUTH_PATH, N_FORMS, PDF_DIR, RANDOM_SEED

# sampling pools

# Pediatric realistic value pools. Kept small/explicit so the synthetic
# distribution is auditable rather than buried inside random number calls.
COMPLAINTS = [
    "fever for 3 days", "cough and difficulty breathing",
    "diarrhea and vomiting", "convulsions", "severe abdominal pain",
    "burns on left arm", "snake bite right leg", "lethargy poor feeding",
    "skin rash with fever", "head injury fall from tree",
]
TRIAGE_COLORS = ["RED", "YELLOW", "GREEN"]

# vitals ranges by age band, in months. Each entry is
# (max_age_months_exclusive, rr_range, hr_range). A patient is matched to
# the first band whose max_age is greater than their age in months.
VITALS_BY_AGE_BAND = [
    (12, (30, 60), (100, 160)),   # infants
    (60, (20, 40), (80, 130)),    # toddlers / preschool
    (999, (15, 30), (70, 110)),   # older children (catch-all)
]


# PDF layout constants

# column x-positions used by the form layout. Adjusting these here
# repositions the matching field on every generated form.
COL_LEFT = 0.7 * inch
COL_MID = 3.2 * inch
COL_RIGHT = 5.2 * inch
COL_VITALS_HR = 2.0 * inch
COL_VITALS_SAT = 3.3 * inch
COL_VITALS_T = 4.5 * inch
COL_VITALS_MUAC = 3.0 * inch

LINE_HEIGHT = 0.28 * inch
SECTION_GAP = LINE_HEIGHT * 1.5
TOP_MARGIN = 0.7 * inch
HEADER_OFFSET = 1.2 * inch


# sampling

def sample_vitals(age_months):
    """Sample pediatric vitals roughly calibrated by age band.

    Parameters
    ----------
    age_months : int
        Patient age in months (2-59).

    Returns
    -------
    dict
        Keys: rr, hr, sat, temperature, weight, muac (all string typed
        so they can be compared directly to OCR output later).
    """
    for max_age, rr_range, hr_range in VITALS_BY_AGE_BAND:
        if age_months < max_age:
            rr = random.randint(*rr_range)
            hr = random.randint(*hr_range)
            break

    return {
        "rr": str(rr),
        "hr": str(hr),
        "sat": str(random.randint(88, 99)),
        "temperature": f"{random.uniform(36.0, 39.5):.1f}",
        "weight": f"{random.uniform(3.0, 25.0):.1f}",
        "muac": str(random.randint(105, 145)),
    }


def sample_form(form_id):
    """Sample one full set of ground truth field values for a form.

    Parameters
    ----------
    form_id : int
        Sequential ID used to construct the patient name.

    Returns
    -------
    dict
        All 14 fields in FIELD_SCHEMA, ready to be rendered onto a PDF.
    """
    age_months = random.randint(2, 59)
    return {
        "date": f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/2024",
        "time": f"{random.randint(7, 18):02d}:{random.randint(0, 59):02d}",
        "ampm": random.choice(["AM", "PM"]),
        "name": f"Patient_{form_id:03d}",
        "age": f"{age_months} months",
        "sex": random.choice(["M", "F"]),
        "presenting_complaint": random.choice(COMPLAINTS),
        "triage_color": random.choice(TRIAGE_COLORS),
        **sample_vitals(age_months),
    }


# PDF rendering

class _Cursor:
    """Tracks the current y-position while drawing rows top to bottom.

    Lets each row say 'next line' or 'next section' instead of every
    caller manually subtracting line_height. Internal helper only.
    """

    def __init__(self, start_y):
        self.y = start_y

    def advance(self, multiplier=1.0):
        """Move down by `multiplier` line heights and return new y."""
        self.y -= LINE_HEIGHT * multiplier
        return self.y


def render_form_pdf(gt, pdf_path):
    """Render the ground truth values onto a one page MSF style triage PDF.

    The layout mirrors the real form's three regions (header, triage marker,
    vitals row). [X] marks denote filled checkboxes, a deliberate
    simplification over Unicode checkbox glyphs, which Tesseract handles
    unpredictably across noise levels.

    Parameters
    ----------
    gt : dict
        Ground truth field values from sample_form.
    pdf_path : pathlib.Path
        Output path for the rendered PDF.
    """
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    _, page_height = letter

    # title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * inch, page_height - TOP_MARGIN,
                 "MSF AWEIL PEDIATRIC TRIAGE")

    cursor = _Cursor(page_height - HEADER_OFFSET)

    # header block: date, time, name, age, sex, complaint
    c.setFont("Helvetica", 10)
    c.drawString(COL_LEFT, cursor.y, f"DATE: {gt['date']}")
    c.drawString(COL_MID, cursor.y, f"TIME: {gt['time']} [X] {gt['ampm']}")
    cursor.advance()

    c.drawString(COL_LEFT, cursor.y, f"NAME: {gt['name']}")
    c.drawString(COL_MID, cursor.y, f"AGE: {gt['age']}")
    c.drawString(COL_RIGHT, cursor.y, f"SEX: [X] {gt['sex']}")
    cursor.advance()

    c.drawString(COL_LEFT, cursor.y,
                 f"PRESENTING COMPLAINT: {gt['presenting_complaint']}")
    cursor.advance(multiplier=1.5)

    # triage marker
    c.setFont("Helvetica-Bold", 11)
    c.drawString(COL_LEFT, cursor.y, f"TRIAGE: [X] {gt['triage_color']}")
    cursor.advance(multiplier=1.5)

    # vitals row
    c.setFont("Helvetica-Bold", 10)
    c.drawString(COL_LEFT, cursor.y, "VITAL SIGNS")
    cursor.advance()

    c.setFont("Helvetica", 10)
    c.drawString(COL_LEFT, cursor.y, f"RR: {gt['rr']}")
    c.drawString(COL_VITALS_HR, cursor.y, f"HR: {gt['hr']}")
    c.drawString(COL_VITALS_SAT, cursor.y, f"Sat: {gt['sat']}")
    c.drawString(COL_VITALS_T, cursor.y, f"T: {gt['temperature']}")
    cursor.advance()

    c.drawString(COL_LEFT, cursor.y, f"WEIGHT: {gt['weight']} kg")
    c.drawString(COL_VITALS_MUAC, cursor.y, f"MUAC: {gt['muac']} mm")

    c.save()


# entry point

def main():
    """Generate N_FORMS synthetic PDFs and save ground truth to disk."""
    # seed before any sampling so reruns produce identical forms, required
    # for reproducibility when N_FORMS is later bumped from 10 to 30.
    random.seed(RANDOM_SEED)

    ground_truth = {}
    for form_id in range(N_FORMS):
        gt = sample_form(form_id)
        render_form_pdf(gt, PDF_DIR / f"form_{form_id:03d}.pdf")
        ground_truth[str(form_id)] = gt

    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated {N_FORMS} forms in {PDF_DIR}")
    print(f"Ground truth: {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
