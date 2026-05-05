"""
Generate synthetic Monrovia Pediatric Triage forms as PDFs.

The form layout is modeled on the MSF Aweil Pediatric Triage form, 
adapted for the Monrovia clinical context that motivates this project. 
Each form is filled with medically plausible patient data; the ground 
truth values for the 14 evaluation fields are saved to 
ground_truth.json so the evaluator can score predictions.

The full form layout, three color coded clinical assessment regions,
nested checkbox grids, OPD/referral fields, is visual context that
creates realistic OCR failure conditions. The evaluation schema
itself remains the 14 fields defined in config.FIELD_SCHEMA.
"""

import json
import random

from reportlab.lib.colors import Color
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from config import GROUND_TRUTH_PATH, N_FORMS, PDF_DIR, RANDOM_SEED

# sampling pools

COMPLAINTS = [
    "fever for 3 days", "cough and difficulty breathing",
    "diarrhea and vomiting", "convulsions", "severe abdominal pain",
    "burns on left arm", "snake bite right leg", "lethargy poor feeding",
    "skin rash with fever", "head injury fall from tree",
]
TRIAGE_COLORS = ["RED", "YELLOW", "GREEN"]

# vitals ranges by age band, in months. (max_age_exclusive, rr, hr).
VITALS_BY_AGE_BAND = [
    (12, (30, 60), (100, 160)),
    (60, (20, 40), (80, 130)),
    (999, (15, 30), (70, 110)),
]

# Decorative checkboxes, these are layout context, not part of the evaluation schema. I sample plausible values 
# so the form looks realistically filled out, but I don't track them in ground_truth.
EMERGENCY_AB = [
    "NOT BREATHING", "CENTRAL CYANOSIS", "SEVERE RESPIRATORY DISTRESS"
]
EMERGENCY_D = [
    "COMA (AVPU = P or U)", "CONVULSIONS",
    "SEVERE TRAUMA", "SEVERE BURNS", "CONTAGIOUS DISEASES -> ISOLATE"
]
PRIORITY_SIGNS = [
    "TINY BABY (<2 months)", "TEMPERATURE <36C OR >39C",
    "TRAUMA/SURG EMERGENCY (non-severe)",
    "POISONING (snake/scorpion bite)", "PAIN (severe)", "PALLOR (severe)",
    "RESP DISTRESS (non-severe)", "RESTLESSNESS, IRRITABILITY, LETHARGY",
    "REFERRAL FROM ANOTHER CENTRE", "BURNS (non-severe)"
]


# layout constants

# page geometry, letter is 8.5" x 11"
PAGE_MARGIN = 0.4 * inch
SIDEBAR_WIDTH = 0.35 * inch          # vertical "TRIAGE" label column
CONTENT_LEFT = PAGE_MARGIN + SIDEBAR_WIDTH + 0.05 * inch
CONTENT_WIDTH = 7.6 * inch - SIDEBAR_WIDTH

# section colors (light tints, printable, won't ruin OCR contrast)
COLOR_RED = Color(0.98, 0.85, 0.85)
COLOR_YELLOW = Color(1.0, 0.95, 0.78)
COLOR_GREEN = Color(0.85, 0.94, 0.85)
COLOR_BORDER = Color(0.2, 0.2, 0.2)

# typography
LINE_HEIGHT = 0.20 * inch
CHECKBOX_SIZE = 0.11 * inch


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
        Keys: rr, hr, sat, temperature, weight, muac (string typed).
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
    """Sample one full set of ground truth values for the 14 schema fields."""
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


def sample_decorative_checkboxes(triage_color):
    """Sample plausible decorative checkbox states based on triage color.

    These are NOT in the evaluation schema, they're visual context
    that makes the form look realistically filled. Returns three sets:
    (emergency_marks, priority_marks, non_urgent_marked).
    """
    if triage_color == "RED":
        # 1-2 emergency signs marked, no priority signs
        emergency = set(random.sample(EMERGENCY_AB + EMERGENCY_D,
                                      k=random.randint(1, 2)))
        priority = set()
        non_urgent = False
    elif triage_color == "YELLOW":
        emergency = set()
        priority = set(random.sample(PRIORITY_SIGNS,
                                     k=random.randint(1, 3)))
        non_urgent = False
    else:  # GREEN
        emergency = set()
        priority = set()
        non_urgent = True

    return emergency, priority, non_urgent


# drawing primitives

def _checkbox(c, x, y, filled, label=None, label_size=8):
    """Square checkbox; X if filled. Optional label drawn to the right."""
    c.setStrokeColor(COLOR_BORDER)
    c.setLineWidth(0.5)
    c.rect(x, y, CHECKBOX_SIZE, CHECKBOX_SIZE)
    if filled:
        c.setFont("Helvetica-Bold", label_size)
        c.drawString(x + 0.018 * inch, y + 0.02 * inch, "X")
    if label:
        c.setFont("Helvetica", label_size)
        c.drawString(x + CHECKBOX_SIZE + 0.04 * inch, y + 0.01 * inch, label)


def _field(c, x, y, label, value, label_w, value_w, label_size=9):
    """LABEL: value________ bolded label, value on an underline."""
    c.setFont("Helvetica-Bold", label_size)
    c.drawString(x, y, f"{label}:")
    c.setFont("Helvetica", label_size + 1)
    c.drawString(x + label_w, y, str(value))
    c.setStrokeColor(COLOR_BORDER)
    c.setLineWidth(0.4)
    c.line(x + label_w, y - 0.025 * inch,
           x + label_w + value_w, y - 0.025 * inch)


def _filled_box(c, x, y, w, h, fill_color):
    """Border + light fill, for the colored section regions."""
    c.setFillColor(fill_color)
    c.setStrokeColor(COLOR_BORDER)
    c.setLineWidth(0.6)
    c.rect(x, y, w, h, fill=1, stroke=1)
    c.setFillColor(Color(0, 0, 0))   # restore black for text


def _vertical_label(c, x, y, text, height):
    """Draw rotated text for the TRIAGE sidebar label."""
    c.saveState()
    c.translate(x + SIDEBAR_WIDTH / 2, y - height / 2)
    c.rotate(90)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(0, 0, text)
    c.restoreState()


# PDF rendering

def render_form_pdf(gt, decorative, pdf_path):
    """Render one Monrovia Pediatric Triage form to PDF.

    Parameters
    ----------
    gt : dict
        14 evaluation schema fields from sample_form.
    decorative : tuple
        (emergency_marks_set, priority_marks_set, non_urgent_bool).
    pdf_path : pathlib.Path
        Output path.
    """
    emergency_marks, priority_marks, non_urgent_marked = decorative

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    page_w, page_h = letter

    #  title bar 
    c.setFont("Helvetica", 7)
    c.drawString(PAGE_MARGIN, page_h - 0.25 * inch, "V1_2024")

    title_y = page_h - 0.5 * inch
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(page_w / 2, title_y, "MONROVIA PEDIATRIC TRIAGE")
    c.setFont("Helvetica", 8)
    c.drawCentredString(page_w / 2, title_y - 0.18 * inch,
                        "OPD Triage Record  |  Pediatric (under 5)")

    # Mock institutional header (no logos — synthetic data only)
    c.setFont("Helvetica-Bold", 8)
    c.drawRightString(page_w - PAGE_MARGIN, title_y - 0.05 * inch,
                      "MONROVIA HEALTH")
    c.setFont("Helvetica", 7)
    c.drawRightString(page_w - PAGE_MARGIN, title_y - 0.18 * inch,
                      "Synthetic record - not for clinical use")

    # header block: date, time, name, age, sex, opd, referral 
    header_top = title_y - 0.4 * inch
    header_h = 1.5 * inch
    header_bottom = header_top - header_h
    _filled_box(c, CONTENT_LEFT, header_bottom,
                CONTENT_WIDTH, header_h, Color(1, 1, 1))

    y = header_top - 0.22 * inch
    _field(c, CONTENT_LEFT + 0.1 * inch, y, "DATE", gt["date"],
           0.45 * inch, 1.1 * inch)
    _field(c, CONTENT_LEFT + 2.1 * inch, y, "TIME", gt["time"],
           0.45 * inch, 0.7 * inch)
    ampm_x = CONTENT_LEFT + 3.7 * inch
    for i, lab in enumerate(["AM", "PM"]):
        _checkbox(c, ampm_x + i * 0.55 * inch, y - 0.01 * inch,
                  filled=(gt["ampm"] == lab), label=lab)

    y -= LINE_HEIGHT * 1.4
    _field(c, CONTENT_LEFT + 0.1 * inch, y, "NAME", gt["name"],
           0.5 * inch, 2.0 * inch)
    _field(c, CONTENT_LEFT + 3.0 * inch, y, "AGE", gt["age"],
           0.4 * inch, 1.0 * inch)
    sex_x = CONTENT_LEFT + 5.1 * inch
    c.setFont("Helvetica-Bold", 9)
    c.drawString(sex_x, y, "SEX:")
    for i, lab in enumerate(["M", "F"]):
        _checkbox(c, sex_x + 0.4 * inch + i * 0.5 * inch, y - 0.01 * inch,
                  filled=(gt["sex"] == lab), label=lab)

    y -= LINE_HEIGHT * 1.4
    # OPD case + referral, decorative, not in eval schema
    opd_yes = random.random() < 0.7
    referral_yes = random.random() < 0.3
    c.setFont("Helvetica-Bold", 9)
    c.drawString(CONTENT_LEFT + 0.1 * inch, y, "OPD CASE:")
    _checkbox(c, CONTENT_LEFT + 1.0 * inch, y - 0.01 * inch,
              filled=not opd_yes, label="NO")
    _checkbox(c, CONTENT_LEFT + 1.55 * inch, y - 0.01 * inch,
              filled=opd_yes, label="YES")
    c.drawString(CONTENT_LEFT + 2.5 * inch, y, "REFERRAL:")
    _checkbox(c, CONTENT_LEFT + 3.4 * inch, y - 0.01 * inch,
              filled=not referral_yes, label="NO")
    _checkbox(c, CONTENT_LEFT + 3.95 * inch, y - 0.01 * inch,
              filled=referral_yes, label="YES, FROM:")
    c.setLineWidth(0.4)
    c.line(CONTENT_LEFT + 4.95 * inch, y - 0.025 * inch,
           CONTENT_LEFT + 7.4 * inch, y - 0.025 * inch)

    y -= LINE_HEIGHT * 1.4
    _field(c, CONTENT_LEFT + 0.1 * inch, y, "PRESENTING COMPLAINT",
           gt["presenting_complaint"], 1.7 * inch, 5.4 * inch)

    # vertical TRIAGE sidebar 
    sidebar_top = header_bottom
    sidebar_bottom = PAGE_MARGIN + 0.4 * inch
    sidebar_h = sidebar_top - sidebar_bottom
    c.setStrokeColor(COLOR_BORDER)
    c.setLineWidth(0.6)
    c.rect(PAGE_MARGIN, sidebar_bottom, SIDEBAR_WIDTH, sidebar_h)
    _vertical_label(c, PAGE_MARGIN, sidebar_top, "TRIAGE", sidebar_h)

    # RED: Emergency Signs 
    red_top = header_bottom - 0.05 * inch
    red_h = 1.55 * inch
    red_bottom = red_top - red_h
    _filled_box(c, CONTENT_LEFT, red_bottom,
                CONTENT_WIDTH, red_h, COLOR_RED)

    # action panel on the right
    action_w = 1.7 * inch
    action_x = CONTENT_LEFT + CONTENT_WIDTH - action_w
    c.setStrokeColor(COLOR_BORDER)
    c.line(action_x, red_bottom, action_x, red_top)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(action_x + action_w / 2, red_top - 0.25 * inch, "RED")
    c.setFont("Helvetica-Bold", 7)
    c.drawCentredString(action_x + action_w / 2, red_top - 0.42 * inch,
                        "IMMEDIATELY")
    c.setFont("Helvetica", 7)
    c.drawCentredString(action_x + action_w / 2, red_top - 0.55 * inch,
                        "TRANSFER TO THE")
    c.drawCentredString(action_x + action_w / 2, red_top - 0.66 * inch,
                        "RESUSCITATION AREA")
    c.drawCentredString(action_x + action_w / 2, red_top - 0.85 * inch,
                        "TREAT IMMEDIATELY")
    c.drawCentredString(action_x + action_w / 2, red_top - 1.05 * inch,
                        "WEIGH IF POSSIBLE")

    # emergency Signs checkboxes (left side of red panel)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(CONTENT_LEFT + 0.1 * inch, red_top - 0.18 * inch,
                 "EMERGENCY SIGNS")
    sy = red_top - 0.35 * inch
    for sign in EMERGENCY_AB:
        _checkbox(c, CONTENT_LEFT + 0.15 * inch, sy,
                  filled=(sign in emergency_marks), label=sign, label_size=7)
        sy -= 0.16 * inch

    sy -= 0.05 * inch
    for sign in EMERGENCY_D:
        _checkbox(c, CONTENT_LEFT + 0.15 * inch, sy,
                  filled=(sign in emergency_marks), label=sign, label_size=7)
        sy -= 0.14 * inch

    # VITAL SIGNS strip (between RED and YELLOW) 
    vit_top = red_bottom - 0.05 * inch
    vit_h = 0.65 * inch
    vit_bottom = vit_top - vit_h
    _filled_box(c, CONTENT_LEFT, vit_bottom,
                CONTENT_WIDTH, vit_h, Color(1, 1, 1))

    vy = vit_top - 0.18 * inch
    c.setFont("Helvetica-Bold", 9)
    c.drawString(CONTENT_LEFT + 0.1 * inch, vy, "VITAL SIGNS")
    _field(c, CONTENT_LEFT + 1.0 * inch, vy, "RR", gt["rr"],
           0.25 * inch, 0.6 * inch, label_size=8)
    _field(c, CONTENT_LEFT + 2.1 * inch, vy, "HR", gt["hr"],
           0.25 * inch, 0.6 * inch, label_size=8)
    _field(c, CONTENT_LEFT + 3.2 * inch, vy, "Sat", gt["sat"],
           0.3 * inch, 0.6 * inch, label_size=8)
    _field(c, CONTENT_LEFT + 4.3 * inch, vy, "T", gt["temperature"],
           0.18 * inch, 0.6 * inch, label_size=8)

    vy -= 0.28 * inch
    _field(c, CONTENT_LEFT + 0.1 * inch, vy, "WEIGHT (kg)", gt["weight"],
           0.95 * inch, 1.0 * inch, label_size=8)
    _field(c, CONTENT_LEFT + 3.0 * inch, vy, "MUAC (mm)", gt["muac"],
           0.85 * inch, 1.0 * inch, label_size=8)

    # YELLOW: Priority Signs 
    yel_top = vit_bottom - 0.05 * inch
    yel_h = 1.85 * inch
    yel_bottom = yel_top - yel_h
    _filled_box(c, CONTENT_LEFT, yel_bottom,
                CONTENT_WIDTH, yel_h, COLOR_YELLOW)

    # yellow action panel
    c.line(action_x, yel_bottom, action_x, yel_top)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(action_x + action_w / 2, yel_top - 0.25 * inch,
                        "YELLOW")
    c.setFont("Helvetica-Bold", 7)
    c.drawCentredString(action_x + action_w / 2, yel_top - 0.42 * inch,
                        "PLACE IN PRIORITY IN")
    c.drawCentredString(action_x + action_w / 2, yel_top - 0.53 * inch,
                        "THE QUEUE")
    c.setFont("Helvetica", 7)
    c.drawCentredString(action_x + action_w / 2, yel_top - 0.75 * inch,
                        "RE-EVALUATE EVERY")
    c.drawCentredString(action_x + action_w / 2, yel_top - 0.86 * inch,
                        "20mins")
    c.drawCentredString(action_x + action_w / 2, yel_top - 1.05 * inch,
                        "MAX. WAIT: 1hr")

    c.setFont("Helvetica-Bold", 8)
    c.drawString(CONTENT_LEFT + 0.1 * inch, yel_top - 0.18 * inch,
                 "PRIORITY SIGNS")
    sy = yel_top - 0.35 * inch
    for sign in PRIORITY_SIGNS:
        _checkbox(c, CONTENT_LEFT + 0.15 * inch, sy,
                  filled=(sign in priority_marks), label=sign, label_size=7)
        sy -= 0.14 * inch

    # GREEN: Non-Urgent 
    grn_top = yel_bottom - 0.05 * inch
    grn_h = 0.65 * inch
    grn_bottom = grn_top - grn_h
    _filled_box(c, CONTENT_LEFT, grn_bottom,
                CONTENT_WIDTH, grn_h, COLOR_GREEN)

    c.line(action_x, grn_bottom, action_x, grn_top)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(action_x + action_w / 2, grn_top - 0.22 * inch,
                        "GREEN -> BACK TO MOH")
    c.setFont("Helvetica", 7)
    c.drawCentredString(action_x + action_w / 2, grn_top - 0.4 * inch,
                        "REEVALUATE EVERY 60mins")
    c.drawCentredString(action_x + action_w / 2, grn_top - 0.52 * inch,
                        "MAX WAIT: 4 hours")

    c.setFont("Helvetica-Bold", 8)
    c.drawString(CONTENT_LEFT + 0.1 * inch, grn_top - 0.18 * inch, "NON-URGENT")
    _checkbox(c, CONTENT_LEFT + 0.15 * inch, grn_top - 0.42 * inch,
              filled=non_urgent_marked,
              label="ABSENCE OF EMERGENCY AND PRIORITY SIGNS", label_size=7)

    # TRIAGE classification line (the schema field) 
    tri_y = grn_bottom - 0.25 * inch
    c.setFont("Helvetica-Bold", 9)
    c.drawString(CONTENT_LEFT + 0.1 * inch, tri_y, "TRIAGE CLASSIFICATION:")
    tri_x = CONTENT_LEFT + 1.9 * inch
    for i, color in enumerate(TRIAGE_COLORS):
        cx = tri_x + i * 1.5 * inch
        _checkbox(c, cx, tri_y - 0.01 * inch,
                  filled=(gt["triage_color"] == color),
                  label=color, label_size=9)

    # nurse signature footer 
    foot_y = PAGE_MARGIN + 0.15 * inch
    c.setFont("Helvetica-Bold", 8)
    c.drawString(CONTENT_LEFT + 0.1 * inch, foot_y, "NURSE:")
    c.setLineWidth(0.4)
    c.line(CONTENT_LEFT + 0.6 * inch, foot_y - 0.025 * inch,
           CONTENT_LEFT + 3.5 * inch, foot_y - 0.025 * inch)
    c.drawString(CONTENT_LEFT + 3.7 * inch, foot_y, "SIGNATURE:")
    c.line(CONTENT_LEFT + 4.45 * inch, foot_y - 0.025 * inch,
           CONTENT_LEFT + 7.5 * inch, foot_y - 0.025 * inch)

    c.save()


# entry point

def main():
    """Generate N_FORMS synthetic PDFs and save ground truth to disk."""
    random.seed(RANDOM_SEED)

    ground_truth = {}
    for form_id in range(N_FORMS):
        gt = sample_form(form_id)
        decorative = sample_decorative_checkboxes(gt["triage_color"])
        render_form_pdf(gt, decorative, PDF_DIR / f"form_{form_id:03d}.pdf")
        ground_truth[str(form_id)] = gt

    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated {N_FORMS} forms in {PDF_DIR}")
    print(f"Ground truth: {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()