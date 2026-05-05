"""
Generate synthetic Monrovia Pediatric Triage forms as PDFs.

The form layout is modeled on the MSF Aweil Pediatric Triage form
(V1, April 2023), adapted for the Monrovia clinical context that
motivates this project. Each form is filled with medically plausible
patient data; the ground truth values for the 14 evaluation fields
are saved to ground_truth.json so the evaluator can score predictions.

The full form layout; three color-coded clinical assessment regions,
nested checkbox grids, OPD/referral fields, test-and-treat panel, and
observation table, is visual context that creates realistic OCR
failure conditions. The evaluation schema itself remains the 14 fields
defined in config.FIELD_SCHEMA.
"""

import json
import random
from dataclasses import dataclass

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

# (max_age_exclusive_months, rr_range, hr_range)
VITALS_BY_AGE_BAND = [
    (12, (30, 60), (100, 160)),
    (60, (20, 40), (80, 130)),
    (999, (15, 30), (70, 110)),
]

# decorative checkbox labels 
EMERGENCY_AB_SIGNS = [
    "NOT BREATHING", "CENTRAL CYANOSIS", "SEVERE RESPIRATORY DISTRESS",
]
EMERGENCY_D_SIGNS = [
    "COMA (AVPU = P or U)", "CONVULSIONS",
    "SEVERE TRAUMA", "SEVERE BURNS", "CONTAGIOUS DISEASES -> ISOLATE",
]
PRIORITY_SIGNS = [
    "TINY BABY (<2 months)", "TEMPERATURE <36C OR >39C",
    "TRAUMA/SURG EMERGENCY (non-severe)",
    "POISONING (snake/scorpion bite)", "PAIN (severe)", "PALLOR (severe)",
    "RESP DISTRESS (non-severe)", "RESTLESSNESS, IRRITABILITY, LETHARGY",
    "REFERRAL FROM ANOTHER CENTRE", "BURNS (non-severe)",
]

REFERRAL_SOURCES = [
    "Redemption Hospital", "JFK Medical Center", "ELWA Hospital",
    "Phebe Hospital", "Community Clinic - Paynesville",
]


# layout constants

PAGE_MARGIN = 0.4 * inch
SIDEBAR_WIDTH = 0.35 * inch
CONTENT_LEFT = PAGE_MARGIN + SIDEBAR_WIDTH + 0.05 * inch
CONTENT_WIDTH = 7.6 * inch - SIDEBAR_WIDTH
CONTENT_RIGHT = CONTENT_LEFT + CONTENT_WIDTH

# section colors
COLOR_RED = Color(0.98, 0.85, 0.85)
COLOR_YELLOW = Color(1.0, 0.95, 0.78)
COLOR_GREEN = Color(0.85, 0.94, 0.85)
COLOR_GREY = Color(0.94, 0.94, 0.94)
COLOR_WHITE = Color(1, 1, 1)
COLOR_BORDER = Color(0.2, 0.2, 0.2)

LINE_HEIGHT = 0.20 * inch
CHECKBOX_SIZE = 0.11 * inch
ACTION_PANEL_WIDTH = 1.7 * inch
SECTION_GAP = 0.05 * inch


# sampling

@dataclass
class Decorative:
    """Decorative form state, visual context only, not in FIELD_SCHEMA.

    These values fill out the form to look realistic and give OCR
    realistic failure surface, but the evaluator does not score them.
    """
    emergency: set       # subset of EMERGENCY_AB_SIGNS + EMERGENCY_D_SIGNS
    priority: set        # subset of PRIORITY_SIGNS
    non_urgent: bool
    opd_yes: bool
    referral_yes: bool
    referral_source: str
    malaria_positive: bool
    act_treated: bool
    bsl: str
    hemoglobin: str
    gave_paracetamol: bool
    gave_glucose: bool
    gave_ors: bool
    obs_rows: list       # list of (time_str, pews_str, bsl_str)


def sample_vitals(age_months):
    """Sample pediatric vitals roughly calibrated by age band."""
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
    """Sample one full set of ground-truth values for the 14 schema fields."""
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


def sample_decorative(triage_color):
    """Sample plausible decorative state consistent with the triage color."""
    if triage_color == "RED":
        emergency = set(random.sample(
            EMERGENCY_AB_SIGNS + EMERGENCY_D_SIGNS, k=random.randint(1, 2)))
        priority = set()
        non_urgent = False
    elif triage_color == "YELLOW":
        emergency = set()
        priority = set(random.sample(PRIORITY_SIGNS, k=random.randint(1, 3)))
        non_urgent = False
    else:
        emergency = set()
        priority = set()
        non_urgent = True

    referral_yes = random.random() < 0.3
    return Decorative(
        emergency=emergency,
        priority=priority,
        non_urgent=non_urgent,
        opd_yes=random.random() < 0.7,
        referral_yes=referral_yes,
        referral_source=random.choice(REFERRAL_SOURCES) if referral_yes else "",
        malaria_positive=random.random() < 0.3,
        act_treated=random.random() < 0.2,
        bsl=str(random.randint(45, 110)),
        hemoglobin=f"{random.uniform(7.0, 13.5):.1f}",
        gave_paracetamol=random.random() < 0.4,
        gave_glucose=random.random() < 0.15,
        gave_ors=random.random() < 0.3,
        obs_rows=[
            (f"{random.randint(7, 18):02d}:{random.randint(0, 59):02d}",
             str(random.randint(0, 5)), str(random.randint(60, 120)))
            for _ in range(random.randint(1, 3))
        ],
    )


# drawing primitives

def draw_checkbox(c, x, y, filled, label=None, label_size=8):
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


def draw_field(c, x, y, label, value, label_w, value_w, label_size=9):
    """LABEL: value________ bolded label, value on an underline."""
    c.setFont("Helvetica-Bold", label_size)
    c.drawString(x, y, f"{label}:")
    c.setFont("Helvetica", label_size + 1)
    c.drawString(x + label_w, y, str(value))
    c.setStrokeColor(COLOR_BORDER)
    c.setLineWidth(0.4)
    c.line(x + label_w, y - 0.025 * inch,
           x + label_w + value_w, y - 0.025 * inch)


def draw_section_box(c, top, height, fill_color):
    """Bordered, filled rectangle spanning the content width.

    Returns the bottom y-coordinate (top - height) for chaining.
    """
    bottom = top - height
    c.setFillColor(fill_color)
    c.setStrokeColor(COLOR_BORDER)
    c.setLineWidth(0.6)
    c.rect(CONTENT_LEFT, bottom, CONTENT_WIDTH, height, fill=1, stroke=1)
    c.setFillColor(Color(0, 0, 0))
    return bottom


def draw_section_title(c, top, title):
    """Bold section title positioned at the top left of a section."""
    c.setFont("Helvetica-Bold", 9)
    c.drawString(CONTENT_LEFT + 0.1 * inch, top - 0.18 * inch, title)


def draw_action_panel(c, top, height, header_lines, body_lines):
    """Right side colored action panel (e.g. 'RED IMMEDIATELY...').

    Parameters
    ----------
    header_lines : list of (text, font_size)
        Bold header lines at the top of the panel.
    body_lines : list of (text, font_size, bold)
        Body lines below the header. Empty strings produce vertical space.
    """
    panel_x = CONTENT_RIGHT - ACTION_PANEL_WIDTH
    panel_cx = panel_x + ACTION_PANEL_WIDTH / 2

    c.setStrokeColor(COLOR_BORDER)
    c.setLineWidth(0.6)
    c.line(panel_x, top - height, panel_x, top)

    # headers
    y = top - 0.22 * inch
    for text, size in header_lines:
        c.setFont("Helvetica-Bold", size)
        c.drawCentredString(panel_cx, y, text)
        y -= size * 1.4 / 72 * inch

    # body
    y -= 0.05 * inch
    for text, size, bold in body_lines:
        font = "Helvetica-Bold" if bold else "Helvetica"
        c.setFont(font, size)
        c.drawCentredString(panel_cx, y, text)
        y -= size * 1.4 / 72 * inch


def draw_vertical_label(c, x, y, text, height):
    """Rotated text for the TRIAGE sidebar."""
    c.saveState()
    c.translate(x + SIDEBAR_WIDTH / 2, y - height / 2)
    c.rotate(90)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(0, 0, text)
    c.restoreState()


def draw_checkbox_list(c, x, top_y, items, marked_set, row_height=0.14 * inch,
                       label_size=7):
    """Vertical list of labeled checkboxes. Returns the next y-coordinate."""
    y = top_y
    for label in items:
        draw_checkbox(c, x, y, filled=(label in marked_set),
                      label=label, label_size=label_size)
        y -= row_height
    return y


# section renderers

def render_title(c, page_w, page_h):
    """Page header: version tag, title, subtitle, institutional label."""
    c.setFont("Helvetica", 7)
    c.drawString(PAGE_MARGIN, page_h - 0.25 * inch, "V1_2024")

    title_y = page_h - 0.5 * inch
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(page_w / 2, title_y, "MONROVIA PEDIATRIC TRIAGE")
    c.setFont("Helvetica", 8)
    c.drawCentredString(page_w / 2, title_y - 0.18 * inch,
                        "OPD Triage Record  |  Pediatric (under 5)")

    c.setFont("Helvetica-Bold", 8)
    c.drawRightString(page_w - PAGE_MARGIN, title_y - 0.05 * inch,
                      "MONROVIA HEALTH")
    c.setFont("Helvetica", 7)
    c.drawRightString(page_w - PAGE_MARGIN, title_y - 0.18 * inch,
                      "Synthetic record - not for clinical use")

    return title_y - 0.4 * inch


def render_header_block(c, gt, dec, top):
    """Top header: date, time, name, age, sex, opd, referral, complaint."""
    bottom = draw_section_box(c, top, 1.5 * inch, COLOR_WHITE)

    # Row 1: DATE | TIME | AM/PM
    y = top - 0.22 * inch
    draw_field(c, CONTENT_LEFT + 0.1 * inch, y, "DATE", gt["date"],
               0.45 * inch, 1.1 * inch)
    draw_field(c, CONTENT_LEFT + 2.1 * inch, y, "TIME", gt["time"],
               0.45 * inch, 0.7 * inch)
    ampm_x = CONTENT_LEFT + 3.7 * inch
    for i, lab in enumerate(["AM", "PM"]):
        draw_checkbox(c, ampm_x + i * 0.55 * inch, y - 0.01 * inch,
                      filled=(gt["ampm"] == lab), label=lab)

    # row 2: NAME | AGE | SEX
    y -= LINE_HEIGHT * 1.4
    draw_field(c, CONTENT_LEFT + 0.1 * inch, y, "NAME", gt["name"],
               0.5 * inch, 2.0 * inch)
    draw_field(c, CONTENT_LEFT + 3.0 * inch, y, "AGE", gt["age"],
               0.4 * inch, 1.0 * inch)
    sex_x = CONTENT_LEFT + 5.1 * inch
    c.setFont("Helvetica-Bold", 9)
    c.drawString(sex_x, y, "SEX:")
    for i, lab in enumerate(["M", "F"]):
        draw_checkbox(c, sex_x + 0.4 * inch + i * 0.5 * inch, y - 0.01 * inch,
                      filled=(gt["sex"] == lab), label=lab)

    # row 3: OPD CASE | REFERRAL
    y -= LINE_HEIGHT * 1.4
    c.setFont("Helvetica-Bold", 9)
    c.drawString(CONTENT_LEFT + 0.1 * inch, y, "OPD CASE:")
    draw_checkbox(c, CONTENT_LEFT + 1.0 * inch, y - 0.01 * inch,
                  filled=not dec.opd_yes, label="NO")
    draw_checkbox(c, CONTENT_LEFT + 1.55 * inch, y - 0.01 * inch,
                  filled=dec.opd_yes, label="YES")
    c.drawString(CONTENT_LEFT + 2.5 * inch, y, "REFERRAL:")
    draw_checkbox(c, CONTENT_LEFT + 3.4 * inch, y - 0.01 * inch,
                  filled=not dec.referral_yes, label="NO")
    draw_checkbox(c, CONTENT_LEFT + 3.95 * inch, y - 0.01 * inch,
                  filled=dec.referral_yes, label="YES, FROM:")

    # referral source line clipped to right edge of content area
    line_start = CONTENT_LEFT + 4.95 * inch
    line_end = CONTENT_RIGHT - 0.1 * inch
    c.setLineWidth(0.4)
    c.line(line_start, y - 0.025 * inch, line_end, y - 0.025 * inch)
    c.setFont("Helvetica", 9)
    c.drawString(line_start + 0.05 * inch, y, dec.referral_source)

    # row 4: PRESENTING COMPLAINT
    y -= LINE_HEIGHT * 1.4
    draw_field(c, CONTENT_LEFT + 0.1 * inch, y, "PRESENTING COMPLAINT",
               gt["presenting_complaint"], 1.7 * inch, 5.4 * inch)

    return bottom


def render_red_block(c, dec, top):
    """RED Emergency Signs panel."""
    height = 1.55 * inch
    bottom = draw_section_box(c, top, height, COLOR_RED)
    draw_section_title(c, top, "EMERGENCY SIGNS")

    draw_action_panel(c, top, height,
                      header_lines=[("RED", 12)],
                      body_lines=[
                          ("IMMEDIATELY", 7, True),
                          ("TRANSFER TO THE", 7, False),
                          ("RESUSCITATION AREA", 7, False),
                          ("", 7, False),
                          ("TREAT IMMEDIATELY", 7, True),
                          ("", 7, False),
                          ("WEIGH IF POSSIBLE", 7, False),
                      ])

    sy = top - 0.35 * inch
    sy = draw_checkbox_list(c, CONTENT_LEFT + 0.15 * inch, sy,
                            EMERGENCY_AB_SIGNS, dec.emergency,
                            row_height=0.16 * inch)
    sy -= 0.05 * inch
    draw_checkbox_list(c, CONTENT_LEFT + 0.15 * inch, sy,
                       EMERGENCY_D_SIGNS, dec.emergency)

    return bottom


def render_vitals_block(c, gt, top):
    """Vitals strip between RED and YELLOW."""
    bottom = draw_section_box(c, top, 0.65 * inch, COLOR_WHITE)

    vy = top - 0.18 * inch
    c.setFont("Helvetica-Bold", 9)
    c.drawString(CONTENT_LEFT + 0.1 * inch, vy, "VITAL SIGNS")
    draw_field(c, CONTENT_LEFT + 1.0 * inch, vy, "RR", gt["rr"],
               0.25 * inch, 0.6 * inch, label_size=8)
    draw_field(c, CONTENT_LEFT + 2.1 * inch, vy, "HR", gt["hr"],
               0.25 * inch, 0.6 * inch, label_size=8)
    draw_field(c, CONTENT_LEFT + 3.2 * inch, vy, "Sat", gt["sat"],
               0.3 * inch, 0.6 * inch, label_size=8)
    draw_field(c, CONTENT_LEFT + 4.3 * inch, vy, "T", gt["temperature"],
               0.18 * inch, 0.6 * inch, label_size=8)

    vy -= 0.28 * inch
    draw_field(c, CONTENT_LEFT + 0.1 * inch, vy, "WEIGHT (kg)", gt["weight"],
               0.95 * inch, 1.0 * inch, label_size=8)
    draw_field(c, CONTENT_LEFT + 3.0 * inch, vy, "MUAC (mm)", gt["muac"],
               0.85 * inch, 1.0 * inch, label_size=8)

    return bottom


def render_yellow_block(c, dec, top):
    """YELLOW Priority Signs panel."""
    height = 1.85 * inch
    bottom = draw_section_box(c, top, height, COLOR_YELLOW)
    draw_section_title(c, top, "PRIORITY SIGNS")

    draw_action_panel(c, top, height,
                      header_lines=[("YELLOW", 12)],
                      body_lines=[
                          ("PLACE IN PRIORITY IN", 7, True),
                          ("THE QUEUE", 7, True),
                          ("", 7, False),
                          ("RE-EVALUATE EVERY", 7, False),
                          ("20mins", 7, False),
                          ("", 7, False),
                          ("MAX. WAIT: 1hr", 7, False),
                      ])

    draw_checkbox_list(c, CONTENT_LEFT + 0.15 * inch, top - 0.35 * inch,
                       PRIORITY_SIGNS, dec.priority)

    return bottom


def render_green_block(c, dec, top):
    """GREEN Non-Urgent panel."""
    height = 0.65 * inch
    bottom = draw_section_box(c, top, height, COLOR_GREEN)
    draw_section_title(c, top, "NON-URGENT")

    draw_action_panel(c, top, height,
                      header_lines=[("GREEN", 9), ("BACK TO MOH", 8)],
                      body_lines=[
                          ("REEVALUATE EVERY 60mins", 6.5, False),
                          ("MAX WAIT: 4 hours", 6.5, False),
                      ])

    draw_checkbox(c, CONTENT_LEFT + 0.15 * inch, top - 0.42 * inch,
                  filled=dec.non_urgent,
                  label="ABSENCE OF EMERGENCY AND PRIORITY SIGNS",
                  label_size=7)

    return bottom


def render_triage_classification(c, gt, top):
    """The TRIAGE CLASSIFICATION line, the schema field."""
    bottom = top - 0.4 * inch

    y = top - 0.22 * inch
    c.setFont("Helvetica-Bold", 9)
    c.drawString(CONTENT_LEFT + 0.1 * inch, y, "TRIAGE CLASSIFICATION:")

    tri_x = CONTENT_LEFT + 1.9 * inch
    for i, color in enumerate(TRIAGE_COLORS):
        draw_checkbox(c, tri_x + i * 1.5 * inch, y - 0.01 * inch,
                      filled=(gt["triage_color"] == color),
                      label=color, label_size=9)

    return bottom


def render_test_and_treat(c, dec, top):
    """TEST AND TREAT panel, modeled on page 2 of the real MSF form."""
    bottom = draw_section_box(c, top, 1.2 * inch, COLOR_GREY)
    draw_section_title(c, top, "TEST AND TREAT")

    # row 1: Malaria RDT + ACT history + BSL
    y = top - 0.4 * inch
    c.setFont("Helvetica-Bold", 7)
    c.drawString(CONTENT_LEFT + 0.1 * inch, y, "Malaria RDT:")
    draw_checkbox(c, CONTENT_LEFT + 1.0 * inch, y - 0.01 * inch,
                  filled=dec.malaria_positive, label="Positive", label_size=7)
    draw_checkbox(c, CONTENT_LEFT + 1.85 * inch, y - 0.01 * inch,
                  filled=not dec.malaria_positive, label="Negative",
                  label_size=7)

    c.setFont("Helvetica-Bold", 7)
    c.drawString(CONTENT_LEFT + 2.85 * inch, y, "ACT (last 3 wks):")
    draw_checkbox(c, CONTENT_LEFT + 3.95 * inch, y - 0.01 * inch,
                  filled=dec.act_treated, label="Yes", label_size=7)
    draw_checkbox(c, CONTENT_LEFT + 4.55 * inch, y - 0.01 * inch,
                  filled=not dec.act_treated, label="No", label_size=7)

    draw_field(c, CONTENT_LEFT + 5.2 * inch, y, "BSL (mg/dL)", dec.bsl,
               0.85 * inch, 0.5 * inch, label_size=7)

    # row 2: Hemoglobin
    y -= 0.28 * inch
    draw_field(c, CONTENT_LEFT + 0.1 * inch, y, "Hemoglobin (g/dL)",
               dec.hemoglobin, 1.1 * inch, 0.7 * inch, label_size=7)

    # row 3: Treatments given
    y -= 0.3 * inch
    c.setFont("Helvetica-Bold", 7)
    c.drawString(CONTENT_LEFT + 0.1 * inch, y, "Treatment given:")
    draw_checkbox(c, CONTENT_LEFT + 1.2 * inch, y - 0.01 * inch,
                  filled=dec.gave_paracetamol,
                  label="PARACETAMOL 15mg/kg PO", label_size=7)
    draw_checkbox(c, CONTENT_LEFT + 3.4 * inch, y - 0.01 * inch,
                  filled=dec.gave_glucose,
                  label="ORAL G10% 10mL/kg PO", label_size=7)
    draw_checkbox(c, CONTENT_LEFT + 5.4 * inch, y - 0.01 * inch,
                  filled=dec.gave_ors, label="ORS 10mL/kg/loose stool",
                  label_size=7)

    return bottom


def render_observation(c, dec, top):
    """OBSERVATION table, TIME / PEWS / BSL / NURSE columns."""
    bottom = draw_section_box(c, top, 0.85 * inch, COLOR_WHITE)
    draw_section_title(c, top, "OBSERVATION")

    headers = ["TIME", "PEWS", "BSL", "NURSE"]
    col_x = [CONTENT_LEFT + 0.15 * inch + i * 1.35 * inch for i in range(4)]

    hy = top - 0.36 * inch
    c.setFont("Helvetica-Bold", 7)
    for x, h in zip(col_x, headers):
        c.drawString(x, hy, h)

    c.setLineWidth(0.4)
    c.line(CONTENT_LEFT + 0.1 * inch, hy - 0.04 * inch,
           CONTENT_RIGHT - 0.1 * inch, hy - 0.04 * inch)

    ry = hy - 0.18 * inch
    c.setFont("Helvetica", 8)
    for time_s, pews, bsl in dec.obs_rows:
        c.drawString(col_x[0], ry, time_s)
        c.drawString(col_x[1], ry, pews)
        c.drawString(col_x[2], ry, bsl)
        c.drawString(col_x[3], ry, "K. Yeaher")
        ry -= 0.15 * inch

    return bottom


def render_footer(c):
    """NURSE and SIGNATURE footer at the bottom of the page."""
    y = PAGE_MARGIN + 0.15 * inch
    c.setFont("Helvetica-Bold", 8)
    c.drawString(CONTENT_LEFT + 0.1 * inch, y, "NURSE:")
    c.setLineWidth(0.4)
    c.line(CONTENT_LEFT + 0.6 * inch, y - 0.025 * inch,
           CONTENT_LEFT + 3.5 * inch, y - 0.025 * inch)
    c.drawString(CONTENT_LEFT + 3.7 * inch, y, "SIGNATURE:")
    c.line(CONTENT_LEFT + 4.45 * inch, y - 0.025 * inch,
           CONTENT_RIGHT - 0.1 * inch, y - 0.025 * inch)


def render_triage_sidebar(c, top, bottom):
    """Vertical TRIAGE sidebar spanning the clinical assessment region."""
    height = top - bottom
    c.setStrokeColor(COLOR_BORDER)
    c.setLineWidth(0.6)
    c.rect(PAGE_MARGIN, bottom, SIDEBAR_WIDTH, height)
    draw_vertical_label(c, PAGE_MARGIN, top, "TRIAGE", height)


# top level rendering

def render_form_pdf(gt, dec, pdf_path):
    """Render one Monrovia Pediatric Triage form to PDF.

    The pipeline is intentionally readable as a vertical recipe:
    each render_* function returns the y-coordinate where it ended,
    which becomes the starting y-coordinate of the next section.
    """
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    page_w, page_h = letter

    top = render_title(c, page_w, page_h)
    top = render_header_block(c, gt, dec, top)

    # sidebar starts here and runs through the GREEN block
    sidebar_top = top

    top = render_red_block(c, dec, top - SECTION_GAP)
    top = render_vitals_block(c, gt, top - SECTION_GAP)
    top = render_yellow_block(c, dec, top - SECTION_GAP)
    top = render_green_block(c, dec, top - SECTION_GAP)
    top = render_triage_classification(c, gt, top - 0.1 * inch)

    sidebar_bottom = top + SECTION_GAP
    render_triage_sidebar(c, sidebar_top, sidebar_bottom)

    # decorative bottom sections fill what would otherwise be empty space
    top = render_test_and_treat(c, dec, top - 0.08 * inch)
    render_observation(c, dec, top - SECTION_GAP)

    render_footer(c)
    c.save()


# entry point

def main():
    """Generate N_FORMS synthetic PDFs and save ground truth to disk."""
    random.seed(RANDOM_SEED)

    ground_truth = {}
    for form_id in range(N_FORMS):
        gt = sample_form(form_id)
        dec = sample_decorative(gt["triage_color"])
        render_form_pdf(gt, dec, PDF_DIR / f"form_{form_id:03d}.pdf")
        ground_truth[str(form_id)] = gt

    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated {N_FORMS} forms in {PDF_DIR}")
    print(f"Ground truth: {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()