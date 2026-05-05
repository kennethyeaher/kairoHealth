"""
Run Tesseract OCR on each rendered image and record per document
character and word error rates against the known source text.

CER/WER are computed against a reconstruction of the *intended* readable
text on the form, what a perfect OCR pass would return given the form
layout. This calibrates the three noise tiers as a controlled variable
for the downstream extraction comparison; it is NOT the field level
extraction metric (that lives in evaluate.py).

The reference text combines all visible static labels on the form
(section titles, checkbox labels, action panels, table headers) with
the per form ground truth field values.
"""

import json

import pandas as pd
import pytesseract
from jiwer import cer, wer
from PIL import Image, ImageEnhance

from config import (GROUND_TRUTH_PATH, IMAGE_DIR, N_FORMS, NOISE_LEVELS,
                    OCR_RESULTS_PATH, RESULTS_DIR)


# source text reconstruction

# static labels appear on every form regardless of field values.
# order roughly matches top-to-bottom layout, but jiwer.cer is order insensitive at the character level, so this is just for readability.
STATIC_LABELS = (
    # title bar
    "V1_2024 MONROVIA PEDIATRIC TRIAGE OPD Triage Record Pediatric "
    "under 5 MONROVIA HEALTH Synthetic record not for clinical use "
    # header block
    "DATE TIME AM PM NAME AGE SEX M F "
    "OPD CASE NO YES REFERRAL NO YES FROM "
    "PRESENTING COMPLAINT "
    # vertical sidebar
    "TRIAGE "
    # RED Emergency signs section
    "EMERGENCY SIGNS "
    "NOT BREATHING CENTRAL CYANOSIS SEVERE RESPIRATORY DISTRESS "
    "COMA AVPU P or U CONVULSIONS SEVERE TRAUMA SEVERE BURNS "
    "CONTAGIOUS DISEASES ISOLATE "
    "RED IMMEDIATELY TRANSFER TO THE RESUSCITATION AREA "
    "TREAT IMMEDIATELY WEIGH IF POSSIBLE "
    # vitals strip
    "VITAL SIGNS RR HR Sat T WEIGHT kg MUAC mm "
    # YELLOW Priority signs section
    "PRIORITY SIGNS "
    "TINY BABY 2 months TEMPERATURE 36C OR 39C "
    "TRAUMA SURG EMERGENCY non-severe "
    "POISONING snake scorpion bite PAIN severe PALLOR severe "
    "RESP DISTRESS non-severe RESTLESSNESS IRRITABILITY LETHARGY "
    "REFERRAL FROM ANOTHER CENTRE BURNS non-severe "
    "YELLOW PLACE IN PRIORITY IN THE QUEUE "
    "RE-EVALUATE EVERY 20mins MAX WAIT 1hr "
    # GREEN non-urgent section
    "NON-URGENT ABSENCE OF EMERGENCY AND PRIORITY SIGNS "
    "GREEN BACK TO MOH REEVALUATE EVERY 60mins MAX WAIT 4 hours "
    # triage classification
    "TRIAGE CLASSIFICATION RED YELLOW GREEN "
    # test and treat section
    "TEST AND TREAT Malaria RDT Positive Negative "
    "ACT last 3 wks Yes No BSL mg dL "
    "Hemoglobin g dL Treatment given "
    "PARACETAMOL 15mg kg PO ORAL G10 10mL kg PO ORS 10mL kg loose stool "
    # observation table
    "OBSERVATION TIME PEWS BSL NURSE "
    # footer
    "NURSE SIGNATURE"
)


def reconstruct_source_text(gt):
    """Build the expected readable text for a filled form.

    Combines static layout labels (constant across forms) with the 14
    ground truth field values for this specific form. This is the
    reference string CER/WER are computed against.
    """
    field_text = (
        f"{gt['date']} {gt['time']} {gt['ampm']} "
        f"{gt['name']} {gt['age']} {gt['sex']} "
        f"{gt['presenting_complaint']} "
        f"{gt['rr']} {gt['hr']} {gt['sat']} {gt['temperature']} "
        f"{gt['weight']} {gt['muac']} "
        f"{gt['triage_color']}"
    )
    return f"{STATIC_LABELS} {field_text}"


# OCR

def preprocess_for_ocr(img):
    """Grayscale + mild contrast bump.

    Matches the chain Hsu et al. (2022) report as effective for clinical
    OCR. Deliberately light, heavier preprocessing would mask the noise
    tier differences we're trying to measure.
    """
    img = img.convert("L")
    return ImageEnhance.Contrast(img).enhance(1.2)


def ocr_image(image_path):
    """Run Tesseract on one image, return raw extracted text."""
    img = Image.open(image_path)
    return pytesseract.image_to_string(preprocess_for_ocr(img))


# entry point

def main():
    """OCR every image and write results + calibration summary."""
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    ocr_results = {}
    error_rate_rows = []

    for form_id in range(N_FORMS):
        gt = ground_truth[str(form_id)]
        source_text = reconstruct_source_text(gt)

        for level in NOISE_LEVELS:
            image_path = IMAGE_DIR / level / f"form_{form_id:03d}.png"
            extracted_text = ocr_image(image_path)

            # CER and WER measure how far OCR output is from the
            # intended readable text, this calibrates the noise tier, not the downstream extractors.
            char_err = cer(source_text, extracted_text)
            word_err = wer(source_text, extracted_text)

            ocr_results[f"{form_id}_{level}"] = {
                "form_id": form_id,
                "noise": level,
                "text": extracted_text,
                "cer": char_err,
                "wer": word_err,
            }
            error_rate_rows.append({
                "form_id": form_id,
                "noise": level,
                "cer": char_err,
                "wer": word_err,
            })

    # save raw OCR text for downstream extractors
    with open(OCR_RESULTS_PATH, "w") as f:
        json.dump(ocr_results, f, indent=2)

    # calibration summary: mean CER/WER per tier
    df = pd.DataFrame(error_rate_rows)
    summary = df.groupby("noise")[["cer", "wer"]].mean().round(3)
    summary = summary.reindex(NOISE_LEVELS)
    summary.to_csv(RESULTS_DIR / "ocr_calibration.csv")

    print("OCR error rates by noise level (mean across all forms):")
    print(summary)
    print(f"\nFull OCR output: {OCR_RESULTS_PATH}")
    print(f"Calibration table: {RESULTS_DIR / 'ocr_calibration.csv'}")


if __name__ == "__main__":
    main()