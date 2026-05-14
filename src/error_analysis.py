"""
error_analysis.py

Per field failure categorization at heavy noise for Kairo Health.

For each wrong field prediction at the heavy noise tier, this module assigns
one of four failure categories. It then writes a per method, per field breakdown
and a deployment guide showing which method performs better for each field.

Categories:
    missing          Method returned nothing, but gold had a value.
    substitution     Method returned a wrong value that appears in the OCR text.
                     This usually means the OCR text had the value but the
                     extractor picked the wrong version.
    hallucination    Method returned a wrong value that does not appear in the
                     OCR text. This is more likely to happen with the LLM.
    layout_artifact  Method returned nothing and the field label keyword is
                     absent from the OCR text. This is more likely to affect
                     the rules method because the regex needs a label anchor.

Run after evaluate.py:
    python -m src.error_analysis

Outputs written to results:
    error_categories.csv     One row per method, field, and category.
    deployment_guide.csv     One row per field with the recommended method.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.significance import (
    GROUND_TRUTH_PATH,
    PREDICTION_PATHS,
    RESULTS_DIR,
    normalize_value,
)

# label keywords used by the rules extractor to anchor each field
# keep this aligned with extract_rules.py
FIELD_LABEL_KEYWORDS = {
    "date": ["date"],
    "time": ["time"],
    "ampm": ["am", "pm"],
    "name": ["name"],
    "age": ["age"],
    "sex": ["sex", "m/f"],
    "presenting_complaint": ["complaint", "presenting"],
    "triage_color": ["red", "yellow", "green", "triage"],
    "rr": ["rr", "resp"],
    "hr": ["hr", "heart"],
    "sat": ["sat", "spo2"],
    "temperature": ["temp"],
    "weight": ["wt", "weight"],
    "muac": ["muac"],
}

OCR_RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "ocr_results.json"
TARGET_NOISES = ("heavy", "severe", "extreme")


def categorize_failure(prediction, gold, ocr_text: str, field: str) -> str | None:
    """
    Assign one failure category to a wrong field prediction.

    Returns None when the prediction is correct so callers can loop through
    every field and only keep actual failures.
    """
    pred = normalize_value(prediction)
    gold = normalize_value(gold)

    if pred == gold:
        return None

    ocr_lower = ocr_text.lower()
    label_keywords = FIELD_LABEL_KEYWORDS.get(field, [field])
    label_present = any(keyword in ocr_lower for keyword in label_keywords)

    if pred is None:
        # empty prediction with no visible label usually means the layout broke the rules extractor
        return "layout_artifact" if not label_present else "missing"

    # wrong value found in OCR is a substitution, otherwise treat it as a hallucination
    return "substitution" if pred in ocr_lower else "hallucination"


def analyze_failures(predictions_path: Path) -> pd.DataFrame:
    """
    Categorize every heavy noise failure for one extraction method.

    Returns a long form table with method, field, category, and count.
    """
    method = predictions_path.stem.replace("predictions_", "")

    gold_data = json.loads(GROUND_TRUTH_PATH.read_text())
    pred_data = json.loads(predictions_path.read_text())
    ocr_data = json.loads(OCR_RESULTS_PATH.read_text())

    rows = []

    for doc_id, pred_fields in pred_data.items():
        if not any(doc_id.endswith(f"_{t}") for t in TARGET_NOISES):
            continue

        form_id = doc_id.rsplit("_", 1)[0]
        gold_fields = gold_data.get(form_id, {})
        ocr_text = ocr_data.get(doc_id, {}).get("text", "")

        for field, gold_value in gold_fields.items():
            category = categorize_failure(
                pred_fields.get(field),
                gold_value,
                ocr_text,
                field,
            )

            if category is not None:
                tier = doc_id.rsplit("_", 1)[1]
                rows.append({
                    "method": method,
                    "field": field,
                    "category": category,
                    "noise": tier,
                })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return (
        df.groupby(["method", "noise", "field", "category"])
        .size()
        .reset_index(name="count")
        .sort_values(["method", "noise", "field", "category"])
    )


def build_deployment_guide(
    error_df: pd.DataFrame,
    gold_data: dict,
) -> pd.DataFrame:
    """
    Recommend the better method for each field at heavy noise.

    For each evaluation field, this counts total errors by method and picks
    the method with fewer errors. Ties are labeled as either.
    """
    all_fields = sorted({
        field
        for fields in gold_data.values()
        for field in fields
    })

    error_totals = (
        error_df.groupby(["method", "field"])["count"]
        .sum()
        .unstack("method", fill_value=0)
    )

    rows = []

    for field in all_fields:
        rules_errors = (
            int(error_totals.at[field, "rules"])
            if "rules" in error_totals.columns and field in error_totals.index
            else 0
        )
        llm_errors = (
            int(error_totals.at[field, "llm"])
            if "llm" in error_totals.columns and field in error_totals.index
            else 0
        )

        if rules_errors < llm_errors:
            winner = "rules"
        elif llm_errors < rules_errors:
            winner = "llm"
        else:
            winner = "either"

        rows.append({
            "field": field,
            "rules_errors": rules_errors,
            "llm_errors": llm_errors,
            "recommended_method": winner,
        })

    return pd.DataFrame(rows)


def main() -> None:
    """
    Run heavy noise error analysis for both methods and write CSV outputs.

    Missing prediction files are skipped so a partial run can still produce
    useful outputs for the files that exist.
    """
    error_frames = [
        analyze_failures(path)
        for path in PREDICTION_PATHS.values()
        if path.exists()
    ]

    error_df = (
        pd.concat(error_frames, ignore_index=True)
        if error_frames
        else pd.DataFrame()
    )

    if error_df.empty:
        print("No heavy noise failures found. Nothing to write.")
        return

    error_df.to_csv(RESULTS_DIR / "error_categories.csv", index=False)

    gold_data = json.loads(GROUND_TRUTH_PATH.read_text())
    guide_df = build_deployment_guide(error_df, gold_data)
    guide_df.to_csv(RESULTS_DIR / "deployment_guide.csv", index=False)

    print(f"Wrote error_categories.csv and deployment_guide.csv to {RESULTS_DIR}")


if __name__ == "__main__":
    main()