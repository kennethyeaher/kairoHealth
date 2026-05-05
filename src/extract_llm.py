"""
LLM based field extraction using zero shot JSON mode prompting.

A structured prompt asks Claude to read OCR text and emit a JSON
object matching FIELD_SCHEMA. Hallucinated keys (fields not in the
schema) are dropped during postprocessing. Output is written to
results/predictions_llm.json with the same per document keys as the
rule based predictions, so the evaluator can compare them directly.
"""

import json
import os
import re
import time

from anthropic import Anthropic
from dotenv import load_dotenv

from config import FIELD_SCHEMA, OCR_RESULTS_PATH, RESULTS_DIR


# API setup

load_dotenv()  # pulls my ANTHROPIC_API_KEY from .env

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# model name, swapping models is a easy one line change.
MODEL = "claude-sonnet-4-5"

# brief delay between calls so I don't hammer the API. 30 calls * 0.3s
API_DELAY_SECONDS = 0.3


# Prompt

# Zero shot extraction prompt. Design choices:
#   - Schema is explicit with format hints, so Claude knows the expected
#     shape rather than inventing keys.
#   - I instruct null for missing fields so the JSON is always valid
#     against the schema (the evaluator handles None gracefully).
#   - "Return ONLY the JSON object" prevents prose wrapping.
#   - I deliberately do NOT ask Claude to correct OCR errors, the
#     experiment is about how the LLM handles raw OCR, not how well
#     it can repair the input.
PROMPT_TEMPLATE = """Extracting structured data from an OCR processed pediatric triage form. The OCR text may contain character errors, smudged or misread checkboxes, and out-of-order text from the form's multi-column layout.

Return a JSON object with exactly these fields, using null for any field you cannot find with reasonable confidence:

{{
  "date": "DD/MM/YYYY format",
  "time": "HH:MM 24hour format",
  "ampm": "AM or PM",
  "name": "patient name",
  "age": "age with unit (e.g. '24 months')",
  "sex": "M or F",
  "presenting_complaint": "free text describing chief complaint",
  "triage_color": "RED, YELLOW, or GREEN, whichever is marked with X",
  "rr": "respiratory rate as integer string",
  "hr": "heart rate as integer string",
  "sat": "oxygen saturation as integer string",
  "temperature": "temperature as decimal string (such as '38.5')",
  "weight": "weight in kg as decimal string",
  "muac": "MUAC in mm as integer string"
}}

Return ONLY the JSON object, no other text, no markdown fencing.

OCR TEXT:
{ocr_text}
"""


# extraction

def extract_llm(ocr_text):
    """Call Claude to extract fields from one OCR text.

    Returns
    -------
    dict
        14 fields matching FIELD_SCHEMA. On API or parse failure,
        every field is None, this counts as a total miss in evaluation.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[
            {"role": "user",
             "content": PROMPT_TEMPLATE.format(ocr_text=ocr_text)}
        ],
    )
    raw = response.content[0].text.strip()

    # strip markdown code fences if Claude added them despite the "no fencing" instruction, observed occasionally in practice.
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # total parse failure counts as a fully null prediction.
        return {field: None for field in FIELD_SCHEMA}

    # drop hallucinated keys, normalize None for missing schema keys.
    return {field: parsed.get(field) for field in FIELD_SCHEMA}


# entry point

def main():
    """Extract fields from every OCR result via Claude and save predictions."""
    with open(OCR_RESULTS_PATH) as f:
        ocr_results = json.load(f)

    predictions = {}
    total = len(ocr_results)

    for i, (key, data) in enumerate(ocr_results.items(), start=1):
        print(f"[{i}/{total}] Extracting {key}...")
        predictions[key] = extract_llm(data["text"])
        time.sleep(API_DELAY_SECONDS)

    out_path = RESULTS_DIR / "predictions_llm.json"
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nLLM predictions written to {out_path}")
    print(f"Predicted {len(predictions)} documents "
          f"x {len(FIELD_SCHEMA)} fields each")


if __name__ == "__main__":
    main()
