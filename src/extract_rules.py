"""
Rule based field extraction using regex patterns.

Patterns are tuned against clean OCR output and applied unchanged at
higher noise tiers. This is the realistic deployment story for
rule based systems: rules are written once, then encounter input
they weren't tuned for. Per the paper's experimental design, we do
not add fuzzy matching fallbacks, watching the regex baseline
degrade with noise is the comparison's headline finding.
"""

import json
import re

from config import FIELD_SCHEMA, OCR_RESULTS_PATH, RESULTS_DIR


# postprocessors

def _normalize_date(s):
    """Tesseract sometimes reads / as -. Normalize back to / for matching."""
    return s.strip().replace("-", "/")


def _upper_strip(s):
    return s.strip().upper()


def _strip(s):
    return s.strip()


def _clean_complaint(s):
    """Cut the complaint at the first newline followed by uppercase label.

    OCR reading order on a multicolumn form puts unrelated label text
    after the complaint, so we trim aggressively.
    """
    s = s.split("\n\n")[0]
    s = re.split(r"\n\s*(?:NO|SEX|TRIAGE|EMERGENCY|VITAL|REFERRAL|OPD)\b",
                 s, flags=re.IGNORECASE)[0]
    return s.strip().rstrip("_,. ")


# Field patterns
# Each entry: (field, pattern, group_strategy, postprocess).
# group_strategy: int = group index, "last" = last match, "first" = first match

FIELD_PATTERNS = [
    ("date",
     r"DATE[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", 1, _normalize_date),

    ("time",
     r"TIME[:\s]+(\d{1,2}:\d{2})", 1, _strip),

    # AM/PM: standalone word match. Brittle by design, when OCR reads
    ("ampm",
     r"\b(AM|PM)\b", 1, _upper_strip),

    ("name",
     r"NAME[:\s]+(\S+)", 1, _strip),

    ("age",
     r"AGE[:\s]+(\d+)\s*months?", 1, lambda s: f"{s.strip()} months"),

    # SEX: allow up to 80 characters between SEX: label and the M/F value, since OCR reading order may insert other content between them in multi column forms.
    ("sex",
     r"SEX[:\s][\s\S]{0,80}?\[?[Xx]\]?\s*([MF])\b", 1, _upper_strip),

    # presenting complaint: capture from label until end of paragraph. _clean_complaint trims OCR'd cross-column noise.
    ("presenting_complaint",
     r"PRESENTING COMPLAINT[:\s]+([^\n]*(?:\n[^\n]+){0,3})",
     1, _clean_complaint),

    # triage color: find the color adjacent to a filled checkbox X.
    ("triage_color",
     r"\[?[XxKk]\]?\s*(RED|YELLOW|GREEN)\b", 1, _upper_strip),

    # vitals: short labels = brittle. RR is unambiguous, but T must be carefully bounded so it doesn't match T in TRIAGE, TIME, etc.
    ("rr",
     r"\bRR[:\s]+(\d+)", 1, _strip),
    ("hr",
     r"\bHR[:\s]+(\d+)", 1, _strip),
    ("sat",
     r"\bSat[:\s]+(\d+)", 1, _strip),

    # temperature: requires a decimal point to distinguish "T: 39.1" from spurious matches like "T 99" elsewhere.
    ("temperature",
     r"\bT[:\s]+(\d+\.\d+)", 1, _strip),

    ("weight",
     r"WEIGHT(?:\s*\(kg\))?[:\s]+(\d+\.\d+)", 1, _strip),
    ("muac",
     r"MUAC(?:\s*\(mm\))?[:\s]+(\d+)", 1, _strip),
]


# extraction

def extract_field(text, pattern, group_strategy, postprocess):
    """Run one pattern against the OCR text. Returns the extracted value
    or None if no match.

    group_strategy of "last" returns the last match; an integer returns
    that group from the first match.
    """
    flags = re.IGNORECASE | re.DOTALL

    if group_strategy == "last":
        matches = re.findall(pattern, text, flags)
        if not matches:
            return None
        value = matches[-1]
    else:
        m = re.search(pattern, text, flags)
        if not m:
            return None
        value = m.group(group_strategy)

    return postprocess(value) if postprocess else value


def extract_all_fields(text):
    """Extract every field in FIELD_SCHEMA from one OCR text.

    Returns a dict with all 14 fields. Missing fields are None.
    """
    out = {field: None for field in FIELD_SCHEMA}
    for field, pattern, group_strategy, postprocess in FIELD_PATTERNS:
        out[field] = extract_field(text, pattern, group_strategy, postprocess)
    return out


# entry point

def main():
    """Extract fields from every OCR result and save predictions."""
    with open(OCR_RESULTS_PATH) as f:
        ocr_results = json.load(f)

    predictions = {}
    for key, data in ocr_results.items():
        predictions[key] = extract_all_fields(data["text"])

    out_path = RESULTS_DIR / "predictions_rules.json"
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Rule-based predictions written to {out_path}")
    print(f"Predicted {len(predictions)} documents "
          f"x {len(FIELD_SCHEMA)} fields each")


if __name__ == "__main__":
    main()
