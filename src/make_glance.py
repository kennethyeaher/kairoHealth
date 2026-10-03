"""
Build the README at a glance strip from data already produced by the pipeline.

Every number on the strip is read from results/ or computed from the committed
prediction files with the same helpers make_figures.py uses, so the strip can
never drift from the tables in the README. Text uses a system font stack and
the strip carries its own background, so it reads in light and dark themes.

Run after evaluate.py and error_analysis.py:
    python -m src.make_glance

Output written to docs/assets:
    at_a_glance.svg
"""

from __future__ import annotations

import csv
from pathlib import Path

from config import FIELD_SCHEMA, GROUND_TRUTH_PATH, NOISE_LEVELS, RESULTS_DIR
from src.make_figures import (ASSET_DIR, classify_triage_cell, load_csv_by_noise,
                              load_json, triage_confusion)
from src.make_figures import COLOR_LLM

# navy surface and ink taken from the pipeline diagram palette
COLOR_BACKGROUND = "#10141b"
COLOR_CARD = "#1a2130"
COLOR_BORDER = "#2c3546"
COLOR_RULE = "#1c5cab"
COLOR_VALUE = "#ffffff"
COLOR_LABEL = "#e4edf9"
COLOR_NOTE = "#9aa3b5"
FONT_STACK = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"

GLANCE_TIER = "heavy"
CARD_WIDTH = 282
CARD_HEIGHT = 150
CARD_GAP = 12
PADDING = 12


def hallucination_counts():
    """Sum LLM hallucinations per tier from error_categories.csv.

    Returns
    counts
        Dict mapping noise tier to the number of hallucinated LLM values.
    """
    counts = {tier: 0 for tier in NOISE_LEVELS}
    with open(RESULTS_DIR / "error_categories.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "llm" and row["category"] == "hallucination":
                counts[row["noise"]] += int(row["count"])
    return counts


def build_cards():
    """Compute the four headline results the strip shows.

    Returns
    cards
        List of (value, label, note) tuples, one per card.
    """
    comparison = load_csv_by_noise("f1_comparison.csv")[GLANCE_TIER]
    mcnemar = load_csv_by_noise("mcnemar.csv")[GLANCE_TIER]

    ground_truth = load_json(GROUND_TRUTH_PATH)
    llm_predictions = load_json(RESULTS_DIR / "predictions_llm.json")

    hallucinations = hallucination_counts()
    attempts = len(llm_predictions) * len(FIELD_SCHEMA)
    first_tier = next(tier for tier in NOISE_LEVELS if hallucinations[tier])

    triage = triage_confusion(llm_predictions, ground_truth, GLANCE_TIER)
    triage_total = sum(triage.values())
    triage_correct = sum(n for (gold, predicted), n in triage.items() if gold == predicted)
    red_under = sum(n for (gold, predicted), n in triage.items()
                    if gold == "RED" and classify_triage_cell(gold, predicted) == "under")

    return [
        (f"{float(comparison['llm_recall']):.2f} vs {float(comparison['rules_recall']):.2f}",
         f"Recall at {GLANCE_TIER} noise",
         "LLM against regex"),
        (f"{mcnemar['llm_only_correct']} of {mcnemar['n_disagree']}",
         "Disagreements the LLM won",
         f"{GLANCE_TIER.capitalize()} noise, McNemar p = {float(mcnemar['p_value']):.1e}"),
        (f"{sum(hallucinations.values())} of {attempts:,}",
         "Values the LLM invented",
         f"None below {first_tier} noise"),
        (f"{triage_correct} of {triage_total}",
         "Triage colors correct",
         f"{GLANCE_TIER.capitalize()} noise, {red_under} RED under triaged"),
    ]


def escape(text):
    """Escape the characters SVG text cannot hold literally."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_svg(cards):
    """Lay the cards out in one row on a navy background.

    Parameters
    cards
        List of (value, label, note) tuples from build_cards.

    Returns
    svg
        The complete SVG document as a string.
    """
    width = PADDING * 2 + CARD_WIDTH * len(cards) + CARD_GAP * (len(cards) - 1)
    height = PADDING * 2 + CARD_HEIGHT
    summary = "; ".join(f"{value}: {label}. {note}" for value, label, note in cards)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        '  <title id="title">Kairo Health results at a glance</title>',
        f'  <desc id="description">{escape(summary)}</desc>',
        f'  <rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="16" '
        f'fill="{COLOR_BACKGROUND}" stroke="{COLOR_BORDER}"/>',
    ]
    for index, (value, label, note) in enumerate(cards):
        x = PADDING + index * (CARD_WIDTH + CARD_GAP)
        accent = COLOR_LLM if index < len(cards) - 1 else COLOR_RULE
        parts += [
            f'  <rect x="{x}" y="{PADDING}" width="{CARD_WIDTH}" height="{CARD_HEIGHT}" rx="10" fill="{COLOR_CARD}"/>',
            f'  <rect x="{x}" y="{PADDING}" width="4" height="{CARD_HEIGHT}" rx="2" fill="{accent}"/>',
            f'  <text x="{x + 24}" y="{PADDING + 58}" fill="{COLOR_VALUE}" font-family="{FONT_STACK}" '
            f'font-size="36" font-weight="700">{escape(value)}</text>',
            f'  <text x="{x + 24}" y="{PADDING + 96}" fill="{COLOR_LABEL}" font-family="{FONT_STACK}" '
            f'font-size="17" font-weight="600">{escape(label)}</text>',
            f'  <text x="{x + 24}" y="{PADDING + 126}" fill="{COLOR_NOTE}" font-family="{FONT_STACK}" '
            f'font-size="15">{escape(note)}</text>',
        ]
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    """Write the strip and print the values it carries."""
    cards = build_cards()
    output_path = Path(ASSET_DIR) / "at_a_glance.svg"
    output_path.write_text(render_svg(cards))
    for value, label, note in cards:
        print(f"{value:>14}  {label}  ({note})")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
