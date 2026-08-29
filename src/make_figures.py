"""
Build the portfolio figure set from data already produced by the pipeline.

This module reads the tables in results/ and the prediction files, then writes
four figures to docs/assets/. It does not re run any pipeline stage, call the
Anthropic API, or recompute anything that evaluate.py already publishes. Recall
values and significance flags are read from the CSVs so the figures can never
drift from the numbers in the README.

Run after evaluate.py and significance.py:
    python -m src.make_figures

Outputs written to docs/assets:
    recall_by_noise.png         Recall per method per tier, significance marked
    triage_safety_matrix.png    Gold against predicted triage color, two tiers
    field_errors_heavy.png      Errors per field at heavy noise, both methods
    noise_ladder.png            One form header rendered at all five tiers

The noise ladder needs data/images/, which is gitignored because it is
regeneratable. If those images are absent the ladder is skipped and the other
three figures are still written.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

from config import (FIELD_SCHEMA, GROUND_TRUTH_PATH, IMAGE_DIR, NOISE_LEVELS,
                    RESULTS_DIR)
from src.significance import normalize_value


# output location and figure settings

ASSET_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"
FIGURE_DPI = 200

# Shared palette. Rules and LLM keep the same two colors in every figure so the
# reader learns the mapping once. The triage colors are reserved for triage
# outcomes and are never used to distinguish methods.
COLOR_SURFACE = "#fcfcfb"
COLOR_INK = "#0b0b0b"
COLOR_INK_SOFT = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_RULES = "#888888"
COLOR_LLM = "#3b82f6"

FILL_CORRECT = "#e8f4ea"
FILL_OVER_TRIAGE = "#fff6e0"
FILL_UNDER_TRIAGE = "#fde9e9"
FILL_NOT_CLASSIFIED = "#f0efec"
TEXT_CORRECT = "#1d7a41"
TEXT_ALERT = "#c0392f"

# Triage classes ordered by clinical urgency. The rank decides whether a wrong
# prediction was over triage or under triage.
TRIAGE_URGENCY = {"RED": 3, "YELLOW": 2, "GREEN": 1}
NOT_CLASSIFIED = "not classified"
TRIAGE_COLUMNS = ["RED", "YELLOW", "GREEN", NOT_CLASSIFIED]

# Tiers shown in the triage matrix figure. Clean and moderate are near perfect
# and extreme is almost entirely blank, so neither is informative here.
TRIAGE_FIGURE_TIERS = ["heavy", "severe"]
FIELD_FIGURE_TIER = "heavy"

# Crop box on the rendered form, in pixels at 200 DPI. Covers the header block
# holding date, time, name, age and sex.
LADDER_CROP_BOX = (150, 180, 1650, 300)
LADDER_PANEL_WIDTH = 980
LADDER_PADDING = 14
LADDER_LABEL_HEIGHT = 34
LADDER_FORM_ID = 0

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.facecolor": COLOR_SURFACE,
    "axes.facecolor": COLOR_SURFACE,
    "savefig.facecolor": COLOR_SURFACE,
    "text.color": COLOR_INK,
    "axes.labelcolor": COLOR_INK_SOFT,
    "xtick.color": COLOR_MUTED,
    "ytick.color": COLOR_MUTED,
})


# loading

def load_json(path):
    """Read one JSON file and return the parsed object.

    Parameters
    path
        Path to a JSON file written by an earlier pipeline stage.

    Returns
    parsed
        The decoded JSON contents.

    Raises
        FileNotFoundError when the file is missing, with the path in the
        message so the caller knows which stage has not been run.
    """
    try:
        with open(path) as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{path} not found. Run the earlier pipeline stages first."
        ) from None


def load_csv_by_noise(filename):
    """Read a results CSV that has one row per noise tier.

    Parameters
    filename
        File name inside the results directory, such as f1_comparison.csv.

    Returns
    rows_by_noise
        Dict mapping noise tier name to that tier's row as a dict of strings.
    """
    path = RESULTS_DIR / filename
    try:
        with open(path, newline="") as handle:
            return {row["noise"]: row for row in csv.DictReader(handle)}
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{path} not found. Run evaluate.py and significance.py first."
        ) from None


def parse_noise_id(document_id):
    """Split a prediction key such as '12_heavy' into its two parts.

    Parameters
    document_id
        Prediction dict key in the form {form_id}_{noise_level}.

    Returns
    form_id, noise_level
        The form identifier as a string and the noise tier name.
    """
    form_id, noise_level = document_id.rsplit("_", 1)
    return form_id, noise_level


# scoring helpers

def count_field_errors(predictions, ground_truth, noise_level):
    """Count wrong predictions per field for one method at one noise tier.

    A field counts as an error when the normalized prediction does not equal
    the normalized gold value. Missing predictions and wrong values are both
    errors here, because this figure answers which fields need a better
    extractor, not how each field failed.

    Parameters
    predictions
        Predictions dict keyed by {form_id}_{noise_level}.
    ground_truth
        Ground truth dict keyed by form_id.
    noise_level
        Tier to count, such as 'heavy'.

    Returns
    error_counts
        Dict mapping field name to the number of forms that field got wrong.
    """
    error_counts = {field: 0 for field in FIELD_SCHEMA}

    for document_id, predicted_fields in predictions.items():
        form_id, tier = parse_noise_id(document_id)
        if tier != noise_level:
            continue

        gold_fields = ground_truth[form_id]
        for field in FIELD_SCHEMA:
            predicted = normalize_value(predicted_fields.get(field))
            gold = normalize_value(gold_fields.get(field))
            if predicted != gold:
                error_counts[field] += 1

    return error_counts


def triage_confusion(predictions, ground_truth, noise_level):
    """Build the gold against predicted table for the triage color field.

    Parameters
    predictions
        Predictions dict keyed by {form_id}_{noise_level}.
    ground_truth
        Ground truth dict keyed by form_id.
    noise_level
        Tier to tabulate, such as 'heavy'.

    Returns
    counts
        Dict mapping (gold_class, predicted_class) to a count. A prediction of
        None is recorded under the not classified column.
    """
    counts = {}

    for document_id, predicted_fields in predictions.items():
        form_id, tier = parse_noise_id(document_id)
        if tier != noise_level:
            continue

        gold = ground_truth[form_id].get("triage_color")
        gold = str(gold).upper() if gold else None
        if gold not in TRIAGE_URGENCY:
            continue

        predicted = normalize_value(predicted_fields.get("triage_color"))
        predicted = predicted.upper() if predicted else NOT_CLASSIFIED
        if predicted not in TRIAGE_COLUMNS:
            predicted = NOT_CLASSIFIED

        counts[(gold, predicted)] = counts.get((gold, predicted), 0) + 1

    return counts


def classify_triage_cell(gold, predicted):
    """Label one confusion cell by its clinical consequence.

    Parameters
    gold
        Triage class the clinician recorded.
    predicted
        Triage class the model returned, or the not classified marker.

    Returns
    label
        One of 'correct', 'over', 'under', or 'not_classified'. Leaving a RED
        or YELLOW child unclassified is treated as an under triage failure
        because a null triage color propagates downstream as default routing.
    """
    if predicted == NOT_CLASSIFIED:
        return "under" if gold in ("RED", "YELLOW") else "not_classified"
    if predicted == gold:
        return "correct"
    if TRIAGE_URGENCY[predicted] < TRIAGE_URGENCY[gold]:
        return "under"
    return "over"


# shared plotting helpers

def style_axes(axis, show_y_grid=True):
    """Apply the shared axis treatment: no top or right spine, faint grid.

    Parameters
    axis
        Matplotlib axes to restyle in place.
    show_y_grid
        Whether to draw horizontal gridlines behind the marks.

    Returns
    None
    """
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.spines["left"].set_color(COLOR_AXIS)
    axis.spines["bottom"].set_color(COLOR_AXIS)

    if show_y_grid:
        axis.grid(axis="y", color=COLOR_GRID, linewidth=1)
        axis.set_axisbelow(True)


def save_figure(figure, filename):
    """Write one figure to the asset directory and close it.

    Parameters
    figure
        Matplotlib figure to save.
    filename
        File name to write inside docs/assets.

    Returns
    output_path
        Path the figure was written to.
    """
    output_path = ASSET_DIR / filename
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)
    return output_path


# figures

def plot_recall_by_noise(comparison_rows, difference_rows):
    """Draw recall per method per tier, marking the significant tiers.

    Recall is the clearest view of the headline result because both methods
    stay precise. What separates them is how often they answer at all.

    Parameters
    comparison_rows
        Rows of f1_comparison.csv keyed by noise tier.
    difference_rows
        Rows of bootstrap_diff.csv keyed by noise tier, used only for the
        significance flag.

    Returns
    output_path
        Path the figure was written to.
    """
    rules_recall = [float(comparison_rows[t]["rules_recall"]) for t in NOISE_LEVELS]
    llm_recall = [float(comparison_rows[t]["llm_recall"]) for t in NOISE_LEVELS]
    is_significant = [
        difference_rows[t]["significant_at_95"].strip().lower() == "true"
        for t in NOISE_LEVELS
    ]

    figure, axis = plt.subplots(figsize=(9, 4.6))
    positions = np.arange(len(NOISE_LEVELS))
    bar_width = 0.36

    axis.bar(positions - bar_width / 2, rules_recall, bar_width,
             color=COLOR_RULES, zorder=3, label="Rule based (regex)")
    axis.bar(positions + bar_width / 2, llm_recall, bar_width,
             color=COLOR_LLM, zorder=3, label="LLM (zero shot)")

    # Value labels sit above each bar so the reader never has to read the axis.
    for index, (rules_value, llm_value) in enumerate(zip(rules_recall, llm_recall)):
        axis.text(index - bar_width / 2, rules_value + 0.018, f"{rules_value:.2f}",
                  ha="center", fontsize=10, color=COLOR_INK_SOFT)
        axis.text(index + bar_width / 2, llm_value + 0.018, f"{llm_value:.2f}",
                  ha="center", fontsize=10, color=COLOR_INK_SOFT)

        if is_significant[index]:
            axis.text(index, max(rules_value, llm_value) + 0.075,
                      "significant\ndifference", ha="center", va="bottom",
                      fontsize=8.5, color=TEXT_ALERT, style="italic",
                      linespacing=1.25)

    axis.set_xticks(positions)
    axis.set_xticklabels([tier.capitalize() for tier in NOISE_LEVELS])
    axis.set_ylim(0, 1.22)
    axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_ylabel("Recall (fields correctly extracted)")
    axis.set_title("Recall holds for the LLM through heavy noise; regex falls away",
                   fontsize=13, fontweight="bold", color=COLOR_INK,
                   loc="left", pad=34)
    style_axes(axis)
    axis.legend(frameon=False, ncol=2, fontsize=10,
                loc="lower left", bbox_to_anchor=(0, 1.0))

    figure.tight_layout()
    return save_figure(figure, "recall_by_noise.png")


def plot_triage_confusion(predictions, ground_truth):
    """Draw the triage confusion matrix for each tier in TRIAGE_FIGURE_TIERS.

    Cell color carries the clinical consequence rather than the count, because
    for this field the direction of an error matters more than its size.

    Parameters
    predictions
        LLM predictions dict keyed by {form_id}_{noise_level}.
    ground_truth
        Ground truth dict keyed by form_id.

    Returns
    output_path
        Path the figure was written to.
    """
    gold_classes = list(TRIAGE_URGENCY)
    cell_fills = {
        "correct": FILL_CORRECT,
        "over": FILL_OVER_TRIAGE,
        "under": FILL_UNDER_TRIAGE,
        "not_classified": FILL_NOT_CLASSIFIED,
    }

    figure, axes = plt.subplots(1, len(TRIAGE_FIGURE_TIERS), figsize=(11, 4.2))

    for axis, noise_level in zip(np.atleast_1d(axes), TRIAGE_FIGURE_TIERS):
        counts = triage_confusion(predictions, ground_truth, noise_level)

        for row, gold in enumerate(gold_classes):
            for column, predicted in enumerate(TRIAGE_COLUMNS):
                count = counts.get((gold, predicted), 0)
                fill = (COLOR_SURFACE if count == 0
                        else cell_fills[classify_triage_cell(gold, predicted)])

                axis.add_patch(Rectangle((column - 0.5, row - 0.5), 1, 1,
                                         facecolor=fill, edgecolor=COLOR_SURFACE,
                                         linewidth=3, zorder=1))
                if count:
                    axis.text(column, row, str(count), ha="center", va="center",
                              fontsize=15, fontweight="bold", color=COLOR_INK,
                              zorder=3)

        axis.set_xticks(range(len(TRIAGE_COLUMNS)))
        axis.set_xticklabels(TRIAGE_COLUMNS, fontsize=9.5)
        axis.set_yticks(range(len(gold_classes)))
        axis.set_yticklabels(gold_classes, fontsize=9.5)
        axis.set_xlabel("LLM predicted", color=COLOR_INK_SOFT, fontsize=10)
        axis.set_ylabel("Clinician (gold)", color=COLOR_INK_SOFT, fontsize=10)
        axis.set_title(f"{noise_level.capitalize()} noise", fontsize=12,
                       fontweight="bold", color=COLOR_INK, loc="left")
        axis.set_xlim(-0.5, len(TRIAGE_COLUMNS) - 0.5)
        axis.set_ylim(len(gold_classes) - 0.5, -0.5)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)

    figure.suptitle(
        "Triage classification: where the model is wrong, and in which direction",
        fontsize=13.5, fontweight="bold", color=COLOR_INK, x=0.012, ha="left", y=0.99)
    figure.text(
        0.012, 0.015,
        "Green, correct   ·   Amber, over triage in the safe direction   ·   "
        "Red, under triage or an urgent child left unclassified   ·   "
        "Grey, a non urgent child left unclassified",
        fontsize=9.5, color=COLOR_INK_SOFT)

    figure.tight_layout(rect=[0, 0.045, 1, 0.94])
    return save_figure(figure, "triage_safety_matrix.png")


def plot_field_errors(rules_errors, llm_errors, form_count):
    """Draw errors per field for both methods, sorted by the LLM advantage.

    Fields where both methods reach zero errors are called out in place rather
    than drawn as empty bars, since an absent bar reads as missing data.

    Parameters
    rules_errors
        Field to error count mapping for the rule based method.
    llm_errors
        Field to error count mapping for the LLM.
    form_count
        Number of forms scored, used to label the axis.

    Returns
    output_path
        Path the figure was written to.
    """
    ordered_fields = sorted(
        FIELD_SCHEMA,
        key=lambda field: rules_errors[field] - llm_errors[field],
        reverse=True,
    )

    figure, axis = plt.subplots(figsize=(9, 5.6))
    positions = np.arange(len(ordered_fields))
    bar_height = 0.36

    axis.barh(positions + bar_height / 2,
              [rules_errors[field] for field in ordered_fields],
              bar_height, color=COLOR_RULES, zorder=3, label="Rule based errors")
    axis.barh(positions - bar_height / 2,
              [llm_errors[field] for field in ordered_fields],
              bar_height, color=COLOR_LLM, zorder=3, label="LLM errors")

    for index, field in enumerate(ordered_fields):
        rules_count = rules_errors[field]
        llm_count = llm_errors[field]

        if rules_count == 0 and llm_count == 0:
            axis.text(0.5, index, "both perfect, regex is free here",
                      va="center", fontsize=9.5, color=TEXT_CORRECT, style="italic")
        elif rules_count == llm_count:
            # One shared label, so equal bars do not print two colliding numbers.
            axis.text(rules_count + 0.5, index, f"{rules_count} / {llm_count}",
                      va="center", fontsize=9.5, color=COLOR_INK_SOFT)
        else:
            axis.text(rules_count + 0.5, index + bar_height / 2, str(rules_count),
                      va="center", fontsize=9.5, color=COLOR_INK_SOFT)
            axis.text(llm_count + 0.5, index - bar_height / 2, str(llm_count),
                      va="center", fontsize=9.5, color=COLOR_INK_SOFT)

    axis.set_yticks(positions)
    axis.set_yticklabels(ordered_fields, fontsize=10)
    axis.invert_yaxis()
    axis.set_xlabel(f"Errors out of {form_count} forms at "
                    f"{FIELD_FIGURE_TIER} noise")
    axis.set_title("Which fields actually need an LLM", fontsize=13,
                   fontweight="bold", color=COLOR_INK, loc="left", pad=12)
    style_axes(axis, show_y_grid=False)
    axis.grid(axis="x", color=COLOR_GRID, linewidth=1)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower right", fontsize=10)

    figure.tight_layout()
    return save_figure(figure, "field_errors_heavy.png")


def load_ladder_font(size, bold=False):
    """Return a DejaVu font at the requested size, or the PIL default.

    Parameters
    size
        Point size to load.
    bold
        Whether to load the bold face.

    Returns
    font
        A PIL font object. Falls back to the bitmap default when DejaVu is not
        installed, which changes the label size but not the figure content.
    """
    face = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{face}", size)
    except OSError:
        return ImageFont.load_default()


def build_noise_ladder(calibration_rows):
    """Stack one form header rendered at every tier into a single image.

    Parameters
    calibration_rows
        Rows of ocr_calibration.csv keyed by noise tier, used to label each
        panel with the sorted character error rate for that tier.

    Returns
    output_path
        Path the image was written to, or None when the rendered form images
        are absent. Those images are gitignored because degrade_images.py
        regenerates them.
    """
    panels = []
    for noise_level in NOISE_LEVELS:
        image_path = (IMAGE_DIR / noise_level /
                      f"form_{LADDER_FORM_ID:03d}.png")
        if not image_path.exists():
            print(f"Skipping the noise ladder. {image_path} is missing. "
                  "Run generate_forms.py and degrade_images.py to rebuild it.")
            return None

        cropped = Image.open(image_path).convert("RGB").crop(LADDER_CROP_BOX)
        scaled_height = int(cropped.height * LADDER_PANEL_WIDTH / cropped.width)
        panels.append(cropped.resize((LADDER_PANEL_WIDTH, scaled_height),
                                     Image.LANCZOS))

    panel_height = panels[0].height
    row_height = panel_height + LADDER_LABEL_HEIGHT + LADDER_PADDING
    sheet = Image.new(
        "RGB",
        (LADDER_PANEL_WIDTH + 2 * LADDER_PADDING,
         LADDER_PADDING + len(panels) * row_height),
        COLOR_SURFACE,
    )

    canvas = ImageDraw.Draw(sheet)
    label_font = load_ladder_font(20, bold=True)
    value_font = load_ladder_font(18)
    offset_y = LADDER_PADDING

    for noise_level, panel in zip(NOISE_LEVELS, panels):
        sorted_cer = float(calibration_rows[noise_level]["cer_sorted"])
        canvas.text((LADDER_PADDING, offset_y), noise_level.capitalize(),
                    font=label_font, fill=COLOR_INK)
        canvas.text((LADDER_PADDING + 120, offset_y),
                    f"sorted CER {sorted_cer:.2f}",
                    font=value_font, fill=COLOR_MUTED)

        offset_y += LADDER_LABEL_HEIGHT
        sheet.paste(panel, (LADDER_PADDING, offset_y))
        canvas.rectangle(
            [LADDER_PADDING, offset_y,
             LADDER_PADDING + LADDER_PANEL_WIDTH - 1, offset_y + panel_height - 1],
            outline=COLOR_AXIS, width=1)
        offset_y += panel_height + LADDER_PADDING

    output_path = ASSET_DIR / "noise_ladder.png"
    sheet.save(output_path)
    return output_path


# entry point

def main():
    """Build every figure in the portfolio set and report what was written."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    ground_truth = load_json(GROUND_TRUTH_PATH)
    rules_predictions = load_json(RESULTS_DIR / "predictions_rules.json")
    llm_predictions = load_json(RESULTS_DIR / "predictions_llm.json")

    comparison_rows = load_csv_by_noise("f1_comparison.csv")
    difference_rows = load_csv_by_noise("bootstrap_diff.csv")
    calibration_rows = load_csv_by_noise("ocr_calibration.csv")

    written = [
        plot_recall_by_noise(comparison_rows, difference_rows),
        plot_triage_confusion(llm_predictions, ground_truth),
        plot_field_errors(
            count_field_errors(rules_predictions, ground_truth, FIELD_FIGURE_TIER),
            count_field_errors(llm_predictions, ground_truth, FIELD_FIGURE_TIER),
            form_count=len(ground_truth),
        ),
        build_noise_ladder(calibration_rows),
    ]

    for output_path in written:
        if output_path is not None:
            print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
