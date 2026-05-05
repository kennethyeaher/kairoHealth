"""
SScore rule-based and LLM predictions against ground truth.

Computes precision, recall, and F1 at the field level, aggregated to
per method per noise level for the headline comparison and broken out
per field for error analysis.

Outputs:
- results/f1_comparison.csv         : headline table (method x noise x P/R/F1)
- results/per_field_f1.csv          : per method, per noise, per field F1
- results/f1_by_noise.png           : headline bar chart (F1)
- results/recall_by_noise.png       : secondary chart (recall) for discussion
- results/rules_field_heatmap.png   : per-field F1 heatmap, regex
- results/llm_field_heatmap.png     : per-field F1 heatmap, LLM
"""

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (FIELD_SCHEMA, GROUND_TRUTH_PATH, NOISE_LEVELS, RESULTS_DIR)


# constants for figures
# consistent color palette across all figures
COLOR_RULES = "#888888"
COLOR_LLM = "#3b82f6"

# figure DPI,  high enough for paper print, not so high that file sizes balloon
FIGURE_DPI = 150


# scoring

def normalize(value):
    """Lowercase + trim for forgiving string equality.

    Handles None gracefully so callers can compare without isinstance checks.
    """
    if value is None:
        return None
    return str(value).strip().lower()


def tally_predictions(predictions, ground_truth):
    """Count TP/FP/FN per (noise, field) for one extraction method.

    Scoring rules:
    - Predicted value == ground truth (and ground truth is not None): TP
    - Predicted None, ground truth has a value: FN (missed extraction)
    - Predicted non-null wrong value when ground truth has a value: FP + FN
      (wrong value is worse than missing, counts as both a miss of the
      correct answer and an emission of a wrong one)
    - Predicted non-null when ground truth is None: FP

    Returns
    -------
    dict
        Mapping (noise, field) -> {tp, fp, fn} integer counts.
    """
    tallies = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for key, pred in predictions.items():
        # keys look like "0_clean", "12_heavy", split on first underscore
        form_id, noise = key.split("_", 1)
        gt = ground_truth[form_id]

        for field in FIELD_SCHEMA:
            gt_val = normalize(gt.get(field))
            pred_val = normalize(pred.get(field))

            tally = tallies[(noise, field)]
            if pred_val == gt_val and gt_val is not None:
                tally["tp"] += 1
            elif pred_val is None and gt_val is not None:
                tally["fn"] += 1
            elif pred_val is not None and gt_val is None:
                tally["fp"] += 1
            elif pred_val != gt_val:
                # Both methods produced something, but they disagree.
                # Count as FP (wrong emission) + FN (correct value missed).
                tally["fp"] += 1
                tally["fn"] += 1

    return tallies


def precision_recall_f1(tp, fp, fn):
    """Compute precision, recall, F1 from raw counts. All return 0 when
    undefined (zero denominators)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def aggregate_by_noise(tallies):
    """Sum tallies across fields to get one row per noise level.

    Returns
    -------
    pd.DataFrame
        Columns: noise, precision, recall, f1.
    """
    rows = []
    for noise in NOISE_LEVELS:
        tp = sum(tallies[(noise, f)]["tp"] for f in FIELD_SCHEMA)
        fp = sum(tallies[(noise, f)]["fp"] for f in FIELD_SCHEMA)
        fn = sum(tallies[(noise, f)]["fn"] for f in FIELD_SCHEMA)
        p, r, f = precision_recall_f1(tp, fp, fn)
        rows.append({
            "noise": noise,
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f, 3),
        })
    return pd.DataFrame(rows)


def per_field_f1(tallies):
    """Return per (noise, field) F1 as a DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: noise, field, f1.
    """
    rows = []
    for noise in NOISE_LEVELS:
        for field in FIELD_SCHEMA:
            t = tallies[(noise, field)]
            _, _, f = precision_recall_f1(t["tp"], t["fp"], t["fn"])
            rows.append({
                "noise": noise,
                "field": field,
                "f1": round(f, 3),
            })
    return pd.DataFrame(rows)



# Plotting
def plot_metric_by_noise(rules_summary, llm_summary, metric, title, out_path):
    """Side by side bar chart of one metric across noise tiers.

    Used for both F1 (headline) and recall (secondary). Same shape so
    figures look consistent in the paper.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(NOISE_LEVELS))
    width = 0.35

    ax.bar(x - width / 2, rules_summary[metric], width, label="Rule based (regex)", color=COLOR_RULES)
    ax.bar(x + width / 2, llm_summary[metric], width, label="LLM (zero shot)", color=COLOR_LLM)

    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in NOISE_LEVELS])
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    # label each bar with its value
    for i, (r, l) in enumerate(zip(rules_summary[metric], llm_summary[metric])):
        ax.text(i - width / 2, r + 0.01, f"{r:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, l + 0.01, f"{l:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI)
    plt.close()


def plot_field_heatmap(per_field_df, method_label, out_path):
    """Per field F1 heatmap: rows = field, columns = noise level."""
    pivot = per_field_df.pivot(index="field", columns="noise", values="f1")
    # enforce consistent ordering: noise tiers L to R, fields top to bottom
    pivot = pivot[NOISE_LEVELS]
    pivot = pivot.reindex(FIELD_SCHEMA)

    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "F1 score"})
    ax.set_title(f"{method_label} F1 by field x noise level")
    ax.set_xlabel("Noise level")
    ax.set_ylabel("Field")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI)
    plt.close()


# entry point

def main():
    """Score both methods, write tables and figures."""
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)
    with open(RESULTS_DIR / "predictions_rules.json") as f:
        rules_pred = json.load(f)
    with open(RESULTS_DIR / "predictions_llm.json") as f:
        llm_pred = json.load(f)

    rules_tallies = tally_predictions(rules_pred, ground_truth)
    llm_tallies = tally_predictions(llm_pred, ground_truth)

    rules_summary = aggregate_by_noise(rules_tallies)
    llm_summary = aggregate_by_noise(llm_tallies)

    # combined comparison table for the paper
    comparison = pd.DataFrame({
        "noise": NOISE_LEVELS,
        "rules_precision": rules_summary["precision"].values,
        "rules_recall": rules_summary["recall"].values,
        "rules_f1": rules_summary["f1"].values,
        "llm_precision": llm_summary["precision"].values,
        "llm_recall": llm_summary["recall"].values,
        "llm_f1": llm_summary["f1"].values,
    })
    comparison.to_csv(RESULTS_DIR / "f1_comparison.csv", index=False)

    # per field detail
    rules_per_field = per_field_f1(rules_tallies)
    llm_per_field = per_field_f1(llm_tallies)
    rules_per_field["method"] = "rules"
    llm_per_field["method"] = "llm"
    pd.concat([rules_per_field, llm_per_field]).to_csv(
        RESULTS_DIR / "per_field_f1.csv", index=False)

    # headline figure: F1 by noise (the standard metric)
    plot_metric_by_noise(
        rules_summary, llm_summary, metric="f1",
        title="Field extraction F1 by OCR noise level",
        out_path=RESULTS_DIR / "f1_by_noise.png",
    )

    # secondary figure: recall by noise 
    plot_metric_by_noise(
        rules_summary, llm_summary, metric="recall",
        title="Field extraction recall by OCR noise level",
        out_path=RESULTS_DIR / "recall_by_noise.png",
    )

    # per field heatmaps
    plot_field_heatmap(rules_per_field, "Rule-based", RESULTS_DIR / "rules_field_heatmap.png")
    plot_field_heatmap(llm_per_field, "LLM", RESULTS_DIR / "llm_field_heatmap.png")

    print("=== Headline F1 comparison ===")
    print(comparison.to_string(index=False))
    print()
    print(f"Tables and figures written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
