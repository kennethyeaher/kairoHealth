"""
significance.py

Statistical significance testing for the Kairo Health pipeline.

Reads prediction JSON files produced by evaluate.py and adds two forms of
statistical evidence that point estimates alone cannot provide:

1. Bootstrap 95 percent confidence intervals on pooled F1.
2. Exact McNemar tests on field level correctness.

Run after evaluate.py:
    python -m src.significance

Outputs written to results:
    bootstrap_ci.csv    F1 with 95 percent CI by method and noise tier
    bootstrap_diff.csv  Paired LLM minus Rules F1 difference with 95 percent CI
    mcnemar.csv         Exact McNemar p value by noise tier
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


# configuration
# declared at module level so paths and settings are easy to override
REPO_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_PATH = REPO_ROOT / "data" / "ground_truth.json"
PREDICTION_PATHS = {
    "rules": REPO_ROOT / "results" / "predictions_rules.json",
    "llm": REPO_ROOT / "results" / "predictions_llm.json",
}
RESULTS_DIR = REPO_ROOT / "results"

# add a new tier here if the experiment gets expanded
NOISE_TIERS = ("clean", "moderate", "heavy", "severe", "extreme")

N_BOOTSTRAP_ITERATIONS = 1000
RANDOM_SEED = 42


@dataclass(frozen=True)
class ScoredDoc:
    """
    One document scored against the gold standard for one extraction method.

    Keeps pooled counts for bootstrap and per field correctness flags for
    McNemar, so both tests use the same scoring pass.
    """

    noise: str
    true_positive: int
    false_positive: int
    false_negative: int
    correct_fields: dict[str, bool]


def normalize_value(value) -> str | None:
    """Normalize empty values to None and compare everything else as lowercase text."""
    if value in (None, "", "null"):
        return None

    return str(value).strip().lower()


def score_one_field(prediction, gold) -> tuple[int, int, int, bool]:
    """
    Score one field and return true positive, false positive, false negative, and correctness.

    A wrong prediction counts as both false positive and false negative because
    this is a clinical extraction task. The model added the wrong value and also
    missed the correct one.
    """
    pred, gold = normalize_value(prediction), normalize_value(gold)

    if pred is None and gold is None:
        return 0, 0, 0, True

    if pred is None:
        return 0, 0, 1, False

    if gold is None:
        return 0, 1, 0, False

    if pred == gold:
        return 1, 0, 0, True

    return 0, 1, 1, False


def f1_score(tp: int, fp: int, fn: int) -> float:
    """Compute F1 from true positive, false positive, and false negative counts."""
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def extract_noise_tier(doc_id: str) -> str | None:
    """Parse noise tier from a doc id following the pattern {form_number}_{tier}.

    Parameters
        doc_id  Document identifier string e.g. '0_clean' or '7_heavy'.

    Returns
        Tier name as a string or None if the last token is unrecognized.
    """
    tier = doc_id.split("_")[-1].lower()
    return tier if tier in NOISE_TIERS else None


def score_predictions(predictions_path: Path) -> dict[str, ScoredDoc]:
    """Load a predictions file, score every document, and return the scored dict.

    Gold IDs are form numbers like '0'. Prediction IDs add a tier suffix
    like '0_clean'. We iterate predictions and look up gold by stripping
    the suffix, so one gold record correctly scores against three
    prediction records (one per tier).

    Parameters predictions_path  Path to a predictions JSON file produced by
    extract_rules.py or extract_llm.py.

    Returns
        Dict mapping prediction doc id to ScoredDoc.
    """
    gold_data = json.loads(GROUND_TRUTH_PATH.read_text())
    prediction_data = json.loads(predictions_path.read_text())

    scored = {}
    for doc_id, pred_fields in prediction_data.items():
        # Strip the tier suffix to look up the matching gold form.
        form_id = doc_id.rsplit("_", 1)[0]
        gold_fields = gold_data.get(form_id, {})
        if not gold_fields:
            continue
        field_scores = {
            field: score_one_field(pred_fields.get(field), gold_value)
            for field, gold_value in gold_fields.items()
        }
        scored[doc_id] = ScoredDoc(
            noise=extract_noise_tier(doc_id),
            true_positive=sum(s[0] for s in field_scores.values()),
            false_positive=sum(s[1] for s in field_scores.values()),
            false_negative=sum(s[2] for s in field_scores.values()),
            correct_fields={field: s[3] for field, s in field_scores.items()},
        )
    return scored


def filter_by_tier(docs: dict[str, ScoredDoc], tier: str) -> dict[str, ScoredDoc]:
    """Return only documents from one noise tier."""
    return {
        doc_id: doc
        for doc_id, doc in docs.items()
        if doc.noise == tier
    }


def pooled_f1(documents: list[ScoredDoc]) -> float:
    """
    Pool counts across documents and then compute F1.

    This matches the evaluation setup and avoids giving equal weight to documents
    with different numbers of fields.
    """
    return f1_score(
        sum(doc.true_positive for doc in documents),
        sum(doc.false_positive for doc in documents),
        sum(doc.false_negative for doc in documents),
    )


def bootstrap_f1(docs: list[ScoredDoc], rng: np.random.Generator) -> np.ndarray:
    """Return the bootstrap distribution of pooled F1 for one method and noise tier."""
    doc_array = np.array(docs, dtype=object)
    indices = rng.integers(
        0,
        len(docs),
        size=(N_BOOTSTRAP_ITERATIONS, len(docs)),
    )

    return np.array([
        pooled_f1(doc_array[idx].tolist())
        for idx in indices
    ])


def bootstrap_diff(
    rules_docs: list[ScoredDoc],
    llm_docs: list[ScoredDoc],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return the paired bootstrap distribution of LLM minus Rules F1.

    Both methods use the same sampled document indices each round, which controls
    for document difficulty.
    """
    rules_array = np.array(rules_docs, dtype=object)
    llm_array = np.array(llm_docs, dtype=object)

    indices = rng.integers(
        0,
        len(rules_docs),
        size=(N_BOOTSTRAP_ITERATIONS, len(rules_docs)),
    )

    return np.array([
        pooled_f1(llm_array[idx].tolist()) - pooled_f1(rules_array[idx].tolist())
        for idx in indices
    ])


def ci_95(distribution: np.ndarray) -> tuple[float, float]:
    """Return the 2.5 and 97.5 percentile bounds for a bootstrap distribution."""
    return tuple(float(x) for x in np.percentile(distribution, [2.5, 97.5]))


def mcnemar(
    rules_docs: dict[str, ScoredDoc],
    llm_docs: dict[str, ScoredDoc],
) -> dict:
    """
    Run an exact McNemar test comparing Rules and LLM correctness.

    Each field in each document is treated as one trial. The test focuses on
    disagreements: fields only the LLM got right and fields only the rules method
    got right.
    """
    llm_only = 0
    rules_only = 0

    for doc_id, rules_doc in rules_docs.items():
        llm_doc = llm_docs[doc_id]

        for field, rules_correct in rules_doc.correct_fields.items():
            llm_correct = llm_doc.correct_fields.get(field, False)

            if llm_correct and not rules_correct:
                llm_only += 1
            elif rules_correct and not llm_correct:
                rules_only += 1

    n_disagree = llm_only + rules_only

    # no disagreements means there is no imbalance to test, so p equals 1
    p_value = 1.0 if n_disagree == 0 else float(
        binomtest(
            min(llm_only, rules_only),
            n_disagree,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )

    return {
        "llm_only_correct": llm_only,
        "rules_only_correct": rules_only,
        "n_disagree": n_disagree,
        "p_value": p_value,
    }


def main() -> None:
    """
    Score predictions, run bootstrap and McNemar tests, and write CSV outputs.

    Tiers with no documents are skipped instead of failing the full run.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    rules_docs = score_predictions(PREDICTION_PATHS["rules"])
    llm_docs = score_predictions(PREDICTION_PATHS["llm"])

    ci_rows = []
    diff_rows = []
    mc_rows = []

    for tier in NOISE_TIERS:
        rules_tier = filter_by_tier(rules_docs, tier)
        llm_tier = filter_by_tier(llm_docs, tier)

        if not rules_tier or not llm_tier:
            print(f"Skipping {tier}. No documents found.")
            continue

        for method, tier_docs in (("rules", rules_tier), ("llm", llm_tier)):
            f1_dist = bootstrap_f1(list(tier_docs.values()), rng)
            lo, hi = ci_95(f1_dist)

            ci_rows.append({
                "method": method,
                "noise": tier,
                "f1_mean": float(f1_dist.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "n_docs": len(tier_docs),
            })

        diff_dist = bootstrap_diff(
            list(rules_tier.values()),
            list(llm_tier.values()),
            rng,
        )
        lo, hi = ci_95(diff_dist)

        diff_rows.append({
            "noise": tier,
            "f1_diff_mean": float(diff_dist.mean()),
            "ci_low": lo,
            "ci_high": hi,
            "significant_at_95": lo > 0 or hi < 0,
        })

        mc = mcnemar(rules_tier, llm_tier)
        mc_rows.append({
            **mc,
            "noise": tier,
            "significant_at_05": mc["p_value"] < 0.05,
        })

    pd.DataFrame(ci_rows).to_csv(RESULTS_DIR / "bootstrap_ci.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(RESULTS_DIR / "bootstrap_diff.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(RESULTS_DIR / "mcnemar.csv", index=False)

    print(f"Wrote bootstrap_ci.csv, bootstrap_diff.csv, and mcnemar.csv to {RESULTS_DIR}")


if __name__ == "__main__":
    main()