"""
evaluate.py — Metric computation for FiNER-ORD token classification.

Provides:
  - Seqeval entity-level metrics (what Daniel's 0.9 F1 refers to)
  - Token-level weighted F1 (what the paper reports)
  - Per-class entity F1 (PER, LOC, ORG)
  - Token-level confusion matrix (7x7)
  - Entity-span level confusion matrix (strict match)
  - Pretty-printing helpers for human inspection

All functions operate on seqeval-format string sequences:
  true_labels: list[list[str]]  # e.g., [['O', 'B-ORG', 'I-ORG', 'O'], ...]
  predictions: list[list[str]]  # same shape

Predictions and labels MUST be aligned: same outer length (number of
sentences), same inner length per sentence (number of real tokens,
with -100/special positions already removed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import precision_score as seqeval_precision
from seqeval.metrics import recall_score as seqeval_recall
from seqeval.metrics.sequence_labeling import get_entities
from sklearn.metrics import confusion_matrix, f1_score as sklearn_f1


ORDERED_LABELS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
ENTITY_TYPES = ["PER", "LOC", "ORG"]


# -------------------------------------------------------------------
# Seqeval wrappers — entity-level (the real metric)
# -------------------------------------------------------------------
def compute_seqeval_metrics(
    true_labels: list[list[str]], predictions: list[list[str]]
) -> dict:
    """
    Compute entity-level metrics using seqeval.

    Returns a dict with:
      - overall_precision, overall_recall, overall_f1, overall_accuracy
      - per-entity-type dicts keyed by PER/LOC/ORG, each with precision/recall/f1/number
    """
    report = seqeval_classification_report(
        true_labels,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    # seqeval's 'accuracy' in classification_report is sequence-level; compute token-level too
    overall_accuracy = _token_level_accuracy(true_labels, predictions)

    results = {
        "overall_precision": seqeval_precision(
            true_labels, predictions, zero_division=0
        ),
        "overall_recall": seqeval_recall(true_labels, predictions, zero_division=0),
        "overall_f1": seqeval_f1(true_labels, predictions, zero_division=0),
        "overall_accuracy": overall_accuracy,
    }

    # Add per-entity-type breakdown
    for entity_type in ENTITY_TYPES:
        if entity_type in report:
            results[entity_type] = {
                "precision": report[entity_type]["precision"],
                "recall": report[entity_type]["recall"],
                "f1": report[entity_type]["f1-score"],
                "number": report[entity_type]["support"],
            }

    return results


def _token_level_accuracy(
    true_labels: list[list[str]], predictions: list[list[str]]
) -> float:
    """Token-level accuracy across all sentences."""
    correct = 0
    total = 0
    for t_seq, p_seq in zip(true_labels, predictions):
        for t, p in zip(t_seq, p_seq):
            correct += int(t == p)
            total += 1
    return correct / total if total > 0 else 0.0


# -------------------------------------------------------------------
# Token-level weighted F1 (what the paper reports)
# -------------------------------------------------------------------
def compute_token_weighted_f1(
    true_labels: list[list[str]], predictions: list[list[str]]
) -> float:
    """
    Token-level weighted F1 across all 7 labels.

    This is the metric the FiNER-ORD paper reports in Table 3. It's inflated
    by the O-majority, so it should always be reported alongside entity F1.
    """
    flat_true = [t for seq in true_labels for t in seq]
    flat_pred = [p for seq in predictions for p in seq]
    return sklearn_f1(flat_true, flat_pred, average="weighted", zero_division=0)


# -------------------------------------------------------------------
# Confusion matrices
# -------------------------------------------------------------------
def token_confusion_matrix(
    true_labels: list[list[str]],
    predictions: list[list[str]],
) -> pd.DataFrame:
    """7x7 token-level confusion matrix across all labels."""
    flat_true = [t for seq in true_labels for t in seq]
    flat_pred = [p for seq in predictions for p in seq]
    cm = confusion_matrix(flat_true, flat_pred, labels=ORDERED_LABELS)
    return pd.DataFrame(cm, index=ORDERED_LABELS, columns=ORDERED_LABELS)


def entity_span_confusion_matrix(
    true_labels: list[list[str]],
    predictions: list[list[str]],
) -> pd.DataFrame:
    """
    Entity-span level confusion matrix with strict span matching.

    Rows: true entity type (or 'Missed' if a gold entity has no matching prediction)
    Cols: predicted entity type (or 'Spurious' if a predicted entity has no matching gold)

    A "match" requires both the type AND the exact (start, end) boundary to agree.
    This is the matrix that makes the ORG boundary problem visible.
    """
    classes = ENTITY_TYPES + ["Spurious_or_Missed"]
    cm = pd.DataFrame(0, index=classes, columns=classes)

    for t_seq, p_seq in zip(true_labels, predictions):
        t_ents = get_entities(t_seq)  # list of (type, start, end)
        p_ents = get_entities(p_seq)

        t_by_span = {(s, e): ent for ent, s, e in t_ents}
        p_by_span = {(s, e): ent for ent, s, e in p_ents}

        # Match gold against predictions
        for span, t_type in t_by_span.items():
            if span in p_by_span:
                p_type = p_by_span[span]
                cm.loc[t_type, p_type] += 1
                del p_by_span[span]
            else:
                # Gold entity with no exact-span match in predictions
                cm.loc[t_type, "Spurious_or_Missed"] += 1

        # Remaining predictions are spurious (no matching gold span)
        for span, p_type in p_by_span.items():
            cm.loc["Spurious_or_Missed", p_type] += 1

    return cm


# -------------------------------------------------------------------
# Top-level: the thing you actually call
# -------------------------------------------------------------------
def compute_detailed_metrics(
    true_labels: list[list[str]],
    predictions: list[list[str]],
    verbose: bool = True,
) -> dict:
    """
    Compute the full suite of metrics in one call.

    Returns a dict suitable for JSON serialization:
      - token_weighted_f1: float
      - entity_overall_f1: float
      - entity_overall_precision: float
      - entity_overall_recall: float
      - entity_per_class: {'PER': float, 'LOC': float, 'ORG': float}
      - token_confusion_matrix: dict (DataFrame.to_dict)
      - entity_span_confusion_matrix: dict (DataFrame.to_dict)

    If verbose=True, prints a human-readable report.
    """
    token_f1 = compute_token_weighted_f1(true_labels, predictions)
    seqeval_results = compute_seqeval_metrics(true_labels, predictions)
    token_cm = token_confusion_matrix(true_labels, predictions)
    span_cm = entity_span_confusion_matrix(true_labels, predictions)

    per_class_f1 = {
        et: seqeval_results[et]["f1"] for et in ENTITY_TYPES if et in seqeval_results
    }

    metrics = {
        "token_weighted_f1": float(token_f1),
        "entity_overall_f1": float(seqeval_results["overall_f1"]),
        "entity_overall_precision": float(seqeval_results["overall_precision"]),
        "entity_overall_recall": float(seqeval_results["overall_recall"]),
        "entity_per_class": {k: float(v) for k, v in per_class_f1.items()},
        "token_confusion_matrix": token_cm.to_dict(),
        "entity_span_confusion_matrix": span_cm.to_dict(),
    }

    if verbose:
        print_metrics_report(metrics, token_cm, span_cm)

    return metrics


# -------------------------------------------------------------------
# Pretty-printing
# -------------------------------------------------------------------
def print_metrics_report(
    metrics: dict,
    token_cm: pd.DataFrame | None = None,
    span_cm: pd.DataFrame | None = None,
) -> None:
    """Print a human-readable metrics report."""
    print("\n" + "=" * 60)
    print("Metrics Report")
    print("=" * 60)
    print(f"Token-level Weighted F1: {metrics['token_weighted_f1']:.4f}")
    print(f"Entity-level Overall F1: {metrics['entity_overall_f1']:.4f}")
    print(f"  Precision: {metrics['entity_overall_precision']:.4f}")
    print(f"  Recall:    {metrics['entity_overall_recall']:.4f}")

    print("\nPer-class Entity F1:")
    for entity_type in ENTITY_TYPES:
        val = metrics["entity_per_class"].get(entity_type, None)
        if val is not None:
            print(f"  {entity_type}: {val:.4f}")
        else:
            print(f"  {entity_type}: N/A")

    if token_cm is not None:
        print("\nToken Confusion Matrix:")
        print(token_cm)

    if span_cm is not None:
        print("\nEntity-Span Confusion Matrix (strict match):")
        print(span_cm)


# -------------------------------------------------------------------
# Delta computation (useful for BIO repair and ablation comparisons)
# -------------------------------------------------------------------
def compute_metrics_delta(
    metrics_before: dict,
    metrics_after: dict,
    tag_before: str = "before",
    tag_after: str = "after",
) -> None:
    """Pretty-print the delta between two metric dicts."""
    print("\n" + "=" * 60)
    print(f"Delta: {tag_before} -> {tag_after}")
    print("=" * 60)

    keys = [
        ("token_weighted_f1", "Token F1"),
        ("entity_overall_f1", "Entity F1"),
        ("entity_overall_precision", "Entity Precision"),
        ("entity_overall_recall", "Entity Recall"),
    ]
    for key, label in keys:
        b = metrics_before.get(key, 0)
        a = metrics_after.get(key, 0)
        delta = a - b
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {label:20s} {b:.4f} -> {a:.4f}  ({arrow} {delta:+.4f})")

    print("\n  Per-class entity F1:")
    for entity_type in ENTITY_TYPES:
        b = metrics_before.get("entity_per_class", {}).get(entity_type, 0)
        a = metrics_after.get("entity_per_class", {}).get(entity_type, 0)
        delta = a - b
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"    {entity_type}: {b:.4f} -> {a:.4f}  ({arrow} {delta:+.4f})")


# -------------------------------------------------------------------
# Loading predictions from disk (used by BIO repair, ablation scripts)
# -------------------------------------------------------------------
def load_predictions(path: str) -> tuple[list[list[str]], list[list[str]]]:
    """Load saved predictions JSON and return (true_labels, predictions)."""
    import json

    with open(path) as f:
        data = json.load(f)
    return data["true_labels"], data["predictions"]


if __name__ == "__main__":
    # Smoke test: synthetic data
    true_labels = [
        ["O", "B-PER", "I-PER", "O", "B-ORG", "I-ORG", "I-ORG", "O"],
        ["B-LOC", "I-LOC", "O", "B-ORG", "O"],
    ]
    predictions = [
        ["O", "B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O", "O"],  # ORG boundary error
        ["B-LOC", "I-LOC", "O", "B-LOC", "O"],  # ORG→LOC type error
    ]
    compute_detailed_metrics(true_labels, predictions, verbose=True)
