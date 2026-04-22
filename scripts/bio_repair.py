"""
Apply BIO validity repair to existing model predictions.
Measures the delta in entity F1 from post-hoc sequence repair.
"""

import json
from pathlib import Path
from seqeval.metrics import classification_report, f1_score
from seqeval.metrics.sequence_labeling import get_entities


def repair_bio(labels):
    """Convert any I-X not preceded by B-X or I-X into B-X."""
    repaired = []
    prev = "O"
    for lbl in labels:
        if lbl.startswith("I-"):
            entity_type = lbl[2:]
            if prev == "O" or (not prev.endswith(entity_type)):
                lbl = f"B-{entity_type}"
        repaired.append(lbl)
        prev = lbl
    return repaired


def load_predictions(path):
    """Expect a JSON with {'true_labels': [[...]], 'predictions': [[...]]}."""
    with open(path) as f:
        data = json.load(f)
    return data["true_labels"], data["predictions"]


def evaluate(true_labels, predictions, tag):
    f1 = f1_score(true_labels, predictions)
    per_class = {}
    for entity in ["PER", "LOC", "ORG"]:
        # Filter spans by type for per-class
        from seqeval.metrics import f1_score as f1

        report = classification_report(true_labels, predictions, output_dict=True)
        if entity in report:
            per_class[entity] = report[entity]["f1-score"]
    print(f"\n--- {tag} ---")
    print(f"Overall entity F1: {f1:.4f}")
    for k, v in per_class.items():
        print(f"  {k}: {v:.4f}")
    return f1, per_class


if __name__ == "__main__":
    # Load existing predictions from your RoBERTa-large run
    true_labels, predictions = load_predictions(
        "results/roberta-large_seed88/predictions.json"
    )

    # Baseline
    f1_before, pc_before = evaluate(true_labels, predictions, "Before repair")

    # Repaired
    repaired = [repair_bio(seq) for seq in predictions]
    f1_after, pc_after = evaluate(true_labels, repaired, "After BIO repair")

    print(f"\n=== Delta ===")
    print(
        f"Overall F1: {f1_before:.4f} -> {f1_after:.4f} (Δ {f1_after - f1_before:+.4f})"
    )
    for k in pc_before:
        print(
            f"  {k}: {pc_before[k]:.4f} -> {pc_after[k]:.4f} (Δ {pc_after[k] - pc_before[k]:+.4f})"
        )
