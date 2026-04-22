"""
bio_repair.py — Post-hoc BIO validity repair for saved model predictions.

Applies the rule: any `I-X` not preceded by `B-X` or `I-X` becomes `B-X`.
Reports the entity-F1 delta before and after repair.

Usage:
    python scripts/bio_repair.py --predictions results/<run_id>/predictions.json --save
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluate import (
    compute_detailed_metrics,
    compute_metrics_delta,
    load_predictions,
)


def repair_bio(labels: list[str]) -> list[str]:
    """Convert any `I-X` not preceded by `B-X` or `I-X` into `B-X`."""
    repaired: list[str] = []
    prev = "O"
    for lbl in labels:
        if lbl.startswith("I-"):
            entity_type = lbl[2:]
            if prev == "O" or prev[2:] != entity_type:
                lbl = f"B-{entity_type}"
        repaired.append(lbl)
        prev = lbl
    return repaired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply BIO validity repair to predictions."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to a predictions.json with {true_labels, predictions}.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Also write the repaired predictions to predictions_repaired.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = Path(args.predictions)

    true_labels, predictions = load_predictions(str(pred_path))
    repaired = [repair_bio(seq) for seq in predictions]

    metrics_before = compute_detailed_metrics(true_labels, predictions, verbose=False)
    metrics_after = compute_detailed_metrics(true_labels, repaired, verbose=False)

    compute_metrics_delta(
        metrics_before, metrics_after, "before repair", "after repair"
    )

    if args.save:
        out_path = pred_path.with_name("predictions_repaired.json")
        with out_path.open("w") as f:
            json.dump({"true_labels": true_labels, "predictions": repaired}, f)
        print(f"\nSaved repaired predictions to: {out_path}")


if __name__ == "__main__":
    main()
