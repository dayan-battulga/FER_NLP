"""
Extract train-split CRF emissions from locked teacher checkpoints.

This script reads existing teacher checkpoints only. It does not retrain or
modify the locked teacher pipeline. The saved train_emissions.npz schema matches
the existing val_emissions.npz and test_emissions.npz files:
emissions, attention_mask, labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.crf_model import (  # noqa: E402
    RobertaCrfForTokenClassification,
    extract_crf_emissions,
    save_emissions_npz,
)
from src.data import get_dataset_and_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Teacher run directory names under results/ or absolute paths.",
    )
    parser.add_argument(
        "--output-root",
        default="results",
        help="Root directory containing run folders.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size. Defaults to the batch_size in summary.json.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="roberta-large",
        help="Tokenizer used for emission alignment. Defaults to roberta-large.",
    )
    return parser.parse_args()


def resolve_run_dir(run: str, output_root: Path) -> Path:
    candidate = Path(run)
    if not candidate.is_absolute():
        candidate = output_root / run
    if not candidate.exists():
        raise FileNotFoundError(f"Run directory does not exist: {candidate}")
    return candidate


def load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_crf_checkpoint(run_dir: Path, summary: dict[str, Any]) -> RobertaCrfForTokenClassification:
    config = summary.get("config", {})
    model_name = config.get("model_name")
    if not model_name:
        raise ValueError(f"summary.json for {run_dir.name} does not contain config.model_name")

    checkpoint_dir = Path(summary.get("checkpoint_best_path") or run_dir / "checkpoint-best")
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = REPO_ROOT / checkpoint_dir
    state_path = checkpoint_dir / "pytorch_model.bin"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing CRF checkpoint state: {state_path}")

    model = RobertaCrfForTokenClassification(model_name)
    state_dict = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def describe_npz(path: Path) -> None:
    with np.load(path, allow_pickle=True) as handle:
        print(f"\nWrote {path}")
        for key in handle.files:
            array = handle[key]
            first_shape = None
            first_dtype = None
            if len(array) > 0:
                first = array[0]
                first_shape = getattr(first, "shape", None)
                first_dtype = getattr(first, "dtype", None)
            print(
                f"  {key}: dtype={array.dtype}, len={len(array)}, "
                f"first_shape={first_shape}, first_dtype={first_dtype}"
            )


def extract_one_run(
    run_dir: Path,
    dataset: Any,
    tokenizer: Any,
    batch_size_override: int | None,
) -> Path:
    summary = load_summary(run_dir)
    if not summary.get("runtime", {}).get("uses_crf", False):
        raise ValueError(f"{run_dir.name} is not marked as a CRF run in summary.json")

    config = summary.get("config", {})
    batch_size = int(batch_size_override or config.get("batch_size", 8))
    model = load_crf_checkpoint(run_dir, summary)
    emissions, masks, labels = extract_crf_emissions(
        model=model,
        dataset=dataset["train"],
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    output_path = run_dir / "train_emissions.npz"
    save_emissions_npz(output_path, emissions, masks, labels)
    describe_npz(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    dataset, tokenizer, _ = get_dataset_and_tokenizer(
        args.tokenizer_name,
        max_length=256,
        run_checks=False,
        label_all_subwords=False,
    )
    for run in args.runs:
        extract_one_run(
            run_dir=resolve_run_dir(run, output_root),
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size_override=args.batch_size,
        )


if __name__ == "__main__":
    main()
