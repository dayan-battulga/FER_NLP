"""
Apply dynamic INT8 quantization to distilled student checkpoints.

Uses torch.quantization.quantize_dynamic on nn.Linear modules with
dtype=torch.qint8. The resulting checkpoint is CPU-targeted. Measure INT8
latency separately with scripts/measure_latency.py using --device cpu.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import ID2LABEL, get_dataset_and_tokenizer  # noqa: E402
from src.evaluate import compute_detailed_metrics  # noqa: E402
from src.train import make_json_safe, write_json_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Student run directory names under results/ or absolute paths.",
    )
    parser.add_argument(
        "--output-root",
        default="results",
        help="Root directory containing run folders.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="CPU inference batch size for the post-quantization test pass.",
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
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def decode_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[list[list[str]], list[list[str]]]:
    predicted = torch.argmax(logits, dim=-1).cpu().numpy()
    label_ids = labels.cpu().numpy()
    true_labels: list[list[str]] = []
    predictions: list[list[str]] = []
    for pred_sequence, label_sequence in zip(predicted, label_ids):
        true_sequence: list[str] = []
        pred_sequence_out: list[str] = []
        for pred_id, label_id in zip(pred_sequence, label_sequence):
            label_id = int(label_id)
            if label_id == -100:
                continue
            true_sequence.append(ID2LABEL[label_id])
            pred_sequence_out.append(ID2LABEL[int(pred_id)])
        true_labels.append(true_sequence)
        predictions.append(pred_sequence_out)
    return true_labels, predictions


def evaluate_quantized_model(
    model: torch.nn.Module,
    tokenizer: Any,
    model_name: str,
    max_seq_length: int,
    label_all_subwords: bool,
    batch_size: int,
) -> dict[str, Any]:
    dataset, _, _ = get_dataset_and_tokenizer(
        model_name,
        max_length=max_seq_length,
        run_checks=False,
        label_all_subwords=label_all_subwords,
    )
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model.eval()
    true_labels_all: list[list[str]] = []
    predictions_all: list[list[str]] = []

    for start in range(0, len(dataset["test"]), batch_size):
        examples = [
            dataset["test"][i]
            for i in range(start, min(start + batch_size, len(dataset["test"])))
        ]
        batch = collator(examples)
        labels = batch.pop("labels")
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        true_labels, predictions = decode_logits(outputs.logits, labels)
        true_labels_all.extend(true_labels)
        predictions_all.extend(predictions)

    return {
        "metrics": compute_detailed_metrics(true_labels_all, predictions_all, verbose=True),
        "predictions": {
            "true_labels": true_labels_all,
            "predictions": predictions_all,
        },
    }


def quantize_one_run(run_dir: Path, batch_size: int) -> Path:
    summary = load_summary(run_dir)
    if summary.get("runtime", {}).get("uses_crf", False):
        raise ValueError(f"{run_dir.name} is a CRF run; quantize only distilled students.")

    config = summary.get("config", {})
    checkpoint_dir = run_dir / "checkpoint-best"
    output_dir = run_dir / "checkpoint-best-int8"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, add_prefix_space=True)
    torch.backends.quantized.engine = 'qnnpack'
    quantized_model = torch.quantization.quantize_dynamic(
        model.cpu(),
        {nn.Linear},
        dtype=torch.qint8,
    )
    import shutil
    torch.save(quantized_model.state_dict(), output_dir / "pytorch_model.bin")
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]:
        src = checkpoint_dir / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)
    tokenizer.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    model_name = str(checkpoint_dir)
    eval_payload = evaluate_quantized_model(
        model=quantized_model,
        tokenizer=tokenizer,
        model_name=model_name,
        max_seq_length=int(config.get("max_seq_length", 256)),
        label_all_subwords=bool(config.get("label_all_subwords", False)),
        batch_size=batch_size,
    )
    summary_int8 = {
        "run_id": f"{run_dir.name}_int8",
        "source_run": run_dir.name,
        "checkpoint_path": str(output_dir),
        "quantization": {
            "method": "torch.quantization.quantize_dynamic",
            "modules": ["nn.Linear"],
            "dtype": "torch.qint8",
            "latency_note": "Measure on CPU separately with scripts/measure_latency.py.",
        },
        "test_metrics": eval_payload["metrics"],
    }
    write_json_file(run_dir / "summary_int8.json", make_json_safe(summary_int8))
    write_json_file(output_dir / "summary_int8.json", make_json_safe(summary_int8))
    write_json_file(output_dir / "predictions_int8.json", eval_payload["predictions"])
    print(f"Wrote {output_dir}")
    print(f"Wrote {run_dir / 'summary_int8.json'}")
    return output_dir


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    for run in args.runs:
        quantize_one_run(resolve_run_dir(run, output_root), batch_size=args.batch_size)


if __name__ == "__main__":
    main()
