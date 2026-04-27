"""
Measure model latency for FiNER-ORD checkpoints.

Reports parameter count, checkpoint size, median latency, and p95 latency at
batch sizes 1 and 8 after 10 warmup batches and 100 measured batches. Dynamic
INT8 checkpoints should be measured on CPU.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.crf_model import RobertaCrfForTokenClassification  # noqa: E402
from src.data import get_dataset_and_tokenizer  # noqa: E402


BATCH_SIZES = [1, 8]
WARMUP_BATCHES = 10
MEASURE_BATCHES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directory names under results/ or absolute paths.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        required=True,
        help="Device used for latency measurement.",
    )
    parser.add_argument(
        "--output-root",
        default="results",
        help="Root directory containing run folders.",
    )
    return parser.parse_args()


def resolve_run(run: str, output_root: Path) -> tuple[str, Path, Path]:
    run_name = Path(run).name
    candidate = Path(run)
    if not candidate.is_absolute():
        candidate = output_root / run
    if candidate.exists():
        return run_name, candidate, candidate / "checkpoint-best"

    if run_name.endswith("_int8"):
        base_name = run_name[: -len("_int8")]
        base_dir = output_root / base_name
        checkpoint = base_dir / "checkpoint-best-int8"
        if checkpoint.exists():
            return run_name, base_dir, checkpoint

    raise FileNotFoundError(f"Could not resolve run or INT8 checkpoint: {run}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cpu_model_name() -> str:
    if platform.system() == "Darwin":
        try:
            output = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
            if output:
                return output
        except Exception:
            pass
    return platform.processor() or platform.machine()


def directory_model_size_mb(checkpoint_dir: Path) -> float:
    preferred = ["model.safetensors", "pytorch_model.bin"]
    for filename in preferred:
        path = checkpoint_dir / filename
        if path.exists():
            return path.stat().st_size / (1024 * 1024)
    total = sum(path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file())
    return total / (1024 * 1024)


def load_model_and_tokenizer(
    run_dir: Path,
    checkpoint_dir: Path,
) -> tuple[torch.nn.Module, Any, bool, str]:
    summary = load_json(run_dir / "summary.json")
    config = summary.get("config", {})
    uses_crf = bool(summary.get("runtime", {}).get("uses_crf", False))
    model_name = config.get("model_name", str(checkpoint_dir))

    if uses_crf:
        model = RobertaCrfForTokenClassification(model_name)
        state_path = checkpoint_dir / "pytorch_model.bin"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing CRF state file: {state_path}")
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
        tokenizer_name = model_name
    elif checkpoint_dir.name == "checkpoint-best-int8":
        fp32_checkpoint = run_dir / "checkpoint-best"
        if not fp32_checkpoint.exists():
            raise FileNotFoundError(
                f"Missing FP32 source checkpoint for INT8 measurement: {fp32_checkpoint}"
            )
        fp32_model = AutoModelForTokenClassification.from_pretrained(fp32_checkpoint)
        torch.backends.quantized.engine = 'qnnpack'
        model = torch.quantization.quantize_dynamic(
            fp32_model.cpu(),
            {nn.Linear},
            dtype=torch.qint8,
        )
        tokenizer_name = str(checkpoint_dir)
    else:
        model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
        tokenizer_name = str(checkpoint_dir)

    _, tokenizer, _ = get_dataset_and_tokenizer(
        tokenizer_name,
        max_length=int(config.get("max_seq_length", 256)),
        run_checks=False,
        label_all_subwords=bool(config.get("label_all_subwords", False)),
    )
    return model, tokenizer, uses_crf, str(model_name)


def make_batches(tokenizer: Any, batch_size: int) -> list[dict[str, torch.Tensor]]:
    dataset, _, _ = get_dataset_and_tokenizer(
        tokenizer.name_or_path,
        max_length=256,
        run_checks=False,
        label_all_subwords=False,
    )
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    test_split = dataset["test"]
    batches = []
    for start in range(0, len(test_split), batch_size):
        examples = [test_split[i] for i in range(start, min(start + batch_size, len(test_split)))]
        batches.append(collator(examples))
    if not batches:
        raise ValueError("Test split produced no batches.")
    return batches


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask"}
    }


def run_forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    uses_crf: bool,
) -> None:
    with torch.no_grad():
        if uses_crf:
            model.decode(batch["input_ids"], batch["attention_mask"])
        else:
            model(**batch)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def measure_for_batch_size(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    uses_crf: bool,
    batch_size: int,
) -> dict[str, float]:
    batches = make_batches(tokenizer, batch_size)
    model.eval()
    timings_ms: list[float] = []

    total_steps = WARMUP_BATCHES + MEASURE_BATCHES
    for step in range(total_steps):
        batch = move_batch(batches[step % len(batches)], device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        run_forward(model, batch, uses_crf)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if step >= WARMUP_BATCHES:
            timings_ms.append(elapsed_ms)

    median_ms = statistics.median(timings_ms)
    p95_ms = percentile(timings_ms, 95)
    examples_per_second = batch_size / (median_ms / 1000.0)
    return {
        "batch_size": batch_size,
        "warmup_batches": WARMUP_BATCHES,
        "measured_batches": MEASURE_BATCHES,
        "median_ms": float(median_ms),
        "p95_ms": float(p95_ms),
        "examples_per_second_median": float(examples_per_second),
    }


def measure_one_run(run_name: str, run_dir: Path, checkpoint_dir: Path, device_name: str) -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    if checkpoint_dir.name == "checkpoint-best-int8" and device_name != "cpu":
        raise ValueError("INT8 dynamic quantization must be measured with --device cpu.")

    device = torch.device(device_name)
    model, tokenizer, uses_crf, model_name = load_model_and_tokenizer(run_dir, checkpoint_dir)
    model.to(device)
    param_count = sum(parameter.numel() for parameter in model.parameters())

    measurements = {}
    for batch_size in BATCH_SIZES:
        measurements[str(batch_size)] = measure_for_batch_size(
            model=model,
            tokenizer=tokenizer,
            device=device,
            uses_crf=uses_crf,
            batch_size=batch_size,
        )

    hardware = {
        "torch_version": torch.__version__,
        "device": device_name,
        "cpu_model": cpu_model_name(),
        "gpu_model": torch.cuda.get_device_name(0)
        if device_name == "cuda" and torch.cuda.is_available()
        else None,
    }
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "model_name": model_name,
        "uses_crf": uses_crf,
        "param_count": int(param_count),
        "checkpoint_mb": float(directory_model_size_mb(checkpoint_dir)),
        "hardware": hardware,
        "latency": measurements,
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    latency_dir = output_root / "latency"
    latency_dir.mkdir(parents=True, exist_ok=True)

    print(f"torch: {torch.__version__}")
    print(f"cpu: {cpu_model_name()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    else:
        print("gpu: none")

    for run in args.runs:
        run_name, run_dir, checkpoint_dir = resolve_run(run, output_root)
        payload = measure_one_run(run_name, run_dir, checkpoint_dir, args.device)
        output_path = latency_dir / f"{run_name}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
