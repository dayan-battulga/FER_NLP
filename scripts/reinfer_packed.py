"""
reinfer_packed.py - Phase 5c packed-window 512-length re-inference.

Inference-only re-runs an existing fine-tuned checkpoint on FiNER-ORD by
greedy-packing consecutive sentences from the same document into 512-token
windows, then slicing the per-window outputs back to per-sentence sequences
for downstream metric computation.

Algorithm (per the Phase 5c plan):

  1. Load FiNER-ORD via src.data.load_finer_ord; group sentences by
     doc_idx, sort by sent_idx.
  2. Tokenize each sentence with `is_split_into_words=True` and
     `add_special_tokens=False`.
  3. Greedy packing within each doc_idx: pack subword sequences until the
     next sentence would overflow the 512-token budget (minus 2 for
     <s>/</s>); flush the window and start a new one. Never pack across
     doc_idx boundaries.
  4. For each window, run a single encoder forward.
  5. Slice the resulting emissions/logits per sentence using the recorded
     spans.
  6. CRF: Viterbi-decode each sentence's slice independently using the
     run's saved `crf_transitions.npz`. Vanilla: argmax along last axis.
     Never run Viterbi across a packed boundary.
  7. Drop -100 positions and call `compute_detailed_metrics`.

Caveat documented in the plan: RoBERTa-large position embeddings 0-511 saw
pretrain context but positions 256-511 never saw FiNER fine-tune context.
The lift may be modest or negative. The plan calls for running the
validation split first and only running the test split if validation
delta is non-negative vs the same checkpoint at 256.

Usage:

    # Validation pass first (safe, burns the val signal not the test signal)
    python scripts/reinfer_packed.py \
        --runs efficient_after_dapt_v2_seed88 \
               efficient_after_dapt_v2_seed5768 \
               efficient_after_dapt_v2_seed78516 \
        --mode crf --split val

    # Test pass only if validation delta is non-negative
    python scripts/reinfer_packed.py \
        --runs efficient_after_dapt_v2_seed88 \
               efficient_after_dapt_v2_seed5768 \
               efficient_after_dapt_v2_seed78516 \
        --mode crf --split test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    logging as hf_logging,
)

# Make `src.*` importable when run directly from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.crf_model import RobertaCrfForTokenClassification  # noqa: E402
from src.data import (  # noqa: E402
    ID2LABEL,
    align_labels_to_subwords,
    load_finer_ord,
)
from src.evaluate import compute_detailed_metrics  # noqa: E402
from src.train import (  # noqa: E402
    COMPAT_RESULTS_HEADER,
    DETAILED_RESULTS_HEADER,
    append_csv_row,
    write_json_file,
)


PACKED_MAX_LENGTH = 512


# ---------------------------------------------------------------------------
# Window packing
# ---------------------------------------------------------------------------


def _pretokenize_sentence(
    tokenizer: Any,
    tokens: list[str],
    word_labels: list[int],
    label_all_subwords: bool,
    truncation_budget: int,
) -> tuple[list[int], list[int]]:
    """Tokenize one sentence without specials and align per-subword labels.

    `truncation_budget` caps the per-sentence subword count so a single
    pathological sentence cannot exceed the window. The standard FiNER
    sentences are well under 512 subwords, but the budget keeps the
    invariant explicit.
    """
    encoded = tokenizer(
        [tokens],
        truncation=True,
        is_split_into_words=True,
        max_length=truncation_budget,
        add_special_tokens=False,
    )
    word_ids = encoded.word_ids(batch_index=0)
    subword_ids = encoded["input_ids"][0]
    aligned_labels = align_labels_to_subwords(
        word_ids,
        word_labels,
        label_all_subwords=label_all_subwords,
    )
    return subword_ids, aligned_labels


def build_packed_windows(
    grouped: Any,
    tokenizer: Any,
    split_name: str,
    label_all_subwords: bool,
    max_seq_length: int = PACKED_MAX_LENGTH,
) -> list[dict[str, Any]]:
    """Greedy-pack sentences within each doc_idx into <=max_seq_length windows.

    Returns a list of windows, each containing:
      - input_ids: packed token ids including <s>/</s> brackets
      - attention_mask: 1s for real positions, 0s for padding (added later)
      - labels: per-subword aligned labels with -100 at specials and continuation
      - sentence_spans: list of (sent_idx, start_in_window, end_in_window)
      - doc_idx: int
    """
    if split_name not in grouped:
        raise ValueError(f"Split `{split_name}` not present in dataset.")

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    if bos_id is None:
        bos_id = tokenizer.cls_token_id
    if eos_id is None:
        eos_id = tokenizer.sep_token_id
    if bos_id is None or eos_id is None:
        raise RuntimeError(
            "Tokenizer must expose either bos/eos or cls/sep special tokens."
        )

    rows_by_doc: dict[int, list[tuple[int, list[str], list[int]]]] = {}
    for row in grouped[split_name]:
        doc_idx = int(row["doc_idx"])
        sent_idx = int(row["sent_idx"])
        tokens = [str(t) for t in row["gold_token"]]
        labels = [int(label) for label in row["gold_label"]]
        rows_by_doc.setdefault(doc_idx, []).append((sent_idx, tokens, labels))

    # Sentence-level slices have a hard cap of (max_seq_length - 2) so that
    # any single sentence still fits in a window with the BOS/EOS overhead.
    per_sentence_budget = max_seq_length - 2

    windows: list[dict[str, Any]] = []

    for doc_idx in sorted(rows_by_doc):
        sentences = sorted(rows_by_doc[doc_idx], key=lambda triple: triple[0])
        cur_input_ids: list[int] = [bos_id]
        cur_labels: list[int] = [-100]
        cur_spans: list[tuple[int, int, int]] = []

        def flush() -> None:
            cur_input_ids.append(eos_id)
            cur_labels.append(-100)
            windows.append(
                {
                    "input_ids": list(cur_input_ids),
                    "labels": list(cur_labels),
                    "attention_mask": [1] * len(cur_input_ids),
                    "sentence_spans": list(cur_spans),
                    "doc_idx": doc_idx,
                }
            )

        for sent_idx, tokens, sent_labels in sentences:
            subword_ids, aligned = _pretokenize_sentence(
                tokenizer=tokenizer,
                tokens=tokens,
                word_labels=sent_labels,
                label_all_subwords=label_all_subwords,
                truncation_budget=per_sentence_budget,
            )
            sent_len = len(subword_ids)

            # +1 here accounts for the trailing </s> we will append at flush.
            if len(cur_input_ids) + sent_len + 1 > max_seq_length:
                if cur_spans:
                    flush()
                cur_input_ids = [bos_id]
                cur_labels = [-100]
                cur_spans = []

            start = len(cur_input_ids)
            cur_input_ids.extend(subword_ids)
            cur_labels.extend(aligned)
            end = len(cur_input_ids)
            cur_spans.append((sent_idx, start, end))

        if cur_spans:
            flush()

    return windows


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------


def _select_inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pad_batch(
    windows: list[dict[str, Any]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Pad a batch of windows to the same length and return tensors."""
    lengths = [len(w["input_ids"]) for w in windows]
    max_len = max(lengths)
    input_ids = torch.full(
        (len(windows), max_len), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((len(windows), max_len), dtype=torch.long)
    for row, window in enumerate(windows):
        L = len(window["input_ids"])
        input_ids[row, :L] = torch.tensor(window["input_ids"], dtype=torch.long)
        attention_mask[row, :L] = 1
    return input_ids, attention_mask, lengths


def run_packed_forward_crf(
    model: RobertaCrfForTokenClassification,
    windows: list[dict[str, Any]],
    tokenizer: Any,
    batch_size: int,
) -> list[np.ndarray]:
    """Run the CRF wrapper's encoder/classifier forward over each window.

    Returns one float32 ndarray per window, of shape (window_len, NUM_LABELS).
    """
    device = _select_inference_device()
    model = model.to(device)
    model.eval()
    pad_id = tokenizer.pad_token_id

    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            input_ids, attention_mask, lengths = _pad_batch(batch, pad_id)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            emissions = outputs["emissions"].detach().cpu().numpy()
            for row, length in enumerate(lengths):
                out.append(emissions[row, :length].astype(np.float32))
    return out


def run_packed_forward_vanilla(
    model: AutoModelForTokenClassification,
    windows: list[dict[str, Any]],
    tokenizer: Any,
    batch_size: int,
) -> list[np.ndarray]:
    """Run a vanilla token-classifier forward over each window."""
    device = _select_inference_device()
    model = model.to(device)
    model.eval()
    pad_id = tokenizer.pad_token_id

    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            input_ids, attention_mask, lengths = _pad_batch(batch, pad_id)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu().numpy()
            for row, length in enumerate(lengths):
                out.append(logits[row, :length].astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# Per-sentence decode
# ---------------------------------------------------------------------------


def _viterbi_decode(
    emissions: np.ndarray,
    transitions: np.ndarray,
    start_transitions: np.ndarray,
    end_transitions: np.ndarray,
) -> np.ndarray:
    """Pure-numpy Viterbi for one sentence's emissions of shape (T, K)."""
    seq_len, _ = emissions.shape
    score = start_transitions + emissions[0]
    backpointers = np.zeros((seq_len, emissions.shape[1]), dtype=np.int64)
    for t in range(1, seq_len):
        broadcast = score[:, None] + transitions + emissions[t][None, :]
        backpointers[t] = np.argmax(broadcast, axis=0)
        score = np.max(broadcast, axis=0)
    score = score + end_transitions
    last_label = int(np.argmax(score))
    best_path = [last_label]
    for t in range(seq_len - 1, 0, -1):
        last_label = int(backpointers[t, last_label])
        best_path.append(last_label)
    best_path.reverse()
    return np.array(best_path, dtype=np.int64)


def slice_per_sentence_crf(
    windows: list[dict[str, Any]],
    window_emissions: list[np.ndarray],
    transitions: np.ndarray,
    start_transitions: np.ndarray,
    end_transitions: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Slice window emissions per sentence and Viterbi-decode each slice."""
    sentence_emissions: list[np.ndarray] = []
    sentence_predictions: list[np.ndarray] = []
    sentence_labels: list[np.ndarray] = []

    for window, emissions in zip(windows, window_emissions):
        for _sent_idx, start, end in window["sentence_spans"]:
            slice_em = emissions[start:end]
            if slice_em.shape[0] == 0:
                continue
            decoded = _viterbi_decode(
                slice_em.astype(np.float32),
                transitions,
                start_transitions,
                end_transitions,
            )
            label_slice = np.array(window["labels"][start:end], dtype=np.int64)
            sentence_emissions.append(slice_em.astype(np.float16))
            sentence_predictions.append(decoded)
            sentence_labels.append(label_slice)

    return sentence_emissions, sentence_predictions, sentence_labels


def slice_per_sentence_vanilla(
    windows: list[dict[str, Any]],
    window_logits: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Slice window logits per sentence and argmax."""
    sentence_logits: list[np.ndarray] = []
    sentence_predictions: list[np.ndarray] = []
    sentence_labels: list[np.ndarray] = []

    for window, logits in zip(windows, window_logits):
        for _sent_idx, start, end in window["sentence_spans"]:
            slice_logits = logits[start:end]
            if slice_logits.shape[0] == 0:
                continue
            decoded = np.argmax(slice_logits, axis=-1).astype(np.int64)
            label_slice = np.array(window["labels"][start:end], dtype=np.int64)
            sentence_logits.append(slice_logits.astype(np.float16))
            sentence_predictions.append(decoded)
            sentence_labels.append(label_slice)

    return sentence_logits, sentence_predictions, sentence_labels


def to_seqeval_strings(
    predictions_per_sentence: list[np.ndarray],
    labels_per_sentence: list[np.ndarray],
) -> tuple[list[list[str]], list[list[str]]]:
    """Drop -100 positions and convert to seqeval string sequences."""
    true_labels: list[list[str]] = []
    predicted_labels: list[list[str]] = []
    for pred_ids, label_ids in zip(predictions_per_sentence, labels_per_sentence):
        decoded_true: list[str] = []
        decoded_pred: list[str] = []
        for pred_id, label_id in zip(pred_ids, label_ids):
            label_id = int(label_id)
            if label_id == -100:
                continue
            decoded_true.append(ID2LABEL[label_id])
            decoded_pred.append(ID2LABEL[int(pred_id)])
        true_labels.append(decoded_true)
        predicted_labels.append(decoded_pred)
    return true_labels, predicted_labels


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_crf_model(checkpoint_dir: Path) -> RobertaCrfForTokenClassification:
    """Reconstruct a CRF wrapper from a `checkpoint-best/` directory.

    The wrapper saves the full state dict (with `backbone.`, `classifier.`,
    `crf.` prefixes) under `pytorch_model.bin`. The constructor instantiates
    a fresh wrapper from the backbone config in `config.json`; the matching
    `AutoModel.from_pretrained` call inside the constructor will warn that
    backbone weights cannot be loaded by name (they are prefixed). We
    silence the warning and let `load_state_dict` overwrite everything.
    """
    hf_logging.set_verbosity_error()
    model = RobertaCrfForTokenClassification(model_name=str(checkpoint_dir))
    state_path = checkpoint_dir / "pytorch_model.bin"
    if not state_path.exists():
        raise FileNotFoundError(
            f"Expected CRF state dict at {state_path}. "
            "Make sure the run was trained via the CRF path."
        )
    state = torch.load(state_path, map_location="cpu")
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        raise RuntimeError(
            f"Missing keys when loading CRF state from {checkpoint_dir}: "
            f"{result.missing_keys[:10]}"
        )
    return model


def load_crf_transitions(
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load saved CRF transition matrices for Viterbi re-decoding."""
    path = run_dir / "crf_transitions.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing CRF transitions at {path}. The CRF training path saves "
            "this artifact at training-end; older runs may need a re-extract."
        )
    with np.load(path) as handle:
        return (
            handle["transitions"].astype(np.float32),
            handle["start_transitions"].astype(np.float32),
            handle["end_transitions"].astype(np.float32),
        )


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    """Read the run's summary.json to recover model_name and label_all_subwords."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Per-run driver
# ---------------------------------------------------------------------------


def reinfer_one_run(
    run_dir: Path,
    output_root: Path,
    mode: str,
    split: str,
    batch_size: int,
    label_all_subwords_override: bool | None,
    write_csv_rows: bool,
) -> dict[str, Any]:
    """Run packed-window re-inference for a single run and write artifacts."""
    if mode not in {"crf", "vanilla"}:
        raise ValueError("mode must be `crf` or `vanilla`.")
    if split not in {"validation", "test"}:
        raise ValueError("split must be `validation` or `test`.")

    summary_meta = load_run_metadata(run_dir)
    config_meta = summary_meta.get("config", {}) if summary_meta else {}
    model_name = config_meta.get("model_name") or str(run_dir / "checkpoint-best")
    if label_all_subwords_override is not None:
        label_all_subwords = label_all_subwords_override
    else:
        label_all_subwords = bool(config_meta.get("label_all_subwords", False))

    checkpoint_dir = run_dir / "checkpoint-best"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, add_prefix_space=True)
    grouped = load_finer_ord()

    print(f"[{run_dir.name}] Building {split} windows (max_seq_length={PACKED_MAX_LENGTH})...")
    windows = build_packed_windows(
        grouped=grouped,
        tokenizer=tokenizer,
        split_name=split,
        label_all_subwords=label_all_subwords,
    )
    n_sentences = sum(len(w["sentence_spans"]) for w in windows)
    print(
        f"[{run_dir.name}] {len(windows)} windows for {n_sentences} sentences "
        f"(packing ratio {n_sentences / max(len(windows), 1):.2f} sent/window)."
    )

    start_time = time.perf_counter()
    if mode == "crf":
        model = load_crf_model(checkpoint_dir)
        window_outputs = run_packed_forward_crf(model, windows, tokenizer, batch_size)
        transitions, start_t, end_t = load_crf_transitions(run_dir)
        sentence_emissions, sentence_preds, sentence_labels = slice_per_sentence_crf(
            windows, window_outputs, transitions, start_t, end_t
        )
        artifact_name = f"{split}_packed_emissions.npz"
        artifact_key = "emissions"
    else:
        model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
        window_outputs = run_packed_forward_vanilla(model, windows, tokenizer, batch_size)
        sentence_emissions, sentence_preds, sentence_labels = slice_per_sentence_vanilla(
            windows, window_outputs
        )
        artifact_name = f"{split}_packed_logits.npz"
        artifact_key = "logits"
    inference_time_min = (time.perf_counter() - start_time) / 60.0

    true_labels, predicted_labels = to_seqeval_strings(sentence_preds, sentence_labels)
    metrics = compute_detailed_metrics(true_labels, predicted_labels, verbose=False)
    print(
        f"[{run_dir.name}] {split} packed-512 entity F1: "
        f"{metrics['entity_overall_f1']:.4f}"
    )

    artifact_path = run_dir / artifact_name
    np.savez_compressed(
        artifact_path,
        **{
            artifact_key: np.array(sentence_emissions, dtype=object),
            "attention_mask": np.array(
                [np.ones(arr.shape[0], dtype=np.uint8) for arr in sentence_emissions],
                dtype=object,
            ),
            "labels": np.array(sentence_labels, dtype=object),
        },
    )

    summary_payload = {
        "run_id": run_dir.name,
        "mode": mode,
        "split": split,
        "max_seq_length": PACKED_MAX_LENGTH,
        "label_all_subwords": label_all_subwords,
        "num_windows": len(windows),
        "num_sentences": n_sentences,
        "inference_time_min": inference_time_min,
        "metrics": metrics,
        "artifact_path": str(artifact_path),
    }
    summary_path = run_dir / f"summary_packed_{split}.json"
    write_json_file(summary_path, summary_payload)

    if write_csv_rows:
        original_model = model_name
        compat_row = {
            "experiment_id": f"{run_dir.name}_packed512_{split}",
            "model": f"{original_model}_packed512",
            "seed": "",
            "config_hash": f"packed512_{mode}",
            "train_f1_val": "",
            "test_entity_f1": (
                float(metrics["entity_overall_f1"]) if split == "test" else ""
            ),
            "test_per_f1": (
                float(metrics["entity_per_class"].get("PER", 0.0))
                if split == "test"
                else ""
            ),
            "test_loc_f1": (
                float(metrics["entity_per_class"].get("LOC", 0.0))
                if split == "test"
                else ""
            ),
            "test_org_f1": (
                float(metrics["entity_per_class"].get("ORG", 0.0))
                if split == "test"
                else ""
            ),
            "params": "",
            "train_time_min": "",
            "notes": (
                f"packed512 | mode={mode} | split={split} | "
                f"source_run={run_dir.name}"
            ),
        }
        detailed_row = {
            "run_id": f"{run_dir.name}_packed512_{split}",
            "model": f"{original_model}_packed512",
            "seed": "",
            "warmup_ratio": "",
            "lr_scheduler": "",
            "label_smoothing": "",
            "test_token_f1": (
                float(metrics["token_weighted_f1"]) if split == "test" else ""
            ),
            "test_entity_f1": (
                float(metrics["entity_overall_f1"]) if split == "test" else ""
            ),
            "test_per_f1": (
                float(metrics["entity_per_class"].get("PER", 0.0))
                if split == "test"
                else ""
            ),
            "test_loc_f1": (
                float(metrics["entity_per_class"].get("LOC", 0.0))
                if split == "test"
                else ""
            ),
            "test_org_f1": (
                float(metrics["entity_per_class"].get("ORG", 0.0))
                if split == "test"
                else ""
            ),
            "train_time_min": "",
        }
        append_csv_row(output_root / "results.csv", COMPAT_RESULTS_HEADER, compat_row)
        append_csv_row(
            output_root / "results_detailed.csv",
            DETAILED_RESULTS_HEADER,
            detailed_row,
        )

    return summary_payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directory names under `results/` (or absolute paths).",
    )
    parser.add_argument(
        "--mode",
        choices=["crf", "vanilla"],
        required=True,
        help="`crf` loads the CRF wrapper + saved transitions; "
        "`vanilla` loads AutoModelForTokenClassification.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "validation"],
        required=True,
        help="`val` and `validation` are equivalent.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Window batch size for the forward pass.",
    )
    parser.add_argument(
        "--output-root",
        default="results",
        help="Root directory for run artifacts (defaults to `results/`).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip appending rows to results.csv / results_detailed.csv.",
    )
    parser.add_argument(
        "--label-all-subwords",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override the run's saved `label_all_subwords` flag.",
    )
    return parser.parse_args()


def resolve_run_dirs(run_arguments: list[str], output_root: Path) -> list[Path]:
    resolved: list[Path] = []
    for token in run_arguments:
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = output_root / token
        if not candidate.exists():
            raise FileNotFoundError(f"Run directory does not exist: {candidate}")
        resolved.append(candidate)
    return resolved


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    run_dirs = resolve_run_dirs(args.runs, output_root)
    split = "validation" if args.split == "val" else args.split

    if args.label_all_subwords == "auto":
        label_all_subwords_override: bool | None = None
    else:
        label_all_subwords_override = args.label_all_subwords == "true"

    summaries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = reinfer_one_run(
            run_dir=run_dir,
            output_root=output_root,
            mode=args.mode,
            split=split,
            batch_size=args.batch_size,
            label_all_subwords_override=label_all_subwords_override,
            write_csv_rows=not args.no_csv,
        )
        summaries.append(summary)

    if len(summaries) > 1:
        f1_values = [s["metrics"]["entity_overall_f1"] for s in summaries]
        f1_mean = sum(f1_values) / len(f1_values)
        print(f"\nPacked-512 {split} entity F1 across {len(summaries)} runs:")
        for summary, f1 in zip(summaries, f1_values):
            print(f"  {summary['run_id']:50s} {f1:.4f}")
        print(f"  {'mean':50s} {f1_mean:.4f}")


if __name__ == "__main__":
    main()
