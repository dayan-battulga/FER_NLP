"""
ensemble_logits.py - Logit/emission/vote ensembling across multiple seeds.

Two ensemble modes are supported:

  1. logit  - Average per-token logits or CRF emissions across N seed runs,
              then decode (argmax for vanilla, Viterbi for CRF). Requires
              `test_logits.npz` (vanilla) or `test_emissions.npz` plus
              `crf_transitions.npz` (CRF) saved per run.

  2. vote   - Per-token majority vote across the saved `predictions.json`
              files of N seed runs. This works on the artifacts already
              committed to the repo and does not require model checkpoints.

Both modes write a summary.json compatible with the rest of the pipeline,
plus a row to results/results.csv via append_csv_row.

Examples:

  # Logit ensemble of CRF teachers (requires test_emissions.npz per seed)
  python scripts/ensemble_logits.py \
      --runs teacher_crf_seed88 teacher_crf_seed5768 teacher_crf_seed78516 \
      --mode logit \
      --use-crf \
      --output-name teacher_crf_ensemble

  # Vote ensemble works on existing predictions.json files
  python scripts/ensemble_logits.py \
      --runs teacher_crf_seed88 teacher_crf_seed5768 teacher_crf_seed78516 \
      --mode vote \
      --output-name teacher_crf_vote_ensemble
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# Make `src.*` importable when run directly as a script from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import ID2LABEL, NUM_LABELS  # noqa: E402
from src.evaluate import compute_detailed_metrics  # noqa: E402
from src.train import (  # noqa: E402
    COMPAT_RESULTS_HEADER,
    DETAILED_RESULTS_HEADER,
    append_csv_row,
    write_json_file,
)


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------


def load_npz_object_arrays(path: Path) -> dict[str, list[np.ndarray]]:
    """Load object-dtype npz produced by save_emissions_npz / save_logits_npz."""

    with np.load(path, allow_pickle=True) as handle:
        return {key: list(handle[key]) for key in handle.files}


def load_predictions_json(path: Path) -> tuple[list[list[str]], list[list[str]]]:
    """Load decoded BIO sequences from predictions.json."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["true_labels"], payload["predictions"]


# ----------------------------------------------------------------------------
# Logit / emission ensemble
# ----------------------------------------------------------------------------


def _assert_aligned(
    arrays_by_run: list[list[np.ndarray]],
    name: str,
) -> None:
    """All runs must produce the same number of examples and same per-example length."""

    first_run = arrays_by_run[0]
    expected_n = len(first_run)
    expected_lengths = [a.shape[0] for a in first_run]
    for run_idx, run in enumerate(arrays_by_run[1:], start=1):
        if len(run) != expected_n:
            raise ValueError(
                f"Run {run_idx} has {len(run)} examples for `{name}`, "
                f"expected {expected_n}."
            )
        run_lengths = [a.shape[0] for a in run]
        if run_lengths != expected_lengths:
            raise ValueError(
                f"Run {run_idx} per-example lengths for `{name}` differ from run 0. "
                "Tokenization must match across seeds for a logit ensemble."
            )


def _assert_labels_match(labels_by_run: list[list[np.ndarray]]) -> None:
    """Gold labels must be byte-for-byte identical across runs."""

    first_run = labels_by_run[0]
    for run_idx, run in enumerate(labels_by_run[1:], start=1):
        for ex_idx, (a, b) in enumerate(zip(first_run, run)):
            if not np.array_equal(a, b):
                raise ValueError(
                    f"Run {run_idx} disagrees with run 0 on gold labels for "
                    f"example {ex_idx}; the same tokenized inputs must be used."
                )


def _stack_to_strings(
    decoded_per_example: list[np.ndarray],
    labels_per_example: list[np.ndarray],
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert (predicted_ids, label_ids) pairs to seqeval string sequences."""

    true_labels: list[list[str]] = []
    predicted_labels: list[list[str]] = []
    for predicted_ids, label_ids in zip(decoded_per_example, labels_per_example):
        decoded_true: list[str] = []
        decoded_pred: list[str] = []
        for pred_id, label_id in zip(predicted_ids, label_ids):
            label_id = int(label_id)
            if label_id == -100:
                continue
            decoded_true.append(ID2LABEL[label_id])
            decoded_pred.append(ID2LABEL[int(pred_id)])
        true_labels.append(decoded_true)
        predicted_labels.append(decoded_pred)
    return true_labels, predicted_labels


def ensemble_vanilla_logits(run_dirs: list[Path], split: str = "test") -> tuple[
    list[list[str]], list[list[str]]
]:
    """Average logits across vanilla runs and argmax."""

    payloads = [
        load_npz_object_arrays(run_dir / f"{split}_logits.npz") for run_dir in run_dirs
    ]
    logits_by_run = [p["logits"] for p in payloads]
    labels_by_run = [p["labels"] for p in payloads]
    _assert_aligned(logits_by_run, "logits")
    _assert_aligned(labels_by_run, "labels")
    _assert_labels_match(labels_by_run)

    n_examples = len(logits_by_run[0])
    decoded_per_example: list[np.ndarray] = []
    for ex_idx in range(n_examples):
        stacked = np.stack(
            [run_logits[ex_idx].astype(np.float32) for run_logits in logits_by_run],
            axis=0,
        )
        averaged = stacked.mean(axis=0)
        decoded_per_example.append(np.argmax(averaged, axis=-1))

    return _stack_to_strings(decoded_per_example, labels_by_run[0])


def _viterbi_decode(
    emissions: np.ndarray,
    transitions: np.ndarray,
    start_transitions: np.ndarray,
    end_transitions: np.ndarray,
) -> np.ndarray:
    """Pure-numpy Viterbi for one sequence of emissions of shape (T, K)."""

    seq_len, num_labels = emissions.shape
    score = start_transitions + emissions[0]
    backpointers = np.zeros((seq_len, num_labels), dtype=np.int64)
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


def ensemble_crf_emissions(run_dirs: list[Path], split: str = "test") -> tuple[
    list[list[str]], list[list[str]]
]:
    """Average emissions and CRF transitions across CRF runs, then Viterbi-decode."""

    emission_payloads = [
        load_npz_object_arrays(run_dir / f"{split}_emissions.npz")
        for run_dir in run_dirs
    ]
    emissions_by_run = [p["emissions"] for p in emission_payloads]
    labels_by_run = [p["labels"] for p in emission_payloads]
    _assert_aligned(emissions_by_run, "emissions")
    _assert_aligned(labels_by_run, "labels")
    _assert_labels_match(labels_by_run)

    transitions_stack = []
    start_stack = []
    end_stack = []
    for run_dir in run_dirs:
        with np.load(run_dir / "crf_transitions.npz") as handle:
            transitions_stack.append(handle["transitions"].astype(np.float32))
            start_stack.append(handle["start_transitions"].astype(np.float32))
            end_stack.append(handle["end_transitions"].astype(np.float32))
    avg_transitions = np.mean(np.stack(transitions_stack, axis=0), axis=0)
    avg_start = np.mean(np.stack(start_stack, axis=0), axis=0)
    avg_end = np.mean(np.stack(end_stack, axis=0), axis=0)

    n_examples = len(emissions_by_run[0])
    decoded_per_example: list[np.ndarray] = []
    for ex_idx in range(n_examples):
        stacked = np.stack(
            [run_em[ex_idx].astype(np.float32) for run_em in emissions_by_run],
            axis=0,
        )
        averaged = stacked.mean(axis=0)
        decoded_per_example.append(
            _viterbi_decode(averaged, avg_transitions, avg_start, avg_end)
        )

    return _stack_to_strings(decoded_per_example, labels_by_run[0])


# ----------------------------------------------------------------------------
# Vote ensemble (works on saved predictions.json)
# ----------------------------------------------------------------------------


def ensemble_votes(run_dirs: list[Path]) -> tuple[list[list[str]], list[list[str]]]:
    """Per-token majority vote across saved BIO predictions."""

    runs = [load_predictions_json(run_dir / "predictions.json") for run_dir in run_dirs]

    first_true = runs[0][0]
    for run_idx, (true_labels, predictions) in enumerate(runs):
        if len(true_labels) != len(first_true):
            raise ValueError(
                f"Run {run_idx} has {len(true_labels)} sentences, "
                f"expected {len(first_true)}."
            )
        for sent_idx, (t_seq, ref_seq) in enumerate(zip(true_labels, first_true)):
            if len(t_seq) != len(ref_seq):
                raise ValueError(
                    f"Run {run_idx} sentence {sent_idx} has length "
                    f"{len(t_seq)} != reference {len(ref_seq)}; "
                    "predictions are not aligned."
                )
            if t_seq != ref_seq:
                raise ValueError(
                    f"Run {run_idx} sentence {sent_idx} disagrees with the "
                    "reference gold labels; predictions must come from the "
                    "same dataset/tokenization."
                )
        for sent_idx, (p_seq, t_seq) in enumerate(zip(predictions, true_labels)):
            if len(p_seq) != len(t_seq):
                raise ValueError(
                    f"Run {run_idx} sentence {sent_idx}: prediction length "
                    f"{len(p_seq)} != gold length {len(t_seq)}."
                )

    voted_predictions: list[list[str]] = []
    for sent_idx in range(len(first_true)):
        sentence_length = len(first_true[sent_idx])
        voted_sentence: list[str] = []
        for token_idx in range(sentence_length):
            tally = Counter(run[1][sent_idx][token_idx] for run in runs)
            best_label, _count = tally.most_common(1)[0]
            voted_sentence.append(best_label)
        voted_predictions.append(voted_sentence)

    return list(first_true), voted_predictions


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------


def _seed_from_run_id(run_id: str) -> int | None:
    """Extract the trailing seed integer from a run_id like `..._seed5768`."""

    if "_seed" not in run_id:
        return None
    suffix = run_id.rsplit("_seed", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def write_ensemble_artifacts(
    output_root: Path,
    output_name: str,
    run_dirs: list[Path],
    mode: str,
    use_crf: bool,
    metrics: dict[str, Any],
    write_csv_rows: bool,
) -> Path:
    """Persist summary.json (and optional CSV rows) for one ensemble run."""

    out_dir = output_root / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "run_id": output_name,
        "ensemble_mode": mode,
        "use_crf": use_crf,
        "source_runs": [str(p) for p in run_dirs],
        "test_metrics": metrics,
    }
    summary_path = out_dir / "summary.json"
    write_json_file(summary_path, summary_payload)

    if write_csv_rows:
        first_run_id = run_dirs[0].name
        seed_value = _seed_from_run_id(first_run_id)
        compat_row = {
            "experiment_id": output_name,
            "model": "ensemble",
            "seed": seed_value if seed_value is not None else "",
            "config_hash": f"ensemble_{mode}",
            "train_f1_val": "",
            "test_entity_f1": float(metrics["entity_overall_f1"]),
            "test_per_f1": float(metrics["entity_per_class"].get("PER", 0.0)),
            "test_loc_f1": float(metrics["entity_per_class"].get("LOC", 0.0)),
            "test_org_f1": float(metrics["entity_per_class"].get("ORG", 0.0)),
            "params": "",
            "train_time_min": "",
            "notes": f"ensemble | mode={mode} | use_crf={use_crf} | runs={','.join(p.name for p in run_dirs)}",
        }
        detailed_row = {
            "run_id": output_name,
            "model": "ensemble",
            "seed": "",
            "warmup_ratio": "",
            "lr_scheduler": "",
            "label_smoothing": "",
            "test_token_f1": float(metrics["token_weighted_f1"]),
            "test_entity_f1": float(metrics["entity_overall_f1"]),
            "test_per_f1": float(metrics["entity_per_class"].get("PER", 0.0)),
            "test_loc_f1": float(metrics["entity_per_class"].get("LOC", 0.0)),
            "test_org_f1": float(metrics["entity_per_class"].get("ORG", 0.0)),
            "train_time_min": "",
        }
        append_csv_row(output_root / "results.csv", COMPAT_RESULTS_HEADER, compat_row)
        append_csv_row(
            output_root / "results_detailed.csv", DETAILED_RESULTS_HEADER, detailed_row
        )

    return summary_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directory names under `results/` (or absolute paths) to ensemble.",
    )
    parser.add_argument(
        "--mode",
        choices=["logit", "vote"],
        default="logit",
        help="`logit` averages emissions/logits before decoding; "
        "`vote` does per-token majority vote on predictions.json.",
    )
    parser.add_argument(
        "--use-crf",
        action="store_true",
        help="Logit mode only: load test_emissions.npz + crf_transitions.npz "
        "and Viterbi-decode the averaged emissions.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "validation"],
        default="test",
        help="Which split to evaluate. `val` is treated as `validation`.",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Subdirectory name under `results/` for the ensemble artifacts.",
    )
    parser.add_argument(
        "--output-root",
        default="results",
        help="Root directory for results (defaults to `results/`).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip appending rows to results.csv / results_detailed.csv.",
    )
    return parser.parse_args()


def resolve_run_dirs(run_arguments: list[str], output_root: Path) -> list[Path]:
    """Resolve each --runs token as either an absolute path or `output_root/<name>`."""

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

    if args.mode == "logit":
        if args.use_crf:
            true_labels, predictions = ensemble_crf_emissions(run_dirs, split=split)
        else:
            true_labels, predictions = ensemble_vanilla_logits(run_dirs, split=split)
    else:
        if split != "test":
            raise ValueError(
                "--mode vote currently only operates on test predictions.json. "
                "Use --mode logit for validation-split ensembling."
            )
        true_labels, predictions = ensemble_votes(run_dirs)

    metrics = compute_detailed_metrics(true_labels, predictions, verbose=True)
    summary_path = write_ensemble_artifacts(
        output_root=output_root,
        output_name=args.output_name,
        run_dirs=run_dirs,
        mode=args.mode,
        use_crf=args.use_crf,
        metrics=metrics,
        write_csv_rows=not args.no_csv,
    )
    print(f"\nWrote {summary_path}")
    print(f"Test entity F1: {metrics['entity_overall_f1']:.4f}")
    print(f"Token weighted F1: {metrics['token_weighted_f1']:.4f}")
    for entity_type in ["PER", "LOC", "ORG"]:
        f1 = metrics["entity_per_class"].get(entity_type, 0.0)
        print(f"  {entity_type}: {f1:.4f}")


if __name__ == "__main__":
    main()
