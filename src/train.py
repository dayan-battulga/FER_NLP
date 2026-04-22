"""
train.py - Vanilla FiNER-ORD training entrypoint.

This file stays config-driven, but it is intentionally written in a
top-down, script-like style so the main training flow is easy to follow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
import yaml
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import ID2LABEL, LABEL2ID, NUM_LABELS, get_dataset_and_tokenizer
from src.evaluate import (
    compute_detailed_metrics,
    compute_seqeval_metrics,
    compute_token_weighted_f1,
)


# -----------------------------------------------------------------------------
# CSV schemas
# -----------------------------------------------------------------------------

COMPAT_RESULTS_HEADER = [
    "experiment_id",
    "model",
    "seed",
    "config_hash",
    "train_f1_val",
    "test_entity_f1",
    "test_per_f1",
    "test_loc_f1",
    "test_org_f1",
    "params",
    "train_time_min",
    "notes",
]

DETAILED_RESULTS_HEADER = [
    "run_id",
    "model",
    "seed",
    "warmup_ratio",
    "lr_scheduler",
    "label_smoothing",
    "test_token_f1",
    "test_entity_f1",
    "test_per_f1",
    "test_loc_f1",
    "test_org_f1",
    "train_time_min",
]


# -----------------------------------------------------------------------------
# Training config
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Structured representation of one YAML training config."""

    model_name: str
    seeds: list[int]
    learning_rate: float
    batch_size: int
    num_epochs: int
    weight_decay: float
    early_stopping_patience: int
    early_stopping_threshold: float
    max_seq_length: int
    fp16: bool
    warmup_ratio: float
    lr_scheduler_type: str
    label_smoothing_factor: float
    wandb_project: str | None = None
    wandb_tags: list[str] | None = None
    output_dir: str = "./results"
    use_crf: bool = False
    crf_learning_rate: float | None = None
    use_distillation: bool = False
    teacher_checkpoint_path: str | None = None
    distillation_temperature: float | None = None
    distillation_alpha: float | None = None
    save_total_limit: int = 2

    def __post_init__(self) -> None:
        # YAML can load numbers and lists in slightly inconsistent ways, so we
        # normalize everything here once. That keeps the rest of the file simple
        # because later code can trust the types on this object.
        self.seeds = [int(seed) for seed in self.seeds]
        self.learning_rate = float(self.learning_rate)
        self.batch_size = int(self.batch_size)
        self.num_epochs = int(self.num_epochs)
        self.weight_decay = float(self.weight_decay)
        self.early_stopping_patience = int(self.early_stopping_patience)
        self.early_stopping_threshold = float(self.early_stopping_threshold)
        self.max_seq_length = int(self.max_seq_length)
        self.fp16 = bool(self.fp16)
        self.warmup_ratio = float(self.warmup_ratio)
        self.label_smoothing_factor = float(self.label_smoothing_factor)
        self.save_total_limit = int(self.save_total_limit)
        self.use_crf = bool(self.use_crf)
        self.use_distillation = bool(self.use_distillation)

        # Treat missing tags as an empty list so later logging code does not
        # need to branch on None vs [].
        if self.wandb_tags is None:
            self.wandb_tags = []
        else:
            self.wandb_tags = [str(tag) for tag in self.wandb_tags]

        # Fail fast on invalid configs before we download models or datasets.
        if not self.model_name:
            raise ValueError("`model_name` must be provided in the config.")
        if not self.seeds:
            raise ValueError("`seeds` must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be > 0.")
        if self.num_epochs <= 0:
            raise ValueError("`num_epochs` must be > 0.")
        if self.early_stopping_patience < 0:
            raise ValueError("`early_stopping_patience` must be >= 0.")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("`warmup_ratio` must be in [0, 1].")
        if self.label_smoothing_factor < 0.0:
            raise ValueError("`label_smoothing_factor` must be >= 0.")
        if self.save_total_limit <= 0:
            raise ValueError("`save_total_limit` must be > 0.")


# -----------------------------------------------------------------------------
# CLI and config loading
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse the CLI flags for the training entrypoint."""

    parser = argparse.ArgumentParser(description="Train FiNER-ORD token classifiers.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--run-checks",
        action="store_true",
        help="Run dataset/tokenization sanity checks before training.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B even if the config enables it.",
    )
    return parser.parse_args()


def load_train_config(config_path: str | Path) -> TrainConfig:
    """Load one YAML config and validate its keys against TrainConfig."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")

    # Reject unknown keys up front so typos never silently change an experiment.
    allowed_keys = {field.name for field in fields(TrainConfig)}
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(f"Unknown config keys in {path}: {joined}")

    return TrainConfig(**payload)


# -----------------------------------------------------------------------------
# Serialization and prediction helpers
# -----------------------------------------------------------------------------


def make_json_safe(value: Any) -> Any:
    """Recursively convert values into JSON-serializable Python objects."""

    if isinstance(value, dict):
        return {str(key): make_json_safe(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [make_json_safe(inner_value) for inner_value in value]
    if isinstance(value, tuple):
        return [make_json_safe(inner_value) for inner_value in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Create a short stable fingerprint for one resolved config."""

    # Sort keys so the same config hashes identically even if dict ordering changes.
    canonical_json = json.dumps(make_json_safe(config_dict), sort_keys=True)

    # A short prefix is easier to read in CSV logs while still distinguishing runs.
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:12]


def decode_token_predictions(
    predictions: np.ndarray | tuple[np.ndarray, ...],
    label_ids: np.ndarray,
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert raw logits and aligned labels into seqeval-ready strings."""

    # Trainer can return logits directly or wrap them in a tuple, so normalize that
    # shape before decoding.
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    predicted_ids = np.argmax(logits, axis=-1)

    true_labels: list[list[str]] = []
    predicted_labels: list[list[str]] = []

    for predicted_sequence, label_sequence in zip(predicted_ids, label_ids):
        decoded_true_sequence: list[str] = []
        decoded_pred_sequence: list[str] = []

        for predicted_id, label_id in zip(predicted_sequence, label_sequence):
            label_id = int(label_id)

            # Hugging Face uses -100 for positions that should not contribute to the
            # loss or metrics, such as special tokens and continuation subwords.
            if label_id == -100:
                continue

            decoded_true_sequence.append(ID2LABEL[label_id])
            decoded_pred_sequence.append(ID2LABEL[int(predicted_id)])

        true_labels.append(decoded_true_sequence)
        predicted_labels.append(decoded_pred_sequence)

    return true_labels, predicted_labels


def write_json_file(path: Path, payload: Any) -> None:
    """Write a JSON file and create parent directories if they do not exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(payload), handle, indent=2)


def append_csv_row(csv_path: Path, header: list[str], row: dict[str, Any]) -> None:
    """Append one row to a CSV file, self-healing if the existing file is
    malformed. Schemas that truly don't match are backed up to a .bak file
    rather than silently overwritten."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    def write_fresh(rows_to_preserve: list[dict[str, Any]]) -> None:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            for preserved in rows_to_preserve:
                writer.writerow({key: preserved.get(key, "") for key in header})

    def append_row() -> None:
        # Make sure the file ends with a newline so the new row can't ever be
        # glued onto the previous line (this is what caused the original bug).
        with csv_path.open("rb") as handle:
            try:
                handle.seek(-1, 2)
                ends_with_newline = handle.read(1) in (b"\n", b"\r")
            except OSError:
                ends_with_newline = True
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            if not ends_with_newline:
                handle.write("\n")
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writerow({key: row.get(key, "") for key in header})

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        write_fresh([])
        append_row()
        return

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        write_fresh([])
        append_row()
        return

    existing_header = [column.strip() for column in rows[0]]
    normalized_expected = [column.strip() for column in header]

    if existing_header == normalized_expected:
        append_row()
        return

    n = len(normalized_expected)

    # Case: header got glued to the first data row (no newline between them),
    # producing a single oversized "header" like [..., 'notesfoo', 'distilbert', ...].
    # The last name in the expected header appears as a prefix on element n-1 of
    # the existing header, and elements n.. are the first data row's values.
    if (
        len(existing_header) >= 2 * n
        and existing_header[: n - 1] == normalized_expected[: n - 1]
        and existing_header[n - 1].startswith(normalized_expected[n - 1])
    ):
        first_value = existing_header[n - 1][len(normalized_expected[n - 1]) :]
        recovered_row_values = [first_value] + existing_header[n : 2 * n - 1]
        recovered_first = dict(zip(normalized_expected, recovered_row_values))
        remaining_rows = [dict(zip(normalized_expected, r)) for r in rows[1:] if r]
        write_fresh([recovered_first, *remaining_rows])
        append_row()
        return

    # Case: schemas are genuinely different. Back up the old file instead of
    # destroying it, then start fresh with the current schema.
    backup_path = csv_path.with_suffix(
        csv_path.suffix + f".bak.{int(time.time())}"
    )
    csv_path.rename(backup_path)
    print(
        f"[append_csv_row] Incompatible existing schema in {csv_path.name}; "
        f"backed up to {backup_path.name} and starting fresh."
    )
    write_fresh([])
    append_row()


def summarize_seed_values(values: list[float]) -> dict[str, float]:
    """Summarize a metric across seeds as mean and std."""

    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(mean(values)), "std": float(stdev(values))}


# -----------------------------------------------------------------------------
# Trainer construction
# -----------------------------------------------------------------------------


def create_trainer(
    config: TrainConfig,
    seed: int,
    run_id: str,
    run_dir: Path,
    dataset: Any,
    tokenizer: Any,
    model: torch.nn.Module,
    use_wandb: bool,
) -> Trainer:
    """Create the Hugging Face Trainer for one seed."""

    def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
        # Trainer calls this at every evaluation step, so it should stay cheap.
        # We intentionally use scalar metrics here instead of the full detailed
        # report because confusion matrices are expensive and only needed once at
        # the end of training.
        true_labels, predicted_labels = decode_token_predictions(
            eval_prediction.predictions,
            eval_prediction.label_ids,
        )
        token_f1 = compute_token_weighted_f1(true_labels, predicted_labels)
        seqeval_metrics = compute_seqeval_metrics(true_labels, predicted_labels)

        return {
            "token_weighted_f1": float(token_f1),
            "entity_overall_f1": float(seqeval_metrics["overall_f1"]),
            "entity_overall_precision": float(seqeval_metrics["overall_precision"]),
            "entity_overall_recall": float(seqeval_metrics["overall_recall"]),
            "entity_per_f1": float(seqeval_metrics.get("PER", {}).get("f1", 0.0)),
            "entity_loc_f1": float(seqeval_metrics.get("LOC", {}).get("f1", 0.0)),
            "entity_org_f1": float(seqeval_metrics.get("ORG", {}).get("f1", 0.0)),
        }

    # Mixed precision is only enabled on CUDA here. That avoids accidental fp16
    # settings on CPU or MPS where the behavior is different or unsupported.
    use_fp16 = bool(config.fp16 and torch.cuda.is_available())

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        run_name=run_id,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=float(config.num_epochs),
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_entity_overall_f1",
        greater_is_better=True,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        label_smoothing_factor=config.label_smoothing_factor,
        save_total_limit=config.save_total_limit,
        fp16=use_fp16,
        report_to=["wandb"] if use_wandb else [],
        seed=seed,
        data_seed=seed,
        remove_unused_columns=True,
        save_safetensors=True,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        ],
    )


# -----------------------------------------------------------------------------
# Seed-level execution
# -----------------------------------------------------------------------------


def run_single_seed(
    config: TrainConfig,
    config_path: str | Path,
    config_stem: str,
    seed: int,
    dataset: Any,
    tokenizer: Any,
    output_root: Path,
    disable_wandb: bool,
) -> dict[str, Any]:
    """Train, evaluate, and log one seed from the config."""

    # Make the entire run reproducible before any model or dataloader state exists.
    set_seed(seed)

    # Use the config stem in the run id so different experiment files for the same
    # backbone do not overwrite each other.
    run_id = f"{config_stem}_seed{seed}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # This is the exact config snapshot that will be written into artifacts for
    # this seed. Keeping the resolved seed here makes later debugging much easier.
    resolved_config = asdict(config)
    resolved_config["seeds"] = [seed]
    resolved_config["resolved_seed"] = seed
    resolved_config["config_path"] = str(config_path)
    resolved_config["run_id"] = run_id

    # -------------------------------------------------------------------------
    # Optional W&B setup
    # -------------------------------------------------------------------------

    use_wandb = False
    if not disable_wandb and config.wandb_project:
        try:
            import wandb

            # Group all seeds from the same config together so the W&B UI mirrors
            # the experiment structure in this repo.
            wandb.init(
                project=config.wandb_project,
                name=run_id,
                group=config_stem,
                tags=config.wandb_tags,
                config=make_json_safe(resolved_config),
                reinit=True,
            )
            use_wandb = True
        except Exception as exc:
            # Tracking should never prevent the actual model run from happening.
            print(f"W&B unavailable for {run_id}: {exc}. Continuing without W&B.")

    # -------------------------------------------------------------------------
    # Model and trainer setup
    # -------------------------------------------------------------------------

    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    trainer = create_trainer(
        config=config,
        seed=seed,
        run_id=run_id,
        run_dir=run_dir,
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        use_wandb=use_wandb,
    )

    try:
        # ---------------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------------

        start_time = time.perf_counter()
        trainer.train()
        train_time_min = (time.perf_counter() - start_time) / 60.0

        # Save a stable "best checkpoint" folder even though Trainer also writes
        # numbered checkpoints. This gives later scripts one predictable path.
        best_checkpoint_dir = run_dir / "checkpoint-best"
        best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(best_checkpoint_dir)
        tokenizer.save_pretrained(best_checkpoint_dir)

        # ---------------------------------------------------------------------
        # Final validation and test evaluation
        # ---------------------------------------------------------------------

        # We recompute final predictions after training so the saved artifacts and
        # summaries always reflect the final best-loaded model, not just metrics
        # cached during an intermediate evaluation step.
        val_output = trainer.predict(
            dataset["validation"],
            metric_key_prefix="validation",
        )
        test_output = trainer.predict(dataset["test"], metric_key_prefix="test")

        val_true_labels, val_predictions = decode_token_predictions(
            val_output.predictions,
            val_output.label_ids,
        )
        test_true_labels, test_predictions = decode_token_predictions(
            test_output.predictions,
            test_output.label_ids,
        )

        # The detailed metric bundle includes confusion matrices and per-class
        # breakdowns, which are useful for reports and error analysis but too
        # heavy to compute inside Trainer every epoch.
        val_metrics = compute_detailed_metrics(
            val_true_labels,
            val_predictions,
            verbose=False,
        )
        test_metrics = compute_detailed_metrics(
            test_true_labels,
            test_predictions,
            verbose=False,
        )

        # ---------------------------------------------------------------------
        # Artifact writing
        # ---------------------------------------------------------------------

        predictions_path = run_dir / "predictions.json"
        summary_path = run_dir / "summary.json"

        write_json_file(
            predictions_path,
            {"true_labels": test_true_labels, "predictions": test_predictions},
        )

        # Parameter count is logged because it feeds directly into later
        # efficiency comparisons and Pareto chart generation.
        param_count = sum(parameter.numel() for parameter in trainer.model.parameters())
        config_hash = compute_config_hash(resolved_config)

        # Prefer Trainer's best metric when available because it comes from the
        # model-selection criterion, but fall back to the final recomputed metric
        # so summary writing is never blocked.
        best_val_f1 = trainer.state.best_metric
        if best_val_f1 is None:
            best_val_f1 = val_metrics["entity_overall_f1"]

        if torch.cuda.is_available():
            device_name = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"

        summary_payload = {
            "run_id": run_id,
            "config": resolved_config,
            "runtime": {
                "device": device_name,
                "fp16_requested": config.fp16,
                "fp16_enabled": bool(config.fp16 and device_name == "cuda"),
            },
            "best_validation_entity_f1": float(best_val_f1),
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "checkpoint_best_path": str(best_checkpoint_dir),
            "param_count": param_count,
            "train_time_min": float(train_time_min),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "predictions_path": str(predictions_path),
        }
        write_json_file(summary_path, summary_payload)

        # ---------------------------------------------------------------------
        # Master CSV logging
        # ---------------------------------------------------------------------

        compat_notes = config_stem
        if config.wandb_tags:
            compat_notes = f"{config_stem} | tags={','.join(config.wandb_tags)}"

        compat_row = {
            "experiment_id": run_id,
            "model": config.model_name,
            "seed": seed,
            "config_hash": config_hash,
            "train_f1_val": float(best_val_f1),
            "test_entity_f1": float(test_metrics["entity_overall_f1"]),
            "test_per_f1": float(test_metrics["entity_per_class"].get("PER", 0.0)),
            "test_loc_f1": float(test_metrics["entity_per_class"].get("LOC", 0.0)),
            "test_org_f1": float(test_metrics["entity_per_class"].get("ORG", 0.0)),
            "params": param_count,
            "train_time_min": float(train_time_min),
            "notes": compat_notes,
        }

        detailed_row = {
            "run_id": run_id,
            "model": config.model_name,
            "seed": seed,
            "warmup_ratio": config.warmup_ratio,
            "lr_scheduler": config.lr_scheduler_type,
            "label_smoothing": config.label_smoothing_factor,
            "test_token_f1": float(test_metrics["token_weighted_f1"]),
            "test_entity_f1": float(test_metrics["entity_overall_f1"]),
            "test_per_f1": float(test_metrics["entity_per_class"].get("PER", 0.0)),
            "test_loc_f1": float(test_metrics["entity_per_class"].get("LOC", 0.0)),
            "test_org_f1": float(test_metrics["entity_per_class"].get("ORG", 0.0)),
            "train_time_min": float(train_time_min),
        }

        append_csv_row(output_root / "results.csv", COMPAT_RESULTS_HEADER, compat_row)
        append_csv_row(
            output_root / "results_detailed.csv",
            DETAILED_RESULTS_HEADER,
            detailed_row,
        )

        return {
            "run_id": run_id,
            "seed": seed,
            "config_hash": config_hash,
            "train_time_min": float(train_time_min),
            "param_count": param_count,
            "summary_path": str(summary_path),
            "predictions_path": str(predictions_path),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
    finally:
        # Close the W&B run explicitly so multi-seed experiments do not leak state
        # across seeds in the same process.
        if use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Multi-seed orchestration
# -----------------------------------------------------------------------------


def run_training(
    config: TrainConfig,
    config_path: str | Path,
    run_checks: bool,
    disable_wandb: bool,
) -> None:
    """Run the full experiment across every seed in the config."""

    # Keep future experiment modes explicit instead of silently ignoring them.
    if config.use_crf:
        raise NotImplementedError(
            "This config enables `use_crf=true`, but CRF training is reserved for "
            "`src/crf_model.py` and is not implemented in `src.train.py` yet."
        )
    if config.use_distillation:
        raise NotImplementedError(
            "This config enables `use_distillation=true`, but distillation is "
            "reserved for `src/distill.py` and is not implemented in `src.train.py` yet."
        )

    config_stem = Path(config_path).stem
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load and tokenize the dataset once because the same tokenized splits are
    # reused for every seed in the experiment.
    dataset, tokenizer, _ = get_dataset_and_tokenizer(
        config.model_name,
        max_length=config.max_seq_length,
        run_checks=run_checks,
    )

    seed_results = []
    for seed in config.seeds:
        seed_results.append(
            run_single_seed(
                config=config,
                config_path=config_path,
                config_stem=config_stem,
                seed=seed,
                dataset=dataset,
                tokenizer=tokenizer,
                output_root=output_root,
                disable_wandb=disable_wandb,
            )
        )

    # Aggregate seed-level results into a small summary file for downstream
    # reporting and quick experiment comparison.
    aggregate_summary = {
        "config_path": str(config_path),
        "config": asdict(config),
        "num_seeds": len(seed_results),
        "run_ids": [result["run_id"] for result in seed_results],
        "seeds": [result["seed"] for result in seed_results],
        "test_metrics": {
            "token_weighted_f1": summarize_seed_values(
                [result["test_metrics"]["token_weighted_f1"] for result in seed_results]
            ),
            "entity_overall_f1": summarize_seed_values(
                [result["test_metrics"]["entity_overall_f1"] for result in seed_results]
            ),
            "entity_per_class": {
                "PER": summarize_seed_values(
                    [
                        result["test_metrics"]["entity_per_class"].get("PER", 0.0)
                        for result in seed_results
                    ]
                ),
                "LOC": summarize_seed_values(
                    [
                        result["test_metrics"]["entity_per_class"].get("LOC", 0.0)
                        for result in seed_results
                    ]
                ),
                "ORG": summarize_seed_values(
                    [
                        result["test_metrics"]["entity_per_class"].get("ORG", 0.0)
                        for result in seed_results
                    ]
                ),
            },
        },
        "validation_metrics": {
            "entity_overall_f1": summarize_seed_values(
                [result["val_metrics"]["entity_overall_f1"] for result in seed_results]
            )
        },
        "train_time_min": summarize_seed_values(
            [result["train_time_min"] for result in seed_results]
        ),
    }
    write_json_file(output_root / f"{config_stem}_aggregate.json", aggregate_summary)


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint used by `python -m src.train`."""

    args = parse_args()
    config = load_train_config(args.config)
    run_training(
        config=config,
        config_path=args.config,
        run_checks=args.run_checks,
        disable_wandb=args.no_wandb,
    )


if __name__ == "__main__":
    main()
