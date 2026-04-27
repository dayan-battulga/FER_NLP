"""
distill.py - Offline student distillation for FiNER-ORD.

The Phase C student intentionally does not use a CRF. The locked teacher gets
its quality from RoBERTa-large + FiNER DAPT + CRF, but the deployable student
story is a vanilla DistilRoBERTa token classifier that is smaller, faster, and
straightforward to quantize.

Distillation uses saved teacher emissions only. The teacher is not forwarded
during student training. Teacher emissions must be aligned to the student
tokens: the locked extraction path uses RoBERTa BPE with add_prefix_space=True,
max_seq_length=256, and label_all_subwords=false. distilroberta-base shares the
same BPE vocabulary, so positions align. A future student with a different
tokenizer must re-extract emissions.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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
from src.train import (
    COMPAT_RESULTS_HEADER,
    DETAILED_RESULTS_HEADER,
    TrainConfig,
    append_csv_row,
    compute_config_hash,
    decode_token_predictions,
    extract_token_logits,
    load_train_config,
    make_json_safe,
    save_logits_npz,
    summarize_seed_values,
    write_json_file,
)


def _load_npz_object_arrays(path: Path) -> dict[str, list[np.ndarray]]:
    """Load the object-array schema used by CRF emission artifacts."""
    with np.load(path, allow_pickle=True) as handle:
        return {key: list(handle[key]) for key in handle.files}


def _assert_aligned(arrays_by_run: list[list[np.ndarray]], name: str) -> None:
    """Require identical example counts and per-example lengths across runs."""
    first = arrays_by_run[0]
    expected_lengths = [array.shape[0] for array in first]
    for run_idx, arrays in enumerate(arrays_by_run[1:], start=1):
        if len(arrays) != len(first):
            raise ValueError(
                f"Teacher run {run_idx} has {len(arrays)} examples for `{name}`, "
                f"expected {len(first)}."
            )
        lengths = [array.shape[0] for array in arrays]
        if lengths != expected_lengths:
            raise ValueError(
                f"Teacher run {run_idx} has different per-example lengths for `{name}`."
            )


def _assert_labels_match(labels_by_run: list[list[np.ndarray]]) -> None:
    """Require byte-identical gold labels across teacher emission dumps."""
    reference = labels_by_run[0]
    for run_idx, labels in enumerate(labels_by_run[1:], start=1):
        for example_idx, (left, right) in enumerate(zip(reference, labels)):
            if not np.array_equal(left, right):
                raise ValueError(
                    f"Teacher run {run_idx} label mismatch at train example {example_idx}."
                )


def load_teacher_emissions(
    output_root: Path,
    teacher_runs: list[str],
    teacher_mode: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load and optionally average teacher train emissions."""
    if teacher_mode == "single" and len(teacher_runs) != 1:
        raise ValueError("`teacher_mode=single` requires exactly one teacher run.")

    payloads = []
    for run in teacher_runs:
        run_path = Path(run)
        if not run_path.is_absolute():
            run_path = output_root / run
        emission_path = run_path / "train_emissions.npz"
        if not emission_path.exists():
            raise FileNotFoundError(f"Missing teacher emissions: {emission_path}")
        payloads.append(_load_npz_object_arrays(emission_path))

    emissions_by_run = [payload["emissions"] for payload in payloads]
    labels_by_run = [payload["labels"] for payload in payloads]
    _assert_aligned(emissions_by_run, "emissions")
    _assert_aligned(labels_by_run, "labels")
    _assert_labels_match(labels_by_run)

    if teacher_mode == "single":
        averaged = [array.astype(np.float32) for array in emissions_by_run[0]]
    else:
        averaged = []
        for example_idx in range(len(emissions_by_run[0])):
            stacked = np.stack(
                [
                    run_emissions[example_idx].astype(np.float32)
                    for run_emissions in emissions_by_run
                ],
                axis=0,
            )
            averaged.append(stacked.mean(axis=0).astype(np.float32))

    return averaged, [label.astype(np.int64) for label in labels_by_run[0]]


def validate_teacher_alignment(dataset: Any, teacher_labels: list[np.ndarray]) -> None:
    """Check saved teacher labels against the tokenized student train split."""
    if len(dataset["train"]) != len(teacher_labels):
        raise ValueError(
            f"Student train split has {len(dataset['train'])} examples, "
            f"teacher emissions have {len(teacher_labels)}."
        )

    for idx, teacher_label in enumerate(teacher_labels):
        student_labels = np.asarray(dataset["train"][idx]["labels"], dtype=np.int64)
        if not np.array_equal(student_labels, teacher_label):
            raise ValueError(
                "Teacher emissions are not aligned to the student tokenization at "
                f"train example {idx}."
            )


def add_teacher_indices(dataset: Any) -> Any:
    """Add stable train indices so the collator can look up emissions."""

    def with_index(_example: dict[str, Any], idx: int) -> dict[str, int]:
        return {"teacher_index": idx}

    dataset = dataset.copy()
    dataset["train"] = dataset["train"].map(with_index, with_indices=True)
    return dataset


class DistillationCollator:
    """Token collator that pads teacher emissions alongside normal labels."""

    def __init__(self, tokenizer: Any, teacher_emissions: list[np.ndarray]) -> None:
        self.base_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        self.teacher_emissions = teacher_emissions

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        copied = [dict(feature) for feature in features]
        has_teacher_indices = all("teacher_index" in feature for feature in copied)
        teacher_indices = []
        if has_teacher_indices:
            teacher_indices = [int(feature.pop("teacher_index")) for feature in copied]
        batch = self.base_collator(copied)
        if not has_teacher_indices:
            return batch

        batch_size, seq_len = batch["input_ids"].shape
        padded = torch.zeros((batch_size, seq_len, NUM_LABELS), dtype=torch.float32)
        for row, teacher_idx in enumerate(teacher_indices):
            emissions = torch.as_tensor(
                self.teacher_emissions[teacher_idx],
                dtype=torch.float32,
            )
            length = min(seq_len, emissions.shape[0])
            padded[row, :length] = emissions[:length]
        batch["teacher_emissions"] = padded
        return batch


class DistillationTrainer(Trainer):
    """Trainer using alpha * CE + (1 - alpha) * T^2 * KL."""

    def __init__(
        self,
        *args: Any,
        temperature: float,
        alpha: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.temperature = float(temperature)
        self.alpha = float(alpha)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        labels = inputs.pop("labels")
        teacher_emissions = inputs.pop("teacher_emissions", None)
        outputs = model(**inputs)
        logits = outputs.logits

        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        if teacher_emissions is None:
            inputs["labels"] = labels
            return (ce_loss, outputs) if return_outputs else ce_loss

        teacher_emissions = teacher_emissions.to(model.device)
        mask = labels != -100
        if mask.any():
            student_active = logits[mask] / self.temperature
            teacher_active = teacher_emissions[mask] / self.temperature
            soft_loss = F.kl_div(
                F.log_softmax(student_active, dim=-1),
                F.softmax(teacher_active, dim=-1),
                reduction="batchmean",
            )
            soft_loss = soft_loss * (self.temperature**2)
        else:
            soft_loss = ce_loss.new_zeros(())

        loss = self.alpha * ce_loss + (1.0 - self.alpha) * soft_loss
        inputs["labels"] = labels
        inputs["teacher_emissions"] = teacher_emissions
        return (loss, outputs) if return_outputs else loss


def create_distillation_trainer(
    config: TrainConfig,
    seed: int,
    run_id: str,
    run_dir: Path,
    dataset: Any,
    tokenizer: Any,
    model: torch.nn.Module,
    teacher_emissions: list[np.ndarray],
    use_wandb: bool,
) -> DistillationTrainer:
    """Build the Hugging Face Trainer for one distilled student seed."""

    def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
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
        remove_unused_columns=False,
        save_safetensors=True,
    )

    return DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DistillationCollator(tokenizer, teacher_emissions),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        ],
        temperature=config.distill_temperature,
        alpha=config.distill_alpha,
    )


def run_single_seed_distillation(
    config: TrainConfig,
    config_path: str | Path,
    config_stem: str,
    seed: int,
    dataset: Any,
    tokenizer: Any,
    teacher_emissions: list[np.ndarray],
    output_root: Path,
    disable_wandb: bool,
) -> dict[str, Any]:
    """Train, evaluate, and log one distilled student seed."""
    set_seed(seed)
    run_id = f"{config_stem}_seed{seed}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = asdict(config)
    resolved_config["seeds"] = [seed]
    resolved_config["resolved_seed"] = seed
    resolved_config["config_path"] = str(config_path)
    resolved_config["run_id"] = run_id

    use_wandb = False
    if not disable_wandb and config.wandb_project:
        try:
            import wandb

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
            print(f"W&B unavailable for {run_id}: {exc}. Continuing without W&B.")

    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    trainer = create_distillation_trainer(
        config=config,
        seed=seed,
        run_id=run_id,
        run_dir=run_dir,
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        teacher_emissions=teacher_emissions,
        use_wandb=use_wandb,
    )

    try:
        start_time = time.perf_counter()
        trainer.train()
        train_time_min = (time.perf_counter() - start_time) / 60.0

        best_checkpoint_dir = run_dir / "checkpoint-best"
        best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(best_checkpoint_dir)
        tokenizer.save_pretrained(best_checkpoint_dir)

        val_output = trainer.predict(dataset["validation"], metric_key_prefix="validation")
        test_output = trainer.predict(dataset["test"], metric_key_prefix="test")
        val_true_labels, val_predictions = decode_token_predictions(
            val_output.predictions,
            val_output.label_ids,
        )
        test_true_labels, test_predictions = decode_token_predictions(
            test_output.predictions,
            test_output.label_ids,
        )
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

        predictions_path = run_dir / "predictions.json"
        summary_path = run_dir / "summary.json"
        write_json_file(
            predictions_path,
            {"true_labels": test_true_labels, "predictions": test_predictions},
        )

        try:
            test_logits, test_masks, test_labels = extract_token_logits(
                trainer.model,
                dataset["test"],
                tokenizer,
                batch_size=config.batch_size,
            )
            save_logits_npz(
                run_dir / "test_logits.npz",
                test_logits,
                test_masks,
                test_labels,
            )
            val_logits, val_masks, val_labels = extract_token_logits(
                trainer.model,
                dataset["validation"],
                tokenizer,
                batch_size=config.batch_size,
            )
            save_logits_npz(
                run_dir / "val_logits.npz",
                val_logits,
                val_masks,
                val_labels,
            )
        except Exception as exc:
            print(f"[{run_id}] Failed to save logits: {exc}")

        param_count = sum(parameter.numel() for parameter in trainer.model.parameters())
        config_hash = compute_config_hash(resolved_config)
        best_val_f1 = trainer.state.best_metric
        if best_val_f1 is None:
            best_val_f1 = val_metrics["entity_overall_f1"]

        if torch.cuda.is_available():
            device_name = "cuda"
            gpu_type = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_name = "mps"
            gpu_type = None
        else:
            device_name = "cpu"
            gpu_type = None

        summary_payload = {
            "run_id": run_id,
            "config": resolved_config,
            "runtime": {
                "device": device_name,
                "gpu_type": gpu_type,
                "fp16_requested": config.fp16,
                "fp16_enabled": bool(config.fp16 and device_name == "cuda"),
                "uses_distillation": True,
                "teacher_mode": config.teacher_mode,
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
            "notes": f"{config_stem} | distillation | teacher_mode={config.teacher_mode}",
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
        if use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass


def run_distillation(
    config: TrainConfig,
    config_path: str | Path,
    run_checks: bool,
    disable_wandb: bool,
    smoke: bool = False,
) -> None:
    """Run offline distillation across every requested student seed."""
    if not config.use_distillation:
        raise ValueError("run_distillation called with use_distillation=False.")
    if config.use_crf:
        raise ValueError("Distillation is incompatible with use_crf=True.")

    if smoke:
        config = replace(config, seeds=[config.seeds[0]], num_epochs=1)

    config_stem = Path(config_path).stem
    if smoke:
        config_stem = f"{config_stem}_smoke"

    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset, tokenizer, _ = get_dataset_and_tokenizer(
        config.model_name,
        max_length=config.max_seq_length,
        run_checks=run_checks,
        label_all_subwords=config.label_all_subwords,
    )
    teacher_emissions, teacher_labels = load_teacher_emissions(
        output_root=output_root,
        teacher_runs=config.teacher_runs,
        teacher_mode=config.teacher_mode,
    )
    validate_teacher_alignment(dataset, teacher_labels)
    dataset = add_teacher_indices(dataset)

    if smoke:
        smoke_n = min(100, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(smoke_n))
        teacher_emissions = teacher_emissions[:smoke_n]
        dataset["validation"] = dataset["validation"].select(
            range(min(100, len(dataset["validation"])))
        )
        dataset["test"] = dataset["test"].select(range(min(100, len(dataset["test"])))
        )

    seed_results = []
    for seed in config.seeds:
        seed_results.append(
            run_single_seed_distillation(
                config=config,
                config_path=config_path,
                config_stem=config_stem,
                seed=seed,
                dataset=dataset,
                tokenizer=tokenizer,
                teacher_emissions=teacher_emissions,
                output_root=output_root,
                disable_wandb=disable_wandb,
            )
        )

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


def parse_args() -> argparse.Namespace:
    """Parse the distillation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    parser.add_argument(
        "--run-checks",
        action="store_true",
        help="Run dataset/tokenization sanity checks before training.",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one seed for one epoch on 100 training sentences.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint used by `python -m src.distill`."""
    args = parse_args()
    config = load_train_config(args.config)
    run_distillation(
        config=config,
        config_path=args.config,
        run_checks=args.run_checks,
        disable_wandb=args.no_wandb,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
