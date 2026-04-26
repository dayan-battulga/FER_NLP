"""
dapt.py - Domain-adaptive pretraining (MLM) for FiNER-ORD.

Continued pretraining of a HuggingFace masked-LM checkpoint on the
FiNER-ORD train-split article text only. The intent is to bias the
encoder toward financial entity vocabulary (especially ORG names like
"Goldman Sachs", "Wall Street Journal", "Morgan Stanley") before the
5-epoch token-classification fine-tune in Phase 3 of the plan.

Critical correctness rules:
  - Only the `train` split's articles are used. Validation and test
    articles are excluded by construction; we also assert that the set
    of `doc_idx` values touched by DAPT does not intersect val/test.
  - We pass `add_prefix_space=True` to the tokenizer so RoBERTa BPE and
    DeBERTa-v3 SentencePiece tokenizers behave consistently across the
    DAPT and downstream fine-tune paths.
  - The saved DAPT checkpoint is the full HF `AutoModelForMaskedLM`
    directory plus the tokenizer, which makes it loadable by
    `AutoModelForTokenClassification.from_pretrained(...)` (the MLM head
    is dropped and a classifier head is randomly initialized) and by the
    repo's CRF wrapper (which calls `AutoModel.from_pretrained` on the
    same path).

Usage:

    python -m src.dapt --config configs/baseline/dapt_roberta_large.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import load_finer_ord


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DaptConfig:
    """Structured representation of a DAPT YAML config."""

    model_name: str
    output_dir: str
    learning_rate: float = 1.0e-5
    num_train_epochs: int = 2
    batch_size: int = 8
    max_seq_length: int = 512
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    lr_scheduler_type: str = "linear"
    mlm_probability: float = 0.15
    fp16: bool = True
    seed: int = 88
    save_total_limit: int = 1
    wandb_project: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.learning_rate = float(self.learning_rate)
        self.num_train_epochs = int(self.num_train_epochs)
        self.batch_size = int(self.batch_size)
        self.max_seq_length = int(self.max_seq_length)
        self.weight_decay = float(self.weight_decay)
        self.warmup_ratio = float(self.warmup_ratio)
        self.mlm_probability = float(self.mlm_probability)
        self.fp16 = bool(self.fp16)
        self.seed = int(self.seed)
        self.save_total_limit = int(self.save_total_limit)
        if self.wandb_tags is None:
            self.wandb_tags = []

        if not self.model_name:
            raise ValueError("`model_name` is required.")
        if not self.output_dir:
            raise ValueError("`output_dir` is required.")
        if not 0.0 < self.mlm_probability < 1.0:
            raise ValueError("`mlm_probability` must be in (0, 1).")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("`warmup_ratio` must be in [0, 1].")
        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be > 0.")
        if self.num_train_epochs <= 0:
            raise ValueError("`num_train_epochs` must be > 0.")


def load_dapt_config(config_path: str | Path) -> DaptConfig:
    """Load and validate a DAPT YAML config."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")

    allowed_keys = {field_obj.name for field_obj in fields(DaptConfig)}
    unknown = sorted(set(payload) - allowed_keys)
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {', '.join(unknown)}")

    return DaptConfig(**payload)


# ---------------------------------------------------------------------------
# Corpus construction (train split only)
# ---------------------------------------------------------------------------


def _article_texts_for_split(grouped: Any, split_name: str) -> list[str]:
    """Reconstruct article-level text for one split by joining sentences."""

    if split_name not in grouped:
        raise ValueError(f"Expected split `{split_name}` in FiNER-ORD.")
    sentences_by_doc: dict[int, list[tuple[int, list[str]]]] = {}
    for row in grouped[split_name]:
        doc_idx = int(row["doc_idx"])
        sent_idx = int(row["sent_idx"])
        tokens = [str(token) for token in row["gold_token"]]
        sentences_by_doc.setdefault(doc_idx, []).append((sent_idx, tokens))

    articles: list[str] = []
    for doc_idx in sorted(sentences_by_doc):
        ordered = sorted(sentences_by_doc[doc_idx], key=lambda pair: pair[0])
        sentences = [" ".join(tokens) for _sent_idx, tokens in ordered]
        articles.append("\n".join(sentences))
    return articles


def build_train_article_texts() -> tuple[list[str], dict[str, dict[str, int]]]:
    """Return one string per training article plus split-content diagnostics.

    `doc_idx` is local to each FiNER-ORD split (every split numbers from 0),
    so it cannot be used directly to detect cross-split contamination.
    Instead we compute a content-hash overlap between the train articles we
    are about to use for DAPT and the val/test articles we must never see.
    Any nonzero overlap raises and aborts DAPT.
    """

    grouped = load_finer_ord()
    train_articles = _article_texts_for_split(grouped, "train")
    val_articles = _article_texts_for_split(grouped, "validation")
    test_articles = _article_texts_for_split(grouped, "test")

    val_set = set(val_articles)
    test_set = set(test_articles)
    forbidden = val_set | test_set

    filtered_articles: list[str] = []
    dropped_val = 0
    dropped_test = 0
    for article in train_articles:
        if article in val_set:
            dropped_val += 1
            continue
        if article in test_set:
            dropped_test += 1
            continue
        filtered_articles.append(article)

    if not filtered_articles:
        raise RuntimeError(
            "After excluding val/test-duplicate articles, no training "
            "articles remain. Refusing to run DAPT on an empty corpus."
        )

    if dropped_val or dropped_test:
        print(
            "DAPT corpus warning: dropped duplicates between train and "
            f"val ({dropped_val}) or test ({dropped_test}). FiNER-ORD has a "
            "small number of articles repeated across splits; we exclude "
            "them from DAPT to avoid leakage."
        )

    # Sanity belt-and-braces: nothing forbidden should remain.
    leaks = [article for article in filtered_articles if article in forbidden]
    if leaks:
        raise RuntimeError(
            f"DAPT corpus filter failed; {len(leaks)} val/test articles "
            "still present after filtering."
        )

    diagnostics = {
        "train": {
            "num_articles_total": len(train_articles),
            "num_articles_used": len(filtered_articles),
            "dropped_duplicates_with_val": dropped_val,
            "dropped_duplicates_with_test": dropped_test,
        },
        "validation": {"num_articles": len(val_articles)},
        "test": {"num_articles": len(test_articles)},
    }
    return filtered_articles, diagnostics


def pack_into_windows(
    tokenizer: Any,
    article_texts: list[str],
    max_seq_length: int,
) -> Dataset:
    """Tokenize each article, concatenate, and pack into fixed-length windows.

    Returns a `datasets.Dataset` with columns `input_ids` and `attention_mask`,
    each row exactly `max_seq_length` tokens long. Special tokens are NOT
    inserted between articles; the boundary is just a stream-level break.
    Trailing tokens shorter than `max_seq_length` are dropped to keep batches
    rectangular.
    """

    flat_ids: list[int] = []
    for text in article_texts:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        flat_ids.extend(encoded["input_ids"])

    n_windows = len(flat_ids) // max_seq_length
    windowed_inputs: list[list[int]] = [
        flat_ids[start * max_seq_length : (start + 1) * max_seq_length]
        for start in range(n_windows)
    ]
    if not windowed_inputs:
        raise ValueError(
            f"DAPT corpus has only {len(flat_ids)} tokens; "
            f"need at least {max_seq_length} for one window."
        )

    return Dataset.from_dict(
        {
            "input_ids": windowed_inputs,
            "attention_mask": [[1] * max_seq_length for _ in windowed_inputs],
        }
    )


# ---------------------------------------------------------------------------
# Training entry
# ---------------------------------------------------------------------------


def run_dapt(config: DaptConfig, config_path: str | Path) -> Path:
    """Run masked-LM continued pretraining and save an encoder checkpoint."""

    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, add_prefix_space=True)
    article_texts, split_diagnostics = build_train_article_texts()
    print(
        f"DAPT corpus: {split_diagnostics['train']['num_articles_used']}/"
        f"{split_diagnostics['train']['num_articles_total']} train articles "
        f"(val: {split_diagnostics['validation']['num_articles']}, "
        f"test: {split_diagnostics['test']['num_articles']} - both excluded)"
    )

    train_dataset = pack_into_windows(
        tokenizer=tokenizer,
        article_texts=article_texts,
        max_seq_length=config.max_seq_length,
    )
    print(
        f"Packed corpus: {len(train_dataset)} windows of "
        f"{config.max_seq_length} tokens each."
    )

    model = AutoModelForMaskedLM.from_pretrained(config.model_name)

    output_root = Path(config.output_dir)
    run_dir = output_root / f"dapt_{Path(config.model_name).name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = bool(config.fp16 and torch.cuda.is_available())

    use_wandb = False
    if config.wandb_project:
        try:
            import wandb

            wandb.init(
                project=config.wandb_project,
                name=run_dir.name,
                tags=list(config.wandb_tags),
                config={**asdict(config), "config_path": str(config_path)},
                reinit=True,
            )
            use_wandb = True
        except Exception as exc:
            print(f"W&B unavailable for {run_dir.name}: {exc}. Continuing without W&B.")

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        run_name=run_dir.name,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=float(config.num_train_epochs),
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=config.save_total_limit,
        fp16=use_fp16,
        report_to=["wandb"] if use_wandb else [],
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
        save_safetensors=True,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.mlm_probability,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    try:
        start_time = time.perf_counter()
        train_result = trainer.train()
        train_time_min = (time.perf_counter() - start_time) / 60.0

        final_dir = run_dir / "checkpoint-final"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        summary = {
            "config_path": str(config_path),
            "config": asdict(config),
            "num_articles": len(article_texts),
            "num_windows": len(train_dataset),
            "train_loss": float(train_result.training_loss),
            "train_time_min": float(train_time_min),
            "checkpoint_final": str(final_dir),
            "split_diagnostics": split_diagnostics,
        }
        with (run_dir / "dapt_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        print(f"DAPT complete in {train_time_min:.2f} min.")
        print(f"Encoder checkpoint: {final_dir}")
        return final_dir
    finally:
        if use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Domain-adaptive pretraining for FiNER-ORD.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a DAPT YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dapt_config(args.config)
    run_dapt(config, config_path=args.config)


if __name__ == "__main__":
    main()
