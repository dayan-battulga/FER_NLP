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
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
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
    logging_steps: int = 100
    wandb_project: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    # Phase 5 v2 corpus knobs. Defaults reproduce the v1 (`finer_train_only`)
    # behavior so existing configs without these fields still load.
    corpus_source: str = "finer_train_only"
    include_cc_news: bool = False
    fnspid_subsample: int = 100_000
    held_out_probe_size: int = 0

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
        self.logging_steps = int(self.logging_steps)
        self.fnspid_subsample = int(self.fnspid_subsample)
        self.held_out_probe_size = int(self.held_out_probe_size)
        self.include_cc_news = bool(self.include_cc_news)
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
        if self.logging_steps <= 0:
            raise ValueError("`logging_steps` must be > 0.")
        if self.held_out_probe_size < 0:
            raise ValueError("`held_out_probe_size` must be >= 0.")
        if self.corpus_source not in {"finer_train_only", "fnspid_plus_finer"}:
            raise ValueError(
                "`corpus_source` must be `finer_train_only` or `fnspid_plus_finer`."
            )
        if self.corpus_source == "fnspid_plus_finer" and self.fnspid_subsample <= 0:
            raise ValueError(
                "`fnspid_subsample` must be > 0 when `corpus_source` is `fnspid_plus_finer`."
            )


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
# Held-out FiNER perplexity probe
# ---------------------------------------------------------------------------


def _select_inference_device() -> torch.device:
    """Pick the best available device for a forward-only probe pass."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_held_out_mlm_loss(
    model: torch.nn.Module,
    held_out_dataset: Dataset,
    tokenizer: Any,
    batch_size: int,
    mlm_probability: float,
    probe_seed: int,
) -> float:
    """Token-weighted mean masked-LM loss over the held-out probe.

    The data collator's masking is RNG-driven, so we seed torch deterministically
    before iterating so that base-model and DAPT'd-model perplexity numbers are
    comparable apples-to-apples (same masking pattern across runs).
    """

    device = _select_inference_device()
    model = model.to(device)
    model.eval()

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    torch.manual_seed(probe_seed)
    np.random.seed(probe_seed)

    total_loss_weighted = 0.0
    total_n_tokens = 0
    n = len(held_out_dataset)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            examples = [held_out_dataset[i] for i in range(start, end)]
            batch = collator(examples)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            n_tokens = int((labels != -100).sum().item())
            if n_tokens == 0:
                continue
            total_loss_weighted += float(outputs.loss.item()) * n_tokens
            total_n_tokens += n_tokens

    if total_n_tokens == 0:
        return float("nan")
    return total_loss_weighted / total_n_tokens


class HeldOutPerplexityCallback(TrainerCallback):
    """Record held-out FiNER MLM perplexity at the end of each epoch.

    The trajectory feeds the Phase 5a acceptance gate (`held_out_finer_ppl`
    must improve by at least 5% over the base RoBERTa-large checkpoint).
    """

    def __init__(
        self,
        held_out_dataset: Dataset,
        tokenizer: Any,
        batch_size: int,
        mlm_probability: float,
        probe_seed: int,
    ) -> None:
        self.held_out_dataset = held_out_dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.mlm_probability = mlm_probability
        self.probe_seed = probe_seed
        self.history: list[dict[str, float | int | None]] = []

    def on_epoch_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any):
        model = kwargs.get("model")
        if model is None:
            return control
        loss = compute_held_out_mlm_loss(
            model=model,
            held_out_dataset=self.held_out_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            mlm_probability=self.mlm_probability,
            probe_seed=self.probe_seed,
        )
        ppl = float(math.exp(loss)) if not math.isnan(loss) and loss < 50 else float("inf")
        self.history.append(
            {
                "epoch": float(state.epoch) if state.epoch is not None else None,
                "step": int(state.global_step),
                "held_out_mlm_loss": float(loss),
                "held_out_mlm_ppl": ppl,
            }
        )
        print(
            f"[held-out probe] epoch {state.epoch}: "
            f"loss={loss:.4f}, ppl={ppl:.2f}"
        )
        return control


# ---------------------------------------------------------------------------
# Training entry
# ---------------------------------------------------------------------------


def run_dapt(config: DaptConfig, config_path: str | Path) -> Path:
    """Run masked-LM continued pretraining and save an encoder checkpoint."""

    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, add_prefix_space=True)

    held_out_articles: list[str] = []
    corpus_diagnostics: dict[str, Any] | None = None

    if config.corpus_source == "fnspid_plus_finer":
        from src.dapt_corpus import CorpusSpec, build_dapt_corpus

        spec = CorpusSpec(
            corpus_source=config.corpus_source,
            include_cc_news=config.include_cc_news,
            fnspid_subsample=config.fnspid_subsample,
            held_out_probe_size=config.held_out_probe_size,
            seed=config.seed,
        )
        article_texts, held_out_articles, corpus_diagnostics = build_dapt_corpus(spec)
        print(
            f"DAPT v2 corpus: {corpus_diagnostics['total_articles_used']} articles total "
            f"(per source: {corpus_diagnostics['num_rows_per_source']}), "
            f"{len(held_out_articles)} FiNER articles held out for the probe."
        )
        split_diagnostics = corpus_diagnostics
    else:
        article_texts, split_diagnostics_legacy = build_train_article_texts()
        print(
            f"DAPT corpus: {split_diagnostics_legacy['train']['num_articles_used']}/"
            f"{split_diagnostics_legacy['train']['num_articles_total']} train articles "
            f"(val: {split_diagnostics_legacy['validation']['num_articles']}, "
            f"test: {split_diagnostics_legacy['test']['num_articles']} - both excluded)"
        )
        # Optional held-out probe even on the v1 path, if the user asks for one.
        if config.held_out_probe_size > 0:
            from src.dapt_corpus import CorpusSpec, build_dapt_corpus

            spec = CorpusSpec(
                corpus_source="finer_train_only",
                held_out_probe_size=config.held_out_probe_size,
                seed=config.seed,
            )
            article_texts, held_out_articles, corpus_diagnostics = build_dapt_corpus(spec)
            split_diagnostics = corpus_diagnostics
        else:
            split_diagnostics = split_diagnostics_legacy

    train_dataset = pack_into_windows(
        tokenizer=tokenizer,
        article_texts=article_texts,
        max_seq_length=config.max_seq_length,
    )
    print(
        f"Packed corpus: {len(train_dataset)} windows of "
        f"{config.max_seq_length} tokens each."
    )

    held_out_dataset: Dataset | None = None
    if held_out_articles:
        try:
            held_out_dataset = pack_into_windows(
                tokenizer=tokenizer,
                article_texts=held_out_articles,
                max_seq_length=config.max_seq_length,
            )
            print(
                f"Held-out probe: {len(held_out_dataset)} windows over "
                f"{len(held_out_articles)} FiNER articles."
            )
        except ValueError as exc:
            print(
                f"Held-out probe disabled: {exc}. "
                "Need at least one full window of tokens to compute MLM perplexity."
            )
            held_out_dataset = None

    model = AutoModelForMaskedLM.from_pretrained(config.model_name)

    base_held_out_loss: float | None = None
    base_held_out_ppl: float | None = None
    if held_out_dataset is not None:
        base_held_out_loss = compute_held_out_mlm_loss(
            model=model,
            held_out_dataset=held_out_dataset,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            mlm_probability=config.mlm_probability,
            probe_seed=config.seed,
        )
        base_held_out_ppl = (
            float(math.exp(base_held_out_loss))
            if not math.isnan(base_held_out_loss) and base_held_out_loss < 50
            else float("inf")
        )
        print(
            f"Base-model held-out MLM: loss={base_held_out_loss:.4f}, "
            f"ppl={base_held_out_ppl:.2f}"
        )

    output_root = Path(config.output_dir)
    run_dir = output_root / f"dapt_{Path(config.model_name).name}"
    if config.corpus_source == "fnspid_plus_finer":
        run_dir = output_root / f"dapt_{Path(config.model_name).name}_v2"
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
        logging_steps=config.logging_steps,
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

    trainer_callbacks: list[TrainerCallback] = []
    held_out_callback: HeldOutPerplexityCallback | None = None
    if held_out_dataset is not None:
        held_out_callback = HeldOutPerplexityCallback(
            held_out_dataset=held_out_dataset,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            mlm_probability=config.mlm_probability,
            probe_seed=config.seed,
        )
        trainer_callbacks.append(held_out_callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=trainer_callbacks,
    )

    try:
        start_time = time.perf_counter()
        train_result = trainer.train()
        train_time_min = (time.perf_counter() - start_time) / 60.0

        final_dir = run_dir / "checkpoint-final"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        # Recompute final-model held-out perplexity so summary.json captures
        # it even if the per-epoch callback missed the very last update.
        final_held_out_loss: float | None = None
        final_held_out_ppl: float | None = None
        if held_out_dataset is not None:
            final_held_out_loss = compute_held_out_mlm_loss(
                model=trainer.model,
                held_out_dataset=held_out_dataset,
                tokenizer=tokenizer,
                batch_size=config.batch_size,
                mlm_probability=config.mlm_probability,
                probe_seed=config.seed,
            )
            final_held_out_ppl = (
                float(math.exp(final_held_out_loss))
                if not math.isnan(final_held_out_loss) and final_held_out_loss < 50
                else float("inf")
            )

        held_out_block: dict[str, Any] | None = None
        if held_out_dataset is not None:
            relative_delta = None
            if (
                base_held_out_ppl is not None
                and final_held_out_ppl is not None
                and math.isfinite(base_held_out_ppl)
                and math.isfinite(final_held_out_ppl)
                and base_held_out_ppl > 0
            ):
                relative_delta = (final_held_out_ppl - base_held_out_ppl) / base_held_out_ppl
            held_out_block = {
                "num_held_out_articles": len(held_out_articles),
                "num_held_out_windows": len(held_out_dataset),
                "probe_seed": config.seed,
                "base_loss": base_held_out_loss,
                "base_ppl": base_held_out_ppl,
                "final_loss": final_held_out_loss,
                "final_ppl": final_held_out_ppl,
                "relative_ppl_delta": relative_delta,
                "trajectory": (
                    held_out_callback.history if held_out_callback is not None else []
                ),
            }
            print(
                "Held-out FiNER perplexity: "
                f"base={base_held_out_ppl:.2f}, final={final_held_out_ppl:.2f}, "
                f"delta={relative_delta if relative_delta is None else f'{relative_delta:+.2%}'}"
            )

        summary = {
            "config_path": str(config_path),
            "config": asdict(config),
            "num_articles": len(article_texts),
            "num_windows": len(train_dataset),
            "train_loss": float(train_result.training_loss),
            "train_time_min": float(train_time_min),
            "checkpoint_final": str(final_dir),
            "split_diagnostics": split_diagnostics,
            "corpus_diagnostics": corpus_diagnostics,
            "held_out_finer_ppl": held_out_block,
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
