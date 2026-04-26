"""
crf_model.py - RoBERTa + linear-chain CRF training path for FiNER-ORD.

Phase B step 2 in CLAUDE.md. A CRF head replaces the independent softmax
classifier and produces BIO-valid sequences by construction via Viterbi
decoding. This file mirrors the top-down structure of src/train.py so the
CRF run produces the same artifacts (summary.json, predictions.json,
checkpoint-best/, master CSV rows) as the vanilla run.

Critical rules honored here (see CLAUDE.md and docs/PROJECT_CONTEXT.md):
  - Use `attention_mask` as the CRF mask so the first timestep is always
    valid. Replace -100 in labels with 0 before the CRF forward; rely on
    the mask to ignore special / continuation subwords at loss time.
    Filter -100 positions out of the final metric computation using the
    original label_ids.
  - Parameter groups: backbone LR (config.learning_rate, e.g. 1e-5) and
    classifier + CRF transitions LR (config.crf_learning_rate, e.g. 1e-4).
  - CRF transitions can NaN in fp16; run emissions in float32 even when
    the backbone is in fp16 via autocast-off in the CRF region.
  - Save decoded (Viterbi) predictions, not logits.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchcrf import CRF
from transformers import (
    AutoConfig,
    AutoModel,
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
    _is_no_decay_parameter,
    append_csv_row,
    compute_config_hash,
    make_json_safe,
    summarize_seed_values,
    write_json_file,
)


# -----------------------------------------------------------------------------
# Emission extraction (used for logit-level ensembling across seeds)
# -----------------------------------------------------------------------------


def _select_inference_device() -> torch.device:
    """Pick the best available device for a forward-only emission pass."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_crf_emissions(
    model: "RobertaCrfForTokenClassification",
    dataset: Any,
    tokenizer: Any,
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Run forward-only over `dataset` and return per-example emissions.

    The CRF transition matrix is intentionally NOT applied here; we save the
    raw classifier emissions per token so multiple seeds can be averaged
    before Viterbi decoding at ensemble time.

    Returns three parallel lists, each of length len(dataset). For example i,
    emissions[i] has shape (L_i, NUM_LABELS) and L_i = attention_mask.sum().
    Labels include -100 sentinels for special tokens and continuation
    subwords; ensemble code drops them at metric time.
    """

    device = _select_inference_device()
    model = model.to(device)
    model.eval()

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    all_emissions: list[np.ndarray] = []
    all_attention_masks: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    n = len(dataset)
    for start in range(0, n, batch_size):
        batch_examples = [dataset[i] for i in range(start, min(start + batch_size, n))]
        batch = collator(batch_examples)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_cpu = batch["labels"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            emissions = outputs["emissions"]

        emissions_np = emissions.detach().cpu().numpy()
        attention_mask_np = attention_mask.detach().cpu().numpy()
        labels_np = labels_cpu.numpy()

        for row in range(emissions_np.shape[0]):
            length = int(attention_mask_np[row].sum())
            if length == 0:
                continue
            all_emissions.append(emissions_np[row, :length].astype(np.float16))
            all_attention_masks.append(attention_mask_np[row, :length].astype(np.uint8))
            all_labels.append(labels_np[row, :length].astype(np.int64))

    return all_emissions, all_attention_masks, all_labels


def save_emissions_npz(
    path: Path,
    emissions: list[np.ndarray],
    attention_masks: list[np.ndarray],
    labels: list[np.ndarray],
) -> None:
    """Persist per-example emissions, masks, and labels as object arrays."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        emissions=np.array(emissions, dtype=object),
        attention_mask=np.array(attention_masks, dtype=object),
        labels=np.array(labels, dtype=object),
    )


def save_crf_transitions(path: Path, model: "RobertaCrfForTokenClassification") -> None:
    """Persist CRF transition matrices so seeds can be averaged at decode time."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        transitions=model.crf.transitions.detach().cpu().numpy().astype(np.float32),
        start_transitions=model.crf.start_transitions.detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        end_transitions=model.crf.end_transitions.detach()
        .cpu()
        .numpy()
        .astype(np.float32),
    )


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class RobertaCrfForTokenClassification(nn.Module):
    """Backbone encoder + dropout + linear classifier + linear-chain CRF.

    The backbone is any HuggingFace encoder (RoBERTa-large for the teacher).
    The classifier produces emissions; the CRF turns emissions + labels into
    a log-likelihood loss at training time and Viterbi-decoded tag sequences
    at inference time.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()

        self.backbone_config = AutoConfig.from_pretrained(
            model_name,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.backbone = AutoModel.from_pretrained(model_name, config=self.backbone_config)

        dropout_prob = getattr(self.backbone_config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.backbone_config.hidden_size, NUM_LABELS)
        self.crf = CRF(NUM_LABELS, batch_first=True)

    def _emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the backbone and classifier; return emissions in float32.

        The backbone and classifier run under whatever autocast context the
        Trainer set up (fp16 on CUDA). We only cast the emissions to fp32
        on the way out because the CRF log-space math below is what's
        numerically fragile, not the encoder.
        """

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(outputs.last_hidden_state)
        return self.classifier(hidden).float()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **_unused: Any,
    ) -> dict[str, torch.Tensor | None]:
        """Return a dict with `loss`, `emissions`, `mask`.

        The CRF mask is `attention_mask` so position 0 (the <s>/[CLS] token)
        is always included; that satisfies pytorch-crf's requirement that
        the first timestep of the mask be True. Special-token and
        continuation labels (-100) are replaced with 0 for the CRF forward,
        and later filtered out at metric time using the original label_ids.
        """

        mask = attention_mask.bool()
        emissions = self._emissions(input_ids, attention_mask)

        loss: torch.Tensor | None = None
        if labels is not None:
            labels_for_crf = labels.clone()
            labels_for_crf[labels_for_crf == -100] = 0
            # Disable autocast only around the CRF, which is log-space and
            # prone to NaN in fp16. The backbone/classifier above already
            # benefit from the outer fp16 autocast context on CUDA.
            with autocast(enabled=False):
                loss = -self.crf(
                    emissions,
                    labels_for_crf,
                    mask=mask,
                    reduction="mean",
                )

        return {"loss": loss, "emissions": emissions, "mask": mask}

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[list[int]]:
        """Viterbi-decode the most likely tag sequence per example."""

        mask = attention_mask.bool()
        emissions = self._emissions(input_ids, attention_mask)
        with autocast(enabled=False):
            return self.crf.decode(emissions, mask=mask)


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------


def _build_crf_llrd_param_groups(
    model: "RobertaCrfForTokenClassification",
    head_lr: float,
    crf_lr: float,
    decay: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """AdamW param groups for CRF + LLRD.

    Backbone is split into per-encoder-layer + embeddings buckets, each
    assigned LR = head_lr * decay^k descending from the top. Classifier
    and CRF head share `crf_lr`, preserving the existing CRF semantics
    where the head trains at a higher LR than the backbone.

    Behavior on a 24-layer RoBERTa-large with `head_lr=3e-5`, `decay=0.9`:
      - top encoder layer (23): 3e-5 * 0.9^1  = 2.7e-5
      - bottom encoder layer (0): 3e-5 * 0.9^24 = 2.4e-6
      - embeddings:               3e-5 * 0.9^24 = 2.4e-6
      - classifier + crf:         crf_lr (e.g., 1e-4)
    """
    backbone = model.backbone
    if not hasattr(backbone, "encoder") or not hasattr(backbone.encoder, "layer"):
        raise ValueError(
            "CRF + LLRD requires `model.backbone.encoder.layer`; "
            "backbone structure is incompatible."
        )
    num_layers = len(backbone.encoder.layer)

    lr_by_bucket: dict[str, float] = {}
    for layer_idx in range(num_layers):
        exponent = num_layers - layer_idx
        lr_by_bucket[f"encoder.layer.{layer_idx}"] = head_lr * (decay ** exponent)
    lr_by_bucket["embeddings"] = head_lr * (decay ** num_layers)

    bucket_params: dict[str, dict[str, list[nn.Parameter]]] = {
        b: {"decay": [], "no_decay": []} for b in lr_by_bucket
    }
    head_params_decay: list[nn.Parameter] = []
    head_params_no_decay: list[nn.Parameter] = []

    layer_prefix = "backbone.encoder.layer."
    embeddings_prefix = "backbone.embeddings."
    assigned_param_ids: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in assigned_param_ids:
            raise ValueError(f"Parameter assigned multiple times: {name}")
        assigned_param_ids.add(param_id)

        no_decay = _is_no_decay_parameter(name)
        decay_key = "no_decay" if no_decay else "decay"

        if name.startswith(layer_prefix):
            layer_suffix = name[len(layer_prefix) :]
            layer_idx_str = layer_suffix.split(".", 1)[0]
            if not layer_idx_str.isdigit():
                raise ValueError(f"Could not parse encoder layer index from `{name}`.")
            bucket_params[f"encoder.layer.{int(layer_idx_str)}"][decay_key].append(param)
        elif name.startswith(embeddings_prefix):
            bucket_params["embeddings"][decay_key].append(param)
        elif name.startswith("backbone."):
            # Other backbone params (pooler, rel_embeddings, etc.) sit at the
            # bottom of the LLRD ladder alongside embeddings. This matches
            # the convention in `src/train.py::_assign_llrd_bucket`.
            bucket_params["embeddings"][decay_key].append(param)
        elif name.startswith("classifier.") or name.startswith("crf."):
            if no_decay:
                head_params_no_decay.append(param)
            else:
                head_params_decay.append(param)
        else:
            raise ValueError(f"Unexpected param name in CRF + LLRD grouping: {name}")

    param_groups: list[dict[str, Any]] = []
    for bucket, lr in lr_by_bucket.items():
        if bucket_params[bucket]["decay"]:
            param_groups.append(
                {
                    "params": bucket_params[bucket]["decay"],
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )
        if bucket_params[bucket]["no_decay"]:
            param_groups.append(
                {
                    "params": bucket_params[bucket]["no_decay"],
                    "lr": lr,
                    "weight_decay": 0.0,
                }
            )
    if head_params_decay:
        param_groups.append(
            {
                "params": head_params_decay,
                "lr": crf_lr,
                "weight_decay": weight_decay,
            }
        )
    if head_params_no_decay:
        param_groups.append(
            {
                "params": head_params_no_decay,
                "lr": crf_lr,
                "weight_decay": 0.0,
            }
        )
    return param_groups


def _log_crf_llrd_lrs(
    model: "RobertaCrfForTokenClassification",
    head_lr: float,
    decay: float,
    crf_lr: float,
) -> None:
    """Print the CRF + LLRD ladder once per run for visibility."""
    backbone = model.backbone
    num_layers = len(backbone.encoder.layer)
    print("=" * 60)
    print(f"CRF + LLRD learning rates (decay={decay}, head_lr={head_lr})")
    print("=" * 60)
    print(f"  classifier + crf       {crf_lr:.10g}")
    for layer_idx in range(num_layers - 1, -1, -1):
        exponent = num_layers - layer_idx
        lr = head_lr * (decay ** exponent)
        print(f"  encoder.layer.{layer_idx:<7d}{lr:.10g}")
    embed_lr = head_lr * (decay ** num_layers)
    print(f"  embeddings + other     {embed_lr:.10g}")


class CrfTrainer(Trainer):
    """Trainer that uses the CRF log-likelihood loss and Viterbi decoding.

    Without LLRD: two parameter groups, backbone at `args.learning_rate` and
    classifier + CRF at `crf_learning_rate`.

    With LLRD (`llrd_decay` set): the backbone is split into per-encoder-layer
    + embeddings buckets stepping down from `head_lr * decay`; the classifier
    + CRF head stays at `crf_learning_rate`. `head_lr` defaults to
    `args.learning_rate` if not provided.

    Predictions are decoded tag IDs, not logits, so `prediction_step` runs
    Viterbi and returns the decoded tags padded to the batch's seq length
    with -100.
    """

    def __init__(
        self,
        *args: Any,
        crf_learning_rate: float,
        llrd_decay: float | None = None,
        head_lr: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.crf_learning_rate = float(crf_learning_rate)
        self.llrd_decay = float(llrd_decay) if llrd_decay is not None else None
        self.head_lr = float(head_lr) if head_lr is not None else None

    # ---- optimizer ---------------------------------------------------------

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW with backbone vs CRF-head LRs, optionally with LLRD."""

        if self.optimizer is not None:
            return self.optimizer

        if self.llrd_decay is not None:
            head_lr = (
                self.head_lr if self.head_lr is not None else self.args.learning_rate
            )
            _log_crf_llrd_lrs(
                model=self.model,
                head_lr=head_lr,
                decay=self.llrd_decay,
                crf_lr=self.crf_learning_rate,
            )
            param_groups = _build_crf_llrd_param_groups(
                model=self.model,
                head_lr=head_lr,
                crf_lr=self.crf_learning_rate,
                decay=self.llrd_decay,
                weight_decay=self.args.weight_decay,
            )
            self.optimizer = torch.optim.AdamW(param_groups)
            return self.optimizer

        decay = self.args.weight_decay
        backbone_params: list[nn.Parameter] = []
        head_params: list[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(param)
            else:
                # classifier.* and crf.* share the faster head LR.
                head_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.args.learning_rate, "weight_decay": decay},
                {"params": head_params, "lr": self.crf_learning_rate, "weight_decay": decay},
            ]
        )
        return self.optimizer

    # ---- loss --------------------------------------------------------------

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs.get("labels"),
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    # ---- prediction --------------------------------------------------------

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        labels = inputs.get("labels")
        has_labels = labels is not None

        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
            )
            loss = outputs["loss"] if has_labels else None
            decoded_batches = model.crf.decode(outputs["emissions"], mask=outputs["mask"])

        if prediction_loss_only:
            return (loss, None, None)

        # Pad each decoded sequence back to the batch seq length with -100 so
        # HF Trainer's nested_concat can stack variable-length batches.
        reference = labels if labels is not None else inputs["input_ids"]
        batch_size, seq_len = reference.shape
        predictions = torch.full(
            (batch_size, seq_len),
            fill_value=-100,
            dtype=torch.long,
            device=reference.device,
        )
        for row, decoded_sequence in enumerate(decoded_batches):
            length = len(decoded_sequence)
            if length > 0:
                predictions[row, :length] = torch.tensor(
                    decoded_sequence,
                    dtype=torch.long,
                    device=reference.device,
                )

        return (loss, predictions, labels)


# -----------------------------------------------------------------------------
# Decoding helpers
# -----------------------------------------------------------------------------


def decode_crf_predictions(
    predicted_ids: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert padded predicted-tag and label arrays into seqeval strings.

    Mirrors src.train.decode_token_predictions but expects already-decoded
    tag IDs (from Viterbi) rather than raw logits. -100 positions in
    `label_ids` are skipped so the output matches the vanilla format.
    """

    true_labels: list[list[str]] = []
    predicted_labels: list[list[str]] = []

    for pred_sequence, label_sequence in zip(predicted_ids, label_ids):
        decoded_true: list[str] = []
        decoded_pred: list[str] = []
        for pred_id, label_id in zip(pred_sequence, label_sequence):
            label_id = int(label_id)
            if label_id == -100:
                continue
            decoded_true.append(ID2LABEL[label_id])
            decoded_pred.append(ID2LABEL[int(pred_id)])
        true_labels.append(decoded_true)
        predicted_labels.append(decoded_pred)

    return true_labels, predicted_labels


def compute_metrics_crf(eval_prediction: EvalPrediction) -> dict[str, float]:
    """Per-epoch metric callback for the CRF path."""

    predictions = eval_prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    true_labels, predicted_labels = decode_crf_predictions(
        predictions, eval_prediction.label_ids
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


# -----------------------------------------------------------------------------
# Trainer construction
# -----------------------------------------------------------------------------


def create_crf_trainer(
    config: TrainConfig,
    seed: int,
    run_id: str,
    run_dir: Path,
    dataset: Any,
    tokenizer: Any,
    model: nn.Module,
    use_wandb: bool,
) -> CrfTrainer:
    """Build the Trainer wired for CRF training."""

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
        # Gradient clipping guards CRF transitions from exploding updates.
        max_grad_norm=1.0,
        # Non-PreTrainedModel wrapper; safetensors can't serialize the CRF.
        save_safetensors=False,
        report_to=["wandb"] if use_wandb else [],
        seed=seed,
        data_seed=seed,
        remove_unused_columns=True,
    )

    crf_lr = config.crf_learning_rate if config.crf_learning_rate is not None else 1e-4

    return CrfTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics_crf,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        ],
        crf_learning_rate=crf_lr,
        llrd_decay=config.llrd_decay,
        head_lr=config.head_lr,
    )


# -----------------------------------------------------------------------------
# Seed-level execution
# -----------------------------------------------------------------------------


def run_single_seed_crf(
    config: TrainConfig,
    config_path: str | Path,
    config_stem: str,
    seed: int,
    dataset: Any,
    tokenizer: Any,
    output_root: Path,
    disable_wandb: bool,
) -> dict[str, Any]:
    """Train + evaluate one seed of the CRF variant and persist all artifacts."""

    set_seed(seed)

    run_id = f"{config_stem}_seed{seed}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = asdict(config)
    resolved_config["seeds"] = [seed]
    resolved_config["resolved_seed"] = seed
    resolved_config["config_path"] = str(config_path)
    resolved_config["run_id"] = run_id

    # ---- W&B (optional) ----------------------------------------------------

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

    # ---- Model + Trainer ---------------------------------------------------

    model = RobertaCrfForTokenClassification(config.model_name)
    trainer = create_crf_trainer(
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
        # ---- Training ------------------------------------------------------

        start_time = time.perf_counter()
        trainer.train()
        train_time_min = (time.perf_counter() - start_time) / 60.0

        # Stable best-checkpoint path for downstream scripts.
        best_checkpoint_dir = run_dir / "checkpoint-best"
        best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.model.state_dict(), best_checkpoint_dir / "pytorch_model.bin")
        trainer.model.backbone_config.save_pretrained(best_checkpoint_dir)
        tokenizer.save_pretrained(best_checkpoint_dir)

        # ---- Final evaluation ---------------------------------------------

        val_output = trainer.predict(
            dataset["validation"], metric_key_prefix="validation"
        )
        test_output = trainer.predict(dataset["test"], metric_key_prefix="test")

        val_true_labels, val_predictions = decode_crf_predictions(
            val_output.predictions, val_output.label_ids
        )
        test_true_labels, test_predictions = decode_crf_predictions(
            test_output.predictions, test_output.label_ids
        )

        val_metrics = compute_detailed_metrics(
            val_true_labels, val_predictions, verbose=False
        )
        test_metrics = compute_detailed_metrics(
            test_true_labels, test_predictions, verbose=False
        )

        # ---- Artifacts -----------------------------------------------------

        predictions_path = run_dir / "predictions.json"
        summary_path = run_dir / "summary.json"

        write_json_file(
            predictions_path,
            {"true_labels": test_true_labels, "predictions": test_predictions},
        )

        try:
            test_emissions, test_masks, test_labels = extract_crf_emissions(
                trainer.model,
                dataset["test"],
                tokenizer,
                batch_size=config.batch_size,
            )
            save_emissions_npz(
                run_dir / "test_emissions.npz",
                test_emissions,
                test_masks,
                test_labels,
            )
            val_emissions, val_masks, val_labels = extract_crf_emissions(
                trainer.model,
                dataset["validation"],
                tokenizer,
                batch_size=config.batch_size,
            )
            save_emissions_npz(
                run_dir / "val_emissions.npz",
                val_emissions,
                val_masks,
                val_labels,
            )
            save_crf_transitions(run_dir / "crf_transitions.npz", trainer.model)
        except Exception as exc:
            print(f"[{run_id}] Failed to save emissions/transitions: {exc}")

        param_count = sum(p.numel() for p in trainer.model.parameters())
        config_hash = compute_config_hash(resolved_config)

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
                "uses_crf": True,
                "crf_learning_rate": float(
                    config.crf_learning_rate if config.crf_learning_rate is not None else 1e-4
                ),
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

        # ---- Master CSVs ---------------------------------------------------

        notes_parts = [config_stem, "crf"]
        if config.wandb_tags:
            notes_parts.append(f"tags={','.join(config.wandb_tags)}")
        compat_notes = " | ".join(notes_parts)

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
            output_root / "results_detailed.csv", DETAILED_RESULTS_HEADER, detailed_row
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


# -----------------------------------------------------------------------------
# Multi-seed orchestration (public entrypoint)
# -----------------------------------------------------------------------------


def run_crf_training(
    config: TrainConfig,
    config_path: str | Path,
    run_checks: bool,
    disable_wandb: bool,
) -> None:
    """Run the CRF variant across every seed in `config`."""

    if not config.use_crf:
        raise ValueError("run_crf_training called with use_crf=False.")
    if config.use_distillation:
        raise NotImplementedError(
            "CRF + distillation is not supported; run Phase B first, then Phase C."
        )

    config_stem = Path(config_path).stem
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset, tokenizer, _ = get_dataset_and_tokenizer(
        config.model_name,
        max_length=config.max_seq_length,
        run_checks=run_checks,
        label_all_subwords=config.label_all_subwords,
    )

    seed_results = []
    for seed in config.seeds:
        seed_results.append(
            run_single_seed_crf(
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

    aggregate_summary = {
        "config_path": str(config_path),
        "config": asdict(config),
        "num_seeds": len(seed_results),
        "run_ids": [r["run_id"] for r in seed_results],
        "seeds": [r["seed"] for r in seed_results],
        "test_metrics": {
            "token_weighted_f1": summarize_seed_values(
                [r["test_metrics"]["token_weighted_f1"] for r in seed_results]
            ),
            "entity_overall_f1": summarize_seed_values(
                [r["test_metrics"]["entity_overall_f1"] for r in seed_results]
            ),
            "entity_per_class": {
                "PER": summarize_seed_values(
                    [r["test_metrics"]["entity_per_class"].get("PER", 0.0) for r in seed_results]
                ),
                "LOC": summarize_seed_values(
                    [r["test_metrics"]["entity_per_class"].get("LOC", 0.0) for r in seed_results]
                ),
                "ORG": summarize_seed_values(
                    [r["test_metrics"]["entity_per_class"].get("ORG", 0.0) for r in seed_results]
                ),
            },
        },
        "validation_metrics": {
            "entity_overall_f1": summarize_seed_values(
                [r["val_metrics"]["entity_overall_f1"] for r in seed_results]
            )
        },
        "train_time_min": summarize_seed_values(
            [r["train_time_min"] for r in seed_results]
        ),
    }
    write_json_file(output_root / f"{config_stem}_aggregate.json", aggregate_summary)
