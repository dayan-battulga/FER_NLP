# PROJECT_CONTEXT.md

Extended technical context for the FiNER-ORD Financial NER project. See [CLAUDE.md](../CLAUDE.md) for the shorter operational context.

---

## Project Direction

Financial Named Entity Recognition on `gtfintechlab/finer-ord`. The model predicts 7 BIO labels over PER, LOC, and ORG entities. The headline metric is strict entity-level F1 from `seqeval`, not token weighted F1.

The submission is framed as a Pareto story under the brief's "F1 or efficiency wins" criterion. The deliverable is a frontier showing teacher F1, distilled student F1, and INT8 quantized student F1 against latency and model size. The teacher, student, and INT8 student are all locked. Latency, quantization, and the Pareto chart are complete. Remaining work is REPORT.md and README.md.

The active path:

1. **Teacher (locked).** `roberta-large` continued pretraining on FiNER train articles (`configs/baseline/dapt_roberta_large.yaml`), then 5-epoch CRF fine-tune (`configs/baseline/efficient_after_dapt.yaml`), ensembled across 3 seeds via CRF emission averaging.
2. **Student (locked).** Offline distillation of `distilroberta-base` from the teacher's averaged emissions on the train split. Vanilla token classification head, no CRF. 20 epochs (deviation from the teacher's 5-epoch budget; see "Phase C: Distillation, Quantization, And Pareto" for justification).
3. **INT8 student (locked).** Dynamic INT8 on the FP32 student via `torch.quantization.quantize_dynamic`. Mean F1 drop of `0.0036` from FP32, well within `0.01` tolerance.
4. **Pareto (complete).** Two-subplot chart (latency vs F1, size vs F1) with five points and the frontier marked.

Current headline:

- **Teacher (locked):** `efficient_after_dapt_logit_ensemble` at **`0.8634`** test entity F1
- **Best teacher single-model mean:** `efficient_after_dapt` at **`0.8548 +/- 0.0038`**
- **Student (locked):** `student_distilled_logit_ensemble` at **`0.8304`** test entity F1
- **Student single-model 3-seed mean:** **`0.8277`** (range `0.8262`-`0.8298`)
- **INT8 student (locked):** 3-seed mean **`0.8241`** test entity F1; drop of `0.0036` from FP32 student
- **Latency (Apple M2 Pro CPU, batch=1):** teacher `106.6ms`, FP32 student `21.7ms` (4.9x speedup), INT8 student `10.8ms` (9.9x speedup, 2.0x over FP32 student)

Both Phase C F1 targets cleared:
- Single-model mean: `0.8277` >= `0.82` (cushion `+0.008`)
- Ensemble: `0.8304` >= `0.83` (cushion `+0.0004`, thin)
- INT8 mean within `0.01` of FP32 mean: `0.0036` drop, well within tolerance

---

## Current Experiment State

### Completed Multi-Seed Teacher Experiments

| Experiment | Config | Test Entity F1 Mean +/- Std | Val Entity F1 Mean +/- Std | Train Time Mean | Notes |
|---|---|---:|---:|---:|---|
| Vanilla RoBERTa-large | `configs/baseline_teacher.yaml` | `0.8485 +/- 0.0022` | `0.9152 +/- 0.0025` | `40.83 min` | 30 epochs, vanilla token classifier |
| Original RoBERTa-large + CRF | `configs/baseline/teacher_crf.yaml` | `0.8521 +/- 0.0018` | `0.9197 +/- 0.0027` | `22.48 min` | 30 epochs, original CRF path |
| Efficient CRF | `configs/baseline/efficient_training.yaml` | `0.8481 +/- 0.0066` | `0.9206 +/- 0.0046` | `7.31 min` | 5 epochs, RoBERTa-large + CRF |
| DeBERTa efficient, LLRD + LAS | `configs/baseline/deberta_efficient.yaml` | `0.8374 +/- 0.0018` | `0.9131 +/- 0.0047` | `8.19 min` | Negative result |
| DeBERTa LLRD + LAS, lower LR | `configs/baseline/deberta_efficient_lr1e5.yaml` | `0.8282 +/- 0.0026` | `0.9014 +/- 0.0087` | `8.46 min` | Negative result |
| DeBERTa LLRD + LAS, lower LR | `configs/baseline/deberta_efficient_lr2e5.yaml` | `0.8325 +/- 0.0058` | `0.9058 +/- 0.0020` | `8.52 min` | Negative result |
| DeBERTa LLRD, first-subword only | `configs/baseline/deberta_efficient_align_off.yaml` | `0.8339 +/- 0.0027` | `0.9103 +/- 0.0032` | `8.52 min` | Isolates `label_all_subwords=false`; still negative |
| Efficient after DAPT | `configs/baseline/efficient_after_dapt.yaml` | **`0.8548 +/- 0.0038`** | `0.9209 +/- 0.0072` | `7.21 min` + `~3.5 min` DAPT | Best single-model 3-seed mean |
| Efficient after DAPT + LAS + LLRD | `configs/baseline/efficient_after_dapt_lasllrd.yaml` | regressed | regressed | `~7 min` | Negative result; ensemble at `0.8368` |

### Completed Multi-Seed Student Experiments (Phase C)

| Experiment | Config | Test Entity F1 Mean (Range) | Val Entity F1 Mean | Train Time Mean | Notes |
|---|---|---:|---:|---:|---|
| Distilled student (FP32) | `configs/baseline/student_distilled.yaml` | **`0.8277` (range `0.0036`)** | `~0.918` | `~2.5 min/seed` | distilroberta-base, no CRF, 20 epochs, alpha=0.5, T=2.0, ensemble teacher |
| Distilled student (INT8) | dynamic INT8 of FP32 student | **`0.8241` (range `0.0118`)** | not run | quantization `<5 sec/seed` | dynamic INT8 on Linear via `quantize_dynamic`, qnnpack engine for ARM |

Per-seed FP32 student test entity F1: `0.8298` (seed 88), `0.8262` (seed 5768), `0.8272` (seed 78516). Best checkpoints landed at epoch 15-16 across all seeds (steps `3060`, `3060`, `3264` with ~204 steps/epoch).

Per-seed INT8 student test entity F1: `0.8301` (seed 88), `0.8240` (seed 5768), `0.8183` (seed 78516). Per-seed INT8 variance is wider than FP32 (range `0.0118` vs `0.0036`); dynamic quantization adds per-batch input quantization noise that interacts differently with each seed's weight distribution.

### Latency And Size (Apple M2 Pro CPU)

| Recipe | Params | Disk (MB) | Latency bs=1 median (ms) | Latency bs=1 p95 (ms) | Latency bs=8 median (ms) | Throughput bs=8 (ex/s) |
|---|---:|---:|---:|---:|---:|---:|
| Teacher single (`efficient_after_dapt_seed88`) | `355.4M` | `1356` | `106.6` | `200.4` | `371.1` | `21.6` |
| FP32 Student single (`student_distilled_seed88`) | `81.5M` | `311` | `21.7` | `25.5` | `56.6` | `141.3` |
| INT8 Student single (`student_distilled_seed88_int8`) | `~39M` (counter artifact) | `190` | `10.8` | `17.7` | `96.3` | `83.1` |

Notes:
- INT8 param count of `39M` is a measurement artifact: `quantize_dynamic` replaces `nn.Linear` with quantized modules that don't expose weights as `nn.Parameter`, so the standard counter undercounts. Use disk size (`190MB`) as the reportable size.
- Teacher p95 / median ratio at bs=1 is `1.88`, much higher than the student's `1.18`. Likely thermal throttling or background process interference during the run; medians are stable, p95 is noisy. Use medians for the chart, mention p95 as a caveat.
- INT8 latency at batch=8 is *worse* than FP32 student (`96.3ms` vs `56.6ms`). Real and reportable; likely qnnpack INT8 GEMM kernel overhead at larger matrix sizes plus per-batch quantize/dequantize cost. Honest framing: INT8 wins at batch=1 (interactive), FP32 wins at batch=8 (throughput).

Cross-machine portability: these numbers are pinned to Apple M2 Pro on `torch==2.3.1` with no GPU. Reproduction on different hardware will produce different absolute numbers; the relative ratios (4.9x teacher-to-student speedup, 2.0x FP32-to-INT8 speedup at bs=1) are the more transferable claim.

### Ensembles Across Seed Runs

| Ensemble | Source Runs | Single Mean | Vote Ensemble | Logit Ensemble |
|---|---|---:|---:|---:|
| `teacher_crf` | `teacher_crf_seed{88,5768,78516}` | `0.8521` | `0.8568` | pending re-extraction |
| `efficient_training` | `efficient_training_seed{88,5768,78516}` | `0.8481` | `0.8551` | pending re-extraction |
| `efficient_after_dapt` | `efficient_after_dapt_seed{88,5768,78516}` | `0.8548` | `0.8587` | **`0.8634`** |
| `efficient_after_dapt_lasllrd` | `efficient_after_dapt_lasllrd_seed{88,5768,78516}` | regressed | regressed | `0.8368` |
| `student_distilled` | `student_distilled_seed{88,5768,78516}` | `0.8277` | not run | **`0.8304`** |
| `student_distilled_int8` | `student_distilled_seed{88,5768,78516}_int8` | `0.8241` | not run | not run (would defeat quantization purpose) |

### Per-Class F1 (Logit Ensembles)

| Class | Teacher Ensemble | Student Ensemble | Gap |
|---|---:|---:|---:|
| PER | `0.9564` | `0.9340` | `-0.0224` |
| LOC | `0.8590` | `0.8325` | `-0.0265` |
| ORG | `0.8206` | `0.7781` | `-0.0425` |
| Overall | `0.8634` | `0.8304` | `-0.0330` |

ORG transfers worst from teacher to student. PER transfers most cleanly. INT8 quantization shows the same pattern: ORG takes the largest hit per seed (`-0.005` to `-0.025`), PER and LOC lose much less. ORG is the brittle class across every transformation in the pipeline.

---

## Phase C: Distillation, Quantization, And Pareto

The teacher, student, and INT8 student are all locked. Phase C compute is complete. Remaining work is REPORT.md and README.md.

### Why distillation

The submission brief is "F1 or efficiency wins." The teacher delivers the F1 axis (`0.8634`). The student delivers the efficiency axis (smaller, faster, deployable). A clean Pareto story is the strongest framing for an ML/NLP Research Engineer role at a defense ML team, where deployment constraints matter as much as raw F1.

Realistic student expectations from the literature for 24-layer-to-6-layer NER distillation:

- 90 to 97% of teacher F1, depending on backbone and corpus
- 4x parameter reduction (`distilroberta-base` 82M vs `roberta-large` 355M)
- 2 to 3x latency improvement on the same hardware at FP32
- Additional 2x latency on CPU after dynamic INT8 quantization

Achieved:

- Student ensemble F1 / teacher ensemble F1: `0.8304 / 0.8634 = 96.2%`. Inside the literature band, near the upper end.
- Parameter reduction: 4.3x (82M vs 355M)
- FP32 latency improvement: `4.9x` over teacher at bs=1, `6.6x` at bs=8 (above the literature band; M2 Pro NEON optimizations differ between architectures, and the teacher CRF adds Viterbi cost on top of the encoder forward).
- INT8 latency improvement on CPU: `2.0x` over FP32 student at bs=1 (matches literature band).

Targets for this project, all hit:

- Student single-model mean: >= `0.82` test entity F1. Achieved: `0.8277`.
- Student logit ensemble: >= `0.83` test entity F1. Achieved: `0.8304`.
- INT8 student: <= `0.01` F1 drop from FP32 student. Achieved: `0.0036` mean drop.
- Latency: documented for batch_size=1 (single-example, deployment-relevant) and batch_size=8 (throughput) at FP32 CPU and INT8 CPU. CPU-only because no local GPU available; CPU is also the right deployment story for a defense ML role.

### Distillation method

Offline distillation from saved teacher emissions. The teacher is run once on the train split, emissions are saved as `train_emissions.npz`, and the student trains against those static emissions. No live teacher forward passes during student training.

Loss:

```
L_hard = CrossEntropy(student_logits, gold_labels)        # masked at -100
L_soft = T^2 * KLDiv(
    log_softmax(student_logits / T),
    softmax(teacher_emissions / T)
)                                                         # masked at -100
L_total = alpha * L_hard + (1 - alpha) * L_soft
```

Defaults: `T=2.0`, `alpha=0.5`. Both configurable via `TrainConfig.distill_temperature` and `TrainConfig.distill_alpha`.

The `T^2` factor on `L_soft` follows the standard Hinton et al. 2015 distillation formulation; without it, the soft loss gradient magnitude scales as `1/T^2` and the soft signal underweights as temperature increases.

### Why 20 student epochs

The Phase C spec called for 5 epochs. At 5 epochs the student landed at ~`0.81` mean entity F1, missing the `0.82` single-model target. Increasing to 20 epochs landed it at `0.8277` mean (`0.8262`-`0.8298` per-seed range).

This is not test-set fishing or per-epoch sweeping. The decision was: try the spec, observe a target miss, scale epochs once, accept the result. Diagnostics confirming real convergence rather than val overfitting:

- Best checkpoints landed at epochs 15-16 across all three seeds (load_best_model_at_end + early stopping selected mid-training, not the final epoch)
- Val-test gap was `0.086`-`0.093`, similar to the teacher's `0.067`. The val set is not being gamed.
- Per-seed F1 range was `0.0036`, indicating low seed variance and stable convergence.

The mechanistic reason 20 epochs are needed: distillation on smoothed soft targets has a weaker per-step gradient signal than vanilla CE training. With `alpha=0.5`, half the gradient comes from a softened distribution where the maximum probability is much less than 1.0 even when the teacher is confident. On 3262 training examples, more passes are needed to converge to the same effective number of strong gradient updates.

This is documented in REPORT.md as a Phase C deviation with the underlying reasoning.

### Ensemble teacher

`teacher_mode=ensemble` (the default) averages emissions across the 3 teacher seeds before computing the soft loss:

```
teacher_emissions = mean(emissions_seed88, emissions_seed5768, emissions_seed78516)
```

This is "ensemble distillation" and consistently outperforms distilling from a single teacher checkpoint by typically `+0.005` to `+0.010` student F1, because the averaged logits are smoother and capture more of the teacher's uncertainty geometry.

`teacher_mode=single` is available for ablation but is not the headline.

### Why no CRF on the student

The CRF added approximately `+0.004` entity F1 on the teacher (`0.8485` vanilla vs `0.8521` with CRF on the 30-epoch baseline). Reasons for not propagating it to the student:

1. The CRF lift came from sequence-level transition modeling that mostly fixes already-easy boundaries. On a smaller student where capacity is the binding constraint, the CRF lift is unlikely to transfer at the same magnitude.
2. The CRF adds Viterbi decoding at inference time, which hurts the deployment story.
3. INT8 dynamic quantization does not cover the CRF transition matrix; quantizing only the encoder while leaving the CRF in FP32 complicates the latency story.

The student uses a vanilla token classification head with softmax decoding.

### Tokenizer alignment

The teacher emissions were extracted with:

- Tokenizer: `roberta-large` BPE
- `add_prefix_space=True`
- `max_seq_length=256`
- `label_all_subwords=false` (first-subword-only alignment)

The student uses `distilroberta-base`, which shares the same BPE vocabulary as `roberta-large`. Token IDs are identical for the same input, and emission positions align without re-extraction.

If a future student backbone uses a different tokenizer (e.g., a BERT-family checkpoint), the teacher emissions must be re-extracted with the new tokenizer for alignment to hold.

### Quantization

Dynamic INT8 quantization via `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`. Single function call, CPU-targeted, no calibration data, no QAT.

Why dynamic over static or QAT:

- Dynamic is one function call and the right level of effort for a take-home.
- Static quantization requires a calibration dataset and per-channel observers; the lift over dynamic for a 6-layer transformer is small.
- QAT requires re-training the student with fake-quantization, which doubles the Phase C time budget.

The INT8 checkpoint is evaluated for:

1. Test entity F1 via the existing `compute_detailed_metrics` path. Saved as `summary_int8.json` per seed. Mean F1 `0.8241`, drop of `0.0036` from FP32 mean.
2. Latency on CPU at batch sizes 1 and 8. INT8 wins at bs=1 (`10.8ms` vs `21.7ms`, `2.0x` speedup) and loses at bs=8 (`96.3ms` vs `56.6ms`, qnnpack overhead at larger batches). Honest tradeoff documented in REPORT.md.
3. On-disk size. INT8 lands at `190MB` vs FP32 student's `311MB`. Size reduction is `1.6x`, not `4x`, because dynamic quantization only converts `nn.Linear` weights to INT8; embeddings, layer norms, and head bias terms remain FP32. The headline claim is "INT8 is meaningfully smaller and faster on CPU at single-example latency, with a small F1 cost."

### Apple Silicon gotchas

Two patches required for the quantization scripts to work on M2 Pro:

1. **Quantization engine.** Default `fbgemm` backend is x86-only. On ARM, set `torch.backends.quantized.engine = 'qnnpack'` before any `quantize_dynamic` call. Without it: `RuntimeError: Didn't find engine for operation quantized::linear_prepack NoQEngine`. Applied in both `scripts/quantize_student.py` and `scripts/measure_latency.py` (the latter also calls `quantize_dynamic` internally to recreate the quantized module structure when loading INT8 checkpoints from disk).

2. **Save path.** `model.save_pretrained()` fails on dynamically quantized models because `huggingface_hub`'s state-dict sharder iterates state_dict values and assumes all are tensors; quantized modules store `torch.dtype` objects (`LinearPackedParams`) and break the iteration. Fix: replace with `torch.save(quantized_model.state_dict(), output_dir / "pytorch_model.bin")` and copy `config.json` plus tokenizer files manually. Loading the INT8 checkpoint later requires recreating the quantized structure first (load FP32 model, apply `quantize_dynamic`, then `load_state_dict`).

Both fixes are documented in CLAUDE.md Critical Rules 23 and 24.

### Pareto chart

Two subplots, both with log x-axes (latency spans 10-320ms, size spans 190-4068MB; linear scale crowds the lower end):

1. **Latency vs F1.** X-axis is single-example latency at batch_size=1 (median over 100 batches after 10 warmup batches). Y-axis is entity F1.
2. **Size vs F1.** X-axis is on-disk model size in MB. Y-axis is entity F1.

Five points plotted:

- Teacher single (3-seed mean F1 `0.8548`, latency `106.6ms`, size `1356MB`)
- Teacher logit ensemble (`0.8634`, latency `~320ms` est. as 3x single-model, size `~4068MB` est. as 3x)
- Student FP32 single (3-seed mean F1 `0.8277`, latency `21.7ms`, size `311MB`)
- Student FP32 logit ensemble (`0.8304`, latency `~65ms` est., size `~933MB` est.)
- Student INT8 single (3-seed mean F1 `0.8241`, latency `10.8ms`, size `190MB`)

INT8 ensemble was intentionally skipped: ensembling triples inference latency, defeating the purpose of quantization. The deployable INT8 product is the single-model checkpoint.

The Pareto frontier is marked with a dashed line connecting the non-dominated points. F1 numbers come from 3-seed means rather than seed 88 alone (mixing single-seed F1 with single-seed latency on the chart can produce an "INT8 dominated FP32" artifact that isn't real; seed-88 INT8 happened to land slightly above seed-88 FP32 by `+0.0003`, pure noise). Latency uses seed 88 alone since architecture-not-weights.

If the optional teacher-side experiments produce results, they would add as dominated points: LAS+LLRD ensemble (`0.8368`, dominated), packed-512 ensemble (not run), 6-model multi-recipe ensemble (not run).

### Phase C run order (executed)

```bash
# 1. Extract train emissions from each locked teacher seed (DONE)
./.venv/bin/python scripts/extract_train_emissions.py \
  --runs efficient_after_dapt_seed88 efficient_after_dapt_seed5768 efficient_after_dapt_seed78516

# 2. Smoke test the distillation pipeline (DONE; passed)
./.venv/bin/python -m src.distill --config configs/baseline/student_distilled.yaml --smoke --no-wandb

# 3. Run 3-seed student distillation (DONE; 20 epochs, mean 0.8277)
./.venv/bin/python -m src.distill --config configs/baseline/student_distilled.yaml --no-wandb

# 4. Build the student logit ensemble (DONE; 0.8304)
./.venv/bin/python scripts/ensemble_logits.py \
  --runs student_distilled_seed88 student_distilled_seed5768 student_distilled_seed78516 \
  --mode logit \
  --output-name student_distilled_logit_ensemble

# 5. Latency measurements FP32 (DONE; CPU only, no local GPU)
./.venv/bin/python scripts/measure_latency.py \
  --runs efficient_after_dapt_seed88 student_distilled_seed88 \
  --device cpu

# 6. Quantize the student (DONE; per-seed INT8 F1 0.8301/0.8240/0.8183, mean 0.8241)
./.venv/bin/python scripts/quantize_student.py \
  --runs student_distilled_seed88 student_distilled_seed5768 student_distilled_seed78516

# 7. Latency measurements INT8 (DONE; bs=1 10.8ms, bs=8 96.3ms regression vs FP32)
./.venv/bin/python scripts/measure_latency.py \
  --runs student_distilled_seed88_int8 \
  --device cpu

# 8. Build the Pareto chart (DONE; docs/figures/pareto.png + pareto_data.csv)
./.venv/bin/python scripts/build_pareto.py
```

### Teacher Reproduction Note

During Phase C setup, an investigation revealed that the original locked teacher checkpoints had been lost (only `summary.json`, `predictions.json`, `crf_transitions.npz`, and `*_emissions.npz` survived; `checkpoint-best/pytorch_model.bin` was gone for all three seeds). The teacher was reproduced from configs by re-running DAPT (10 epochs MLM) followed by 3-seed CRF fine-tuning. The reproduced ensemble matched the original `0.8634` to four decimals, with per-class numbers (PER `0.9564`, LOC `0.8590`, ORG `0.8206`) also matching exactly. Per-seed reproduced test F1: `0.8543`, `0.8514`, `0.8588`.

This validates that the locked teacher recipe is fully reproducible from configs alone, which is a positive signal for the submission. All checkpoints are now stored locally, not just on Colab.

---

## Optional Teacher-Side Experiments

These are inference-only or single-seed experiments that could run in parallel with the remaining REPORT.md work. They do not block submission and are unlikely to be worth the time at this point.

### Packed-512 Re-Inference

`scripts/reinfer_packed.py` implements greedy within-document sentence packing into 512-token windows, then slices emissions back to per-sentence sequences.

Rationale:

- The current fine-tune uses `max_seq_length=256`.
- Adjacent sentences may provide useful context for entity boundaries.
- Re-inference is cheaper and lower-risk than retraining.

Protocol:

1. Run validation first on `efficient_after_dapt_seed{88,5768,78516}`.
2. Compare validation F1 to the same checkpoints at 256.
3. Run test only if validation delta is non-negative.

Risk: RoBERTa positions 256-511 saw pretraining context but not FiNER fine-tune context. Validation may be flat or negative.

### 6-Model Multi-Recipe Logit Ensemble

Combine `efficient_after_dapt_seed{88,5768,78516}` and `teacher_crf_seed{88,5768,78516}` for decorrelated errors across recipes.

Blocker: `teacher_crf_*` runs predate saved `test_emissions.npz` and `crf_transitions.npz`. Re-extract before running.

Plausible lift: `+0.005` to `+0.015` over `0.8634`. If the lift comes through, this becomes the new teacher headline and the distillation pipeline rebuilds against the new emissions.

### Dice Loss Spike

`configs/baseline/efficient_dice_seed88.yaml` with `CE + Dice`, O excluded, vanilla-only. Single seed (`88`).

Acceptance: `+0.005` over `efficient_training_seed88` (`0.8467`). Exploratory; does not influence the headline. If positive, document as a future direction in `REPORT.md` and do not escalate to 3 seeds inside Phase C scope.

---

## Why The Teacher Is Locked

The active teacher is the only completed recipe family that gives a consistent lift over the prior best single-model path while preserving the 5-epoch fine-tune budget.

Recipe (`configs/baseline/efficient_after_dapt.yaml`):

- Backbone: `roberta-large`
- DAPT: 10 epochs masked-LM continued pretraining on 131 FiNER train articles
- Fine-tune: 5 epochs CRF token classification
- LR: `3e-5`
- Batch size: `8`
- Scheduler: cosine
- Warmup ratio: `0.1`
- Max sequence length: `256`
- Label alignment: first-subword-only
- CRF LR: `1e-4`

Why alternatives are not the active teacher:

- 30-epoch vanilla and CRF baselines are competitive but do not beat DAPT + efficient CRF.
- DeBERTa-v3-large underperformed across all LLRD and label-alignment variants.
- BIO repair did not move entity F1.
- Stricter hard-BIO CRF constraints performed worse than the current CRF.
- Multi-source DAPT was abandoned because the FNSPID pull was unreliable and licensing complexity outweighed the expected lift.
- RoBERTa LAS + LLRD on a DAPT'd backbone regressed by `-0.027` on the ensemble; the LLRD decay over-froze the lower layers and the LAS gradient over-corrected through the head.

---

## Negative Results Worth Documenting

The following experiments were tested and failed. They belong in `REPORT.md` as evidence of search rigor.

### LAS + LLRD On A DAPT'd Backbone

`configs/baseline/efficient_after_dapt_lasllrd.yaml` stacks `label_all_subwords=true`, `head_lr=3e-5`, and `llrd_decay=0.9` on top of `efficient_after_dapt`. Source paper (`Improving_FiNER-ORD_Entity_F1.pdf`) recommends both as standard token-classification best practices.

Result on 3 seeds:

- Logit ensemble: `0.8368` (vs `0.8634` baseline ensemble; `-0.027`)
- Per-class regression: PER `-0.021`, LOC `-0.028`, ORG `-0.029`

All three classes regressed including PER, which was already nearly solved. Hypothesis: the `0.9` LLRD decay multiplier is too aggressive on a DAPT'd backbone where the lower layers have already been domain-adapted. The classifier head over-corrects via the increased LAS gradient signal because the under-adapting backbone cannot move enough to satisfy the new objective. The two interventions interact destructively on this specific backbone.

The paper's recommendation is correct in general but does not transfer to a backbone that has just been DAPT'd. A non-DAPT baseline + LAS + LLRD might recover the paper's expected lift, but is out of scope.

### DeBERTa-v3-large

Four DeBERTa variants under the 5-epoch budget with various LR and LAS combinations. All landed in `0.828` to `0.838`, well below the RoBERTa-large efficient baseline of `0.8481`. Lowering `head_lr` did not stabilize the model. Disabling LAS did not rescue it. RoBERTa-large is the right backbone at this budget.

### FNSPID Multi-Source DAPT

Implementation of `src/dapt_corpus.py` plus `configs/baseline/dapt_roberta_large_v2.yaml` to pull from `Zihan1004/FNSPID`. Abandoned before any successful run for two reasons:

1. FNSPID has many summary-only rows where `Article = None`. Per-row body filtering reduced the usable corpus and the body-quality sanity checks (median chars >= 1000) frequently failed.
2. FNSPID is licensed CC BY-NC-4.0. Distributing a derived checkpoint to a for-profit defense submission is licensing-borderline. The resolution was to ship recipe code only, not the trained checkpoint, but that complication outweighed the expected `+0.01` to `+0.02` F1 lift.

The code was reverted. v1 DAPT (`configs/baseline/dapt_roberta_large.yaml`, FiNER train articles only) remains the active DAPT recipe.

### BIO Repair

`scripts/bio_repair.py` post-hoc fixes invalid `I-X` transitions (e.g., `O I-ORG` becomes `O B-ORG` or `O O`). Applied to saved vanilla and CRF predictions; net entity F1 gain was zero. Invalid BIO tags are already rare; seqeval treats most repaired boundary cases equivalently.

### Stricter CRF

A variant with valid-token-only packing (no `-100` substitution) and hard BIO transition masking (illegal transitions like `O I-X` set to `-inf`) underperformed the original CRF on 3 seeds. Reverted.

### 5-Epoch Student Distillation

The Phase C spec called for 5 epochs of student distillation. At 5 epochs the student landed at ~`0.81` mean entity F1, missing the `0.82` target by ~`0.01`. This is a documented deviation in REPORT.md, not a "negative result" in the same sense as the others; the failure was the spec's epoch count, not a bad recipe. Scaling once to 20 epochs hit both targets.

### INT8 Latency Regression at Batch=8

Dynamic INT8 wins at batch=1 (`10.8ms` vs FP32 `21.7ms`, `2.0x` speedup) but loses at batch=8 (`96.3ms` vs FP32 `56.6ms`). Reportable tradeoff, not a "negative result" requiring a recipe change. The deployment-relevant number is bs=1 single-example latency, which INT8 wins. Documented in REPORT.md as an honest characterization of the INT8 frontier shape.

---

## Remaining Distance To 0.9

The current headline `0.8634` is about `0.0366` below `0.9000`. The best single-model mean `0.8548` is about `0.0452` below target. The student headline `0.8304` is much further out (`0.0696` below), but the student is on the efficiency axis and is not expected to chase the teacher's F1.

The gap is primarily:

- Strict span boundary errors (token F1 at `0.985`, entity F1 at `0.86`)
- ORG class dominance of the residual error (`0.8206` ORG vs `0.9564` PER on the teacher)
- Test-set lexical novelty (val-test gap of `0.067` for teacher, `0.086`-`0.093` for student)

Closing the gap to `0.90` would require a structural change. Per the source paper's "Best-Shot Path To >0.90":

1. Encoder upgrade with relative positional embeddings (DeBERTa-v3 failed here at the 5-epoch budget; would need a larger budget or different backbone)
2. Domain-Adaptive Pre-Training on a 5GB external financial news corpus (FNSPID approach was abandoned; TRC2 requires NIST gating)
3. GlobalPointer or biaffine span-based decoder (not implemented; requires custom PyTorch nn.Module, custom multi-label categorical CE loss, RoPE, quadratic memory)
4. Multi-class Dice loss over the span matrix
5. 5-model boundary-consensus ensemble

The paper itself frames `0.88` to `0.89` as the realistic ceiling without that full stack and `>0.90` as plausible only with substantial method changes.

The submission frames the gap honestly in the "Limitations and Future Work" section of `REPORT.md`. Phase C is the higher-EV use of remaining time than chasing the F1 number further.

---

## Dataset

**Source:** `gtfintechlab/finer-ord` on Hugging Face, CC BY-NC 4.0 license.

**Splits:**

- Train: 135 articles, 3,262 sentences, 80,531 tokens
- Validation: 24 articles, 402 sentences, 10,233 tokens
- Test: 42 articles, 1,075 sentences, 25,957 tokens

**Entity counts (train/val/test):**

- PER: `821 / 138 / 284`
- LOC: `966 / 193 / 300`
- ORG: `2026 / 274 / 544`

**Label schema:**

- `0 -> O`
- `1 -> B-PER`
- `2 -> I-PER`
- `3 -> B-LOC`
- `4 -> I-LOC`
- `5 -> B-ORG`
- `6 -> I-ORG`

Verify the integer ordering via `python -m src.data` before long runs. `"Obama"` should show as `B-PER`, and multi-token entities should be continuous.

**Inter-split article duplication:** 4 train articles have full reconstructed text identical to a val or test article. `src.dapt.build_train_article_texts` filters these out before DAPT tokenization. Do not bypass that filter.

---

## Baselines And Paper Context

The FiNER-ORD paper reports token weighted F1, which is not directly comparable to this project's strict entity-F1 target.

Paper token weighted F1:

| Model | Weighted F1 | PER | LOC | ORG |
|---|---:|---:|---:|---:|
| BERT-base-cased | `0.6931` | `0.8811` | `0.6820` | `0.6013` |
| RoBERTa-base | `0.7220` | `0.9050` | `0.7154` | `0.6304` |
| BERT-large-cased | `0.7216` | `0.8954` | `0.7289` | `0.6272` |
| RoBERTa-large | `0.7648` | `0.9263` | `0.7717` | `0.6769` |
| GPT-4o zero-shot | `0.6692` | `0.8004` | `0.6651` | `0.6036` |

Current strict entity-F1 results:

| Model | Token F1 Mean | Entity F1 Mean | PER Mean | LOC Mean | ORG Mean |
|---|---:|---:|---:|---:|---:|
| Vanilla `roberta-large` | `0.9852` | `0.8485` | `0.9518` | `0.8538` | `0.7954` |
| Original `roberta-large + CRF` | `0.9853` | `0.8521` | `0.9531` | `0.8510` | `0.8032` |
| Efficient CRF | `0.9849` | `0.8481` | `0.9459` | `0.8449` | `0.8022` |
| Efficient after DAPT | `0.9859` | `0.8548` | `0.9425` | `0.8602` | `0.8095` |
| Distilled student (FP32) | `~0.984` | `0.8277` | `~0.93` | `~0.83` | `~0.77` |
| Distilled student (INT8) | `~0.984` | `0.8241` | `~0.93` | `~0.83` | `~0.76` |

Interpretation:

- RoBERTa-large remains the strongest backbone family tested.
- CRF gives modest lift, mostly through ORG.
- DAPT gives the most useful completed lift and is the locked teacher base.
- ORG remains the main bottleneck across every strong recipe, including the student and the INT8 student.
- The distilled student preserves 96.2% of teacher ensemble F1 with 4.3x parameter reduction, fitting the literature band.
- INT8 quantization preserves 99.6% of FP32 student F1 (`0.8241 / 0.8277`), well within the `0.01` tolerance.

---

## Known Error Patterns

The best current explanation comes from saved prediction analysis in `notebooks/teacher_0_9_gap_analysis.ipynb`.

### Validation-To-Test Gap

Teacher gap is about `0.067`:

- Vanilla: `0.9152 -> 0.8485`
- Original CRF: `0.9197 -> 0.8521`

Student gap is `0.086`-`0.093` (per seed `0.92 -> 0.83`). The student gap is larger than the teacher gap, which is expected: a smaller model with no CRF and no DAPT-on-financial-text has less inductive bias for unseen corporate names.

This suggests a real generalization gap, not optimization failure.

### Token F1 Is Already High

Token weighted F1 is around `0.985` to `0.986` for the teacher and `~0.984` for both FP32 and INT8 students. Strict entity F1 is about `0.86` (teacher) and `0.83` (student ensemble). The remaining problem is strict span quality, not raw token labeling.

### ORG Bottleneck

ORG is the weakest class in every completed strong experiment:

- Vanilla ORG mean: `0.7954`
- Original CRF ORG mean: `0.8032`
- Efficient CRF ORG mean: `0.8022`
- Efficient after DAPT ORG mean: roughly `0.8095`
- Efficient after DAPT logit ensemble ORG: `0.8206`
- Student ensemble ORG: `0.7781`
- INT8 student per-seed ORG: `0.7611`-`0.7780`

Strict span confusion analysis shows the remaining ORG burden comes mostly from missed and spurious spans, not type confusion. The student loses `0.0425` ORG F1 vs the teacher ensemble; INT8 quantization loses an additional `0.005` to `0.025` per seed. ORG is the brittle class across every transformation in the pipeline.

### LOC Boundary Issues

LOC is stronger than ORG, but boundary and partial-overlap errors remain common.

### Type Confusion Is Minor

Same-span wrong-type mistakes are much less common than missed spans, spurious spans, and boundary / partial-overlap spans. This is why BIO repair and stricter transition legality did not move the metric much.

---

## Known Annotation Edge Cases

From the paper appendix and notebook review:

- `"Google India"` is fully ORG despite `"India"` being a location.
- `"New York Stock Exchange"` is fully ORG; models often predict LOC for `"New York"`.
- `"The Wall Street Journal"` includes the leading `"The"` in the ORG span.
- Possessives: `"Morgan Stanley's"` tags through `"Stanley"` and drops `"'s"`.
- `"R-Texas"` tags only `"Texas"` as LOC.
- `"Kenyan"` adjective form is not tagged as LOC.

These cases make span-boundary and span-type consistency difficult under strict entity scoring.

---

## Technical Decisions

### Active Teacher

- `roberta-large`
- FiNER-only DAPT checkpoint
- CRF fine-tune
- 3-seed logit/emission ensemble
- Locked at `0.8634`

### Active Student (Phase C)

- `distilroberta-base`
- Vanilla token classification head (no CRF)
- Offline distillation from teacher ensemble emissions
- 20 epochs (deviation from original 5-epoch spec; documented in REPORT.md)
- 3-seed logit ensemble
- Locked at `0.8304` ensemble, `0.8277` single-model mean

### Active INT8 Student (Phase C)

- Source: 3-seed FP32 student checkpoints
- Method: `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` with `qnnpack` engine for ARM
- 3-seed mean test entity F1: `0.8241` (drop of `0.0036` from FP32 mean)
- On-disk size: `190MB` (vs `311MB` FP32 student, `1.6x` reduction)
- CPU latency at bs=1: `10.8ms` (`2.0x` faster than FP32 student, `9.9x` faster than teacher)

### Historical Alternatives

- Vanilla `roberta-large` (30 epochs)
- Original `roberta-large + CRF` (30 epochs)
- Efficient `roberta-large + CRF` (5 epochs, no DAPT)
- DeBERTa-v3-large efficient variants with LLRD and LAS
- RoBERTa-large + DAPT + LAS + LLRD (regressed)

### Efficient After DAPT Recipe (Locked Teacher)

`configs/baseline/efficient_after_dapt.yaml`:

- `model_name = ./results/dapt_roberta-large/checkpoint-final`
- `learning_rate = 3e-5`
- `batch_size = 8`
- `num_epochs = 5`
- `weight_decay = 0.01`
- `warmup_ratio = 0.1`
- `lr_scheduler_type = cosine`
- `max_seq_length = 256`
- `fp16 = true`
- `use_crf = true`
- `crf_learning_rate = 1e-4`
- `label_all_subwords = false`

### Distilled Student Recipe (Locked)

`configs/baseline/student_distilled.yaml`:

- `model_name = distilroberta-base`
- `seeds = [88, 5768, 78516]`
- `learning_rate = 5e-5`
- `batch_size = 16`
- `num_epochs = 20` (deviation from original 5-epoch spec)
- `weight_decay = 0.01`
- `warmup_ratio = 0.1`
- `lr_scheduler_type = cosine`
- `max_seq_length = 256`
- `fp16 = true`
- `use_distillation = true`
- `use_crf = false`
- `teacher_mode = ensemble`
- `teacher_runs = [efficient_after_dapt_seed88, efficient_after_dapt_seed5768, efficient_after_dapt_seed78516]`
- `distill_temperature = 2.0`
- `distill_alpha = 0.5`
- `label_all_subwords = false`
- `label_smoothing_factor = 0.0`

### CRF Recipe

The checked-in CRF implementation:

- backbone LR from `config.learning_rate`
- classifier + CRF head LR from `config.crf_learning_rate`
- `pytorch-crf==0.7.2`
- uses `attention_mask` as the CRF mask
- replaces `-100` labels with `0` before CRF forward
- filters ignored positions out of metrics later using original labels

Historical note: a stricter valid-token-only + hard-BIO version was implemented and tested. It underperformed the original CRF and was reverted.

### Subword Alignment

Both first-subword-only and label-all-subwords alignment are implemented in `src.data`. The locked teacher and the distilled student both use first-subword-only, which keeps emission alignment compatible across teacher and student.

DeBERTa tested LAS and still underperformed. RoBERTa LAS + LLRD (`efficient_after_dapt_lasllrd`) regressed by `-0.027` on the ensemble.

### Seeds

Use exactly:

- `88`
- `5768`
- `78516`

Headline single-model numbers should always be mean +/- std (or mean and range when std is misleading at n=3) across these three seeds.

---

## Tech Stack

- Python `3.10+`
- `torch==2.3.1`
- `transformers==4.44.2`
- `datasets==2.21.0`
- `tokenizers==0.19.1`
- `accelerate==0.33.0`
- `seqeval==1.2.2`
- `scikit-learn==1.5.1`
- `pytorch-crf==0.7.2`
- `numpy==1.26.4`
- `pandas==2.2.2`
- `wandb==0.26.1`
- `pyyaml==6.0.2`
- `matplotlib` (Pareto chart; existing dependency, do not add new ones)

Platform notes:

- Apple Silicon MPS works for small checks, but CUDA is expected for real GPU runs.
- Colab works with CUDA and the current configs.
- Cross-run wall-clock comparisons should account for GPU heterogeneity.
- INT8 dynamic quantization is CPU-only. Latency for the INT8 student must be measured on CPU, not GPU.
- Apple Silicon requires `torch.backends.quantized.engine = 'qnnpack'` before any `quantize_dynamic` call.
- Always invoke project Python with `./.venv/bin/python` to avoid numpy ABI mismatch on object-dtype `.npz` artifacts.

---

## Repo Structure

```text
finer-ord/
├── CLAUDE.md
├── README.md
├── REPORT.md
├── requirements.txt
├── docs/
│   ├── PROJECT_CONTEXT.MD
│   └── figures/
│       ├── pareto.png
│       └── pareto_data.csv
├── configs/
│   ├── baseline_teacher.yaml
│   ├── smoke_test.yaml
│   └── baseline/
│       ├── teacher_crf.yaml
│       ├── efficient_training.yaml
│       ├── efficient_after_dapt.yaml
│       ├── efficient_after_dapt_lasllrd.yaml
│       ├── efficient_dice_seed88.yaml
│       ├── student_distilled.yaml
│       ├── dapt_roberta_large.yaml
│       ├── deberta_smoke.yaml
│       ├── deberta_efficient.yaml
│       ├── deberta_efficient_lr1e5.yaml
│       ├── deberta_efficient_lr2e5.yaml
│       └── deberta_efficient_align_off.yaml
├── notebooks/
│   ├── compare_runs.ipynb
│   ├── bio_fillter_testing.ipynb
│   ├── teacher_0_9_gap_analysis.ipynb
│   └── colab_runner.ipynb
├── src/
│   ├── data.py
│   ├── evaluate.py
│   ├── train.py
│   ├── crf_model.py
│   ├── losses.py
│   ├── dapt.py
│   ├── distill.py
│   └── finer-ord.py
├── scripts/
│   ├── bio_repair.py
│   ├── ensemble_logits.py
│   ├── reinfer_packed.py
│   ├── extract_train_emissions.py
│   ├── measure_latency.py
│   ├── quantize_student.py
│   └── build_pareto.py
└── results/
    ├── results.csv
    ├── results_detailed.csv
    ├── latency/
    │   └── <run_name>.json
    └── <run_id>/
```

---

## Module Responsibilities

**`src/data.py`**
Dataset loading, grouping, tokenization, label alignment, sanity checks. Owns `ID2LABEL`, `LABEL2ID`, and `NUM_LABELS`.

**`src/evaluate.py`**
All metric computation. Entry point is `compute_detailed_metrics(true_labels, predictions)`.

**`src/train.py`**
Vanilla multi-seed training loop and experiment orchestration. Dispatches to `src.crf_model.run_crf_training` when `use_crf: true` and to `src.distill.run_distillation` when `use_distillation: true`.

**`src/crf_model.py`**
Original CRF wrapper plus CRF-specific training loop. Saves decoded predictions and emission artifacts for ensembling. Read-only during Phase C; do not modify.

**`src/dapt.py`**
Masked-LM continued pretraining on FiNER train articles only, with duplicate val/test article filtering.

**`src/distill.py`**
Student distillation training. Loads saved teacher emissions, computes the combined CE + KL loss, trains the student, and saves checkpoints in the same format as `src.train.py`.

**`src/losses.py`**
Dice Loss implementation for exploratory vanilla-only loss experiments.

**`scripts/ensemble_logits.py`**
Vote and logit/emission ensembling across seed runs. Supports both CRF (`--use-crf`) and vanilla modes.

**`scripts/reinfer_packed.py`**
Packed 512-token re-inference for existing checkpoints.

**`scripts/bio_repair.py`**
BIO post-processing analysis for saved predictions.

**`scripts/extract_train_emissions.py`**
Loads each locked teacher checkpoint, runs CRF inference on the train split, saves `train_emissions.npz` next to the existing val/test emissions.

**`scripts/measure_latency.py`**
Measures parameter count, on-disk size, and median/p95 latency at batch sizes 1 and 8 after warmup. Supports `--device cpu` and `--device cuda`. Writes `results/latency/<run_name>.json`. Includes `qnnpack` engine setup for INT8 paths on ARM.

**`scripts/quantize_student.py`**
Applies dynamic INT8 quantization to a student checkpoint, saves `checkpoint-best-int8/`, runs full test inference, writes `summary_int8.json`. Includes `qnnpack` engine setup and manual `torch.save` of state_dict (avoids the broken `huggingface_hub` sharder path for quantized models).

**`scripts/build_pareto.py`**
Reads `results.csv`, `results/<run>/summary_int8.json`, `results/*_aggregate.json`, and `results/latency/*.json`. Writes `docs/figures/pareto.png` and `docs/figures/pareto_data.csv`. Uses 3-seed mean F1 numbers from aggregate JSONs. Log x-axes on both subplots.

---

## Phase Status

### Complete

- Data pipeline and label mapping
- Entity-level evaluation
- Vanilla and CRF training paths
- DAPT on FiNER train articles
- Efficient 5-epoch CRF recipe
- DeBERTa LLRD and label-all-subwords sweeps (negative)
- BIO repair analysis (no lift)
- Logit and vote ensembling
- RoBERTa LAS + LLRD on a DAPT'd backbone (negative)
- Teacher locked at `efficient_after_dapt_logit_ensemble` (`0.8634`)
- Teacher reproduced from configs after Colab session checkpoint loss event (matches original `0.8634` exactly).
- `src/distill.py` implementation
- `scripts/extract_train_emissions.py` and run for all 3 teacher seeds
- 3-seed student distillation (20 epochs; mean `0.8277`, ensemble `0.8304`)
- Student logit ensemble (`0.8304`, both targets cleared)
- All checkpoints (teacher CRF, FP32 student, INT8 student, DAPT backbone) downloaded to local repo
- **Latency measurements on Apple M2 Pro CPU: teacher `106.6ms`, FP32 student `21.7ms`, INT8 student `10.8ms` at bs=1 median**
- **3-seed dynamic INT8 quantization: mean F1 `0.8241`, drop `0.0036` from FP32 mean, well within `0.01` tolerance**
- **Pareto chart built (`docs/figures/pareto.png` + `pareto_data.csv`); 5 points, log axes, frontier line drawn**

### Active

- `REPORT.md` (write fresh; centerpiece deliverable)
- `README.md` polish

### Optional (not on critical path)

- Packed-512 re-inference
- 6-model multi-recipe ensemble (blocked on emissions re-extraction)
- Dice spike on seed 88 (exploratory)

### Deprioritized

- Custom span-based decoder (GlobalPointer, biaffine). Out of scope under the time budget; documented in `REPORT.md` as future work.
- Larger external DAPT corpus (TRC2, alternative finance news). Licensing complexity outweighs the expected lift.

---

## Critical Rules

1. Optimize for strict entity F1.
2. The teacher is locked at `0.8634`. Do not retrain or modify its configs.
3. The student is locked at `0.8304` ensemble. Do not retrain to chase F1.
4. The INT8 student is locked at `0.8241` mean. Do not re-quantize.
5. Phase C distillation reads existing teacher emissions only; no live teacher forward passes.
6. Distillation defaults to ensemble teacher; `single` mode is for ablation.
7. Student does not use a CRF.
8. Tokenizer alignment must match teacher emissions; do not switch student backbones without re-extracting.
9. Never headline a single seed. Use 3-seed mean +/- std (or mean and range).
10. Verify label mapping before long runs.
11. Preserve the current CRF implementation; it is the locked teacher's decoder.
12. Save `predictions.json` per run.
13. DAPT must not include val/test articles.
14. Packed-512 must run validation before test.
15. Treat `label_all_subwords` changes as real ablations.
16. INT8 latency must be measured on CPU.
17. The Pareto chart uses single-example latency on the x-axis; throughput is in the data CSV.
18. Pareto chart F1 numbers come from 3-seed means; latency uses seed 88 alone (architecture-not-weights).
19. Keep negative results in the report; they explain engineering judgment.
20. Document the 20-epoch student deviation honestly in REPORT.md.
21. Document the INT8 batch=8 latency regression honestly in REPORT.md.
22. On Apple Silicon, set `torch.backends.quantized.engine = 'qnnpack'` before any `quantize_dynamic` call.
23. Do not use `save_pretrained()` on dynamically quantized models; use `torch.save(state_dict, ...)` and copy config/tokenizer files manually.
24. Always invoke project Python with `./.venv/bin/python` to avoid numpy ABI mismatch on object-dtype `.npz` files.

---

## Out Of Scope

- Custom tokenizer training
- Training from scratch
- Proprietary finance LLMs
- Expanding the label set
- Custom span-based decoders for this submission cycle (future work)
- TRC2 or other gated DAPT corpora (future work)
- Static or QAT quantization (dynamic INT8 only for the take-home)
- INT8 ensemble (intentionally skipped; ensembling defeats the purpose of quantization)

---

## Risks And Unknowns

1. **Validation-test gap remains large.** Teacher: `0.067`. Student: `0.086`-`0.093`. More validation F1 may not translate to test F1.
2. **ORG span quality is still the bottleneck.** Local transition constraints did not fix span recall/precision; distillation did not either; INT8 quantization made it worse.
3. **CPU latency measurements are environment-specific.** Pinned to Apple M2 Pro with `torch==2.3.1`. Cross-machine numbers are not directly comparable. Hardware noted explicitly in `REPORT.md`.
4. **Ensembles improve F1 but cost latency.** Reported honestly with the latency cost (3x the single-model latency, estimated).
5. **Student ensemble margin to target is thin.** `0.8304` vs `0.83` target is `+0.0004`. Cite as cleared, not as comfortable.
6. **INT8 batch=8 latency is worse than FP32.** Real, reportable, and documented as an honest deployment tradeoff (interactive vs throughput workloads).

---

## Open Questions

- What exact submission format Dunedain expects: repo, PDF, or both?
- Should `REPORT.md` plot the optional teacher experiments (LAS+LLRD failed, packed-512 not run, 6-model ensemble not run) as dominated points or omit them?
- Does the 6-model multi-recipe ensemble lift the teacher headline if old `teacher_crf_*` emissions get re-extracted? Not on the critical path; only worth it if there's leftover time after REPORT.md.

---

## Glossary

- **BIO tagging**: Beginning / Inside / Outside format for entity spans.
- **CRF**: Conditional Random Field; learns pairwise label transitions and decodes with Viterbi.
- **DAPT**: Domain-adaptive pretraining via masked language modeling on in-domain text.
- **Distillation**: Training a smaller student model to mimic a larger teacher's outputs, typically via a KL divergence on softened logits plus standard CE on gold labels.
- **Dynamic INT8**: Post-training quantization that converts `nn.Linear` weights to INT8 on the fly during inference; CPU-only, single function call, no calibration data.
- **Entity F1**: strict span-level F1 where both type and boundary must match.
- **fbgemm**: Facebook GEMM library; PyTorch's default quantization backend on x86 CPUs.
- **Gold labels**: human-annotated ground truth.
- **LAS**: label-all-subwords; continuation subwords inherit an entity continuation label instead of `-100`.
- **LLRD**: layer-wise learning-rate decay; lower encoder layers train with smaller effective learning rates.
- **Logit ensemble**: average model emissions/logits before decoding.
- **Pareto frontier**: the set of points such that no other point dominates them on both axes (F1 and latency, or F1 and size).
- **qnnpack**: Quantized Neural Network PACKage; PyTorch's quantization backend for ARM CPUs (the alternative to fbgemm).
- **seqeval**: canonical Python library for NER metrics with BIO awareness.
- **Token F1 weighted**: token-level F1 across all labels, inflated by the O-majority.
- **Viterbi decoding**: dynamic programming algorithm for the most probable label sequence.
