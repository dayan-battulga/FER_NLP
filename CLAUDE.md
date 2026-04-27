# CLAUDE.md

Working context for the FiNER-ORD Financial NER project. Keep this file current. Update the "Current Status" section first when the repo state changes.

---

## Project

Take-home challenge for the ML/NLP Research Engineer internship at Dunedain. Build a Financial NER model on the `gtfintechlab/finer-ord` dataset. The task is BIO-tagged token classification with 7 labels across PER / LOC / ORG entities.

**Primary metric:** entity-level F1 via `seqeval` (confirmed with Daniel).

**Submission framing:** "F1 or efficiency wins." We optimize for both. The deliverable is a Pareto frontier showing teacher F1, distilled student F1, and INT8 quantized student F1 against latency and model size.

**Current strategy:** the teacher is locked. All remaining performance work is on the student / distillation / quantization track. Cheap remaining teacher experiments (multi-recipe 6-model ensemble, packed-512 re-inference) may be run as inference-only side experiments while Phase C work proceeds, but they do not block Phase C delivery.

**Locked teacher:** `efficient_after_dapt_logit_ensemble` at **`0.8634`** test entity F1. Best single-model 3-seed mean is `efficient_after_dapt` at **`0.8548 +/- 0.0038`**.

**Student target:** distilled student at >= `0.82` single-model 3-seed mean, >= `0.83` logit ensemble. INT8 student at <= `0.01` F1 drop from FP32 student. Pareto frontier should show student down-and-to-the-left of teacher.

---

## Current Status

### Completed teacher work

- [x] Initial one-seed baselines (BERT-base, RoBERTa-base, RoBERTa-large). Best of that pass: RoBERTa-large at `0.8471` entity F1 on seed `88`.
- [x] Metric confirmed with Daniel: strict entity-level `seqeval` F1, not the paper's token weighted F1.
- [x] Repo scaffolded around config-driven vanilla and CRF training.
- [x] `src/data.py`, `src/evaluate.py`, `src/train.py`, `src/crf_model.py`, `src/dapt.py`, `src/losses.py`, and the analysis scripts are written.
- [x] Smoke test passed.
- [x] Phase B vanilla teacher: 3 seeds, `0.8485 +/- 0.0022` (30 epochs).
- [x] Phase B CRF: 3 seeds, `0.8521 +/- 0.0018` (30 epochs).
- [x] Efficient 5-epoch CRF baseline: 3 seeds, `0.8481 +/- 0.0066`.
- [x] DeBERTa-v3-large efficient sweep: all variants regressed below RoBERTa-large.
  - `deberta_efficient`: `0.8374 +/- 0.0018`
  - `deberta_efficient_lr1e5`: `0.8282 +/- 0.0026`
  - `deberta_efficient_lr2e5`: `0.8325 +/- 0.0058`
  - `deberta_efficient_align_off`: `0.8339 +/- 0.0027`
- [x] BIO repair analyzed on saved predictions. No entity-F1 gain.
- [x] Stricter CRF (valid-token-only packing, hard BIO transitions) tested locally and reverted; underperformed the original CRF.
- [x] Phase 3 DAPT (10 epochs MLM on 131/135 FiNER train articles, 180 windows of 512 tokens) plus 5-epoch CRF fine-tune: `efficient_after_dapt` at `0.8548 +/- 0.0038`. Mean fine-tune time `7.21` minutes per seed plus `~3.5` minutes one-time DAPT. All three DAPT seeds beat the non-DAPT efficient mean.
- [x] Phase 4 ensembles on top of `efficient_after_dapt`:
  - `efficient_after_dapt_logit_ensemble`: **`0.8634`** test entity F1 (PER `0.9564`, LOC `0.8590`, ORG `0.8206`)
  - `efficient_after_dapt_vote_ensemble`: `0.8587`
- [x] FNSPID-based DAPT v2 attempted in code, abandoned before running. Many summary-only rows; CC BY-NC license was incompatible with shipping a derived checkpoint to a for-profit submission.
- [x] RoBERTa LAS + LLRD ablation (`efficient_after_dapt_lasllrd.yaml`) completed for 3 seeds. Logit ensemble landed at `0.8368`, a `-0.027` regression vs the `efficient_after_dapt` ensemble. All three classes regressed (PER `-0.021`, LOC `-0.028`, ORG `-0.029`). Hypothesis: `llrd_decay=0.9` is too aggressive on a DAPT'd backbone where the lower layers were already domain-adapted; combined with LAS, the head over-corrects for an under-adapting backbone. Not promoted; documented as a negative result.
- [x] Teacher locked at `efficient_after_dapt_logit_ensemble` (`0.8634`).

### Active Phase C work

- [ ] Implement `src/distill.py` for offline distillation from saved teacher emissions. KL on softened teacher emissions plus CE on gold labels, weighted `alpha * CE + (1 - alpha) * KL * T^2`. Defaults: `T=2.0`, `alpha=0.5`.
- [ ] Add `scripts/extract_train_emissions.py`. Loads each locked teacher checkpoint, runs inference on the train split, saves `results/<run>/train_emissions.npz` next to the existing val/test emissions.
- [ ] Update `configs/baseline/student_distilled.yaml` with the real student recipe: `model_name=distilroberta-base`, `seeds=[88, 5768, 78516]`, `lr=5e-5`, `batch_size=16`, `num_epochs=5`, `use_distillation=true`, `use_crf=false`, `teacher_mode=ensemble`, `teacher_runs=[efficient_after_dapt_seed88, efficient_after_dapt_seed5768, efficient_after_dapt_seed78516]`, `distill_temperature=2.0`, `distill_alpha=0.5`.
- [ ] Update `TrainConfig` in `src/train.py` to add the four distillation fields and replace the `use_distillation` `NotImplementedError` gate with delegation to `src.distill.run_distillation`. Match the existing `use_crf` dispatch pattern.
- [ ] Run 3-seed student distillation. Ensemble via `scripts/ensemble_logits.py` in `--mode logit` (vanilla, no `--use-crf`).
- [ ] Add `scripts/measure_latency.py`. Reports parameter count, checkpoint MB, median and p95 latency at batch sizes 1 and 8 after a 10-batch warmup. Supports CPU and GPU via `--device`. Writes `results/latency/<run_name>.json`.
- [ ] Add `scripts/quantize_student.py`. Applies dynamic INT8 to `nn.Linear` modules, saves `checkpoint-best-int8/`, runs full test inference, writes `summary_int8.json`. INT8 latency must be measured on CPU; dynamic quantization is a no-op on GPU.
- [ ] Add `scripts/build_pareto.py`. Builds `docs/figures/pareto.png` (latency-vs-F1 and size-vs-F1 subplots) plus `docs/figures/pareto_data.csv`. Uses single-example latency (batch_size=1) on the x-axis; throughput is in the CSV but not on the chart.
- [ ] Write `REPORT.md` and a real `README.md`.

### Optional teacher-side experiments (do not block Phase C)

- [ ] Phase 5c packed-512 re-inference. Run validation first on `efficient_after_dapt_seed*`. Test only if validation delta is non-negative.
- [ ] Phase 5d Dice spike. CE + Dice with O excluded, vanilla-only. Single seed (88). Acceptance: `+0.005` over `efficient_training_seed88` (`0.8467`). Exploratory; not part of the headline.
- [ ] Phase 5e 6-model multi-recipe ensemble. Blocked on re-extracting `*_emissions.npz` and `crf_transitions.npz` for `teacher_crf_seed*`. Cheap once unblocked.

---

## Critical Rules

These are easy to get wrong. Do not.

1. **Optimize for entity F1 on the teacher and the student.** Token weighted F1 is useful context but not the headline metric.

2. **The teacher is locked.** Do not retrain `efficient_after_dapt_seed*`, do not change `configs/baseline/efficient_after_dapt.yaml`, and do not modify `src/crf_model.py`. Phase C reads existing teacher checkpoints only.

3. **Distillation is offline, from saved emissions.** Generate `train_emissions.npz` once via `scripts/extract_train_emissions.py` and reuse it across student seeds. No live teacher forward passes during student training.

4. **Distillation defaults to ensemble teacher.** Average emissions across the 3 teacher seeds before computing the soft loss. `teacher_mode=single` exists for ablation; do not use it for the headline.

5. **Student does not use a CRF.** Vanilla token classification head. The CRF added marginal lift on the teacher and adds inference-time cost the deployment story does not want. Document this choice in `src/distill.py`.

6. **Tokenizer alignment must match.** Teacher emissions were generated with `label_all_subwords=false`, `max_seq_length=256`, and the RoBERTa BPE tokenizer with `add_prefix_space=True`. The student uses `distilroberta-base`, which shares the same BPE; alignment is preserved. Do not switch to a different student tokenizer without re-extracting emissions.

7. **`add_prefix_space=True` when `is_split_into_words=True`.** Applies to RoBERTa and DeBERTa families uniformly.

8. **Label mapping integers preserve the HF dataset's original ordering.** `0 -> O`, `1 -> B-PER`, `2 -> I-PER`, `3 -> B-LOC`, `4 -> I-LOC`, `5 -> B-ORG`, `6 -> I-ORG`.

9. **Sanity check label mapping before any long run.** `python -m src.data` should show `"Obama" -> B-PER` and continuous multi-token entities.

10. **The checked-in CRF implementation is the original one.** Uses `attention_mask` as the CRF mask, replaces `-100` with `0` before CRF forward. Do not assume any stricter variant is active.

11. **Save `predictions.json` per run.** BIO repair, error analysis, and the Pareto chart depend on it.

12. **Three seeds for headline single-model numbers.** Use `88, 5768, 78516` and report mean +/- std. Do not headline a single lucky seed.

13. **Always annotate F1 comparisons with epoch budget and recipe family.** Vanilla and original CRF baselines ran for 30 epochs; efficient recipes and the student run for 5 epochs.

14. **DAPT corpus must never include val/test articles.** `src.dapt.build_train_article_texts` filters out the 4 FiNER train articles whose reconstructed text exactly matches a val/test article. Do not bypass that filter.

15. **DAPT stays on FiNER train articles only.** Multi-source DAPT was attempted and abandoned for licensing and quality reasons.

16. **Ensemble logit mode requires byte-identical gold labels and per-example lengths across seed dumps.** `scripts/ensemble_logits.py` asserts both.

17. **Dice Loss and CRF stay separate.** `TrainConfig.__post_init__` raises if `loss_type=dice` and `use_crf=true`. Distillation also forbids `use_crf=true` with `use_distillation=true`.

18. **Dice Loss runs as `CE + Dice`, not pure Dice.** Pure dice on FiNER's O-heavy distribution collapsed to predicting O everywhere.

19. **Packed-512 re-inference runs validation before test.** Position embeddings 256-511 saw pretrain context but not FiNER fine-tune context. Only run on test if the validation delta is non-negative.

20. **INT8 quantization latency must be measured on CPU.** `torch.quantization.quantize_dynamic` is CPU-targeted; on GPU the quantized checkpoint runs at FP32 speed and the Pareto chart will look identical to the unquantized student.

---

## Repo Map

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
    └── latency/
        └── <run_name>.json
```

**Single source of truth for label mapping:** `src/data.py` defines `ID2LABEL`, `LABEL2ID`, and `NUM_LABELS`.

**Single entry point for metrics:** `src/evaluate.py::compute_detailed_metrics()` returns a JSON-serializable dict with token F1, entity F1, both confusion matrices, and per-class F1.

---

## Run Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login

# Verify data and pipeline
python -m src.data
python -m src.train --config configs/smoke_test.yaml --run-checks

# Historical baselines (for context, do not re-run)
python -m src.train --config configs/baseline_teacher.yaml
python -m src.train --config configs/baseline/teacher_crf.yaml
python -m src.train --config configs/baseline/efficient_training.yaml

# Locked teacher path (already complete, do not re-run)
python -m src.dapt  --config configs/baseline/dapt_roberta_large.yaml
python -m src.train --config configs/baseline/efficient_after_dapt.yaml

# Locked teacher ensemble (already complete; this is the headline)
python scripts/ensemble_logits.py \
  --runs efficient_after_dapt_seed88 efficient_after_dapt_seed5768 efficient_after_dapt_seed78516 \
  --mode logit \
  --use-crf \
  --output-name efficient_after_dapt_logit_ensemble

# Phase C: extract train emissions, then run student distillation
python scripts/extract_train_emissions.py \
  --runs efficient_after_dapt_seed88 efficient_after_dapt_seed5768 efficient_after_dapt_seed78516

python -m src.distill --config configs/baseline/student_distilled.yaml --smoke
python -m src.distill --config configs/baseline/student_distilled.yaml

# Student ensemble (vanilla, no --use-crf)
python scripts/ensemble_logits.py \
  --runs student_distilled_seed88 student_distilled_seed5768 student_distilled_seed78516 \
  --mode logit \
  --output-name student_distilled_logit_ensemble

# Latency, quantization, Pareto
python scripts/measure_latency.py --runs efficient_after_dapt_seed88 student_distilled_seed88 --device cuda
python scripts/quantize_student.py --runs student_distilled_seed88 student_distilled_seed5768 student_distilled_seed78516
python scripts/measure_latency.py --runs student_distilled_seed88_int8 --device cpu
python scripts/build_pareto.py

# Optional teacher-side side experiments
python scripts/reinfer_packed.py \
  --runs efficient_after_dapt_seed88 efficient_after_dapt_seed5768 efficient_after_dapt_seed78516 \
  --mode crf --split val
python -m src.train --config configs/baseline/efficient_dice_seed88.yaml
```

Add `--no-wandb` to training commands when logging is not desired.

---

## Key Technical Choices

**Locked teacher:** `roberta-large + FiNER-only DAPT + CRF`, ensemble of 3 seeds. Headline is `efficient_after_dapt_logit_ensemble` at `0.8634` test entity F1.

**Why this teacher path:** DAPT gave a consistent lift across all three seeds and kept the 5-epoch fine-tune budget. DeBERTa did not improve under the same budget. BIO repair, stricter CRF, FNSPID, and RoBERTa LAS + LLRD all failed to improve the recipe.

**Known teacher bottleneck:** ORG is the weakest class. The best ensemble ORG F1 is `0.8206`, far below PER `0.9564`. Most of the residual error is missed and spurious ORG spans, not type confusion.

**Student backbone:** `distilroberta-base`. 82M parameters vs RoBERTa-large's 355M (4.3x reduction). Same BPE tokenizer as the teacher, so emission alignment transfers without re-extraction.

**Distillation loss:** weighted combination of `alpha * CE(student_logits, gold)` and `(1 - alpha) * T^2 * KL(softmax(student_logits / T) || softmax(teacher_emissions / T))`. Defaults `T=2.0`, `alpha=0.5`. Mask `-100` positions identically in both terms.

**Ensemble teacher:** average teacher emissions across the 3 seeds before computing the soft loss. Distilling from the ensemble consistently outperforms distilling from a single teacher (typically by `+0.005` to `+0.010` student F1) because the averaged logits are smoother and more informative.

**Student does not use a CRF.** The CRF added `+0.004` on the teacher; on a smaller student where capacity is the binding constraint, that lift is unlikely to transfer, and the CRF adds inference-time Viterbi cost that hurts the deployment story.

**Quantization:** dynamic INT8 via `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`. Single function call, CPU-targeted, no QAT or static quantization. Documented in `REPORT.md` as the appropriate level for a take-home.

**Pareto chart:** two subplots. Latency-vs-F1 uses single-example latency at batch size 1 (the deployment-relevant number). Size-vs-F1 uses on-disk model.safetensors size. Both axes plot all six points: teacher single, teacher ensemble, student single, student ensemble, INT8 student single, INT8 student ensemble. Frontier marked explicitly.

**Pinned versions that matter:** `transformers==4.44.2`, `seqeval==1.2.2`, `numpy==1.26.4`, `torch==2.3.1`, `sentencepiece==0.2.0`. Do not upgrade blindly.

---

## Known Error Patterns

From saved prediction analysis and strict span confusion notebooks:

- **Validation-to-test gap is large.** Vanilla and CRF both drop by about `0.067` entity F1 from validation to test.
- **Token F1 is already very high.** Models sit around `0.985` token weighted F1, so the remaining problem is strict entity-span quality.
- **ORG is the main bottleneck.** Missed and spurious ORG spans dominate the remaining error burden.
- **LOC is the second bottleneck.** Boundary and partial-overlap errors are still common.
- **Type confusion is small.** Same-span wrong-type mistakes are much less important than missed, spurious, and boundary-overlap errors.
- **BIO repair does not help.** Invalid BIO tags are not the limiting issue.

Interventions tied to those patterns:

- **DAPT + efficient CRF** completed and is the locked teacher path.
- **Logit ensemble** completed and is the headline.
- **LAS + LLRD on a DAPT'd backbone** failed; documented as negative.
- **Distillation from the ensemble** is the next active intervention.
- **Span-aware decoding** remains the most plausible larger change for future work.

---

## Style And Workflow Preferences

- **No em dashes.** Applies to report writing too.
- **Direct tone, grounded in specifics.** Avoid corporate jargon.
- **Pareto-first framing.** The submission is a frontier story (teacher F1, student F1, INT8 F1, latency, size), not a single-number story.
- **Validation before test for inference experiments.** Do not burn the test set on speculative decoding changes.
- **Honest about negative results.** Failed ablations are useful evidence and should be in `REPORT.md`.

---

## Open Questions

- Does the student logit ensemble clear `0.83` test entity F1?
- Does INT8 quantization stay within `0.01` F1 of the FP32 student?
- What are the realistic CPU and GPU latency numbers for teacher single, teacher ensemble, student, and INT8 student?
- Should `scripts/build_pareto.py` plot the optional teacher experiments (LAS+LLRD failed, packed-512 pending) as dominated points, or omit them?
- Does the 6-model multi-recipe ensemble lift the teacher headline if old `teacher_crf_*` emissions get re-extracted? Cheap to check; not on the critical path.
- What exact submission format Dunedain expects: repo link, PDF, or both?

---

## Notes For Future Sessions

- The deeper technical reference lives in `docs/PROJECT_CONTEXT.MD`. Keep `CLAUDE.md` short and operational.
- `notebooks/teacher_0_9_gap_analysis.ipynb` is the best artifact for explaining the teacher's residual ORG error.
- `notebooks/bio_fillter_testing.ipynb` confirms BIO repair did not move entity F1.
- The current best single efficient-after-dapt seed (`0.8553`) is not a headline result by itself. Use 3-seed mean +/- std for claims.
- Phase C is contractually required and is the centerpiece of the submission. Do not let teacher-side experiments compete with it for time.
- Any future model family beyond RoBERTa, DeBERTa-v3, and DistilRoBERTa should re-verify label alignment with `python -m src.data`.
