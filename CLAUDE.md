# CLAUDE.md

Working context for the FiNER-ORD Financial NER project. Keep this file current. Update the "Current Status" section first when the repo state changes.

---

## Project

Take-home challenge for the ML/NLP Research Engineer internship at Dunedain. Build a Financial NER model on the `gtfintechlab/finer-ord` dataset. The task is BIO-tagged token classification with 7 labels across PER / LOC / ORG entities.

**Primary metric:** entity-level F1 via `seqeval` (confirmed with Daniel).

**Submission framing:** "F1 or efficiency wins." We optimize for both. The deliverable is a Pareto frontier showing teacher F1, distilled student F1, and INT8 quantized student F1 against latency and model size.

**Current strategy:** the teacher and the student are both locked. All remaining work is on latency measurement, quantization, the Pareto chart, and the report. Cheap remaining teacher experiments (multi-recipe 6-model ensemble, packed-512 re-inference) remain optional but are unlikely to be worth the time given the submission deadline.

**Locked teacher:** `efficient_after_dapt_logit_ensemble` at **`0.8634`** test entity F1. Best single-model 3-seed mean is `efficient_after_dapt` at **`0.8548 +/- 0.0038`**.

**Locked student:** `student_distilled_logit_ensemble` at **`0.8304`** test entity F1. Single-model 3-seed mean: **`0.8277`** (range `0.8262`-`0.8298`). Both Phase C targets hit (>= `0.82` single, >= `0.83` ensemble).

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
- [x] Phase 3 DAPT (10 epochs MLM on 131/135 FiNER train articles, 180 windows of 512 tokens) plus 5-epoch CRF fine-tune: `efficient_after_dapt` at `0.8548 +/- 0.0038`.
- [x] Phase 4 ensembles on top of `efficient_after_dapt`:
  - `efficient_after_dapt_logit_ensemble`: **`0.8634`** test entity F1 (PER `0.9564`, LOC `0.8590`, ORG `0.8206`)
  - `efficient_after_dapt_vote_ensemble`: `0.8587`
- [x] FNSPID-based DAPT v2 attempted in code, abandoned before running. Many summary-only rows; CC BY-NC license was incompatible with shipping a derived checkpoint to a for-profit submission.
- [x] RoBERTa LAS + LLRD ablation (`efficient_after_dapt_lasllrd.yaml`) completed for 3 seeds. Logit ensemble landed at `0.8368`, a `-0.027` regression vs the `efficient_after_dapt` ensemble. All three classes regressed (PER `-0.021`, LOC `-0.028`, ORG `-0.029`). Hypothesis: `llrd_decay=0.9` is too aggressive on a DAPT'd backbone where the lower layers were already domain-adapted; combined with LAS, the head over-corrects for an under-adapting backbone. Not promoted; documented as a negative result.
- [x] Teacher locked at `efficient_after_dapt_logit_ensemble` (`0.8634`).
- [x] **Teacher reproduced from configs after a Colab session checkpoint loss event.** Initial Colab session saved emissions but the teacher checkpoint directories were lost. After fresh DAPT (10 epochs MLM) and 3-seed CRF fine-tunes, the reproduced ensemble F1 matched `0.8634` exactly to four decimals, with per-class numbers also matching. Per-seed test F1: `0.8543` (seed 88), `0.8514` (seed 5768), `0.8588` (seed 78516).

### Completed Phase C work

- [x] `src/distill.py` implemented for offline distillation from saved teacher emissions. KL on softened teacher emissions plus CE on gold labels, weighted `alpha * CE + (1 - alpha) * T^2 * KL`. Defaults: `T=2.0`, `alpha=0.5`. `-100` masking applied identically to both terms.
- [x] `scripts/extract_train_emissions.py` written and run. Each locked teacher seed has `train_emissions.npz` (3262 examples, dtype=object, shape `(seq_len, 7)` per example) saved alongside the existing val/test emissions.
- [x] `configs/baseline/student_distilled.yaml` updated to the Phase C spec: `model_name=distilroberta-base`, `seeds=[88, 5768, 78516]`, `lr=5e-5`, `batch_size=16`, `use_distillation=true`, `use_crf=false`, `teacher_mode=ensemble`, `teacher_runs=[efficient_after_dapt_seed88, efficient_after_dapt_seed5768, efficient_after_dapt_seed78516]`, `distill_temperature=2.0`, `distill_alpha=0.5`. `num_epochs=20` (see deviation note below).
- [x] `TrainConfig` in `src/train.py` extended with the four distillation fields and dispatch to `src.distill.run_distillation` when `use_distillation=true`.
- [x] 3-seed student distillation complete. Per-seed test entity F1: `0.8298` (seed 88), `0.8262` (seed 5768), `0.8272` (seed 78516). Mean `0.8277`, range `0.0036`. Best checkpoints landed at epoch 15-16 across all three seeds (steps `3060`, `3060`, `3264`). Train time ~2-3 min/seed on Colab GPU.
- [x] Student logit ensemble: **`0.8304`** test entity F1 (PER `0.9340`, LOC `0.8325`, ORG `0.7781`). Both targets cleared.
- [x] All Phase C artifacts (teacher checkpoints, student checkpoints, DAPT backbone, emissions, summaries) downloaded from Colab to local repo.

### Active Phase C work

- [ ] Run `scripts/measure_latency.py` against teacher single, student single (FP32), and INT8 student. CPU is the headline number; CUDA is bonus context. Batch sizes 1 and 8 with 10-batch warmup.
- [ ] Run `scripts/quantize_student.py` on each student seed. Acceptance: INT8 F1 within `0.01` of FP32 student (i.e., >= ~`0.82`).
- [ ] Run `scripts/build_pareto.py`. Produces `docs/figures/pareto.png` (latency-vs-F1 and size-vs-F1 subplots) and `docs/figures/pareto_data.csv`.
- [ ] Write `REPORT.md`. This is the bulk of remaining work.
- [ ] Polish `README.md`.

### Spec deviations (document in REPORT.md)

- **Student trained for 20 epochs, not 5.** Initial 5-epoch run landed at ~`0.81` mean entity F1, missing the `0.82` target. Distillation on smoothed soft targets has a weaker per-step gradient signal than vanilla CE fine-tuning; at 3262 train examples with `alpha=0.5`, convergence happens later than for a CRF fine-tune. With 20 epochs and HF `EarlyStoppingCallback` plus `load_best_model_at_end`, all three seeds selected checkpoints from epoch 15-16. Val-test gap was `0.086`-`0.093` (similar to the teacher's `0.067`), so the late-epoch selection reflects real convergence, not val overfitting. Train time per seed: ~2-3 min on Colab GPU. Cost of the deviation: ~8 min total wall-clock.

### Optional teacher-side experiments (not on the critical path)

- [ ] Phase 5c packed-512 re-inference. Run validation first on `efficient_after_dapt_seed*`. Test only if validation delta is non-negative. Unlikely to be worth the time given submission deadline.
- [ ] Phase 5d Dice spike. CE + Dice with O excluded, vanilla-only. Single seed (88). Acceptance: `+0.005` over `efficient_training_seed88` (`0.8467`). Exploratory; not part of the headline.
- [ ] Phase 5e 6-model multi-recipe ensemble. Blocked on re-extracting `*_emissions.npz` and `crf_transitions.npz` for `teacher_crf_seed*`. Cheap once unblocked, plausible `+0.005` to `+0.015` over `0.8634`.

---

## Critical Rules

These are easy to get wrong. Do not.

1. **Optimize for entity F1 on the teacher and the student.** Token weighted F1 is useful context but not the headline metric.

2. **The teacher is locked.** Do not retrain `efficient_after_dapt_seed*`, do not change `configs/baseline/efficient_after_dapt.yaml`, and do not modify `src/crf_model.py`.

3. **The student is also locked at `0.8304` ensemble.** Do not retrain to chase F1. Epoch sweeps that select on test F1 are test-set fishing and not reportable.

4. **Distillation is offline, from saved emissions.** Generate `train_emissions.npz` once via `scripts/extract_train_emissions.py` and reuse it across student seeds. No live teacher forward passes during student training.

5. **Distillation defaults to ensemble teacher.** Average emissions across the 3 teacher seeds before computing the soft loss. `teacher_mode=single` exists for ablation; do not use it for the headline.

6. **Student does not use a CRF.** Vanilla token classification head. The CRF added marginal lift on the teacher and adds inference-time cost the deployment story does not want.

7. **Tokenizer alignment must match.** Teacher emissions were generated with `label_all_subwords=false`, `max_seq_length=256`, and the RoBERTa BPE tokenizer with `add_prefix_space=True`. The student uses `distilroberta-base`, which shares the same BPE; alignment is preserved.

8. **`add_prefix_space=True` when `is_split_into_words=True`.** Applies to RoBERTa and DeBERTa families uniformly.

9. **Label mapping integers preserve the HF dataset's original ordering.** `0 -> O`, `1 -> B-PER`, `2 -> I-PER`, `3 -> B-LOC`, `4 -> I-LOC`, `5 -> B-ORG`, `6 -> I-ORG`.

10. **Sanity check label mapping before any long run.** `python -m src.data` should show `"Obama" -> B-PER` and continuous multi-token entities.

11. **The checked-in CRF implementation is the original one.** Uses `attention_mask` as the CRF mask, replaces `-100` with `0` before CRF forward.

12. **Save `predictions.json` per run.** BIO repair, error analysis, and the Pareto chart depend on it.

13. **Three seeds for headline single-model numbers.** Use `88, 5768, 78516` and report mean +/- std (or mean and range when std is misleading at n=3). Do not headline a single lucky seed.

14. **Always annotate F1 comparisons with epoch budget and recipe family.** Vanilla and original CRF baselines ran for 30 epochs; efficient teacher recipes run for 5; the student runs for 20 (longer because distillation needs it).

15. **DAPT corpus must never include val/test articles.** `src.dapt.build_train_article_texts` filters out the 4 FiNER train articles whose reconstructed text exactly matches a val/test article.

16. **DAPT stays on FiNER train articles only.** Multi-source DAPT was attempted and abandoned for licensing and quality reasons.

17. **Ensemble logit mode requires byte-identical gold labels and per-example lengths across seed dumps.** `scripts/ensemble_logits.py` asserts both.

18. **Dice Loss and CRF stay separate.** `TrainConfig.__post_init__` raises if `loss_type=dice` and `use_crf=true`. Distillation also forbids `use_crf=true` with `use_distillation=true`.

19. **Dice Loss runs as `CE + Dice`, not pure Dice.** Pure dice on FiNER's O-heavy distribution collapsed to predicting O everywhere.

20. **Packed-512 re-inference runs validation before test.** Position embeddings 256-511 saw pretrain context but not FiNER fine-tune context. Only run on test if the validation delta is non-negative.

21. **INT8 quantization latency must be measured on CPU.** `torch.quantization.quantize_dynamic` is CPU-targeted; on GPU the quantized checkpoint runs at FP32 speed and the Pareto chart will look identical to the unquantized student.

22. **Always invoke project Python with `./.venv/bin/python` to avoid numpy version mismatch.** Object-array `.npz` files saved by venv numpy cannot always be read by system numpy across the 1.26/2.0 boundary; failure mode is `TypeError: _reconstruct: First argument must be a sub-type of ndarray` at `np.load` time.

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
./.venv/bin/python -m src.data
./.venv/bin/python -m src.train --config configs/smoke_test.yaml --run-checks

# Historical baselines (for context, do not re-run)
./.venv/bin/python -m src.train --config configs/baseline_teacher.yaml
./.venv/bin/python -m src.train --config configs/baseline/teacher_crf.yaml
./.venv/bin/python -m src.train --config configs/baseline/efficient_training.yaml

# Locked teacher path (already complete; reproduces 0.8634 ensemble)
./.venv/bin/python -m src.dapt  --config configs/baseline/dapt_roberta_large.yaml
./.venv/bin/python -m src.train --config configs/baseline/efficient_after_dapt.yaml

# Locked teacher ensemble (already complete; this is the headline)
./.venv/bin/python scripts/ensemble_logits.py \
  --runs efficient_after_dapt_seed88 efficient_after_dapt_seed5768 efficient_after_dapt_seed78516 \
  --mode logit \
  --use-crf \
  --output-name efficient_after_dapt_logit_ensemble

# Phase C: extract train emissions, distill student, ensemble
./.venv/bin/python scripts/extract_train_emissions.py \
  --runs efficient_after_dapt_seed88 efficient_after_dapt_seed5768 efficient_after_dapt_seed78516

./.venv/bin/python -m src.distill --config configs/baseline/student_distilled.yaml --smoke
./.venv/bin/python -m src.distill --config configs/baseline/student_distilled.yaml

# Student ensemble (vanilla, no --use-crf; reproduces 0.8304)
./.venv/bin/python scripts/ensemble_logits.py \
  --runs student_distilled_seed88 student_distilled_seed5768 student_distilled_seed78516 \
  --mode logit \
  --output-name student_distilled_logit_ensemble

# Latency, quantization, Pareto (active queue)
./.venv/bin/python scripts/measure_latency.py --runs efficient_after_dapt_seed88 student_distilled_seed88 --device cuda
./.venv/bin/python scripts/measure_latency.py --runs efficient_after_dapt_seed88 student_distilled_seed88 --device cpu
./.venv/bin/python scripts/quantize_student.py --runs student_distilled_seed88 student_distilled_seed5768 student_distilled_seed78516
./.venv/bin/python scripts/measure_latency.py --runs student_distilled_seed88_int8 --device cpu
./.venv/bin/python scripts/build_pareto.py

# Optional teacher-side side experiments
./.venv/bin/python scripts/reinfer_packed.py \
  --runs efficient_after_dapt_seed88 efficient_after_dapt_seed5768 efficient_after_dapt_seed78516 \
  --mode crf --split val
./.venv/bin/python -m src.train --config configs/baseline/efficient_dice_seed88.yaml
```

Add `--no-wandb` to training commands when logging is not desired.

---

## Key Technical Choices

**Locked teacher:** `roberta-large + FiNER-only DAPT + CRF`, ensemble of 3 seeds. Headline is `efficient_after_dapt_logit_ensemble` at `0.8634` test entity F1.

**Why this teacher path:** DAPT gave a consistent lift across all three seeds and kept the 5-epoch fine-tune budget. DeBERTa did not improve under the same budget. BIO repair, stricter CRF, FNSPID, and RoBERTa LAS + LLRD all failed to improve the recipe.

**Known teacher bottleneck:** ORG is the weakest class. The best ensemble ORG F1 is `0.8206`, far below PER `0.9564`. Most of the residual error is missed and spurious ORG spans, not type confusion.

**Student backbone:** `distilroberta-base`. 82M parameters vs RoBERTa-large's 355M (4.3x reduction). Same BPE tokenizer as the teacher, so emission alignment transfers without re-extraction.

**Student headline:** `student_distilled_logit_ensemble` at `0.8304` test entity F1. Single-model 3-seed mean `0.8277` (range `0.8262`-`0.8298`). Both targets cleared. Teacher-to-student ensemble gap of `0.0330`. ORG bottleneck transfers: student ORG `0.7781` vs teacher ensemble ORG `0.8206` (`-0.0425`); PER and LOC distill more cleanly (`-0.0224` and `-0.0265` respectively).

**Distillation loss:** weighted combination of `alpha * CE(student_logits, gold)` and `(1 - alpha) * T^2 * KL(softmax(student_logits / T) || softmax(teacher_emissions / T))`. Defaults `T=2.0`, `alpha=0.5`. Mask `-100` positions identically in both terms.

**Ensemble teacher:** average teacher emissions across the 3 seeds before computing the soft loss.

**Student does not use a CRF.** The CRF added `+0.004` on the teacher; on a smaller student where capacity is the binding constraint, that lift is unlikely to transfer, and the CRF adds inference-time Viterbi cost that hurts the deployment story.

**Why 20 epochs for the student.** 5 epochs landed at ~`0.81` mean (below target). Distillation on smoothed soft targets has a weaker per-step gradient signal than vanilla CE; convergence happens later. Best checkpoints landed at epoch 15-16 across all three seeds, with val-test gap matching the teacher's; this is real convergence, not val overfitting.

**Quantization:** dynamic INT8 via `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`. Single function call, CPU-targeted, no QAT or static quantization.

**Pareto chart:** two subplots. Latency-vs-F1 uses single-example latency at batch size 1 (the deployment-relevant number). Size-vs-F1 uses on-disk model size. Both axes plot all six points: teacher single, teacher ensemble, student single, student ensemble, INT8 student single, INT8 student ensemble. Frontier marked explicitly.

**Pinned versions that matter:** `transformers==4.44.2`, `seqeval==1.2.2`, `numpy==1.26.4`, `torch==2.3.1`, `sentencepiece==0.2.0`. Do not upgrade blindly.

---

## Known Error Patterns

From saved prediction analysis and strict span confusion notebooks:

- **Validation-to-test gap is large.** Teacher: `0.067`. Student: `0.086`-`0.093`. Diagnosed as lexical novelty in the test set (out-of-vocabulary corporate names).
- **Token F1 is already very high.** Teacher: `0.985`+. Student: `0.984`+. Residual error is strict span quality, not generic token labeling.
- **ORG is the main bottleneck on both teacher and student.** Missed and spurious ORG spans dominate the remaining error burden.
- **LOC is the second bottleneck.** Boundary and partial-overlap errors are still common.
- **Type confusion is small.** Same-span wrong-type mistakes are much less important than missed, spurious, and boundary-overlap errors.
- **BIO repair does not help.** Invalid BIO tags are not the limiting issue.

Interventions tied to those patterns:

- **DAPT + efficient CRF** completed and is the locked teacher path.
- **Logit ensemble** completed for both teacher (`0.8634`) and student (`0.8304`).
- **LAS + LLRD on a DAPT'd backbone** failed; documented as negative.
- **Distillation from the ensemble** completed; both targets hit.
- **Span-aware decoding** remains the most plausible larger change for future work; out of scope under this take-home time budget.

---

## Style And Workflow Preferences

- **No em dashes.** Applies to report writing too.
- **Direct tone, grounded in specifics.** Avoid corporate jargon.
- **Pareto-first framing.** The submission is a frontier story (teacher F1, student F1, INT8 F1, latency, size), not a single-number story.
- **Validation before test for inference experiments.** Do not burn the test set on speculative decoding changes.
- **Honest about negative results.** Failed ablations are useful evidence and should be in `REPORT.md`.

---

## Open Questions

- What CPU / CUDA latency numbers do teacher single, student single, and INT8 student actually produce on the local machine and on Colab?
- Does INT8 quantization stay within `0.01` F1 of the FP32 student?
- Should `scripts/build_pareto.py` plot the optional teacher experiments (LAS+LLRD failed, packed-512 pending) as dominated points, or omit them?
- Does the 6-model multi-recipe ensemble lift the teacher headline if old `teacher_crf_*` emissions get re-extracted? Cheap to check; not on the critical path.
- What exact submission format Dunedain expects: repo link, PDF, or both?

---

## Notes For Future Sessions

- The deeper technical reference lives in `docs/PROJECT_CONTEXT.MD`. Keep `CLAUDE.md` short and operational.
- `notebooks/teacher_0_9_gap_analysis.ipynb` is the best artifact for explaining the teacher's residual ORG error.
- `notebooks/bio_fillter_testing.ipynb` confirms BIO repair did not move entity F1.
- The current best single efficient-after-dapt seed (`0.8588`) is not a headline result by itself. Use 3-seed mean +/- std for claims.
- All checkpoints (teacher CRF, FP32 student, DAPT backbone) are now stored locally in `results/`. Colab is treated as ephemeral compute; recovery from Colab eviction has been validated end-to-end.
- Phase C is the centerpiece of the submission. The remaining work (latency, quantization, Pareto, REPORT.md) is what produces the actual submission deliverables.
- Any future model family beyond RoBERTa, DeBERTa-v3, and DistilRoBERTa should re-verify label alignment with `python -m src.data`.
