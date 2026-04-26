# CLAUDE.md

Working context for the FiNER-ORD Financial NER project. Keep this file current. Update the "Current Status" section first when the repo state changes.

---

## Project

Take-home challenge for the ML/NLP Research Engineer internship at Dunedain. Build a Financial NER model on the `gtfintechlab/finer-ord` dataset. The task is BIO-tagged token classification with 7 labels across PER / LOC / ORG entities.

**Primary metric:** entity-level F1 via `seqeval` (confirmed with Daniel).  
**Target:** reach or approach `0.9` test entity F1.  
**Current strategy:** abandon the teacher/student distillation pipeline. Focus solely on improving the strongest model family: `efficient_after_dapt` (`roberta-large` continued pretraining on FiNER train articles, then 5-epoch CRF fine-tune).  
**Current headline:** `efficient_after_dapt_logit_ensemble` at **`0.8634`** test entity F1. Best single-model 3-seed mean is `efficient_after_dapt` at **`0.8548 +/- 0.0038`**.

---

## Current Status

- [x] Initial one-seed baselines were run earlier (BERT-base, RoBERTa-base, RoBERTa-large). Best of that pass: RoBERTa-large at `0.8471` entity F1 on seed `88`.
- [x] Metric confirmed with Daniel: use strict entity-level `seqeval` F1, not the paper's token weighted F1.
- [x] Repo scaffolded around config-driven vanilla and CRF training.
- [x] `src/data.py`, `src/evaluate.py`, `src/train.py`, `src/crf_model.py`, `src/dapt.py`, `src/losses.py`, and the analysis scripts are written.
- [x] Smoke test passed. `smoke_test_seed88` completed and wrote `summary.json`, `predictions.json`, and CSV rows.
- [x] Phase B vanilla teacher completed for 3 seeds on CUDA. Aggregate test entity F1 is `0.8485 +/- 0.0022` (30 epochs).
- [x] Phase B CRF comparison completed for 3 seeds. Aggregate test entity F1 is `0.8521 +/- 0.0018` (30 epochs).
- [x] Efficient 5-epoch CRF baseline completed for 3 seeds. Aggregate test entity F1 is `0.8481 +/- 0.0066`.
- [x] DeBERTa-v3-large efficient recipe sweep completed. Negative result: all tested DeBERTa variants underperform RoBERTa-large at the 5-epoch budget.
  - `deberta_efficient`: `0.8374 +/- 0.0018`
  - `deberta_efficient_lr1e5`: `0.8282 +/- 0.0026`
  - `deberta_efficient_lr2e5`: `0.8325 +/- 0.0058`
  - `deberta_efficient_align_off`: `0.8339 +/- 0.0027`
- [x] BIO repair was analyzed on committed vanilla and CRF predictions. It produced essentially no entity-F1 gain.
- [x] The stricter CRF variant with valid-token-only packing and hard BIO constraints was tried locally, performed worse than the original CRF, and was reverted.
- [x] Phase 3 DAPT (10 epochs of MLM on 131/135 FiNER train articles, 180 windows of 512 tokens) plus 5-epoch CRF fine-tune completed for 3 seeds.
  - `efficient_after_dapt`: test entity F1 `0.8548 +/- 0.0038`
  - Mean fine-tune time: `7.21` minutes per seed plus `~3.5` minutes one-time DAPT
  - All three DAPT seeds beat the non-DAPT efficient-training mean, so the lift is real.
- [x] Phase 4 ensembles on top of `efficient_after_dapt` completed:
  - `efficient_after_dapt_logit_ensemble`: **`0.8634`** test entity F1 (PER `0.9564`, LOC `0.8590`, ORG `0.8206`)
  - `efficient_after_dapt_vote_ensemble`: `0.8587`
- [x] FNSPID-based DAPT v2 was attempted in code but abandoned before running. The dataset had many summary-only rows, and the license/distribution complexity outweighed the likely lift. The FNSPID/cc_news scaffolding was reverted.
- [x] Teacher/student distillation, student configs, and placeholder distillation code have been removed from the plan and repo. Do not reintroduce them unless explicitly requested.
- [ ] Next performance work should start from `configs/baseline/efficient_after_dapt.yaml` and its saved runs.
- [ ] Phase 5c packed-512 re-inference has not been run. Validation pass first on `efficient_after_dapt_seed*`; only run on test if validation delta is non-negative.
- [ ] Phase 5d Dice spike has not been run successfully yet. The current config uses `CE + Dice`, excludes `O` from the dice mean, and is vanilla-only.
- [ ] Phase 5e 6-model multi-recipe ensemble (`efficient_after_dapt_*` + `teacher_crf_*`) is blocked on extracting emissions/transitions for old `teacher_crf_*` runs.
- [ ] `README.md` is still a placeholder.
- [ ] `REPORT.md` does not exist yet.

---

## Critical Rules

These are easy to get wrong. Do not.

1. **Optimize for entity F1.** Token weighted F1 is useful context but not the headline metric.

2. **`efficient_after_dapt` is the current best model path.** Treat `configs/baseline/efficient_after_dapt.yaml` as the baseline for future F1 improvements.

3. **No teacher/student distillation work.** The project plan has changed. Do not add student configs, distillation trainers, latency/quantization/Pareto work, or a `src/distill.py` pipeline unless explicitly requested.

4. **Tokenizer always uses `add_prefix_space=True` when `is_split_into_words=True`.** This applies uniformly to RoBERTa and DeBERTa-family tokenizers; the data pipeline passes it unconditionally.

5. **Label mapping integers preserve the HF dataset's original ordering.** `0 -> O`, `1 -> B-PER`, `2 -> I-PER`, `3 -> B-LOC`, `4 -> I-LOC`, `5 -> B-ORG`, `6 -> I-ORG`.

6. **Sanity check label mapping before any long run.** `python -m src.data` should show sensible entities such as `"Obama" -> B-PER` and continuous multi-token entities.

7. **The checked-in CRF implementation is the original one.** It uses `attention_mask` as the CRF mask and replaces `-100` with `0` before the CRF forward pass. Do not assume the stricter packed / hard-BIO variant is active.

8. **Save `predictions.json` per run.** BIO repair and error analysis depend on it.

9. **Three seeds for headline single-model numbers.** Use `88, 5768, 78516` and report mean +/- std. Do not headline a single lucky seed.

10. **Always annotate F1 comparisons with epoch budget.** Vanilla and original CRF baselines ran for 30 epochs; efficient recipes run for 5 epochs.

11. **DAPT corpus must never include val/test articles.** `src.dapt.build_train_article_texts` filters out the 4 FiNER train articles whose reconstructed text exactly matches a val/test article. Do not bypass that filter.

12. **DAPT stays on FiNER train articles only unless explicitly approved.** Multi-source DAPT was attempted and abandoned.

13. **Ensemble logit mode requires byte-identical gold labels and per-example lengths across seed dumps.** `scripts/ensemble_logits.py` asserts both.

14. **Dice Loss and CRF stay separate.** `TrainConfig.__post_init__` raises if `loss_type=dice` and `use_crf=true`.

15. **Dice Loss runs as `CE + Dice`, not pure Dice.** Pure dice on FiNER's O-heavy distribution collapsed to predicting O everywhere.

16. **Packed-512 re-inference runs validation before test.** Position embeddings 256-511 saw pretrain context but not FiNER fine-tune context. Only run on test if the validation delta is non-negative.

---

## Repo Map

```text
finer-ord/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── docs/
│   └── PROJECT_CONTEXT.MD
├── configs/
│   ├── baseline_teacher.yaml
│   ├── smoke_test.yaml
│   └── baseline/
│       ├── teacher_crf.yaml
│       ├── efficient_training.yaml
│       ├── efficient_after_dapt.yaml
│       ├── efficient_after_dapt_lasllrd.yaml
│       ├── efficient_dice_seed88.yaml
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
│   └── finer-ord.py
├── scripts/
│   ├── bio_repair.py
│   ├── ensemble_logits.py
│   └── reinfer_packed.py
└── results/
    ├── results.csv
    └── results_detailed.csv
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

# Historical baselines
python -m src.train --config configs/baseline_teacher.yaml
python -m src.train --config configs/baseline/teacher_crf.yaml
python -m src.train --config configs/baseline/efficient_training.yaml

# Current best path: DAPT then 5-epoch CRF fine-tune
python -m src.dapt  --config configs/baseline/dapt_roberta_large.yaml
python -m src.train --config configs/baseline/efficient_after_dapt.yaml

# Current best ensemble, after efficient_after_dapt emits *_emissions.npz
python scripts/ensemble_logits.py \
  --runs efficient_after_dapt_seed88 efficient_after_dapt_seed5768 efficient_after_dapt_seed78516 \
  --mode logit \
  --use-crf \
  --output-name efficient_after_dapt_logit_ensemble

# Packed-512 re-inference on current best checkpoints: validation first
python scripts/reinfer_packed.py \
  --runs efficient_after_dapt_seed88 efficient_after_dapt_seed5768 efficient_after_dapt_seed78516 \
  --mode crf \
  --split val

# Test only if validation delta is non-negative
python scripts/reinfer_packed.py \
  --runs efficient_after_dapt_seed88 efficient_after_dapt_seed5768 efficient_after_dapt_seed78516 \
  --mode crf \
  --split test

# Dice Loss spike, single seed only unless it clearly wins
python -m src.train --config configs/baseline/efficient_dice_seed88.yaml

# Optional 6-model ensemble after old teacher_crf emissions are re-extracted
python scripts/ensemble_logits.py \
  --runs efficient_after_dapt_seed88 efficient_after_dapt_seed5768 efficient_after_dapt_seed78516 teacher_crf_seed88 teacher_crf_seed5768 teacher_crf_seed78516 \
  --mode logit \
  --use-crf \
  --output-name multi_recipe_logit_ensemble_6m
```

Add `--no-wandb` to training commands when logging is not desired.

---

## Key Technical Choices

**Current best model:** `roberta-large + FiNER-only DAPT + CRF`, implemented by `configs/baseline/efficient_after_dapt.yaml`.

**Best current result:** `efficient_after_dapt_logit_ensemble` at `0.8634` test entity F1.

**Best current single-model mean:** `efficient_after_dapt` at `0.8548 +/- 0.0038`.

**Why this path:** DAPT gave a consistent lift across all three seeds and kept the 5-epoch fine-tune budget. DeBERTa did not improve under the same budget. BIO repair and stricter CRF constraints did not address the main error mass.

**Known bottleneck:** ORG is still the weakest class. The best ensemble ORG F1 is `0.8206`, far below PER `0.9564`.

**Implemented training paths:** `src.train.py` handles vanilla multi-seed training and dispatches to `src.crf_model.py` when `use_crf: true`.

**DAPT:** `src.dapt.py` runs masked-LM continued pretraining on FiNER-ORD train article text only and excludes train articles duplicated in val/test.

**Ensembling:** `scripts/ensemble_logits.py` supports vote and logit/emission averaging. The headline path uses CRF emission averaging plus Viterbi re-decode.

**Pinned versions that matter:** `transformers==4.44.2`, `seqeval==1.2.2`, `numpy==1.26.4`, `torch==2.3.1`, `sentencepiece==0.2.0`. Do not upgrade blindly.

---

## Known Error Patterns

From saved prediction analysis and strict span confusion notebooks:

- **Validation-to-test gap is large.** Vanilla and CRF both drop by about `0.067` entity F1 from validation to test.
- **Token F1 is already very high.** Models sit around `0.985` token weighted F1, so the remaining problem is strict entity-span quality.
- **ORG is the main bottleneck.** Missed and spurious ORG spans dominate the remaining error burden.
- **LOC is the second bottleneck.** Boundary and partial-overlap errors are still common.
- **Type confusion is small.** Exact same-span wrong-type mistakes are much less important than missed, spurious, and boundary-overlap errors.
- **BIO repair does not help.** Invalid BIO tags are not the limiting issue.

Interventions tied to those patterns:

- **DAPT + efficient CRF** completed and is the current best path.
- **Logit ensemble** completed and is the current headline.
- **Packed-512 re-inference** is the next low-risk inference-only lever.
- **Multi-recipe logit ensemble** may lift F1 if old CRF teacher emissions are extracted.
- **Span-aware decoding or span-aware loss** remains the most plausible larger change if the 0.9 target remains out of reach.

---

## Style And Workflow Preferences

- **No em dashes.** Applies to report writing too.
- **Direct tone, grounded in specifics.** Avoid corporate jargon.
- **Performance-first.** Prefer changes that can plausibly lift entity F1 over efficiency-only work.
- **Validation before test for inference experiments.** Do not burn the test set on speculative decoding changes.
- **Honest about negative results.** Failed ablations are useful evidence.

---

## Open Questions

- Can packed-512 re-inference lift validation F1 for `efficient_after_dapt_seed*`?
- Does `efficient_after_dapt_lasllrd.yaml` improve the 3-seed mean enough to replace `efficient_after_dapt.yaml`?
- Can a 6-model multi-recipe logit ensemble narrow the gap to `0.9` without changing training?
- Would a span-based decoder or span-aware auxiliary loss address the ORG span boundary errors better than BIO + CRF?
- What exact submission format Dunedain expects: repo link, PDF, or both?

---

## Notes For Future Sessions

- The deeper technical reference lives in `docs/PROJECT_CONTEXT.MD`. Keep `CLAUDE.md` short and operational.
- `notebooks/teacher_0_9_gap_analysis.ipynb` is the best artifact for explaining why the models are still below `0.90`.
- `notebooks/bio_fillter_testing.ipynb` confirms BIO repair did not move entity F1.
- The current best single efficient-training seed (`0.8553`) is not a headline result by itself. Use 3-seed mean +/- std for claims.
- Results belong in the report. Keep this file focused on workflow, constraints, and current repo state.
- Any future model family beyond RoBERTa and DeBERTa-v3 should re-verify label alignment with `python -m src.data`.
